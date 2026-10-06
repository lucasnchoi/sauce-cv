"""Scores predicted tracks against ground truth: IDF1 and detection counts."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

from tracker.schemas import validate

BOX_COLUMNS = ["x1", "y1", "x2", "y2"]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection-over-union of every box in a with every box in b. Zero-area boxes give 0."""
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    left = np.maximum(a[:, None, 0], b[None, :, 0])
    top = np.maximum(a[:, None, 1], b[None, :, 1])
    right = np.minimum(a[:, None, 2], b[None, :, 2])
    bottom = np.minimum(a[:, None, 3], b[None, :, 3])
    intersection = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


@dataclass(frozen=True)
class TrackingCounts:
    idtp: int       # boxes matched under the best one-to-one pairing of tracks
    n_gt: int       # ground-truth boxes
    n_pred: int     # predicted boxes
    tp: int         # boxes matched frame by frame, ignoring identity

    @property
    def fp(self) -> int:
        return self.n_pred - self.tp

    @property
    def fn(self) -> int:
        return self.n_gt - self.tp

    @property
    def idf1(self) -> float:
        total = self.n_gt + self.n_pred
        return 2 * self.idtp / total if total else 0.0

    @property
    def precision(self) -> float:
        return self.tp / self.n_pred if self.n_pred else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.n_gt if self.n_gt else 0.0

    def __add__(self, other: "TrackingCounts") -> "TrackingCounts":
        return TrackingCounts(
            idtp=self.idtp + other.idtp,
            n_gt=self.n_gt + other.n_gt,
            n_pred=self.n_pred + other.n_pred,
            tp=self.tp + other.tp,
        )


def _by_frame(df: pl.DataFrame, label: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if df.select("frame", "track_id").is_duplicated().any():
        raise ValueError(f"{label}: a track id appears more than once in the same frame")
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for key, part in df.partition_by("frame", as_dict=True).items():
        frame = key[0] if isinstance(key, tuple) else key
        out[int(frame)] = (part["track_id"].to_numpy(), part.select(BOX_COLUMNS).to_numpy())
    return out


def score_tracking(gt: pl.DataFrame, pred: pl.DataFrame, iou_threshold: float = 0.5) -> TrackingCounts:
    gt = validate(gt, "tracks")
    pred = validate(pred, "tracks")
    if gt.height == 0:
        raise ValueError("no ground truth boxes to score against")
    gt_frames = _by_frame(gt, "ground truth")
    pred_frames = _by_frame(pred, "predictions")

    pair_frames: dict[tuple[int, int], int] = defaultdict(int)
    tp = 0
    for frame, (gt_ids, gt_boxes) in gt_frames.items():
        if frame not in pred_frames:
            continue
        pred_ids, pred_boxes = pred_frames[frame]
        iou = iou_matrix(gt_boxes, pred_boxes)
        overlaps = iou >= iou_threshold
        for g, p in zip(*np.nonzero(overlaps)):
            pair_frames[(int(gt_ids[g]), int(pred_ids[p]))] += 1
        # One-to-one pairing within the frame; the large cost keeps non-overlapping
        # pairs from being preferred, so the number of overlapping pairs is maximal.
        cost = np.where(overlaps, 1.0 - iou, 1e6)
        rows, cols = linear_sum_assignment(cost)
        tp += int(overlaps[rows, cols].sum())

    idtp = 0
    if pair_frames:
        gt_index = {t: i for i, t in enumerate(sorted({g for g, _ in pair_frames}))}
        pred_index = {t: i for i, t in enumerate(sorted({p for _, p in pair_frames}))}
        weights = np.zeros((len(gt_index), len(pred_index)))
        for (g, p), count in pair_frames.items():
            weights[gt_index[g], pred_index[p]] = count
        rows, cols = linear_sum_assignment(-weights)
        idtp = int(weights[rows, cols].sum())

    return TrackingCounts(idtp=idtp, n_gt=gt.height, n_pred=pred.height, tp=tp)
