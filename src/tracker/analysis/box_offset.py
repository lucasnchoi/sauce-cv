"""Why do tracked boxes miss ground-truth boxes? Offsets, sizes, and the overlap rule.

Investigation 1 of Plan 2a. Every function is pure table logic so the
findings note can cite numbers that a test pins.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import polars as pl

from tracker.datasets.mot import load_mot_ground_truth
from tracker.evaluate.tracking import TrackingCounts, iou_matrix, score_tracking
from tracker.pipeline import read_clip_info
from tracker.schemas import validate
from tracker.tables import read_table

BOX = ["x1", "y1", "x2", "y2"]
IOU_BINS = [0.0, 0.1, 0.3, 0.5, 1.0]
BIN_LABELS = ["<0.1", "0.1-0.3", "0.3-0.5", ">=0.5"]
DELTAS = ["dx_centre", "dy_centre", "d_bottom", "width_ratio", "height_ratio", "pred_area"]


def _boxes(df: pl.DataFrame) -> pl.DataFrame:
    """Accept a tracks or detections table; keep frame and box."""
    table = "tracks" if "track_id" in df.columns else "detections"
    return validate(df, table).select(["frame"] + BOX)


def _by_frame(df: pl.DataFrame) -> dict[int, np.ndarray]:
    return {
        int(key[0] if isinstance(key, tuple) else key): part.select(BOX).to_numpy().astype(float)
        for key, part in df.partition_by("frame", as_dict=True).items()
    }


def nearest_pairs(pred: pl.DataFrame, gt: pl.DataFrame) -> pl.DataFrame:
    """For every predicted box, the ground-truth box in the same frame it overlaps most.

    Matching is many-to-one: each prediction picks its own best ground-truth box
    independently. On a row with best_iou == 0 the gt_* columns hold an arbitrary box
    from that frame (argmax of zeros), so only best_iou and the null deltas mean anything there.
    """
    pred, gt = _boxes(pred), _boxes(gt)
    gt_frames = _by_frame(gt)
    rows: list[dict] = []
    for frame, part in sorted(_by_frame(pred).items()):
        truth = gt_frames.get(frame)
        iou = iou_matrix(part, truth) if truth is not None and len(truth) else np.zeros((len(part), 0))
        for i, box in enumerate(part):
            best = int(iou[i].argmax()) if iou.shape[1] else -1
            best_iou = float(iou[i, best]) if best >= 0 else 0.0
            row = {"frame": frame, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                   "best_iou": best_iou, "pred_area": float((box[2] - box[0]) * (box[3] - box[1]))}
            if best >= 0:
                g = truth[best]
                row.update(gt_x1=g[0], gt_y1=g[1], gt_x2=g[2], gt_y2=g[3])
            else:
                row.update(gt_x1=None, gt_y1=None, gt_x2=None, gt_y2=None)
            if best_iou > 0:
                gw, gh = g[2] - g[0], g[3] - g[1]
                row.update(
                    dx_centre=float((box[0] + box[2]) / 2 - (g[0] + g[2]) / 2),
                    dy_centre=float((box[1] + box[3]) / 2 - (g[1] + g[3]) / 2),
                    d_bottom=float(g[3] - box[3]),
                    width_ratio=float((box[2] - box[0]) / gw) if gw > 0 else None,
                    height_ratio=float((box[3] - box[1]) / gh) if gh > 0 else None,
                )
            else:
                row.update(dx_centre=None, dy_centre=None, d_bottom=None, width_ratio=None, height_ratio=None)
            rows.append(row)
    schema = {"frame": pl.Int64, "x1": pl.Float64, "y1": pl.Float64, "x2": pl.Float64, "y2": pl.Float64,
              "best_iou": pl.Float64, "pred_area": pl.Float64,
              "gt_x1": pl.Float64, "gt_y1": pl.Float64, "gt_x2": pl.Float64, "gt_y2": pl.Float64,
              "dx_centre": pl.Float64, "dy_centre": pl.Float64, "d_bottom": pl.Float64,
              "width_ratio": pl.Float64, "height_ratio": pl.Float64}
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def summarize(pairs: pl.DataFrame) -> dict:
    n = pairs.height
    if n == 0:
        return {"n_pred": 0, "bin_share": {label: 0.0 for label in BIN_LABELS},
                "matched_medians": {name: None for name in DELTAS},
                "iou_by_area_tercile": {"small": None, "medium": None, "large": None}}
    iou = pairs["best_iou"].to_numpy()
    counts = np.histogram(np.clip(iou, 0.0, 1.0), bins=IOU_BINS)[0]
    bin_share = {label: float(c) / n for label, c in zip(BIN_LABELS, counts)}
    matched = pairs.filter(pl.col("best_iou") >= 0.3)
    medians = {name: (float(matched[name].median()) if matched.height and matched[name].null_count() < matched.height else None)
               for name in DELTAS}
    cuts = np.quantile(pairs["pred_area"].to_numpy(), [1 / 3, 2 / 3])
    terciles = {}
    for label, low, high in (("small", -np.inf, cuts[0]), ("medium", cuts[0], cuts[1]), ("large", cuts[1], np.inf)):
        part = pairs.filter((pl.col("pred_area") > low) & (pl.col("pred_area") <= high))
        terciles[label] = float(part["best_iou"].median()) if part.height else None
    return {"n_pred": n, "bin_share": bin_share, "matched_medians": medians, "iou_by_area_tercile": terciles}


def box_agreement(a: pl.DataFrame, b: pl.DataFrame) -> dict:
    """How closely the boxes in a match the boxes in b, frame by frame (e.g. tracks vs detections)."""
    pairs = nearest_pairs(a, b)
    if pairs.height == 0:
        return {"n": 0, "share_identical": None, "median_iou": None}
    iou = pairs["best_iou"].to_numpy()
    return {"n": pairs.height, "share_identical": float((iou > 0.999).mean()), "median_iou": float(np.median(iou))}


def threshold_sweep(gt: pl.DataFrame, pred: pl.DataFrame, thresholds=(0.3, 0.4, 0.5)) -> dict[float, TrackingCounts]:
    return {t: score_tracking(gt, pred, iou_threshold=t) for t in thresholds}


def shifted_share(pred: pl.DataFrame, gt: pl.DataFrame, k: int, iou_threshold: float = 0.5) -> float:
    """Share of predicted boxes whose best IoU is at least the threshold when read against shifted ground truth.

    The prediction at frame f is compared with the ground truth labelled f - k (the ground truth's
    frame column is moved up by k), so positive k means the ground truth reads k frames earlier,
    the convention of docs/investigation-box-offset.md. A prediction whose shifted frame has no
    ground truth counts as not matched. k = 0 is the real labelling.
    """
    shifted = gt.with_columns((pl.col("frame") + k).alias("frame"))
    iou = nearest_pairs(pred, shifted)["best_iou"].to_numpy()
    return float((iou >= iou_threshold).mean()) if len(iou) else 0.0


def drift_by_range(pred: pl.DataFrame, gt: pl.DataFrame, n_ranges: int = 5, ks=range(-2, 9),
                   iou_threshold: float = 0.5) -> list[dict]:
    """Does the best shift change along the clip? Equal frame ranges over the prediction's frames.

    Each prediction's best IoU is computed once per k; ranges only group those rows. Returns, per
    range with at least one frame, its first and last frame, the k with the highest share (the lowest k on a tie), that
    share, and the share at k = 0.
    """
    if pred.height == 0:
        return []
    edges = np.linspace(pred["frame"].min(), pred["frame"].max() + 1, n_ranges + 1).astype(int)
    shares = {}                                      # k -> share per range
    for k in sorted({*ks, 0}):
        pairs = nearest_pairs(pred, gt.with_columns((pl.col("frame") + k).alias("frame")))
        frames, hit = pairs["frame"].to_numpy(), pairs["best_iou"].to_numpy() >= iou_threshold
        shares[k] = [float(hit[(frames >= first) & (frames < stop)].mean()) if ((frames >= first) & (frames < stop)).any() else 0.0
                     for first, stop in zip(edges[:-1], edges[1:])]
    ranges = []
    for i, (first, stop) in enumerate(zip(edges[:-1], edges[1:])):
        if stop <= first:                            # fewer frames than ranges
            continue
        best = max(sorted(ks), key=lambda k: shares[k][i])
        ranges.append({"range": (int(first), int(stop) - 1), "best_k": best,
                       "share_at_best": shares[best][i], "share_at_zero": shares[0][i]})
    return ranges


def diagnose_clip(run_dir, clip_id: str, seq_dir) -> dict:
    run_dir = Path(run_dir)
    tracks = read_table("tracks", run_dir / clip_id / "tracks.parquet")
    detections = read_table("detections", run_dir / clip_id / "detections.parquet")
    gt = load_mot_ground_truth(seq_dir)
    sweep = threshold_sweep(gt, tracks)
    return {
        "clip_id": clip_id,
        "fps": read_clip_info(run_dir / clip_id / "clip.json").fps,
        "tracks": summarize(nearest_pairs(tracks, gt)),
        "detections": summarize(nearest_pairs(detections, gt)),
        "tracks_vs_detections": box_agreement(tracks, detections),
        "sweep": {str(t): {"idf1": c.idf1, "precision": c.precision, "recall": c.recall} for t, c in sweep.items()},
    }


def shift_clip(run_dir, clip_id: str, seq_dir, n_ranges: int = 5, ks=range(-2, 9)) -> dict:
    """The frame-shift tables for one clip: share at IoU 0.5 for each k, and the drift along the clip."""
    tracks = read_table("tracks", Path(run_dir) / clip_id / "tracks.parquet")
    gt = load_mot_ground_truth(seq_dir)
    return {
        "clip_id": clip_id,
        "shifted_share": {str(k): shifted_share(tracks, gt, k) for k in ks},
        "drift_by_range": drift_by_range(tracks, gt, n_ranges, ks),
    }


def draw_comparison(clip, frame: int, gt: pl.DataFrame, tracks: pl.DataFrame, detections: pl.DataFrame, out_path) -> None:
    """Ground truth green, tracked boxes red with their id, raw detections thin blue."""
    image = clip.read_frame(frame)
    for row in detections.filter(pl.col("frame") == frame).iter_rows(named=True):
        cv2.rectangle(image, (int(row["x1"]), int(row["y1"])), (int(row["x2"]), int(row["y2"])), (255, 128, 0), 1)
    for row in gt.filter(pl.col("frame") == frame).iter_rows(named=True):
        cv2.rectangle(image, (int(row["x1"]), int(row["y1"])), (int(row["x2"]), int(row["y2"])), (0, 200, 0), 2)
    for row in tracks.filter(pl.col("frame") == frame).iter_rows(named=True):
        p = (int(row["x1"]), int(row["y1"]))
        cv2.rectangle(image, p, (int(row["x2"]), int(row["y2"])), (0, 0, 255), 2)
        cv2.putText(image, str(row["track_id"]), (p[0], max(12, p[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)
