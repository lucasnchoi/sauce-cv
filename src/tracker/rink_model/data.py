"""The owner's labels as Keypoint R-CNN training samples.

One sample per labelled frame that has at least one visible point. The target
is a single "rink" box around the visible points, and the 33 keypoints with
COCO visibility 2 (visible) or 0 (not). Frames are read from the dataset on
demand; images are never copied out of data/raw.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import torch

from tracker.rink import KEYPOINTS
from tracker.video import Clip

BOX_PAD_PX = 20.0
N = len(KEYPOINTS)


@dataclass(frozen=True)
class Sample:
    clip_id: str
    frame: int
    points: np.ndarray      # (33, 3): x_px, y_px, visible (1.0 or 0.0)


def samples_from_labels(labels: pl.DataFrame) -> list[Sample]:
    out = []
    for key, part in sorted(labels.partition_by("clip_id", "frame", as_dict=True).items()):
        clip_id, frame = key
        points = np.zeros((N, 3), dtype=float)
        for row in part.iter_rows(named=True):
            if row["visible"]:
                points[row["keypoint_id"]] = (row["x_px"], row["y_px"], 1.0)
        if points[:, 2].sum() >= 1:
            out.append(Sample(str(clip_id), int(frame), points))
    return out


def split_validation(samples: list[Sample], seed: int, share: float = 0.2) -> tuple[list[Sample], list[Sample]]:
    """A seeded validation slice from the same (training) clips; never the held-out clips."""
    order = list(samples)
    random.Random(seed).shuffle(order)
    n_val = max(1, round(len(order) * share)) if len(order) > 1 else 0
    key = lambda s: (s.clip_id, s.frame)
    return sorted(order[n_val:], key=key), sorted(order[:n_val], key=key)


def target_for(points: np.ndarray, width: int, height: int) -> dict[str, torch.Tensor]:
    visible = points[:, 2] > 0
    xs, ys = points[visible, 0], points[visible, 1]
    box = [max(0.0, float(xs.min()) - BOX_PAD_PX), max(0.0, float(ys.min()) - BOX_PAD_PX),
           min(float(width), float(xs.max()) + BOX_PAD_PX), min(float(height), float(ys.max()) + BOX_PAD_PX)]
    keypoints = np.zeros((N, 3), dtype=np.float32)
    keypoints[:, :2] = points[:, :2]
    keypoints[:, 2] = np.where(visible, 2.0, 0.0)
    return {
        "boxes": torch.tensor([box], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
        "keypoints": torch.from_numpy(keypoints[None]),
    }


def image_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def augment(image: np.ndarray, points: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Photometric jitter and a small scale change. Never a flip: it would swap left and right."""
    scale = rng.uniform(0.9, 1.1)
    height, width = image.shape[:2]
    image = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))))
    points = points.copy()
    points[:, :2] *= scale
    image = cv2.convertScaleAbs(image, alpha=rng.uniform(0.8, 1.2), beta=rng.uniform(-20, 20))
    return image, points


class LabelledFrames(torch.utils.data.Dataset):
    def __init__(self, samples: list[Sample], sequences: dict[str, Path], augment: bool = False, seed: int = 0):
        self.samples = list(samples)
        self.sequences = sequences
        self.augment = augment
        self._rng = random.Random(seed)
        self._clips: dict[str, Clip] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _clip(self, clip_id: str) -> Clip:
        if clip_id not in self._clips:
            self._clips[clip_id] = Clip(self.sequences[clip_id])
        return self._clips[clip_id]

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = self._clip(sample.clip_id).read_frame(sample.frame)
        points = sample.points
        if self.augment:
            image, points = augment(image, points, self._rng)
        height, width = image.shape[:2]
        return image_tensor(image), target_for(points, width, height)


def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
