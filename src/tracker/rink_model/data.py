"""The owner's labels as samples for the heatmap regressor.

Every labelled frame is a sample, including frames with nothing visible: the
target for an invisible keypoint is an all-zero heatmap, which is what
teaches the model absence. Visible keypoints get a Gaussian peak. Frames are
read from the dataset on demand and never copied out of data/raw.
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

N = len(KEYPOINTS)
INPUT_HW = (576, 1024)      # rows, cols: 16:9 like the clips, divisible by 32
STRIDE = 4                  # heatmap cell size in input pixels
SIGMA_CELLS = 2.0
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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
        out.append(Sample(str(clip_id), int(frame), points))
    return out


def split_validation(samples: list[Sample], seed: int, share: float = 0.2) -> tuple[list[Sample], list[Sample]]:
    order = list(samples)
    random.Random(seed).shuffle(order)
    n_val = max(1, round(len(order) * share)) if len(order) > 1 else 0
    key = lambda s: (s.clip_id, s.frame)
    return sorted(order[n_val:], key=key), sorted(order[:n_val], key=key)


def heatmap_targets(points: np.ndarray, width: int, height: int, input_hw=INPUT_HW, stride: int = STRIDE,
                    sigma: float = SIGMA_CELLS) -> np.ndarray:
    rows, cols = input_hw[0] // stride, input_hw[1] // stride
    target = np.zeros((N, rows, cols), dtype=np.float32)
    ys, xs = np.mgrid[0:rows, 0:cols]
    for k in range(N):
        x, y, visible = points[k]
        if visible <= 0:
            continue
        cx, cy = x / width * cols - 0.5, y / height * rows - 0.5
        if not (-0.5 <= cx < cols - 0.5 and -0.5 <= cy < rows - 0.5):
            continue                                            # off the image: absent
        target[k] = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))
    return target


def image_tensor(image_bgr: np.ndarray, input_hw=INPUT_HW) -> torch.Tensor:
    resized = cv2.resize(image_bgr, (input_hw[1], input_hw[0]), interpolation=cv2.INTER_AREA)
    rgb = np.ascontiguousarray(resized[:, :, ::-1]).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


def augment(image: np.ndarray, points: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """A small zoom and shift plus photometric jitter. Never a flip: it would swap left and right."""
    height, width = image.shape[:2]
    scale = rng.uniform(0.9, 1.1)
    tx = (1 - scale) * width / 2 + rng.uniform(-0.05, 0.05) * width
    ty = (1 - scale) * height / 2 + rng.uniform(-0.05, 0.05) * height
    matrix = np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32)
    image = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    points = points.copy()
    xy = points[:, :2] * scale + matrix[:, 2]
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    points[:, :2] = xy
    points[:, 2] = np.where(inside, points[:, 2], 0.0)
    image = cv2.convertScaleAbs(image, alpha=rng.uniform(0.8, 1.2), beta=rng.uniform(-20, 20))
    return image, points


class LabelledFrames(torch.utils.data.Dataset):
    def __init__(self, samples: list[Sample], sequences: dict[str, Path], augment: bool = False, seed: int = 0, input_hw=INPUT_HW):
        self.samples = list(samples)
        self.sequences = sequences
        self.augment = augment
        self.input_hw = tuple(input_hw)
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
        target = heatmap_targets(points, width, height, self.input_hw)
        return image_tensor(image, self.input_hw), torch.from_numpy(target)


def collate(batch):
    images, targets = zip(*batch)
    return torch.stack(list(images)), torch.stack(list(targets))
