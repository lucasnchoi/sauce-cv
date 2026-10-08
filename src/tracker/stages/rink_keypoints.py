"""Stage 1: rink keypoints on every frame, from the project's own model."""
from __future__ import annotations

import numpy as np
import polars as pl

from tracker.schemas import from_rows, validate


def keypoint_rows(points: np.ndarray, clip_id: str, frame: int) -> list[dict]:
    return [
        {"clip_id": clip_id, "frame": int(frame), "keypoint_id": int(k),
         "x_px": float(points[k, 0]), "y_px": float(points[k, 1]), "conf": float(points[k, 2])}
        for k in range(points.shape[0]) if points[k, 2] > 0
    ]


def run_keypoints(clip, detector) -> pl.DataFrame:
    """Every frame: stage 1 runs before the footage filter, which needs these points."""
    clip_id = clip.info.clip_id
    rows: list[dict] = []
    for index, image in clip.frames():
        rows.extend(keypoint_rows(detector.detect(image), clip_id, index))
    return validate(from_rows("keypoints", rows), "keypoints")


def keypoints_for_frames(clip, frames: list[int], detector) -> pl.DataFrame:
    """Listed frames only, in order: pre-labelling reads a few frames per clip."""
    clip_id = clip.info.clip_id
    rows: list[dict] = []
    for frame in sorted(set(int(f) for f in frames)):
        rows.extend(keypoint_rows(detector.detect(clip.read_frame(frame)), clip_id, frame))
    return validate(from_rows("keypoints", rows), "keypoints")
