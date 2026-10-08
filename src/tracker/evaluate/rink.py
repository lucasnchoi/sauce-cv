"""Scores the rink transform: labelled rink points pushed through each frame's homography, against the template.

Distances are in metres. Labelled frames the footage filter called non-play
are a filter matter, reported separately; labelled play frames with no valid
transform are counted, never skipped silently.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from tracker.rink import template_points
from tracker.schemas import validate
from tracker.stages.rink_transform import project, transforms_by_frame

FT_TO_M = 0.3048


@dataclass(frozen=True)
class RinkCounts:
    distances_m: tuple[float, ...]
    n_play_frames: int
    n_valid_play_frames: int
    n_labelled_frames: int
    n_labelled_scored: int
    n_labelled_no_transform: int
    n_labelled_non_play: int

    @property
    def n_points(self) -> int:
        return len(self.distances_m)

    @property
    def median_m(self) -> float | None:
        return float(np.median(self.distances_m)) if self.distances_m else None

    @property
    def valid_share(self) -> float | None:
        return self.n_valid_play_frames / self.n_play_frames if self.n_play_frames else None

    def __add__(self, other: "RinkCounts") -> "RinkCounts":
        return RinkCounts(
            self.distances_m + other.distances_m,
            self.n_play_frames + other.n_play_frames,
            self.n_valid_play_frames + other.n_valid_play_frames,
            self.n_labelled_frames + other.n_labelled_frames,
            self.n_labelled_scored + other.n_labelled_scored,
            self.n_labelled_no_transform + other.n_labelled_no_transform,
            self.n_labelled_non_play + other.n_labelled_non_play,
        )


def score_rink(labels: pl.DataFrame, frames: pl.DataFrame, rink_transforms: pl.DataFrame) -> RinkCounts:
    labels = validate(labels, "rink_keypoint_labels")
    frames = validate(frames, "frames")
    play = {int(f) for f in frames.filter(pl.col("is_play"))["frame"].to_list()}
    by_frame = transforms_by_frame(rink_transforms)
    template = template_points()
    distances: list[float] = []
    scored = no_transform = non_play = 0
    labelled = labels.partition_by("frame", as_dict=True) if labels.height else {}
    for key, part in labelled.items():
        frame = int(key[0] if isinstance(key, tuple) else key)
        if frame not in play:
            non_play += 1
            continue
        if frame not in by_frame:
            no_transform += 1
            continue
        scored += 1
        H, _ = by_frame[frame]
        visible = part.filter(pl.col("visible"))
        if visible.height:
            feet = project(H, visible.select("x_px", "y_px").to_numpy())
            truth = template[visible["keypoint_id"].to_numpy()]
            distances.extend((np.linalg.norm(feet - truth, axis=1) * FT_TO_M).tolist())
    return RinkCounts(tuple(distances), len(play), sum(1 for f in play if f in by_frame),
                      len(labelled), scored, no_transform, non_play)
