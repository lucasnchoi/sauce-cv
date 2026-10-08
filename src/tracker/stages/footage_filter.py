"""Stage 2: footage filter.

The spec's rule, all four numbers settings: a frame is main-camera play when
enough rink keypoints are confident; runs of play frames form segments; short
gaps are bridged; short segments are dropped. The Plan 1 pass-through stays
for comparison (`--footage-filter passthrough`).
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import polars as pl

from tracker.schemas import from_rows, validate
from tracker.video import ClipInfo


@dataclass(frozen=True)
class FootageFilterSettings:
    min_keypoints: int = 6
    min_conf: float = 0.5
    bridge_gap_s: float = 0.5
    min_segment_s: float = 1.0

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, values: dict) -> "FootageFilterSettings":
        known = {f.name: f.type for f in fields(cls)}
        unknown = sorted(set(values) - set(known))
        if unknown:
            raise ValueError(f"unknown footage filter settings {unknown}; known: {sorted(known)}")
        casts = {"int": int, "float": float}
        return cls(**{key: casts[known[key]](value) for key, value in values.items()})


DEFAULT_FOOTAGE_FILTER = FootageFilterSettings()


def passthrough_frames(info: ClipInfo) -> pl.DataFrame:
    rows = [
        {
            "clip_id": info.clip_id,
            "frame": frame,
            "time_s": frame / info.fps,
            "is_play": True,
            "segment_id": 0,
        }
        for frame in range(info.n_frames)
    ]
    return validate(from_rows("frames", rows), "frames")


def _runs(flags: list[bool], value: bool) -> list[tuple[int, int]]:
    """(start, stop) of every maximal run of `value`, stop exclusive."""
    runs, start = [], None
    for i, flag in enumerate(flags + [not value]):
        if flag == value and start is None:
            start = i
        elif flag != value and start is not None:
            runs.append((start, i))
            start = None
    return runs


def keypoint_frames(keypoints: pl.DataFrame, info: ClipInfo, settings: FootageFilterSettings = DEFAULT_FOOTAGE_FILTER) -> pl.DataFrame:
    keypoints = validate(keypoints, "keypoints")
    counts = (keypoints.filter(pl.col("conf") >= settings.min_conf)
              .group_by("frame").agg(pl.len().alias("n")))
    confident = {int(f): int(n) for f, n in counts.iter_rows()}
    play = [confident.get(frame, 0) >= settings.min_keypoints for frame in range(info.n_frames)]
    bridge = round(settings.bridge_gap_s * info.fps)
    for start, stop in _runs(play, False):
        inside = start > 0 and stop < info.n_frames        # a gap, not a leading or trailing stretch
        if inside and stop - start < bridge:
            play[start:stop] = [True] * (stop - start)
    shortest = round(settings.min_segment_s * info.fps)
    for start, stop in _runs(play, True):
        if stop - start < shortest:
            play[start:stop] = [False] * (stop - start)
    segment: list[int | None] = [None] * info.n_frames
    for index, (start, stop) in enumerate(_runs(play, True)):
        segment[start:stop] = [index] * (stop - start)
    rows = [{"clip_id": info.clip_id, "frame": frame, "time_s": frame / info.fps, "is_play": play[frame], "segment_id": segment[frame]}
            for frame in range(info.n_frames)]
    return validate(from_rows("frames", rows), "frames")
