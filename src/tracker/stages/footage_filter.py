"""Stage 2: footage filter.

Three modes. `transform` (default, Plan 2c): a frame is main-camera play when
its own rink transform fitted (stage 3a). `keypoints` (Plan 2's rule): play
when enough rink keypoints are confident. `passthrough` (Plan 1): every frame
is play. In the first two, runs of play frames form segments, short gaps are
bridged and short segments are dropped. The bridge default is 2.0 s, chosen on the
training clips in Plan 2c (`docs/filter-results.md`); Plan 2's rule used 0.5 s.
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
    bridge_gap_s: float = 2.0
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
MODES = ("transform", "keypoints", "passthrough")


def recorded_settings(mode: str, settings: FootageFilterSettings) -> dict | None:
    """What run.json records for the filter: only the numbers the mode uses."""
    if mode == "keypoints":
        return settings.as_dict()
    if mode == "transform":
        return {"bridge_gap_s": settings.bridge_gap_s, "min_segment_s": settings.min_segment_s}
    return None


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


def _frames_from_flags(play: list[bool], info: ClipInfo, settings: FootageFilterSettings) -> pl.DataFrame:
    """Bridge short gaps inside play, drop short segments, number the segments, build the frames table."""
    play = list(play)
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


def keypoint_frames(keypoints: pl.DataFrame, info: ClipInfo, settings: FootageFilterSettings = DEFAULT_FOOTAGE_FILTER) -> pl.DataFrame:
    keypoints = validate(keypoints, "keypoints")
    counts = (keypoints.filter(pl.col("conf") >= settings.min_conf)
              .group_by("frame").agg(pl.len().alias("n")))
    confident = {int(f): int(n) for f, n in counts.iter_rows()}
    play = [confident.get(frame, 0) >= settings.min_keypoints for frame in range(info.n_frames)]
    return _frames_from_flags(play, info, settings)


def transform_frames(fits: pl.DataFrame, info: ClipInfo, settings: FootageFilterSettings = DEFAULT_FOOTAGE_FILTER) -> pl.DataFrame:
    """Play when the frame's own fit is `fitted`: the stage-3 validity gate is the filter's test."""
    fits = validate(fits, "rink_fits")
    fitted = {int(f) for f in fits.filter(pl.col("status") == "fitted")["frame"].to_list()}
    play = [frame in fitted for frame in range(info.n_frames)]
    return _frames_from_flags(play, info, settings)


def frames_for(mode: str, info: ClipInfo, settings: FootageFilterSettings,
               fits: pl.DataFrame | None = None, keypoints: pl.DataFrame | None = None) -> pl.DataFrame:
    if mode not in MODES:
        raise ValueError(f"unknown footage filter mode '{mode}'; use one of {MODES}")
    if mode == "passthrough":
        return passthrough_frames(info)
    if mode == "keypoints":
        if keypoints is None:
            raise ValueError("footage filter mode 'keypoints' needs the keypoints table")
        return keypoint_frames(keypoints, info, settings)
    if fits is None:
        raise ValueError("footage filter mode 'transform' needs the rink_fits table")
    return transform_frames(fits, info, settings)
