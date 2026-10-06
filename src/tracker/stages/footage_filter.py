"""Footage filter.

Stand-in for the keypoint rule in the spec: until rink keypoints exist, every
frame counts as main-camera play, in a single segment.
"""
from __future__ import annotations

import polars as pl

from tracker.schemas import from_rows, validate
from tracker.video import ClipInfo


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
