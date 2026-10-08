"""Stage 9: build the tracking table.

The only place pixels become rink positions: the box's foot through the
frame's homography. A frame with no valid transform keeps its row with the
position missing; smoothing runs only across consecutive observed or
interpolated frames of one player within one segment, so it never fills a
gap. Team, jersey number and possession stay unknown until their stages.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from tracker.schemas import H_COLUMNS, validate
from tracker.stages.rink_transform import project

SMOOTH_WINDOW = 5


def position_conf(status: str, n_inliers: int, reproj_error_px: float | None, max_reproj_px: float = 10.0) -> float:
    if status == "fitted":
        error = 0.0 if reproj_error_px is None else float(reproj_error_px)
        return float(min(1.0, n_inliers / 8) * max(0.0, 1.0 - error / max_reproj_px))
    if status == "interpolated":
        return 0.5
    return 0.0


def _positions(tracks: pl.DataFrame, rink_transforms: pl.DataFrame) -> pl.DataFrame:
    """x_ft, y_ft, position_status, position_conf per track row, before smoothing."""
    by_frame = {}
    for row in rink_transforms.filter(pl.col("status") != "invalid").iter_rows(named=True):
        by_frame[int(row["frame"])] = (np.array([row[h] for h in H_COLUMNS], dtype=float).reshape(3, 3),
                                       row["status"], int(row["n_inliers"]), row["reproj_error_px"])
    xs, ys, statuses, confs = [], [], [], []
    for row in tracks.iter_rows(named=True):
        hit = by_frame.get(int(row["frame"]))
        if hit is None:
            xs.append(None); ys.append(None); statuses.append("missing"); confs.append(0.0)
            continue
        H, status, n_inliers, error = hit
        feet = project(H, np.array([[(row["x1"] + row["x2"]) / 2, row["y2"]]]))[0]
        xs.append(float(feet[0])); ys.append(float(feet[1]))
        statuses.append("observed" if status == "fitted" else "interpolated")
        confs.append(position_conf(status, n_inliers, error))
    return pl.DataFrame({"x_ft": pl.Series(xs, dtype=pl.Float64), "y_ft": pl.Series(ys, dtype=pl.Float64),
                         "position_status": pl.Series(statuses, dtype=pl.String), "position_conf": pl.Series(confs, dtype=pl.Float64)})


def assemble_tracking(tracks: pl.DataFrame, frames: pl.DataFrame, rink_transforms: pl.DataFrame,
                      smooth_window: int = SMOOTH_WINDOW) -> pl.DataFrame:
    tracks = validate(tracks, "tracks").sort("frame", "track_id")
    frames = validate(frames, "frames")
    rink_transforms = validate(rink_transforms, "rink_transforms")
    base = tracks.join(frames.select("frame", "time_s", "segment_id"), on="frame", how="left", maintain_order="left")
    if base["time_s"].null_count():
        missing = base.filter(pl.col("time_s").is_null())["frame"].unique().to_list()
        raise ValueError(f"tracks reference frames absent from the frames table: {missing[:5]}")
    table = pl.concat([base, _positions(tracks, rink_transforms)], how="horizontal")
    table = table.with_columns(pl.format("trk{}", pl.col("track_id")).alias("player_key"))
    # Smoothing runs: consecutive non-missing frames of one player within one segment.
    table = table.sort("player_key", "frame").with_columns(
        ((pl.col("frame").diff().over("player_key") != 1) | (pl.col("position_status") == "missing")
         | (pl.col("position_status").shift(1).over("player_key") == "missing")
         | (pl.col("segment_id") != pl.col("segment_id").shift(1).over("player_key"))
         ).fill_null(True).cast(pl.Int64).cum_sum().over("player_key").alias("_run")
    )
    window = max(1, int(smooth_window))
    table = table.with_columns(
        pl.when(pl.col("position_status") != "missing")
        .then(pl.col("x_ft").rolling_mean(window_size=window, center=True, min_samples=1).over(["player_key", "_run"]))
        .otherwise(None).alias("x_ft"),
        pl.when(pl.col("position_status") != "missing")
        .then(pl.col("y_ft").rolling_mean(window_size=window, center=True, min_samples=1).over(["player_key", "_run"]))
        .otherwise(None).alias("y_ft"),
    )
    out = table.select(
        pl.col("clip_id"), pl.col("frame"), pl.col("time_s"),
        pl.col("class").alias("entity"), pl.col("player_key"),
        pl.lit("unknown").alias("team"),
        pl.lit(None, dtype=pl.Int64).alias("jersey_number"),
        pl.lit(0.0).alias("identity_conf"),
        pl.col("x_ft"), pl.col("y_ft"), pl.col("position_status"), pl.col("position_conf"),
        pl.lit(None, dtype=pl.String).alias("possessor_key"),
        pl.lit(None, dtype=pl.Float64).alias("possession_conf"),
        pl.lit(None, dtype=pl.String).alias("possession_source"),
    ).sort("frame", "player_key")
    return validate(out, "tracking")
