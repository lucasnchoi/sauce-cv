"""Stage 9: build the tracking table.

At this point only tracks exist, so team, jersey number, rink position and
possession are left unknown or missing. Later stages fill them; they are never
guessed here.
"""
from __future__ import annotations

import polars as pl

from tracker.schemas import validate


def assemble_tracking(tracks: pl.DataFrame, fps: float) -> pl.DataFrame:
    tracks = validate(tracks, "tracks")
    if fps <= 0:
        raise ValueError(f"frame rate must be positive, got {fps}")
    table = tracks.select(
        pl.col("clip_id"),
        pl.col("frame"),
        (pl.col("frame") / fps).alias("time_s"),
        pl.col("class").alias("entity"),
        pl.format("trk{}", pl.col("track_id")).alias("player_key"),
        pl.lit("unknown").alias("team"),
        pl.lit(None, dtype=pl.Int64).alias("jersey_number"),
        pl.lit(0.0).alias("identity_conf"),
        pl.lit(None, dtype=pl.Float64).alias("x_ft"),
        pl.lit(None, dtype=pl.Float64).alias("y_ft"),
        pl.lit("missing").alias("position_status"),
        pl.lit(0.0).alias("position_conf"),
        pl.lit(None, dtype=pl.String).alias("possessor_key"),
        pl.lit(None, dtype=pl.Float64).alias("possession_conf"),
        pl.lit(None, dtype=pl.String).alias("possession_source"),
    ).sort("frame", "player_key")
    return validate(table, "tracking")
