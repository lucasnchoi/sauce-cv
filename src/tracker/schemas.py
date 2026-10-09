"""Table definitions: the contracts between pipeline stages.

Every stage validates its input tables before running and its output tables
before writing. Frame indices are 0-based. Pixel coordinates have the origin
at the top-left of the image, x to the right and y down.
"""
from __future__ import annotations

import polars as pl

from tracker.rink import KEYPOINT_IDS


class SchemaError(ValueError):
    """A table does not match its definition."""


FRAMES = {
    "clip_id": pl.String,
    "frame": pl.Int64,
    "time_s": pl.Float64,
    "is_play": pl.Boolean,
    "segment_id": pl.Int64,
}

TRACKS = {
    "clip_id": pl.String,
    "frame": pl.Int64,
    "track_id": pl.Int64,
    "class": pl.String,
    "x1": pl.Float64,
    "y1": pl.Float64,
    "x2": pl.Float64,
    "y2": pl.Float64,
    "conf": pl.Float64,
}

DETECTIONS = {
    "clip_id": pl.String,
    "frame": pl.Int64,
    "x1": pl.Float64,
    "y1": pl.Float64,
    "x2": pl.Float64,
    "y2": pl.Float64,
    "conf": pl.Float64,
}

TRACKING = {
    "clip_id": pl.String,
    "frame": pl.Int64,
    "time_s": pl.Float64,
    "entity": pl.String,
    "player_key": pl.String,
    "team": pl.String,
    "jersey_number": pl.Int64,
    "identity_conf": pl.Float64,
    "x_ft": pl.Float64,
    "y_ft": pl.Float64,
    "position_status": pl.String,
    "position_conf": pl.Float64,
    "possessor_key": pl.String,
    "possession_conf": pl.Float64,
    "possession_source": pl.String,
}

KEYPOINTS = {
    "clip_id": pl.String,
    "frame": pl.Int64,
    "keypoint_id": pl.Int64,
    "x_px": pl.Float64,
    "y_px": pl.Float64,
    "conf": pl.Float64,
}

RINK_KEYPOINT_LABELS = {
    "clip_id": pl.String,
    "frame": pl.Int64,
    "keypoint_id": pl.Int64,
    "x_px": pl.Float64,
    "y_px": pl.Float64,
    "visible": pl.Boolean,
}

H_COLUMNS = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]

RINK_TRANSFORMS = {
    "clip_id": pl.String,
    "frame": pl.Int64,
    **{h: pl.Float64 for h in H_COLUMNS},
    "n_inliers": pl.Int64,
    "reproj_error_px": pl.Float64,
    "status": pl.String,
}

SCHEMAS = {
    "frames": FRAMES,
    "detections": DETECTIONS,
    "tracks": TRACKS,
    "tracking": TRACKING,
    "keypoints": KEYPOINTS,
    "rink_keypoint_labels": RINK_KEYPOINT_LABELS,
    "rink_transforms": RINK_TRANSFORMS,
    "rink_fits": RINK_TRANSFORMS,
}

# Columns that may never be null.
REQUIRED = {
    "frames": ["clip_id", "frame", "time_s", "is_play"],
    "detections": list(DETECTIONS),
    "tracks": list(TRACKS),
    "tracking": ["clip_id", "frame", "time_s", "entity", "position_status", "position_conf"],
    "keypoints": list(KEYPOINTS),
    "rink_keypoint_labels": ["clip_id", "frame", "keypoint_id", "visible"],
    "rink_transforms": ["clip_id", "frame", "n_inliers", "status"],
    "rink_fits": ["clip_id", "frame", "n_inliers", "status"],
}

# Columns restricted to a fixed set of values. Nulls are checked by REQUIRED.
ALLOWED_VALUES = {
    "tracks": {"class": {"player", "goalie", "referee"}},
    "tracking": {
        "entity": {"player", "goalie", "referee", "puck"},
        "team": {"team_a", "team_b", "official", "unknown"},
        "position_status": {"observed", "interpolated", "missing"},
        "possession_source": {"observed", "inferred"},
    },
    "keypoints": {"keypoint_id": set(KEYPOINT_IDS)},
    "rink_keypoint_labels": {"keypoint_id": set(KEYPOINT_IDS)},
    "rink_transforms": {"status": {"fitted", "interpolated", "invalid"}},
    "rink_fits": {"status": {"fitted", "invalid"}},
}


def _schema(table: str) -> dict:
    if table not in SCHEMAS:
        raise SchemaError(f"unknown table '{table}'; known tables: {sorted(SCHEMAS)}")
    return SCHEMAS[table]


def validate(df: pl.DataFrame, table: str) -> pl.DataFrame:
    """Check a table against its definition and return it in schema column order."""
    schema = _schema(table)
    missing = [c for c in schema if c not in df.columns]
    extra = [c for c in df.columns if c not in schema]
    if missing or extra:
        raise SchemaError(
            f"table '{table}': missing columns {missing}, unexpected columns {extra}"
        )
    wrong = [
        f"{c} is {df.schema[c]}, expected {dtype}"
        for c, dtype in schema.items()
        if df.schema[c] != dtype
    ]
    if wrong:
        raise SchemaError(f"table '{table}': wrong column types: {'; '.join(wrong)}")
    for column in REQUIRED[table]:
        nulls = df[column].null_count()
        if nulls:
            raise SchemaError(f"table '{table}': column '{column}' has {nulls} null values")
        if df.schema[column] == pl.Float64:
            nans = df[column].is_nan().sum()
            if nans:
                raise SchemaError(f"table '{table}': column '{column}' has {nans} NaN values")
    for column, allowed in ALLOWED_VALUES.get(table, {}).items():
        bad = (
            df.filter(pl.col(column).is_not_null() & ~pl.col(column).is_in(sorted(allowed)))
            [column].unique().to_list()
        )
        if bad:
            raise SchemaError(
                f"table '{table}': column '{column}' has values {sorted(bad)}; "
                f"allowed: {sorted(allowed)}"
            )
    return df.select(list(schema))


def empty(table: str) -> pl.DataFrame:
    """A table with the right columns and no rows."""
    return pl.DataFrame(schema=_schema(table))


def from_rows(table: str, rows: list[dict]) -> pl.DataFrame:
    """Build a table from dictionaries. Columns absent from a row become null."""
    schema = _schema(table)
    for row in rows:
        unknown = sorted(set(row) - set(schema))
        if unknown:
            raise SchemaError(f"table '{table}': unknown columns in row: {unknown}")
    if not rows:
        return empty(table)
    columns = {name: [row.get(name) for row in rows] for name in schema}
    return pl.DataFrame(columns, schema=schema, strict=False)
