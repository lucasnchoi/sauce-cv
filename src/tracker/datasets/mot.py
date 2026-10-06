"""Loads tracking ground truth stored in the layout VIP-HTD uses.

Each annotation row is: frame,id,x,y,w,h,conf,class,vis with 1-based frames
and boxes given as a top-left corner plus width and height.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from tracker.schemas import empty, validate

# Named here because polars numbers headerless columns from 0 in 2.x and from 1 in 1.x.
GT_COLUMNS = ["frame", "id", "x", "y", "w", "h", "conf", "class", "vis"]


def load_mot_ground_truth(seq_dir: Path) -> pl.DataFrame:
    seq_dir = Path(seq_dir)
    gt_path = seq_dir / "gt" / "gt.txt"
    if not gt_path.is_file():
        raise FileNotFoundError(f"sequence '{seq_dir.name}': no ground truth at {gt_path}")
    try:
        raw = pl.read_csv(gt_path, has_header=False, infer_schema_length=None)
    except pl.exceptions.NoDataError:
        return empty("tracks")
    raw = raw.rename(dict(zip(raw.columns, GT_COLUMNS)))
    if raw.width < 6:
        raise ValueError(
            f"sequence '{seq_dir.name}': expected at least 6 columns in {gt_path}, found {raw.width}"
        )
    x, y, w, h = (pl.col(c).cast(pl.Float64) for c in ("x", "y", "w", "h"))
    keep = (w > 0) & (h > 0)
    if raw.width >= 7:
        keep = keep & (pl.col("conf").cast(pl.Float64) != 0)   # conf 0 means ignore
    table = raw.filter(keep).select(
        pl.lit(seq_dir.name).alias("clip_id"),
        (pl.col("frame").cast(pl.Int64) - 1).alias("frame"),
        pl.col("id").cast(pl.Int64).alias("track_id"),
        pl.lit("player").alias("class"),
        x.alias("x1"),
        y.alias("y1"),
        (x + w).alias("x2"),
        (y + h).alias("y2"),
        pl.lit(1.0).alias("conf"),
    )
    return validate(table.sort("frame", "track_id"), "tracks")


def list_sequences(dataset_root: Path, annotation: str = "mot-challenge-format") -> dict[str, Path]:
    """Map clip id to sequence folder for every annotated sequence in the dataset."""
    base = Path(dataset_root) / annotation
    if not base.is_dir():
        raise FileNotFoundError(f"no '{annotation}' folder under {dataset_root}")
    found: dict[str, Path] = {}
    for gt_path in sorted(base.glob("*/*/gt/gt.txt")):
        seq_dir = gt_path.parent.parent
        if seq_dir.name in found:
            raise ValueError(
                f"sequence name '{seq_dir.name}' appears twice: {found[seq_dir.name]} and {seq_dir}"
            )
        found[seq_dir.name] = seq_dir
    return found
