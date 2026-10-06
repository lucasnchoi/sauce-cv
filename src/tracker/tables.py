"""Reads and writes pipeline tables as Parquet, validating on the way in and out."""
from __future__ import annotations

from pathlib import Path

import polars as pl

from tracker.schemas import validate


def write_table(df: pl.DataFrame, table: str, path: Path) -> None:
    validated = validate(df, table)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validated.write_parquet(path)


def read_table(table: str, path: Path) -> pl.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"table '{table}' not found at {path}; run the stage that writes it first"
        )
    return validate(pl.read_parquet(path), table)
