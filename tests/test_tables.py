import polars as pl
import pytest

from tracker.schemas import SchemaError, from_rows
from tracker.tables import read_table, write_table

TRACK_ROW = {
    "clip_id": "c1", "frame": 0, "track_id": 1, "class": "player",
    "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0, "conf": 0.9,
}


def test_round_trip_keeps_rows(tmp_path):
    path = tmp_path / "run" / "c1" / "tracks.parquet"
    write_table(from_rows("tracks", [TRACK_ROW]), "tracks", path)
    assert read_table("tracks", path).to_dicts() == [TRACK_ROW]


def test_writing_an_invalid_table_raises_and_writes_nothing(tmp_path):
    path = tmp_path / "tracks.parquet"
    with pytest.raises(SchemaError):
        write_table(from_rows("tracks", [TRACK_ROW]).drop("conf"), "tracks", path)
    assert not path.exists()


def test_reading_a_missing_file_names_the_table(tmp_path):
    with pytest.raises(FileNotFoundError, match="tracks"):
        read_table("tracks", tmp_path / "nope.parquet")


def test_reading_a_file_with_the_wrong_columns_raises(tmp_path):
    path = tmp_path / "tracks.parquet"
    pl.DataFrame({"a": [1]}).write_parquet(path)
    with pytest.raises(SchemaError):
        read_table("tracks", path)
