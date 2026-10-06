import polars as pl
import pytest

from tracker.schemas import SCHEMAS, SchemaError, empty, from_rows, validate

TRACK_ROW = {
    "clip_id": "c1", "frame": 0, "track_id": 1, "class": "player",
    "x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0, "conf": 0.9,
}


def test_valid_table_passes_and_columns_follow_schema_order():
    shuffled = from_rows("tracks", [TRACK_ROW]).select(sorted(SCHEMAS["tracks"]))
    result = validate(shuffled, "tracks")
    assert result.columns == list(SCHEMAS["tracks"])


def test_missing_column_is_named():
    df = from_rows("tracks", [TRACK_ROW]).drop("conf")
    with pytest.raises(SchemaError, match="conf"):
        validate(df, "tracks")


def test_unexpected_column_is_named():
    df = from_rows("tracks", [TRACK_ROW]).with_columns(pl.lit(1).alias("extra"))
    with pytest.raises(SchemaError, match="extra"):
        validate(df, "tracks")


def test_wrong_type_is_named():
    df = from_rows("tracks", [TRACK_ROW]).with_columns(pl.col("frame").cast(pl.Float64))
    with pytest.raises(SchemaError, match="frame"):
        validate(df, "tracks")


def test_value_outside_allowed_set_is_rejected():
    df = from_rows("tracks", [{**TRACK_ROW, "class": "skater"}])
    with pytest.raises(SchemaError, match="skater"):
        validate(df, "tracks")


def test_null_in_required_column_is_rejected():
    df = from_rows("tracks", [{**TRACK_ROW, "clip_id": None}])
    with pytest.raises(SchemaError, match="clip_id"):
        validate(df, "tracks")


def test_unknown_table_is_rejected():
    with pytest.raises(SchemaError, match="nonsense"):
        validate(pl.DataFrame(), "nonsense")


@pytest.mark.parametrize("table", sorted(SCHEMAS))
def test_empty_table_is_valid(table):
    assert validate(empty(table), table).height == 0


def test_from_rows_fills_absent_nullable_columns_with_null():
    row = {
        "clip_id": "c1", "frame": 0, "time_s": 0.0, "entity": "player",
        "team": "unknown", "identity_conf": 0.0,
        "position_status": "missing", "position_conf": 0.0,
    }
    df = validate(from_rows("tracking", [row]), "tracking")
    assert df["x_ft"][0] is None
    assert df["jersey_number"][0] is None


def test_from_rows_rejects_a_misspelt_column():
    with pytest.raises(SchemaError, match="trackid"):
        from_rows("tracks", [{**TRACK_ROW, "trackid": 5}])
