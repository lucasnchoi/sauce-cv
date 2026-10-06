import pytest

from tracker.schemas import SCHEMAS, from_rows
from tracker.stages.assemble import assemble_tracking


def tracks(rows):
    return from_rows("tracks", [
        {"clip_id": "c", "frame": f, "track_id": t, "class": "player",
         "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 20.0, "conf": 0.8}
        for f, t in rows
    ])


def test_each_track_row_becomes_a_tracking_row_with_honest_blanks():
    table = assemble_tracking(tracks([(0, 4), (1, 4)]), fps=30.0)
    assert table.columns == list(SCHEMAS["tracking"])
    assert table.to_dicts()[1] == {
        "clip_id": "c", "frame": 1, "time_s": 1 / 30, "entity": "player",
        "player_key": "trk4", "team": "unknown", "jersey_number": None,
        "identity_conf": 0.0, "x_ft": None, "y_ft": None,
        "position_status": "missing", "position_conf": 0.0,
        "possessor_key": None, "possession_conf": None, "possession_source": None,
    }


def test_time_uses_the_clip_frame_rate():
    assert assemble_tracking(tracks([(60, 1)]), fps=60.0)["time_s"][0] == 1.0
    assert assemble_tracking(tracks([(60, 1)]), fps=30.0)["time_s"][0] == 2.0


def test_rows_are_ordered_by_frame_then_player():
    table = assemble_tracking(tracks([(1, 2), (0, 9), (0, 3)]), fps=30.0)
    assert list(zip(table["frame"], table["player_key"])) == [(0, "trk3"), (0, "trk9"), (1, "trk2")]


def test_no_tracks_gives_an_empty_valid_table():
    table = assemble_tracking(tracks([]), fps=30.0)
    assert table.height == 0
    assert table.columns == list(SCHEMAS["tracking"])


def test_a_frame_rate_of_zero_is_an_error():
    with pytest.raises(ValueError, match="frame rate"):
        assemble_tracking(tracks([(0, 1)]), fps=0.0)
