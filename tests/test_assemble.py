import numpy as np
import pytest

from tracker.schemas import H_COLUMNS, SCHEMAS, from_rows
from tracker.stages.assemble import SMOOTH_WINDOW, assemble_tracking, position_conf

IDENTITY = dict(zip(H_COLUMNS, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]))
HALF = dict(zip(H_COLUMNS, [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]))      # feet = pixels / 2


def tracks(rows):
    return from_rows("tracks", [
        {"clip_id": "c", "frame": f, "track_id": t, "class": "player", "x1": x, "y1": 0.0, "x2": x + 10.0, "y2": 20.0, "conf": 0.8}
        for f, t, x in rows
    ])


def frames(n, fps=30.0, play=None, segment=None):
    play = set(range(n)) if play is None else play
    return from_rows("frames", [{"clip_id": "c", "frame": f, "time_s": f / fps, "is_play": f in play,
                                 "segment_id": (segment(f) if segment else 0) if f in play else None} for f in range(n)])


def transforms(rows):
    return from_rows("rink_transforms", [
        {"clip_id": "c", "frame": f, **(h if h is not None else {k: None for k in H_COLUMNS}),
         "n_inliers": n, "reproj_error_px": e, "status": s} for f, h, n, e, s in rows
    ])


def test_each_track_row_becomes_a_tracking_row_with_honest_blanks():
    table = assemble_tracking(tracks([(0, 4, 0.0), (1, 4, 0.0)]), frames(2), transforms([]))
    assert table.columns == list(SCHEMAS["tracking"])
    assert table.to_dicts()[1] == {
        "clip_id": "c", "frame": 1, "time_s": 1 / 30, "entity": "player",
        "player_key": "trk4", "team": "unknown", "jersey_number": None,
        "identity_conf": 0.0, "x_ft": None, "y_ft": None,
        "position_status": "missing", "position_conf": 0.0,
        "possessor_key": None, "possession_conf": None, "possession_source": None,
    }


def test_time_comes_from_the_frames_table():
    assert assemble_tracking(tracks([(60, 1, 0.0)]), frames(61, fps=60.0), transforms([]))["time_s"][0] == 1.0
    assert assemble_tracking(tracks([(60, 1, 0.0)]), frames(61, fps=30.0), transforms([]))["time_s"][0] == 2.0


def test_positions_are_the_box_foot_through_the_transform_with_status_and_conf():
    table = assemble_tracking(
        tracks([(0, 1, 10.0), (1, 1, 10.0), (2, 1, 10.0)]), frames(3),
        transforms([(0, HALF, 8, 2.0, "fitted"), (1, HALF, 0, None, "interpolated"), (2, None, 2, None, "invalid")]),
        smooth_window=1,
    )
    rows = table.to_dicts()
    assert (rows[0]["x_ft"], rows[0]["y_ft"]) == (7.5, 10.0)                       # foot (15, 20) halved
    assert rows[0]["position_status"] == "observed" and rows[0]["position_conf"] == pytest.approx(0.8)
    assert rows[1]["position_status"] == "interpolated" and rows[1]["position_conf"] == 0.5
    assert rows[2]["position_status"] == "missing" and rows[2]["x_ft"] is None and rows[2]["position_conf"] == 0.0


def test_position_conf_is_bounded():
    assert position_conf("fitted", 8, 0.0) == 1.0 and position_conf("fitted", 16, 0.0) == 1.0
    assert position_conf("fitted", 4, 5.0) == pytest.approx(0.25) and position_conf("fitted", 4, 12.0) == 0.0
    assert position_conf("interpolated", 0, None) == 0.5 and position_conf("invalid", 9, 1.0) == 0.0


def test_smoothing_runs_within_a_player_and_segment_and_never_fills_a_gap():
    """Review Focus 2 (assemble side): a missing frame stays missing and splits the smoothing run."""
    xs = [0.0, 10.0, 0.0, 10.0, 0.0, 10.0, 0.0]
    rows = [(f, 1, x) for f, x in enumerate(xs)]
    fitted = [(f, IDENTITY, 8, 1.0, "fitted") for f in range(7) if f != 3]
    table = assemble_tracking(tracks(rows), frames(7), transforms(fitted), smooth_window=3)
    out = table.sort("frame")
    assert out["position_status"][3] == "missing" and out["x_ft"][3] is None
    # frame 1 averages frames 0-2 (feet 5, 15, 5 -> 8.33); frame 2 averages frames 1-2 only (the gap at 3 ends the run)
    assert out["x_ft"][1] == pytest.approx((5 + 15 + 5) / 3) and out["x_ft"][2] == pytest.approx((15 + 5) / 2)
    assert out["x_ft"][4] == pytest.approx((5 + 15) / 2)
    # another player in the same frames is smoothed on its own
    two = assemble_tracking(tracks(rows + [(f, 2, 100.0) for f in range(7)]), frames(7), transforms(fitted), smooth_window=3)
    assert two.filter(two["player_key"] == "trk2")["x_ft"].to_list()[:2] == [105.0, 105.0]
    # a segment boundary ends the run too
    seg = assemble_tracking(tracks(rows), frames(7, segment=lambda f: 0 if f < 2 else 1), transforms(fitted), smooth_window=3)
    assert seg.sort("frame")["x_ft"][1] == pytest.approx((5 + 15) / 2)


def test_rows_are_ordered_by_frame_then_player_and_empty_input_is_valid():
    table = assemble_tracking(tracks([(1, 2, 0.0), (0, 9, 0.0), (0, 3, 0.0)]), frames(2), transforms([]))
    assert list(zip(table["frame"], table["player_key"])) == [(0, "trk3"), (0, "trk9"), (1, "trk2")]
    empty = assemble_tracking(tracks([]), frames(2), transforms([]))
    assert empty.height == 0 and empty.columns == list(SCHEMAS["tracking"])
