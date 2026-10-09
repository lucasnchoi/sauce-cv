import polars as pl
import pytest

from tracker.schemas import H_COLUMNS, SCHEMAS, from_rows
from tracker.stages.footage_filter import (
    DEFAULT_FOOTAGE_FILTER,
    MODES,
    FootageFilterSettings,
    frames_for,
    keypoint_frames,
    passthrough_frames,
    recorded_settings,
    transform_frames,
)
from tracker.video import ClipInfo

INFO = ClipInfo(clip_id="c", fps=10.0, n_frames=40, width=64, height=48)


def keypoints(play_frames, n_points=6, conf=0.9):
    return from_rows("keypoints", [
        {"clip_id": "c", "frame": f, "keypoint_id": k, "x_px": 1.0, "y_px": 1.0, "conf": conf}
        for f in play_frames for k in range(n_points)
    ])


def test_defaults_match_the_spec():
    assert DEFAULT_FOOTAGE_FILTER == FootageFilterSettings(min_keypoints=6, min_conf=0.5, bridge_gap_s=0.5, min_segment_s=1.0)
    assert FootageFilterSettings.from_dict({"min_keypoints": "8"}).min_keypoints == 8


def test_a_frame_needs_enough_confident_keypoints():
    table = keypoint_frames(pl.concat([keypoints(range(0, 20)), keypoints(range(20, 40), n_points=5)]), INFO)
    assert table.columns == list(SCHEMAS["frames"]) and table.height == 40
    assert table["is_play"].to_list() == [True] * 20 + [False] * 20
    assert table["segment_id"].to_list() == [0] * 20 + [None] * 20
    assert table["time_s"][10] == 1.0
    low = keypoint_frames(keypoints(range(0, 40), conf=0.4), INFO)
    assert not low["is_play"].any() and low["segment_id"].null_count() == 40


def test_short_gaps_are_bridged_and_long_gaps_split_segments():
    # play 0-14, gap 15-18 (4 frames < 5 = 0.5 s at 10 fps): bridged; play 19-24; gap 25-29 (5 frames): not bridged; play 30-39
    table = keypoint_frames(keypoints(list(range(0, 15)) + list(range(19, 25)) + list(range(30, 40))), INFO)
    assert table["is_play"].to_list() == [True] * 25 + [False] * 5 + [True] * 10
    assert table["segment_id"].to_list() == [0] * 25 + [None] * 5 + [1] * 10


def test_segments_shorter_than_a_second_are_dropped():
    # play 0-8 (9 frames < 10): dropped; play 20-39 (20 frames): kept as segment 0
    table = keypoint_frames(keypoints(list(range(0, 9)) + list(range(20, 40))), INFO)
    assert table["is_play"].to_list() == [False] * 20 + [True] * 20
    assert table["segment_id"].to_list() == [None] * 20 + [0] * 20


def test_no_keypoints_at_all_gives_all_non_play():
    table = keypoint_frames(from_rows("keypoints", []), INFO)
    assert table.height == 40 and not table["is_play"].any()


def test_settings_change_the_rule():
    strict = FootageFilterSettings(min_keypoints=7)
    assert not keypoint_frames(keypoints(range(0, 40)), INFO, strict)["is_play"].any()
    no_drop = FootageFilterSettings(min_segment_s=0.0)
    assert keypoint_frames(keypoints(range(0, 3)), INFO, no_drop)["is_play"].to_list()[:4] == [True, True, True, False]


def fits(fitted_frames, n=INFO.n_frames):
    """A rink_fits table: `fitted` on the listed frames (identity H), `invalid` elsewhere."""
    identity = dict(zip(H_COLUMNS, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]))
    blank = {h: None for h in H_COLUMNS}
    return from_rows("rink_fits", [
        {"clip_id": "c", "frame": f, **(identity if f in fitted_frames else blank),
         "n_inliers": 8 if f in fitted_frames else 0, "reproj_error_px": 1.0 if f in fitted_frames else None,
         "status": "fitted" if f in fitted_frames else "invalid"}
        for f in range(n)
    ])


def test_transform_mode_calls_a_frame_play_when_its_fit_is_valid():
    table = transform_frames(fits(set(range(0, 20))), INFO)
    assert table.columns == list(SCHEMAS["frames"]) and table.height == 40
    assert table["is_play"].to_list() == [True] * 20 + [False] * 20
    assert table["segment_id"].to_list() == [0] * 20 + [None] * 20
    assert table["time_s"][10] == 1.0
    assert not transform_frames(fits(set()), INFO)["is_play"].any()


def test_transform_mode_drops_short_segments_and_bridges_short_gaps():
    """Review Focus 1: one good frame among bad ones is not a segment; a short gap inside play is."""
    lone = transform_frames(fits({5} | set(range(20, 40))), INFO)
    assert lone["is_play"].to_list() == [False] * 20 + [True] * 20
    # play 0-14, gap 15-18 (4 frames < 5): bridged; gap 25-29 (5 frames): split
    bridged = transform_frames(fits(set(range(0, 15)) | set(range(19, 25)) | set(range(30, 40))), INFO)
    assert bridged["is_play"].to_list() == [True] * 25 + [False] * 5 + [True] * 10
    assert bridged["segment_id"].to_list() == [0] * 25 + [None] * 5 + [1] * 10


def test_transform_mode_never_bridges_a_leading_or_trailing_stretch():
    """Review Focus 2."""
    table = transform_frames(fits(set(range(2, 38))), INFO)
    assert table["is_play"].to_list() == [False, False] + [True] * 36 + [False, False]


def test_transform_mode_uses_the_bridge_and_segment_settings():
    no_rules = FootageFilterSettings(bridge_gap_s=0.0, min_segment_s=0.0)
    table = transform_frames(fits({3, 5}), INFO, no_rules)
    assert table["is_play"].to_list()[:7] == [False, False, False, True, False, True, False]
    assert table["segment_id"].to_list()[:7] == [None, None, None, 0, None, 1, None]


def test_frames_for_dispatches_on_the_mode():
    assert MODES == ("transform", "keypoints", "passthrough")
    assert frames_for("passthrough", INFO, DEFAULT_FOOTAGE_FILTER)["is_play"].all()
    assert frames_for("transform", INFO, DEFAULT_FOOTAGE_FILTER, fits=fits(set(range(0, 20))))["is_play"].sum() == 20
    assert frames_for("keypoints", INFO, DEFAULT_FOOTAGE_FILTER, keypoints=keypoints(range(0, 20)))["is_play"].sum() == 20
    with pytest.raises(ValueError, match="mode"):
        frames_for("magic", INFO, DEFAULT_FOOTAGE_FILTER)
    with pytest.raises(ValueError, match="rink_fits"):
        frames_for("transform", INFO, DEFAULT_FOOTAGE_FILTER)
    with pytest.raises(ValueError, match="keypoints"):
        frames_for("keypoints", INFO, DEFAULT_FOOTAGE_FILTER)


def test_recorded_settings_depend_on_the_mode():
    s = FootageFilterSettings(min_keypoints=7, bridge_gap_s=1.0)
    assert recorded_settings("keypoints", s) == {"min_keypoints": 7, "min_conf": 0.5, "bridge_gap_s": 1.0, "min_segment_s": 1.0}
    assert recorded_settings("transform", s) == {"bridge_gap_s": 1.0, "min_segment_s": 1.0}
    assert recorded_settings("passthrough", s) is None
