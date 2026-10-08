import polars as pl
import pytest

from tracker.schemas import SCHEMAS, from_rows
from tracker.stages.footage_filter import DEFAULT_FOOTAGE_FILTER, FootageFilterSettings, keypoint_frames
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
