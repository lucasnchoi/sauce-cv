import numpy as np
import polars as pl
import pytest

from tracker.labels.rink_keypoints import frame_rows, load_labels, upsert_frame
from tracker.labels.session import LabelSession
from tracker.rink import KEYPOINTS, KEYPOINT_BY_NAME
from tracker.schemas import from_rows

FRAMES = [("A_001", 3), ("A_001", 9), ("B_001", 0)]


def fresh():
    return LabelSession(FRAMES, load_labels("/nonexistent/labels.parquet"))


def test_starts_at_the_first_frame_and_keypoint_zero():
    s = fresh()
    assert s.current == ("A_001", 3) and s.keypoint == 0 and s.progress == (0, 3)


def test_place_sets_the_current_keypoint_and_advances_to_the_next_unset():
    s = fresh()
    s.place(100.0, 200.0)
    assert s.current_points == {0: (100.0, 200.0)} and s.keypoint == 1
    s.select(10)
    s.mark_not_visible()
    assert s.current_points[10] is None and s.keypoint == 11
    s.select(10); s.place(5.0, 6.0)                        # re-placing overwrites
    assert s.current_points[10] == (5.0, 6.0)


def test_advance_skips_set_keypoints_and_wraps():
    s = fresh()
    for k in range(33):
        s.select(k); s.place(1.0 * k, 2.0)
    assert len(s.current_points) == 33
    s.select(5); s.clear_current()
    assert 5 not in s.current_points
    s.select(30); s.place(0.0, 0.0)                        # advancing from 30 wraps to the only unset id, 5
    assert s.keypoint == 5


def test_undo_reverts_the_last_edit_in_this_frame():
    s = fresh()
    s.place(1.0, 1.0); s.place(2.0, 2.0)
    s.undo()
    assert s.current_points == {0: (1.0, 1.0)} and s.keypoint == 1
    s.undo(); s.undo()                                     # extra undo is harmless
    assert s.current_points == {}


def test_frame_navigation_is_clamped_and_resets_the_keypoint():
    s = fresh()
    s.place(1.0, 1.0)
    s.next_frame()
    assert s.current == ("A_001", 9) and s.keypoint == 0
    s.prev_frame()
    assert s.current == ("A_001", 3) and s.keypoint == 1        # first unset id in this frame
    s.prev_frame()
    assert s.current == ("A_001", 3)
    s.next_frame(); s.next_frame(); s.next_frame()
    assert s.current == ("B_001", 0)


def test_frame_table_upserts_edited_frames_only_and_counts_progress():
    existing = upsert_frame(load_labels("/nonexistent"), "B_001", 0, frame_rows("B_001", 0, {10: (7.0, 8.0)}))
    s = LabelSession(FRAMES, existing)
    assert s.is_labelled(("B_001", 0)) and s.progress == (1, 3)
    s.place(1.0, 2.0)                                      # A_001 frame 3, keypoint 0
    table = s.frame_table()
    assert table.height == 66
    a = table.filter((pl.col("clip_id") == "A_001") & (pl.col("frame") == 3))
    assert a.filter(pl.col("visible"))["keypoint_id"].to_list() == [0]
    assert a.filter(pl.col("keypoint_id") == 1)["visible"].to_list() == [False]       # Review Focus 1: unset is recorded
    b = table.filter(pl.col("clip_id") == "B_001")
    assert b.filter(pl.col("visible"))["x_px"].to_list() == [7.0]                     # untouched frame kept
    assert s.progress == (2, 3)


def test_prelabels_fill_only_unlabelled_frames_with_the_best_row_per_keypoint():
    s = fresh()
    s.place(1.0, 1.0)                                      # A_001 frame 3 is edited: must not be overwritten
    pre = from_rows("keypoints", [
        {"clip_id": "A_001", "frame": 3, "keypoint_id": 0, "x_px": 99.0, "y_px": 99.0, "conf": 0.9},
        {"clip_id": "A_001", "frame": 9, "keypoint_id": 4, "x_px": 10.0, "y_px": 20.0, "conf": 0.6},
        {"clip_id": "A_001", "frame": 9, "keypoint_id": 4, "x_px": 11.0, "y_px": 21.0, "conf": 0.8},
        {"clip_id": "A_001", "frame": 9, "keypoint_id": 8, "x_px": 1.0, "y_px": 1.0, "conf": 0.2},   # below min_conf
    ])
    s.load_prelabels(pre, min_conf=0.5)
    assert s.points[("A_001", 3)] == {0: (1.0, 1.0)}
    assert s.points[("A_001", 9)] == {4: (11.0, 21.0)}
    assert not s.is_labelled(("A_001", 9))                 # prelabels are suggestions until the owner saves the frame


def test_render_draws_points_and_legend_highlights_the_current_keypoint():
    s = fresh()
    s.place(32.0, 24.0)
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    out = s.render(image)
    assert out.shape == image.shape and image.sum() == 0            # input untouched
    assert out[24, 32].tolist() != [0, 0, 0]
    legend = s.legend()
    assert legend.ndim == 3 and legend.shape[2] == 3


def test_placing_the_last_unset_keypoint_keeps_the_cursor_there():
    s = fresh()
    for k in range(33):
        s.select(k); s.place(1.0 * k, 2.0)
    assert s.keypoint == 32


def test_undoing_the_only_edit_leaves_the_frame_unlabelled():
    s = fresh()
    s.place(1.0, 1.0)
    s.undo()
    assert not s.is_labelled(("A_001", 3))
    assert s.progress == (0, 3)
    assert s.frame_table().height == 0


def test_prelabels_fill_an_untouched_starting_frame():
    s = fresh()
    pre = from_rows("keypoints", [
        {"clip_id": "A_001", "frame": 3, "keypoint_id": 0, "x_px": 12.0, "y_px": 34.0, "conf": 0.9},
    ])
    s.load_prelabels(pre)
    assert s.points[("A_001", 3)] == {0: (12.0, 34.0)}
    assert not s.is_labelled(("A_001", 3))
