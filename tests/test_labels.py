import json

import polars as pl
import pytest

from tracker.labels.rink_keypoints import (
    build_frame_list, frame_rows, labelled_frames, load_frame_list, load_labels, sample_frames,
    save_frame_list, save_labels, upsert_frame,
)
from tracker.rink import KEYPOINTS
from tracker.schemas import SCHEMAS


def test_frame_rows_cover_all_33_keypoints_with_absent_ones_not_visible():
    rows = frame_rows("c", 4, {10: (640.0, 360.0), 8: None})
    assert len(rows) == 33 and [r["keypoint_id"] for r in rows] == list(range(33))
    by_id = {r["keypoint_id"]: r for r in rows}
    assert by_id[10] == {"clip_id": "c", "frame": 4, "keypoint_id": 10, "x_px": 640.0, "y_px": 360.0, "visible": True}
    assert by_id[8]["visible"] is False and by_id[8]["x_px"] is None
    assert by_id[0]["visible"] is False


def test_a_frame_with_nothing_visible_is_still_recorded(tmp_path):
    """Review Focus 1."""
    labels = upsert_frame(load_labels(tmp_path / "none.parquet"), "c", 7, frame_rows("c", 7, {}))
    save_labels(labels, tmp_path / "labels.parquet")
    again = load_labels(tmp_path / "labels.parquet")
    assert again.height == 33 and not again["visible"].any()
    assert labelled_frames(again) == {("c", 7)}


def test_upsert_replaces_a_frame_without_duplicating_it(tmp_path):
    """Review Focus 3."""
    labels = upsert_frame(load_labels(tmp_path / "x"), "c", 1, frame_rows("c", 1, {10: (1.0, 2.0)}))
    labels = upsert_frame(labels, "c", 2, frame_rows("c", 2, {10: (3.0, 4.0)}))
    labels = upsert_frame(labels, "c", 1, frame_rows("c", 1, {10: (5.0, 6.0), 27: (7.0, 8.0)}))
    assert labels.height == 66
    f1 = labels.filter((pl.col("frame") == 1) & pl.col("visible"))
    assert f1["keypoint_id"].to_list() == [10, 27] and f1["x_px"].to_list() == [5.0, 7.0]


def test_save_rejects_visible_without_pixels_or_outside_the_image_and_writes_nothing(tmp_path):
    """Review Focus 2."""
    rows = frame_rows("c", 0, {10: (640.0, 360.0)})
    bad = [dict(r, x_px=None) if r["keypoint_id"] == 10 else r for r in rows]
    with pytest.raises(ValueError, match=r"c.*frame 0.*keypoint 10"):
        save_labels(upsert_frame(load_labels(tmp_path / "x"), "c", 0, bad), tmp_path / "labels.parquet")
    assert not (tmp_path / "labels.parquet").exists()
    outside = frame_rows("c", 0, {10: (1280.0, 10.0)})
    with pytest.raises(ValueError, match="outside"):
        save_labels(upsert_frame(load_labels(tmp_path / "x"), "c", 0, outside), tmp_path / "labels.parquet")


def test_save_rejects_a_frame_with_a_missing_keypoint_row(tmp_path):
    rows = frame_rows("c", 0, {})[:-1]
    labels = pl.DataFrame(rows, schema=SCHEMAS["rink_keypoint_labels"])
    with pytest.raises(ValueError, match="33"):
        save_labels(labels, tmp_path / "labels.parquet")


def test_sample_frames_is_deterministic_spread_and_capped():
    a = sample_frames(1000, 10, seed=1, clip_id="A_001")
    assert a == sample_frames(1000, 10, seed=1, clip_id="A_001")
    assert a != sample_frames(1000, 10, seed=1, clip_id="A_002")
    assert a == sorted(a) and len(a) == 10
    assert all(100 * i <= f < 100 * (i + 1) for i, f in enumerate(a))     # one per tenth
    assert sample_frames(3, 10, seed=1, clip_id="x") == [0, 1, 2]


def test_build_frame_list_uses_training_clips_for_training_and_heldout_for_evaluation(make_sequence, tmp_path):
    root = tmp_path / "ds"
    for name in ("A_001", "A_002", "B_001"):
        make_sequence(name=name, n_frames=12, gt_rows=[], parent=root / "mot-challenge-format" / "train")
    from tracker.datasets.mot import list_sequences
    sequences = list_sequences(root)
    split = {"seed": 1, "heldout": ["B_001"], "train": ["A_001", "A_002"]}
    data = build_frame_list(sequences, split, per_clip_train=3, per_clip_eval=2, seed=5)
    assert {c for c, _ in data["training"]} == {"A_001", "A_002"} and len(data["training"]) == 6
    assert {c for c, _ in data["evaluation"]} == {"B_001"} and len(data["evaluation"]) == 2
    assert all(0 <= f < 12 for _, f in data["training"] + data["evaluation"])
    path = tmp_path / "labels" / "rink_frames.json"
    save_frame_list(data, path)
    assert load_frame_list(path) == data and json.loads(path.read_text())["seed"] == 5
