import pytest

from tracker.datasets.mot import list_sequences, load_mot_ground_truth
from tracker.schemas import SCHEMAS


def test_rows_become_zero_based_frames_and_corner_boxes(make_sequence):
    seq = make_sequence(name="GAME_A_001", gt_rows=[(1, 35, 289, 237, 99, 106), (2, 35, 290, 237, 99, 106)])
    df = load_mot_ground_truth(seq)
    assert df.columns == list(SCHEMAS["tracks"])
    assert df.to_dicts()[0] == {
        "clip_id": "GAME_A_001", "frame": 0, "track_id": 35, "class": "player",
        "x1": 289.0, "y1": 237.0, "x2": 388.0, "y2": 343.0, "conf": 1.0,
    }
    assert df["frame"].to_list() == [0, 1]


def test_degenerate_boxes_are_dropped_and_partial_boxes_kept(make_sequence):
    seq = make_sequence(gt_rows=[
        (1, 1, 10, 10, 0, 20),      # zero width: dropped
        (1, 2, 10, 10, 20, -5),     # negative height: dropped
        (1, 3, -8, 10, 20, 20),     # partly outside the frame: kept
    ])
    df = load_mot_ground_truth(seq)
    assert df["track_id"].to_list() == [3]
    assert df["x1"][0] == -8.0


def test_rows_flagged_to_ignore_are_dropped(make_sequence):
    seq = make_sequence(gt_rows=[(1, 1, 10, 10, 20, 20)])
    with (seq / "gt" / "gt.txt").open("a") as handle:
        handle.write("1,2,30,30,20,20,0,1,1\n")     # conf 0 means ignore
    assert load_mot_ground_truth(seq)["track_id"].to_list() == [1]


def test_a_fractional_value_after_row_100_does_not_break_loading(make_sequence):
    seq = make_sequence(gt_rows=[(f, 1, 10, 10, 20, 30) for f in range(1, 151)])
    with (seq / "gt" / "gt.txt").open("a") as handle:
        handle.write("151,1,10.5,10,20,30,1,1,1\n")    # first fractional value, row 151
    df = load_mot_ground_truth(seq)
    assert df.height == 151
    assert df["x1"][150] == 10.5 and df["x2"][150] == 30.5


def test_empty_annotation_file_gives_an_empty_table(make_sequence):
    seq = make_sequence(gt_rows=[])
    assert load_mot_ground_truth(seq).height == 0


def test_missing_annotation_file_names_the_sequence(make_sequence):
    seq = make_sequence(name="GAME_C_003")
    with pytest.raises(FileNotFoundError, match="GAME_C_003"):
        load_mot_ground_truth(seq)


def test_list_sequences_finds_every_split(make_sequence, tmp_path):
    root = tmp_path / "ds"
    make_sequence(name="GAME_A_001", gt_rows=[], parent=root / "mot-challenge-format" / "train")
    make_sequence(name="GAME_A_002", gt_rows=[], parent=root / "mot-challenge-format" / "test")
    make_sequence(name="GAME_B_001", parent=root / "mot-challenge-format" / "train")  # no gt: skipped
    found = list_sequences(root)
    assert sorted(found) == ["GAME_A_001", "GAME_A_002"]
    assert found["GAME_A_002"].name == "GAME_A_002"


def test_list_sequences_rejects_a_name_used_twice(make_sequence, tmp_path):
    root = tmp_path / "ds"
    make_sequence(name="GAME_A_001", gt_rows=[], parent=root / "mot-challenge-format" / "train")
    make_sequence(name="GAME_A_001", gt_rows=[], parent=root / "mot-challenge-format" / "test")
    with pytest.raises(ValueError, match="GAME_A_001"):
        list_sequences(root)


def test_list_sequences_on_a_missing_dataset_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="mot-challenge-format"):
        list_sequences(tmp_path / "nowhere")
