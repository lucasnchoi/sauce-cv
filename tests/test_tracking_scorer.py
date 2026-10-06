import numpy as np
import pytest

from tracker.evaluate.tracking import TrackingCounts, iou_matrix, score_tracking
from tracker.schemas import from_rows

BOX_A = (0.0, 0.0, 10.0, 10.0)
BOX_B = (100.0, 100.0, 110.0, 110.0)
FAR = (500.0, 500.0, 510.0, 510.0)


def tracks(rows):
    """rows: (frame, track_id, box)."""
    return from_rows("tracks", [
        {"clip_id": "c", "frame": f, "track_id": t, "class": "player",
         "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "conf": 1.0}
        for f, t, b in rows
    ])


def test_iou_of_identical_boxes_is_one():
    a = np.array([BOX_A])
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)


def test_iou_of_half_overlapping_boxes():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[5.0, 0.0, 15.0, 10.0]])       # intersection 50, union 150
    assert iou_matrix(a, b)[0, 0] == pytest.approx(1 / 3)


def test_iou_of_disjoint_boxes_is_zero():
    assert iou_matrix(np.array([BOX_A]), np.array([BOX_B]))[0, 0] == 0.0


def test_iou_with_a_zero_area_box_is_zero_not_an_error():
    flat = np.array([[5.0, 5.0, 5.0, 5.0]])
    assert iou_matrix(flat, flat)[0, 0] == 0.0


def test_iou_with_no_boxes_has_the_right_shape():
    assert iou_matrix(np.zeros((0, 4)), np.array([BOX_A])).shape == (0, 1)


def test_perfect_tracking_scores_one():
    gt = tracks([(f, 1, BOX_A) for f in range(4)])
    pred = tracks([(f, 7, BOX_A) for f in range(4)])
    counts = score_tracking(gt, pred)
    assert counts == TrackingCounts(idtp=4, n_gt=4, n_pred=4, tp=4)
    assert counts.idf1 == 1.0
    assert counts.precision == 1.0 and counts.recall == 1.0


def test_one_identity_switch_halves_idf1_but_not_detection():
    # One real player for 4 frames; the tracker changes its number halfway.
    gt = tracks([(f, 1, BOX_A) for f in range(4)])
    pred = tracks([(0, 10, BOX_A), (1, 10, BOX_A), (2, 11, BOX_A), (3, 11, BOX_A)])
    counts = score_tracking(gt, pred)
    assert counts.idtp == 2                       # best single pairing covers 2 frames
    assert counts.idf1 == pytest.approx(2 * 2 / (4 + 4))   # 0.5
    assert counts.tp == 4 and counts.fp == 0 and counts.fn == 0


def test_two_players_swapping_numbers_halfway():
    gt = tracks([(f, 1, BOX_A) for f in range(4)] + [(f, 2, BOX_B) for f in range(4)])
    pred = tracks(
        [(0, 10, BOX_A), (1, 10, BOX_A), (2, 10, BOX_B), (3, 10, BOX_B)]
        + [(0, 11, BOX_B), (1, 11, BOX_B), (2, 11, BOX_A), (3, 11, BOX_A)]
    )
    counts = score_tracking(gt, pred)
    assert counts.idtp == 4                       # 2 frames for each pairing
    assert counts.idf1 == pytest.approx(2 * 4 / (8 + 8))   # 0.5
    assert counts.tp == 8


def test_missed_frames_lower_recall():
    gt = tracks([(f, 1, BOX_A) for f in range(4)])
    pred = tracks([(f, 7, BOX_A) for f in range(3)])
    counts = score_tracking(gt, pred)
    assert counts.idf1 == pytest.approx(2 * 3 / (4 + 3))   # 6/7
    assert counts.recall == pytest.approx(3 / 4)
    assert counts.precision == 1.0
    assert counts.fn == 1


def test_a_false_track_lowers_precision():
    gt = tracks([(f, 1, BOX_A) for f in range(4)])
    pred = tracks([(f, 7, BOX_A) for f in range(4)] + [(f, 8, FAR) for f in range(4)])
    counts = score_tracking(gt, pred)
    assert counts.idf1 == pytest.approx(2 * 4 / (4 + 8))   # 2/3
    assert counts.precision == pytest.approx(4 / 8)
    assert counts.fp == 4


def test_overlap_below_the_threshold_does_not_count():
    gt = tracks([(f, 1, (0.0, 0.0, 10.0, 10.0)) for f in range(4)])
    pred = tracks([(f, 7, (5.0, 0.0, 15.0, 10.0)) for f in range(4)])   # IoU 1/3
    counts = score_tracking(gt, pred)
    assert counts.idtp == 0 and counts.tp == 0
    assert counts.idf1 == 0.0


def test_one_box_cannot_match_two_ground_truth_boxes_in_a_frame():
    near_a = (1.0, 0.0, 11.0, 10.0)               # overlaps BOX_A strongly
    gt = tracks([(0, 1, BOX_A), (0, 2, near_a)])
    pred = tracks([(0, 7, BOX_A)])
    counts = score_tracking(gt, pred)
    assert counts.tp == 1 and counts.fn == 1


def test_no_predictions_scores_zero_without_error():
    gt = tracks([(f, 1, BOX_A) for f in range(4)])
    counts = score_tracking(gt, tracks([]))
    assert counts == TrackingCounts(idtp=0, n_gt=4, n_pred=0, tp=0)
    assert counts.idf1 == 0.0 and counts.precision == 0.0 and counts.recall == 0.0


def test_no_ground_truth_is_an_error():
    with pytest.raises(ValueError, match="no ground truth"):
        score_tracking(tracks([]), tracks([(0, 7, BOX_A)]))


def test_a_track_id_used_twice_in_one_frame_is_an_error():
    gt = tracks([(0, 1, BOX_A)])
    pred = tracks([(0, 7, BOX_A), (0, 7, BOX_B)])
    with pytest.raises(ValueError, match="more than once"):
        score_tracking(gt, pred)


def test_counts_add_up_across_clips():
    a = TrackingCounts(idtp=2, n_gt=4, n_pred=4, tp=4)
    b = TrackingCounts(idtp=6, n_gt=6, n_pred=6, tp=6)
    total = a + b
    assert total == TrackingCounts(idtp=8, n_gt=10, n_pred=10, tp=10)
    assert total.idf1 == pytest.approx(2 * 8 / 20)   # pooled, not the mean of 0.5 and 1.0
