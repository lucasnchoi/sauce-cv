import numpy as np
import polars as pl
import pytest

from tracker.analysis.box_offset import (
    IOU_BINS, box_agreement, nearest_pairs, summarize, threshold_sweep,
)
from tracker.schemas import from_rows


def tracks(rows, conf=1.0):
    """rows: (frame, track_id, (x1, y1, x2, y2))."""
    return from_rows("tracks", [
        {"clip_id": "c", "frame": f, "track_id": t, "class": "player",
         "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "conf": conf}
        for f, t, b in rows
    ])


def dets(rows):
    """rows: (frame, (x1, y1, x2, y2))."""
    return from_rows("detections", [
        {"clip_id": "c", "frame": f, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "conf": 0.9}
        for f, b in rows
    ])


A = (0.0, 0.0, 10.0, 20.0)          # width 10, height 20, bottom 20
A_SHIFTED = (2.0, 4.0, 12.0, 24.0)  # same size, centre +2,+4, bottom 24
A_SMALLER = (0.0, 0.0, 5.0, 10.0)   # half the size, same top-left
FAR = (500.0, 500.0, 510.0, 520.0)


def test_nearest_pairs_report_offsets_against_the_best_ground_truth_box():
    pairs = nearest_pairs(tracks([(0, 1, A_SHIFTED)]), tracks([(0, 1, A), (0, 2, FAR)]))
    row = pairs.row(0, named=True)
    # intersection: x 2..10 (8) * y 4..20 (16) = 128; union = 200 + 200 - 128 = 272
    assert row["best_iou"] == pytest.approx(128 / 272)
    assert (row["gt_x1"], row["gt_y1"], row["gt_x2"], row["gt_y2"]) == A
    assert row["dx_centre"] == pytest.approx(2.0) and row["dy_centre"] == pytest.approx(4.0)
    assert row["d_bottom"] == pytest.approx(-4.0)            # gt bottom 20 - pred bottom 24
    assert row["width_ratio"] == pytest.approx(1.0) and row["height_ratio"] == pytest.approx(1.0)
    assert row["pred_area"] == pytest.approx(200.0)


def test_size_ratios_are_pred_over_gt():
    row = nearest_pairs(tracks([(0, 1, A_SMALLER)]), tracks([(0, 1, A)])).row(0, named=True)
    assert row["best_iou"] == pytest.approx(50 / 200)
    assert row["width_ratio"] == pytest.approx(0.5) and row["height_ratio"] == pytest.approx(0.5)


def test_a_prediction_with_no_ground_truth_anywhere_is_zero_iou_with_null_deltas():
    """Review Focus 3: no gt in the frame at all, and gt far away."""
    pairs = nearest_pairs(tracks([(0, 1, A), (1, 1, A)]), tracks([(1, 9, FAR)]))
    assert pairs["best_iou"].to_list() == [0.0, 0.0]
    assert pairs["gt_x1"].to_list() == [None, 500.0]
    assert pairs["dx_centre"].null_count() == 2


def test_no_predictions_gives_an_empty_frame_and_an_honest_summary():
    pairs = nearest_pairs(tracks([]), tracks([(0, 1, A)]))
    assert pairs.height == 0
    summary = summarize(pairs)
    assert summary["n_pred"] == 0
    assert summary["bin_share"] == {"<0.1": 0.0, "0.1-0.3": 0.0, "0.3-0.5": 0.0, ">=0.5": 0.0}
    assert summary["matched_medians"]["d_bottom"] is None


def test_summary_bins_and_medians():
    pairs = nearest_pairs(
        tracks([(0, 1, A), (0, 2, A_SHIFTED), (0, 3, A_SMALLER), (0, 4, FAR)]),
        tracks([(0, 1, A)]),
    )
    summary = summarize(pairs)
    assert summary["n_pred"] == 4
    assert IOU_BINS == [0.0, 0.1, 0.3, 0.5, 1.0]
    # ious: 1.0, 128/272=0.47, 0.25, 0.0 -> one in each bin
    assert summary["bin_share"] == {"<0.1": 0.25, "0.1-0.3": 0.25, "0.3-0.5": 0.25, ">=0.5": 0.25}
    # matched (>=0.3): A and A_SHIFTED -> d_bottom 0 and -4 -> median -2
    assert summary["matched_medians"]["d_bottom"] == pytest.approx(-2.0)
    assert set(summary["iou_by_area_tercile"]) == {"small", "medium", "large"}


def test_box_agreement_measures_identical_and_shifted_boxes():
    agreement = box_agreement(tracks([(0, 1, A), (1, 1, A_SHIFTED)]), dets([(0, A), (1, A)]))
    assert agreement["n"] == 2
    assert agreement["share_identical"] == 0.5
    assert agreement["median_iou"] == pytest.approx((1.0 + 128 / 272) / 2)


def test_threshold_sweep_counts_the_shifted_box_only_below_half():
    gt = tracks([(f, 1, A) for f in range(2)])
    pred = tracks([(f, 7, A_SHIFTED) for f in range(2)])       # IoU 0.47
    sweep = threshold_sweep(gt, pred)
    assert sweep[0.3].tp == 2 and sweep[0.4].tp == 2 and sweep[0.5].tp == 0
    assert sweep[0.3].idf1 == 1.0 and sweep[0.5].idf1 == 0.0
