import pytest

from tracker.analysis.box_offset import (
    IOU_BINS, box_agreement, drift_by_range, nearest_pairs, shifted_share, summarize, threshold_sweep,
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


def test_iou_by_area_tercile_medians():
    # Six predictions of height 10 and widths 2..12 (areas 20..120), each alone in its frame
    # against a ground-truth box of the same height and the width below; with equal heights
    # IoU = min(w, g) / max(w, g).
    widths_and_gt = [(2, 4), (4, 5), (6, 10), (8, 8), (10, 40), (12, 16)]
    # ious: 0.5, 0.8, 0.6, 1.0, 0.25, 0.75
    pred = tracks([(f, 1, (0.0, 0.0, float(w), 10.0)) for f, (w, _) in enumerate(widths_and_gt)])
    gt = tracks([(f, 1, (0.0, 0.0, float(g), 10.0)) for f, (_, g) in enumerate(widths_and_gt)])
    terciles = summarize(nearest_pairs(pred, gt))["iou_by_area_tercile"]
    # area cut points 53.3 and 86.7 -> two boxes per tercile
    assert terciles["small"] == pytest.approx(0.65)     # areas 20, 40: median of 0.5, 0.8
    assert terciles["medium"] == pytest.approx(0.8)     # areas 60, 80: median of 0.6, 1.0
    assert terciles["large"] == pytest.approx(0.5)      # areas 100, 120: median of 0.25, 0.75


def moving(positions):
    """rows: (frame, x position) -> a 10x20 box at that x, one id."""
    return tracks([(f, 1, (float(x), 0.0, float(x) + 10.0, 20.0)) for f, x in positions])


def test_shifted_share_reads_ground_truth_k_frames_earlier():
    # The box moves 5 px per frame. Predictions are frames 1..20 (x = 5f); the ground truth labelled g
    # (frames 0..19) holds the box where the prediction has it one frame later, x = 5(g + 1).
    pred = moving([(f, 5 * f) for f in range(1, 21)])
    gt = moving([(g, 5 * (g + 1)) for g in range(20)])
    # k=1: prediction f against ground truth f-1, x = 5f: identical, IoU 1 for all 20.
    assert shifted_share(pred, gt, 1) == 1.0
    # k=0: ground truth f sits 5 px right: overlap 5 * 20 = 100, union 200 + 200 - 100 = 300, IoU 1/3 < 0.5 (frame 20 has none).
    assert shifted_share(pred, gt, 0) == 0.0
    # k=2 (ground truth f-2 is 5 px left, IoU 1/3) and k=-1 (ground truth f+1 is 10 px right, no overlap).
    assert shifted_share(pred, gt, 2) == 0.0 and shifted_share(pred, gt, -1) == 0.0
    # A lower threshold counts the 1/3 overlaps at k=0: frames 1..19 have ground truth, frame 20 does not -> 19/20.
    assert shifted_share(pred, gt, 0, iou_threshold=0.3) == pytest.approx(19 / 20)


def test_a_prediction_whose_shifted_frame_has_no_ground_truth_is_not_matched():
    # Predictions are frames 0..19 (x = 5f); k=1 reads ground truth f-1, and frame 0 has none: 19 of 20.
    pred = moving([(f, 5 * f) for f in range(20)])
    gt = moving([(g, 5 * (g + 1)) for g in range(20)])
    assert shifted_share(pred, gt, 1) == pytest.approx(19 / 20)
    assert shifted_share(tracks([]), gt, 1) == 0.0


def test_drift_by_range_finds_the_best_shift_in_each_half():
    # Predictions are frames 10..29 (x = 5f), two ranges of 10 frames: (10, 19) and (20, 29).
    pred = moving([(f, 5 * f) for f in range(10, 30)])
    # Ground truth frames 10..19 hold the prediction's own position (aligned, k=0);
    # frames 19..29 hold it one frame later, x = 5(g + 1) (k=1), as a second id so frame 19 has both.
    aligned = [(g, 1, (5.0 * g, 0.0, 5.0 * g + 10.0, 20.0)) for g in range(10, 20)]
    late = [(g, 2, (5.0 * (g + 1), 0.0, 5.0 * (g + 1) + 10.0, 20.0)) for g in range(19, 30)]
    result = drift_by_range(pred, tracks(aligned + late), n_ranges=2)
    assert [r["range"] for r in result] == [(10, 19), (20, 29)]
    # first range: k=0 matches all 10 (share 1.0); any other k is 5+ px off (IoU 1/3 or less) or has no frame
    assert (result[0]["best_k"], result[0]["share_at_best"], result[0]["share_at_zero"]) == (0, 1.0, 1.0)
    # second range: k=1 matches all 10 (prediction f against ground truth f-1 at x = 5f); k=0 is 5 px off, IoU 1/3
    assert (result[1]["best_k"], result[1]["share_at_best"], result[1]["share_at_zero"]) == (1, 1.0, 0.0)
