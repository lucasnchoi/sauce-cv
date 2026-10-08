import numpy as np
import polars as pl
import pytest

from tracker.rink import template_points
from tracker.schemas import H_COLUMNS, SCHEMAS, from_rows
from tracker.stages.rink_transform import (
    DEFAULT_RINK_TRANSFORM, RinkTransformSettings, fit_frame, fit_transforms, invert, project, transforms_by_frame,
)

# A camera-like feet -> pixel homography: 6 px/ft along, foreshortened across, slight perspective.
H_TRUE = np.array([[6.0, 0.5, 640.0], [0.0, -4.0, 400.0], [0.0, 0.0005, 1.0]])
FEET = template_points()
VISIBLE = np.array([0, 2, 4, 6, 8, 10, 11, 13, 15, 17, 19, 23, 27, 29, 31])     # one end plus the neutral zone


def to_px(feet, H=H_TRUE):
    return project(H, feet)


def test_project_and_invert_round_trip():
    px = to_px(FEET[VISIBLE])
    back = project(invert(H_TRUE), px)
    assert np.allclose(back, FEET[VISIBLE], atol=1e-6)
    assert invert(H_TRUE)[2, 2] == pytest.approx(1.0)


def test_fit_recovers_a_known_homography_despite_noise_and_outliers():
    rng = np.random.default_rng(0)
    px = to_px(FEET[VISIBLE]) + rng.normal(0, 1.0, (len(VISIBLE), 2))
    px[0] += (80.0, -60.0)                                   # two gross outliers
    px[5] += (-70.0, 50.0)
    H, n_inliers, error = fit_frame(px, VISIBLE, DEFAULT_RINK_TRANSFORM)
    assert H is not None and H[2, 2] == 1.0
    assert n_inliers >= len(VISIBLE) - 2 and error < 3.0
    feet = project(H, to_px(FEET[VISIBLE]))
    assert np.abs(feet - FEET[VISIBLE]).max() < 0.5          # within half a foot everywhere


def test_too_few_or_collinear_points_are_invalid():
    """Review Focus 1."""
    three = VISIBLE[:3]
    assert fit_frame(to_px(FEET[three]), three, DEFAULT_RINK_TRANSFORM) == (None, 0, None)
    line = np.array([11, 15, 17, 13])                         # all on y = 22 ft: collinear
    H, n, _ = fit_frame(to_px(FEET[line]), line, DEFAULT_RINK_TRANSFORM)
    assert H is None


def test_a_bad_fit_is_invalid_but_keeps_its_numbers():
    rng = np.random.default_rng(1)
    px = to_px(FEET[VISIBLE]) + rng.normal(0, 25.0, (len(VISIBLE), 2))     # far too noisy
    H, n_inliers, error = fit_frame(px, VISIBLE, RinkTransformSettings(ransac_px=40.0, max_reproj_px=10.0))
    assert H is None and n_inliers >= 4 and error is not None and error >= 10.0


def keypoints_for(frames_points: dict[int, np.ndarray | None], ids=VISIBLE):
    rows = []
    for frame, px in frames_points.items():
        if px is None:
            continue
        for k, (x, y) in zip(ids, px):
            rows.append({"clip_id": "c", "frame": frame, "keypoint_id": int(k), "x_px": float(x), "y_px": float(y), "conf": 0.9})
    return from_rows("keypoints", rows)


def frames_table(n, play):
    return from_rows("frames", [{"clip_id": "c", "frame": f, "time_s": f / 10.0, "is_play": f in play,
                                 "segment_id": 0 if f in play else None} for f in range(n)])


def test_fit_transforms_writes_play_frames_only_and_interpolates_short_gaps():
    """Review Focus 2: gaps of 0.5 s or more stay invalid."""
    good = to_px(FEET[VISIBLE])
    none = {k: None for k in range(20)}
    pts = {**none, 2: good, 3: good, 4: good, 7: good, 8: good, 15: good, 16: good, 17: good}
    # play frames 2..17 in one segment; gap 5-6 (2 frames < 5): interpolated; gap 9-14 (6 frames): invalid
    table = fit_transforms(keypoints_for(pts), frames_table(20, set(range(2, 18))), fps=10.0)
    assert table.columns == list(SCHEMAS["rink_transforms"])
    assert table["frame"].to_list() == list(range(2, 18))
    status = dict(zip(table["frame"], table["status"]))
    assert status[2] == "fitted" and status[5] == "interpolated" and status[6] == "interpolated"
    assert all(status[f] == "invalid" for f in range(9, 15)) and status[17] == "fitted"
    inter = table.filter(pl.col("frame") == 5).row(0, named=True)
    fitted = table.filter(pl.col("frame") == 4).row(0, named=True)
    assert inter["n_inliers"] == 0 and inter["reproj_error_px"] is None and inter["h22"] == 1.0
    assert fitted["n_inliers"] >= 13 and fitted["reproj_error_px"] < 1.0
    assert all(table.filter(pl.col("status") == "invalid")[h].null_count() == 6 for h in H_COLUMNS)
    by_frame = transforms_by_frame(table)
    assert set(by_frame) == {2, 3, 4, 5, 6, 7, 8, 15, 16, 17} and by_frame[5][1] == "interpolated"


def test_smoothing_averages_fitted_frames_within_a_segment_only():
    rng = np.random.default_rng(2)
    pts = {f: to_px(FEET[VISIBLE]) + rng.normal(0, 2.0, (len(VISIBLE), 2)) for f in range(10)}
    frames = from_rows("frames", [{"clip_id": "c", "frame": f, "time_s": f / 10.0, "is_play": True, "segment_id": 0 if f < 5 else 1} for f in range(10)])
    raw = fit_transforms(keypoints_for(pts), frames, fps=10.0, settings=RinkTransformSettings(smooth_window=1))
    smooth = fit_transforms(keypoints_for(pts), frames, fps=10.0, settings=RinkTransformSettings(smooth_window=5))
    raw_h = raw.select(H_COLUMNS).to_numpy()
    smooth_h = smooth.select(H_COLUMNS).to_numpy()
    assert np.std(smooth_h[:5], axis=0).sum() < np.std(raw_h[:5], axis=0).sum()       # smoother within segment 0
    assert np.allclose(smooth_h[4], np.mean(raw_h[2:5], axis=0))                        # frame 4 averages frames 2-4 only: 5-9 are another segment


def test_invalid_frames_never_get_a_guess_and_a_non_play_clip_is_empty():
    table = fit_transforms(keypoints_for({0: None}), frames_table(5, set()), fps=10.0)
    assert table.height == 0 and table.columns == list(SCHEMAS["rink_transforms"])
    lone = fit_transforms(keypoints_for({3: to_px(FEET[VISIBLE[:3]])}, ids=VISIBLE[:3]), frames_table(5, {3}), fps=10.0)
    assert lone["status"].to_list() == ["invalid"] and lone["n_inliers"].to_list() == [0]
