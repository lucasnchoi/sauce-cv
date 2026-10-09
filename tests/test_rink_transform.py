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


def test_random_pixels_are_invalid():
    rng = np.random.default_rng(5)
    for _ in range(10):
        pixels = rng.uniform([0, 0], [1280, 720], (15, 2))
        assert fit_frame(pixels, VISIBLE, DEFAULT_RINK_TRANSFORM)[0] is None


def test_a_minority_of_correct_points_is_invalid():
    rng = np.random.default_rng(6)
    px = rng.uniform([0, 0], [1280, 720], (len(VISIBLE), 2))
    px[:4] = to_px(FEET[VISIBLE[:4]])
    H, _, _ = fit_frame(px, VISIBLE, DEFAULT_RINK_TRANSFORM)
    assert H is None


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


from tracker.stages.rink_transform import fit_all, smooth_transforms


def test_fit_all_gives_every_frame_a_row_even_without_keypoints():
    """Review Focus 5."""
    good = to_px(FEET[VISIBLE])
    table = fit_all(keypoints_for({1: good, 3: good}), "c", 5)
    assert table.columns == list(SCHEMAS["rink_fits"])
    assert table["frame"].to_list() == [0, 1, 2, 3, 4]
    assert table["status"].to_list() == ["invalid", "fitted", "invalid", "fitted", "invalid"]
    empty_rows = table.filter(pl.col("status") == "invalid")
    assert empty_rows["n_inliers"].to_list() == [0, 0, 0] and all(empty_rows[h].null_count() == 3 for h in H_COLUMNS)
    assert table.filter(pl.col("frame") == 1).row(0, named=True)["h22"] == 1.0
    assert fit_all(from_rows("keypoints", []), "c", 3)["status"].to_list() == ["invalid"] * 3


def test_fit_all_keeps_the_numbers_of_a_bad_fit():
    few = to_px(FEET[VISIBLE[:3]])
    table = fit_all(keypoints_for({0: few}, ids=VISIBLE[:3]), "c", 1)
    assert table["status"].to_list() == ["invalid"] and table["n_inliers"].to_list() == [0]


def test_smooth_transforms_reproduces_fit_transforms():
    good = to_px(FEET[VISIBLE])
    none = {k: None for k in range(20)}
    pts = {**none, 2: good, 3: good, 4: good, 7: good, 8: good, 15: good, 16: good, 17: good}
    frames = frames_table(20, set(range(2, 18)))
    fits = fit_all(keypoints_for(pts), "c", 20)
    assert fits.height == 20
    composed = smooth_transforms(fits, frames, fps=10.0)
    direct = fit_transforms(keypoints_for(pts), frames, fps=10.0)
    assert composed.to_dicts() == direct.to_dicts()
    assert composed["frame"].to_list() == list(range(2, 18))
    status = dict(zip(composed["frame"], composed["status"]))
    assert status[5] == "interpolated" and all(status[f] == "invalid" for f in range(9, 15))


def test_smooth_transforms_needs_a_fit_row_for_every_play_frame():
    fits = fit_all(keypoints_for({0: to_px(FEET[VISIBLE])}), "c", 3)
    with pytest.raises(ValueError, match="play frames"):
        smooth_transforms(fits, frames_table(5, {0, 1, 2, 3, 4}), fps=10.0)


def test_smooth_transforms_on_no_play_frames_is_empty():
    fits = fit_all(keypoints_for({0: to_px(FEET[VISIBLE])}), "c", 3)
    table = smooth_transforms(fits, frames_table(3, set()), fps=10.0)
    assert table.height == 0 and table.columns == list(SCHEMAS["rink_transforms"])
