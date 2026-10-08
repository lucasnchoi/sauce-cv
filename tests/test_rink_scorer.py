import numpy as np
import pytest

from tracker.evaluate.rink import FT_TO_M, RinkCounts, score_rink
from tracker.labels.rink_keypoints import frame_rows, load_labels, upsert_frame
from tracker.rink import template_points
from tracker.schemas import H_COLUMNS, from_rows
from tracker.stages.rink_transform import invert, project

H_TRUE = np.array([[6.0, 0.5, 640.0], [0.0, -4.0, 400.0], [0.0, 0.0005, 1.0]])    # feet -> px
H_PX2FT = invert(H_TRUE)
FEET = template_points()


def labels_for(frames: dict[int, dict[int, tuple[float, float]]]):
    table = load_labels("/nonexistent")
    for frame, points in frames.items():
        table = upsert_frame(table, "c", frame, frame_rows("c", frame, points))
    return table


def frames_table(n, play):
    return from_rows("frames", [{"clip_id": "c", "frame": f, "time_s": f / 10.0, "is_play": f in play,
                                 "segment_id": 0 if f in play else None} for f in range(n)])


def transforms(rows: dict[int, tuple[np.ndarray | None, str]]):
    out = []
    for frame, (H, status) in rows.items():
        values = {h: None for h in H_COLUMNS} if H is None else dict(zip(H_COLUMNS, H.ravel().tolist()))
        out.append({"clip_id": "c", "frame": frame, **values, "n_inliers": 0 if H is None else 8,
                    "reproj_error_px": None if H is None else 1.0, "status": status})
    return from_rows("rink_transforms", out)


def px_of(keypoint_id, H=H_TRUE):
    return tuple(project(H, FEET[[keypoint_id]])[0].tolist())


def test_distances_are_measured_in_metres_against_the_template():
    labels = labels_for({3: {10: px_of(10), 27: px_of(27)}})
    shifted = H_TRUE.copy(); shifted[0, 2] += 6.0            # the transform is one foot off along the rink
    counts = score_rink(labels, frames_table(5, {3}), transforms({3: (invert(shifted), "fitted")}))
    assert counts.n_points == 2 and counts.median_m == pytest.approx(1.0 * FT_TO_M, abs=0.02)
    assert counts.n_labelled_frames == 1 and counts.n_labelled_scored == 1 and counts.n_labelled_no_transform == 0
    assert counts.n_play_frames == 1 and counts.n_valid_play_frames == 1 and counts.valid_share == 1.0


def test_interpolated_transforms_count_as_valid_and_invalid_ones_as_no_transform():
    labels = labels_for({1: {10: px_of(10)}, 2: {10: px_of(10)}, 4: {}})
    rows = transforms({1: (H_PX2FT, "interpolated"), 2: (None, "invalid"), 3: (H_PX2FT, "fitted"), 4: (H_PX2FT, "fitted")})
    counts = score_rink(labels, frames_table(5, {1, 2, 3, 4}), rows)
    assert counts.n_points == 1 and counts.median_m == pytest.approx(0.0, abs=1e-6)
    assert counts.n_labelled_scored == 2 and counts.n_labelled_no_transform == 1      # frame 4 scored with 0 points
    assert counts.n_play_frames == 4 and counts.n_valid_play_frames == 3 and counts.valid_share == 0.75


def test_a_labelled_frame_the_filter_called_non_play_is_reported_separately():
    """Review Focus 3."""
    labels = labels_for({0: {10: px_of(10)}, 2: {10: px_of(10)}})
    counts = score_rink(labels, frames_table(3, {2}), transforms({2: (H_PX2FT, "fitted")}))
    assert counts.n_labelled_non_play == 1 and counts.n_labelled_scored == 1 and counts.n_points == 1


def test_a_labelled_play_frame_with_no_transform_row_counts_as_no_transform():
    labels = labels_for({2: {10: px_of(10)}})
    counts = score_rink(labels, frames_table(3, {2}), transforms({}))
    assert counts.n_labelled_no_transform == 1 and counts.n_points == 0 and counts.median_m is None


def test_counts_add_and_pool_distances():
    a = RinkCounts((0.1, 0.5), 10, 9, 2, 2, 0, 0)
    b = RinkCounts((0.9,), 10, 6, 1, 1, 0, 1)
    pooled = a + b
    assert pooled.distances_m == (0.1, 0.5, 0.9) and pooled.median_m == 0.5 and pooled.valid_share == 0.75
    assert pooled.n_labelled_non_play == 1 and RinkCounts((), 0, 0, 0, 0, 0, 0).valid_share is None
