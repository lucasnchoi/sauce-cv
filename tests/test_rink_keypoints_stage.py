import numpy as np

from fakes import FakeClip, FakeKeypointDetector
from tracker.schemas import SCHEMAS
from tracker.stages.rink_keypoints import keypoint_rows, keypoints_for_frames, run_keypoints

POINTS = {10: (640.0, 360.0, 0.9), 27: (640.0, 200.0, 0.4)}


def test_rows_skip_keypoints_with_zero_confidence():
    points = np.zeros((33, 3))
    points[10] = (1.0, 2.0, 0.9)
    assert keypoint_rows(points, "c", 4) == [{"clip_id": "c", "frame": 4, "keypoint_id": 10, "x_px": 1.0, "y_px": 2.0, "conf": 0.9}]


def test_stage_runs_every_frame_and_writes_a_valid_table():
    clip = FakeClip(clip_id="c", n_frames=3)
    detector = FakeKeypointDetector(POINTS, per_frame={1: {}})
    table = run_keypoints(clip, detector)
    assert table.columns == list(SCHEMAS["keypoints"]) and detector.calls == 3
    assert table["frame"].to_list() == [0, 0, 2, 2] and table["keypoint_id"].to_list() == [10, 27, 10, 27]


def test_listed_frames_only_are_read_for_prelabels(make_sequence):
    from tracker.video import Clip
    clip = Clip(make_sequence(n_frames=5))
    detector = FakeKeypointDetector(POINTS)
    table = keypoints_for_frames(clip, [4, 1], detector)
    assert detector.calls == 2 and table["frame"].to_list() == [1, 1, 4, 4]
