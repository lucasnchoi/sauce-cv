import pytest

from fakes import FakeClip, FakeDetector, FakeTrackerFactory, detections
from tracker.schemas import SCHEMAS, from_rows
from tracker.stages.detect_track import bytetrack_factory, run_detect_track
from tracker.stages.footage_filter import passthrough_frames
from tracker.video import ClipInfo

BOX_A = [10.0, 10.0, 30.0, 40.0]
BOX_B = [50.0, 10.0, 60.0, 40.0]


def frames_table(clip_id, rows, fps=30.0):
    """rows: (frame, is_play, segment_id or None)."""
    return from_rows("frames", [
        {"clip_id": clip_id, "frame": f, "time_s": f / fps, "is_play": play, "segment_id": seg}
        for f, play, seg in rows
    ])


def test_passthrough_marks_every_frame_as_play_in_one_segment():
    info = ClipInfo(clip_id="c", fps=60.0, n_frames=3, width=64, height=48)
    df = passthrough_frames(info)
    assert df.columns == list(SCHEMAS["frames"])
    assert df["frame"].to_list() == [0, 1, 2]
    assert df["time_s"].to_list() == [0.0, 1 / 60, 2 / 60]
    assert df["is_play"].to_list() == [True, True, True]
    assert df["segment_id"].to_list() == [0, 0, 0]


def test_passthrough_for_a_clip_with_no_frames_is_empty():
    info = ClipInfo(clip_id="c", fps=30.0, n_frames=0, width=64, height=48)
    assert passthrough_frames(info).height == 0


def test_every_play_frame_gets_a_row_per_tracked_box():
    clip = FakeClip(clip_id="c", n_frames=3)
    tracks = run_detect_track(
        clip, passthrough_frames(clip.info), FakeDetector([BOX_A, BOX_B]), FakeTrackerFactory()
    )
    assert tracks.columns == list(SCHEMAS["tracks"])
    assert tracks.height == 6
    assert tracks.filter(frame=0).sort("track_id").to_dicts()[0] == {
        "clip_id": "c", "frame": 0, "track_id": 0, "class": "player",
        "x1": 10.0, "y1": 10.0, "x2": 30.0, "y2": 40.0, "conf": 0.9,
    }


def test_frames_that_are_not_play_are_skipped_and_not_sent_to_the_detector():
    clip = FakeClip(n_frames=4)
    frames = frames_table("c", [(0, True, 0), (1, False, None), (2, False, None), (3, True, 1)])
    detector = FakeDetector([BOX_A])
    tracks = run_detect_track(clip, frames, detector, FakeTrackerFactory())
    assert detector.calls == 2
    assert tracks["frame"].to_list() == [0, 3]


def test_a_new_segment_gets_a_fresh_tracker_and_new_track_ids():
    clip = FakeClip(n_frames=4)
    frames = frames_table("c", [(0, True, 0), (1, True, 0), (2, True, 1), (3, True, 1)])
    factory = FakeTrackerFactory()
    tracks = run_detect_track(clip, frames, FakeDetector([BOX_A, BOX_B]), factory)
    assert len(factory.fps_seen) == 2
    first = set(tracks.filter(frame=0)["track_id"])
    second = set(tracks.filter(frame=2)["track_id"])
    assert first == {0, 1}
    assert first.isdisjoint(second)          # tracks never cross a segment boundary


def test_the_tracker_is_given_the_clip_frame_rate():
    for fps in (30.0, 60.0):
        clip = FakeClip(fps=fps, n_frames=2)
        factory = FakeTrackerFactory()
        run_detect_track(clip, passthrough_frames(clip.info), FakeDetector([BOX_A]), factory)
        assert factory.fps_seen == [fps]


def test_unconfirmed_tracks_are_left_out():
    clip = FakeClip(n_frames=2)
    tracks = run_detect_track(
        clip, passthrough_frames(clip.info), FakeDetector([BOX_A, BOX_B]),
        FakeTrackerFactory(unconfirmed={1}),
    )
    assert set(tracks["track_id"]) == {0}


def test_no_detections_gives_an_empty_valid_table():
    clip = FakeClip(n_frames=3)
    tracks = run_detect_track(
        clip, passthrough_frames(clip.info), FakeDetector([]), FakeTrackerFactory()
    )
    assert tracks.height == 0
    assert tracks.columns == list(SCHEMAS["tracks"])


def test_frames_table_for_another_clip_is_rejected():
    clip = FakeClip(clip_id="c", n_frames=2)
    frames = frames_table("other", [(0, True, 0), (1, True, 0)])
    with pytest.raises(ValueError, match="other"):
        run_detect_track(clip, frames, FakeDetector([BOX_A]), FakeTrackerFactory())


def test_real_bytetrack_keeps_one_id_for_a_smoothly_moving_box():
    class MovingBoxDetector:
        name = "moving-box"

        def __init__(self):
            self.frame = 0

        def detect(self, image_bgr):
            x = 100.0 + 5.0 * self.frame
            self.frame += 1
            return detections([[x, 100.0, x + 40.0, 180.0]])

    clip = FakeClip(n_frames=10, width=640, height=360)
    tracks = run_detect_track(
        clip, passthrough_frames(clip.info), MovingBoxDetector(), bytetrack_factory
    )
    assert tracks["track_id"].n_unique() == 1
    assert tracks.height >= 8                 # the first frame is not yet confirmed
