import numpy as np
import pytest

from fakes import FakeClip, FakeDetector, FakeTrackerFactory, detections
from tracker.schemas import SCHEMAS, from_rows
from tracker.stages.detect_track import bytetrack_factory, run_detect, run_detect_track, run_track
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


def test_non_play_frames_are_detected_but_not_tracked():
    clip = FakeClip(n_frames=4)
    frames = frames_table("c", [(0, True, 0), (1, False, None), (2, False, None), (3, True, 1)])
    detector = FakeDetector([BOX_A])
    tracks = run_detect_track(clip, frames, detector, FakeTrackerFactory())
    assert detector.calls == 4                       # the cache covers every frame (refilter needs it)
    assert tracks["frame"].to_list() == [0, 3]       # tracks exist on play frames only


def test_one_tracker_serves_the_whole_clip_across_segments():
    clip = FakeClip(n_frames=4)
    frames = frames_table("c", [(0, True, 0), (1, True, 0), (2, True, 1), (3, True, 1)])
    factory = FakeTrackerFactory()
    tracks = run_detect_track(clip, frames, FakeDetector([BOX_A, BOX_B]), factory)
    assert len(factory.fps_seen) == 1                # created once, not once per segment
    assert set(tracks.filter(frame=0)["track_id"]) == {0, 1}
    assert set(tracks.filter(frame=2)["track_id"]) == {0, 1}     # the tracker's own ids, no per-segment offset


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


def test_run_detect_writes_one_row_per_box_on_every_frame():
    clip = FakeClip(clip_id="c", n_frames=3)
    dets = run_detect(clip, FakeDetector([BOX_A, BOX_B]))
    assert dets.columns == list(SCHEMAS["detections"])
    assert dets["frame"].to_list() == [0, 0, 1, 1, 2, 2]
    assert dets.filter(frame=1).sort("x1").to_dicts()[0] == {
        "clip_id": "c", "frame": 1, "x1": 10.0, "y1": 10.0, "x2": 30.0, "y2": 40.0, "conf": 0.9,
    }


def test_run_detect_with_nothing_found_is_an_empty_valid_table():
    clip = FakeClip(n_frames=2)
    dets = run_detect(clip, FakeDetector([]))
    assert dets.height == 0 and dets.columns == list(SCHEMAS["detections"])


def test_run_track_reproduces_run_detect_track():
    clip = FakeClip(clip_id="c", n_frames=4)
    frames = frames_table("c", [(0, True, 0), (1, True, 0), (2, True, 1), (3, True, 1)])
    direct = run_detect_track(clip, frames, FakeDetector([BOX_A, BOX_B]), FakeTrackerFactory())
    dets = run_detect(clip, FakeDetector([BOX_A, BOX_B]))
    again = run_track(dets, frames, clip.info.fps, FakeTrackerFactory())
    assert again.to_dicts() == direct.to_dicts()


def test_run_track_updates_the_tracker_on_play_frames_with_no_detections():
    """A frame with nothing detected still reaches the tracker, so lost-track
    bookkeeping advances (Review Focus 1)."""
    class CountingFactory:
        def __init__(self):
            self.updates = []

        def __call__(self, fps):
            factory = self

            class _Tracker:
                def update(self, dets):
                    factory.updates.append(len(dets))
                    dets.tracker_id = np.arange(len(dets))
                    return dets
            return _Tracker()

    clip = FakeClip(n_frames=3)
    frames = passthrough_frames(clip.info)
    dets = from_rows("detections", [
        {"clip_id": "c", "frame": 0, "x1": 1.0, "y1": 1.0, "x2": 5.0, "y2": 5.0, "conf": 0.9},
        {"clip_id": "c", "frame": 2, "x1": 1.0, "y1": 1.0, "x2": 5.0, "y2": 5.0, "conf": 0.9},
    ])
    factory = CountingFactory()
    tracks = run_track(dets, frames, clip.info.fps, factory)
    assert factory.updates == [1, 0, 1]
    assert tracks["frame"].to_list() == [0, 2]


def test_run_track_rejects_detections_for_another_clip():
    clip = FakeClip(clip_id="c", n_frames=2)
    dets = from_rows("detections", [
        {"clip_id": "other", "frame": 0, "x1": 1.0, "y1": 1.0, "x2": 5.0, "y2": 5.0, "conf": 0.9},
    ])
    with pytest.raises(ValueError, match="other"):
        run_track(dets, passthrough_frames(clip.info), clip.info.fps, FakeTrackerFactory())


def test_real_bytetrack_survives_an_empty_frame_and_keeps_the_id():
    rows = []
    for frame in range(10):
        if frame == 4:
            continue                                   # nothing detected on frame 4
        x = 100.0 + 5.0 * frame
        rows.append({"clip_id": "c", "frame": frame, "x1": x, "y1": 100.0, "x2": x + 40.0, "y2": 180.0, "conf": 0.9})
    clip = FakeClip(n_frames=10, width=640, height=360)
    tracks = run_track(from_rows("detections", rows), passthrough_frames(clip.info), clip.info.fps, bytetrack_factory)
    assert tracks["track_id"].n_unique() == 1
    assert 4 not in tracks["frame"].to_list()
    assert tracks.height >= 7


def test_run_track_skips_non_play_frames_entirely():
    """Non-play frames are not fed as empty frames: the tracker never sees them."""
    class CountingFactory:
        def __init__(self):
            self.updates = []
            self.made = 0

        def __call__(self, fps):
            self.made += 1
            factory = self

            class _Tracker:
                def update(self, dets):
                    factory.updates.append(len(dets))
                    dets.tracker_id = np.arange(len(dets))
                    return dets
            return _Tracker()

    frames = frames_table("c", [(0, True, 0), (1, False, None), (2, False, None), (3, True, 1), (4, True, 1)])
    dets = from_rows("detections", [
        {"clip_id": "c", "frame": f, "x1": 1.0, "y1": 1.0, "x2": 5.0, "y2": 5.0, "conf": 0.9} for f in (0, 1, 2, 3)
    ])
    factory = CountingFactory()
    tracks = run_track(dets, frames, 30.0, factory)
    assert factory.made == 1
    assert factory.updates == [1, 1, 0]              # frames 0, 3, 4; the boxes on 1 and 2 are never shown
    assert tracks["frame"].to_list() == [0, 3]


@pytest.mark.slow
def test_real_bytetrack_keeps_the_id_across_a_skipped_gap():
    """Review Focus 3: a box that is where it was before a gap keeps its id."""
    box = {"clip_id": "c", "x1": 100.0, "y1": 100.0, "x2": 140.0, "y2": 180.0, "conf": 0.95}
    rows = [{**box, "frame": f, "x1": 100.0 + f, "x2": 140.0 + f} for f in list(range(0, 6)) + list(range(10, 16))]
    frames = frames_table("c", [(f, f < 6 or f >= 10, 0 if f < 6 else (1 if f >= 10 else None)) for f in range(16)], fps=30.0)
    tracks = run_track(from_rows("detections", rows), frames, 30.0, bytetrack_factory)
    assert tracks["frame"].to_list()[0] >= 1                 # confirmation needs a second frame
    assert tracks["track_id"].n_unique() == 1
    assert {0, 1, 2, 3, 4, 5} - set(tracks["frame"]) <= {0} and set(range(10, 16)) <= set(tracks["frame"])
