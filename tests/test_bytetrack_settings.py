import pytest

from tracker.stages.detect_track import DEFAULT_BYTETRACK, ByteTrackSettings, bytetrack_factory


def test_defaults_match_the_library_defaults_we_verified():
    assert DEFAULT_BYTETRACK.as_dict() == {
        "lost_track_buffer": 30,
        "track_activation_threshold": 0.7,
        "minimum_consecutive_frames": 2,
        "minimum_iou_threshold": 0.1,
        "high_conf_det_threshold": 0.6,
    }
    assert DEFAULT_BYTETRACK.label() == "baseline"


def test_label_names_only_what_differs_from_the_defaults():
    settings = ByteTrackSettings(lost_track_buffer=60, minimum_iou_threshold=0.2)
    assert settings.label() == "lost_track_buffer=60,minimum_iou_threshold=0.2"


def test_from_dict_rejects_unknown_keys_by_name():
    with pytest.raises(ValueError, match="lost_track_bufer"):
        ByteTrackSettings.from_dict({"lost_track_bufer": 10})


def test_from_dict_round_trips_and_casts_types():
    settings = ByteTrackSettings.from_dict({"lost_track_buffer": "45", "track_activation_threshold": "0.5"})
    assert settings.lost_track_buffer == 45 and settings.track_activation_threshold == 0.5
    assert ByteTrackSettings.from_dict(settings.as_dict()) == settings


def test_factory_passes_the_settings_to_the_real_tracker():
    """A track is first returned unconfirmed on the frame it appears, and the library
    counts that frame, so the default of 2 confirms it on the second frame. Requiring 3
    delays confirmation to the third."""
    from fakes import FakeClip
    from tracker.schemas import from_rows
    from tracker.stages.detect_track import run_track
    from tracker.stages.footage_filter import passthrough_frames

    clip = FakeClip(clip_id="c", n_frames=3)
    frames = passthrough_frames(clip.info)
    dets = from_rows("detections", [
        {"clip_id": "c", "frame": f, "x1": 100.0, "y1": 100.0, "x2": 140.0, "y2": 180.0, "conf": 0.9}
        for f in range(3)
    ])
    assert run_track(dets, frames, 30.0, bytetrack_factory)["frame"].to_list() == [1, 2]
    patient = ByteTrackSettings(minimum_consecutive_frames=3).factory()
    assert run_track(dets, frames, 30.0, patient)["frame"].to_list() == [2]


def test_default_factory_is_unchanged_in_shape():
    assert hasattr(bytetrack_factory(60.0), "update")
