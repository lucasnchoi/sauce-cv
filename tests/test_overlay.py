import cv2
import pytest

from tracker.overlay import render_overlay
from tracker.schemas import from_rows
from tracker.video import Clip, ClipError


def one_track(n_frames):
    return from_rows("tracks", [
        {"clip_id": "GAME_A_001", "frame": f, "track_id": 3, "class": "player",
         "x1": 20.0, "y1": 10.0, "x2": 40.0, "y2": 40.0, "conf": 0.9}
        for f in range(n_frames)
    ])


def read_video(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, image = capture.read()
        if not ok:
            break
        frames.append(image)
    capture.release()
    return frames


def test_overlay_has_one_frame_per_clip_frame_and_draws_the_box(make_sequence, tmp_path):
    clip = Clip(make_sequence(name="GAME_A_001", n_frames=5, width=64, height=48))
    out = tmp_path / "out" / "overlay.mp4"
    render_overlay(clip, one_track(5), out)
    frames = read_video(out)
    assert len(frames) == 5
    assert frames[0].shape == (48, 64, 3)
    original = next(clip.frames())[1]
    # The box edge at x=20 is drawn in a colour unlike the flat grey frame.
    assert abs(frames[0][25, 20].astype(int) - original[25, 20].astype(int)).sum() > 60


def test_overlay_with_no_tracks_still_writes_every_frame(make_sequence, tmp_path):
    clip = Clip(make_sequence(n_frames=4))
    out = tmp_path / "overlay.mp4"
    render_overlay(clip, one_track(0), out)
    assert len(read_video(out)) == 4


def test_overlay_into_an_unwritable_place_is_an_error(make_sequence, tmp_path):
    clip = Clip(make_sequence(n_frames=2))
    blocker = tmp_path / "file.txt"
    blocker.write_text("x")
    with pytest.raises(ClipError, match="overlay"):
        render_overlay(clip, one_track(2), blocker / "overlay.mp4")


def test_overlay_with_positions_adds_the_rink_view_beside_the_frame(make_sequence, tmp_path):
    from tracker.schemas import H_COLUMNS
    from tracker.stages.assemble import assemble_tracking
    clip = Clip(make_sequence(name="GAME_A_001", n_frames=2, width=64, height=48))
    frames = from_rows("frames", [{"clip_id": "GAME_A_001", "frame": f, "time_s": f / 30, "is_play": True, "segment_id": 0} for f in range(2)])
    half = dict(zip(H_COLUMNS, [0.5, 0.0, -10.0, 0.0, 0.5, -10.0, 0.0, 0.0, 1.0]))
    transforms = from_rows("rink_transforms", [{"clip_id": "GAME_A_001", "frame": f, **half, "n_inliers": 8, "reproj_error_px": 1.0, "status": "fitted"} for f in range(2)])
    tracking = assemble_tracking(one_track(2), frames, transforms)
    out = tmp_path / "overlay.mp4"
    render_overlay(clip, one_track(2), out, tracking=tracking, rink_scale=2.0)
    video = read_video(out)
    assert len(video) == 2 and video[0].shape[0] == 48 and video[0].shape[1] == 64 + int(200 * 2.0 + 20)
    assert video[0][:, 64:].mean() > 100                     # the rink view is mostly white ice
