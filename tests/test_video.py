import cv2
import numpy as np
import pytest

from tracker.video import Clip, ClipError


def test_sequence_folder_reports_info_from_files_not_metadata(make_sequence):
    seq = make_sequence(name="GAME_A_001", n_frames=5, fps=60, width=64, height=48)
    info = Clip(seq).info
    assert info.clip_id == "GAME_A_001"      # folder name, not the seqinfo name
    assert info.n_frames == 5                # files on disk, not seqLength
    assert info.fps == 60.0
    assert (info.width, info.height) == (64, 48)


def test_frames_are_yielded_in_order_from_zero(make_sequence):
    clip = Clip(make_sequence(n_frames=4))
    frames = list(clip.frames())
    assert [index for index, _ in frames] == [0, 1, 2, 3]
    assert frames[0][1].shape == (48, 64, 3)
    assert frames[3][1].mean() > frames[0][1].mean()   # brighter later frames


def test_explicit_fps_overrides_seqinfo(make_sequence):
    assert Clip(make_sequence(fps=60), fps=25).info.fps == 25.0


def test_plain_image_folder_needs_an_fps(make_sequence):
    seq = make_sequence()
    with pytest.raises(ClipError, match="frame rate"):
        Clip(seq / "img1")
    assert Clip(seq / "img1", fps=30).info.clip_id == "img1"


def test_unreadable_frame_names_the_clip_and_frame(make_sequence):
    seq = make_sequence(name="GAME_B_002", n_frames=4)
    (seq / "img1" / "000003.jpg").write_bytes(b"not a jpeg")
    clip = Clip(seq)
    with pytest.raises(ClipError, match=r"GAME_B_002.*frame 2"):
        list(clip.frames())


def test_empty_folder_is_an_error(tmp_path):
    (tmp_path / "EMPTY_001").mkdir()
    with pytest.raises(ClipError, match="no image files"):
        Clip(tmp_path / "EMPTY_001", fps=30)


def test_missing_path_is_an_error(tmp_path):
    with pytest.raises(ClipError, match="does not exist"):
        Clip(tmp_path / "nope")


def test_file_that_is_not_a_video_is_an_error(tmp_path):
    path = tmp_path / "notes.mp4"
    path.write_text("hello")
    with pytest.raises(ClipError, match="could not open"):
        Clip(path)


def test_video_file_is_read(tmp_path):
    path = tmp_path / "shift.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for i in range(6):
        writer.write(np.full((48, 64, 3), 30 * i, dtype=np.uint8))
    writer.release()
    clip = Clip(path)
    assert clip.info.clip_id == "shift"
    assert clip.info.fps == pytest.approx(30.0, abs=0.5)
    assert (clip.info.width, clip.info.height) == (64, 48)
    assert [index for index, _ in clip.frames()] == [0, 1, 2, 3, 4, 5]
