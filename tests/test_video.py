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


def _write_video(path, n_frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for i in range(n_frames):
        writer.write(np.full((48, 64, 3), 30 * i, dtype=np.uint8))
    writer.release()


def test_video_frame_count_is_what_decodes_not_what_the_container_states(tmp_path, monkeypatch):
    """Real files state more frames than they hold (2657 stated, 2644 decoded)."""
    path = tmp_path / "overstated.mp4"
    _write_video(path, 6)

    real_capture = cv2.VideoCapture

    class OverReporting:
        def __init__(self, filename):
            self._capture = real_capture(filename)

        def get(self, prop):
            value = self._capture.get(prop)
            return value + 7 if prop == cv2.CAP_PROP_FRAME_COUNT else value

        def __getattr__(self, name):
            return getattr(self._capture, name)

    monkeypatch.setattr(cv2, "VideoCapture", OverReporting)
    assert OverReporting(str(path)).get(cv2.CAP_PROP_FRAME_COUNT) == 13   # 6 written + 7
    clip = Clip(path)
    assert clip.info.n_frames == 6
    assert [index for index, _ in clip.frames()] == [0, 1, 2, 3, 4, 5]


def test_video_decoder_failing_mid_stream_raises_clip_error(tmp_path, monkeypatch):
    """The count pass succeeds; a later pass that stops after 3 frames is an error."""
    path = tmp_path / "flaky.mp4"
    _write_video(path, 6)
    clip = Clip(path)
    assert clip.info.n_frames == 6

    real_capture = cv2.VideoCapture

    class FailsAfterThree:
        def __init__(self, filename):
            self._capture = real_capture(filename)
            self._reads = 0

        def read(self):
            if self._reads >= 3:
                return False, None
            self._reads += 1
            return self._capture.read()

        def __getattr__(self, name):
            return getattr(self._capture, name)

    monkeypatch.setattr(cv2, "VideoCapture", FailsAfterThree)
    with pytest.raises(ClipError, match=r"flaky.*frame 3.*3 of 6 frames"):
        list(clip.frames())


def test_malformed_seqinfo_non_numeric_framerate(make_sequence):
    """Non-numeric frameRate in seqinfo.ini raises ClipError."""
    seq = make_sequence()
    (seq / "seqinfo.ini").write_text(
        "[Sequence]\n"
        "name=TEST\n"
        "frameRate=not_a_number\n"
        "imDir=img1\n"
    )
    with pytest.raises(ClipError, match=r"GAME_A_001.*seqinfo.ini"):
        Clip(seq)


def test_malformed_seqinfo_no_section(make_sequence):
    """seqinfo.ini with no section header raises ClipError."""
    seq = make_sequence()
    (seq / "seqinfo.ini").write_text("frameRate=30\n")
    with pytest.raises(ClipError, match=r"GAME_A_001.*seqinfo.ini"):
        Clip(seq)


def test_malformed_seqinfo_with_explicit_fps_opens_fine(make_sequence):
    """Malformed seqinfo.ini is ignored when fps is explicitly passed."""
    seq = make_sequence()
    (seq / "seqinfo.ini").write_text("frameRate=not_a_number\n")
    clip = Clip(seq, fps=30)
    assert clip.info.fps == 30.0
    assert clip.info.clip_id == "GAME_A_001"


def test_read_frame_returns_the_asked_frame_of_an_image_folder(make_sequence):
    clip = Clip(make_sequence(n_frames=4))
    assert clip.read_frame(3).mean() > clip.read_frame(0).mean()      # brighter later frames
    assert clip.read_frame(2).shape == (48, 64, 3)


def test_read_frame_out_of_range_or_unreadable_names_the_clip_and_frame(make_sequence):
    seq = make_sequence(name="GAME_D_004", n_frames=3)
    (seq / "img1" / "000002.jpg").write_bytes(b"not a jpeg")
    clip = Clip(seq)
    with pytest.raises(ClipError, match=r"GAME_D_004.*frame 1"):
        clip.read_frame(1)
    with pytest.raises(ClipError, match=r"GAME_D_004.*frame 7"):
        clip.read_frame(7)


def test_read_frame_from_a_video_file(tmp_path):
    path = tmp_path / "shift.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for i in range(6):
        writer.write(np.full((48, 64, 3), 30 * i, dtype=np.uint8))
    writer.release()
    clip = Clip(path)
    assert clip.read_frame(5).mean() > clip.read_frame(1).mean()
    with pytest.raises(ClipError, match="frame 6"):
        clip.read_frame(6)
