import pytest

from fakes import FakeDetector, FakeTrackerFactory
from tracker.pipeline import StageError, run_clip, run_details, run_many, score_run
from tracker.tables import read_table

BOX = [10.0, 10.0, 30.0, 40.0]
GT_ROWS = [(f, 1, 10, 10, 20, 30) for f in (1, 2, 3)]      # same box as BOX, frames 1..3


def test_run_clip_writes_every_table_and_the_overlay(make_sequence, tmp_path):
    seq = make_sequence(name="GAME_A_001", n_frames=3)
    run_dir = tmp_path / "run"
    clip_id = run_clip(seq, run_dir, FakeDetector([BOX]), FakeTrackerFactory())
    assert clip_id == "GAME_A_001"
    out = run_dir / "GAME_A_001"
    assert read_table("frames", out / "frames.parquet").height == 3
    assert read_table("tracks", out / "tracks.parquet").height == 3
    assert read_table("tracking", out / "tracking.parquet").height == 3
    assert (out / "overlay.mp4").stat().st_size > 0


def test_overlay_can_be_turned_off(make_sequence, tmp_path):
    seq = make_sequence(n_frames=2)
    run_clip(seq, tmp_path / "run", FakeDetector([BOX]), FakeTrackerFactory(), overlay=False)
    assert not (tmp_path / "run" / seq.name / "overlay.mp4").exists()


def test_a_missing_clip_fails_with_the_stage_and_clip_named(tmp_path):
    with pytest.raises(StageError, match=r"stage 'open clip' failed on clip 'GHOST_001'"):
        run_clip(tmp_path / "GHOST_001", tmp_path / "run", FakeDetector([BOX]), FakeTrackerFactory())


def test_one_bad_clip_does_not_stop_the_others(make_sequence, tmp_path):
    good = make_sequence(name="GAME_A_001", n_frames=3)
    bad = make_sequence(name="GAME_B_001", n_frames=3)
    (bad / "img1" / "000002.jpg").write_bytes(b"not a jpeg")
    results = run_many([bad, good], tmp_path / "run", FakeDetector([BOX]), FakeTrackerFactory())
    assert results["GAME_A_001"] is None
    assert "stage 'detect and track' failed on clip 'GAME_B_001'" in results["GAME_B_001"]
    assert "frame 1" in results["GAME_B_001"]
    assert (tmp_path / "run" / "GAME_A_001" / "tracks.parquet").exists()


def test_rerunning_a_clip_replaces_its_tables(make_sequence, tmp_path):
    seq = make_sequence(n_frames=3)
    run_dir = tmp_path / "run"
    run_clip(seq, run_dir, FakeDetector([BOX, [40.0, 5.0, 50.0, 30.0]]), FakeTrackerFactory(), overlay=False)
    run_clip(seq, run_dir, FakeDetector([BOX]), FakeTrackerFactory(), overlay=False)
    assert read_table("tracks", run_dir / seq.name / "tracks.parquet").height == 3


def test_run_details_record_the_detector_and_package_versions():
    details = run_details(FakeDetector([BOX]))
    assert details["detector"] == "fake-detector"
    assert details["detector_threshold"] == 0.5
    assert "polars" in details["packages"]
    assert details["started_at"].endswith("+00:00")


def test_score_run_scores_good_clips_and_lists_the_rest(make_sequence, tmp_path):
    scored = make_sequence(name="GAME_A_001", n_frames=3, gt_rows=GT_ROWS)
    not_run = make_sequence(name="GAME_B_001", n_frames=3, gt_rows=GT_ROWS)
    run_dir = tmp_path / "run"
    run_clip(scored, run_dir, FakeDetector([BOX]), FakeTrackerFactory(), overlay=False)
    counts, failed = score_run(
        run_dir,
        {"GAME_A_001": scored, "GAME_B_001": not_run},
        ["GAME_A_001", "GAME_B_001", "GAME_C_001"],
    )
    assert counts["GAME_A_001"].idf1 == 1.0
    assert "tracks" in failed["GAME_B_001"]               # no tracks table for it
    assert "no ground truth sequence" in failed["GAME_C_001"]
