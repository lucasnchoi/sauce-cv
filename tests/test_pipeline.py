import pytest

from fakes import FakeDetector, FakeTrackerFactory
from tracker.pipeline import StageError, run_clip, run_details, run_many, score_run
from tracker.schemas import from_rows
from tracker.tables import read_table, write_table

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
    assert "stage 'detect' failed on clip 'GAME_B_001'" in results["GAME_B_001"]
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
    assert counts["GAME_A_001"][0.5].idf1 == 1.0
    assert "tracks" in failed["GAME_B_001"]               # no tracks table for it
    assert "no ground truth sequence" in failed["GAME_C_001"]


from tracker.pipeline import read_clip_info, write_clip_info
from tracker.video import ClipInfo


def test_run_clip_also_writes_detections_and_clip_info(make_sequence, tmp_path):
    seq = make_sequence(name="GAME_A_001", n_frames=3, fps=60)
    run_dir = tmp_path / "run"
    run_clip(seq, run_dir, FakeDetector([BOX]), FakeTrackerFactory(), overlay=False)
    out = run_dir / "GAME_A_001"
    assert read_table("detections", out / "detections.parquet").height == 3
    info = read_clip_info(out / "clip.json")
    assert info == ClipInfo(clip_id="GAME_A_001", fps=60.0, n_frames=3, width=64, height=48)


def test_clip_info_round_trips(tmp_path):
    info = ClipInfo(clip_id="x", fps=29.97, n_frames=12, width=640, height=360)
    write_clip_info(info, tmp_path / "deep" / "clip.json")
    assert read_clip_info(tmp_path / "deep" / "clip.json") == info


def test_a_failing_detector_names_the_detect_stage(make_sequence, tmp_path):
    class Exploding:
        name = "boom"

        def detect(self, image_bgr):
            raise RuntimeError("cuda fell over")

    with pytest.raises(StageError, match=r"stage 'detect' failed on clip 'GAME_A_001'.*cuda fell over"):
        run_clip(make_sequence(name="GAME_A_001", n_frames=2), tmp_path / "run", Exploding(), FakeTrackerFactory())


import json

from tracker.pipeline import retrack_run
from tracker.stages.detect_track import ByteTrackSettings


def test_run_details_record_the_tracker_settings():
    details = run_details(FakeDetector([BOX]), ByteTrackSettings(lost_track_buffer=5))
    assert details["tracker"]["name"] == "bytetrack"
    assert details["tracker"]["settings"]["lost_track_buffer"] == 5
    assert run_details(FakeDetector([BOX]))["tracker"]["settings"]["lost_track_buffer"] == 30


def test_retrack_reproduces_tracks_and_records_where_it_came_from(make_sequence, tmp_path):
    seq = make_sequence(name="GAME_A_001", n_frames=6)
    src = tmp_path / "src"
    run_clip(seq, src, FakeDetector([BOX]), overlay=False)        # real ByteTrack, defaults
    (src / "run.json").write_text(json.dumps({"run": {"detector": "fake-detector"}, "clips": {"GAME_A_001": None}}))
    out = tmp_path / "out"
    results = retrack_run(src, out, ByteTrackSettings())
    assert results == {"GAME_A_001": None}
    original = read_table("tracks", src / "GAME_A_001" / "tracks.parquet")
    again = read_table("tracks", out / "GAME_A_001" / "tracks.parquet")
    assert again.to_dicts() == original.to_dicts()                # Review Focus 2
    assert read_table("tracking", out / "GAME_A_001" / "tracking.parquet").height == again.height
    recorded = json.loads((out / "run.json").read_text())
    assert recorded["run"]["detector"] == "fake-detector"
    assert recorded["run"]["tracker"]["settings"]["lost_track_buffer"] == 30
    assert recorded["run"]["retracked_from"] == str(src)
    assert not (out / "GAME_A_001" / "overlay.mp4").exists()


def test_retrack_with_other_settings_can_change_the_tracks(make_sequence, tmp_path):
    seq = make_sequence(name="GAME_A_001", n_frames=6)
    src = tmp_path / "src"
    run_clip(seq, src, FakeDetector([BOX]), overlay=False)
    retrack_run(src, tmp_path / "out", ByteTrackSettings(minimum_consecutive_frames=3))
    # defaults confirm from frame 1 (5 rows of 6); requiring 3 confirms from frame 2 (4 rows)
    assert read_table("tracks", src / "GAME_A_001" / "tracks.parquet").height == 5
    assert read_table("tracks", tmp_path / "out" / "GAME_A_001" / "tracks.parquet").height == 4


def test_retrack_skips_clip_folders_without_detections_and_reports_them(make_sequence, tmp_path):
    seq = make_sequence(name="GAME_A_001", n_frames=3)
    src = tmp_path / "src"
    run_clip(seq, src, FakeDetector([BOX]), overlay=False)
    (src / "GAME_B_001").mkdir()
    results = retrack_run(src, tmp_path / "out", ByteTrackSettings())
    assert results["GAME_A_001"] is None
    assert "stage 'read cached run' failed on clip 'GAME_B_001'" in results["GAME_B_001"]


def test_retrack_accepts_a_plan_1_run_json_whose_tracker_is_a_string(make_sequence, tmp_path):
    src = tmp_path / "src"
    run_clip(make_sequence(name="GAME_A_001", n_frames=3), src, FakeDetector([BOX]), overlay=False)
    (src / "run.json").write_text(json.dumps({"run": {"tracker": "ByteTrack, default settings"}, "clips": {}}))
    retrack_run(src, tmp_path / "out", ByteTrackSettings())
    recorded = json.loads((tmp_path / "out" / "run.json").read_text())["run"]
    assert recorded["tracker"]["settings"]["lost_track_buffer"] == 30


def test_score_run_scores_at_three_thresholds(make_sequence, tmp_path):
    scored = make_sequence(name="GAME_A_001", n_frames=3, gt_rows=GT_ROWS)
    run_dir = tmp_path / "run"
    run_clip(scored, run_dir, FakeDetector([BOX]), FakeTrackerFactory(), overlay=False)
    counts, failed = score_run(run_dir, {"GAME_A_001": scored}, ["GAME_A_001"])
    assert set(counts["GAME_A_001"]) == {0.5, 0.4, 0.3}
    assert counts["GAME_A_001"][0.5].idf1 == 1.0 and failed == {}


def test_run_many_refuses_two_clips_with_the_same_id(make_sequence, tmp_path):
    """Review Focus 5."""
    a = make_sequence(name="002", parent=tmp_path / "A")
    b = make_sequence(name="002", parent=tmp_path / "B")
    with pytest.raises(ValueError, match=r"002.*A.*B|002.*B.*A"):
        run_many([a, b], tmp_path / "run", FakeDetector([BOX]), FakeTrackerFactory())
    assert not (tmp_path / "run").exists()


def test_retrack_refuses_to_overwrite_its_source(make_sequence, tmp_path):
    src = tmp_path / "src"
    run_clip(make_sequence(n_frames=2), src, FakeDetector([BOX]), overlay=False)
    with pytest.raises(ValueError, match="same folder"):
        retrack_run(src, src, ByteTrackSettings())


def test_retrack_fails_fast_on_a_corrupt_source_run_json(make_sequence, tmp_path):
    src = tmp_path / "src"
    run_clip(make_sequence(n_frames=2), src, FakeDetector([BOX]), overlay=False)
    (src / "run.json").write_text("{not json")
    with pytest.raises(ValueError, match="run.json"):
        retrack_run(src, tmp_path / "out", ByteTrackSettings())
    assert not (tmp_path / "out").exists()


def test_retrack_carries_over_every_table_the_tracker_does_not_produce(make_sequence, tmp_path):
    src = tmp_path / "src"
    run_clip(make_sequence(name="GAME_A_001", n_frames=2), src, FakeDetector([BOX]), overlay=False)
    extra = from_rows("keypoints", [{"clip_id": "GAME_A_001", "frame": 0, "keypoint_id": 10, "x_px": 1.0, "y_px": 2.0, "conf": 0.9}])
    write_table(extra, "keypoints", src / "GAME_A_001" / "keypoints.parquet")
    retrack_run(src, tmp_path / "out", ByteTrackSettings())
    assert read_table("keypoints", tmp_path / "out" / "GAME_A_001" / "keypoints.parquet").height == 1


def test_score_run_lists_a_corrupt_tracks_table_as_failed(make_sequence, tmp_path):
    seq = make_sequence(name="GAME_A_001", n_frames=3, gt_rows=GT_ROWS)
    run_dir = tmp_path / "run"
    run_clip(seq, run_dir, FakeDetector([BOX]), FakeTrackerFactory(), overlay=False)
    (run_dir / "GAME_A_001" / "tracks.parquet").write_bytes(b"garbage")
    counts, failed = score_run(run_dir, {"GAME_A_001": seq}, ["GAME_A_001"])
    assert counts == {} and "GAME_A_001" in failed


def test_score_rink_run_reads_each_clip_and_names_a_missing_transforms_table(make_sequence, tmp_path):
    from tracker.labels.rink_keypoints import frame_rows, load_labels, upsert_frame
    from tracker.pipeline import score_rink_run
    from tracker.schemas import H_COLUMNS, from_rows
    run_dir = tmp_path / "run"
    run_clip(make_sequence(name="GAME_A_001", n_frames=3), run_dir, FakeDetector([BOX]), FakeTrackerFactory(), overlay=False)
    run_clip(make_sequence(name="GAME_A_002", n_frames=3), run_dir, FakeDetector([BOX]), FakeTrackerFactory(), overlay=False)
    identity = dict(zip(H_COLUMNS, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]))
    write_table(from_rows("rink_transforms", [{"clip_id": "GAME_A_001", "frame": f, **identity, "n_inliers": 8, "reproj_error_px": 1.0, "status": "fitted"} for f in range(3)]),
                "rink_transforms", run_dir / "GAME_A_001" / "rink_transforms.parquet")
    labels = upsert_frame(load_labels("/nonexistent"), "GAME_A_001", 1, frame_rows("GAME_A_001", 1, {10: (0.0, 0.0)}))
    counts, failed = score_rink_run(run_dir, ["GAME_A_001", "GAME_A_002"], labels)
    assert counts["GAME_A_001"].n_points == 1 and counts["GAME_A_001"].valid_share == 1.0
    assert "GAME_A_002" in failed and "rink_transforms.parquet" in failed["GAME_A_002"]
