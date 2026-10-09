import pytest

from fakes import FakeDetector, FakeTrackerFactory
from tracker.pipeline import StageError, retrack_run, run_clip, run_details, run_many, score_run
from tracker.stages.detect_track import ByteTrackSettings
from tracker.stages.footage_filter import FootageFilterSettings
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
    (run_dir / "GAME_A_002" / "rink_transforms.parquet").unlink()      # run_clip always writes it; remove to test the missing case
    labels = upsert_frame(load_labels("/nonexistent"), "GAME_A_001", 1, frame_rows("GAME_A_001", 1, {10: (0.0, 0.0)}))
    counts, failed = score_rink_run(run_dir, ["GAME_A_001", "GAME_A_002"], labels)
    assert counts["GAME_A_001"].n_points == 1 and counts["GAME_A_001"].valid_share == 1.0
    assert "GAME_A_002" in failed and "rink_transforms.parquet" in failed["GAME_A_002"]


from fakes import FakeKeypointDetector
from tracker.pipeline import RinkStages

RINK_POINTS = {k: (float(5 + 3 * k), float(10 + (k % 5) * 6), 0.9) for k in (0, 2, 4, 6, 8, 10, 11, 13, 15, 17, 19, 23)}

import numpy as np

from tracker.rink import template_points
from tracker.stages.rink_transform import project

H_CAMERA = np.array([[6.0, 0.5, 640.0], [0.0, -4.0, 400.0], [0.0, 0.0005, 1.0]])
GOOD_IDS = (0, 2, 4, 6, 8, 10, 11, 13, 15, 17, 19, 23, 27, 29, 31)
GOOD_POINTS = {k: (float(x), float(y), 0.9) for k, (x, y) in zip(GOOD_IDS, project(H_CAMERA, template_points()[list(GOOD_IDS)]))}


def test_run_clip_with_rink_stages_writes_keypoints_frames_transforms_and_positions(make_sequence, tmp_path):
    seq = make_sequence(name="GAME_A_001", n_frames=3)
    detector = FakeKeypointDetector(RINK_POINTS)
    rink = RinkStages(make_keypoint_detector=lambda: detector, footage_filter=FootageFilterSettings(min_segment_s=0.0, bridge_gap_s=0.0), footage_mode="keypoints")
    run_clip(seq, tmp_path / "run", FakeDetector([BOX]), FakeTrackerFactory(), overlay=True, rink=rink)
    out = tmp_path / "run" / "GAME_A_001"
    assert read_table("keypoints", out / "keypoints.parquet").height == 3 * len(RINK_POINTS)
    fits = read_table("rink_fits", out / "rink_fits.parquet")
    assert fits["frame"].to_list() == [0, 1, 2] and set(fits["status"].to_list()) <= {"fitted", "invalid"}
    frames = read_table("frames", out / "frames.parquet")
    assert frames["is_play"].all()
    transforms = read_table("rink_transforms", out / "rink_transforms.parquet")
    assert transforms.height == 3 and set(transforms["status"].to_list()) <= {"fitted", "invalid"}
    tracking = read_table("tracking", out / "tracking.parquet")
    assert set(tracking["position_status"].to_list()) <= {"observed", "interpolated", "missing"}
    assert (out / "overlay.mp4").stat().st_size > 0 and detector.calls == 3


def test_a_clip_with_no_play_frames_still_succeeds_with_everything_missing(make_sequence, tmp_path):
    """Review Focus 4."""
    seq = make_sequence(name="GAME_A_001", n_frames=3)
    rink = RinkStages(make_keypoint_detector=lambda: FakeKeypointDetector({10: (5.0, 5.0, 0.9)}))     # one point: never play
    clip_id = run_clip(seq, tmp_path / "run", FakeDetector([BOX]), FakeTrackerFactory(), overlay=False, rink=rink)
    out = tmp_path / "run" / clip_id
    frames = read_table("frames", out / "frames.parquet")
    assert not frames["is_play"].any()
    assert read_table("rink_transforms", out / "rink_transforms.parquet").height == 0
    assert read_table("detections", out / "detections.parquet").height == 3     # the detector still ran on every frame
    assert read_table("tracks", out / "tracks.parquet").height == 0
    assert read_table("rink_fits", out / "rink_fits.parquet")["status"].to_list() == ["invalid"] * 3
    assert read_table("tracking", out / "tracking.parquet").height == 0


def test_passthrough_mode_ignores_the_keypoint_rule_but_still_fits_transforms(make_sequence, tmp_path):
    seq = make_sequence(name="GAME_A_001", n_frames=2)
    rink = RinkStages(make_keypoint_detector=lambda: FakeKeypointDetector({10: (5.0, 5.0, 0.9)}), footage_mode="passthrough")
    run_clip(seq, tmp_path / "run", FakeDetector([BOX]), FakeTrackerFactory(), overlay=False, rink=rink)
    out = tmp_path / "run" / "GAME_A_001"
    assert read_table("frames", out / "frames.parquet")["is_play"].all()
    assert read_table("rink_transforms", out / "rink_transforms.parquet")["status"].to_list() == ["invalid", "invalid"]


def test_missing_weights_fail_the_keypoint_stage_on_every_clip_with_the_path(make_sequence, tmp_path):
    """Review Focus 5."""
    def no_weights():
        raise FileNotFoundError("rink keypoint weights not found at /models/nope.pt")
    a = make_sequence(name="GAME_A_001", n_frames=2)
    b = make_sequence(name="GAME_A_002", n_frames=2)
    results = run_many([a, b], tmp_path / "run", FakeDetector([BOX]), FakeTrackerFactory(), overlay=False, rink=RinkStages(make_keypoint_detector=no_weights))
    assert set(results) == {"GAME_A_001", "GAME_A_002"}
    assert all("rink keypoints" in m and "/models/nope.pt" in m for m in results.values())


def test_run_details_record_the_rink_settings(tmp_path):
    rink = RinkStages(make_keypoint_detector=lambda: FakeKeypointDetector({}), footage_filter=FootageFilterSettings(min_keypoints=7), footage_mode="keypoints")
    details = run_details(FakeDetector([BOX]), rink=rink)
    assert details["footage_filter"] == {"mode": "keypoints", "settings": {"min_keypoints": 7, "min_conf": 0.5, "bridge_gap_s": 2.0, "min_segment_s": 1.0}}
    assert details["rink_transform"]["max_reproj_px"] == 10.0 and details["rink_keypoints"]["name"] == "fake-keypoints"
    assert run_details(FakeDetector([BOX]))["footage_filter"] == {"mode": "passthrough", "settings": None}
    assert run_details(FakeDetector([BOX]))["rink_keypoints"] is None
    default = RinkStages(make_keypoint_detector=lambda: FakeKeypointDetector({}))
    assert run_details(FakeDetector([BOX]), rink=default)["footage_filter"] == {"mode": "transform", "settings": {"bridge_gap_s": 2.0, "min_segment_s": 1.0}}
    assert run_details(FakeDetector([BOX]))["detector_frames"] == "all"


def test_retrack_keeps_rink_transforms_and_positions(make_sequence, tmp_path):
    seq = make_sequence(name="GAME_A_001", n_frames=3)
    rink = RinkStages(make_keypoint_detector=lambda: FakeKeypointDetector(GOOD_POINTS), footage_filter=FootageFilterSettings(min_segment_s=0.0))
    run_clip(seq, tmp_path / "src", FakeDetector([BOX]), overlay=False, rink=rink)
    retrack_run(tmp_path / "src", tmp_path / "out", ByteTrackSettings())
    assert read_table("rink_transforms", tmp_path / "out" / "GAME_A_001" / "rink_transforms.parquet").height == 3
    before = read_table("tracking", tmp_path / "src" / "GAME_A_001" / "tracking.parquet")["position_status"].to_list()
    after = read_table("tracking", tmp_path / "out" / "GAME_A_001" / "tracking.parquet")["position_status"].to_list()
    assert before == after


def test_transform_mode_calls_a_frame_play_only_when_its_fit_is_valid(make_sequence, tmp_path):
    seq = make_sequence(name="GAME_A_001", n_frames=3)
    detector = FakeKeypointDetector(GOOD_POINTS, per_frame={2: {10: (5.0, 5.0, 0.9)}})     # frame 2: one point, no fit
    rink = RinkStages(make_keypoint_detector=lambda: detector, footage_filter=FootageFilterSettings(min_segment_s=0.0, bridge_gap_s=0.0))
    run_clip(seq, tmp_path / "run", FakeDetector([BOX]), FakeTrackerFactory(), overlay=False, rink=rink)
    out = tmp_path / "run" / "GAME_A_001"
    assert read_table("rink_fits", out / "rink_fits.parquet")["status"].to_list() == ["fitted", "fitted", "invalid"]
    frames = read_table("frames", out / "frames.parquet")
    assert frames["is_play"].to_list() == [True, True, False]
    assert read_table("rink_transforms", out / "rink_transforms.parquet")["frame"].to_list() == [0, 1]
    assert read_table("detections", out / "detections.parquet")["frame"].to_list() == [0, 1, 2]
    assert read_table("tracks", out / "tracks.parquet")["frame"].to_list() == [0, 1]
    assert read_table("tracking", out / "tracking.parquet")["position_status"].to_list() == ["observed", "observed"]


from tracker.pipeline import refilter_run
from tracker.stages.rink_transform import RinkTransformSettings


def _source_run(make_sequence, tmp_path, detector_frames="all", keypoints=GOOD_POINTS, per_frame=None):
    seq = make_sequence(name="GAME_A_001", n_frames=4)
    rink = RinkStages(make_keypoint_detector=lambda: FakeKeypointDetector(keypoints, per_frame=per_frame),
                      footage_filter=FootageFilterSettings(min_segment_s=0.0, bridge_gap_s=0.0))
    run_clip(seq, tmp_path / "src", FakeDetector([BOX]), FakeTrackerFactory(), overlay=False, rink=rink)
    details = run_details(FakeDetector([BOX]), rink=rink)
    if detector_frames is None:
        details.pop("detector_frames")
    (tmp_path / "src" / "run.json").write_text(json.dumps({"run": details, "clips": {"GAME_A_001": None}}))
    return tmp_path / "src"


def test_refilter_reproduces_a_fresh_run_under_the_same_settings(make_sequence, tmp_path):
    src = _source_run(make_sequence, tmp_path, per_frame={2: {10: (5.0, 5.0, 0.9)}})
    results = refilter_run(src, tmp_path / "out", "transform", FootageFilterSettings(min_segment_s=0.0, bridge_gap_s=0.0),
                           RinkTransformSettings(), ByteTrackSettings(), tracker_factory=FakeTrackerFactory())
    assert results == {"GAME_A_001": None}
    for table in ("frames", "rink_transforms", "tracks", "tracking", "detections", "rink_fits", "keypoints"):
        before = read_table(table, src / "GAME_A_001" / f"{table}.parquet").to_dicts()
        after = read_table(table, tmp_path / "out" / "GAME_A_001" / f"{table}.parquet").to_dicts()
        assert before == after, table
    assert read_table("frames", tmp_path / "out" / "GAME_A_001" / "frames.parquet")["is_play"].to_list() == [True, True, False, True]
    recorded = json.loads((tmp_path / "out" / "run.json").read_text())["run"]
    assert recorded["refiltered_from"] == str(src) and "refiltered_at" in recorded
    assert recorded["footage_filter"] == {"mode": "transform", "settings": {"bridge_gap_s": 0.0, "min_segment_s": 0.0}}
    assert recorded["detector"] == "fake-detector" and recorded["detector_frames"] == "all"


def test_refilter_can_change_the_mode_and_the_tracks_follow(make_sequence, tmp_path):
    src = _source_run(make_sequence, tmp_path, per_frame={2: {10: (5.0, 5.0, 0.9)}})
    refilter_run(src, tmp_path / "out", "passthrough", FootageFilterSettings(), RinkTransformSettings(), ByteTrackSettings(),
                 tracker_factory=FakeTrackerFactory())
    out = tmp_path / "out" / "GAME_A_001"
    assert read_table("frames", out / "frames.parquet")["is_play"].all()
    assert read_table("tracks", out / "tracks.parquet")["frame"].to_list() == [0, 1, 2, 3]        # frame 2 is back, from the cache
    assert read_table("rink_transforms", out / "rink_transforms.parquet")["status"].to_list() == ["fitted", "fitted", "interpolated", "fitted"]   # one bad frame inside play is interpolated
    assert json.loads((tmp_path / "out" / "run.json").read_text())["run"]["footage_filter"] == {"mode": "passthrough", "settings": None}


def test_refilter_refuses_a_run_detected_on_play_frames_only(make_sequence, tmp_path):
    """Review Focus 4."""
    src = _source_run(make_sequence, tmp_path, detector_frames=None)
    with pytest.raises(ValueError, match="every frame"):
        refilter_run(src, tmp_path / "out", "transform", FootageFilterSettings(), RinkTransformSettings(), ByteTrackSettings())
    assert not (tmp_path / "out").exists()


def test_refilter_names_a_clip_without_keypoints_and_refuses_its_own_source(make_sequence, tmp_path):
    src = _source_run(make_sequence, tmp_path)
    (src / "GAME_A_001" / "keypoints.parquet").unlink()
    results = refilter_run(src, tmp_path / "out", "transform", FootageFilterSettings(), RinkTransformSettings(), ByteTrackSettings(),
                           tracker_factory=FakeTrackerFactory())
    assert "keypoints.parquet" in results["GAME_A_001"] and "read cached run" in results["GAME_A_001"]
    with pytest.raises(ValueError, match="same folder"):
        refilter_run(src, src, "transform", FootageFilterSettings(), RinkTransformSettings(), ByteTrackSettings())
    with pytest.raises(ValueError, match="mode"):
        refilter_run(src, tmp_path / "out2", "magic", FootageFilterSettings(), RinkTransformSettings(), ByteTrackSettings())


def test_refilter_applies_rink_fit_settings_and_records_them(make_sequence, tmp_path):
    src = _source_run(make_sequence, tmp_path)
    strict = RinkTransformSettings(max_reproj_px=0.0)          # nothing can pass a zero error limit
    results = refilter_run(src, tmp_path / "out", "transform", FootageFilterSettings(), strict, ByteTrackSettings(),
                           tracker_factory=FakeTrackerFactory())
    assert results == {"GAME_A_001": None}
    out = tmp_path / "out" / "GAME_A_001"
    assert read_table("rink_fits", out / "rink_fits.parquet")["status"].to_list() == ["invalid"] * 4
    assert not read_table("frames", out / "frames.parquet")["is_play"].any()
    assert json.loads((tmp_path / "out" / "run.json").read_text())["run"]["rink_transform"]["max_reproj_px"] == 0.0


def test_refilter_refuses_a_source_without_run_json(make_sequence, tmp_path):
    src = _source_run(make_sequence, tmp_path)
    (src / "run.json").unlink()
    with pytest.raises(ValueError, match="no run.json"):
        refilter_run(src, tmp_path / "out", "transform", FootageFilterSettings(), RinkTransformSettings(), ByteTrackSettings())
