import json

import pytest

from fakes import FakeDetector
from tracker.analysis.tracker_grid import FACTOR_VALUES, load_grid, one_factor_grid, run_grid
from tracker.datasets.mot import list_sequences
from tracker.pipeline import run_clip
from tracker.stages.detect_track import ByteTrackSettings

BOX = [10.0, 10.0, 30.0, 40.0]
GT_ROWS = [(f, 1, 10, 10, 20, 30) for f in (1, 2, 3, 4)]


def test_one_factor_grid_is_baseline_plus_each_single_change():
    points = one_factor_grid()
    assert points[0] == ByteTrackSettings()
    assert len(points) == 1 + sum(len(v) for v in FACTOR_VALUES.values())
    assert len(set(points)) == len(points)
    for point in points[1:]:
        assert point.label().count(",") == 0          # exactly one setting differs


def test_load_grid_prepends_the_baseline_and_drops_duplicates(tmp_path):
    path = tmp_path / "grid.json"
    path.write_text(json.dumps([{"lost_track_buffer": 60}, {"lost_track_buffer": 60}, {}]))
    points = load_grid(path)
    assert points == [ByteTrackSettings(), ByteTrackSettings(lost_track_buffer=60)]


def test_load_grid_names_a_bad_key_and_its_position(tmp_path):
    path = tmp_path / "grid.json"
    path.write_text(json.dumps([{"lost_track_buffer": 60}, {"lost_track_bufer": 10}]))
    with pytest.raises(ValueError, match=r"entry 1.*lost_track_bufer"):
        load_grid(path)


@pytest.fixture
def cached_run(make_sequence, tmp_path):
    root = tmp_path / "ds"
    for name in ("A_001", "B_001"):
        make_sequence(name=name, n_frames=4, gt_rows=GT_ROWS, parent=root / "mot-challenge-format" / "train")
    sequences = list_sequences(root)
    src = tmp_path / "src"
    for name, seq in sequences.items():
        run_clip(seq, src, FakeDetector([BOX]), overlay=False)
    (src / "run.json").write_text(json.dumps({"run": {"detector": "fake-detector"}, "clips": {}}))
    return src, sequences


def test_run_grid_scores_each_point_and_sorts_by_idf1(cached_run, tmp_path):
    src, sequences = cached_run
    points = [ByteTrackSettings(), ByteTrackSettings(minimum_consecutive_frames=3)]
    records = run_grid(src, tmp_path / "grid", points, sequences, ["A_001", "B_001"])
    by_label = {r["label"]: r for r in records}
    # Ground truth covers frames 0-3 (4 per clip); the detector returns the same box every frame.
    # Baseline: a track is confirmed on its second frame, so frames 1, 2, 3 are tracked:
    #   per clip idtp 3, n_gt 4, n_pred 3; pooled over two clips idtp 6, n_gt 8, n_pred 6
    #   IDF1 = 2 * 6 / (8 + 6) = 6/7.
    assert by_label["baseline"]["idf1"] == pytest.approx(6 / 7)
    # minimum_consecutive_frames=3: confirmed on the third frame, so frames 2, 3 are tracked:
    #   per clip idtp 2, n_pred 2; pooled idtp 4, n_gt 8, n_pred 4 -> IDF1 = 2 * 4 / (8 + 4) = 2/3.
    assert by_label["minimum_consecutive_frames=3"]["idf1"] == pytest.approx(2 / 3)
    assert records[0]["label"] == "baseline"                                      # best first
    assert records[0]["n_clips"] == 2 and records[0]["failed"] == {}
    assert (tmp_path / "grid" / "grid.json").exists() and (tmp_path / "grid" / "grid.md").exists()
    assert "minimum_consecutive_frames=3" in (tmp_path / "grid" / "grid.md").read_text()


def test_a_point_that_fails_on_one_clip_still_scores_the_others(cached_run, tmp_path, monkeypatch):
    """Review Focus 4."""
    src, sequences = cached_run
    import tracker.analysis.tracker_grid as grid_module
    real = grid_module.retrack_run

    def flaky(src_run, out_run, settings):
        results = real(src_run, out_run, settings)
        if settings.lost_track_buffer == 5:
            (out_run / "B_001" / "tracks.parquet").unlink()
            results["B_001"] = "stage 'track' failed on clip 'B_001': boom"
        return results

    monkeypatch.setattr(grid_module, "retrack_run", flaky)
    records = run_grid(src, tmp_path / "grid", [ByteTrackSettings(), ByteTrackSettings(lost_track_buffer=5)],
                       sequences, ["A_001", "B_001"])
    flaky_record = next(r for r in records if r["label"] == "lost_track_buffer=5")
    assert flaky_record["n_clips"] == 1
    assert "B_001" in flaky_record["failed"]
    # the scorer also reports B_001 (no tracks table); the re-track message is the one kept
    assert "stage 'track' failed" in flaky_record["failed"]["B_001"]
    assert next(r for r in records if r["label"] == "baseline")["n_clips"] == 2
