import json

import pytest

from fakes import FakeDetector
from tracker.cli import main

BOX = [10.0, 10.0, 30.0, 40.0]
GT_ROWS = [(f, 1, 10, 10, 20, 30) for f in (1, 2, 3)]


def fake_detector(threshold):
    detector = FakeDetector([BOX])
    detector.threshold = threshold
    return detector


@pytest.fixture
def dataset(make_sequence, tmp_path):
    """Seven clips from four games, in the dataset's folder layout."""
    root = tmp_path / "ds"
    names = ["A_001", "A_002", "B_001", "B_002", "C_001", "C_002", "D_001"]
    for name in names:
        make_sequence(name=name, n_frames=3, gt_rows=GT_ROWS,
                      parent=root / "mot-challenge-format" / "train")
    return root


def test_split_writes_a_file_with_whole_games(dataset, tmp_path):
    out = tmp_path / "splits" / "heldout.json"
    assert main(["split", "--dataset", str(dataset), "--out", str(out), "--heldout", "2"]) == 0
    split = json.loads(out.read_text())
    assert len(split["heldout"]) == 2
    assert len(split["heldout"]) + len(split["train"]) == 7
    assert len({name.split("_")[0] for name in split["heldout"]}) == 1


def test_run_then_score_on_the_heldout_clips(dataset, tmp_path, capsys):
    split = tmp_path / "heldout.json"
    run_dir = tmp_path / "run"
    main(["split", "--dataset", str(dataset), "--out", str(split), "--heldout", "2"])
    code = main(
        ["run", "--out", str(run_dir), "--dataset", str(dataset), "--split", str(split), "--no-overlay"],
        make_detector=fake_detector,
    )
    assert code == 0
    recorded = json.loads((run_dir / "run.json").read_text())
    assert recorded["run"]["detector"] == "fake-detector"
    assert sorted(recorded["clips"]) == sorted(json.loads(split.read_text())["heldout"])

    assert main(["score", "--run", str(run_dir), "--dataset", str(dataset), "--split", str(split)]) == 0
    card = json.loads((run_dir / "scorecard.json").read_text())
    # Real ByteTrack confirms a track on a clip's second frame, so 2 of the 3 frames
    # are tracked: IDF1 = 2 * 2 / (3 + 2) = 0.8 per clip and pooled.
    assert card["tracking"]["overall"]["idf1"] == pytest.approx(0.8)
    assert card["run"]["detector"] == "fake-detector"
    assert "IDF1 80.0%" in capsys.readouterr().out


def test_run_on_named_clips_and_a_threshold(dataset, tmp_path):
    run_dir = tmp_path / "run"
    clip = dataset / "mot-challenge-format" / "train" / "A_001"
    code = main(
        ["run", "--out", str(run_dir), "--clip", str(clip), "--threshold", "0.3", "--no-overlay"],
        make_detector=fake_detector,
    )
    assert code == 0
    assert json.loads((run_dir / "run.json").read_text())["run"]["detector_threshold"] == 0.3
    assert (run_dir / "A_001" / "tracks.parquet").exists()


def test_run_reports_failure_with_a_nonzero_exit(dataset, tmp_path, capsys):
    clip = dataset / "mot-challenge-format" / "train" / "A_001"
    (clip / "img1" / "000002.jpg").write_bytes(b"not a jpeg")
    code = main(
        ["run", "--out", str(tmp_path / "run"), "--clip", str(clip), "--no-overlay"],
        make_detector=fake_detector,
    )
    assert code == 1
    assert "FAILED" in capsys.readouterr().out


def test_run_json_with_the_detector_is_written_even_when_every_clip_fails(dataset, tmp_path):
    clip = dataset / "mot-challenge-format" / "train" / "A_001"
    (clip / "img1" / "000002.jpg").write_bytes(b"not a jpeg")
    run_dir = tmp_path / "run"
    main(["run", "--out", str(run_dir), "--clip", str(clip), "--no-overlay"], make_detector=fake_detector)
    recorded = json.loads((run_dir / "run.json").read_text())
    assert recorded["run"]["detector"] == "fake-detector"
    assert list(recorded["clips"]) == ["A_001"] and recorded["clips"]["A_001"] is not None


def test_run_without_clips_or_a_dataset_is_a_usage_error(tmp_path, capsys):
    assert main(["run", "--out", str(tmp_path / "run")], make_detector=fake_detector) == 2
    assert "--clip" in capsys.readouterr().err


def test_run_with_a_split_naming_an_unknown_clip_is_an_error(dataset, tmp_path, capsys):
    split = tmp_path / "heldout.json"
    split.write_text(json.dumps({"seed": 1, "heldout": ["Z_009"], "train": []}))
    code = main(
        ["run", "--out", str(tmp_path / "run"), "--dataset", str(dataset), "--split", str(split)],
        make_detector=fake_detector,
    )
    assert code == 2
    assert "Z_009" in capsys.readouterr().err


def test_score_with_nothing_scored_exits_nonzero(dataset, tmp_path):
    split = tmp_path / "heldout.json"
    main(["split", "--dataset", str(dataset), "--out", str(split), "--heldout", "2"])
    code = main(["score", "--run", str(tmp_path / "empty-run"), "--dataset", str(dataset), "--split", str(split)])
    assert code == 1
    assert (tmp_path / "empty-run" / "scorecard.md").exists()


def test_run_accepts_tracker_settings_and_records_them(dataset, tmp_path):
    run_dir = tmp_path / "run"
    clip = dataset / "mot-challenge-format" / "train" / "A_001"
    code = main(
        ["run", "--out", str(run_dir), "--clip", str(clip), "--no-overlay",
         "--set", "lost_track_buffer=12", "--set", "minimum_consecutive_frames=1"],
        make_detector=fake_detector,
    )
    assert code == 0
    recorded = json.loads((run_dir / "run.json").read_text())["run"]["tracker"]["settings"]
    assert recorded["lost_track_buffer"] == 12 and recorded["minimum_consecutive_frames"] == 1


def test_run_with_a_bad_setting_is_a_usage_error(dataset, tmp_path, capsys):
    clip = dataset / "mot-challenge-format" / "train" / "A_001"
    code = main(["run", "--out", str(tmp_path / "run"), "--clip", str(clip), "--set", "nope=1"],
                make_detector=fake_detector)
    assert code == 2
    assert "nope" in capsys.readouterr().err


def test_retrack_command_writes_a_new_run(dataset, tmp_path, capsys):
    src = tmp_path / "src"
    clip = dataset / "mot-challenge-format" / "train" / "A_001"
    main(["run", "--out", str(src), "--clip", str(clip), "--no-overlay"], make_detector=fake_detector)
    code = main(["retrack", "--run", str(src), "--out", str(tmp_path / "out"), "--set", "minimum_consecutive_frames=1"])
    assert code == 0
    assert (tmp_path / "out" / "A_001" / "tracks.parquet").exists()
    assert "A_001: ok" in capsys.readouterr().out
    recorded = json.loads((tmp_path / "out" / "run.json").read_text())["run"]
    assert recorded["tracker"]["settings"]["minimum_consecutive_frames"] == 1
    assert recorded["detector"] == "fake-detector"


def test_diagnose_boxes_writes_json_per_clip_a_summary_and_frames(dataset, tmp_path):
    split = tmp_path / "heldout.json"
    run_dir = tmp_path / "run"
    main(["split", "--dataset", str(dataset), "--out", str(split), "--heldout", "2"])
    main(["run", "--out", str(run_dir), "--dataset", str(dataset), "--split", str(split), "--no-overlay"],
         make_detector=fake_detector)
    held = json.loads(split.read_text())["heldout"]
    out = tmp_path / "diag"
    code = main(["diagnose-boxes", "--run", str(run_dir), "--dataset", str(dataset), "--split", str(split),
                 "--out", str(out), "--frames", "2", "--draw", held[0]])
    assert code == 0
    report = json.loads((out / f"{held[0]}.json").read_text())
    assert report["tracks"]["n_pred"] == 2                   # ByteTrack confirms on frame 1 of 3
    agreement = report["tracks_vs_detections"]
    assert agreement["n"] == 2 and 0.0 <= agreement["share_identical"] <= 1.0   # whether tracked boxes equal raw ones is Investigation 1's question, not this test's
    assert "0.5" in report["sweep"]
    text = (out / "summary.md").read_text()
    assert held[0] in text and held[1] in text
    assert len(list((out / "frames" / held[0]).glob("*.jpg"))) == 2


def test_diagnose_boxes_shift_writes_the_shift_tables(dataset, tmp_path):
    split = tmp_path / "heldout.json"
    run_dir = tmp_path / "run"
    main(["split", "--dataset", str(dataset), "--out", str(split), "--heldout", "2"])
    main(["run", "--out", str(run_dir), "--dataset", str(dataset), "--split", str(split), "--no-overlay"],
         make_detector=fake_detector)
    held = json.loads(split.read_text())["heldout"]
    out = tmp_path / "diag"
    assert main(["diagnose-boxes", "--run", str(run_dir), "--dataset", str(dataset), "--split", str(split),
                 "--out", str(out), "--shift"]) == 0
    shift = json.loads((out / f"{held[0]}.shift.json").read_text())
    # The ground-truth box equals the detector's box and 2 frames are tracked, so both match at k=0.
    assert shift["shifted_share"]["0"] == 1.0
    assert shift["drift_by_range"][0]["share_at_zero"] == 1.0
    text = (out / "shift.md").read_text()
    assert held[0] in text and held[1] in text and "k=+4" in text
    assert (out / "ntsc.md").exists()


def test_diagnose_boxes_without_shift_writes_no_shift_files(dataset, tmp_path):
    split = tmp_path / "heldout.json"
    run_dir = tmp_path / "run"
    main(["split", "--dataset", str(dataset), "--out", str(split), "--heldout", "2"])
    main(["run", "--out", str(run_dir), "--dataset", str(dataset), "--split", str(split), "--no-overlay"],
         make_detector=fake_detector)
    out = tmp_path / "diag"
    assert main(["diagnose-boxes", "--run", str(run_dir), "--dataset", str(dataset), "--split", str(split),
                 "--out", str(out)]) == 0
    assert not (out / "shift.md").exists()


def test_diagnose_boxes_keeps_going_when_one_draw_clip_cannot_be_drawn(dataset, tmp_path, capsys):
    split = tmp_path / "heldout.json"
    run_dir = tmp_path / "run"
    main(["split", "--dataset", str(dataset), "--out", str(split), "--heldout", "2"])
    main(["run", "--out", str(run_dir), "--dataset", str(dataset), "--split", str(split), "--no-overlay"],
         make_detector=fake_detector)
    held = json.loads(split.read_text())["heldout"]
    # Diagnosis needs only tables and ground truth, so it succeeds; drawing needs the images.
    (dataset / "mot-challenge-format" / "train" / held[0] / "img1" / "000003.jpg").write_bytes(b"not a jpeg")
    out = tmp_path / "diag"
    code = main(["diagnose-boxes", "--run", str(run_dir), "--dataset", str(dataset), "--split", str(split),
                 "--out", str(out), "--frames", "2", "--draw", held[0], "--draw", held[1]])
    assert code == 0
    printed = capsys.readouterr().out
    assert f"{held[0]}: not drawn" in printed
    assert f"{held[1]}: drew 2 frames" in printed


def test_grid_command_runs_a_small_grid(dataset, tmp_path, capsys):
    split = tmp_path / "heldout.json"
    src = tmp_path / "src"
    main(["split", "--dataset", str(dataset), "--out", str(split), "--heldout", "2"])
    main(["run", "--out", str(src), "--dataset", str(dataset), "--split", str(split), "--subset", "train", "--no-overlay"],
         make_detector=fake_detector)
    grid_file = tmp_path / "grid.json"
    grid_file.write_text(json.dumps([{"minimum_consecutive_frames": 3}]))
    code = main(["grid", "--run", str(src), "--dataset", str(dataset), "--split", str(split), "--subset", "train",
                 "--out", str(tmp_path / "grid"), "--grid", str(grid_file)])
    assert code == 0
    records = json.loads((tmp_path / "grid" / "grid.json").read_text())
    assert {r["label"] for r in records} == {"baseline", "minimum_consecutive_frames=3"}
    # 3-frame clips, ground truth on frames 0-2: baseline tracks frames 1, 2 (IDF1 0.8); mcf=3 tracks frame 2 only
    # (idtp 1, n_pred 1: IDF1 = 2 * 1 / (3 + 1) = 0.5). Baseline is best.
    assert "best: baseline" in capsys.readouterr().out


def test_grid_command_refuses_the_heldout_subset(dataset, tmp_path, capsys):
    split = tmp_path / "heldout.json"
    main(["split", "--dataset", str(dataset), "--out", str(split), "--heldout", "2"])
    code = main(["grid", "--run", str(tmp_path / "src"), "--dataset", str(dataset), "--split", str(split),
                 "--subset", "heldout", "--out", str(tmp_path / "grid"), "--one-factor"])
    assert code == 2
    assert "held-out" in capsys.readouterr().err


def test_label_frames_writes_a_frame_list_and_refuses_to_overwrite(dataset, tmp_path, capsys):
    split = tmp_path / "heldout.json"
    main(["split", "--dataset", str(dataset), "--out", str(split), "--heldout", "2"])
    out = tmp_path / "labels" / "rink_frames.json"
    assert main(["label-frames", "--dataset", str(dataset), "--split", str(split), "--out", str(out),
                 "--per-clip-train", "2", "--per-clip-eval", "1", "--seed", "3"]) == 0
    data = json.loads(out.read_text())
    assert len(data["training"]) == 2 * 5 and len(data["evaluation"]) == 1 * 2 and data["seed"] == 3
    assert main(["label-frames", "--dataset", str(dataset), "--split", str(split), "--out", str(out)]) == 2
    assert "exists" in capsys.readouterr().err


def test_label_rink_render_only_writes_an_image_without_a_window(dataset, tmp_path):
    split = tmp_path / "heldout.json"
    main(["split", "--dataset", str(dataset), "--out", str(split), "--heldout", "2"])
    frames = tmp_path / "rink_frames.json"
    main(["label-frames", "--dataset", str(dataset), "--split", str(split), "--out", str(frames),
          "--per-clip-train", "1", "--per-clip-eval", "1"])
    out = tmp_path / "view.jpg"
    code = main(["label-rink", "--dataset", str(dataset), "--frames", str(frames), "--labels", str(tmp_path / "labels.parquet"),
                 "--render-only", str(out)])
    assert code == 0 and out.stat().st_size > 0
    import cv2
    image = cv2.imread(str(out))
    assert image.shape[1] > 64            # frame (64 px) plus the legend side by side


def test_score_reports_lower_thresholds_and_frame_rates(dataset, tmp_path, capsys):
    split = tmp_path / "heldout.json"
    run_dir = tmp_path / "run"
    main(["split", "--dataset", str(dataset), "--out", str(split), "--heldout", "2"])
    main(["run", "--out", str(run_dir), "--dataset", str(dataset), "--split", str(split), "--no-overlay"], make_detector=fake_detector)
    assert main(["score", "--run", str(run_dir), "--dataset", str(dataset), "--split", str(split)]) == 0
    card = json.loads((run_dir / "scorecard.json").read_text())
    assert card["tracking"]["overall"]["idf1_at_0.3"] == pytest.approx(0.8)      # same as 0.5 here: boxes are exact
    assert card["tracking"]["by_frame_rate"]["30"]["n_clips"] == 2
    assert "IDF1@0.3 80.0%" in capsys.readouterr().out


def test_a_non_numeric_setting_names_the_key(dataset, tmp_path, capsys):
    clip = dataset / "mot-challenge-format" / "train" / "A_001"
    code = main(["run", "--out", str(tmp_path / "run"), "--clip", str(clip), "--set", "lost_track_buffer=abc"], make_detector=fake_detector)
    assert code == 2 and "lost_track_buffer" in capsys.readouterr().err


def test_grid_with_a_missing_file_is_a_usage_error(dataset, tmp_path, capsys):
    split = tmp_path / "heldout.json"
    main(["split", "--dataset", str(dataset), "--out", str(split), "--heldout", "2"])
    code = main(["grid", "--run", str(tmp_path / "src"), "--dataset", str(dataset), "--split", str(split),
                 "--subset", "train", "--out", str(tmp_path / "grid"), "--grid", str(tmp_path / "nope.json")])
    assert code == 2 and "nope.json" in capsys.readouterr().err


def test_run_json_survives_an_interrupted_run(dataset, tmp_path, monkeypatch):
    import tracker.cli as cli_module

    def killed(*args, **kwargs):
        raise RuntimeError("killed")

    monkeypatch.setattr(cli_module, "run_many", killed)
    clip = dataset / "mot-challenge-format" / "train" / "A_001"
    with pytest.raises(RuntimeError):
        main(["run", "--out", str(tmp_path / "run"), "--clip", str(clip), "--no-overlay"], make_detector=fake_detector)
    recorded = json.loads((tmp_path / "run" / "run.json").read_text())
    assert recorded["run"]["detector"] == "fake-detector" and recorded["clips"] == {}


def test_train_rink_refuses_labels_from_heldout_clips(dataset, tmp_path, capsys):
    from tracker.labels.rink_keypoints import frame_rows, load_labels, save_labels, upsert_frame
    split = tmp_path / "heldout.json"
    main(["split", "--dataset", str(dataset), "--out", str(split), "--heldout", "2"])
    held = json.loads(split.read_text())["heldout"][0]
    labels = upsert_frame(load_labels("/nonexistent"), held, 0, frame_rows(held, 0, {10: (5.0, 5.0)}))
    save_labels(labels, tmp_path / "labels.parquet", image_size=(64, 48))
    code = main(["train-rink", "--labels", str(tmp_path / "labels.parquet"), "--dataset", str(dataset),
                 "--split", str(split), "--out", str(tmp_path / "model"), "--epochs", "1"])
    assert code == 2 and held in capsys.readouterr().err
    assert not (tmp_path / "model").exists()
