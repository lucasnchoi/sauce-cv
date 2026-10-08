import json

import numpy as np
import pytest
import torch

from tracker.rink_model.data import Sample
from tracker.rink_model.train import TrainSettings, TrainingTooSlow, pck, predict_samples, train

TINY = (32, 64)


def sample(clip_id, frame, **points):
    pts = np.zeros((33, 3))
    for k, (x, y) in points.items():
        pts[int(k)] = (x, y, 1.0)
    return Sample(clip_id, frame, pts)


def test_pck_counts_hits_within_the_limit_and_unpredicted_points_as_misses():
    samples = [sample("A", 0, **{"10": (100.0, 100.0), "27": (200.0, 50.0), "0": (5.0, 5.0)})]
    pred = np.zeros((33, 3))
    pred[10] = (104.0, 103.0, 0.9)        # 5 px off: hit
    pred[27] = (215.0, 50.0, 0.8)         # 15 px off: miss
    # keypoint 0 not predicted (conf 0): miss, not counted in mean_px
    result = pck({("A", 0): pred}, samples, px=10.0)
    assert result == {"pck": pytest.approx(1 / 3), "mean_px": pytest.approx(10.0), "n_points": 3, "n_predicted": 2}
    assert pck({}, samples, px=10.0) == {"pck": 0.0, "mean_px": None, "n_points": 3, "n_predicted": 0}


def test_settings_round_trip_and_cast():
    s = TrainSettings.from_dict({"epochs": "3", "lr": "0.01"})
    assert s.epochs == 3 and s.lr == 0.01 and s.batch_size == 4
    assert TrainSettings.from_dict(TrainSettings().as_dict()) == TrainSettings()
    with pytest.raises(ValueError, match="unknown"):
        TrainSettings.from_dict({"bogus": 1})


def test_one_tiny_epoch_writes_weights_and_a_run_record(make_sequence, tmp_path):
    seq = make_sequence(name="A_001", n_frames=4)
    sequences = {"A_001": seq}
    samples = [sample("A_001", f, **{"10": (30.0, 20.0), "27": (40.0, 30.0), "0": (10.0, 10.0), "8": (50.0, 5.0)}) for f in range(4)]
    settings = TrainSettings(epochs=1, batch_size=2, val_share=0.25, seed=1)
    lines = []
    record = train(samples[:3], samples[3:], sequences, tmp_path / "run", settings, device=torch.device("cpu"),
                   pretrained=False, input_hw=TINY, log=lines.append)
    assert (tmp_path / "run" / "last.pt").is_file() and (tmp_path / "run" / "best.pt").is_file()
    saved = json.loads((tmp_path / "run" / "run.json").read_text())
    assert saved == record
    assert saved["settings"]["epochs"] == 1 and saved["n_train"] == 3 and saved["n_val"] == 1
    assert saved["val_frames"] == [["A_001", 3]] and saved["history"][0]["epoch"] == 1
    assert "loss" in saved["history"][0] and "pck" in saved["history"][0] and saved["stopped"] is None
    assert saved["best_epoch"] == 1 and "torch" in saved["packages"]
    assert "vis_recall" in saved["history"][0] and saved["input_hw"] == [32, 64]
    assert len(lines) == 1 and json.loads(lines[0])["epoch"] == 1


def test_predict_samples_returns_one_array_per_sample(make_sequence, tmp_path):
    from tracker.rink_model.model import build_model
    seq = make_sequence(name="A_001", n_frames=2)
    model = build_model(pretrained=False).eval()
    out = predict_samples(model, [sample("A_001", 1, **{"10": (1.0, 1.0)})], {"A_001": seq}, torch.device("cpu"), input_hw=TINY)
    assert set(out) == {("A_001", 1)} and out[("A_001", 1)].shape == (33, 3)


def test_a_slow_first_epoch_stops_with_the_record_written(make_sequence, tmp_path, monkeypatch):
    seq = make_sequence(name="A_001", n_frames=2)
    samples = [sample("A_001", f, **{"10": (30.0, 20.0), "27": (40.0, 30.0)}) for f in range(2)]
    settings = TrainSettings(epochs=2, batch_size=2, max_epoch_minutes=0.0)
    with pytest.raises(TrainingTooSlow, match="minutes"):
        train(samples, [], {"A_001": seq}, tmp_path / "run", settings, device=torch.device("cpu"), pretrained=False, input_hw=TINY, log=lambda s: None)
    saved = json.loads((tmp_path / "run" / "run.json").read_text())
    assert saved["stopped"].startswith("first epoch took") and len(saved["history"]) == 1


def test_check_frames_fits_transforms_from_detections_and_measures_labelled_points(make_sequence):
    from fakes import FakeKeypointDetector
    from tracker.rink import template_points
    from tracker.rink_model.train import check_frames
    from tracker.stages.rink_transform import project
    H_true = np.array([[0.25, 0.0, 32.0], [0.0, -0.25, 24.0], [0.0, 0.0, 1.0]])      # feet -> px inside a 64x48 frame
    ids = [0, 2, 4, 6, 8, 10, 11, 13, 15, 17, 19, 23]
    px = project(H_true, template_points()[ids])
    detector = FakeKeypointDetector({k: (float(x), float(y), 0.9) for k, (x, y) in zip(ids, px)})
    seq = make_sequence(name="A_001", n_frames=2)
    labelled = Sample("A_001", 1, np.array([[*p, 1.0] if k in ids[:4] else [0.0, 0.0, 0.0] for k, p in enumerate(project(H_true, template_points()))]))
    result = check_frames(detector, [labelled], {"A_001": seq})
    assert result["n_frames"] == 1 and result["n_valid"] == 1 and result["valid_share"] == 1.0
    assert result["n_points"] == 4 and result["median_m"] < 0.05 and result["pck"] == 1.0
