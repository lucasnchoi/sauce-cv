import numpy as np
import polars as pl
import pytest
import torch

from tracker.rink_model.data import (
    BOX_PAD_PX, LabelledFrames, Sample, augment, collate, image_tensor, samples_from_labels, split_validation, target_for,
)
from tracker.rink_model.model import N_KEYPOINTS, RinkKeypointDetector, build_model, load_model, pick_device, save_weights
from tracker.labels.rink_keypoints import frame_rows, load_labels, upsert_frame

TINY = {"min_size": 32, "max_size": 48}


def labels_with(frames: dict[tuple[str, int], dict[int, tuple[float, float]]]) -> pl.DataFrame:
    table = load_labels("/nonexistent")
    for (clip_id, frame), points in frames.items():
        table = upsert_frame(table, clip_id, frame, frame_rows(clip_id, frame, points))
    return table


def test_samples_skip_frames_with_nothing_visible_and_keep_clip_frame_order():
    labels = labels_with({("B", 2): {10: (30.0, 20.0)}, ("A", 5): {}, ("A", 1): {0: (1.0, 2.0), 27: (3.0, 4.0)}})
    samples = samples_from_labels(labels)
    assert [(s.clip_id, s.frame) for s in samples] == [("A", 1), ("B", 2)]
    assert samples[0].points.shape == (33, 3)
    assert samples[0].points[27].tolist() == [3.0, 4.0, 1.0] and samples[0].points[1].tolist() == [0.0, 0.0, 0.0]


def test_validation_split_is_seeded_disjoint_and_sized():
    samples = [Sample("A", f, np.zeros((33, 3))) for f in range(10)]
    train, val = split_validation(samples, seed=1, share=0.2)
    assert len(val) == 2 and len(train) == 8
    assert {s.frame for s in train} | {s.frame for s in val} == set(range(10))
    assert [s.frame for s in val] == [s.frame for s in split_validation(samples, seed=1, share=0.2)[1]]
    assert [s.frame for s in val] != [s.frame for s in split_validation(samples, seed=2, share=0.2)[1]]
    assert split_validation(samples[:1], seed=1)[1] == []


def test_target_box_pads_the_visible_points_and_clips_to_the_image():
    points = np.zeros((33, 3))
    points[10] = (10.0, 5.0, 1.0)
    points[27] = (600.0, 300.0, 1.0)
    target = target_for(points, width=640, height=320)
    assert target["boxes"].tolist() == [[0.0, 0.0, 620.0, 320.0]]
    assert target["labels"].tolist() == [1]
    assert target["keypoints"].shape == (1, 33, 3)
    assert target["keypoints"][0, 10].tolist() == [10.0, 5.0, 2.0] and target["keypoints"][0, 0].tolist() == [0.0, 0.0, 0.0]


def test_augment_scales_points_with_the_image_and_never_flips():
    import random
    image = np.full((48, 64, 3), 100, dtype=np.uint8)
    points = np.zeros((33, 3)); points[10] = (32.0, 24.0, 1.0)
    out, pts = augment(image, points, random.Random(3))
    scale = out.shape[1] / 64
    assert 0.9 <= scale <= 1.1 and abs(out.shape[0] / 48 - scale) < 0.03
    assert pts[10, 0] == pytest.approx(32.0 * scale, abs=0.5) and pts[10, 2] == 1.0
    assert pts[10, 0] > 0.4 * out.shape[1]                                   # still on the right half: no flip


def test_dataset_yields_tensors_and_targets(make_sequence):
    seq = make_sequence(name="A_001", n_frames=3)
    points = np.zeros((33, 3)); points[10] = (32.0, 24.0, 1.0)
    data = LabelledFrames([Sample("A_001", 2, points)], {"A_001": seq})
    image, target = data[0]
    assert image.shape == (3, 48, 64) and image.dtype == torch.float32 and float(image.max()) <= 1.0
    assert target["boxes"].shape == (1, 4)
    images, targets = collate([data[0], data[0]])
    assert len(images) == 2 and len(targets) == 2


def test_untrained_model_runs_forward_and_has_33_keypoints():
    model = build_model(pretrained=False, **TINY).eval()
    with torch.no_grad():
        out = model([torch.rand(3, 32, 48)])[0]
    assert out["keypoints"].shape[1:] == (N_KEYPOINTS, 3) and out["keypoints_scores"].shape[1:] == (N_KEYPOINTS,)


def test_weights_round_trip_and_the_detector_returns_33_rows(tmp_path):
    model = build_model(pretrained=False, **TINY)
    save_weights(model, tmp_path / "w" / "best.pt", {"round": 1})
    loaded, meta = load_model(tmp_path / "w" / "best.pt", device="cpu", **TINY)
    assert meta == {"round": 1} and not loaded.training
    detector = RinkKeypointDetector(tmp_path / "w" / "best.pt", threshold=0.3, device="cpu", **TINY)
    points = detector.detect(np.zeros((32, 48, 3), dtype=np.uint8))
    assert points.shape == (33, 3) and (0 <= points[:, 2]).all() and (points[:, 2] <= 1).all()
    assert ((points[:, 2] == 0) | (points[:, 2] >= 0.3)).all()
    assert detector.name == "keypointrcnn-resnet50-fpn-rink33" and detector.threshold == 0.3


def test_missing_weights_name_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.pt"):
        load_model(tmp_path / "nope.pt", device="cpu")


def test_pick_device_honours_an_explicit_name():
    assert pick_device("cpu").type == "cpu"
    assert pick_device().type in ("mps", "cpu")
