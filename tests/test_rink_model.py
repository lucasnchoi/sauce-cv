import random

import numpy as np
import polars as pl
import pytest
import torch

from tracker.labels.rink_keypoints import frame_rows, load_labels, upsert_frame
from tracker.rink_model.data import (
    INPUT_HW, STRIDE, LabelledFrames, Sample, augment, collate, heatmap_targets, image_tensor, samples_from_labels, split_validation,
)
from tracker.rink_model.model import N_KEYPOINTS, RinkKeypointDetector, build_model, decode, load_model, pick_device, save_weights

TINY = (32, 64)      # rows, cols: heatmaps 8 x 16


def labels_with(frames):
    table = load_labels("/nonexistent")
    for (clip_id, frame), points in frames.items():
        table = upsert_frame(table, clip_id, frame, frame_rows(clip_id, frame, points))
    return table


def test_samples_keep_every_labelled_frame_in_clip_frame_order():
    labels = labels_with({("B", 2): {10: (30.0, 20.0)}, ("A", 5): {}, ("A", 1): {0: (1.0, 2.0), 27: (3.0, 4.0)}})
    samples = samples_from_labels(labels)
    assert [(s.clip_id, s.frame) for s in samples] == [("A", 1), ("A", 5), ("B", 2)]
    assert samples[1].points[:, 2].sum() == 0                                  # nothing visible: still a sample
    assert samples[0].points[27].tolist() == [3.0, 4.0, 1.0] and samples[0].points[1].tolist() == [0.0, 0.0, 0.0]


def test_validation_split_is_seeded_disjoint_and_sized():
    samples = [Sample("A", f, np.zeros((33, 3))) for f in range(10)]
    train, val = split_validation(samples, seed=1, share=0.2)
    assert len(val) == 2 and len(train) == 8
    assert {s.frame for s in train} | {s.frame for s in val} == set(range(10))
    assert [s.frame for s in val] == [s.frame for s in split_validation(samples, seed=1, share=0.2)[1]]
    assert split_validation(samples[:1], seed=1)[1] == []


def test_heatmap_targets_peak_at_visible_points_and_are_zero_for_absent_ones():
    points = np.zeros((33, 3))
    points[10] = (320.0, 180.0, 1.0)            # centre of a 640 x 360 frame
    points[27] = (700.0, 100.0, 1.0)            # off the image: treated as absent
    target = heatmap_targets(points, width=640, height=360, input_hw=TINY, stride=STRIDE)
    assert target.shape == (33, 8, 16) and target.dtype == np.float32
    r, c = np.unravel_index(target[10].argmax(), target[10].shape)
    assert (r, c) == (3, 7) or (r, c) == (4, 8)                                # cx = 320/640*16 - 0.5 = 7.5, cy = 3.5
    assert target[10].max() == pytest.approx(1.0, abs=0.15) and target[0].sum() == 0 and target[27].sum() == 0


def test_augment_moves_points_with_the_image_and_never_flips():
    image = np.full((48, 64, 3), 100, dtype=np.uint8)
    points = np.zeros((33, 3)); points[10] = (48.0, 24.0, 1.0); points[11] = (63.0, 47.0, 1.0)
    out, pts = augment(image, points, random.Random(3))
    assert out.shape == image.shape
    assert pts[10, 0] > 32 and 0 <= pts[10, 2] <= 1                           # still on the right half: no flip
    assert pts[11, 2] in (0.0, 1.0)                                             # a corner point may leave the image and become absent


def test_image_tensor_is_resized_and_normalised():
    t = image_tensor(np.full((48, 64, 3), 128, dtype=np.uint8), input_hw=TINY)
    assert t.shape == (3, 32, 64) and t.dtype == torch.float32 and abs(float(t.mean())) < 1.0


def test_dataset_yields_stacked_batches(make_sequence):
    seq = make_sequence(name="A_001", n_frames=3)
    points = np.zeros((33, 3)); points[10] = (32.0, 24.0, 1.0)
    data = LabelledFrames([Sample("A_001", 2, points), Sample("A_001", 1, points)], {"A_001": seq}, input_hw=TINY)
    image, target = data[0]
    assert image.shape == (3, 32, 64) and target.shape == (33, 8, 16)
    images, targets = collate([data[0], data[1]])
    assert images.shape == (2, 3, 32, 64) and targets.shape == (2, 33, 8, 16)


def test_untrained_model_maps_an_image_to_33_heatmaps_at_stride_4():
    model = build_model(pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 32, 64))
    assert out.shape == (1, N_KEYPOINTS, 8, 16)


def test_decode_finds_the_peak_and_maps_it_back_to_pixels():
    logits = torch.full((33, 8, 16), -8.0)
    logits[10, 3, 7] = 6.0; logits[10, 3, 8] = 2.0                             # peak with a higher right neighbour
    points = decode(logits, width=640, height=360)
    assert points[10, 2] == pytest.approx(torch.sigmoid(torch.tensor(6.0)).item())
    assert points[10, 0] == pytest.approx((7 + 0.5 + 0.25) / 16 * 640) and points[10, 1] == pytest.approx((3 + 0.5) / 8 * 360)
    assert points[0, 2] < 0.001


def test_weights_round_trip_and_the_detector_thresholds_its_rows(tmp_path):
    model = build_model(pretrained=False)
    save_weights(model, tmp_path / "w" / "best.pt", {"round": 1})
    loaded, meta = load_model(tmp_path / "w" / "best.pt", device="cpu")
    assert meta == {"round": 1} and not loaded.training
    detector = RinkKeypointDetector(tmp_path / "w" / "best.pt", threshold=0.3, device="cpu", input_hw=TINY)
    points = detector.detect(np.zeros((48, 64, 3), dtype=np.uint8))
    assert points.shape == (33, 3) and ((points[:, 2] == 0) | (points[:, 2] >= 0.3)).all()
    assert detector.name == "resnet50-heatmap-rink33" and detector.threshold == 0.3


def test_detector_uses_the_models_heatmaps(tmp_path):
    model = build_model(pretrained=False)
    save_weights(model, tmp_path / "best.pt", {})
    detector = RinkKeypointDetector(tmp_path / "best.pt", threshold=0.3, device="cpu", input_hw=TINY)
    logits = torch.full((1, 33, 8, 16), -8.0); logits[0, 5, 2, 3] = 5.0
    detector._model = lambda images: logits
    points = detector.detect(np.zeros((48, 64, 3), dtype=np.uint8))
    assert points[5, 2] == pytest.approx(torch.sigmoid(torch.tensor(5.0)).item()) and points[5, 0] == pytest.approx((3.5) / 16 * 64)
    assert (points[[k for k in range(33) if k != 5], :] == 0).all()


def test_missing_weights_name_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.pt"):
        load_model(tmp_path / "nope.pt", device="cpu")


def test_pick_device_honours_an_explicit_name():
    assert pick_device("cpu").type == "cpu" and pick_device().type in ("mps", "cpu")
