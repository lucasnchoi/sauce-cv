from pathlib import Path

import cv2
import pytest

pytestmark = pytest.mark.slow

DATASET = Path("data/raw/VIP-HTD/mot-challenge-format")
WEIGHTS = Path("data/models/rink-keypoints/r1/best.pt")


def test_trained_rink_model_finds_points_on_a_real_frame():
    from tracker.rink_model.model import RinkKeypointDetector
    frames = sorted(DATASET.glob("*/*/img1/000001.jpg"))
    if not frames or not WEIGHTS.is_file():
        pytest.skip("VIP-HTD or the round-1 weights are not on disk")
    points = RinkKeypointDetector(WEIGHTS, threshold=0.3).detect(cv2.imread(str(frames[0])))
    assert points.shape == (33, 3) and (points[:, 2] > 0).sum() >= 4
