from pathlib import Path

import cv2
import pytest

from tracker.stages.detect_track import RfDetrPersonDetector

pytestmark = pytest.mark.slow

DATASET = Path("data/raw/VIP-HTD/mot-challenge-format")


def test_person_detector_finds_players_on_a_real_frame():
    frames = sorted(DATASET.glob("*/*/img1/000001.jpg"))
    if not frames:
        pytest.skip("VIP-HTD is not downloaded")
    image = cv2.imread(str(frames[0]))
    found = RfDetrPersonDetector(threshold=0.5).detect(image)
    assert len(found) >= 3
    assert found.xyxy.shape[1] == 4
    assert (found.xyxy[:, 2] > found.xyxy[:, 0]).all()
    assert (found.xyxy[:, 3] > found.xyxy[:, 1]).all()
