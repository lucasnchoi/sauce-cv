from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def make_sequence(tmp_path):
    """Build a dataset-style sequence folder and return its path."""

    def _make(name="GAME_A_001", n_frames=5, fps=30, width=64, height=48,
              gt_rows=None, parent=None) -> Path:
        seq = Path(parent or tmp_path) / name
        (seq / "img1").mkdir(parents=True)
        for i in range(n_frames):
            image = np.full((height, width, 3), 40 + 10 * i, dtype=np.uint8)
            cv2.imwrite(str(seq / "img1" / f"{i + 1:06d}.jpg"), image)
        # name and seqLength are wrong on purpose: readers must not trust them.
        (seq / "seqinfo.ini").write_text(
            "[Sequence]\n"
            f"name={name}_WRONG\n"
            "imDir=img1\n"
            f"frameRate={fps}\n"
            f"seqLength={n_frames + 100}\n"
            f"imWidth={width}\n"
            f"imHeight={height}\n"
            "imExt=.jpg\n"
        )
        if gt_rows is not None:
            (seq / "gt").mkdir()
            (seq / "gt" / "gt.txt").write_text(
                "".join(f"{f},{i},{x},{y},{w},{h},1,1,1\n" for f, i, x, y, w, h in gt_rows)
            )
        return seq

    return _make
