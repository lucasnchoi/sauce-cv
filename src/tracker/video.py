"""Reads frames from a dataset sequence folder, an image folder or a video file.

The clip identifier is the folder or file name. Names and lengths stored inside
metadata files are ignored, because the open datasets contain mismatches.
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class ClipError(RuntimeError):
    """A clip could not be opened or read."""


@dataclass(frozen=True)
class ClipInfo:
    clip_id: str
    fps: float
    n_frames: int
    width: int
    height: int


def _seqinfo_fps(path: Path) -> float | None:
    parser = configparser.ConfigParser()
    parser.read(path)
    if parser.has_option("Sequence", "frameRate"):
        return parser.getfloat("Sequence", "frameRate")
    return None


class Clip:
    def __init__(self, path: Path, clip_id: str | None = None, fps: float | None = None):
        path = Path(path)
        if not path.exists():
            raise ClipError(f"clip path does not exist: {path}")
        self._images: list[Path] = []
        self._video_path: Path | None = None

        if path.is_dir():
            name = clip_id or path.name
            image_dir = path / "img1" if (path / "img1").is_dir() else path
            self._images = sorted(
                p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
            )
            if not self._images:
                raise ClipError(f"clip '{name}': no image files in {image_dir}")
            first = cv2.imread(str(self._images[0]))
            if first is None:
                raise ClipError(
                    f"clip '{name}': frame 0 ({self._images[0].name}) could not be read"
                )
            height, width = first.shape[:2]
            n_frames = len(self._images)
            seqinfo = path / "seqinfo.ini"
            found_fps = _seqinfo_fps(seqinfo) if seqinfo.is_file() else None
        else:
            name = clip_id or path.stem
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise ClipError(f"clip '{name}': could not open video {path}")
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            found_fps = capture.get(cv2.CAP_PROP_FPS) or None
            capture.release()
            self._video_path = path

        chosen_fps = fps if fps is not None else found_fps
        if chosen_fps is None or chosen_fps <= 0:
            raise ClipError(
                f"clip '{name}': no usable frame rate; pass fps or provide seqinfo.ini"
            )
        self.info = ClipInfo(
            clip_id=name, fps=float(chosen_fps), n_frames=n_frames, width=width, height=height
        )

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield (0-based frame index, BGR image) for every frame in order."""
        if self._video_path is None:
            for index, image_path in enumerate(self._images):
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ClipError(
                        f"clip '{self.info.clip_id}': frame {index} "
                        f"({image_path.name}) could not be read"
                    )
                yield index, image
            return
        capture = cv2.VideoCapture(str(self._video_path))
        try:
            index = 0
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                yield index, image
                index += 1
        finally:
            capture.release()
