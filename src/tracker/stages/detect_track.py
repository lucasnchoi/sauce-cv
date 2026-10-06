"""Stage 4: find people in each play frame and link them into tracks within a segment."""
from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import polars as pl
import supervision as sv

from tracker.schemas import from_rows, validate


class Detector(Protocol):
    name: str

    def detect(self, image_bgr: np.ndarray) -> sv.Detections: ...


class Tracker(Protocol):
    def update(self, detections: sv.Detections) -> sv.Detections: ...


TrackerFactory = Callable[[float], Tracker]


def bytetrack_factory(fps: float) -> Tracker:
    from trackers import ByteTrackTracker

    return ByteTrackTracker(frame_rate=fps)


class RfDetrPersonDetector:
    """RF-DETR Medium with its pretrained weights, keeping only the 'person' class.

    The pretrained model cannot tell players, goalies and referees apart, so the
    stage labels every track 'player' until a fine-tuned detector replaces this.
    """

    name = "rfdetr-medium-coco-person"

    def __init__(self, threshold: float = 0.5):
        from rfdetr import RFDETRMedium
        from rfdetr.assets.coco_classes import COCO_CLASSES

        self.threshold = threshold
        self._model = RFDETRMedium()
        self._person_ids = [i for i, label in COCO_CLASSES.items() if label == "person"]

    def detect(self, image_bgr: np.ndarray) -> sv.Detections:
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        found = self._model.predict(rgb, threshold=self.threshold, include_source_image=False)
        return found[np.isin(found.class_id, self._person_ids)]


def run_detect_track(
    clip,
    frames: pl.DataFrame,
    detector: Detector,
    tracker_factory: TrackerFactory = bytetrack_factory,
) -> pl.DataFrame:
    frames = validate(frames, "frames")
    clip_id = clip.info.clip_id
    other = sorted(set(frames["clip_id"].unique().to_list()) - {clip_id})
    if other:
        raise ValueError(f"frames table is for clip(s) {other}, not '{clip_id}'")
    play = frames.filter(pl.col("is_play"))
    segment_of = dict(zip(play["frame"].to_list(), play["segment_id"].to_list()))

    rows: list[dict] = []
    tracker = None
    current_segment = None
    id_offset = 0
    next_free_id = 0
    for index, image in clip.frames():
        if index not in segment_of:
            continue
        segment = segment_of[index]
        if tracker is None or segment != current_segment:
            tracker = tracker_factory(clip.info.fps)
            current_segment = segment
            id_offset = next_free_id
        tracked = tracker.update(detector.detect(image))
        if tracked.tracker_id is None or len(tracked) == 0:
            continue
        confidences = (
            tracked.confidence if tracked.confidence is not None else np.ones(len(tracked))
        )
        for box, track_id, conf in zip(tracked.xyxy, tracked.tracker_id, confidences):
            if track_id < 0:                  # not yet confirmed by the tracker
                continue
            global_id = id_offset + int(track_id)
            next_free_id = max(next_free_id, global_id + 1)
            rows.append({
                "clip_id": clip_id,
                "frame": index,
                "track_id": global_id,
                "class": "player",
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
                "conf": float(conf),
            })
    return validate(from_rows("tracks", rows), "tracks")
