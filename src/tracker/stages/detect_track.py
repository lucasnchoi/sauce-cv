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


BOX_COLUMNS = ["x1", "y1", "x2", "y2"]


def _play_segments(frames: pl.DataFrame, clip_id: str) -> dict[int, int]:
    """Map play frame -> segment id, after checking the table belongs to this clip."""
    frames = validate(frames, "frames")
    other = sorted(set(frames["clip_id"].unique().to_list()) - {clip_id})
    if other:
        raise ValueError(f"frames table is for clip(s) {other}, not '{clip_id}'")
    play = frames.filter(pl.col("is_play")).sort("frame")
    return dict(zip(play["frame"].to_list(), play["segment_id"].to_list()))


def _rows_from(found: sv.Detections, clip_id: str, frame: int) -> list[dict]:
    if len(found) == 0:
        return []
    confidences = found.confidence if found.confidence is not None else np.ones(len(found))
    return [
        {
            "clip_id": clip_id, "frame": frame,
            "x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3]),
            "conf": float(conf),
        }
        for box, conf in zip(found.xyxy, confidences)
    ]


def _detections_from(part: pl.DataFrame | None) -> sv.Detections:
    if part is None or part.height == 0:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=part.select(BOX_COLUMNS).to_numpy().astype(float),
        confidence=part["conf"].to_numpy().astype(float),
        class_id=np.zeros(part.height, dtype=int),
    )


def run_detect(clip, frames: pl.DataFrame, detector: Detector) -> pl.DataFrame:
    """Run the detector on every play frame; one row per box found."""
    clip_id = clip.info.clip_id
    segment_of = _play_segments(frames, clip_id)
    rows: list[dict] = []
    for index, image in clip.frames():
        if index not in segment_of:
            continue
        rows.extend(_rows_from(detector.detect(image), clip_id, index))
    return validate(from_rows("detections", rows), "detections")


def run_track(
    detections: pl.DataFrame,
    frames: pl.DataFrame,
    fps: float,
    tracker_factory: TrackerFactory = bytetrack_factory,
) -> pl.DataFrame:
    """Link cached detections into tracks, one tracker per segment.

    Every play frame is fed to the tracker, including frames with no
    detections, so lost-track bookkeeping advances exactly as it did live.
    """
    detections = validate(detections, "detections")
    frames = validate(frames, "frames")
    clip_ids = set(frames["clip_id"].unique().to_list())
    clip_id = next(iter(clip_ids)) if len(clip_ids) == 1 else None
    if clip_id is None:
        raise ValueError(f"frames table must hold exactly one clip, found {sorted(clip_ids)}")
    other = sorted(set(detections["clip_id"].unique().to_list()) - {clip_id})
    if other:
        raise ValueError(f"detections table is for clip(s) {other}, not '{clip_id}'")
    segment_of = _play_segments(frames, clip_id)
    by_frame = {
        (key[0] if isinstance(key, tuple) else key): part
        for key, part in detections.partition_by("frame", as_dict=True).items()
    }

    rows: list[dict] = []
    tracker = None
    current_segment = None
    id_offset = 0
    next_free_id = 0
    for index, segment in segment_of.items():
        if tracker is None or segment != current_segment:
            tracker = tracker_factory(fps)
            current_segment = segment
            id_offset = next_free_id
        tracked = tracker.update(_detections_from(by_frame.get(index)))
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


def run_detect_track(
    clip,
    frames: pl.DataFrame,
    detector: Detector,
    tracker_factory: TrackerFactory = bytetrack_factory,
) -> pl.DataFrame:
    """Detect then track, as one step. Equivalent to run_track(run_detect(...))."""
    detections = run_detect(clip, frames, detector)
    return run_track(detections, frames, clip.info.fps, tracker_factory)
