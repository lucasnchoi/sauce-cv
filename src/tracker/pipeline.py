"""Runs the stages over clips and scores a finished run.

Stages talk only through the tables written here. A stage that fails on one
clip stops that clip; other clips continue.
"""
from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Callable

import polars as pl

from tracker.datasets.mot import load_mot_ground_truth
from tracker.evaluate.rink import RinkCounts, score_rink
from tracker.evaluate.tracking import TrackingCounts, score_tracking
from tracker.overlay import render_overlay
from tracker.stages.assemble import assemble_tracking
from tracker.stages.detect_track import DEFAULT_BYTETRACK, ByteTrackSettings, bytetrack_factory, run_detect, run_track
from tracker.schemas import empty
from tracker.stages.footage_filter import DEFAULT_FOOTAGE_FILTER, FootageFilterSettings, keypoint_frames, passthrough_frames
from tracker.stages.rink_keypoints import run_keypoints
from tracker.stages.rink_transform import DEFAULT_RINK_TRANSFORM, RinkTransformSettings, fit_transforms
from tracker.tables import read_table, write_table
from tracker.video import Clip, ClipInfo

PACKAGES = ["rfdetr", "trackers", "supervision", "torch", "polars", "opencv-python", "numpy", "scipy", "torchvision"]


class StageError(RuntimeError):
    def __init__(self, stage: str, clip_id: str, cause: BaseException):
        super().__init__(f"stage '{stage}' failed on clip '{clip_id}': {cause}")
        self.stage = stage
        self.clip_id = clip_id


@contextmanager
def _stage(name: str, clip_id: str):
    try:
        yield
    except StageError:
        raise
    except Exception as error:
        raise StageError(name, clip_id, error) from error


def write_clip_info(info: ClipInfo, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(info), indent=2) + "\n")


def read_clip_info(path) -> ClipInfo:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"clip info not found at {path}; run the pipeline on this clip first")
    return ClipInfo(**json.loads(path.read_text()))


@dataclass
class RinkStages:
    """Stages 1–3 for a run. The detector is made inside the first clip's stage, so a missing weights file
    fails that stage (naming the path) instead of the whole command, and the model loads once."""

    make_keypoint_detector: Callable[[], object] | None = None
    footage_filter: FootageFilterSettings = DEFAULT_FOOTAGE_FILTER
    footage_mode: str = "keypoints"
    rink_transform: RinkTransformSettings = DEFAULT_RINK_TRANSFORM
    _detector: object = field(default=None, repr=False)

    def keypoint_detector(self):
        if self._detector is None and self.make_keypoint_detector is not None:
            self._detector = self.make_keypoint_detector()
        return self._detector

    def details(self) -> dict:
        """For run.json. A factory that fails (missing weights) is recorded as unavailable here; the
        per-clip stage still raises, so the run lists every clip's failure with the path."""
        detector = None
        unavailable = None
        if self.make_keypoint_detector is not None:
            try:
                detector = self.keypoint_detector()
            except Exception as error:          # noqa: BLE001 - recorded, and raised again by the stage
                detector = None
                unavailable = {"name": "unavailable", "weights": None, "threshold": None, "error": str(error)}
        uses_rule = detector is not None and self.footage_mode == "keypoints"
        if detector is None and self.make_keypoint_detector is not None:
            rink_keypoints = unavailable
        elif detector is None:
            rink_keypoints = None
        else:
            rink_keypoints = {"name": detector.name, "weights": str(detector.weights_path), "threshold": detector.threshold}
        return {
            "rink_keypoints": rink_keypoints,
            "footage_filter": {"mode": "keypoints" if uses_rule else "passthrough",
                               "settings": self.footage_filter.as_dict() if uses_rule else None},
            "rink_transform": self.rink_transform.as_dict(),
        }


def _clip_id_of(path: Path) -> str:
    return path.name if path.is_dir() or not path.suffix else path.stem


def run_clip(clip_path, run_dir, detector, tracker_factory=bytetrack_factory,
             fps: float | None = None, overlay: bool = True, rink: RinkStages | None = None) -> str:
    clip_path = Path(clip_path)
    fallback_id = _clip_id_of(clip_path)
    with _stage("open clip", fallback_id):
        clip = Clip(clip_path, fps=fps)
        clip_id = clip.info.clip_id
        out = Path(run_dir) / clip_id
        write_clip_info(clip.info, out / "clip.json")
    keypoints = None
    if rink is not None and rink.make_keypoint_detector is not None:
        with _stage("rink keypoints", clip_id):
            keypoints = run_keypoints(clip, rink.keypoint_detector())
            write_table(keypoints, "keypoints", out / "keypoints.parquet")
    with _stage("footage filter", clip_id):
        if keypoints is not None and rink.footage_mode == "keypoints":
            frames = keypoint_frames(keypoints, clip.info, rink.footage_filter)
        else:
            frames = passthrough_frames(clip.info)
        write_table(frames, "frames", out / "frames.parquet")
    with _stage("detect", clip_id):
        detections = run_detect(clip, frames, detector)
        write_table(detections, "detections", out / "detections.parquet")
    with _stage("track", clip_id):
        tracks = run_track(detections, frames, clip.info.fps, tracker_factory)
        write_table(tracks, "tracks", out / "tracks.parquet")
    with _stage("rink transform", clip_id):
        settings = rink.rink_transform if rink is not None else DEFAULT_RINK_TRANSFORM
        transforms = fit_transforms(keypoints, frames, clip.info.fps, settings) if keypoints is not None else empty("rink_transforms")
        write_table(transforms, "rink_transforms", out / "rink_transforms.parquet")
    with _stage("assemble", clip_id):
        tracking = assemble_tracking(tracks, frames, transforms)
        write_table(tracking, "tracking", out / "tracking.parquet")
    if overlay:
        with _stage("overlay", clip_id):
            render_overlay(clip, tracks, out / "overlay.mp4", tracking=tracking)
    return clip_id


def run_many(clip_paths, run_dir, detector, tracker_factory=bytetrack_factory,
             fps: float | None = None, overlay: bool = True, rink: RinkStages | None = None) -> dict[str, str | None]:
    paths = [Path(p) for p in clip_paths]
    seen: dict[str, Path] = {}
    for path in paths:
        clip_id = _clip_id_of(path)
        if clip_id in seen:
            raise ValueError(f"two clips would share the id '{clip_id}': {seen[clip_id]} and {path}")
        seen[clip_id] = path
    results: dict[str, str | None] = {}
    for clip_path in paths:
        try:
            results[run_clip(clip_path, run_dir, detector, tracker_factory, fps, overlay, rink)] = None
        except StageError as error:
            results[error.clip_id] = str(error)
    return results


def retrack_run(src_run, out_run, settings: ByteTrackSettings) -> dict[str, str | None]:
    """Re-link the cached detections of a finished run under other tracker settings.

    The detector is not run again, so every difference in the result comes
    from the tracker. No overlay is written.
    """
    src_run, out_run = Path(src_run), Path(out_run)
    if src_run.resolve() == out_run.resolve():
        raise ValueError(f"retrack output is the same folder as its source: {src_run}")
    source_file = src_run / "run.json"
    try:
        details = json.loads(source_file.read_text())["run"] if source_file.is_file() else {}
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"cannot read {source_file}: {error}") from error
    factory = settings.factory()
    results: dict[str, str | None] = {}
    for clip_dir in sorted(p for p in src_run.iterdir() if p.is_dir()):
        clip_id = clip_dir.name
        out = out_run / clip_id
        try:
            with _stage("read cached run", clip_id):
                info = read_clip_info(clip_dir / "clip.json")
                frames = read_table("frames", clip_dir / "frames.parquet")
                detections = read_table("detections", clip_dir / "detections.parquet")
                transforms_file = clip_dir / "rink_transforms.parquet"
                transforms = read_table("rink_transforms", transforms_file) if transforms_file.is_file() else empty("rink_transforms")
                write_clip_info(info, out / "clip.json")
                for table_file in sorted(clip_dir.glob("*.parquet")):
                    if table_file.name not in ("tracks.parquet", "tracking.parquet"):
                        shutil.copyfile(table_file, out / table_file.name)   # every table the tracker does not produce
            with _stage("track", clip_id):
                tracks = run_track(detections, frames, info.fps, factory)
                write_table(tracks, "tracks", out / "tracks.parquet")
            with _stage("assemble", clip_id):
                write_table(assemble_tracking(tracks, frames, transforms), "tracking", out / "tracking.parquet")
            results[clip_id] = None
        except StageError as error:
            results[clip_id] = str(error)

    earlier = details.get("tracker")                 # Plan 1 runs recorded a plain string here
    details["tracker"] = {**(earlier if isinstance(earlier, dict) else {}), "name": "bytetrack", "library": "trackers",
                          "settings": settings.as_dict(), "frame_rate": "set per clip"}
    details["retracked_from"] = str(src_run)
    details["retracked_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out_run.mkdir(parents=True, exist_ok=True)
    (out_run / "run.json").write_text(json.dumps({"run": details, "clips": results}, indent=2) + "\n")
    return results


def run_details(detector, tracker_settings: ByteTrackSettings = DEFAULT_BYTETRACK, rink: RinkStages | None = None) -> dict:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return {
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "detector": detector.name,
        "detector_threshold": getattr(detector, "threshold", None),
        "tracker": {
            "name": "bytetrack",
            "library": "trackers",
            "settings": tracker_settings.as_dict(),
            "frame_rate": "set per clip",
        },
        **(rink.details() if rink is not None else RinkStages().details()),
        "packages": versions,
    }


THRESHOLDS = (0.5, 0.4, 0.3)


def score_run(run_dir, sequences: dict[str, Path], clip_ids: list[str], thresholds=THRESHOLDS
              ) -> tuple[dict[str, dict[float, TrackingCounts]], dict[str, str]]:
    counts: dict[str, dict[float, TrackingCounts]] = {}
    failed: dict[str, str] = {}
    for clip_id in clip_ids:
        if clip_id not in sequences:
            failed[clip_id] = "no ground truth sequence with this name in the dataset"
            continue
        try:
            predicted = read_table("tracks", Path(run_dir) / clip_id / "tracks.parquet")
            truth = load_mot_ground_truth(sequences[clip_id])
            counts[clip_id] = {t: score_tracking(truth, predicted, iou_threshold=t) for t in thresholds}
        except (FileNotFoundError, ValueError, pl.exceptions.PolarsError) as error:   # SchemaError is a ValueError
            failed[clip_id] = str(error)
    return counts, failed


def score_rink_run(run_dir, clip_ids: list[str], labels: pl.DataFrame) -> tuple[dict[str, RinkCounts], dict[str, str]]:
    """Rink-transform counts per clip from the run's frames and rink_transforms tables and the committed labels."""
    counts: dict[str, RinkCounts] = {}
    failed: dict[str, str] = {}
    for clip_id in clip_ids:
        clip_dir = Path(run_dir) / clip_id
        try:
            frames = read_table("frames", clip_dir / "frames.parquet")
            transforms = read_table("rink_transforms", clip_dir / "rink_transforms.parquet")
            counts[clip_id] = score_rink(labels.filter(pl.col("clip_id") == clip_id), frames, transforms)
        except (FileNotFoundError, ValueError, pl.exceptions.PolarsError) as error:
            failed[clip_id] = str(error)
    return counts, failed
