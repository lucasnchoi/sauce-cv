"""Runs the stages over clips and scores a finished run.

Stages talk only through the tables written here. A stage that fails on one
clip stops that clip; other clips continue.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from tracker.datasets.mot import load_mot_ground_truth
from tracker.evaluate.tracking import TrackingCounts, score_tracking
from tracker.overlay import render_overlay
from tracker.stages.assemble import assemble_tracking
from tracker.stages.detect_track import bytetrack_factory, run_detect_track
from tracker.stages.footage_filter import passthrough_frames
from tracker.tables import read_table, write_table
from tracker.video import Clip

PACKAGES = ["rfdetr", "trackers", "supervision", "torch", "polars", "opencv-python", "numpy", "scipy"]


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


def run_clip(clip_path, run_dir, detector, tracker_factory=bytetrack_factory,
             fps: float | None = None, overlay: bool = True) -> str:
    clip_path = Path(clip_path)
    fallback_id = clip_path.name if clip_path.is_dir() or not clip_path.suffix else clip_path.stem
    with _stage("open clip", fallback_id):
        clip = Clip(clip_path, fps=fps)
    clip_id = clip.info.clip_id
    out = Path(run_dir) / clip_id

    with _stage("footage filter", clip_id):
        frames = passthrough_frames(clip.info)
        write_table(frames, "frames", out / "frames.parquet")
    with _stage("detect and track", clip_id):
        tracks = run_detect_track(clip, frames, detector, tracker_factory)
        write_table(tracks, "tracks", out / "tracks.parquet")
    with _stage("assemble", clip_id):
        write_table(assemble_tracking(tracks, clip.info.fps), "tracking", out / "tracking.parquet")
    if overlay:
        with _stage("overlay", clip_id):
            render_overlay(clip, tracks, out / "overlay.mp4")
    return clip_id


def run_many(clip_paths, run_dir, detector, tracker_factory=bytetrack_factory,
             fps: float | None = None, overlay: bool = True) -> dict[str, str | None]:
    results: dict[str, str | None] = {}
    for clip_path in clip_paths:
        try:
            results[run_clip(clip_path, run_dir, detector, tracker_factory, fps, overlay)] = None
        except StageError as error:
            results[error.clip_id] = str(error)
    return results


def run_details(detector) -> dict:
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
        "tracker": "ByteTrack from the trackers library, default settings, frame rate set per clip",
        "footage_filter": "stand-in: every frame is play",
        "packages": versions,
    }


def score_run(run_dir, sequences: dict[str, Path], clip_ids: list[str]
              ) -> tuple[dict[str, TrackingCounts], dict[str, str]]:
    counts: dict[str, TrackingCounts] = {}
    failed: dict[str, str] = {}
    for clip_id in clip_ids:
        if clip_id not in sequences:
            failed[clip_id] = "no ground truth sequence with this name in the dataset"
            continue
        try:
            predicted = read_table("tracks", Path(run_dir) / clip_id / "tracks.parquet")
            truth = load_mot_ground_truth(sequences[clip_id])
            counts[clip_id] = score_tracking(truth, predicted)
        except (FileNotFoundError, ValueError) as error:      # SchemaError is a ValueError
            failed[clip_id] = str(error)
    return counts, failed
