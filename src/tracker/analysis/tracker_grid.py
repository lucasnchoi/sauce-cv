"""Investigation 2 of Plan 2a: re-track a finished run under each ByteTrack setting and score it.

The detector never runs here. Held-out clips are never passed in by the
callers in this plan; the function itself does not know which clips are
which, so the caller is responsible for that rule.
"""
from __future__ import annotations

import json
from pathlib import Path

from tracker.evaluate.tracking import TrackingCounts
from tracker.pipeline import retrack_run, score_run
from tracker.stages.detect_track import ByteTrackSettings

FACTOR_VALUES = {
    "track_activation_threshold": [0.5, 0.6],
    "minimum_iou_threshold": [0.2, 0.3],
    "lost_track_buffer": [15, 60, 90],
    "minimum_consecutive_frames": [3, 4],
    "high_conf_det_threshold": [0.5, 0.7],
}


def one_factor_grid() -> list[ByteTrackSettings]:
    """The baseline plus every single-setting change in FACTOR_VALUES."""
    points = [ByteTrackSettings()]
    for name, values in FACTOR_VALUES.items():
        for value in values:
            points.append(ByteTrackSettings.from_dict({name: value}))
    return points


def load_grid(path) -> list[ByteTrackSettings]:
    """A JSON list of settings dicts. The baseline is always included once."""
    entries = json.loads(Path(path).read_text())
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a JSON list of settings objects")
    points = [ByteTrackSettings()]
    for index, entry in enumerate(entries):
        try:
            point = ByteTrackSettings.from_dict(entry)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{path}: entry {index}: {error}") from error
        if point not in points:
            points.append(point)
    return points


def _pooled(counts: dict[str, TrackingCounts]) -> TrackingCounts | None:
    total = None
    for c in counts.values():
        total = c if total is None else total + c
    return total


def run_grid(src_run, out_dir, points: list[ByteTrackSettings], sequences: dict, clip_ids: list[str]) -> list[dict]:
    out_dir = Path(out_dir)
    records = []
    for point in points:
        point_dir = out_dir / point.label()
        failed = {clip: message for clip, message in retrack_run(src_run, point_dir, point).items() if message}
        counts, not_scored = score_run(point_dir, sequences, clip_ids)
        for clip, message in not_scored.items():
            failed.setdefault(clip, message)
        total = _pooled({c: v[0.5] for c, v in counts.items()})
        records.append({
            "label": point.label(),
            "settings": point.as_dict(),
            "idf1": total.idf1 if total else None,
            "precision": total.precision if total else None,
            "recall": total.recall if total else None,
            "n_clips": len(counts),
            "per_clip": {clip: c[0.5].idf1 for clip, c in sorted(counts.items())},
            "failed": failed,
        })
    records.sort(key=lambda r: (r["idf1"] is not None, r["idf1"] or 0.0), reverse=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "grid.json").write_text(json.dumps(records, indent=2) + "\n")
    lines = ["# Tracker settings grid", "", f"Source run: `{src_run}`; {len(clip_ids)} clips asked for.", "",
             "| Setting | IDF1 | Precision | Recall | Clips scored | Failed |", "|---|---|---|---|---|---|"]
    for r in records:
        pct = lambda v: "—" if v is None else f"{v:.1%}"
        lines.append(f"| {r['label']} | {pct(r['idf1'])} | {pct(r['precision'])} | {pct(r['recall'])} | {r['n_clips']} | {len(r['failed'])} |")
    (out_dir / "grid.md").write_text("\n".join(lines) + "\n")
    return records
