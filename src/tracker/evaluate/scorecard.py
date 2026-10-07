"""Builds the scorecard: each stage's measures overall and per clip, with the run's settings.

The tracking measure is IDF1 at IoU 0.5 (the spec's bar). IDF1 at 0.4 and
0.3 are reported beside it, never instead of it: the gap between them marks
how far boxes sit from the ground truth (decision point 1b). The by-frame-rate
table is secondary: the held-out set has no 30 fps clip, so only training
clips say anything about 30 fps.
"""
from __future__ import annotations

import json
from pathlib import Path

from tracker.evaluate.tracking import TrackingCounts

# First-version accuracy bars from the spec, section 7.
BARS = {"tracking_idf1": 0.85}
MEASURE = 0.5
EXTRA_THRESHOLDS = (0.4, 0.3)


def _normalise(per_clip: dict) -> dict[str, dict[float, TrackingCounts]]:
    return {clip: (v if isinstance(v, dict) else {MEASURE: v}) for clip, v in per_clip.items()}


def _pool(items: list[dict[float, TrackingCounts]]) -> dict[float, TrackingCounts]:
    pooled: dict[float, TrackingCounts] = {}
    for by_threshold in items:
        for t, counts in by_threshold.items():
            pooled[t] = counts if t not in pooled else pooled[t] + counts
    return pooled


def _entry(by_threshold: dict[float, TrackingCounts], n_clips: int | None = None) -> dict:
    counts = by_threshold[MEASURE]
    entry = {
        "idf1": counts.idf1,
        "precision": counts.precision,
        "recall": counts.recall,
        "idtp": counts.idtp,
        "n_gt": counts.n_gt,
        "n_pred": counts.n_pred,
        "tp": counts.tp,
        "fp": counts.fp,
        "fn": counts.fn,
        "meets_bar": counts.idf1 >= BARS["tracking_idf1"],
    }
    for t in EXTRA_THRESHOLDS:
        entry[f"idf1_at_{t}"] = by_threshold[t].idf1 if t in by_threshold else None
    if n_clips is not None:
        entry["n_clips"] = n_clips
    return entry


def read_fps_by_clip(run_dir, clip_ids) -> dict[str, float]:
    """Frame rate per clip from each clip.json that exists; clips without one are left out."""
    from tracker.pipeline import read_clip_info          # local import: pipeline imports this module's siblings

    out = {}
    for clip_id in clip_ids:
        path = Path(run_dir) / clip_id / "clip.json"
        if path.is_file():
            out[clip_id] = read_clip_info(path).fps
    return out


def build_scorecard(per_clip: dict, failed: dict[str, str], run: dict,
                    fps_by_clip: dict[str, float] | None = None, sections: dict[str, dict] | None = None) -> dict:
    per_clip = _normalise(per_clip)
    overall = _entry(_pool(list(per_clip.values()))) if per_clip else None
    groups: dict[str, list] = {}
    for clip, by_threshold in per_clip.items():
        fps = (fps_by_clip or {}).get(clip)
        key = f"{fps:g}" if fps is not None else "unknown"
        groups.setdefault(key, []).append(by_threshold)
    by_frame_rate = {"secondary": True}
    for key in sorted(groups, key=lambda k: (k == "unknown", k)):
        by_frame_rate[key] = _entry(_pool(groups[key]), n_clips=len(groups[key]))
    card = {
        "run": run,
        "bars": dict(BARS),
        "tracking": {
            "overall": overall,
            "per_clip": {clip: _entry(per_clip[clip]) for clip in sorted(per_clip)},
            "by_frame_rate": by_frame_rate,
        },
        "failed_clips": dict(failed),
    }
    for name, section in (sections or {}).items():
        card[name] = section
    return card


def _pct(value) -> str:
    return "—" if value is None else f"{value:.1%}"


def _row(name: str, entry: dict) -> str:
    verdict = "yes" if entry["meets_bar"] else "no"
    return (
        f"| {name} | {_pct(entry['idf1'])} | {_pct(entry['idf1_at_0.4'])} | {_pct(entry['idf1_at_0.3'])} "
        f"| {_pct(entry['precision'])} | {_pct(entry['recall'])} "
        f"| {entry['n_gt']} | {entry['fp']} | {entry['fn']} | {verdict} |"
    )


def _markdown(scorecard: dict) -> str:
    lines = ["# Scorecard", ""]
    tracking = scorecard["tracking"]
    lines.append(f"Tracking bar: IDF1 at IoU 0.5 of {scorecard['bars']['tracking_idf1']:.1%} or better. "
                 "IDF1 at 0.4 and 0.3 are shown beside it; the gap marks boxes that sit near but not on the ground truth.")
    lines.append("")
    if tracking["overall"] is None:
        lines.append("No clip could be scored.")
    else:
        lines.append("| Clip | IDF1 | IDF1@0.4 | IDF1@0.3 | Precision | Recall | Ground-truth boxes | False boxes | Missed boxes | Meets bar |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        lines.append(_row("**Overall**", tracking["overall"]))
        for clip, entry in tracking["per_clip"].items():
            lines.append(_row(clip, entry))
        rates = {k: v for k, v in tracking["by_frame_rate"].items() if k != "secondary"}
        if rates:
            lines += ["", "## By frame rate (secondary)", "",
                      "The held-out set has no 30 fps clip, so only training-clip rows say anything about 30 fps.", "",
                      "| Frame rate | Clips | IDF1 | IDF1@0.4 | IDF1@0.3 | Precision | Recall |", "|---|---|---|---|---|---|---|"]
            for key, entry in rates.items():
                lines.append(f"| {key} | {entry['n_clips']} | {_pct(entry['idf1'])} | {_pct(entry['idf1_at_0.4'])} | "
                             f"{_pct(entry['idf1_at_0.3'])} | {_pct(entry['precision'])} | {_pct(entry['recall'])} |")
    if scorecard["failed_clips"]:
        lines += ["", "## Clips that failed", ""]
        for clip, message in sorted(scorecard["failed_clips"].items()):
            lines.append(f"- {clip}: {message}")
    for name, section in scorecard.items():
        if name in ("run", "bars", "tracking", "failed_clips"):
            continue
        lines += ["", f"## {name}", "", "```json", json.dumps(section, indent=2), "```"]
    lines += ["", "## Run", "", "```json", json.dumps(scorecard["run"], indent=2), "```", ""]
    return "\n".join(lines)


def write_scorecard(scorecard: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scorecard.json").write_text(json.dumps(scorecard, indent=2) + "\n")
    (out_dir / "scorecard.md").write_text(_markdown(scorecard))
