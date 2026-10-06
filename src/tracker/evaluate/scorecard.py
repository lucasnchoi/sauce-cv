"""Builds the scorecard: each stage's measures overall and per clip, with the run's settings."""
from __future__ import annotations

import json
from pathlib import Path

from tracker.evaluate.tracking import TrackingCounts

# First-version accuracy bars from the spec, section 7.
BARS = {"tracking_idf1": 0.85}


def _entry(counts: TrackingCounts) -> dict:
    return {
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


def build_scorecard(per_clip: dict[str, TrackingCounts], failed: dict[str, str], run: dict) -> dict:
    overall = None
    if per_clip:
        total = None
        for counts in per_clip.values():
            total = counts if total is None else total + counts
        overall = _entry(total)
    return {
        "run": run,
        "bars": dict(BARS),
        "tracking": {
            "overall": overall,
            "per_clip": {clip: _entry(per_clip[clip]) for clip in sorted(per_clip)},
        },
        "failed_clips": dict(failed),
    }


def _row(name: str, entry: dict) -> str:
    verdict = "yes" if entry["meets_bar"] else "no"
    return (
        f"| {name} | {entry['idf1']:.1%} | {entry['precision']:.1%} | {entry['recall']:.1%} "
        f"| {entry['n_gt']} | {entry['fp']} | {entry['fn']} | {verdict} |"
    )


def _markdown(scorecard: dict) -> str:
    lines = ["# Scorecard", ""]
    tracking = scorecard["tracking"]
    lines.append(f"Tracking bar: IDF1 of {scorecard['bars']['tracking_idf1']:.1%} or better.")
    lines.append("")
    if tracking["overall"] is None:
        lines.append("No clip could be scored.")
    else:
        lines.append(
            "| Clip | IDF1 | Precision | Recall | Ground-truth boxes | False boxes | Missed boxes | Meets bar |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        lines.append(_row("**Overall**", tracking["overall"]))
        for clip, entry in tracking["per_clip"].items():
            lines.append(_row(clip, entry))
    if scorecard["failed_clips"]:
        lines += ["", "## Clips that failed", ""]
        for clip, message in sorted(scorecard["failed_clips"].items()):
            lines.append(f"- {clip}: {message}")
    lines += ["", "## Run", "", "```json", json.dumps(scorecard["run"], indent=2), "```", ""]
    return "\n".join(lines)


def write_scorecard(scorecard: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scorecard.json").write_text(json.dumps(scorecard, indent=2) + "\n")
    (out_dir / "scorecard.md").write_text(_markdown(scorecard))
