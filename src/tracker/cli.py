"""Command line: choose the held-out split, run the pipeline, score a run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tracker.analysis.box_offset import diagnose_clip, draw_comparison
from tracker.analysis.tracker_grid import load_grid, one_factor_grid, run_grid
from tracker.datasets.mot import list_sequences, load_mot_ground_truth
from tracker.datasets.split import DEFAULT_SEED, choose_heldout, load_split, save_split
from tracker.evaluate.scorecard import build_scorecard, write_scorecard
from tracker.pipeline import retrack_run, run_details, run_many, score_run
from tracker.stages.detect_track import ByteTrackSettings
from tracker.tables import read_table
from tracker.video import Clip, ClipError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracker", description="Turn hockey broadcast clips into tracking tables."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    split = commands.add_parser("split", help="choose and record the held-out clips")
    split.add_argument("--dataset", type=Path, required=True)
    split.add_argument("--out", type=Path, required=True)
    split.add_argument("--seed", type=int, default=DEFAULT_SEED)
    split.add_argument("--heldout", type=int, default=5)

    run = commands.add_parser("run", help="run the pipeline on clips")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--clip", type=Path, action="append", default=[])
    run.add_argument("--dataset", type=Path)
    run.add_argument("--split", type=Path)
    run.add_argument("--subset", choices=["heldout", "train"], default="heldout")
    run.add_argument("--threshold", type=float, default=0.5)
    run.add_argument("--fps", type=float)
    run.add_argument("--no-overlay", action="store_true")
    run.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                     help="a ByteTrack setting, e.g. --set lost_track_buffer=60 (repeatable)")

    retrack = commands.add_parser("retrack", help="re-link a finished run's detections with other tracker settings")
    retrack.add_argument("--run", type=Path, required=True, help="a run that holds detections.parquet per clip")
    retrack.add_argument("--out", type=Path, required=True)
    retrack.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    score = commands.add_parser("score", help="score a finished run against ground truth")
    score.add_argument("--run", type=Path, required=True)
    score.add_argument("--dataset", type=Path, required=True)
    score.add_argument("--split", type=Path, required=True)
    score.add_argument("--subset", choices=["heldout", "train"], default="heldout")

    diagnose = commands.add_parser("diagnose-boxes", help="compare tracked boxes, raw detections and ground truth")
    diagnose.add_argument("--run", type=Path, required=True)
    diagnose.add_argument("--dataset", type=Path, required=True)
    diagnose.add_argument("--split", type=Path, required=True)
    diagnose.add_argument("--subset", choices=["heldout", "train"], default="heldout")
    diagnose.add_argument("--out", type=Path, required=True)
    diagnose.add_argument("--frames", type=int, default=10, help="comparison frames to draw per --draw clip")
    diagnose.add_argument("--draw", action="append", default=[], metavar="CLIP_ID")

    grid = commands.add_parser("grid", help="score a finished run under each ByteTrack setting (training clips only)")
    grid.add_argument("--run", type=Path, required=True)
    grid.add_argument("--dataset", type=Path, required=True)
    grid.add_argument("--split", type=Path, required=True)
    grid.add_argument("--subset", choices=["heldout", "train"], default="train")
    grid.add_argument("--out", type=Path, required=True)
    source = grid.add_mutually_exclusive_group(required=True)
    source.add_argument("--grid", type=Path, help="JSON list of settings objects")
    source.add_argument("--one-factor", action="store_true", help="baseline plus each single-setting change")
    return parser


def _usage_error(message: str) -> int:
    print(f"tracker: {message}", file=sys.stderr)
    return 2


def _settings(pairs: list[str]) -> ByteTrackSettings:
    values = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"setting '{pair}' is not KEY=VALUE")
        values[key] = value
    return ByteTrackSettings.from_dict(values)


def _split(args) -> int:
    clips = sorted(list_sequences(args.dataset))
    held = choose_heldout(clips, n_heldout=args.heldout, seed=args.seed)
    save_split(args.out, clips, held, args.seed)
    print(f"held out {len(held)} of {len(clips)} clips: {', '.join(held)}")
    return 0


def _run(args, make_detector) -> int:
    if args.clip:
        clip_paths = list(args.clip)
    elif args.dataset and args.split:
        sequences = list_sequences(args.dataset)
        wanted = load_split(args.split)[args.subset]
        unknown = [clip for clip in wanted if clip not in sequences]
        if unknown:
            return _usage_error(f"split names clips that are not in the dataset: {unknown}")
        clip_paths = [sequences[clip] for clip in wanted]
    else:
        return _usage_error("give --clip PATH, or both --dataset DIR and --split FILE")

    try:
        settings = _settings(args.set)
    except ValueError as error:
        return _usage_error(str(error))

    if make_detector is None:
        from tracker.stages.detect_track import RfDetrPersonDetector as make_detector
    detector = make_detector(args.threshold)
    details = run_details(detector, settings)
    # Written first so a run killed part-way still leaves the run details behind.
    args.out.mkdir(parents=True, exist_ok=True)
    run_file = args.out / "run.json"
    run_file.write_text(json.dumps({"run": details, "clips": {}}, indent=2) + "\n")
    results = run_many(clip_paths, args.out, detector, settings.factory(), fps=args.fps, overlay=not args.no_overlay)
    run_file.write_text(json.dumps({"run": details, "clips": results}, indent=2) + "\n")
    for clip, error in results.items():
        print(f"{clip}: {'ok' if error is None else 'FAILED: ' + error}")
    return 1 if any(error is not None for error in results.values()) else 0


def _retrack(args) -> int:
    try:
        settings = _settings(args.set)
    except ValueError as error:
        return _usage_error(str(error))
    if not args.run.is_dir():
        return _usage_error(f"run folder not found: {args.run}")
    results = retrack_run(args.run, args.out, settings)
    for clip, error in results.items():
        print(f"{clip}: {'ok' if error is None else 'FAILED: ' + error}")
    print(f"tracker settings: {settings.label()}; see {args.out / 'run.json'}")
    return 1 if any(error is not None for error in results.values()) else 0


def _diagnose(args) -> int:
    sequences = list_sequences(args.dataset)
    clip_ids = load_split(args.split)[args.subset]
    args.out.mkdir(parents=True, exist_ok=True)
    reports, failed = {}, {}
    for clip_id in clip_ids:
        if clip_id not in sequences:
            failed[clip_id] = "no ground truth sequence with this name in the dataset"
            continue
        try:
            reports[clip_id] = diagnose_clip(args.run, clip_id, sequences[clip_id])
        except (FileNotFoundError, ValueError) as error:
            failed[clip_id] = str(error)
            continue
        (args.out / f"{clip_id}.json").write_text(json.dumps(reports[clip_id], indent=2) + "\n")
    lines = ["# Box diagnosis", "", f"Run: `{args.run}`", "",
             "| Clip | fps | boxes | <0.1 | 0.1-0.3 | 0.3-0.5 | >=0.5 | med dx | med dy | med d_bottom | med w ratio | med h ratio | tracks==dets | IDF1@0.3 | @0.4 | @0.5 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for clip_id, r in reports.items():
        t, m, s = r["tracks"], r["tracks"]["matched_medians"], r["sweep"]
        fmt = lambda v, f="{:.1f}": "—" if v is None else f.format(v)
        lines.append(
            f"| {clip_id} | {r['fps']:.0f} | {t['n_pred']} | " + " | ".join(f"{t['bin_share'][b]:.1%}" for b in ("<0.1", "0.1-0.3", "0.3-0.5", ">=0.5"))
            + f" | {fmt(m['dx_centre'])} | {fmt(m['dy_centre'])} | {fmt(m['d_bottom'])} | {fmt(m['width_ratio'], '{:.2f}')} | {fmt(m['height_ratio'], '{:.2f}')}"
            + f" | {fmt(r['tracks_vs_detections']['share_identical'], '{:.0%}')} | {s['0.3']['idf1']:.1%} | {s['0.4']['idf1']:.1%} | {s['0.5']['idf1']:.1%} |"
        )
    if failed:
        lines += ["", "## Not diagnosed", ""] + [f"- {c}: {m}" for c, m in sorted(failed.items())]
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")

    for clip_id in args.draw:
        if clip_id not in reports:
            print(f"{clip_id}: not drawn ({failed.get(clip_id, 'not in the subset')})")
            continue
        try:
            clip = Clip(sequences[clip_id])
            tracks = read_table("tracks", args.run / clip_id / "tracks.parquet")
            detections = read_table("detections", args.run / clip_id / "detections.parquet")
            gt = load_mot_ground_truth(sequences[clip_id])
            frames = sorted({int(round(x)) for x in np.linspace(0, clip.info.n_frames - 1, args.frames)})
            for frame in frames:
                draw_comparison(clip, frame, gt, tracks, detections, args.out / "frames" / clip_id / f"{frame:06d}.jpg")
        except (ValueError, FileNotFoundError, ClipError) as error:
            print(f"{clip_id}: not drawn ({error})")
            continue
        print(f"{clip_id}: drew {len(frames)} frames under {args.out / 'frames' / clip_id}")
    print(f"diagnosed {len(reports)} clips ({len(failed)} not); see {args.out / 'summary.md'}")
    return 0 if reports else 1


def _grid(args) -> int:
    if args.subset == "heldout":
        return _usage_error("the grid tunes settings, so it never scores the held-out clips; use --subset train")
    try:
        points = one_factor_grid() if args.one_factor else load_grid(args.grid)
    except ValueError as error:
        return _usage_error(str(error))
    sequences = list_sequences(args.dataset)
    clip_ids = load_split(args.split)[args.subset]
    records = run_grid(args.run, args.out, points, sequences, clip_ids)
    for r in records:
        idf1 = "—" if r["idf1"] is None else f"{r['idf1']:.1%}"
        print(f"{r['label']}: IDF1 {idf1} over {r['n_clips']} clips ({len(r['failed'])} failed)")
    print(f"best: {records[0]['label']}; see {args.out / 'grid.md'}")
    return 0


def _score(args) -> int:
    sequences = list_sequences(args.dataset)
    clip_ids = load_split(args.split)[args.subset]
    counts, failed = score_run(args.run, sequences, clip_ids)
    run_file = args.run / "run.json"
    details = json.loads(run_file.read_text())["run"] if run_file.is_file() else {}
    scorecard = build_scorecard(counts, failed, details)
    write_scorecard(scorecard, args.run)
    overall = scorecard["tracking"]["overall"]
    if overall is None:
        print(f"no clip could be scored; see {args.run / 'scorecard.md'}")
        return 1
    print(
        f"IDF1 {overall['idf1']:.1%}, precision {overall['precision']:.1%}, "
        f"recall {overall['recall']:.1%} over {len(counts)} clips "
        f"({len(failed)} failed); see {args.run / 'scorecard.md'}"
    )
    return 0


def main(argv: list[str] | None = None, make_detector=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "split":
        return _split(args)
    if args.command == "run":
        return _run(args, make_detector)
    if args.command == "retrack":
        return _retrack(args)
    if args.command == "diagnose-boxes":
        return _diagnose(args)
    if args.command == "grid":
        return _grid(args)
    return _score(args)


if __name__ == "__main__":
    sys.exit(main())
