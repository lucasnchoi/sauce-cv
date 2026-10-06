"""Command line: choose the held-out split, run the pipeline, score a run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tracker.datasets.mot import list_sequences
from tracker.datasets.split import DEFAULT_SEED, choose_heldout, load_split, save_split
from tracker.evaluate.scorecard import build_scorecard, write_scorecard
from tracker.pipeline import run_details, run_many, score_run


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

    score = commands.add_parser("score", help="score a finished run against ground truth")
    score.add_argument("--run", type=Path, required=True)
    score.add_argument("--dataset", type=Path, required=True)
    score.add_argument("--split", type=Path, required=True)
    score.add_argument("--subset", choices=["heldout", "train"], default="heldout")
    return parser


def _usage_error(message: str) -> int:
    print(f"tracker: {message}", file=sys.stderr)
    return 2


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

    if make_detector is None:
        from tracker.stages.detect_track import RfDetrPersonDetector as make_detector
    detector = make_detector(args.threshold)
    details = run_details(detector)
    results = run_many(clip_paths, args.out, detector, fps=args.fps, overlay=not args.no_overlay)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run.json").write_text(json.dumps({"run": details, "clips": results}, indent=2) + "\n")
    for clip, error in results.items():
        print(f"{clip}: {'ok' if error is None else 'FAILED: ' + error}")
    return 1 if any(error is not None for error in results.values()) else 0


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
    return _score(args)


if __name__ == "__main__":
    sys.exit(main())
