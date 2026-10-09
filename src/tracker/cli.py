"""Command line: choose the held-out split, run the pipeline, score a run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

from tracker.analysis.box_offset import NTSC_RATIO, diagnose_clip, draw_comparison, ntsc_fit, shift_clip
from tracker.analysis.tracker_grid import load_grid, one_factor_grid, run_grid
from tracker.datasets.mot import list_sequences, load_mot_ground_truth
from tracker.datasets.split import DEFAULT_SEED, choose_heldout, load_split, save_split
from tracker.labels.rink_keypoints import build_frame_list, interleave_frames, load_frame_list, load_labels, save_frame_list
from tracker.labels.session import LabelSession
from tracker.labels.tool import render_current, run_tool
from tracker.evaluate.scorecard import build_scorecard, read_fps_by_clip, write_scorecard
from tracker.pipeline import RinkStages, read_clip_info, refilter_run, retrack_run, run_details, run_many, score_rink_run, score_run
from tracker.stages.detect_track import ByteTrackSettings
from tracker.stages.footage_filter import MODES, FootageFilterSettings
from tracker.stages.rink_transform import RinkTransformSettings
from tracker.tables import read_table, write_table
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
                     help="a setting: ByteTrack KEY=VALUE, or footage.KEY=VALUE / rink.KEY=VALUE for the footage filter and rink transform (repeatable)")
    run.add_argument("--rink-weights", type=Path, help="rink keypoint weights; enables stages 1-3")
    run.add_argument("--rink-threshold", type=float, default=0.3)
    run.add_argument("--footage-filter", choices=list(MODES), default="transform")
    run.add_argument("--device", choices=["mps", "cpu"])

    retrack = commands.add_parser("retrack", help="re-link a finished run's detections with other tracker settings")
    retrack.add_argument("--run", type=Path, required=True, help="a run that holds detections.parquet per clip")
    retrack.add_argument("--out", type=Path, required=True)
    retrack.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    refilter = commands.add_parser("refilter", help="re-run the footage filter, rink transform, tracking and positions from a finished run's cache")
    refilter.add_argument("--run", type=Path, required=True, help="a run detected on every frame (Plan 2c or later); its keypoints and detections are re-used, everything else is re-run under the settings given")
    refilter.add_argument("--out", type=Path, required=True)
    refilter.add_argument("--footage-filter", choices=list(MODES), default="transform")
    refilter.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                          help="a setting: ByteTrack KEY=VALUE, or footage.KEY=VALUE / rink.KEY=VALUE (repeatable); defaults are the project's, not the source run's")

    score = commands.add_parser("score", help="score a finished run against ground truth")
    score.add_argument("--run", type=Path, required=True)
    score.add_argument("--dataset", type=Path, required=True)
    score.add_argument("--split", type=Path, required=True)
    score.add_argument("--subset", choices=["heldout", "train"], default="heldout")
    score.add_argument("--labels", type=Path, default=Path("labels/rink_keypoints.parquet"))

    diagnose = commands.add_parser("diagnose-boxes", help="compare tracked boxes, raw detections and ground truth")
    diagnose.add_argument("--run", type=Path, required=True)
    diagnose.add_argument("--dataset", type=Path, required=True)
    diagnose.add_argument("--split", type=Path, required=True)
    diagnose.add_argument("--subset", choices=["heldout", "train"], default="heldout")
    diagnose.add_argument("--out", type=Path, required=True)
    diagnose.add_argument("--frames", type=int, default=10, help="comparison frames to draw per --draw clip")
    diagnose.add_argument("--shift", action="store_true", help="also test the ground truth read k frames earlier (shift.md)")
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
    frames_cmd = commands.add_parser("label-frames", help="choose and record the frames to label for rink keypoints")
    frames_cmd.add_argument("--dataset", type=Path, required=True)
    frames_cmd.add_argument("--split", type=Path, required=True)
    frames_cmd.add_argument("--out", type=Path, required=True)
    frames_cmd.add_argument("--per-clip-train", type=int, default=24)
    frames_cmd.add_argument("--per-clip-eval", type=int, default=10)
    frames_cmd.add_argument("--seed", type=int, default=20261007)
    frames_cmd.add_argument("--force", action="store_true", help="overwrite an existing frame list")

    label = commands.add_parser("label-rink", help="label rink keypoints on the recorded frames")
    label.add_argument("--dataset", type=Path, required=True)
    label.add_argument("--frames", type=Path, required=True)
    label.add_argument("--labels", type=Path, required=True)
    label.add_argument("--subset", choices=["training", "evaluation", "all"], default="training")
    label.add_argument("--prelabels", type=Path, help="a run folder holding <clip>/keypoints.parquet to pre-fill from")
    label.add_argument("--min-conf", type=float, default=0.5)
    label.add_argument("--start", type=int, default=None,
                       help="position in the displayed order, 0-based (default: the first unlabelled frame)")
    label.add_argument("--render-only", type=Path, help="write the current view to this image and exit (no window)")
    train_cmd = commands.add_parser("train-rink", help="train the rink keypoint model on the committed labels")
    train_cmd.add_argument("--labels", type=Path, default=Path("labels/rink_keypoints.parquet"))
    train_cmd.add_argument("--dataset", type=Path, required=True)
    train_cmd.add_argument("--split", type=Path, required=True, help="labels on held-out clips are dropped, never trained on")
    train_cmd.add_argument("--out", type=Path, required=True)
    train_cmd.add_argument("--epochs", type=int)
    train_cmd.add_argument("--batch", type=int)
    train_cmd.add_argument("--lr", type=float)
    train_cmd.add_argument("--pos-weight", type=float)
    train_cmd.add_argument("--seed", type=int)
    train_cmd.add_argument("--val-share", type=float)
    train_cmd.add_argument("--device", choices=["mps", "cpu"])
    train_cmd.add_argument("--max-epoch-minutes", type=float)
    train_cmd.add_argument("--no-pretrained", action="store_true", help="start from random weights (tests only)")
    prelabel = commands.add_parser("prelabel-rink", help="run the rink keypoint model on the listed label frames so the owner corrects instead of placing")
    prelabel.add_argument("--weights", type=Path, required=True)
    prelabel.add_argument("--dataset", type=Path, required=True)
    prelabel.add_argument("--frames", type=Path, required=True)
    prelabel.add_argument("--out", type=Path, required=True)
    prelabel.add_argument("--subset", choices=["training", "evaluation", "all"], default="all")
    prelabel.add_argument("--threshold", type=float, default=0.3)
    prelabel.add_argument("--device", choices=["mps", "cpu"])
    check = commands.add_parser("rink-check", help="PCK and transform quality of rink weights on a training run's validation frames")
    check.add_argument("--weights", type=Path, required=True)
    check.add_argument("--dataset", type=Path, required=True)
    check.add_argument("--labels", type=Path, default=Path("labels/rink_keypoints.parquet"))
    check.add_argument("--frames-from", type=Path, required=True, help="a training run.json; its val_frames are checked")
    check.add_argument("--split", type=Path, help="when given, held-out clips in the frame list are refused")
    check.add_argument("--threshold", type=float, default=0.3)
    check.add_argument("--device", choices=["mps", "cpu"])
    return parser


def _usage_error(message: str) -> int:
    print(f"tracker: {message}", file=sys.stderr)
    return 2


def _settings(pairs: list[str]) -> tuple[ByteTrackSettings, FootageFilterSettings, RinkTransformSettings]:
    groups: dict[str, dict] = {"tracker": {}, "footage": {}, "rink": {}}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise ValueError(f"setting '{pair}' is not KEY=VALUE")
        group, dot, name = key.partition(".")
        if dot and group in ("footage", "rink"):
            groups[group][name] = value
        elif dot:
            raise ValueError(f"setting '{key}' has an unknown group '{group}'; use footage.NAME or rink.NAME")
        else:
            groups["tracker"][key] = value
    bad = next((k for g in groups.values() for k, v in g.items() if not _looks_numeric(v)), None)
    if bad is not None:
        raise ValueError(f"setting '{bad}' needs a number")
    return (ByteTrackSettings.from_dict(groups["tracker"]), FootageFilterSettings.from_dict(groups["footage"]),
            RinkTransformSettings.from_dict(groups["rink"]))


def _looks_numeric(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


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
        settings, footage, rink_settings = _settings(args.set)
    except ValueError as error:
        return _usage_error(str(error))

    if make_detector is None:
        from tracker.stages.detect_track import RfDetrPersonDetector as make_detector
    detector = make_detector(args.threshold)
    rink = RinkStages(
        make_keypoint_detector=(lambda: _keypoint_detector(args.rink_weights, args.rink_threshold, args.device)) if args.rink_weights else None,
        footage_filter=footage, footage_mode=args.footage_filter, rink_transform=rink_settings,
    )
    if args.rink_weights is None:
        print("no --rink-weights: footage filter is passthrough, no rink transform, every position is missing")
    details = run_details(detector, settings, rink)
    # Written first so a run killed part-way still leaves the run details behind.
    args.out.mkdir(parents=True, exist_ok=True)
    run_file = args.out / "run.json"
    run_file.write_text(json.dumps({"run": details, "clips": {}}, indent=2) + "\n")
    try:
        results = run_many(clip_paths, args.out, detector, settings.factory(), fps=args.fps, overlay=not args.no_overlay, rink=rink)
    except ValueError as error:
        return _usage_error(str(error))
    run_file.write_text(json.dumps({"run": details, "clips": results}, indent=2) + "\n")
    for clip, error in results.items():
        print(f"{clip}: {'ok' if error is None else 'FAILED: ' + error}")
    return 1 if any(error is not None for error in results.values()) else 0


def _retrack(args) -> int:
    try:
        settings = _settings(args.set)[0]
    except ValueError as error:
        return _usage_error(str(error))
    if not args.run.is_dir():
        return _usage_error(f"run folder not found: {args.run}")
    results = retrack_run(args.run, args.out, settings)
    for clip, error in results.items():
        print(f"{clip}: {'ok' if error is None else 'FAILED: ' + error}")
    print(f"tracker settings: {settings.label()}; see {args.out / 'run.json'}")
    return 1 if any(error is not None for error in results.values()) else 0


def _refilter(args) -> int:
    try:
        tracker_settings, footage, rink_settings = _settings(args.set)
    except ValueError as error:
        return _usage_error(str(error))
    if not args.run.is_dir():
        return _usage_error(f"run folder not found: {args.run}")
    try:
        results = refilter_run(args.run, args.out, args.footage_filter, footage, rink_settings, tracker_settings)
    except ValueError as error:
        return _usage_error(str(error))
    for clip, error in results.items():
        print(f"{clip}: {'ok' if error is None else 'FAILED: ' + error}")
    print(f"footage filter: {args.footage_filter}; tracker settings: {tracker_settings.label()}; see {args.out / 'run.json'}")
    return 1 if any(error is not None for error in results.values()) else 0


def _write_shift_table(path: Path, run: Path, shifts: dict) -> None:
    """Share of tracked boxes at IoU 0.5 or more per k, then the best k in each fifth of each clip."""
    ks = list(next(iter(shifts.values()))["shifted_share"]) if shifts else []
    lines = ["# Frame-shift test", "",
             f"Run: `{run}`. The tracked box at frame f is compared with the ground truth labelled f-k; "
             "cells are the share of tracked boxes at IoU 0.5 or more. k=0 is the real labelling.", "",
             "| Clip | " + " | ".join(f"k={int(k):+d}" for k in ks) + " |", "|---|" + "---|" * len(ks)]
    for clip_id, s in shifts.items():
        lines.append(f"| {clip_id} | " + " | ".join(f"{s['shifted_share'][k]:.1%}" for k in ks) + " |")
    lines += ["", "Best k in each fifth of the clip's tracked frames (share at that k, share at k=0):", "",
              "| Clip | 1st | 2nd | 3rd | 4th | 5th |", "|---|---|---|---|---|---|"]
    for clip_id, s in shifts.items():
        cells = [f"{r['best_k']:+d} ({r['share_at_best']:.1%}, {r['share_at_zero']:.1%})" for r in s["drift_by_range"]]
        lines.append(f"| {clip_id} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def _diagnose(args) -> int:
    sequences = list_sequences(args.dataset)
    clip_ids = load_split(args.split)[args.subset]
    args.out.mkdir(parents=True, exist_ok=True)
    reports, shifts, failed = {}, {}, {}
    for clip_id in clip_ids:
        if clip_id not in sequences:
            failed[clip_id] = "no ground truth sequence with this name in the dataset"
            continue
        try:
            reports[clip_id] = diagnose_clip(args.run, clip_id, sequences[clip_id])
            if args.shift:
                shifts[clip_id] = shift_clip(args.run, clip_id, sequences[clip_id])
        except (FileNotFoundError, ValueError) as error:
            failed[clip_id] = str(error)
            continue
        (args.out / f"{clip_id}.json").write_text(json.dumps(reports[clip_id], indent=2) + "\n")
        if args.shift:
            (args.out / f"{clip_id}.shift.json").write_text(json.dumps(shifts[clip_id], indent=2) + "\n")
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
    if args.shift:
        _write_shift_table(args.out / "shift.md", args.run, shifts)
        lines = ["# NTSC check", "",
                 f"Run: `{args.run}`. Predicted k at a range's mid-frame is mid / {NTSC_RATIO:g}; a clip fits when every "
                 "|best k - predicted| <= 0.75 and best k never decreases. 'no drift' means best k is 0 in every fifth.", "",
                 "| Clip | frames | fps | best k per fifth | predicted k per fifth | max abs error | verdict |", "|---|---|---|---|---|---|---|"]
        for clip_id, s in shifts.items():
            fit = ntsc_fit(s["drift_by_range"])
            info = read_clip_info(args.run / clip_id / "clip.json")
            verdict = "no drift" if fit["no_drift"] else ("fits" if fit["fits"] else "drifts, does not fit")
            lines.append(f"| {clip_id} | {info.n_frames} | {info.fps:g} | " + ", ".join(f"{r['best_k']:+d}" for r in fit["ranges"])
                         + " | " + ", ".join(f"{r['predicted_k']:.1f}" for r in fit["ranges"]) + f" | {fit['max_abs_error']:.2f} | {verdict} |")
        (args.out / "ntsc.md").write_text("\n".join(lines) + "\n")

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
    if not args.one_factor and args.grid is not None and not args.grid.is_file():
        return _usage_error(f"grid file not found: {args.grid}")
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
    rink_per_clip = rink_failed = None
    if args.labels.is_file():
        rink_per_clip, rink_failed = score_rink_run(args.run, clip_ids, load_labels(args.labels))
    else:
        print(f"no label file at {args.labels}: the rink transform is not scored")
    scorecard = build_scorecard(counts, failed, details, fps_by_clip=read_fps_by_clip(args.run, clip_ids),
                                rink_per_clip=rink_per_clip, rink_failed=rink_failed)
    write_scorecard(scorecard, args.run)
    overall = scorecard["tracking"]["overall"]
    if overall is None:
        print(f"no clip could be scored; see {args.run / 'scorecard.md'}")
        return 1
    print(
        f"IDF1 {overall['idf1']:.1%}, precision {overall['precision']:.1%}, "
        f"recall {overall['recall']:.1%}, IDF1@0.3 {overall['idf1_at_0.3']:.1%} over {len(counts)} clips "
        f"({len(failed)} failed); see {args.run / 'scorecard.md'}"
    )
    if rink_per_clip is not None and scorecard["rink"]["overall"] is not None:
        rink = scorecard["rink"]["overall"]
        median = "—" if rink["median_m"] is None else "{:.2f}".format(rink["median_m"])
        valid = "—" if rink["valid_share"] is None else "{:.1%}".format(rink["valid_share"])
        print(f"rink median {median} m, valid {valid} over {len(rink_per_clip)} clips ({len(rink_failed)} without a rink score)")
    return 0


def _label_frames(args) -> int:
    if args.out.exists() and not args.force:
        return _usage_error(f"{args.out} exists; the frame list is a permanent record (use --force to replace it)")
    sequences = list_sequences(args.dataset)
    split = load_split(args.split)
    data = build_frame_list(sequences, split, args.per_clip_train, args.per_clip_eval, args.seed)
    save_frame_list(data, args.out)
    print(f"{len(data['training'])} training frames and {len(data['evaluation'])} evaluation frames recorded in {args.out}")
    return 0


def _label_rink(args) -> int:
    sequences = list_sequences(args.dataset)
    data = load_frame_list(args.frames)
    training = interleave_frames([(c, int(f)) for c, f in data["training"]])
    evaluation = interleave_frames([(c, int(f)) for c, f in data["evaluation"]])
    frames = {"training": training, "evaluation": evaluation, "all": training + evaluation}[args.subset]
    unknown = sorted({c for c, _ in frames} - set(sequences))
    if unknown:
        return _usage_error(f"frame list names clips that are not in the dataset: {unknown}")
    session = LabelSession(frames, load_labels(args.labels))
    if args.prelabels:
        for clip_id in sorted({c for c, _ in frames}):
            path = args.prelabels / clip_id / "keypoints.parquet"
            if path.is_file():
                session.load_prelabels(read_table("keypoints", path), args.min_conf)
    if args.start is None:
        session.start_at_first_unlabelled()
    else:
        session.jump_to(args.start)
    if args.render_only:
        import cv2
        args.render_only.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.render_only), render_current(session, sequences))
        print(f"rendered {session.current} to {args.render_only}")
        return 0
    run_tool(session, sequences, args.labels)
    return 0


def _train_rink(args) -> int:
    from tracker.labels.rink_keypoints import load_labels
    from tracker.rink_model.data import samples_from_labels, split_validation
    from tracker.rink_model.model import pick_device
    from tracker.rink_model.train import TrainSettings, TrainingTooSlow, train

    sequences = list_sequences(args.dataset)
    split = load_split(args.split)
    labels = load_labels(args.labels)
    clips = sorted(set(labels["clip_id"].to_list()))
    held = sorted(set(clips) & set(split["heldout"]))
    if held:
        n_held = labels.filter(pl.col("clip_id").is_in(held)).select("clip_id", "frame").unique().height
        labels = labels.filter(~pl.col("clip_id").is_in(held))
        clips = sorted(set(clips) - set(held))
        print(f"dropping {n_held} labelled frames on held-out clips (never trained on): {held}")
    missing = sorted(set(clips) - set(sequences))
    if missing:
        return _usage_error(f"labels name clips that are not in the dataset: {missing}")
    overrides = {k: v for k, v in {"epochs": args.epochs, "batch_size": args.batch, "lr": args.lr, "pos_weight": args.pos_weight, "seed": args.seed,
                                   "val_share": args.val_share, "max_epoch_minutes": args.max_epoch_minutes}.items()
                 if v is not None}
    settings = TrainSettings.from_dict({**TrainSettings().as_dict(), **overrides})
    samples = samples_from_labels(labels)
    if not samples:
        return _usage_error(f"{args.labels} holds no labelled training frame (after dropping held-out clips)")
    train_samples, val_samples = split_validation(samples, settings.seed, settings.val_share)
    device = pick_device(args.device)
    print(f"training on {len(train_samples)} frames, validating on {len(val_samples)}, device {device}")
    try:
        record = train(train_samples, val_samples, sequences, args.out, settings, device, pretrained=not args.no_pretrained)
    except TrainingTooSlow as error:
        print(f"tracker: {error}", file=sys.stderr)
        return 1
    n_points = record["history"][-1].get("n_points", 0)
    print(f"best epoch {record['best_epoch']}, PCK@{settings.pck_px:g} {record['best_pck'] if record['best_pck'] is not None else '—'} "
          f"over {n_points} validation points; weights in {args.out / 'best.pt'}")
    return 0


def _keypoint_detector(weights: Path, threshold: float, device: str | None):
    from tracker.rink_model.model import RinkKeypointDetector

    return RinkKeypointDetector(weights, threshold=threshold, device=device)


def _prelabel_rink(args) -> int:
    from tracker.stages.rink_keypoints import keypoints_for_frames

    if args.weights != Path("fake.pt") and not args.weights.is_file():
        return _usage_error(f"rink keypoint weights not found at {args.weights}")
    sequences = list_sequences(args.dataset)
    data = load_frame_list(args.frames)
    listed = {"training": data["training"], "evaluation": data["evaluation"], "all": data["training"] + data["evaluation"]}[args.subset]
    by_clip: dict[str, list[int]] = {}
    for clip_id, frame in listed:
        by_clip.setdefault(clip_id, []).append(int(frame))
    unknown = sorted(set(by_clip) - set(sequences))
    if unknown:
        return _usage_error(f"frame list names clips that are not in the dataset: {unknown}")
    detector = _keypoint_detector(args.weights, args.threshold, args.device)
    n_frames = 0
    for clip_id, frames in sorted(by_clip.items()):
        table = keypoints_for_frames(Clip(sequences[clip_id]), frames, detector)
        write_table(table, "keypoints", args.out / clip_id / "keypoints.parquet")
        n_frames += len(frames)
    print(f"pre-labelled {n_frames} frames over {len(by_clip)} clips into {args.out}")
    return 0


def _rink_check(args) -> int:
    from tracker.rink_model.data import samples_from_labels
    from tracker.rink_model.train import check_frames

    if not args.frames_from.is_file():
        return _usage_error(f"frame list not found: {args.frames_from}")
    if not args.weights.is_file():
        return _usage_error(f"weights not found: {args.weights}")
    wanted = {(c, int(f)) for c, f in json.loads(args.frames_from.read_text())["val_frames"]}
    if args.split:
        held = sorted({c for c, _ in wanted} & set(load_split(args.split)["heldout"]))
        if held:
            return _usage_error(f"validation frames name held-out clips: {held}")
    samples = [s for s in samples_from_labels(load_labels(args.labels)) if (s.clip_id, s.frame) in wanted]
    sequences = list_sequences(args.dataset)
    detector = _keypoint_detector(args.weights, args.threshold, args.device)
    result = check_frames(detector, samples, sequences)
    result.pop("distances_m")
    print(json.dumps(result))
    return 0


def main(argv: list[str] | None = None, make_detector=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "split":
        return _split(args)
    if args.command == "run":
        return _run(args, make_detector)
    if args.command == "retrack":
        return _retrack(args)
    if args.command == "refilter":
        return _refilter(args)
    if args.command == "diagnose-boxes":
        return _diagnose(args)
    if args.command == "grid":
        return _grid(args)
    if args.command == "label-frames":
        return _label_frames(args)
    if args.command == "label-rink":
        return _label_rink(args)
    if args.command == "train-rink":
        return _train_rink(args)
    if args.command == "prelabel-rink":
        return _prelabel_rink(args)
    if args.command == "rink-check":
        return _rink_check(args)
    return _score(args)


if __name__ == "__main__":
    sys.exit(main())
