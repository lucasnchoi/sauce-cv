"""The owner's rink-keypoint labels: a committed table, and the recorded list of frames to label.

Every labelled frame holds exactly one row per template keypoint, so a frame
with nothing visible is still on record. Pixels are null unless visible.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import polars as pl

from tracker.rink import KEYPOINTS
from tracker.schemas import empty, validate
from tracker.tables import read_table, write_table
from tracker.video import Clip

TABLE = "rink_keypoint_labels"


def load_labels(path) -> pl.DataFrame:
    path = Path(path)
    return read_table(TABLE, path) if path.is_file() else empty(TABLE)


def _check(labels: pl.DataFrame, image_size: tuple[int, int]) -> None:
    width, height = image_size
    for row in labels.iter_rows(named=True):
        where = f"clip '{row['clip_id']}' frame {row['frame']} keypoint {row['keypoint_id']}"
        if row["visible"]:
            if row["x_px"] is None or row["y_px"] is None:
                raise ValueError(f"{where}: visible but has no pixel position")
            if not (0 <= row["x_px"] < width and 0 <= row["y_px"] < height):
                raise ValueError(f"{where}: pixel ({row['x_px']}, {row['y_px']}) is outside the {width}x{height} image")
        elif row["x_px"] is not None or row["y_px"] is not None:
            raise ValueError(f"{where}: not visible but has a pixel position")
    counts = labels.group_by("clip_id", "frame").agg(pl.len().alias("n"), pl.col("keypoint_id").n_unique().alias("u"))
    bad = counts.filter((pl.col("n") != len(KEYPOINTS)) | (pl.col("u") != len(KEYPOINTS)))
    if bad.height:
        r = bad.row(0, named=True)
        raise ValueError(f"clip '{r['clip_id']}' frame {r['frame']}: expected {len(KEYPOINTS)} keypoint rows, found {r['n']}")


def save_labels(labels: pl.DataFrame, path, image_size: tuple[int, int] = (1280, 720)) -> None:
    labels = validate(labels, TABLE).sort("clip_id", "frame", "keypoint_id")
    _check(labels, image_size)
    write_table(labels, TABLE, path)


def frame_rows(clip_id: str, frame: int, points: dict[int, tuple[float, float] | None]) -> list[dict]:
    rows = []
    for k in KEYPOINTS:
        p = points.get(k.id)
        rows.append({"clip_id": clip_id, "frame": frame, "keypoint_id": k.id,
                     "x_px": float(p[0]) if p else None, "y_px": float(p[1]) if p else None,
                     "visible": p is not None})
    return rows


def upsert_frame(labels: pl.DataFrame, clip_id: str, frame: int, rows: list[dict]) -> pl.DataFrame:
    kept = labels.filter(~((pl.col("clip_id") == clip_id) & (pl.col("frame") == frame)))
    new = pl.DataFrame(rows, schema=kept.schema)
    return pl.concat([kept, new]).sort("clip_id", "frame", "keypoint_id")


def labelled_frames(labels: pl.DataFrame) -> set[tuple[str, int]]:
    return {(c, int(f)) for c, f in labels.select("clip_id", "frame").unique().iter_rows()}


def interleave_frames(frames: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Round-robin across clips: every clip's first frame, then every clip's second, and so on."""
    seen: dict[str, int] = {}
    ranked = []
    for clip_id, frame in frames:
        rank = seen.get(clip_id, 0)
        seen[clip_id] = rank + 1
        ranked.append((rank, clip_id, frame))
    return [(c, f) for _, c, f in sorted(ranked, key=lambda t: (t[0], t[1]))]


def sample_frames(n_frames: int, count: int, seed: int, clip_id: str) -> list[int]:
    """One frame from each of `count` equal intervals, fixed by seed and clip id."""
    count = min(count, n_frames)
    rng = random.Random(f"{seed}:{clip_id}")
    picks = []
    for i in range(count):
        first = (i * n_frames) // count
        stop = ((i + 1) * n_frames) // count
        picks.append(rng.randrange(first, max(stop, first + 1)))
    return sorted(picks)


def build_frame_list(sequences: dict[str, Path], split: dict, per_clip_train: int, per_clip_eval: int, seed: int) -> dict:
    def pick(clip_ids, per_clip):
        out = []
        for clip_id in sorted(clip_ids):
            n = Clip(sequences[clip_id]).info.n_frames
            out += [[clip_id, f] for f in sample_frames(n, per_clip, seed, clip_id)]
        return out
    return {"seed": seed, "per_clip_train": per_clip_train, "per_clip_eval": per_clip_eval,
            "training": pick(split["train"], per_clip_train), "evaluation": pick(split["heldout"], per_clip_eval)}


def save_frame_list(data: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_frame_list(path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"frame list not found at {path}; run 'tracker label-frames' first")
    return json.loads(path.read_text())
