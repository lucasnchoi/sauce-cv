"""Draws tracked boxes on a clip so results can be checked by eye."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import polars as pl

from tracker.rink import draw_rink, rink_to_px
from tracker.schemas import validate
from tracker.video import ClipError


def _colour(track_id: int) -> tuple[int, int, int]:
    """A stable, distinct colour per track."""
    return (37 * track_id % 200 + 55, 91 * track_id % 200 + 55, 173 * track_id % 200 + 55)


def _rink_view(positions: list[dict], scale: float, margin: int) -> np.ndarray:
    image = draw_rink(scale=scale, margin=margin)
    for row in positions:
        if row["x_ft"] is None or row["y_ft"] is None:
            continue
        track_id = int(row["player_key"][3:]) if row["player_key"].startswith("trk") else 0
        cv2.circle(image, rink_to_px(row["x_ft"], row["y_ft"], scale, margin), max(3, int(scale)), _colour(track_id), -1)
    return image


def render_overlay(clip, tracks: pl.DataFrame, out_path: Path, tracking: pl.DataFrame | None = None, rink_scale: float = 3.0) -> None:
    tracks = validate(tracks, "tracks")
    margin = 10
    rink_width = int(round(200 * rink_scale + 2 * margin)) if tracking is not None else 0
    out_path = Path(out_path)
    info = clip.info
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ClipError(f"clip '{info.clip_id}': cannot write overlay to {out_path}: {error}")
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (info.width + rink_width, info.height)
    )
    if not writer.isOpened():
        raise ClipError(f"clip '{info.clip_id}': cannot write overlay to {out_path}")

    by_frame: dict[int, list[dict]] = {}
    for row in tracks.to_dicts():
        by_frame.setdefault(row["frame"], []).append(row)
    positions_by_frame: dict[int, list[dict]] = {}
    if tracking is not None:
        for row in validate(tracking, "tracking").select("frame", "player_key", "x_ft", "y_ft").to_dicts():
            positions_by_frame.setdefault(row["frame"], []).append(row)
    try:
        for index, image in clip.frames():
            if image.shape[1] != info.width or image.shape[0] != info.height:
                raise ClipError(
                    f"clip '{info.clip_id}': frame {index} is {image.shape[1]}x{image.shape[0]}, "
                    f"expected {info.width}x{info.height}"
                )
            for row in by_frame.get(index, []):
                colour = _colour(row["track_id"])
                top_left = (int(row["x1"]), int(row["y1"]))
                bottom_right = (int(row["x2"]), int(row["y2"]))
                cv2.rectangle(image, top_left, bottom_right, colour, 2)
                cv2.putText(
                    image, str(row["track_id"]), (top_left[0], max(12, top_left[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2,
                )
            if tracking is not None:
                rink = _rink_view(positions_by_frame.get(index, []), rink_scale, margin)
                panel = np.zeros((info.height, rink_width, 3), dtype=np.uint8)
                h = min(info.height, rink.shape[0])
                panel[:h, : rink.shape[1]] = rink[:h]
                image = np.hstack([image, panel])
            writer.write(image)
    finally:
        writer.release()
