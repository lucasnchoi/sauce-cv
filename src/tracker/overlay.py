"""Draws tracked boxes on a clip so results can be checked by eye."""
from __future__ import annotations

from pathlib import Path

import cv2
import polars as pl

from tracker.schemas import validate
from tracker.video import ClipError


def _colour(track_id: int) -> tuple[int, int, int]:
    """A stable, distinct colour per track."""
    return (37 * track_id % 200 + 55, 91 * track_id % 200 + 55, 173 * track_id % 200 + 55)


def render_overlay(clip, tracks: pl.DataFrame, out_path: Path) -> None:
    tracks = validate(tracks, "tracks")
    out_path = Path(out_path)
    info = clip.info
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ClipError(f"clip '{info.clip_id}': cannot write overlay to {out_path}: {error}")
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (info.width, info.height)
    )
    if not writer.isOpened():
        raise ClipError(f"clip '{info.clip_id}': cannot write overlay to {out_path}")

    by_frame: dict[int, list[dict]] = {}
    for row in tracks.to_dicts():
        by_frame.setdefault(row["frame"], []).append(row)
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
            writer.write(image)
    finally:
        writer.release()
