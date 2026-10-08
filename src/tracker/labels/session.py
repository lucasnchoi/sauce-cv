"""State of a labelling session, kept apart from the window so it can be tested."""
from __future__ import annotations

import cv2
import numpy as np
import polars as pl

from tracker.labels.rink_keypoints import frame_rows, labelled_frames, upsert_frame
from tracker.rink import KEYPOINTS, draw_rink

Point = tuple[float, float] | None


class LabelSession:
    def __init__(self, frames: list[tuple[str, int]], labels: pl.DataFrame):
        self.frames = [(str(c), int(f)) for c, f in frames]
        self._labels = labels
        self._saved = labelled_frames(labels)
        self.points: dict[tuple[str, int], dict[int, Point]] = {}
        for clip_id, frame in self.frames:
            part = labels.filter((pl.col("clip_id") == clip_id) & (pl.col("frame") == frame))
            if part.height:
                self.points[(clip_id, frame)] = {
                    int(r["keypoint_id"]): ((r["x_px"], r["y_px"]) if r["visible"] else None)
                    for r in part.iter_rows(named=True)
                }
        self._edited: set[tuple[str, int]] = set()
        self._history: dict[tuple[str, int], list[tuple[dict[int, Point], bool]]] = {}
        self.index = 0
        self.keypoint = self._first_unset()

    # -- frame and keypoint selection -------------------------------------------------------
    @property
    def current(self) -> tuple[str, int]:
        return self.frames[self.index]

    @property
    def current_points(self) -> dict[int, Point]:
        return self.points.setdefault(self.current, {})

    def is_labelled(self, frame: tuple[str, int]) -> bool:
        return frame in self._saved or frame in self._edited

    @property
    def progress(self) -> tuple[int, int]:
        return sum(self.is_labelled(f) for f in self.frames), len(self.frames)

    def _first_unset(self, start: int = 0) -> int:
        ids = [k.id for k in KEYPOINTS]
        order = ids[start:] + ids[:start]
        for k in order:
            if k not in self.current_points:
                return k
        return start % len(ids)

    def select(self, keypoint_id: int) -> None:
        if keypoint_id not in {k.id for k in KEYPOINTS}:
            raise ValueError(f"no keypoint {keypoint_id}")
        self.keypoint = keypoint_id

    def next_keypoint(self) -> None:
        self.keypoint = (self.keypoint + 1) % len(KEYPOINTS)

    def prev_keypoint(self) -> None:
        self.keypoint = (self.keypoint - 1) % len(KEYPOINTS)

    def jump_to(self, index: int) -> None:
        """Go to a frame position (clamped) and start at its first unset keypoint."""
        self.index = max(0, min(index, len(self.frames) - 1))
        self.keypoint = self._first_unset()

    def start_at_first_unlabelled(self) -> None:
        """Open at the first frame not yet labelled, or at position 0 when every frame is labelled."""
        self.jump_to(0)
        if self.is_labelled(self.current):
            self.next_unlabelled()

    def next_frame(self) -> None:
        self.jump_to(self.index + 1)

    def prev_frame(self) -> None:
        self.jump_to(self.index - 1)

    def next_unlabelled(self) -> bool:
        """Jump to the next frame after this one that is not labelled; False if there is none."""
        for i in list(range(self.index + 1, len(self.frames))) + list(range(0, self.index)):
            if not self.is_labelled(self.frames[i]):
                self.jump_to(i)
                return True
        return False

    # -- edits -------------------------------------------------------------------------------
    def _edit(self, value: Point | type(...)) -> None:
        self._history.setdefault(self.current, []).append((dict(self.current_points), self.current in self._edited))
        if value is ...:
            self.current_points.pop(self.keypoint, None)
        else:
            self.current_points[self.keypoint] = value
        self._edited.add(self.current)
        if value is not ...:
            candidate = self._first_unset(self.keypoint + 1)
            if candidate not in self.current_points:
                self.keypoint = candidate

    def place(self, x: float, y: float) -> None:
        self._edit((float(x), float(y)))

    def mark_not_visible(self) -> None:
        self._edit(None)

    def clear_current(self) -> None:
        self._edit(...)

    def accept(self) -> None:
        """Record the frame as it stands (its suggestions included) without changing any point; undoable."""
        self._history.setdefault(self.current, []).append((dict(self.current_points), self.current in self._edited))
        self._edited.add(self.current)

    def undo(self) -> None:
        history = self._history.get(self.current)
        if not history:
            return
        snapshot, was_edited = history.pop()
        self.points[self.current] = snapshot
        if not was_edited:
            self._edited.discard(self.current)
        self.keypoint = self._first_unset()

    # -- prelabels and output ----------------------------------------------------------------
    def load_prelabels(self, keypoints: pl.DataFrame, min_conf: float = 0.5) -> None:
        """Suggest points for frames with no entry yet; the frame counts as labelled only once edited or saved."""
        for clip_id, frame in self.frames:
            if self.is_labelled((clip_id, frame)) or self.points.get((clip_id, frame)):
                continue
            part = (keypoints.filter((pl.col("clip_id") == clip_id) & (pl.col("frame") == frame) & (pl.col("conf") >= min_conf))
                    .sort("conf", descending=True).unique("keypoint_id", keep="first"))
            if part.height:
                self.points[(clip_id, frame)] = {int(r["keypoint_id"]): (r["x_px"], r["y_px"]) for r in part.iter_rows(named=True)}
        self.keypoint = self._first_unset()

    def frame_table(self) -> pl.DataFrame:
        table = self._labels
        for clip_id, frame in sorted(self._edited):
            table = upsert_frame(table, clip_id, frame, frame_rows(clip_id, frame, self.points.get((clip_id, frame), {})))
        return table

    # -- drawing -----------------------------------------------------------------------------
    def render(self, image: np.ndarray) -> np.ndarray:
        out = image.copy()
        cv2.rectangle(out, (0, 0), (min(out.shape[1], 460), 46), (0, 0, 0), -1)
        for k, p in self.current_points.items():
            if p is None:
                continue
            x, y = int(round(p[0])), int(round(p[1]))
            colour = (0, 0, 255) if k == self.keypoint else (0, 200, 0)
            cv2.circle(out, (x, y), 4, colour, -1)
            cv2.putText(out, str(k), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1)
        clip_id, frame = self.current
        done, total = self.progress
        name = KEYPOINTS[self.keypoint].name
        cv2.putText(out, f"{clip_id} f{frame}  [{self.index + 1}/{len(self.frames)}]  labelled {done}/{total}",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(out, f"keypoint {self.keypoint}: {name}", (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
        return out

    def legend(self) -> np.ndarray:
        pts = self.current_points
        return draw_rink(highlight=self.keypoint,
                         labelled={k for k, p in pts.items() if p is not None},
                         not_visible={k for k, p in pts.items() if p is None})
