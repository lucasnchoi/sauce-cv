"""The labelling window: an OpenCV loop around LabelSession. Not unit-tested; use --render-only to smoke-check."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tracker.labels.rink_keypoints import save_labels
from tracker.labels.session import LabelSession
from tracker.rink import KEYPOINTS, rink_to_px
from tracker.video import Clip

KEYS = {
    "left click": "place the current keypoint",
    "left click on the legend": "select that keypoint",
    "right click": "mark the current keypoint not visible",
    "x": "mark the current keypoint not visible (same as right click)",
    "a": "accept the frame as it stands (records it without changing any point)",
    "]": "next keypoint", "[": "previous keypoint",
    "n": "next frame", "p": "previous frame", "g": "jump to the next unlabelled frame",
    "u": "undo the last edit in this frame", "c": "clear the current keypoint",
    "s": "save", "q": "save and quit",
}


def side_by_side(frame: np.ndarray, legend: np.ndarray) -> np.ndarray:
    height = max(frame.shape[0], legend.shape[0])
    def pad(img):
        out = np.zeros((height, img.shape[1], 3), dtype=np.uint8)
        out[: img.shape[0], : img.shape[1]] = img
        return out
    return np.hstack([pad(frame), pad(legend)])


def inside_frame(x: int, y: int, shape) -> bool:
    """True when (x, y) lies on the frame image, not on the legend panel beside it."""
    return 0 <= x < shape[1] and 0 <= y < shape[0]


def legend_keypoint_at(x: int, y: int, scale: float = 4.0, margin: int = 20, max_distance: float = 12.0) -> int | None:
    """Id of the keypoint nearest to legend-local pixel (x, y), or None if none is within max_distance."""
    best, best_d = None, max_distance
    for k in KEYPOINTS:
        px, py = rink_to_px(k.x_ft, k.y_ft, scale, margin)
        d = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
        if d <= best_d:
            best, best_d = k.id, d
    return best


def render_current(session: LabelSession, sequences: dict[str, Path]) -> np.ndarray:
    clip_id, frame = session.current
    image = Clip(sequences[clip_id]).read_frame(frame)
    return side_by_side(session.render(image), session.legend())


def run_tool(session: LabelSession, sequences: dict[str, Path], labels_path, image_size=(1280, 720), window="label-rink") -> None:
    clips: dict[str, Clip] = {}
    frame_shape = [0, 0]
    legend_shape = [0, 0]

    def clip_for(clip_id):
        if clip_id not in clips:
            clips[clip_id] = Clip(sequences[clip_id])
        return clips[clip_id]

    def on_mouse(event, x, y, flags, param):
        if inside_frame(x, y, frame_shape):
            if event == cv2.EVENT_LBUTTONDOWN:
                session.place(x, y)
            elif event == cv2.EVENT_RBUTTONDOWN:
                session.mark_not_visible()
        elif event == cv2.EVENT_LBUTTONDOWN and x >= frame_shape[1] and y >= 0:
            local_x = x - frame_shape[1]
            if local_x < legend_shape[1] and y < legend_shape[0]:
                found = legend_keypoint_at(local_x, y)
                if found is not None:
                    session.select(found)

    def save() -> bool:
        try:
            save_labels(session.frame_table(), labels_path, image_size)
            print(f"saved {session.progress[0]}/{session.progress[1]} frames to {labels_path}")
            return True
        except (ValueError, OSError) as error:
            print(f"NOT saved: {error}")
            return False

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            clip_id, frame = session.current
            image = clip_for(clip_id).read_frame(frame)
            frame_shape[:] = image.shape[:2]
            legend = session.legend()
            legend_shape[:] = legend.shape[:2]
            cv2.imshow(window, side_by_side(session.render(image), legend))
            key = cv2.waitKey(30) & 0xFF
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("]"): session.next_keypoint()
            elif key == ord("["): session.prev_keypoint()
            elif key == ord("n"): session.next_frame(); save()
            elif key == ord("p"): session.prev_frame(); save()
            elif key == ord("u"): session.undo()
            elif key == ord("c"): session.clear_current()
            elif key == ord("x"): session.mark_not_visible()
            elif key == ord("a"): session.accept()
            elif key == ord("g"):
                if not session.next_unlabelled():
                    print("every frame is labelled")
                save()
            elif key == ord("s"): save()
            elif key == ord("q"):
                if save():
                    break
    finally:
        save()
        cv2.destroyWindow(window)
