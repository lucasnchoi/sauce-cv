"""The labelling window: an OpenCV loop around LabelSession. Not unit-tested; use --render-only to smoke-check."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tracker.labels.rink_keypoints import save_labels
from tracker.labels.session import LabelSession
from tracker.video import Clip

KEYS = {
    "left click": "place the current keypoint",
    "right click": "mark the current keypoint not visible",
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


def render_current(session: LabelSession, sequences: dict[str, Path]) -> np.ndarray:
    clip_id, frame = session.current
    image = Clip(sequences[clip_id]).read_frame(frame)
    return side_by_side(session.render(image), session.legend())


def run_tool(session: LabelSession, sequences: dict[str, Path], labels_path, image_size=(1280, 720), window="label-rink") -> None:
    clips: dict[str, Clip] = {}

    def clip_for(clip_id):
        if clip_id not in clips:
            clips[clip_id] = Clip(sequences[clip_id])
        return clips[clip_id]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            session.place(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            session.mark_not_visible()

    def save() -> bool:
        try:
            save_labels(session.frame_table(), labels_path, image_size)
            print(f"saved {session.progress[0]}/{session.progress[1]} frames to {labels_path}")
            return True
        except ValueError as error:
            print(f"NOT saved: {error}")
            return False

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        clip_id, frame = session.current
        image = clip_for(clip_id).read_frame(frame)
        cv2.imshow(window, side_by_side(session.render(image), session.legend()))
        key = cv2.waitKey(30) & 0xFF
        if key == ord("]"): session.next_keypoint()
        elif key == ord("["): session.prev_keypoint()
        elif key == ord("n"): session.next_frame()
        elif key == ord("p"): session.prev_frame()
        elif key == ord("u"): session.undo()
        elif key == ord("c"): session.clear_current()
        elif key == ord("g"):
            if not session.next_unlabelled():
                print("every frame is labelled")
        elif key == ord("s"): save()
        elif key == ord("q"):
            if save():
                break
    cv2.destroyWindow(window)
