"""Stand-ins for the detector, tracker and clip, so stage logic can be tested without models."""
import numpy as np
import supervision as sv

from tracker.video import ClipInfo


def detections(boxes, conf=0.9) -> sv.Detections:
    if not boxes:
        return sv.Detections.empty()
    n = len(boxes)
    return sv.Detections(
        xyxy=np.array(boxes, dtype=float),
        confidence=np.full(n, conf),
        class_id=np.zeros(n, dtype=int),
    )


class FakeDetector:
    """Returns the same boxes for every frame and counts how often it was called."""

    name = "fake-detector"
    threshold = 0.5

    def __init__(self, boxes):
        self._boxes = boxes
        self.calls = 0

    def detect(self, image_bgr):
        self.calls += 1
        return detections(self._boxes)


class _FakeTracker:
    def __init__(self, unconfirmed):
        self._unconfirmed = unconfirmed

    def update(self, dets):
        dets.tracker_id = np.array(
            [-1 if i in self._unconfirmed else i for i in range(len(dets))], dtype=int
        )
        return dets


class FakeTrackerFactory:
    """Each tracker numbers detections 0, 1, 2… by position; listed positions get -1."""

    def __init__(self, unconfirmed=()):
        self._unconfirmed = set(unconfirmed)
        self.fps_seen = []

    def __call__(self, fps):
        self.fps_seen.append(fps)
        return _FakeTracker(self._unconfirmed)


class FakeClip:
    def __init__(self, clip_id="c", fps=30.0, n_frames=4, width=64, height=48):
        self.info = ClipInfo(clip_id=clip_id, fps=fps, n_frames=n_frames, width=width, height=height)

    def frames(self):
        for index in range(self.info.n_frames):
            yield index, np.zeros((self.info.height, self.info.width, 3), dtype=np.uint8)


class FakeKeypointDetector:
    """Returns fixed rink keypoints; per_frame overrides by call index (frame order)."""

    name = "fake-keypoints"
    threshold = 0.3
    weights_path = "fake"

    def __init__(self, points, per_frame=None):
        self._points = points
        self._per_frame = per_frame or {}
        self.calls = 0

    def detect(self, image_bgr):
        chosen = self._per_frame.get(self.calls, self._points)
        self.calls += 1
        out = np.zeros((33, 3), dtype=float)
        for keypoint_id, (x, y, conf) in chosen.items():
            out[keypoint_id] = (x, y, conf)
        return out
