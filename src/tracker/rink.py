"""The NHL rink as data: 33 named keypoints in feet, and a top-down drawing.

Coordinates follow the design spec: feet, origin at centre ice, x along the
rink (camera-left negative), y across the rink (far side positive). "L"/"R"
are the camera's left and right; "far"/"near" are +y and -y; a circle's
"top" is its +y extreme.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

RINK_LENGTH_FT = 200.0
RINK_WIDTH_FT = 85.0
CORNER_RADIUS_FT = 28.0
GOAL_LINE_X_FT = 89.0          # 11 ft from the end boards
BLUE_LINE_X_FT = 25.0          # 64 ft from the goal line
END_DOT_X_FT = 69.0            # 20 ft from the goal line
NEUTRAL_DOT_X_FT = 20.0        # 5 ft from the blue line
DOT_Y_FT = 22.0
CIRCLE_RADIUS_FT = 15.0
GOAL_HALF_WIDTH_FT = 3.0
HALF_LENGTH = RINK_LENGTH_FT / 2
HALF_WIDTH = RINK_WIDTH_FT / 2
# The goal line meets the boards inside the corner arc, whose centre is 28 ft in from each end and side.
GOAL_LINE_BOARDS_Y_FT = 36.75  # where the goal line meets the boards: the corner arc (centre 28 ft in from end and side, radius 28) at x = 89 gives 14.5 + sqrt(28^2 - 17^2) = 36.7486, rounded to the quarter-foot


@dataclass(frozen=True)
class Keypoint:
    id: int
    name: str
    x_ft: float
    y_ft: float


def _sided(base: str, x: float, y: float, start_id: int, far_near: bool = True) -> list[Keypoint]:
    """L/R (and far/near) variants of one feature, ids in the order L_far, L_near, R_far, R_near."""
    points = []
    i = start_id
    for side, sx in (("L", -1.0), ("R", 1.0)):
        for fn, sy in ((("far", 1.0), ("near", -1.0)) if far_near else (("", 1.0),)):
            name = f"{base}_{side}" + (f"_{fn}" if fn else "")
            points.append(Keypoint(i, name, sx * x, sy * y))
            i += 1
    return points


KEYPOINTS: tuple[Keypoint, ...] = tuple(
    _sided("goal_line_boards", GOAL_LINE_X_FT, GOAL_LINE_BOARDS_Y_FT, 0)        # 0-3
    + _sided("blue_line_boards", BLUE_LINE_X_FT, HALF_WIDTH, 4)                 # 4-7
    + [Keypoint(8, "centre_line_boards_far", 0.0, HALF_WIDTH),
       Keypoint(9, "centre_line_boards_near", 0.0, -HALF_WIDTH),
       Keypoint(10, "faceoff_dot_centre", 0.0, 0.0)]
    + _sided("faceoff_dot_end", END_DOT_X_FT, DOT_Y_FT, 11)                     # 11-14
    + _sided("faceoff_dot_neutral", NEUTRAL_DOT_X_FT, DOT_Y_FT, 15)             # 15-18
    + [Keypoint(19, "circle_end_L_far_top", -END_DOT_X_FT, DOT_Y_FT + CIRCLE_RADIUS_FT),
       Keypoint(20, "circle_end_L_far_bottom", -END_DOT_X_FT, DOT_Y_FT - CIRCLE_RADIUS_FT),
       Keypoint(21, "circle_end_L_near_top", -END_DOT_X_FT, -DOT_Y_FT + CIRCLE_RADIUS_FT),
       Keypoint(22, "circle_end_L_near_bottom", -END_DOT_X_FT, -DOT_Y_FT - CIRCLE_RADIUS_FT),
       Keypoint(23, "circle_end_R_far_top", END_DOT_X_FT, DOT_Y_FT + CIRCLE_RADIUS_FT),
       Keypoint(24, "circle_end_R_far_bottom", END_DOT_X_FT, DOT_Y_FT - CIRCLE_RADIUS_FT),
       Keypoint(25, "circle_end_R_near_top", END_DOT_X_FT, -DOT_Y_FT + CIRCLE_RADIUS_FT),
       Keypoint(26, "circle_end_R_near_bottom", END_DOT_X_FT, -DOT_Y_FT - CIRCLE_RADIUS_FT),
       Keypoint(27, "centre_circle_top", 0.0, CIRCLE_RADIUS_FT),
       Keypoint(28, "centre_circle_bottom", 0.0, -CIRCLE_RADIUS_FT)]
    + _sided("goal_post", GOAL_LINE_X_FT, GOAL_HALF_WIDTH_FT, 29)               # 29-32
)
assert [k.id for k in KEYPOINTS] == list(range(33))

KEYPOINT_BY_NAME = {k.name: k for k in KEYPOINTS}
KEYPOINT_IDS = frozenset(k.id for k in KEYPOINTS)


def template_points() -> np.ndarray:
    """(33, 2) array of (x_ft, y_ft), row i for keypoint i."""
    return np.array([[k.x_ft, k.y_ft] for k in KEYPOINTS], dtype=float)


def mirror_name(name: str) -> str:
    if "_L_" in name:
        return name.replace("_L_", "_R_", 1)
    if name.endswith("_L"):
        return name[:-2] + "_R"
    if "_R_" in name:
        return name.replace("_R_", "_L_", 1)
    if name.endswith("_R"):
        return name[:-2] + "_L"
    return name


def rink_to_px(x_ft: float, y_ft: float, scale: float, margin: int) -> tuple[int, int]:
    """Top-down image coordinates: camera-left at the left, far side at the top."""
    return (int(round(margin + (x_ft + HALF_LENGTH) * scale)),
            int(round(margin + (HALF_WIDTH - y_ft) * scale)))


RED, BLUE, WHITE, GREY, DARK, GREEN, BLACK = (0, 0, 220), (220, 80, 0), (245, 245, 245), (150, 150, 150), (60, 60, 60), (0, 200, 0), (0, 0, 0)
HIGHLIGHT = (0, 0, 255)


def draw_rink(scale: float = 4.0, margin: int = 20, highlight: int | None = None,
              labelled: set[int] = frozenset(), not_visible: set[int] = frozenset()) -> np.ndarray:
    width = int(round(RINK_LENGTH_FT * scale + 2 * margin))
    height = int(round(RINK_WIDTH_FT * scale + 2 * margin))
    image = np.full((height, width, 3), WHITE, dtype=np.uint8)
    px = lambda x, y: rink_to_px(x, y, scale, margin)
    r = int(round(CORNER_RADIUS_FT * scale))
    # Boards: straight segments plus four quarter arcs.
    cx_l, cx_r = -HALF_LENGTH + CORNER_RADIUS_FT, HALF_LENGTH - CORNER_RADIUS_FT
    cy_f, cy_n = HALF_WIDTH - CORNER_RADIUS_FT, -HALF_WIDTH + CORNER_RADIUS_FT
    cv2.line(image, px(cx_l, HALF_WIDTH), px(cx_r, HALF_WIDTH), BLACK, 2)
    cv2.line(image, px(cx_l, -HALF_WIDTH), px(cx_r, -HALF_WIDTH), BLACK, 2)
    cv2.line(image, px(-HALF_LENGTH, cy_f), px(-HALF_LENGTH, cy_n), BLACK, 2)
    cv2.line(image, px(HALF_LENGTH, cy_f), px(HALF_LENGTH, cy_n), BLACK, 2)
    for (cx, cy), angle in (((cx_l, cy_f), 180), ((cx_r, cy_f), 270), ((cx_r, cy_n), 0), ((cx_l, cy_n), 90)):
        cv2.ellipse(image, px(cx, cy), (r, r), 0, angle, angle + 90, BLACK, 2)
    # Lines.
    for x in (-GOAL_LINE_X_FT, GOAL_LINE_X_FT):
        cv2.line(image, px(x, GOAL_LINE_BOARDS_Y_FT), px(x, -GOAL_LINE_BOARDS_Y_FT), RED, 2)
    for x in (-BLUE_LINE_X_FT, BLUE_LINE_X_FT):
        cv2.line(image, px(x, HALF_WIDTH), px(x, -HALF_WIDTH), BLUE, 3)
    cv2.line(image, px(0, HALF_WIDTH), px(0, -HALF_WIDTH), RED, 3)
    # Circles and dots.
    cr = int(round(CIRCLE_RADIUS_FT * scale))
    for cx, cy in ((0, 0), (-END_DOT_X_FT, DOT_Y_FT), (-END_DOT_X_FT, -DOT_Y_FT), (END_DOT_X_FT, DOT_Y_FT), (END_DOT_X_FT, -DOT_Y_FT)):
        cv2.circle(image, px(cx, cy), cr, RED, 1)
    for cx, cy in ((0, 0), (-END_DOT_X_FT, DOT_Y_FT), (-END_DOT_X_FT, -DOT_Y_FT), (END_DOT_X_FT, DOT_Y_FT), (END_DOT_X_FT, -DOT_Y_FT),
                   (-NEUTRAL_DOT_X_FT, DOT_Y_FT), (-NEUTRAL_DOT_X_FT, -DOT_Y_FT), (NEUTRAL_DOT_X_FT, DOT_Y_FT), (NEUTRAL_DOT_X_FT, -DOT_Y_FT)):
        cv2.circle(image, px(cx, cy), max(2, int(scale)), RED, -1)
    for x in (-GOAL_LINE_X_FT, GOAL_LINE_X_FT):
        cv2.line(image, px(x, GOAL_HALF_WIDTH_FT), px(x, -GOAL_HALF_WIDTH_FT), BLACK, 3)
    # Keypoints.
    for k in KEYPOINTS:
        p = px(k.x_ft, k.y_ft)
        if k.id == highlight:
            cv2.circle(image, p, max(5, int(1.5 * scale)), HIGHLIGHT, -1)
        elif k.id in labelled:
            cv2.circle(image, p, max(3, int(scale)), GREEN, -1)
        elif k.id in not_visible:
            cv2.circle(image, p, max(3, int(scale)), DARK, 1)
            cv2.line(image, (p[0] - 3, p[1] - 3), (p[0] + 3, p[1] + 3), DARK, 1)
        else:
            cv2.circle(image, p, max(3, int(scale)), GREY, 1)
    return image
