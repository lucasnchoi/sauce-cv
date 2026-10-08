import numpy as np
import pytest

from tracker import rink
from tracker.rink import KEYPOINTS, KEYPOINT_BY_NAME, Keypoint, draw_rink, mirror_name, rink_to_px, template_points


def test_there_are_33_keypoints_with_ids_in_order_and_unique_names():
    assert len(KEYPOINTS) == 33
    assert [k.id for k in KEYPOINTS] == list(range(33))
    assert len({k.name for k in KEYPOINTS}) == 33


def test_known_coordinates():
    assert KEYPOINT_BY_NAME["goal_line_boards_L_far"] == Keypoint(0, "goal_line_boards_L_far", -89.0, 36.75)
    assert KEYPOINT_BY_NAME["blue_line_boards_R_near"] == Keypoint(7, "blue_line_boards_R_near", 25.0, -42.5)
    assert KEYPOINT_BY_NAME["faceoff_dot_centre"] == Keypoint(10, "faceoff_dot_centre", 0.0, 0.0)
    assert KEYPOINT_BY_NAME["faceoff_dot_end_L_near"] == Keypoint(12, "faceoff_dot_end_L_near", -69.0, -22.0)
    assert KEYPOINT_BY_NAME["faceoff_dot_neutral_R_far"] == Keypoint(17, "faceoff_dot_neutral_R_far", 20.0, 22.0)
    assert KEYPOINT_BY_NAME["circle_end_L_far_top"] == Keypoint(19, "circle_end_L_far_top", -69.0, 37.0)
    assert KEYPOINT_BY_NAME["circle_end_R_near_bottom"] == Keypoint(26, "circle_end_R_near_bottom", 69.0, -37.0)
    assert KEYPOINT_BY_NAME["centre_circle_bottom"] == Keypoint(28, "centre_circle_bottom", 0.0, -15.0)
    assert KEYPOINT_BY_NAME["goal_post_R_far"] == Keypoint(31, "goal_post_R_far", 89.0, 3.0)


def test_goal_line_meets_the_boards_inside_the_corner_arc():
    # Corner arc centre (72, 14.5), radius 28; at x = 89 the arc is at y = 14.5 + sqrt(28^2 - 17^2)
    assert rink.GOAL_LINE_BOARDS_Y_FT == pytest.approx(14.5 + (28**2 - 17**2) ** 0.5, abs=0.01)
    assert rink.GOAL_LINE_BOARDS_Y_FT == pytest.approx(36.75, abs=0.01)


def test_every_sided_point_has_its_mirror_with_x_negated():
    for k in KEYPOINTS:
        other = KEYPOINT_BY_NAME[mirror_name(k.name)]
        assert other.x_ft == -k.x_ft and other.y_ft == k.y_ft
    assert mirror_name("goal_post_L_far") == "goal_post_R_far"
    assert mirror_name("centre_circle_top") == "centre_circle_top"


def test_all_points_lie_inside_the_rink():
    pts = template_points()
    assert pts.shape == (33, 2)
    assert (np.abs(pts[:, 0]) <= 100).all() and (np.abs(pts[:, 1]) <= 42.5).all()
    assert pts[0].tolist() == [-89.0, 36.75]


def test_rink_to_px_puts_the_far_side_at_the_top():
    assert rink_to_px(-100, 42.5, scale=4.0, margin=20) == (20, 20)
    assert rink_to_px(100, -42.5, scale=4.0, margin=20) == (820, 360)
    assert rink_to_px(0, 0, scale=4.0, margin=20) == (420, 190)


def test_draw_rink_has_the_right_size_and_coloured_lines():
    image = draw_rink(scale=4.0, margin=20)
    assert image.shape == (380, 840, 3)
    cx, cy = rink_to_px(0, 30, 4.0, 20)                      # on the centre line, between the circles
    b, g, r = image[cy, cx].tolist()
    assert r > 150 and b < 100                                  # red
    bx, by = rink_to_px(25, 30, 4.0, 20)                     # on a blue line
    b, g, r = image[by, bx].tolist()
    assert b > 150 and r < 100                                  # blue


def test_draw_rink_highlights_and_marks_points():
    plain = draw_rink()
    marked = draw_rink(highlight=10, labelled={0}, not_visible={1})
    cx, cy = rink_to_px(0, 0, 4.0, 20)
    assert marked[cy, cx].tolist() != plain[cy, cx].tolist()   # highlight changes the centre dot
    assert marked[cy, cx][2] > 150                              # red


def test_draw_rink_numbers_write_text_beside_the_points():
    plain = draw_rink()
    numbered = draw_rink(numbers={k.id: str(k.id) for k in KEYPOINTS})
    assert numbered.shape == plain.shape and (numbered != plain).any()
