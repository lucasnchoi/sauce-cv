from tracker.labels.tool import inside_frame


def test_inside_frame_rejects_clicks_on_the_legend_panel():
    shape = (48, 64, 3)
    assert inside_frame(0, 0, shape) and inside_frame(63, 47, shape)
    assert not inside_frame(64, 10, shape) and not inside_frame(10, 48, shape)
    assert not inside_frame(-1, 5, shape)


def test_legend_keypoint_at_picks_the_nearest_keypoint_within_reach():
    from tracker.labels.tool import legend_keypoint_at
    assert legend_keypoint_at(420, 190) == 10
    assert legend_keypoint_at(423, 188) == 10
    assert legend_keypoint_at(5, 5) is None
