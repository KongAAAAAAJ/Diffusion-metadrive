import numpy as np

from metadrive.policy.diffusion_policy.eval_transfuser_open_loop import (
    _extract_multi_point_guidance_points,
    _matplotlib_color,
    _mode_name,
)
from metadrive.policy.diffusion_policy.mode_definitions import build_mode_slots


def test_extract_multi_point_guidance_points_uses_anchor_endpoints() -> None:
    coarse = np.asarray(
        [
            [[0.0, 0.0], [4.0, 1.0]],
            [[0.0, 0.0], [5.0, -2.0]],
        ],
        dtype=np.float32,
    )

    points = _extract_multi_point_guidance_points(coarse)

    np.testing.assert_allclose(points, np.asarray([[4.0, 1.0], [5.0, -2.0]], dtype=np.float64))


def test_extract_multi_point_guidance_points_rejects_invalid_shapes() -> None:
    assert _extract_multi_point_guidance_points(None) is None
    assert _extract_multi_point_guidance_points(np.zeros((8, 2), dtype=np.float32)) is None


def test_matplotlib_color_converts_opencv_bgr_to_rgb_unit_float() -> None:
    assert _matplotlib_color((245, 200, 11)) == (11 / 255.0, 200 / 255.0, 245 / 255.0)
    assert _matplotlib_color("#CC79A7") == "#CC79A7"


def test_mode_name_uses_runtime_mode_slots() -> None:
    slots = build_mode_slots(
        keep_lane_count=5,
        lane_change_left_count=5,
        lane_change_right_count=5,
        emergency_stop_count=1,
    )

    assert _mode_name(15, slots) == "STOP"
