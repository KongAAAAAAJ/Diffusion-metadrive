from __future__ import annotations

import numpy as np

from metadrive.policy.diffusion_policy.transfuser_config import build_transfuser_config
from metadrive.policy.diffusion_policy.transfuser_features import build_status_feature
from metadrive.policy.diffusion_policy.transfuser_policy import compute_trajectory_control


def test_build_status_feature_keeps_navigation_intent():
    config = build_transfuser_config("small")
    ego_state = np.arange(19, dtype=np.float32)

    status = build_status_feature(ego_state, config).numpy()

    np.testing.assert_allclose(status, np.asarray([2, 3, 4, 5, 6, 8, 9, 10], dtype=np.float32))


def test_compute_trajectory_control_stabilized_has_deadzone_for_small_bias():
    trajectory = np.asarray(
        [
            [3.0, 0.05, 0.01],
            [6.0, 0.08, 0.02],
            [9.0, 0.10, 0.02],
        ],
        dtype=np.float32,
    )

    action, debug = compute_trajectory_control(
        trajectory=trajectory,
        lookahead_index=1,
        current_speed_km_h=20.0,
        target_speed_km_h=30.0,
        controller_type="stabilized",
    )

    assert float(action[0]) == 0.0
    assert debug["steering"] == 0.0


def test_compute_trajectory_control_preserves_left_right_sign():
    right_traj = np.asarray([[8.0, 1.5, 0.15]], dtype=np.float32)
    left_traj = np.asarray([[8.0, -1.5, -0.15]], dtype=np.float32)

    right_action, right_debug = compute_trajectory_control(
        trajectory=right_traj,
        lookahead_index=0,
        current_speed_km_h=20.0,
        target_speed_km_h=30.0,
        controller_type="stabilized",
    )
    left_action, left_debug = compute_trajectory_control(
        trajectory=left_traj,
        lookahead_index=0,
        current_speed_km_h=20.0,
        target_speed_km_h=30.0,
        controller_type="stabilized",
    )

    assert right_action[0] > 0.0
    assert left_action[0] < 0.0
    assert right_debug["waypoint_y"] > 0.0
    assert left_debug["waypoint_y"] < 0.0
