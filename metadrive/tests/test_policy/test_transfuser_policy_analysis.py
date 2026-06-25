from __future__ import annotations

import numpy as np
import torch

from models.diffusion.transfuser_config import build_transfuser_config
from models.diffusion.transfuser_features import build_status_feature
from models.diffusion.transfuser_policy import TransfuserPolicy, compute_trajectory_control


def test_build_status_feature_keeps_navigation_intent():
    config = build_transfuser_config("small")
    ego_state = np.arange(19, dtype=np.float32)

    status = build_status_feature(ego_state, config).numpy()

    np.testing.assert_allclose(status, ego_state)


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


def test_compute_trajectory_control_uses_waypoint_spacing_for_reference_speed():
    trajectory = np.asarray(
        [
            [0.25, 0.0, 0.0],
            [0.50, 0.0, 0.0],
            [0.75, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    action, debug = compute_trajectory_control(
        trajectory=trajectory,
        lookahead_index=1,
        current_speed_km_h=10.0,
        target_speed_km_h=30.0,
        controller_type="stabilized",
    )

    assert np.isclose(debug["trajectory_target_speed_km_h"], 1.8, atol=1e-3)
    assert debug["speed_error"] < 0.0
    assert action[1] < 0.0


def test_compute_trajectory_control_short_stop_like_trajectory_brakes_hard():
    trajectory = np.asarray(
        [
            [0.05, 0.0, 0.0],
            [0.08, 0.0, 0.0],
            [0.10, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    action, debug = compute_trajectory_control(
        trajectory=trajectory,
        lookahead_index=1,
        current_speed_km_h=29.6,
        target_speed_km_h=30.0,
        controller_type="stabilized",
    )

    assert debug["trajectory_target_speed_km_h"] < 1.0
    assert debug["speed_error"] < -20.0
    assert action[1] < -0.9


def test_transfuser_policy_act_uses_multimodal_inference_and_exposes_candidates(monkeypatch):
    policy = object.__new__(TransfuserPolicy)
    policy._device = torch.device("cpu")
    policy._model_config = object()
    policy._lookahead_index = 1
    policy._target_speed_km_h = 30.0
    policy._controller_type = "stabilized"
    policy.action_info = {}
    policy.control_object = type("Ego", (), {"name": "agent0", "speed_km_h": 20.0})()
    policy._update_trajectory_visualization = lambda trajectory: None
    policy._trajectory_to_action = lambda trajectory: (
        np.asarray([0.1, 0.2], dtype=np.float32),
        {"steering": 0.1, "throttle": 0.2},
    )

    class FakeObs:
        def __init__(self):
            self.current_observation = {"obs": "cached-agent0"}

        def observe(self, control_object):
            return {"obs": control_object.name}

    fake_obs = FakeObs()
    fake_engine = type(
        "Engine",
        (),
        {
            "agent_manager": type("AgentManager", (), {"observations": {"agent0": fake_obs}})(),
        },
    )()
    monkeypatch.setattr("metadrive.policy.base_policy.get_engine", lambda: fake_engine)

    fake_camera = torch.zeros((3, 256, 768), dtype=torch.float32)
    fake_lidar = torch.zeros((1, 256, 256), dtype=torch.float32)
    fake_status = torch.zeros((19,), dtype=torch.float32)
    fake_ego_state = torch.zeros((19,), dtype=torch.float32)
    monkeypatch.setattr(
        "models.diffusion.transfuser_policy.observation_to_features",
        lambda observation, config, vehicle=None: (
            {
                "camera_feature": fake_camera,
                "lidar_feature": fake_lidar,
                "status_feature": fake_status,
                "ego_state": fake_ego_state,
                "target_point": torch.tensor([3.0, 0.0], dtype=torch.float32),
            }
            if observation == {"obs": "cached-agent0"}
            else (_ for _ in ()).throw(AssertionError("policy should use cached observation"))
        ),
    )

    class FakeModel:
        def infer_multimodal(self, features):
            return {
                "trajectory": torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.5, 0.1]]], dtype=torch.float32),
                "trajectory_candidates": torch.tensor(
                    [[
                        [[1.0, 0.0, 0.0], [2.0, 0.5, 0.1]],
                        [[1.0, 1.0, 0.0], [2.0, 1.5, 0.2]],
                    ]],
                    dtype=torch.float32,
                ),
                "trajectory_mode_idx": torch.tensor([1], dtype=torch.int64),
                "trajectory_mode_logits": torch.tensor([[0.2, 0.8]], dtype=torch.float32),
            }

    policy._model = FakeModel()

    action = policy.act()

    assert np.allclose(action, np.asarray([0.1, 0.2], dtype=np.float32))
    assert "trajectory_candidates" in policy.action_info
    assert policy.action_info["trajectory_candidates"].shape == (2, 2, 3)
    assert policy.action_info["trajectory_mode_idx"] == 1
    assert policy.action_info["camera_feature"].shape == (3, 256, 768)
    assert policy.action_info["lidar_feature"].shape == (1, 256, 256)
    assert policy.action_info["status_feature"].shape == (19,)
    assert policy.action_info["ego_state"].shape == (19,)
    assert policy.action_info["target_point"].shape == (2,)
