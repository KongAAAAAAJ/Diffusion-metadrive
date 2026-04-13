from __future__ import annotations

import numpy as np
import torch

from metadrive.policy.diffusion_policy.transfuser_config import build_transfuser_config
from metadrive.policy.diffusion_policy.transfuser_features import (
    compute_target_point,
    compute_target_point_from_sample,
    sample_to_features_targets,
)
from metadrive.policy.diffusion_policy.transfuser_model_v2 import DiffMotionPlanningRefinementModule


class _FakeLane:
    def __init__(self, length: float):
        self.length = float(length)

    def local_coordinates(self, position):
        return float(position[0]), float(position[1])

    def position(self, longitudinal, lateral):
        return np.asarray([float(longitudinal), float(lateral)], dtype=np.float32)


class _FakeNavigation:
    def __init__(self, lane):
        self.current_ref_lanes = [lane]


class _FakeVehicle:
    def __init__(self, lane, position, heading_theta: float, speed_km_h: float):
        self.navigation = _FakeNavigation(lane)
        self.position = np.asarray(position, dtype=np.float32)
        self.heading_theta = float(heading_theta)
        self.speed_km_h = float(speed_km_h)


def _raw_sample() -> dict:
    return {
        "left_camera": np.zeros((10, 16, 3), dtype=np.uint8),
        "front_camera": np.zeros((10, 16, 3), dtype=np.uint8),
        "right_camera": np.zeros((10, 16, 3), dtype=np.uint8),
        "lidar": np.zeros((32,), dtype=np.float32),
        "ego_state": np.zeros((19,), dtype=np.float32),
        "trajectory": np.zeros((8, 3), dtype=np.float32),
        "agent_states": np.zeros((16, 5), dtype=np.float32),
        "agent_labels": np.zeros((16,), dtype=bool),
        "bev_raster": np.zeros((3, 16, 16), dtype=np.uint8),
        "reference_pose_world": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "future_reference_pose_world": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [7.0, 0.0, 0.0],
                [8.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "reference_lane_index": np.asarray(0, dtype=np.int16),
        "future_reference_lane_index": np.asarray([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int16),
        "ego_speed_km_h": np.asarray(1.0, dtype=np.float32),
    }


def test_compute_target_point_uses_min_forward_distance_for_low_speed():
    config = build_transfuser_config("small", target_point_min_forward_distance_m=3.0)
    vehicle = _FakeVehicle(_FakeLane(length=50.0), position=[2.0, 0.0], heading_theta=0.0, speed_km_h=1.0)

    target_point = compute_target_point(vehicle, config).numpy()

    np.testing.assert_allclose(target_point, np.asarray([3.0, 0.0], dtype=np.float32))


def test_compute_target_point_clamps_at_current_lane_end():
    config = build_transfuser_config("small", target_point_min_forward_distance_m=3.0)
    vehicle = _FakeVehicle(_FakeLane(length=50.0), position=[49.0, 0.0], heading_theta=0.0, speed_km_h=50.0)

    target_point = compute_target_point(vehicle, config).numpy()

    np.testing.assert_allclose(target_point, np.asarray([1.0, 0.0], dtype=np.float32))


def test_compute_target_point_from_sample_stays_on_current_lane():
    config = build_transfuser_config("small", target_point_min_forward_distance_m=3.0)
    sample = {
        "reference_pose_world": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "future_reference_pose_world": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "reference_lane_index": np.asarray(0, dtype=np.int16),
        "future_reference_lane_index": np.asarray([0, 0, 1], dtype=np.int16),
        "ego_speed_km_h": np.asarray(30.0, dtype=np.float32),
    }

    target_point = compute_target_point_from_sample(sample, config).numpy()

    np.testing.assert_allclose(target_point, np.asarray([2.0, 0.0], dtype=np.float32))


def test_sample_to_features_targets_adds_target_point():
    config = build_transfuser_config("small", target_point_min_forward_distance_m=3.0)

    features, targets = sample_to_features_targets(_raw_sample(), config)

    assert tuple(features["target_point"].shape) == (2,)
    np.testing.assert_allclose(features["target_point"].numpy(), np.asarray([3.0, 0.0], dtype=np.float32))
    assert tuple(targets["trajectory"].shape) == (8, 3)


def test_refinement_module_accepts_target_point_zero_fallback():
    module = DiffMotionPlanningRefinementModule(
        embed_dims=16,
        ego_fut_ts=8,
        ego_fut_mode=4,
        target_point_dim=6,
    )
    traj_feature = torch.zeros((2, 4, 16), dtype=torch.float32)

    plan_reg_none, plan_cls_none = module(traj_feature, target_point_embed=None)
    plan_reg_zero, plan_cls_zero = module(traj_feature, target_point_embed=torch.zeros((2, 6), dtype=torch.float32))

    assert tuple(plan_reg_none.shape) == (2, 4, 8, 3)
    assert tuple(plan_cls_none.shape) == (2, 4)
    torch.testing.assert_close(plan_reg_none, plan_reg_zero)
    torch.testing.assert_close(plan_cls_none, plan_cls_zero)
