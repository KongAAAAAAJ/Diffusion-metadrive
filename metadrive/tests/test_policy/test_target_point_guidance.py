from __future__ import annotations

import numpy as np
import torch
from types import SimpleNamespace

from models.diffusion.transfuser_config import build_transfuser_config
import models.diffusion.transfuser_features as transfuser_features
from models.diffusion.transfuser_features import (
    LaneDecision,
    _build_live_target_lane_polyline,
    _build_topology_polyline_from_live_vehicle,
    compute_target_point,
    compute_target_point_from_sample,
    observation_to_features,
    sample_to_features_targets,
)
from models.diffusion.transfuser_model_v2 import DiffMotionPlanningRefinementModule


class _FakeLane:
    def __init__(self, length: float, lateral_offset: float = 0.0, lane_index=None):
        self.length = float(length)
        self._lateral_offset = float(lateral_offset)
        self.index = lane_index

    def local_coordinates(self, position):
        return float(position[0]), float(position[1] - self._lateral_offset)

    def position(self, longitudinal, lateral):
        return np.asarray([float(longitudinal), float(lateral) + self._lateral_offset], dtype=np.float32)

    def width_at(self, longitudinal):
        return 3.5


class _FakeNavigation:
    def __init__(self, lane, next_ref_lanes=None):
        self.current_ref_lanes = [lane]
        self.next_ref_lanes = list(next_ref_lanes) if next_ref_lanes is not None else None


class _FakeVehicle:
    def __init__(self, lane, position, heading_theta: float, speed_km_h: float, next_ref_lanes=None):
        self.navigation = _FakeNavigation(lane, next_ref_lanes=next_ref_lanes)
        self.position = np.asarray(position, dtype=np.float32)
        self.heading_theta = float(heading_theta)
        self.speed_km_h = float(speed_km_h)
        self.lane = lane
        self.engine = None


class _FakeRoadNetwork:
    def __init__(self, graph):
        self.graph = graph


class _FakeMap:
    def __init__(self, road_graph):
        self.road_network = _FakeRoadNetwork(road_graph)


class _FakeLidar:
    def __init__(self, objects):
        self._objects = list(objects)

    def get_surrounding_objects(self, vehicle):
        return list(self._objects)


class _FakeFrontBackObjects:
    def __init__(self, front_obj=None, front_dist=999.0):
        self._front_obj = front_obj
        self._front_dist = float(front_dist)

    def has_front_object(self):
        return self._front_obj is not None

    def front_object(self):
        return self._front_obj

    def front_min_distance(self):
        return self._front_dist

    def left_lane_exist(self):
        return False

    def right_lane_exist(self):
        return False


class _FakeDynamicObject:
    def __init__(
        self,
        lane,
        position,
        speed_km_h: float = 0.0,
        acceleration: float = 0.0,
        length: float = 4.5,
        width: float = 2.0,
    ):
        self.lane = lane
        self.position = np.asarray(position, dtype=np.float32)
        self.speed_km_h = float(speed_km_h)
        self.acceleration = float(acceleration)
        self.LENGTH = float(length)
        self.WIDTH = float(width)


def _raw_sample() -> dict:
    return {
        "left_camera": np.zeros((10, 16, 3), dtype=np.uint8),
        "front_camera": np.zeros((10, 16, 3), dtype=np.uint8),
        "right_camera": np.zeros((10, 16, 3), dtype=np.uint8),
        "lidar": np.zeros((32,), dtype=np.float32),
        "ego_state": np.zeros((19,), dtype=np.float32),
        "trajectory": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.2, 0.0],
                [3.0, 0.4, 0.0],
                [4.0, 0.6, 0.0],
                [5.0, 0.8, 0.0],
                [6.0, 1.0, 0.0],
                [7.0, 1.2, 0.0],
                [8.0, 1.5, 0.0],
            ],
            dtype=np.float32,
        ),
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


def test_compute_target_point_without_front_vehicle_uses_idm_free_road_progress():
    config = build_transfuser_config("small", target_point_min_forward_distance_m=3.0)
    vehicle = _FakeVehicle(_FakeLane(length=50.0), position=[2.0, 0.0], heading_theta=0.0, speed_km_h=1.0)

    target_point = compute_target_point(vehicle, config).numpy()

    assert float(target_point[0]) > 3.0
    np.testing.assert_allclose(target_point[1], np.asarray(0.0, dtype=np.float32), atol=1e-5)


def test_compute_target_point_clamps_at_current_lane_end():
    config = build_transfuser_config("small", target_point_min_forward_distance_m=3.0)
    vehicle = _FakeVehicle(_FakeLane(length=50.0), position=[49.0, 0.0], heading_theta=0.0, speed_km_h=50.0)

    target_point = compute_target_point(vehicle, config).numpy()

    np.testing.assert_allclose(target_point, np.asarray([1.0, 0.0], dtype=np.float32))


def test_compute_target_point_respects_reachable_limit_when_stationary(monkeypatch):
    config = build_transfuser_config(
        "small",
        target_point_min_forward_distance_m=3.0,
        target_point_prediction_horizon_s=4.0,
        target_point_max_reachable_accel_mps2=0.25,
        target_point_max_reachable_jerk_mps3=0.25,
    )
    current_poly = np.stack([np.linspace(0.0, 40.0, 41), np.zeros((41,), dtype=np.float32)], axis=1).astype(np.float32)
    monkeypatch.setattr(
        transfuser_features,
        "_build_mode_context_for_target_point",
        lambda vehicle, current_map=None: SimpleNamespace(
            current_lane_polyline=current_poly,
            left_lane_polyline=None,
            right_lane_polyline=None,
            has_left_adjacent=False,
            has_right_adjacent=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        transfuser_features,
        "_find_front_back_objects_for_target_lane",
        lambda vehicle, lane_decision, mode_context=None: _FakeFrontBackObjects(front_obj=None, front_dist=999.0),
        raising=False,
    )
    vehicle = _FakeVehicle(_FakeLane(length=50.0), position=[0.0, 0.0], heading_theta=0.0, speed_km_h=0.0)

    target_point = compute_target_point(vehicle, config).numpy()

    assert 0.0 < float(target_point[0]) <= 3.0
    np.testing.assert_allclose(target_point[1], 0.0, atol=1e-5)


def test_compute_target_point_keeps_front_safe_gap(monkeypatch):
    config = build_transfuser_config(
        "small",
        target_point_prediction_horizon_s=4.0,
        target_point_front_safe_gap_m=10.0,
        target_point_max_reachable_accel_mps2=4.0,
        target_point_max_reachable_jerk_mps3=10.0,
    )
    current_poly = np.stack([np.linspace(0.0, 80.0, 81), np.zeros((81,), dtype=np.float32)], axis=1).astype(np.float32)
    monkeypatch.setattr(
        transfuser_features,
        "_build_mode_context_for_target_point",
        lambda vehicle, current_map=None: SimpleNamespace(
            current_lane_polyline=current_poly,
            left_lane_polyline=None,
            right_lane_polyline=None,
            has_left_adjacent=False,
            has_right_adjacent=False,
        ),
        raising=False,
    )
    lane = _FakeLane(length=100.0)
    vehicle = _FakeVehicle(lane, position=[0.0, 0.0], heading_theta=0.0, speed_km_h=36.0)
    front_obj = _FakeDynamicObject(lane, position=[15.0, 0.0], speed_km_h=9.0, acceleration=0.0)
    monkeypatch.setattr(
        transfuser_features,
        "_find_front_back_objects_for_target_lane",
        lambda vehicle, lane_decision, mode_context=None: _FakeFrontBackObjects(front_obj=front_obj, front_dist=15.0),
        raising=False,
    )

    target_point = compute_target_point(vehicle, config).numpy()

    assert float(target_point[0]) <= 12.8
    np.testing.assert_allclose(target_point[1], 0.0, atol=1e-5)


def test_compute_target_point_avoids_future_overlap_with_non_front_vehicle(monkeypatch):
    config = build_transfuser_config(
        "small",
        target_point_prediction_horizon_s=4.0,
        target_point_non_front_overlap_buffer_m=1.0,
        target_point_max_reachable_accel_mps2=4.0,
        target_point_max_reachable_jerk_mps3=10.0,
    )
    current_poly = np.stack([np.linspace(0.0, 80.0, 81), np.zeros((81,), dtype=np.float32)], axis=1).astype(np.float32)
    monkeypatch.setattr(
        transfuser_features,
        "_build_mode_context_for_target_point",
        lambda vehicle, current_map=None: SimpleNamespace(
            current_lane_polyline=current_poly,
            left_lane_polyline=None,
            right_lane_polyline=None,
            has_left_adjacent=False,
            has_right_adjacent=False,
        ),
        raising=False,
    )
    lane = _FakeLane(length=100.0)
    vehicle = _FakeVehicle(lane, position=[0.0, 0.0], heading_theta=0.0, speed_km_h=36.0)
    vehicle.lidar = _FakeLidar([_FakeDynamicObject(lane, position=[39.0, 0.0], speed_km_h=0.0, acceleration=0.0)])
    monkeypatch.setattr(
        transfuser_features,
        "_find_front_back_objects_for_target_lane",
        lambda vehicle, lane_decision, mode_context=None: _FakeFrontBackObjects(front_obj=None, front_dist=999.0),
        raising=False,
    )

    target_point = compute_target_point(vehicle, config).numpy()

    assert float(target_point[0]) < 39.0
    np.testing.assert_allclose(target_point[1], 0.0, atol=1e-5)


def test_compute_target_point_lane_change_still_follows_adjacent_lane(monkeypatch):
    config = build_transfuser_config(
        "small",
        target_point_prediction_horizon_s=4.0,
        target_point_max_reachable_accel_mps2=4.0,
        target_point_max_reachable_jerk_mps3=10.0,
    )
    current_poly = np.stack([np.linspace(0.0, 30.0, 31), np.zeros((31,), dtype=np.float32)], axis=1).astype(np.float32)
    left_poly = np.stack([np.linspace(0.0, 30.0, 31), np.full((31,), 3.5, dtype=np.float32)], axis=1).astype(np.float32)
    monkeypatch.setattr(
        transfuser_features,
        "_build_mode_context_for_target_point",
        lambda vehicle, current_map=None: SimpleNamespace(
            current_lane_polyline=current_poly,
            left_lane_polyline=left_poly,
            right_lane_polyline=None,
            has_left_adjacent=True,
            has_right_adjacent=False,
        ),
        raising=False,
    )
    lane = _FakeLane(length=50.0)
    vehicle = _FakeVehicle(lane, position=[0.0, 0.0], heading_theta=0.0, speed_km_h=18.0)
    monkeypatch.setattr(
        transfuser_features,
        "_find_front_back_objects_for_target_lane",
        lambda vehicle, lane_decision, mode_context=None: _FakeFrontBackObjects(front_obj=None, front_dist=999.0),
        raising=False,
    )

    target_point = compute_target_point(vehicle, config, lane_decision=LaneDecision.CHANGE_LEFT).numpy()

    assert float(target_point[0]) > 0.0
    np.testing.assert_allclose(target_point[1], 3.5, atol=1e-5)


def test_observation_to_features_target_point_uses_live_expert_lane_decision(monkeypatch):
    config = build_transfuser_config("small")
    lane = _FakeLane(length=50.0, lane_index=("s", "e", 1))
    vehicle = _FakeVehicle(lane, position=[0.0, 0.0], heading_theta=0.0, speed_km_h=18.0)
    observation = {
        "rgb_left": np.zeros((4, 4, 3), dtype=np.uint8),
        "rgb_front": np.zeros((4, 4, 3), dtype=np.uint8),
        "rgb_right": np.zeros((4, 4, 3), dtype=np.uint8),
        "lidar": np.zeros((32,), dtype=np.float32),
        "ego_state": np.zeros((19,), dtype=np.float32),
    }
    calls = []

    def _fake_compute_target_point(vehicle_arg, config_arg, lane_decision=LaneDecision.KEEP):
        calls.append(lane_decision)
        return torch.tensor([1.0, 2.0], dtype=torch.float32)

    monkeypatch.setattr(transfuser_features, "compute_target_point", _fake_compute_target_point, raising=False)

    features = observation_to_features(
        observation,
        config,
        vehicle=vehicle,
        lane_decision=LaneDecision.CHANGE_LEFT,
    )

    np.testing.assert_allclose(features["target_point"].numpy(), np.asarray([1.0, 2.0], dtype=np.float32))
    assert calls == [LaneDecision.KEEP]


def test_observation_to_features_topology_polyline_uses_keep_lane_geometry(monkeypatch):
    config = build_transfuser_config("small")
    lane = _FakeLane(length=50.0, lane_index=("s", "e", 1))
    vehicle = _FakeVehicle(lane, position=[0.0, 0.0], heading_theta=0.0, speed_km_h=18.0)
    observation = {
        "rgb_left": np.zeros((4, 4, 3), dtype=np.uint8),
        "rgb_front": np.zeros((4, 4, 3), dtype=np.uint8),
        "rgb_right": np.zeros((4, 4, 3), dtype=np.uint8),
        "lidar": np.zeros((32,), dtype=np.float32),
        "ego_state": np.zeros((19,), dtype=np.float32),
    }
    calls = []

    monkeypatch.setattr(
        transfuser_features,
        "_build_topology_polyline_from_live_vehicle",
        lambda vehicle_arg, lane_decision: calls.append(lane_decision) or torch.tensor(
            [[0.0, 0.0], [1.0, 0.0]],
            dtype=torch.float32,
        ),
        raising=False,
    )

    features = observation_to_features(
        observation,
        config,
        vehicle=vehicle,
        lane_decision=LaneDecision.CHANGE_LEFT,
    )

    np.testing.assert_allclose(features["topology_polyline"].numpy(), np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32))
    assert calls == [LaneDecision.KEEP]


def test_build_topology_polyline_from_live_vehicle_prefers_live_target_lane_geometry(monkeypatch):
    lane = _FakeLane(length=50.0, lane_index=("s", "e", 1))
    vehicle = _FakeVehicle(lane, position=[0.0, 0.0], heading_theta=0.0, speed_km_h=18.0)
    live_polyline = np.asarray([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(
        transfuser_features,
        "_build_live_target_lane_polyline",
        lambda vehicle_arg, lane_decision: live_polyline,
        raising=False,
    )
    monkeypatch.setattr(
        transfuser_features,
        "_build_mode_context_for_target_point",
        lambda vehicle_arg: SimpleNamespace(
            current_lane_polyline=np.asarray([[0.0, 0.0], [2.0, -2.0], [4.0, -4.0]], dtype=np.float32),
            left_lane_polyline=None,
            right_lane_polyline=None,
            has_left_adjacent=False,
            has_right_adjacent=False,
        ),
        raising=False,
    )

    topology_polyline = _build_topology_polyline_from_live_vehicle(vehicle, LaneDecision.KEEP).numpy()

    np.testing.assert_allclose(topology_polyline, live_polyline)


def test_compute_target_point_uses_expert_longitudinal_result_for_slow_front_vehicle(monkeypatch):
    config = build_transfuser_config(
        "small",
        target_point_prediction_horizon_s=4.0,
        target_point_front_safe_gap_m=10.0,
        target_point_max_reachable_accel_mps2=4.0,
        target_point_max_reachable_jerk_mps3=10.0,
    )
    current_poly = np.stack([np.linspace(0.0, 80.0, 81), np.zeros((81,), dtype=np.float32)], axis=1).astype(np.float32)
    monkeypatch.setattr(
        transfuser_features,
        "_build_mode_context_for_target_point",
        lambda vehicle, current_map=None: SimpleNamespace(
            current_lane_polyline=current_poly,
            left_lane_polyline=None,
            right_lane_polyline=None,
            has_left_adjacent=False,
            has_right_adjacent=False,
        ),
        raising=False,
    )
    lane = _FakeLane(length=100.0)
    vehicle = _FakeVehicle(lane, position=[0.0, 0.0], heading_theta=0.0, speed_km_h=30.0)
    front_obj = _FakeDynamicObject(lane, position=[20.0, 0.0], speed_km_h=10.0, acceleration=0.0)
    monkeypatch.setattr(
        transfuser_features,
        "_find_front_back_objects_for_target_lane",
        lambda vehicle, lane_decision, mode_context=None: _FakeFrontBackObjects(front_obj=front_obj, front_dist=20.0),
        raising=False,
    )

    target_point = compute_target_point(vehicle, config).numpy()

    assert float(target_point[0]) < 15.0
    np.testing.assert_allclose(target_point[1], 0.0, atol=1e-5)


def test_compute_target_point_on_curve_stays_near_lane_centerline(monkeypatch):
    config = build_transfuser_config(
        "small",
        target_point_prediction_horizon_s=4.0,
        target_point_max_reachable_accel_mps2=4.0,
        target_point_max_reachable_jerk_mps3=10.0,
    )
    angles = np.deg2rad(np.asarray([0.0, 20.0, 40.0, 60.0], dtype=np.float32))
    radius = 10.0
    current_poly = np.stack([radius * np.sin(angles), radius * (1.0 - np.cos(angles))], axis=1).astype(np.float32)
    monkeypatch.setattr(
        transfuser_features,
        "_build_mode_context_for_target_point",
        lambda vehicle: SimpleNamespace(
            current_lane_polyline=current_poly,
            left_lane_polyline=None,
            right_lane_polyline=None,
            has_left_adjacent=False,
            has_right_adjacent=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        transfuser_features,
        "_find_front_back_objects_for_target_lane",
        lambda vehicle, lane_decision, mode_context=None: _FakeFrontBackObjects(front_obj=None, front_dist=999.0),
        raising=False,
    )
    monkeypatch.setattr(
        transfuser_features,
        "_bounded_target_progress_distance",
        lambda vehicle, config, target_polyline, lane_decision, mode_context=None: 3.0,
        raising=False,
    )
    lane = _FakeLane(length=50.0)
    vehicle = _FakeVehicle(lane, position=[0.0, 0.0], heading_theta=0.0, speed_km_h=18.0)

    target_point = compute_target_point(vehicle, config).numpy()

    expected_s = 3.0
    expected_angle = expected_s / radius
    expected = np.asarray(
        [radius * np.sin(expected_angle), radius * (1.0 - np.cos(expected_angle))],
        dtype=np.float32,
    )
    assert np.linalg.norm(target_point - expected) < 0.25


def test_compute_target_point_prefers_live_lane_geometry_over_stale_mode_context(monkeypatch):
    config = build_transfuser_config(
        "small",
        target_point_prediction_horizon_s=4.0,
        target_point_max_reachable_accel_mps2=4.0,
        target_point_max_reachable_jerk_mps3=10.0,
    )
    wrong_poly = np.stack(
        [np.linspace(0.0, 30.0, 31), np.full((31,), 6.0, dtype=np.float32)],
        axis=1,
    ).astype(np.float32)
    monkeypatch.setattr(
        transfuser_features,
        "_build_mode_context_for_target_point",
        lambda vehicle, current_map=None: SimpleNamespace(
            current_lane_polyline=wrong_poly,
            left_lane_polyline=None,
            right_lane_polyline=None,
            has_left_adjacent=False,
            has_right_adjacent=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        transfuser_features,
        "_find_front_back_objects_for_target_lane",
        lambda vehicle, lane_decision, mode_context=None: _FakeFrontBackObjects(front_obj=None, front_dist=999.0),
        raising=False,
    )
    lane = _FakeLane(length=50.0, lane_index=("s", "e", 0))
    vehicle = _FakeVehicle(lane, position=[0.0, 0.0], heading_theta=0.0, speed_km_h=18.0)
    vehicle.engine = SimpleNamespace(current_map=_FakeMap({"s": {"e": [lane]}}))

    target_point = compute_target_point(vehicle, config).numpy()

    np.testing.assert_allclose(target_point[1], 0.0, atol=1e-5)
    assert float(target_point[0]) > 0.0


def test_compute_target_point_lane_change_prefers_live_adjacent_lane_geometry(monkeypatch):
    config = build_transfuser_config(
        "small",
        target_point_prediction_horizon_s=4.0,
        target_point_max_reachable_accel_mps2=4.0,
        target_point_max_reachable_jerk_mps3=10.0,
    )
    wrong_left_poly = np.stack(
        [np.linspace(0.0, 30.0, 31), np.full((31,), 9.0, dtype=np.float32)],
        axis=1,
    ).astype(np.float32)
    monkeypatch.setattr(
        transfuser_features,
        "_build_mode_context_for_target_point",
        lambda vehicle, current_map=None: SimpleNamespace(
            current_lane_polyline=np.stack([np.linspace(0.0, 30.0, 31), np.zeros((31,), dtype=np.float32)], axis=1),
            left_lane_polyline=wrong_left_poly,
            right_lane_polyline=None,
            has_left_adjacent=True,
            has_right_adjacent=False,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        transfuser_features,
        "_find_front_back_objects_for_target_lane",
        lambda vehicle, lane_decision, mode_context=None: _FakeFrontBackObjects(front_obj=None, front_dist=999.0),
        raising=False,
    )
    left_lane = _FakeLane(length=50.0, lateral_offset=3.5, lane_index=("s", "e", 0))
    current_lane = _FakeLane(length=50.0, lateral_offset=0.0, lane_index=("s", "e", 1))
    current_map = _FakeMap({"s": {"e": [left_lane, current_lane]}})
    vehicle = _FakeVehicle(current_lane, position=[0.0, 0.0], heading_theta=0.0, speed_km_h=18.0)
    vehicle.engine = SimpleNamespace(current_map=current_map)

    target_point = compute_target_point(vehicle, config, lane_decision=LaneDecision.CHANGE_LEFT).numpy()

    np.testing.assert_allclose(target_point[1], 3.5, atol=1e-5)
    assert float(target_point[0]) > 0.0


def test_compute_target_point_prefers_navigation_next_ref_continuation_for_keep_lane(monkeypatch):
    config = build_transfuser_config(
        "small",
        target_point_prediction_horizon_s=4.0,
        target_point_max_reachable_accel_mps2=4.0,
        target_point_max_reachable_jerk_mps3=10.0,
    )
    monkeypatch.setattr(
        transfuser_features,
        "_find_front_back_objects_for_target_lane",
        lambda vehicle, lane_decision, mode_context=None: _FakeFrontBackObjects(front_obj=None, front_dist=999.0),
        raising=False,
    )
    current_lane = _FakeLane(length=10.0, lateral_offset=0.0, lane_index=("a", "b", 0))
    correct_next_lane = _FakeLane(length=40.0, lateral_offset=0.0, lane_index=("b", "c", 0))
    wrong_topology_lane = _FakeLane(length=40.0, lateral_offset=6.0, lane_index=("b", "d", 0))
    current_map = _FakeMap({"a": {"b": [current_lane]}, "b": {"c": [correct_next_lane], "d": [wrong_topology_lane]}})
    vehicle = _FakeVehicle(
        current_lane,
        position=[8.0, 0.0],
        heading_theta=0.0,
        speed_km_h=18.0,
        next_ref_lanes=[correct_next_lane],
    )
    vehicle.engine = SimpleNamespace(current_map=current_map)

    target_point = compute_target_point(vehicle, config, lane_decision=LaneDecision.KEEP).numpy()

    np.testing.assert_allclose(target_point[1], 0.0, atol=1e-5)
    assert float(target_point[0]) >= 2.0


def test_compute_target_point_keep_lane_ignores_stale_outer_next_ref_lane(monkeypatch):
    config = build_transfuser_config(
        "small",
        target_point_prediction_horizon_s=4.0,
        target_point_max_reachable_accel_mps2=4.0,
        target_point_max_reachable_jerk_mps3=10.0,
    )
    monkeypatch.setattr(
        transfuser_features,
        "_find_front_back_objects_for_target_lane",
        lambda vehicle, lane_decision, mode_context=None: _FakeFrontBackObjects(front_obj=None, front_dist=999.0),
        raising=False,
    )
    current_lane = _FakeLane(length=10.0, lateral_offset=0.0, lane_index=("a", "b", 0))
    correct_next_lane = _FakeLane(length=40.0, lateral_offset=0.0, lane_index=("b", "c", 0))
    stale_outer_lane = _FakeLane(length=40.0, lateral_offset=3.5, lane_index=("b", "c", 1))
    current_map = _FakeMap({"a": {"b": [current_lane]}, "b": {"c": [correct_next_lane, stale_outer_lane]}})
    vehicle = _FakeVehicle(
        current_lane,
        position=[8.0, 0.0],
        heading_theta=0.0,
        speed_km_h=18.0,
        next_ref_lanes=[stale_outer_lane],
    )
    vehicle.engine = SimpleNamespace(current_map=current_map)

    target_point = compute_target_point(vehicle, config, lane_decision=LaneDecision.KEEP).numpy()

    np.testing.assert_allclose(target_point[1], 0.0, atol=1e-5)
    assert float(target_point[0]) >= 2.0


def test_build_live_target_lane_polyline_keep_lane_ignores_stale_outer_next_ref_lane():
    current_lane = _FakeLane(length=10.0, lateral_offset=0.0, lane_index=("a", "b", 0))
    correct_next_lane = _FakeLane(length=40.0, lateral_offset=0.0, lane_index=("b", "c", 0))
    stale_outer_lane = _FakeLane(length=40.0, lateral_offset=3.5, lane_index=("b", "c", 1))
    current_map = _FakeMap({"a": {"b": [current_lane]}, "b": {"c": [correct_next_lane, stale_outer_lane]}})
    vehicle = _FakeVehicle(
        current_lane,
        position=[8.0, 0.0],
        heading_theta=0.0,
        speed_km_h=18.0,
        next_ref_lanes=[stale_outer_lane],
    )
    vehicle.engine = SimpleNamespace(current_map=current_map)

    polyline = _build_live_target_lane_polyline(vehicle, LaneDecision.KEEP)

    assert polyline is not None
    assert float(polyline[min(5, polyline.shape[0] - 1), 1]) == 0.0


def test_compute_target_point_ignores_adjacent_lane_vehicle_for_no_front_progress(monkeypatch):
    config = build_transfuser_config(
        "small",
        target_point_prediction_horizon_s=4.0,
        target_point_max_reachable_accel_mps2=4.0,
        target_point_max_reachable_jerk_mps3=10.0,
    )
    lane = _FakeLane(length=100.0, lateral_offset=0.0, lane_index=("s", "e", 0))
    vehicle = _FakeVehicle(lane, position=[0.0, 0.0], heading_theta=0.0, speed_km_h=18.0)
    vehicle.lidar = _FakeLidar([
        _FakeDynamicObject(
            _FakeLane(length=100.0, lateral_offset=3.5, lane_index=("s", "e", 1)),
            position=[20.0, 3.5],
            speed_km_h=18.0,
        )
    ])
    vehicle.engine = SimpleNamespace(current_map=_FakeMap({"s": {"e": [lane]}}))
    monkeypatch.setattr(
        transfuser_features,
        "_find_front_back_objects_for_target_lane",
        lambda vehicle, lane_decision, mode_context=None: _FakeFrontBackObjects(front_obj=None, front_dist=999.0),
        raising=False,
    )

    target_point = compute_target_point(vehicle, config, lane_decision=LaneDecision.KEEP).numpy()

    assert float(target_point[0]) > 20.0
    np.testing.assert_allclose(target_point[1], 0.0, atol=1e-5)


def test_front_vehicle_positive_accel_uses_constant_speed_prediction():
    config = build_transfuser_config(
        "small",
        target_point_prediction_horizon_s=4.0,
        target_point_vehicle_prediction_use_constant_accel=True,
    )
    lane = _FakeLane(length=100.0)
    vehicle = _FakeVehicle(lane, position=[0.0, 0.0], heading_theta=0.0, speed_km_h=0.0)
    front_obj = _FakeDynamicObject(lane, position=[10.0, 0.0], speed_km_h=36.0, acceleration=2.0)

    predicted = transfuser_features._predict_world_position(
        front_obj,
        horizon_s=config.target_point_prediction_horizon_s,
        use_constant_accel=config.target_point_vehicle_prediction_use_constant_accel,
    )

    np.testing.assert_allclose(predicted, np.asarray([50.0, 0.0], dtype=np.float32), atol=1e-4)


def test_front_safe_distance_limit_uses_constant_speed_for_positive_accel():
    config = build_transfuser_config(
        "small",
        target_point_prediction_horizon_s=4.0,
        target_point_front_safe_gap_m=10.0,
        target_point_vehicle_prediction_use_constant_accel=True,
    )
    lane = _FakeLane(length=100.0)
    vehicle = _FakeVehicle(lane, position=[0.0, 0.0], heading_theta=0.0, speed_km_h=0.0)
    front_obj = _FakeDynamicObject(lane, position=[15.0, 0.0], speed_km_h=36.0, acceleration=2.0)
    target_polyline = np.stack([np.linspace(0.0, 80.0, 81), np.zeros((81,), dtype=np.float32)], axis=1).astype(np.float32)

    limit = transfuser_features._front_safe_distance_limit(vehicle, front_obj, config, target_polyline=target_polyline)

    expected = 15.0 + 40.0 - 10.0 - (0.5 * max(front_obj.LENGTH, front_obj.WIDTH))
    np.testing.assert_allclose(limit, expected, atol=1e-4)


def test_compute_target_point_from_sample_uses_gt_trajectory_endpoint():
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
        "trajectory": np.asarray(
            [
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.5, 0.2, 0.0],
                [2.0, 0.5, 0.0],
            ],
            dtype=np.float32,
        ),
    }

    target_point = compute_target_point_from_sample(sample, config).numpy()

    np.testing.assert_allclose(target_point, np.asarray([2.0, 0.5], dtype=np.float32))


def test_sample_to_features_targets_adds_target_point():
    config = build_transfuser_config("small", target_point_min_forward_distance_m=3.0)

    features, targets = sample_to_features_targets(_raw_sample(), config)

    assert tuple(features["target_point"].shape) == (2,)
    np.testing.assert_allclose(features["target_point"].numpy(), np.asarray([8.0, 1.5], dtype=np.float32))
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
