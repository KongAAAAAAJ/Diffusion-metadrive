from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

import expert_dataset.hierarchical_expert as hierarchical_expert
from metadrive.engine.base_engine import BaseEngine
from metadrive.component.lane.straight_lane import StraightLane
from expert_dataset.expert_idm_policy import ExpertIDMPolicy, MockVehicle
from expert_dataset.hierarchical_expert import IntersectionConflictRegulator
from expert_dataset.hierarchical_expert.driving_style import STYLE_PRESETS
from expert_dataset.hierarchical_expert.hierarchical_policy import HierarchicalExpertIDMPolicy
from expert_dataset.hierarchical_expert.lane_change_manager import ManeuverCommand, ManeuverState
from expert_dataset.hierarchical_expert.rear_end_guard import RearEndGuardRegulator
from expert_dataset.hierarchical_expert.roundabout_regulator import RoundaboutRegulator


class MockLidar:
    def __init__(self, objects):
        self.objects = objects

    def get_surrounding_objects(self, ego):
        return [obj for obj in self.objects if obj is not ego]


class MockRoadNetwork:
    def has_connection(self, current, nxt):
        return False


class MockMap:
    def __init__(self):
        self.road_network = MockRoadNetwork()


class MockNavigation:
    def __init__(self, lanes):
        self.current_ref_lanes = lanes
        self.next_ref_lanes = lanes
        self.map = MockMap()


def build_policy(style_key="normal"):
    BaseEngine.singleton = SimpleNamespace(global_config={"show_policy_mark": False})
    lane = StraightLane([0.0, 0.0], [80.0, 0.0], width=3.7)
    lane.index = ("a", "b", 0)
    lane.is_previous_lane_of = lambda other: False
    vehicle = MockVehicle(position=np.asarray(lane.position(5.0, 0.0)), heading_theta=0.0, speed=8.0)
    vehicle.lane = lane
    vehicle.navigation = MockNavigation([lane])
    vehicle.heading = np.asarray([1.0, 0.0], dtype=np.float64)
    vehicle.velocity_km_h = np.asarray([vehicle.speed_km_h, 0.0], dtype=np.float64)
    vehicle.last_current_action = [None, [0.0, 0.0]]
    vehicle.engine = SimpleNamespace(global_config={"enable_idm_lane_change": True, "disable_idm_deceleration": False})
    vehicle.lidar = MockLidar([vehicle])
    policy = HierarchicalExpertIDMPolicy(
        vehicle,
        random_seed=0,
        style_profile=STYLE_PRESETS[style_key],
    )
    vehicle.policy = policy
    return vehicle, policy, lane


def test_hierarchical_policy_is_expert_subclass():
    _, policy, _ = build_policy()
    assert isinstance(policy, ExpertIDMPolicy)


def test_act_returns_finite_action_and_action_info():
    _, policy, _ = build_policy()

    action = policy.act()

    assert len(action) == 2
    assert np.isfinite(action[0])
    assert np.isfinite(action[1])
    assert "maneuver_state" in policy.action_info
    assert "style_aggression" in policy.action_info
    assert "intersection_conflict_active" in policy.action_info
    assert "intersection_conflict_count" in policy.action_info
    assert "intersection_conflict_acc" in policy.action_info
    assert "roundabout_phase" in policy.action_info
    assert "roundabout_active" in policy.action_info
    assert "roundabout_acc" in policy.action_info


def test_policy_returns_low_level_action_shape():
    _, policy, _ = build_policy()

    action = policy.act()

    assert len(action) == 2
    assert np.isfinite(action[0])
    assert np.isfinite(action[1])


def test_policy_initializes_intersection_regulator():
    _, policy, _ = build_policy()

    assert isinstance(policy.intersection_regulator, IntersectionConflictRegulator)


def test_package_exports_rear_end_guard_regulator():
    assert hierarchical_expert.RearEndGuardRegulator is RearEndGuardRegulator


def test_policy_initializes_rear_end_guard():
    _, policy, _ = build_policy()

    assert isinstance(policy.rear_end_guard, RearEndGuardRegulator)


def test_package_exports_roundabout_regulator():
    assert hierarchical_expert.RoundaboutRegulator is RoundaboutRegulator


def test_policy_initializes_roundabout_regulator():
    _, policy, _ = build_policy()

    assert isinstance(policy.roundabout_regulator, RoundaboutRegulator)


def test_reset_clears_manager_runtime_state():
    _, policy, lane = build_policy()
    policy.manager.state = ManeuverState.EXECUTING
    policy.manager.active_trajectory = lane
    policy.manager.active_command = ManeuverCommand(-1, lane, lane, 0.0, False)
    policy.manager.source_lane = lane
    policy.manager.cooldown_timer = 10

    policy.reset()

    assert policy.manager.state == ManeuverState.IDLE
    assert policy.manager.active_trajectory is None
    assert policy.manager.active_command is None
    assert policy.manager.source_lane is None
    assert policy.manager.cooldown_timer == 0


def test_different_styles_apply_different_idm_parameters():
    _, conservative, _ = build_policy("conservative")
    _, aggressive, _ = build_policy("aggressive")

    assert conservative.DISTANCE_WANTED != aggressive.DISTANCE_WANTED
    assert conservative.TIME_WANTED != aggressive.TIME_WANTED


def test_act_uses_trajectory_tracker_when_executing():
    _, policy, lane = build_policy()
    policy.manager.update = Mock(return_value=(lane, None, 999.0))
    policy.manager.state = ManeuverState.EXECUTING
    policy.trajectory_tracker = Mock()
    policy.trajectory_tracker.compute_steering = Mock(return_value=0.12)
    policy.steering_control = Mock(return_value=0.0)

    action = policy.act()

    assert action[0] == 0.12
    policy.trajectory_tracker.compute_steering.assert_called_once()
    policy.steering_control.assert_not_called()


def test_act_uses_lane_follow_controller_when_idle():
    _, policy, lane = build_policy()
    policy.manager.update = Mock(return_value=(lane, None, 999.0))
    policy.manager.state = ManeuverState.IDLE
    policy.trajectory_tracker = Mock()
    policy.trajectory_tracker.compute_steering = Mock(return_value=0.12)
    policy.steering_control = Mock(return_value=0.03)

    action = policy.act()

    assert action[0] == 0.03
    policy.steering_control.assert_called_once_with(lane)
    policy.trajectory_tracker.compute_steering.assert_not_called()


def test_pid_controllers_reset_when_returning_from_trajectory_tracking_to_idle():
    _, policy, lane = build_policy()
    policy.manager.update = Mock(return_value=(lane, None, 999.0))
    policy.manager.state = ManeuverState.EXECUTING
    policy.trajectory_tracker = Mock()
    policy.trajectory_tracker.compute_steering = Mock(return_value=0.08)
    policy.heading_pid.p_error = 1.0
    policy.heading_pid.i_error = 1.0
    policy.lateral_pid.p_error = 1.0
    policy.lateral_pid.i_error = 1.0

    policy.act()
    policy.manager.state = ManeuverState.IDLE
    policy.steering_control = Mock(return_value=0.0)
    policy.act()

    assert policy.heading_pid.p_error == 0
    assert policy.heading_pid.i_error == 0
    assert policy.lateral_pid.p_error == 0
    assert policy.lateral_pid.i_error == 0


def test_policy_reset_clears_tracker_state():
    _, policy, _ = build_policy()
    policy.trajectory_tracker._last_steering = 0.12

    policy.reset()

    assert policy.trajectory_tracker._last_steering is None


def test_lane_change_does_not_reduce_target_speed_during_maneuvers():
    _, policy, lane = build_policy()
    policy.manager.update = Mock(return_value=(lane, None, 999.0))
    policy.manager.state = ManeuverState.EXECUTING
    policy.trajectory_tracker.compute_steering = Mock(return_value=0.01)
    original_target_speed = policy.target_speed

    captured = {}

    def fake_acceleration(front_obj, dist):
        captured["target_speed"] = policy.target_speed
        return 0.2

    policy.acceleration = Mock(side_effect=fake_acceleration)

    action = policy.act()

    assert action[1] == 0.2
    assert captured["target_speed"] == original_target_speed
    assert policy.target_speed == original_target_speed


def test_idle_state_keeps_original_target_speed_during_acceleration():
    _, policy, lane = build_policy()
    policy.manager.update = Mock(return_value=(lane, None, 999.0))
    policy.manager.state = ManeuverState.IDLE
    original_target_speed = policy.target_speed

    captured = {}

    def fake_acceleration(front_obj, dist):
        captured["target_speed"] = policy.target_speed
        return 0.1

    policy.acceleration = Mock(side_effect=fake_acceleration)
    policy.steering_control = Mock(return_value=0.0)

    policy.act()

    assert captured["target_speed"] == original_target_speed
    assert policy.target_speed == original_target_speed


def test_regulator_is_called_after_idm_acceleration():
    _, policy, lane = build_policy()
    policy.manager.update = Mock(return_value=(lane, None, 999.0))
    policy.manager.state = ManeuverState.IDLE
    policy.steering_control = Mock(return_value=0.0)

    call_order = []

    def fake_acceleration(front_obj, dist):
        call_order.append(("idm", policy.target_speed))
        return 0.35

    def fake_adjustment(**kwargs):
        call_order.append(("regulator", kwargs["idm_acc"]))
        return -0.25

    policy.acceleration = Mock(side_effect=fake_acceleration)
    policy.intersection_regulator.adjust_acceleration = Mock(side_effect=fake_adjustment)

    action = policy.act()

    assert action == [0.0, -0.25]
    assert call_order == [("idm", policy.target_speed), ("regulator", 0.35)]
    assert policy.action_info["intersection_conflict_acc"] == -0.25


def test_rear_end_guard_runs_after_idm_and_before_intersection_regulator():
    _, policy, lane = build_policy()
    policy.manager.update = Mock(return_value=(lane, SimpleNamespace(speed=4.0), 12.0))
    policy.manager.state = ManeuverState.IDLE
    policy.steering_control = Mock(return_value=0.0)

    call_order = []

    def fake_acceleration(front_obj, dist):
        call_order.append(("idm", dist))
        return 0.35

    def fake_rear_end_adjustment(**kwargs):
        call_order.append(("rear_end_guard", kwargs["idm_acc"]))
        return -1.8

    def fake_roundabout_adjustment(**kwargs):
        call_order.append(("roundabout", kwargs["idm_acc"]))
        return -2.0

    def fake_intersection_adjustment(**kwargs):
        call_order.append(("intersection", kwargs["idm_acc"]))
        return -2.5

    policy.acceleration = Mock(side_effect=fake_acceleration)
    policy.rear_end_guard = Mock()
    policy.rear_end_guard.adjust_acceleration = Mock(side_effect=fake_rear_end_adjustment)
    policy.rear_end_guard.last_diagnostics = {"active": True, "acc": -1.8, "ttc": 1.2, "gap": 12.0}
    policy.roundabout_regulator = Mock()
    policy.roundabout_regulator.adjust_acceleration = Mock(side_effect=fake_roundabout_adjustment)
    policy.roundabout_regulator.last_diagnostics = {"phase": "APPROACHING", "active": True, "acc": -2.0}
    policy.intersection_regulator.adjust_acceleration = Mock(side_effect=fake_intersection_adjustment)

    action = policy.act()

    assert action == [0.0, -2.5]
    assert call_order == [("idm", 12.0), ("rear_end_guard", 0.35), ("roundabout", -1.8), ("intersection", -2.0)]


def test_action_info_contains_rear_end_guard_diagnostics():
    _, policy, lane = build_policy()
    policy.manager.update = Mock(return_value=(lane, SimpleNamespace(speed=4.0), 10.0))
    policy.manager.state = ManeuverState.IDLE
    policy.steering_control = Mock(return_value=0.0)
    policy.acceleration = Mock(return_value=0.3)
    policy.rear_end_guard = Mock()
    policy.rear_end_guard.adjust_acceleration = Mock(return_value=-1.8)
    policy.rear_end_guard.last_diagnostics = {"active": True, "acc": -1.8, "ttc": 1.5, "gap": 10.0}
    policy.intersection_regulator.adjust_acceleration = Mock(return_value=-2.0)

    policy.act()

    assert policy.action_info["rear_end_guard_active"] is True
    assert policy.action_info["rear_end_guard_acc"] == -1.8
    assert policy.action_info["rear_end_guard_ttc"] == 1.5
    assert policy.action_info["rear_end_guard_gap"] == 10.0


def test_action_info_contains_roundabout_diagnostics():
    _, policy, lane = build_policy()
    policy.manager.update = Mock(return_value=(lane, SimpleNamespace(speed=4.0), 10.0))
    policy.manager.state = ManeuverState.IDLE
    policy.steering_control = Mock(return_value=0.0)
    policy.acceleration = Mock(return_value=0.3)
    policy.rear_end_guard = Mock()
    policy.rear_end_guard.adjust_acceleration = Mock(return_value=0.3)
    policy.rear_end_guard.last_diagnostics = {"active": False, "acc": 0.3, "ttc": None, "gap": 10.0}
    policy.roundabout_regulator = Mock()
    policy.roundabout_regulator.adjust_acceleration = Mock(return_value=-0.8)
    policy.roundabout_regulator.last_diagnostics = {"phase": "APPROACHING", "active": True, "acc": -0.8}
    policy.intersection_regulator.adjust_acceleration = Mock(return_value=-1.2)

    policy.act()

    assert policy.action_info["roundabout_phase"] == "APPROACHING"
    assert policy.action_info["roundabout_active"] is True
    assert policy.action_info["roundabout_acc"] == -0.8


def test_no_front_object_path_exposes_inactive_rear_end_guard_diagnostics():
    _, policy, lane = build_policy()
    policy.manager.update = Mock(return_value=(lane, None, 7.5))
    policy.manager.state = ManeuverState.IDLE
    policy.steering_control = Mock(return_value=0.0)
    policy.acceleration = Mock(return_value=0.2)
    policy.rear_end_guard = Mock()
    policy.rear_end_guard.adjust_acceleration = Mock(return_value=0.2)
    policy.rear_end_guard.last_diagnostics = {"active": False, "acc": 0.2, "ttc": None, "gap": 7.5}
    policy.intersection_regulator.adjust_acceleration = Mock(return_value=0.2)

    policy.act()

    assert policy.action_info["rear_end_guard_active"] is False
    assert policy.action_info["rear_end_guard_ttc"] is None
    assert policy.action_info["rear_end_guard_gap"] == 7.5


def test_reset_clears_rear_end_guard_state():
    _, policy, _ = build_policy()
    policy.rear_end_guard = RearEndGuardRegulator()
    policy.rear_end_guard._last_accel = -1.2

    policy.reset()

    assert policy.rear_end_guard._last_accel is None


def test_reset_clears_roundabout_regulator_state():
    _, policy, _ = build_policy()
    policy.roundabout_regulator = RoundaboutRegulator()
    policy.roundabout_regulator._last_accel = -0.9

    policy.reset()

    assert policy.roundabout_regulator._last_accel is None


def test_lane_change_keeps_target_speed_before_regulator():
    _, policy, lane = build_policy()
    policy.manager.update = Mock(return_value=(lane, None, 999.0))
    policy.manager.state = ManeuverState.EXECUTING
    policy.trajectory_tracker.compute_steering = Mock(return_value=0.0)

    original_target_speed = policy.target_speed
    captured = {}

    def fake_acceleration(front_obj, dist):
        captured["target_speed_during_idm"] = policy.target_speed
        return 0.3

    def fake_adjustment(**kwargs):
        captured["idm_acc"] = kwargs["idm_acc"]
        captured["state"] = kwargs["maneuver_state"]
        return 0.1

    policy.acceleration = Mock(side_effect=fake_acceleration)
    policy.intersection_regulator.adjust_acceleration = Mock(side_effect=fake_adjustment)

    action = policy.act()

    assert captured["target_speed_during_idm"] == original_target_speed
    assert captured["idm_acc"] == 0.3
    assert captured["state"] == ManeuverState.EXECUTING
    assert action[1] == 0.1
    assert policy.target_speed == original_target_speed
