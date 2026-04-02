from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np


def _load_module():
    point_lane_module = ModuleType("metadrive.component.lane.point_lane")
    point_lane_module.PointLane = type("PointLane", (), {})
    straight_lane_module = ModuleType("metadrive.component.lane.straight_lane")
    straight_lane_module.StraightLane = type("StraightLane", (), {})
    create_block_module = ModuleType("metadrive.component.pgblock.create_pg_block_utils")
    create_block_module.create_bend_straight = lambda *args, **kwargs: (None, None)

    pid_module = ModuleType("metadrive.component.vehicle.PID_controller")

    class PIDController:
        def __init__(self, k_p, k_i, k_d):
            self.k_p = k_p
            self.k_i = k_i
            self.k_d = k_d

    pid_module.PIDController = PIDController

    idm_module = ModuleType("metadrive.policy.idm_policy")

    class IDMPolicy:
        MAX_STEERING_ANGLE = 1.0

        def __init__(self, control_object, random_seed):
            self.control_object = control_object
            self.random_seed = random_seed
            self.engine = SimpleNamespace(global_config={})
            self.heading_pid = PIDController(1.7, 0.01, 3.5)
            self.lateral_pid = PIDController(0.3, 0.002, 0.05)
            self.enable_lane_change = self.engine.global_config.get("enable_idm_lane_change", True)
            self.action_info = {}
            self.routing_target_lane = "target-lane"

        def steering_control(self, target_lane):
            return ("idm", target_lane)

        def act(self, *args, **kwargs):
            action = [0.25, 0.5]
            self.action_info["action"] = action
            return action

        def reset(self):
            self.action_info.clear()

    idm_module.IDMPolicy = IDMPolicy

    math_module = ModuleType("metadrive.utils.math")
    math_module.wrap_to_pi = lambda value: value

    sys.modules["metadrive.component.lane.point_lane"] = point_lane_module
    sys.modules["metadrive.component.lane.straight_lane"] = straight_lane_module
    sys.modules["metadrive.component.pgblock.create_pg_block_utils"] = create_block_module
    sys.modules["metadrive.component.vehicle.PID_controller"] = pid_module
    sys.modules["metadrive.policy.idm_policy"] = idm_module
    sys.modules["metadrive.utils.math"] = math_module

    module_path = Path(__file__).resolve().parents[1] / "expert_idm_policy.py"
    spec = importlib.util.spec_from_file_location("expert_idm_policy_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, PIDController, IDMPolicy


expert_idm_policy, PIDController, IDMPolicy = _load_module()


def test_expert_idm_policy_inherits_idm_steering_control():
    assert expert_idm_policy.ExpertIDMPolicy.steering_control is IDMPolicy.steering_control


def test_build_standalone_policy_uses_idm_pid_gains():
    vehicle = expert_idm_policy.MockVehicle(position=[0.0, 0.0], heading_theta=0.0, speed=0.0)

    policy = expert_idm_policy._build_standalone_policy(vehicle)

    expected_heading = PIDController(1.7, 0.01, 3.5)
    expected_lateral = PIDController(0.3, 0.002, 0.05)
    assert policy.heading_pid.k_p == expected_heading.k_p
    assert policy.heading_pid.k_i == expected_heading.k_i
    assert policy.heading_pid.k_d == expected_heading.k_d
    assert policy.lateral_pid.k_p == expected_lateral.k_p
    assert policy.lateral_pid.k_i == expected_lateral.k_i
    assert policy.lateral_pid.k_d == expected_lateral.k_d
    assert policy.enable_lane_change is True


def test_expert_idm_policy_keeps_parent_lane_change_setting():
    vehicle = object()

    policy = expert_idm_policy.ExpertIDMPolicy(control_object=vehicle, random_seed=0)

    assert policy.enable_lane_change is True


def test_expert_idm_policy_initializes_intersection_regulator():
    vehicle = SimpleNamespace(lidar=SimpleNamespace(get_surrounding_objects=lambda _: []))

    policy = expert_idm_policy.ExpertIDMPolicy(control_object=vehicle, random_seed=0)

    assert isinstance(policy.intersection_regulator, expert_idm_policy.IntersectionSpeedRegulator)


def test_expert_idm_policy_act_applies_intersection_adjustment():
    surrounding = [object()]
    vehicle = SimpleNamespace(lidar=SimpleNamespace(get_surrounding_objects=lambda _: surrounding))
    policy = expert_idm_policy.ExpertIDMPolicy(control_object=vehicle, random_seed=0)
    policy.routing_target_lane = "ref-lane"
    policy.intersection_regulator.adjust = lambda ego, all_objects, target_lane, idm_acc: -1.25

    action = policy.act()

    assert action == [0.25, -1.25]
    assert policy.action_info["action"] == [0.25, -1.25]


def test_expert_idm_policy_reset_clears_intersection_regulator_state():
    vehicle = SimpleNamespace(lidar=SimpleNamespace(get_surrounding_objects=lambda _: []))
    policy = expert_idm_policy.ExpertIDMPolicy(control_object=vehicle, random_seed=0)
    policy.intersection_regulator._last_accel = -2.0
    policy.action_info["action"] = [0.1, -0.2]

    policy.reset()

    assert policy.intersection_regulator._last_accel is None
    assert policy.action_info == {}


def test_intersection_regulator_returns_idm_acc_without_conflict():
    regulator = expert_idm_policy.IntersectionSpeedRegulator()
    ego = SimpleNamespace(
        position=np.asarray([0.0, 0.0], dtype=np.float64),
        speed=8.0,
        heading_theta=0.0,
    )

    adjusted = regulator.adjust(ego=ego, all_objects=[], target_lane=None, idm_acc=0.4)

    assert adjusted == 0.4
    assert regulator._last_accel is None


def test_intersection_regulator_detects_crossing_conflict_and_brakes():
    regulator = expert_idm_policy.IntersectionSpeedRegulator()
    ego = SimpleNamespace(
        position=np.asarray([0.0, 0.0], dtype=np.float64),
        speed=8.0,
        heading_theta=0.0,
    )
    crossing = SimpleNamespace(
        position=np.asarray([12.0, -12.0], dtype=np.float64),
        speed=8.0,
        heading_theta=np.pi / 2,
    )
    lane = SimpleNamespace(
        length=60.0,
        local_coordinates=lambda position: (float(position[0]), float(position[1])),
        position=lambda longitudinal, lateral: np.asarray([longitudinal, lateral], dtype=np.float64),
    )
    crossing_lane = SimpleNamespace(
        length=60.0,
        local_coordinates=lambda position: (float(position[1] + 20.0), float(position[0] - 10.0)),
        position=lambda longitudinal, lateral: np.asarray([10.0 + lateral, -20.0 + longitudinal], dtype=np.float64),
    )
    crossing.navigation = SimpleNamespace(current_ref_lanes=[crossing_lane])
    crossing.lane_index = ("c", "d", 0)

    adjusted = regulator.adjust(ego=ego, all_objects=[crossing], target_lane=lane, idm_acc=0.4)

    assert adjusted < 0.0
    assert adjusted < 0.4
