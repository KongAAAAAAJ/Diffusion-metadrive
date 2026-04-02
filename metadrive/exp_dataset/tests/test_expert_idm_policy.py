from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


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

        def steering_control(self, target_lane):
            return ("idm", target_lane)

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
