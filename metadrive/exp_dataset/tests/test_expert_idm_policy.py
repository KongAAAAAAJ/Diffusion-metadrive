from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np


def _load_module():
    metadrive_pkg = ModuleType("metadrive")
    metadrive_pkg.__path__ = []
    component_pkg = ModuleType("metadrive.component")
    component_pkg.__path__ = []
    lane_pkg = ModuleType("metadrive.component.lane")
    lane_pkg.__path__ = []
    pgblock_pkg = ModuleType("metadrive.component.pgblock")
    pgblock_pkg.__path__ = []
    vehicle_pkg = ModuleType("metadrive.component.vehicle")
    vehicle_pkg.__path__ = []
    policy_pkg = ModuleType("metadrive.policy")
    policy_pkg.__path__ = []
    utils_pkg = ModuleType("metadrive.utils")
    utils_pkg.__path__ = []
    sys.modules["metadrive"] = metadrive_pkg
    sys.modules["metadrive.component"] = component_pkg
    sys.modules["metadrive.component.lane"] = lane_pkg
    sys.modules["metadrive.component.pgblock"] = pgblock_pkg
    sys.modules["metadrive.component.vehicle"] = vehicle_pkg
    sys.modules["metadrive.policy"] = policy_pkg
    sys.modules["metadrive.utils"] = utils_pkg

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

    class FrontBackObjects:
        @staticmethod
        def get_find_front_back_objs(objs, lane, position, max_distance):
            return SimpleNamespace(front_object=lambda: None, front_min_distance=lambda: max_distance)

    class IDMPolicy:
        MAX_STEERING_ANGLE = 1.0
        DISTANCE_WANTED = 10.0
        TIME_WANTED = 1.5
        DELTA = 10.0
        ACC_FACTOR = 1.0
        DEACC_FACTOR = -5.0
        NORMAL_SPEED = 30.0
        MAX_SPEED = 100.0
        LANE_CHANGE_FREQ = 50
        LANE_CHANGE_SPEED_INCREASE = 10.0
        SAFE_LANE_CHANGE_DISTANCE = 15.0
        MAX_LONG_DIST = 30.0

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
    idm_module.FrontBackObjects = FrontBackObjects

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


def test_expert_idm_policy_applies_instance_level_idm_config():
    vehicle = SimpleNamespace(lidar=SimpleNamespace(get_surrounding_objects=lambda _: []))
    config = expert_idm_policy.ExpertIDMConfig(
        distance_wanted=6.5,
        time_wanted=0.9,
        delta=6.0,
        acc_factor=1.8,
        deacc_factor=-3.5,
        normal_speed_kmh=42.0,
        max_speed_kmh=88.0,
        enable_lane_change=False,
        lane_change_freq=13,
        lane_change_speed_increase=4.0,
        safe_lane_change_distance=9.0,
        max_long_dist=55.0,
        heading_pid_kp=2.4,
        heading_pid_ki=0.12,
        heading_pid_kd=4.8,
        lateral_pid_kp=0.55,
        lateral_pid_ki=0.01,
        lateral_pid_kd=0.09,
    )

    policy = expert_idm_policy.ExpertIDMPolicy(control_object=vehicle, random_seed=0, idm_config=config)

    assert policy.DISTANCE_WANTED == 6.5
    assert policy.TIME_WANTED == 0.9
    assert policy.DELTA == 6.0
    assert policy.ACC_FACTOR == 1.8
    assert policy.DEACC_FACTOR == -3.5
    assert policy.NORMAL_SPEED == 42.0
    assert policy.MAX_SPEED == 88.0
    assert policy.enable_lane_change is False
    assert policy.LANE_CHANGE_FREQ == 13
    assert policy.LANE_CHANGE_SPEED_INCREASE == 4.0
    assert policy.SAFE_LANE_CHANGE_DISTANCE == 9.0
    assert policy.MAX_LONG_DIST == 55.0
    assert policy.target_speed == 42.0
    assert policy.heading_pid.k_p == 2.4
    assert policy.heading_pid.k_i == 0.12
    assert policy.heading_pid.k_d == 4.8
    assert policy.lateral_pid.k_p == 0.55
    assert policy.lateral_pid.k_i == 0.01
    assert policy.lateral_pid.k_d == 0.09


def test_expert_idm_policy_config_does_not_mutate_idm_policy_class_attributes():
    vehicle = SimpleNamespace(lidar=SimpleNamespace(get_surrounding_objects=lambda _: []))
    original_time_wanted = IDMPolicy.TIME_WANTED
    original_distance_wanted = getattr(IDMPolicy, "DISTANCE_WANTED", None)

    expert_idm_policy.ExpertIDMPolicy(
        control_object=vehicle,
        random_seed=0,
        idm_config=expert_idm_policy.ExpertIDMConfig(
            time_wanted=0.7,
            distance_wanted=5.0,
        ),
    )

    assert IDMPolicy.TIME_WANTED == original_time_wanted
    assert IDMPolicy.DISTANCE_WANTED == original_distance_wanted


def test_expert_idm_policy_instances_do_not_share_config_state():
    vehicle = SimpleNamespace(lidar=SimpleNamespace(get_surrounding_objects=lambda _: []))

    policy_a = expert_idm_policy.ExpertIDMPolicy(
        control_object=vehicle,
        random_seed=0,
        idm_config=expert_idm_policy.ExpertIDMConfig(time_wanted=0.8, enable_lane_change=False),
    )
    policy_b = expert_idm_policy.ExpertIDMPolicy(
        control_object=vehicle,
        random_seed=1,
        idm_config=expert_idm_policy.ExpertIDMConfig(time_wanted=1.9, enable_lane_change=True),
    )

    assert policy_a.TIME_WANTED == 0.8
    assert policy_b.TIME_WANTED == 1.9
    assert policy_a.enable_lane_change is False
    assert policy_b.enable_lane_change is True


def test_expert_idm_policy_instances_do_not_share_pid_state():
    vehicle = SimpleNamespace(lidar=SimpleNamespace(get_surrounding_objects=lambda _: []))

    policy_a = expert_idm_policy.ExpertIDMPolicy(
        control_object=vehicle,
        random_seed=0,
        idm_config=expert_idm_policy.ExpertIDMConfig(
            heading_pid_kp=2.1,
            heading_pid_ki=0.05,
            heading_pid_kd=4.2,
            lateral_pid_kp=0.4,
            lateral_pid_ki=0.01,
            lateral_pid_kd=0.08,
        ),
    )
    policy_b = expert_idm_policy.ExpertIDMPolicy(
        control_object=vehicle,
        random_seed=1,
        idm_config=expert_idm_policy.ExpertIDMConfig(
            heading_pid_kp=1.3,
            heading_pid_ki=0.02,
            heading_pid_kd=2.7,
            lateral_pid_kp=0.25,
            lateral_pid_ki=0.003,
            lateral_pid_kd=0.04,
        ),
    )

    assert policy_a.heading_pid.k_p == 2.1
    assert policy_a.heading_pid.k_i == 0.05
    assert policy_a.heading_pid.k_d == 4.2
    assert policy_b.heading_pid.k_p == 1.3
    assert policy_b.heading_pid.k_i == 0.02
    assert policy_b.heading_pid.k_d == 2.7
    assert policy_a.lateral_pid.k_p == 0.4
    assert policy_a.lateral_pid.k_i == 0.01
    assert policy_a.lateral_pid.k_d == 0.08
    assert policy_b.lateral_pid.k_p == 0.25
    assert policy_b.lateral_pid.k_i == 0.003
    assert policy_b.lateral_pid.k_d == 0.04


def test_expert_idm_policy_pid_config_does_not_mutate_parent_defaults():
    vehicle = SimpleNamespace(lidar=SimpleNamespace(get_surrounding_objects=lambda _: []))
    original_heading = (1.7, 0.01, 3.5)
    original_lateral = (0.3, 0.002, 0.05)

    policy = expert_idm_policy.ExpertIDMPolicy(
        control_object=vehicle,
        random_seed=0,
        idm_config=expert_idm_policy.ExpertIDMConfig(
            heading_pid_kp=3.0,
            heading_pid_ki=0.2,
            heading_pid_kd=5.0,
            lateral_pid_kp=0.6,
            lateral_pid_ki=0.02,
            lateral_pid_kd=0.1,
        ),
    )

    assert (policy.heading_pid.k_p, policy.heading_pid.k_i, policy.heading_pid.k_d) != original_heading
    assert (policy.lateral_pid.k_p, policy.lateral_pid.k_i, policy.lateral_pid.k_d) != original_lateral

    fresh_parent = IDMPolicy(control_object=vehicle, random_seed=1)
    assert (fresh_parent.heading_pid.k_p, fresh_parent.heading_pid.k_i, fresh_parent.heading_pid.k_d) == original_heading
    assert (fresh_parent.lateral_pid.k_p, fresh_parent.lateral_pid.k_i, fresh_parent.lateral_pid.k_d) == original_lateral


def test_expert_idm_policy_rejects_opposite_direction_lane_change():
    vehicle = SimpleNamespace(
        lidar=SimpleNamespace(get_surrounding_objects=lambda _: []),
        position=np.asarray([0.0, 0.0], dtype=np.float64),
        heading_theta=0.0,
        on_yellow_continuous_line=False,
        on_white_continuous_line=False,
    )
    current_lane = SimpleNamespace(
        index=("A", "B", 0),
        local_coordinates=lambda position: (0.0, 0.0),
        heading_theta_at=lambda longitudinal: 0.0,
    )
    opposite_lane = SimpleNamespace(
        index=("B", "A", 0),
        local_coordinates=lambda position: (0.0, 0.0),
        heading_theta_at=lambda longitudinal: np.pi,
    )

    policy = expert_idm_policy.ExpertIDMPolicy(control_object=vehicle, random_seed=0)
    policy.routing_target_lane = current_lane
    policy.move_to_next_road = lambda: True
    policy.lane_change_policy = lambda all_objects: (None, 12.0, opposite_lane)
    policy.acceleration = lambda front_obj, dist_to_front: 0.5
    policy.intersection_regulator.adjust = lambda ego, all_objects, target_lane, idm_acc: idm_acc

    action = policy.act()

    assert action[0] == ("idm", current_lane)
    assert policy.action_info["action"][0] == ("idm", current_lane)


def test_expert_idm_policy_keeps_legal_same_direction_lane_change():
    vehicle = SimpleNamespace(
        lidar=SimpleNamespace(get_surrounding_objects=lambda _: []),
        position=np.asarray([0.0, 0.0], dtype=np.float64),
        heading_theta=0.0,
        on_yellow_continuous_line=False,
        on_white_continuous_line=False,
    )
    current_lane = SimpleNamespace(
        index=("A", "B", 0),
        local_coordinates=lambda position: (0.0, 0.0),
        heading_theta_at=lambda longitudinal: 0.0,
    )
    right_lane = SimpleNamespace(
        index=("A", "B", 1),
        local_coordinates=lambda position: (0.0, 0.0),
        heading_theta_at=lambda longitudinal: 0.0,
    )

    policy = expert_idm_policy.ExpertIDMPolicy(control_object=vehicle, random_seed=0)
    policy.routing_target_lane = current_lane
    policy.move_to_next_road = lambda: True
    policy.lane_change_policy = lambda all_objects: (None, 12.0, right_lane)
    policy.acceleration = lambda front_obj, dist_to_front: 0.5
    policy.intersection_regulator.adjust = lambda ego, all_objects, target_lane, idm_acc: idm_acc

    action = policy.act()

    assert action[0] == ("idm", right_lane)


def test_expert_idm_policy_rejects_lane_change_while_on_yellow_continuous_line():
    vehicle = SimpleNamespace(
        lidar=SimpleNamespace(get_surrounding_objects=lambda _: []),
        position=np.asarray([0.0, 0.0], dtype=np.float64),
        heading_theta=0.0,
        on_yellow_continuous_line=True,
        on_white_continuous_line=False,
    )
    current_lane = SimpleNamespace(
        index=("A", "B", 0),
        local_coordinates=lambda position: (0.0, 0.0),
        heading_theta_at=lambda longitudinal: 0.0,
    )
    opposite_lane = SimpleNamespace(
        index=("B", "A", 0),
        local_coordinates=lambda position: (0.0, 0.0),
        heading_theta_at=lambda longitudinal: np.pi,
    )

    policy = expert_idm_policy.ExpertIDMPolicy(control_object=vehicle, random_seed=0)
    policy.routing_target_lane = current_lane
    policy.move_to_next_road = lambda: True
    policy.lane_change_policy = lambda all_objects: (None, 12.0, opposite_lane)
    policy.acceleration = lambda front_obj, dist_to_front: 0.5
    policy.intersection_regulator.adjust = lambda ego, all_objects, target_lane, idm_acc: idm_acc

    action = policy.act()

    assert action[0] == ("idm", current_lane)


def test_expert_idm_policy_act_applies_intersection_adjustment():
    surrounding = [object()]
    vehicle = SimpleNamespace(
        lidar=SimpleNamespace(get_surrounding_objects=lambda _: surrounding),
        position=np.asarray([0.0, 0.0], dtype=np.float64),
        heading_theta=0.0,
        on_yellow_continuous_line=False,
        on_white_continuous_line=False,
    )
    current_lane = SimpleNamespace(
        index=("A", "B", 0),
        local_coordinates=lambda position: (0.0, 0.0),
        heading_theta_at=lambda longitudinal: 0.0,
    )
    policy = expert_idm_policy.ExpertIDMPolicy(control_object=vehicle, random_seed=0)
    policy.routing_target_lane = current_lane
    policy.move_to_next_road = lambda: True
    policy.lane_change_policy = lambda all_objects: (None, 12.0, current_lane)
    policy.acceleration = lambda front_obj, dist_to_front: 0.5
    policy.intersection_regulator.adjust = lambda ego, all_objects, target_lane, idm_acc: -1.25

    action = policy.act()

    assert action == [("idm", current_lane), -1.25]
    assert policy.action_info["action"] == [("idm", current_lane), -1.25]


def test_expert_idm_policy_falls_back_to_front_vehicle_lookup_when_lane_change_path_fails():
    lead_vehicle = SimpleNamespace(speed_km_h=28.0)
    surrounding_objects = SimpleNamespace(
        front_object=lambda: lead_vehicle,
        front_min_distance=lambda: 18.0,
    )

    class _FrontBackObjects:
        @staticmethod
        def get_find_front_back_objs(objs, lane, position, max_distance):
            return surrounding_objects

    expert_idm_policy.FrontBackObjects = _FrontBackObjects

    vehicle = SimpleNamespace(
        lidar=SimpleNamespace(get_surrounding_objects=lambda _: [lead_vehicle]),
        position=np.asarray([0.0, 0.0], dtype=np.float64),
        heading_theta=0.0,
        on_yellow_continuous_line=False,
        on_white_continuous_line=False,
    )
    current_lane = SimpleNamespace(
        index=("A", "B", 0),
        local_coordinates=lambda position: (0.0, 0.0),
        heading_theta_at=lambda longitudinal: 0.0,
    )
    policy = expert_idm_policy.ExpertIDMPolicy(control_object=vehicle, random_seed=0)
    policy.routing_target_lane = current_lane
    policy.move_to_next_road = lambda: True
    policy.lane_change_policy = lambda all_objects: (_ for _ in ()).throw(RuntimeError("lane change failed"))
    policy.acceleration = lambda front_obj, dist_to_front: 0.25 if front_obj is lead_vehicle and dist_to_front == 18.0 else -9.0
    policy.intersection_regulator.adjust = lambda ego, all_objects, target_lane, idm_acc: idm_acc

    action = policy.act()

    assert action == [("idm", current_lane), 0.25]
    assert policy.action_info["front_object_detected"] is True
    assert policy.action_info["front_object_distance"] == 18.0
    assert policy.action_info["front_lookup_fallback_used"] is True


def test_expert_idm_policy_reset_clears_intersection_regulator_state():
    vehicle = SimpleNamespace(lidar=SimpleNamespace(get_surrounding_objects=lambda _: []))
    policy = expert_idm_policy.ExpertIDMPolicy(control_object=vehicle, random_seed=0)
    policy.intersection_regulator._last_accel = -2.0
    policy.action_info["action"] = [0.1, -0.2]

    policy.reset()

    assert policy.intersection_regulator._last_accel is None
    assert policy.action_info == {}


def test_expert_idm_policy_set_idm_config_replaces_runtime_parameters():
    vehicle = SimpleNamespace(lidar=SimpleNamespace(get_surrounding_objects=lambda _: []))
    policy = expert_idm_policy.ExpertIDMPolicy(control_object=vehicle, random_seed=0)
    new_config = expert_idm_policy.ExpertIDMConfig(
        normal_speed_kmh=48.0,
        enable_lane_change=False,
        heading_pid_kp=2.8,
        heading_pid_ki=0.2,
        heading_pid_kd=5.2,
    )

    policy.set_idm_config(new_config)

    assert policy.NORMAL_SPEED == 48.0
    assert policy.target_speed == 48.0
    assert policy.enable_lane_change is False
    assert policy.heading_pid.k_p == 2.8
    assert policy.heading_pid.k_i == 0.2
    assert policy.heading_pid.k_d == 5.2


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
