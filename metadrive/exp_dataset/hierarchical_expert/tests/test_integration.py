from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from metadrive.engine.base_engine import BaseEngine
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.exp_dataset.expert_idm_policy import MockVehicle, _simulate_step
from metadrive.exp_dataset.hierarchical_expert.driving_style import STYLE_PRESETS
from metadrive.exp_dataset.hierarchical_expert.hierarchical_policy import HierarchicalExpertIDMPolicy
from metadrive.exp_dataset.hierarchical_expert.lane_change_manager import ManeuverState


@dataclass
class MockNeighbor:
    position: np.ndarray
    speed: float
    lane: object
    heading_vector: np.ndarray | None = None

    def __post_init__(self):
        if self.heading_vector is None:
            self.heading_vector = np.asarray([1.0, 0.0], dtype=np.float64)

    @property
    def speed_km_h(self):
        return float(self.speed * 3.6)

    @property
    def heading(self):
        direction = np.asarray(self.heading_vector, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm <= 1e-6:
            return np.asarray([1.0, 0.0], dtype=np.float64)
        return direction / norm

    @property
    def velocity_km_h(self):
        return self.heading * self.speed_km_h


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
    def __init__(self, current_ref_lanes, next_ref_lanes):
        self.current_ref_lanes = current_ref_lanes
        self.next_ref_lanes = next_ref_lanes
        self.map = MockMap()


def build_vehicle(policy_cls, lane, current_lanes, next_lanes, style_profile):
    BaseEngine.singleton = SimpleNamespace(global_config={"show_policy_mark": False})
    vehicle = MockVehicle(position=np.asarray(lane.position(5.0, 0.0)), heading_theta=0.0, speed=8.0)
    vehicle.lane = lane
    vehicle.navigation = MockNavigation(current_lanes, next_lanes)
    vehicle.heading = np.asarray([1.0, 0.0], dtype=np.float64)
    vehicle.velocity_km_h = np.asarray([vehicle.speed_km_h, 0.0], dtype=np.float64)
    vehicle.lidar = MockLidar([vehicle])
    vehicle.last_current_action = [None, [0.0, 0.0]]
    vehicle.engine = SimpleNamespace(global_config={"enable_idm_lane_change": True, "disable_idm_deceleration": False})
    policy = policy_cls(vehicle, random_seed=0, style_profile=style_profile)
    vehicle.policy = policy
    return vehicle, policy


def update_vehicle_kinematics(vehicle):
    vehicle.heading = np.asarray([math.cos(vehicle.heading_theta), math.sin(vehicle.heading_theta)], dtype=np.float64)
    vehicle.velocity_km_h = vehicle.heading * vehicle.speed_km_h


def assign_lane_by_position(vehicle, lanes):
    distances = []
    for lane in lanes:
        _, lateral = lane.local_coordinates(vehicle.position)
        distances.append(abs(float(lateral)))
    vehicle.lane = lanes[int(np.argmin(distances))]


def simulate(policy, vehicle, lanes, other_objects, steps=160, inject=None):
    states = []
    for step in range(steps):
        if inject is not None:
            inject(step, other_objects)
        vehicle.lidar.objects = [vehicle] + other_objects
        action = policy.act()
        steering, _ = action
        assert np.isfinite(steering)
        _simulate_step(vehicle, steering, dt=0.05, wheel_base=2.8)
        active_trajectory = policy.manager.active_trajectory
        active_lateral_error = None
        if active_trajectory is not None:
            _, active_lateral_error = active_trajectory.local_coordinates(vehicle.position)
        assign_lane_by_position(vehicle, lanes)
        update_vehicle_kinematics(vehicle)
        states.append((policy.manager.state.name, vehicle.position.copy(), steering, active_lateral_error))
    return states


def first_completion(history):
    seen_executing = False
    for idx, (state, position, _, _) in enumerate(history):
        if state == "EXECUTING":
            seen_executing = True
        if seen_executing and state == "IDLE":
            return idx, position
    return None, None


def build_two_lane_scene():
    width = 3.7
    left = StraightLane([0.0, width], [120.0, width], width=width)
    right = StraightLane([0.0, 0.0], [120.0, 0.0], width=width)
    left.index = ("a", "b", 0)
    right.index = ("a", "b", 1)
    left.is_previous_lane_of = lambda other: False
    right.is_previous_lane_of = lambda other: False
    return [left, right]


def build_single_lane_scene():
    lane = StraightLane([0.0, 0.0], [120.0, 0.0], width=3.7)
    lane.index = ("a", "b", 0)
    lane.is_previous_lane_of = lambda other: False
    return lane


def simulate_crossing(policy, vehicle, ego_lane, other_objects, steps=40, dt=0.1):
    policy.manager.update = lambda **kwargs: (ego_lane, None, 999.0)
    policy.manager.state = ManeuverState.IDLE
    policy.steering_control = lambda target_lane: 0.0

    history = []
    for _ in range(steps):
        vehicle.lidar.objects = [vehicle] + other_objects
        steering, acc = policy.act()
        vehicle.speed = max(0.0, float(vehicle.speed + acc * dt))
        _simulate_step(vehicle, steering=steering, dt=dt, wheel_base=2.8)
        assign_lane_by_position(vehicle, [ego_lane])
        update_vehicle_kinematics(vehicle)
        for obj in other_objects:
            obj.position = obj.position + obj.heading * obj.speed * dt
        history.append(
            {
                "position": vehicle.position.copy(),
                "speed": float(vehicle.speed),
                "acc": float(acc),
                "conflict_active": bool(policy.action_info["intersection_conflict_active"]),
                "rear_end_guard_active": bool(policy.action_info["rear_end_guard_active"]),
            }
        )
    return history


def simulate_following(policy, vehicle, ego_lane, front_obj, steps=35, dt=0.1):
    def update_manager(**kwargs):
        gap = max(float(front_obj.position[0] - vehicle.position[0]), 0.0)
        return ego_lane, front_obj, gap

    policy.manager.update = update_manager
    policy.manager.state = ManeuverState.IDLE
    policy.steering_control = lambda target_lane: 0.0

    history = []
    for _ in range(steps):
        vehicle.lidar.objects = [vehicle, front_obj]
        steering, acc = policy.act()
        vehicle.speed = max(0.0, float(vehicle.speed + acc * dt))
        _simulate_step(vehicle, steering=steering, dt=dt, wheel_base=2.8)
        assign_lane_by_position(vehicle, [ego_lane])
        update_vehicle_kinematics(vehicle)
        front_obj.position = front_obj.position + front_obj.heading * front_obj.speed * dt
        history.append(
            {
                "position": vehicle.position.copy(),
                "speed": float(vehicle.speed),
                "acc": float(acc),
                "gap": float(front_obj.position[0] - vehicle.position[0]),
                "rear_end_guard_active": bool(policy.action_info["rear_end_guard_active"]),
            }
        )
    return history


def test_lane_change_completes_in_simple_two_lane_scene():
    lanes = build_two_lane_scene()
    vehicle, policy = build_vehicle(
        HierarchicalExpertIDMPolicy, lanes[1], lanes, lanes, STYLE_PRESETS["aggressive"]
    )
    slow_front = MockNeighbor(position=np.asarray(lanes[1].position(35.0, 0.0)), speed=2.0, lane=lanes[1])

    history = simulate(policy, vehicle, lanes, [slow_front], steps=180)

    completion_step, completion_position = first_completion(history)
    assert any(state == "EXECUTING" for state, _, _, _ in history)
    assert completion_step is not None
    _, completion_lateral = lanes[0].local_coordinates(completion_position)
    steering_history = np.asarray([steering for _, _, steering, _ in history], dtype=np.float64)
    maneuver_steering = np.asarray(
        [steering for state, _, steering, _ in history if state in {"EXECUTING", "ABORTING"}],
        dtype=np.float64,
    )
    active_errors = np.asarray(
        [abs(error) for state, _, _, error in history if state == "EXECUTING" and error is not None],
        dtype=np.float64,
    )
    assert abs(completion_lateral) < 0.8
    assert steering_history.size > 0
    assert maneuver_steering.size > 1
    assert np.max(np.abs(np.diff(maneuver_steering))) < 0.25
    assert active_errors.size > 0
    assert np.max(active_errors) < 1.0


def test_normal_style_still_overtakes_slow_front_vehicle():
    lanes = build_two_lane_scene()
    vehicle, policy = build_vehicle(
        HierarchicalExpertIDMPolicy, lanes[1], lanes, lanes, STYLE_PRESETS["normal"]
    )
    slow_front = MockNeighbor(position=np.asarray(lanes[1].position(35.0, 0.0)), speed=2.0, lane=lanes[1])

    history = simulate(policy, vehicle, lanes, [slow_front], steps=220)

    completion_step, completion_position = first_completion(history)
    assert any(state == "EXECUTING" for state, _, _, _ in history)
    assert completion_step is not None
    _, completion_lateral = lanes[0].local_coordinates(completion_position)
    assert abs(completion_lateral) < 0.8


def test_lane_change_can_abort_and_return_to_source_lane():
    lanes = build_two_lane_scene()
    vehicle, policy = build_vehicle(
        HierarchicalExpertIDMPolicy, lanes[1], lanes, lanes, STYLE_PRESETS["aggressive"]
    )
    slow_front = MockNeighbor(position=np.asarray(lanes[1].position(35.0, 0.0)), speed=2.0, lane=lanes[1])
    fast_rear = MockNeighbor(position=np.asarray(lanes[0].position(0.0, 0.0)), speed=25.0, lane=lanes[0])

    def inject(step, objects):
        if step == 20 and all(obj is not fast_rear for obj in objects):
            objects.append(fast_rear)

    history = simulate(policy, vehicle, lanes, [slow_front], steps=200, inject=inject)

    completion_step, completion_position = first_completion(history)
    assert any(state == "ABORTING" for state, _, _, _ in history)
    assert completion_step is not None
    _, completion_lateral = lanes[1].local_coordinates(completion_position)
    abort_steering = np.asarray(
        [steering for state, _, steering, _ in history if state == "ABORTING"],
        dtype=np.float64,
    )
    assert abs(completion_lateral) < 0.8
    assert abort_steering.size >= 1
    if abort_steering.size > 1:
        assert np.max(np.abs(np.diff(abort_steering))) < 0.25


def test_mandatory_lane_change_exposes_mandatory_command():
    lanes = build_two_lane_scene()
    lanes[0].is_previous_lane_of = lambda other: True
    vehicle, policy = build_vehicle(
        HierarchicalExpertIDMPolicy, lanes[1], lanes, [lanes[0]], STYLE_PRESETS["normal"]
    )
    vehicle.position = np.asarray(lanes[1].position(118.0, 0.0))

    simulate(policy, vehicle, lanes, [], steps=10)

    assert policy.manager.active_command is not None
    assert policy.manager.active_command.is_mandatory is True


def test_aggressive_style_finishes_in_fewer_steps_than_conservative():
    lanes = build_two_lane_scene()
    slow_front = MockNeighbor(position=np.asarray(lanes[1].position(35.0, 0.0)), speed=2.0, lane=lanes[1])

    conservative_vehicle, conservative_policy = build_vehicle(
        HierarchicalExpertIDMPolicy, lanes[1], lanes, lanes, STYLE_PRESETS["conservative"]
    )
    aggressive_vehicle, aggressive_policy = build_vehicle(
        HierarchicalExpertIDMPolicy, lanes[1], lanes, lanes, STYLE_PRESETS["aggressive"]
    )

    conservative_history = simulate(conservative_policy, conservative_vehicle, lanes, [slow_front], steps=220)
    aggressive_history = simulate(aggressive_policy, aggressive_vehicle, lanes, [slow_front], steps=220)

    def completion_step(history):
        idx, _ = first_completion(history)
        return 9999 if idx is None else idx

    assert completion_step(aggressive_history) < completion_step(conservative_history)


def test_crossing_vehicle_makes_policy_more_conservative_than_baseline():
    ego_lane = build_single_lane_scene()
    cross_lane = StraightLane([10.0, -30.0], [10.0, 30.0], width=3.7)
    cross_lane.index = ("c", "d", 0)
    cross_lane.is_previous_lane_of = lambda other: False

    vehicle, policy = build_vehicle(
        HierarchicalExpertIDMPolicy, ego_lane, [ego_lane], [ego_lane], STYLE_PRESETS["normal"]
    )
    baseline_vehicle, baseline_policy = build_vehicle(
        HierarchicalExpertIDMPolicy, ego_lane, [ego_lane], [ego_lane], STYLE_PRESETS["normal"]
    )
    baseline_policy.intersection_regulator.adjust_acceleration = lambda **kwargs: kwargs["idm_acc"]

    conflict_vehicle = MockNeighbor(
        position=np.asarray([10.0, -15.0], dtype=np.float64),
        speed=12.0,
        lane=cross_lane,
        heading_vector=np.asarray([0.0, 1.0], dtype=np.float64),
    )
    baseline_conflict_vehicle = MockNeighbor(
        position=np.asarray([10.0, -15.0], dtype=np.float64),
        speed=12.0,
        lane=cross_lane,
        heading_vector=np.asarray([0.0, 1.0], dtype=np.float64),
    )

    regulated = simulate_crossing(policy, vehicle, ego_lane, [conflict_vehicle], steps=30)
    baseline = simulate_crossing(baseline_policy, baseline_vehicle, ego_lane, [baseline_conflict_vehicle], steps=30)

    regulated_acc = np.asarray([item["acc"] for item in regulated], dtype=np.float64)
    baseline_acc = np.asarray([item["acc"] for item in baseline], dtype=np.float64)
    regulated_speed = np.asarray([item["speed"] for item in regulated], dtype=np.float64)
    baseline_speed = np.asarray([item["speed"] for item in baseline], dtype=np.float64)

    assert np.min(regulated_acc) < np.min(baseline_acc)
    assert np.min(regulated_speed) < np.min(baseline_speed)
    assert any(item["conflict_active"] for item in regulated)


def test_non_conflicting_cross_traffic_does_not_limit_longitudinal_control():
    ego_lane = build_single_lane_scene()
    cross_lane = StraightLane([60.0, -30.0], [60.0, 30.0], width=3.7)
    cross_lane.index = ("c", "d", 0)
    cross_lane.is_previous_lane_of = lambda other: False

    vehicle, policy = build_vehicle(
        HierarchicalExpertIDMPolicy, ego_lane, [ego_lane], [ego_lane], STYLE_PRESETS["normal"]
    )
    baseline_vehicle, baseline_policy = build_vehicle(
        HierarchicalExpertIDMPolicy, ego_lane, [ego_lane], [ego_lane], STYLE_PRESETS["normal"]
    )
    baseline_policy.intersection_regulator.adjust_acceleration = lambda **kwargs: kwargs["idm_acc"]

    far_cross_vehicle = MockNeighbor(
        position=np.asarray([60.0, -20.0], dtype=np.float64),
        speed=4.0,
        lane=cross_lane,
        heading_vector=np.asarray([0.0, 1.0], dtype=np.float64),
    )
    baseline_far_cross_vehicle = MockNeighbor(
        position=np.asarray([60.0, -20.0], dtype=np.float64),
        speed=4.0,
        lane=cross_lane,
        heading_vector=np.asarray([0.0, 1.0], dtype=np.float64),
    )

    regulated = simulate_crossing(policy, vehicle, ego_lane, [far_cross_vehicle], steps=25)
    baseline = simulate_crossing(baseline_policy, baseline_vehicle, ego_lane, [baseline_far_cross_vehicle], steps=25)

    regulated_acc = np.asarray([item["acc"] for item in regulated], dtype=np.float64)
    baseline_acc = np.asarray([item["acc"] for item in baseline], dtype=np.float64)

    assert np.allclose(regulated_acc, baseline_acc)
    assert not any(item["conflict_active"] for item in regulated)


def test_acceleration_recovers_smoothly_after_conflict_clears():
    ego_lane = build_single_lane_scene()
    cross_lane = StraightLane([10.0, -30.0], [10.0, 30.0], width=3.7)
    cross_lane.index = ("c", "d", 0)
    cross_lane.is_previous_lane_of = lambda other: False

    vehicle, policy = build_vehicle(
        HierarchicalExpertIDMPolicy, ego_lane, [ego_lane], [ego_lane], STYLE_PRESETS["normal"]
    )
    conflict_vehicle = MockNeighbor(
        position=np.asarray([10.0, -15.0], dtype=np.float64),
        speed=12.0,
        lane=cross_lane,
        heading_vector=np.asarray([0.0, 1.0], dtype=np.float64),
    )

    history = simulate_crossing(policy, vehicle, ego_lane, [conflict_vehicle], steps=35)
    accelerations = np.asarray([item["acc"] for item in history], dtype=np.float64)
    active_steps = np.asarray([item["conflict_active"] for item in history], dtype=bool)

    assert np.any(active_steps)
    last_active = int(np.where(active_steps)[0][-1])
    assert not np.any(active_steps[last_active + 3 :])
    assert np.max(np.abs(np.diff(accelerations[last_active:]))) <= 1.01
    assert accelerations[-1] >= np.min(accelerations)


def test_rear_end_guard_makes_policy_more_conservative_than_longitudinal_baseline():
    ego_lane = build_single_lane_scene()
    front_vehicle = MockNeighbor(position=np.asarray(ego_lane.position(18.0, 0.0)), speed=1.0, lane=ego_lane)
    baseline_front_vehicle = MockNeighbor(position=np.asarray(ego_lane.position(18.0, 0.0)), speed=1.0, lane=ego_lane)

    vehicle, policy = build_vehicle(
        HierarchicalExpertIDMPolicy, ego_lane, [ego_lane], [ego_lane], STYLE_PRESETS["normal"]
    )
    baseline_vehicle, baseline_policy = build_vehicle(
        HierarchicalExpertIDMPolicy, ego_lane, [ego_lane], [ego_lane], STYLE_PRESETS["normal"]
    )
    baseline_policy.rear_end_guard.adjust_acceleration = lambda **kwargs: kwargs["idm_acc"]
    baseline_policy.rear_end_guard.last_diagnostics = {"active": False, "acc": 0.0, "ttc": None, "gap": None}
    baseline_policy.intersection_regulator.adjust_acceleration = lambda **kwargs: kwargs["idm_acc"]
    policy.intersection_regulator.adjust_acceleration = lambda **kwargs: kwargs["idm_acc"]
    baseline_policy.acceleration = lambda front_obj, dist: 0.4
    policy.acceleration = lambda front_obj, dist: 0.4

    guarded = simulate_following(policy, vehicle, ego_lane, front_vehicle, steps=30)
    baseline = simulate_following(baseline_policy, baseline_vehicle, ego_lane, baseline_front_vehicle, steps=30)

    guarded_acc = np.asarray([item["acc"] for item in guarded], dtype=np.float64)
    baseline_acc = np.asarray([item["acc"] for item in baseline], dtype=np.float64)

    assert np.min(guarded_acc) < np.min(baseline_acc)
    assert any(item["rear_end_guard_active"] for item in guarded)


def test_rear_end_guard_does_not_weaken_existing_strong_idm_brake():
    ego_lane = build_single_lane_scene()
    vehicle, policy = build_vehicle(
        HierarchicalExpertIDMPolicy, ego_lane, [ego_lane], [ego_lane], STYLE_PRESETS["normal"]
    )
    front_vehicle = MockNeighbor(position=np.asarray(ego_lane.position(7.5, 0.0)), speed=0.0, lane=ego_lane)
    phase = {"step": 0}

    def update_manager(**kwargs):
        if phase["step"] == 0:
            phase["step"] += 1
            return ego_lane, None, 100.0
        return ego_lane, front_vehicle, 2.5

    policy.manager.update = update_manager
    policy.manager.state = ManeuverState.IDLE
    policy.steering_control = lambda target_lane: 0.0
    policy.intersection_regulator.adjust_acceleration = lambda **kwargs: kwargs["idm_acc"]
    policy.acceleration = lambda front_obj, dist: 0.5 if front_obj is None else -5.0

    first_action = policy.act()
    second_action = policy.act()

    assert first_action[1] > -5.0
    assert second_action[1] == -5.0


def test_repeated_rear_end_guard_steps_remain_rate_limited():
    ego_lane = build_single_lane_scene()
    vehicle, policy = build_vehicle(
        HierarchicalExpertIDMPolicy, ego_lane, [ego_lane], [ego_lane], STYLE_PRESETS["normal"]
    )
    front_vehicle = MockNeighbor(position=np.asarray(ego_lane.position(16.0, 0.0)), speed=0.5, lane=ego_lane)
    policy.intersection_regulator.adjust_acceleration = lambda **kwargs: kwargs["idm_acc"]

    history = simulate_following(policy, vehicle, ego_lane, front_vehicle, steps=25)
    accelerations = np.asarray([item["acc"] for item in history], dtype=np.float64)

    assert any(item["rear_end_guard_active"] for item in history)
    assert np.max(np.abs(np.diff(accelerations))) <= 1.01


def test_intersection_conflict_still_activates_without_rear_end_guard_activation():
    ego_lane = build_single_lane_scene()
    cross_lane = StraightLane([10.0, -30.0], [10.0, 30.0], width=3.7)
    cross_lane.index = ("c", "d", 0)
    cross_lane.is_previous_lane_of = lambda other: False

    vehicle, policy = build_vehicle(
        HierarchicalExpertIDMPolicy, ego_lane, [ego_lane], [ego_lane], STYLE_PRESETS["normal"]
    )
    conflict_vehicle = MockNeighbor(
        position=np.asarray([10.0, -15.0], dtype=np.float64),
        speed=12.0,
        lane=cross_lane,
        heading_vector=np.asarray([0.0, 1.0], dtype=np.float64),
    )

    history = simulate_crossing(policy, vehicle, ego_lane, [conflict_vehicle], steps=30)

    assert any(item["conflict_active"] for item in history)
    assert not any(item["rear_end_guard_active"] for item in history)
