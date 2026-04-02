from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from metadrive.component.lane.straight_lane import StraightLane
from metadrive.exp_dataset.hierarchical_expert.driving_style import DrivingStyleProfile
from metadrive.exp_dataset.hierarchical_expert.lane_change_manager import (
    LaneChangeManager,
    ManeuverCommand,
    ManeuverState,
)


@dataclass
class MockVehicle:
    position: np.ndarray
    heading_theta: float
    speed: float
    lane: object

    @property
    def speed_km_h(self):
        return float(self.speed * 3.6)


class StubSafety:
    def __init__(self, acceptable=True, score=1.0, ongoing=True):
        self.acceptable = acceptable
        self.score = score
        self.ongoing = ongoing

    def is_gap_acceptable(self, direction, ego_speed, perception):
        return self.acceptable

    def gap_acceptance_score(self, direction, ego_speed, perception):
        return self.score

    def monitor_ongoing(self, direction, ego_speed, perception):
        return self.ongoing


class StubPlanner:
    def __init__(self, trajectory=None, abort_trajectory=None):
        self.trajectory = trajectory
        self.abort_trajectory = abort_trajectory

    def plan(self, **kwargs):
        return self.trajectory

    def plan_abort(self, **kwargs):
        return self.abort_trajectory


class MockFrontBackObjects:
    def left_lane_exist(self):
        return True

    def right_lane_exist(self):
        return True

    def has_front_object(self):
        return False

    def front_object(self):
        return None

    def front_min_distance(self):
        return 999.0

    def has_left_front_object(self):
        return False

    def has_right_front_object(self):
        return False

    def left_front_object(self):
        return None

    def right_front_object(self):
        return None

    def left_front_min_distance(self):
        return 999.0

    def right_front_min_distance(self):
        return 999.0


def build_lanes(width=3.7):
    left = StraightLane([0.0, width], [80.0, width], width=width)
    right = StraightLane([0.0, 0.0], [80.0, 0.0], width=width)
    left.index = ("a", "b", 0)
    right.index = ("a", "b", 1)
    left.is_previous_lane_of = lambda other: False
    right.is_previous_lane_of = lambda other: False
    return [left, right]


def test_initial_state_is_idle():
    manager = LaneChangeManager(StubSafety(), StubPlanner(), DrivingStyleProfile())
    assert manager.state == ManeuverState.IDLE


def test_idle_to_executing_when_utility_gap_and_trajectory_are_available():
    lanes = build_lanes()
    ego = MockVehicle(position=np.asarray([5.0, 0.0]), heading_theta=0.0, speed=8.0, lane=lanes[1])
    planner = StubPlanner(trajectory=lanes[0])
    manager = LaneChangeManager(
        StubSafety(acceptable=True, score=1.0),
        planner,
        DrivingStyleProfile(lane_change_threshold=0.1),
    )

    steering_target, _, _ = manager.update(
        ego=ego,
        all_objects=[],
        routing_target_lane=lanes[1],
        current_lanes=lanes,
        next_lanes=lanes,
    )

    assert manager.state == ManeuverState.EXECUTING
    assert steering_target is lanes[0]
    assert isinstance(manager.active_command, ManeuverCommand)


def test_executing_to_idle_when_target_lane_reached():
    lanes = build_lanes()
    ego = MockVehicle(position=np.asarray(lanes[0].position(10.0, 0.1)), heading_theta=0.0, speed=8.0, lane=lanes[0])
    manager = LaneChangeManager(StubSafety(), StubPlanner(trajectory=lanes[0]), DrivingStyleProfile())
    manager.state = ManeuverState.EXECUTING
    manager.active_trajectory = lanes[0]
    manager.active_command = ManeuverCommand(direction=-1, target_lane=lanes[0], source_lane=lanes[1], urgency=0.0, is_mandatory=False)
    manager.source_lane = lanes[1]

    steering_target, _, _ = manager.update(
        ego=ego,
        all_objects=[],
        routing_target_lane=lanes[1],
        current_lanes=lanes,
        next_lanes=lanes,
    )

    assert manager.state == ManeuverState.IDLE
    assert steering_target is lanes[0]
    assert manager.cooldown_timer > 0


def test_executing_to_aborting_when_monitor_flags_unsafe():
    lanes = build_lanes()
    ego = MockVehicle(position=np.asarray(lanes[1].position(10.0, 0.2)), heading_theta=0.0, speed=8.0, lane=lanes[1])
    abort_lane = lanes[1]
    manager = LaneChangeManager(StubSafety(ongoing=False), StubPlanner(trajectory=lanes[0], abort_trajectory=abort_lane), DrivingStyleProfile())
    manager.state = ManeuverState.EXECUTING
    manager.active_trajectory = lanes[0]
    manager.active_command = ManeuverCommand(direction=-1, target_lane=lanes[0], source_lane=lanes[1], urgency=0.0, is_mandatory=False)
    manager.source_lane = lanes[1]

    steering_target, _, _ = manager.update(
        ego=ego,
        all_objects=[],
        routing_target_lane=lanes[1],
        current_lanes=lanes,
        next_lanes=lanes,
    )

    assert manager.state == ManeuverState.ABORTING
    assert steering_target is abort_lane


def test_aborting_to_idle_when_back_on_source_lane():
    lanes = build_lanes()
    ego = MockVehicle(position=np.asarray(lanes[1].position(10.0, 0.1)), heading_theta=0.0, speed=8.0, lane=lanes[1])
    manager = LaneChangeManager(StubSafety(), StubPlanner(abort_trajectory=lanes[1]), DrivingStyleProfile())
    manager.state = ManeuverState.ABORTING
    manager.active_trajectory = lanes[1]
    manager.source_lane = lanes[1]
    manager.active_command = ManeuverCommand(direction=-1, target_lane=lanes[0], source_lane=lanes[1], urgency=0.0, is_mandatory=False)

    steering_target, _, _ = manager.update(
        ego=ego,
        all_objects=[],
        routing_target_lane=lanes[1],
        current_lanes=lanes,
        next_lanes=lanes,
    )

    assert manager.state == ManeuverState.IDLE
    assert steering_target is lanes[1]


def test_cooldown_blocks_immediate_new_lane_change():
    lanes = build_lanes()
    ego = MockVehicle(position=np.asarray(lanes[1].position(10.0, 0.0)), heading_theta=0.0, speed=8.0, lane=lanes[1])
    manager = LaneChangeManager(StubSafety(acceptable=True), StubPlanner(trajectory=lanes[0]), DrivingStyleProfile())
    manager.cooldown_timer = 5

    steering_target, _, _ = manager.update(
        ego=ego,
        all_objects=[],
        routing_target_lane=lanes[1],
        current_lanes=lanes,
        next_lanes=lanes,
    )

    assert manager.state == ManeuverState.IDLE
    assert steering_target is lanes[1]
    assert manager.cooldown_timer == 4


def test_mandatory_lane_change_skips_utility_threshold():
    lanes = build_lanes()
    lanes[0].is_previous_lane_of = lambda other: True
    ego = MockVehicle(position=np.asarray(lanes[1].position(195.0, 0.0)), heading_theta=0.0, speed=8.0, lane=lanes[1])
    planner = StubPlanner(trajectory=lanes[0])
    style = DrivingStyleProfile(lane_change_threshold=99.0)
    manager = LaneChangeManager(StubSafety(acceptable=True, score=-1.0), planner, style)

    steering_target, _, _ = manager.update(
        ego=ego,
        all_objects=[],
        routing_target_lane=lanes[1],
        current_lanes=lanes,
        next_lanes=[lanes[0]],
    )

    assert manager.state == ManeuverState.EXECUTING
    assert manager.active_command.is_mandatory is True
    assert steering_target is lanes[0]


def test_compute_route_urgency_returns_zero_without_lane_drop():
    lanes = build_lanes()
    ego = MockVehicle(position=np.asarray(lanes[1].position(20.0, 0.0)), heading_theta=0.0, speed=8.0, lane=lanes[1])
    manager = LaneChangeManager(StubSafety(), StubPlanner(), DrivingStyleProfile())

    urgency, mandatory, direction = manager._compute_route_urgency(lanes[1], lanes, lanes, ego)

    assert urgency == 0.0
    assert mandatory is False
    assert direction == 0
