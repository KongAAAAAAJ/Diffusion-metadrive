from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metadrive.component.lane.straight_lane import StraightLane
from metadrive.exp_dataset.hierarchical_expert import IntersectionConflictRegulator
from metadrive.exp_dataset.hierarchical_expert.lane_change_manager import ManeuverState


@dataclass
class MockActor:
    position: np.ndarray
    speed: float
    heading_vector: np.ndarray

    @property
    def heading(self):
        direction = np.asarray(self.heading_vector, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm <= 1e-6:
            return np.asarray([1.0, 0.0], dtype=np.float64)
        return direction / norm

    @property
    def heading_theta(self):
        return float(np.arctan2(self.heading[1], self.heading[0]))

    @property
    def speed_km_h(self):
        return float(self.speed * 3.6)

    @property
    def velocity_km_h(self):
        return self.heading * self.speed_km_h


def build_reference_lane():
    lane = StraightLane([0.0, 0.0], [60.0, 0.0], width=3.7)
    lane.index = ("a", "b", 0)
    lane.is_previous_lane_of = lambda other: False
    return lane


def build_ego(speed=8.0):
    return MockActor(
        position=np.asarray([0.0, 0.0], dtype=np.float64),
        speed=float(speed),
        heading_vector=np.asarray([1.0, 0.0], dtype=np.float64),
    )


def build_crossing_vehicle(x=12.0, y=-12.0, speed=8.0):
    return MockActor(
        position=np.asarray([x, y], dtype=np.float64),
        speed=float(speed),
        heading_vector=np.asarray([0.0, 1.0], dtype=np.float64),
    )


def build_vertical_lane():
    lane = StraightLane([10.0, -20.0], [10.0, 20.0], width=3.7)
    lane.index = ("c", "d", 0)
    return lane


def test_package_exports_intersection_regulator():
    assert IntersectionConflictRegulator.__name__ == "IntersectionConflictRegulator"


def test_adjust_acceleration_returns_idm_acc_when_no_candidates():
    regulator = IntersectionConflictRegulator()
    ego = build_ego()
    lane = build_reference_lane()

    adjusted = regulator.adjust_acceleration(
        ego=ego,
        all_objects=[],
        reference_target=lane,
        idm_acc=0.35,
        maneuver_state=ManeuverState.IDLE,
    )

    assert adjusted == 0.35
    assert regulator.last_diagnostics["active"] is False
    assert regulator.last_diagnostics["conflict_count"] == 0


def test_adjust_acceleration_returns_idm_acc_when_paths_do_not_conflict():
    regulator = IntersectionConflictRegulator()
    ego = build_ego()
    lane = build_reference_lane()
    distant_parallel = MockActor(
        position=np.asarray([12.0, 14.0], dtype=np.float64),
        speed=6.0,
        heading_vector=np.asarray([0.0, 1.0], dtype=np.float64),
    )

    adjusted = regulator.adjust_acceleration(
        ego=ego,
        all_objects=[distant_parallel],
        reference_target=lane,
        idm_acc=0.25,
        maneuver_state=ManeuverState.IDLE,
    )

    assert adjusted == 0.25
    assert regulator.last_diagnostics["active"] is False


def test_single_crossing_conflict_returns_more_conservative_acceleration():
    regulator = IntersectionConflictRegulator()
    ego = build_ego()
    lane = build_reference_lane()
    crossing_vehicle = build_crossing_vehicle()

    adjusted = regulator.adjust_acceleration(
        ego=ego,
        all_objects=[crossing_vehicle],
        reference_target=lane,
        idm_acc=0.4,
        maneuver_state=ManeuverState.IDLE,
    )

    assert adjusted < 0.0
    assert adjusted < 0.4
    assert regulator.last_diagnostics["active"] is True
    assert regulator.last_diagnostics["conflict_count"] == 1


def test_multiple_conflicts_choose_the_most_restrictive_acceleration():
    ego = build_ego()
    lane = build_reference_lane()
    near_conflict = build_crossing_vehicle(x=10.0, y=-10.0, speed=8.0)
    far_conflict = build_crossing_vehicle(x=16.0, y=-14.0, speed=7.0)

    single_near = IntersectionConflictRegulator().adjust_acceleration(
        ego=ego,
        all_objects=[near_conflict],
        reference_target=lane,
        idm_acc=0.2,
        maneuver_state=ManeuverState.IDLE,
    )
    single_far = IntersectionConflictRegulator().adjust_acceleration(
        ego=ego,
        all_objects=[far_conflict],
        reference_target=lane,
        idm_acc=0.2,
        maneuver_state=ManeuverState.IDLE,
    )

    combined_regulator = IntersectionConflictRegulator()
    combined = combined_regulator.adjust_acceleration(
        ego=ego,
        all_objects=[near_conflict, far_conflict],
        reference_target=lane,
        idm_acc=0.2,
        maneuver_state=ManeuverState.IDLE,
    )

    assert combined == min(single_near, single_far)
    assert combined_regulator.last_diagnostics["conflict_count"] == 2


def test_output_is_finite_and_change_limited_between_steps():
    regulator = IntersectionConflictRegulator()
    ego = build_ego(speed=10.0)
    lane = build_reference_lane()
    emergency_conflict = build_crossing_vehicle(x=6.0, y=-3.0, speed=12.0)

    first = regulator.adjust_acceleration(
        ego=ego,
        all_objects=[],
        reference_target=lane,
        idm_acc=0.5,
        maneuver_state=ManeuverState.IDLE,
    )
    second = regulator.adjust_acceleration(
        ego=ego,
        all_objects=[emergency_conflict],
        reference_target=lane,
        idm_acc=0.5,
        maneuver_state=ManeuverState.IDLE,
    )

    assert np.isfinite(first)
    assert np.isfinite(second)
    assert second >= regulator.max_braking
    assert abs(second - first) <= regulator.accel_rate_limit + 1e-6


def test_near_term_conflict_no_longer_holds_course_with_small_ttc():
    regulator = IntersectionConflictRegulator()
    ego = build_ego(speed=8.0)
    lane = build_reference_lane()
    mild_conflict = build_crossing_vehicle(x=8.0, y=-4.0, speed=7.5)

    adjusted = regulator.adjust_acceleration(
        ego=ego,
        all_objects=[mild_conflict],
        reference_target=lane,
        idm_acc=0.15,
        maneuver_state=ManeuverState.IDLE,
    )

    assert adjusted < 0.15
    assert regulator.last_diagnostics["active"] is True


def test_predict_object_path_prefers_lane_centerline_over_current_heading():
    regulator = IntersectionConflictRegulator(horizon=0.4, dt=0.2)
    lane = build_vertical_lane()
    obj = MockActor(
        position=np.asarray([10.0, -10.0], dtype=np.float64),
        speed=5.0,
        heading_vector=np.asarray([1.0, 1.0], dtype=np.float64),
    )
    obj.lane = lane

    path = regulator._predict_object_path(obj)

    np.testing.assert_allclose(path[:, 0], np.full(path.shape[0], 10.0), atol=1e-6)
    assert np.all(np.diff(path[:, 1]) > 0.0)


def test_find_most_dangerous_conflicts_returns_top_two_sorted_by_danger():
    regulator = IntersectionConflictRegulator()
    conflicts = [
        {"time_gap": 1.8, "min_distance": 1.0},
        {"time_gap": 0.6, "min_distance": 3.0},
        {"time_gap": 0.6, "min_distance": 2.0},
        {"time_gap": 1.1, "min_distance": 0.5},
    ]

    selected = regulator._find_most_dangerous_conflicts(conflicts)

    assert selected == [
        {"time_gap": 0.6, "min_distance": 2.0},
        {"time_gap": 0.6, "min_distance": 3.0},
    ]


def test_filter_candidate_vehicles_keeps_stationary_vehicle_in_conflict_zone():
    regulator = IntersectionConflictRegulator(candidate_radius=20.0)
    ego = build_ego()
    stopped_vehicle = build_crossing_vehicle(x=5.0, y=-3.0, speed=0.0)

    candidates = regulator._filter_candidate_vehicles(ego, [stopped_vehicle])

    assert candidates == [stopped_vehicle]


def test_evaluate_conflict_detects_asynchronous_arrival_conflicts():
    regulator = IntersectionConflictRegulator(dt=0.5, conflict_distance=0.75, safe_time_gap=1.5)
    ego = build_ego(speed=4.0)
    obj = build_crossing_vehicle(speed=2.0)
    ego_path = np.asarray(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [6.0, 0.0],
        ],
        dtype=np.float64,
    )
    obj_path = np.asarray(
        [
            [4.0, -3.0],
            [4.0, -2.0],
            [4.0, -1.0],
            [4.0, 0.0],
        ],
        dtype=np.float64,
    )
    regulator._predict_object_path = lambda _: obj_path

    conflict = regulator._evaluate_conflict(ego, ego_path, obj)

    assert conflict is not None
    assert conflict["time_gap"] == 1.0
    assert conflict["arrival_diff"] == 0.5
    assert conflict["s_ego"] == 4.0
    assert conflict["s_obj"] == 3.0


def test_should_hold_course_uses_speed_adaptive_ttc_check():
    regulator = IntersectionConflictRegulator()
    slow_close_conflict = {"time_gap": 0.5, "min_distance": 3.0}
    fast_far_conflict = {"time_gap": 0.5, "min_distance": 15.0}

    assert regulator._should_hold_course(build_ego(speed=5.0), slow_close_conflict) is False
    assert regulator._should_hold_course(build_ego(speed=15.0), fast_far_conflict) is True
