from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from metadrive.component.lane.straight_lane import StraightLane
from metadrive.exp_dataset.hierarchical_expert.roundabout_regulator import RoundaboutPhase, RoundaboutRegulator


class CircularStubLane:
    def __init__(self, radius: float = 12.0, center=(0.0, 0.0), start_angle: float = -np.pi / 2):
        self.radius = float(radius)
        self.center = np.asarray(center, dtype=np.float64)
        self.start_angle = float(start_angle)
        self.length = 2 * np.pi * self.radius

    def position(self, longitudinal, lateral):
        theta = self.start_angle + float(longitudinal) / self.radius
        offset_radius = self.radius + float(lateral)
        point = self.center + np.asarray([offset_radius * np.cos(theta), offset_radius * np.sin(theta)], dtype=np.float64)
        return point

    def local_coordinates(self, point):
        point = np.asarray(point, dtype=np.float64)
        rel = point - self.center
        theta = np.arctan2(rel[1], rel[0])
        delta = theta - self.start_angle
        while delta < 0.0:
            delta += 2 * np.pi
        longitudinal = delta * self.radius
        lateral = np.linalg.norm(rel) - self.radius
        return longitudinal, lateral


class FakeRoundaboutLane:
    def __init__(self, length=50.0, origin=(0.0, 0.0), direction=(1.0, 0.0)):
        self.length = float(length)
        self.origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        norm = np.linalg.norm(direction)
        self.direction = direction / max(norm, 1e-6)
        self.direction_lateral = np.asarray([self.direction[1], -self.direction[0]], dtype=np.float64)

    def position(self, longitudinal, lateral):
        return self.origin + float(longitudinal) * self.direction + float(lateral) * self.direction_lateral

    def local_coordinates(self, point):
        point = np.asarray(point, dtype=np.float64)
        delta = point - self.origin
        longitudinal = float(delta[0] * self.direction[0] + delta[1] * self.direction[1])
        lateral = float(delta[0] * self.direction_lateral[0] + delta[1] * self.direction_lateral[1])
        return longitudinal, lateral


def build_straight_lane(start=(0.0, 0.0), end=(40.0, 0.0), width=3.7):
    lane = StraightLane(start, end, width=width)
    lane.index = ("a", "b", 0)
    lane.is_previous_lane_of = lambda other: False
    return lane


def make_actor(position, speed=8.0, heading=(1.0, 0.0), lane=None, current_ref_lanes=None, next_ref_lanes=None):
    heading = np.asarray(heading, dtype=np.float64)
    norm = np.linalg.norm(heading)
    if norm <= 1e-6:
        heading = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        heading = heading / norm
    navigation = SimpleNamespace(
        current_ref_lanes=list(current_ref_lanes or []),
        next_ref_lanes=list(next_ref_lanes or []),
    )
    actor = SimpleNamespace(
        position=np.asarray(position, dtype=np.float64),
        speed=float(speed),
        speed_km_h=float(speed) * 3.6,
        heading=heading,
        heading_theta=float(np.arctan2(heading[1], heading[0])),
        velocity_km_h=heading * float(speed) * 3.6,
        navigation=navigation,
        lane=lane,
    )
    return actor


def test_detect_phase_returns_none_on_straight_road():
    regulator = RoundaboutRegulator()
    current_lane = build_straight_lane()
    ego = make_actor(position=current_lane.position(5.0, 0.0), lane=current_lane, current_ref_lanes=[current_lane], next_ref_lanes=[current_lane])

    phase = regulator._detect_phase(ego, current_lane)

    assert phase == RoundaboutPhase.NONE


def test_is_roundabout_lane_true_for_circular_lane_name():
    regulator = RoundaboutRegulator()

    assert regulator._is_roundabout_lane(CircularStubLane()) is True


def test_is_roundabout_lane_false_for_straight_lane():
    regulator = RoundaboutRegulator()

    assert regulator._is_roundabout_lane(build_straight_lane()) is False


def test_adjust_acceleration_yields_when_approaching_and_gap_is_tight():
    regulator = RoundaboutRegulator(horizon=4.0, dt=0.2)
    approach_lane = build_straight_lane(start=(0.0, 0.0), end=(40.0, 0.0))
    roundabout_lane = FakeRoundaboutLane(length=60.0, origin=(20.0, -7.5), direction=(0.0, 1.0))
    ego = make_actor(
        position=(15.0, 0.0),
        speed=10.0,
        lane=approach_lane,
        current_ref_lanes=[approach_lane],
        next_ref_lanes=[roundabout_lane],
    )
    circulating = make_actor(
        position=(20.0, -7.5),
        speed=5.0,
        heading=(0.0, 1.0),
        lane=roundabout_lane,
    )

    adjusted = regulator.adjust_acceleration(
        ego=ego,
        all_objects=[circulating],
        reference_target=approach_lane,
        idm_acc=0.3,
    )

    assert adjusted < 0.3


def test_adjust_acceleration_keeps_idm_acc_when_approach_gap_is_clear():
    regulator = RoundaboutRegulator(horizon=6.0, dt=0.2)
    approach_lane = build_straight_lane(start=(0.0, 0.0), end=(60.0, 0.0))
    roundabout_lane = FakeRoundaboutLane(length=80.0, origin=(20.0, -50.0), direction=(0.0, 1.0))
    ego = make_actor(
        position=(15.0, 0.0),
        speed=10.0,
        lane=approach_lane,
        current_ref_lanes=[approach_lane],
        next_ref_lanes=[roundabout_lane],
    )
    circulating = make_actor(
        position=(20.0, -50.0),
        speed=5.0,
        heading=(0.0, 1.0),
        lane=roundabout_lane,
    )

    adjusted = regulator.adjust_acceleration(
        ego=ego,
        all_objects=[circulating],
        reference_target=approach_lane,
        idm_acc=0.25,
    )

    assert adjusted == 0.25


def test_in_roundabout_conflict_acc_ignores_far_straight_road_vehicle():
    regulator = RoundaboutRegulator()
    roundabout_lane = CircularStubLane()
    ego = make_actor(
        position=roundabout_lane.position(0.0, 0.0),
        speed=6.0,
        heading=(1.0, 0.0),
        lane=roundabout_lane,
    )
    far_lane = build_straight_lane(start=(100.0, 100.0), end=(160.0, 100.0))
    far_vehicle = make_actor(
        position=(120.0, 100.0),
        speed=8.0,
        heading=(1.0, 0.0),
        lane=far_lane,
    )

    adjusted = regulator._circulating_conflict_acc(
        ego=ego,
        all_objects=[far_vehicle],
        reference_target=roundabout_lane,
        idm_acc=0.2,
    )

    assert adjusted == 0.2


def test_in_roundabout_returns_idm_acc_directly():
    regulator = RoundaboutRegulator()
    roundabout_lane = CircularStubLane()
    ego = make_actor(
        position=roundabout_lane.position(5.0, 0.0),
        speed=6.0,
        heading=(0.0, 1.0),
        lane=roundabout_lane,
        current_ref_lanes=[roundabout_lane],
        next_ref_lanes=[roundabout_lane],
    )
    conflict_vehicle = make_actor(
        position=roundabout_lane.position(8.0, 0.0),
        speed=4.0,
        heading=(-1.0, 0.0),
        lane=roundabout_lane,
    )

    adjusted = regulator.adjust_acceleration(
        ego=ego,
        all_objects=[conflict_vehicle],
        reference_target=roundabout_lane,
        idm_acc=0.5,
    )

    assert adjusted == 0.5
    assert regulator.last_diagnostics["phase"] == "IN_ROUNDABOUT"
    assert regulator.last_diagnostics["active"] is False


def test_reset_clears_last_accel():
    regulator = RoundaboutRegulator()
    regulator._last_accel = -1.5

    regulator.reset()

    assert regulator._last_accel is None


def test_adjust_acceleration_returns_idm_acc_when_internal_error_occurs():
    regulator = RoundaboutRegulator()
    regulator._detect_phase = lambda ego, reference_target: (_ for _ in ()).throw(RuntimeError("boom"))
    ego = make_actor(position=(0.0, 0.0))

    adjusted = regulator.adjust_acceleration(
        ego=ego,
        all_objects=[],
        reference_target=None,
        idm_acc=0.4,
    )

    assert adjusted == 0.4


def test_in_roundabout_conflict_acc_ignores_exit_lane_vehicle():
    regulator = RoundaboutRegulator(conflict_distance=6.0)
    roundabout_lane = CircularStubLane()
    exit_lane = build_straight_lane(start=(12.0, 0.0), end=(32.0, 0.0))
    ego = make_actor(
        position=roundabout_lane.position(0.0, 0.0),
        speed=6.0,
        heading=(1.0, 0.0),
        lane=roundabout_lane,
    )
    exit_vehicle = make_actor(
        position=(12.0, 0.0),
        speed=5.0,
        heading=(1.0, 0.0),
        lane=exit_lane,
    )

    adjusted = regulator._circulating_conflict_acc(
        ego=ego,
        all_objects=[exit_vehicle],
        reference_target=roundabout_lane,
        idm_acc=0.2,
    )

    assert adjusted == 0.2


def test_predict_path_along_lane_extrapolates_past_lane_end_instead_of_piling_up():
    regulator = RoundaboutRegulator(horizon=1.0, dt=0.25)
    short_lane = build_straight_lane(start=(0.0, 0.0), end=(1.0, 0.0))
    actor = make_actor(position=(0.8, 0.0), speed=2.0, lane=short_lane)

    path = regulator._predict_path_along_lane(actor, short_lane, speed=2.0)

    assert np.all(np.diff(path[:, 0]) > 0.0)
    assert path[-1, 0] > 1.0


def test_finalize_resets_rate_limiter_when_switching_from_in_roundabout_to_exiting():
    regulator = RoundaboutRegulator(accel_rate_limit=1.0)
    regulator._last_accel = -3.0
    regulator._last_phase = RoundaboutPhase.IN_ROUNDABOUT

    adjusted = regulator._finalize(idm_acc=1.0, accel_cmd=1.0, phase=RoundaboutPhase.EXITING)

    assert adjusted == 1.0
    assert regulator._last_phase == RoundaboutPhase.EXITING


def test_finalize_resets_immediately_when_not_intervening():
    regulator = RoundaboutRegulator(accel_rate_limit=1.0)
    regulator._last_accel = -3.0
    regulator._last_phase = RoundaboutPhase.APPROACHING

    result = regulator._finalize(idm_acc=2.0, accel_cmd=2.0, phase=RoundaboutPhase.APPROACHING)

    assert result == 2.0
    assert regulator._last_accel is None
    assert regulator.last_diagnostics["active"] is False


def test_finalize_applies_rate_limit_when_intervening():
    regulator = RoundaboutRegulator(accel_rate_limit=1.0)
    regulator._last_accel = 0.0

    result = regulator._finalize(idm_acc=2.0, accel_cmd=-3.0, phase=RoundaboutPhase.APPROACHING)

    assert result == -1.0
    assert regulator.last_diagnostics["active"] is True


def test_gap_acceptance_ignores_same_direction_vehicle():
    regulator = RoundaboutRegulator(horizon=4.0, dt=0.2)
    approach_lane = build_straight_lane(start=(0.0, 0.0), end=(40.0, 0.0))
    roundabout_lane = FakeRoundaboutLane(length=60.0, origin=(35.0, 0.0), direction=(1.0, 0.0))
    ego = make_actor(
        position=(15.0, 0.0),
        speed=10.0,
        heading=(1.0, 0.0),
        lane=approach_lane,
        current_ref_lanes=[approach_lane],
        next_ref_lanes=[roundabout_lane],
    )
    same_dir_vehicle = make_actor(
        position=(35.0, 0.0),
        speed=8.0,
        heading=(1.0, 0.0),
        lane=roundabout_lane,
    )
    ego_path = np.asarray(
        [
            [15.0, 0.0],
            [17.0, 0.0],
            [19.0, 0.0],
            [21.0, 0.0],
            [23.0, 0.0],
        ],
        dtype=np.float64,
    )
    same_dir_path = np.asarray(
        [
            [11.0, 0.0],
            [13.0, 0.0],
            [15.0, 0.0],
            [17.0, 0.0],
            [19.0, 0.0],
        ],
        dtype=np.float64,
    )

    regulator._predict_path_along_lane = lambda obj, lane, speed: ego_path if obj is ego else same_dir_path  # noqa: E731

    adjusted = regulator._gap_acceptance_acc(
        ego=ego,
        all_objects=[same_dir_vehicle],
        reference_target=approach_lane,
        idm_acc=0.3,
    )

    assert adjusted == 0.3


def test_gap_acceptance_still_yields_to_cross_traffic():
    regulator = RoundaboutRegulator(horizon=4.0, dt=0.2)
    approach_lane = build_straight_lane(start=(0.0, 0.0), end=(40.0, 0.0))
    roundabout_lane = FakeRoundaboutLane(length=60.0, origin=(20.0, -7.5), direction=(0.0, 1.0))
    ego = make_actor(
        position=(15.0, 0.0),
        speed=10.0,
        heading=(1.0, 0.0),
        lane=approach_lane,
        current_ref_lanes=[approach_lane],
        next_ref_lanes=[roundabout_lane],
    )
    cross_vehicle = make_actor(
        position=(20.0, -7.5),
        speed=5.0,
        heading=(0.0, 1.0),
        lane=roundabout_lane,
    )
    adjusted = regulator.adjust_acceleration(
        ego=ego,
        all_objects=[cross_vehicle],
        reference_target=approach_lane,
        idm_acc=0.3,
    )

    assert adjusted < 0.3
