from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from metadrive.component.lane.point_lane import PointLane
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.pgblock.create_pg_block_utils import create_bend_straight
from metadrive.component.vehicle.PID_controller import PIDController
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.utils.math import wrap_to_pi


class IntersectionSpeedRegulator:
    """Inline intersection-conflict speed regulator for ExpertIDMPolicy."""

    HORIZON = 3.0
    DT = 0.2
    CANDIDATE_RADIUS = 40.0
    CONFLICT_DIST = 4.5
    SAFE_TIME_GAP = 1.5
    MAX_BRAKING = -4.5
    ACCEL_RATE_LIMIT = 1.0
    COMMITTED_TIME = 0.8
    COMMITTED_SPEED = 3.0
    EMERGENCY_DIST = 2.5
    SPEED_EPS = 1e-3

    def __init__(self):
        self._last_accel = None

    def reset(self):
        self._last_accel = None

    def adjust(self, ego, all_objects, target_lane, idm_acc: float) -> float:
        candidates = self._filter_candidates(ego, all_objects)
        ego_path = self._predict_path_on_lane(ego, target_lane)
        conflicts = []
        for obj in candidates:
            conflict = self._evaluate_conflict(ego, ego_path, obj)
            if conflict is not None:
                conflicts.append(conflict)

        target_acc = float(idm_acc)
        if conflicts:
            target_acc = min(self._compute_target_acc(ego, conflict, idm_acc) for conflict in conflicts)

        if target_acc >= float(idm_acc) - 1e-6:
            self._last_accel = None
            return float(idm_acc)
        return self._limit_rate(target_acc)

    def _filter_candidates(self, ego, all_objects) -> list:
        candidates = []
        ego_position = np.asarray(getattr(ego, "position", np.zeros(2)), dtype=np.float64)
        ego_dir = self._get_direction(ego)
        for obj in all_objects:
            if obj is ego or not hasattr(obj, "position"):
                continue
            position = np.asarray(obj.position, dtype=np.float64)
            if not np.all(np.isfinite(position)):
                continue
            if np.linalg.norm(position - ego_position) > self.CANDIDATE_RADIUS:
                continue
            if float(np.dot(ego_dir, self._get_direction(obj))) > 0.7:
                continue
            candidates.append(obj)
        return candidates

    def _predict_path_on_lane(self, obj, lane) -> np.ndarray:
        speed = self._get_speed(obj)
        if lane is not None and hasattr(lane, "local_coordinates") and hasattr(lane, "position"):
            try:
                long, _ = lane.local_coordinates(np.asarray(obj.position, dtype=np.float64))
                long = float(max(long, 0.0))
                lane_length = float(getattr(lane, "length", long + self.HORIZON * max(speed, 1.0)))
                samples = []
                for t in self._time_samples():
                    sample_long = float(np.clip(long + max(speed, 0.5) * t, 0.0, lane_length))
                    samples.append(np.asarray(lane.position(sample_long, 0.0), dtype=np.float64))
                return np.stack(samples, axis=0)
            except Exception:
                pass
        return self._predict_path_straight(obj)

    def _predict_path_straight(self, obj) -> np.ndarray:
        origin = np.asarray(obj.position, dtype=np.float64)
        direction = self._get_direction(obj)
        speed = max(self._get_speed(obj), 0.5)
        return np.stack([origin + direction * speed * t for t in self._time_samples()], axis=0)

    def _get_obj_lane(self, obj):
        navigation = getattr(obj, "navigation", None)
        current_ref_lanes = getattr(navigation, "current_ref_lanes", None) or []
        if current_ref_lanes:
            lane_index = getattr(obj, "lane_index", None)
            if isinstance(lane_index, (tuple, list)) and lane_index:
                candidate_idx = lane_index[-1]
                if isinstance(candidate_idx, (int, np.integer)) and 0 <= int(candidate_idx) < len(current_ref_lanes):
                    return current_ref_lanes[int(candidate_idx)]
            current_lane = getattr(obj, "lane", None)
            if current_lane in current_ref_lanes:
                return current_lane
            return current_ref_lanes[0]
        return getattr(obj, "lane", None)

    def _get_direction(self, obj) -> np.ndarray:
        heading = getattr(obj, "heading", None)
        if heading is not None:
            direction = np.asarray(heading, dtype=np.float64)
            norm = np.linalg.norm(direction)
            if norm > self.SPEED_EPS:
                return direction / norm
        velocity = getattr(obj, "velocity_km_h", None)
        if velocity is not None:
            direction = np.asarray(velocity, dtype=np.float64)
            norm = np.linalg.norm(direction)
            if norm > self.SPEED_EPS:
                return direction / norm
        heading_theta = getattr(obj, "heading_theta", None)
        if heading_theta is not None:
            return np.asarray([math.cos(float(heading_theta)), math.sin(float(heading_theta))], dtype=np.float64)
        return np.asarray([1.0, 0.0], dtype=np.float64)

    def _get_speed(self, obj) -> float:
        if hasattr(obj, "speed"):
            return max(float(obj.speed), 0.0)
        if hasattr(obj, "speed_km_h"):
            return max(float(obj.speed_km_h) / 3.6, 0.0)
        velocity = getattr(obj, "velocity_km_h", None)
        if velocity is not None:
            return max(float(np.linalg.norm(np.asarray(velocity, dtype=np.float64))) / 3.6, 0.0)
        return 0.0

    def _evaluate_conflict(self, ego, ego_path, obj) -> dict | None:
        obj_path = self._predict_path_on_lane(obj, self._get_obj_lane(obj))
        speed_ego = max(self._get_speed(ego), self.SPEED_EPS)
        speed_obj = max(self._get_speed(obj), self.SPEED_EPS)
        diff = ego_path[:, np.newaxis, :] - obj_path[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        min_distance = float(np.min(dist_matrix))
        if min_distance >= self.CONFLICT_DIST:
            return None
        ego_indices, obj_indices = np.where(dist_matrix < self.CONFLICT_DIST)
        if ego_indices.size == 0:
            return None
        ego_times = ego_indices.astype(np.float64) * self.DT
        obj_times = obj_indices.astype(np.float64) * self.DT
        arrival_diff = np.abs(ego_times - obj_times)
        valid = arrival_diff < (self.SAFE_TIME_GAP + 1.0)
        if not np.any(valid):
            return None
        valid_indices = np.flatnonzero(valid)
        order = np.lexsort((arrival_diff[valid_indices], ego_times[valid_indices]))
        best = int(valid_indices[int(order[0])])
        conflict_ego_time = float(ego_times[best])
        conflict_obj_time = float(obj_times[best])
        return {
            "time_gap": conflict_ego_time,
            "min_distance": min_distance,
            "s_ego": speed_ego * conflict_ego_time,
            "s_obj": speed_obj * conflict_obj_time,
            "obj_speed": speed_obj,
            "arrival_diff": float(arrival_diff[best]),
        }

    def _compute_target_acc(self, ego, conflict, idm_acc) -> float:
        if self._should_hold_course(ego, conflict):
            return float(idm_acc)
        if conflict["time_gap"] < self.COMMITTED_TIME and conflict["min_distance"] < self.EMERGENCY_DIST:
            return self.MAX_BRAKING
        v_obj = max(float(conflict["obj_speed"]), self.SPEED_EPS)
        t_obj = float(conflict["s_obj"] / v_obj)
        t_target = max(t_obj + self.SAFE_TIME_GAP, self.DT)
        v_ego = max(self._get_speed(ego), 0.0)
        target_acc = 2.0 * (float(conflict["s_ego"]) - v_ego * t_target) / (t_target ** 2)
        target_acc = min(float(target_acc), float(idm_acc))
        return float(np.clip(target_acc, self.MAX_BRAKING, np.inf))

    def _should_hold_course(self, ego, conflict) -> bool:
        if conflict["time_gap"] >= self.COMMITTED_TIME:
            return False
        ego_speed = max(self._get_speed(ego), self.SPEED_EPS)
        if ego_speed <= self.COMMITTED_SPEED:
            return False
        return float(conflict["min_distance"]) / ego_speed > self.COMMITTED_TIME

    def _limit_rate(self, acc) -> float:
        acc = float(acc)
        if self._last_accel is None:
            self._last_accel = acc
            return acc
        limited = float(np.clip(acc, self._last_accel - self.ACCEL_RATE_LIMIT, self._last_accel + self.ACCEL_RATE_LIMIT))
        self._last_accel = limited
        return limited

    def _time_samples(self) -> np.ndarray:
        return np.arange(0.0, self.HORIZON + self.DT * 0.5, self.DT, dtype=np.float64)


@dataclass(frozen=True)
class ExpertIDMConfig:
    # IDM parameters
    distance_wanted: float = float(getattr(IDMPolicy, "DISTANCE_WANTED", 10.0))
    time_wanted: float = float(getattr(IDMPolicy, "TIME_WANTED", 1.0))
    delta: float = float(getattr(IDMPolicy, "DELTA", 10.0))
    acc_factor: float = float(getattr(IDMPolicy, "ACC_FACTOR", 1.0))
    deacc_factor: float = float(getattr(IDMPolicy, "DEACC_FACTOR", -5.0))
    normal_speed_kmh: float = float(getattr(IDMPolicy, "NORMAL_SPEED", 30.0))
    max_speed_kmh: float = float(getattr(IDMPolicy, "MAX_SPEED", 100.0))
    enable_lane_change: bool = True
    lane_change_freq: int = int(getattr(IDMPolicy, "LANE_CHANGE_FREQ", 50))
    lane_change_speed_increase: float = float(getattr(IDMPolicy, "LANE_CHANGE_SPEED_INCREASE", 10.0))
    safe_lane_change_distance: float = float(getattr(IDMPolicy, "SAFE_LANE_CHANGE_DISTANCE", 15.0))
    max_long_dist: float = float(getattr(IDMPolicy, "MAX_LONG_DIST", 30.0))
    heading_pid_kp: float = 1.7
    heading_pid_ki: float = 0.01
    heading_pid_kd: float = 3.5
    lateral_pid_kp: float = 0.3
    lateral_pid_ki: float = 0.002
    lateral_pid_kd: float = 0.05


class ExpertIDMPolicy(IDMPolicy):
    """Dataset-facing IDM policy aligned with the background-traffic IDM."""
    MAX_STEERING = IDMPolicy.MAX_STEERING_ANGLE

    def __init__(self, control_object, random_seed: int = 0, idm_config: ExpertIDMConfig | None = None):
        super().__init__(control_object=control_object, random_seed=random_seed)
        self.idm_config = idm_config or ExpertIDMConfig(enable_lane_change=bool(self.enable_lane_change))
        self._apply_idm_config(self.idm_config)
        self.intersection_regulator = IntersectionSpeedRegulator()

    def _apply_idm_config(self, idm_config: ExpertIDMConfig) -> None:
        self.DISTANCE_WANTED = float(idm_config.distance_wanted)
        self.TIME_WANTED = float(idm_config.time_wanted)
        self.DELTA = float(idm_config.delta)
        self.ACC_FACTOR = float(idm_config.acc_factor)
        self.DEACC_FACTOR = float(idm_config.deacc_factor)
        self.NORMAL_SPEED = float(idm_config.normal_speed_kmh)
        self.MAX_SPEED = float(idm_config.max_speed_kmh)
        self.enable_lane_change = bool(idm_config.enable_lane_change)
        self.LANE_CHANGE_FREQ = int(idm_config.lane_change_freq)
        self.LANE_CHANGE_SPEED_INCREASE = float(idm_config.lane_change_speed_increase)
        self.SAFE_LANE_CHANGE_DISTANCE = float(idm_config.safe_lane_change_distance)
        self.MAX_LONG_DIST = float(idm_config.max_long_dist)
        self.target_speed = self.NORMAL_SPEED
        self.heading_pid = PIDController(
            float(idm_config.heading_pid_kp),
            float(idm_config.heading_pid_ki),
            float(idm_config.heading_pid_kd),
        )
        self.lateral_pid = PIDController(
            float(idm_config.lateral_pid_kp),
            float(idm_config.lateral_pid_ki),
            float(idm_config.lateral_pid_kd),
        )

    def _fallback_steering_lane(self, proposed_lane):
        current_lane = getattr(self.control_object, "lane", None)
        if current_lane is not None:
            return current_lane
        if self.routing_target_lane is not None:
            return self.routing_target_lane
        return proposed_lane

    def _lane_heading_at_vehicle_position(self, lane) -> float | None:
        if lane is None or not hasattr(lane, "local_coordinates") or not hasattr(lane, "heading_theta_at"):
            return None
        try:
            longitudinal, _ = lane.local_coordinates(self.control_object.position)
            return float(lane.heading_theta_at(float(longitudinal)))
        except Exception:
            return None

    @staticmethod
    def _lane_is_opposite_by_index(current_lane, target_lane) -> bool:
        current_index = getattr(current_lane, "index", None)
        target_index = getattr(target_lane, "index", None)
        if not (
            isinstance(current_index, (tuple, list))
            and isinstance(target_index, (tuple, list))
            and len(current_index) >= 2
            and len(target_index) >= 2
        ):
            return False
        return tuple(current_index[:2]) == tuple(reversed(tuple(target_index[:2])))

    def _is_illegal_crossing_lane_change(self, target_lane) -> bool:
        current_lane = self._fallback_steering_lane(target_lane)
        if target_lane is None or current_lane is None or target_lane is current_lane:
            return False
        if bool(getattr(self.control_object, "on_yellow_continuous_line", False)) or bool(
            getattr(self.control_object, "on_white_continuous_line", False)
        ):
            return True
        if self._lane_is_opposite_by_index(current_lane, target_lane):
            return True
        current_heading = self._lane_heading_at_vehicle_position(current_lane)
        target_heading = self._lane_heading_at_vehicle_position(target_lane)
        if current_heading is None or target_heading is None:
            return False
        return abs(float(wrap_to_pi(target_heading - current_heading))) > (np.pi / 2.0)

    def _guard_steering_target_lane(self, target_lane):
        if self._is_illegal_crossing_lane_change(target_lane):
            return self._fallback_steering_lane(target_lane)
        return target_lane

    def act(self, *args, **kwargs):
        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)
        try:
            success = self.move_to_next_road()
            if success and self.enable_lane_change:
                acc_front_obj, acc_front_dist, steering_target_lane = self.lane_change_policy(all_objects)
            else:
                surrounding_objects = FrontBackObjects.get_find_front_back_objs(
                    all_objects,
                    self.routing_target_lane,
                    self.control_object.position,
                    max_distance=self.MAX_LONG_DIST,
                )
                acc_front_obj = surrounding_objects.front_object()
                acc_front_dist = surrounding_objects.front_min_distance()
                steering_target_lane = self.routing_target_lane
        except Exception:
            acc_front_obj = None
            acc_front_dist = 5
            steering_target_lane = self.routing_target_lane

        steering_target_lane = self._guard_steering_target_lane(steering_target_lane)
        steering = self.steering_control(steering_target_lane)
        acc = self.acceleration(acc_front_obj, acc_front_dist)
        action = [steering, acc]
        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)
        action[1] = self.intersection_regulator.adjust(
            ego=self.control_object,
            all_objects=all_objects,
            target_lane=steering_target_lane,
            idm_acc=action[1],
        )
        self.action_info["action"] = action
        return action

    def reset(self):
        super().reset()
        self.intersection_regulator.reset()



@dataclass
class MockVehicle:
    position: np.ndarray
    heading_theta: float
    speed: float

    @property
    def speed_km_h(self) -> float:
        return float(self.speed * 3.6)


@dataclass
class TrackingCaseResult:
    case_name: str
    figure_path: Path
    steps: int
    travelled_longitudinal: float
    final_lateral_error: float
    max_abs_lateral_error: float
    second_half_max_abs_lateral_error: float
    mean_abs_lateral_error: float
    rmse_lateral_error: float
    max_abs_heading_error_deg: float
    max_abs_steering: float


@dataclass
class SimulationConfig:
    speed_kmh: float = 30.0
    dt: float = 0.05
    max_steps: int = 400
    wheel_base: float = 2.8
    save_dir: Path = Path("tmp/expert_idm_policy_debug")

    @property
    def speed_m_s(self) -> float:
        return float(self.speed_kmh / 3.6)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone tracking test for ExpertIDMPolicy steering controller.")
    parser.add_argument(
        "--case",
        choices=("all", "curve_tracking", "intersection_turn", "roundabout_tracking"),
        default="all",
    )
    parser.add_argument("--speed-kmh", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--wheel-base", type=float, default=2.8)
    parser.add_argument("--save-dir", type=Path, default=Path("tmp/expert_idm_policy_debug"))
    return parser.parse_args()


def _build_standalone_policy(vehicle: MockVehicle) -> ExpertIDMPolicy:
    policy = ExpertIDMPolicy.__new__(ExpertIDMPolicy)
    policy.control_object = vehicle
    policy.action_info = {}
    policy.heading_pid = PIDController(1.7, 0.01, 3.5)
    policy.lateral_pid = PIDController(0.3, 0.002, 0.05)
    policy.enable_lane_change = True
    return policy


def _simulate_step(vehicle: MockVehicle, steering: float, dt: float, wheel_base: float) -> None:
    delta = float(np.clip(steering, -1.0, 1.0)) * IDMPolicy.MAX_STEERING_ANGLE
    yaw_rate = float(vehicle.speed / wheel_base * math.tan(delta))
    vehicle.heading_theta = wrap_to_pi(vehicle.heading_theta + yaw_rate * dt)
    direction = np.array([math.cos(vehicle.heading_theta), math.sin(vehicle.heading_theta)], dtype=np.float64)
    vehicle.position = vehicle.position + vehicle.speed * direction * dt


def _sample_lane_centerline(lane, num_samples: int = 200) -> np.ndarray:
    longs = np.linspace(0.0, float(lane.length), num_samples)
    return np.stack([np.asarray(lane.position(longitudinal, 0.0), dtype=np.float64) for longitudinal in longs], axis=0)


def _sample_lane_sequence(lanes: list, samples_per_lane: int = 80) -> np.ndarray:
    points = []
    for lane_index, lane in enumerate(lanes):
        num_samples = max(samples_per_lane, int(math.ceil(float(lane.length) * 2.0)))
        longs = np.linspace(0.0, float(lane.length), num_samples)
        lane_points = [np.asarray(lane.position(longitudinal, 0.0), dtype=np.float64) for longitudinal in longs]
        if lane_index > 0:
            lane_points = lane_points[1:]
        points.extend(lane_points)
    return np.stack(points, axis=0)


def _build_curve_lane() -> PointLane:
    lane_width = 4.0
    approach = StraightLane([0.0, 0.0], [20.0, 0.0], width=lane_width)
    curve, exit_straight = create_bend_straight(
        approach, 40.0, 25.0, math.radians(90.0), False, width=lane_width
    )
    centerline = _sample_lane_sequence([approach, curve, exit_straight])
    return PointLane(center_line_points=centerline, width=lane_width)


def _build_intersection_turn_lane() -> PointLane:
    lane_width = 4.0
    intersection_radius = 10.0
    lane_num = 2
    left_turn_radius = intersection_radius + lane_num * lane_width
    approach = StraightLane([0.0, 0.0], [20.0, 0.0], width=lane_width)
    turn, exit_straight = create_bend_straight(
        approach, 35.0, left_turn_radius, math.radians(90.0), False, width=lane_width
    )
    centerline = _sample_lane_sequence([approach, turn, exit_straight])
    return PointLane(center_line_points=centerline, width=lane_width)


def _build_roundabout_lane() -> PointLane:
    lane_width = 4.0
    exit_radius = 10.0
    inner_radius = 30.0
    angle_deg = 70.0
    exit_length = 60.0
    positive_lane_num = 2

    approach = StraightLane([0.0, 0.0], [20.0, 0.0], width=lane_width)
    bend_0, helper_straight_0 = create_bend_straight(
        approach, 10.0, exit_radius, math.radians(angle_deg), True, width=lane_width
    )

    radius_big = (positive_lane_num * 2 - 1) * lane_width + inner_radius
    tool_lane_1 = StraightLane(helper_straight_0.position(-5.0, 0.0), helper_straight_0.position(0.0, 0.0), width=lane_width)
    bend_1, helper_straight_1 = create_bend_straight(
        tool_lane_1, 10.0, radius_big, math.radians(2 * angle_deg - 90.0), False, width=lane_width
    )

    tool_lane_2 = StraightLane(helper_straight_1.position(-5.0, 0.0), helper_straight_1.position(0.0, 0.0), width=lane_width)
    bend_2, exit_straight = create_bend_straight(
        tool_lane_2, exit_length, exit_radius, math.radians(angle_deg), True, width=lane_width
    )

    centerline = _sample_lane_sequence([approach, bend_0, bend_1, bend_2, exit_straight])
    return PointLane(center_line_points=centerline, width=lane_width)


def _make_cases() -> dict[str, tuple[object, np.ndarray, float]]:
    curve_lane = _build_curve_lane()
    intersection_lane = _build_intersection_turn_lane()
    roundabout_lane = _build_roundabout_lane()

    curve_initial = np.array([0.0, 0.8], dtype=np.float64)
    curve_heading = math.radians(4.0)

    intersection_initial = np.array([0.0, 0.6], dtype=np.float64)
    intersection_heading = math.radians(3.0)

    roundabout_initial = np.array([0.0, 0.6], dtype=np.float64)
    roundabout_heading = math.radians(3.0)

    return {
        "curve_tracking": (curve_lane, curve_initial, curve_heading),
        "intersection_turn": (intersection_lane, intersection_initial, intersection_heading),
        "roundabout_tracking": (roundabout_lane, roundabout_initial, roundabout_heading),
    }


def _run_tracking_case(case_name: str, lane, initial_position: np.ndarray, initial_heading: float, cfg: SimulationConfig):
    vehicle = MockVehicle(position=initial_position.copy(), heading_theta=float(initial_heading), speed=cfg.speed_m_s)
    policy = _build_standalone_policy(vehicle)

    positions = [vehicle.position.copy()]
    lateral_errors = []
    heading_errors = []
    steerings = []
    longitudinals = []

    for _ in range(cfg.max_steps):
        longitudinal, lateral_error = lane.local_coordinates(vehicle.position)
        clamped_long = float(np.clip(longitudinal, 0.0, float(lane.length)))
        heading_error = wrap_to_pi(float(lane.heading_theta_at(clamped_long)) - float(vehicle.heading_theta))

        lateral_errors.append(float(lateral_error))
        heading_errors.append(float(heading_error))
        longitudinals.append(float(clamped_long))

        if clamped_long >= float(lane.length):
            break

        steering = float(policy.steering_control(lane))
        steerings.append(steering)
        _simulate_step(vehicle, steering=steering, dt=cfg.dt, wheel_base=cfg.wheel_base)
        positions.append(vehicle.position.copy())

    positions_array = np.stack(positions, axis=0)
    lateral_errors_array = np.asarray(lateral_errors, dtype=np.float64)
    heading_errors_array = np.asarray(heading_errors, dtype=np.float64)
    steerings_array = np.asarray(steerings if steerings else [0.0], dtype=np.float64)
    second_half_lateral_errors = lateral_errors_array[len(lateral_errors_array) // 2:]

    figure_path = cfg.save_dir / f"{case_name}.png"
    _save_case_figure(
        case_name=case_name,
        lane=lane,
        positions=positions_array,
        lateral_errors=lateral_errors_array,
        figure_path=figure_path,
    )

    return TrackingCaseResult(
        case_name=case_name,
        figure_path=figure_path,
        steps=int(len(lateral_errors_array)),
        travelled_longitudinal=float(longitudinals[-1] if longitudinals else 0.0),
        final_lateral_error=float(lateral_errors_array[-1] if lateral_errors_array.size else 0.0),
        max_abs_lateral_error=float(np.max(np.abs(lateral_errors_array)) if lateral_errors_array.size else 0.0),
        second_half_max_abs_lateral_error=float(np.max(np.abs(second_half_lateral_errors)) if second_half_lateral_errors.size else 0.0),
        mean_abs_lateral_error=float(np.mean(np.abs(lateral_errors_array)) if lateral_errors_array.size else 0.0),
        rmse_lateral_error=float(np.sqrt(np.mean(np.square(lateral_errors_array))) if lateral_errors_array.size else 0.0),
        max_abs_heading_error_deg=float(np.rad2deg(np.max(np.abs(heading_errors_array)) if heading_errors_array.size else 0.0)),
        max_abs_steering=float(np.max(np.abs(steerings_array))),
    )


def _save_case_figure(case_name: str, lane, positions: np.ndarray, lateral_errors: np.ndarray, figure_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for expert_idm_policy standalone plotting. Please install it first.") from exc

    reference = _sample_lane_centerline(lane)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_traj, ax_lat) = plt.subplots(2, 1, figsize=(8, 10))
    ax_traj.plot(reference[:, 0], reference[:, 1], label="reference", linewidth=2.0)
    ax_traj.plot(positions[:, 0], positions[:, 1], label="tracked", linewidth=2.0)
    ax_traj.scatter(positions[0, 0], positions[0, 1], label="start", s=40)
    ax_traj.set_title(f"{case_name}: trajectory")
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend()

    ax_lat.plot(np.arange(len(lateral_errors)), lateral_errors, linewidth=2.0)
    ax_lat.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax_lat.set_title(f"{case_name}: lateral error")
    ax_lat.set_xlabel("step")
    ax_lat.set_ylabel("lateral error [m]")
    ax_lat.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)


def _print_case_result(result: TrackingCaseResult) -> None:
    print(
        f"[case={result.case_name}] \n"
        f"steps={result.steps} \n"
        f"travelled_longitudinal={result.travelled_longitudinal:.3f} \n"
        f"final_lateral_error={result.final_lateral_error:+.3f} \n"
        f"max_abs_lateral_error={result.max_abs_lateral_error:.3f} \n"
        f"second_half_max_abs_lateral_error={result.second_half_max_abs_lateral_error:.3f} \n"
        f"mean_abs_lateral_error={result.mean_abs_lateral_error:.3f} \n"
        f"rmse_lateral_error={result.rmse_lateral_error:.3f} \n"
        f"max_abs_heading_error_deg={result.max_abs_heading_error_deg:.3f} \n"
        f"max_abs_steering={result.max_abs_steering:.3f} \n"
        f"figure_path={result.figure_path} \n"
    )


def main() -> None:
    args = _parse_args()
    cfg = SimulationConfig(
        speed_kmh=float(args.speed_kmh),
        dt=float(args.dt),
        max_steps=int(args.max_steps),
        wheel_base=float(args.wheel_base),
        save_dir=Path(args.save_dir),
    )

    cases = _make_cases()
    selected_cases = list(cases.keys()) if args.case == "all" else [args.case]

    for case_name in selected_cases:
        lane, initial_position, initial_heading = cases[case_name]
        result = _run_tracking_case(case_name, lane, initial_position, initial_heading, cfg)
        _print_case_result(result)


if __name__ == "__main__":
    main()
