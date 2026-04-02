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


class ExpertIDMPolicy(IDMPolicy):
    """Dataset-facing IDM policy aligned with the background-traffic IDM."""

    def __init__(self, control_object, random_seed: int = 0):
        super().__init__(control_object=control_object, random_seed=random_seed)



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
