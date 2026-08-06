"""Isolated trajectory-control benchmark for the BEV expert chain.

The benchmark deliberately bypasses RuleMaker and the Normal planner.  Every
reference is audited before it is handed to the real sensorless environment,
so a controller failure cannot be hidden by an invalid test trajectory.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from evaluation.joint_simulator_branch import (
    JointEpisodeSpec,
    JointSimulatorBranchEvaluator,
    SimulatorBranchResult,
    capture_joint_pose_global,
)
from evaluation.longitudinal_tracking_diagnostics import (
    build_longitudinal_tracking_report,
)
from expert_dataset.collect_joint_bev import SensorlessJointBEVPlatoonEnv
from models.bev_planner.mode_contract import validate_trajectory_kinematics
from models.controller.longitudinal_reference import (
    LongitudinalCascadeController,
    signed_longitudinal_speed_mps,
    trajectory_to_longitudinal_reference,
)


AGENT_IDS = ("agent0", "agent1", "agent2")
DEFAULT_SCENARIO = "S1_free_cruise_straight"
DEFAULT_ROUTE = "R3_mainline_straight"


class ControlBenchmarkError(RuntimeError):
    """Raised when the isolated control benchmark contract is violated."""


class _IndependentControlEnv(SensorlessJointBEVPlatoonEnv):
    """Production sensorless environment with follower gap feedback disabled."""

    def trajectory_reference_to_control(
        self,
        agent_id: str,
        trajectory: np.ndarray,
        longitudinal_reference,
    ) -> np.ndarray:
        value = np.asarray(trajectory)
        if value.shape != (8, 3) or not np.isfinite(value).all():
            raise ControlBenchmarkError("independent trajectory must be finite [8,3]")
        steering = self._lateral_preview_pid(agent_id, value)
        current_speed = signed_longitudinal_speed_mps(self.agents[agent_id])
        controller = getattr(self, "_trajectory_longitudinal_controller", None)
        if controller is None:
            controller = LongitudinalCascadeController(
                dt_s=self._cfg_float("physics_world_step_size", 0.02)
                * self._cfg_int("decision_repeat", 5)
            )
            self._trajectory_longitudinal_controller = controller
        throttle, _ = controller.compute(
            agent_id,
            current_speed,
            longitudinal_reference,
            gap_acceleration_mps2=0.0,
        )
        return np.asarray([steering, throttle], dtype=np.float32)


@dataclass(frozen=True)
class ControlTrackingCase:
    name: str
    initial_speed_mps: float
    trajectory: np.ndarray
    category: str

    def __post_init__(self) -> None:
        value = np.asarray(self.trajectory)
        speed = float(self.initial_speed_mps)
        if (
            not self.name
            or not self.category
            or value.shape != (8, 3)
            or not np.issubdtype(value.dtype, np.floating)
            or not np.isfinite(value).all()
            or not np.isfinite(speed)
            or speed < 0.0
        ):
            raise ControlBenchmarkError("invalid control benchmark case")
        audit = validate_trajectory_kinematics(
            value,
            speed,
            np.zeros((3,), dtype=np.float64),
        )
        if not audit.valid:
            raise ControlBenchmarkError(
                f"case {self.name!r} violates trajectory contract: "
                f"{audit.reason}"
            )
        frozen = np.ascontiguousarray(value, dtype=np.float32)
        frozen.setflags(write=False)
        object.__setattr__(self, "initial_speed_mps", speed)
        object.__setattr__(self, "trajectory", frozen)


def _times() -> np.ndarray:
    return np.arange(1, 9, dtype=np.float64) * 0.5


def _straight_profile(
    initial_speed_mps: float,
    *,
    acceleration_mps2: float = 0.0,
    minimum_speed_mps: float = 0.0,
    maximum_speed_mps: float | None = None,
) -> np.ndarray:
    times = _times()
    speed0 = float(initial_speed_mps)
    accel = float(acceleration_mps2)
    if accel < 0.0:
        stop_time = max((speed0 - minimum_speed_mps) / -accel, 0.0)
        active = np.minimum(times, stop_time)
        distance = speed0 * active + 0.5 * accel * active**2
        distance += minimum_speed_mps * np.maximum(times - stop_time, 0.0)
    elif accel > 0.0 and maximum_speed_mps is not None:
        cap_time = max((float(maximum_speed_mps) - speed0) / accel, 0.0)
        active = np.minimum(times, cap_time)
        distance = speed0 * active + 0.5 * accel * active**2
        distance += float(maximum_speed_mps) * np.maximum(times - cap_time, 0.0)
    else:
        distance = speed0 * times + 0.5 * accel * times**2
    result = np.zeros((8, 3), dtype=np.float32)
    result[:, 0] = distance.astype(np.float32)
    return result


def _lane_change_profile(
    speed_mps: float,
    lateral_distance_m: float,
    *,
    duration_s: float = 4.0,
) -> np.ndarray:
    times = _times()
    tau = np.clip(times / float(duration_s), 0.0, 1.0)
    blend = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    derivative = (30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4) / float(
        duration_s
    )
    result = np.zeros((8, 3), dtype=np.float32)
    result[:, 0] = (float(speed_mps) * times).astype(np.float32)
    result[:, 1] = (float(lateral_distance_m) * blend).astype(np.float32)
    result[:, 2] = np.arctan2(
        float(lateral_distance_m) * derivative,
        float(speed_mps),
    ).astype(np.float32)
    return result


def _constant_curvature_profile(speed_mps: float, radius_m: float) -> np.ndarray:
    times = _times()
    arc = float(speed_mps) * times
    angle = arc / float(radius_m)
    result = np.zeros((8, 3), dtype=np.float32)
    result[:, 0] = (float(radius_m) * np.sin(angle)).astype(np.float32)
    result[:, 1] = (float(radius_m) * (1.0 - np.cos(angle))).astype(np.float32)
    result[:, 2] = angle.astype(np.float32)
    return result


def standard_control_cases() -> tuple[ControlTrackingCase, ...]:
    """Return deterministic, hard-valid controller references."""

    return (
        ControlTrackingCase(
            "constant_4", 4.0, _straight_profile(4.0), "longitudinal"
        ),
        ControlTrackingCase(
            "constant_8", 8.0, _straight_profile(8.0), "longitudinal"
        ),
        ControlTrackingCase(
            "accelerate_4_plus_0p4",
            4.0,
            _straight_profile(4.0, acceleration_mps2=0.4),
            "longitudinal",
        ),
        ControlTrackingCase(
            "brake_8_to_4",
            8.0,
            _straight_profile(8.0, acceleration_mps2=-2.0, minimum_speed_mps=4.0),
            "longitudinal",
        ),
        ControlTrackingCase(
            "stop_6",
            6.0,
            _straight_profile(6.0, acceleration_mps2=-2.0),
            "stop",
        ),
        ControlTrackingCase(
            "lane_change_left",
            6.0,
            _lane_change_profile(6.0, 3.5),
            "lateral",
        ),
        ControlTrackingCase(
            "lane_change_right",
            6.0,
            _lane_change_profile(6.0, -3.5),
            "lateral",
        ),
        ControlTrackingCase(
            "curve_radius_500",
            6.0,
            _constant_curvature_profile(6.0, 500.0),
            "curve",
        ),
        ControlTrackingCase(
            "merge_continuous",
            6.0,
            _lane_change_profile(6.0, 2.5),
            "merge",
        ),
    )


def evaluate_gap_feedback_contract() -> dict[str, object]:
    """Exercise independent/locked follower corrections without a planner.

    The two zero-correction cases distinguish the emergency-independent and
    nominal locked contracts.  The signed extreme cases prove direction and
    the public ``+/-1 m/s2`` bound before any simulator rollout is attempted.
    """

    trajectory = _straight_profile(6.0)
    reference = trajectory_to_longitudinal_reference(
        trajectory,
        6.0,
        source="control_benchmark_gap_contract",
    )
    rows = []
    for name, correction in (
        ("independent", 0.0),
        ("locked_nominal", 0.0),
        ("follower_too_close", -1.0),
        ("follower_too_far", 1.0),
    ):
        controller = LongitudinalCascadeController()
        throttle, debug = controller.compute(
            "agent1",
            6.0,
            reference,
            gap_acceleration_mps2=correction,
        )
        rows.append(
            {
                "name": name,
                "requested_gap_feedback_mps2": correction,
                "applied_gap_feedback_mps2": float(
                    debug["gap_feedback_mps2"]
                ),
                "normalized_throttle": float(throttle),
            }
        )
    return {
        "cases": rows,
        "passed": (
            rows[0]["applied_gap_feedback_mps2"] == 0.0
            and rows[1]["applied_gap_feedback_mps2"] == 0.0
            and rows[2]["applied_gap_feedback_mps2"] == -1.0
            and rows[3]["applied_gap_feedback_mps2"] == 1.0
            and rows[2]["normalized_throttle"]
            < rows[0]["normalized_throttle"]
            < rows[3]["normalized_throttle"]
        ),
    }


def _percentile(values: Iterable[float], percentile: float) -> float:
    array = np.abs(np.asarray(tuple(values), dtype=np.float64))
    if array.size == 0 or not np.isfinite(array).all():
        raise ControlBenchmarkError("tracking trace is empty or non-finite")
    return float(np.percentile(array, percentile))


def summarize_control_result(
    case: ControlTrackingCase,
    result: SimulatorBranchResult,
    *,
    control_mode: str = "locked",
) -> dict[str, object]:
    if result.reward.rewards.shape != (1,) or len(result.tracking_traces) != 1:
        raise ControlBenchmarkError("one control case must produce one branch")
    longitudinal = build_longitudinal_tracking_report(
        result,
        np.stack([[case.trajectory] * 3]).astype(np.float32),
    )
    role_rows = []
    lateral_all: list[float] = []
    heading_all: list[float] = []
    speed_error_all: list[float] = []
    follower_gap_error_all: list[float] = []
    for role, trace in enumerate(result.tracking_traces[0]):
        lateral = np.asarray(trace["lateral_errors_m"], dtype=np.float64)
        heading = np.asarray(trace["heading_errors_rad"], dtype=np.float64)
        lateral_all.extend(lateral.tolist())
        heading_all.extend(heading.tolist())
        actual_speed = np.asarray(trace["actual_speed_mps"], dtype=np.float64)
        reference_speed = np.asarray(
            trace["reference_feedforward_speed_mps"], dtype=np.float64
        )
        if actual_speed.shape != reference_speed.shape:
            raise ControlBenchmarkError("speed tracking trace shape mismatch")
        speed_error = reference_speed - actual_speed
        speed_error_all.extend(speed_error.tolist())
        gap_error = np.asarray(
            trace["formation_gap_error_m"], dtype=np.float64
        )
        if role > 0:
            follower_gap_error_all.extend(gap_error.tolist())
        role_rows.append(
            {
                "role": role,
                "longitudinal_p95_m": _percentile(
                    trace["longitudinal_errors_m"], 95
                ),
                "lateral_p95_m": _percentile(lateral, 95),
                "heading_p95_rad": _percentile(heading, 95),
                "maximum_continuous_saturation_s": float(
                    trace["maximum_continuous_saturation_s"]
                ),
                "terminal_speed_mps": float(trace["actual_speed_mps"][-1]),
                "signed_speed_error_p95_mps": _percentile(speed_error, 95),
                "desired_center_gap_error_p95_m": (
                    _percentile(gap_error, 95) if role > 0 else 0.0
                ),
                "terminal_desired_center_gap_error_m": (
                    float(abs(gap_error[-1])) if role > 0 else 0.0
                ),
                "formation_control_increment_abs_max_normalized": float(
                    np.max(np.abs(trace["formation_control_increment"]))
                ),
            }
        )
    lateral_p95 = _percentile(lateral_all, 95)
    heading_p95 = _percentile(heading_all, 95)
    unsafe = bool(result.reward.unsafe[0])
    stop_speed = (
        max(abs(float(row["terminal_speed_mps"])) for row in role_rows)
        if case.category == "stop"
        else 0.0
    )
    speed_error_p95 = _percentile(speed_error_all, 95)
    gap_error_p95 = (
        _percentile(follower_gap_error_all, 95)
        if follower_gap_error_all
        else 0.0
    )
    terminal_gap_error = max(
        (float(row["terminal_desired_center_gap_error_m"]) for row in role_rows),
        default=0.0,
    )
    reverse_motion = any(
        np.any(np.asarray(trace["actual_speed_mps"], dtype=np.float64) < -0.05)
        for trace in result.tracking_traces[0]
    )
    blockers: list[str] = []
    if control_mode == "independent":
        if speed_error_p95 > 0.5:
            blockers.append("signed_speed_error_p95")
    elif control_mode == "locked":
        if gap_error_p95 > 1.5:
            blockers.append("desired_center_gap_error_p95")
        if terminal_gap_error > 1.0:
            blockers.append("terminal_desired_center_gap_error")
    else:
        raise ControlBenchmarkError(f"unknown control mode {control_mode!r}")
    if longitudinal.maximum_continuous_saturation_s > 1.0:
        blockers.append("continuous_control_saturation")
    if lateral_p95 > 0.5:
        blockers.append("lateral_p95")
    if heading_p95 > 0.1:
        blockers.append("heading_p95")
    if stop_speed > 0.3:
        blockers.append("stop_terminal_speed")
    if reverse_motion:
        blockers.append("reverse_motion")
    if longitudinal.target_reference_speed_delta_p95_mps > 0.1:
        blockers.append("target_reference_speed_pollution")
    if unsafe:
        blockers.append("simulator_unsafe")
    return {
        "name": case.name,
        "control_mode": str(control_mode),
        "category": case.category,
        "initial_speed_mps": case.initial_speed_mps,
        "trajectory_contract_valid": True,
        "longitudinal_p95_m": longitudinal.longitudinal_error_p95_m,
        "longitudinal_p99_m": longitudinal.longitudinal_error_p99_m,
        "signed_speed_error_p95_mps": speed_error_p95,
        "desired_center_gap_error_p95_m": gap_error_p95,
        "terminal_desired_center_gap_error_m": terminal_gap_error,
        "lateral_p95_m": lateral_p95,
        "heading_p95_rad": heading_p95,
        "target_reference_speed_delta_p95_mps": (
            longitudinal.target_reference_speed_delta_p95_mps
        ),
        "maximum_continuous_saturation_s": (
            longitudinal.maximum_continuous_saturation_s
        ),
        "stop_terminal_speed_mps": stop_speed,
        "reverse_motion": reverse_motion,
        "unsafe": unsafe,
        "failure_reasons": list(result.failure_reasons[0]),
        "roles": role_rows,
        "passed": not blockers,
        "blockers": sorted(set(blockers)),
    }


def _episode_spec(
    evaluator: JointSimulatorBranchEvaluator,
    case: ControlTrackingCase,
    *,
    seed: int,
) -> JointEpisodeSpec:
    config = {"initial_speed_km_h": case.initial_speed_mps * 3.6}
    provisional = JointEpisodeSpec(
        DEFAULT_SCENARIO,
        DEFAULT_ROUTE,
        int(seed),
        np.zeros((3, 3), dtype=np.float64),
        env_config=config,
    )
    env = evaluator._make_env(provisional)
    try:
        reference = capture_joint_pose_global(env)
    finally:
        env.close()
    return JointEpisodeSpec(
        DEFAULT_SCENARIO,
        DEFAULT_ROUTE,
        int(seed),
        reference,
        env_config=config,
    )


def run_control_tracking_benchmark(
    output_root: Path | str,
    *,
    case_names: Sequence[str] | None = None,
    seed: int = 17,
    evaluator: JointSimulatorBranchEvaluator | None = None,
    control_modes: Sequence[str] | None = None,
) -> dict[str, object]:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    selected = standard_control_cases()
    if case_names is not None:
        requested = tuple(str(value) for value in case_names)
        by_name = {case.name: case for case in selected}
        unknown = sorted(set(requested) - set(by_name))
        if unknown:
            raise ControlBenchmarkError(f"unknown control cases: {unknown}")
        selected = tuple(by_name[name] for name in requested)
    modes = tuple(str(value) for value in (control_modes or ("locked", "independent")))
    if evaluator is not None and control_modes is None:
        modes = ("locked",)
    if not modes or any(value not in {"locked", "independent"} for value in modes):
        raise ControlBenchmarkError(
            "control_modes must contain locked and/or independent"
        )
    evaluators = {
        "locked": evaluator or JointSimulatorBranchEvaluator(),
        "independent": (
            evaluator
            if evaluator is not None
            else JointSimulatorBranchEvaluator(env_factory=_IndependentControlEnv)
        ),
    }
    rows = []
    for control_mode in modes:
        branch = evaluators[control_mode]
        for case in selected:
            spec = _episode_spec(branch, case, seed=seed)
            candidates = np.stack([[case.trajectory] * 3]).astype(np.float32)
            result = branch.evaluate(spec, (), candidates)
            row = summarize_control_result(
                case, result, control_mode=control_mode
            )
            rows.append(row)
            traces = result.tracking_traces[0]
            np.savez_compressed(
                output / f"{control_mode}_{case.name}.npz",
                trajectory=case.trajectory,
                reference_world=np.asarray(
                    [trace["reference_world"] for trace in traces],
                    dtype=np.float64,
                ),
                actual_world=np.asarray(
                    [trace["actual_world"] for trace in traces],
                    dtype=np.float64,
                ),
                longitudinal_error_m=np.asarray(
                    [trace["longitudinal_errors_m"] for trace in traces],
                    dtype=np.float64,
                ),
                lateral_error_m=np.asarray(
                    [trace["lateral_errors_m"] for trace in traces],
                    dtype=np.float64,
                ),
                heading_error_rad=np.asarray(
                    [trace["heading_errors_rad"] for trace in traces],
                    dtype=np.float64,
                ),
                steering=np.asarray(
                    [trace["steering"] for trace in traces], dtype=np.float64
                ),
                throttle=np.asarray(
                    [trace["throttle"] for trace in traces], dtype=np.float64
                ),
                actual_speed_mps=np.asarray(
                    [trace["actual_speed_mps"] for trace in traces],
                    dtype=np.float64,
                ),
                actual_acceleration_mps2=np.asarray(
                    [trace["actual_acceleration_mps2"] for trace in traces],
                    dtype=np.float64,
                ),
                reference_speed_mps=np.asarray(
                    [trace["reference_feedforward_speed_mps"] for trace in traces],
                    dtype=np.float64,
                ),
                reference_acceleration_mps2=np.asarray(
                    [
                        trace["reference_feedforward_acceleration_mps2"]
                        for trace in traces
                    ],
                    dtype=np.float64,
                ),
                position_error_speed_increment_mps=np.asarray(
                    [
                        trace["position_error_speed_increment_mps"]
                        for trace in traces
                    ],
                    dtype=np.float64,
                ),
                formation_control_increment=np.asarray(
                    [trace["formation_control_increment"] for trace in traces],
                    dtype=np.float64,
                ),
                formation_gap_error_m=np.asarray(
                    [trace["formation_gap_error_m"] for trace in traces],
                    dtype=np.float64,
                ),
                control_saturated=np.asarray(
                    [trace["control_saturated"] for trace in traces],
                    dtype=np.bool_,
                ),
            )
    gap_contract = evaluate_gap_feedback_contract()
    report = {
        "format": "bev_control_tracking_benchmark_v1",
        "diagnostic_only": True,
        "scenario": DEFAULT_SCENARIO,
        "route": DEFAULT_ROUTE,
        "seed": int(seed),
        "control_modes": list(modes),
        "thresholds": {
            "longitudinal_p95_m": 1.0,
            "longitudinal_p99_m": 1.5,
            "independent_signed_speed_error_p95_mps": 0.5,
            "locked_desired_center_gap_error_p95_m": 1.5,
            "locked_terminal_desired_center_gap_error_m": 1.0,
            "lateral_p95_m": 0.5,
            "heading_p95_rad": 0.1,
            "stop_terminal_speed_mps": 0.3,
            "maximum_continuous_saturation_s": 1.0,
            "target_reference_speed_delta_p95_mps": 0.1,
        },
        "cases": rows,
        "gap_feedback_contract": gap_contract,
        "passed": (
            bool(rows)
            and all(bool(row["passed"]) for row in rows)
            and bool(gap_contract["passed"])
        ),
    }
    (output / "control_benchmark_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cases", nargs="+", default=None)
    parser.add_argument(
        "--control-modes",
        nargs="+",
        choices=("locked", "independent"),
        default=None,
    )
    args = parser.parse_args()
    report = run_control_tracking_benchmark(
        args.output_root,
        case_names=args.cases,
        seed=args.seed,
        control_modes=args.control_modes,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "ControlBenchmarkError",
    "ControlTrackingCase",
    "run_control_tracking_benchmark",
    "standard_control_cases",
    "summarize_control_result",
    "evaluate_gap_feedback_contract",
]
