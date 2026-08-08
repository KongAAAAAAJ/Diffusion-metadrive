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
from expert_dataset.collect_joint_bev import (
    SensorlessJointBEVPlatoonEnv,
    simulator_decision_dt_s,
)
from models.bev_planner.mode_contract import validate_trajectory_kinematics
from models.controller.longitudinal_reference import (
    BRAKE_ACCELERATION_SCALE_MPS2,
    DRIVE_ACCELERATION_SCALE_MPS2,
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
                * self._cfg_int("decision_repeat", 5),
                acceleration_bias_mps2=self._cfg_float(
                    "acceleration_bias_mps2", 0.0
                ),
                drive_acceleration_scale_mps2=self._cfg_float(
                    "drive_acceleration_scale_mps2",
                    DRIVE_ACCELERATION_SCALE_MPS2,
                ),
                brake_acceleration_scale_mps2=self._cfg_float(
                    "brake_acceleration_scale_mps2",
                    BRAKE_ACCELERATION_SCALE_MPS2,
                ),
            )
            self._trajectory_longitudinal_controller = controller
        throttle, longitudinal_debug = controller.compute(
            agent_id,
            current_speed,
            longitudinal_reference,
            gap_acceleration_mps2=0.0,
        )
        self._last_longitudinal_control_debug[agent_id] = dict(
            longitudinal_debug
        )
        return np.asarray([steering, throttle], dtype=np.float32)


class _ControlBenchmarkBranchEvaluator(JointSimulatorBranchEvaluator):
    """Branch evaluator that additionally replays low-level warm-start controls.

    The production reward evaluator deliberately keeps its trajectory-only
    prefix contract.  This diagnostic subclass broadens that contract only for
    the isolated controller benchmark.
    """

    @staticmethod
    def _validate_prefix(
        prefix_actions: Sequence[Mapping[str, np.ndarray]],
    ) -> tuple[dict[str, np.ndarray], ...]:
        checked: list[dict[str, np.ndarray]] = []
        for step in prefix_actions:
            if not isinstance(step, Mapping) or set(step) != set(AGENT_IDS):
                raise ControlBenchmarkError(
                    "each warm-start action must contain exactly agent0/1/2"
                )
            values: dict[str, np.ndarray] = {}
            for agent_id in AGENT_IDS:
                control = np.asarray(step[agent_id])
                if (
                    control.shape != (2,)
                    or not np.issubdtype(control.dtype, np.floating)
                    or not np.isfinite(control).all()
                ):
                    raise ControlBenchmarkError(
                        "warm-start controls must be finite floating-point [2]"
                    )
                values[agent_id] = np.ascontiguousarray(
                    control, dtype=np.float32
                )
            checked.append(values)
        return tuple(checked)


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
        stop_requested=np.full(
            (1, 3), case.category == "stop", dtype=np.bool_
        ),
        tracking_group_mask=np.ones(1, dtype=np.bool_),
    )
    role_rows = []
    lateral_all: list[float] = []
    heading_all: list[float] = []
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
        gap_error = np.asarray(
            trace["formation_gap_error_m"], dtype=np.float64
        )
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
    # Acceptance is role-wise: pooling three traces can hide a single bad
    # vehicle behind the other two.  Independent mode judges the worst speed
    # tracker; locked mode judges the worst follower gap tracker.
    speed_error_p95 = max(
        float(row["signed_speed_error_p95_mps"]) for row in role_rows
    )
    gap_error_p95 = max(
        (
            float(row["desired_center_gap_error_p95_m"])
            for row in role_rows
            if int(row["role"]) > 0
        ),
        default=0.0,
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


def _prepare_natural_start(
    evaluator: JointSimulatorBranchEvaluator,
    case: ControlTrackingCase,
    *,
    seed: int,
    acceleration_bias_mps2: float,
    drive_acceleration_scale_mps2: float,
    brake_acceleration_scale_mps2: float,
) -> tuple[
    JointEpisodeSpec,
    tuple[Mapping[str, np.ndarray], ...],
    dict[str, object],
]:
    """Reach the case initial speed through vehicle dynamics, never teleporting it.

    The returned low-level prefix is replayed by every simulator branch.  Its
    samples are diagnostic initialization data and are deliberately excluded
    from the four-second trajectory tracking metrics.
    """

    config = {
        "initial_speed_km_h": 0.0,
        # R3 has three lanes.  Starting on the centre lane makes both signed
        # lane-change references physically meaningful instead of driving one
        # direction out of the road from edge lane 0.
        "platoon_route_spawn_lane_index": 1,
        "acceleration_bias_mps2": float(acceleration_bias_mps2),
        "drive_acceleration_scale_mps2": float(
            drive_acceleration_scale_mps2
        ),
        "brake_acceleration_scale_mps2": float(
            brake_acceleration_scale_mps2
        ),
    }
    provisional = JointEpisodeSpec(
        DEFAULT_SCENARIO,
        DEFAULT_ROUTE,
        int(seed),
        np.zeros((3, 3), dtype=np.float64),
        env_config=config,
    )
    env = evaluator._make_env(provisional)
    try:
        # Lightweight test evaluators do not expose a real simulation step.
        # Preserve their isolated summary contract without pretending that a
        # physical warm start was performed.
        if not callable(getattr(env, "step", None)):
            reference = capture_joint_pose_global(env)
            direct_config = dict(config)
            direct_config["initial_speed_km_h"] = case.initial_speed_mps * 3.6
            return (
                JointEpisodeSpec(
                    DEFAULT_SCENARIO,
                    DEFAULT_ROUTE,
                    int(seed),
                    reference,
                    env_config=direct_config,
                ),
                (),
                {
                    "mode": "test_double_direct_start",
                    "target_speed_mps": case.initial_speed_mps,
                    "steps": 0,
                    "duration_s": 0.0,
                    "terminal_speed_mps": [case.initial_speed_mps] * 3,
                    "speed_history_mps": [],
                    "throttle_history": [],
                },
            )

        target = float(case.initial_speed_mps)
        dt_s = float(simulator_decision_dt_s(env))
        maximum_steps = max(1, int(math.ceil(20.0 / dt_s)))
        required_stable_steps = max(1, int(math.ceil(1.0 / dt_s)))
        stable_steps = 0
        prefix: list[Mapping[str, np.ndarray]] = []
        speed_history: list[list[float]] = []
        throttle_history: list[list[float]] = []
        for _ in range(maximum_steps):
            speeds = [
                float(signed_longitudinal_speed_mps(env.agents[agent_id]))
                for agent_id in AGENT_IDS
            ]
            controls: dict[str, np.ndarray] = {}
            throttles: list[float] = []
            for agent_id, speed in zip(AGENT_IDS, speeds):
                error = target - speed
                # This is only a deterministic physical-state initializer.  It
                # intentionally does not reuse or tune the controller under
                # test, and permits mild braking to settle an overshoot.
                throttle = float(np.clip(0.05 + 0.22 * error, -0.25, 0.65))
                controls[agent_id] = np.asarray(
                    [0.0, throttle], dtype=np.float32
                )
                throttles.append(throttle)
            prefix.append(
                {key: value.copy() for key, value in controls.items()}
            )
            speed_history.append(speeds)
            throttle_history.append(throttles)
            _, _, terminated, truncated, _ = env.step(controls)
            terminated_all = (
                bool(terminated.get("__all__", False))
                if isinstance(terminated, Mapping)
                else bool(terminated)
            )
            truncated_all = (
                bool(truncated.get("__all__", False))
                if isinstance(truncated, Mapping)
                else bool(truncated)
            )
            if terminated_all or truncated_all:
                raise ControlBenchmarkError(
                    "natural-start initialization terminated before target speed"
                )
            next_speeds = [
                float(signed_longitudinal_speed_mps(env.agents[agent_id]))
                for agent_id in AGENT_IDS
            ]
            if max(abs(target - value) for value in next_speeds) <= 0.15:
                stable_steps += 1
            else:
                stable_steps = 0
            if stable_steps >= required_stable_steps:
                break
        else:
            raise ControlBenchmarkError(
                "natural-start initialization did not stabilize within 20s"
            )

        reference = capture_joint_pose_global(env)
        terminal_speeds = [
            float(signed_longitudinal_speed_mps(env.agents[agent_id]))
            for agent_id in AGENT_IDS
        ]
        dynamics_keys = (
            "max_engine_force",
            "max_brake_force",
            "wheel_friction",
            "mass",
        )
        vehicle_dynamics = []
        for agent_id in AGENT_IDS:
            parameters = env.agents[agent_id].get_dynamics_parameters()
            vehicle_dynamics.append(
                {
                    key: float(parameters[key])
                    for key in dynamics_keys
                }
            )
    finally:
        env.close()
    spec = JointEpisodeSpec(
        DEFAULT_SCENARIO, DEFAULT_ROUTE, int(seed), reference, env_config=config
    )
    diagnostics = {
        "mode": "natural_from_rest",
        "target_speed_mps": target,
        "steps": len(prefix),
        "duration_s": len(prefix) * dt_s,
        "terminal_speed_mps": terminal_speeds,
        "maximum_terminal_speed_error_mps": max(
            abs(target - value) for value in terminal_speeds
        ),
        "vehicle_dynamics": vehicle_dynamics,
        "speed_history_mps": speed_history,
        "throttle_history": throttle_history,
    }
    return spec, tuple(prefix), diagnostics


def run_control_tracking_benchmark(
    output_root: Path | str,
    *,
    case_names: Sequence[str] | None = None,
    seed: int = 17,
    evaluator: JointSimulatorBranchEvaluator | None = None,
    control_modes: Sequence[str] | None = None,
    acceleration_bias_mps2: float = 0.0,
    drive_acceleration_scale_mps2: float = DRIVE_ACCELERATION_SCALE_MPS2,
    brake_acceleration_scale_mps2: float = BRAKE_ACCELERATION_SCALE_MPS2,
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
    if not np.isfinite(float(acceleration_bias_mps2)):
        raise ControlBenchmarkError("acceleration_bias_mps2 must be finite")
    if (
        not np.isfinite(float(drive_acceleration_scale_mps2))
        or float(drive_acceleration_scale_mps2) <= 0.0
    ):
        raise ControlBenchmarkError(
            "drive_acceleration_scale_mps2 must be finite and positive"
        )
    if (
        not np.isfinite(float(brake_acceleration_scale_mps2))
        or float(brake_acceleration_scale_mps2) <= 0.0
    ):
        raise ControlBenchmarkError(
            "brake_acceleration_scale_mps2 must be finite and positive"
        )
    evaluators = {
        "locked": evaluator or _ControlBenchmarkBranchEvaluator(),
        "independent": (
            evaluator
            if evaluator is not None
            else _ControlBenchmarkBranchEvaluator(
                env_factory=_IndependentControlEnv
            )
        ),
    }
    rows = []
    for control_mode in modes:
        branch = evaluators[control_mode]
        for case in selected:
            spec, prefix_actions, initialization = _prepare_natural_start(
                branch,
                case,
                seed=seed,
                acceleration_bias_mps2=float(acceleration_bias_mps2),
                drive_acceleration_scale_mps2=float(
                    drive_acceleration_scale_mps2
                ),
                brake_acceleration_scale_mps2=float(
                    brake_acceleration_scale_mps2
                ),
            )
            candidates = np.stack([[case.trajectory] * 3]).astype(np.float32)
            result = branch.evaluate(spec, prefix_actions, candidates)
            row = summarize_control_result(
                case, result, control_mode=control_mode
            )
            row["initialization"] = {
                key: value
                for key, value in initialization.items()
                if key not in {"speed_history_mps", "throttle_history"}
            }
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
                desired_acceleration_mps2=np.asarray(
                    [trace["desired_acceleration_mps2"] for trace in traces],
                    dtype=np.float64,
                ),
                compensated_acceleration_mps2=np.asarray(
                    [trace["compensated_acceleration_mps2"] for trace in traces],
                    dtype=np.float64,
                ),
                raw_desired_acceleration_mps2=np.asarray(
                    [trace["raw_desired_acceleration_mps2"] for trace in traces],
                    dtype=np.float64,
                ),
                acceleration_bias_mps2=np.asarray(
                    [trace["acceleration_bias_mps2"] for trace in traces],
                    dtype=np.float64,
                ),
                brake_acceleration_scale_mps2=np.asarray(
                    [
                        trace["brake_acceleration_scale_mps2"]
                        for trace in traces
                    ],
                    dtype=np.float64,
                ),
                controller_speed_error_mps=np.asarray(
                    [trace["controller_speed_error_mps"] for trace in traces],
                    dtype=np.float64,
                ),
                controller_preview_speed_mps=np.asarray(
                    [trace["controller_preview_speed_mps"] for trace in traces],
                    dtype=np.float64,
                ),
                controller_preview_acceleration_mps2=np.asarray(
                    [
                        trace["controller_preview_acceleration_mps2"]
                        for trace in traces
                    ],
                    dtype=np.float64,
                ),
                position_feedback_mps2=np.asarray(
                    [trace["position_feedback_mps2"] for trace in traces],
                    dtype=np.float64,
                ),
                gap_feedback_mps2=np.asarray(
                    [trace["gap_feedback_mps2"] for trace in traces],
                    dtype=np.float64,
                ),
                speed_integral=np.asarray(
                    [trace["speed_integral"] for trace in traces],
                    dtype=np.float64,
                ),
                speed_overzero_guard=np.asarray(
                    [trace["speed_overzero_guard"] for trace in traces],
                    dtype=np.bool_,
                ),
                control_regime=np.asarray(
                    [trace["control_regime"] for trace in traces],
                    dtype=np.str_,
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
                warmup_speed_mps=np.asarray(
                    initialization["speed_history_mps"], dtype=np.float64
                ),
                warmup_throttle=np.asarray(
                    initialization["throttle_history"], dtype=np.float64
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
        "acceleration_bias_mps2": float(acceleration_bias_mps2),
        "drive_acceleration_scale_mps2": float(
            drive_acceleration_scale_mps2
        ),
        "brake_acceleration_scale_mps2": float(
            brake_acceleration_scale_mps2
        ),
        "initialization": "natural_from_rest",
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
    parser.add_argument("--acceleration-bias-mps2", type=float, default=0.0)
    parser.add_argument(
        "--drive-acceleration-scale-mps2",
        type=float,
        default=DRIVE_ACCELERATION_SCALE_MPS2,
    )
    parser.add_argument(
        "--brake-acceleration-scale-mps2",
        type=float,
        default=BRAKE_ACCELERATION_SCALE_MPS2,
    )
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
        acceleration_bias_mps2=args.acceleration_bias_mps2,
        drive_acceleration_scale_mps2=args.drive_acceleration_scale_mps2,
        brake_acceleration_scale_mps2=args.brake_acceleration_scale_mps2,
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
