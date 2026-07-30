"""Longitudinal feasibility and closed-loop tracking diagnostics.

This module is intentionally observational.  It never projects trajectories,
changes controller commands, or selects a fallback mode.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from models.bev_planner.mode_contract import (
    HardModeMaskConfig,
    TRAJECTORY_DIM,
    TRAJECTORY_STEPS,
    validate_trajectory_kinematics,
)


class LongitudinalDiagnosticError(RuntimeError):
    """Raised when longitudinal diagnostic inputs violate the strict contract."""


def _readonly(array: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    value = np.ascontiguousarray(array, dtype=dtype)
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class LongitudinalAuditConfig:
    dt_s: float = 0.5
    max_speed_mps: float = 100.0 / 3.6
    min_accel_mps2: float = -8.0
    max_accel_mps2: float = 5.0
    max_yaw_rate_rad_s: float = 1.0
    max_curvature_per_m: float = 0.25
    max_lateral_accel_mps2: float = 6.0
    max_heading_alignment_error_rad: float = 0.6
    min_forward_step_m: float = -1.0e-3
    movement_epsilon_m: float = 1.0e-3

    def __post_init__(self) -> None:
        hard = HardModeMaskConfig(
            dt_s=self.dt_s,
            max_speed_mps=self.max_speed_mps,
            min_accel_mps2=self.min_accel_mps2,
            max_accel_mps2=self.max_accel_mps2,
            max_yaw_rate_rad_s=self.max_yaw_rate_rad_s,
            max_curvature_per_m=self.max_curvature_per_m,
            max_lateral_accel_mps2=self.max_lateral_accel_mps2,
            max_heading_alignment_error_rad=(
                self.max_heading_alignment_error_rad
            ),
            min_forward_step_m=self.min_forward_step_m,
            movement_epsilon_m=self.movement_epsilon_m,
        )
        for name in (
            "dt_s",
            "max_speed_mps",
            "min_accel_mps2",
            "max_accel_mps2",
            "max_yaw_rate_rad_s",
            "max_curvature_per_m",
            "max_lateral_accel_mps2",
            "max_heading_alignment_error_rad",
            "min_forward_step_m",
            "movement_epsilon_m",
        ):
            object.__setattr__(self, name, float(getattr(hard, name)))


@dataclass(frozen=True)
class LongitudinalTrajectoryAudit:
    sample_times_s: np.ndarray
    segment_distance_m: np.ndarray
    forward_step_m: np.ndarray
    speed_mps: np.ndarray
    acceleration_mps2: np.ndarray
    yaw_rate_rad_s: np.ndarray
    curvature_per_m: np.ndarray
    lateral_acceleration_mps2: np.ndarray
    heading_alignment_error_rad: np.ndarray
    cumulative_distance_m: np.ndarray
    reachable_min_distance_m: np.ndarray
    reachable_max_distance_m: np.ndarray
    violations: tuple[str, ...]

    def __post_init__(self) -> None:
        vector_names = (
            "sample_times_s",
            "segment_distance_m",
            "forward_step_m",
            "speed_mps",
            "acceleration_mps2",
            "yaw_rate_rad_s",
            "curvature_per_m",
            "lateral_acceleration_mps2",
            "heading_alignment_error_rad",
            "cumulative_distance_m",
            "reachable_min_distance_m",
            "reachable_max_distance_m",
        )
        for name in vector_names:
            value = np.asarray(getattr(self, name))
            if value.shape != (TRAJECTORY_STEPS,) or not np.isfinite(
                value
            ).all():
                raise LongitudinalDiagnosticError(
                    f"{name} must be finite [{TRAJECTORY_STEPS}]"
                )
            object.__setattr__(
                self, name, _readonly(value, dtype=np.dtype(np.float64))
            )
        if (
            not isinstance(self.violations, tuple)
            or any(not isinstance(value, str) for value in self.violations)
            or len(set(self.violations)) != len(self.violations)
        ):
            raise LongitudinalDiagnosticError(
                "violations must be a unique tuple of strings"
            )

    @property
    def valid(self) -> bool:
        return not self.violations

    def as_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "violations": list(self.violations),
            "sample_times_s": self.sample_times_s.tolist(),
            "segment_distance_m": self.segment_distance_m.tolist(),
            "forward_step_m": self.forward_step_m.tolist(),
            "speed_mps": self.speed_mps.tolist(),
            "acceleration_mps2": self.acceleration_mps2.tolist(),
            "yaw_rate_rad_s": self.yaw_rate_rad_s.tolist(),
            "curvature_per_m": self.curvature_per_m.tolist(),
            "lateral_acceleration_mps2": (
                self.lateral_acceleration_mps2.tolist()
            ),
            "heading_alignment_error_rad": (
                self.heading_alignment_error_rad.tolist()
            ),
            "cumulative_distance_m": self.cumulative_distance_m.tolist(),
            "reachable_min_distance_m": (
                self.reachable_min_distance_m.tolist()
            ),
            "reachable_max_distance_m": (
                self.reachable_max_distance_m.tolist()
            ),
        }


def audit_longitudinal_trajectory(
    trajectory: np.ndarray,
    current_speed_mps: float,
    *,
    config: LongitudinalAuditConfig | None = None,
) -> LongitudinalTrajectoryAudit:
    """Audit one fixed-time ego-local trajectory without changing it."""

    values = np.asarray(trajectory)
    if (
        values.shape != (TRAJECTORY_STEPS, TRAJECTORY_DIM)
        or not np.issubdtype(values.dtype, np.floating)
        or not np.isfinite(values).all()
    ):
        raise LongitudinalDiagnosticError(
            "trajectory must be finite floating-point [8,3]"
        )
    if (
        isinstance(current_speed_mps, (bool, np.bool_))
        or not np.isfinite(current_speed_mps)
        or float(current_speed_mps) < 0.0
    ):
        raise LongitudinalDiagnosticError(
            "current_speed_mps must be finite and non-negative"
        )
    cfg = config or LongitudinalAuditConfig()
    hard_config = HardModeMaskConfig(
        dt_s=cfg.dt_s,
        max_speed_mps=cfg.max_speed_mps,
        min_accel_mps2=cfg.min_accel_mps2,
        max_accel_mps2=cfg.max_accel_mps2,
        max_yaw_rate_rad_s=cfg.max_yaw_rate_rad_s,
        max_curvature_per_m=cfg.max_curvature_per_m,
        max_lateral_accel_mps2=cfg.max_lateral_accel_mps2,
        max_heading_alignment_error_rad=cfg.max_heading_alignment_error_rad,
        min_forward_step_m=cfg.min_forward_step_m,
        movement_epsilon_m=cfg.movement_epsilon_m,
    )
    result = validate_trajectory_kinematics(
        values,
        float(current_speed_mps),
        np.zeros((TRAJECTORY_DIM,), dtype=np.float64),
        hard_config,
    )
    return LongitudinalTrajectoryAudit(
        sample_times_s=result.sample_times_s,
        segment_distance_m=result.segment_distance_m,
        forward_step_m=result.forward_step_m,
        speed_mps=result.speed_mps,
        acceleration_mps2=result.acceleration_mps2,
        yaw_rate_rad_s=result.yaw_rate_rad_s,
        curvature_per_m=result.curvature_per_m,
        lateral_acceleration_mps2=result.lateral_acceleration_mps2,
        heading_alignment_error_rad=result.heading_alignment_error_rad,
        cumulative_distance_m=result.cumulative_distance_m,
        reachable_min_distance_m=result.reachable_min_distance_m,
        reachable_max_distance_m=result.reachable_max_distance_m,
        violations=result.violations,
    )


def summarize_trajectory_audits(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate audits while preserving source/scenario/role/mode attribution."""

    if not rows:
        raise LongitudinalDiagnosticError("trajectory audit rows are empty")
    counts: dict[str, int] = {}
    breakdown: dict[str, dict[str, int]] = {}
    source_counts: dict[str, dict[str, int]] = {}
    source_breakdown: dict[str, dict[str, object]] = {}
    scenario_counts: dict[str, dict[str, int]] = {}
    role_counts: dict[str, dict[str, int]] = {}
    mode_counts: dict[str, dict[str, int]] = {}
    valid = 0
    for row in rows:
        audit = row.get("audit")
        if not isinstance(audit, LongitudinalTrajectoryAudit):
            raise LongitudinalDiagnosticError(
                "each trajectory audit row requires an audit result"
            )
        if audit.valid:
            valid += 1
        for reason in audit.violations:
            counts[reason] = counts.get(reason, 0) + 1
        source = str(row.get("source", "unknown"))
        source_slot = source_breakdown.setdefault(
            source,
            {
                "total": 0,
                "valid": 0,
                "violation_counts": {},
                "scenarios": {},
                "roles": {},
                "modes": {},
            },
        )
        source_slot["total"] += 1
        source_slot["valid"] += int(audit.valid)
        source_violations = source_slot["violation_counts"]
        for reason in audit.violations:
            source_violations[reason] = (
                int(source_violations.get(reason, 0)) + 1
            )
        for aggregate_name, row_name in (
            ("scenarios", "scenario"),
            ("roles", "role"),
            ("modes", "mode"),
        ):
            aggregate = source_slot[aggregate_name]
            aggregate_key = str(row.get(row_name, "unknown"))
            counts_slot = aggregate.setdefault(
                aggregate_key, {"total": 0, "valid": 0}
            )
            counts_slot["total"] += 1
            counts_slot["valid"] += int(audit.valid)
        key = "/".join(
            str(row.get(name, "unknown"))
            for name in ("source", "scenario", "role", "mode")
        )
        slot = breakdown.setdefault(key, {"total": 0, "valid": 0})
        slot["total"] += 1
        slot["valid"] += int(audit.valid)
        for collection, name in (
            (source_counts, "source"),
            (scenario_counts, "scenario"),
            (role_counts, "role"),
            (mode_counts, "mode"),
        ):
            aggregate_key = str(row.get(name, "unknown"))
            aggregate = collection.setdefault(
                aggregate_key, {"total": 0, "valid": 0}
            )
            aggregate["total"] += 1
            aggregate["valid"] += int(audit.valid)
    total = len(rows)
    return {
        "total": total,
        "valid": valid,
        "invalid": total - valid,
        "valid_rate": valid / total,
        "violation_counts": dict(sorted(counts.items())),
        "breakdown": dict(sorted(breakdown.items())),
        "sources": dict(sorted(source_counts.items())),
        "source_breakdown": {
            source: {
                **{
                    key: value
                    for key, value in slot.items()
                    if key not in {
                        "violation_counts",
                        "scenarios",
                        "roles",
                        "modes",
                    }
                },
                "violation_counts": dict(
                    sorted(slot["violation_counts"].items())
                ),
                "scenarios": dict(sorted(slot["scenarios"].items())),
                "roles": dict(sorted(slot["roles"].items())),
                "modes": dict(sorted(slot["modes"].items())),
            }
            for source, slot in sorted(source_breakdown.items())
        },
        "scenarios": dict(sorted(scenario_counts.items())),
        "roles": dict(sorted(role_counts.items())),
        "modes": dict(sorted(mode_counts.items())),
    }


@dataclass(frozen=True)
class LongitudinalTrackingReport:
    trajectory_audit: Mapping[str, object]
    clean_sample_count: int
    contaminated_sample_count: int
    longitudinal_error_p95_m: float
    longitudinal_error_p99_m: float
    target_reference_speed_delta_p95_mps: float
    maximum_continuous_saturation_s: float
    maximum_stop_terminal_speed_mps: float
    control_decomposition: Mapping[str, object]
    passed: bool
    blockers: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "trajectory_audit": dict(self.trajectory_audit),
            "clean_sample_count": self.clean_sample_count,
            "contaminated_sample_count": self.contaminated_sample_count,
            "longitudinal_error_p95_m": self.longitudinal_error_p95_m,
            "longitudinal_error_p99_m": self.longitudinal_error_p99_m,
            "target_reference_speed_delta_p95_mps": (
                self.target_reference_speed_delta_p95_mps
            ),
            "maximum_continuous_saturation_s": (
                self.maximum_continuous_saturation_s
            ),
            "maximum_stop_terminal_speed_mps": (
                self.maximum_stop_terminal_speed_mps
            ),
            "control_decomposition": dict(self.control_decomposition),
            "passed": self.passed,
            "blockers": list(self.blockers),
        }


def summarize_longitudinal_control(
    role_samples: Sequence[Mapping[str, Sequence[object]]],
) -> dict[str, object]:
    """Summarize leader/follower control effects without mixing role samples."""

    if len(role_samples) != 3:
        raise LongitudinalDiagnosticError(
            "control decomposition requires leader, middle and rear samples"
        )

    def _role(values: Mapping[str, Sequence[object]]) -> dict[str, float]:
        required = {
            "target_delta",
            "actual_acceleration",
            "formation_increment",
            "saturation",
            "clean_longitudinal_error",
            "contamination",
        }
        if set(values) != required:
            raise LongitudinalDiagnosticError(
                "control decomposition fields do not match the strict contract"
            )
        target = np.asarray(values["target_delta"], dtype=np.float64)
        acceleration = np.asarray(
            values["actual_acceleration"], dtype=np.float64
        )
        formation = np.asarray(
            values["formation_increment"], dtype=np.float64
        )
        saturated = np.asarray(values["saturation"], dtype=np.bool_)
        clean_error = np.asarray(
            values["clean_longitudinal_error"], dtype=np.float64
        )
        contamination = np.asarray(
            values["contamination"], dtype=np.bool_
        )
        if not (
            target.size
            and target.size == acceleration.size == formation.size
            and saturated.size == target.size
            and np.isfinite(target).all()
            and np.isfinite(acceleration).all()
            and np.isfinite(formation).all()
            and np.isfinite(clean_error).all()
        ):
            raise LongitudinalDiagnosticError(
                "control decomposition trace is incomplete or non-finite"
            )
        return {
            "samples": int(target.size),
            "target_reference_speed_delta_mean_mps": float(target.mean()),
            "target_reference_speed_delta_p05_mps": float(
                np.percentile(target, 5)
            ),
            "target_reference_speed_delta_p50_mps": float(
                np.percentile(target, 50)
            ),
            "target_reference_speed_delta_p95_mps": float(
                np.percentile(target, 95)
            ),
            "target_reference_speed_delta_abs_p95_mps": float(
                np.percentile(np.abs(target), 95)
            ),
            "actual_acceleration_abs_p95_mps2": float(
                np.percentile(np.abs(acceleration), 95)
            ),
            "formation_control_increment_abs_p95": float(
                np.percentile(np.abs(formation), 95)
            ),
            "control_saturation_rate": float(saturated.mean()),
            "clean_longitudinal_samples": int(clean_error.size),
            "contaminated_samples": int(contamination.sum()),
            "clean_longitudinal_error_p95_m": (
                float(np.percentile(clean_error, 95))
                if clean_error.size
                else float("inf")
            ),
            "clean_longitudinal_error_p99_m": (
                float(np.percentile(clean_error, 99))
                if clean_error.size
                else float("inf")
            ),
        }

    combined = {
        key: [
            item
            for role_values in role_samples
            for item in role_values[key]
        ]
        for key in role_samples[0]
    }
    return {
        "overall": _role(combined),
        "leader": _role(role_samples[0]),
        "middle": _role(role_samples[1]),
        "rear": _role(role_samples[2]),
    }


def build_longitudinal_tracking_report(
    branch_result: object,
    trajectories: np.ndarray,
) -> LongitudinalTrackingReport:
    candidates = np.asarray(trajectories)
    initial_speed = np.asarray(getattr(branch_result, "initial_speed_mps", ()))
    if (
        candidates.ndim != 4
        or candidates.shape[1:] != (3, TRAJECTORY_STEPS, TRAJECTORY_DIM)
        or initial_speed.shape != candidates.shape[:2]
    ):
        raise LongitudinalDiagnosticError(
            "branch trajectories and initial speeds do not match [G,3,8,3]"
        )
    rows = []
    clean_errors = []
    contaminated = 0
    target_delta = []
    saturation = []
    stop_terminal_speed = []
    role_control = [
        {
            "target_delta": [],
            "actual_acceleration": [],
            "formation_increment": [],
            "saturation": [],
            "clean_longitudinal_error": [],
            "contamination": [],
        }
        for _ in range(3)
    ]
    traces = getattr(branch_result, "tracking_traces", ())
    if len(traces) != candidates.shape[0]:
        raise LongitudinalDiagnosticError("branch tracking traces are incomplete")
    for group in range(candidates.shape[0]):
        for role in range(3):
            audit = audit_longitudinal_trajectory(
                candidates[group, role], float(initial_speed[group, role])
            )
            rows.append(
                {
                    "source": "branch",
                    "scenario": "unknown",
                    "role": role,
                    "mode": "selected",
                    "audit": audit,
                }
            )
            trace = traces[group][role]
            errors = np.abs(
                np.asarray(trace["longitudinal_errors_m"], dtype=np.float64)
            )
            mask = ~np.asarray(
                trace["lateral_heading_contaminated"], dtype=bool
            )
            if errors.shape != mask.shape:
                raise LongitudinalDiagnosticError(
                    "branch contamination mask does not match tracking errors"
                )
            clean_errors.extend(errors[mask].tolist())
            contaminated += int((~mask).sum())
            target_delta.extend(
                np.asarray(
                    trace["position_error_speed_increment_mps"],
                    dtype=np.float64,
                ).tolist()
            )
            role_control[role]["target_delta"].extend(
                np.asarray(
                    trace["position_error_speed_increment_mps"],
                    dtype=np.float64,
                ).tolist()
            )
            role_control[role]["actual_acceleration"].extend(
                np.asarray(
                    trace["actual_acceleration_mps2"], dtype=np.float64
                ).tolist()
            )
            role_control[role]["formation_increment"].extend(
                np.asarray(
                    trace["formation_control_increment"], dtype=np.float64
                ).tolist()
            )
            role_control[role]["saturation"].extend(
                np.asarray(trace["control_saturated"], dtype=bool).tolist()
            )
            role_control[role]["clean_longitudinal_error"].extend(
                errors[mask].tolist()
            )
            role_control[role]["contamination"].extend((~mask).tolist())
            saturation.append(
                float(trace["maximum_continuous_saturation_s"])
            )
            if float(audit.speed_mps[-1]) <= 0.3:
                actual = np.asarray(
                    trace["actual_speed_mps"], dtype=np.float64
                )
                if actual.size:
                    stop_terminal_speed.append(float(actual[-1]))
    summary = summarize_trajectory_audits(rows)
    clean = np.asarray(clean_errors, dtype=np.float64)
    if clean.size:
        p95 = float(np.percentile(clean, 95))
        p99 = float(np.percentile(clean, 99))
    else:
        p95 = p99 = float("inf")
    speed_delta_p95 = (
        float(np.percentile(np.abs(np.asarray(target_delta)), 95))
        if target_delta
        else float("inf")
    )
    maximum_saturation = max(saturation, default=float("inf"))
    maximum_stop_speed = max(stop_terminal_speed, default=0.0)
    blockers = []
    if summary["invalid"]:
        blockers.append("invalid_trajectory")
    if clean.size == 0:
        blockers.append("no_uncontaminated_longitudinal_samples")
    if p95 > 1.0:
        blockers.append("longitudinal_p95")
    if p99 > 1.5:
        blockers.append("longitudinal_p99")
    if maximum_saturation > 1.0:
        blockers.append("continuous_control_saturation")
    if maximum_stop_speed > 0.3:
        blockers.append("stop_terminal_speed")

    control_decomposition = summarize_longitudinal_control(role_control)
    return LongitudinalTrackingReport(
        trajectory_audit=summary,
        clean_sample_count=int(clean.size),
        contaminated_sample_count=contaminated,
        longitudinal_error_p95_m=p95,
        longitudinal_error_p99_m=p99,
        target_reference_speed_delta_p95_mps=speed_delta_p95,
        maximum_continuous_saturation_s=maximum_saturation,
        maximum_stop_terminal_speed_mps=maximum_stop_speed,
        control_decomposition=control_decomposition,
        passed=not blockers,
        blockers=tuple(blockers),
    )


def audit_live_normal_planner(
    scenarios: Sequence[tuple[str, str]],
    seeds: Sequence[int],
    *,
    states_per_episode: int = 3,
    maximum_steps: int = 200,
) -> dict[str, object]:
    """Audit current native RuleMaker+Normal-planner output in live scenarios."""

    if states_per_episode <= 0 or maximum_steps <= 0:
        raise LongitudinalDiagnosticError(
            "states_per_episode and maximum_steps must be positive"
        )
    # Imports remain local because the online GRPO module imports this
    # diagnostics module.  The dependency is only needed by this executable
    # audit path and does not affect training/model imports.
    from expert_dataset.collect_joint_bev import (
        JointCollectionError,
        RulePlannerExpert,
        simulator_decision_dt_s,
    )
    from models.controller.PIDController import (
        _world_trajectory_to_ego_local,
    )
    from train.train_bev_joint_grpo_online import (
        _new_env,
        episode_has_ended,
    )

    agent_ids = ("agent0", "agent1", "agent2")
    rows = []
    diagnostic_rows = []
    episodes = []
    for scenario in scenarios:
        if len(scenario) != 2:
            raise LongitudinalDiagnosticError(
                "each scenario must be a (scenario_id, route) pair"
            )
        for seed in seeds:
            env = _new_env((str(scenario[0]), str(scenario[1])), int(seed))
            expert = RulePlannerExpert(env, agent_ids)
            dt_s = simulator_decision_dt_s(env)
            collected = 0
            failure = None
            try:
                for state_index in range(maximum_steps):
                    try:
                        step = expert.plan(env)
                    except JointCollectionError as exc:
                        failure = f"{exc.reason_code}:{exc}"
                        break
                    controls = {}
                    for role, agent_id in enumerate(agent_ids):
                        vehicle = env.agents[agent_id]
                        local = _world_trajectory_to_ego_local(
                            vehicle, step.trajectories_world[agent_id]
                        )
                        speed = float(
                            getattr(vehicle, "speed_km_h", 0.0)
                        ) / 3.6
                        audit = audit_longitudinal_trajectory(local, speed)
                        row = {
                            "source": "normal_planner_native",
                            "scenario": scenario[0],
                            "role": role,
                            "mode": int(step.rule_actions[agent_id]),
                            "audit": audit,
                        }
                        rows.append(row)
                        diagnostic_rows.append(
                            {
                                "scenario": str(scenario[0]),
                                "route": str(scenario[1]),
                                "seed": int(seed),
                                "state_index": state_index,
                                "role": role,
                                "mode": int(step.rule_actions[agent_id]),
                                "current_speed_mps": speed,
                                "audit": audit.as_dict(),
                                "trajectory": (
                                    local.tolist() if not audit.valid else None
                                ),
                            }
                        )
                        controls[agent_id] = np.array(
                            step.controls[agent_id],
                            dtype=np.float32,
                            copy=True,
                        )
                    collected += 1
                    if collected >= states_per_episode:
                        break
                    _, _, terminated, truncated, info = env.step(controls)
                    if episode_has_ended(terminated, truncated, info):
                        failure = "episode_ended"
                        break
                if collected < states_per_episode and failure is None:
                    failure = "insufficient_native_planner_states"
            finally:
                env.close()
            episodes.append(
                {
                    "scenario": str(scenario[0]),
                    "route": str(scenario[1]),
                    "seed": int(seed),
                    "states": collected,
                    "failure": failure,
                    "duration_s": collected * dt_s,
                }
            )
    summary = summarize_trajectory_audits(rows) if rows else {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "valid_rate": 0.0,
        "violation_counts": {},
        "breakdown": {},
    }
    return {
        "trajectory_audit": summary,
        "episodes": episodes,
        "diagnostic_rows": diagnostic_rows,
        "passed": (
            bool(rows)
            and summary["invalid"] == 0
            and all(
                item["states"] == states_per_episode
                for item in episodes
            )
        ),
    }


def run_longitudinal_tracking_benchmark(
    episode_spec: object,
    prefix_actions: Sequence[Mapping[str, np.ndarray]],
    trajectories: np.ndarray,
    *,
    evaluator: object | None = None,
) -> LongitudinalTrackingReport:
    """Run the existing branch controller and diagnose its longitudinal trace."""

    if evaluator is None:
        from evaluation.joint_simulator_branch import (
            JointSimulatorBranchEvaluator,
        )

        evaluator = JointSimulatorBranchEvaluator()
    evaluate = getattr(evaluator, "evaluate", None)
    if not callable(evaluate):
        raise LongitudinalDiagnosticError(
            "evaluator must provide evaluate(spec, prefix, trajectories)"
        )
    result = evaluate(episode_spec, prefix_actions, trajectories)
    return build_longitudinal_tracking_report(result, trajectories)


def audit_joint_dataset(dataset_root: Path | str) -> dict[str, object]:
    """Audit every expert trajectory and every hard-valid dynamic anchor."""

    from expert_dataset.joint_bev_dataset import (
        JointBEVDataset,
        JointBEVDatasetConfig,
    )

    rows = []
    for split in ("train", "val", "test"):
        dataset = JointBEVDataset(
            JointBEVDatasetConfig(dataset_root=dataset_root, split=split)
        )
        try:
            for sample_index in range(len(dataset)):
                sample = dataset[sample_index]
                ego_speed = sample["ego_state"][:, 0].numpy()
                expert = sample["expert_trajectory"].numpy()
                anchors = sample["coarse_trajectories"].numpy()
                valid_mask = sample["mode_valid_mask"].numpy()
                gt_mode = sample["gt_mode"].numpy()
                for role in range(3):
                    rows.append(
                        {
                            "source": "expert",
                            "scenario": split,
                            "role": role,
                            "mode": int(gt_mode[role]),
                            "audit": audit_longitudinal_trajectory(
                                expert[role], float(ego_speed[role])
                            ),
                        }
                    )
                    for mode in np.flatnonzero(valid_mask[role]):
                        rows.append(
                            {
                                "source": "hard_valid_anchor",
                                "scenario": split,
                                "role": role,
                                "mode": int(mode),
                                "audit": audit_longitudinal_trajectory(
                                    anchors[role, mode],
                                    float(ego_speed[role]),
                                ),
                            }
                        )
        finally:
            dataset.close()
    report = summarize_trajectory_audits(rows)
    source_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        source = str(row["source"])
        audit = row["audit"]
        slot = source_counts.setdefault(source, {"total": 0, "valid": 0})
        slot["total"] += 1
        slot["valid"] += int(audit.valid)
    report["sources"] = source_counts
    report["passed"] = all(
        value["valid"] == value["total"] for value in source_counts.values()
    )
    return report


def _main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit expert trajectories and hard-valid anchors in a packed "
            "joint BEV dataset"
        )
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--normal-planner-s5-s9",
        action="store_true",
        help="also audit live native Normal-planner output on S5--S9",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.dataset_root is None and not arguments.normal_planner_s5_s9:
        parser.error(
            "at least one of --dataset-root or --normal-planner-s5-s9 is required"
        )
    dataset_report = (
        audit_joint_dataset(arguments.dataset_root)
        if arguments.dataset_root is not None
        else None
    )
    normal_report = None
    if arguments.normal_planner_s5_s9:
        from scenarios.bev_round13_contract import (
            DEVELOPMENT_SEEDS,
            PRIMARY_S5_S9_SCENARIOS,
        )

        normal_report = audit_live_normal_planner(
            PRIMARY_S5_S9_SCENARIOS,
            DEVELOPMENT_SEEDS,
        )
    report = {
        "format": "bev_longitudinal_dataset_audit_v1",
        "diagnostic_only": True,
        "dataset_root": (
            str(arguments.dataset_root.resolve())
            if arguments.dataset_root is not None
            else None
        ),
        "dataset_report": dataset_report,
        "normal_planner_report": normal_report,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


__all__ = [
    "LongitudinalAuditConfig",
    "LongitudinalDiagnosticError",
    "LongitudinalTrackingReport",
    "LongitudinalTrajectoryAudit",
    "audit_joint_dataset",
    "audit_live_normal_planner",
    "audit_longitudinal_trajectory",
    "build_longitudinal_tracking_report",
    "run_longitudinal_tracking_benchmark",
    "summarize_longitudinal_control",
    "summarize_trajectory_audits",
]


if __name__ == "__main__":
    raise SystemExit(_main())
