"""Fair closed-loop evaluation for one or more Stage 1 and GRPO models."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
from scipy.stats import beta as beta_distribution

# Must be set before the first CUDA context is created.  The evaluator uses
# fixed diffusion noise, so allowing non-deterministic GEMM kernels would make
# that fairness contract incomplete near mode-logit ties.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from expert_dataset.collect_joint_bev import (
    JointBEVSampleBuilder,
    SensorlessJointBEVPlatoonEnv,
    simulator_decision_dt_s,
)
from evaluation.bev_model_manifest import (
    ComparisonSpec,
    ModelSpec,
    file_sha256,
    load_model_manifest,
)
from evaluation.bev_evaluation_artifacts import ClosedLoopArtifactWriter
from models.decisioner.rule_decisioner import (
    LaneChangeCommitmentError,
    diffusion_mode_feedback_actions,
    hard_valid_modes_by_rule_action,
    joint_proposal_actions,
    make_rule_maker,
    match_joint_action_proposal,
)
from train.bev_joint_grpo import (
    load_grpo_b_checkpoint,
    load_grpo_checkpoint,
    load_stage1_a_for_grpo,
    load_stage1_b_for_grpo,
)
from train.train_bev_diffusion_stage1 import planner_forward_from_batch
from train.train_bev_joint_grpo_online import (
    AGENT_IDS,
    constant_velocity_actions,
    execution_mode_valid_mask,
    episode_has_ended,
    joint_trajectory_action,
    model_inputs_to_batch,
    optimize_selected_model_trajectories,
    optimize_safe_stop_trajectories,
)
from models.bev_planner.joint_reward import (
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256,
    JOINT_REWARD_CONTRACT,
    JOINT_REWARD_CONTRACT_SHA256,
    JointRewardConfig,
    JointRewardError,
    joint_reward_config_sha256,
)
from models.bev_planner.trajectory_optimizer import (
    KinematicTrajectoryOptimizer,
    KinematicTrajectoryOptimizerConfig,
    TrajectoryOptimizationError,
)
from models.platoon_planner.collision_geometry import shared_corridor_gap_series
from scenarios.bev_round13_contract import (
    BEVScenarioContractError,
    HOLDOUT_SEEDS,
    PRIMARY_S5_S9_SCENARIOS,
    primary_scenario_contract,
    deterministic_initial_speed_km_h,
)


DIAGNOSTIC_EVAL_SCENARIOS = PRIMARY_S5_S9_SCENARIOS
FORMAL_EVAL_SCENARIOS = PRIMARY_S5_S9_SCENARIOS
FORMAL_EVAL_SEEDS = (17, 23, 31, 47, 59)
BACKGROUND_GAP_THRESHOLD_M = 5.0
PLATOON_GAP_THRESHOLD_M = 7.0
TTC_OBSERVATION_THRESHOLD_S = 1.5
NO_RISK_GAP_M = 1.0e6
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260827
GATE_FAILURE_EXIT_CODE = 2


class ModelEvaluationError(RuntimeError):
    """Raised when a model evaluation is incomplete or unfair."""


@dataclass(frozen=True)
class ModelEvaluationConfig:
    run_mode: Literal["diagnostic", "formal"] = "diagnostic"
    device: str = "cuda"
    seeds: tuple[int, ...] = HOLDOUT_SEEDS
    scenarios: tuple[tuple[str, str], ...] = DIAGNOSTIC_EVAL_SCENARIOS
    max_steps: int = 100
    v2_planning_tick_p95_limit_ms: float = 200.0
    artifact_root: Path | None = None
    save_visualizations: bool = False
    video_fps: int = 10
    visualization_interval: int = 1
    topdown_screen_size: int = 800
    topdown_film_size: int = 3000

    def __post_init__(self) -> None:
        if self.run_mode not in ("diagnostic", "formal"):
            raise ModelEvaluationError(
                "evaluation run_mode must be diagnostic or formal"
            )
        if self.device not in ("cpu", "cuda"):
            raise ModelEvaluationError("evaluation device must be cpu or cuda")
        if not self.seeds or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in self.seeds
        ):
            raise ModelEvaluationError("evaluation seeds must be integers")
        if not self.scenarios:
            raise ModelEvaluationError("evaluation scenarios cannot be empty")
        try:
            primary_scenario_contract(self.scenarios)
        except BEVScenarioContractError as exc:
            raise ModelEvaluationError(str(exc)) from exc
        if isinstance(self.max_steps, bool) or self.max_steps <= 0:
            raise ModelEvaluationError("max_steps must be positive")
        if (
            not math.isfinite(self.v2_planning_tick_p95_limit_ms)
            or self.v2_planning_tick_p95_limit_ms <= 0.0
        ):
            raise ModelEvaluationError(
                "v2_planning_tick_p95_limit_ms must be positive and finite"
            )
        if self.save_visualizations and self.artifact_root is None:
            raise ModelEvaluationError("save_visualizations requires an artifact_root")
        if isinstance(self.video_fps, bool) or self.video_fps <= 0:
            raise ModelEvaluationError("video_fps must be positive")
        if (
            isinstance(self.visualization_interval, bool)
            or self.visualization_interval <= 0
        ):
            raise ModelEvaluationError("visualization_interval must be positive")
        if isinstance(self.topdown_screen_size, bool) or self.topdown_screen_size <= 0:
            raise ModelEvaluationError("topdown_screen_size must be positive")
        if isinstance(self.topdown_film_size, bool) or self.topdown_film_size <= 0:
            raise ModelEvaluationError("topdown_film_size must be positive")


@dataclass(frozen=True)
class ReproducibilityToleranceConfig:
    """Engineering tolerances for closed-loop repeat comparisons.

    These tolerances do not relax collision, out-of-road or execution-rejection
    outcomes.  They only distinguish harmless floating-point/physics drift from
    a changed experimental conclusion.
    """

    distance_m: float = 0.10
    speed_km_h: float = 0.25
    rate: float = 0.10
    reward: float = 0.10
    comfort: float = 0.10
    fraction: float = 0.02

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value < 0.0:
                raise ModelEvaluationError(
                    f"reproducibility tolerance {field.name} must be finite and non-negative"
                )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _configure_deterministic_inference(device: torch.device) -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ModelEvaluationError(
            "model evaluation requires PYTHONHASHSEED=0 at process startup"
        )
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def _nullable_percentile(values: Sequence[float], q: float) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.percentile(finite, q)) if finite else None


def _exact_binomial_ci(
    event_count: int, sample_count: int, *, confidence: float = 0.95
) -> list[float]:
    """Return the two-sided Clopper--Pearson interval for one event rate."""

    events = int(event_count)
    samples = int(sample_count)
    if samples <= 0 or events < 0 or events > samples:
        raise ModelEvaluationError("binomial counts are invalid")
    alpha = 1.0 - float(confidence)
    lower = (
        0.0
        if events == 0
        else float(beta_distribution.ppf(alpha / 2.0, events, samples - events + 1))
    )
    upper = (
        1.0
        if events == samples
        else float(
            beta_distribution.ppf(
                1.0 - alpha / 2.0, events + 1, samples - events
            )
        )
    )
    return [lower, upper]


def _vehicle_dimensions(vehicle: object) -> tuple[float, float]:
    if isinstance(vehicle, Mapping):
        length = float(vehicle.get("length_m", 5.74))
        width = float(vehicle.get("width_m", 2.3))
    else:
        length = float(getattr(vehicle, "LENGTH", 5.74))
        width = float(getattr(vehicle, "WIDTH", 2.3))
    if not math.isfinite(length) or length <= 0.0 or not math.isfinite(width) or width <= 0.0:
        raise ModelEvaluationError("vehicle dimensions must be positive and finite")
    return length, width


def _vehicle_pose(vehicle: object) -> np.ndarray:
    if isinstance(vehicle, Mapping):
        position_value = vehicle.get("position_xy_m")
        heading_value = vehicle.get("heading_rad")
    else:
        position_value = getattr(vehicle, "position")
        heading_value = getattr(vehicle, "heading_theta")
    position = np.asarray(position_value, dtype=np.float64).reshape(-1)
    if position.size < 2:
        raise ModelEvaluationError("vehicle position must contain x and y")
    pose = np.asarray(
        [position[0], position[1], float(heading_value)],
        dtype=np.float64,
    )
    if not np.isfinite(pose).all():
        raise ModelEvaluationError("vehicle pose must be finite")
    return pose


def _corridor_gap(first: object, second: object) -> float | None:
    gap = float(
        shared_corridor_gap_series(
            _vehicle_pose(first)[None, :],
            _vehicle_dimensions(first),
            _vehicle_pose(second)[None, :],
            _vehicle_dimensions(second),
            no_risk_gap_m=NO_RISK_GAP_M,
        )[0]
    )
    return None if gap >= NO_RISK_GAP_M else gap


def _live_gap_observations(
    env: object,
    *,
    agent_states: Mapping[str, object] | None = None,
) -> tuple[dict[tuple[str, int], float], dict[tuple[str, str], float]]:
    """Return same-corridor OBB gaps for background and adjacent platoon pairs."""

    live_agents = dict(getattr(env, "agents", {}) or {})
    agents = dict(agent_states) if agent_states is not None else live_agents
    platoon_objects = {id(value) for value in live_agents.values()}
    background: dict[tuple[str, int], float] = {}
    for other in _surrounding_vehicles(env):
        if id(other) in platoon_objects:
            continue
        for agent_id in AGENT_IDS:
            agent = agents.get(agent_id)
            if agent is None:
                continue
            gap = _corridor_gap(agent, other)
            if gap is not None:
                background[(agent_id, id(other))] = float(gap)
    platoon: dict[tuple[str, str], float] = {}
    for first_id, second_id in zip(AGENT_IDS, AGENT_IDS[1:]):
        first = agents.get(first_id)
        second = agents.get(second_id)
        if first is None or second is None:
            continue
        gap = _corridor_gap(first, second)
        if gap is not None:
            platoon[(first_id, second_id)] = float(gap)
    return background, platoon


def _closing_risk_observations(
    current: Mapping[object, float],
    previous: Mapping[object, float],
    *,
    dt_s: float,
) -> tuple[list[float], list[float]]:
    """Return TTC and DRAC for pairs that remain in-corridor and are closing."""

    ttc_values: list[float] = []
    drac_values: list[float] = []
    for key in current.keys() & previous.keys():
        gap = float(current[key])
        closing_speed = (float(previous[key]) - gap) / dt_s
        if closing_speed <= 0.0:
            continue
        effective_gap = max(gap, 0.0)
        ttc_values.append(effective_gap / closing_speed)
        drac_values.append((closing_speed * closing_speed) / (2.0 * max(effective_gap, 0.1)))
    return ttc_values, drac_values


def _navigation_snapshot(env: object) -> dict[str, tuple[float, float]]:
    snapshots = {}
    for agent_id in AGENT_IDS:
        vehicle = (getattr(env, "agents", {}) or {}).get(agent_id)
        navigation = getattr(vehicle, "navigation", None)
        travelled = getattr(navigation, "travelled_length", None)
        total = getattr(navigation, "total_length", None)
        if (
            isinstance(travelled, bool)
            or not isinstance(travelled, (int, float))
            or isinstance(total, bool)
            or not isinstance(total, (int, float))
            or not math.isfinite(float(travelled))
            or not math.isfinite(float(total))
            or float(total) <= 0.0
        ):
            raise ModelEvaluationError(
                f"navigation progress is unavailable for {agent_id}"
            )
        snapshots[agent_id] = (float(travelled), float(total))
    return snapshots


def _post_step_agent_states(
    env: object, step_info: Mapping[str, object]
) -> dict[str, object]:
    """Resolve the true post-step state, including agents removed on termination."""

    live_agents = dict(getattr(env, "agents", {}) or {})
    resolved: dict[str, object] = {}
    required_snapshot_fields = {
        "navigation_travelled_length_m",
        "navigation_total_length_m",
        "position_xy_m",
        "heading_rad",
        "speed_km_h",
        "length_m",
        "width_m",
    }
    for agent_id in AGENT_IDS:
        vehicle = live_agents.get(agent_id)
        if vehicle is not None:
            resolved[agent_id] = vehicle
            continue
        agent_info = step_info.get(agent_id)
        snapshot = (
            agent_info.get("evaluation_metric_snapshot")
            if isinstance(agent_info, Mapping)
            else None
        )
        if not isinstance(snapshot, Mapping) or not required_snapshot_fields.issubset(
            snapshot
        ):
            raise ModelEvaluationError(
                f"terminal evaluation metric snapshot is unavailable for {agent_id}"
            )
        # Validate geometry here. Navigation is validated by the cache updater.
        _vehicle_pose(snapshot)
        _vehicle_dimensions(snapshot)
        resolved[agent_id] = snapshot
    return resolved


def _scenario_summary_from_step_info(
    step_info: Mapping[str, object],
) -> Mapping[str, object]:
    """Read the one summary already attached by PlatoonEnv._build_info_dict."""

    for agent_id in AGENT_IDS:
        agent_info = step_info.get(agent_id)
        if (
            isinstance(agent_info, Mapping)
            and "scenario_realized" in agent_info
            and "functional_success" in agent_info
        ):
            return agent_info
    raise ModelEvaluationError("step info has no S5--S9 scenario summary")


def _absolute_distribution(
    values: Sequence[float], stem: str, unit: str
) -> dict[str, float | None]:
    finite = np.abs(np.asarray([float(value) for value in values], dtype=np.float64))
    if finite.size == 0:
        return {
            f"{stem}_abs_mean_{unit}": None,
            f"{stem}_abs_p95_{unit}": None,
            f"{stem}_abs_max_{unit}": None,
        }
    return {
        f"{stem}_abs_mean_{unit}": float(np.mean(finite)),
        f"{stem}_abs_p95_{unit}": float(np.percentile(finite, 95)),
        f"{stem}_abs_max_{unit}": float(np.max(finite)),
    }


def _timing_distribution(values: Sequence[float]) -> dict[str, float]:
    finite = [float(value) for value in values]
    return {
        "p50_ms": _percentile(finite, 50),
        "p95_ms": _percentile(finite, 95),
        "p99_ms": _percentile(finite, 99),
        "max_ms": max(finite, default=0.0),
    }


def _closed_loop_metric_policy() -> dict[str, dict[str, object]]:
    """Frozen v3 comparison metrics, addressed from one episode row."""

    policy: dict[str, dict[str, object]] = {
        "safety.collision_rate": {
            "episode_path": ("safety", "collision"),
            "unit": "fraction",
            "direction": "lower",
        },
        "safety.out_of_road_rate": {
            "episode_path": ("safety", "out_of_road"),
            "unit": "fraction",
            "direction": "lower",
        },
        "safety.background_gap_violation_rate": {
            "episode_path": ("safety", "background_gap_violation"),
            "unit": "fraction",
            "direction": "lower",
        },
        "safety.background_gap_exposure_fraction_mean": {
            "episode_path": ("safety", "background_gap_exposure_fraction"),
            "unit": "fraction",
            "direction": "lower",
        },
        "safety.background_gap_deficit_integral_m_s_mean": {
            "episode_path": ("safety", "background_gap_deficit_integral_m_s"),
            "unit": "m*s",
            "direction": "lower",
        },
        "safety.minimum_background_gap_m_mean": {
            "episode_path": ("safety", "minimum_background_gap_m"),
            "unit": "m",
            "direction": "higher",
        },
        "safety.platoon_gap_violation_rate": {
            "episode_path": ("safety", "platoon_gap_violation"),
            "unit": "fraction",
            "direction": "lower",
        },
        "safety.platoon_gap_exposure_fraction_mean": {
            "episode_path": ("safety", "platoon_gap_exposure_fraction"),
            "unit": "fraction",
            "direction": "lower",
        },
        "safety.platoon_gap_deficit_integral_m_s_mean": {
            "episode_path": ("safety", "platoon_gap_deficit_integral_m_s"),
            "unit": "m*s",
            "direction": "lower",
        },
        "safety.minimum_platoon_gap_m_mean": {
            "episode_path": ("safety", "minimum_platoon_gap_m"),
            "unit": "m",
            "direction": "higher",
        },
        "safety.minimum_ttc_s_mean": {
            "episode_path": ("safety", "minimum_ttc_s"),
            "unit": "s",
            "direction": "higher",
        },
        "safety.ttc_below_1_5_s_exposure_fraction_mean": {
            "episode_path": ("safety", "ttc_below_1_5_s_exposure_fraction"),
            "unit": "fraction",
            "direction": "lower",
        },
        "safety.max_drac_mps2_mean": {
            "episode_path": ("safety", "max_drac_mps2"),
            "unit": "m/s^2",
            "direction": "lower",
        },
        "safety.execution_rejection_episode_rate": {
            "episode_path": ("safety", "execution_rejected"),
            "unit": "fraction",
            "direction": "lower",
        },
        "safety.execution_rejection_tick_fraction_mean": {
            "episode_path": ("safety", "execution_rejection_tick_fraction"),
            "unit": "fraction",
            "direction": "lower",
        },
        "efficiency.scenario_realized_rate": {
            "episode_path": ("efficiency", "scenario_realized"),
            "unit": "fraction",
            "direction": "higher",
        },
        "efficiency.functional_success_final_rate": {
            "episode_path": ("efficiency", "functional_success_final"),
            "unit": "fraction",
            "direction": "higher",
        },
        "efficiency.functional_success_ever_rate": {
            "episode_path": ("efficiency", "functional_success_ever"),
            "unit": "fraction",
            "direction": "higher",
        },
        "efficiency.functional_success_stable_rate": {
            "episode_path": ("efficiency", "functional_success_stable"),
            "unit": "fraction",
            "direction": "higher",
        },
        "efficiency.first_stable_success_time_s_mean": {
            "episode_path": ("efficiency", "first_stable_success_time_s"),
            "unit": "s",
            "direction": "lower",
        },
        "efficiency.team_mean_route_progress_m_mean": {
            "episode_path": ("efficiency", "team_mean_route_progress_m"),
            "unit": "m",
            "direction": "higher",
        },
        "efficiency.team_min_route_progress_m_mean": {
            "episode_path": ("efficiency", "team_min_route_progress_m"),
            "unit": "m",
            "direction": "higher",
        },
        "efficiency.team_mean_route_progress_fraction_mean": {
            "episode_path": ("efficiency", "team_mean_route_progress_fraction"),
            "unit": "fraction",
            "direction": "higher",
        },
        "efficiency.team_min_route_progress_fraction_mean": {
            "episode_path": ("efficiency", "team_min_route_progress_fraction"),
            "unit": "fraction",
            "direction": "higher",
        },
        "efficiency.completion_time_s_mean": {
            "episode_path": ("efficiency", "completion_time_s"),
            "unit": "s",
            "direction": "lower",
        },
        "efficiency.episode_duration_s_mean": {
            "episode_path": ("efficiency", "episode_duration_s"),
            "unit": "s",
            "direction": "descriptive",
        },
        "cooperation.locked_spacing_error_mean_m": {
            "episode_path": ("cooperation", "locked_spacing_error_mean_m"),
            "unit": "m",
            "direction": "lower",
        },
        "cooperation.locked_spacing_error_p95_m_mean": {
            "episode_path": ("cooperation", "locked_spacing_error_p95_m"),
            "unit": "m",
            "direction": "lower",
        },
        "cooperation.locked_spacing_error_max_m_mean": {
            "episode_path": ("cooperation", "locked_spacing_error_max_m"),
            "unit": "m",
            "direction": "lower",
        },
        "cooperation.locked_speed_spread_mean_mps": {
            "episode_path": ("cooperation", "locked_speed_spread_mean_mps"),
            "unit": "m/s",
            "direction": "lower",
        },
        "cooperation.unlock_count_mean": {
            "episode_path": ("cooperation", "unlock_count"),
            "unit": "count",
            "direction": "lower",
        },
        "cooperation.unlocked_duration_s_mean": {
            "episode_path": ("cooperation", "unlocked_duration_s"),
            "unit": "s",
            "direction": "lower",
        },
        "cooperation.relock_recovery_success_rate": {
            "episode_path": ("cooperation", "relock_recovery_success"),
            "unit": "fraction",
            "direction": "higher",
        },
        "cooperation.recovery_censored_rate": {
            "episode_path": ("cooperation", "recovery_censored"),
            "unit": "fraction",
            "direction": "lower",
        },
        "cooperation.relock_recovery_time_s_mean": {
            "episode_path": ("cooperation", "relock_recovery_time_s"),
            "unit": "s",
            "direction": "lower",
        },
    }
    for agent_id in AGENT_IDS:
        policy[f"safety.per_agent.{agent_id}.collision_rate"] = {
            "episode_path": ("safety", "per_agent", agent_id, "collision"),
            "unit": "fraction",
            "direction": "lower",
        }
        policy[f"safety.per_agent.{agent_id}.out_of_road_rate"] = {
            "episode_path": ("safety", "per_agent", agent_id, "out_of_road"),
            "unit": "fraction",
            "direction": "lower",
        }
        policy[f"efficiency.per_agent.{agent_id}.route_progress_m_mean"] = {
            "episode_path": (
                "efficiency",
                "per_agent",
                agent_id,
                "route_progress_m",
            ),
            "unit": "m",
            "direction": "higher",
        }
        policy[
            f"efficiency.per_agent.{agent_id}.route_progress_fraction_mean"
        ] = {
            "episode_path": (
                "efficiency",
                "per_agent",
                agent_id,
                "route_progress_fraction",
            ),
            "unit": "fraction",
            "direction": "higher",
        }
    comfort_fields = {
        "longitudinal_acceleration_abs_mean_mps2": "m/s^2",
        "longitudinal_acceleration_abs_p95_mps2": "m/s^2",
        "longitudinal_acceleration_abs_max_mps2": "m/s^2",
        "lateral_acceleration_abs_mean_mps2": "m/s^2",
        "lateral_acceleration_abs_p95_mps2": "m/s^2",
        "lateral_acceleration_abs_max_mps2": "m/s^2",
        "longitudinal_jerk_abs_mean_mps3": "m/s^3",
        "longitudinal_jerk_abs_p95_mps3": "m/s^3",
        "longitudinal_jerk_abs_max_mps3": "m/s^3",
        "yaw_rate_abs_mean_rad_s": "rad/s",
        "yaw_rate_abs_p95_rad_s": "rad/s",
        "yaw_rate_abs_max_rad_s": "rad/s",
        "yaw_acceleration_abs_mean_rad_s2": "rad/s^2",
        "yaw_acceleration_abs_p95_rad_s2": "rad/s^2",
        "yaw_acceleration_abs_max_rad_s2": "rad/s^2",
        "steering_command_slew_abs_mean_per_s": "command/s",
        "steering_command_slew_abs_p95_per_s": "command/s",
        "steering_command_slew_abs_max_per_s": "command/s",
        "throttle_command_slew_abs_mean_per_s": "command/s",
        "throttle_command_slew_abs_p95_per_s": "command/s",
        "throttle_command_slew_abs_max_per_s": "command/s",
    }
    for agent_id in AGENT_IDS:
        for field, unit in comfort_fields.items():
            policy[f"comfort.per_agent.{agent_id}.{field}"] = {
                "episode_path": ("comfort", "per_agent", agent_id, field),
                "unit": unit,
                "direction": "lower",
            }
    for component in (
        "bev_build_ms",
        "model_inference_ms",
        "control_mapping_ms",
        "planning_tick_ms",
        "trajectory_optimizer_ms",
        "rule_maker_ms",
    ):
        for percentile in ("p50_ms", "p95_ms"):
            policy[f"timing.{component}_{percentile}_mean"] = {
                "episode_path": ("timing", component, percentile),
                "unit": "ms",
                "direction": "lower",
            }
    for field, unit in (
        ("p99_ms", "ms"),
        ("max_ms", "ms"),
        ("over_200_ms_fraction", "fraction"),
    ):
        policy[f"timing.planning_tick_ms_{field}_mean"] = {
            "episode_path": ("timing", "planning_tick_ms", field),
            "unit": unit,
            "direction": "lower",
        }
    return policy


def _metric_definitions() -> dict[str, dict[str, object]]:
    return {
        name: {
            "unit": str(spec["unit"]),
            "direction": str(spec["direction"]),
            "statistical_unit": "episode",
            "episode_source_path": list(spec["episode_path"]),
        }
        for name, spec in _closed_loop_metric_policy().items()
    }


def _episode_value(
    episode: Mapping[str, object], path: Sequence[str]
) -> bool | float | int | None:
    current: object = episode
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise ModelEvaluationError(
                f"episode metric is missing {'.'.join(path)}"
            )
        current = current[key]
    if current is None or isinstance(current, bool):
        return current
    if not isinstance(current, (int, float)) or not math.isfinite(float(current)):
        raise ModelEvaluationError(
            f"episode metric {'.'.join(path)} must be finite numeric, boolean, or null"
        )
    return current


def _mean_episode_path(
    episodes: Sequence[Mapping[str, object]], path: Sequence[str]
) -> float | None:
    values = [
        float(value)
        for episode in episodes
        if (value := _episode_value(episode, path)) is not None
    ]
    return float(np.mean(values)) if values else None


def _rate_summary(
    episodes: Sequence[Mapping[str, object]],
    path: Sequence[str],
    *,
    stem: str,
) -> dict[str, object]:
    values = [
        bool(value)
        for episode in episodes
        if (value := _episode_value(episode, path)) is not None
    ]
    events = sum(values)
    samples = len(values)
    if samples == 0:
        return {
            f"{stem}_count": 0,
            f"{stem}_sample_count": 0,
            f"{stem}_rate": None,
            f"{stem}_rate_ci95": None,
        }
    return {
        f"{stem}_count": int(events),
        f"{stem}_sample_count": int(samples),
        f"{stem}_rate": float(events / samples),
        f"{stem}_rate_ci95": _exact_binomial_ci(events, samples),
    }


def _aggregate_episode_metrics(
    episodes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not episodes:
        raise ModelEvaluationError("cannot aggregate an empty episode set")
    safety: dict[str, object] = {}
    safety.update(_rate_summary(episodes, ("safety", "collision"), stem="collision"))
    safety.update(
        _rate_summary(episodes, ("safety", "out_of_road"), stem="out_of_road")
    )
    safety.update(
        _rate_summary(
            episodes,
            ("safety", "background_gap_violation"),
            stem="background_gap_violation",
        )
    )
    safety.update(
        _rate_summary(
            episodes,
            ("safety", "platoon_gap_violation"),
            stem="platoon_gap_violation",
        )
    )
    safety.update(
        _rate_summary(
            episodes,
            ("safety", "execution_rejected"),
            stem="execution_rejection_episode",
        )
    )
    for field in (
        "background_gap_exposure_fraction",
        "background_gap_deficit_integral_m_s",
        "minimum_background_gap_m",
        "platoon_gap_exposure_fraction",
        "platoon_gap_deficit_integral_m_s",
        "minimum_platoon_gap_m",
        "minimum_ttc_s",
        "ttc_below_1_5_s_exposure_fraction",
        "max_drac_mps2",
        "execution_rejection_tick_fraction",
    ):
        safety[f"{field}_mean"] = _mean_episode_path(
            episodes, ("safety", field)
        )
    safety["per_agent"] = {}
    for agent_id in AGENT_IDS:
        agent = {}
        agent.update(
            _rate_summary(
                episodes,
                ("safety", "per_agent", agent_id, "collision"),
                stem="collision",
            )
        )
        agent.update(
            _rate_summary(
                episodes,
                ("safety", "per_agent", agent_id, "out_of_road"),
                stem="out_of_road",
            )
        )
        safety["per_agent"][agent_id] = agent

    efficiency: dict[str, object] = {}
    for field in (
        "scenario_realized",
        "functional_success_final",
        "functional_success_ever",
        "functional_success_stable",
    ):
        efficiency.update(
            _rate_summary(episodes, ("efficiency", field), stem=field)
        )
    for field in (
        "first_stable_success_time_s",
        "team_mean_route_progress_m",
        "team_min_route_progress_m",
        "team_mean_route_progress_fraction",
        "team_min_route_progress_fraction",
        "completion_time_s",
        "episode_duration_s",
    ):
        efficiency[f"{field}_mean"] = _mean_episode_path(
            episodes, ("efficiency", field)
        )
    efficiency["per_agent"] = {
        agent_id: {
            "route_progress_m_mean": _mean_episode_path(
                episodes,
                ("efficiency", "per_agent", agent_id, "route_progress_m"),
            ),
            "route_progress_fraction_mean": _mean_episode_path(
                episodes,
                ("efficiency", "per_agent", agent_id, "route_progress_fraction"),
            ),
        }
        for agent_id in AGENT_IDS
    }

    cooperation: dict[str, object] = {}
    for field in (
        "locked_spacing_error_mean_m",
        "locked_spacing_error_p95_m",
        "locked_spacing_error_max_m",
        "locked_speed_spread_mean_mps",
        "unlock_count",
        "unlocked_duration_s",
        "relock_recovery_time_s",
    ):
        output_field = (
            field
            if field in ("locked_spacing_error_mean_m", "locked_speed_spread_mean_mps")
            else f"{field}_mean"
        )
        cooperation[output_field] = _mean_episode_path(
            episodes, ("cooperation", field)
        )
    cooperation["relock_recovery_applicable_count"] = sum(
        _episode_value(episode, ("cooperation", "relock_recovery_success"))
        is not None
        for episode in episodes
    )
    cooperation.update(
        _rate_summary(
            episodes,
            ("cooperation", "relock_recovery_success"),
            stem="relock_recovery_success",
        )
    )
    cooperation.update(
        _rate_summary(
            episodes,
            ("cooperation", "recovery_censored"),
            stem="recovery_censored",
        )
    )

    comfort = {"per_agent": {}}
    for agent_id in AGENT_IDS:
        source = _episode_value_mapping(
            episodes[0], ("comfort", "per_agent", agent_id)
        )
        comfort["per_agent"][agent_id] = {
            field: _mean_episode_path(
                episodes, ("comfort", "per_agent", agent_id, field)
            )
            for field in source
        }

    timing: dict[str, object] = {}
    timing_source = _episode_value_mapping(episodes[0], ("timing",))
    for component, component_value in timing_source.items():
        if not isinstance(component_value, Mapping):
            raise ModelEvaluationError("episode timing component must be a mapping")
        for field in component_value:
            timing[f"{component}_{field}_mean"] = _mean_episode_path(
                episodes, ("timing", component, field)
            )
    return {
        "episodes": len(episodes),
        "safety": safety,
        "efficiency": efficiency,
        "cooperation": cooperation,
        "comfort": comfort,
        "timing": timing,
    }


def _episode_value_mapping(
    episode: Mapping[str, object], path: Sequence[str]
) -> Mapping[str, object]:
    current: object = episode
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise ModelEvaluationError(
                f"episode metric is missing {'.'.join(path)}"
            )
        current = current[key]
    if not isinstance(current, Mapping):
        raise ModelEvaluationError(f"episode metric {'.'.join(path)} must be an object")
    return current


def _build_model_aggregates(
    episodes: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for episode in episodes:
        scenario = episode.get("scenario")
        if not isinstance(scenario, str):
            raise ModelEvaluationError("episode scenario is invalid")
        grouped.setdefault(scenario, []).append(episode)
    counts = {len(rows) for rows in grouped.values()}
    if len(counts) != 1:
        raise ModelEvaluationError(
            "scenario macro aggregation requires equal episode counts per scenario"
        )
    by_scenario = {
        scenario: _aggregate_episode_metrics(rows)
        for scenario, rows in sorted(grouped.items())
    }
    overall = _aggregate_episode_metrics(episodes)
    _apply_scenario_macro(overall, tuple(by_scenario.values()))
    overall["aggregation"] = "equal_weight_scenario_macro"
    overall["scenario_count"] = len(by_scenario)
    return by_scenario, overall


def _apply_scenario_macro(
    target: dict[str, object], scenario_values: Sequence[Mapping[str, object]]
) -> None:
    """Replace scalar estimates by equal-scenario means, retaining pooled counts/CIs."""

    for key, value in tuple(target.items()):
        if isinstance(value, dict):
            nested = [
                scenario[key]
                for scenario in scenario_values
                if isinstance(scenario.get(key), Mapping)
            ]
            if nested:
                _apply_scenario_macro(value, nested)
            continue
        if (
            key in ("episodes", "scenario_count")
            or key.endswith("_count")
            or key.endswith("_sample_count")
            or key.endswith("_rate")
            or key.endswith("_ci95")
            or key == "aggregation"
        ):
            continue
        values = [
            float(scenario[key])
            for scenario in scenario_values
            if isinstance(scenario.get(key), (int, float))
            and not isinstance(scenario.get(key), bool)
        ]
        target[key] = float(np.mean(values)) if values else None


def _validate_grpo_evaluation_eligibility(
    payload: Mapping[str, object], *, formal: bool, model_id: str
) -> None:
    """Reject diagnostic/formal flag combinations that could overclaim a run."""

    expected = {
        "run_mode": "formal" if formal else "smoke",
        "diagnostic_only": not formal,
        "eligible_for_formal_training": formal,
        "calibration_required": False,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ModelEvaluationError(
                f"{model_id} checkpoint evaluation eligibility mismatch: {field}"
            )


def _validate_grpo_application_contract(
    payload: Mapping[str, object], *, model_id: str
) -> None:
    raw_contract = payload.get("reward_application_contract")
    contract_sha = payload.get("reward_application_contract_sha256")
    if not isinstance(raw_contract, Mapping) or not isinstance(contract_sha, str):
        raise ModelEvaluationError(
            f"{model_id} reward application contract is missing"
        )
    canonical_sha = hashlib.sha256(
        json.dumps(
            dict(raw_contract),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if contract_sha != canonical_sha:
        raise ModelEvaluationError(
            f"{model_id} reward application contract SHA mismatch"
        )
    bound_fields = (
        "reward_input_domain",
        "candidate_selection_domain",
        "execution_input_domain",
        "best_checkpoint_metric",
        "tracking_expansion_enabled",
        "calibration_required",
    )
    if any(payload.get(name) != raw_contract.get(name) for name in bound_fields):
        raise ModelEvaluationError(
            f"{model_id} reward application metadata mismatch"
        )
    if model_id == "grpo_open":
        if (
            dict(raw_contract) != GRPO_OPEN_REWARD_APPLICATION_CONTRACT
            or contract_sha != GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256
        ):
            raise ModelEvaluationError(
                "grpo_open must use the frozen tau_d application contract"
            )
    elif model_id == "grpo_exec":
        expected = {
            "reward_input_domain": "tau_a",
            "candidate_selection_domain": "tau_a",
            "execution_input_domain": "tau_cmd",
            "calibration_required": False,
        }
        if any(raw_contract.get(name) != value for name, value in expected.items()):
            raise ModelEvaluationError(
                "grpo_exec must use an execution-domain application contract"
            )


def _load_policy(
    spec: ModelSpec,
    *,
    device: torch.device,
    formal: bool,
    reward_bindings: dict[str, tuple[str, str, str]] | None = None,
):
    expected_variant = spec.variant
    expected_kind = spec.kind
    checkpoint = spec.checkpoint
    source_checkpoint = (
        checkpoint if expected_kind == "stage1" else spec.source_checkpoint
    )
    if source_checkpoint is None:
        raise ModelEvaluationError(f"{spec.model_id} source checkpoint is missing")
    loader = (
        load_stage1_a_for_grpo if expected_variant == "A" else load_stage1_b_for_grpo
    )
    trainer, source_payload, source_sha = loader(
        source_checkpoint,
        device=device,
        allow_diagnostic_source=not formal,
    )
    if formal and source_payload.get("eligible_for_formal_training") is not True:
        raise ModelEvaluationError(
            f"{spec.model_id} source is not eligible for formal evaluation"
        )
    if expected_kind == "grpo":
        checkpoint_loader = (
            load_grpo_checkpoint if expected_variant == "A" else load_grpo_b_checkpoint
        )
        grpo_payload = checkpoint_loader(
            checkpoint,
            trainer,
            expected_source_stage1_sha256=source_sha,
        )
        for field in (
            "run_mode",
            "reward_contract_version",
            "reward_contract_sha256",
            "reward_config",
            "reward_config_sha256",
            "reward_application_contract",
            "reward_application_contract_sha256",
            "reward_input_domain",
            "candidate_selection_domain",
            "execution_input_domain",
            "best_checkpoint_metric",
            "tracking_expansion_enabled",
            "calibration_required",
            "diagnostic_only",
            "eligible_for_formal_training",
            "scenario_seeds",
            "environment_steps",
            "scenario_contract_sha256",
            "trajectory_optimizer_config",
            "trajectory_optimizer_sha256",
        ):
            if field not in grpo_payload:
                raise ModelEvaluationError(
                    f"{spec.model_id} is not a contract-bound online GRPO checkpoint"
                )
        _validate_grpo_evaluation_eligibility(
            grpo_payload, formal=formal, model_id=spec.model_id
        )
        _validate_grpo_application_contract(
            grpo_payload, model_id=spec.model_id
        )
        expected_reward_version = JOINT_REWARD_CONTRACT.get("version")
        raw_reward_config = grpo_payload.get("reward_config")
        try:
            parsed_reward_config = (
                JointRewardConfig(**dict(raw_reward_config))
                if isinstance(raw_reward_config, Mapping)
                else None
            )
        except (TypeError, ValueError, JointRewardError) as exc:
            raise ModelEvaluationError(
                f"{spec.model_id} reward config contract mismatch"
            ) from exc
        if (
            not isinstance(expected_reward_version, str)
            or parsed_reward_config is None
            or dict(raw_reward_config) != dataclasses.asdict(parsed_reward_config)
            or grpo_payload.get("reward_contract_version")
            != expected_reward_version
            or grpo_payload.get("reward_contract_sha256")
            != JOINT_REWARD_CONTRACT_SHA256
            or grpo_payload.get("reward_config_sha256")
            != joint_reward_config_sha256(parsed_reward_config)
        ):
            raise ModelEvaluationError(
                f"{spec.model_id} reward contract mismatch"
            )
        if reward_bindings is not None:
            reward_bindings[spec.model_id] = (
                expected_reward_version,
                JOINT_REWARD_CONTRACT_SHA256,
                joint_reward_config_sha256(parsed_reward_config),
            )
        optimizer_config = KinematicTrajectoryOptimizerConfig()
        if (
            grpo_payload.get("trajectory_optimizer_config")
            != dataclasses.asdict(optimizer_config)
            or grpo_payload.get("trajectory_optimizer_sha256")
            != optimizer_config.sha256()
        ):
            raise ModelEvaluationError(
                f"{spec.model_id} trajectory optimizer contract mismatch"
            )
        expected_contract = primary_scenario_contract()
        if grpo_payload.get("scenario_contract_sha256") != expected_contract["sha256"]:
            raise ModelEvaluationError(
                f"{spec.model_id} scenario contract no longer matches frozen S5--S9"
            )
    trainer.planner.eval()
    return trainer.planner


def _validate_common_reward_binding(
    bindings: Mapping[str, tuple[str, str, str]],
) -> dict[str, str] | None:
    """Reject comparisons between GRPO checkpoints trained under mixed rewards."""

    if not bindings:
        return None
    distinct = set(bindings.values())
    if len(distinct) != 1:
        raise ModelEvaluationError(
            "GRPO checkpoints mix reward contract/config hashes"
        )
    version, contract_sha, config_sha = next(iter(distinct))
    return {
        "reward_contract_version": version,
        "reward_contract_sha256": contract_sha,
        "reward_config_sha256": config_sha,
    }


def _empty_metrics() -> dict[str, object]:
    return {
        "roles": {
            agent_id: {
                "collision": 0,
                "out_of_road": 0,
                "progress": [],
                "speed_km_h": [],
                "minimum_gap_m": [],
                "selected_modes": [],
                "stop": 0,
                "acceleration_mps2": [],
                "jerk": [],
                "yaw_rate_rad_s": [],
                "steering_change": [],
            }
            for agent_id in AGENT_IDS
        },
        "episode_collision": 0,
        "episode_out_of_road": 0,
        "episode_completed": 0,
        "execution_rejections": [],
        "gap_5m_violation": 0,
        "gap_7m_violation": 0,
        "formation_error": [],
        "formation_spread": [],
        "recovery_time_s": [],
        "joint_reward": [],
        "episode_lengths": [],
        "mode_rewards": {},
        "timing": {
            "bev_build_ms": [],
            "model_inference_ms": [],
            "control_mapping_ms": [],
            "planning_tick_ms": [],
            "trajectory_optimizer_ms": [],
            "rule_maker_ms": [],
        },
        "trajectory_intervention_ade_m": [],
        "trajectory_intervention_fde_m": [],
        "trajectory_retained_raw_fraction": [],
        "execution_modes_removed": 0,
        "execution_mode_masks_built": 0,
        "rule_proposal_match_attempts": 0,
        "rule_proposal_matches": 0,
        "rule_accepted_ranks": [],
        "rule_condition_failures": 0,
        "s7_feedback_exception_hits": 0,
        "episode_outcomes": [],
        "episode_metrics": [],
    }


def _summarize(raw: dict[str, object], episode_count: int) -> dict[str, object]:
    roles = {}
    for agent_id, values in raw["roles"].items():
        modes = values["selected_modes"]
        roles[agent_id] = {
            "collision_rate": values["collision"] / episode_count,
            "out_of_road_rate": values["out_of_road"] / episode_count,
            "progress_mean_m": (
                float(np.mean(values["progress"])) if values["progress"] else 0.0
            ),
            "speed_mean_km_h": (
                float(np.mean(values["speed_km_h"])) if values["speed_km_h"] else 0.0
            ),
            "minimum_gap_m": min(values["minimum_gap_m"], default=1.0e6),
            "mode_distribution": {
                str(mode): int(modes.count(mode)) for mode in sorted(set(modes))
            },
            "stop_rate": values["stop"] / max(len(modes), 1),
            "acceleration_abs_mean_mps2": (
                float(np.mean(np.abs(values["acceleration_mps2"])))
                if values["acceleration_mps2"]
                else 0.0
            ),
            "jerk_abs_mean": (
                float(np.mean(np.abs(values["jerk"]))) if values["jerk"] else 0.0
            ),
            "yaw_rate_abs_mean_rad_s": (
                float(np.mean(np.abs(values["yaw_rate_rad_s"])))
                if values["yaw_rate_rad_s"]
                else 0.0
            ),
            "steering_change_abs_mean": (
                float(np.mean(np.abs(values["steering_change"])))
                if values["steering_change"]
                else 0.0
            ),
        }
    timing = {name: _timing_distribution(values) for name, values in raw["timing"].items()}
    timing["planning_tick_ms"]["over_200_ms_fraction"] = (
        sum(float(value) > 200.0 for value in raw["timing"]["planning_tick_ms"])
        / max(len(raw["timing"]["planning_tick_ms"]), 1)
    )
    joint_reward = list(raw["joint_reward"])
    legacy_mode_rewards = {
        str(mode): {
            "count": len(values),
            "mean_reward": float(np.mean(values)) if values else 0.0,
        }
        for mode, values in sorted(raw.get("mode_rewards", {}).items())
    }
    execution_rejections = list(raw.get("execution_rejections", ()))
    rejection_episode_keys = {
        (
            str(value.get("scenario")),
            str(value.get("route")),
            int(value.get("seed")),
        )
        for value in execution_rejections
    }
    rejection_tick_keys = {
        (*episode_key, int(value.get("step", -1)))
        for value in execution_rejections
        for episode_key in [
            (
                str(value.get("scenario")),
                str(value.get("route")),
                int(value.get("seed")),
            )
        ]
    }
    result = {
        "episodes": episode_count,
        "roles": roles,
        "joint_safety": {
            "collision_rate": raw["episode_collision"] / episode_count,
            "out_of_road_rate": raw["episode_out_of_road"] / episode_count,
            "gap_5m_violation_rate": raw["gap_5m_violation"] / episode_count,
            "gap_7m_violation_rate": raw["gap_7m_violation"] / episode_count,
        },
        "formation": {
            "mean_error_m": (
                float(np.mean(raw["formation_error"]))
                if raw["formation_error"]
                else 0.0
            ),
            "p95_error_m": _percentile(raw["formation_error"], 95),
            "maximum_spread_m": max(raw["formation_spread"], default=0.0),
            "recovery_time_mean_s": (
                float(np.mean(raw["recovery_time_s"]))
                if raw["recovery_time_s"]
                else 0.0
            ),
        },
        "efficiency": {
            "completion_rate": raw["episode_completed"] / episode_count,
            "joint_reward_mean": (
                float(np.mean(raw["joint_reward"])) if raw["joint_reward"] else 0.0
            ),
        },
        "execution": {
            "rejection_count": len(execution_rejections),
            "rejection_episode_count": len(rejection_episode_keys),
            "rejection_episode_rate": len(rejection_episode_keys) / episode_count,
            "rejection_tick_count": len(rejection_tick_keys),
            "rejection_tick_fraction": (
                sum(
                    float(episode["safety"]["execution_rejection_tick_fraction"])
                    for episode in raw.get("episode_metrics", ())
                )
                / max(len(raw.get("episode_metrics", ())), 1)
            ),
            "rejections": execution_rejections,
            "mean_modes_removed_per_joint_state": (
                raw.get("execution_modes_removed", 0)
                / max(raw.get("execution_mode_masks_built", 0), 1)
            ),
        },
        "rule_maker": {
            "proposal_match_rate": raw.get("rule_proposal_matches", 0)
            / max(raw.get("rule_proposal_match_attempts", 0), 1),
            "accepted_proposal_rank_mean": (
                float(np.mean(raw.get("rule_accepted_ranks", ())))
                if raw.get("rule_accepted_ranks")
                else None
            ),
            "condition_failure_count": int(
                raw.get("rule_condition_failures", 0)
            ),
            "s7_left_to_keep_exception_hits": int(
                raw.get("s7_feedback_exception_hits", 0)
            ),
        },
        "trajectory_optimization": {
            "intervention_ade_mean_m": (
                float(np.mean(raw.get("trajectory_intervention_ade_m", ())))
                if raw.get("trajectory_intervention_ade_m")
                else 0.0
            ),
            "intervention_fde_mean_m": (
                float(np.mean(raw.get("trajectory_intervention_fde_m", ())))
                if raw.get("trajectory_intervention_fde_m")
                else 0.0
            ),
            "retained_raw_fraction_mean": (
                float(np.mean(raw.get("trajectory_retained_raw_fraction", ())))
                if raw.get("trajectory_retained_raw_fraction")
                else 0.0
            ),
        },
        "episode_outcomes": list(raw.get("episode_outcomes", ())),
        "timing": timing,
        "legacy_closed_loop": {
            "num_episodes": episode_count,
            "success_rate": raw["episode_completed"] / episode_count,
            "crash_rate": raw["episode_collision"] / episode_count,
            "out_of_road_rate": raw["episode_out_of_road"] / episode_count,
            "average_reward_per_step": (
                float(np.mean(joint_reward)) if joint_reward else 0.0
            ),
            "average_episode_length": (
                float(np.mean(raw.get("episode_lengths", ())))
                if raw.get("episode_lengths")
                else 0.0
            ),
            "mode_reward_stats": legacy_mode_rewards,
        },
    }
    episode_metrics = list(raw.get("episode_metrics", ()))
    if episode_metrics:
        by_scenario, overall = _build_model_aggregates(episode_metrics)
        result["episode_metrics"] = episode_metrics
        result["by_scenario"] = by_scenario
        result["overall"] = overall
    return result


def _surrounding_vehicles(env: object) -> tuple[object, ...]:
    """Return controlled and traffic vehicles without planner ownership."""

    seen: set[int] = set()
    vehicles: list[object] = []
    agents = getattr(env, "agents", {}) or {}
    agent_values = agents.values() if isinstance(agents, Mapping) else agents
    engine = getattr(env, "engine", None)
    traffic_manager = getattr(engine, "traffic_manager", None)
    traffic = getattr(traffic_manager, "traffic_vehicles", None)
    if traffic is None:
        traffic = getattr(traffic_manager, "_traffic_vehicles", ()) or ()
    try:
        traffic_values = tuple(traffic)
    except TypeError:
        traffic_values = ()
    for vehicle in (*tuple(agent_values), *traffic_values):
        if vehicle is None or id(vehicle) in seen:
            continue
        seen.add(id(vehicle))
        vehicles.append(vehicle)
    return tuple(vehicles)


def _minimum_background_gap(env: object) -> float:
    background, _ = _live_gap_observations(env)
    return min(background.values(), default=float("inf"))


def _new_episode_state(env: object, rule_maker: object) -> dict[str, object]:
    navigation = _navigation_snapshot(env)
    previous_velocity = {}
    previous_heading = {}
    for agent_id in AGENT_IDS:
        vehicle = env.agents[agent_id]
        heading = float(vehicle.heading_theta)
        speed_mps = float(vehicle.speed_km_h) / 3.6
        previous_velocity[agent_id] = speed_mps * np.asarray(
            [math.cos(heading), math.sin(heading)], dtype=np.float64
        )
        previous_heading[agent_id] = heading
    return {
        "initial_navigation": dict(navigation),
        "last_navigation": dict(navigation),
        "background_gap_step_min": [],
        "platoon_gap_step_min": [],
        "background_gap_deficit_integral_m_s": 0.0,
        "platoon_gap_deficit_integral_m_s": 0.0,
        "background_gap_exposure_steps": 0,
        "platoon_gap_exposure_steps": 0,
        "previous_background_gaps": {},
        "previous_platoon_gaps": {},
        "minimum_ttc_by_step": [],
        "drac_values": [],
        "ttc_exposure_steps": 0,
        "scenario_realized": [],
        "functional_success": [],
        "formation_lock_states": [bool(rule_maker.is_formation_locked)],
        "locked_spacing_errors_m": [],
        "locked_speed_spreads_mps": [],
        "previous_velocity": previous_velocity,
        "previous_heading": previous_heading,
        "previous_longitudinal_acceleration": {
            agent_id: None for agent_id in AGENT_IDS
        },
        "previous_yaw_rate": {agent_id: None for agent_id in AGENT_IDS},
        "previous_control": {
            agent_id: np.zeros((2,), dtype=np.float64) for agent_id in AGENT_IDS
        },
        "comfort": {
            agent_id: {
                "longitudinal_acceleration": [],
                "lateral_acceleration": [],
                "longitudinal_jerk": [],
                "yaw_rate": [],
                "yaw_acceleration": [],
                "steering_command_slew": [],
                "throttle_command_slew": [],
            }
            for agent_id in AGENT_IDS
        },
        "timing": {
            "bev_build_ms": [],
            "model_inference_ms": [],
            "control_mapping_ms": [],
            "planning_tick_ms": [],
            "trajectory_optimizer_ms": [],
            "rule_maker_ms": [],
        },
        "model_ready_ticks": 0,
    }


def _update_navigation_cache(
    agent_states: Mapping[str, object], cache: dict[str, tuple[float, float]]
) -> None:
    for agent_id in AGENT_IDS:
        vehicle = agent_states.get(agent_id)
        if isinstance(vehicle, Mapping):
            travelled = vehicle.get("navigation_travelled_length_m")
            total = vehicle.get("navigation_total_length_m")
        else:
            navigation = getattr(vehicle, "navigation", None)
            travelled = getattr(navigation, "travelled_length", None)
            total = getattr(navigation, "total_length", None)
        if (
            isinstance(travelled, bool)
            or not isinstance(travelled, (int, float))
            or isinstance(total, bool)
            or not isinstance(total, (int, float))
            or not math.isfinite(float(travelled))
            or not math.isfinite(float(total))
            or float(total) <= 0.0
        ):
            raise ModelEvaluationError(
                f"navigation progress is unavailable for {agent_id}"
            )
        cache[agent_id] = (float(travelled), float(total))


def _update_episode_state(
    state: dict[str, object],
    *,
    env: object,
    rule_maker: object,
    step_info: Mapping[str, object],
    pre_poses: np.ndarray,
    dt_s: float,
) -> None:
    agent_states = _post_step_agent_states(env, step_info)
    background, platoon = _live_gap_observations(
        env, agent_states=agent_states
    )
    background_min = min(background.values(), default=None)
    platoon_min = min(platoon.values(), default=None)
    state["current_background_gap"] = (
        float(background_min) if background_min is not None else float("inf")
    )
    state["current_platoon_gap"] = (
        float(platoon_min) if platoon_min is not None else float("inf")
    )
    if background_min is not None:
        state["background_gap_step_min"].append(float(background_min))
        deficit = max(BACKGROUND_GAP_THRESHOLD_M - float(background_min), 0.0)
        state["background_gap_deficit_integral_m_s"] += deficit * dt_s
        state["background_gap_exposure_steps"] += int(deficit > 0.0)
    if platoon_min is not None:
        state["platoon_gap_step_min"].append(float(platoon_min))
        deficit = max(PLATOON_GAP_THRESHOLD_M - float(platoon_min), 0.0)
        state["platoon_gap_deficit_integral_m_s"] += deficit * dt_s
        state["platoon_gap_exposure_steps"] += int(deficit > 0.0)
    background_ttc, background_drac = _closing_risk_observations(
        background,
        state["previous_background_gaps"],
        dt_s=dt_s,
    )
    platoon_ttc, platoon_drac = _closing_risk_observations(
        platoon,
        state["previous_platoon_gaps"],
        dt_s=dt_s,
    )
    step_ttc = [*background_ttc, *platoon_ttc]
    if step_ttc:
        minimum_ttc = min(step_ttc)
        state["minimum_ttc_by_step"].append(float(minimum_ttc))
        state["ttc_exposure_steps"] += int(
            minimum_ttc < TTC_OBSERVATION_THRESHOLD_S
        )
    state["drac_values"].extend([*background_drac, *platoon_drac])
    state["previous_background_gaps"] = dict(background)
    state["previous_platoon_gaps"] = dict(platoon)

    _update_navigation_cache(agent_states, state["last_navigation"])
    summary = _scenario_summary_from_step_info(step_info)
    state["scenario_realized"].append(
        bool(summary.get("scenario_realized", False))
    )
    state["functional_success"].append(
        bool(summary.get("functional_success", False))
    )
    locked = bool(rule_maker.is_formation_locked)
    state["formation_lock_states"].append(locked)
    if locked:
        ideal_gap = float(getattr(rule_maker, "ideal_following_distance_m", 10.0))
        state["locked_spacing_errors_m"].extend(
            abs(float(gap) - ideal_gap) for gap in platoon.values()
        )
        speeds = [
            (
                float(agent_states[agent_id]["speed_km_h"])
                if isinstance(agent_states[agent_id], Mapping)
                else float(agent_states[agent_id].speed_km_h)
            )
            / 3.6
            for agent_id in AGENT_IDS
        ]
        state["locked_speed_spreads_mps"].append(max(speeds) - min(speeds))

    controls = getattr(env, "_pending_low_level_actions", {}) or {}
    for role, agent_id in enumerate(AGENT_IDS):
        vehicle = agent_states[agent_id]
        post_pose = _vehicle_pose(vehicle)
        velocity = (post_pose[:2] - pre_poses[role, :2]) / dt_s
        acceleration_world = (
            velocity - state["previous_velocity"][agent_id]
        ) / dt_s
        forward = np.asarray(
            [math.cos(post_pose[2]), math.sin(post_pose[2])], dtype=np.float64
        )
        lateral = np.asarray([-forward[1], forward[0]], dtype=np.float64)
        longitudinal_acceleration = float(np.dot(acceleration_world, forward))
        lateral_acceleration = float(np.dot(acceleration_world, lateral))
        heading_delta = math.atan2(
            math.sin(post_pose[2] - state["previous_heading"][agent_id]),
            math.cos(post_pose[2] - state["previous_heading"][agent_id]),
        )
        yaw_rate = heading_delta / dt_s
        comfort = state["comfort"][agent_id]
        comfort["longitudinal_acceleration"].append(longitudinal_acceleration)
        comfort["lateral_acceleration"].append(lateral_acceleration)
        prior_acceleration = state["previous_longitudinal_acceleration"][agent_id]
        if prior_acceleration is not None:
            comfort["longitudinal_jerk"].append(
                (longitudinal_acceleration - float(prior_acceleration)) / dt_s
            )
        comfort["yaw_rate"].append(yaw_rate)
        prior_yaw_rate = state["previous_yaw_rate"][agent_id]
        if prior_yaw_rate is not None:
            comfort["yaw_acceleration"].append(
                (yaw_rate - float(prior_yaw_rate)) / dt_s
            )
        control = np.asarray(
            controls.get(agent_id, np.zeros((2,), dtype=np.float64)),
            dtype=np.float64,
        ).reshape(-1)
        if control.shape != (2,) or not np.isfinite(control).all():
            raise ModelEvaluationError("executed control must be finite [steer, throttle]")
        slew = (control - state["previous_control"][agent_id]) / dt_s
        comfort["steering_command_slew"].append(float(slew[0]))
        comfort["throttle_command_slew"].append(float(slew[1]))
        state["previous_velocity"][agent_id] = velocity
        state["previous_heading"][agent_id] = float(post_pose[2])
        state["previous_longitudinal_acceleration"][agent_id] = (
            longitudinal_acceleration
        )
        state["previous_yaw_rate"][agent_id] = yaw_rate
        state["previous_control"][agent_id] = control


def _stable_success_time(
    values: Sequence[bool], *, dt_s: float
) -> tuple[bool, float | None]:
    if not values or not values[-1]:
        return False, None
    index = len(values) - 1
    while index > 0 and values[index - 1]:
        index -= 1
    return True, float((index + 1) * dt_s)


def _formation_recovery(
    lock_states: Sequence[bool], *, dt_s: float
) -> dict[str, object]:
    unlock_count = 0
    unlock_start: int | None = None
    completed_durations = []
    for index, (previous, current) in enumerate(zip(lock_states, lock_states[1:]), start=1):
        if previous and not current:
            unlock_count += 1
            unlock_start = index
        elif not previous and current and unlock_start is not None:
            completed_durations.append((index - unlock_start) * dt_s)
            unlock_start = None
    if unlock_count == 0:
        success: bool | None = None
        censored: bool | None = None
        recovery_time: float | None = None
    else:
        success = unlock_start is None
        censored = not success
        recovery_time = (
            float(np.mean(completed_durations))
            if success and completed_durations
            else None
        )
    return {
        "unlock_count": int(unlock_count),
        "unlocked_duration_s": float(sum(not value for value in lock_states[1:]) * dt_s),
        "relock_recovery_success": success,
        "recovery_censored": censored,
        "relock_recovery_time_s": recovery_time,
    }


def _finish_episode_metrics(
    state: Mapping[str, object],
    *,
    model_id: str,
    scenario: Sequence[str],
    seed: int,
    dt_s: float,
    executed_steps: int,
    collision: bool,
    out_of_road: bool,
    role_collision: Mapping[str, bool],
    role_out: Mapping[str, bool],
    execution_rejected: bool,
    rejection_steps: Sequence[int],
) -> dict[str, object]:
    initial_navigation = state["initial_navigation"]
    last_navigation = state["last_navigation"]
    per_agent_progress = {}
    route_progress_m = []
    route_progress_fraction = []
    for agent_id in AGENT_IDS:
        initial, total = initial_navigation[agent_id]
        final, final_total = last_navigation[agent_id]
        if not math.isclose(total, final_total, rel_tol=0.0, abs_tol=1.0e-6):
            raise ModelEvaluationError(f"navigation total length changed for {agent_id}")
        progress = float(final - initial)
        remaining = max(total - initial, 1.0e-6)
        fraction = float(np.clip(progress / remaining, 0.0, 1.0))
        per_agent_progress[agent_id] = {
            "route_progress_m": progress,
            "route_progress_fraction": fraction,
        }
        route_progress_m.append(progress)
        route_progress_fraction.append(fraction)
    functional_values = list(state["functional_success"])
    stable_success, first_stable_time = _stable_success_time(
        functional_values, dt_s=dt_s
    )
    background_gaps = list(state["background_gap_step_min"])
    platoon_gaps = list(state["platoon_gap_step_min"])
    minimum_ttc = min(state["minimum_ttc_by_step"], default=None)
    comfort = {"per_agent": {}}
    comfort_specs = (
        ("longitudinal_acceleration", "longitudinal_acceleration", "mps2"),
        ("lateral_acceleration", "lateral_acceleration", "mps2"),
        ("longitudinal_jerk", "longitudinal_jerk", "mps3"),
        ("yaw_rate", "yaw_rate", "rad_s"),
        ("yaw_acceleration", "yaw_acceleration", "rad_s2"),
        ("steering_command_slew", "steering_command_slew", "per_s"),
        ("throttle_command_slew", "throttle_command_slew", "per_s"),
    )
    for agent_id in AGENT_IDS:
        values = state["comfort"][agent_id]
        summary = {}
        for source, stem, unit in comfort_specs:
            summary.update(_absolute_distribution(values[source], stem, unit))
        comfort["per_agent"][agent_id] = summary
    timing = {
        component: _timing_distribution(values)
        for component, values in state["timing"].items()
    }
    timing["planning_tick_ms"]["over_200_ms_fraction"] = (
        sum(value > 200.0 for value in state["timing"]["planning_tick_ms"])
        / max(len(state["timing"]["planning_tick_ms"]), 1)
    )
    recovery = _formation_recovery(state["formation_lock_states"], dt_s=dt_s)
    spacing = list(state["locked_spacing_errors_m"])
    speed_spread = list(state["locked_speed_spreads_mps"])
    model_ready_ticks = int(state["model_ready_ticks"])
    rejection_tick_count = len(set(int(value) for value in rejection_steps))
    return {
        "model": model_id,
        "scenario": str(scenario[0]),
        "route": str(scenario[1]),
        "seed": int(seed),
        "dt_s": float(dt_s),
        "steps": int(executed_steps),
        "safety": {
            "collision": bool(collision),
            "out_of_road": bool(out_of_road),
            "per_agent": {
                agent_id: {
                    "collision": bool(role_collision[agent_id]),
                    "out_of_road": bool(role_out[agent_id]),
                }
                for agent_id in AGENT_IDS
            },
            "background_gap_violation": bool(
                background_gaps and min(background_gaps) < BACKGROUND_GAP_THRESHOLD_M
            ),
            "background_gap_exposure_fraction": float(
                state["background_gap_exposure_steps"] / max(executed_steps, 1)
            ),
            "background_gap_deficit_integral_m_s": float(
                state["background_gap_deficit_integral_m_s"]
            ),
            "minimum_background_gap_m": min(background_gaps, default=None),
            "platoon_gap_violation": bool(
                platoon_gaps and min(platoon_gaps) < PLATOON_GAP_THRESHOLD_M
            ),
            "platoon_gap_exposure_fraction": float(
                state["platoon_gap_exposure_steps"] / max(executed_steps, 1)
            ),
            "platoon_gap_deficit_integral_m_s": float(
                state["platoon_gap_deficit_integral_m_s"]
            ),
            "minimum_platoon_gap_m": min(platoon_gaps, default=None),
            "minimum_ttc_s": minimum_ttc,
            "ttc_below_1_5_s_exposure_fraction": float(
                state["ttc_exposure_steps"] / max(executed_steps, 1)
            ),
            "max_drac_mps2": max(state["drac_values"], default=0.0),
            "execution_rejected": bool(execution_rejected),
            "execution_rejection_tick_fraction": float(
                rejection_tick_count / max(model_ready_ticks, 1)
            ),
        },
        "efficiency": {
            "scenario_realized": bool(any(state["scenario_realized"])),
            "functional_success_final": bool(
                functional_values[-1] if functional_values else False
            ),
            "functional_success_ever": bool(any(functional_values)),
            "functional_success_stable": bool(stable_success),
            "first_stable_success_time_s": first_stable_time,
            "per_agent": per_agent_progress,
            "team_mean_route_progress_m": float(np.mean(route_progress_m)),
            "team_min_route_progress_m": float(min(route_progress_m)),
            "team_mean_route_progress_fraction": float(
                np.mean(route_progress_fraction)
            ),
            "team_min_route_progress_fraction": float(min(route_progress_fraction)),
            "completion_time_s": first_stable_time,
            "episode_duration_s": float(executed_steps * dt_s),
        },
        "cooperation": {
            "locked_spacing_error_mean_m": (
                float(np.mean(spacing)) if spacing else None
            ),
            "locked_spacing_error_p95_m": _nullable_percentile(spacing, 95),
            "locked_spacing_error_max_m": max(spacing, default=None),
            "locked_speed_spread_mean_mps": (
                float(np.mean(speed_spread)) if speed_spread else None
            ),
            **recovery,
        },
        "comfort": comfort,
        "timing": timing,
    }


def _initial_state_signature(env: object) -> np.ndarray:
    rows = []
    for agent_id in AGENT_IDS:
        if agent_id not in env.agents:
            raise ModelEvaluationError(
                f"initial state is missing required agent {agent_id}"
            )
        vehicle = env.agents[agent_id]
        position = np.asarray(vehicle.position, dtype=np.float64).reshape(-1)
        rows.append(
            (
                float(position[0]),
                float(position[1]),
                float(vehicle.heading_theta),
                float(vehicle.speed_km_h),
            )
        )
    signature = np.asarray(rows, dtype=np.float64)
    if signature.shape != (3, 4) or not np.isfinite(signature).all():
        raise ModelEvaluationError("initial vehicle state is invalid")
    return signature


def _initial_states_sha256(
    values: Mapping[tuple[str, str, int], np.ndarray],
) -> str:
    digest = hashlib.sha256()
    for key in sorted(values):
        digest.update(json.dumps(key, separators=(",", ":")).encode("utf-8"))
        signature = np.ascontiguousarray(values[key], dtype=np.float64)
        digest.update(signature.tobytes())
    return digest.hexdigest()


def _initial_scene_sha256(env: object) -> str:
    """Hash platoon and background actors without unstable object UUIDs."""

    agent_objects = {
        id(vehicle): agent_id
        for agent_id, vehicle in (getattr(env, "agents", {}) or {}).items()
    }
    rows = []
    engine = getattr(env, "engine", None)
    get_policy = getattr(engine, "get_policy", None)
    for vehicle in _surrounding_vehicles(env):
        position = np.asarray(getattr(vehicle, "position", ()), dtype=np.float64)
        if position.shape[0] < 2 or not np.isfinite(position[:2]).all():
            raise ModelEvaluationError("initial scene vehicle position is invalid")
        lane = getattr(vehicle, "lane", None)
        lane_index = getattr(lane, "index", getattr(vehicle, "lane_index", None))
        policy = None
        if callable(get_policy):
            try:
                policy = get_policy(getattr(vehicle, "name", ""))
            except Exception:
                policy = None
        rows.append(
            {
                "role": agent_objects.get(id(vehicle), "background"),
                "vehicle_class": type(vehicle).__name__,
                "position": [float(position[0]), float(position[1])],
                "heading": float(getattr(vehicle, "heading_theta", 0.0)),
                "speed_km_h": float(getattr(vehicle, "speed_km_h", 0.0)),
                "lane_index": repr(lane_index),
                "policy_class": type(policy).__name__ if policy is not None else None,
            }
        )
    rows.sort(
        key=lambda row: (
            row["role"],
            row["vehicle_class"],
            row["position"][0],
            row["position"][1],
            row["lane_index"],
        )
    )
    encoded = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_mapping_sha256(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name]
        if not isinstance(value, torch.Tensor):
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _tensor_mapping_exact(
    first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]
) -> bool:
    tensor_keys = {
        name for name, value in first.items() if isinstance(value, torch.Tensor)
    }
    if tensor_keys != {
        name for name, value in second.items() if isinstance(value, torch.Tensor)
    }:
        return False
    return all(torch.equal(first[name], second[name]) for name in tensor_keys)


def _execution_mask_or_record_rejection(
    values: object,
    *,
    optimizer: KinematicTrajectoryOptimizer,
    raw: dict[str, object],
    model_id: str,
    scenario: Sequence[str],
    seed: int,
    step_index: int,
) -> np.ndarray | None:
    try:
        return execution_mode_valid_mask(values, optimizer=optimizer)
    except TrajectoryOptimizationError as exc:
        raw["execution_rejections"].append(
            {
                "model": model_id,
                "scenario": str(scenario[0]),
                "route": str(scenario[1]),
                "seed": int(seed),
                "step": int(step_index),
                "stage": "execution_mode_mask",
                "reason": str(exc),
            }
        )
        return None


def _record_safe_stop_rejection(
    raw: dict[str, object], rejection: Mapping[str, object]
) -> bool:
    """Record a RuleMaker/model mismatch and mark its episode as rejected."""

    raw["execution_rejections"].append(dict(rejection))
    return True


def _formal_conclusions_eligible(run_mode: str, all_gates_passed: bool) -> bool:
    return run_mode == "formal" and bool(all_gates_passed)


def _timed_optimizer_call(
    elapsed_ms: list[float],
    function: Callable[..., object],
    *args: object,
    **kwargs: object,
) -> object:
    start = time.perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        elapsed_ms[0] += (time.perf_counter() - start) * 1000.0


def _map_trajectory_controls(
    env: object,
    trajectories: Mapping[str, np.ndarray],
    *,
    model_id: str,
    scenario: Sequence[str],
    seed: int,
    step_index: int,
) -> dict[str, np.ndarray]:
    controls = {}
    for agent_id in AGENT_IDS:
        try:
            control = np.asarray(
                env.trajectory_to_control(agent_id, trajectories[agent_id]),
                dtype=np.float32,
            )
        except Exception as exc:
            raise ModelEvaluationError(
                "trajectory control failed for "
                f"model={model_id} scenario={scenario[0]} "
                f"seed={seed} step={step_index} agent={agent_id}: {exc}"
            ) from exc
        if control.shape != (2,) or not np.isfinite(control).all():
            raise ModelEvaluationError("trajectory control must be finite [steer, throttle]")
        controls[agent_id] = control
    return controls


def _step_precomputed_trajectory_controls(
    env: object,
    trajectories: Mapping[str, np.ndarray],
    controls: Mapping[str, np.ndarray],
) -> tuple[object, object, object, object, object]:
    """Execute exactly the controls already timed by the evaluator."""

    env._pending_step_trajectories = {
        agent_id: np.asarray(trajectories[agent_id], dtype=np.float32)
        for agent_id in AGENT_IDS
    }
    return env.low_level_step(dict(controls), control_mode="trajectory")


@torch.no_grad()
def evaluate_models(
    manifest_path: Path,
    output_path: Path,
    config: ModelEvaluationConfig | None = None,
) -> dict[str, object]:
    cfg = config or ModelEvaluationConfig()
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise ModelEvaluationError("CUDA evaluation requested but unavailable")
    device = torch.device(cfg.device)
    _configure_deterministic_inference(device)
    manifest = load_model_manifest(Path(manifest_path))
    reward_bindings: dict[str, tuple[str, str, str]] = {}
    models = {
        spec.model_id: _load_policy(
            spec,
            device=device,
            formal=cfg.run_mode == "formal",
            reward_bindings=reward_bindings,
        )
        for spec in manifest.models
    }
    common_reward_binding = _validate_common_reward_binding(reward_bindings)
    model_reports = {}
    episode_count = len(cfg.scenarios) * len(cfg.seeds)
    reference_initial_states: dict[tuple[str, str, int], np.ndarray] = {}
    reference_initial_scenes: dict[tuple[str, str, int], str] = {}

    for name, planner in models.items():
        raw = _empty_metrics()
        trajectory_optimizer = KinematicTrajectoryOptimizer()
        deterministic_probe: dict[str, object] | None = None
        artifact_writer = (
            ClosedLoopArtifactWriter(
                cfg.artifact_root,
                name,
                video_fps=cfg.video_fps,
                frame_interval=cfg.visualization_interval,
                topdown_screen_size=cfg.topdown_screen_size,
                topdown_film_size=cfg.topdown_film_size,
            )
            if cfg.save_visualizations and cfg.artifact_root is not None
            else None
        )
        for scenario in cfg.scenarios:
            for seed in cfg.seeds:
                env = SensorlessJointBEVPlatoonEnv(
                    {
                        "num_agents": 3,
                        "traffic_density": 0.0,
                        "initial_speed_km_h": deterministic_initial_speed_km_h(
                            scenario[0], int(seed)
                        ),
                        "start_seed": int(seed),
                        "num_scenarios": 1,
                    }
                )
                env.set_runtime_scenario_route(*scenario)
                spawn_manager = getattr(
                    getattr(env, "engine", None), "spawn_manager", None
                )
                set_spawn_seed = getattr(spawn_manager, "set_episode_spawn_seed", None)
                if callable(set_spawn_seed):
                    set_spawn_seed(int(seed))
                env.reset(seed=int(seed))
                initial_key = (scenario[0], scenario[1], int(seed))
                initial_state = _initial_state_signature(env)
                reference_initial_state = reference_initial_states.setdefault(
                    initial_key, initial_state.copy()
                )
                if not np.allclose(
                    initial_state,
                    reference_initial_state,
                    rtol=0.0,
                    atol=1.0e-8,
                ):
                    raise ModelEvaluationError(
                        "models did not receive identical reset state for "
                        f"scenario={scenario[0]} seed={seed}"
                    )
                initial_scene = _initial_scene_sha256(env)
                reference_initial_scene = reference_initial_scenes.setdefault(
                    initial_key, initial_scene
                )
                if initial_scene != reference_initial_scene:
                    raise ModelEvaluationError(
                        "models did not receive identical initial scene for "
                        f"scenario={scenario[0]} seed={seed}"
                    )
                builder = JointBEVSampleBuilder(AGENT_IDS)
                builder.reset()
                if getattr(planner.config, "model_version", None) != "v2":
                    raise ModelEvaluationError(
                        "S5-S9 evaluation requires a v2 planner"
                    )
                rule_maker = make_rule_maker(dict(env.config))
                rule_maker.reset(env, list(AGENT_IDS))
                episode_state = _new_episode_state(env, rule_maker)
                episode_rejection_start = len(raw["execution_rejections"])
                committed_execution_id: int | None = None
                committed_plan_actions: dict[str, int] | None = None
                dt_s = simulator_decision_dt_s(env)
                generator = torch.Generator(device=device)
                generator.manual_seed(int(seed))
                episode_collision = False
                episode_out = False
                episode_gap5 = False
                episode_gap7 = False
                episode_completed = False
                episode_execution_rejected = False
                episode_min_background_gap = float("inf")
                episode_min_platoon_gap = float("inf")
                role_collision = {agent_id: False for agent_id in AGENT_IDS}
                role_out = {agent_id: False for agent_id in AGENT_IDS}
                previous_speed = {agent_id: None for agent_id in AGENT_IDS}
                previous_heading = {
                    agent_id: float(env.agents[agent_id].heading_theta)
                    for agent_id in AGENT_IDS
                }
                executed_steps = 0
                artifact_episode_id = (
                    artifact_writer.start_episode(scenario, int(seed))
                    if artifact_writer is not None
                    else None
                )
                try:
                    for step_index in range(cfg.max_steps):
                        pre_poses = np.asarray(
                            [
                                [
                                    *np.asarray(
                                        env.agents[agent_id].position, dtype=np.float64
                                    )[:2],
                                    float(env.agents[agent_id].heading_theta),
                                ]
                                for agent_id in AGENT_IDS
                            ],
                            dtype=np.float64,
                        )
                        trajectories = None
                        precomputed_controls = None
                        step_bev = None
                        builder.capture_state(env, step_index * dt_s)
                        if not builder.history_ready():
                            action = constant_velocity_actions(env)
                            selected_modes = None
                        else:
                            episode_state["model_ready_ticks"] += 1
                            tick_start = time.perf_counter()
                            bev_start = tick_start
                            values = builder.build_model_inputs(env)
                            step_bev = values.bev
                            bev_ms = (time.perf_counter() - bev_start) * 1000.0
                            rule_batch = None
                            rule_condition: dict[str, int] | None = None
                            rule_condition_is_commitment = False
                            rule_start = time.perf_counter()
                            hard_modes = hard_valid_modes_by_rule_action(
                                AGENT_IDS, values.mode_valid_mask
                            )
                            if rule_maker.has_active_lane_change_commitments:
                                if (
                                    committed_execution_id is None
                                    or committed_plan_actions is None
                                ):
                                    raise ModelEvaluationError(
                                        "online RuleMaker commitment has no execution state"
                                    )
                                try:
                                    rule_maker.advance_committed_execution(
                                        env,
                                        list(AGENT_IDS),
                                        committed_execution_id,
                                    )
                                except LaneChangeCommitmentError as exc:
                                    raise ModelEvaluationError(
                                        f"online RuleMaker commitment failed: {exc}"
                                    ) from exc
                                if rule_maker.has_active_lane_change_commitments:
                                    rule_condition = (
                                        rule_maker.committed_execution_rule_actions(
                                            env, committed_plan_actions
                                        )
                                    )
                                    rule_condition_is_commitment = True
                                else:
                                    committed_execution_id = None
                                    committed_plan_actions = None
                            if rule_condition is None:
                                try:
                                    rule_batch = rule_maker.propose_joint_actions(
                                        env,
                                        list(AGENT_IDS),
                                        getattr(env, "_last_planner_batch", None) or {},
                                        hard_valid_modes_by_action=hard_modes,
                                    )
                                except LaneChangeCommitmentError as exc:
                                    raise ModelEvaluationError(
                                        f"online RuleMaker proposal failed: {exc}"
                                    ) from exc
                                rule_condition = (
                                    joint_proposal_actions(
                                        rule_batch.proposals[0], AGENT_IDS
                                    )
                                    if rule_batch.proposals
                                    else {agent_id: 0 for agent_id in AGENT_IDS}
                                )
                            rule_maker_ms = (
                                time.perf_counter() - rule_start
                            ) * 1000.0
                            values = builder.augment_v2_model_inputs(
                                env,
                                values,
                                rule_action_condition=rule_condition,
                                rule_formation_state=rule_maker.is_formation_locked,
                            )
                            execution_mask = _execution_mask_or_record_rejection(
                                values,
                                optimizer=trajectory_optimizer,
                                raw=raw,
                                model_id=name,
                                scenario=scenario,
                                seed=int(seed),
                                step_index=step_index,
                            )
                            if execution_mask is None:
                                episode_execution_rejected = True
                                break
                            raw["execution_modes_removed"] += int(
                                np.count_nonzero(
                                    values.mode_valid_mask & ~execution_mask
                                )
                            )
                            raw["execution_mode_masks_built"] += 1
                            batch = model_inputs_to_batch(
                                values,
                                device,
                                mode_valid_mask=execution_mask,
                            )
                            noise = torch.randn(
                                (1, 3, 10, 8, 2),
                                dtype=torch.float32,
                                device=device,
                                generator=generator,
                            )
                            _sync(device)
                            inference_start = time.perf_counter()
                            output = planner_forward_from_batch(
                                planner, batch, diffusion_noise=noise
                            )
                            _sync(device)
                            inference_ms = (
                                time.perf_counter() - inference_start
                            ) * 1000.0
                            deterministic_probe_ms = 0.0
                            if deterministic_probe is None:
                                probe_start = time.perf_counter()
                                repeated_output = planner_forward_from_batch(
                                    planner, batch, diffusion_noise=noise
                                )
                                deterministic_probe = {
                                    "input_and_noise_sha256": _tensor_mapping_sha256(
                                        {**batch, "diffusion_noise": noise}
                                    ),
                                    "output_sha256": _tensor_mapping_sha256(output),
                                    "identical_replay": _tensor_mapping_exact(
                                        output, repeated_output
                                    ),
                                }
                                if not deterministic_probe["identical_replay"]:
                                    raise ModelEvaluationError(
                                        f"{name} is not exact for identical input and noise"
                                    )
                                deterministic_probe_ms = (
                                    time.perf_counter() - probe_start
                                ) * 1000.0
                            raw_trajectories = (
                                output["selected_trajectory"][0].detach().cpu().numpy()
                            )
                            selected_mode_array = (
                                output["selected_mode"][0].detach().cpu().numpy()
                            )
                            pending_rule_acceptance = None
                            forced_safe_stop = False
                            optimizer_elapsed_ms = [0.0]
                            try:
                                optimization = _timed_optimizer_call(
                                    optimizer_elapsed_ms,
                                    optimize_selected_model_trajectories,
                                    values,
                                    raw_trajectories,
                                    selected_mode_array,
                                    optimizer=trajectory_optimizer,
                                )
                            except TrajectoryOptimizationError as exc:
                                raw["execution_rejections"].append(
                                    {
                                        "scenario": scenario[0],
                                        "route": scenario[1],
                                        "seed": int(seed),
                                        "step": int(step_index),
                                        "selected_modes": selected_mode_array.tolist(),
                                        "reason": str(exc),
                                    }
                                )
                                episode_execution_rejected = True
                                raw["rule_condition_failures"] += 1
                                optimization = _timed_optimizer_call(
                                    optimizer_elapsed_ms,
                                    optimize_safe_stop_trajectories,
                                    values, optimizer=trajectory_optimizer
                                )
                                forced_safe_stop = True
                                selected_mode_array = np.full(
                                    (3,), 9, dtype=np.int64
                                )
                            if not forced_safe_stop:
                                assert rule_condition is not None
                                physical_actions, feedback_actions, s7_exceptions = (
                                    diffusion_mode_feedback_actions(
                                        selected_mode_array.tolist(),
                                        AGENT_IDS,
                                        scenario_id=scenario[0],
                                        local_route=scenario[1],
                                    )
                                )
                                raw["s7_feedback_exception_hits"] += sum(
                                    int(value) for value in s7_exceptions.values()
                                )
                                if rule_condition_is_commitment:
                                    compatible = all(
                                        feedback_actions[agent_id]
                                        == int(rule_condition[agent_id])
                                        for agent_id in AGENT_IDS
                                    )
                                    if not compatible:
                                        raw["rule_condition_failures"] += 1
                                        episode_execution_rejected = (
                                            _record_safe_stop_rejection(
                                                raw,
                                                {
                                                    "scenario": scenario[0],
                                                    "route": scenario[1],
                                                    "seed": int(seed),
                                                    "step": int(step_index),
                                                    "stage": "active_commitment_compatibility",
                                                    "physical_actions": physical_actions,
                                                    "feedback_actions": feedback_actions,
                                                    "required_actions": rule_condition,
                                                },
                                            )
                                        )
                                        optimization = _timed_optimizer_call(
                                            optimizer_elapsed_ms,
                                            optimize_safe_stop_trajectories,
                                            values, optimizer=trajectory_optimizer
                                        )
                                        forced_safe_stop = True
                                        selected_mode_array = np.full(
                                            (3,), 9, dtype=np.int64
                                        )
                                else:
                                    raw["rule_proposal_match_attempts"] += 1
                                    matched = (
                                        None
                                        if rule_batch is None
                                        else match_joint_action_proposal(
                                            rule_batch,
                                            feedback_actions,
                                            AGENT_IDS,
                                        )
                                    )
                                    if matched is None:
                                        raw["rule_condition_failures"] += 1
                                        episode_execution_rejected = (
                                            _record_safe_stop_rejection(
                                                raw,
                                                {
                                                    "scenario": scenario[0],
                                                    "route": scenario[1],
                                                    "seed": int(seed),
                                                    "step": int(step_index),
                                                    "stage": "proposal_action_match",
                                                    "physical_actions": physical_actions,
                                                    "feedback_actions": feedback_actions,
                                                },
                                            )
                                        )
                                        optimization = _timed_optimizer_call(
                                            optimizer_elapsed_ms,
                                            optimize_safe_stop_trajectories,
                                            values, optimizer=trajectory_optimizer
                                        )
                                        forced_safe_stop = True
                                        selected_mode_array = np.full(
                                            (3,), 9, dtype=np.int64
                                        )
                                    else:
                                        pending_rule_acceptance = matched
                            trajectories = optimization.optimized_trajectories
                            selected_modes = selected_mode_array.tolist()
                            action = joint_trajectory_action(trajectories)
                            control_start = time.perf_counter()
                            precomputed_controls = _map_trajectory_controls(
                                env,
                                action,
                                model_id=name,
                                scenario=scenario,
                                seed=int(seed),
                                step_index=step_index,
                            )
                            control_ms = (
                                time.perf_counter() - control_start
                            ) * 1000.0
                            if pending_rule_acceptance is not None:
                                assert rule_batch is not None
                                acceptance_start = time.perf_counter()
                                try:
                                    rule_maker.accept_joint_action(
                                        rule_batch.batch_id,
                                        pending_rule_acceptance.proposal_id,
                                    )
                                except LaneChangeCommitmentError as exc:
                                    raise ModelEvaluationError(
                                        f"diffusion proposal acceptance failed: {exc}"
                                    ) from exc
                                rule_maker_ms += (
                                    time.perf_counter() - acceptance_start
                                ) * 1000.0
                                raw["rule_proposal_matches"] += 1
                                raw["rule_accepted_ranks"].append(
                                    int(pending_rule_acceptance.rank)
                                )
                                if rule_maker.has_active_lane_change_commitments:
                                    committed_execution_id = int(rule_batch.batch_id)
                                    committed_plan_actions = joint_proposal_actions(
                                        pending_rule_acceptance, AGENT_IDS
                                    )
                            timing_values = {
                                "bev_build_ms": bev_ms,
                                "model_inference_ms": inference_ms,
                                "control_mapping_ms": control_ms,
                                "planning_tick_ms": max(
                                    0.0,
                                    (time.perf_counter() - tick_start) * 1000.0
                                    - deterministic_probe_ms,
                                ),
                                "trajectory_optimizer_ms": optimizer_elapsed_ms[0],
                                "rule_maker_ms": rule_maker_ms,
                            }
                            for timing_name, timing_value in timing_values.items():
                                raw["timing"][timing_name].append(timing_value)
                                episode_state["timing"][timing_name].append(
                                    timing_value
                                )
                            raw["trajectory_intervention_ade_m"].extend(
                                optimization.intervention_ade_m.tolist()
                            )
                            raw["trajectory_intervention_fde_m"].extend(
                                optimization.intervention_fde_m.tolist()
                            )
                            raw["trajectory_retained_raw_fraction"].extend(
                                optimization.retained_raw_fraction.tolist()
                            )

                        if (
                            artifact_writer is not None
                            and artifact_episode_id is not None
                        ):
                            artifact_writer.capture_topdown_frame(
                                artifact_episode_id,
                                env=env,
                                step_index=step_index,
                                pre_poses=pre_poses,
                                trajectories=trajectories,
                            )
                        if precomputed_controls is None:
                            _, reward, terminated, truncated, info = env.step(action)
                        else:
                            (
                                _,
                                reward,
                                terminated,
                                truncated,
                                info,
                            ) = _step_precomputed_trajectory_controls(
                                env, action, precomputed_controls
                            )
                        executed_steps += 1
                        live_agents = dict(env.agents)
                        _update_episode_state(
                            episode_state,
                            env=env,
                            rule_maker=rule_maker,
                            step_info=info,
                            pre_poses=pre_poses,
                            dt_s=dt_s,
                        )
                        background_gap = episode_state["current_background_gap"]
                        platoon_gap = episode_state["current_platoon_gap"]
                        episode_min_background_gap = min(
                            episode_min_background_gap, background_gap
                        )
                        episode_min_platoon_gap = min(
                            episode_min_platoon_gap, platoon_gap
                        )
                        episode_gap5 |= background_gap < BACKGROUND_GAP_THRESHOLD_M
                        episode_gap7 |= platoon_gap < PLATOON_GAP_THRESHOLD_M
                        formation_values = []
                        for role, agent_id in enumerate(AGENT_IDS):
                            agent_info = info.get(agent_id, {})
                            role_values = raw["roles"][agent_id]
                            crash = any(
                                bool(agent_info.get(key, False))
                                for key in (
                                    "crash",
                                    "crash_vehicle",
                                    "crash_object",
                                    "crash_building",
                                    "crash_human",
                                )
                            )
                            left = bool(agent_info.get("out_of_road", False)) or bool(
                                agent_info.get("out_of_route", False)
                            )
                            episode_collision |= crash
                            episode_out |= left
                            role_collision[agent_id] |= crash
                            role_out[agent_id] |= left
                            role_values["progress"].append(
                                float(agent_info.get("progress", 0.0))
                            )
                            speed = float(
                                agent_info.get(
                                    "speed_km_h",
                                    getattr(
                                        live_agents.get(agent_id), "speed_km_h", 0.0
                                    ),
                                )
                            )
                            role_values["speed_km_h"].append(speed)
                            role_minimum_gap = min(platoon_gap, background_gap)
                            if math.isfinite(role_minimum_gap):
                                role_values["minimum_gap_m"].append(
                                    role_minimum_gap
                                )
                            formation_value = float(
                                agent_info.get("formation_error", 0.0)
                            )
                            formation_values.append(formation_value)
                            role_values["jerk"].append(
                                float(agent_info.get("jerk", 0.0))
                            )
                            role_values["steering_change"].append(
                                float(agent_info.get("delta_steering", 0.0))
                            )
                            prior_speed = previous_speed[agent_id]
                            if prior_speed is not None:
                                role_values["acceleration_mps2"].append(
                                    (speed / 3.6 - prior_speed) / dt_s
                                )
                            previous_speed[agent_id] = speed / 3.6
                            live_agent = live_agents.get(agent_id)
                            if live_agent is not None:
                                heading = float(live_agent.heading_theta)
                                heading_delta = math.atan2(
                                    math.sin(heading - previous_heading[agent_id]),
                                    math.cos(heading - previous_heading[agent_id]),
                                )
                                role_values["yaw_rate_rad_s"].append(
                                    heading_delta / dt_s
                                )
                                previous_heading[agent_id] = heading
                            if selected_modes is not None:
                                mode = int(selected_modes[role])
                                role_values["selected_modes"].append(mode)
                                role_values["stop"] += int(mode == 9)
                                raw["mode_rewards"].setdefault(mode, []).append(
                                    float(reward.get(agent_id, 0.0))
                                )
                        raw["formation_error"].extend(formation_values)
                        raw["formation_spread"].append(
                            float(max(formation_values, default=0.0))
                        )
                        raw["joint_reward"].append(
                            float(
                                np.mean(
                                    [
                                        float(reward.get(agent_id, 0.0))
                                        for agent_id in AGENT_IDS
                                    ]
                                )
                            )
                        )
                        if (
                            artifact_writer is not None
                            and artifact_episode_id is not None
                        ):
                            post_poses = np.asarray(
                                [
                                    (
                                        [
                                            *np.asarray(
                                                live_agents.get(
                                                    agent_id, env.agents.get(agent_id)
                                                ).position,
                                                dtype=np.float64,
                                            )[:2],
                                            float(
                                                live_agents.get(
                                                    agent_id, env.agents.get(agent_id)
                                                ).heading_theta
                                            ),
                                        ]
                                        if live_agents.get(
                                            agent_id, env.agents.get(agent_id)
                                        )
                                        is not None
                                        else pre_poses[role].tolist()
                                    )
                                    for role, agent_id in enumerate(AGENT_IDS)
                                ],
                                dtype=np.float64,
                            )
                            artifact_writer.record_step(
                                artifact_episode_id,
                                step_index=step_index,
                                dt_s=dt_s,
                                pre_poses=pre_poses,
                                post_poses=post_poses,
                                trajectories=trajectories,
                                selected_modes=selected_modes,
                                rewards=reward,
                                controls=getattr(env, "_pending_low_level_actions", {}),
                                bev=step_bev,
                            )
                        if episode_has_ended(terminated, truncated, info):
                            if all(
                                bool(info.get(agent_id, {}).get("arrive_dest", False))
                                for agent_id in AGENT_IDS
                            ):
                                raw["episode_completed"] += 1
                                episode_completed = True
                            break
                    episode_rejections = raw["execution_rejections"][
                        episode_rejection_start:
                    ]
                    rejection_steps = [
                        int(value.get("step", -1)) for value in episode_rejections
                    ]
                    episode_metrics = _finish_episode_metrics(
                        episode_state,
                        model_id=name,
                        scenario=scenario,
                        seed=int(seed),
                        dt_s=dt_s,
                        executed_steps=executed_steps,
                        collision=episode_collision,
                        out_of_road=episode_out,
                        role_collision=role_collision,
                        role_out=role_out,
                        execution_rejected=episode_execution_rejected,
                        rejection_steps=rejection_steps,
                    )
                    raw["episode_metrics"].append(episode_metrics)
                    recovery_time = episode_metrics["cooperation"][
                        "relock_recovery_time_s"
                    ]
                    if recovery_time is not None:
                        raw["recovery_time_s"].append(float(recovery_time))
                    raw["episode_lengths"].append(executed_steps)
                    raw["episode_collision"] += int(episode_collision)
                    raw["episode_out_of_road"] += int(episode_out)
                    raw["gap_5m_violation"] += int(episode_gap5)
                    raw["gap_7m_violation"] += int(episode_gap7)
                    for agent_id in AGENT_IDS:
                        raw["roles"][agent_id]["collision"] += int(
                            role_collision[agent_id]
                        )
                        raw["roles"][agent_id]["out_of_road"] += int(role_out[agent_id])
                    raw["episode_outcomes"].append(
                        {
                            "scenario": scenario[0],
                            "route": scenario[1],
                            "seed": int(seed),
                            "collision": bool(episode_collision),
                            "out_of_road": bool(episode_out),
                            "completed": bool(episode_completed),
                            "execution_rejected": bool(episode_execution_rejected),
                            "gap_5m_violation": bool(episode_gap5),
                            "gap_7m_violation": bool(episode_gap7),
                            "minimum_background_gap_m": (
                                float(episode_min_background_gap)
                                if math.isfinite(episode_min_background_gap)
                                else None
                            ),
                            "minimum_platoon_gap_m": (
                                float(episode_min_platoon_gap)
                                if math.isfinite(episode_min_platoon_gap)
                                else None
                            ),
                            "functional_success_stable": bool(
                                episode_metrics["efficiency"][
                                    "functional_success_stable"
                                ]
                            ),
                        }
                    )
                    if artifact_writer is not None and artifact_episode_id is not None:
                        artifact_writer.finish_episode(artifact_episode_id)
                finally:
                    env.close()
        summary = _summarize(raw, episode_count)
        if artifact_writer is not None:
            summary["artifacts"] = artifact_writer.finalize()
        planning_p95 = summary["timing"]["planning_tick_ms"]["p95_ms"]
        latency_gate_passed = planning_p95 <= cfg.v2_planning_tick_p95_limit_ms
        summary["gate_results"] = {
            "planning_tick_p95_ms": {
                "threshold_ms": cfg.v2_planning_tick_p95_limit_ms,
                "observed_ms": planning_p95,
                "passed": latency_gate_passed,
            }
        }
        summary["online_rule_maker_contract"] = {
            "implementation": "MultiAgentRuleMaker",
            "normal_planner_called": False,
            "planning_tick_p95_limit_ms": cfg.v2_planning_tick_p95_limit_ms,
        }
        if deterministic_probe is None:
            raise ModelEvaluationError(
                f"{name} evaluation never reached a model-ready state"
            )
        summary["deterministic_probe"] = deterministic_probe
        model_reports[name] = summary

    all_gates_passed = all(
        bool(
            model["gate_results"]["planning_tick_p95_ms"]["passed"]
        )
        for model in model_reports.values()
    )
    dt_values = {
        float(episode["dt_s"])
        for model in model_reports.values()
        for episode in model["episode_metrics"]
    }
    if len(dt_values) != 1:
        raise ModelEvaluationError("evaluation decision dt changed across episodes")
    report = {
        "format": "bev_model_evaluation_v3",
        "evaluation_status": (
            "completed" if all_gates_passed else "completed_with_gate_failure"
        ),
        "all_gates_passed": all_gates_passed,
        "run_mode": cfg.run_mode,
        "diagnostic_only": cfg.run_mode != "formal",
        "eligible_for_formal_conclusions": _formal_conclusions_eligible(
            cfg.run_mode, all_gates_passed
        ),
        "common_scenarios": [list(value) for value in cfg.scenarios],
        "common_seeds": list(cfg.seeds),
        "common_noise_seed_by_episode": True,
        "common_initial_state_verified": True,
        "initial_state_sha256": _initial_states_sha256(reference_initial_states),
        "initial_scene_sha256": hashlib.sha256(
            json.dumps(
                sorted(reference_initial_scenes.items()),
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "deterministic_inference": {
            "torch_deterministic_algorithms": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "python_hash_seed": os.environ["PYTHONHASHSEED"],
            "tf32": False,
        },
        "scenario_contract": primary_scenario_contract(cfg.scenarios),
        "evaluation_protocol": {
            "scenarios": [list(value) for value in cfg.scenarios],
            "seeds": list(cfg.seeds),
            "max_steps": int(cfg.max_steps),
            "decision_dt_s": next(iter(dt_values)),
            "statistical_unit": "episode",
            "pairing_key": ["scenario", "route", "seed"],
            "overall_aggregation": "episode_first_equal_weight_scenario_macro",
            "background_gap_threshold_m": BACKGROUND_GAP_THRESHOLD_M,
            "platoon_gap_threshold_m": PLATOON_GAP_THRESHOLD_M,
            "gap_geometry": (
                "shared-corridor oriented-box bumper gap with relative-heading support"
            ),
            "gap_exposure_denominator": "all executed simulator steps",
            "gap_deficit_integral": (
                "sum(max(threshold - minimum in-corridor gap at tick, 0) * dt)"
            ),
            "ttc_observation_threshold_s": TTC_OBSERVATION_THRESHOLD_S,
            "ttc_no_closing_value": None,
            "ttc_definition": (
                "current bumper gap divided by positive closing speed for the same "
                "pair in consecutive shared-corridor samples"
            ),
            "drac_definition": (
                "closing_speed^2 / (2 * max(current_gap, 0.1m))"
            ),
            "route_progress_source": (
                "navigation.travelled_length delta divided by reset-time remaining "
                "navigation.total_length"
            ),
            "functional_success_stable": (
                "earliest successful step whose suffix remains successful through "
                "episode end"
            ),
            "formation_recovery": (
                "RuleMaker locked-to-unlocked transition through subsequent relock; "
                "episodes without unlock are not applicable and unfinished intervals "
                "are censored"
            ),
            "comfort_source": (
                "world pose, speed-equivalent reset velocity, decision dt, and executed "
                "low-level control commands"
            ),
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "rate_confidence_interval": "two-sided 95% Clopper-Pearson",
            "comparison_confidence_interval": (
                "95% scenario-stratified paired bootstrap"
            ),
            "planning_tick_p95_limit_ms": cfg.v2_planning_tick_p95_limit_ms,
            "repeats_are_independent_samples": False,
        },
        "metric_definitions": _metric_definitions(),
        "grpo_reward_binding": common_reward_binding,
        "manifest": str(manifest.path),
        "manifest_sha256": file_sha256(manifest.path),
        "model_order": list(manifest.model_ids),
        "model_checkpoints": {
            spec.model_id: {
                "kind": spec.kind,
                "variant": spec.variant,
                "reward_domain": spec.reward_domain,
                "checkpoint": str(spec.checkpoint),
                "checkpoint_sha256": spec.checkpoint_sha256,
            }
            for spec in manifest.models
        },
        "models": model_reports,
        "comparisons": compare_models(model_reports, manifest.comparisons),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _behavior_sha256(report: Mapping[str, object]) -> str:
    models = report.get("models")
    if not isinstance(models, Mapping):
        raise ModelEvaluationError("evaluation report has no model metrics")
    model_order = report.get("model_order")
    if (
        not isinstance(model_order, list)
        or not model_order
        or any(not isinstance(name, str) for name in model_order)
        or set(model_order) != set(models)
    ):
        raise ModelEvaluationError("evaluation report model_order is invalid")
    behavior = {}
    for name in model_order:
        value = models.get(name)
        if not isinstance(value, Mapping):
            raise ModelEvaluationError(f"evaluation report is missing {name}")
        behavior[name] = _without_timing(value)
    encoded = json.dumps(
        behavior, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _without_timing(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _without_timing(item)
            for key, item in value.items()
            if key not in ("timing", "gate_results", "artifacts")
        }
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


def _conclusion_label(delta: float, tolerance: float) -> str:
    epsilon = 1.0e-12 * max(1.0, abs(delta), abs(tolerance))
    if delta > tolerance + epsilon:
        return "better"
    if delta < -tolerance - epsilon:
        return "worse"
    return "equivalent"


def _exceeds_tolerance(delta: float, tolerance: float) -> bool:
    epsilon = 1.0e-12 * max(1.0, abs(delta), abs(tolerance))
    return delta > tolerance + epsilon


def _repeat_metric_tolerance(
    metric: str, unit: str, cfg: ReproducibilityToleranceConfig
) -> float:
    if unit == "fraction":
        return cfg.rate if metric.endswith("_rate") else cfg.fraction
    if unit == "m":
        return cfg.distance_m
    if unit == "km/h":
        return cfg.speed_km_h
    if unit == "reward":
        return cfg.reward
    return cfg.comfort


def _critical_repeat_episode_paths() -> tuple[tuple[str, ...], ...]:
    paths: list[tuple[str, ...]] = [
        ("safety", "collision"),
        ("safety", "out_of_road"),
        ("safety", "execution_rejected"),
        ("efficiency", "scenario_realized"),
        ("efficiency", "functional_success_final"),
        ("efficiency", "functional_success_ever"),
        ("efficiency", "functional_success_stable"),
        ("cooperation", "relock_recovery_success"),
        ("cooperation", "recovery_censored"),
    ]
    for agent_id in AGENT_IDS:
        paths.extend(
            (
                ("safety", "per_agent", agent_id, "collision"),
                ("safety", "per_agent", agent_id, "out_of_road"),
            )
        )
    return tuple(paths)


def compare_models(
    models: Mapping[str, object],
    comparisons: Sequence[ComparisonSpec],
    tolerance: ReproducibilityToleranceConfig | None = None,
) -> dict[str, object]:
    del tolerance
    policy = _closed_loop_metric_policy()
    results = {}
    for comparison in comparisons:
        baseline = models.get(comparison.baseline)
        candidate = models.get(comparison.candidate)
        if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
            raise ModelEvaluationError("comparison model set is incomplete")
        baseline_episodes = _paired_episode_index(baseline)
        candidate_episodes = _paired_episode_index(candidate)
        if baseline_episodes.keys() != candidate_episodes.keys():
            raise ModelEvaluationError(
                f"comparison {comparison.comparison_id} episode sets do not match"
            )
        metric_results = {}
        for metric, rule in policy.items():
            path = rule["episode_path"]
            direction = str(rule["direction"])
            paired = []
            for key in baseline_episodes:
                baseline_value = _episode_value(baseline_episodes[key], path)
                candidate_value = _episode_value(candidate_episodes[key], path)
                if baseline_value is None or candidate_value is None:
                    continue
                paired.append(
                    (
                        key[0],
                        float(baseline_value),
                        float(candidate_value),
                    )
                )
            baseline_value, candidate_value, raw_delta, ci95 = (
                _paired_scenario_macro_bootstrap(paired, metric=metric)
            )
            metric_results[metric] = {
                "baseline": baseline_value,
                "candidate": candidate_value,
                "candidate_minus_baseline": raw_delta,
                "unit": str(rule["unit"]),
                "direction": direction,
                "n_pairs": len(paired),
                "ci95": ci95,
            }
        results[comparison.comparison_id] = {
            "baseline": comparison.baseline,
            "candidate": comparison.candidate,
            "metrics": metric_results,
        }
    return results


def _paired_episode_index(
    model: Mapping[str, object],
) -> dict[tuple[str, str, int], Mapping[str, object]]:
    episodes = model.get("episode_metrics")
    if not isinstance(episodes, list):
        raise ModelEvaluationError("comparison model has no episode_metrics")
    indexed = {}
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise ModelEvaluationError("episode_metrics row must be an object")
        key = (
            str(episode.get("scenario")),
            str(episode.get("route")),
            int(episode.get("seed")),
        )
        if key in indexed:
            raise ModelEvaluationError(f"duplicate comparison episode {key}")
        indexed[key] = episode
    return indexed


def _paired_scenario_macro_bootstrap(
    paired: Sequence[tuple[str, float, float]], *, metric: str
) -> tuple[float | None, float | None, float | None, list[float] | None]:
    if not paired:
        return None, None, None, None
    grouped: dict[str, list[tuple[float, float]]] = {}
    for scenario, baseline, candidate in paired:
        grouped.setdefault(scenario, []).append((baseline, candidate))
    baseline_value = float(
        np.mean(
            [np.mean([value[0] for value in values]) for values in grouped.values()]
        )
    )
    candidate_value = float(
        np.mean(
            [np.mean([value[1] for value in values]) for values in grouped.values()]
        )
    )
    raw_delta = candidate_value - baseline_value
    seed_offset = int(hashlib.sha256(metric.encode("utf-8")).hexdigest()[:8], 16)
    generator = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    arrays = {
        scenario: np.asarray(values, dtype=np.float64)
        for scenario, values in grouped.items()
    }
    scenario_bootstrap = []
    for values in arrays.values():
        deltas = values[:, 1] - values[:, 0]
        indices = generator.integers(
            0, len(values), size=(BOOTSTRAP_SAMPLES, len(values))
        )
        scenario_bootstrap.append(np.mean(deltas[indices], axis=1))
    bootstrap = np.mean(np.stack(scenario_bootstrap, axis=0), axis=0)
    return (
        baseline_value,
        candidate_value,
        raw_delta,
        [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
    )


def _report_conclusions(
    report: Mapping[str, object], cfg: ReproducibilityToleranceConfig
) -> dict[str, str]:
    comparisons = report.get("comparisons")
    if not isinstance(comparisons, Mapping):
        raise ModelEvaluationError("repeat report has no comparisons")
    conclusions = {}
    for comparison_id, comparison in comparisons.items():
        if not isinstance(comparison, Mapping) or not isinstance(
            comparison.get("metrics"), Mapping
        ):
            raise ModelEvaluationError("repeat report comparison is invalid")
        for metric, metric_result in comparison["metrics"].items():
            if not isinstance(metric_result, Mapping):
                raise ModelEvaluationError("repeat comparison metric is invalid")
            if str(metric).startswith("timing."):
                continue
            delta = metric_result["candidate_minus_baseline"]
            if delta is None:
                conclusions[f"{comparison_id}/{metric}"] = "not_applicable"
                continue
            unit = str(metric_result["unit"])
            tolerance = _repeat_metric_tolerance(str(metric), unit, cfg)
            direction = str(metric_result["direction"])
            if direction == "descriptive":
                conclusions[f"{comparison_id}/{metric}"] = (
                    "stable"
                    if not _exceeds_tolerance(abs(float(delta)), tolerance)
                    else "changed"
                )
            else:
                improvement = float(delta) if direction == "higher" else -float(delta)
                conclusions[f"{comparison_id}/{metric}"] = _conclusion_label(
                    improvement, tolerance
                )
    return conclusions


def compare_repeated_reports(
    reports: Sequence[Mapping[str, object]],
    tolerance: ReproducibilityToleranceConfig | None = None,
) -> dict[str, object]:
    """Compare every v3 non-timing episode metric across fixed-process repeats."""

    if len(reports) < 2:
        raise ModelEvaluationError("repeat comparison requires at least two reports")
    cfg = tolerance or ReproducibilityToleranceConfig()
    baseline = reports[0]
    baseline_models = baseline.get("models")
    if not isinstance(baseline_models, Mapping):
        raise ModelEvaluationError("repeat report has no models")
    model_order = baseline.get("model_order")
    if (
        not isinstance(model_order, list)
        or not model_order
        or any(not isinstance(name, str) for name in model_order)
        or set(model_order) != set(baseline_models)
    ):
        raise ModelEvaluationError("baseline model_order is invalid")
    baseline_conclusions = _report_conclusions(baseline, cfg)
    for repeat_index, report in enumerate(reports[1:], start=2):
        if report.get("model_order") != model_order:
            raise ModelEvaluationError(
                f"repeat {repeat_index} model_order does not match baseline"
            )
        if set(_report_conclusions(report, cfg)) != set(baseline_conclusions):
            raise ModelEvaluationError(
                f"repeat {repeat_index} comparison set does not match baseline"
            )

    initial_hashes = [str(report.get("initial_state_sha256", "")) for report in reports]
    initial_scene_hashes = [
        str(report.get("initial_scene_sha256", "")) for report in reports
    ]
    initial_state_exact = (
        bool(initial_hashes[0])
        and len(set(initial_hashes)) == 1
        and bool(initial_scene_hashes[0])
        and len(set(initial_scene_hashes)) == 1
    )
    critical_mismatches: list[str] = []
    boundary_disagreements: list[str] = []
    continuous_violations: list[str] = []
    probe_evidence: dict[str, object] = {}
    critical_paths = set(_critical_repeat_episode_paths())
    metric_policy = {
        metric: rule
        for metric, rule in _closed_loop_metric_policy().items()
        if not metric.startswith("timing.")
    }

    for model_name in model_order:
        baseline_model = baseline_models.get(model_name)
        if not isinstance(baseline_model, Mapping):
            raise ModelEvaluationError(f"baseline is missing {model_name}")
        baseline_episodes = _paired_episode_index(baseline_model)
        baseline_probe = baseline_model.get("deterministic_probe")
        if not isinstance(baseline_probe, Mapping):
            raise ModelEvaluationError("deterministic probe is missing")
        probe_rows = [baseline_probe]
        for repeat_index, report in enumerate(reports[1:], start=2):
            models = report.get("models")
            model = models.get(model_name) if isinstance(models, Mapping) else None
            if not isinstance(model, Mapping):
                raise ModelEvaluationError(
                    f"repeat {repeat_index} is missing {model_name}"
                )
            episodes = _paired_episode_index(model)
            if episodes.keys() != baseline_episodes.keys():
                critical_mismatches.append(
                    f"{model_name}/repeat_{repeat_index}/episode_set"
                )
            for key in baseline_episodes.keys() & episodes.keys():
                first = baseline_episodes[key]
                second = episodes[key]
                key_label = f"{key[0]}/{key[1]}/{key[2]}"
                for path in critical_paths:
                    if _episode_value(first, path) != _episode_value(second, path):
                        critical_mismatches.append(
                            f"{model_name}/repeat_{repeat_index}/{key_label}/"
                            f"{'.'.join(path)}"
                        )
                gap_boundary_paths = (
                    (
                        ("safety", "background_gap_violation"),
                        ("safety", "minimum_background_gap_m"),
                        BACKGROUND_GAP_THRESHOLD_M,
                    ),
                    (
                        ("safety", "platoon_gap_violation"),
                        ("safety", "minimum_platoon_gap_m"),
                        PLATOON_GAP_THRESHOLD_M,
                    ),
                )
                for flag_path, distance_path, threshold in gap_boundary_paths:
                    first_flag = _episode_value(first, flag_path)
                    second_flag = _episode_value(second, flag_path)
                    if first_flag == second_flag:
                        continue
                    distances = (
                        _episode_value(first, distance_path),
                        _episode_value(second, distance_path),
                    )
                    target = (
                        boundary_disagreements
                        if all(
                            value is not None
                            and abs(float(value) - threshold) <= cfg.distance_m
                            for value in distances
                        )
                        else critical_mismatches
                    )
                    target.append(
                        f"{model_name}/repeat_{repeat_index}/{key_label}/"
                        f"{'.'.join(flag_path)}"
                    )
                for metric, rule in metric_policy.items():
                    path = tuple(rule["episode_path"])
                    if path in critical_paths or path in {
                        ("safety", "background_gap_violation"),
                        ("safety", "platoon_gap_violation"),
                    }:
                        continue
                    first_value = _episode_value(first, path)
                    second_value = _episode_value(second, path)
                    if first_value is None and second_value is None:
                        continue
                    allowed = _repeat_metric_tolerance(
                        metric, str(rule["unit"]), cfg
                    )
                    if first_value is None or second_value is None:
                        continuous_violations.append(
                            f"{model_name}/repeat_{repeat_index}/{key_label}/{metric}="
                            "applicability_changed"
                        )
                        continue
                    delta = abs(float(first_value) - float(second_value))
                    if _exceeds_tolerance(delta, allowed):
                        continuous_violations.append(
                            f"{model_name}/repeat_{repeat_index}/{key_label}/{metric}="
                            f"{delta:.6g}>{allowed:.6g}"
                        )
            probe = model.get("deterministic_probe")
            if not isinstance(probe, Mapping):
                raise ModelEvaluationError("deterministic probe is missing")
            probe_rows.append(probe)
        input_hashes = [
            str(value.get("input_and_noise_sha256", "")) for value in probe_rows
        ]
        output_hashes = [str(value.get("output_sha256", "")) for value in probe_rows]
        probe_evidence[model_name] = {
            "identical_replay_in_every_run": all(
                value.get("identical_replay") is True for value in probe_rows
            ),
            "cross_repeat_input_exact": bool(input_hashes[0])
            and len(set(input_hashes)) == 1,
            "cross_repeat_output_exact": bool(output_hashes[0])
            and len(set(output_hashes)) == 1,
            "input_and_noise_sha256": input_hashes,
            "output_sha256": output_hashes,
        }

    conclusions = [_report_conclusions(report, cfg) for report in reports]
    conclusion_changes = {
        name: [value[name] for value in conclusions]
        for name in conclusions[0]
        if len({value[name] for value in conclusions}) != 1
    }
    deterministic_model_replay = all(
        value["identical_replay_in_every_run"] for value in probe_evidence.values()
    )
    return {
        "contract": dataclasses.asdict(cfg),
        "initial_state_exact": initial_state_exact,
        "initial_state_sha256": initial_hashes,
        "initial_scene_sha256": initial_scene_hashes,
        "deterministic_model_replay": deterministic_model_replay,
        "model_probe_evidence": probe_evidence,
        "critical_discrete_outcomes_exact": not critical_mismatches,
        "critical_mismatches": critical_mismatches,
        "threshold_boundary_disagreements": boundary_disagreements,
        "continuous_metrics_within_tolerance": not continuous_violations,
        "continuous_tolerance_violations": continuous_violations,
        "statistical_conclusions_stable": not conclusion_changes,
        "conclusion_changes": conclusion_changes,
        "conclusions_by_repeat": conclusions,
        "tolerance_gate_passed": (
            initial_state_exact
            and deterministic_model_replay
            and not critical_mismatches
            and not continuous_violations
            and not conclusion_changes
        ),
    }


def evaluate_models_repeated(
    manifest_path: Path,
    output_path: Path,
    config: ModelEvaluationConfig,
    *,
    repeats: int,
    tolerance: ReproducibilityToleranceConfig | None = None,
) -> dict[str, object]:
    """Run complete evaluations sequentially inside one fixed process."""

    if isinstance(repeats, bool) or repeats < 2:
        raise ModelEvaluationError("fixed-process repeats must be at least two")
    output = Path(output_path)
    reports = []
    hashes = []
    for repeat_index in range(repeats):
        repeat_output = output.with_name(
            f"{output.stem}.repeat_{repeat_index + 1}{output.suffix}"
        )
        repeat_config = (
            dataclasses.replace(
                config,
                artifact_root=Path(config.artifact_root) / f"repeat_{repeat_index + 1}",
            )
            if config.artifact_root is not None
            else config
        )
        report = evaluate_models(manifest_path, repeat_output, repeat_config)
        reports.append(report)
        hashes.append(_behavior_sha256(report))
    exact_match = len(set(hashes)) == 1
    tolerance_comparison = compare_repeated_reports(reports, tolerance)
    all_gates_passed = all(
        bool(report["all_gates_passed"]) for report in reports
    ) and bool(tolerance_comparison["tolerance_gate_passed"])
    combined = {
        "format": "bev_model_fixed_process_repeat_v3",
        "evaluation_status": (
            "completed" if all_gates_passed else "completed_with_gate_failure"
        ),
        "all_gates_passed": all_gates_passed,
        "run_mode": config.run_mode,
        "diagnostic_only": config.run_mode != "formal",
        "eligible_for_formal_conclusions": _formal_conclusions_eligible(
            config.run_mode, all_gates_passed
        ),
        "repeat_count": int(repeats),
        "fixed_process_reproducibility": {
            "exact_behavior_match": exact_match,
            "behavior_sha256": hashes,
            "timing_excluded_from_hash": True,
            "tolerance_comparison": tolerance_comparison,
        },
        "model_order": reports[0]["model_order"],
        "evaluation_protocol": reports[0]["evaluation_protocol"],
        "metric_definitions": reports[0]["metric_definitions"],
        "grpo_reward_binding": reports[0].get("grpo_reward_binding"),
        "models": reports[0]["models"],
        "comparisons": reports[0]["comparisons"],
        "repeat_reports": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--run-mode", choices=("diagnostic", "formal"), default="diagnostic"
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--visualization-interval", type=int, default=1)
    parser.add_argument("--topdown-screen-size", type=int, default=800)
    parser.add_argument("--topdown-film-size", type=int, default=3000)
    arguments = parser.parse_args()
    config = ModelEvaluationConfig(
        run_mode=arguments.run_mode,
        device=arguments.device,
        seeds=(FORMAL_EVAL_SEEDS if arguments.run_mode == "formal" else HOLDOUT_SEEDS),
        scenarios=(
            FORMAL_EVAL_SCENARIOS
            if arguments.run_mode == "formal"
            else DIAGNOSTIC_EVAL_SCENARIOS
        ),
        max_steps=arguments.max_steps,
        artifact_root=arguments.artifact_root,
        save_visualizations=arguments.save_visualizations,
        video_fps=arguments.video_fps,
        visualization_interval=arguments.visualization_interval,
        topdown_screen_size=arguments.topdown_screen_size,
        topdown_film_size=arguments.topdown_film_size,
    )
    report = (
        evaluate_models(
            arguments.manifest,
            arguments.output,
            config,
        )
        if arguments.repeats == 1
        else evaluate_models_repeated(
            arguments.manifest,
            arguments.output,
            config,
            repeats=arguments.repeats,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_gates_passed"] else GATE_FAILURE_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
