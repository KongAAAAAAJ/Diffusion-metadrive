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
from typing import Literal, Mapping, Sequence

import numpy as np

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
from models.platoon_planner.platoon_normal_planner import PlatoonNormalPlanner
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


class ModelEvaluationError(RuntimeError):
    """Raised when a model evaluation is incomplete or unfair."""


@dataclass(frozen=True)
class ModelEvaluationConfig:
    run_mode: Literal["diagnostic", "formal"] = "diagnostic"
    device: str = "cuda"
    seeds: tuple[int, ...] = HOLDOUT_SEEDS
    scenarios: tuple[tuple[str, str], ...] = DIAGNOSTIC_EVAL_SCENARIOS
    max_steps: int = 100
    inference_p95_limit_ms: float = 100.0
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
            not math.isfinite(self.inference_p95_limit_ms)
            or self.inference_p95_limit_ms <= 0.0
        ):
            raise ModelEvaluationError(
                "inference_p95_limit_ms must be positive and finite"
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
        },
        "trajectory_intervention_ade_m": [],
        "trajectory_intervention_fde_m": [],
        "trajectory_retained_raw_fraction": [],
        "execution_modes_removed": 0,
        "execution_mode_masks_built": 0,
        "episode_outcomes": [],
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
    timing = {
        name: {
            "p50_ms": _percentile(values, 50),
            "p95_ms": _percentile(values, 95),
        }
        for name, values in raw["timing"].items()
    }
    joint_reward = list(raw["joint_reward"])
    legacy_mode_rewards = {
        str(mode): {
            "count": len(values),
            "mean_reward": float(np.mean(values)) if values else 0.0,
        }
        for mode, values in sorted(raw.get("mode_rewards", {}).items())
    }
    return {
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
            "rejection_count": len(raw.get("execution_rejections", ())),
            "rejection_rate": len(raw.get("execution_rejections", ())) / episode_count,
            "rejections": list(raw.get("execution_rejections", ())),
            "mean_modes_removed_per_joint_state": (
                raw.get("execution_modes_removed", 0)
                / max(raw.get("execution_mode_masks_built", 0), 1)
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


def _minimum_background_gap(env: object) -> float:
    helper = PlatoonNormalPlanner()
    platoon_objects = set(id(value) for value in env.agents.values())
    minimum = float("inf")
    for _, other in helper._surrounding_vehicles(env):
        if id(other) in platoon_objects:
            continue
        other_position = np.asarray(other.position, dtype=np.float64)[:2]
        other_length = float(getattr(other, "LENGTH", 5.74))
        for agent_id in AGENT_IDS:
            if agent_id not in env.agents:
                continue
            position = np.asarray(env.agents[agent_id].position, dtype=np.float64)[:2]
            minimum = min(
                minimum,
                float(np.linalg.norm(position - other_position))
                - 0.5 * (5.74 + other_length),
            )
    return minimum


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

    helper = PlatoonNormalPlanner()
    agent_objects = {
        id(vehicle): agent_id
        for agent_id, vehicle in (getattr(env, "agents", {}) or {}).items()
    }
    rows = []
    engine = getattr(env, "engine", None)
    get_policy = getattr(engine, "get_policy", None)
    for _, vehicle in helper._surrounding_vehicles(env):
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
                recovered_at = None
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
                        step_bev = None
                        builder.capture_state(env, step_index * dt_s)
                        if not builder.history_ready():
                            action = constant_velocity_actions(env)
                            selected_modes = None
                        else:
                            tick_start = time.perf_counter()
                            bev_start = tick_start
                            values = builder.build_model_inputs(env)
                            step_bev = values.bev
                            bev_ms = (time.perf_counter() - bev_start) * 1000.0
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
                            if deterministic_probe is None:
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
                            raw_trajectories = (
                                output["selected_trajectory"][0].detach().cpu().numpy()
                            )
                            selected_mode_array = (
                                output["selected_mode"][0].detach().cpu().numpy()
                            )
                            try:
                                optimization = optimize_selected_model_trajectories(
                                    values,
                                    raw_trajectories,
                                    selected_mode_array,
                                    optimizer=trajectory_optimizer,
                                )
                            except TrajectoryOptimizationError as exc:
                                # An unprojectable policy output is a measured
                                # closed-loop planning failure. End this episode
                                # without executing a fallback, while preserving
                                # a complete and comparable four-model report.
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
                                break
                            trajectories = optimization.optimized_trajectories
                            selected_modes = selected_mode_array.tolist()
                            action = joint_trajectory_action(trajectories)
                            control_start = time.perf_counter()
                            for agent_id in AGENT_IDS:
                                try:
                                    control = env.trajectory_to_control(
                                        agent_id, action[agent_id]
                                    )
                                except Exception as exc:
                                    raise ModelEvaluationError(
                                        "trajectory control failed for "
                                        f"model={name} scenario={scenario[0]} "
                                        f"seed={seed} step={step_index} "
                                        f"agent={agent_id}: {exc}"
                                    ) from exc
                                if not np.isfinite(control).all():
                                    raise ModelEvaluationError(
                                        "trajectory control is non-finite"
                                    )
                            control_ms = (time.perf_counter() - control_start) * 1000.0
                            raw["timing"]["bev_build_ms"].append(bev_ms)
                            raw["timing"]["model_inference_ms"].append(inference_ms)
                            raw["timing"]["control_mapping_ms"].append(control_ms)
                            raw["timing"]["planning_tick_ms"].append(
                                (time.perf_counter() - tick_start) * 1000.0
                            )
                            raw["timing"]["trajectory_optimizer_ms"].append(
                                optimization.elapsed_ms
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
                        _, reward, terminated, truncated, info = env.step(action)
                        executed_steps += 1
                        live_agents = dict(env.agents)
                        poses = {
                            agent_id: np.asarray(
                                live_agents[agent_id].position, dtype=np.float64
                            )
                            for agent_id in AGENT_IDS
                            if agent_id in live_agents
                        }
                        adjacent_gaps = []
                        for first, second in zip(AGENT_IDS, AGENT_IDS[1:]):
                            if first in poses and second in poses:
                                adjacent_gaps.append(
                                    float(
                                        np.linalg.norm(poses[first] - poses[second])
                                        - 5.74
                                    )
                                )
                            else:
                                # Removal is represented by the authoritative
                                # crash/out/arrival terminal flags below.  Do
                                # not manufacture a geometric gap sample after
                                # MetaDrive has deleted an agent object.
                                adjacent_gaps.append(float("inf"))
                        background_gap = _minimum_background_gap(env)
                        episode_min_background_gap = min(
                            episode_min_background_gap, background_gap
                        )
                        episode_min_platoon_gap = min(
                            episode_min_platoon_gap, min(adjacent_gaps)
                        )
                        episode_gap5 |= background_gap < 5.0
                        episode_gap7 |= min(adjacent_gaps) < 7.0
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
                            role_values["minimum_gap_m"].append(
                                min(min(adjacent_gaps), background_gap)
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
                        if (
                            recovered_at is None
                            and max(formation_values, default=0.0) <= 2.0
                        ):
                            recovered_at = step_index * dt_s
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
                    raw["recovery_time_s"].append(
                        float(
                            recovered_at
                            if recovered_at is not None
                            else cfg.max_steps * dt_s
                        )
                    )
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
                        }
                    )
                    if artifact_writer is not None and artifact_episode_id is not None:
                        artifact_writer.finish_episode(artifact_episode_id)
                finally:
                    env.close()
        summary = _summarize(raw, episode_count)
        if artifact_writer is not None:
            summary["artifacts"] = artifact_writer.finalize()
        inference_p95 = summary["timing"]["model_inference_ms"]["p95_ms"]
        if inference_p95 > cfg.inference_p95_limit_ms:
            raise ModelEvaluationError(
                f"{name} three-role inference P95 {inference_p95:.2f}ms exceeds "
                f"{cfg.inference_p95_limit_ms:.2f}ms"
            )
        if deterministic_probe is None:
            raise ModelEvaluationError(
                f"{name} evaluation never reached a model-ready state"
            )
        summary["deterministic_probe"] = deterministic_probe
        model_reports[name] = summary

    report = {
        "format": "bev_model_evaluation_v2",
        "run_mode": cfg.run_mode,
        "diagnostic_only": cfg.run_mode != "formal",
        "eligible_for_formal_conclusions": cfg.run_mode == "formal",
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
        "comparison_policy": _closed_loop_metric_policy(),
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
        behavior[name] = {
            key: item
            for key, item in value.items()
            if key not in ("timing", "artifacts")
        }
    encoded = json.dumps(
        behavior, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nested_float(value: Mapping[str, object], path: Sequence[str]) -> float:
    current: object = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise ModelEvaluationError(
                f"repeat report is missing metric {'.'.join(path)}"
            )
        current = current[key]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise ModelEvaluationError(f"repeat metric {'.'.join(path)} is not numeric")
    result = float(current)
    if not math.isfinite(result):
        raise ModelEvaluationError(f"repeat metric {'.'.join(path)} is non-finite")
    return result


def _normalized_mode_distribution(
    model: Mapping[str, object], agent_id: str
) -> dict[str, float]:
    roles = model.get("roles")
    if not isinstance(roles, Mapping) or not isinstance(roles.get(agent_id), Mapping):
        raise ModelEvaluationError(f"repeat report is missing role {agent_id}")
    distribution = roles[agent_id].get("mode_distribution")
    if not isinstance(distribution, Mapping):
        raise ModelEvaluationError("mode_distribution is missing")
    counts = {str(key): int(value) for key, value in distribution.items()}
    total = sum(counts.values())
    return {key: value / max(total, 1) for key, value in counts.items()}


def _episode_outcome_index(
    model: Mapping[str, object],
) -> dict[tuple[str, str, int], Mapping[str, object]]:
    outcomes = model.get("episode_outcomes")
    if not isinstance(outcomes, list):
        raise ModelEvaluationError("repeat report has no episode outcomes")
    indexed = {}
    for item in outcomes:
        if not isinstance(item, Mapping):
            raise ModelEvaluationError("episode outcome must be an object")
        key = (
            str(item.get("scenario")),
            str(item.get("route")),
            int(item.get("seed")),
        )
        if key in indexed:
            raise ModelEvaluationError(f"duplicate episode outcome {key}")
        indexed[key] = item
    return indexed


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


def _closed_loop_metric_policy(
    tolerance: ReproducibilityToleranceConfig | None = None,
) -> dict[str, dict[str, object]]:
    cfg = tolerance or ReproducibilityToleranceConfig()
    return {
        "joint_safety.collision_rate": {
            "path": ("joint_safety", "collision_rate"),
            "direction": "lower",
            "equivalence_tolerance": 0.0,
        },
        "joint_safety.out_of_road_rate": {
            "path": ("joint_safety", "out_of_road_rate"),
            "direction": "lower",
            "equivalence_tolerance": 0.0,
        },
        "joint_safety.gap_5m_violation_rate": {
            "path": ("joint_safety", "gap_5m_violation_rate"),
            "direction": "lower",
            "equivalence_tolerance": cfg.rate,
        },
        "joint_safety.gap_7m_violation_rate": {
            "path": ("joint_safety", "gap_7m_violation_rate"),
            "direction": "lower",
            "equivalence_tolerance": cfg.rate,
        },
        "formation.mean_error_m": {
            "path": ("formation", "mean_error_m"),
            "direction": "lower",
            "equivalence_tolerance": cfg.distance_m,
        },
        "efficiency.completion_rate": {
            "path": ("efficiency", "completion_rate"),
            "direction": "higher",
            "equivalence_tolerance": cfg.rate,
        },
        "efficiency.joint_reward_mean": {
            "path": ("efficiency", "joint_reward_mean"),
            "direction": "higher",
            "equivalence_tolerance": cfg.reward,
        },
    }


def compare_models(
    models: Mapping[str, object],
    comparisons: Sequence[ComparisonSpec],
    tolerance: ReproducibilityToleranceConfig | None = None,
) -> dict[str, object]:
    policy = _closed_loop_metric_policy(tolerance)
    results = {}
    for comparison in comparisons:
        baseline = models.get(comparison.baseline)
        candidate = models.get(comparison.candidate)
        if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
            raise ModelEvaluationError("comparison model set is incomplete")
        metric_results = {}
        for metric, rule in policy.items():
            path = rule["path"]
            direction = str(rule["direction"])
            equivalence_tolerance = float(rule["equivalence_tolerance"])
            baseline_value = _nested_float(baseline, path)
            candidate_value = _nested_float(candidate, path)
            raw_delta = candidate_value - baseline_value
            improvement = raw_delta if direction == "higher" else -raw_delta
            metric_results[metric] = {
                "baseline": baseline_value,
                "candidate": candidate_value,
                "candidate_minus_baseline": raw_delta,
                "improvement": improvement,
                "direction": direction,
                "equivalence_tolerance": equivalence_tolerance,
                "conclusion": _conclusion_label(improvement, equivalence_tolerance),
            }
        results[comparison.comparison_id] = {
            "baseline": comparison.baseline,
            "candidate": comparison.candidate,
            "metrics": metric_results,
        }
    return results


def _report_comparison_specs(
    report: Mapping[str, object],
) -> tuple[ComparisonSpec, ...]:
    values = report.get("comparisons")
    if not isinstance(values, Mapping):
        raise ModelEvaluationError("evaluation report comparisons are missing")
    specs = []
    for comparison_id, value in values.items():
        if not isinstance(comparison_id, str) or not isinstance(value, Mapping):
            raise ModelEvaluationError("evaluation report comparison is invalid")
        baseline = value.get("baseline")
        candidate = value.get("candidate")
        if not isinstance(baseline, str) or not isinstance(candidate, str):
            raise ModelEvaluationError("evaluation report comparison ids are invalid")
        specs.append(ComparisonSpec(comparison_id, baseline, candidate))
    return tuple(specs)


def _report_conclusions(
    report: Mapping[str, object], cfg: ReproducibilityToleranceConfig
) -> dict[str, str]:
    models = report.get("models")
    if not isinstance(models, Mapping):
        raise ModelEvaluationError("repeat report has no models")
    comparisons = compare_models(models, _report_comparison_specs(report), cfg)
    return {
        f"{comparison_id}/{metric}": str(metric_result["conclusion"])
        for comparison_id, comparison in comparisons.items()
        for metric, metric_result in comparison["metrics"].items()
    }


def compare_repeated_reports(
    reports: Sequence[Mapping[str, object]],
    tolerance: ReproducibilityToleranceConfig | None = None,
) -> dict[str, object]:
    """Compare closed-loop repeats without requiring bit-exact physics."""

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
    for repeat_index, report in enumerate(reports[1:], start=2):
        if report.get("model_order") != model_order:
            raise ModelEvaluationError(
                f"repeat {repeat_index} model_order does not match baseline"
            )
        if set(_report_conclusions(report, cfg)) != set(
            _report_conclusions(baseline, cfg)
        ):
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
    critical_mismatches = []
    boundary_disagreements = []
    continuous_violations = []
    probe_evidence = {}

    scalar_metrics = (
        (("joint_safety", "gap_5m_violation_rate"), cfg.rate),
        (("joint_safety", "gap_7m_violation_rate"), cfg.rate),
        (("formation", "mean_error_m"), cfg.distance_m),
        (("formation", "p95_error_m"), cfg.distance_m),
        (("formation", "maximum_spread_m"), cfg.distance_m),
        (("formation", "recovery_time_mean_s"), cfg.comfort),
        (("efficiency", "joint_reward_mean"), cfg.reward),
        (("execution", "mean_modes_removed_per_joint_state"), cfg.fraction),
        (("trajectory_optimization", "intervention_ade_mean_m"), cfg.distance_m),
        (("trajectory_optimization", "intervention_fde_mean_m"), cfg.distance_m),
        (("trajectory_optimization", "retained_raw_fraction_mean"), cfg.fraction),
    )
    role_metrics = (
        ("progress_mean_m", cfg.distance_m),
        ("speed_mean_km_h", cfg.speed_km_h),
        ("minimum_gap_m", cfg.distance_m),
        ("stop_rate", cfg.fraction),
        ("acceleration_abs_mean_mps2", cfg.comfort),
        ("jerk_abs_mean", cfg.comfort),
        ("yaw_rate_abs_mean_rad_s", cfg.comfort),
        ("steering_change_abs_mean", cfg.comfort),
    )

    for model_name in model_order:
        baseline_model = baseline_models.get(model_name)
        if not isinstance(baseline_model, Mapping):
            raise ModelEvaluationError(f"baseline is missing {model_name}")
        baseline_outcomes = _episode_outcome_index(baseline_model)
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
            outcomes = _episode_outcome_index(model)
            if outcomes.keys() != baseline_outcomes.keys():
                critical_mismatches.append(
                    f"{model_name}/repeat_{repeat_index}/episode_set"
                )
            for key in baseline_outcomes.keys() & outcomes.keys():
                first = baseline_outcomes[key]
                second = outcomes[key]
                for field in (
                    "collision",
                    "out_of_road",
                    "completed",
                    "execution_rejected",
                ):
                    if bool(first.get(field)) != bool(second.get(field)):
                        critical_mismatches.append(
                            f"{model_name}/repeat_{repeat_index}/{key}/{field}"
                        )
                for field, threshold in (
                    ("gap_5m_violation", 5.0),
                    ("gap_7m_violation", 7.0),
                ):
                    if bool(first.get(field)) != bool(second.get(field)):
                        distance_field = (
                            "minimum_background_gap_m"
                            if field == "gap_5m_violation"
                            else "minimum_platoon_gap_m"
                        )
                        values = (first.get(distance_field), second.get(distance_field))
                        near_boundary = all(
                            isinstance(value, (int, float))
                            and abs(float(value) - threshold) <= cfg.distance_m
                            for value in values
                        )
                        target = (
                            boundary_disagreements
                            if near_boundary
                            else critical_mismatches
                        )
                        target.append(
                            f"{model_name}/repeat_{repeat_index}/{key}/{field}"
                        )
                for distance_field in (
                    "minimum_background_gap_m",
                    "minimum_platoon_gap_m",
                ):
                    first_value = first.get(distance_field)
                    second_value = second.get(distance_field)
                    if first_value is None and second_value is None:
                        continue
                    if (
                        first_value is None
                        or second_value is None
                        or _exceeds_tolerance(
                            abs(float(first_value) - float(second_value)),
                            cfg.distance_m,
                        )
                    ):
                        continuous_violations.append(
                            f"{model_name}/repeat_{repeat_index}/{key}/{distance_field}"
                        )
            for path, allowed in scalar_metrics:
                delta = abs(
                    _nested_float(model, path) - _nested_float(baseline_model, path)
                )
                if _exceeds_tolerance(delta, allowed):
                    continuous_violations.append(
                        f"{model_name}/repeat_{repeat_index}/{'.'.join(path)}={delta:.6g}>{allowed:.6g}"
                    )
            for agent_id in AGENT_IDS:
                for metric, allowed in role_metrics:
                    path = ("roles", agent_id, metric)
                    delta = abs(
                        _nested_float(model, path) - _nested_float(baseline_model, path)
                    )
                    if _exceeds_tolerance(delta, allowed):
                        continuous_violations.append(
                            f"{model_name}/repeat_{repeat_index}/{'.'.join(path)}={delta:.6g}>{allowed:.6g}"
                        )
                first_modes = _normalized_mode_distribution(baseline_model, agent_id)
                second_modes = _normalized_mode_distribution(model, agent_id)
                total_variation = 0.5 * sum(
                    abs(first_modes.get(mode, 0.0) - second_modes.get(mode, 0.0))
                    for mode in first_modes.keys() | second_modes.keys()
                )
                if _exceeds_tolerance(total_variation, cfg.fraction):
                    continuous_violations.append(
                        f"{model_name}/repeat_{repeat_index}/{agent_id}/mode_total_variation="
                        f"{total_variation:.6g}>{cfg.fraction:.6g}"
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
    combined = {
        "format": "bev_model_fixed_process_repeat_v2",
        "run_mode": config.run_mode,
        "diagnostic_only": config.run_mode != "formal",
        "eligible_for_formal_conclusions": config.run_mode == "formal",
        "repeat_count": int(repeats),
        "fixed_process_reproducibility": {
            "exact_behavior_match": exact_match,
            "behavior_sha256": hashes,
            "timing_excluded_from_hash": True,
            "tolerance_comparison": tolerance_comparison,
        },
        "model_order": reports[0]["model_order"],
        "grpo_reward_binding": reports[0].get("grpo_reward_binding"),
        "models": reports[0]["models"],
        "comparison_policy": reports[0]["comparison_policy"],
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
