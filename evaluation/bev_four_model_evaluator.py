"""Fair closed-loop evaluation for Stage 1 and GRPO Variants A/B."""

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
from typing import Literal, Mapping

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
    episode_has_ended,
    joint_trajectory_action,
    model_inputs_to_batch,
    optimize_selected_model_trajectories,
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


MODEL_NAMES = ("A", "B", "A_GRPO", "B_GRPO")
DIAGNOSTIC_EVAL_SCENARIOS = PRIMARY_S5_S9_SCENARIOS
FORMAL_EVAL_SCENARIOS = PRIMARY_S5_S9_SCENARIOS
FORMAL_EVAL_SEEDS = (17, 23, 31, 47, 59)


class FourModelEvaluationError(RuntimeError):
    """Raised when a four-model evaluation is incomplete or unfair."""


@dataclass(frozen=True)
class FourModelEvaluationConfig:
    run_mode: Literal["diagnostic", "formal"] = "diagnostic"
    device: str = "cuda"
    seeds: tuple[int, ...] = HOLDOUT_SEEDS
    scenarios: tuple[tuple[str, str], ...] = DIAGNOSTIC_EVAL_SCENARIOS
    max_steps: int = 100
    inference_p95_limit_ms: float = 100.0

    def __post_init__(self) -> None:
        if self.run_mode not in ("diagnostic", "formal"):
            raise FourModelEvaluationError(
                "evaluation run_mode must be diagnostic or formal"
            )
        if self.device not in ("cpu", "cuda"):
            raise FourModelEvaluationError("evaluation device must be cpu or cuda")
        if (
            not self.seeds
            or any(isinstance(value, bool) or not isinstance(value, int) for value in self.seeds)
        ):
            raise FourModelEvaluationError("evaluation seeds must be integers")
        if not self.scenarios:
            raise FourModelEvaluationError("evaluation scenarios cannot be empty")
        try:
            primary_scenario_contract(self.scenarios)
        except BEVScenarioContractError as exc:
            raise FourModelEvaluationError(str(exc)) from exc
        if isinstance(self.max_steps, bool) or self.max_steps <= 0:
            raise FourModelEvaluationError("max_steps must be positive")
        if (
            not math.isfinite(self.inference_p95_limit_ms)
            or self.inference_p95_limit_ms <= 0.0
        ):
            raise FourModelEvaluationError(
                "inference_p95_limit_ms must be positive and finite"
            )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _configure_deterministic_inference(device: torch.device) -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise FourModelEvaluationError(
            "four-model evaluation requires PYTHONHASHSEED=0 at process startup"
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


def _load_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FourModelEvaluationError(f"invalid model manifest: {path}") from exc
    if not isinstance(payload, dict) or payload.get("format") != "bev_four_model_manifest_v1":
        raise FourModelEvaluationError("four-model manifest format mismatch")
    models = payload.get("models")
    if not isinstance(models, Mapping) or set(models) != set(MODEL_NAMES):
        raise FourModelEvaluationError(
            "manifest must contain exactly A, B, A_GRPO and B_GRPO"
        )
    return payload


def _file_sha256(path: Path) -> str:
    try:
        stream = path.open("rb")
    except OSError as exc:
        raise FourModelEvaluationError(f"unable to read checkpoint: {path}") from exc
    digest = hashlib.sha256()
    with stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checkpoint_hash(
    spec: Mapping[str, object], field: str, hash_field: str
) -> Path:
    path = Path(str(spec.get(field, "")))
    expected = spec.get(hash_field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise FourModelEvaluationError(f"manifest {hash_field} is invalid")
    if _file_sha256(path) != expected:
        raise FourModelEvaluationError(f"manifest {field} SHA256 mismatch")
    return path


def _load_policy(
    name: str,
    spec: Mapping[str, object],
    *,
    device: torch.device,
    formal: bool,
):
    expected_variant = "A" if name.startswith("A") else "B"
    expected_kind = "grpo" if name.endswith("_GRPO") else "stage1"
    if spec.get("variant") != expected_variant or spec.get("kind") != expected_kind:
        raise FourModelEvaluationError(f"{name} manifest type mismatch")
    checkpoint = _validate_checkpoint_hash(
        spec, "checkpoint", "checkpoint_sha256"
    )
    source_checkpoint = (
        checkpoint
        if expected_kind == "stage1"
        else _validate_checkpoint_hash(
            spec, "source_checkpoint", "source_checkpoint_sha256"
        )
    )
    loader = (
        load_stage1_a_for_grpo
        if expected_variant == "A"
        else load_stage1_b_for_grpo
    )
    trainer, source_payload, source_sha = loader(
        source_checkpoint,
        device=device,
        allow_diagnostic_source=not formal,
    )
    if formal and source_payload.get("eligible_for_formal_training") is not True:
        raise FourModelEvaluationError(
            f"{name} source is not eligible for formal evaluation"
        )
    if expected_kind == "grpo":
        checkpoint_loader = (
            load_grpo_checkpoint
            if expected_variant == "A"
            else load_grpo_b_checkpoint
        )
        grpo_payload = checkpoint_loader(
            checkpoint,
            trainer,
            expected_source_stage1_sha256=source_sha,
        )
        for field in (
            "run_mode",
            "reward_config",
            "calibration_report_sha256",
            "scenario_seeds",
            "environment_steps",
            "scenario_contract_sha256",
            "trajectory_optimizer_config",
            "trajectory_optimizer_sha256",
        ):
            if field not in grpo_payload:
                raise FourModelEvaluationError(
                    f"{name} is not an online-calibrated GRPO checkpoint"
                )
        optimizer_config = KinematicTrajectoryOptimizerConfig()
        if (
            grpo_payload.get("trajectory_optimizer_config")
            != dataclasses.asdict(optimizer_config)
            or grpo_payload.get("trajectory_optimizer_sha256")
            != optimizer_config.sha256()
        ):
            raise FourModelEvaluationError(
                f"{name} trajectory optimizer contract mismatch"
            )
        if formal and grpo_payload.get("eligible_for_formal_training") is not True:
            raise FourModelEvaluationError(
                f"{name} checkpoint is diagnostic-only"
            )
        expected_contract = primary_scenario_contract()
        if (
            grpo_payload.get("scenario_contract_sha256")
            != expected_contract["sha256"]
        ):
            raise FourModelEvaluationError(
                f"{name} scenario contract no longer matches frozen S5--S9"
            )
    trainer.planner.eval()
    return trainer.planner


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
    }


def _summarize(raw: dict[str, object], episode_count: int) -> dict[str, object]:
    roles = {}
    for agent_id, values in raw["roles"].items():
        modes = values["selected_modes"]
        roles[agent_id] = {
            "collision_rate": values["collision"] / episode_count,
            "out_of_road_rate": values["out_of_road"] / episode_count,
            "progress_mean_m": float(np.mean(values["progress"])) if values["progress"] else 0.0,
            "speed_mean_km_h": float(np.mean(values["speed_km_h"])) if values["speed_km_h"] else 0.0,
            "minimum_gap_m": min(values["minimum_gap_m"], default=1.0e6),
            "mode_distribution": {
                str(mode): int(modes.count(mode)) for mode in sorted(set(modes))
            },
            "stop_rate": values["stop"] / max(len(modes), 1),
            "acceleration_abs_mean_mps2": float(np.mean(np.abs(values["acceleration_mps2"]))) if values["acceleration_mps2"] else 0.0,
            "jerk_abs_mean": float(np.mean(np.abs(values["jerk"]))) if values["jerk"] else 0.0,
            "yaw_rate_abs_mean_rad_s": float(np.mean(np.abs(values["yaw_rate_rad_s"]))) if values["yaw_rate_rad_s"] else 0.0,
            "steering_change_abs_mean": float(np.mean(np.abs(values["steering_change"]))) if values["steering_change"] else 0.0,
        }
    timing = {
        name: {
            "p50_ms": _percentile(values, 50),
            "p95_ms": _percentile(values, 95),
        }
        for name, values in raw["timing"].items()
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
            "mean_error_m": float(np.mean(raw["formation_error"])) if raw["formation_error"] else 0.0,
            "p95_error_m": _percentile(raw["formation_error"], 95),
            "maximum_spread_m": max(raw["formation_spread"], default=0.0),
            "recovery_time_mean_s": float(np.mean(raw["recovery_time_s"])) if raw["recovery_time_s"] else 0.0,
        },
        "efficiency": {
            "completion_rate": raw["episode_completed"] / episode_count,
            "joint_reward_mean": float(np.mean(raw["joint_reward"])) if raw["joint_reward"] else 0.0,
        },
        "execution": {
            "rejection_count": len(raw.get("execution_rejections", ())),
            "rejection_rate": len(raw.get("execution_rejections", ()))
            / episode_count,
            "rejections": list(raw.get("execution_rejections", ())),
        },
        "trajectory_optimization": {
            "intervention_ade_mean_m": float(
                np.mean(raw.get("trajectory_intervention_ade_m", ()))
            )
            if raw.get("trajectory_intervention_ade_m")
            else 0.0,
            "intervention_fde_mean_m": float(
                np.mean(raw.get("trajectory_intervention_fde_m", ()))
            )
            if raw.get("trajectory_intervention_fde_m")
            else 0.0,
            "retained_raw_fraction_mean": float(
                np.mean(raw.get("trajectory_retained_raw_fraction", ()))
            )
            if raw.get("trajectory_retained_raw_fraction")
            else 0.0,
        },
        "timing": timing,
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
            position = np.asarray(
                env.agents[agent_id].position, dtype=np.float64
            )[:2]
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
            raise FourModelEvaluationError(
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
        raise FourModelEvaluationError("initial vehicle state is invalid")
    return signature


@torch.no_grad()
def evaluate_four_models(
    manifest_path: Path,
    output_path: Path,
    config: FourModelEvaluationConfig | None = None,
) -> dict[str, object]:
    cfg = config or FourModelEvaluationConfig()
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise FourModelEvaluationError("CUDA evaluation requested but unavailable")
    device = torch.device(cfg.device)
    _configure_deterministic_inference(device)
    manifest = _load_manifest(Path(manifest_path))
    models = {
        name: _load_policy(
            name,
            manifest["models"][name],
            device=device,
            formal=cfg.run_mode == "formal",
        )
        for name in MODEL_NAMES
    }
    model_reports = {}
    episode_count = len(cfg.scenarios) * len(cfg.seeds)
    reference_initial_states: dict[tuple[str, str, int], np.ndarray] = {}

    for name, planner in models.items():
        raw = _empty_metrics()
        trajectory_optimizer = KinematicTrajectoryOptimizer()
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
                set_spawn_seed = getattr(
                    spawn_manager, "set_episode_spawn_seed", None
                )
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
                    raise FourModelEvaluationError(
                        "models did not receive identical reset state for "
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
                role_collision = {agent_id: False for agent_id in AGENT_IDS}
                role_out = {agent_id: False for agent_id in AGENT_IDS}
                recovered_at = None
                previous_speed = {agent_id: None for agent_id in AGENT_IDS}
                previous_heading = {
                    agent_id: float(env.agents[agent_id].heading_theta)
                    for agent_id in AGENT_IDS
                }
                try:
                    for step_index in range(cfg.max_steps):
                        builder.capture_state(env, step_index * dt_s)
                        if not builder.history_ready():
                            action = constant_velocity_actions(env)
                            selected_modes = None
                        else:
                            tick_start = time.perf_counter()
                            bev_start = tick_start
                            values = builder.build_model_inputs(env)
                            bev_ms = (time.perf_counter() - bev_start) * 1000.0
                            batch = model_inputs_to_batch(values, device)
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
                            raw_trajectories = (
                                output["selected_trajectory"][0]
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            selected_mode_array = (
                                output["selected_mode"][0]
                                .detach()
                                .cpu()
                                .numpy()
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
                                    raise FourModelEvaluationError(
                                        "trajectory control failed for "
                                        f"model={name} scenario={scenario[0]} "
                                        f"seed={seed} step={step_index} "
                                        f"agent={agent_id}: {exc}"
                                    ) from exc
                                if not np.isfinite(control).all():
                                    raise FourModelEvaluationError(
                                        "trajectory control is non-finite"
                                    )
                            control_ms = (
                                time.perf_counter() - control_start
                            ) * 1000.0
                            raw["timing"]["bev_build_ms"].append(bev_ms)
                            raw["timing"]["model_inference_ms"].append(
                                inference_ms
                            )
                            raw["timing"]["control_mapping_ms"].append(
                                control_ms
                            )
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

                        _, reward, terminated, truncated, info = env.step(action)
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
                                    getattr(live_agents.get(agent_id), "speed_km_h", 0.0),
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
                        raw["formation_error"].extend(formation_values)
                        raw["formation_spread"].append(
                            float(max(formation_values, default=0.0))
                        )
                        if recovered_at is None and max(
                            formation_values, default=0.0
                        ) <= 2.0:
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
                        if episode_has_ended(terminated, truncated, info):
                            if all(
                                bool(info.get(agent_id, {}).get("arrive_dest", False))
                                for agent_id in AGENT_IDS
                            ):
                                raw["episode_completed"] += 1
                            break
                    raw["recovery_time_s"].append(
                        float(recovered_at if recovered_at is not None else cfg.max_steps * dt_s)
                    )
                    raw["episode_collision"] += int(episode_collision)
                    raw["episode_out_of_road"] += int(episode_out)
                    raw["gap_5m_violation"] += int(episode_gap5)
                    raw["gap_7m_violation"] += int(episode_gap7)
                    for agent_id in AGENT_IDS:
                        raw["roles"][agent_id]["collision"] += int(
                            role_collision[agent_id]
                        )
                        raw["roles"][agent_id]["out_of_road"] += int(
                            role_out[agent_id]
                        )
                finally:
                    env.close()
        summary = _summarize(raw, episode_count)
        inference_p95 = summary["timing"]["model_inference_ms"]["p95_ms"]
        if inference_p95 > cfg.inference_p95_limit_ms:
            raise FourModelEvaluationError(
                f"{name} three-role inference P95 {inference_p95:.2f}ms exceeds "
                f"{cfg.inference_p95_limit_ms:.2f}ms"
            )
        model_reports[name] = summary

    report = {
        "format": "bev_four_model_evaluation_v1",
        "run_mode": cfg.run_mode,
        "diagnostic_only": cfg.run_mode != "formal",
        "eligible_for_formal_conclusions": cfg.run_mode == "formal",
        "common_scenarios": [list(value) for value in cfg.scenarios],
        "common_seeds": list(cfg.seeds),
        "common_noise_seed_by_episode": True,
        "common_initial_state_verified": True,
        "deterministic_inference": {
            "torch_deterministic_algorithms": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "python_hash_seed": os.environ["PYTHONHASHSEED"],
            "tf32": False,
        },
        "scenario_contract": primary_scenario_contract(cfg.scenarios),
        "manifest": str(Path(manifest_path).resolve()),
        "manifest_sha256": _file_sha256(Path(manifest_path)),
        "model_checkpoints": {
            name: {
                "checkpoint": str(
                    Path(str(manifest["models"][name]["checkpoint"])).resolve()
                ),
                "checkpoint_sha256": manifest["models"][name][
                    "checkpoint_sha256"
                ],
            }
            for name in MODEL_NAMES
        },
        "models": model_reports,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-mode", choices=("diagnostic", "formal"), default="diagnostic")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-steps", type=int, default=100)
    arguments = parser.parse_args()
    report = evaluate_four_models(
        arguments.manifest,
        arguments.output,
        FourModelEvaluationConfig(
            run_mode=arguments.run_mode,
            device=arguments.device,
            seeds=(
                FORMAL_EVAL_SEEDS
                if arguments.run_mode == "formal"
                else HOLDOUT_SEEDS
            ),
            scenarios=(
                FORMAL_EVAL_SCENARIOS
                if arguments.run_mode == "formal"
                else DIAGNOSTIC_EVAL_SCENARIOS
            ),
            max_steps=arguments.max_steps,
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
