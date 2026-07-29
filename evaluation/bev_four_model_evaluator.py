"""Fair closed-loop evaluation for Stage 1 and GRPO Variants A/B."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
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
    checkpoint = Path(str(spec.get("checkpoint", "")))
    loader = (
        load_stage1_a_for_grpo
        if expected_variant == "A"
        else load_stage1_b_for_grpo
    )
    trainer, source_payload, source_sha = loader(
        checkpoint if expected_kind == "stage1" else Path(str(spec.get("source_checkpoint", ""))),
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
        ):
            if field not in grpo_payload:
                raise FourModelEvaluationError(
                    f"{name} is not an online-calibrated GRPO checkpoint"
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
        },
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
            position = np.asarray(
                env.agents[agent_id].position, dtype=np.float64
            )[:2]
            minimum = min(
                minimum,
                float(np.linalg.norm(position - other_position))
                - 0.5 * (5.74 + other_length),
            )
    return minimum


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

    for name, planner in models.items():
        raw = _empty_metrics()
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
                            trajectories = (
                                output["selected_trajectory"][0]
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            selected_modes = (
                                output["selected_mode"][0].detach().cpu().tolist()
                            )
                            action = joint_trajectory_action(trajectories)
                            control_start = time.perf_counter()
                            for agent_id in AGENT_IDS:
                                control = env.trajectory_to_control(
                                    agent_id, action[agent_id]
                                )
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

                        _, reward, terminated, truncated, info = env.step(action)
                        poses = [
                            np.asarray(env.agents[agent_id].position, dtype=np.float64)
                            for agent_id in AGENT_IDS
                        ]
                        adjacent_gaps = [
                            float(np.linalg.norm(poses[index] - poses[index + 1]) - 5.74)
                            for index in (0, 1)
                        ]
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
                                    getattr(env.agents[agent_id], "speed_km_h", 0.0),
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
                            heading = float(env.agents[agent_id].heading_theta)
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
        "scenario_contract": primary_scenario_contract(cfg.scenarios),
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
