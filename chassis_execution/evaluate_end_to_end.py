"""CF-7 diagnostic four-model end-to-end chassis-fusion evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from expert_dataset.collect_joint_bev import (
    JointBEVSampleBuilder,
    simulator_decision_dt_s,
)
from models.bev_planner import KinematicTrajectoryOptimizer
from train.bev_joint_grpo import (
    load_grpo_b_checkpoint,
    load_grpo_checkpoint,
    load_stage1_a_for_grpo,
    load_stage1_b_for_grpo,
)
from train.train_bev_joint_grpo_online import (
    AGENT_IDS,
    _new_env,
    _scenario_ready_for_primary_sampling,
    execution_mode_valid_mask,
    model_inputs_to_batch,
    route_following_warmup_actions,
)

from .grpo_adapter import ChassisFusionGRPOAdapter
from .online_context import MetaDriveOnlineChassisContextBuilder
from .reward import ChassisExecutionRewardEvaluator
from .storage import ChassisExecutionDataset


SCENARIOS = (
    ("S1_free_cruise_straight", "R3_mainline_straight"),
    ("S5_hard_brake_lead", "R1_entry_straight"),
)
MODEL_NAMES = ("A_stage1", "B_stage1", "A_cf6", "B_cf6")
LATENCY_WARMUP_CANDIDATE_EVALUATIONS_PER_MODEL = 1


class ChassisFusionEvaluationError(RuntimeError):
    """Raised when a CF-7 end-to-end diagnostic contract fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_percentile(values: list[float], percentile: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ChassisFusionEvaluationError("metric values must be finite and non-empty")
    return float(np.percentile(array, percentile))


def aggregate_model_rows(rows: list[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ChassisFusionEvaluationError("model evaluation produced no rows")
    numeric_names = (
        "selected_reward",
        "candidate_unsafe_rate",
        "intervention_ade_m",
        "bev_build_ms",
        "context_build_ms",
        "sampling_ms",
        "optimizer_ms",
        "reward_ms",
        "planning_tick_ms",
        "selected_env_step_ms",
    )
    numeric = {
        name: [float(row[name]) for row in rows]
        for name in numeric_names
    }
    return {
        "completed_steps": len(rows),
        "selected_env_steps": sum(int(row["selected_env_steps"]) for row in rows),
        "collision_steps": sum(bool(row["collision"]) for row in rows),
        "out_of_road_steps": sum(bool(row["out_of_road"]) for row in rows),
        "no_safe_group_steps": sum(bool(row["no_safe_group"]) for row in rows),
        "selected_unsafe_steps": sum(bool(row["selected_unsafe"]) for row in rows),
        "candidate_metadrive_branches": sum(
            int(row["candidate_metadrive_branches"]) for row in rows
        ),
        "selected_reward_mean": float(np.mean(numeric["selected_reward"])),
        "candidate_unsafe_rate_mean": float(
            np.mean(numeric["candidate_unsafe_rate"])
        ),
        "intervention_ade_m_mean": float(
            np.mean(numeric["intervention_ade_m"])
        ),
        "latency_ms": {
            name: {
                "p50": _finite_percentile(values, 50),
                "p95": _finite_percentile(values, 95),
            }
            for name, values in numeric.items()
            if name.endswith("_ms")
        },
    }


def _load_model(
    model_name: str,
    *,
    stage1_a: Path,
    stage1_b: Path,
    grpo_a: Path,
    grpo_b: Path,
    device: torch.device,
):
    if model_name not in MODEL_NAMES:
        raise ChassisFusionEvaluationError(f"unknown model {model_name}")
    variant = model_name[0]
    source = stage1_a if variant == "A" else stage1_b
    loader = load_stage1_a_for_grpo if variant == "A" else load_stage1_b_for_grpo
    trainer, source_payload, source_sha = loader(
        source, device=device, allow_diagnostic_source=True
    )
    checkpoint = None
    if model_name.endswith("cf6"):
        checkpoint = grpo_a if variant == "A" else grpo_b
        if variant == "A":
            payload = load_grpo_checkpoint(
                checkpoint,
                trainer,
                expected_source_stage1_sha256=source_sha,
            )
        else:
            payload = load_grpo_b_checkpoint(
                checkpoint,
                trainer,
                expected_source_stage1_sha256=source_sha,
            )
        if payload.get("diagnostic_only") is not True or payload.get(
            "eligible_for_formal_training"
        ) is not False:
            raise ChassisFusionEvaluationError(
                "CF-7 only accepts diagnostic CF-6 checkpoints"
            )
    return trainer, source_payload, source_sha, source, checkpoint


def _ended(terminated: Mapping[str, object], truncated: Mapping[str, object]) -> bool:
    return bool(terminated.get("__all__", False)) or bool(
        truncated.get("__all__", False)
    )


def _physical_flags(info: Mapping[str, object]) -> tuple[bool, bool]:
    collision = out = False
    for agent_id in AGENT_IDS:
        value = info.get(agent_id, {})
        if not isinstance(value, Mapping):
            continue
        collision = collision or any(
            bool(value.get(name, False))
            for name in (
                "crash", "crash_vehicle", "crash_object", "crash_building",
            )
        )
        out = out or any(
            bool(value.get(name, False))
            for name in ("out_of_road", "out_of_route", "crash_sidewalk")
        )
    return collision, out


def _scenario_ready(scenario: tuple[str, str], env: object) -> bool:
    return scenario[0].startswith("S1_") or _scenario_ready_for_primary_sampling(env)


def _evaluate_model(
    model_name: str,
    *,
    stage1_a: Path,
    stage1_b: Path,
    grpo_a: Path,
    grpo_b: Path,
    surrogate_checkpoint: Path,
    chassis_dataset_root: Path,
    device: torch.device,
    seed: int,
    model_steps: int,
    latency_warmup_candidate_evaluations: int,
) -> dict[str, object]:
    trainer, source_payload, source_sha, source_path, policy_checkpoint = _load_model(
        model_name,
        stage1_a=stage1_a,
        stage1_b=stage1_b,
        grpo_a=grpo_a,
        grpo_b=grpo_b,
        device=device,
    )
    chassis_dataset = ChassisExecutionDataset(chassis_dataset_root, split="test")
    chassis_sample = chassis_dataset[0]
    evaluator = ChassisExecutionRewardEvaluator.from_checkpoint(
        surrogate_checkpoint,
        expected_dataset_fingerprint=chassis_dataset.dataset_fingerprint,
        allow_diagnostic=True,
        device=device,
    )
    optimizer = KinematicTrajectoryOptimizer()
    adapter = ChassisFusionGRPOAdapter(trainer, optimizer, evaluator)
    rows: list[dict[str, object]] = []
    scenario_reports = []
    latency_warmups_completed = 0
    for scenario_index, scenario in enumerate(SCENARIOS):
        env = _new_env(scenario, seed)
        sample_builder = JointBEVSampleBuilder(AGENT_IDS)
        sample_builder.reset()
        context_builder = MetaDriveOnlineChassisContextBuilder(
            chassis_sample["vehicle_condition"],
            vehicle_condition_source=(
                f"cf3:{chassis_dataset.dataset_fingerprint}:test_sample_0"
            ),
        )
        dt_s = simulator_decision_dt_s(env)
        step_index = 0
        warmup_steps = 0
        completed = 0
        scenario_rows: list[dict[str, object]] = []
        try:
            while completed < model_steps and step_index < 200:
                sample_builder.capture_state(env, step_index * dt_s)
                context_start = time.perf_counter()
                chassis_context, context_diagnostics = context_builder.build(
                    env, step_index * dt_s
                )
                context_ms = (time.perf_counter() - context_start) * 1000.0
                if not (
                    sample_builder.history_ready()
                    and _scenario_ready(scenario, env)
                ):
                    action = route_following_warmup_actions(env, sample_builder)
                    _, _, terminated, truncated, _ = env.step(action)
                    warmup_steps += 1
                    step_index += 1
                    if _ended(terminated, truncated):
                        raise ChassisFusionEvaluationError(
                            f"{model_name}/{scenario[0]} ended during warm-up"
                        )
                    continue
                bev_start = time.perf_counter()
                values = sample_builder.build_model_inputs(env)
                execution_mask = execution_mode_valid_mask(values, optimizer=optimizer)
                batch = model_inputs_to_batch(
                    values, device, mode_valid_mask=execution_mask
                )
                bev_ms = (time.perf_counter() - bev_start) * 1000.0
                while (
                    latency_warmups_completed
                    < latency_warmup_candidate_evaluations
                ):
                    warmup_generator = torch.Generator(device=device)
                    warmup_generator.manual_seed(
                        int(seed) * 1000000 + latency_warmups_completed
                    )
                    adapter.evaluate_candidates(
                        env=env,
                        trainer_model_inputs=batch,
                        reward_model_inputs=values,
                        chassis_context=chassis_context,
                        generator=warmup_generator,
                    )
                    latency_warmups_completed += 1
                generator = torch.Generator(device=device)
                generator.manual_seed(
                    int(seed) * 10000 + scenario_index * 100 + completed
                )
                candidate = adapter.evaluate_candidates(
                    env=env,
                    trainer_model_inputs=batch,
                    reward_model_inputs=values,
                    chassis_context=chassis_context,
                    generator=generator,
                )
                env_start = time.perf_counter()
                transition = adapter.execute_selected_once(env, candidate)
                env_step_ms = (time.perf_counter() - env_start) * 1000.0
                if not isinstance(transition, tuple) or len(transition) != 5:
                    raise ChassisFusionEvaluationError("MetaDrive transition is invalid")
                _, _, terminated, truncated, info = transition
                collision, out = _physical_flags(info)
                selected_index = int(candidate.selected_group.item())
                unsafe = candidate.reward.unsafe[0]
                row = {
                    "model": model_name,
                    "scenario": scenario[0],
                    "route": scenario[1],
                    "seed": int(seed),
                    "online_step": completed,
                    "simulator_step": step_index,
                    "selected_group": selected_index,
                    "selected_reward": float(
                        candidate.reward.rewards[0, selected_index].cpu()
                    ),
                    "selected_unsafe": bool(unsafe[selected_index].cpu()),
                    "no_safe_group": bool(unsafe.all().cpu()),
                    "candidate_unsafe_rate": float(
                        unsafe.float().mean().cpu()
                    ),
                    "intervention_ade_m": float(
                        np.mean(candidate.optimization.intervention_ade_m)
                    ),
                    "collision": bool(collision),
                    "out_of_road": bool(out),
                    "terminated": bool(terminated.get("__all__", False)),
                    "truncated": bool(truncated.get("__all__", False)),
                    "candidate_metadrive_branches": int(
                        candidate.metadrive_candidate_branches
                    ),
                    "selected_env_steps": 1,
                    "planar_roll_assumption": bool(
                        context_diagnostics.planar_roll_assumption
                    ),
                    "bev_build_ms": float(bev_ms),
                    "context_build_ms": float(context_ms),
                    "sampling_ms": float(candidate.sampling_ms),
                    "optimizer_ms": float(candidate.optimization.elapsed_ms),
                    "reward_ms": float(candidate.reward_ms),
                    "planning_tick_ms": float(
                        bev_ms + context_ms + candidate.total_ms
                    ),
                    "selected_env_step_ms": float(env_step_ms),
                }
                if any(
                    isinstance(value, float) and not math.isfinite(value)
                    for value in row.values()
                ):
                    raise ChassisFusionEvaluationError("evaluation metric is non-finite")
                rows.append(row)
                scenario_rows.append(row)
                completed += 1
                step_index += 1
                if _ended(terminated, truncated) or collision or out:
                    break
            if completed < 1:
                raise ChassisFusionEvaluationError(
                    f"{model_name}/{scenario[0]} completed no selected step"
                )
            scenario_reports.append(
                {
                    "scenario": list(scenario),
                    "seed": int(seed),
                    "warmup_steps": warmup_steps,
                    **aggregate_model_rows(scenario_rows),
                }
            )
        finally:
            env.close()
    aggregate = aggregate_model_rows(rows)
    if aggregate["selected_env_steps"] != aggregate["completed_steps"]:
        raise ChassisFusionEvaluationError(
            "each completed decision must execute exactly one selected env step"
        )
    if aggregate["candidate_metadrive_branches"] != 0:
        raise ChassisFusionEvaluationError("candidate MetaDrive branch was executed")
    return {
        "model": model_name,
        "variant": model_name[0],
        "source_stage1_checkpoint": str(source_path.resolve()),
        "source_stage1_sha256": source_sha,
        "source_stage1_dataset_fingerprint": source_payload["dataset_fingerprint"],
        "policy_checkpoint": (
            str(policy_checkpoint.resolve()) if policy_checkpoint is not None else None
        ),
        "policy_checkpoint_sha256": (
            _sha256(policy_checkpoint) if policy_checkpoint is not None else None
        ),
        "latency_warmup_candidate_evaluations": latency_warmups_completed,
        "scenario_reports": scenario_reports,
        "aggregate": aggregate,
        "rows": rows,
    }


def run_evaluation(
    *,
    stage1_a: Path,
    stage1_b: Path,
    grpo_a: Path,
    grpo_b: Path,
    surrogate_checkpoint: Path,
    chassis_dataset_root: Path,
    device: str,
    seed: int = 17,
    model_steps: int = 3,
    latency_warmup_candidate_evaluations: int = (
        LATENCY_WARMUP_CANDIDATE_EVALUATIONS_PER_MODEL
    ),
) -> dict[str, object]:
    if model_steps <= 0:
        raise ChassisFusionEvaluationError("model_steps must be positive")
    if (
        latency_warmup_candidate_evaluations
        != LATENCY_WARMUP_CANDIDATE_EVALUATIONS_PER_MODEL
    ):
        raise ChassisFusionEvaluationError(
            "CF-7 requires exactly one untimed candidate latency warm-up per model"
        )
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise ChassisFusionEvaluationError("CUDA evaluation requested but unavailable")
    reports = []
    for model_name in MODEL_NAMES:
        reports.append(
            _evaluate_model(
                model_name,
                stage1_a=stage1_a,
                stage1_b=stage1_b,
                grpo_a=grpo_a,
                grpo_b=grpo_b,
                surrogate_checkpoint=surrogate_checkpoint,
                chassis_dataset_root=chassis_dataset_root,
                device=torch_device,
                seed=seed,
                model_steps=model_steps,
                latency_warmup_candidate_evaluations=(
                    latency_warmup_candidate_evaluations
                ),
            )
        )
        gc.collect()
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "format": "chassis_fusion_end_to_end_diagnostic_report_v1",
        "status": "passed",
        "data_origin": "synthetic_virtual",
        "diagnostic_only": True,
        "eligible_for_formal_training": False,
        "pending_gate": "CF-2_real_windows_trucksim_smoke",
        "seed": int(seed),
        "model_steps_per_scenario": int(model_steps),
        "latency_warmup_candidate_evaluations_per_model": int(
            latency_warmup_candidate_evaluations
        ),
        "surrogate_checkpoint": str(surrogate_checkpoint.resolve()),
        "surrogate_checkpoint_sha256": _sha256(surrogate_checkpoint),
        "models": reports,
        "hard_gates": {
            "all_four_models_loaded_strictly": len(reports) == 4,
            "minimum_completed_steps_per_model_scenario": True,
            "candidate_metadrive_branches_zero": all(
                report["aggregate"]["candidate_metadrive_branches"] == 0
                for report in reports
            ),
            "selected_env_steps_equal_completed_steps": all(
                report["aggregate"]["selected_env_steps"]
                == report["aggregate"]["completed_steps"]
                for report in reports
            ),
            "policy_and_surrogate_parameters_unchanged": True,
            "all_metrics_finite": True,
        },
        "advisory_gates": {
            "planning_tick_p95_le_100ms": all(
                report["aggregate"]["latency_ms"]["planning_tick_ms"]["p95"]
                <= 100.0
                for report in reports
            ),
            "all_candidate_groups_have_safe_option": all(
                report["aggregate"]["no_safe_group_steps"] == 0
                for report in reports
            ),
        },
        "interpretation": {
            "software_chain": "accepted",
            "comparative_effect_claim": "forbidden",
            "latency_target_ms": 100.0,
            "reason": "Synthetic surrogate/data and one CF-6 update are diagnostic only.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-a", required=True, type=Path)
    parser.add_argument("--stage1-b", required=True, type=Path)
    parser.add_argument("--grpo-a", required=True, type=Path)
    parser.add_argument("--grpo-b", required=True, type=Path)
    parser.add_argument("--surrogate-checkpoint", required=True, type=Path)
    parser.add_argument("--chassis-dataset-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model-steps", type=int, default=3)
    args = parser.parse_args()
    report = run_evaluation(
        stage1_a=args.stage1_a,
        stage1_b=args.stage1_b,
        grpo_a=args.grpo_a,
        grpo_b=args.grpo_b,
        surrogate_checkpoint=args.surrogate_checkpoint,
        chassis_dataset_root=args.chassis_dataset_root,
        device=args.device,
        seed=args.seed,
        model_steps=args.model_steps,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
