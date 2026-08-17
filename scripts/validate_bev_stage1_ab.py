#!/usr/bin/env python3
"""Open-loop and one-step closed-loop validation for one or more Stage 1 models."""

# flake8: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Mapping

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch import Tensor

from evaluation.bev_model_manifest import file_sha256, load_model_manifest
from evaluation.bev_evaluation_artifacts import (
    ClosedLoopArtifactWriter,
    OpenLoopArtifactCollector,
)
from expert_dataset.collect_joint_bev import (
    JointBEVModelInputs,
    JointBEVSampleBuilder,
    RulePlannerExpert,
    SensorlessJointBEVPlatoonEnv,
    simulator_decision_dt_s,
)
from expert_dataset.joint_bev_dataset import (
    JointBEVDataset,
    JointBEVDatasetConfig,
    build_joint_bev_dataloader,
)
from models.bev_planner import (
    BEVOnlyDiffusionPlanner,
    BEVOnlyDiffusionPlannerConfig,
    JointStage1Loss,
    KinematicTrajectoryOptimizer,
    Stage1LossConfig,
)
from train.train_bev_diffusion_stage1 import (
    Stage1TrainingError,
    load_stage1_checkpoint,
    loss_from_batch,
    move_joint_batch,
    planner_forward_from_batch,
    resolve_device,
)


AGENT_IDS = ("agent0", "agent1", "agent2")
VARIANT_CONDITION = {"A": "none", "B": "predicted_detached"}
OPEN_LOOP_COMPARISON_POLICY = {
    "loss/total": ("lower", 1.0e-4),
    "metric/mode_accuracy": ("higher", 1.0e-3),
    "metric/gt_mode_ade": ("lower", 1.0e-2),
    "metric/gt_mode_fde": ("lower", 1.0e-2),
    "metric/selected_ade": ("lower", 1.0e-2),
    "metric/selected_fde": ("lower", 1.0e-2),
}


def _checkpoint_header(path: Path) -> Mapping[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise Stage1TrainingError(f"unable to read checkpoint header: {path}") from exc
    if not isinstance(payload, Mapping):
        raise Stage1TrainingError("Stage 1 checkpoint must be a mapping")
    return payload


def load_planner(
    path: Path, device: torch.device
) -> tuple[BEVOnlyDiffusionPlanner, dict]:
    header = _checkpoint_header(path)
    variant = header.get("variant")
    if variant not in VARIANT_CONDITION:
        raise Stage1TrainingError("Stage 1 checkpoint variant must be A or B")
    planner = BEVOnlyDiffusionPlanner(
        BEVOnlyDiffusionPlannerConfig(
            predecessor_condition=VARIANT_CONDITION[str(variant)]
        )
    )
    payload = load_stage1_checkpoint(path, planner)
    planner.to(device).eval()
    return planner, payload


def _explicit_noise(
    batch: Mapping[str, Tensor], device: torch.device, *, seed: int = 17
) -> Tensor:
    shape = (*batch["coarse_trajectories"].shape[:-1], 2)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return torch.randn(shape, generator=generator, device=device, dtype=torch.float32)


@torch.no_grad()
def validate_open_loop(
    planner: BEVOnlyDiffusionPlanner,
    payload: Mapping[str, object],
    *,
    dataset_root: Path,
    device: torch.device,
    max_samples: int = 1500,
    artifact_root: Path | None = None,
    save_visualizations: bool = False,
) -> dict[str, object]:
    dataset = JointBEVDataset(JointBEVDatasetConfig(dataset_root, "val"))
    loader_dataset = None
    collector = (
        OpenLoopArtifactCollector(
            artifact_root, save_visualizations=save_visualizations
        )
        if artifact_root is not None
        else None
    )
    try:
        if len(dataset) <= 0:
            raise Stage1TrainingError(
                "open-loop validation requires a non-empty val split"
            )
        sample_limit = min(int(max_samples), len(dataset))
        if sample_limit <= 0:
            raise Stage1TrainingError("open-loop max_samples must be positive")
        loader = build_joint_bev_dataloader(
            dataset_root,
            "val",
            batch_size=min(8, len(dataset)),
            shuffle=False,
            num_workers=0,
            seed=17,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        loader_dataset = loader.dataset
        dataset_fingerprint = loader_dataset.contract["dataset_fingerprint"]
        if payload.get("dataset_fingerprint") != dataset_fingerprint:
            raise Stage1TrainingError(
                "checkpoint and open-loop dataset fingerprints do not match"
            )
        training_config = payload.get("training_config")
        if not isinstance(training_config, Mapping) or not isinstance(
            training_config.get("loss"), Mapping
        ):
            raise Stage1TrainingError("checkpoint loss config is invalid")
        loss_module = JointStage1Loss(
            Stage1LossConfig(**dict(training_config["loss"]))
        ).to(device)
        processed = 0
        metric_totals: dict[str, float] = {}
        selected_modes: list[object] = []
        deterministic = False
        for batch_index, raw_batch in enumerate(loader):
            remaining = sample_limit - processed
            if remaining <= 0:
                break
            batch = move_joint_batch(raw_batch, device)
            batch_size = int(batch["bev"].shape[0])
            if batch_size > remaining:
                batch = {name: value[:remaining] for name, value in batch.items()}
                batch_size = remaining
            noise = _explicit_noise(batch, device, seed=17 + processed)
            first = planner_forward_from_batch(planner, batch, diffusion_noise=noise)
            if batch_index == 0:
                second = planner_forward_from_batch(
                    planner, batch, diffusion_noise=noise
                )
                for name in (
                    "trajectory_candidates",
                    "mode_logits",
                    "selected_mode",
                    "selected_trajectory",
                ):
                    torch.testing.assert_close(
                        first[name], second[name], rtol=0.0, atol=0.0
                    )
                deterministic = True
            if not bool(torch.isfinite(first["trajectory_candidates"]).all()):
                raise Stage1TrainingError(
                    "open-loop trajectory candidates are non-finite"
                )
            selected_valid = batch["mode_valid_mask"].gather(
                -1, first["selected_mode"].unsqueeze(-1)
            )
            if not bool(selected_valid.all()):
                raise Stage1TrainingError(
                    "open-loop planner selected an invalid hard-mask mode"
                )
            result = loss_from_batch(loss_module, first, batch)
            for name, value in result.scalar_metrics().items():
                metric_totals[name] = (
                    metric_totals.get(name, 0.0) + float(value) * batch_size
                )
            selected_modes.extend(first["selected_mode"].detach().cpu().tolist())
            if collector is not None:
                collector.add_batch(batch, first)
            processed += batch_size
        if processed != sample_limit:
            raise Stage1TrainingError(
                f"open-loop processed {processed} samples, expected {sample_limit}"
            )
    finally:
        dataset.close()
        close_loader_dataset = getattr(loader_dataset, "close", None)
        if callable(close_loader_dataset):
            close_loader_dataset()

    return {
        "samples": processed,
        "selected_mode": selected_modes,
        "metrics": {name: total / processed for name, total in metric_totals.items()},
        "deterministic": deterministic,
        "legacy_open_loop": collector.finalize() if collector is not None else None,
    }


def _model_batch(
    values: JointBEVModelInputs, device: torch.device
) -> dict[str, Tensor]:
    return {
        name: torch.from_numpy(np.array(array, copy=True)).unsqueeze(0).to(device)
        for name, array in values.as_dict().items()
    }


def _has_failure(info: Mapping[str, object]) -> bool:
    failure_keys = (
        "crash",
        "crash_vehicle",
        "crash_object",
        "crash_building",
        "crash_human",
        "out_of_road",
        "out_of_route",
    )
    for agent_id in AGENT_IDS:
        agent_info = info.get(agent_id)
        if isinstance(agent_info, Mapping) and any(
            bool(agent_info.get(key, False)) for key in failure_keys
        ):
            return True
    return False


def _global_agent_poses(env: object) -> np.ndarray:
    poses = np.asarray(
        [
            [
                *np.asarray(env.agents[agent_id].position, dtype=np.float64)[:2],
                float(env.agents[agent_id].heading_theta),
            ]
            for agent_id in AGENT_IDS
        ],
        dtype=np.float64,
    )
    if poses.shape != (3, 3) or not np.isfinite(poses).all():
        raise Stage1TrainingError("sensorless S1 agent poses are invalid")
    return poses


@torch.no_grad()
def validate_closed_loop(
    planner: BEVOnlyDiffusionPlanner,
    *,
    device: torch.device,
    artifact_root: Path | None = None,
    model_id: str = "stage1",
    topdown_screen_size: int = 800,
    topdown_film_size: int = 3000,
) -> dict[str, object]:
    env = SensorlessJointBEVPlatoonEnv(
        {
            "num_agents": 3,
            "traffic_density": 0.0,
            "start_seed": 17,
            "num_scenarios": 1,
        }
    )
    try:
        env.set_runtime_scenario_route(
            "S1_free_cruise_straight", "R3_mainline_straight"
        )
        env.reset()
        builder = JointBEVSampleBuilder(AGENT_IDS)
        builder.reset()
        expert = RulePlannerExpert(env, AGENT_IDS)
        dt_s = simulator_decision_dt_s(env)
        step_index = 0
        while True:
            builder.capture_state(env, step_index * dt_s)
            if builder.history_ready():
                break
            expert_step = expert.plan(env)
            _, _, terminated, truncated, info = env.low_level_step(
                dict(expert_step.controls)
            )
            if (
                bool(terminated.get("__all__", False))
                or bool(truncated.get("__all__", False))
                or _has_failure(info)
            ):
                raise Stage1TrainingError(
                    "sensorless S1 history warm-up ended before one second"
                )
            step_index += 1
            if step_index > 20:
                raise Stage1TrainingError(
                    "sensorless S1 history did not become ready within 20 steps"
                )

        inputs = builder.build_model_inputs(env)
        batch = _model_batch(inputs, device)
        output = planner_forward_from_batch(planner, batch)
        raw_trajectories = (
            output["selected_trajectory"].squeeze(0).detach().cpu().numpy()
        )
        selected_modes = output["selected_mode"].squeeze(0).detach().cpu().numpy()
        optimization = KinematicTrajectoryOptimizer().optimize(
            raw_trajectories,
            inputs.coarse_trajectories,
            inputs.ego_state[:, 0],
            selected_modes,
        )
        trajectories = optimization.optimized_trajectories
        if trajectories.shape != (3, 8, 3) or not np.isfinite(trajectories).all():
            raise Stage1TrainingError(
                "closed-loop planner produced invalid trajectories"
            )
        actions = {
            agent_id: np.ascontiguousarray(trajectories[index], dtype=np.float32)
            for index, agent_id in enumerate(AGENT_IDS)
        }
        pre_poses = _global_agent_poses(env)
        writer = None
        episode_id = None
        if artifact_root is not None:
            writer = ClosedLoopArtifactWriter(
                artifact_root,
                model_id,
                topdown_screen_size=topdown_screen_size,
                topdown_film_size=topdown_film_size,
            )
            episode_id = writer.start_episode(
                ("S1_free_cruise_straight", "R3_mainline_straight"), 17
            )
            writer.capture_topdown_frame(
                episode_id,
                env=env,
                step_index=step_index,
                pre_poses=pre_poses,
                trajectories=trajectories,
            )
        started = time.perf_counter()
        observation, reward, terminated, truncated, info = env.step(actions)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not all(
            isinstance(value, Mapping)
            for value in (observation, reward, terminated, truncated, info)
        ):
            raise Stage1TrainingError(
                "closed-loop environment returned invalid mappings"
            )
        if _has_failure(info):
            raise Stage1TrainingError(
                "sensorless S1 model step produced crash/out-of-road"
            )
        if any(agent_id not in observation for agent_id in AGENT_IDS):
            raise Stage1TrainingError(
                "sensorless S1 model step did not retain all three agents"
            )
        if any(
            not np.isfinite(float(reward[agent_id]))
            for agent_id in AGENT_IDS
            if agent_id in reward
        ):
            raise Stage1TrainingError("sensorless S1 returned a non-finite reward")
        low_level = getattr(env, "_pending_low_level_actions", {})
        if set(low_level) != set(AGENT_IDS) or any(
            np.asarray(low_level[agent_id]).shape != (2,)
            or not np.isfinite(np.asarray(low_level[agent_id])).all()
            for agent_id in AGENT_IDS
        ):
            raise Stage1TrainingError(
                "sensorless S1 trajectory controller produced invalid controls"
            )
        artifact_report = None
        if writer is not None and episode_id is not None:
            post_poses = _global_agent_poses(env)
            writer.record_step(
                episode_id,
                step_index=step_index,
                dt_s=dt_s,
                pre_poses=pre_poses,
                post_poses=post_poses,
                trajectories=trajectories,
                selected_modes=selected_modes.tolist(),
                rewards=reward,
                controls=low_level,
                bev=inputs.bev,
            )
            writer.finish_episode(episode_id)
            artifact_report = writer.finalize()
        return {
            "warmup_steps": step_index,
            "selected_mode": output["selected_mode"].squeeze(0).cpu().tolist(),
            "environment_step_ms": elapsed_ms,
            "terminated": bool(terminated.get("__all__", False)),
            "truncated": bool(truncated.get("__all__", False)),
            "trajectory_optimizer": {
                "config_sha256": optimization.config_sha256,
                "elapsed_ms": optimization.elapsed_ms,
                "raw_valid": optimization.raw_valid.tolist(),
                "optimized_valid": optimization.optimized_valid.tolist(),
                "intervention_ade_m": optimization.intervention_ade_m.tolist(),
                "intervention_fde_m": optimization.intervention_fde_m.tolist(),
                "retained_raw_fraction": optimization.retained_raw_fraction.tolist(),
            },
            "artifacts": artifact_report,
        }
    finally:
        env.close()


def _comparison_label(improvement: float, tolerance: float) -> str:
    if improvement > tolerance:
        return "better"
    if improvement < -tolerance:
        return "worse"
    return "equivalent"


def compare_open_loop_models(
    models: Mapping[str, object], comparisons: tuple[object, ...]
) -> dict[str, object]:
    results = {}
    for comparison in comparisons:
        baseline_id = comparison.baseline
        candidate_id = comparison.candidate
        baseline = models[baseline_id]
        candidate = models[candidate_id]
        baseline_metrics = baseline["open_loop"]["metrics"]
        candidate_metrics = candidate["open_loop"]["metrics"]
        metric_results = {}
        for metric, (direction, tolerance) in OPEN_LOOP_COMPARISON_POLICY.items():
            baseline_value = float(baseline_metrics[metric])
            candidate_value = float(candidate_metrics[metric])
            raw_delta = candidate_value - baseline_value
            improvement = raw_delta if direction == "higher" else -raw_delta
            metric_results[metric] = {
                "baseline": baseline_value,
                "candidate": candidate_value,
                "candidate_minus_baseline": raw_delta,
                "improvement": improvement,
                "direction": direction,
                "equivalence_tolerance": tolerance,
                "conclusion": _comparison_label(improvement, tolerance),
            }
        results[comparison.comparison_id] = {
            "baseline": baseline_id,
            "candidate": candidate_id,
            "metrics": metric_results,
        }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--open-loop-num-samples", type=int, default=1500)
    parser.add_argument("--save-visualizations", action="store_true")
    parser.add_argument("--topdown-screen-size", type=int, default=800)
    parser.add_argument("--topdown-film-size", type=int, default=3000)
    args = parser.parse_args()
    if args.topdown_screen_size <= 0:
        parser.error("--topdown-screen-size must be positive")
    if args.topdown_film_size <= 0:
        parser.error("--topdown-film-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    manifest = load_model_manifest(args.manifest)
    artifact_root = args.artifact_root or args.output.parent / "artifacts"
    non_stage1 = [model.model_id for model in manifest.models if model.kind != "stage1"]
    if non_stage1:
        raise Stage1TrainingError(
            "Stage 1 validation manifest contains non-Stage1 models: "
            + ", ".join(non_stage1)
        )
    models: dict[str, object] = {}
    for model in manifest.models:
        planner, payload = load_planner(model.checkpoint, device)
        if payload["variant"] != model.variant:
            raise Stage1TrainingError(
                f"model {model.model_id} checkpoint variant does not match manifest"
            )
        models[model.model_id] = {
            "kind": model.kind,
            "variant": model.variant,
            "checkpoint": str(model.checkpoint),
            "checkpoint_sha256": model.checkpoint_sha256,
            "checkpoint_metadata": {
                "run_mode": payload.get("run_mode"),
                "diagnostic_only": payload.get("diagnostic_only"),
                "eligible_for_formal_training": payload.get(
                    "eligible_for_formal_training"
                ),
            },
            "open_loop": validate_open_loop(
                planner,
                payload,
                dataset_root=args.dataset_root,
                device=device,
                max_samples=args.open_loop_num_samples,
                artifact_root=artifact_root / "open_loop" / model.model_id,
                save_visualizations=args.save_visualizations,
            ),
            "closed_loop": validate_closed_loop(
                planner,
                device=device,
                artifact_root=artifact_root / "s1_closed_loop",
                model_id=model.model_id,
                topdown_screen_size=args.topdown_screen_size,
                topdown_film_size=args.topdown_film_size,
            ),
        }
        del planner
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report: dict[str, object] = {
        "format": "bev_stage1_model_validation_v2",
        "seed": 17,
        "device": str(device),
        "diagnostic_only": True,
        "eligible_for_formal_conclusions": False,
        "manifest": str(manifest.path),
        "manifest_sha256": file_sha256(manifest.path),
        "model_order": list(manifest.model_ids),
        "comparison_policy": {
            metric: {"direction": value[0], "equivalence_tolerance": value[1]}
            for metric, value in OPEN_LOOP_COMPARISON_POLICY.items()
        },
        "models": models,
        "comparisons": compare_open_loop_models(models, manifest.comparisons),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
