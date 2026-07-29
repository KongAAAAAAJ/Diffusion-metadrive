#!/usr/bin/env python3
"""Open-loop and one-step sensorless closed-loop validation for Stage 1 A/B."""

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


def _checkpoint_header(path: Path) -> Mapping[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise Stage1TrainingError(f"unable to read checkpoint header: {path}") from exc
    if not isinstance(payload, Mapping):
        raise Stage1TrainingError("Stage 1 checkpoint must be a mapping")
    return payload


def load_planner(path: Path, device: torch.device) -> tuple[BEVOnlyDiffusionPlanner, dict]:
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


def _explicit_noise(batch: Mapping[str, Tensor], device: torch.device) -> Tensor:
    shape = (*batch["coarse_trajectories"].shape[:-1], 2)
    generator = torch.Generator(device=device)
    generator.manual_seed(17)
    return torch.randn(shape, generator=generator, device=device, dtype=torch.float32)


@torch.no_grad()
def validate_open_loop(
    planner: BEVOnlyDiffusionPlanner,
    payload: Mapping[str, object],
    *,
    dataset_root: Path,
    device: torch.device,
) -> dict[str, object]:
    dataset = JointBEVDataset(JointBEVDatasetConfig(dataset_root, "val"))
    loader_dataset = None
    try:
        if len(dataset) <= 0:
            raise Stage1TrainingError("open-loop validation requires a non-empty val split")
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
        batch = move_joint_batch(next(iter(loader)), device)
    finally:
        dataset.close()
        close_loader_dataset = getattr(loader_dataset, "close", None)
        if callable(close_loader_dataset):
            close_loader_dataset()

    if payload.get("dataset_fingerprint") != dataset_fingerprint:
        raise Stage1TrainingError(
            "checkpoint and open-loop dataset fingerprints do not match"
        )
    noise = _explicit_noise(batch, device)
    first = planner_forward_from_batch(planner, batch, diffusion_noise=noise)
    second = planner_forward_from_batch(planner, batch, diffusion_noise=noise)
    for name in (
        "trajectory_candidates",
        "mode_logits",
        "selected_mode",
        "selected_trajectory",
    ):
        torch.testing.assert_close(first[name], second[name], rtol=0.0, atol=0.0)
    if not bool(torch.isfinite(first["trajectory_candidates"]).all()):
        raise Stage1TrainingError("open-loop trajectory candidates are non-finite")
    selected_valid = batch["mode_valid_mask"].gather(
        -1, first["selected_mode"].unsqueeze(-1)
    )
    if not bool(selected_valid.all()):
        raise Stage1TrainingError("open-loop planner selected an invalid hard-mask mode")

    training_config = payload.get("training_config")
    if not isinstance(training_config, Mapping) or not isinstance(
        training_config.get("loss"), Mapping
    ):
        raise Stage1TrainingError("checkpoint loss config is invalid")
    result = loss_from_batch(
        JointStage1Loss(Stage1LossConfig(**dict(training_config["loss"]))).to(device),
        first,
        batch,
    )
    return {
        "samples": int(batch["bev"].shape[0]),
        "selected_mode": first["selected_mode"].detach().cpu().tolist(),
        "metrics": result.scalar_metrics(),
        "deterministic": True,
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


@torch.no_grad()
def validate_closed_loop(
    planner: BEVOnlyDiffusionPlanner,
    *,
    device: torch.device,
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
        trajectories = output["selected_trajectory"].squeeze(0).detach().cpu().numpy()
        if trajectories.shape != (3, 8, 3) or not np.isfinite(trajectories).all():
            raise Stage1TrainingError("closed-loop planner produced invalid trajectories")
        actions = {
            agent_id: np.ascontiguousarray(trajectories[index], dtype=np.float32)
            for index, agent_id in enumerate(AGENT_IDS)
        }
        started = time.perf_counter()
        observation, reward, terminated, truncated, info = env.step(actions)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not all(
            isinstance(value, Mapping)
            for value in (observation, reward, terminated, truncated, info)
        ):
            raise Stage1TrainingError("closed-loop environment returned invalid mappings")
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
        return {
            "warmup_steps": step_index,
            "selected_mode": output["selected_mode"].squeeze(0).cpu().tolist(),
            "environment_step_ms": elapsed_ms,
            "terminated": bool(terminated.get("__all__", False)),
            "truncated": bool(truncated.get("__all__", False)),
        }
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    report: dict[str, object] = {"seed": 17, "device": str(device), "variants": {}}
    for variant, checkpoint in (
        ("A", args.checkpoint_a),
        ("B", args.checkpoint_b),
    ):
        planner, payload = load_planner(checkpoint, device)
        if payload["variant"] != variant:
            raise Stage1TrainingError(
                f"checkpoint-{variant.lower()} does not contain variant {variant}"
            )
        report["variants"][variant] = {
            "checkpoint": str(checkpoint),
            "open_loop": validate_open_loop(
                planner,
                payload,
                dataset_root=args.dataset_root,
                device=device,
            ),
            "closed_loop": validate_closed_loop(planner, device=device),
        }
        del planner
        if device.type == "cuda":
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
