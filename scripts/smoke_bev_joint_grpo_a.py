#!/usr/bin/env python3
"""Run one diagnostic Variant-A joint GRPO update on a packed real sample."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch

from expert_dataset.joint_bev_dataset import (
    JointBEVDataset,
    JointBEVDatasetConfig,
)
from models.bev_planner import JointGRPOError
from train.bev_joint_grpo import (
    grpo_checkpoint_payload,
    load_grpo_checkpoint,
    load_stage1_a_for_grpo,
    save_grpo_checkpoint,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def _snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _max_delta(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
) -> float:
    if before.keys() != after.keys():
        raise JointGRPOError("diagnostic state keys changed during update")
    return max(
        (
            float(
                (after[name].detach().cpu().float() - before[name].float())
                .abs()
                .max()
            )
            for name in before
        ),
        default=0.0,
    )


def _model_inputs(
    sample: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    names = (
        "bev",
        "ego_state",
        "formation_relation_state",
        "relation_valid_mask",
        "agent_role",
        "coarse_trajectories",
        "mode_valid_mask",
    )
    return {
        name: sample[name].unsqueeze(0).to(
            device=device,
            non_blocking=device.type == "cuda",
        )
        for name in names
    }


def main() -> int:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise JointGRPOError("CUDA diagnostic requested but CUDA is unavailable")
    device = torch.device(args.device)
    trainer, source_payload, source_sha256 = load_stage1_a_for_grpo(
        args.stage1_checkpoint,
        device=device,
        allow_diagnostic_source=True,
    )
    dataset = JointBEVDataset(
        JointBEVDatasetConfig(
            dataset_root=args.dataset_root,
            split="train",
            mmap_cache_episodes=1,
        )
    )
    if len(dataset) <= 0:
        raise JointGRPOError("diagnostic dataset train split is empty")
    if dataset.contract["dataset_fingerprint"] != source_payload[
        "dataset_fingerprint"
    ]:
        raise JointGRPOError(
            "diagnostic dataset fingerprint does not match Stage 1 source"
        )
    model_inputs = _model_inputs(dataset[0], device)

    before = {
        "diffusion_decoder": _snapshot(trainer.planner.diffusion_decoder),
        "mode_head": _snapshot(trainer.planner.mode_head),
        "backbone": _snapshot(trainer.planner.backbone),
        "bev_fusion": _snapshot(trainer.planner.bev_fusion),
        "context_encoder": _snapshot(trainer.planner.context_encoder),
        "reference": _snapshot(trainer.reference),
    }
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    rollout = trainer.sample_groups(
        model_inputs,
        generator=torch.Generator(device=device).manual_seed(args.seed),
    )
    rewards = torch.tensor(
        [[-1.5, -0.5, 0.5, 1.5]],
        dtype=torch.float32,
        device=device,
    )
    update = trainer.update(rollout, rewards)
    post_update_loss = trainer.compute_loss(rollout, rewards)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    after = {
        "diffusion_decoder": _snapshot(trainer.planner.diffusion_decoder),
        "mode_head": _snapshot(trainer.planner.mode_head),
        "backbone": _snapshot(trainer.planner.backbone),
        "bev_fusion": _snapshot(trainer.planner.bev_fusion),
        "context_encoder": _snapshot(trainer.planner.context_encoder),
        "reference": _snapshot(trainer.reference),
    }
    deltas = {
        name: _max_delta(before[name], after[name])
        for name in before
    }
    if deltas["diffusion_decoder"] <= 0.0 or deltas["mode_head"] <= 0.0:
        raise JointGRPOError("diagnostic trainable policy did not update")
    for name in ("backbone", "bev_fusion", "context_encoder", "reference"):
        if deltas[name] != 0.0:
            raise JointGRPOError(f"frozen diagnostic module changed: {name}")

    metrics = update.loss.scalar_metrics()
    metrics.update(
        {
            "replay/mode_log_prob_max_abs": float(
                (
                    update.loss.new_mode_log_prob
                    - rollout.old_mode_log_prob
                )
                .abs()
                .max()
                .detach()
                .cpu()
            ),
            "replay/trajectory_log_prob_max_abs": float(
                (
                    update.loss.new_trajectory_log_prob
                    - rollout.old_trajectory_log_prob
                )
                .abs()
                .max()
                .detach()
                .cpu()
            ),
            "post_update/reference_kl": float(
                post_update_loss.reference_kl.detach().cpu()
            ),
            "gradient/diffusion_decoder": update.gradient_norms[
                "diffusion_decoder"
            ],
            "gradient/mode_head": update.gradient_norms["mode_head"],
            "gradient/total_before_clip": update.total_gradient_norm,
        }
    )
    checkpoint_payload = grpo_checkpoint_payload(
        trainer=trainer,
        source_stage1_sha256=source_sha256,
        source_stage1_payload=source_payload,
        metrics=metrics,
        diagnostic_only=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_grpo_checkpoint(
        args.output / "diagnostic_grpo_a.pt",
        checkpoint_payload,
    )

    restored, _, restored_source_sha = load_stage1_a_for_grpo(
        args.stage1_checkpoint,
        device=device,
        allow_diagnostic_source=True,
    )
    load_grpo_checkpoint(
        checkpoint_path,
        restored,
        expected_source_stage1_sha256=restored_source_sha,
    )
    if _max_delta(
        _snapshot(trainer.planner),
        _snapshot(restored.planner),
    ) != 0.0:
        raise JointGRPOError("diagnostic checkpoint planner round-trip failed")
    if _max_delta(
        _snapshot(trainer.reference),
        _snapshot(restored.reference),
    ) != 0.0:
        raise JointGRPOError("diagnostic checkpoint reference round-trip failed")

    report = {
        "format": "bev_joint_grpo_a_smoke_v1",
        "diagnostic_only": True,
        "eligible_for_formal_training": False,
        "reward_source": "external_ordered_signed_test_values",
        "source_stage1_checkpoint": str(args.stage1_checkpoint.resolve()),
        "source_stage1_sha256": source_sha256,
        "source_dataset_fingerprint": source_payload["dataset_fingerprint"],
        "dataset_root": str(args.dataset_root.resolve()),
        "device": str(device),
        "seed": int(args.seed),
        "optimizer_steps": trainer.optimizer_step,
        "roll_timesteps": list(trainer.config.roll_timesteps),
        "stochastic_timesteps": list(trainer.config.stochastic_timesteps),
        "rollout_shapes": {
            "chains_normalized": list(rollout.chains_normalized.shape),
            "sampled_modes": list(rollout.sampled_modes.shape),
            "selected_trajectories": list(
                rollout.selected_trajectories.shape
            ),
            "old_mode_log_prob": list(rollout.old_mode_log_prob.shape),
            "old_trajectory_log_prob": list(
                rollout.old_trajectory_log_prob.shape
            ),
        },
        "rewards": rewards.detach().cpu().tolist(),
        "advantages": update.loss.advantages.detach().cpu().tolist(),
        "metrics": metrics,
        "parameter_max_abs_delta": deltas,
        "peak_cuda_memory_mib": (
            float(torch.cuda.max_memory_allocated(device) / (1024**2))
            if device.type == "cuda"
            else 0.0
        ),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_round_trip": True,
    }
    report_path = args.output / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
