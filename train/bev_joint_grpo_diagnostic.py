"""Shared one-step GPU diagnostic for Variant-A/B joint GRPO."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping

import torch

from expert_dataset.joint_bev_dataset import (
    JointBEVDataset,
    JointBEVDatasetConfig,
)
from models.bev_planner import (
    JointGRPOError,
    JointGRPORolloutB,
    JointGRPOTrainerA,
    JointGRPOTrainerB,
)
from train.bev_joint_grpo import (
    grpo_b_checkpoint_payload,
    grpo_checkpoint_payload,
    load_grpo_b_checkpoint,
    load_grpo_checkpoint,
    load_stage1_a_for_grpo,
    load_stage1_b_for_grpo,
    save_grpo_checkpoint,
)


def _parse_args(variant: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run one diagnostic Variant-{variant} joint GRPO update"
    )
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


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_source(
    variant: str,
    checkpoint: Path,
    device: torch.device,
) -> tuple[
    JointGRPOTrainerA | JointGRPOTrainerB,
    dict[str, object],
    str,
]:
    loader = (
        load_stage1_a_for_grpo
        if variant == "A"
        else load_stage1_b_for_grpo
    )
    return loader(
        checkpoint,
        device=device,
        allow_diagnostic_source=True,
    )


def run_diagnostic(variant: str, args: argparse.Namespace) -> dict[str, object]:
    if variant not in ("A", "B"):
        raise JointGRPOError("diagnostic variant must be A or B")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise JointGRPOError("CUDA diagnostic requested but CUDA is unavailable")
    device = torch.device(args.device)
    trainer, source_payload, source_sha256 = _load_source(
        variant,
        args.stage1_checkpoint,
        device,
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
    if variant == "B":
        encoder = trainer.planner.diffusion_decoder.predecessor_action_encoder
        gate = trainer.planner.diffusion_decoder.predecessor_residual_gate
        if encoder is None or gate is None:
            raise JointGRPOError("Variant B condition modules are missing")
        before["predecessor_action_encoder"] = _snapshot(encoder)
        gate_before = gate.detach().cpu().clone()
    else:
        gate_before = None

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    sample_start = time.perf_counter()
    rollout = trainer.sample_groups(
        model_inputs,
        generator=torch.Generator(device=device).manual_seed(args.seed),
    )
    _synchronize(device)
    sample_ms = 1000.0 * (time.perf_counter() - sample_start)
    rewards = torch.tensor(
        [[-1.5, -0.5, 0.5, 1.5]],
        dtype=torch.float32,
        device=device,
    )
    update_start = time.perf_counter()
    update = trainer.update(rollout, rewards)
    _synchronize(device)
    update_ms = 1000.0 * (time.perf_counter() - update_start)
    post_update_loss = trainer.compute_loss(rollout, rewards)
    _synchronize(device)

    after = {
        "diffusion_decoder": _snapshot(trainer.planner.diffusion_decoder),
        "mode_head": _snapshot(trainer.planner.mode_head),
        "backbone": _snapshot(trainer.planner.backbone),
        "bev_fusion": _snapshot(trainer.planner.bev_fusion),
        "context_encoder": _snapshot(trainer.planner.context_encoder),
        "reference": _snapshot(trainer.reference),
    }
    if variant == "B":
        encoder = trainer.planner.diffusion_decoder.predecessor_action_encoder
        gate = trainer.planner.diffusion_decoder.predecessor_residual_gate
        if encoder is None or gate is None or gate_before is None:
            raise JointGRPOError("Variant B condition modules disappeared")
        after["predecessor_action_encoder"] = _snapshot(encoder)
        gate_delta = float(
            (gate.detach().cpu().float() - gate_before.float()).abs()
        )
    else:
        gate_delta = 0.0
    deltas = {
        name: _max_delta(before[name], after[name])
        for name in before
    }
    deltas["predecessor_residual_gate"] = gate_delta
    if deltas["diffusion_decoder"] <= 0.0 or deltas["mode_head"] <= 0.0:
        raise JointGRPOError("diagnostic trainable policy did not update")
    for name in ("backbone", "bev_fusion", "context_encoder", "reference"):
        if deltas[name] != 0.0:
            raise JointGRPOError(f"frozen diagnostic module changed: {name}")
    if variant == "B" and (
        deltas["predecessor_action_encoder"] <= 0.0
        or deltas["predecessor_residual_gate"] <= 0.0
    ):
        raise JointGRPOError("Variant B condition modules did not update")

    mode_replay_error = float(
        (
            update.loss.new_mode_log_prob - rollout.old_mode_log_prob
        )
        .abs()
        .max()
        .detach()
        .cpu()
    )
    trajectory_replay_error = float(
        (
            update.loss.new_trajectory_log_prob
            - rollout.old_trajectory_log_prob
        )
        .abs()
        .max()
        .detach()
        .cpu()
    )
    if mode_replay_error > 2e-5 or trajectory_replay_error > 2e-5:
        raise JointGRPOError("fixed-chain log-prob replay exceeded tolerance")
    metrics = update.loss.scalar_metrics()
    metrics.update(
        {
            "replay/mode_log_prob_max_abs": mode_replay_error,
            "replay/trajectory_log_prob_max_abs": trajectory_replay_error,
            "post_update/reference_kl": float(
                post_update_loss.reference_kl.detach().cpu()
            ),
            "gradient/total_before_clip": update.total_gradient_norm,
            "timing/sample_groups_ms": sample_ms,
            "timing/update_ms": update_ms,
        }
    )
    metrics.update(
        {
            f"gradient/{name}": value
            for name, value in update.gradient_norms.items()
        }
    )
    payload_builder = (
        grpo_checkpoint_payload
        if variant == "A"
        else grpo_b_checkpoint_payload
    )
    checkpoint_payload = payload_builder(
        trainer=trainer,
        source_stage1_sha256=source_sha256,
        source_stage1_payload=source_payload,
        metrics=metrics,
        diagnostic_only=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_grpo_checkpoint(
        args.output / f"diagnostic_grpo_{variant.lower()}.pt",
        checkpoint_payload,
    )

    restored, _, restored_source_sha = _load_source(
        variant,
        args.stage1_checkpoint,
        device,
    )
    checkpoint_loader = (
        load_grpo_checkpoint
        if variant == "A"
        else load_grpo_b_checkpoint
    )
    checkpoint_loader(
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

    rollout_shapes = {
        "chains_normalized": list(rollout.chains_normalized.shape),
        "sampled_modes": list(rollout.sampled_modes.shape),
        "selected_trajectories": list(
            rollout.selected_trajectories.shape
        ),
        "old_mode_log_prob": list(rollout.old_mode_log_prob.shape),
        "old_trajectory_log_prob": list(
            rollout.old_trajectory_log_prob.shape
        ),
    }
    if isinstance(rollout, JointGRPORolloutB):
        rollout_shapes["predecessor_action_history_normalized"] = list(
            rollout.predecessor_action_history_normalized.shape
        )
    report: dict[str, object] = {
        "format": f"bev_joint_grpo_{variant.lower()}_smoke_v1",
        "variant": variant,
        "predecessor_condition": (
            "none" if variant == "A" else "predicted_detached"
        ),
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
        "rollout_shapes": rollout_shapes,
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
    return report


def diagnostic_main(variant: str) -> int:
    report = run_diagnostic(variant, _parse_args(variant))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


__all__ = ["diagnostic_main", "run_diagnostic"]
