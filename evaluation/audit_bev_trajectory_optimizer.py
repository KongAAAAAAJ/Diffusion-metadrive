"""Audit raw GRPO actions and the deterministic execution transform."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from expert_dataset.joint_bev_dataset import JointBEVDataset, JointBEVDatasetConfig
from models.bev_planner.trajectory_optimizer import KinematicTrajectoryOptimizer
from train.bev_joint_grpo import load_stage1_a_for_grpo, load_stage1_b_for_grpo
from train.train_bev_joint_grpo_online import (
    model_inputs_to_batch,
    optimize_selected_model_trajectories,
)


MODEL_FIELDS = (
    "bev",
    "ego_state",
    "formation_relation_state",
    "relation_valid_mask",
    "agent_role",
    "coarse_trajectories",
    "mode_valid_mask",
)


def run_audit(
    *,
    variant: str,
    checkpoint: Path,
    dataset_root: Path,
    output: Path,
    device: str,
    max_samples: int,
    seed: int,
) -> dict[str, object]:
    torch_device = torch.device(device)
    loader = load_stage1_a_for_grpo if variant == "A" else load_stage1_b_for_grpo
    trainer, payload, checkpoint_sha = loader(
        checkpoint,
        device=torch_device,
        allow_diagnostic_source=True,
    )
    dataset = JointBEVDataset(
        JointBEVDatasetConfig(dataset_root=dataset_root, split="train")
    )
    optimizer = KinematicTrajectoryOptimizer()
    generator = torch.Generator(device=torch_device)
    generator.manual_seed(seed)
    raw_valid = []
    optimized_valid = []
    intervention_ade = []
    intervention_fde = []
    elapsed_ms = []
    retained_fraction = []
    coarse_ade = []
    coarse_fde = []
    lane_direction_matches = []
    raw_violations: dict[str, int] = {}
    evaluated = min(max_samples, len(dataset))
    for index in range(evaluated):
        item = dataset[index]
        values = SimpleNamespace(
            **{name: item[name].cpu().numpy() for name in MODEL_FIELDS}
        )
        values.as_dict = lambda values=values: {
            name: getattr(values, name) for name in MODEL_FIELDS
        }
        batch = model_inputs_to_batch(values, torch_device)
        with torch.no_grad():
            rollout = trainer.sample_groups(batch, generator=generator)
        raw = rollout.selected_trajectories[0].detach().cpu().numpy().copy()
        raw_tensor_before = rollout.selected_trajectories.detach().clone()
        modes = rollout.sampled_modes[0].detach().cpu().numpy()
        result = optimize_selected_model_trajectories(
            values, raw, modes, optimizer=optimizer
        )
        if not torch.equal(raw_tensor_before, rollout.selected_trajectories):
            raise RuntimeError("execution optimizer mutated the raw GRPO rollout")
        raw_valid.extend(result.raw_valid.reshape(-1).tolist())
        optimized_valid.extend(result.optimized_valid.reshape(-1).tolist())
        intervention_ade.extend(result.intervention_ade_m.reshape(-1).tolist())
        intervention_fde.extend(result.intervention_fde_m.reshape(-1).tolist())
        selected_coarse = np.empty_like(raw)
        for group in range(raw.shape[0]):
            for role in range(3):
                selected_coarse[group, role] = values.coarse_trajectories[
                    role, int(modes[group, role])
                ]
                mode = int(modes[group, role])
                if mode in (3, 4, 5, 6, 7, 8):
                    lane_direction_matches.append(
                        bool(
                            np.sign(result.optimized_trajectories[group, role, -1, 1])
                            == np.sign(selected_coarse[group, role, -1, 1])
                        )
                    )
        coarse_delta = np.linalg.norm(
            result.optimized_trajectories[..., :2] - selected_coarse[..., :2],
            axis=-1,
        )
        coarse_ade.extend(coarse_delta.mean(axis=-1).reshape(-1).tolist())
        coarse_fde.extend(coarse_delta[..., -1].reshape(-1).tolist())
        elapsed_ms.append(float(result.elapsed_ms))
        retained_fraction.extend(result.retained_raw_fraction.reshape(-1).tolist())
        for violations in result.raw_violations:
            for reason in violations:
                raw_violations[reason] = raw_violations.get(reason, 0) + 1

    report = {
        "format": "bev_trajectory_optimizer_audit_v1",
        "variant": variant,
        "source_checkpoint": str(checkpoint.resolve()),
        "source_checkpoint_sha256": checkpoint_sha,
        "source_dataset_fingerprint": payload["dataset_fingerprint"],
        "dataset_root": str(dataset_root.resolve()),
        "samples": evaluated,
        "trajectories": len(raw_valid),
        "raw_valid_rate": float(np.mean(raw_valid)),
        "optimized_valid_rate": float(np.mean(optimized_valid)),
        "intervention_ade_mean_m": float(np.mean(intervention_ade)),
        "intervention_fde_mean_m": float(np.mean(intervention_fde)),
        "selected_coarse_ade_mean_m": float(np.mean(coarse_ade)),
        "selected_coarse_fde_mean_m": float(np.mean(coarse_fde)),
        "lane_change_direction_match_rate": float(np.mean(lane_direction_matches))
        if lane_direction_matches
        else 1.0,
        "optimizer_latency_p50_ms": float(np.percentile(elapsed_ms, 50)),
        "optimizer_latency_p95_ms": float(np.percentile(elapsed_ms, 95)),
        "retained_raw_fraction_mean": float(np.mean(retained_fraction)),
        "raw_violations": dict(sorted(raw_violations.items())),
        "raw_grpo_rollout_unchanged": True,
        "optimizer_config_sha256": optimizer.config.sha256(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("A", "B"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    report = run_audit(**vars(args))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
