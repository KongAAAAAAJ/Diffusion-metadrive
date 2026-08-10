"""Run the CF-6 A/B diagnostic chain with one real selected MetaDrive step."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from expert_dataset.collect_joint_bev import JointBEVSampleBuilder
from models.bev_planner import KinematicTrajectoryOptimizer
from scenarios.bev_round13_contract import INTERFACE_SMOKE_SCENARIOS
from train.bev_joint_grpo import (
    grpo_b_checkpoint_payload,
    grpo_checkpoint_payload,
    load_grpo_b_checkpoint,
    load_grpo_checkpoint,
    load_stage1_a_for_grpo,
    load_stage1_b_for_grpo,
    save_grpo_checkpoint,
)
from train.train_bev_joint_grpo_online import (
    AGENT_IDS,
    _new_env,
    execution_mode_valid_mask,
    model_inputs_to_batch,
    route_following_warmup_actions,
)

from .grpo_adapter import ChassisExecutionContext, ChassisFusionGRPOAdapter
from .reward import ChassisExecutionRewardEvaluator
from .storage import ChassisExecutionDataset


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _context_from_dataset(dataset_root: Path) -> tuple[ChassisExecutionContext, str]:
    dataset = ChassisExecutionDataset(dataset_root, split="test")
    sample = dataset[0]
    return (
        ChassisExecutionContext(
            initial_state=sample["initial_state"].unsqueeze(0),
            vehicle_condition=sample["vehicle_condition"].unsqueeze(0),
            controller_context=sample["controller_context"].unsqueeze(0),
            controller_mode=sample["controller_mode"].unsqueeze(0),
            agent_role=sample["agent_role"].unsqueeze(0),
        ),
        dataset.dataset_fingerprint,
    )


def _load_variant(variant: str, checkpoint: Path, device: torch.device):
    loader = load_stage1_a_for_grpo if variant == "A" else load_stage1_b_for_grpo
    return loader(
        checkpoint,
        device=device,
        allow_diagnostic_source=True,
    )


def _save_and_reload(
    *,
    variant: str,
    trainer: object,
    source_payload: dict[str, object],
    source_sha: str,
    checkpoint: Path,
    metrics: dict[str, float],
) -> int:
    payload_fn = grpo_checkpoint_payload if variant == "A" else grpo_b_checkpoint_payload
    payload = payload_fn(
        trainer=trainer,
        source_stage1_sha256=source_sha,
        source_stage1_payload=source_payload,
        metrics=metrics,
        diagnostic_only=True,
    )
    save_grpo_checkpoint(checkpoint, payload)
    del payload
    if variant == "A":
        loaded = load_grpo_checkpoint(
            checkpoint, trainer, expected_source_stage1_sha256=source_sha
        )
    else:
        loaded = load_grpo_b_checkpoint(
            checkpoint, trainer, expected_source_stage1_sha256=source_sha
        )
    step = int(loaded["optimizer_step"])
    del loaded
    return step


def _run_variant(
    *,
    variant: str,
    source_checkpoint: Path,
    surrogate_checkpoint: Path,
    chassis_dataset_root: Path,
    output_root: Path,
    device: torch.device,
) -> dict[str, object]:
    trainer, source_payload, source_sha = _load_variant(
        variant, source_checkpoint, device
    )
    optimizer = KinematicTrajectoryOptimizer()
    chassis_context, chassis_fingerprint = _context_from_dataset(
        chassis_dataset_root
    )
    evaluator = ChassisExecutionRewardEvaluator.from_checkpoint(
        surrogate_checkpoint,
        expected_dataset_fingerprint=chassis_fingerprint,
        allow_diagnostic=True,
        device=device,
    )
    adapter = ChassisFusionGRPOAdapter(trainer, optimizer, evaluator)
    scenario = tuple(INTERFACE_SMOKE_SCENARIOS[0])
    seed = 17
    env = _new_env(scenario, seed)
    builder = JointBEVSampleBuilder(AGENT_IDS)
    builder.reset()
    warmup_steps = 0
    try:
        for step_index in range(40):
            builder.capture_state(env, float(step_index) * 0.1)
            if builder.history_ready():
                break
            _, _, terminated, truncated, _ = env.step(
                route_following_warmup_actions(env, builder)
            )
            warmup_steps += 1
            if bool(terminated.get("__all__", False)) or bool(
                truncated.get("__all__", False)
            ):
                raise RuntimeError("CF-6 interface smoke ended during warm-up")
        else:
            raise RuntimeError("CF-6 interface smoke never became history-ready")
        values = builder.build_model_inputs(env)
        execution_mask = execution_mode_valid_mask(values, optimizer=optimizer)
        batch = model_inputs_to_batch(
            values, device, mode_valid_mask=execution_mask
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(1700 + (0 if variant == "A" else 1))
        result = adapter.run_update(
            env=env,
            trainer_model_inputs=batch,
            reward_model_inputs=values,
            chassis_context=chassis_context,
            generator=generator,
        )
        transition = adapter.execute_selected_once(env, result)
        if not isinstance(transition, tuple) or len(transition) != 5:
            raise RuntimeError("MetaDrive selected action returned an invalid transition")
        _, _, terminated, truncated, info = transition
        if any(
            bool(info.get(agent_id, {}).get(flag, False))
            for agent_id in AGENT_IDS
            for flag in ("crash", "out_of_road", "out_of_route")
        ):
            raise RuntimeError("CF-6 selected one-step interface smoke was unsafe")
        metrics = result.update.loss.scalar_metrics()
        metrics.update(
            {
                "reward/mean": float(result.reward.rewards.mean().cpu()),
                "reward/min": float(result.reward.rewards.min().cpu()),
                "reward/max": float(result.reward.rewards.max().cpu()),
                "trajectory_optimizer/intervention_ade_m": float(
                    np.mean(result.optimization.intervention_ade_m)
                ),
            }
        )
        variant_root = output_root / variant
        checkpoint = variant_root / "checkpoints" / "last.pt"
        reloaded_step = _save_and_reload(
            variant=variant,
            trainer=trainer,
            source_payload=source_payload,
            source_sha=source_sha,
            checkpoint=checkpoint,
            metrics=metrics,
        )
        if reloaded_step != result.update.optimizer_step:
            raise RuntimeError("CF-6 strict checkpoint reload step mismatch")
        return {
            "variant": variant,
            "source_stage1_checkpoint": str(source_checkpoint.resolve()),
            "source_stage1_sha256": source_sha,
            "source_stage1_dataset_fingerprint": source_payload[
                "dataset_fingerprint"
            ],
            "surrogate_checkpoint": str(surrogate_checkpoint.resolve()),
            "surrogate_checkpoint_sha256": _file_sha256(surrogate_checkpoint),
            "chassis_dataset_fingerprint": chassis_fingerprint,
            "data_origin": "synthetic_virtual",
            "diagnostic_only": True,
            "eligible_for_formal_training": False,
            "tau_d_is_rollout_probability_action": True,
            "tau_cmd_is_only_surrogate_input": True,
            "reward_requires_grad": result.reward.rewards.requires_grad,
            "policy_parameters_changed": (
                result.policy_sha256_before != result.policy_sha256_after
            ),
            "surrogate_parameters_changed": False,
            "trajectory_optimizer_config_sha256": result.optimizer_config_sha256,
            "metadrive_candidate_branches": result.metadrive_candidate_branches,
            "selected_metadrive_env_steps": 1,
            "selected_group": int(result.selected_group.item()),
            "rewards": result.reward.rewards[0].cpu().tolist(),
            "unsafe": result.reward.unsafe[0].cpu().tolist(),
            "optimizer_step": result.update.optimizer_step,
            "strict_checkpoint_reload_step": reloaded_step,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": _file_sha256(checkpoint),
            "scenario": list(scenario),
            "seed": seed,
            "warmup_env_steps": warmup_steps,
            "selected_step_terminated": bool(terminated.get("__all__", False)),
            "selected_step_truncated": bool(truncated.get("__all__", False)),
            "intervention_ade_m": float(
                np.mean(result.optimization.intervention_ade_m)
            ),
            "loss": metrics,
        }
    finally:
        env.close()


def run_diagnostic(
    *,
    stage1_a: Path,
    stage1_b: Path,
    surrogate_checkpoint: Path,
    chassis_dataset_root: Path,
    output_root: Path,
    device: str,
) -> dict[str, object]:
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CF-6 CUDA diagnostic requested but CUDA is unavailable")
    rows = []
    for variant, checkpoint in (("A", stage1_a), ("B", stage1_b)):
        rows.append(
            _run_variant(
                variant=variant,
                source_checkpoint=checkpoint,
                surrogate_checkpoint=surrogate_checkpoint,
                chassis_dataset_root=chassis_dataset_root,
                output_root=output_root,
                device=torch_device,
            )
        )
        gc.collect()
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
    if not all(row["policy_parameters_changed"] for row in rows):
        raise RuntimeError("CF-6 A/B policies must both change after one update")
    if any(row["metadrive_candidate_branches"] != 0 for row in rows):
        raise RuntimeError("CF-6 candidate MetaDrive branch execution detected")
    return {
        "format": "chassis_fusion_grpo_adapter_diagnostic_report_v1",
        "status": "passed",
        "data_origin": "synthetic_virtual",
        "diagnostic_only": True,
        "eligible_for_formal_training": False,
        "pending_gate": "CF-2_real_windows_trucksim_smoke",
        "variants": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-a", required=True, type=Path)
    parser.add_argument("--stage1-b", required=True, type=Path)
    parser.add_argument("--surrogate-checkpoint", required=True, type=Path)
    parser.add_argument("--chassis-dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    report = run_diagnostic(
        stage1_a=args.stage1_a,
        stage1_b=args.stage1_b,
        surrogate_checkpoint=args.surrogate_checkpoint,
        chassis_dataset_root=args.chassis_dataset_root,
        output_root=args.output_root,
        device=args.device,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
