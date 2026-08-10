"""CF-5 diagnostic reward smoke using the trained surrogate and CF-3 context."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .contracts import NUM_GROUPS, ChassisExecutionCommand
from .reward import ChassisExecutionRewardEvaluator
from .storage import ChassisExecutionDataset


class _NoStepEnvironment(SimpleNamespace):
    def __init__(self) -> None:
        agents = {}
        for role, x in enumerate((0.0, -15.74, -31.48)):
            agents[f"agent{role}"] = SimpleNamespace(
                name=f"agent{role}",
                position=np.asarray([x, 0.0], dtype=np.float32),
                heading_theta=0.0,
                velocity=np.asarray([6.0, 0.0], dtype=np.float32),
                LENGTH=5.74,
                WIDTH=2.3,
            )
        super().__init__(
            agents=agents,
            step_calls=0,
            _desired_center_spacing_m=lambda *_: 15.74,
        )

    def step(self, *_args, **_kwargs):
        self.step_calls += 1
        raise RuntimeError("diagnostic reward must not call env.step")


def _model_inputs() -> SimpleNamespace:
    relation = np.zeros((3, 12), dtype=np.float32)
    relation[0, 4], relation[0, 10] = -15.74, -31.48
    relation[1, 4], relation[1, 10] = 15.74, -15.74
    relation[2, 4], relation[2, 10] = 31.48, 15.74
    bev = np.zeros((3, 8, 256, 256), dtype=np.uint8)
    bev[:, 0] = 255
    return SimpleNamespace(bev=bev, formation_relation_state=relation)


def _state_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_diagnostic(
    checkpoint: Path,
    dataset_root: Path,
    *,
    device: str,
    latency_iterations: int = 50,
) -> dict[str, object]:
    if latency_iterations <= 0:
        raise ValueError("latency_iterations must be positive")
    dataset = ChassisExecutionDataset(dataset_root, split="test")
    sample = dataset[0]
    torch_device = torch.device(device)
    base = sample["tau_cmd"].to(torch_device)
    scales = torch.tensor([0.55, 0.85, 1.15, 1.45], device=torch_device)
    tau_cmd = base.unsqueeze(0).repeat(NUM_GROUPS, 1, 1, 1)
    tau_cmd[..., :2] *= scales.reshape(NUM_GROUPS, 1, 1, 1)
    command = ChassisExecutionCommand(
        tau_cmd=tau_cmd.unsqueeze(0),
        initial_state=sample["initial_state"].unsqueeze(0).to(torch_device),
        vehicle_condition=sample["vehicle_condition"].unsqueeze(0).to(torch_device),
        controller_context=sample["controller_context"].unsqueeze(0).to(torch_device),
        controller_mode=sample["controller_mode"].unsqueeze(0).to(torch_device),
        agent_role=sample["agent_role"].unsqueeze(0).to(torch_device),
    )
    evaluator = ChassisExecutionRewardEvaluator.from_checkpoint(
        checkpoint,
        expected_dataset_fingerprint=dataset.dataset_fingerprint,
        allow_diagnostic=True,
        device=torch_device,
    )
    evaluator.geometric._prediction_planner._predicted_obstacles = lambda *args, **kwargs: []
    before = _state_sha256(evaluator.surrogate)
    env = _NoStepEnvironment()
    result = evaluator.score(env, _model_inputs(), command)
    with torch.inference_mode():
        for _ in range(5):
            evaluator.score(env, _model_inputs(), command)
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        latency_ms: list[float] = []
        for _ in range(latency_iterations):
            start = time.perf_counter()
            evaluator.score(env, _model_inputs(), command)
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            latency_ms.append((time.perf_counter() - start) * 1000.0)
    after = _state_sha256(evaluator.surrogate)
    rewards = result.rewards[0].cpu().numpy()
    if not np.isfinite(rewards).all() or float(np.ptp(rewards)) <= 1e-6:
        raise RuntimeError("execution-aware diagnostic rewards are not informative")
    if before != after:
        raise RuntimeError("frozen surrogate parameters changed during reward evaluation")
    if env.step_calls != 0 or evaluator.metadrive_candidate_branch_count != 0:
        raise RuntimeError("candidate MetaDrive execution occurred during CF-5 reward")
    p50_ms = float(np.percentile(latency_ms, 50))
    p95_ms = float(np.percentile(latency_ms, 95))
    if p95_ms > 100.0:
        raise RuntimeError(f"CF-5 reward P95 {p95_ms:.3f}ms exceeds 100ms")
    return {
        "format": "chassis_execution_reward_diagnostic_report_v1",
        "status": "passed",
        "checkpoint": str(checkpoint.resolve()),
        "dataset_fingerprint": dataset.dataset_fingerprint,
        "data_origin": evaluator.checkpoint_metadata["data_origin"],
        "diagnostic_only": evaluator.checkpoint_metadata["diagnostic_only"],
        "eligible_for_formal_training": evaluator.checkpoint_metadata[
            "eligible_for_formal_training"
        ],
        "command_source": "tau_cmd",
        "geometric_trajectory_source": "predicted_tau_a",
        "rewards": rewards.tolist(),
        "reward_span": float(np.ptp(rewards)),
        "unsafe": result.unsafe[0].cpu().tolist(),
        "components": {
            name: value[0].cpu().tolist() for name, value in result.components.items()
        },
        "surrogate_state_sha256_before": before,
        "surrogate_state_sha256_after": after,
        "surrogate_parameters_changed": False,
        "surrogate_evaluations": evaluator.evaluation_count,
        "metadrive_candidate_branches": evaluator.metadrive_candidate_branch_count,
        "env_step_calls": env.step_calls,
        "background_fixture": "empty",
        "latency": {
            "device": str(torch_device),
            "iterations": latency_iterations,
            "p50_ms": p50_ms,
            "p95_ms": p95_ms,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--latency-iterations", type=int, default=50)
    args = parser.parse_args()
    report = run_diagnostic(
        args.checkpoint,
        args.dataset_root,
        device=args.device,
        latency_iterations=args.latency_iterations,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
