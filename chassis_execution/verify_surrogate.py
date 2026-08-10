"""Strict CF-4 checkpoint, determinism and inference-latency verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .contracts import NUM_GROUPS, ChassisExecutionCommand
from .storage import ChassisExecutionDataset, verify_chassis_execution_dataset
from .training import load_surrogate_checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def verify_surrogate_checkpoint(
    checkpoint: Path | str,
    dataset_root: Path | str,
    *,
    allow_diagnostic: bool,
    device: str = "cpu",
    latency_iterations: int = 50,
) -> dict[str, Any]:
    if latency_iterations <= 0:
        raise ValueError("latency_iterations must be positive")
    dataset_report = verify_chassis_execution_dataset(dataset_root)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, payload = load_surrogate_checkpoint(
        checkpoint,
        expected_dataset_fingerprint=dataset_report["dataset_fingerprint"],
        allow_diagnostic=allow_diagnostic,
        map_location=target_device,
    )
    model = model.frozen_copy().to(target_device).eval()
    dataset = ChassisExecutionDataset(dataset_root, split="test")
    raw = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    batch = {name: value.to(target_device) for name, value in raw.items()}
    command = ChassisExecutionCommand(
        tau_cmd=batch["tau_cmd"].unsqueeze(1).expand(-1, NUM_GROUPS, -1, -1, -1),
        initial_state=batch["initial_state"],
        vehicle_condition=batch["vehicle_condition"],
        controller_context=batch["controller_context"],
        controller_mode=batch["controller_mode"],
        agent_role=batch["agent_role"],
    )
    with torch.inference_mode():
        first = model.predict(command)
        second = model.predict(command)
    maximum_repeat_error = 0.0
    for field in first.__dataclass_fields__:
        first_value = getattr(first, field)
        second_value = getattr(second, field)
        if not bool(torch.isfinite(first_value).all()):
            raise RuntimeError(f"non-finite surrogate prediction: {field}")
        maximum_repeat_error = max(
            maximum_repeat_error,
            float((first_value - second_value).abs().max().cpu()),
        )
    if maximum_repeat_error != 0.0:
        raise RuntimeError("surrogate inference is not deterministic for a fixed input")
    with torch.inference_mode():
        for _ in range(10):
            model.predict(command)
        _synchronize(target_device)
        latencies_ms: list[float] = []
        for _ in range(latency_iterations):
            start = time.perf_counter()
            model.predict(command)
            _synchronize(target_device)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    if p95 > 100.0:
        raise RuntimeError(f"surrogate B=1,G=4 inference P95 {p95:.3f}ms exceeds 100ms")
    acceptance = payload.get("metrics", {}).get("acceptance", {})
    if acceptance.get("passed") is not True:
        raise RuntimeError("checkpoint training acceptance did not pass")
    path = Path(checkpoint).expanduser().resolve()
    return {
        "format": "chassis_execution_surrogate_verification_report_v1",
        "status": "passed",
        "checkpoint": str(path),
        "checkpoint_sha256": _sha256(path),
        "dataset_fingerprint": payload["dataset_fingerprint"],
        "data_origin": payload["data_origin"],
        "diagnostic_only": payload["diagnostic_only"],
        "eligible_for_formal_training": payload["eligible_for_formal_training"],
        "all_parameters_frozen": not any(
            parameter.requires_grad for parameter in model.parameters()
        ),
        "maximum_repeat_error": maximum_repeat_error,
        "latency": {
            "device": str(target_device),
            "batch": 1,
            "groups": NUM_GROUPS,
            "iterations": latency_iterations,
            "p50_ms": p50,
            "p95_ms": p95,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-diagnostic", action="store_true")
    parser.add_argument("--latency-iterations", type=int, default=50)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = verify_surrogate_checkpoint(
        args.checkpoint,
        args.dataset_root,
        allow_diagnostic=args.allow_diagnostic,
        device=args.device,
        latency_iterations=args.latency_iterations,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
