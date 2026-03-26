from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "outputs" / "phase5" / "toy-single_summary.json"


def test_phase5_toy_training_metrics():
    assert SUMMARY_PATH.exists(), SUMMARY_PATH
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    losses = summary["loss"]
    rewards = summary["mean_reward"]
    kls = summary["kl"]
    grad_norms = summary["grad_norm"]
    gpu_peak_gb = float(summary.get("gpu_peak_gb", 0.0))

    assert len(losses) >= 100
    assert all(value == value and abs(value) < 1e6 for value in losses)
    assert rewards[-1] > rewards[0]
    assert max(kls) < 1.0
    assert max(grad_norms) < 100.0
    assert gpu_peak_gb < 10.0
