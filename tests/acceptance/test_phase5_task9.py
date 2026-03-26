from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "outputs" / "phase5" / "real_single_summary.json"


def _summary() -> dict:
    assert SUMMARY_PATH.exists(), SUMMARY_PATH
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def test_phase5_task9_completed_20_steps():
    summary = _summary()
    assert int(summary["steps"]) == 20


def test_phase5_task9_losses_finite():
    summary = _summary()
    losses = summary["loss"]
    assert all(value == value and abs(value) < 1e6 for value in losses)


def test_phase5_task9_gpu_budget():
    summary = _summary()
    assert float(summary.get("gpu_peak_gb", 0.0)) < 14.0


def test_phase5_task9_grad_norm_bound():
    summary = _summary()
    assert max(summary["grad_norm"]) < 200.0


def test_phase5_task9_kl_bound():
    summary = _summary()
    assert max(summary["kl"]) < 10.0
