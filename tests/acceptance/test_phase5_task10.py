from __future__ import annotations

import json
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "outputs" / "phase5" / "real_platoon_summary.json"
CHECKPOINT_DIR = ROOT / "checkpoints" / "platoon_rl_real"
TB_DIR = ROOT / "logs" / "platoon_rl_real"


def _summary() -> dict:
    assert SUMMARY_PATH.exists(), SUMMARY_PATH
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def test_phase5_task10_completed_100_steps():
    summary = _summary()
    assert int(summary["steps"]) == 100


def test_phase5_task10_no_nan_or_oom():
    summary = _summary()
    assert not summary["has_nan_loss"]
    assert not summary["has_oom"]


def test_phase5_task10_gpu_budget():
    summary = _summary()
    assert float(summary.get("gpu_peak_gb", 0.0)) < 16.0


def test_phase5_task10_checkpoint_count():
    ckpts = sorted(CHECKPOINT_DIR.glob("step_*.ckpt"))
    assert len(ckpts) >= 2, len(ckpts)


def test_phase5_task10_tensorboard_curves():
    event_files = sorted(TB_DIR.rglob("events.out.tfevents.*"))
    assert event_files, TB_DIR
    acc = event_accumulator.EventAccumulator(str(event_files[-1]))
    acc.Reload()
    tags = set(acc.Tags().get("scalars", []))
    required = {
        "loss",
        "rl_loss",
        "il_loss",
        "kl",
        "mean_reward",
        "formation_error",
        "collision_rate",
        "grad_norm",
    }
    assert required.issubset(tags), tags


def test_phase5_task10_reward_not_persistently_worse():
    summary = _summary()
    rewards = summary["mean_reward"]
    assert len(rewards) >= 100
    assert sum(rewards[-20:]) / 20.0 >= sum(rewards[:20]) / 20.0 - 1.0
