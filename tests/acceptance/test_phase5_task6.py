from __future__ import annotations

import json
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = ROOT / "outputs" / "phase5" / "platoon_summary.json"
CHECKPOINT_DIR = ROOT / "checkpoints" / "platoon_rl"
TB_DIR = ROOT / "logs" / "platoon_rl"


def test_phase5_platoon_training_metrics():
    assert SUMMARY_PATH.exists(), SUMMARY_PATH
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    assert int(summary["steps"]) >= 500
    assert not summary["has_nan_loss"]
    assert not summary["has_oom"]
    assert float(summary.get("gpu_peak_gb", 0.0)) < 15.0

    formation = summary["formation_error"]
    collision = summary["collision_rate"]
    assert len(formation) >= 500
    assert len(collision) >= 500
    assert sum(formation[-50:]) / 50.0 < sum(formation[:50]) / 50.0
    assert sum(collision[-50:]) / 50.0 <= sum(collision[:50]) / 50.0

    ckpts = sorted(CHECKPOINT_DIR.glob("step_*.ckpt"))
    assert len(ckpts) >= 5, len(ckpts)

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
