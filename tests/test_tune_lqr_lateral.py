from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_tuner_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "tune_lqr_lateral.py"
    spec = importlib.util.spec_from_file_location("tune_lqr_lateral", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_collect_episode_lqr_stats_scores_follower_lqr_steps() -> None:
    module = _load_tuner_module()
    payload = {
        "steps": [
            {
                "info": {"agent1": {"crash": False, "out_of_road": False}},
                "pdms": {"agent1": {"formation_lat": 0.8}},
                "control_debug": {
                    "agent0": {"mode": "leader_pid"},
                    "agent1": {
                        "mode": "follower_lqr",
                        "lat_error": 0.3,
                        "heading_error": 0.05,
                        "clipped_steering": 0.2,
                        "K_lat": [2.0, 3.0],
                    },
                },
            }
            for _ in range(20)
        ]
    }

    stats = module.collect_episode_lqr_stats(payload)

    assert stats["failed"] is False
    assert stats["follower_lqr_steps"] == 20
    assert stats["rms_lat_error"] == pytest.approx(0.3)
    assert stats["rms_heading_error"] == pytest.approx(0.05)
    assert stats["mean_pdms_formation_lat"] == pytest.approx(0.8)
    assert stats["K_lat_mean"] == pytest.approx([2.0, 3.0])
    assert stats["cost"] < 1.0


def test_reference_lat_k_returns_two_gain_values() -> None:
    module = _load_tuner_module()
    k = module.reference_lat_k(1.0, 1.0, 0.1)

    assert len(k) == 2
    assert k[0] == pytest.approx(3.1623, rel=1e-3)
