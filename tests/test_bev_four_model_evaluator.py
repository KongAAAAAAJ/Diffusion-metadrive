from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.bev_four_model_evaluator import (
    FourModelEvaluationConfig,
    FourModelEvaluationError,
    _load_manifest,
    _summarize,
)


def test_evaluation_config_and_manifest_are_strict(tmp_path: Path) -> None:
    with pytest.raises(FourModelEvaluationError):
        FourModelEvaluationConfig(device="auto")
    with pytest.raises(FourModelEvaluationError):
        FourModelEvaluationConfig(max_steps=0)

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"format": "bev_four_model_manifest_v1", "models": {}}))
    with pytest.raises(FourModelEvaluationError, match="exactly"):
        _load_manifest(bad)

    models = {
        "A": {},
        "B": {},
        "A_GRPO": {},
        "B_GRPO": {},
    }
    good = tmp_path / "good.json"
    good.write_text(
        json.dumps({"format": "bev_four_model_manifest_v1", "models": models})
    )
    assert _load_manifest(good)["models"].keys() == models.keys()


def test_metric_aggregation_computes_rates_and_p95() -> None:
    raw = {
        "roles": {
            name: {
                "collision": 1,
                "out_of_road": 0,
                "progress": [1.0, 3.0],
                "speed_km_h": [10.0, 20.0],
                "minimum_gap_m": [6.0, 8.0],
                "selected_modes": [0, 9],
                "stop": 1,
                "acceleration_mps2": [1.0],
                "jerk": [0.2],
                "yaw_rate_rad_s": [0.1],
                "steering_change": [0.05],
            }
            for name in ("agent0", "agent1", "agent2")
        },
        "episode_collision": 1,
        "episode_out_of_road": 0,
        "episode_completed": 1,
        "gap_5m_violation": 0,
        "gap_7m_violation": 1,
        "formation_error": [1.0, 2.0],
        "formation_spread": [2.0],
        "recovery_time_s": [1.0],
        "joint_reward": [0.5, 1.5],
        "timing": {
            "bev_build_ms": [1.0, 3.0],
            "model_inference_ms": [10.0, 20.0],
            "control_mapping_ms": [1.0, 2.0],
            "planning_tick_ms": [12.0, 25.0],
        },
    }
    result = _summarize(raw, 2)
    assert result["joint_safety"]["collision_rate"] == pytest.approx(0.5)
    assert result["roles"]["agent0"]["stop_rate"] == pytest.approx(0.5)
    assert result["timing"]["model_inference_ms"]["p95_ms"] == pytest.approx(
        19.5
    )
