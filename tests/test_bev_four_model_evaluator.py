from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from evaluation.bev_four_model_evaluator import (
    DIAGNOSTIC_EVAL_SCENARIOS,
    FourModelEvaluationConfig,
    FourModelEvaluationError,
    _file_sha256,
    _configure_deterministic_inference,
    _behavior_sha256,
    _initial_state_signature,
    _load_manifest,
    _validate_checkpoint_hash,
    _summarize,
)
from scenarios.bev_round13_contract import HOLDOUT_SEEDS, PRIMARY_S5_S9_SCENARIOS


def test_evaluation_config_and_manifest_are_strict(tmp_path: Path) -> None:
    with pytest.raises(FourModelEvaluationError):
        FourModelEvaluationConfig(device="auto")
    with pytest.raises(FourModelEvaluationError):
        FourModelEvaluationConfig(max_steps=0)
    config = FourModelEvaluationConfig(device="cpu")
    assert config.scenarios == PRIMARY_S5_S9_SCENARIOS
    assert config.seeds == HOLDOUT_SEEDS
    assert DIAGNOSTIC_EVAL_SCENARIOS == PRIMARY_S5_S9_SCENARIOS
    with pytest.raises(FourModelEvaluationError):
        FourModelEvaluationConfig(
            device="cpu",
            scenarios=(("S1_free_cruise_straight", "R3_mainline_straight"),),
        )

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
        "execution_rejections": [
            {"scenario": "S8", "seed": 31, "step": 12, "reason": "infeasible"}
        ],
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
    assert result["execution"]["rejection_count"] == 1
    assert result["execution"]["rejection_rate"] == pytest.approx(0.5)
    assert result["timing"]["model_inference_ms"]["p95_ms"] == pytest.approx(
        19.5
    )


def test_checkpoint_hash_is_strict(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"round13.93")
    digest = _file_sha256(checkpoint)
    assert _validate_checkpoint_hash(
        {"checkpoint": str(checkpoint), "checkpoint_sha256": digest},
        "checkpoint",
        "checkpoint_sha256",
    ) == checkpoint
    with pytest.raises(FourModelEvaluationError, match="SHA256 mismatch"):
        _validate_checkpoint_hash(
            {"checkpoint": str(checkpoint), "checkpoint_sha256": "0" * 64},
            "checkpoint",
            "checkpoint_sha256",
        )


def test_deterministic_inference_requires_process_hash_seed(monkeypatch) -> None:
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    with pytest.raises(FourModelEvaluationError, match="PYTHONHASHSEED=0"):
        _configure_deterministic_inference(torch.device("cpu"))
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    _configure_deterministic_inference(torch.device("cpu"))
    assert torch.are_deterministic_algorithms_enabled()


def test_initial_state_signature_is_joint_first_and_strict() -> None:
    env = SimpleNamespace(
        agents={
            f"agent{role}": SimpleNamespace(
                position=(float(role), -float(role)),
                heading_theta=0.1 * role,
                speed_km_h=20.0 + role,
            )
            for role in range(3)
        }
    )
    signature = _initial_state_signature(env)
    assert signature.shape == (3, 4)
    assert signature[2].tolist() == pytest.approx([2.0, -2.0, 0.2, 22.0])
    del env.agents["agent1"]
    with pytest.raises(FourModelEvaluationError, match="agent1"):
        _initial_state_signature(env)


def test_behavior_hash_excludes_timing_but_not_policy_metrics() -> None:
    report = {
        "models": {
            name: {"joint_safety": {"collision_rate": 0.0}, "timing": {"p95": 1.0}}
            for name in ("A", "B", "A_GRPO", "B_GRPO")
        }
    }
    baseline = _behavior_sha256(report)
    report["models"]["A"]["timing"]["p95"] = 99.0
    assert _behavior_sha256(report) == baseline
    report["models"]["A"]["joint_safety"]["collision_rate"] = 1.0
    assert _behavior_sha256(report) != baseline
