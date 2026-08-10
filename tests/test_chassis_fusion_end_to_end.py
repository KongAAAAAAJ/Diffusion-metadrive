from __future__ import annotations

import json
from pathlib import Path

import pytest

from chassis_execution.evaluate_end_to_end import (
    ChassisFusionEvaluationError,
    aggregate_model_rows,
)


ROOT = Path(__file__).resolve().parents[1]


def _row(index: int):
    return {
        "selected_reward": float(index),
        "candidate_unsafe_rate": 0.25 * index,
        "intervention_ade_m": 0.1 * index,
        "bev_build_ms": 1.0 + index,
        "context_build_ms": 0.5 + index,
        "sampling_ms": 3.0 + index,
        "optimizer_ms": 2.0 + index,
        "reward_ms": 4.0 + index,
        "planning_tick_ms": 10.0 + index,
        "selected_env_step_ms": 5.0 + index,
        "selected_env_steps": 1,
        "collision": index == 1,
        "out_of_road": False,
        "no_safe_group": index == 2,
        "selected_unsafe": index == 2,
        "candidate_metadrive_branches": 0,
    }


def test_cf7_machine_protocol_is_diagnostic_and_four_model() -> None:
    payload = json.loads(
        (ROOT / "schemas" / "chassis_fusion_evaluation_v1.json").read_text()
    )
    assert payload["models"] == ["A_stage1", "B_stage1", "A_cf6", "B_cf6"]
    assert payload["execution"]["candidate_metadrive_branches"] == 0
    assert payload["execution"]["selected_tau_cmd_env_steps_per_decision"] == 1
    assert payload["execution"]["surrogate_input"] == "tau_cmd"
    assert payload["fairness"][
        "latency_warmup_candidate_evaluations_per_model"
    ] == 1
    assert payload["advisory_gates"][
        "three_vehicle_planning_tick_p95_ms"
    ] == 100.0
    assert payload["provenance"]["diagnostic_only"] is True
    assert payload["provenance"]["eligible_for_formal_training"] is False
    assert payload["provenance"]["research_effect_claim"] is False


def test_aggregate_reports_safety_execution_and_latency() -> None:
    result = aggregate_model_rows([_row(0), _row(1), _row(2)])
    assert result["completed_steps"] == result["selected_env_steps"] == 3
    assert result["collision_steps"] == 1
    assert result["no_safe_group_steps"] == 1
    assert result["selected_unsafe_steps"] == 1
    assert result["candidate_metadrive_branches"] == 0
    assert result["selected_reward_mean"] == pytest.approx(1.0)
    assert result["latency_ms"]["planning_tick_ms"]["p95"] > 10.0


def test_aggregate_rejects_empty_or_nonfinite_metrics() -> None:
    with pytest.raises(ChassisFusionEvaluationError, match="no rows"):
        aggregate_model_rows([])
    bad = _row(0)
    bad["reward_ms"] = float("nan")
    with pytest.raises(ChassisFusionEvaluationError, match="finite"):
        aggregate_model_rows([bad])
