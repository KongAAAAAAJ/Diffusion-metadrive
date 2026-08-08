from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.round13_closeout import (
    Round13CloseoutError,
    build_round13_closeout,
    verify_round13_closeout,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_round13_closeout_freezes_formal_boundary() -> None:
    payload = build_round13_closeout(REPO_ROOT)
    assert payload["status"] == "infrastructure_accepted"
    assert payload["reproducibility_policy"]["bit_exact_closed_loop_required"] is False
    assert payload["formal_eligibility"]["current_checkpoints_eligible"] is False
    assert payload["data_contract"]["storage_schema_version"] == 2
    assert payload["data_contract"]["joint_first"] is True
    assert payload["mode_contract"]["num_modes"] == 10
    assert payload["mode_contract"]["trajectory_steps"] == 8
    assert payload["latency_evidence_ms"]["full_planning_tick_p95_max"] > 100.0
    assert payload["next_hard_gate"]["name"] == "formal_joint_bev_15000_step_pilot"


def test_round13_closeout_round_trip_and_drift_rejection() -> None:
    payload = build_round13_closeout(REPO_ROOT)
    encoded = json.loads(json.dumps(payload))
    assert verify_round13_closeout(encoded, REPO_ROOT) == payload
    encoded["formal_eligibility"]["current_checkpoints_eligible"] = True
    with pytest.raises(Round13CloseoutError, match="no longer matches"):
        verify_round13_closeout(encoded, REPO_ROOT)
