from __future__ import annotations

import copy

import pytest

from scenarios.bev_round13_contract import (
    BEVScenarioContractError,
    DEVELOPMENT_SEEDS,
    HOLDOUT_SEEDS,
    INTERFACE_SMOKE_SCENARIOS,
    PRIMARY_S5_S9_SCENARIOS,
    candidate_scenario_contract_v2,
    primary_scenario_contract,
    validate_primary_scenario_contract,
)
from scenarios.definitions import SCENARIO_BY_ID


def test_primary_contract_is_complete_stable_and_separate_from_smoke() -> None:
    assert [value[0][:2] for value in PRIMARY_S5_S9_SCENARIOS] == [
        "S5",
        "S6",
        "S7",
        "S8",
        "S9",
    ]
    assert all(value not in PRIMARY_S5_S9_SCENARIOS for value in INTERFACE_SMOKE_SCENARIOS)
    assert DEVELOPMENT_SEEDS == (17, 23)
    assert HOLDOUT_SEEDS == (31, 47)
    first = primary_scenario_contract()
    second = primary_scenario_contract()
    assert first == second
    assert len(first["sha256"]) == 64
    assert first["sha256"] == "e70b07d7e73d6969f90417441f983b49ecd6bfe48003be310fde5860e24e8e34"
    assert validate_primary_scenario_contract(first) == first["sha256"]
    assert all(
        SCENARIO_BY_ID[scenario_id].independent_trajectory_control_after_realization
        for scenario_id, _ in PRIMARY_S5_S9_SCENARIOS
    )
    candidate = candidate_scenario_contract_v2()
    assert candidate["format"] == "bev_primary_s5_s9_contract_v2_candidate"
    assert candidate["sha256"] != first["sha256"]


def test_contract_rejects_partial_or_tampered_scenario_sets() -> None:
    with pytest.raises(BEVScenarioContractError):
        primary_scenario_contract(PRIMARY_S5_S9_SCENARIOS[:-1])
    payload = copy.deepcopy(primary_scenario_contract())
    payload["definitions"][0]["definition"]["description"] = "changed"
    with pytest.raises(BEVScenarioContractError):
        validate_primary_scenario_contract(payload)
