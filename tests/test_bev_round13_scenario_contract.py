from __future__ import annotations

import copy

import pytest

from scenarios.bev_round13_contract import (
    BEVScenarioContractError,
    DEVELOPMENT_SEEDS,
    HOLDOUT_SEEDS,
    INTERFACE_SMOKE_SCENARIOS,
    PRIMARY_S5_S9_SCENARIOS,
    primary_scenario_contract,
    validate_primary_scenario_contract,
)


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
    assert validate_primary_scenario_contract(first) == first["sha256"]


def test_contract_rejects_partial_or_tampered_scenario_sets() -> None:
    with pytest.raises(BEVScenarioContractError):
        primary_scenario_contract(PRIMARY_S5_S9_SCENARIOS[:-1])
    payload = copy.deepcopy(primary_scenario_contract())
    payload["definitions"][0]["definition"]["description"] = "changed"
    with pytest.raises(BEVScenarioContractError):
        validate_primary_scenario_contract(payload)
