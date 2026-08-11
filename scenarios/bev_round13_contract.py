"""Frozen S5--S9 scenario contract for BEV joint reward work."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Sequence

from scenarios.definitions import SCENARIO_BY_ID


PRIMARY_S5_S9_SCENARIOS: tuple[tuple[str, str], ...] = (
    ("S5_hard_brake_lead", "R1_entry_straight"),
    ("S6_background_merge_in", "R6_mainline_merge_approach"),
    ("S7_ego_merge_from_ramp", "R7_merge_core"),
    ("S8_ego_exit_to_ramp", "R6_exit_to_ramp"),
    ("S9_narrow_channel_negotiation", "R8_narrow_channel"),
)
INTERFACE_SMOKE_SCENARIOS: tuple[tuple[str, str], ...] = (
    ("S1_free_cruise_straight", "R3_mainline_straight"),
)
DEVELOPMENT_SEEDS: tuple[int, ...] = (17, 23)
HOLDOUT_SEEDS: tuple[int, ...] = (31, 47)
_V1_SNAPSHOT = Path(__file__).resolve().parent / "contracts" / "bev_primary_s5_s9_v1.json"


class BEVScenarioContractError(RuntimeError):
    """Raised when a run is not bound to the frozen S5--S9 contract."""


def _normalise(value: object) -> object:
    if dataclasses.is_dataclass(value):
        return _normalise(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _normalise(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise BEVScenarioContractError(
        f"scenario contract contains unsupported value {type(value).__name__}"
    )


def primary_scenario_contract(
    scenarios: Sequence[tuple[str, str]] = PRIMARY_S5_S9_SCENARIOS,
) -> dict[str, object]:
    pairs = tuple((str(scenario), str(route)) for scenario, route in scenarios)
    if pairs != PRIMARY_S5_S9_SCENARIOS:
        raise BEVScenarioContractError(
            "primary reward work requires the complete ordered S5--S9 contract"
        )
    return json.loads(_V1_SNAPSHOT.read_text(encoding="utf-8"))


def candidate_scenario_contract_v2(
    scenarios: Sequence[tuple[str, str]] = PRIMARY_S5_S9_SCENARIOS,
    *,
    frozen: bool = False,
) -> dict[str, object]:
    pairs = tuple((str(scenario), str(route)) for scenario, route in scenarios)
    if pairs != PRIMARY_S5_S9_SCENARIOS:
        raise BEVScenarioContractError(
            "candidate reward work requires the complete ordered S5--S9 contract"
        )
    definitions = []
    for scenario_id, route in pairs:
        definition = SCENARIO_BY_ID.get(scenario_id)
        if definition is None:
            raise BEVScenarioContractError(f"unknown scenario {scenario_id}")
        if (
            route not in definition.allowed_local_routes
            or route not in definition.trigger_by_local_route
        ):
            raise BEVScenarioContractError(
                f"{scenario_id}/{route} violates the runtime route contract"
            )
        definitions.append(
            {
                "scenario": scenario_id,
                "route": route,
                "definition": _normalise(definition),
            }
        )
    payload: dict[str, object] = {
        "format": (
            "bev_primary_s5_s9_contract_v2"
            if frozen else "bev_primary_s5_s9_contract_v2_candidate"
        ),
        "scenarios": [list(value) for value in pairs],
        "definitions": definitions,
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    payload["sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def validate_primary_scenario_contract(
    payload: object,
    scenarios: Sequence[tuple[str, str]] = PRIMARY_S5_S9_SCENARIOS,
) -> str:
    expected = primary_scenario_contract(scenarios)
    if not isinstance(payload, dict) or payload != expected:
        raise BEVScenarioContractError(
            "scenario contract/hash no longer matches frozen S5--S9 definitions"
        )
    return str(expected["sha256"])


def deterministic_initial_speed_km_h(
    scenario_id: str, seed: int
) -> float:
    rows = {
        str(row["scenario"]): row["definition"]
        for row in primary_scenario_contract()["definitions"]
    }
    definition = rows.get(str(scenario_id))
    if definition is None:
        raise BEVScenarioContractError(f"unknown scenario {scenario_id}")
    value = definition.get("ego_initial_speed_km_h")
    if value is None:
        return 25.0
    if isinstance(value, (tuple, list)) and len(value) == 2:
        digest = hashlib.sha256(
            f"{scenario_id}:{int(seed)}:initial_speed".encode("ascii")
        ).digest()
        rng = np.random.RandomState(
            int.from_bytes(digest[:4], byteorder="little", signed=False)
        )
        return float(rng.uniform(float(value[0]), float(value[1])))
    return float(value)


def deterministic_candidate_initial_speed_km_h(scenario_id: str, seed: int) -> float:
    """Candidate-v2 speed resolver; v1 resolver remains stable for old data."""

    return _deterministic_speed_from_definition(
        SCENARIO_BY_ID.get(str(scenario_id)), str(scenario_id), int(seed)
    )


def _deterministic_speed_from_definition(definition, scenario_id: str, seed: int) -> float:
    if definition is None:
        raise BEVScenarioContractError(f"unknown scenario {scenario_id}")
    value = definition.ego_initial_speed_km_h
    if value is None:
        return 25.0
    if isinstance(value, (tuple, list)) and len(value) == 2:
        digest = hashlib.sha256(
            f"{scenario_id}:{int(seed)}:initial_speed".encode("ascii")
        ).digest()
        rng = np.random.RandomState(int.from_bytes(digest[:4], "little"))
        return float(rng.uniform(float(value[0]), float(value[1])))
    return float(value)


__all__ = [
    "BEVScenarioContractError",
    "DEVELOPMENT_SEEDS",
    "HOLDOUT_SEEDS",
    "INTERFACE_SMOKE_SCENARIOS",
    "PRIMARY_S5_S9_SCENARIOS",
    "primary_scenario_contract",
    "candidate_scenario_contract_v2",
    "deterministic_initial_speed_km_h",
    "deterministic_candidate_initial_speed_km_h",
    "validate_primary_scenario_contract",
]
