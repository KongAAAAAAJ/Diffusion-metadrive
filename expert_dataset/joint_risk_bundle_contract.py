"""Machine contract shared by Diffusion-MetaDrive and RiskEntry datasets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from expert_dataset.joint_bev_storage import joint_sample_storage_contract
from scenarios.bev_round13_contract import (
    PRIMARY_S5_S9_SCENARIOS,
    primary_scenario_contract,
)
from expert_dataset.joint_risk_identity import (
    EXTERNAL_ACTOR_ID_PATTERN,
    PLATOON_ACTOR_IDS,
    PLATOON_AGENT_TO_ACTOR_ID,
)


BUNDLE_FORMAT = "metadrive-joint-planning-risk-bundle"
BUNDLE_SCHEMA_VERSION = "1.0.0"
SIDECAR_FORMAT = "riskentry-metadrive-actor-sidecar"
SIDECAR_SCHEMA_VERSION = "1.0.0"
PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas"
    / "metadrive_joint_risk_bundle_v1.json"
)
RULE_CONDITIONED_V2_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "schemas"
    / "metadrive_joint_risk_bundle_rule_conditioned_v2.json"
)


class JointRiskBundleContractError(ValueError):
    """Raised when the frozen cross-project protocol has drifted."""


def _read_protocol(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JointRiskBundleContractError(f"invalid bundle protocol JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise JointRiskBundleContractError("bundle protocol root must be an object")
    return payload


def validate_bundle_protocol(
    payload: object, *, planner_version: str = "v1"
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise JointRiskBundleContractError("bundle protocol must be an object")
    protocol = dict(payload)
    if protocol.get("format") != BUNDLE_FORMAT:
        raise JointRiskBundleContractError("bundle format mismatch")
    if protocol.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise JointRiskBundleContractError("bundle schema version mismatch")

    components = protocol.get("components")
    if not isinstance(components, Mapping):
        raise JointRiskBundleContractError("components must be an object")
    base = components.get("base")
    sidecar = components.get("sidecar")
    if not isinstance(base, Mapping) or not isinstance(sidecar, Mapping):
        raise JointRiskBundleContractError("base and sidecar component contracts are required")
    storage_contract = joint_sample_storage_contract(planner_version)
    if (
        base.get("format") != storage_contract.storage_format
        or base.get("schema_version") != storage_contract.schema_version
    ):
        raise JointRiskBundleContractError("base storage contract mismatch")
    expected_fields = {
        name: {
            "dtype": str(storage_contract.sample_dtypes[name]),
            "shape": list(storage_contract.sample_shapes[name]),
        }
        for name in storage_contract.sample_shapes
    }
    if base.get("sample_fields") != expected_fields:
        raise JointRiskBundleContractError("base tensor contract mismatch")
    if (
        sidecar.get("format") != SIDECAR_FORMAT
        or sidecar.get("schema_version") != SIDECAR_SCHEMA_VERSION
    ):
        raise JointRiskBundleContractError("RiskEntry sidecar contract mismatch")

    identity = protocol.get("actor_identity")
    if not isinstance(identity, Mapping):
        raise JointRiskBundleContractError("actor_identity must be an object")
    platoon = identity.get("platoon")
    if not isinstance(platoon, list):
        raise JointRiskBundleContractError("actor_identity.platoon must be a list")
    observed_mapping = {
        str(item.get("source_agent_id")): str(item.get("actor_id"))
        for item in platoon
        if isinstance(item, Mapping)
    }
    if observed_mapping != PLATOON_AGENT_TO_ACTOR_ID:
        raise JointRiskBundleContractError("platoon identity must be agent0/1/2 -> P0/P1/P2")
    if identity.get("external_id_pattern") != EXTERNAL_ACTOR_ID_PATTERN.pattern:
        raise JointRiskBundleContractError("external actor ID pattern mismatch")
    if identity.get("legacy_p1_p2_p3_allowed") is not False:
        raise JointRiskBundleContractError("legacy P1/P2/P3 identities must be disabled")

    binding = protocol.get("formal_research_binding")
    if not isinstance(binding, Mapping):
        raise JointRiskBundleContractError("formal_research_binding must be an object")
    expected_scenarios = [list(value) for value in PRIMARY_S5_S9_SCENARIOS]
    if binding.get("scenarios") != expected_scenarios:
        raise JointRiskBundleContractError("formal S5--S9 scenario list mismatch")
    if binding.get("scenario_contract_sha256") != primary_scenario_contract()["sha256"]:
        raise JointRiskBundleContractError("formal scenario contract hash mismatch")
    if float(binding.get("decision_dt_s", 0.0)) != 0.1:
        raise JointRiskBundleContractError("decision_dt_s must be 0.1")
    if int(binding.get("split_seed", -1)) != 17:
        raise JointRiskBundleContractError("split_seed must be 17")

    join = protocol.get("join")
    if not isinstance(join, Mapping):
        raise JointRiskBundleContractError("join must be an object")
    if join.get("base_without_sidecar_allowed") is not False:
        raise JointRiskBundleContractError("base-only episodes are forbidden")
    if join.get("sidecar_without_base_allowed") is not True:
        raise JointRiskBundleContractError("sidecar-only dangerous episodes must be allowed")
    return protocol


def load_bundle_protocol(
    path: Path | str | None = None, *, planner_version: str = "v1"
) -> dict[str, object]:
    protocol_path = (
        Path(path)
        if path is not None
        else (
            RULE_CONDITIONED_V2_PROTOCOL_PATH
            if planner_version == "v2"
            else PROTOCOL_PATH
        )
    )
    return validate_bundle_protocol(
        _read_protocol(protocol_path), planner_version=planner_version
    )


def bundle_protocol_sha256(
    path: Path | str | None = None, *, planner_version: str = "v1"
) -> str:
    protocol_path = (
        Path(path)
        if path is not None
        else (
            RULE_CONDITIONED_V2_PROTOCOL_PATH
            if planner_version == "v2"
            else PROTOCOL_PATH
        )
    )
    load_bundle_protocol(protocol_path, planner_version=planner_version)
    return hashlib.sha256(protocol_path.read_bytes()).hexdigest()


__all__ = [
    "BUNDLE_FORMAT",
    "BUNDLE_SCHEMA_VERSION",
    "EXTERNAL_ACTOR_ID_PATTERN",
    "JointRiskBundleContractError",
    "PLATOON_ACTOR_IDS",
    "PLATOON_AGENT_TO_ACTOR_ID",
    "PROTOCOL_PATH",
    "RULE_CONDITIONED_V2_PROTOCOL_PATH",
    "SIDECAR_FORMAT",
    "SIDECAR_SCHEMA_VERSION",
    "bundle_protocol_sha256",
    "load_bundle_protocol",
    "validate_bundle_protocol",
]
