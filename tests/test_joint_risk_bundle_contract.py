from __future__ import annotations

import copy

import pytest

from expert_dataset.collect_joint_bev import JOINT_SAMPLE_SHAPES
from expert_dataset.joint_risk_bundle_contract import (
    EXTERNAL_ACTOR_ID_PATTERN,
    JointRiskBundleContractError,
    PLATOON_AGENT_TO_ACTOR_ID,
    bundle_protocol_sha256,
    load_bundle_protocol,
    validate_bundle_protocol,
)


def test_frozen_bundle_protocol_matches_base_and_identity_contracts():
    protocol = load_bundle_protocol()
    base_fields = protocol["components"]["base"]["sample_fields"]
    assert tuple(base_fields) == tuple(JOINT_SAMPLE_SHAPES)
    assert PLATOON_AGENT_TO_ACTOR_ID == {
        "agent0": "P0",
        "agent1": "P1",
        "agent2": "P2",
    }
    assert EXTERNAL_ACTOR_ID_PATTERN.fullmatch("V000")
    assert EXTERNAL_ACTOR_ID_PATTERN.fullmatch("V1234")
    assert not EXTERNAL_ACTOR_ID_PATTERN.fullmatch("V1")
    assert len(bundle_protocol_sha256()) == 64


def test_protocol_rejects_legacy_platoon_identity():
    protocol = copy.deepcopy(load_bundle_protocol())
    protocol["actor_identity"]["platoon"][0]["actor_id"] = "P1"
    protocol["actor_identity"]["platoon"][1]["actor_id"] = "P2"
    protocol["actor_identity"]["platoon"][2]["actor_id"] = "P3"
    with pytest.raises(JointRiskBundleContractError, match="P0/P1/P2"):
        validate_bundle_protocol(protocol)


def test_protocol_rejects_base_tensor_schema_drift():
    protocol = copy.deepcopy(load_bundle_protocol())
    protocol["components"]["base"]["sample_fields"]["bev"]["shape"][-1] = 128
    with pytest.raises(JointRiskBundleContractError, match="base tensor"):
        validate_bundle_protocol(protocol)


def test_protocol_requires_sidecar_superset_retention():
    protocol = copy.deepcopy(load_bundle_protocol())
    protocol["join"]["sidecar_without_base_allowed"] = False
    with pytest.raises(JointRiskBundleContractError, match="sidecar-only"):
        validate_bundle_protocol(protocol)
