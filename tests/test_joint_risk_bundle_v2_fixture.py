from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from expert_dataset.joint_risk_bundle_v2_fixture import (
    BundleV2FixtureError,
    generate_fixture,
    verify_fixture,
)


EXPECTED_PROTOCOL_SHA256 = "21f9a8a193e1c0ca9e85ac648a0a3f9663f979ad0b3ef9518cda6f5750a7b89c"
RISKENTRY_ROOT = Path("/home/kong/RiskModeCodes")


def test_protocol_copy_is_the_frozen_riskentry_v2_protocol() -> None:
    local = Path("schemas/metadrive_joint_risk_bundle_v2.json")
    assert hashlib.sha256(local.read_bytes()).hexdigest() == EXPECTED_PROTOCOL_SHA256
    upstream = RISKENTRY_ROOT / "schemas/metadrive_joint_risk_bundle_v2.json"
    if upstream.is_file():
        assert local.read_bytes() == upstream.read_bytes()


def test_generate_and_verify_complete_bundle_v2_fixture(tmp_path: Path) -> None:
    output = tmp_path / "bundle-v2-fixture"
    report = generate_fixture(output)
    assert report["status"] == "pass"
    assert report["eligible_for_formal_training"] is False
    assert set(report["partitions"]) == {"id", "compositional_ood", "topology_ood"}
    assert all(item["episodes"] == 8 for item in report["partitions"].values())
    assert verify_fixture(output)["report_sha256"] == report["report_sha256"]

    for partition in ("id", "compositional_ood", "topology_ood"):
        root = output / f"riskentry_fixture_{partition}_v2"
        manifest = json.loads((root / "dataset_bundle_manifest.json").read_text())
        assert manifest["schema_version"] == "2.0.0"
        assert manifest["benchmark_partition"] == partition
        assert manifest["diagnostic_fixture"] is True
        assert manifest["eligible_for_formal_training"] is False


def test_fixture_verifier_rejects_observed_ground_truth_leak(tmp_path: Path) -> None:
    output = tmp_path / "bundle-v2-fixture"
    generate_fixture(output)
    episode = next(
        (output / "riskentry_fixture_id_v2" / "riskentry_actor_sidecar").glob(
            "*/episodes/*"
        )
    )
    mask = np.load(episode / "actor_observation_mask.npy", allow_pickle=False)
    observed = np.load(episode / "observed_actor_state.npy", allow_pickle=False)
    valid = np.load(
        episode / "observed_actor_state_valid_mask.npy", allow_pickle=False
    )
    mask[0, 3] = False
    valid[0, 3] = False
    observed[0, 3, 0] = 1.0
    np.save(episode / "actor_observation_mask.npy", mask, allow_pickle=False)
    np.save(episode / "observed_actor_state_valid_mask.npy", valid, allow_pickle=False)
    np.save(episode / "observed_actor_state.npy", observed, allow_pickle=False)
    with pytest.raises(BundleV2FixtureError, match="unobserved state must be zero"):
        verify_fixture(output)


def test_riskentry_v2_metadata_and_source_preflight(tmp_path: Path) -> None:
    source_root = RISKENTRY_ROOT / "src"
    if not source_root.is_dir():
        pytest.skip("RiskEntry checkout is unavailable")
    sys.path.insert(0, str(source_root))
    try:
        from riskentry.contracts.metadrive_bundle_v2 import (
            formal_bundle_protocol_sha256,
            validate_formal_episode_metadata,
        )
        from riskentry.data.preflight import audit_source_bundle

        output = tmp_path / "bundle-v2-fixture"
        generate_fixture(output)
        assert formal_bundle_protocol_sha256() == EXPECTED_PROTOCOL_SHA256
        for partition in ("id", "compositional_ood", "topology_ood"):
            root = output / f"riskentry_fixture_{partition}_v2"
            episodes = sorted(
                (root / "riskentry_actor_sidecar").glob("*/episodes/*/episode.json")
            )
            assert len(episodes) == 8
            for path in episodes:
                validate_formal_episode_metadata(json.loads(path.read_text()))
            audit = audit_source_bundle(
                {
                    "source": {
                        "bundle_root": str(root),
                        "read_only": True,
                        "format": "auto",
                        "schema_version": "2.0.0",
                    },
                    "usage": {
                        "development_only": True,
                        "formal_paper_results_allowed": False,
                    },
                }
            )
            assert audit["status"] == "pass"
            assert audit["source"]["benchmark_partition"] == partition
            assert audit["counts"]["committed_base_episodes"] == 8
            assert audit["counts"]["committed_sidecar_episodes"] == 8
    finally:
        sys.path.remove(str(source_root))

