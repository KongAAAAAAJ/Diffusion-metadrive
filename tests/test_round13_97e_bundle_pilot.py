from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import round13_97e_bundle_pilot as pilot


SCENARIOS = (
    "S5_hard_brake_lead",
    "S6_background_merge_in",
    "S7_ego_merge_from_ramp",
    "S8_ego_exit_to_ramp",
    "S9_narrow_channel_negotiation",
)


def _fixture(tmp_path: Path, monkeypatch, *, samples: int = 10) -> Path:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "dataset_bundle_manifest.json").write_text(
        json.dumps(
            {
                "base_directory": "platoon_joint_bev",
                "sidecar_directory": "riskentry_actor_sidecar",
                "scenario_contract_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "scenario_id": name,
        }
        for name in SCENARIOS
    ]
    (root / "bundle_episode_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    scenario_samples = {name: samples // 5 for name in SCENARIOS}
    monkeypatch.setattr(
        pilot,
        "load_bundle_protocol",
        lambda: {"schema_version": "1.0.0"},
    )
    monkeypatch.setattr(
        pilot,
        "read_round13_json",
        lambda path: {"frozen": True},
    )
    monkeypatch.setattr(
        pilot,
        "verify_round13_closeout",
        lambda payload, root: {"status": "infrastructure_accepted"},
    )
    monkeypatch.setattr(
        pilot,
        "verify_joint_risk_bundle",
        lambda path: {
            "base_joint_samples": samples,
            "attempted_episodes": 5,
            "committed_base_episodes": 5,
            "committed_sidecar_episodes": 5,
            "sidecar_only_episodes": 0,
            "sidecar_raw_steps": 405,
            "base_dataset_fingerprint": "b" * 64,
            "sidecar_dataset_fingerprint": "c" * 64,
            "bundle_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(
        pilot,
        "verify_joint_bev_dataset",
        lambda path, min_decode_samples_per_s: {
            "scenario_joint_samples": scenario_samples,
            "decode_joint_samples_per_s": 500.0,
            "splits": {
                "train": {"joint_samples": max(samples - 2, 1)},
                "val": {"joint_samples": 1},
                "test": {"joint_samples": 1},
            },
        },
    )
    monkeypatch.setattr(
        pilot,
        "verify_riskentry_sidecar_dataset",
        lambda path: {"outcomes": {"success": 5}, "events": {}},
    )
    return root


def test_protocol_pilot_is_diagnostic_and_cannot_enable_stage1(
    tmp_path: Path, monkeypatch
) -> None:
    root = _fixture(tmp_path, monkeypatch, samples=10)
    report = pilot.audit_shared_bundle_pilot(
        root,
        minimum_base_samples=10,
        require_all_s5_s9=True,
        require_formal_15000=False,
    )
    assert report["status"] == "protocol_pilot_accepted"
    assert report["diagnostic_only"] is True
    assert report["eligible_for_stage1_formal_training"] is False
    assert report["eligible_for_formal_50k_collection"] is False
    assert report["sidecar_raw_steps"] == 405


def test_formal_gate_is_physical_count_based(tmp_path: Path, monkeypatch) -> None:
    root = _fixture(tmp_path, monkeypatch, samples=15_000)
    report = pilot.audit_shared_bundle_pilot(
        root,
        minimum_base_samples=15_000,
        require_all_s5_s9=True,
        require_formal_15000=True,
    )
    assert report["status"] == "formal_15000_pilot_accepted"
    assert report["eligible_for_formal_50k_collection"] is True
    assert report["eligible_for_stage1_formal_training"] is False


def test_missing_scenario_or_metadata_shortcut_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    root = _fixture(tmp_path, monkeypatch, samples=10)
    original = pilot.verify_joint_bev_dataset
    monkeypatch.setattr(
        pilot,
        "verify_joint_bev_dataset",
        lambda path, min_decode_samples_per_s: {
            **original(path, min_decode_samples_per_s),
            "scenario_joint_samples": {SCENARIOS[0]: 10},
        },
    )
    with pytest.raises(pilot.Round1397ePilotError, match="no committed base sample"):
        pilot.audit_shared_bundle_pilot(
            root,
            minimum_base_samples=10,
            require_all_s5_s9=True,
            require_formal_15000=False,
        )
    with pytest.raises(pilot.Round1397ePilotError, match="at least 15,000"):
        pilot.audit_shared_bundle_pilot(
            root,
            minimum_base_samples=10,
            require_all_s5_s9=False,
            require_formal_15000=True,
        )
