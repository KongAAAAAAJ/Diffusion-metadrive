from __future__ import annotations

import json
from pathlib import Path

import pytest

from expert_dataset.joint_risk_bundle_storage import (
    BundleEpisodeAttempt,
    BundleEpisodeResult,
    JointRiskBundleIndex,
    JointRiskBundleStorageError,
)
from scenarios.bev_round13_contract import primary_scenario_contract


BASE_FP = "1" * 64
SIDECAR_FP = "2" * 64


def _open(root: Path, *, resume: bool) -> JointRiskBundleIndex:
    return JointRiskBundleIndex(
        root,
        base_directory="platoon_joint_bev",
        sidecar_directory="riskentry_actor_sidecar",
        base_dataset_fingerprint=BASE_FP,
        sidecar_dataset_fingerprint=SIDECAR_FP,
        scenario_contract_sha256=primary_scenario_contract()["sha256"],
        split_seed=17,
        resume=resume,
    )


def _attempt(index: int = 0) -> BundleEpisodeAttempt:
    return BundleEpisodeAttempt(
        episode_index=index,
        split="train",
        scenario_id="S5_hard_brake_lead",
        local_route="R1_entry_straight",
        spawn_seed=17 + index,
    )


def test_atomic_bundle_index_accepts_sidecar_only_and_resumes(tmp_path: Path) -> None:
    with _open(tmp_path, resume=False) as index:
        index.begin_attempt(_attempt())
        index.finalize(
            BundleEpisodeResult(
                **_attempt().__dict__,
                base_status="rejected",
                base_rejection_reason="crash_vehicle:agent0",
                sidecar_status="committed",
                sidecar_rejection_reason=None,
                raw_steps=25,
                base_samples=0,
                outcome="collision",
            )
        )
        assert index.next_episode_index == 1
        assert index.pending_attempt is None

    with _open(tmp_path, resume=True) as resumed:
        assert resumed.next_episode_index == 1
        assert resumed.rows[0].outcome == "collision"
    rows = (tmp_path / "bundle_episode_index.jsonl").read_text().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["sidecar_status"] == "committed"


def test_base_only_result_is_forbidden(tmp_path: Path) -> None:
    with _open(tmp_path, resume=False) as index:
        index.begin_attempt(_attempt())
        with pytest.raises(JointRiskBundleStorageError, match="base-only"):
            BundleEpisodeResult(
                **_attempt().__dict__,
                base_status="committed",
                base_rejection_reason=None,
                sidecar_status="rejected",
                sidecar_rejection_reason="missing",
                raw_steps=0,
                base_samples=1,
                outcome="invalid",
            )


def test_pending_attempt_survives_process_boundary(tmp_path: Path) -> None:
    with _open(tmp_path, resume=False) as index:
        index.begin_attempt(_attempt())
    with _open(tmp_path, resume=True) as resumed:
        assert resumed.pending_attempt == _attempt()
        assert resumed.next_episode_index == 0


def test_resume_rejects_manifest_or_index_drift(tmp_path: Path) -> None:
    with _open(tmp_path, resume=False):
        pass
    manifest = json.loads((tmp_path / "dataset_bundle_manifest.json").read_text())
    manifest["split_seed"] = 99
    (tmp_path / "dataset_bundle_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(JointRiskBundleStorageError, match="manifest mismatch"):
        _open(tmp_path, resume=True)

