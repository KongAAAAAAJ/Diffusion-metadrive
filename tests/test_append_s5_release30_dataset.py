from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

from tools import append_s5_release30_dataset as append


def test_release30_target_slots_are_exact_contiguous_24_3_3() -> None:
    slots = append._target_slots()
    assert {name: len(values) for name, values in slots.items()} == {
        "train": 24,
        "val": 3,
        "test": 3,
    }
    assert sorted(index for values in slots.values() for index in values) == list(
        range(401, 431)
    )


def test_release30_swap_paths_are_fixed_siblings(tmp_path: Path) -> None:
    destination = tmp_path / "dataset"
    destination.mkdir()
    staging, backup, failed = append._validate_swap_paths(destination)
    assert staging == tmp_path / "dataset.s5_release30_append_staging"
    assert backup == tmp_path / "dataset.pre_s5_release30_append_b93dfa5c"
    assert failed == tmp_path / "dataset.failed_s5_release30_append"

    staging.symlink_to(destination, target_is_directory=True)
    with pytest.raises(append.S5AppendError, match="symbolic"):
        append._validate_swap_paths(destination)


def test_release_evidence_allows_descriptive_background_distribution() -> None:
    attributes = {
        "targeted_supplement_evidence": {
            "behavior_category": append.RELEASE_CATEGORY,
            "platoon_safety_events": [],
            "incidental_background_actor_count": 3,
            "target_background_condition_realized": False,
            "lateral_mode_runs_by_role": [2, 2, 2],
            "lateral_run_directions_by_role": [
                ["left", "right"],
                ["right", "left"],
                ["left", "right"],
            ],
            "lateral_range_m_by_role": [3.0, 3.1, 3.2],
            "return_error_m_by_role": [0.1, 0.2, 0.3],
        }
    }
    append._validate_release_evidence(attributes)
    attributes["targeted_supplement_evidence"][
        "incidental_background_actor_count"
    ] = 2
    with pytest.raises(append.S5AppendError, match="outside 3-6"):
        append._validate_release_evidence(attributes)


def test_copy_episode_is_physical_and_preserves_prior_provenance(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source_episode = (
        source / "platoon_joint_bev/train/episodes/episode_00000007"
    )
    source_sidecar = (
        source / "riskentry_actor_sidecar/train/episodes/episode_00000007"
    )
    source_episode.mkdir(parents=True)
    source_sidecar.mkdir(parents=True)
    target.mkdir()
    np.save(source_episode / "value.npy", np.arange(4, dtype=np.float32))
    np.save(source_sidecar / "step_index.npy", np.arange(4, dtype=np.int64))
    (source_episode / "episode.json").write_text(
        json.dumps(
            {
                "episode_index": 7,
                "split": "train",
                "joint_samples": 4,
                "attributes": {
                    "curation_source": {"source_id": "older"},
                    "sidecar_dataset_fingerprint": "old",
                },
            }
        ),
        encoding="utf-8",
    )
    (source_sidecar / "episode.json").write_text(
        json.dumps(
            {
                "episode_index": 7,
                "split": "train",
                "base_dataset_fingerprint": "old",
                "scenario_parameters": {
                    "curation_source": {"source_id": "older"}
                },
            }
        ),
        encoding="utf-8",
    )
    inventory = []
    base_metadata = append._copy_episode(
        source_root=source,
        source_id="immediate",
        component="platoon_joint_bev",
        source_split="train",
        source_index=7,
        target_root=target,
        target_split="test",
        target_index=401,
        base_fingerprint="b" * 64,
        sidecar_fingerprint="s" * 64,
        inventory=inventory,
    )
    sidecar_metadata = append._copy_episode(
        source_root=source,
        source_id="immediate",
        component="riskentry_actor_sidecar",
        source_split="train",
        source_index=7,
        target_root=target,
        target_split="test",
        target_index=401,
        base_fingerprint="b" * 64,
        sidecar_fingerprint="s" * 64,
        inventory=inventory,
    )

    assert base_metadata["episode_index"] == 401
    assert base_metadata["split"] == "test"
    assert base_metadata["attributes"]["sidecar_dataset_fingerprint"] == "s" * 64
    assert base_metadata["attributes"]["curation_source"][
        "prior_curation_source"
    ] == {"source_id": "older"}
    assert sidecar_metadata["base_dataset_fingerprint"] == "b" * 64
    assert sidecar_metadata["scenario_parameters"]["curation_source"][
        "prior_curation_source"
    ] == {"source_id": "older"}
    copied = target / "platoon_joint_bev/test/episodes/episode_00000401/value.npy"
    assert np.array_equal(np.load(copied), np.arange(4, dtype=np.float32))
    assert copied.stat().st_ino != (source_episode / "value.npy").stat().st_ino
    assert len(inventory) == 2


def test_install_staged_append_swaps_only_after_staging_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "dataset"
    supplement = tmp_path / "supplement"
    destination.mkdir()
    supplement.mkdir()
    staging, backup, failed = append._swap_paths(destination)
    staging.mkdir()
    (destination / "marker").write_text("old", encoding="utf-8")
    (staging / "marker").write_text("new", encoding="utf-8")
    plan = {
        "contract_payload": {},
        "base_dataset_fingerprint": "base",
        "sidecar_dataset_fingerprint": "sidecar",
        "curation_contract_sha256": "curation",
    }
    verified_roots = []

    monkeypatch.setattr(append, "_exclusive_source_locks", lambda roots: nullcontext())
    monkeypatch.setattr(
        append,
        "_recheck_plan_bindings",
        lambda destination_root, supplement_root, frozen_plan: None,
    )

    def fake_verify(root: Path) -> dict[str, object]:
        verified_roots.append(Path(root))
        return {
            "bundle_root": str(Path(root).resolve()),
            "base_dataset_fingerprint": "base",
            "sidecar_dataset_fingerprint": "sidecar",
            "curation_contract_sha256": "curation",
            "verified": True,
        }

    monkeypatch.setattr(append, "verify_appended_bundle", fake_verify)
    result = append.install_staged_append(destination, supplement, plan)

    assert verified_roots == [staging, destination]
    assert (destination / "marker").read_text(encoding="utf-8") == "new"
    assert (backup / "marker").read_text(encoding="utf-8") == "old"
    assert not staging.exists()
    assert not failed.exists()
    assert result["backup_retained"] is True
    assert result["rollback_performed"] is False


def test_staging_must_match_the_bound_dry_run_plan() -> None:
    report = {
        "base_dataset_fingerprint": "unexpected",
        "sidecar_dataset_fingerprint": "sidecar",
        "curation_contract_sha256": "curation",
    }
    plan = {
        "base_dataset_fingerprint": "base",
        "sidecar_dataset_fingerprint": "sidecar",
        "curation_contract_sha256": "curation",
    }
    with pytest.raises(append.S5AppendError, match="does not match dry-run plan"):
        append._require_staging_matches_plan(report, plan)
