from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from expert_dataset.joint_bev_storage import joint_sample_storage_contract
from expert_dataset.run_joint_bev_collection import load_run_config
from tools import compose_rule_conditioned_v2_dataset as compose


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _episode(
    root: Path,
    component: str,
    split: str,
    index: int,
    *,
    sidecar_fingerprint: str = "old-sidecar",
    base_fingerprint: str = "old-base",
) -> Path:
    path = root / component / split / "episodes" / f"episode_{index:08d}"
    path.mkdir(parents=True)
    if component == "platoon_joint_bev":
        np.save(path / "value.npy", np.arange(4, dtype=np.float32))
        _write_json(
            path / "episode.json",
            {
                "episode_index": index,
                "split": split,
                "joint_samples": 1,
                "attributes": {
                    "scenario_id": "S6_background_merge_in",
                    "spawn_seed": index + 10,
                    "sidecar_dataset_fingerprint": sidecar_fingerprint,
                },
            },
        )
    else:
        np.save(path / "step_index.npy", np.asarray([0], dtype=np.int64))
        np.save(path / "actor_state.npy", np.zeros((1, 3, 8), dtype=np.float32))
        np.save(path / "base_sample_step_index.npy", np.asarray([0], dtype=np.int64))
        _write_json(
            path / "episode.json",
            {
                "episode_index": index,
                "split": split,
                "base_dataset_fingerprint": base_fingerprint,
                "scenario_parameters": {},
                "retention": {"outcome": "success"},
            },
        )
    return path


def _source_root(root: Path, *, split: str, index: int) -> None:
    base_episode = _episode(root, "platoon_joint_bev", split, index)
    sidecar_episode = _episode(root, "riskentry_actor_sidecar", split, index)
    for component in ("platoon_joint_bev", "riskentry_actor_sidecar"):
        (root / component / ".writer.lock").touch()
        for name in compose.SPLIT_NAMES:
            (root / component / name / "episodes").mkdir(parents=True, exist_ok=True)
    (root / ".bundle_writer.lock").touch()
    storage = joint_sample_storage_contract("v2")
    _write_json(
        root / "platoon_joint_bev/dataset_contract.json",
        {
            "format": storage.storage_format,
            "schema_version": storage.schema_version,
            "planner_version": "v2",
            "dataset_fingerprint": "a" * 64,
        },
    )
    _write_json(
        root / "platoon_joint_bev/collection_state.json",
        {
            "schema_version": storage.schema_version,
            "next_episode_index": index + 1,
            "rejection_reasons": {},
        },
    )
    for name in compose.SPLIT_NAMES:
        base_entries = []
        sidecar_entries = []
        if name == split:
            base_entries.append(
                {
                    "episode_index": index,
                    "directory": base_episode.name,
                    "joint_samples": 1,
                    "attributes": _read(base_episode / "episode.json")["attributes"],
                }
            )
            sidecar_entries.append(
                {
                    "episode_index": index,
                    "directory": sidecar_episode.name,
                    "raw_steps": 1,
                    "actor_count": 3,
                    "base_samples": 1,
                    "outcome": "success",
                }
            )
        _write_json(
            root / f"platoon_joint_bev/{name}/manifest.json",
            {"episodes": base_entries},
        )
        _write_json(
            root / f"riskentry_actor_sidecar/{name}/manifest.json",
            {"episodes": sidecar_entries},
        )
    _write_json(root / "dataset_bundle_manifest.json", {"protocol_sha256": "old"})
    row = {
        "episode_index": index,
        "split": split,
        "scenario_id": "S6_background_merge_in",
        "local_route": "R1",
        "spawn_seed": index + 10,
        "base_status": "committed",
        "base_rejection_reason": None,
        "sidecar_status": "committed",
        "sidecar_rejection_reason": None,
        "raw_steps": 1,
        "base_samples": 1,
        "outcome": "success",
    }
    (root / "bundle_episode_index.jsonl").write_text(
        json.dumps(row, sort_keys=True) + "\n", encoding="utf-8"
    )


def _append_rejected_without_sidecar(
    root: Path, index: int, *, spawn_seed: int | None = None
) -> None:
    row = {
        "episode_index": index,
        "split": "val",
        "scenario_id": "S6_background_merge_in",
        "local_route": "R1",
        "spawn_seed": index + 10 if spawn_seed is None else spawn_seed,
        "base_status": "rejected",
        "base_rejection_reason": "trajectory_infeasible",
        "sidecar_status": "rejected",
        "sidecar_rejection_reason": "rollout_failed",
        "raw_steps": 0,
        "base_samples": 0,
        "outcome": "terminated",
    }
    with (root / "bundle_episode_index.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
    state = _read(root / "platoon_joint_bev/collection_state.json")
    state.update(
        {
            "attempted_episodes": index + 1,
            "next_episode_index": index + 1,
            "stored_episodes": 1,
            "rejected_episodes": index,
            "rejection_reasons": {"trajectory_infeasible": index},
        }
    )
    _write_json(root / "platoon_joint_bev/collection_state.json", state)


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_target_slots_are_contiguous_and_match_seed17_split() -> None:
    slots = compose._target_slots(401)
    flattened = sorted(index for values in slots.values() for index in values)
    assert flattened == list(range(401, 401 + compose.APPEND_EPISODES))
    assert sum(len(values) for values in slots.values()) == 53


def test_committed_identity_uniqueness_ignores_rejected_retry() -> None:
    rows = [
        {
            "scenario_id": "S6_background_merge_in",
            "spawn_seed": 31,
            "base_status": "rejected",
        },
        {
            "scenario_id": "S6_background_merge_in",
            "spawn_seed": 31,
            "base_status": "committed",
        },
    ]
    assert compose._duplicate_committed_identities(rows) == []


def test_committed_identity_uniqueness_rejects_two_committed_rows() -> None:
    rows = [
        {
            "scenario_id": "S6_background_merge_in",
            "spawn_seed": 31,
            "base_status": "committed",
        },
        {
            "scenario_id": "S6_background_merge_in",
            "spawn_seed": 31,
            "base_status": "committed",
        },
    ]
    assert compose._duplicate_committed_identities(rows) == [
        ("S6_background_merge_in", 31)
    ]


def test_new_collection_configs_are_v2_candidate_v4_and_loadable() -> None:
    root = Path("configs/dataset")
    for name, total in (
        ("data_collect_bev_rule_conditioned_v2_s6_s9_diagnostic8k.yaml", 8000),
        ("data_collect_bev_rule_conditioned_v2_s6_s9_formal40k.yaml", 40000),
    ):
        payload = yaml.safe_load((root / name).read_text(encoding="utf-8"))
        assert payload["dataset"]["planner_version"] == "v2"
        assert payload["dataset"]["scenario_contract"] == "candidate_v4"
        assert payload["collection"]["target_joint_steps"] == total
        assert payload["collection"]["scenario_max_episode_steps"] == {
            "S6_background_merge_in": 260,
            "S9_narrow_channel_negotiation": 800,
        }
        config = load_run_config(root / name)
        assert tuple(config.formal_scenario_quotas) == compose.S6_S9_SCENARIOS


def test_s5_config_is_independently_runnable_strict_balanced_target_release() -> None:
    path = Path(
        "configs/dataset/data_collect_bev_rule_conditioned_v2_s5_release10070.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = load_run_config(path)
    targeted = payload["targeted_supplement"]
    assert targeted["accepted_episode_quotas"] == {"train": 43, "val": 5, "test": 5}
    assert targeted["require_target_background_condition"] is True
    assert targeted["bootstrap_spawn_seeds"] == list(range(83017, 83027))
    assert config.target_joint_steps == 10_070
    assert config.targeted_supplement is not None
    assert config.targeted_supplement.target_episodes == 53
    assert dict(config.targeted_supplement.accepted_episode_quotas) == {
        "train": 43,
        "val": 5,
        "test": 5,
    }


def test_prepare_composition_contract_freezes_both_sources_and_dynamic_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base"
    supplement = tmp_path / "supplement"
    base.mkdir()
    supplement.mkdir()
    config_path = tmp_path / "s5.yaml"
    config_path.write_text("frozen S5 config\n", encoding="utf-8")
    contract_path = tmp_path / "contract.json"
    start = 417
    monkeypatch.setattr(
        compose, "_validate_source_root", lambda path: Path(path).resolve()
    )
    monkeypatch.setattr(
        compose,
        "_validate_completed_s6_s9",
        lambda root: (
            {"status": "pass", "aligned_samples": 40000, "aligned_episodes": 220},
            {
                "base_dataset_fingerprint": "a" * 64,
                "sidecar_dataset_fingerprint": "b" * 64,
                "bundle_index_sha256": "c" * 64,
                "next_episode_index": start,
            },
        ),
    )
    supplement_binding = {
        "base_dataset_fingerprint": "d" * 64,
        "sidecar_dataset_fingerprint": "e" * 64,
        "bundle_index_sha256": "f" * 64,
        "next_episode_index": 53,
    }
    monkeypatch.setattr(
        compose,
        "_strict_verify",
        lambda root: {
            "status": "pass",
            "aligned_samples": 10_070,
            "aligned_episodes": 53,
        },
    )
    monkeypatch.setattr(compose, "_source_binding", lambda root: supplement_binding)
    config = SimpleNamespace(
        targeted_supplement=SimpleNamespace(
            accepted_episode_quotas=dict(compose.S5_SOURCE_EPISODE_QUOTAS)
        )
    )
    monkeypatch.setattr(
        compose,
        "_validate_s5_config",
        lambda root, path, binding: (
            config,
            {"complete": True, "accepted_episodes": 53, "pending_transaction": False},
        ),
    )

    contract = compose.prepare_composition_contract(
        base, supplement, config_path, contract_path
    )
    expected = compose._target_slots(start)
    assert contract["source_accepted_episode_quotas"] == {"train": 43, "val": 5, "test": 5}
    assert contract["target_split_quotas"] == {
        split: len(expected[split]) for split in compose.SPLIT_NAMES
    }
    assert contract["target_split_quotas"] != contract["source_accepted_episode_quotas"]
    assert [row["episode_index"] for row in contract["append_slots"]] == list(
        range(start, start + 53)
    )
    assert contract["supplement_source"]["config_sha256"] == compose._file_sha256(
        config_path
    )
    assert (
        compose.prepare_composition_contract(
            base, supplement, config_path, contract_path
        )
        == contract
    )

    contract_path.write_text("different\n", encoding="utf-8")
    with pytest.raises(compose.RuleConditionedV2CompositionError, match="different content"):
        compose.prepare_composition_contract(
            base, supplement, config_path, contract_path
        )


def test_strict_s5_evidence_rejects_non_target_or_wrong_profile() -> None:
    attributes = {
        "scenario_id": compose.S5_SCENARIO,
        "rule_maker_profile_id": "balanced",
        "scenario_contract_id": "candidate_v4",
        "targeted_supplement_evidence": {
            "behavior_category": compose.RELEASE_CATEGORY,
            "target_background_condition_sampled": True,
            "target_background_condition_realized": True,
            "platoon_safety_events": [],
            "lateral_mode_runs_by_role": [2, 2, 2],
            "lateral_run_directions_by_role": [
                ["left", "right"],
                ["right", "left"],
                ["left", "right"],
            ],
            "lateral_range_m_by_role": [3.0, 3.1, 3.2],
            "return_error_m_by_role": [0.1, 0.2, 0.3],
        },
    }
    compose._validate_s5_attributes(attributes)
    attributes["targeted_supplement_evidence"][
        "target_background_condition_realized"
    ] = False
    with pytest.raises(compose.RuleConditionedV2CompositionError, match="strict balanced"):
        compose._validate_s5_attributes(attributes)


def test_frozen_contract_rejects_s5_config_or_source_quota_drift(tmp_path: Path) -> None:
    s5_config = tmp_path / "s5.yaml"
    s5_config.write_text("original\n", encoding="utf-8")
    start = 401
    slots = compose._target_slots(start)
    quotas = {split: len(slots[split]) for split in compose.SPLIT_NAMES}
    base_binding = {
        "base_dataset_fingerprint": "a" * 64,
        "bundle_index_sha256": "b" * 64,
        "next_episode_index": start,
    }
    supplement_binding = {
        "base_dataset_fingerprint": "c" * 64,
        "bundle_index_sha256": "d" * 64,
    }
    freeze = {
        "format": compose.PREPARE_CONTRACT_FORMAT,
        "base_source": dict(base_binding),
        "supplement_source": {
            **supplement_binding,
            "config_path": str(s5_config.resolve()),
            "config_sha256": compose._file_sha256(s5_config),
            "accepted_episode_quotas": dict(compose.S5_SOURCE_EPISODE_QUOTAS),
        },
        "source_accepted_episode_quotas": dict(compose.S5_SOURCE_EPISODE_QUOTAS),
        "target_split_quotas": quotas,
        "append_slots": [
            {"episode_index": index, "split": split}
            for split in compose.SPLIT_NAMES
            for index in slots[split]
        ],
    }
    config = SimpleNamespace(
        targeted_supplement=SimpleNamespace(
            accepted_episode_quotas=dict(compose.S5_SOURCE_EPISODE_QUOTAS)
        )
    )
    assert compose._validated_frozen_slots(
        freeze, base_binding, supplement_binding, s5_config.resolve(), config
    ) == slots
    s5_config.write_text("drifted\n", encoding="utf-8")
    with pytest.raises(
        compose.RuleConditionedV2CompositionError, match="base/S5 config drift"
    ):
        compose._validated_frozen_slots(
            freeze, base_binding, supplement_binding, s5_config.resolve(), config
        )

    freeze["supplement_source"]["config_sha256"] = compose._file_sha256(s5_config)
    config.targeted_supplement.accepted_episode_quotas = {"train": 44, "val": 4, "test": 5}
    with pytest.raises(
        compose.RuleConditionedV2CompositionError,
        match="slots/source quotas drifted",
    ):
        compose._validated_frozen_slots(
            freeze, base_binding, supplement_binding, s5_config.resolve(), config
        )


def test_build_staging_physically_copies_and_reindexes_tiny_schema3_bundle(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    supplement = tmp_path / "supplement"
    staging = tmp_path / "staging"
    _source_root(base, split="train", index=0)
    _append_rejected_without_sidecar(base, 1, spawn_seed=10)
    _source_root(supplement, split="test", index=0)
    fingerprint = "d" * 64
    storage = joint_sample_storage_contract("v2")
    sidecar = compose.sidecar_dataset_fingerprint(
        fingerprint,
        base_format=storage.storage_format,
        base_schema_version=storage.schema_version,
    )
    plan = {
        "base_dataset_fingerprint": fingerprint,
        "sidecar_dataset_fingerprint": sidecar,
        "contract_payload": {"tiny": True},
        "supplement_episode_mapping": [
            {
                "source_episode_index": 0,
                "source_split": "test",
                "target_episode_index": 2,
                "target_split": "val",
            }
        ],
        "expected_output": {
            "base_episodes": 2,
            "base_joint_samples": 2,
            "sidecar_episodes": 2,
            "sidecar_only_episodes": 0,
            "sidecar_raw_steps": 2,
            "bundle_index_rows": 3,
            "rejected_episodes": 1,
            "split_episode_counts": {"train": 1, "val": 1, "test": 0},
            "split_joint_samples": {"train": 1, "val": 1, "test": 0},
        },
    }
    source_hash = compose._file_sha256(
        supplement / "platoon_joint_bev/test/episodes/episode_00000000/value.npy"
    )
    compose.build_staging(base, supplement, staging, plan)
    copied = staging / "platoon_joint_bev/val/episodes/episode_00000002/value.npy"
    assert compose._file_sha256(copied) == source_hash
    assert copied.stat().st_ino != (
        supplement / "platoon_joint_bev/test/episodes/episode_00000000/value.npy"
    ).stat().st_ino
    copied_metadata = _read(copied.parent / "episode.json")
    assert copied_metadata["episode_index"] == 2
    assert copied_metadata["split"] == "val"
    provenance = copied_metadata["attributes"]["curation_source"]
    assert provenance["source_split"] == "test"
    assert provenance["target_split"] == "val"
    assert _read(staging / "platoon_joint_bev/train/manifest.json")[
        "schema_version"
    ] == 3
    state = _read(staging / "platoon_joint_bev/collection_state.json")
    assert state["next_episode_index"] == 3
    assert state["rejected_episodes"] == 1
    assert state["rejection_reasons"] == {"trajectory_infeasible": 1}
    staged_rows = [
        json.loads(line)
        for line in (staging / "bundle_episode_index.jsonl").read_text().splitlines()
    ]
    assert len(staged_rows) == 3
    preserved_base_rows = [row for row in staged_rows if row["episode_index"] < 2]
    assert sum(
        row["scenario_id"] == "S6_background_merge_in"
        and row["spawn_seed"] == 10
        for row in preserved_base_rows
    ) == 2
    assert compose._duplicate_committed_identities(preserved_base_rows) == []
    assert compose._file_sha256(
        supplement / "platoon_joint_bev/test/episodes/episode_00000000/value.npy"
    ) == source_hash


def test_stage_refuses_insufficient_space_before_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base"
    supplement = tmp_path / "supplement"
    base.mkdir()
    supplement.mkdir()
    final = tmp_path / "final"
    monkeypatch.setattr(compose, "_required_staging_bytes", lambda *args: 100)
    monkeypatch.setattr(
        compose.shutil, "disk_usage", lambda path: SimpleNamespace(free=99)
    )
    with pytest.raises(compose.RuleConditionedV2CompositionError, match="insufficient"):
        compose.stage_composition(base, supplement, final, {"supplement_episode_mapping": []})


def test_stage_refuses_existing_staging_root(tmp_path: Path) -> None:
    base = tmp_path / "base"
    supplement = tmp_path / "supplement"
    final = tmp_path / "final"
    base.mkdir()
    supplement.mkdir()
    compose._staging_path(final).mkdir()

    with pytest.raises(
        compose.RuleConditionedV2CompositionError,
        match="already exists",
    ):
        compose.stage_composition(
            base, supplement, final, {"supplement_episode_mapping": []}
        )


def test_inspect_sources_rejects_duplicate_s5_scenario_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base"
    supplement = tmp_path / "supplement"
    base.mkdir()
    supplement.mkdir()
    config_path = tmp_path / "s5.yaml"
    config_path.write_text("frozen\n", encoding="utf-8")
    freeze_path = tmp_path / "freeze.json"
    _write_json(freeze_path, {})
    base_binding = {
        "base_dataset_fingerprint": "a" * 64,
        "bundle_index_sha256": "b" * 64,
        "next_episode_index": 100,
    }
    supplement_binding = {
        "base_dataset_fingerprint": "c" * 64,
        "bundle_index_sha256": "d" * 64,
    }
    config = SimpleNamespace(
        targeted_supplement=SimpleNamespace(
            accepted_episode_quotas=dict(compose.S5_SOURCE_EPISODE_QUOTAS)
        ),
        immutable_fingerprint=lambda: supplement_binding[
            "base_dataset_fingerprint"
        ],
    )
    source_entries: dict[int, tuple[str, dict[str, object]]] = {}
    source_index = 0
    for split, count in compose.S5_SOURCE_EPISODE_QUOTAS.items():
        for _ in range(count):
            spawn_seed = 83017 if source_index < 2 else 83017 + source_index
            source_entries[source_index] = (
                split,
                {
                    "joint_samples": compose.SAMPLES_PER_S5_EPISODE,
                    "attributes": {"spawn_seed": spawn_seed},
                },
            )
            source_index += 1

    monkeypatch.setattr(
        compose, "_validate_source_root", lambda path: Path(path).resolve()
    )
    monkeypatch.setattr(
        compose,
        "_validate_completed_s6_s9",
        lambda root: ({"status": "pass"}, dict(base_binding)),
    )
    monkeypatch.setattr(compose, "_strict_verify", lambda root: {"status": "pass"})
    monkeypatch.setattr(compose, "_source_binding", lambda root: supplement_binding)
    monkeypatch.setattr(
        compose,
        "_validate_s5_config",
        lambda root, path, binding: (config, {"complete": True}),
    )
    monkeypatch.setattr(
        compose,
        "_validated_frozen_slots",
        lambda *args: compose._target_slots(100),
    )
    monkeypatch.setattr(compose, "_manifest_entries", lambda *args: source_entries)
    monkeypatch.setattr(compose, "_validate_s5_attributes", lambda attributes: None)
    monkeypatch.setattr(compose, "_episode_hashes", lambda path: {})
    monkeypatch.setattr(compose, "_read_rows", lambda root: [])

    with pytest.raises(
        compose.RuleConditionedV2CompositionError,
        match=r"duplicate \(scenario_id, spawn_seed\)",
    ):
        compose.inspect_sources(
            base,
            supplement,
            config_path,
            freeze_path,
            expected_base_fingerprint="a" * 64,
            expected_base_index_sha256="b" * 64,
            expected_supplement_fingerprint="c" * 64,
            expected_supplement_index_sha256="d" * 64,
        )


def test_install_is_atomic_to_absent_new_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base"
    supplement = tmp_path / "supplement"
    final = tmp_path / "final"
    base.mkdir()
    supplement.mkdir()
    staging = compose._staging_path(final)
    staging.mkdir()
    (staging / "marker").write_text("new", encoding="utf-8")
    contract = {"frozen": True}
    fingerprint = compose._payload_sha256(contract)
    _write_json(staging / "dataset_curation_manifest.json", {"contract_payload": contract})
    plan = {"contract_payload": contract, "base_dataset_fingerprint": fingerprint}
    monkeypatch.setattr(compose, "_exclusive_source_locks", lambda roots: nullcontext())
    monkeypatch.setattr(compose, "_recheck_bindings", lambda *args: None)

    def fake_verify(root: Path) -> dict[str, object]:
        return {"bundle_root": str(Path(root).resolve()), "verified": True}

    monkeypatch.setattr(compose, "verify_composed_bundle", fake_verify)
    result = compose.install_staged_composition(base, supplement, final, plan)
    assert result == {"bundle_root": str(final.resolve()), "verified": True}
    assert (final / "marker").read_text(encoding="utf-8") == "new"
    assert not staging.exists()

    staging.mkdir()
    with pytest.raises(compose.RuleConditionedV2CompositionError, match="absent final"):
        compose.install_staged_composition(base, supplement, final, plan)


def test_source_pending_and_symlink_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / ".bundle_episode_pending.json").write_text("{}", encoding="utf-8")
    with pytest.raises(compose.RuleConditionedV2CompositionError, match="pending"):
        compose._validate_source_root(source)

    clean = tmp_path / "clean"
    clean.mkdir()
    link = tmp_path / "link"
    link.symlink_to(clean, target_is_directory=True)
    with pytest.raises(compose.RuleConditionedV2CompositionError, match="real directory"):
        compose._validate_source_root(link)
