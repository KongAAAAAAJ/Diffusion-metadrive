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


def _append_rejected_without_sidecar(root: Path, index: int) -> None:
    row = {
        "episode_index": index,
        "split": "val",
        "scenario_id": "S6_background_merge_in",
        "local_route": "R1",
        "spawn_seed": index + 10,
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


def test_s5_template_is_strict_balanced_target_release() -> None:
    path = Path(
        "configs/dataset/data_collect_bev_rule_conditioned_v2_s5_release10070.template.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    compose._validate_template(payload)
    targeted = payload["targeted_supplement"]
    assert targeted["accepted_episode_quotas"] == {"train": 0, "val": 0, "test": 0}
    assert targeted["require_target_background_condition"] is True
    assert targeted["bootstrap_spawn_seeds"] == list(range(83017, 83027))


def test_prepare_s5_config_freezes_dynamic_slots_and_refuses_different_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "base"
    base.mkdir()
    template = Path(
        "configs/dataset/data_collect_bev_rule_conditioned_v2_s5_release10070.template.yaml"
    ).resolve()
    output = tmp_path / "generated.yaml"
    contract_path = tmp_path / "contract.json"
    start = 417
    monkeypatch.setattr(compose, "_validate_source_root", lambda path: base)
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
    monkeypatch.setattr(
        compose,
        "load_run_config",
        lambda path: SimpleNamespace(
            targeted_supplement=SimpleNamespace(target_episodes=53)
        ),
    )

    contract = compose.prepare_s5_config(base, template, output, contract_path)
    generated = yaml.safe_load(output.read_text(encoding="utf-8"))
    expected = compose._target_slots(start)
    assert generated["targeted_supplement"]["accepted_episode_quotas"] == {
        split: len(expected[split]) for split in compose.SPLIT_NAMES
    }
    assert [row["episode_index"] for row in contract["append_slots"]] == list(
        range(start, start + 53)
    )
    assert compose.prepare_s5_config(base, template, output, contract_path) == contract

    output.write_text("different\n", encoding="utf-8")
    with pytest.raises(compose.RuleConditionedV2CompositionError, match="different content"):
        compose.prepare_s5_config(base, template, output, contract_path)


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


def test_frozen_contract_rejects_generated_config_hash_drift(tmp_path: Path) -> None:
    generated = tmp_path / "generated.yaml"
    generated.write_text("original\n", encoding="utf-8")
    start = 401
    slots = compose._target_slots(start)
    quotas = {split: len(slots[split]) for split in compose.SPLIT_NAMES}
    base_binding = {
        "base_dataset_fingerprint": "a" * 64,
        "bundle_index_sha256": "b" * 64,
        "next_episode_index": start,
    }
    freeze = {
        "format": compose.PREPARE_CONTRACT_FORMAT,
        "base_source": dict(base_binding),
        "generated_config_path": str(generated.resolve()),
        "generated_config_sha256": compose._file_sha256(generated),
        "accepted_episode_quotas": quotas,
        "append_slots": [
            {"episode_index": index, "split": split}
            for split in compose.SPLIT_NAMES
            for index in slots[split]
        ],
    }
    config = SimpleNamespace(
        targeted_supplement=SimpleNamespace(accepted_episode_quotas=quotas)
    )
    assert compose._validated_frozen_slots(
        freeze, base_binding, generated.resolve(), config
    ) == slots
    generated.write_text("drifted\n", encoding="utf-8")
    with pytest.raises(
        compose.RuleConditionedV2CompositionError, match="base/config drift"
    ):
        compose._validated_frozen_slots(
            freeze, base_binding, generated.resolve(), config
        )


def test_build_staging_physically_copies_and_reindexes_tiny_schema3_bundle(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    supplement = tmp_path / "supplement"
    staging = tmp_path / "staging"
    _source_root(base, split="train", index=0)
    _append_rejected_without_sidecar(base, 1)
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
                "target_split": "test",
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
            "split_episode_counts": {"train": 1, "val": 0, "test": 1},
            "split_joint_samples": {"train": 1, "val": 0, "test": 1},
        },
    }
    source_hash = compose._file_sha256(
        supplement / "platoon_joint_bev/test/episodes/episode_00000000/value.npy"
    )
    compose.build_staging(base, supplement, staging, plan)
    copied = staging / "platoon_joint_bev/test/episodes/episode_00000002/value.npy"
    assert compose._file_sha256(copied) == source_hash
    assert copied.stat().st_ino != (
        supplement / "platoon_joint_bev/test/episodes/episode_00000000/value.npy"
    ).stat().st_ino
    assert _read(copied.parent / "episode.json")["episode_index"] == 2
    assert _read(staging / "platoon_joint_bev/train/manifest.json")[
        "schema_version"
    ] == 3
    state = _read(staging / "platoon_joint_bev/collection_state.json")
    assert state["next_episode_index"] == 3
    assert state["rejected_episodes"] == 1
    assert state["rejection_reasons"] == {"trajectory_infeasible": 1}
    assert len((staging / "bundle_episode_index.jsonl").read_text().splitlines()) == 3
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
