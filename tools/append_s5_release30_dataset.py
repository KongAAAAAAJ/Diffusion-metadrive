"""Build and atomically install the audited S5 release30 append bundle."""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from expert_dataset.finalize_s5_targeted_supplement import (
    audit_targeted_supplement,
)
from expert_dataset.joint_bev_storage import (
    EpisodeSplitAssigner,
    EpisodeSplitConfig,
    SPLIT_NAMES,
    STORAGE_FORMAT,
    STORAGE_SCHEMA_VERSION,
)
from expert_dataset.joint_risk_bundle_contract import (
    BUNDLE_FORMAT,
    BUNDLE_SCHEMA_VERSION,
    bundle_protocol_sha256,
)
from expert_dataset.riskentry_sidecar_storage import (
    sidecar_dataset_contract,
    sidecar_dataset_fingerprint,
)
from expert_dataset.run_joint_bev_collection import load_run_config
from expert_dataset.verify_joint_bev_dataset import verify_joint_bev_dataset
from expert_dataset.verify_joint_risk_bundle import verify_joint_risk_bundle
from expert_dataset.verify_riskentry_sidecar import (
    verify_riskentry_sidecar_dataset,
)
from scenarios.bev_round13_contract import (
    CANDIDATE_V3_CONTRACT_ID,
    CANDIDATE_V4_CONTRACT_ID,
    scenario_contract_for_id,
)
from tools.rebalance_s5_dataset import (
    S5RebalanceError,
    _atomic_json,
    _canonical_json,
    _copy_payload_file,
    _curation_source,
    _exclusive_source_locks,
    _file_sha256,
    _manifest_entries,
    _payload_sha256,
    _read_json,
    _read_rows,
    _validate_no_links,
    _write_manifests,
    verify_curated_bundle,
)


DESTINATION_ROOT = Path(
    "/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/"
    "bev_joint_risk_candidate_v3_formal50k_v1"
)
SUPPLEMENT_ROOT = Path(
    "/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/"
    "bev_joint_risk_candidate_v4_s5_release30_v1"
)
SUPPLEMENT_CONFIG = Path(
    "configs/dataset/data_collect_candidate_v4_s5_release30.yaml"
)
EXPECTED_DESTINATION_FINGERPRINT = (
    "b93dfa5c6e2224aac7ae3518b45216c9f04b4a18b78f0a88aafb0b22657a3b4a"
)
EXPECTED_DESTINATION_INDEX_SHA256 = (
    "b56de9a388d7d2e694a0eebfe722bbe0979f1dd9f74c3052c09bf256822b3778"
)
S5_SCENARIO = "S5_hard_brake_lead"
BRAKE_CATEGORY = "keep_emergency_braking"
RELEASE_CATEGORY = "temporary_formation_release_and_recovery"
APPEND_FORMAT = "s5-release30-appended-curated-bundle-v1"
INVENTORY_FORMAT = "s5-release30-payload-inventory-v1"
SOURCE_ID_DESTINATION = "curated44560_pre_release30_append"
SOURCE_ID_SUPPLEMENT = "candidate_v4_s5_release30"
EXPECTED_DESTINATION_COUNTS = {
    "base_episodes": 233,
    "base_joint_samples": 44560,
    "sidecar_episodes": 333,
    "sidecar_only_episodes": 100,
    "bundle_index_rows": 401,
    "split_episode_counts": {"train": 184, "val": 28, "test": 21},
    "split_joint_samples": {"train": 35220, "val": 5310, "test": 4030},
    "s5_behavior_episode_counts": {
        BRAKE_CATEGORY: 5,
        RELEASE_CATEGORY: 19,
    },
}
EXPECTED_FINAL_COUNTS = {
    "base_episodes": 263,
    "base_joint_samples": 50260,
    "sidecar_episodes": 363,
    "sidecar_only_episodes": 100,
    "bundle_index_rows": 431,
    "split_episode_counts": {"train": 208, "val": 31, "test": 24},
    "split_joint_samples": {"train": 39780, "val": 5880, "test": 4600},
    "s5_behavior_episode_counts": {
        BRAKE_CATEGORY: 5,
        RELEASE_CATEGORY: 49,
    },
}


class S5AppendError(S5RebalanceError):
    """Raised when a release30 source or append safety gate fails."""


def _episode_path(root: Path, component: str, split: str, index: int) -> Path:
    return root / component / split / "episodes" / f"episode_{index:08d}"


def _behavior_category(attributes: Mapping[str, object]) -> str:
    formal = attributes.get("formal_coverage")
    targeted = attributes.get("targeted_supplement_evidence")
    evidence = formal if isinstance(formal, Mapping) else targeted
    if not isinstance(evidence, Mapping):
        return "unknown"
    return str(evidence.get("behavior_category", "unknown"))


def _validate_release_evidence(attributes: Mapping[str, object]) -> None:
    evidence = attributes.get("targeted_supplement_evidence")
    if not isinstance(evidence, Mapping):
        raise S5AppendError("supplement episode lacks targeted evidence")
    if evidence.get("behavior_category") != RELEASE_CATEGORY:
        raise S5AppendError("supplement episode behavior category mismatch")
    if evidence.get("platoon_safety_events") != []:
        raise S5AppendError("supplement episode has a platoon safety event")
    background_count = evidence.get("incidental_background_actor_count")
    if background_count not in (3, 4, 5, 6):
        raise S5AppendError("supplement background count is outside 3-6")
    runs = evidence.get("lateral_mode_runs_by_role")
    directions = evidence.get("lateral_run_directions_by_role")
    ranges = evidence.get("lateral_range_m_by_role")
    returns = evidence.get("return_error_m_by_role")
    if runs != [2, 2, 2] or not isinstance(directions, list) or len(directions) != 3:
        raise S5AppendError("supplement lateral behavior run contract failed")
    for row in directions:
        if (
            not isinstance(row, list)
            or len(row) != 2
            or row[0] == row[1]
            or "mixed" in row
        ):
            raise S5AppendError("supplement lateral directions are not opposite")
    if len({row[0] for row in directions}) < 2:
        raise S5AppendError("supplement lacks mixed-direction platoon behavior")
    if (
        not isinstance(ranges, list)
        or len(ranges) != 3
        or any(float(value) < 2.5 for value in ranges)
        or not isinstance(returns, list)
        or len(returns) != 3
        or any(float(value) > 0.5 for value in returns)
    ):
        raise S5AppendError("supplement physical displacement contract failed")


def _episode_hashes(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_dir():
        raise S5AppendError(f"episode is not a real directory: {path}")
    hashes = {}
    for item in sorted(path.iterdir()):
        if item.is_symlink() or not item.is_file():
            raise S5AppendError(f"invalid episode file: {item}")
        hashes[item.name] = _file_sha256(item)
    return hashes


def _target_slots(start: int = 401, count: int = 30) -> dict[str, list[int]]:
    assigner = EpisodeSplitAssigner(EpisodeSplitConfig(seed=17))
    slots = {split: [] for split in SPLIT_NAMES}
    for index in range(start, start + count):
        slots[assigner.split_for_episode(index)].append(index)
    observed = {split: len(values) for split, values in slots.items()}
    if observed != {"train": 24, "val": 3, "test": 3}:
        raise S5AppendError(f"release30 target slots drifted: {observed}")
    return slots


def _source_statistics(root: Path) -> dict[str, object]:
    base = _manifest_entries(root, "platoon_joint_bev")
    sidecar = _manifest_entries(root, "riskentry_actor_sidecar")
    split_episodes: Counter[str] = Counter()
    split_samples: Counter[str] = Counter()
    behaviors: Counter[str] = Counter()
    for _, (split, entry) in base.items():
        attributes = entry.get("attributes")
        if not isinstance(attributes, Mapping):
            raise S5AppendError("base manifest attributes are missing")
        split_episodes[split] += 1
        split_samples[split] += int(entry["joint_samples"])
        if attributes.get("scenario_id") == S5_SCENARIO:
            behaviors[_behavior_category(attributes)] += 1
    return {
        "base_episodes": len(base),
        "base_joint_samples": sum(split_samples.values()),
        "sidecar_episodes": len(sidecar),
        "sidecar_only_episodes": len(set(sidecar) - set(base)),
        "bundle_index_rows": len(_read_rows(root)),
        "split_episode_counts": {
            split: int(split_episodes[split]) for split in SPLIT_NAMES
        },
        "split_joint_samples": {
            split: int(split_samples[split]) for split in SPLIT_NAMES
        },
        "s5_behavior_episode_counts": {
            BRAKE_CATEGORY: int(behaviors[BRAKE_CATEGORY]),
            RELEASE_CATEGORY: int(behaviors[RELEASE_CATEGORY]),
        },
    }


def inspect_sources(
    destination_root: Path,
    supplement_root: Path,
    supplement_config: Path,
    *,
    expected_destination_fingerprint: str,
    expected_destination_index_sha256: str,
    expected_supplement_fingerprint: str,
    expected_supplement_index_sha256: str,
    verify_destination_payload_hashes: bool = True,
) -> dict[str, object]:
    destination_root = destination_root.resolve()
    supplement_root = supplement_root.resolve()
    if destination_root == supplement_root:
        raise S5AppendError("destination and supplement roots must differ")
    _validate_no_links(destination_root)
    _validate_no_links(supplement_root)
    for root in (destination_root, supplement_root):
        for pending in (".bundle_episode_pending.json", ".targeted_batch_pending.json"):
            if (root / pending).exists():
                raise S5AppendError(f"source has pending state: {root / pending}")

    destination_report = verify_curated_bundle(
        destination_root,
        verify_payload_hashes=verify_destination_payload_hashes,
    )
    if destination_report["base_dataset_fingerprint"] != expected_destination_fingerprint:
        raise S5AppendError("destination base fingerprint drift")
    destination_index_sha = _file_sha256(
        destination_root / "bundle_episode_index.jsonl"
    )
    if destination_index_sha != expected_destination_index_sha256:
        raise S5AppendError("destination bundle index hash drift")
    if _source_statistics(destination_root) != EXPECTED_DESTINATION_COUNTS:
        raise S5AppendError("destination counts drifted from the verified 44560 baseline")

    config = load_run_config(supplement_config)
    if config.bundle_root.resolve() != supplement_root:
        raise S5AppendError("supplement config points to a different bundle root")
    if config.scenario_contract_id != CANDIDATE_V4_CONTRACT_ID:
        raise S5AppendError("supplement config is not candidate-v4")
    if config.immutable_fingerprint() != expected_supplement_fingerprint:
        raise S5AppendError("supplement fingerprint does not match the frozen config")
    supplement_contract = _read_json(
        supplement_root / "platoon_joint_bev/dataset_contract.json"
    )
    if supplement_contract.get("dataset_fingerprint") != expected_supplement_fingerprint:
        raise S5AppendError("supplement base fingerprint drift")
    supplement_index_sha = _file_sha256(
        supplement_root / "bundle_episode_index.jsonl"
    )
    if supplement_index_sha != expected_supplement_index_sha256:
        raise S5AppendError("supplement bundle index hash drift")
    verify_joint_risk_bundle(
        supplement_root, scenario_contract_id=CANDIDATE_V4_CONTRACT_ID
    )
    audit = audit_targeted_supplement(config)
    if (
        audit.get("complete") is not True
        or int(audit.get("accepted_episodes", -1)) != 30
        or audit.get("split_counts") != {"train": 24, "val": 3, "test": 3}
        or audit.get("pending_transaction") is not False
    ):
        raise S5AppendError("supplement audit is not complete release30 evidence")

    destination_base = _manifest_entries(destination_root, "platoon_joint_bev")
    supplement_base = _manifest_entries(supplement_root, "platoon_joint_bev")
    supplement_sidecar = _manifest_entries(
        supplement_root, "riskentry_actor_sidecar"
    )
    if len(supplement_base) != 30 or not set(supplement_base).issubset(supplement_sidecar):
        raise S5AppendError("supplement must contain exactly 30 joined base episodes")
    slots = _target_slots()
    sources_by_split = {split: [] for split in SPLIT_NAMES}
    for index, (split, entry) in sorted(supplement_base.items()):
        attributes = entry.get("attributes")
        if not isinstance(attributes, Mapping):
            raise S5AppendError("supplement base attributes are missing")
        if (
            attributes.get("scenario_id") != S5_SCENARIO
            or attributes.get("rule_maker_profile_id") != "balanced"
            or attributes.get("scenario_contract_id") != CANDIDATE_V4_CONTRACT_ID
            or int(entry["joint_samples"]) != 190
        ):
            raise S5AppendError("supplement episode violates the release30 contract")
        _validate_release_evidence(attributes)
        sources_by_split[split].append(index)

    mapping = []
    for split in SPLIT_NAMES:
        source_ids = sorted(sources_by_split[split])
        if len(source_ids) != len(slots[split]):
            raise S5AppendError(f"supplement split mapping mismatch for {split}")
        for source_index, target_index in zip(source_ids, slots[split]):
            source_base = _episode_path(
                supplement_root, "platoon_joint_bev", split, source_index
            )
            source_sidecar = _episode_path(
                supplement_root, "riskentry_actor_sidecar", split, source_index
            )
            attributes = supplement_base[source_index][1]["attributes"]
            mapping.append(
                {
                    "source_episode_index": source_index,
                    "source_split": split,
                    "target_episode_index": target_index,
                    "target_split": split,
                    "spawn_seed": int(attributes["spawn_seed"]),
                    "base_file_sha256": _episode_hashes(source_base),
                    "sidecar_file_sha256": _episode_hashes(source_sidecar),
                }
            )
    mapping.sort(key=lambda row: int(row["target_episode_index"]))

    scenario_seeds = [
        (
            str(entry["attributes"]["scenario_id"]),
            int(entry["attributes"]["spawn_seed"]),
        )
        for _, entry in destination_base.values()
    ] + [(S5_SCENARIO, int(row["spawn_seed"])) for row in mapping]
    duplicates = sorted(
        key for key, count in Counter(scenario_seeds).items() if count > 1
    )
    if duplicates:
        raise S5AppendError(
            f"duplicate (scenario_id, spawn_seed) pairs: {duplicates}"
        )

    source_contracts = {
        contract_id: scenario_contract_for_id(contract_id)["sha256"]
        for contract_id in (CANDIDATE_V3_CONTRACT_ID, CANDIDATE_V4_CONTRACT_ID)
    }
    scenario_payload = {
        "format": "curated-mixed-scenario-contract-v1",
        "source_contracts": source_contracts,
        "policy": "preserve_per_episode_source_contract_without_relabeling",
    }
    selected_raw_steps = sum(
        int(
            np.load(
                _episode_path(
                    supplement_root,
                    "riskentry_actor_sidecar",
                    str(row["source_split"]),
                    int(row["source_episode_index"]),
                )
                / "step_index.npy",
                mmap_mode="r",
                allow_pickle=False,
            ).shape[0]
        )
        for row in mapping
    )
    contract_payload = {
        "format": APPEND_FORMAT,
        "destination_source": {
            "base_dataset_fingerprint": expected_destination_fingerprint,
            "sidecar_dataset_fingerprint": destination_report[
                "sidecar_dataset_fingerprint"
            ],
            "bundle_index_sha256": destination_index_sha,
            "dataset_curation_manifest_sha256": _file_sha256(
                destination_root / "dataset_curation_manifest.json"
            ),
        },
        "supplement_source": {
            "base_dataset_fingerprint": expected_supplement_fingerprint,
            "bundle_index_sha256": supplement_index_sha,
            "scenario_contract_sha256": scenario_contract_for_id(
                CANDIDATE_V4_CONTRACT_ID
            )["sha256"],
        },
        "selection_policy": "all_30_base_committed_joined_release_episodes",
        "removal_policy": "preserve_every_existing_destination_episode",
        "supplement_episode_mapping": mapping,
        "split_assignment": {
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "test_ratio": 0.1,
            "seed": 17,
        },
        "scenario_contract": scenario_payload,
        "expected_output": {
            **EXPECTED_FINAL_COUNTS,
            "sidecar_raw_steps": int(destination_report["sidecar_raw_steps"])
            + selected_raw_steps,
            "new_release30_episodes": 30,
        },
    }
    base_fingerprint = _payload_sha256(contract_payload)
    return {
        "format": APPEND_FORMAT,
        "destination_root": str(destination_root),
        "supplement_root": str(supplement_root),
        "contract_payload": contract_payload,
        "base_dataset_fingerprint": base_fingerprint,
        "sidecar_dataset_fingerprint": sidecar_dataset_fingerprint(
            base_fingerprint
        ),
        "curation_contract_sha256": _payload_sha256(scenario_payload),
        "supplement_episode_mapping": mapping,
        "removed_episode_indices": [],
        "formal_training_eligibility": False,
    }


def _copy_episode(
    *,
    source_root: Path,
    source_id: str,
    component: str,
    source_split: str,
    source_index: int,
    target_root: Path,
    target_split: str,
    target_index: int,
    base_fingerprint: str,
    sidecar_fingerprint: str,
    inventory: list[dict[str, object]],
) -> dict[str, object]:
    source = _episode_path(source_root, component, source_split, source_index)
    target = _episode_path(target_root, component, target_split, target_index)
    if target.exists():
        raise S5AppendError(f"duplicate target episode path: {target}")
    target.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        if path.name == "episode.json":
            continue
        if path.suffix != ".npy":
            raise S5AppendError(f"unexpected episode payload: {path}")
        _copy_payload_file(
            path,
            target / path.name,
            inventory=inventory,
            source_id=source_id,
            source_relative=str(path.relative_to(source_root)),
            target_relative=str((target / path.name).relative_to(target_root)),
        )
    metadata = _read_json(source / "episode.json")
    provenance = _curation_source(
        source_id, source_root, source_split, source_index
    )
    metadata["episode_index"] = target_index
    metadata["split"] = target_split
    if component == "platoon_joint_bev":
        attributes = metadata.get("attributes")
        if not isinstance(attributes, dict):
            raise S5AppendError("base episode attributes are missing")
        previous = attributes.get("curation_source")
        if isinstance(previous, Mapping):
            provenance["prior_curation_source"] = dict(previous)
        attributes["sidecar_dataset_fingerprint"] = sidecar_fingerprint
        attributes["curation_source"] = provenance
    else:
        metadata["base_dataset_fingerprint"] = base_fingerprint
        parameters = metadata.get("scenario_parameters")
        if not isinstance(parameters, dict):
            raise S5AppendError("sidecar scenario parameters are missing")
        previous = parameters.get("curation_source")
        if isinstance(previous, Mapping):
            provenance["prior_curation_source"] = dict(previous)
        parameters["curation_source"] = provenance
    _atomic_json(target / "episode.json", metadata)
    return metadata


def build_staging(
    destination_root: Path,
    supplement_root: Path,
    staging_root: Path,
    plan: Mapping[str, object],
) -> None:
    if staging_root.exists():
        raise S5AppendError(f"staging root already exists: {staging_root}")
    if staging_root.parent.resolve() != destination_root.parent.resolve():
        raise S5AppendError("staging must be a sibling of the destination")
    staging_root.mkdir()
    base_root = staging_root / "platoon_joint_bev"
    sidecar_root = staging_root / "riskentry_actor_sidecar"
    for component_root in (base_root, sidecar_root):
        component_root.mkdir()
        (component_root / ".writer.lock").touch()
        for split in SPLIT_NAMES:
            (component_root / split / "episodes").mkdir(parents=True)
    (staging_root / ".bundle_writer.lock").touch()

    base_fingerprint = str(plan["base_dataset_fingerprint"])
    sidecar_fingerprint = str(plan["sidecar_dataset_fingerprint"])
    base_contract = _read_json(
        destination_root / "platoon_joint_bev/dataset_contract.json"
    )
    base_contract["dataset_fingerprint"] = base_fingerprint
    _atomic_json(base_root / "dataset_contract.json", base_contract)
    _atomic_json(
        sidecar_root / "dataset_contract.json",
        sidecar_dataset_contract(base_fingerprint),
    )
    bundle_manifest = _read_json(destination_root / "dataset_bundle_manifest.json")
    bundle_manifest["base_dataset_fingerprint"] = base_fingerprint
    bundle_manifest["sidecar_dataset_fingerprint"] = sidecar_fingerprint
    bundle_manifest["scenario_contract_sha256"] = plan[
        "curation_contract_sha256"
    ]
    _atomic_json(staging_root / "dataset_bundle_manifest.json", bundle_manifest)

    destination_base = _manifest_entries(destination_root, "platoon_joint_bev")
    destination_sidecar = _manifest_entries(
        destination_root, "riskentry_actor_sidecar"
    )
    inventory: list[dict[str, object]] = []
    base_metadata: dict[int, dict[str, object]] = {}
    sidecar_metadata: dict[int, dict[str, object]] = {}
    for index, (split, _) in sorted(destination_sidecar.items()):
        sidecar_metadata[index] = _copy_episode(
            source_root=destination_root,
            source_id=SOURCE_ID_DESTINATION,
            component="riskentry_actor_sidecar",
            source_split=split,
            source_index=index,
            target_root=staging_root,
            target_split=split,
            target_index=index,
            base_fingerprint=base_fingerprint,
            sidecar_fingerprint=sidecar_fingerprint,
            inventory=inventory,
        )
        if index in destination_base:
            base_metadata[index] = _copy_episode(
                source_root=destination_root,
                source_id=SOURCE_ID_DESTINATION,
                component="platoon_joint_bev",
                source_split=split,
                source_index=index,
                target_root=staging_root,
                target_split=split,
                target_index=index,
                base_fingerprint=base_fingerprint,
                sidecar_fingerprint=sidecar_fingerprint,
                inventory=inventory,
            )
    supplement_rows = _read_rows(supplement_root)
    for row in plan["supplement_episode_mapping"]:
        source_index = int(row["source_episode_index"])
        source_split = str(row["source_split"])
        target_index = int(row["target_episode_index"])
        target_split = str(row["target_split"])
        base_metadata[target_index] = _copy_episode(
            source_root=supplement_root,
            source_id=SOURCE_ID_SUPPLEMENT,
            component="platoon_joint_bev",
            source_split=source_split,
            source_index=source_index,
            target_root=staging_root,
            target_split=target_split,
            target_index=target_index,
            base_fingerprint=base_fingerprint,
            sidecar_fingerprint=sidecar_fingerprint,
            inventory=inventory,
        )
        sidecar_metadata[target_index] = _copy_episode(
            source_root=supplement_root,
            source_id=SOURCE_ID_SUPPLEMENT,
            component="riskentry_actor_sidecar",
            source_split=source_split,
            source_index=source_index,
            target_root=staging_root,
            target_split=target_split,
            target_index=target_index,
            base_fingerprint=base_fingerprint,
            sidecar_fingerprint=sidecar_fingerprint,
            inventory=inventory,
        )
    _write_manifests(staging_root, base_metadata, sidecar_metadata)

    source_state = _read_json(
        destination_root / "platoon_joint_bev/collection_state.json"
    )
    _atomic_json(
        base_root / "collection_state.json",
        {
            "schema_version": STORAGE_SCHEMA_VERSION,
            "attempted_episodes": 431,
            "next_episode_index": 431,
            "stored_episodes": 263,
            "rejected_episodes": 168,
            "rejection_reasons": dict(source_state.get("rejection_reasons", {})),
            "total_joint_samples": 50260,
        },
    )

    rows = [dict(row) for row in _read_rows(destination_root)]
    by_target = {
        int(row["target_episode_index"]): row
        for row in plan["supplement_episode_mapping"]
    }
    for target_index in range(401, 431):
        mapping = by_target[target_index]
        source_row = dict(
            supplement_rows[int(mapping["source_episode_index"])]
        )
        source_row["episode_index"] = target_index
        source_row["split"] = mapping["target_split"]
        rows.append(source_row)
    if [int(row["episode_index"]) for row in rows] != list(range(431)):
        raise S5AppendError("constructed bundle index is not contiguous")
    (staging_root / "bundle_episode_index.jsonl").write_bytes(
        b"".join(_canonical_json(row) + b"\n" for row in rows)
    )

    inventory_payload = {
        "format": INVENTORY_FORMAT,
        "entries": sorted(
            inventory,
            key=lambda row: (
                str(row["source_id"]),
                str(row["source_path"]),
                str(row["target_path"]),
            ),
        ),
    }
    _atomic_json(staging_root / "payload_inventory.json", inventory_payload)
    _atomic_json(
        staging_root / "dataset_curation_manifest.json",
        {
            "format": APPEND_FORMAT,
            "complete": True,
            "contract_payload": plan["contract_payload"],
            "base_dataset_fingerprint": base_fingerprint,
            "sidecar_dataset_fingerprint": sidecar_fingerprint,
            "curation_contract_sha256": plan["curation_contract_sha256"],
            "payload_inventory_sha256": _file_sha256(
                staging_root / "payload_inventory.json"
            ),
            "destination_source_path_before_in_place_replacement": str(
                destination_root
            ),
            "supplement_source_path": str(supplement_root),
            "formal_training_eligibility": False,
            "existing_destination_episodes_removed": 0,
            "backup_policy": "retain_after_successful_final_verification",
        },
    )


def verify_appended_bundle(
    root: Path | str, *, verify_payload_hashes: bool = True
) -> dict[str, object]:
    root = Path(root).resolve()
    _validate_no_links(root)
    manifest = _read_json(root / "dataset_curation_manifest.json")
    if manifest.get("format") != APPEND_FORMAT or manifest.get("complete") is not True:
        raise S5AppendError("append curation manifest is incomplete")
    contract = manifest.get("contract_payload")
    if not isinstance(contract, Mapping):
        raise S5AppendError("append contract payload is missing")
    base_fingerprint = _payload_sha256(contract)
    sidecar_fingerprint = sidecar_dataset_fingerprint(base_fingerprint)
    if (
        manifest.get("base_dataset_fingerprint") != base_fingerprint
        or manifest.get("sidecar_dataset_fingerprint") != sidecar_fingerprint
    ):
        raise S5AppendError("append dataset fingerprint mismatch")
    scenario_payload = contract.get("scenario_contract")
    if _payload_sha256(scenario_payload) != manifest.get("curation_contract_sha256"):
        raise S5AppendError("append scenario contract hash mismatch")
    source_contracts = scenario_payload.get("source_contracts")
    if not isinstance(source_contracts, Mapping):
        raise S5AppendError("append source contract allowlist is missing")
    allowed_hashes = sorted(str(value) for value in source_contracts.values())
    base_report = verify_joint_bev_dataset(root / "platoon_joint_bev")
    sidecar_report = verify_riskentry_sidecar_dataset(
        root / "riskentry_actor_sidecar",
        allowed_scenario_contract_sha256s=allowed_hashes,
    )
    bundle_manifest = _read_json(root / "dataset_bundle_manifest.json")
    if (
        bundle_manifest.get("format") != BUNDLE_FORMAT
        or bundle_manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION
        or bundle_manifest.get("protocol_sha256") != bundle_protocol_sha256()
        or bundle_manifest.get("base_dataset_fingerprint") != base_fingerprint
        or bundle_manifest.get("sidecar_dataset_fingerprint") != sidecar_fingerprint
        or bundle_manifest.get("scenario_contract_sha256")
        != manifest.get("curation_contract_sha256")
        or float(bundle_manifest.get("decision_dt_s", -1.0)) != 0.1
        or int(bundle_manifest.get("split_seed", -1)) != 17
    ):
        raise S5AppendError("append bundle protocol binding mismatch")

    rows = _read_rows(root)
    base_entries = _manifest_entries(root, "platoon_joint_bev")
    sidecar_entries = _manifest_entries(root, "riskentry_actor_sidecar")
    expected = contract.get("expected_output")
    if not isinstance(expected, Mapping):
        raise S5AppendError("append expected output is missing")
    if {key: expected[key] for key in EXPECTED_FINAL_COUNTS} != EXPECTED_FINAL_COUNTS:
        raise S5AppendError("append expected counts drifted from release30")
    if (
        len(rows) != 431
        or len(base_entries) != 263
        or len(sidecar_entries) != 363
        or len(set(sidecar_entries) - set(base_entries)) != 100
    ):
        raise S5AppendError("append bundle component counts mismatch")

    scenario_seeds = []
    appended_count = 0
    for row in rows:
        index = int(row["episode_index"])
        split = str(row["split"])
        has_base = index in base_entries
        has_sidecar = index in sidecar_entries
        if has_base != (row["base_status"] == "committed"):
            raise S5AppendError("append bundle/base status mismatch")
        if has_sidecar != (row["sidecar_status"] == "committed"):
            raise S5AppendError("append bundle/sidecar status mismatch")
        if not has_base:
            continue
        base_metadata = _read_json(
            _episode_path(root, "platoon_joint_bev", split, index) / "episode.json"
        )
        sidecar_metadata = _read_json(
            _episode_path(root, "riskentry_actor_sidecar", split, index)
            / "episode.json"
        )
        attributes = base_metadata.get("attributes")
        parameters = sidecar_metadata.get("scenario_parameters")
        if not isinstance(attributes, Mapping) or not isinstance(parameters, Mapping):
            raise S5AppendError("append episode provenance metadata is missing")
        if attributes.get("sidecar_dataset_fingerprint") != sidecar_fingerprint:
            raise S5AppendError("append base/sidecar fingerprint binding mismatch")
        if attributes.get("scenario_contract_sha256") not in allowed_hashes:
            raise S5AppendError("append episode uses an undeclared scenario contract")
        if parameters.get("scenario_contract_sha256") != attributes.get(
            "scenario_contract_sha256"
        ):
            raise S5AppendError("append base/sidecar scenario contract mismatch")
        provenance = attributes.get("curation_source")
        sidecar_provenance = parameters.get("curation_source")
        if not isinstance(provenance, Mapping) or not isinstance(
            sidecar_provenance, Mapping
        ):
            raise S5AppendError("append curation provenance is missing")
        if provenance.get("source_id") == SOURCE_ID_SUPPLEMENT:
            _validate_release_evidence(attributes)
            appended_count += 1
        scenario_seeds.append(
            (str(attributes["scenario_id"]), int(attributes["spawn_seed"]))
        )
    if appended_count != 30 or len(scenario_seeds) != len(set(scenario_seeds)):
        raise S5AppendError("release30 count or scenario/seed uniqueness failed")
    if _source_statistics(root) != EXPECTED_FINAL_COUNTS:
        raise S5AppendError("append final statistics mismatch")
    state = _read_json(root / "platoon_joint_bev/collection_state.json")
    if (
        state.get("next_episode_index") != 431
        or state.get("attempted_episodes") != 431
        or state.get("stored_episodes") != 263
        or state.get("rejected_episodes") != 168
        or state.get("total_joint_samples") != 50260
    ):
        raise S5AppendError("append collection state mismatch")
    for pending in (".bundle_episode_pending.json", ".targeted_batch_pending.json"):
        if (root / pending).exists():
            raise S5AppendError(f"append root contains pending state: {pending}")
    if (
        int(base_report["episodes"]) != 263
        or int(base_report["joint_samples"]) != 50260
        or int(sidecar_report["episodes"]) != 363
        or int(sidecar_report["raw_steps"]) != int(expected["sidecar_raw_steps"])
    ):
        raise S5AppendError("append strict verifier totals mismatch")
    inventory_path = root / "payload_inventory.json"
    if _file_sha256(inventory_path) != manifest.get("payload_inventory_sha256"):
        raise S5AppendError("append payload inventory hash mismatch")
    verified_payload_files = 0
    if verify_payload_hashes:
        inventory = _read_json(inventory_path)
        entries = inventory.get("entries")
        if inventory.get("format") != INVENTORY_FORMAT or not isinstance(entries, list):
            raise S5AppendError("append payload inventory format mismatch")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise S5AppendError("append payload inventory entry is invalid")
            target = root / str(entry["target_path"])
            if target.is_symlink() or _file_sha256(target) != entry.get("sha256"):
                raise S5AppendError(f"append payload hash mismatch: {target}")
            verified_payload_files += 1
    return {
        "format": APPEND_FORMAT,
        "bundle_root": str(root),
        "base_dataset_fingerprint": base_fingerprint,
        "sidecar_dataset_fingerprint": sidecar_fingerprint,
        "curation_contract_sha256": manifest["curation_contract_sha256"],
        "bundle_index_sha256": _file_sha256(root / "bundle_episode_index.jsonl"),
        **EXPECTED_FINAL_COUNTS,
        "sidecar_raw_steps": int(expected["sidecar_raw_steps"]),
        "new_release30_episodes": appended_count,
        "verified_payload_files": verified_payload_files,
        "eligible_for_formal_training": False,
    }


def _swap_paths(destination_root: Path) -> tuple[Path, Path, Path]:
    staging = destination_root.with_name(
        f"{destination_root.name}.s5_release30_append_staging"
    )
    backup = destination_root.with_name(
        f"{destination_root.name}.pre_s5_release30_append_b93dfa5c"
    )
    failed = destination_root.with_name(
        f"{destination_root.name}.failed_s5_release30_append"
    )
    return staging, backup, failed


def _validate_swap_paths(destination_root: Path) -> tuple[Path, Path, Path]:
    staging, backup, failed = _swap_paths(destination_root)
    parent = destination_root.parent.resolve()
    if destination_root.is_symlink() or not destination_root.is_dir():
        raise S5AppendError("append destination must be a real directory")
    if any(path.parent.resolve() != parent for path in (staging, backup, failed)):
        raise S5AppendError("append swap paths must share the destination parent")
    if any(path.is_symlink() for path in (destination_root, staging, backup, failed)):
        raise S5AppendError("append swap paths must not be symbolic links")
    return staging, backup, failed


def execute_append(
    destination_root: Path,
    supplement_root: Path,
    plan: Mapping[str, object],
) -> dict[str, object]:
    staging, backup, failed = _validate_swap_paths(destination_root)
    if staging.exists() or backup.exists() or failed.exists():
        raise S5AppendError("append staging, backup, or failed root already exists")
    with _exclusive_source_locks((destination_root, supplement_root)):
        contract = plan["contract_payload"]
        if not isinstance(contract, Mapping):
            raise S5AppendError("append contract payload is missing")
        destination_source = contract["destination_source"]
        supplement_source = contract["supplement_source"]
        if not isinstance(destination_source, Mapping) or not isinstance(
            supplement_source, Mapping
        ):
            raise S5AppendError("append source bindings are missing")
        if (
            _file_sha256(destination_root / "bundle_episode_index.jsonl")
            != destination_source["bundle_index_sha256"]
            or _file_sha256(destination_root / "dataset_curation_manifest.json")
            != destination_source["dataset_curation_manifest_sha256"]
            or _file_sha256(supplement_root / "bundle_episode_index.jsonl")
            != supplement_source["bundle_index_sha256"]
        ):
            raise S5AppendError("append source hashes drifted before staging")
        build_staging(destination_root, supplement_root, staging, plan)
        staging_report = verify_appended_bundle(staging)
        os.replace(destination_root, backup)
        try:
            os.replace(staging, destination_root)
        except Exception:
            os.replace(backup, destination_root)
            raise
        try:
            final_report = verify_appended_bundle(destination_root)
            expected_final = {
                **staging_report,
                "bundle_root": str(destination_root.resolve()),
            }
            if final_report != expected_final:
                raise S5AppendError(
                    "staging and final append verification reports differ"
                )
        except Exception:
            os.replace(destination_root, failed)
            os.replace(backup, destination_root)
            raise
    return {
        **final_report,
        "backup_root": str(backup),
        "backup_retained": backup.is_dir(),
        "rollback_performed": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination-root", type=Path, default=DESTINATION_ROOT)
    parser.add_argument("--supplement-root", type=Path, default=SUPPLEMENT_ROOT)
    parser.add_argument("--supplement-config", type=Path, default=SUPPLEMENT_CONFIG)
    parser.add_argument(
        "--expected-destination-fingerprint",
        default=EXPECTED_DESTINATION_FINGERPRINT,
    )
    parser.add_argument(
        "--expected-destination-index-sha256",
        default=EXPECTED_DESTINATION_INDEX_SHA256,
    )
    parser.add_argument("--expected-supplement-fingerprint", required=True)
    parser.add_argument("--expected-supplement-index-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--replace-in-place", action="store_true")
    parser.add_argument("--result-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    destination_root = args.destination_root.expanduser().resolve()
    supplement_root = args.supplement_root.expanduser().resolve()
    supplement_config = args.supplement_config.expanduser().resolve()
    plan = inspect_sources(
        destination_root,
        supplement_root,
        supplement_config,
        expected_destination_fingerprint=args.expected_destination_fingerprint,
        expected_destination_index_sha256=(
            args.expected_destination_index_sha256
        ),
        expected_supplement_fingerprint=args.expected_supplement_fingerprint,
        expected_supplement_index_sha256=args.expected_supplement_index_sha256,
        verify_destination_payload_hashes=True,
    )
    if args.execute:
        if not args.replace_in_place:
            raise S5AppendError("execution requires --replace-in-place")
        executed = execute_append(destination_root, supplement_root, plan)
        result = {"mode": "executed", **executed}
    else:
        result = {"mode": "dry_run", **plan}
    if args.result_json is not None:
        _atomic_json(args.result_json.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except S5AppendError as exc:
        raise SystemExit(f"error: {exc}") from exc


__all__ = [
    "S5AppendError",
    "build_staging",
    "execute_append",
    "inspect_sources",
    "verify_appended_bundle",
]
