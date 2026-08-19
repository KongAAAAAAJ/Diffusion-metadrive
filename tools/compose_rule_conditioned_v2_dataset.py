"""Freeze S5 quotas and compose the audited rule-conditioned v2 dataset."""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import hashlib
import json
import shutil
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml

from expert_dataset.finalize_s5_targeted_supplement import (
    audit_targeted_supplement,
)
from expert_dataset.joint_bev_storage import (
    EpisodeSplitAssigner,
    EpisodeSplitConfig,
    SPLIT_NAMES,
    joint_sample_storage_contract,
)
from expert_dataset.joint_risk_bundle_contract import bundle_protocol_sha256
from expert_dataset.riskentry_sidecar_storage import (
    SIDECAR_FORMAT,
    SIDECAR_SCHEMA_VERSION,
    sidecar_dataset_contract,
    sidecar_dataset_fingerprint,
)
from expert_dataset.run_joint_bev_collection import load_run_config
from scenarios.bev_round13_contract import (
    CANDIDATE_V4_CONTRACT_ID,
    scenario_contract_for_id,
)
from tools.rebalance_s5_dataset import (
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
)


S5_SCENARIO = "S5_hard_brake_lead"
S6_S9_SCENARIOS = (
    "S6_background_merge_in",
    "S7_ego_merge_from_ramp",
    "S8_ego_exit_to_ramp",
    "S9_narrow_channel_negotiation",
)
RELEASE_CATEGORY = "temporary_formation_release_and_recovery"
APPEND_EPISODES = 53
SAMPLES_PER_S5_EPISODE = 190
S6_S9_SAMPLES_PER_SCENARIO = 10_000
S6_S9_TOTAL_SAMPLES = 40_000
FINAL_TOTAL_SAMPLES = 50_070
COMPOSITION_FORMAT = "rule-conditioned-v2-s5-s9-composition-v1"
INVENTORY_FORMAT = "rule-conditioned-v2-payload-inventory-v1"
PREPARE_CONTRACT_FORMAT = "rule-conditioned-v2-s5-config-contract-v1"
SOURCE_ID_BASE = "rule_conditioned_v2_s6_s9_formal40k"
SOURCE_ID_S5 = "rule_conditioned_v2_s5_release10070"


class RuleConditionedV2CompositionError(RuntimeError):
    """Raised when a frozen source or composition safety gate fails."""


def _strict_verify(root: Path) -> dict[str, object]:
    from expert_dataset.verify_rule_conditioned_v2_bundle import (
        verify_rule_conditioned_v2_bundle,
    )

    report = verify_rule_conditioned_v2_bundle(root)
    if report.get("status") != "pass" or report.get("planner_version") != "v2":
        raise RuleConditionedV2CompositionError(
            f"rule-conditioned v2 verification did not pass: {root}"
        )
    return report


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_new_or_identical(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise RuleConditionedV2CompositionError(
                f"output exists with different content: {path}"
            )
        return
    _atomic_bytes(path, payload)


def _validate_source_root(root: Path) -> Path:
    path = root.expanduser().resolve()
    if root.is_symlink() or not path.is_dir():
        raise RuleConditionedV2CompositionError(
            f"source must be a real directory: {root}"
        )
    _validate_no_links(path)
    for pending in (".bundle_episode_pending.json", ".targeted_batch_pending.json"):
        if (path / pending).exists():
            raise RuleConditionedV2CompositionError(
                f"source contains pending state: {path / pending}"
            )
    return path


def _source_binding(root: Path) -> dict[str, object]:
    base_contract = _read_json(root / "platoon_joint_bev/dataset_contract.json")
    bundle_manifest = _read_json(root / "dataset_bundle_manifest.json")
    storage = joint_sample_storage_contract("v2")
    if (
        base_contract.get("planner_version") != "v2"
        or int(base_contract.get("schema_version", -1)) != storage.schema_version
        or base_contract.get("format") != storage.storage_format
        or bundle_manifest.get("protocol_sha256")
        != bundle_protocol_sha256(planner_version="v2")
        or bundle_manifest.get("scenario_contract_sha256")
        != scenario_contract_for_id(CANDIDATE_V4_CONTRACT_ID)["sha256"]
    ):
        raise RuleConditionedV2CompositionError(
            f"source is not a candidate-v4 rule-conditioned v2 bundle: {root}"
        )
    fingerprint = str(base_contract.get("dataset_fingerprint", ""))
    if bundle_manifest.get("base_dataset_fingerprint") != fingerprint:
        raise RuleConditionedV2CompositionError("source base fingerprint binding mismatch")
    return {
        "base_dataset_fingerprint": fingerprint,
        "sidecar_dataset_fingerprint": str(
            bundle_manifest.get("sidecar_dataset_fingerprint", "")
        ),
        "bundle_index_sha256": _file_sha256(root / "bundle_episode_index.jsonl"),
    }


def _target_slots(start: int, count: int = APPEND_EPISODES) -> dict[str, list[int]]:
    if start < 0 or count <= 0:
        raise RuleConditionedV2CompositionError("target slot range is invalid")
    assigner = EpisodeSplitAssigner(EpisodeSplitConfig(seed=17))
    slots = {split: [] for split in SPLIT_NAMES}
    for index in range(start, start + count):
        slots[assigner.split_for_episode(index)].append(index)
    return slots


def _base_scenario_samples(root: Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for _, (_, entry) in _manifest_entries(root, "platoon_joint_bev").items():
        attributes = entry.get("attributes")
        if not isinstance(attributes, Mapping):
            raise RuleConditionedV2CompositionError("base manifest attributes missing")
        counts[str(attributes.get("scenario_id"))] += int(entry["joint_samples"])
    return dict(counts)


def _validate_completed_s6_s9(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    report = _strict_verify(root)
    binding = _source_binding(root)
    state = _read_json(root / "platoon_joint_bev/collection_state.json")
    rows = _read_rows(root)
    expected = {name: S6_S9_SAMPLES_PER_SCENARIO for name in S6_S9_SCENARIOS}
    stored_episodes = len(_manifest_entries(root, "platoon_joint_bev"))
    if (
        int(state.get("total_joint_samples", -1)) != S6_S9_TOTAL_SAMPLES
        or int(state.get("next_episode_index", -1)) != len(rows)
        or int(state.get("attempted_episodes", -1)) != len(rows)
        or int(state.get("stored_episodes", -1)) != stored_episodes
        or int(state.get("rejected_episodes", -1))
        != len(rows) - stored_episodes
        or _base_scenario_samples(root) != expected
    ):
        raise RuleConditionedV2CompositionError(
            "S6-S9 source is not the completed formal40k bundle"
        )
    return report, {
        **binding,
        "next_episode_index": len(rows),
        "stored_episodes": stored_episodes,
        "rejected_episodes": len(rows) - stored_episodes,
        "rejection_reasons": dict(state.get("rejection_reasons", {})),
    }


def _validate_template(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise RuleConditionedV2CompositionError("S5 template root must be a mapping")
    dataset = payload.get("dataset")
    collection = payload.get("collection")
    targeted = payload.get("targeted_supplement")
    split = payload.get("split")
    if not all(isinstance(value, dict) for value in (dataset, collection, targeted, split)):
        raise RuleConditionedV2CompositionError("S5 template sections are missing")
    assert isinstance(dataset, dict)
    assert isinstance(collection, dict)
    assert isinstance(targeted, dict)
    assert isinstance(split, dict)
    required = (
        dataset.get("planner_version") == "v2"
        and dataset.get("scenario_contract") == CANDIDATE_V4_CONTRACT_ID
        and int(collection.get("target_joint_steps", -1)) == 10_070
        and int(collection.get("start_seed", -1)) == 83_017
        and int(split.get("seed", -1)) == 17
        and targeted.get("scenario_id") == S5_SCENARIO
        and targeted.get("behavior_category") == RELEASE_CATEGORY
        and targeted.get("rule_maker_profile_id") == "balanced"
        and targeted.get("require_target_background_condition") is True
        and targeted.get("finalization_mode") == "audit_only"
        and int(targeted.get("required_samples_per_episode", -1))
        == SAMPLES_PER_S5_EPISODE
        and int(targeted.get("max_simulator_attempts", -1)) == 300
        and targeted.get("bootstrap_spawn_seeds") == list(range(83_017, 83_027))
    )
    if not required:
        raise RuleConditionedV2CompositionError("S5 template contract mismatch")
    return payload


def prepare_s5_config(
    base_root: Path | str,
    template_path: Path | str,
    output_config: Path | str,
    output_contract: Path | str,
) -> dict[str, object]:
    base = _validate_source_root(Path(base_root))
    template = Path(template_path).expanduser().resolve()
    config_output = Path(output_config).expanduser().resolve()
    contract_output = Path(output_contract).expanduser().resolve()
    if config_output == contract_output:
        raise RuleConditionedV2CompositionError("config and contract outputs overlap")
    report, source = _validate_completed_s6_s9(base)
    raw = _validate_template(yaml.safe_load(template.read_text(encoding="utf-8")))
    start = int(source["next_episode_index"])
    slots = _target_slots(start)
    quotas = {split: len(slots[split]) for split in SPLIT_NAMES}
    raw["targeted_supplement"]["accepted_episode_quotas"] = quotas
    config_bytes = yaml.safe_dump(
        raw, allow_unicode=True, sort_keys=False
    ).encode("utf-8")

    descriptor, temporary_name = tempfile.mkstemp(suffix=".yaml")
    try:
        stream = os.fdopen(descriptor, "wb")
        descriptor = -1
        with stream:
            stream.write(config_bytes)
        generated = load_run_config(Path(temporary_name))
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        Path(temporary_name).unlink(missing_ok=True)
    if generated.targeted_supplement is None or generated.targeted_supplement.target_episodes != APPEND_EPISODES:
        raise RuleConditionedV2CompositionError("generated S5 config is invalid")

    contract = {
        "format": PREPARE_CONTRACT_FORMAT,
        "base_source": {
            "bundle_root": str(base),
            **source,
        },
        "strict_verifier": {
            "status": report.get("status"),
            "aligned_samples": report.get("aligned_samples"),
            "aligned_episodes": report.get("aligned_episodes"),
        },
        "split_assignment": EpisodeSplitConfig(seed=17).as_dict(),
        "append_episode_count": APPEND_EPISODES,
        "append_slots": [
            {"episode_index": index, "split": split}
            for split in SPLIT_NAMES
            for index in slots[split]
        ],
        "accepted_episode_quotas": quotas,
        "template_sha256": _file_sha256(template),
        "generated_config_path": str(config_output),
        "generated_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }
    contract["append_slots"].sort(key=lambda row: int(row["episode_index"]))
    contract_bytes = (
        json.dumps(contract, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    for path, payload in (
        (config_output, config_bytes),
        (contract_output, contract_bytes),
    ):
        if path.exists() and (path.is_symlink() or path.read_bytes() != payload):
            raise RuleConditionedV2CompositionError(
                f"output exists with different content: {path}"
            )
    _write_new_or_identical(config_output, config_bytes)
    _write_new_or_identical(contract_output, contract_bytes)
    return contract


def _episode_path(
    root: Path, component: str, split: str, episode_index: int
) -> Path:
    return root / component / split / "episodes" / f"episode_{episode_index:08d}"


def _episode_hashes(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_dir():
        raise RuleConditionedV2CompositionError(f"invalid episode directory: {path}")
    hashes = {}
    for item in sorted(path.iterdir()):
        if item.is_symlink() or not item.is_file():
            raise RuleConditionedV2CompositionError(f"invalid episode payload: {item}")
        hashes[item.name] = _file_sha256(item)
    return hashes


def _validate_s5_attributes(attributes: Mapping[str, object]) -> None:
    evidence = attributes.get("targeted_supplement_evidence")
    if (
        attributes.get("scenario_id") != S5_SCENARIO
        or attributes.get("rule_maker_profile_id") != "balanced"
        or attributes.get("scenario_contract_id") != CANDIDATE_V4_CONTRACT_ID
        or not isinstance(evidence, Mapping)
        or evidence.get("behavior_category") != RELEASE_CATEGORY
        or evidence.get("target_background_condition_sampled") is not True
        or evidence.get("target_background_condition_realized") is not True
        or evidence.get("platoon_safety_events") != []
        or evidence.get("lateral_mode_runs_by_role") != [2, 2, 2]
    ):
        raise RuleConditionedV2CompositionError(
            "S5 episode is not strict balanced target release/recovery evidence"
        )
    directions = evidence.get("lateral_run_directions_by_role")
    ranges = evidence.get("lateral_range_m_by_role")
    returns = evidence.get("return_error_m_by_role")
    if (
        not isinstance(directions, list)
        or len(directions) != 3
        or any(
            not isinstance(row, list)
            or len(row) != 2
            or row[0] == row[1]
            or "mixed" in row
            for row in directions
        )
        or len({row[0] for row in directions}) < 2
        or not isinstance(ranges, list)
        or len(ranges) != 3
        or any(float(value) < 2.5 for value in ranges)
        or not isinstance(returns, list)
        or len(returns) != 3
        or any(float(value) > 0.5 for value in returns)
    ):
        raise RuleConditionedV2CompositionError("S5 physical recovery evidence failed")


def _source_statistics(root: Path) -> dict[str, object]:
    base = _manifest_entries(root, "platoon_joint_bev")
    sidecar = _manifest_entries(root, "riskentry_actor_sidecar")
    split_episodes: Counter[str] = Counter()
    split_samples: Counter[str] = Counter()
    for _, (split, entry) in base.items():
        split_episodes[split] += 1
        split_samples[split] += int(entry["joint_samples"])
    raw_steps = sum(
        int(entry["raw_steps"]) for _, entry in sidecar.values()
    )
    return {
        "base_episodes": len(base),
        "base_joint_samples": sum(split_samples.values()),
        "sidecar_episodes": len(sidecar),
        "sidecar_only_episodes": len(set(sidecar) - set(base)),
        "sidecar_raw_steps": raw_steps,
        "bundle_index_rows": len(_read_rows(root)),
        "split_episode_counts": {
            split: int(split_episodes[split]) for split in SPLIT_NAMES
        },
        "split_joint_samples": {
            split: int(split_samples[split]) for split in SPLIT_NAMES
        },
    }


def _check_expected_binding(
    name: str,
    observed: Mapping[str, object],
    *,
    fingerprint: str,
    index_sha256: str,
) -> None:
    if observed.get("base_dataset_fingerprint") != fingerprint:
        raise RuleConditionedV2CompositionError(f"{name} base fingerprint drift")
    if observed.get("bundle_index_sha256") != index_sha256:
        raise RuleConditionedV2CompositionError(f"{name} bundle index hash drift")


def _validated_frozen_slots(
    freeze: Mapping[str, object],
    base_binding: Mapping[str, object],
    supplement_config: Path,
    config,
) -> dict[str, list[int]]:
    if freeze.get("format") != PREPARE_CONTRACT_FORMAT:
        raise RuleConditionedV2CompositionError("invalid S5 freeze contract")
    frozen_source = freeze.get("base_source")
    frozen_slots = freeze.get("append_slots")
    if (
        not isinstance(frozen_source, Mapping)
        or not isinstance(frozen_slots, list)
        or frozen_source.get("base_dataset_fingerprint")
        != base_binding["base_dataset_fingerprint"]
        or frozen_source.get("bundle_index_sha256")
        != base_binding["bundle_index_sha256"]
        or int(frozen_source.get("next_episode_index", -1))
        != int(base_binding["next_episode_index"])
        or freeze.get("generated_config_path") != str(supplement_config)
        or freeze.get("generated_config_sha256") != _file_sha256(supplement_config)
    ):
        raise RuleConditionedV2CompositionError("S5 freeze contract base/config drift")
    slots = {split: [] for split in SPLIT_NAMES}
    for item in frozen_slots:
        if not isinstance(item, Mapping) or item.get("split") not in SPLIT_NAMES:
            raise RuleConditionedV2CompositionError("invalid frozen append slot")
        slots[str(item["split"])].append(int(item["episode_index"]))
    expected_slots = _target_slots(int(base_binding["next_episode_index"]))
    expected_quotas = {
        split: len(expected_slots[split]) for split in SPLIT_NAMES
    }
    requirements = config.targeted_supplement
    loaded_quotas = (
        None
        if requirements is None
        else dict(requirements.accepted_episode_quotas)
    )
    if (
        slots != expected_slots
        or freeze.get("accepted_episode_quotas") != expected_quotas
        or loaded_quotas != expected_quotas
    ):
        raise RuleConditionedV2CompositionError(
            "frozen append slots/accepted quotas drifted"
        )
    return slots


def inspect_sources(
    base_root: Path | str,
    supplement_root: Path | str,
    supplement_config: Path | str,
    s5_contract_path: Path | str,
    *,
    expected_base_fingerprint: str,
    expected_base_index_sha256: str,
    expected_supplement_fingerprint: str,
    expected_supplement_index_sha256: str,
) -> dict[str, object]:
    base = _validate_source_root(Path(base_root))
    supplement = _validate_source_root(Path(supplement_root))
    if base == supplement:
        raise RuleConditionedV2CompositionError("base and supplement roots overlap")
    _, base_binding = _validate_completed_s6_s9(base)
    _check_expected_binding(
        "base",
        base_binding,
        fingerprint=expected_base_fingerprint,
        index_sha256=expected_base_index_sha256,
    )
    _strict_verify(supplement)
    supplement_binding = _source_binding(supplement)
    _check_expected_binding(
        "supplement",
        supplement_binding,
        fingerprint=expected_supplement_fingerprint,
        index_sha256=expected_supplement_index_sha256,
    )

    supplement_config = Path(supplement_config).expanduser().resolve()
    config = load_run_config(supplement_config)
    if (
        config.bundle_root.resolve() != supplement
        or config.planner_version != "v2"
        or config.scenario_contract_id != CANDIDATE_V4_CONTRACT_ID
        or config.immutable_fingerprint() != expected_supplement_fingerprint
        or config.targeted_supplement is None
        or config.targeted_supplement.target_episodes != APPEND_EPISODES
    ):
        raise RuleConditionedV2CompositionError("S5 config/source binding mismatch")
    audit = audit_targeted_supplement(config)
    if (
        audit.get("complete") is not True
        or int(audit.get("accepted_episodes", -1)) != APPEND_EPISODES
        or audit.get("pending_transaction") is not False
    ):
        raise RuleConditionedV2CompositionError("S5 targeted audit is incomplete")

    freeze = _read_json(Path(s5_contract_path).expanduser().resolve())
    slots = _validated_frozen_slots(
        freeze, base_binding, supplement_config, config
    )

    supplement_base = _manifest_entries(supplement, "platoon_joint_bev")
    supplement_sidecar = _manifest_entries(supplement, "riskentry_actor_sidecar")
    if len(supplement_base) != APPEND_EPISODES or not set(supplement_base) <= set(supplement_sidecar):
        raise RuleConditionedV2CompositionError(
            "S5 supplement must contain exactly 53 joined accepted episodes"
        )
    by_split = {split: [] for split in SPLIT_NAMES}
    for index, (split, entry) in sorted(supplement_base.items()):
        if int(entry["joint_samples"]) != SAMPLES_PER_S5_EPISODE:
            raise RuleConditionedV2CompositionError("S5 episode sample count mismatch")
        attributes = entry.get("attributes")
        if not isinstance(attributes, Mapping):
            raise RuleConditionedV2CompositionError("S5 attributes missing")
        _validate_s5_attributes(attributes)
        by_split[split].append(index)
    if {split: len(by_split[split]) for split in SPLIT_NAMES} != {
        split: len(slots[split]) for split in SPLIT_NAMES
    }:
        raise RuleConditionedV2CompositionError("S5 accepted split quotas drifted")

    mapping = []
    for split in SPLIT_NAMES:
        for source_index, target_index in zip(sorted(by_split[split]), slots[split]):
            attributes = supplement_base[source_index][1]["attributes"]
            mapping.append(
                {
                    "source_episode_index": source_index,
                    "source_split": split,
                    "target_episode_index": target_index,
                    "target_split": split,
                    "spawn_seed": int(attributes["spawn_seed"]),
                    "base_file_sha256": _episode_hashes(
                        _episode_path(
                            supplement, "platoon_joint_bev", split, source_index
                        )
                    ),
                    "sidecar_file_sha256": _episode_hashes(
                        _episode_path(
                            supplement,
                            "riskentry_actor_sidecar",
                            split,
                            source_index,
                        )
                    ),
                }
            )
    mapping.sort(key=lambda row: int(row["target_episode_index"]))

    identities = [
        (
            str(row["scenario_id"]),
            int(row["spawn_seed"]),
        )
        for row in _read_rows(base)
    ] + [(S5_SCENARIO, int(row["spawn_seed"])) for row in mapping]
    duplicates = sorted(key for key, count in Counter(identities).items() if count > 1)
    if duplicates:
        raise RuleConditionedV2CompositionError(
            f"duplicate (scenario_id, spawn_seed) pairs: {duplicates}"
        )

    base_stats = _source_statistics(base)
    selected_raw_steps = sum(
        int(
            np.load(
                _episode_path(
                    supplement,
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
    final_split_episodes = dict(base_stats["split_episode_counts"])
    final_split_samples = dict(base_stats["split_joint_samples"])
    for split in SPLIT_NAMES:
        final_split_episodes[split] += len(slots[split])
        final_split_samples[split] += len(slots[split]) * SAMPLES_PER_S5_EPISODE
    expected_output = {
        "base_episodes": int(base_stats["base_episodes"]) + APPEND_EPISODES,
        "base_joint_samples": FINAL_TOTAL_SAMPLES,
        "sidecar_episodes": int(base_stats["sidecar_episodes"]) + APPEND_EPISODES,
        "sidecar_only_episodes": int(base_stats["sidecar_only_episodes"]),
        "sidecar_raw_steps": int(base_stats["sidecar_raw_steps"]) + selected_raw_steps,
        "bundle_index_rows": int(base_stats["bundle_index_rows"]) + APPEND_EPISODES,
        "rejected_episodes": int(base_stats["bundle_index_rows"])
        - int(base_stats["base_episodes"]),
        "split_episode_counts": final_split_episodes,
        "split_joint_samples": final_split_samples,
        "new_s5_release_episodes": APPEND_EPISODES,
    }
    contract_payload = {
        "format": COMPOSITION_FORMAT,
        "base_source": {**base_binding, "bundle_root": str(base)},
        "supplement_source": {
            **supplement_binding,
            "bundle_root": str(supplement),
            "config_path": str(supplement_config),
            "config_sha256": _file_sha256(supplement_config),
            "freeze_contract_path": str(Path(s5_contract_path).resolve()),
            "freeze_contract_sha256": _file_sha256(Path(s5_contract_path).resolve()),
        },
        "selection_policy": "all_53_strict_balanced_target_release_base_episodes",
        "base_policy": "preserve_all_s6_s9_bundle_rows_and_component_episodes",
        "supplement_episode_mapping": mapping,
        "split_assignment": EpisodeSplitConfig(seed=17).as_dict(),
        "scenario_contract_id": CANDIDATE_V4_CONTRACT_ID,
        "scenario_contract_sha256": scenario_contract_for_id(
            CANDIDATE_V4_CONTRACT_ID
        )["sha256"],
        "expected_output": expected_output,
    }
    fingerprint = _payload_sha256(contract_payload)
    storage = joint_sample_storage_contract("v2")
    return {
        "format": COMPOSITION_FORMAT,
        "base_root": str(base),
        "supplement_root": str(supplement),
        "contract_payload": contract_payload,
        "base_dataset_fingerprint": fingerprint,
        "sidecar_dataset_fingerprint": sidecar_dataset_fingerprint(
            fingerprint,
            base_format=storage.storage_format,
            base_schema_version=storage.schema_version,
        ),
        "supplement_episode_mapping": mapping,
        "expected_output": expected_output,
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
        raise RuleConditionedV2CompositionError(f"duplicate target episode: {target}")
    target.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        if path.name == "episode.json":
            continue
        if path.is_symlink() or path.suffix != ".npy":
            raise RuleConditionedV2CompositionError(f"unexpected payload: {path}")
        _copy_payload_file(
            path,
            target / path.name,
            inventory=inventory,
            source_id=source_id,
            source_relative=str(path.relative_to(source_root)),
            target_relative=str((target / path.name).relative_to(target_root)),
        )
    metadata = _read_json(source / "episode.json")
    provenance = _curation_source(source_id, source_root, source_split, source_index)
    metadata["episode_index"] = target_index
    metadata["split"] = target_split
    if component == "platoon_joint_bev":
        attributes = metadata.get("attributes")
        if not isinstance(attributes, dict):
            raise RuleConditionedV2CompositionError("base attributes missing")
        previous = attributes.get("curation_source")
        if isinstance(previous, Mapping):
            provenance["prior_curation_source"] = dict(previous)
        attributes["sidecar_dataset_fingerprint"] = sidecar_fingerprint
        attributes["curation_source"] = provenance
    else:
        metadata["base_dataset_fingerprint"] = base_fingerprint
        parameters = metadata.get("scenario_parameters")
        if not isinstance(parameters, dict):
            raise RuleConditionedV2CompositionError("sidecar parameters missing")
        previous = parameters.get("curation_source")
        if isinstance(previous, Mapping):
            provenance["prior_curation_source"] = dict(previous)
        parameters["curation_source"] = provenance
    _atomic_json(target / "episode.json", metadata)
    return metadata


def _write_manifests(
    root: Path,
    base_metadata: Mapping[int, Mapping[str, object]],
    sidecar_metadata: Mapping[int, Mapping[str, object]],
) -> None:
    storage = joint_sample_storage_contract("v2")
    for split in SPLIT_NAMES:
        base_entries = [
            {
                "episode_index": index,
                "directory": f"episode_{index:08d}",
                "joint_samples": int(metadata["joint_samples"]),
                "attributes": metadata["attributes"],
            }
            for index, metadata in sorted(base_metadata.items())
            if metadata["split"] == split
        ]
        sidecar_entries = []
        for index, metadata in sorted(sidecar_metadata.items()):
            if metadata["split"] != split:
                continue
            path = _episode_path(root, "riskentry_actor_sidecar", split, index)
            sidecar_entries.append(
                {
                    "episode_index": index,
                    "directory": f"episode_{index:08d}",
                    "raw_steps": int(np.load(path / "step_index.npy", mmap_mode="r").shape[0]),
                    "actor_count": int(np.load(path / "actor_state.npy", mmap_mode="r").shape[1]),
                    "base_samples": int(
                        np.load(path / "base_sample_step_index.npy", mmap_mode="r").shape[0]
                    ),
                    "outcome": str(metadata["retention"]["outcome"]),
                }
            )
        _atomic_json(
            root / "platoon_joint_bev" / split / "manifest.json",
            {
                "schema_version": storage.schema_version,
                "format": storage.storage_format,
                "split": split,
                "episode_count": len(base_entries),
                "joint_samples": sum(int(row["joint_samples"]) for row in base_entries),
                "episodes": base_entries,
            },
        )
        _atomic_json(
            root / "riskentry_actor_sidecar" / split / "manifest.json",
            {
                "format": SIDECAR_FORMAT,
                "schema_version": SIDECAR_SCHEMA_VERSION,
                "split": split,
                "episode_count": len(sidecar_entries),
                "raw_steps": sum(int(row["raw_steps"]) for row in sidecar_entries),
                "base_samples": sum(int(row["base_samples"]) for row in sidecar_entries),
                "episodes": sidecar_entries,
            },
        )


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _required_staging_bytes(
    base_root: Path, supplement_root: Path, plan: Mapping[str, object]
) -> int:
    selected = 0
    for row in plan["supplement_episode_mapping"]:
        for component in ("platoon_joint_bev", "riskentry_actor_sidecar"):
            selected += _tree_bytes(
                _episode_path(
                    supplement_root,
                    component,
                    str(row["source_split"]),
                    int(row["source_episode_index"]),
                )
            )
    return _tree_bytes(base_root) + selected


def build_staging(
    base_root: Path,
    supplement_root: Path,
    staging_root: Path,
    plan: Mapping[str, object],
) -> None:
    if staging_root.exists():
        raise RuleConditionedV2CompositionError(f"staging already exists: {staging_root}")
    staging_root.mkdir()
    storage = joint_sample_storage_contract("v2")
    base_component = staging_root / "platoon_joint_bev"
    sidecar_component = staging_root / "riskentry_actor_sidecar"
    for component in (base_component, sidecar_component):
        component.mkdir()
        (component / ".writer.lock").touch()
        for split in SPLIT_NAMES:
            (component / split / "episodes").mkdir(parents=True)
    (staging_root / ".bundle_writer.lock").touch()

    fingerprint = str(plan["base_dataset_fingerprint"])
    sidecar_fingerprint = str(plan["sidecar_dataset_fingerprint"])
    base_contract = _read_json(base_root / "platoon_joint_bev/dataset_contract.json")
    base_contract["dataset_fingerprint"] = fingerprint
    _atomic_json(base_component / "dataset_contract.json", base_contract)
    _atomic_json(
        sidecar_component / "dataset_contract.json",
        sidecar_dataset_contract(
            fingerprint,
            base_format=storage.storage_format,
            base_schema_version=storage.schema_version,
        ),
    )
    bundle_manifest = _read_json(base_root / "dataset_bundle_manifest.json")
    bundle_manifest["protocol_sha256"] = bundle_protocol_sha256(planner_version="v2")
    bundle_manifest["base_dataset_fingerprint"] = fingerprint
    bundle_manifest["sidecar_dataset_fingerprint"] = sidecar_fingerprint
    _atomic_json(staging_root / "dataset_bundle_manifest.json", bundle_manifest)

    base_entries = _manifest_entries(base_root, "platoon_joint_bev")
    sidecar_entries = _manifest_entries(base_root, "riskentry_actor_sidecar")
    inventory: list[dict[str, object]] = []
    base_metadata: dict[int, dict[str, object]] = {}
    sidecar_metadata: dict[int, dict[str, object]] = {}
    for index, (split, _) in sorted(sidecar_entries.items()):
        sidecar_metadata[index] = _copy_episode(
            source_root=base_root,
            source_id=SOURCE_ID_BASE,
            component="riskentry_actor_sidecar",
            source_split=split,
            source_index=index,
            target_root=staging_root,
            target_split=split,
            target_index=index,
            base_fingerprint=fingerprint,
            sidecar_fingerprint=sidecar_fingerprint,
            inventory=inventory,
        )
        if index in base_entries:
            base_metadata[index] = _copy_episode(
                source_root=base_root,
                source_id=SOURCE_ID_BASE,
                component="platoon_joint_bev",
                source_split=split,
                source_index=index,
                target_root=staging_root,
                target_split=split,
                target_index=index,
                base_fingerprint=fingerprint,
                sidecar_fingerprint=sidecar_fingerprint,
                inventory=inventory,
            )
    supplement_rows = {int(row["episode_index"]): row for row in _read_rows(supplement_root)}
    for row in plan["supplement_episode_mapping"]:
        source_index = int(row["source_episode_index"])
        source_split = str(row["source_split"])
        target_index = int(row["target_episode_index"])
        target_split = str(row["target_split"])
        base_metadata[target_index] = _copy_episode(
            source_root=supplement_root,
            source_id=SOURCE_ID_S5,
            component="platoon_joint_bev",
            source_split=source_split,
            source_index=source_index,
            target_root=staging_root,
            target_split=target_split,
            target_index=target_index,
            base_fingerprint=fingerprint,
            sidecar_fingerprint=sidecar_fingerprint,
            inventory=inventory,
        )
        sidecar_metadata[target_index] = _copy_episode(
            source_root=supplement_root,
            source_id=SOURCE_ID_S5,
            component="riskentry_actor_sidecar",
            source_split=source_split,
            source_index=source_index,
            target_root=staging_root,
            target_split=target_split,
            target_index=target_index,
            base_fingerprint=fingerprint,
            sidecar_fingerprint=sidecar_fingerprint,
            inventory=inventory,
        )
    _write_manifests(staging_root, base_metadata, sidecar_metadata)

    source_state = _read_json(base_root / "platoon_joint_bev/collection_state.json")
    expected = plan["expected_output"]
    next_index = int(expected["bundle_index_rows"])
    _atomic_json(
        base_component / "collection_state.json",
        {
            "schema_version": storage.schema_version,
            "attempted_episodes": next_index,
            "next_episode_index": next_index,
            "stored_episodes": int(expected["base_episodes"]),
            "rejected_episodes": next_index - int(expected["base_episodes"]),
            "rejection_reasons": dict(source_state.get("rejection_reasons", {})),
            "total_joint_samples": int(expected["base_joint_samples"]),
        },
    )

    rows = [dict(row) for row in _read_rows(base_root)]
    for mapping in plan["supplement_episode_mapping"]:
        source_row = dict(supplement_rows[int(mapping["source_episode_index"])])
        source_row["episode_index"] = int(mapping["target_episode_index"])
        source_row["split"] = str(mapping["target_split"])
        rows.append(source_row)
    rows.sort(key=lambda row: int(row["episode_index"]))
    if [int(row["episode_index"]) for row in rows] != list(range(next_index)):
        raise RuleConditionedV2CompositionError("constructed bundle index is not contiguous")
    (staging_root / "bundle_episode_index.jsonl").write_bytes(
        b"".join(_canonical_json(row) + b"\n" for row in rows)
    )
    _atomic_json(
        staging_root / "payload_inventory.json",
        {
            "format": INVENTORY_FORMAT,
            "entries": sorted(
                inventory,
                key=lambda row: (
                    str(row["source_id"]),
                    str(row["source_path"]),
                    str(row["target_path"]),
                ),
            ),
        },
    )
    _atomic_json(
        staging_root / "dataset_curation_manifest.json",
        {
            "format": COMPOSITION_FORMAT,
            "complete": True,
            "contract_payload": plan["contract_payload"],
            "base_dataset_fingerprint": fingerprint,
            "sidecar_dataset_fingerprint": sidecar_fingerprint,
            "payload_inventory_sha256": _file_sha256(
                staging_root / "payload_inventory.json"
            ),
            "formal_training_eligibility": False,
            "source_roots_preserved": True,
        },
    )


def verify_composed_bundle(
    root: Path | str, *, verify_payload_hashes: bool = True
) -> dict[str, object]:
    bundle = Path(root).expanduser().resolve()
    _validate_no_links(bundle)
    manifest = _read_json(bundle / "dataset_curation_manifest.json")
    contract = manifest.get("contract_payload")
    if (
        manifest.get("format") != COMPOSITION_FORMAT
        or manifest.get("complete") is not True
        or not isinstance(contract, Mapping)
    ):
        raise RuleConditionedV2CompositionError("composition manifest is incomplete")
    fingerprint = _payload_sha256(contract)
    storage = joint_sample_storage_contract("v2")
    sidecar_fingerprint = sidecar_dataset_fingerprint(
        fingerprint,
        base_format=storage.storage_format,
        base_schema_version=storage.schema_version,
    )
    if (
        manifest.get("base_dataset_fingerprint") != fingerprint
        or manifest.get("sidecar_dataset_fingerprint") != sidecar_fingerprint
    ):
        raise RuleConditionedV2CompositionError("composition fingerprint mismatch")
    strict = _strict_verify(bundle)
    expected = contract.get("expected_output")
    if not isinstance(expected, Mapping) or _source_statistics(bundle) != {
        key: expected[key]
        for key in (
            "base_episodes",
            "base_joint_samples",
            "sidecar_episodes",
            "sidecar_only_episodes",
            "sidecar_raw_steps",
            "bundle_index_rows",
            "split_episode_counts",
            "split_joint_samples",
        )
    }:
        raise RuleConditionedV2CompositionError("composition output counts mismatch")
    state = _read_json(bundle / "platoon_joint_bev/collection_state.json")
    if (
        int(state.get("next_episode_index", -1)) != int(expected["bundle_index_rows"])
        or int(state.get("attempted_episodes", -1))
        != int(expected["bundle_index_rows"])
        or int(state.get("stored_episodes", -1)) != int(expected["base_episodes"])
        or int(state.get("rejected_episodes", -1))
        != int(expected["rejected_episodes"])
        or state.get("rejection_reasons")
        != contract["base_source"]["rejection_reasons"]
        or int(state.get("total_joint_samples", -1)) != FINAL_TOTAL_SAMPLES
    ):
        raise RuleConditionedV2CompositionError("composition collection state mismatch")

    appended = 0
    identities = [
        (str(row["scenario_id"]), int(row["spawn_seed"]))
        for row in _read_rows(bundle)
    ]
    for _, (_, entry) in _manifest_entries(bundle, "platoon_joint_bev").items():
        attributes = entry.get("attributes")
        if not isinstance(attributes, Mapping):
            raise RuleConditionedV2CompositionError("composed attributes missing")
        provenance = attributes.get("curation_source")
        if isinstance(provenance, Mapping) and provenance.get("source_id") == SOURCE_ID_S5:
            _validate_s5_attributes(attributes)
            appended += 1
    if appended != APPEND_EPISODES or len(identities) != len(set(identities)):
        raise RuleConditionedV2CompositionError("S5 count or scenario/seed uniqueness failed")

    inventory_path = bundle / "payload_inventory.json"
    if _file_sha256(inventory_path) != manifest.get("payload_inventory_sha256"):
        raise RuleConditionedV2CompositionError("payload inventory hash mismatch")
    verified_files = 0
    if verify_payload_hashes:
        inventory = _read_json(inventory_path)
        entries = inventory.get("entries")
        if inventory.get("format") != INVENTORY_FORMAT or not isinstance(entries, list):
            raise RuleConditionedV2CompositionError("payload inventory format mismatch")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise RuleConditionedV2CompositionError("payload inventory entry invalid")
            target = bundle / str(entry["target_path"])
            if target.is_symlink() or _file_sha256(target) != entry.get("sha256"):
                raise RuleConditionedV2CompositionError(f"payload hash mismatch: {target}")
            verified_files += 1
    return {
        "format": COMPOSITION_FORMAT,
        "bundle_root": str(bundle),
        "base_dataset_fingerprint": fingerprint,
        "sidecar_dataset_fingerprint": sidecar_fingerprint,
        "bundle_index_sha256": _file_sha256(bundle / "bundle_episode_index.jsonl"),
        "base_joint_samples": FINAL_TOTAL_SAMPLES,
        "new_s5_release_episodes": appended,
        "aligned_samples": strict.get("aligned_samples"),
        "verified_payload_files": verified_files,
        "eligible_for_formal_training": False,
    }


def _recheck_bindings(
    base_root: Path, supplement_root: Path, plan: Mapping[str, object]
) -> None:
    contract = plan.get("contract_payload")
    if not isinstance(contract, Mapping):
        raise RuleConditionedV2CompositionError("composition contract missing")
    for root, key in ((base_root, "base_source"), (supplement_root, "supplement_source")):
        source = contract.get(key)
        if not isinstance(source, Mapping):
            raise RuleConditionedV2CompositionError("source binding missing")
        binding = _source_binding(root)
        if (
            binding["base_dataset_fingerprint"] != source["base_dataset_fingerprint"]
            or binding["bundle_index_sha256"] != source["bundle_index_sha256"]
        ):
            raise RuleConditionedV2CompositionError("source drifted after dry-run")
        if key == "supplement_source" and (
            _file_sha256(Path(str(source["config_path"])))
            != source["config_sha256"]
            or _file_sha256(Path(str(source["freeze_contract_path"])))
            != source["freeze_contract_sha256"]
        ):
            raise RuleConditionedV2CompositionError(
                "supplement config/freeze contract drifted after dry-run"
            )


def _staging_path(final_root: Path) -> Path:
    return final_root.with_name(f"{final_root.name}.rule_conditioned_v2_compose_staging")


def stage_composition(
    base_root: Path,
    supplement_root: Path,
    final_root: Path,
    plan: Mapping[str, object],
) -> dict[str, object]:
    final = final_root.expanduser().resolve()
    staging = _staging_path(final)
    if final.exists() or final.is_symlink() or staging.exists() or staging.is_symlink():
        raise RuleConditionedV2CompositionError("final or staging root already exists")
    if final.parent.resolve() != staging.parent.resolve() or not final.parent.is_dir():
        raise RuleConditionedV2CompositionError("final/staging parent is invalid")
    required = _required_staging_bytes(base_root, supplement_root, plan)
    if shutil.disk_usage(final.parent).free < required:
        raise RuleConditionedV2CompositionError("insufficient free space for physical staging")
    with _exclusive_source_locks((base_root, supplement_root)):
        _recheck_bindings(base_root, supplement_root, plan)
        build_staging(base_root, supplement_root, staging, plan)
        report = verify_composed_bundle(staging)
        _recheck_bindings(base_root, supplement_root, plan)
    return report


def install_staged_composition(
    base_root: Path,
    supplement_root: Path,
    final_root: Path,
    plan: Mapping[str, object],
) -> dict[str, object]:
    final = final_root.expanduser().resolve()
    staging = _staging_path(final)
    if final.exists() or staging.is_symlink() or not staging.is_dir():
        raise RuleConditionedV2CompositionError(
            "install requires an absent final root and real verified staging root"
        )
    with _exclusive_source_locks((base_root, supplement_root)):
        _recheck_bindings(base_root, supplement_root, plan)
        staged = verify_composed_bundle(staging)
        staged_manifest = _read_json(staging / "dataset_curation_manifest.json")
        if _payload_sha256(staged_manifest["contract_payload"]) != plan.get(
            "base_dataset_fingerprint"
        ):
            raise RuleConditionedV2CompositionError("staging does not match dry-run plan")
        os.replace(staging, final)
        try:
            installed = verify_composed_bundle(final)
        except Exception:
            os.replace(final, staging)
            raise
        if installed != {**staged, "bundle_root": str(final)}:
            os.replace(final, staging)
            raise RuleConditionedV2CompositionError(
                "staging/final verification reports differ"
            )
    return installed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-s5-config")
    prepare.add_argument("--base-root", type=Path, required=True)
    prepare.add_argument("--template", type=Path, required=True)
    prepare.add_argument("--output-config", type=Path, required=True)
    prepare.add_argument("--output-contract", type=Path, required=True)
    prepare.add_argument("--result-json", type=Path)

    compose = commands.add_parser("compose")
    compose.add_argument("--base-root", type=Path, required=True)
    compose.add_argument("--supplement-root", type=Path, required=True)
    compose.add_argument("--supplement-config", type=Path, required=True)
    compose.add_argument("--s5-contract", type=Path, required=True)
    compose.add_argument("--final-root", type=Path, required=True)
    compose.add_argument("--expected-base-fingerprint", required=True)
    compose.add_argument("--expected-base-index-sha256", required=True)
    compose.add_argument("--expected-supplement-fingerprint", required=True)
    compose.add_argument("--expected-supplement-index-sha256", required=True)
    modes = compose.add_mutually_exclusive_group()
    modes.add_argument("--stage-only", action="store_true")
    modes.add_argument(
        "--install-staged",
        "--install-to-new-root",
        dest="install_staged",
        action="store_true",
    )
    compose.add_argument("--result-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare-s5-config":
        result = prepare_s5_config(
            args.base_root, args.template, args.output_config, args.output_contract
        )
        result = {"mode": "prepared", **result}
    else:
        base = _validate_source_root(args.base_root)
        supplement = _validate_source_root(args.supplement_root)
        plan = inspect_sources(
            base,
            supplement,
            args.supplement_config,
            args.s5_contract,
            expected_base_fingerprint=args.expected_base_fingerprint,
            expected_base_index_sha256=args.expected_base_index_sha256,
            expected_supplement_fingerprint=args.expected_supplement_fingerprint,
            expected_supplement_index_sha256=args.expected_supplement_index_sha256,
        )
        if args.stage_only:
            result = {
                "mode": "staged",
                **stage_composition(base, supplement, args.final_root, plan),
            }
        elif args.install_staged:
            result = {
                "mode": "installed",
                **install_staged_composition(base, supplement, args.final_root, plan),
            }
        else:
            result = {"mode": "dry_run", **plan}
    if args.result_json is not None:
        _atomic_json(args.result_json.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuleConditionedV2CompositionError as exc:
        raise SystemExit(f"error: {exc}") from exc


__all__ = [
    "RuleConditionedV2CompositionError",
    "build_staging",
    "inspect_sources",
    "install_staged_composition",
    "prepare_s5_config",
    "stage_composition",
    "verify_composed_bundle",
]
