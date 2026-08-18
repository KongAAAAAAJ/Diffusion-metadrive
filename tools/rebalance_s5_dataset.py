"""Build and atomically install the audited S5-rebalanced dataset bundle."""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import fcntl
import hashlib
import json
import shutil
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

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
from expert_dataset.joint_risk_bundle_storage import BundleEpisodeResult
from expert_dataset.riskentry_sidecar_storage import (
    SIDECAR_FORMAT,
    SIDECAR_SCHEMA_VERSION,
    sidecar_dataset_contract,
    sidecar_dataset_fingerprint,
)
from expert_dataset.verify_joint_bev_dataset import verify_joint_bev_dataset
from expert_dataset.verify_joint_risk_bundle import verify_joint_risk_bundle
from expert_dataset.verify_riskentry_sidecar import verify_riskentry_sidecar_dataset
from scenarios.bev_round13_contract import (
    CANDIDATE_V3_CONTRACT_ID,
    CANDIDATE_V4_CONTRACT_ID,
    scenario_contract_for_id,
)


FORMAL_ROOT = Path(
    "/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/"
    "bev_joint_risk_candidate_v3_formal50k_v1"
)
SUPPLEMENT_ROOT = Path(
    "/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/"
    "bev_joint_risk_candidate_v4_s5_release20_v2"
)
EXPECTED_FORMAL_FINGERPRINT = (
    "4060c643aa15ece2936f98b641c78b6ff75309114ebc58f915d9ce3bb4bfa32c"
)
EXPECTED_SUPPLEMENT_FINGERPRINT = (
    "8e19b181f504bbabd96a93cad2259eba6348ea60f4b7aefa118108f26d8488e1"
)
EXPECTED_FORMAL_INDEX_SHA256 = (
    "5403bca52214e77a93867ca7c245605a4d04bc2f747321626eb22fc890ad2420"
)
EXPECTED_SUPPLEMENT_INDEX_SHA256 = (
    "b310fe8eb5a65ad41aa8c6ff8a3e076dbc4eb68b3b9566f37c6125e225a9c66f"
)
EXPECTED_TARGETED_PENDING_SHA256 = (
    "d2ea9d2276c1c928a9d8ed7e30291e7dad118f92b4d124477e6d42b5caae2216"
)
S5_SCENARIO = "S5_hard_brake_lead"
BRAKE_CATEGORY = "keep_emergency_braking"
RELEASE_CATEGORY = "temporary_formation_release_and_recovery"
KEPT_BRAKE_IDS = (0, 11, 16, 23, 77)
ORIGINAL_RELEASE_ID = 8
SUPPLEMENT_ID_MAP = {
    3: 362,
    4: 387,
    6: 363,
    7: 364,
    8: 366,
    9: 365,
    13: 367,
    15: 369,
    18: 370,
    20: 371,
    25: 372,
    26: 373,
    34: 374,
    38: 375,
    40: 377,
    49: 400,
    58: 378,
    59: 379,
}
EXPECTED_FINAL_SPLIT_EPISODES = {"train": 184, "val": 28, "test": 21}
EXPECTED_FINAL_SPLIT_SAMPLES = {"train": 35_220, "val": 5_310, "test": 4_030}
EXPECTED_FINAL_S5_SPLITS = {"train": 19, "val": 2, "test": 3}
EXPECTED_FINAL_RELEASE_SPLITS = {"train": 15, "val": 2, "test": 2}
EXPECTED_FINAL_BRAKE_SPLITS = {"train": 4, "val": 0, "test": 1}
EXPECTED_FINAL_BASE_EPISODES = 233
EXPECTED_FINAL_BASE_SAMPLES = 44_560
EXPECTED_FINAL_SIDECAR_EPISODES = 333
EXPECTED_FINAL_RAW_STEPS = 97_682
EXPECTED_FINAL_INDEX_ROWS = 401
CURATION_FORMAT = "s5-behavior-rebalanced-curated-bundle-v1"
INVENTORY_FORMAT = "s5-behavior-rebalanced-payload-inventory-v1"


class S5RebalanceError(RuntimeError):
    """Raised when source evidence or a destructive safety boundary fails."""


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _payload_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise S5RebalanceError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise S5RebalanceError(f"JSON root must be an object: {path}")
    return payload


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    encoded = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False
    ).encode("utf-8") + b"\n"
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_rows(root: Path) -> list[dict[str, object]]:
    path = root / "bundle_episode_index.jsonl"
    rows: list[dict[str, object]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise S5RebalanceError(f"invalid bundle row: {path}")
            BundleEpisodeResult.from_mapping(payload)
            rows.append(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise S5RebalanceError(f"invalid bundle index: {path}") from exc
    if [int(row["episode_index"]) for row in rows] != list(range(len(rows))):
        raise S5RebalanceError(f"bundle index is not contiguous: {path}")
    return rows


def _manifest_entries(root: Path, component: str) -> dict[int, tuple[str, dict[str, object]]]:
    result: dict[int, tuple[str, dict[str, object]]] = {}
    for split in SPLIT_NAMES:
        manifest = _read_json(root / component / split / "manifest.json")
        entries = manifest.get("episodes")
        if not isinstance(entries, list):
            raise S5RebalanceError(f"invalid manifest episode list: {component}/{split}")
        for raw_entry in entries:
            if not isinstance(raw_entry, dict):
                raise S5RebalanceError(f"invalid manifest entry: {component}/{split}")
            episode_index = int(raw_entry.get("episode_index", -1))
            if episode_index < 0 or episode_index in result:
                raise S5RebalanceError("duplicate or invalid global episode index")
            result[episode_index] = (split, raw_entry)
    return result


def _behavior_category(attributes: Mapping[str, object]) -> str:
    formal = attributes.get("formal_coverage")
    targeted = attributes.get("targeted_supplement_evidence")
    evidence = formal if isinstance(formal, Mapping) else targeted
    if not isinstance(evidence, Mapping):
        return "unknown"
    return str(evidence.get("behavior_category", "unknown"))


def _validate_targeted_evidence(attributes: Mapping[str, object]) -> None:
    evidence = attributes.get("targeted_supplement_evidence")
    if not isinstance(evidence, Mapping):
        raise S5RebalanceError("supplement episode lacks targeted evidence")
    if evidence.get("behavior_category") != RELEASE_CATEGORY:
        raise S5RebalanceError("supplement episode behavior category mismatch")
    if evidence.get("platoon_safety_events") != []:
        raise S5RebalanceError("supplement episode has a platoon safety event")
    if evidence.get("target_background_condition_realized") is not True:
        raise S5RebalanceError("supplement target background was not realized")
    runs = evidence.get("lateral_mode_runs_by_role")
    directions = evidence.get("lateral_run_directions_by_role")
    ranges = evidence.get("lateral_range_m_by_role")
    returns = evidence.get("return_error_m_by_role")
    if runs != [2, 2, 2] or not isinstance(directions, list) or len(directions) != 3:
        raise S5RebalanceError("supplement lateral behavior run contract failed")
    for row in directions:
        if not isinstance(row, list) or len(row) != 2 or row[0] == row[1]:
            raise S5RebalanceError("supplement lateral directions are not opposite")
    if (
        not isinstance(ranges, list)
        or len(ranges) != 3
        or any(float(value) < 2.5 for value in ranges)
        or not isinstance(returns, list)
        or len(returns) != 3
        or any(float(value) > 0.5 for value in returns)
    ):
        raise S5RebalanceError("supplement physical displacement contract failed")


def _validate_no_links(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise S5RebalanceError(f"source must be a real directory: {root}")
    for current, directories, files in os.walk(root, followlinks=False):
        for name in (*directories, *files):
            path = Path(current) / name
            if path.is_symlink():
                raise S5RebalanceError(f"symbolic links are forbidden: {path}")


def inspect_sources(
    formal_root: Path,
    supplement_root: Path,
    *,
    expected_formal_fingerprint: str,
    expected_supplement_fingerprint: str,
) -> dict[str, object]:
    formal_root = formal_root.resolve()
    supplement_root = supplement_root.resolve()
    if formal_root == supplement_root:
        raise S5RebalanceError("formal and supplement roots must differ")
    _validate_no_links(formal_root)
    _validate_no_links(supplement_root)
    formal_contract = _read_json(formal_root / "platoon_joint_bev/dataset_contract.json")
    supplement_contract = _read_json(
        supplement_root / "platoon_joint_bev/dataset_contract.json"
    )
    if formal_contract.get("dataset_fingerprint") != expected_formal_fingerprint:
        raise S5RebalanceError("formal base fingerprint drift")
    if supplement_contract.get("dataset_fingerprint") != expected_supplement_fingerprint:
        raise S5RebalanceError("supplement base fingerprint drift")
    formal_index_sha = _file_sha256(formal_root / "bundle_episode_index.jsonl")
    supplement_index_sha = _file_sha256(supplement_root / "bundle_episode_index.jsonl")
    if formal_index_sha != EXPECTED_FORMAL_INDEX_SHA256:
        raise S5RebalanceError("formal bundle index hash drift")
    if supplement_index_sha != EXPECTED_SUPPLEMENT_INDEX_SHA256:
        raise S5RebalanceError("supplement bundle index hash drift")
    if (formal_root / ".bundle_episode_pending.json").exists() or (
        formal_root / ".targeted_batch_pending.json"
    ).exists():
        raise S5RebalanceError("formal source has an unfinished transaction")
    if (supplement_root / ".bundle_episode_pending.json").exists():
        raise S5RebalanceError("supplement source has a bundle transaction pending")
    pending_path = supplement_root / ".targeted_batch_pending.json"
    if not pending_path.is_file() or _file_sha256(pending_path) != EXPECTED_TARGETED_PENDING_SHA256:
        raise S5RebalanceError("supplement targeted pending evidence drift")
    pending = _read_json(pending_path)
    if pending.get("entries") != [
        {
            "episode_index": 376,
            "local_route": "R1_entry_straight",
            "scenario_id": S5_SCENARIO,
            "spawn_seed": 340309086,
            "split": "test",
        }
    ]:
        raise S5RebalanceError("unexpected supplement pending entry")

    formal_base = _manifest_entries(formal_root, "platoon_joint_bev")
    formal_sidecar = _manifest_entries(formal_root, "riskentry_actor_sidecar")
    supplement_base = _manifest_entries(supplement_root, "platoon_joint_bev")
    supplement_sidecar = _manifest_entries(supplement_root, "riskentry_actor_sidecar")
    formal_rows = _read_rows(formal_root)
    supplement_rows = _read_rows(supplement_root)
    if len(formal_base) != 262 or len(formal_rows) != 362 or len(formal_sidecar) != 362:
        raise S5RebalanceError("formal source counts drifted from 262/362/362")
    if len(supplement_base) != 18 or len(supplement_rows) != 376:
        raise S5RebalanceError("supplement source counts drifted from 18/376")
    if not set(supplement_base).issubset(supplement_sidecar):
        raise S5RebalanceError("a supplement base episode lacks sidecar data")

    braking: set[int] = set()
    release: set[int] = set()
    for episode_index, (_, entry) in formal_base.items():
        attributes = entry.get("attributes")
        if not isinstance(attributes, Mapping) or attributes.get("scenario_id") != S5_SCENARIO:
            continue
        category = _behavior_category(attributes)
        if category == BRAKE_CATEGORY:
            braking.add(episode_index)
        elif category == RELEASE_CATEGORY:
            release.add(episode_index)
    if len(braking) != 52 or release != {ORIGINAL_RELEASE_ID}:
        raise S5RebalanceError("formal S5 behavior counts drifted from 52:1")
    if not set(KEPT_BRAKE_IDS).issubset(braking):
        raise S5RebalanceError("a required retained brake episode is missing")
    removed = sorted(braking - set(KEPT_BRAKE_IDS))
    removed_samples = sum(int(formal_base[index][1]["joint_samples"]) for index in removed)
    if len(removed) != 47 or removed_samples != 8_860 or 357 not in removed:
        raise S5RebalanceError("removed brake selection does not match the frozen plan")
    if set(supplement_base) != set(SUPPLEMENT_ID_MAP):
        raise S5RebalanceError("supplement committed episode IDs drifted")
    split_assignment = formal_contract.get("split_assignment")
    if split_assignment != {
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "seed": 17,
    }:
        raise S5RebalanceError("formal split assignment drifted")
    assigner = EpisodeSplitAssigner(EpisodeSplitConfig(**split_assignment))
    mapping: list[dict[str, object]] = []
    for source_index, target_index in sorted(SUPPLEMENT_ID_MAP.items()):
        source_split, entry = supplement_base[source_index]
        attributes = entry.get("attributes")
        if not isinstance(attributes, Mapping):
            raise S5RebalanceError("supplement attributes missing")
        _validate_targeted_evidence(attributes)
        target_split = assigner.split_for_episode(target_index)
        mapping.append(
            {
                "source_episode_index": source_index,
                "source_split": source_split,
                "target_episode_index": target_index,
                "target_split": target_split,
                "spawn_seed": int(attributes["spawn_seed"]),
            }
        )
    if Counter(str(item["target_split"]) for item in mapping) != Counter(
        {"train": 14, "val": 2, "test": 2}
    ):
        raise S5RebalanceError("supplement remap does not produce 14/2/2")

    retained_formal = sorted(set(formal_base) - set(removed))
    final_scenario_seeds = [
        (
            str(formal_base[index][1]["attributes"]["scenario_id"]),
            int(formal_base[index][1]["attributes"]["spawn_seed"]),
        )
        for index in retained_formal
    ] + [(S5_SCENARIO, int(item["spawn_seed"])) for item in mapping]
    duplicates = sorted(
        key for key, count in Counter(final_scenario_seeds).items() if count > 1
    )
    if duplicates:
        raise S5RebalanceError(
            "final base episodes have duplicate (scenario_id, spawn_seed) pairs: "
            f"{duplicates}"
        )

    scenario_payload = {
        "format": "curated-mixed-scenario-contract-v1",
        "source_contracts": {
            CANDIDATE_V3_CONTRACT_ID: scenario_contract_for_id(
                CANDIDATE_V3_CONTRACT_ID
            )["sha256"],
            CANDIDATE_V4_CONTRACT_ID: scenario_contract_for_id(
                CANDIDATE_V4_CONTRACT_ID
            )["sha256"],
        },
        "policy": "preserve_per_episode_source_contract_without_relabeling",
    }
    curation_contract_sha256 = _payload_sha256(scenario_payload)
    contract_payload = {
        "format": CURATION_FORMAT,
        "formal_source": {
            "base_dataset_fingerprint": expected_formal_fingerprint,
            "bundle_index_sha256": formal_index_sha,
        },
        "supplement_source": {
            "base_dataset_fingerprint": expected_supplement_fingerprint,
            "bundle_index_sha256": supplement_index_sha,
            "targeted_pending_sha256": EXPECTED_TARGETED_PENDING_SHA256,
        },
        "kept_brake_episode_indices": list(KEPT_BRAKE_IDS),
        "kept_original_release_episode_index": ORIGINAL_RELEASE_ID,
        "removed_brake_episode_indices": removed,
        "supplement_episode_mapping": mapping,
        "split_assignment": split_assignment,
        "scenario_contract": scenario_payload,
        "expected_output": {
            "base_episodes": EXPECTED_FINAL_BASE_EPISODES,
            "base_joint_samples": EXPECTED_FINAL_BASE_SAMPLES,
            "sidecar_episodes": EXPECTED_FINAL_SIDECAR_EPISODES,
            "sidecar_raw_steps": EXPECTED_FINAL_RAW_STEPS,
            "bundle_index_rows": EXPECTED_FINAL_INDEX_ROWS,
            "split_episode_counts": EXPECTED_FINAL_SPLIT_EPISODES,
            "split_joint_samples": EXPECTED_FINAL_SPLIT_SAMPLES,
            "s5_behavior_episode_counts": {
                BRAKE_CATEGORY: 5,
                RELEASE_CATEGORY: 19,
            },
        },
    }
    base_fingerprint = _payload_sha256(contract_payload)
    return {
        "format": CURATION_FORMAT,
        "formal_root": str(formal_root),
        "supplement_root": str(supplement_root),
        "contract_payload": contract_payload,
        "base_dataset_fingerprint": base_fingerprint,
        "sidecar_dataset_fingerprint": sidecar_dataset_fingerprint(base_fingerprint),
        "curation_contract_sha256": curation_contract_sha256,
        "retained_formal_base_episode_indices": retained_formal,
        "removed_brake_episode_indices": removed,
        "supplement_episode_mapping": mapping,
        "supplement_pending_episode_excluded": 376,
        "eligible_for_formal_training": False,
    }


def _episode_path(root: Path, component: str, split: str, episode_index: int) -> Path:
    return root / component / split / "episodes" / f"episode_{episode_index:08d}"


def _curation_source(
    source_id: str,
    source_root: Path,
    source_split: str,
    source_index: int,
) -> dict[str, object]:
    base_path = _episode_path(
        source_root, "platoon_joint_bev", source_split, source_index
    )
    sidecar_path = _episode_path(
        source_root, "riskentry_actor_sidecar", source_split, source_index
    )
    return {
        "source_id": source_id,
        "source_episode_index": source_index,
        "source_split": source_split,
        "source_base_metadata_sha256": (
            _file_sha256(base_path / "episode.json") if base_path.is_dir() else None
        ),
        "source_sidecar_metadata_sha256": _file_sha256(sidecar_path / "episode.json"),
    }


def _copy_payload_file(
    source: Path,
    target: Path,
    *,
    inventory: list[dict[str, object]],
    source_id: str,
    source_relative: str,
    target_relative: str,
) -> None:
    if source.is_symlink() or not source.is_file():
        raise S5RebalanceError(f"payload source is not a regular file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open("rb") as input_stream, target.open("xb") as output_stream:
        for chunk in iter(lambda: input_stream.read(1024 * 1024), b""):
            digest.update(chunk)
            output_stream.write(chunk)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    shutil.copystat(source, target, follow_symlinks=False)
    inventory.append(
        {
            "action": "copied",
            "source_id": source_id,
            "source_path": source_relative,
            "target_path": target_relative,
            "bytes": source.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    )


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
        raise S5RebalanceError(f"duplicate target episode path: {target}")
    target.mkdir(parents=True)
    for path in sorted(source.iterdir()):
        if path.name == "episode.json":
            continue
        if path.suffix != ".npy":
            raise S5RebalanceError(f"unexpected episode payload: {path}")
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
            raise S5RebalanceError("base episode attributes missing")
        attributes["sidecar_dataset_fingerprint"] = sidecar_fingerprint
        attributes["curation_source"] = provenance
    else:
        metadata["base_dataset_fingerprint"] = base_fingerprint
        scenario_parameters = metadata.get("scenario_parameters")
        if not isinstance(scenario_parameters, dict):
            raise S5RebalanceError("sidecar scenario parameters missing")
        scenario_parameters["curation_source"] = provenance
    _atomic_json(target / "episode.json", metadata)
    return metadata


def _record_removed_inventory(
    formal_root: Path,
    removed_ids: Sequence[int],
    formal_base: Mapping[int, tuple[str, dict[str, object]]],
    inventory: list[dict[str, object]],
) -> None:
    for episode_index in removed_ids:
        split = formal_base[episode_index][0]
        for component in ("platoon_joint_bev", "riskentry_actor_sidecar"):
            episode = _episode_path(formal_root, component, split, episode_index)
            for path in sorted(episode.iterdir()):
                inventory.append(
                    {
                        "action": "removed",
                        "source_id": "formal50k_pre_rebalance",
                        "source_path": str(path.relative_to(formal_root)),
                        "bytes": path.stat().st_size,
                        "sha256": _file_sha256(path),
                    }
                )


def _write_manifests(
    root: Path,
    base_metadata: Mapping[int, dict[str, object]],
    sidecar_metadata: Mapping[int, dict[str, object]],
) -> None:
    for split in SPLIT_NAMES:
        base_entries = []
        sidecar_entries = []
        for episode_index, metadata in sorted(base_metadata.items()):
            if metadata["split"] != split:
                continue
            base_entries.append(
                {
                    "episode_index": episode_index,
                    "directory": f"episode_{episode_index:08d}",
                    "joint_samples": int(metadata["joint_samples"]),
                    "attributes": metadata["attributes"],
                }
            )
        for episode_index, metadata in sorted(sidecar_metadata.items()):
            if metadata["split"] != split:
                continue
            path = _episode_path(root, "riskentry_actor_sidecar", split, episode_index)
            sidecar_entries.append(
                {
                    "episode_index": episode_index,
                    "directory": f"episode_{episode_index:08d}",
                    "raw_steps": int(np.load(path / "step_index.npy", mmap_mode="r").shape[0]),
                    "actor_count": int(
                        np.load(path / "actor_state.npy", mmap_mode="r").shape[1]
                    ),
                    "base_samples": int(
                        np.load(
                            path / "base_sample_step_index.npy", mmap_mode="r"
                        ).shape[0]
                    ),
                    "outcome": str(metadata["retention"]["outcome"]),
                }
            )
        _atomic_json(
            root / "platoon_joint_bev" / split / "manifest.json",
            {
                "schema_version": STORAGE_SCHEMA_VERSION,
                "format": STORAGE_FORMAT,
                "split": split,
                "episode_count": len(base_entries),
                "joint_samples": sum(int(item["joint_samples"]) for item in base_entries),
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
                "raw_steps": sum(int(item["raw_steps"]) for item in sidecar_entries),
                "base_samples": sum(int(item["base_samples"]) for item in sidecar_entries),
                "episodes": sidecar_entries,
            },
        )


def build_staging(
    formal_root: Path, supplement_root: Path, staging_root: Path, plan: Mapping[str, object]
) -> None:
    if staging_root.exists():
        raise S5RebalanceError(f"staging root already exists: {staging_root}")
    if staging_root.parent.resolve() != formal_root.parent.resolve():
        raise S5RebalanceError("staging must be a sibling of the formal root")
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
    base_contract = _read_json(formal_root / "platoon_joint_bev/dataset_contract.json")
    base_contract["dataset_fingerprint"] = base_fingerprint
    _atomic_json(base_root / "dataset_contract.json", base_contract)
    _atomic_json(
        sidecar_root / "dataset_contract.json",
        sidecar_dataset_contract(base_fingerprint),
    )
    bundle_manifest = _read_json(formal_root / "dataset_bundle_manifest.json")
    bundle_manifest["base_dataset_fingerprint"] = base_fingerprint
    bundle_manifest["sidecar_dataset_fingerprint"] = sidecar_fingerprint
    bundle_manifest["scenario_contract_sha256"] = plan["curation_contract_sha256"]
    _atomic_json(staging_root / "dataset_bundle_manifest.json", bundle_manifest)

    formal_base = _manifest_entries(formal_root, "platoon_joint_bev")
    formal_sidecar = _manifest_entries(formal_root, "riskentry_actor_sidecar")
    supplement_base = _manifest_entries(supplement_root, "platoon_joint_bev")
    inventory: list[dict[str, object]] = []
    _record_removed_inventory(
        formal_root,
        list(plan["removed_brake_episode_indices"]),
        formal_base,
        inventory,
    )
    base_metadata: dict[int, dict[str, object]] = {}
    sidecar_metadata: dict[int, dict[str, object]] = {}
    removed = set(int(value) for value in plan["removed_brake_episode_indices"])
    for episode_index, (split, _) in sorted(formal_sidecar.items()):
        if episode_index in removed:
            continue
        sidecar_metadata[episode_index] = _copy_episode(
            source_root=formal_root,
            source_id="formal50k_pre_rebalance",
            component="riskentry_actor_sidecar",
            source_split=split,
            source_index=episode_index,
            target_root=staging_root,
            target_split=split,
            target_index=episode_index,
            base_fingerprint=base_fingerprint,
            sidecar_fingerprint=sidecar_fingerprint,
            inventory=inventory,
        )
        if episode_index in formal_base:
            base_metadata[episode_index] = _copy_episode(
                source_root=formal_root,
                source_id="formal50k_pre_rebalance",
                component="platoon_joint_bev",
                source_split=split,
                source_index=episode_index,
                target_root=staging_root,
                target_split=split,
                target_index=episode_index,
                base_fingerprint=base_fingerprint,
                sidecar_fingerprint=sidecar_fingerprint,
                inventory=inventory,
            )
    for mapping in plan["supplement_episode_mapping"]:
        source_index = int(mapping["source_episode_index"])
        source_split = str(mapping["source_split"])
        target_index = int(mapping["target_episode_index"])
        target_split = str(mapping["target_split"])
        base_metadata[target_index] = _copy_episode(
            source_root=supplement_root,
            source_id="candidate_v4_s5_release18",
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
            source_id="candidate_v4_s5_release18",
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

    source_state = _read_json(formal_root / "platoon_joint_bev/collection_state.json")
    rejection_reasons = dict(source_state.get("rejection_reasons", {}))
    rejection_reasons["curation_removed_s5_emergency_braking"] = 47
    rejection_reasons["curation_index_padding"] = 21
    _atomic_json(
        base_root / "collection_state.json",
        {
            "schema_version": STORAGE_SCHEMA_VERSION,
            "attempted_episodes": EXPECTED_FINAL_INDEX_ROWS,
            "next_episode_index": EXPECTED_FINAL_INDEX_ROWS,
            "stored_episodes": EXPECTED_FINAL_BASE_EPISODES,
            "rejected_episodes": 168,
            "rejection_reasons": rejection_reasons,
            "total_joint_samples": EXPECTED_FINAL_BASE_SAMPLES,
        },
    )

    formal_rows = _read_rows(formal_root)
    supplement_rows = _read_rows(supplement_root)
    supplement_by_target = {
        int(item["target_episode_index"]): item
        for item in plan["supplement_episode_mapping"]
    }
    rows: list[dict[str, object]] = []
    for episode_index, source_row in enumerate(formal_rows):
        if episode_index not in removed:
            rows.append(dict(source_row))
            continue
        rows.append(
            {
                "episode_index": episode_index,
                "split": source_row["split"],
                "scenario_id": source_row["scenario_id"],
                "local_route": source_row["local_route"],
                "spawn_seed": source_row["spawn_seed"],
                "base_status": "rejected",
                "base_rejection_reason": "curation_removed_s5_emergency_braking",
                "sidecar_status": "rejected",
                "sidecar_rejection_reason": "curation_removed_s5_emergency_braking",
                "raw_steps": 0,
                "base_samples": 0,
                "outcome": "curation_removed",
            }
        )
    assigner = EpisodeSplitAssigner(EpisodeSplitConfig(seed=17))
    for episode_index in range(len(formal_rows), EXPECTED_FINAL_INDEX_ROWS):
        mapping = supplement_by_target.get(episode_index)
        if mapping is None:
            rows.append(
                {
                    "episode_index": episode_index,
                    "split": assigner.split_for_episode(episode_index),
                    "scenario_id": S5_SCENARIO,
                    "local_route": "R1_entry_straight",
                    "spawn_seed": 2_147_000_000 + episode_index,
                    "base_status": "rejected",
                    "base_rejection_reason": "curation_index_padding",
                    "sidecar_status": "rejected",
                    "sidecar_rejection_reason": "curation_index_padding",
                    "raw_steps": 0,
                    "base_samples": 0,
                    "outcome": "curation_index_padding",
                }
            )
            continue
        source_row = supplement_rows[int(mapping["source_episode_index"])]
        row = dict(source_row)
        row["episode_index"] = episode_index
        row["split"] = mapping["target_split"]
        rows.append(row)
    if len(rows) != EXPECTED_FINAL_INDEX_ROWS:
        raise S5RebalanceError("constructed bundle index row count mismatch")
    encoded_rows = b"".join(_canonical_json(row) + b"\n" for row in rows)
    (staging_root / "bundle_episode_index.jsonl").write_bytes(encoded_rows)

    inventory_payload = {
        "format": INVENTORY_FORMAT,
        "entries": sorted(
            inventory,
            key=lambda item: (
                str(item["action"]),
                str(item["source_id"]),
                str(item["source_path"]),
            ),
        ),
    }
    _atomic_json(staging_root / "payload_inventory.json", inventory_payload)
    inventory_sha = _file_sha256(staging_root / "payload_inventory.json")
    _atomic_json(
        staging_root / "dataset_curation_manifest.json",
        {
            "format": CURATION_FORMAT,
            "complete": True,
            "contract_payload": plan["contract_payload"],
            "base_dataset_fingerprint": base_fingerprint,
            "sidecar_dataset_fingerprint": sidecar_fingerprint,
            "curation_contract_sha256": plan["curation_contract_sha256"],
            "payload_inventory_sha256": inventory_sha,
            "formal_source_path_before_in_place_replacement": str(formal_root),
            "supplement_source_path": str(supplement_root),
            "formal_training_eligibility": False,
            "original_payload_deleted_after_acceptance": False,
        },
    )


def verify_curated_bundle(root: Path | str, *, verify_payload_hashes: bool = True) -> dict[str, object]:
    root = Path(root).resolve()
    manifest = _read_json(root / "dataset_curation_manifest.json")
    if manifest.get("format") != CURATION_FORMAT or manifest.get("complete") is not True:
        raise S5RebalanceError("curation manifest is incomplete")
    contract_payload = manifest.get("contract_payload")
    if not isinstance(contract_payload, Mapping):
        raise S5RebalanceError("curation contract payload missing")
    base_fingerprint = _payload_sha256(contract_payload)
    if base_fingerprint != manifest.get("base_dataset_fingerprint"):
        raise S5RebalanceError("curation base fingerprint mismatch")
    sidecar_fingerprint = sidecar_dataset_fingerprint(base_fingerprint)
    if sidecar_fingerprint != manifest.get("sidecar_dataset_fingerprint"):
        raise S5RebalanceError("curation sidecar fingerprint mismatch")
    scenario_payload = contract_payload.get("scenario_contract")
    if _payload_sha256(scenario_payload) != manifest.get("curation_contract_sha256"):
        raise S5RebalanceError("curation scenario contract hash mismatch")
    source_contracts = scenario_payload.get("source_contracts")
    if not isinstance(source_contracts, Mapping):
        raise S5RebalanceError("curation source contracts missing")
    allowed_hashes = sorted(str(value) for value in source_contracts.values())
    base_report = verify_joint_bev_dataset(root / "platoon_joint_bev")
    sidecar_report = verify_riskentry_sidecar_dataset(
        root / "riskentry_actor_sidecar",
        allowed_scenario_contract_sha256s=allowed_hashes,
    )
    bundle_manifest = _read_json(root / "dataset_bundle_manifest.json")
    if set(bundle_manifest) != {
        "format",
        "schema_version",
        "protocol_sha256",
        "base_directory",
        "sidecar_directory",
        "base_dataset_fingerprint",
        "sidecar_dataset_fingerprint",
        "scenario_contract_sha256",
        "decision_dt_s",
        "split_seed",
    }:
        raise S5RebalanceError("curated bundle manifest fields mismatch")
    if (
        bundle_manifest["format"] != BUNDLE_FORMAT
        or bundle_manifest["schema_version"] != BUNDLE_SCHEMA_VERSION
        or bundle_manifest["protocol_sha256"] != bundle_protocol_sha256()
        or bundle_manifest["base_dataset_fingerprint"] != base_fingerprint
        or bundle_manifest["sidecar_dataset_fingerprint"] != sidecar_fingerprint
        or bundle_manifest["scenario_contract_sha256"]
        != manifest["curation_contract_sha256"]
        or float(bundle_manifest["decision_dt_s"]) != 0.1
        or int(bundle_manifest["split_seed"]) != 17
    ):
        raise S5RebalanceError("curated bundle protocol binding mismatch")
    rows = _read_rows(root)
    if len(rows) != EXPECTED_FINAL_INDEX_ROWS:
        raise S5RebalanceError("curated bundle index must contain 401 rows")
    base_entries = _manifest_entries(root, "platoon_joint_bev")
    sidecar_entries = _manifest_entries(root, "riskentry_actor_sidecar")
    if len(base_entries) != EXPECTED_FINAL_BASE_EPISODES:
        raise S5RebalanceError("curated base episode count mismatch")
    if len(sidecar_entries) != EXPECTED_FINAL_SIDECAR_EPISODES:
        raise S5RebalanceError("curated sidecar episode count mismatch")
    sidecar_only = 0
    for row in rows:
        episode_index = int(row["episode_index"])
        split = str(row["split"])
        has_base = episode_index in base_entries
        has_sidecar = episode_index in sidecar_entries
        if has_base != (row["base_status"] == "committed"):
            raise S5RebalanceError("curated bundle/base status mismatch")
        if has_sidecar != (row["sidecar_status"] == "committed"):
            raise S5RebalanceError("curated bundle/sidecar status mismatch")
        if has_base:
            base_metadata = _read_json(
                _episode_path(root, "platoon_joint_bev", split, episode_index)
                / "episode.json"
            )
            sidecar_metadata = _read_json(
                _episode_path(root, "riskentry_actor_sidecar", split, episode_index)
                / "episode.json"
            )
            attributes = base_metadata.get("attributes")
            parameters = sidecar_metadata.get("scenario_parameters")
            if not isinstance(attributes, Mapping) or not isinstance(parameters, Mapping):
                raise S5RebalanceError("curated episode provenance metadata missing")
            if attributes.get("sidecar_dataset_fingerprint") != sidecar_fingerprint:
                raise S5RebalanceError("curated base/sidecar fingerprint binding mismatch")
            if attributes.get("scenario_contract_sha256") not in allowed_hashes:
                raise S5RebalanceError("curated base uses an undeclared scenario contract")
            if parameters.get("scenario_contract_sha256") != attributes.get(
                "scenario_contract_sha256"
            ):
                raise S5RebalanceError("base/sidecar source scenario contract mismatch")
            if not isinstance(attributes.get("curation_source"), Mapping) or not isinstance(
                parameters.get("curation_source"), Mapping
            ):
                raise S5RebalanceError("curation source provenance is missing")
        elif has_sidecar:
            sidecar_only += 1
    if sidecar_only != 100:
        raise S5RebalanceError("curated sidecar-only count must be 100")
    state = _read_json(root / "platoon_joint_bev/collection_state.json")
    if (
        state.get("next_episode_index") != EXPECTED_FINAL_INDEX_ROWS
        or state.get("stored_episodes") != EXPECTED_FINAL_BASE_EPISODES
        or state.get("rejected_episodes") != 168
        or state.get("total_joint_samples") != EXPECTED_FINAL_BASE_SAMPLES
    ):
        raise S5RebalanceError("curated collection state mismatch")

    split_episodes: Counter[str] = Counter()
    split_samples: Counter[str] = Counter()
    s5_splits: Counter[str] = Counter()
    behavior_splits: dict[str, Counter[str]] = {
        BRAKE_CATEGORY: Counter(),
        RELEASE_CATEGORY: Counter(),
    }
    scenario_seeds: list[tuple[str, int]] = []
    targeted_count = 0
    for episode_index, (split, entry) in base_entries.items():
        attributes = entry.get("attributes")
        if not isinstance(attributes, Mapping):
            raise S5RebalanceError("curated base attributes missing")
        split_episodes[split] += 1
        split_samples[split] += int(entry["joint_samples"])
        scenario_seeds.append(
            (str(attributes["scenario_id"]), int(attributes["spawn_seed"]))
        )
        if attributes.get("scenario_id") != S5_SCENARIO:
            continue
        s5_splits[split] += 1
        category = _behavior_category(attributes)
        if category not in behavior_splits:
            raise S5RebalanceError("unexpected curated S5 behavior category")
        behavior_splits[category][split] += 1
        if isinstance(attributes.get("targeted_supplement_evidence"), Mapping):
            _validate_targeted_evidence(attributes)
            targeted_count += 1
    if dict(split_episodes) != EXPECTED_FINAL_SPLIT_EPISODES:
        raise S5RebalanceError("curated split episode counts mismatch")
    if dict(split_samples) != EXPECTED_FINAL_SPLIT_SAMPLES:
        raise S5RebalanceError("curated split sample counts mismatch")
    if dict(s5_splits) != EXPECTED_FINAL_S5_SPLITS:
        raise S5RebalanceError("curated S5 split counts mismatch")
    if {name: int(behavior_splits[BRAKE_CATEGORY][name]) for name in SPLIT_NAMES} != EXPECTED_FINAL_BRAKE_SPLITS:
        raise S5RebalanceError("curated brake split counts mismatch")
    if {name: int(behavior_splits[RELEASE_CATEGORY][name]) for name in SPLIT_NAMES} != EXPECTED_FINAL_RELEASE_SPLITS:
        raise S5RebalanceError("curated release split counts mismatch")
    if targeted_count != 18 or len(scenario_seeds) != len(set(scenario_seeds)):
        raise S5RebalanceError(
            "targeted count or (scenario_id, spawn_seed) uniqueness failed"
        )
    for pending_name in (".bundle_episode_pending.json", ".targeted_batch_pending.json"):
        if (root / pending_name).exists():
            raise S5RebalanceError(f"curated root contains pending state: {pending_name}")
    if int(base_report["episodes"]) != EXPECTED_FINAL_BASE_EPISODES or int(
        base_report["joint_samples"]
    ) != EXPECTED_FINAL_BASE_SAMPLES:
        raise S5RebalanceError("full base verification totals mismatch")
    if int(sidecar_report["episodes"]) != EXPECTED_FINAL_SIDECAR_EPISODES or int(
        sidecar_report["raw_steps"]
    ) != EXPECTED_FINAL_RAW_STEPS:
        raise S5RebalanceError("full sidecar verification totals mismatch")
    inventory_path = root / "payload_inventory.json"
    if _file_sha256(inventory_path) != manifest.get("payload_inventory_sha256"):
        raise S5RebalanceError("payload inventory hash mismatch")
    verified_payload_files = 0
    if verify_payload_hashes:
        inventory = _read_json(inventory_path)
        entries = inventory.get("entries")
        if inventory.get("format") != INVENTORY_FORMAT or not isinstance(entries, list):
            raise S5RebalanceError("payload inventory format mismatch")
        for entry in entries:
            if not isinstance(entry, Mapping) or entry.get("action") != "copied":
                continue
            target = root / str(entry["target_path"])
            if target.is_symlink() or _file_sha256(target) != entry.get("sha256"):
                raise S5RebalanceError(f"copied payload hash mismatch: {target}")
            verified_payload_files += 1
    return {
        "format": CURATION_FORMAT,
        "bundle_root": str(root),
        "base_dataset_fingerprint": base_fingerprint,
        "sidecar_dataset_fingerprint": sidecar_fingerprint,
        "curation_contract_sha256": manifest["curation_contract_sha256"],
        "bundle_index_sha256": _file_sha256(root / "bundle_episode_index.jsonl"),
        "base_episodes": EXPECTED_FINAL_BASE_EPISODES,
        "base_joint_samples": EXPECTED_FINAL_BASE_SAMPLES,
        "sidecar_episodes": EXPECTED_FINAL_SIDECAR_EPISODES,
        "sidecar_raw_steps": EXPECTED_FINAL_RAW_STEPS,
        "sidecar_only_episodes": 100,
        "bundle_index_rows": EXPECTED_FINAL_INDEX_ROWS,
        "split_episode_counts": EXPECTED_FINAL_SPLIT_EPISODES,
        "split_joint_samples": EXPECTED_FINAL_SPLIT_SAMPLES,
        "s5_behavior_episode_counts": {
            BRAKE_CATEGORY: 5,
            RELEASE_CATEGORY: 19,
        },
        "verified_payload_files": verified_payload_files,
        "eligible_for_formal_training": False,
    }


@contextmanager
def _exclusive_source_locks(roots: Sequence[Path]) -> Iterator[None]:
    streams = []
    try:
        for root in roots:
            for relative in (
                ".bundle_writer.lock",
                "platoon_joint_bev/.writer.lock",
                "riskentry_actor_sidecar/.writer.lock",
            ):
                stream = (root / relative).open("a+b")
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    stream.close()
                    raise S5RebalanceError(f"active writer lock: {root / relative}") from exc
                streams.append(stream)
        yield
    finally:
        for stream in reversed(streams):
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()


def _validate_swap_paths(formal_root: Path, staging_root: Path, backup_root: Path) -> None:
    parent = formal_root.parent.resolve()
    if any(path.parent.resolve() != parent for path in (formal_root, staging_root, backup_root)):
        raise S5RebalanceError("formal, staging, and backup roots must share one parent")
    if staging_root.name != f"{formal_root.name}.s5_rebalance_staging":
        raise S5RebalanceError("unexpected staging directory name")
    if backup_root.name != f"{formal_root.name}.pre_s5_rebalance_4060c643":
        raise S5RebalanceError("unexpected backup directory name")
    if formal_root.is_symlink() or staging_root.is_symlink() or backup_root.is_symlink():
        raise S5RebalanceError("swap paths must not be symbolic links")


def _delete_verified_backup(backup_root: Path, formal_root: Path) -> None:
    _validate_swap_paths(
        formal_root,
        formal_root.with_name(f"{formal_root.name}.s5_rebalance_staging"),
        backup_root,
    )
    if not backup_root.is_dir() or backup_root.is_symlink():
        raise S5RebalanceError("backup deletion target is not the expected real directory")
    _validate_no_links(backup_root)
    shutil.rmtree(backup_root)


def execute_rebalance(
    formal_root: Path,
    supplement_root: Path,
    staging_root: Path,
    backup_root: Path,
    plan: Mapping[str, object],
    *,
    delete_backup_after_verify: bool,
) -> dict[str, object]:
    _validate_swap_paths(formal_root, staging_root, backup_root)
    if staging_root.exists() or backup_root.exists():
        raise S5RebalanceError("staging or backup root already exists")
    with _exclusive_source_locks((formal_root, supplement_root)):
        verify_joint_risk_bundle(
            formal_root, scenario_contract_id=CANDIDATE_V3_CONTRACT_ID
        )
        verify_joint_risk_bundle(
            supplement_root, scenario_contract_id=CANDIDATE_V4_CONTRACT_ID
        )
        build_staging(formal_root, supplement_root, staging_root, plan)
        staging_report = verify_curated_bundle(staging_root)
        os.replace(formal_root, backup_root)
        try:
            os.replace(staging_root, formal_root)
        except Exception:
            os.replace(backup_root, formal_root)
            raise
        try:
            final_report = verify_curated_bundle(formal_root)
        except Exception:
            os.replace(formal_root, staging_root)
            os.replace(backup_root, formal_root)
            raise
        if final_report != {
            **staging_report,
            "bundle_root": str(formal_root.resolve()),
        }:
            os.replace(formal_root, staging_root)
            os.replace(backup_root, formal_root)
            raise S5RebalanceError("staging and final verification reports differ")
        if delete_backup_after_verify:
            _delete_verified_backup(backup_root, formal_root)
        curation_manifest = _read_json(formal_root / "dataset_curation_manifest.json")
        curation_manifest["original_payload_deleted_after_acceptance"] = bool(
            delete_backup_after_verify
        )
        _atomic_json(formal_root / "dataset_curation_manifest.json", curation_manifest)
        final_report["original_payload_deleted_after_acceptance"] = bool(
            delete_backup_after_verify
        )
        final_report["backup_root_exists"] = backup_root.exists()
        return final_report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, default=FORMAL_ROOT)
    parser.add_argument("--supplement-root", type=Path, default=SUPPLEMENT_ROOT)
    parser.add_argument("--expected-formal-fingerprint", required=True)
    parser.add_argument("--expected-supplement-fingerprint", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--replace-in-place", action="store_true")
    parser.add_argument("--delete-backup-after-verify", action="store_true")
    parser.add_argument("--result-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    formal_root = args.formal_root.expanduser().resolve()
    supplement_root = args.supplement_root.expanduser().resolve()
    plan = inspect_sources(
        formal_root,
        supplement_root,
        expected_formal_fingerprint=args.expected_formal_fingerprint,
        expected_supplement_fingerprint=args.expected_supplement_fingerprint,
    )
    if not args.execute:
        result = {"mode": "dry_run", **plan}
    else:
        if not args.replace_in_place or not args.delete_backup_after_verify:
            raise S5RebalanceError(
                "execution requires --replace-in-place and --delete-backup-after-verify"
            )
        staging_root = formal_root.with_name(f"{formal_root.name}.s5_rebalance_staging")
        backup_root = formal_root.with_name(
            f"{formal_root.name}.pre_s5_rebalance_4060c643"
        )
        result = {
            "mode": "executed",
            **execute_rebalance(
                formal_root,
                supplement_root,
                staging_root,
                backup_root,
                plan,
                delete_backup_after_verify=True,
            ),
        }
    if args.result_json is not None:
        _atomic_json(args.result_json.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except S5RebalanceError as exc:
        raise SystemExit(f"error: {exc}") from exc


__all__ = [
    "S5RebalanceError",
    "build_staging",
    "execute_rebalance",
    "inspect_sources",
    "verify_curated_bundle",
]
