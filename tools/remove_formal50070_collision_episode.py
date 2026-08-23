"""Remove the one user-waived collision episode from the formal50070 base set.

The raw RiskEntry sidecar is retained as sidecar-only evidence.  Installation is
performed through a physically copied sibling staging directory and an atomic
directory swap; the original bundle is retained as a rollback backup.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import fcntl
import json
import shutil
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

from expert_dataset.riskentry_sidecar_storage import (
    sidecar_dataset_contract,
    sidecar_dataset_fingerprint,
)
from expert_dataset.verify_rule_conditioned_v2_bundle import (
    verify_rule_conditioned_v2_bundle,
)
from scenarios.bev_round13_contract import CANDIDATE_V4_CONTRACT_ID
from tools.rebalance_s5_dataset import (
    _atomic_json,
    _file_sha256,
    _manifest_entries,
    _payload_sha256,
    _read_json,
    _read_rows,
    _validate_no_links,
)


DEFAULT_ROOT = Path(
    "/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/"
    "bev_joint_rule_conditioned_v2_s5_s9_formal50070_v1"
)
SOURCE_BASE_FINGERPRINT = (
    "88690467a1a65d05c7b724093f2aecbdf565d6e2f456b1e558737a4fb00a4813"
)
SOURCE_SIDECAR_FINGERPRINT = (
    "6903c8e8ca50c2b6d3dee906d8214a4830a584642eeca3167452bbdcf450238b"
)
SOURCE_INDEX_SHA256 = (
    "024b0fd6eb1a4410cb72b8054b12534f6aaafb19641c38c2d605dc9e2a6306e4"
)
TARGET_EPISODE_INDEX = 194
TARGET_SPLIT = "train"
TARGET_SCENARIO = "S6_background_merge_in"
TARGET_ROUTE = "R6_mainline_merge_approach"
TARGET_SEED = 1_751_228_647
TARGET_SAMPLES = 200
REMOVAL_REASON = "curation_removed_user_waived_platoon_collision"
CURATION_FORMAT = "rule-conditioned-v2-formal50070-episode-removal-v1"
INVENTORY_FORMAT = "rule-conditioned-v2-episode-removal-inventory-v1"
EXPECTED_OUTPUT = {
    "base_episodes": 258,
    "base_joint_samples": 49_870,
    "bundle_index_rows": 415,
    "rejected_episodes": 157,
    "sidecar_episodes": 415,
    "sidecar_only_episodes": 157,
    "sidecar_raw_steps": 118_321,
    "split_episode_counts": {"train": 204, "val": 27, "test": 27},
    "split_joint_samples": {"train": 39_470, "val": 5_170, "test": 5_230},
    "scenario_joint_samples": {
        "S5_hard_brake_lead": 10_070,
        "S6_background_merge_in": 9_800,
        "S7_ego_merge_from_ramp": 10_000,
        "S8_ego_exit_to_ramp": 10_000,
        "S9_narrow_channel_negotiation": 10_000,
    },
}


class Formal50070EpisodeRemovalError(RuntimeError):
    """Raised when source evidence or an atomic replacement boundary fails."""


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _target_paths(root: Path) -> tuple[Path, Path]:
    name = f"episode_{TARGET_EPISODE_INDEX:08d}"
    return (
        root / "platoon_joint_bev" / TARGET_SPLIT / "episodes" / name,
        root / "riskentry_actor_sidecar" / TARGET_SPLIT / "episodes" / name,
    )


def _find_platoon_collision(metadata: Mapping[str, object]) -> dict[str, object]:
    actors = metadata.get("actors")
    events = metadata.get("events")
    if not isinstance(actors, list) or not isinstance(events, list):
        raise Formal50070EpisodeRemovalError("target sidecar event evidence is missing")
    platoon_ids = {
        str(actor["actor_id"])
        for actor in actors
        if isinstance(actor, Mapping) and actor.get("actor_type") == "platoon"
    }
    matches = []
    for event in events:
        if not isinstance(event, Mapping) or event.get("event_type") not in {
            "collision_vehicle",
            "collision_object",
            "collision_sidewalk",
        }:
            continue
        actor_ids = event.get("actor_ids")
        if isinstance(actor_ids, list) and platoon_ids.intersection(
            str(value) for value in actor_ids
        ):
            matches.append(dict(event))
    if len(matches) != 1:
        raise Formal50070EpisodeRemovalError(
            f"expected exactly one target platoon collision, found {len(matches)}"
        )
    return matches[0]


def _curated_row(source_row: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "episode_index": TARGET_EPISODE_INDEX,
        "split": TARGET_SPLIT,
        "scenario_id": TARGET_SCENARIO,
        "local_route": TARGET_ROUTE,
        "spawn_seed": TARGET_SEED,
        "base_status": "committed",
        "sidecar_status": "committed",
        "base_samples": TARGET_SAMPLES,
        "outcome": "collision",
    }
    if any(source_row.get(key) != value for key, value in expected.items()):
        raise Formal50070EpisodeRemovalError("target bundle row drifted")
    row = dict(source_row)
    row.update(
        base_status="rejected",
        base_rejection_reason=REMOVAL_REASON,
        base_samples=0,
    )
    return row


def _curated_base_manifest(source: Mapping[str, object]) -> dict[str, object]:
    episodes = source.get("episodes")
    if not isinstance(episodes, list):
        raise Formal50070EpisodeRemovalError("base train manifest is invalid")
    target = [
        row
        for row in episodes
        if isinstance(row, Mapping)
        and row.get("episode_index") == TARGET_EPISODE_INDEX
    ]
    if len(target) != 1 or target[0].get("joint_samples") != TARGET_SAMPLES:
        raise Formal50070EpisodeRemovalError("target base manifest entry drifted")
    result = dict(source)
    result["episodes"] = [
        row
        for row in episodes
        if not isinstance(row, Mapping)
        or row.get("episode_index") != TARGET_EPISODE_INDEX
    ]
    result["episode_count"] = int(source["episode_count"]) - 1
    result["joint_samples"] = int(source["joint_samples"]) - TARGET_SAMPLES
    return result


def _curated_sidecar_manifest(source: Mapping[str, object]) -> dict[str, object]:
    episodes = source.get("episodes")
    if not isinstance(episodes, list):
        raise Formal50070EpisodeRemovalError("sidecar train manifest is invalid")
    updated = []
    changed = 0
    for source_row in episodes:
        if not isinstance(source_row, Mapping):
            raise Formal50070EpisodeRemovalError("sidecar manifest row is invalid")
        row = dict(source_row)
        if row.get("episode_index") == TARGET_EPISODE_INDEX:
            if row.get("base_samples") != TARGET_SAMPLES or row.get("outcome") != "collision":
                raise Formal50070EpisodeRemovalError(
                    "target sidecar manifest entry drifted"
                )
            row["base_samples"] = 0
            changed += 1
        updated.append(row)
    if changed != 1:
        raise Formal50070EpisodeRemovalError("target sidecar manifest entry is missing")
    result = dict(source)
    result["episodes"] = updated
    result["base_samples"] = int(source["base_samples"]) - TARGET_SAMPLES
    return result


def inspect_source(root: Path | str) -> dict[str, object]:
    source = Path(root).expanduser().resolve()
    _validate_no_links(source)
    for pending in (".bundle_episode_pending.json", ".targeted_batch_pending.json"):
        if (source / pending).exists():
            raise Formal50070EpisodeRemovalError(
                f"source has unfinished transaction: {pending}"
            )
    base_contract = _read_json(source / "platoon_joint_bev/dataset_contract.json")
    sidecar_contract = _read_json(
        source / "riskentry_actor_sidecar/dataset_contract.json"
    )
    manifest = _read_json(source / "dataset_bundle_manifest.json")
    if (
        base_contract.get("dataset_fingerprint") != SOURCE_BASE_FINGERPRINT
        or sidecar_contract.get("base_dataset_fingerprint")
        != SOURCE_BASE_FINGERPRINT
        or manifest.get("sidecar_dataset_fingerprint")
        != SOURCE_SIDECAR_FINGERPRINT
        or _file_sha256(source / "bundle_episode_index.jsonl")
        != SOURCE_INDEX_SHA256
    ):
        raise Formal50070EpisodeRemovalError("source fingerprint or index drifted")
    rows = _read_rows(source)
    if len(rows) != EXPECTED_OUTPUT["bundle_index_rows"]:
        raise Formal50070EpisodeRemovalError("source bundle row count drifted")
    _curated_row(rows[TARGET_EPISODE_INDEX])
    base_path, sidecar_path = _target_paths(source)
    if not base_path.is_dir() or not sidecar_path.is_dir():
        raise Formal50070EpisodeRemovalError("target component episode is missing")
    base_metadata = _read_json(base_path / "episode.json")
    sidecar_metadata = _read_json(sidecar_path / "episode.json")
    if (
        base_metadata.get("joint_samples") != TARGET_SAMPLES
        or sidecar_metadata.get("episode_index") != TARGET_EPISODE_INDEX
        or sidecar_metadata.get("base_dataset_fingerprint")
        != SOURCE_BASE_FINGERPRINT
    ):
        raise Formal50070EpisodeRemovalError("target episode metadata drifted")
    event = _find_platoon_collision(sidecar_metadata)
    removal_inventory = []
    for path in sorted(base_path.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise Formal50070EpisodeRemovalError(
                f"target base payload is not a regular file: {path}"
            )
        removal_inventory.append(
            {
                "path": str(path.relative_to(source)),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    contract_payload = {
        "format": CURATION_FORMAT,
        "source": {
            "base_dataset_fingerprint": SOURCE_BASE_FINGERPRINT,
            "sidecar_dataset_fingerprint": SOURCE_SIDECAR_FINGERPRINT,
            "bundle_index_sha256": SOURCE_INDEX_SHA256,
            "dataset_curation_manifest_sha256": _file_sha256(
                source / "dataset_curation_manifest.json"
            ),
        },
        "removal": {
            "episode_index": TARGET_EPISODE_INDEX,
            "split": TARGET_SPLIT,
            "scenario_id": TARGET_SCENARIO,
            "local_route": TARGET_ROUTE,
            "spawn_seed": TARGET_SEED,
            "base_samples": TARGET_SAMPLES,
            "base_policy": "remove_from_training_base",
            "sidecar_policy": "retain_raw_evidence_with_empty_base_mapping",
            "reason": REMOVAL_REASON,
            "collision_event": event,
            "base_payload_inventory": removal_inventory,
            "sidecar_episode_metadata_sha256": _file_sha256(
                sidecar_path / "episode.json"
            ),
        },
        "user_waiver": {
            "authorized_on": "2026-08-21",
            "temporary_training_use_allowed": True,
            "waived_original_total_sample_quota": 50_070,
            "waived_original_s6_sample_quota": 10_000,
            "strict_original_formal50070_contract_passed": False,
        },
        "expected_output": EXPECTED_OUTPUT,
    }
    base_fingerprint = _payload_sha256(contract_payload)
    sidecar_fingerprint = sidecar_dataset_fingerprint(
        base_fingerprint,
        base_format=str(base_contract["format"]),
        base_schema_version=int(base_contract["schema_version"]),
    )
    return {
        "format": CURATION_FORMAT,
        "source_root": str(source),
        "contract_payload": contract_payload,
        "base_dataset_fingerprint": base_fingerprint,
        "sidecar_dataset_fingerprint": sidecar_fingerprint,
        "eligible_for_formal_training_by_user_waiver": True,
        "strict_original_formal50070_contract_passed": False,
    }


def _copy_ignore(source: Path):
    base_parent = source / "platoon_joint_bev" / TARGET_SPLIT / "episodes"

    def ignore(directory: str, names: list[str]) -> set[str]:
        current = Path(directory)
        ignored: set[str] = set()
        if current == source:
            ignored.update(
                name
                for name in ("dataset_curation_manifest.json", "payload_inventory.json")
                if name in names
            )
        if current == base_parent and f"episode_{TARGET_EPISODE_INDEX:08d}" in names:
            ignored.add(f"episode_{TARGET_EPISODE_INDEX:08d}")
        return ignored

    return ignore


def _rewrite_episode_fingerprints(
    root: Path, *, base_fingerprint: str, sidecar_fingerprint: str
) -> None:
    for split in ("train", "val", "test"):
        manifest_path = root / "platoon_joint_bev" / split / "manifest.json"
        manifest = _read_json(manifest_path)
        entries = manifest.get("episodes")
        if not isinstance(entries, list):
            raise Formal50070EpisodeRemovalError("base manifest episodes missing")
        for entry in entries:
            if not isinstance(entry, dict):
                raise Formal50070EpisodeRemovalError("base manifest entry is invalid")
            attributes = entry.get("attributes")
            if not isinstance(attributes, dict):
                raise Formal50070EpisodeRemovalError("base manifest attributes missing")
            attributes["sidecar_dataset_fingerprint"] = sidecar_fingerprint
            episode_index = int(entry["episode_index"])
            path = (
                root
                / "platoon_joint_bev"
                / split
                / "episodes"
                / f"episode_{episode_index:08d}"
                / "episode.json"
            )
            metadata = _read_json(path)
            metadata_attributes = metadata.get("attributes")
            if not isinstance(metadata_attributes, dict):
                raise Formal50070EpisodeRemovalError("base episode attributes missing")
            metadata_attributes["sidecar_dataset_fingerprint"] = sidecar_fingerprint
            _atomic_json(path, metadata)
        _atomic_json(manifest_path, manifest)
    for episode_index, (split, _) in _manifest_entries(
        root, "riskentry_actor_sidecar"
    ).items():
        path = (
            root
            / "riskentry_actor_sidecar"
            / split
            / "episodes"
            / f"episode_{episode_index:08d}"
            / "episode.json"
        )
        metadata = _read_json(path)
        metadata["base_dataset_fingerprint"] = base_fingerprint
        _atomic_json(path, metadata)


def build_staging(source: Path, staging: Path, plan: Mapping[str, object]) -> None:
    if staging.exists() or staging.parent.resolve() != source.parent.resolve():
        raise Formal50070EpisodeRemovalError("invalid or existing staging root")
    source_bytes = sum(
        path.stat().st_size for path in source.rglob("*") if path.is_file()
    )
    if shutil.disk_usage(source.parent).free < source_bytes + 1024**3:
        raise Formal50070EpisodeRemovalError("insufficient free space for physical staging")
    shutil.copytree(
        source,
        staging,
        copy_function=shutil.copy2,
        symlinks=False,
        ignore=_copy_ignore(source),
    )
    base_fingerprint = str(plan["base_dataset_fingerprint"])
    sidecar_fingerprint = str(plan["sidecar_dataset_fingerprint"])

    base_contract = _read_json(staging / "platoon_joint_bev/dataset_contract.json")
    base_contract["dataset_fingerprint"] = base_fingerprint
    _atomic_json(staging / "platoon_joint_bev/dataset_contract.json", base_contract)
    _atomic_json(
        staging / "riskentry_actor_sidecar/dataset_contract.json",
        sidecar_dataset_contract(
            base_fingerprint,
            base_format=str(base_contract["format"]),
            base_schema_version=int(base_contract["schema_version"]),
        ),
    )
    bundle_manifest = _read_json(staging / "dataset_bundle_manifest.json")
    bundle_manifest["base_dataset_fingerprint"] = base_fingerprint
    bundle_manifest["sidecar_dataset_fingerprint"] = sidecar_fingerprint
    _atomic_json(staging / "dataset_bundle_manifest.json", bundle_manifest)

    _atomic_json(
        staging / "platoon_joint_bev/train/manifest.json",
        _curated_base_manifest(
            _read_json(source / "platoon_joint_bev/train/manifest.json")
        ),
    )
    _atomic_json(
        staging / "riskentry_actor_sidecar/train/manifest.json",
        _curated_sidecar_manifest(
            _read_json(source / "riskentry_actor_sidecar/train/manifest.json")
        ),
    )
    _rewrite_episode_fingerprints(
        staging,
        base_fingerprint=base_fingerprint,
        sidecar_fingerprint=sidecar_fingerprint,
    )
    _, sidecar_path = _target_paths(staging)
    _atomic_bytes(
        sidecar_path / "base_sample_step_index.npy",
        _empty_int64_npy(),
    )

    state = _read_json(source / "platoon_joint_bev/collection_state.json")
    reasons = dict(state.get("rejection_reasons", {}))
    reasons[REMOVAL_REASON] = int(reasons.get(REMOVAL_REASON, 0)) + 1
    state.update(
        stored_episodes=EXPECTED_OUTPUT["base_episodes"],
        rejected_episodes=EXPECTED_OUTPUT["rejected_episodes"],
        total_joint_samples=EXPECTED_OUTPUT["base_joint_samples"],
        rejection_reasons=reasons,
    )
    _atomic_json(staging / "platoon_joint_bev/collection_state.json", state)

    rows = _read_rows(source)
    rows[TARGET_EPISODE_INDEX] = _curated_row(rows[TARGET_EPISODE_INDEX])
    encoded = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        + b"\n"
        for row in rows
    )
    _atomic_bytes(staging / "bundle_episode_index.jsonl", encoded)

    inventory = {
        "format": INVENTORY_FORMAT,
        "source_base_dataset_fingerprint": SOURCE_BASE_FINGERPRINT,
        "source_bundle_index_sha256": SOURCE_INDEX_SHA256,
        "excluded_base_payload": plan["contract_payload"]["removal"][
            "base_payload_inventory"
        ],
        "retained_sidecar_episode_index": TARGET_EPISODE_INDEX,
    }
    _atomic_json(staging / "payload_inventory.json", inventory)
    _atomic_json(
        staging / "dataset_curation_manifest.json",
        {
            "format": CURATION_FORMAT,
            "complete": True,
            "contract_payload": plan["contract_payload"],
            "base_dataset_fingerprint": base_fingerprint,
            "sidecar_dataset_fingerprint": sidecar_fingerprint,
            "payload_inventory_sha256": _file_sha256(
                staging / "payload_inventory.json"
            ),
            "formal_training_eligibility": True,
            "eligibility_basis": "explicit_user_temporary_waiver_after_base_episode_removal",
            "strict_original_formal50070_contract_passed": False,
            "source_root_preserved_as_atomic_backup": True,
        },
    )


def _empty_int64_npy() -> bytes:
    import io

    stream = io.BytesIO()
    np.save(stream, np.empty((0,), dtype=np.int64), allow_pickle=False)
    return stream.getvalue()


def verify_curated_bundle(root: Path | str) -> dict[str, object]:
    bundle = Path(root).expanduser().resolve()
    _validate_no_links(bundle)
    manifest = _read_json(bundle / "dataset_curation_manifest.json")
    if manifest.get("format") != CURATION_FORMAT or manifest.get("complete") is not True:
        raise Formal50070EpisodeRemovalError("curation manifest is incomplete")
    contract = manifest.get("contract_payload")
    if not isinstance(contract, Mapping):
        raise Formal50070EpisodeRemovalError("curation contract is missing")
    base_fingerprint = _payload_sha256(contract)
    base_contract = _read_json(bundle / "platoon_joint_bev/dataset_contract.json")
    expected_sidecar = sidecar_dataset_fingerprint(
        base_fingerprint,
        base_format=str(base_contract["format"]),
        base_schema_version=int(base_contract["schema_version"]),
    )
    if (
        manifest.get("base_dataset_fingerprint") != base_fingerprint
        or manifest.get("sidecar_dataset_fingerprint") != expected_sidecar
        or base_contract.get("dataset_fingerprint") != base_fingerprint
        or manifest.get("formal_training_eligibility") is not True
        or manifest.get("strict_original_formal50070_contract_passed") is not False
        or _file_sha256(bundle / "payload_inventory.json")
        != manifest.get("payload_inventory_sha256")
    ):
        raise Formal50070EpisodeRemovalError("curation binding mismatch")
    base_path, sidecar_path = _target_paths(bundle)
    if base_path.exists() or not sidecar_path.is_dir():
        raise Formal50070EpisodeRemovalError("target base/sidecar retention mismatch")
    mapping = np.load(
        sidecar_path / "base_sample_step_index.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    if mapping.dtype != np.int64 or mapping.shape != (0,):
        raise Formal50070EpisodeRemovalError("retained sidecar base mapping is not empty")
    sidecar_metadata = _read_json(sidecar_path / "episode.json")
    _find_platoon_collision(sidecar_metadata)
    rows = _read_rows(bundle)
    target_row = rows[TARGET_EPISODE_INDEX]
    if (
        target_row.get("base_status") != "rejected"
        or target_row.get("base_rejection_reason") != REMOVAL_REASON
        or target_row.get("sidecar_status") != "committed"
        or target_row.get("base_samples") != 0
        or target_row.get("outcome") != "collision"
    ):
        raise Formal50070EpisodeRemovalError("curated target index row mismatch")

    base_entries = _manifest_entries(bundle, "platoon_joint_bev")
    sidecar_entries = _manifest_entries(bundle, "riskentry_actor_sidecar")
    split_episodes: Counter[str] = Counter()
    split_samples: Counter[str] = Counter()
    scenario_samples: Counter[str] = Counter()
    for _, (split, entry) in base_entries.items():
        attributes = entry.get("attributes")
        if not isinstance(attributes, Mapping):
            raise Formal50070EpisodeRemovalError("base manifest attributes missing")
        split_episodes[split] += 1
        samples = int(entry["joint_samples"])
        split_samples[split] += samples
        scenario_samples[str(attributes["scenario_id"])] += samples
    expected = contract.get("expected_output")
    if not isinstance(expected, Mapping) or (
        len(base_entries) != expected["base_episodes"]
        or len(sidecar_entries) != expected["sidecar_episodes"]
        or dict(split_episodes) != expected["split_episode_counts"]
        or dict(split_samples) != expected["split_joint_samples"]
        or dict(scenario_samples) != expected["scenario_joint_samples"]
    ):
        raise Formal50070EpisodeRemovalError("curated count contract mismatch")
    state = _read_json(bundle / "platoon_joint_bev/collection_state.json")
    if (
        state.get("stored_episodes") != expected["base_episodes"]
        or state.get("rejected_episodes") != expected["rejected_episodes"]
        or state.get("total_joint_samples") != expected["base_joint_samples"]
    ):
        raise Formal50070EpisodeRemovalError("curated collection state mismatch")

    conditioned = verify_rule_conditioned_v2_bundle(
        bundle, scenario_contract_id=CANDIDATE_V4_CONTRACT_ID
    )
    sidecar_only = sum(
        row.get("base_status") == "rejected"
        and row.get("sidecar_status") == "committed"
        for row in rows
    )
    if (
        sidecar_only != expected["sidecar_only_episodes"]
        or conditioned["aligned_samples"] != expected["base_joint_samples"]
    ):
        raise Formal50070EpisodeRemovalError("complete verifier totals mismatch")
    return {
        "status": "pass_with_explicit_user_quota_waiver",
        "bundle_root": str(bundle),
        "base_dataset_fingerprint": base_fingerprint,
        "sidecar_dataset_fingerprint": expected_sidecar,
        "bundle_index_sha256": _file_sha256(
            bundle / "bundle_episode_index.jsonl"
        ),
        **dict(expected),
        "strict_original_formal50070_contract_passed": False,
        "eligible_for_formal_training_by_user_waiver": True,
    }


@contextmanager
def _exclusive_locks(root: Path) -> Iterator[None]:
    streams = []
    try:
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
                raise Formal50070EpisodeRemovalError(
                    f"active writer lock: {root / relative}"
                ) from exc
            streams.append(stream)
        yield
    finally:
        for stream in reversed(streams):
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()


def _swap_paths(root: Path) -> tuple[Path, Path, Path]:
    return (
        root.with_name(f"{root.name}.episode194_removal_staging"),
        root.with_name(f"{root.name}.pre_episode194_removal_88690467"),
        root.with_name(f"{root.name}.episode194_removal_failed"),
    )


def execute(source: Path, plan: Mapping[str, object]) -> dict[str, object]:
    root = source.expanduser().resolve()
    staging, backup, failed = _swap_paths(root)
    if any(path.parent.resolve() != root.parent.resolve() for path in (staging, backup, failed)):
        raise Formal50070EpisodeRemovalError("swap roots must share one parent")
    if any(path.exists() or path.is_symlink() for path in (staging, backup, failed)):
        raise Formal50070EpisodeRemovalError("staging, backup, or failed root exists")
    with _exclusive_locks(root):
        current_plan = inspect_source(root)
        if current_plan != plan:
            raise Formal50070EpisodeRemovalError("source drifted after dry-run")
        build_staging(root, staging, plan)
        staging_report = verify_curated_bundle(staging)
        os.replace(root, backup)
        try:
            os.replace(staging, root)
        except Exception:
            os.replace(backup, root)
            raise
        try:
            final_report = verify_curated_bundle(root)
            if final_report != {**staging_report, "bundle_root": str(root)}:
                raise Formal50070EpisodeRemovalError(
                    "staging and installed verification reports differ"
                )
        except Exception:
            os.replace(root, failed)
            os.replace(backup, root)
            raise
    return {
        **final_report,
        "backup_root": str(backup),
        "backup_retained": backup.is_dir(),
        "rollback_performed": False,
    }


def resume_staging(source: Path, plan: Mapping[str, object]) -> dict[str, object]:
    """Repair metadata bindings in this tool's retained staging, then install it."""

    root = source.expanduser().resolve()
    staging, backup, failed = _swap_paths(root)
    if not staging.is_dir() or staging.is_symlink():
        raise Formal50070EpisodeRemovalError("retained staging root is missing")
    if backup.exists() or failed.exists():
        raise Formal50070EpisodeRemovalError("backup or failed root already exists")
    with _exclusive_locks(root):
        current_plan = inspect_source(root)
        if current_plan != plan:
            raise Formal50070EpisodeRemovalError("source drifted after dry-run")
        staging_manifest = _read_json(staging / "dataset_curation_manifest.json")
        if (
            staging_manifest.get("format") != CURATION_FORMAT
            or staging_manifest.get("contract_payload") != plan["contract_payload"]
        ):
            raise Formal50070EpisodeRemovalError(
                "retained staging is not bound to the current plan"
            )
        _rewrite_episode_fingerprints(
            staging,
            base_fingerprint=str(plan["base_dataset_fingerprint"]),
            sidecar_fingerprint=str(plan["sidecar_dataset_fingerprint"]),
        )
        staging_report = verify_curated_bundle(staging)
        os.replace(root, backup)
        try:
            os.replace(staging, root)
        except Exception:
            os.replace(backup, root)
            raise
        try:
            final_report = verify_curated_bundle(root)
            if final_report != {**staging_report, "bundle_root": str(root)}:
                raise Formal50070EpisodeRemovalError(
                    "staging and installed verification reports differ"
                )
        except Exception:
            os.replace(root, failed)
            os.replace(backup, root)
            raise
    return {
        **final_report,
        "backup_root": str(backup),
        "backup_retained": backup.is_dir(),
        "rollback_performed": False,
        "resumed_retained_staging": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-staging", action="store_true")
    parser.add_argument("--result-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.bundle_root.expanduser().resolve()
    plan = inspect_source(root)
    if args.execute and args.resume_staging:
        raise Formal50070EpisodeRemovalError(
            "--execute and --resume-staging are mutually exclusive"
        )
    if args.execute:
        result = execute(root, plan)
    elif args.resume_staging:
        result = resume_staging(root, plan)
    else:
        result = plan
    encoded = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.result_json is not None:
        _atomic_bytes(args.result_json, encoded.encode("utf-8"))
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "Formal50070EpisodeRemovalError",
    "build_staging",
    "execute",
    "inspect_source",
    "resume_staging",
    "verify_curated_bundle",
]
