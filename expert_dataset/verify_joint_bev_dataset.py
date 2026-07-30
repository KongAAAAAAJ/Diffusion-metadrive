"""Verify the packed joint-first BEV dataset and report pilot statistics."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from envs.observations.semantic_bev import BEV_CHANNEL_NAMES
from expert_dataset.collect_joint_bev import AgentRole, JOINT_SAMPLE_DTYPES
from expert_dataset.joint_bev_dataset import (
    JointBEVDataset,
    JointBEVDatasetConfig,
    JointBEVDatasetError,
    validate_dataset_contract,
)
from expert_dataset.joint_bev_storage import (
    EpisodeSplitAssigner,
    EpisodeSplitConfig,
    PACKED_BEV_FIELD,
    SPLIT_NAMES,
    STORAGE_SCHEMA_VERSION,
    STORED_FIELD_SHAPES,
)
from expert_dataset.semantic_bev_codec import (
    BEV_COMPRESSION_RATIO,
    LOGICAL_BEV_SHAPE,
    SemanticBEVCodecError,
    pack_semantic_bev,
    unpack_semantic_bev,
)
from models.bev_planner.mode_contract import (
    MODE_NAMES,
    NUM_MODES,
    ModeContractError,
    ModeIndex,
    validate_trajectory_kinematics,
)
from scenarios.bev_round13_contract import (
    PRIMARY_S5_S9_SCENARIOS,
    primary_scenario_contract,
)


DIAGNOSTIC_64_SCENARIO_COUNTS = {
    "S5_hard_brake_lead": 13,
    "S6_background_merge_in": 13,
    "S7_ego_merge_from_ramp": 13,
    "S8_ego_exit_to_ramp": 13,
    "S9_narrow_channel_negotiation": 12,
}


class JointBEVVerificationError(RuntimeError):
    """Raised when a dataset fails a structural or semantic verification rule."""


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JointBEVVerificationError(f"invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise JointBEVVerificationError(f"JSON root must be an object: {path}")
    return payload


def _split_assigner(contract: Mapping[str, object]) -> EpisodeSplitAssigner:
    raw = contract.get("split_assignment")
    expected = {"train_ratio", "val_ratio", "test_ratio", "seed"}
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise JointBEVVerificationError("invalid split_assignment contract")
    try:
        config = EpisodeSplitConfig(
            train_ratio=float(raw["train_ratio"]),
            val_ratio=float(raw["val_ratio"]),
            test_ratio=float(raw["test_ratio"]),
            seed=int(raw["seed"]),
        )
    except (TypeError, ValueError) as exc:
        raise JointBEVVerificationError("invalid split_assignment contract") from exc
    return EpisodeSplitAssigner(config)


def _validate_root_entries(root: Path) -> dict[str, object]:
    expected = {
        ".writer.lock",
        "collection_state.json",
        "dataset_contract.json",
        *SPLIT_NAMES,
    }
    try:
        actual = {path.name for path in root.iterdir()}
    except OSError as exc:
        raise JointBEVVerificationError(f"unable to inspect dataset root: {root}") from exc
    if actual != expected:
        raise JointBEVVerificationError(
            f"dataset root file set mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    state = _read_json(root / "collection_state.json")
    if int(state.get("schema_version", -1)) != STORAGE_SCHEMA_VERSION:
        raise JointBEVVerificationError("collection state schema version mismatch")
    return state


def _validate_chunk(
    arrays: Mapping[str, np.ndarray],
    start: int,
    stop: int,
    *,
    split: str,
    episode_index: int,
) -> tuple[np.ndarray, float, dict[str, int]]:
    context = f"split={split} episode={episode_index} samples=[{start}:{stop}]"
    packed = np.asarray(arrays[PACKED_BEV_FIELD][start:stop])
    decode_start = time.perf_counter()
    try:
        bev = unpack_semantic_bev(packed)
    except SemanticBEVCodecError as exc:
        raise JointBEVVerificationError(
            f"packed semantic BEV decode failed at {context}: {exc}"
        ) from exc
    decode_seconds = time.perf_counter() - decode_start
    try:
        repacked = pack_semantic_bev(bev)
    except SemanticBEVCodecError as exc:
        raise JointBEVVerificationError(
            f"decoded semantic BEV is invalid at {context}: {exc}"
        ) from exc
    if not np.array_equal(repacked, packed):
        raise JointBEVVerificationError(
            f"semantic BEV round-trip mismatch at {context}"
        )

    for field_name, dtype in JOINT_SAMPLE_DTYPES.items():
        if field_name == "bev":
            continue
        value = np.asarray(arrays[field_name][start:stop])
        if value.dtype != dtype:
            raise JointBEVVerificationError(
                f"{field_name} dtype mismatch at {context}"
            )
        if np.issubdtype(dtype, np.floating) and not np.isfinite(value).all():
            raise JointBEVVerificationError(
                f"{field_name} contains non-finite values at {context}"
            )

    roles = np.asarray(arrays["agent_role"][start:stop])
    expected_roles = np.asarray(list(AgentRole), dtype=np.int64)
    if not np.all(roles == expected_roles):
        raise JointBEVVerificationError(f"agent role order mismatch at {context}")
    mode_mask = np.asarray(arrays["mode_valid_mask"][start:stop])
    gt_mode = np.asarray(arrays["gt_mode"][start:stop])
    if np.any(gt_mode < 0) or np.any(gt_mode >= NUM_MODES):
        raise JointBEVVerificationError(f"gt_mode is outside fixed K=10 at {context}")
    if not np.all(mode_mask[..., int(ModeIndex.STOP)]):
        raise JointBEVVerificationError(f"STOP mode is disabled at {context}")
    selected_valid = np.take_along_axis(
        mode_mask, gt_mode[..., None], axis=-1
    )[..., 0]
    if not np.all(selected_valid):
        raise JointBEVVerificationError(
            f"mode_valid_mask[gt_mode] is false at {context}"
        )
    ego_state = np.asarray(arrays["ego_state"][start:stop])
    coarse = np.asarray(arrays["coarse_trajectories"][start:stop])
    expert = np.asarray(arrays["expert_trajectory"][start:stop])
    kinematic_counts = {
        "expert_total": 0,
        "expert_valid": 0,
        "hard_valid_anchor_total": 0,
        "hard_valid_anchor_valid": 0,
        "stop_total": 0,
        "stop_valid": 0,
    }
    for sample_index in range(stop - start):
        for role_index in range(3):
            speed = float(ego_state[sample_index, role_index, 0])
            try:
                expert_audit = validate_trajectory_kinematics(
                    expert[sample_index, role_index],
                    speed,
                    np.zeros((3,), dtype=np.float64),
                )
            except ModeContractError as exc:
                raise JointBEVVerificationError(
                    f"expert trajectory contract error at {context}, "
                    f"sample={sample_index}, role={role_index}: {exc}"
                ) from exc
            kinematic_counts["expert_total"] += 1
            if not expert_audit.valid:
                raise JointBEVVerificationError(
                    f"expert trajectory is dynamically invalid at {context}, "
                    f"sample={sample_index}, role={role_index}: "
                    f"{','.join(expert_audit.violations)}"
                )
            kinematic_counts["expert_valid"] += 1
            for mode_index in np.flatnonzero(
                mode_mask[sample_index, role_index]
            ):
                anchor_audit = validate_trajectory_kinematics(
                    coarse[sample_index, role_index, int(mode_index)],
                    speed,
                    np.zeros((3,), dtype=np.float64),
                )
                kinematic_counts["hard_valid_anchor_total"] += 1
                if int(mode_index) == int(ModeIndex.STOP):
                    kinematic_counts["stop_total"] += 1
                if not anchor_audit.valid:
                    raise JointBEVVerificationError(
                        f"hard-valid anchor is dynamically invalid at {context}, "
                        f"sample={sample_index}, role={role_index}, "
                        f"mode={int(mode_index)}: "
                        f"{','.join(anchor_audit.violations)}"
                    )
                kinematic_counts["hard_valid_anchor_valid"] += 1
                if int(mode_index) == int(ModeIndex.STOP):
                    kinematic_counts["stop_valid"] += 1
    return bev, decode_seconds, kinematic_counts


def verify_joint_bev_dataset(
    dataset_root: Path | str,
    *,
    splits: Sequence[str] = SPLIT_NAMES,
    max_samples_per_split: int | None = None,
    chunk_size: int = 16,
    min_decode_samples_per_s: float = 0.0,
    diagnostic_64: bool = False,
) -> dict[str, object]:
    """Run strict verification and return a JSON-serializable report."""

    root = Path(dataset_root).expanduser()
    if not root.is_dir():
        raise JointBEVVerificationError(f"dataset root does not exist: {root}")
    selected_splits = tuple(str(split) for split in splits)
    if (
        not selected_splits
        or len(set(selected_splits)) != len(selected_splits)
        or any(split not in SPLIT_NAMES for split in selected_splits)
    ):
        raise JointBEVVerificationError(
            f"splits must be unique values from {SPLIT_NAMES}"
        )
    if max_samples_per_split is not None and max_samples_per_split <= 0:
        raise JointBEVVerificationError("max_samples_per_split must be positive")
    if chunk_size <= 0:
        raise JointBEVVerificationError("chunk_size must be positive")
    if min_decode_samples_per_s < 0.0:
        raise JointBEVVerificationError(
            "min_decode_samples_per_s must be non-negative"
        )

    try:
        contract = validate_dataset_contract(root)
    except JointBEVDatasetError as exc:
        raise JointBEVVerificationError(str(exc)) from exc
    state = _validate_root_entries(root)
    assigner = _split_assigner(contract)

    datasets: dict[str, JointBEVDataset] = {}
    episode_owner: dict[int, str] = {}
    try:
        for split in selected_splits:
            try:
                dataset = JointBEVDataset(
                    JointBEVDatasetConfig(
                        root, split, mmap_cache_episodes=1
                    )
                )
            except JointBEVDatasetError as exc:
                raise JointBEVVerificationError(str(exc)) from exc
            datasets[split] = dataset
            for record in dataset.records:
                previous = episode_owner.get(record.episode_index)
                if previous is not None:
                    raise JointBEVVerificationError(
                        f"episode {record.episode_index} appears in both "
                        f"{previous} and {split}"
                    )
                expected_split = assigner.split_for_episode(record.episode_index)
                if expected_split != split:
                    raise JointBEVVerificationError(
                        f"episode {record.episode_index} is in {split}, "
                        f"expected {expected_split}"
                    )
                episode_owner[record.episode_index] = split

        report_splits: dict[str, object] = {}
        total_scanned = 0
        total_decode_seconds = 0.0
        total_episodes = 0
        total_joint_samples = 0
        total_packed_bytes = 0
        total_raw_bytes = 0
        total_kinematic_counts: Counter[str] = Counter()
        for split, dataset in datasets.items():
            scenario_counts: Counter[str] = Counter()
            gt_counts = np.zeros(NUM_MODES, dtype=np.int64)
            valid_counts = np.zeros(NUM_MODES, dtype=np.int64)
            occupancy = np.zeros(len(BEV_CHANNEL_NAMES), dtype=np.int64)
            occupancy_denominator = 0
            relation_valid = 0
            relation_total = 0
            scanned = 0
            decode_seconds = 0.0
            for record_index, record in enumerate(dataset.records):
                scenario_id = str(record.attributes.get("scenario_id", "")).strip()
                local_route = str(record.attributes.get("local_route", "")).strip()
                if not scenario_id or not local_route:
                    raise JointBEVVerificationError(
                        f"episode {record.episode_index} lacks scenario/route attributes"
                    )
                if diagnostic_64:
                    expected_routes = dict(PRIMARY_S5_S9_SCENARIOS)
                    if expected_routes.get(scenario_id) != local_route:
                        raise JointBEVVerificationError(
                            f"diagnostic episode {record.episode_index} "
                            "violates the S5--S9 route contract"
                        )
                    if record.attributes.get("diagnostic_subsampled") is not True:
                        raise JointBEVVerificationError(
                            "diagnostic episode is not marked as event-subsampled"
                        )
                    if (
                        record.attributes.get("scenario_contract_sha256")
                        != primary_scenario_contract()["sha256"]
                    ):
                        raise JointBEVVerificationError(
                            "diagnostic scenario contract hash mismatch"
                        )
                    selected_steps = record.attributes.get(
                        "selected_sample_steps"
                    )
                    trigger_step = record.attributes.get(
                        "scenario_trigger_step"
                    )
                    if (
                        not isinstance(selected_steps, list)
                        or len(selected_steps) != record.joint_samples
                        or isinstance(trigger_step, bool)
                        or not isinstance(trigger_step, int)
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, int)
                            or value < trigger_step
                            for value in selected_steps
                        )
                    ):
                        raise JointBEVVerificationError(
                            "diagnostic event sample metadata are invalid"
                        )
                episode_root = (
                    dataset.split_root / "episodes" / record.directory
                )
                episode_metadata = _read_json(episode_root / "episode.json")
                if episode_metadata.get("attributes") != dict(record.attributes):
                    raise JointBEVVerificationError(
                        f"episode/manifest attributes mismatch: {record.episode_index}"
                    )
                try:
                    arrays = dataset._load_episode(record_index)
                except JointBEVDatasetError as exc:
                    raise JointBEVVerificationError(str(exc)) from exc
                available = record.joint_samples
                if max_samples_per_split is not None:
                    available = min(
                        available, max_samples_per_split - scanned
                    )
                scenario_counts[scenario_id] += available
                for start in range(0, available, chunk_size):
                    stop = min(start + chunk_size, available)
                    bev, elapsed, kinematic_counts = _validate_chunk(
                        arrays,
                        start,
                        stop,
                        split=split,
                        episode_index=record.episode_index,
                    )
                    count = stop - start
                    scanned += count
                    decode_seconds += elapsed
                    total_kinematic_counts.update(kinematic_counts)
                    occupancy += np.count_nonzero(
                        bev, axis=(0, 1, 3, 4)
                    ).astype(np.int64)
                    occupancy_denominator += (
                        count
                        * 3
                        * LOGICAL_BEV_SHAPE[-2]
                        * LOGICAL_BEV_SHAPE[-1]
                    )
                    gt = np.asarray(arrays["gt_mode"][start:stop])
                    gt_counts += np.bincount(
                        gt.reshape(-1), minlength=NUM_MODES
                    )
                    valid_counts += np.asarray(
                        arrays["mode_valid_mask"][start:stop]
                    ).sum(axis=(0, 1), dtype=np.int64)
                    relation = np.asarray(
                        arrays["relation_valid_mask"][start:stop]
                    )
                    relation_valid += int(relation.sum())
                    relation_total += int(relation.size)
                if (
                    max_samples_per_split is not None
                    and scanned >= max_samples_per_split
                ):
                    break

            packed_bytes = int(
                len(dataset)
                * np.prod(STORED_FIELD_SHAPES[PACKED_BEV_FIELD])
            )
            raw_bytes = int(
                len(dataset) * 3 * np.prod(LOGICAL_BEV_SHAPE)
            )
            role_samples = scanned * 3
            split_rate = (
                scanned / decode_seconds if decode_seconds > 0.0 else 0.0
            )
            report_splits[split] = {
                "episodes": len(dataset.records),
                "joint_samples": len(dataset),
                "scanned_joint_samples": scanned,
                "complete_scan": scanned == len(dataset),
                "scenario_joint_samples": dict(sorted(scenario_counts.items())),
                "gt_mode_counts": {
                    name: int(gt_counts[index])
                    for index, name in enumerate(MODE_NAMES)
                },
                "mode_valid_rate": {
                    name: (
                        float(valid_counts[index] / role_samples)
                        if role_samples
                        else 0.0
                    )
                    for index, name in enumerate(MODE_NAMES)
                },
                "relation_valid_rate": (
                    float(relation_valid / relation_total)
                    if relation_total
                    else 0.0
                ),
                "bev_occupancy_rate": {
                    name: (
                        float(occupancy[index] / occupancy_denominator)
                        if occupancy_denominator
                        else 0.0
                    )
                    for index, name in enumerate(BEV_CHANNEL_NAMES)
                },
                "packed_bev_payload_bytes": packed_bytes,
                "raw_bev_payload_bytes": raw_bytes,
                "bev_compression_ratio": (
                    float(raw_bytes / packed_bytes)
                    if packed_bytes
                    else BEV_COMPRESSION_RATIO
                ),
                "decode_seconds": decode_seconds,
                "decode_joint_samples_per_s": split_rate,
            }
            total_scanned += scanned
            total_decode_seconds += decode_seconds
            total_episodes += len(dataset.records)
            total_joint_samples += len(dataset)
            total_packed_bytes += packed_bytes
            total_raw_bytes += raw_bytes

        state_stored = int(state.get("stored_episodes", -1))
        state_samples = int(state.get("total_joint_samples", -1))
        if set(selected_splits) == set(SPLIT_NAMES):
            if state_stored != total_episodes:
                raise JointBEVVerificationError(
                    "collection state stored_episodes mismatch"
                )
            if state_samples != total_joint_samples:
                raise JointBEVVerificationError(
                    "collection state total_joint_samples mismatch"
                )
            attempted = int(state.get("attempted_episodes", -1))
            rejected = int(state.get("rejected_episodes", -1))
            next_episode = int(state.get("next_episode_index", -1))
            if (
                attempted != next_episode
                or attempted != state_stored + rejected
            ):
                raise JointBEVVerificationError(
                    "collection state episode counters are inconsistent"
                )

        total_rate = (
            total_scanned / total_decode_seconds
            if total_decode_seconds > 0.0
            else 0.0
        )
        if total_scanned and total_rate < min_decode_samples_per_s:
            raise JointBEVVerificationError(
                f"decode throughput {total_rate:.2f} joint samples/s is below "
                f"required {min_decode_samples_per_s:.2f}"
            )
        global_scenario_counts: Counter[str] = Counter()
        for value in report_splits.values():
            global_scenario_counts.update(value["scenario_joint_samples"])
        if (
            total_kinematic_counts["expert_valid"]
            != total_kinematic_counts["expert_total"]
            or total_kinematic_counts["hard_valid_anchor_valid"]
            != total_kinematic_counts["hard_valid_anchor_total"]
            or total_kinematic_counts["stop_valid"]
            != total_kinematic_counts["stop_total"]
        ):
            raise JointBEVVerificationError(
                "trajectory kinematic validation counts are inconsistent"
            )
        if diagnostic_64:
            if total_joint_samples != 64 or total_scanned != 64:
                raise JointBEVVerificationError(
                    "diagnostic_64 requires a complete scan of exactly 64 samples"
                )
            if dict(global_scenario_counts) != DIAGNOSTIC_64_SCENARIO_COUNTS:
                raise JointBEVVerificationError(
                    "diagnostic_64 scenario sample quotas do not match"
                )
            if any(len(dataset) == 0 for dataset in datasets.values()):
                raise JointBEVVerificationError(
                    "diagnostic_64 requires non-empty train/val/test splits"
                )
            if (
                total_kinematic_counts["expert_total"] != 192
                or total_kinematic_counts["stop_total"] != 192
            ):
                raise JointBEVVerificationError(
                    "diagnostic_64 role-level kinematic counts must equal 192"
                )
        return {
            "schema_version": STORAGE_SCHEMA_VERSION,
            "dataset_root": str(root.resolve()),
            "verified_splits": list(selected_splits),
            "complete_scan": all(
                bool(value["complete_scan"])
                for value in report_splits.values()
            ),
            "episodes": total_episodes,
            "joint_samples": total_joint_samples,
            "scanned_joint_samples": total_scanned,
            "packed_bev_payload_bytes": total_packed_bytes,
            "raw_bev_payload_bytes": total_raw_bytes,
            "bev_compression_ratio": (
                float(total_raw_bytes / total_packed_bytes)
                if total_packed_bytes
                else BEV_COMPRESSION_RATIO
            ),
            "decode_seconds": total_decode_seconds,
            "decode_joint_samples_per_s": total_rate,
            "kinematic_validation": dict(
                sorted(total_kinematic_counts.items())
            ),
            "scenario_joint_samples": dict(
                sorted(global_scenario_counts.items())
            ),
            "diagnostic_64": bool(diagnostic_64),
            "splits": report_splits,
        }
    finally:
        for dataset in datasets.values():
            dataset.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLIT_NAMES,
        default=list(SPLIT_NAMES),
    )
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--min-decode-samples-per-s", type=float, default=0.0)
    parser.add_argument("--diagnostic-64", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = verify_joint_bev_dataset(
        args.dataset_root,
        splits=args.splits,
        max_samples_per_split=(
            args.max_samples_per_split or None
        ),
        chunk_size=args.chunk_size,
        min_decode_samples_per_s=args.min_decode_samples_per_s,
        diagnostic_64=args.diagnostic_64,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "JointBEVVerificationError",
    "verify_joint_bev_dataset",
]
