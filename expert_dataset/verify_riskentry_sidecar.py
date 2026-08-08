"""Strict verifier for the RiskEntry MetaDrive actor sidecar dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from expert_dataset.joint_bev_storage import SPLIT_NAMES
from expert_dataset.riskentry_sidecar_storage import (
    EPISODE_FILE_NAMES,
    EPISODE_PATTERN,
    SIDECAR_ARRAY_DTYPES,
    SIDECAR_FORMAT,
    SIDECAR_SCHEMA_VERSION,
    RiskEntrySidecarStorageError,
    sidecar_dataset_contract,
    sidecar_dataset_fingerprint,
    validate_sidecar_episode_payload,
)


class RiskEntrySidecarVerificationError(RuntimeError):
    """Raised when a persisted sidecar fails full verification."""


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RiskEntrySidecarVerificationError(f"invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise RiskEntrySidecarVerificationError(f"JSON root must be an object: {path}")
    return payload


def validate_sidecar_dataset_contract(dataset_root: Path | str) -> dict[str, object]:
    root = Path(dataset_root).expanduser()
    contract = _read_json(root / "dataset_contract.json")
    base_fingerprint = str(contract.get("base_dataset_fingerprint", ""))
    try:
        expected = sidecar_dataset_contract(base_fingerprint)
    except RiskEntrySidecarStorageError as exc:
        raise RiskEntrySidecarVerificationError(str(exc)) from exc
    if contract != expected:
        raise RiskEntrySidecarVerificationError("sidecar dataset contract mismatch")
    return contract


def _load_episode_arrays(path: Path) -> dict[str, np.memmap]:
    actual_names = {item.name for item in path.iterdir()}
    if actual_names != EPISODE_FILE_NAMES:
        raise RiskEntrySidecarVerificationError(
            f"episode file set mismatch at {path}: "
            f"missing={sorted(EPISODE_FILE_NAMES - actual_names)}, "
            f"unexpected={sorted(actual_names - EPISODE_FILE_NAMES)}"
        )
    arrays: dict[str, np.memmap] = {}
    for name in SIDECAR_ARRAY_DTYPES:
        array_path = path / f"{name}.npy"
        try:
            array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise RiskEntrySidecarVerificationError(
                f"unable to mmap sidecar array: {array_path}"
            ) from exc
        if not isinstance(array, np.memmap):
            raise RiskEntrySidecarVerificationError(
                f"sidecar array is not mmap-readable: {array_path}"
            )
        arrays[name] = array
    return arrays


def _manifest_entry(
    metadata: Mapping[str, object], arrays: Mapping[str, np.ndarray], directory: str
) -> dict[str, object]:
    retention = metadata.get("retention")
    if not isinstance(retention, Mapping):
        raise RiskEntrySidecarVerificationError("episode retention must be an object")
    return {
        "episode_index": int(metadata["episode_index"]),
        "directory": directory,
        "raw_steps": int(len(arrays["step_index"])),
        "actor_count": int(arrays["actor_state"].shape[1]),
        "base_samples": int(len(arrays["base_sample_step_index"])),
        "outcome": str(retention["outcome"]),
    }


def verify_riskentry_sidecar_dataset(
    dataset_root: Path | str,
    *,
    splits: Sequence[str] = SPLIT_NAMES,
) -> dict[str, object]:
    root = Path(dataset_root).expanduser()
    if not root.is_dir():
        raise RiskEntrySidecarVerificationError(
            f"sidecar dataset root does not exist: {root}"
        )
    selected = tuple(str(split) for split in splits)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(split not in SPLIT_NAMES for split in selected)
    ):
        raise RiskEntrySidecarVerificationError(
            f"splits must be unique values from {SPLIT_NAMES}"
        )
    contract = validate_sidecar_dataset_contract(root)
    allowed_root = {"dataset_contract.json", ".writer.lock", *SPLIT_NAMES}
    unexpected_root = []
    for path in root.iterdir():
        if path.name in allowed_root or path.name.startswith(".dataset_contract.json.tmp-"):
            continue
        unexpected_root.append(path.name)
    if unexpected_root:
        raise RiskEntrySidecarVerificationError(
            f"unexpected sidecar root entries: {sorted(unexpected_root)}"
        )

    episode_owner: dict[int, str] = {}
    report_splits: dict[str, object] = {}
    total_raw_steps = 0
    total_base_samples = 0
    total_actors = 0
    total_lanes = 0
    total_payload_bytes = 0
    global_outcomes: Counter[str] = Counter()
    global_events: Counter[str] = Counter()
    scenario_hashes: set[str] = set()

    for split in selected:
        split_root = root / split
        episodes_root = split_root / "episodes"
        manifest_path = split_root / "manifest.json"
        if not episodes_root.is_dir() or not manifest_path.is_file():
            raise RiskEntrySidecarVerificationError(
                f"split {split} lacks episodes/manifest.json"
            )
        allowed_split = {"episodes", "manifest.json"}
        unexpected_split = []
        for path in split_root.iterdir():
            if path.name in allowed_split or path.name.startswith(".manifest.json.tmp-"):
                continue
            unexpected_split.append(path.name)
        if unexpected_split:
            raise RiskEntrySidecarVerificationError(
                f"unexpected entries in {split}: {sorted(unexpected_split)}"
            )
        expected_entries: list[dict[str, object]] = []
        ignored_temporary: list[str] = []
        split_outcomes: Counter[str] = Counter()
        split_events: Counter[str] = Counter()
        split_raw_steps = 0
        split_base_samples = 0
        split_actors = 0
        split_lanes = 0
        active_actor_observations = 0
        derivative_observations = 0
        lane_observations = 0
        split_payload_bytes = 0
        for episode_path in sorted(episodes_root.iterdir()):
            if (
                episode_path.is_dir()
                and episode_path.name.startswith(".episode_")
                and ".tmp-" in episode_path.name
            ):
                ignored_temporary.append(episode_path.name)
                continue
            if not episode_path.is_dir():
                raise RiskEntrySidecarVerificationError(
                    f"unexpected file in episode root: {episode_path}"
                )
            match = EPISODE_PATTERN.fullmatch(episode_path.name)
            if match is None:
                raise RiskEntrySidecarVerificationError(
                    f"invalid episode directory name: {episode_path.name}"
                )
            episode_index = int(match.group(1))
            previous = episode_owner.get(episode_index)
            if previous is not None:
                raise RiskEntrySidecarVerificationError(
                    f"episode {episode_index} appears in both {previous} and {split}"
                )
            metadata = _read_json(episode_path / "episode.json")
            arrays = _load_episode_arrays(episode_path)
            try:
                validate_sidecar_episode_payload(metadata, arrays)
            except RiskEntrySidecarStorageError as exc:
                raise RiskEntrySidecarVerificationError(
                    f"episode {episode_index}: {exc}"
                ) from exc
            if (
                int(metadata.get("episode_index", -1)) != episode_index
                or metadata.get("split") != split
                or metadata.get("base_dataset_fingerprint")
                != contract["base_dataset_fingerprint"]
            ):
                raise RiskEntrySidecarVerificationError(
                    f"episode {episode_index} identity/provenance mismatch"
                )
            episode_owner[episode_index] = split
            entry = _manifest_entry(metadata, arrays, episode_path.name)
            expected_entries.append(entry)
            scenario_parameters = metadata["scenario_parameters"]
            scenario_hashes.add(str(scenario_parameters["scenario_contract_sha256"]))
            event_counts = Counter(
                str(event["event_type"]) for event in metadata["events"]
            )
            split_events.update(event_counts)
            split_outcomes[str(entry["outcome"])] += 1
            raw_steps = int(entry["raw_steps"])
            base_samples = int(entry["base_samples"])
            actor_count = int(entry["actor_count"])
            lane_count = len(metadata["lanes"])
            split_raw_steps += raw_steps
            split_base_samples += base_samples
            split_actors += actor_count
            split_lanes += lane_count
            actor_valid = np.asarray(arrays["actor_valid_mask"])
            state_valid = np.asarray(arrays["actor_state_valid_mask"])
            lane_valid = np.asarray(arrays["lane_valid_mask"])
            active_actor_observations += int(actor_valid.sum())
            derivative_observations += int(
                np.logical_and(actor_valid, state_valid[..., 7]).sum()
            )
            lane_observations += int(lane_valid.sum())
            for name in SIDECAR_ARRAY_DTYPES:
                split_payload_bytes += int((episode_path / f"{name}.npy").stat().st_size)

        expected_entries.sort(key=lambda item: int(item["episode_index"]))
        manifest = _read_json(manifest_path)
        expected_manifest = {
            "format": SIDECAR_FORMAT,
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "split": split,
            "episode_count": len(expected_entries),
            "raw_steps": split_raw_steps,
            "base_samples": split_base_samples,
            "episodes": expected_entries,
        }
        if manifest != expected_manifest:
            raise RiskEntrySidecarVerificationError(
                f"{split} manifest does not match committed episode directories"
            )
        report_splits[split] = {
            "episodes": len(expected_entries),
            "raw_steps": split_raw_steps,
            "base_samples": split_base_samples,
            "actor_table_rows": split_actors,
            "lane_table_rows": split_lanes,
            "outcomes": dict(sorted(split_outcomes.items())),
            "events": dict(sorted(split_events.items())),
            "actor_observations": active_actor_observations,
            "derivative_valid_rate": (
                float(derivative_observations / active_actor_observations)
                if active_actor_observations
                else 0.0
            ),
            "lane_valid_rate": (
                float(lane_observations / active_actor_observations)
                if active_actor_observations
                else 0.0
            ),
            "payload_bytes": split_payload_bytes,
            "ignored_temporary_directories": ignored_temporary,
        }
        total_raw_steps += split_raw_steps
        total_base_samples += split_base_samples
        total_actors += split_actors
        total_lanes += split_lanes
        total_payload_bytes += split_payload_bytes
        global_outcomes.update(split_outcomes)
        global_events.update(split_events)

    if len(scenario_hashes) > 1:
        raise RiskEntrySidecarVerificationError(
            "episodes use more than one scenario contract SHA256"
        )
    return {
        "format": SIDECAR_FORMAT,
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "dataset_root": str(root.resolve()),
        "base_dataset_fingerprint": contract["base_dataset_fingerprint"],
        "sidecar_dataset_fingerprint": sidecar_dataset_fingerprint(
            str(contract["base_dataset_fingerprint"])
        ),
        "verified_splits": list(selected),
        "complete_scan": True,
        "episodes": len(episode_owner),
        "raw_steps": total_raw_steps,
        "base_samples": total_base_samples,
        "actor_table_rows": total_actors,
        "lane_table_rows": total_lanes,
        "outcomes": dict(sorted(global_outcomes.items())),
        "events": dict(sorted(global_events.items())),
        "scenario_contract_sha256": (
            next(iter(scenario_hashes)) if scenario_hashes else None
        ),
        "payload_bytes": total_payload_bytes,
        "splits": report_splits,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=SPLIT_NAMES, default=list(SPLIT_NAMES)
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = verify_riskentry_sidecar_dataset(args.dataset_root, splits=args.splits)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RiskEntrySidecarVerificationError",
    "main",
    "validate_sidecar_dataset_contract",
    "verify_riskentry_sidecar_dataset",
]
