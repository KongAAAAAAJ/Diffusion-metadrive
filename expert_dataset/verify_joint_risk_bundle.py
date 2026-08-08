"""Cross-store verifier for the unified joint-BEV/RiskEntry bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from expert_dataset.joint_risk_bundle_contract import (
    BUNDLE_FORMAT,
    BUNDLE_SCHEMA_VERSION,
    bundle_protocol_sha256,
)
from expert_dataset.joint_risk_bundle_storage import (
    BundleEpisodeResult,
    JointRiskBundleStorageError,
)
from expert_dataset.verify_joint_bev_dataset import verify_joint_bev_dataset
from expert_dataset.verify_riskentry_sidecar import verify_riskentry_sidecar_dataset
from scenarios.bev_round13_contract import primary_scenario_contract


class JointRiskBundleVerificationError(RuntimeError):
    """Raised when component datasets cannot be joined losslessly."""


def _read_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JointRiskBundleVerificationError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise JointRiskBundleVerificationError(f"JSON root must be an object: {path}")
    return payload


def _episode_path(root: Path, split: str, episode_index: int) -> Path:
    return root / split / "episodes" / f"episode_{episode_index:08d}"


def verify_joint_risk_bundle(bundle_root: Path | str) -> dict[str, object]:
    root = Path(bundle_root).expanduser()
    manifest = _read_object(root / "dataset_bundle_manifest.json")
    if (root / ".bundle_episode_pending.json").exists():
        raise JointRiskBundleVerificationError(
            "bundle has an unfinished pending episode transaction"
        )
    required_manifest = {
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
    }
    if set(manifest) != required_manifest:
        raise JointRiskBundleVerificationError("bundle manifest fields mismatch")
    if (
        manifest["format"] != BUNDLE_FORMAT
        or manifest["schema_version"] != BUNDLE_SCHEMA_VERSION
        or manifest["protocol_sha256"] != bundle_protocol_sha256()
        or manifest["scenario_contract_sha256"]
        != primary_scenario_contract()["sha256"]
        or float(manifest["decision_dt_s"]) != 0.1
        or int(manifest["split_seed"]) != 17
    ):
        raise JointRiskBundleVerificationError("bundle protocol binding mismatch")
    base_root = root / str(manifest["base_directory"])
    sidecar_root = root / str(manifest["sidecar_directory"])
    if base_root == sidecar_root:
        raise JointRiskBundleVerificationError("base and sidecar roots overlap")

    base_report = verify_joint_bev_dataset(base_root)
    sidecar_report = verify_riskentry_sidecar_dataset(sidecar_root)
    base_contract = _read_object(base_root / "dataset_contract.json")
    if base_contract.get("dataset_fingerprint") != manifest["base_dataset_fingerprint"]:
        raise JointRiskBundleVerificationError("base fingerprint mismatch")
    if sidecar_report["sidecar_dataset_fingerprint"] != manifest["sidecar_dataset_fingerprint"]:
        raise JointRiskBundleVerificationError("sidecar fingerprint mismatch")
    if sidecar_report["base_dataset_fingerprint"] != manifest["base_dataset_fingerprint"]:
        raise JointRiskBundleVerificationError("sidecar/base fingerprint join mismatch")

    index_path = root / "bundle_episode_index.jsonl"
    try:
        lines = index_path.read_text(encoding="utf-8").splitlines()
        rows = []
        for line in lines:
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise JointRiskBundleVerificationError("bundle row must be an object")
            rows.append(BundleEpisodeResult.from_mapping(payload))
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        JointRiskBundleStorageError,
    ) as exc:
        raise JointRiskBundleVerificationError("invalid bundle episode index") from exc
    if [row.episode_index for row in rows] != list(range(len(rows))):
        raise JointRiskBundleVerificationError("bundle rows must be contiguous")

    committed_base = 0
    committed_sidecar = 0
    sidecar_only = 0
    for row in rows:
        base_path = _episode_path(base_root, row.split, row.episode_index)
        sidecar_path = _episode_path(sidecar_root, row.split, row.episode_index)
        if base_path.is_dir() != (row.base_status == "committed"):
            raise JointRiskBundleVerificationError("bundle/base status mismatch")
        if sidecar_path.is_dir() != (row.sidecar_status == "committed"):
            raise JointRiskBundleVerificationError("bundle/sidecar status mismatch")
        if row.base_status == "committed":
            committed_base += 1
            base_metadata = _read_object(base_path / "episode.json")
            sidecar_metadata = _read_object(sidecar_path / "episode.json")
            attributes = base_metadata.get("attributes")
            if not isinstance(attributes, Mapping):
                raise JointRiskBundleVerificationError("base episode attributes missing")
            required_equal = {
                "scenario_id": row.scenario_id,
                "local_route": row.local_route,
                "spawn_seed": row.spawn_seed,
            }
            for name, expected in required_equal.items():
                if attributes.get(name) != expected or sidecar_metadata.get(name) != expected:
                    raise JointRiskBundleVerificationError(
                        f"episode join metadata mismatch: {name}"
                    )
            if attributes.get("sidecar_dataset_fingerprint") != manifest["sidecar_dataset_fingerprint"]:
                raise JointRiskBundleVerificationError("base sidecar fingerprint binding mismatch")
            if attributes.get("scenario_contract_sha256") != manifest["scenario_contract_sha256"]:
                raise JointRiskBundleVerificationError("base scenario hash mismatch")
            sidecar_parameters = sidecar_metadata.get("scenario_parameters")
            if not isinstance(sidecar_parameters, Mapping) or sidecar_parameters.get(
                "scenario_contract_sha256"
            ) != manifest["scenario_contract_sha256"]:
                raise JointRiskBundleVerificationError("sidecar scenario hash mismatch")
            mapping = np.load(
                sidecar_path / "base_sample_step_index.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            selected = np.asarray(attributes.get("selected_sample_steps", []), dtype=np.int64)
            if not np.array_equal(mapping, selected):
                raise JointRiskBundleVerificationError("base sample/raw step mapping mismatch")
            if len(mapping) != int(base_metadata["joint_samples"]):
                raise JointRiskBundleVerificationError("mapping/base sample count mismatch")
            if int(attributes.get("raw_timeline_length", -1)) != row.raw_steps:
                raise JointRiskBundleVerificationError("base raw timeline length mismatch")
        if row.sidecar_status == "committed":
            committed_sidecar += 1
            sidecar_metadata = _read_object(sidecar_path / "episode.json")
            raw_steps = int(
                np.load(sidecar_path / "step_index.npy", mmap_mode="r", allow_pickle=False).shape[0]
            )
            base_steps = int(
                np.load(
                    sidecar_path / "base_sample_step_index.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                ).shape[0]
            )
            if raw_steps != row.raw_steps or base_steps != row.base_samples:
                raise JointRiskBundleVerificationError("bundle row component counts mismatch")
            if sidecar_metadata["retention"]["outcome"] != row.outcome:
                raise JointRiskBundleVerificationError("bundle sidecar outcome mismatch")
            if row.base_status == "rejected":
                sidecar_only += 1
                if base_steps != 0:
                    raise JointRiskBundleVerificationError(
                        "sidecar-only episode must have an empty base mapping"
                    )

    base_state = _read_object(base_root / "collection_state.json")
    if int(base_state.get("next_episode_index", -1)) != len(rows):
        raise JointRiskBundleVerificationError("bundle/base attempted episode count mismatch")
    if committed_base != int(base_report["episodes"]):
        raise JointRiskBundleVerificationError("unindexed base episode detected")
    if committed_sidecar != int(sidecar_report["episodes"]):
        raise JointRiskBundleVerificationError("unindexed sidecar episode detected")

    return {
        "format": BUNDLE_FORMAT,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "bundle_root": str(root.resolve()),
        "bundle_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "attempted_episodes": len(rows),
        "committed_base_episodes": committed_base,
        "committed_sidecar_episodes": committed_sidecar,
        "sidecar_only_episodes": sidecar_only,
        "base_joint_samples": base_report["joint_samples"],
        "sidecar_raw_steps": sidecar_report["raw_steps"],
        "base_dataset_fingerprint": manifest["base_dataset_fingerprint"],
        "sidecar_dataset_fingerprint": manifest["sidecar_dataset_fingerprint"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_root", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(verify_joint_risk_bundle(args.bundle_root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["JointRiskBundleVerificationError", "verify_joint_risk_bundle"]
