"""Round 13.97e end-to-end readiness audit for a shared dataset bundle."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Mapping

from evaluation.round13_closeout import (
    _read_json as read_round13_json,
    verify_round13_closeout,
)
from expert_dataset.joint_risk_bundle_contract import load_bundle_protocol
from expert_dataset.verify_joint_bev_dataset import verify_joint_bev_dataset
from expert_dataset.verify_joint_risk_bundle import verify_joint_risk_bundle
from expert_dataset.verify_riskentry_sidecar import verify_riskentry_sidecar_dataset
from scenarios.bev_round13_contract import PRIMARY_S5_S9_SCENARIOS


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_PILOT_MINIMUM_JOINT_SAMPLES = 15_000


class Round1397ePilotError(RuntimeError):
    """Raised when a bundle is not ready for the next collection gate."""


def _read_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Round1397ePilotError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise Round1397ePilotError(f"JSON root must be an object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    encoded = (
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_bundle_rows(bundle_root: Path) -> tuple[dict[str, object], ...]:
    try:
        lines = (bundle_root / "bundle_episode_index.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        rows = tuple(json.loads(line) for line in lines)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Round1397ePilotError("unable to read bundle episode index") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise Round1397ePilotError("bundle episode rows must be objects")
    return rows


def audit_shared_bundle_pilot(
    bundle_root: Path | str,
    *,
    minimum_base_samples: int,
    require_all_s5_s9: bool,
    require_formal_15000: bool,
) -> dict[str, object]:
    root = Path(bundle_root).expanduser().resolve()
    if minimum_base_samples <= 0:
        raise Round1397ePilotError("minimum_base_samples must be positive")
    if (
        require_formal_15000
        and minimum_base_samples < FORMAL_PILOT_MINIMUM_JOINT_SAMPLES
    ):
        raise Round1397ePilotError(
            "formal pilot must require at least 15,000 joint samples"
        )

    protocol = load_bundle_protocol()
    closeout = verify_round13_closeout(
        read_round13_json(REPO_ROOT / "evaluation/ROUND13_96_FREEZE.json"),
        REPO_ROOT,
    )
    bundle = verify_joint_risk_bundle(root)
    manifest = _read_object(root / "dataset_bundle_manifest.json")
    base_root = root / str(manifest["base_directory"])
    sidecar_root = root / str(manifest["sidecar_directory"])
    base = verify_joint_bev_dataset(
        base_root,
        min_decode_samples_per_s=50.0,
    )
    sidecar = verify_riskentry_sidecar_dataset(sidecar_root)
    rows = _load_bundle_rows(root)

    expected_scenarios = tuple(name for name, _ in PRIMARY_S5_S9_SCENARIOS)
    scenario_samples = {
        name: int(base["scenario_joint_samples"].get(name, 0))
        for name in expected_scenarios
    }
    attempted_scenarios = Counter(str(row["scenario_id"]) for row in rows)
    missing_scenarios = [
        name for name, count in scenario_samples.items() if count == 0
    ]
    if require_all_s5_s9 and missing_scenarios:
        raise Round1397ePilotError(
            f"pilot has no committed base sample for scenarios: {missing_scenarios}"
        )
    if int(bundle["base_joint_samples"]) < minimum_base_samples:
        raise Round1397ePilotError(
            f"pilot has {bundle['base_joint_samples']} base samples, "
            f"requires {minimum_base_samples}"
        )
    if int(bundle["attempted_episodes"]) == 0:
        raise Round1397ePilotError("pilot contains no attempted episodes")
    if int(bundle["committed_base_episodes"]) == 0:
        raise Round1397ePilotError("pilot contains no committed base episode")
    if int(bundle["committed_sidecar_episodes"]) < int(
        bundle["committed_base_episodes"]
    ):
        raise Round1397ePilotError("sidecar coverage is below base coverage")

    non_empty_splits = [
        split
        for split, values in base["splits"].items()
        if int(values["joint_samples"]) > 0
    ]
    if require_formal_15000 and set(non_empty_splits) != {"train", "val", "test"}:
        raise Round1397ePilotError(
            "formal 15k pilot requires non-empty train/val/test splits"
        )

    attempted = int(bundle["attempted_episodes"])
    committed = int(bundle["committed_base_episodes"])
    formal_gate_passed = bool(
        require_formal_15000
        and int(bundle["base_joint_samples"])
        >= FORMAL_PILOT_MINIMUM_JOINT_SAMPLES
        and not missing_scenarios
        and set(non_empty_splits) == {"train", "val", "test"}
    )
    return {
        "format": "bev_risk_shared_bundle_pilot_report_v1",
        "round": "13.97e",
        "status": (
            "formal_15000_pilot_accepted"
            if formal_gate_passed
            else "protocol_pilot_accepted"
        ),
        "diagnostic_only": not formal_gate_passed,
        "eligible_for_stage1_formal_training": False,
        "eligible_for_formal_50k_collection": formal_gate_passed,
        "bundle_root": str(root),
        "minimum_base_samples": int(minimum_base_samples),
        "base_joint_samples": int(bundle["base_joint_samples"]),
        "attempted_episodes": attempted,
        "committed_base_episodes": committed,
        "committed_sidecar_episodes": int(bundle["committed_sidecar_episodes"]),
        "sidecar_only_episodes": int(bundle["sidecar_only_episodes"]),
        "sidecar_raw_steps": int(bundle["sidecar_raw_steps"]),
        "base_episode_acceptance_rate": committed / attempted,
        "sidecar_episode_retention_rate": int(
            bundle["committed_sidecar_episodes"]
        )
        / attempted,
        "scenario_joint_samples": scenario_samples,
        "scenario_attempts": {
            name: int(attempted_scenarios[name]) for name in expected_scenarios
        },
        "missing_scenarios": missing_scenarios,
        "non_empty_splits": non_empty_splits,
        "outcomes": sidecar["outcomes"],
        "events": sidecar["events"],
        "decode_joint_samples_per_s": float(base["decode_joint_samples_per_s"]),
        "base_dataset_fingerprint": bundle["base_dataset_fingerprint"],
        "sidecar_dataset_fingerprint": bundle["sidecar_dataset_fingerprint"],
        "bundle_index_sha256": bundle["bundle_sha256"],
        "scenario_contract_sha256": manifest["scenario_contract_sha256"],
        "protocol_schema_version": protocol["schema_version"],
        "round13_closeout_status": closeout["status"],
        "next_gate": (
            "formal_50k_joint_bundle_collection"
            if formal_gate_passed
            else "formal_joint_bev_15000_step_pilot"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_root", type=Path)
    parser.add_argument("--minimum-base-samples", type=int, default=10)
    parser.add_argument("--require-all-s5-s9", action="store_true")
    parser.add_argument("--require-formal-15000", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = audit_shared_bundle_pilot(
        args.bundle_root,
        minimum_base_samples=args.minimum_base_samples,
        require_all_s5_s9=args.require_all_s5_s9,
        require_formal_15000=args.require_formal_15000,
    )
    if args.output is not None:
        _atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FORMAL_PILOT_MINIMUM_JOINT_SAMPLES",
    "Round1397ePilotError",
    "audit_shared_bundle_pilot",
]
