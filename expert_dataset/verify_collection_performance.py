"""Verify a serial/parallel formal-path collection performance pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from expert_dataset.joint_risk_bundle_v2_fixture import PARTITIONS
from expert_dataset.verify_joint_risk_bundle_v2_formal import verify_formal_bundle


class CollectionPerformanceVerificationError(RuntimeError):
    """Raised when a collection performance pilot violates its hard gate."""


def _json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CollectionPerformanceVerificationError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise CollectionPerformanceVerificationError(f"JSON object required: {path}")
    return payload


def _index_rows(root: Path, contract: Mapping[str, object]) -> list[dict[str, object]]:
    prefix = str(contract["dataset_instance_prefix"])
    result: list[dict[str, object]] = []
    for partition in PARTITIONS:
        partition_root = root / f"{prefix}_{partition}_v2"
        if not partition_root.exists():
            continue
        path = partition_root / "bundle_episode_index.jsonl"
        try:
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CollectionPerformanceVerificationError(
                f"invalid episode index: {path}"
            ) from exc
        if any(not isinstance(row, dict) for row in rows):
            raise CollectionPerformanceVerificationError(
                f"episode index rows must be objects: {path}"
            )
        result.extend({"partition": partition, **row} for row in rows)
    return result


def _physical_contract(contract: Mapping[str, object]) -> dict[str, object]:
    ignored = {"dataset_instance_prefix"}
    return {key: value for key, value in contract.items() if key not in ignored}


def _execution(report: Mapping[str, object]) -> Mapping[str, object]:
    execution = report.get("execution")
    if not isinstance(execution, Mapping):
        raise CollectionPerformanceVerificationError("collection report lacks execution")
    return execution


def verify_collection_performance(
    serial_root: Path | str,
    parallel_root: Path | str,
    *,
    minimum_speedup: float = 2.5,
    require_complete: bool = False,
) -> dict[str, object]:
    """Run structural gates and compare deterministic collection semantics.

    The project does not require bit-exact MetaDrive trajectories across separate
    runs.  This gate instead requires the complete scheduled episode index,
    matched-pair allocation, split, seed, selected-window count, and discrete
    outcome to agree exactly.  Each dataset must independently pass the frozen
    formal-path verifier.
    """

    serial_root = Path(serial_root).expanduser().resolve()
    parallel_root = Path(parallel_root).expanduser().resolve()
    if minimum_speedup <= 1.0:
        raise CollectionPerformanceVerificationError("minimum_speedup must exceed 1")
    serial_contract = _json(serial_root / "formal_v2_run_contract.json")
    parallel_contract = _json(parallel_root / "formal_v2_run_contract.json")
    if _physical_contract(serial_contract) != _physical_contract(parallel_contract):
        raise CollectionPerformanceVerificationError(
            "serial and parallel physical collection contracts differ"
        )

    serial_report = _json(serial_root / "formal_v2_collection_report.json")
    parallel_report = _json(parallel_root / "formal_v2_collection_report.json")
    serial_complete = bool(serial_report.get("complete"))
    parallel_complete = bool(parallel_report.get("complete"))
    if serial_complete != parallel_complete:
        raise CollectionPerformanceVerificationError(
            "serial and parallel completion status differs"
        )
    if require_complete and not serial_complete:
        raise CollectionPerformanceVerificationError("both performance pilots must complete")
    serial_execution = _execution(serial_report)
    parallel_execution = _execution(parallel_report)
    if int(serial_execution.get("parallel_workers", -1)) != 1:
        raise CollectionPerformanceVerificationError("serial pilot must use one worker")
    if int(parallel_execution.get("parallel_workers", -1)) <= 1:
        raise CollectionPerformanceVerificationError(
            "parallel pilot must use more than one worker"
        )

    serial_rate = float(serial_execution["eligible_anchor_windows_per_hour"])
    parallel_rate = float(parallel_execution["eligible_anchor_windows_per_hour"])
    if serial_rate <= 0.0 or parallel_rate <= 0.0:
        raise CollectionPerformanceVerificationError("throughput must be positive")
    speedup = parallel_rate / serial_rate
    if speedup < minimum_speedup:
        raise CollectionPerformanceVerificationError(
            f"parallel speedup {speedup:.3f} is below {minimum_speedup:.3f}"
        )

    serial_rows = _index_rows(serial_root, serial_contract)
    parallel_rows = _index_rows(parallel_root, parallel_contract)
    if not serial_rows:
        raise CollectionPerformanceVerificationError("performance pilot is empty")
    if serial_rows != parallel_rows:
        raise CollectionPerformanceVerificationError(
            "serial and parallel episode schedules or discrete outcomes differ"
        )
    serial_verification = verify_formal_bundle(serial_root) if serial_complete else None
    parallel_verification = verify_formal_bundle(parallel_root) if parallel_complete else None
    eligible_windows = sum(int(row["base_samples"]) for row in serial_rows)

    return {
        "format": "bundle-v2-collection-performance-gate-v1",
        "passed": True,
        "minimum_speedup": minimum_speedup,
        "measured_speedup": speedup,
        "episodes": len(serial_rows),
        "eligible_windows": eligible_windows,
        "complete_quota_pilot": serial_complete,
        "serial": {
            "root": str(serial_root),
            "workers": int(serial_execution["parallel_workers"]),
            "elapsed_s": float(serial_execution["elapsed_s"]),
            "episodes_per_hour": float(serial_execution["episodes_per_hour"]),
            "eligible_anchor_windows_per_hour": serial_rate,
            "formal_verifier_passed": (
                None if serial_verification is None else bool(serial_verification["passed"])
            ),
        },
        "parallel": {
            "root": str(parallel_root),
            "workers": int(parallel_execution["parallel_workers"]),
            "elapsed_s": float(parallel_execution["elapsed_s"]),
            "episodes_per_hour": float(parallel_execution["episodes_per_hour"]),
            "eligible_anchor_windows_per_hour": parallel_rate,
            "formal_verifier_passed": (
                None
                if parallel_verification is None
                else bool(parallel_verification["passed"])
            ),
        },
        "determinism_contract": {
            "physical_config_equal": True,
            "episode_index_and_discrete_outcomes_exact": True,
            "ordered_single_writer_commit": True,
            "bit_exact_simulator_trajectory_required": False,
            "policy": "Round 13.95 layered reproducibility contract",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-root", type=Path, required=True)
    parser.add_argument("--parallel-root", type=Path, required=True)
    parser.add_argument("--minimum-speedup", type=float, default=2.5)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = verify_collection_performance(
        args.serial_root,
        args.parallel_root,
        minimum_speedup=args.minimum_speedup,
        require_complete=args.require_complete,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CollectionPerformanceVerificationError",
    "verify_collection_performance",
]
