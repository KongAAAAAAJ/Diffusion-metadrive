from __future__ import annotations

import json
from pathlib import Path

import pytest

from expert_dataset.verify_collection_performance import (
    CollectionPerformanceVerificationError,
    verify_collection_performance,
)


def _write_pilot(root: Path, *, prefix: str, workers: int, rate: float) -> None:
    root.mkdir()
    contract = {
        "format": "test",
        "schema_version": 1,
        "run_mode": "formal_pilot",
        "target_windows": {"id": 8, "compositional_ood": 8, "topology_ood": 8},
        "dataset_instance_prefix": prefix,
    }
    (root / "formal_v2_run_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    report = {
        "complete": True,
        "execution": {
            "parallel_workers": workers,
            "elapsed_s": 10.0,
            "episodes_per_hour": rate,
            "eligible_anchor_windows_per_hour": rate,
        },
    }
    (root / "formal_v2_collection_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    row = {
        "episode_index": 0,
        "split": "test",
        "spawn_seed": 17,
        "matched_pair_id": "pair_0",
        "outcome": "success",
        "base_samples": 8,
    }
    for partition in ("id", "compositional_ood", "topology_ood"):
        partition_root = root / f"{prefix}_{partition}_v2"
        partition_root.mkdir()
        (partition_root / "bundle_episode_index.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8"
        )


def test_performance_gate_requires_speedup_and_exact_discrete_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    _write_pilot(serial, prefix="serial", workers=1, rate=100.0)
    _write_pilot(parallel, prefix="parallel", workers=4, rate=300.0)
    monkeypatch.setattr(
        "expert_dataset.verify_collection_performance.verify_formal_bundle",
        lambda root: {"passed": True, "eligible_windows_total": 24},
    )
    report = verify_collection_performance(serial, parallel, minimum_speedup=2.5)
    assert report["passed"] is True
    assert report["measured_speedup"] == pytest.approx(3.0)
    assert report["determinism_contract"][
        "episode_index_and_discrete_outcomes_exact"
    ] is True

    parallel_report = json.loads(
        (parallel / "formal_v2_collection_report.json").read_text(encoding="utf-8")
    )
    parallel_report["execution"]["eligible_anchor_windows_per_hour"] = 200.0
    (parallel / "formal_v2_collection_report.json").write_text(
        json.dumps(parallel_report), encoding="utf-8"
    )
    with pytest.raises(CollectionPerformanceVerificationError, match="speedup"):
        verify_collection_performance(serial, parallel, minimum_speedup=2.5)


def test_performance_gate_rejects_changed_discrete_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    _write_pilot(serial, prefix="serial", workers=1, rate=100.0)
    _write_pilot(parallel, prefix="parallel", workers=4, rate=300.0)
    index = parallel / "parallel_id_v2" / "bundle_episode_index.jsonl"
    row = json.loads(index.read_text(encoding="utf-8"))
    row["outcome"] = "collision"
    index.write_text(json.dumps(row) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        "expert_dataset.verify_collection_performance.verify_formal_bundle",
        lambda root: {"passed": True, "eligible_windows_total": 24},
    )
    with pytest.raises(CollectionPerformanceVerificationError, match="discrete"):
        verify_collection_performance(serial, parallel)
