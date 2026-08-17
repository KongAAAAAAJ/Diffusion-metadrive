from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from expert_dataset.riskentry_sidecar_adapter import (
    SidecarActorRecord,
    SidecarActorSnapshot,
    SidecarLaneRecord,
    SidecarRawEvent,
)
from expert_dataset.riskentry_sidecar_storage import (
    EPISODE_FILE_NAMES,
    RiskEntrySidecarDatasetStore,
    RiskEntrySidecarStorageError,
    SIDECAR_ARRAY_DTYPES,
    SidecarEpisodeStart,
    sidecar_dataset_contract,
    sidecar_dataset_fingerprint,
)
from expert_dataset.verify_riskentry_sidecar import (
    RiskEntrySidecarVerificationError,
    main,
    verify_riskentry_sidecar_dataset,
)


BASE_FINGERPRINT = hashlib.sha256(b"base-joint-bev-fixture").hexdigest()
SCENARIO_HASH = hashlib.sha256(b"scenario-contract-fixture").hexdigest()


def _metadata(index: int, split: str = "train") -> SidecarEpisodeStart:
    return SidecarEpisodeStart(
        episode_index=index,
        split=split,
        scenario_id="S5_hard_brake_lead",
        local_route="R1_entry_straight",
        spawn_seed=17 + index,
        decision_dt_s=0.1,
        base_dataset_fingerprint=BASE_FINGERPRINT,
        scenario_parameters={
            "scenario_contract_sha256": SCENARIO_HASH,
            "brake_deceleration_mps2": 6.0,
        },
    )


def _actor_records(*, include_external: bool = True):
    records = [
        SidecarActorRecord(0, "P0", "agent0", "platoon", "leader", 0, 5.74, 2.3),
        SidecarActorRecord(1, "P1", "agent1", "platoon", "middle", 0, 5.74, 2.3),
        SidecarActorRecord(2, "P2", "agent2", "platoon", "rear", 0, 5.74, 2.3),
    ]
    if include_external:
        records.append(
            SidecarActorRecord(3, "V000", "traffic-uuid", "external", None, 0, 4.5, 1.8)
        )
    return tuple(records)


LANES = (SidecarLaneRecord(0, "L000", '["A","B",0]', "mainline"),)


def _snapshot(record: SidecarActorRecord, step: int, *, derivative_valid: bool):
    x = 20.0 - 8.0 * record.actor_index + 0.4 * step
    return SidecarActorSnapshot(
        actor_id=record.actor_id,
        source_object_id=record.source_object_id,
        actor_type=record.actor_type,
        world_x_m=x,
        world_y_m=0.0 if record.actor_type == "platoon" else 3.5,
        heading_rad=0.0,
        velocity_x_mps=4.0,
        velocity_y_mps=0.0,
        length_m=record.length_m,
        width_m=record.width_m,
        acceleration_x_mps2=0.0,
        acceleration_y_mps2=0.0,
        yaw_rate_radps=0.0,
        acceleration_valid=derivative_valid,
        yaw_rate_valid=derivative_valid,
        lane_id="L000",
        lane_s_m=x,
        lane_lateral_m=0.0,
        lane_heading_error_rad=0.0,
        lane_width_m=3.5,
        lane_valid=True,
    )


def _append_episode(
    store: RiskEntrySidecarDatasetStore,
    metadata: SidecarEpisodeStart,
    *,
    collision: bool = False,
    base_steps=(1, 2),
):
    records = _actor_records()
    store.begin_episode(metadata)
    store.update_registry(
        actor_records=records,
        lane_records=LANES,
        key_actor_ids={"lead_braker": "V000"},
    )
    for step in range(3):
        store.append_frame(
            step_index=step,
            timestamp_s=step * 0.1,
            actors=tuple(
                _snapshot(record, step, derivative_valid=step > 0)
                for record in records
            ),
        )
        if step == 1:
            store.append_event(
                SidecarRawEvent(
                    "scenario_trigger",
                    1,
                    0.1,
                    details={"scenario_id": metadata.scenario_id},
                )
            )
        if collision and step == 2:
            store.append_event(
                SidecarRawEvent(
                    "out_of_road",
                    2,
                    0.2,
                    actor_ids=("P0",),
                    details={"out_of_road_source": "agent_geometry_checker"},
                )
            )
            store.append_event(
                SidecarRawEvent(
                    "collision_vehicle",
                    2,
                    0.2,
                    actor_ids=("P0", "V000"),
                    terminal=True,
                    details={"source": "fixture"},
                )
            )
            store.append_event(
                SidecarRawEvent(
                    "terminated", 2, 0.2, actor_ids=("P0",), terminal=True
                )
            )
    return store.commit_episode(base_sample_step_indices=base_steps)


def _episode_path(root: Path, split: str, index: int) -> Path:
    return root / split / "episodes" / f"episode_{index:08d}"


def test_atomic_writer_persists_exact_schema_mmap_arrays_and_manifests(tmp_path: Path):
    root = tmp_path / "riskentry_actor_sidecar"
    with RiskEntrySidecarDatasetStore(
        root, base_dataset_fingerprint=BASE_FINGERPRINT, resume=False
    ) as store:
        success = _append_episode(store, _metadata(0, "train"))
        dangerous = _append_episode(
            store,
            _metadata(1, "test"),
            collision=True,
            base_steps=(),
        )
        assert success.outcome == "success"
        assert dangerous.outcome == "collision"
        assert dangerous.base_samples == 0
        expected_sidecar_fingerprint = sidecar_dataset_fingerprint(BASE_FINGERPRINT)
        assert store.dataset_fingerprint == expected_sidecar_fingerprint

    assert json.loads((root / "dataset_contract.json").read_text()) == sidecar_dataset_contract(
        BASE_FINGERPRINT
    )
    for split, index in (("train", 0), ("test", 1)):
        path = _episode_path(root, split, index)
        assert {item.name for item in path.iterdir()} == EPISODE_FILE_NAMES
        for name, dtype in SIDECAR_ARRAY_DTYPES.items():
            array = np.load(path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            assert isinstance(array, np.memmap)
            assert array.dtype == dtype
    collision_metadata = json.loads(
        (_episode_path(root, "test", 1) / "episode.json").read_text()
    )
    assert collision_metadata["retention"] == {
        "outcome": "collision",
        "kept_despite_dangerous_outcome": True,
    }
    assert collision_metadata["key_actor_ids"] == {"lead_braker": "V000"}
    assert collision_metadata["actors"][0]["actor_id"] == "P0"
    assert collision_metadata["actors"][3]["actor_id"] == "V000"
    out_event = next(
        event
        for event in collision_metadata["events"]
        if event["event_type"] == "out_of_road"
    )
    assert out_event["details"]["out_of_road_source"] == "agent_geometry_checker"

    report = verify_riskentry_sidecar_dataset(root)
    assert report["episodes"] == 2
    assert report["raw_steps"] == 6
    assert report["base_samples"] == 2
    assert report["outcomes"] == {"collision": 1, "success": 1}
    assert report["events"]["collision_vehicle"] == 1
    assert report["events"]["out_of_road"] == 1
    assert report["scenario_contract_sha256"] == SCENARIO_HASH


def test_dense_masks_preserve_disappearance_and_reappearance(tmp_path: Path):
    root = tmp_path / "sidecar"
    records = _actor_records()
    with RiskEntrySidecarDatasetStore(
        root, base_dataset_fingerprint=BASE_FINGERPRINT, resume=False
    ) as store:
        store.begin_episode(_metadata(0))
        store.update_registry(
            actor_records=records,
            lane_records=LANES,
            key_actor_ids={"lead_braker": "V000"},
        )
        store.append_frame(
            step_index=0,
            timestamp_s=0.0,
            actors=tuple(_snapshot(record, 0, derivative_valid=False) for record in records),
        )
        store.append_frame(
            step_index=1,
            timestamp_s=0.1,
            actors=tuple(_snapshot(record, 1, derivative_valid=True) for record in records[:3]),
        )
        store.append_event(
            SidecarRawEvent("actor_despawn", 1, 0.1, actor_ids=("V000",))
        )
        store.append_frame(
            step_index=2,
            timestamp_s=0.2,
            actors=tuple(
                _snapshot(record, 2, derivative_valid=(record.actor_id != "V000"))
                for record in records
            ),
        )
        store.commit_episode(base_sample_step_indices=(1,))

    path = _episode_path(root, "train", 0)
    actor_valid = np.load(path / "actor_valid_mask.npy", mmap_mode="r")
    state_valid = np.load(path / "actor_state_valid_mask.npy", mmap_mode="r")
    actor_state = np.load(path / "actor_state.npy", mmap_mode="r")
    lane_index = np.load(path / "lane_index.npy", mmap_mode="r")
    assert actor_valid[:, 3].tolist() == [True, False, True]
    assert not state_valid[1, 3].any()
    assert not state_valid[2, 3, 5:].any()
    assert np.all(actor_state[1, 3] == 0.0)
    assert lane_index[1, 3] == -1
    verify_riskentry_sidecar_dataset(root)


def test_resume_rebuilds_stale_manifest_and_ignores_uncommitted_directory(tmp_path: Path):
    root = tmp_path / "sidecar"
    with RiskEntrySidecarDatasetStore(
        root, base_dataset_fingerprint=BASE_FINGERPRINT, resume=False
    ) as store:
        _append_episode(store, _metadata(0))
    manifest_path = root / "train" / "manifest.json"
    manifest_path.write_text(json.dumps({"stale": True}), encoding="utf-8")
    temporary = root / "train" / "episodes" / ".episode_00000001.tmp-interrupted"
    temporary.mkdir()

    with RiskEntrySidecarDatasetStore(
        root, base_dataset_fingerprint=BASE_FINGERPRINT, resume=True
    ) as resumed:
        assert resumed.summary()["splits"]["train"]["episodes"] == 1
        assert resumed.summary()["splits"]["train"][
            "ignored_temporary_directories"
        ] == [temporary.name]

    manifest = json.loads(manifest_path.read_text())
    assert manifest["episode_count"] == 1
    report = verify_riskentry_sidecar_dataset(root)
    assert report["splits"]["train"]["ignored_temporary_directories"] == [
        temporary.name
    ]


def test_directory_rename_failure_never_creates_commit_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "sidecar"
    import expert_dataset.riskentry_sidecar_storage as storage

    with RiskEntrySidecarDatasetStore(
        root, base_dataset_fingerprint=BASE_FINGERPRINT, resume=False
    ) as store:
        store.begin_episode(_metadata(0))
        records = _actor_records()
        store.update_registry(
            actor_records=records,
            lane_records=LANES,
            key_actor_ids={"lead_braker": "V000"},
        )
        store.append_frame(
            step_index=0,
            timestamp_s=0.0,
            actors=tuple(_snapshot(record, 0, derivative_valid=False) for record in records),
        )
        real_replace = storage.os.replace

        def fail_directory_rename(source, destination):
            if Path(source).is_dir():
                raise OSError("injected directory rename failure")
            return real_replace(source, destination)

        monkeypatch.setattr(storage.os, "replace", fail_directory_rename)
        with pytest.raises(OSError, match="injected"):
            store.commit_episode(base_sample_step_indices=())
        assert not _episode_path(root, "train", 0).exists()
        store.reject_episode(reason_code="atomic_commit_failed")

    monkeypatch.setattr(storage.os, "replace", os.replace)
    with RiskEntrySidecarDatasetStore(
        root, base_dataset_fingerprint=BASE_FINGERPRINT, resume=True
    ) as resumed:
        assert resumed.summary()["splits"]["train"]["episodes"] == 0
        assert len(
            resumed.summary()["splits"]["train"]["ignored_temporary_directories"]
        ) == 1


def test_reject_episode_writes_no_partial_episode(tmp_path: Path):
    root = tmp_path / "sidecar"
    with RiskEntrySidecarDatasetStore(
        root, base_dataset_fingerprint=BASE_FINGERPRINT, resume=False
    ) as store:
        store.begin_episode(_metadata(7, "val"))
        store.reject_episode(reason_code="required_active_state_missing")
        assert store.last_rejection_reason == "required_active_state_missing"
        assert not _episode_path(root, "val", 7).exists()
        assert store.summary()["splits"]["val"]["episodes"] == 0


def test_writer_rejects_bad_mapping_registry_and_contract(tmp_path: Path):
    root = tmp_path / "sidecar"
    with pytest.raises(RiskEntrySidecarStorageError, match="scenario_contract"):
        SidecarEpisodeStart(
            0,
            "train",
            "S5",
            "R1",
            17,
            0.1,
            BASE_FINGERPRINT,
            {},
        )
    with RiskEntrySidecarDatasetStore(
        root, base_dataset_fingerprint=BASE_FINGERPRINT, resume=False
    ) as store:
        store.begin_episode(_metadata(0))
        records = _actor_records()
        store.update_registry(
            actor_records=records,
            lane_records=LANES,
            key_actor_ids={"lead_braker": "V000"},
        )
        store.append_frame(
            step_index=0,
            timestamp_s=0.0,
            actors=tuple(_snapshot(record, 0, derivative_valid=False) for record in records),
        )
        with pytest.raises(RiskEntrySidecarStorageError, match="base sample"):
            store.commit_episode(base_sample_step_indices=(1,))
        store.reject_episode(reason_code="invalid_base_mapping")

    with pytest.raises(RiskEntrySidecarStorageError, match="contract mismatch"):
        RiskEntrySidecarDatasetStore(
            root,
            base_dataset_fingerprint=hashlib.sha256(b"different").hexdigest(),
            resume=True,
        )


def test_writer_lock_rejects_concurrent_process_local_writer(tmp_path: Path):
    root = tmp_path / "sidecar"
    first = RiskEntrySidecarDatasetStore(
        root, base_dataset_fingerprint=BASE_FINGERPRINT, resume=False
    )
    try:
        with pytest.raises(RiskEntrySidecarStorageError, match="another sidecar writer"):
            RiskEntrySidecarDatasetStore(
                root, base_dataset_fingerprint=BASE_FINGERPRINT, resume=True
            )
    finally:
        first.close()


def test_verifier_detects_corrupt_array_manifest_and_duplicate_split(tmp_path: Path):
    root = tmp_path / "sidecar"
    with RiskEntrySidecarDatasetStore(
        root, base_dataset_fingerprint=BASE_FINGERPRINT, resume=False
    ) as store:
        _append_episode(store, _metadata(0))
    path = _episode_path(root, "train", 0)
    state = np.load(path / "actor_state.npy", mmap_mode="r+")
    state[1, 0, 0] = np.nan
    state.flush()
    with pytest.raises(RiskEntrySidecarVerificationError, match="NaN"):
        verify_riskentry_sidecar_dataset(root)
    state[1, 0, 0] = 20.4
    state.flush()

    manifest_path = root / "train" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["raw_steps"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RiskEntrySidecarVerificationError, match="manifest"):
        verify_riskentry_sidecar_dataset(root)


def test_verifier_rejects_episode_present_in_two_splits(tmp_path: Path):
    root = tmp_path / "sidecar"
    with RiskEntrySidecarDatasetStore(
        root, base_dataset_fingerprint=BASE_FINGERPRINT, resume=False
    ) as store:
        episode = _append_episode(store, _metadata(0, "train"))
    source = _episode_path(root, "train", 0)
    destination = _episode_path(root, "val", 0)
    shutil.copytree(source, destination)
    metadata_path = destination / "episode.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["split"] = "val"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    entry = episode.manifest_entry()
    entry.pop("split")
    (root / "val" / "manifest.json").write_text(
        json.dumps(
            {
                "format": "riskentry-metadrive-actor-sidecar",
                "schema_version": "1.0.0",
                "split": "val",
                "episode_count": 1,
                "raw_steps": episode.raw_steps,
                "base_samples": episode.base_samples,
                "episodes": [entry],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RiskEntrySidecarVerificationError, match="appears in both"):
        verify_riskentry_sidecar_dataset(root)


def test_verifier_cli_prints_json_report(tmp_path: Path, capsys):
    root = tmp_path / "sidecar"
    with RiskEntrySidecarDatasetStore(
        root, base_dataset_fingerprint=BASE_FINGERPRINT, resume=False
    ) as store:
        _append_episode(store, _metadata(0))
    assert main(["--dataset-root", str(root)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["complete_scan"] is True
    assert report["episodes"] == 1
