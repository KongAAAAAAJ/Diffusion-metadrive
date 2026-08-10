from __future__ import annotations

from pathlib import Path
import threading

import pytest
import yaml

from expert_dataset.joint_risk_bundle_v2_formal import (
    CELL_ORDER,
    FORMAL_TARGETS,
    FormalV2CollectionError,
    FormalV2Config,
    FormalV2PartitionWriter,
    _collect_pair_batch,
    _episode_spec,
    _pair_window_allocations,
    load_formal_v2_config,
    run_formal_v2_collection,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path, *, mode: str = "formal_pilot") -> FormalV2Config:
    targets = FORMAL_TARGETS if mode == "formal" else {
        "id": 8,
        "compositional_ood": 8,
        "topology_ood": 8,
    }
    return FormalV2Config(
        output_root=tmp_path,
        run_mode=mode,
        resume=True,
        split_seed=3,
        max_episode_steps=160,
        anchor_steps=(40, 50, 60, 70),
        max_attempts_per_cell=3,
        target_windows=targets,
        dataset_instance_prefix="unit_formal_v2",
    )


def test_pilot_and_70k_configs_share_the_same_collector_contract() -> None:
    pilot = load_formal_v2_config(
        REPO_ROOT / "configs/dataset/data_collect_bundle_v2_formal_path_pilot.yaml"
    )
    formal = load_formal_v2_config(
        REPO_ROOT / "configs/dataset/data_collect_bundle_v2_formal70k.yaml"
    )
    assert pilot.run_mode == "formal_pilot"
    assert formal.run_mode == "formal"
    assert pilot.anchor_steps == formal.anchor_steps
    assert pilot.max_episode_steps == formal.max_episode_steps == 120
    assert formal.target_windows == FORMAL_TARGETS
    assert len(CELL_ORDER) == 8


def test_formal_mode_rejects_non_frozen_quota(tmp_path: Path) -> None:
    with pytest.raises(FormalV2CollectionError, match="50k/10k/10k"):
        FormalV2Config(
            output_root=tmp_path,
            run_mode="formal",
            resume=True,
            split_seed=3,
            max_episode_steps=160,
            anchor_steps=(60,),
            max_attempts_per_cell=3,
            target_windows={"id": 8, "compositional_ood": 8, "topology_ood": 8},
            dataset_instance_prefix="bad",
        )


def test_partition_writer_resume_is_strict_and_non_destructive(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with FormalV2PartitionWriter(config, "id") as writer:
        assert writer.rows == []
        assert writer.cell_counts() == {}
        assert writer._base_contract_payload()["split_assignment"]["seed"] == 3
        assert (
            writer._base_contract_payload()["split_assignment"]["unit"]
            == "matched_pair_id"
        )
        assert (writer.base_root / ".writer.lock").is_file()
        assert (writer.base_root / "collection_state.json").is_file()
    with FormalV2PartitionWriter(config, "id") as resumed:
        assert resumed.rows == []
        assert resumed.state_path.is_file()

    changed = FormalV2Config(
        **{
            **config.__dict__,
            "anchor_steps": (50, 60, 70),
        }
    )
    with pytest.raises(FormalV2CollectionError, match="resume contract mismatch"):
        FormalV2PartitionWriter(changed, "id")


def test_config_loader_rejects_string_resume(tmp_path: Path) -> None:
    payload = {
        "dataset": {
            "output_root": str(tmp_path / "output"),
            "dataset_instance_prefix": "strict",
        },
        "collection": {
            "run_mode": "formal_pilot",
            "resume": "false",
            "split_seed": 3,
            "max_episode_steps": 160,
            "anchor_steps": [40, 50, 60, 70],
            "max_attempts_per_cell": 3,
            "target_windows": {
                "id": 8,
                "compositional_ood": 8,
                "topology_ood": 8,
            },
        },
    }
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(FormalV2CollectionError, match="must be a boolean"):
        load_formal_v2_config(path)


def test_episode_seed_split_and_pair_id_are_deterministic(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = _episode_spec(
        config,
        "id",
        "adjacent_lane_cut_in",
        "control",
        episode_index=0,
        cell_episode_index=0,
        attempt=0,
    )
    repeat = _episode_spec(
        config,
        "id",
        "adjacent_lane_cut_in",
        "control",
        episode_index=0,
        cell_episode_index=0,
        attempt=0,
    )
    assert first == repeat
    assert first.split == "train"
    assert first.matched_pair_id.endswith("pair_000000")
    paired = _episode_spec(
        config,
        "id",
        "adjacent_lane_cut_in",
        "near_critical",
        episode_index=1,
        cell_episode_index=0,
        attempt=0,
    )
    assert paired.split == first.split
    assert paired.spawn_seed == first.spawn_seed
    assert first.spawn_seed != _episode_spec(
        config,
        "compositional_ood",
        "adjacent_lane_cut_in",
        "control",
        episode_index=0,
        cell_episode_index=0,
        attempt=0,
    ).spawn_seed


def test_pair_window_allocations_never_overfill_quota_or_episode_budget() -> None:
    assert _pair_window_allocations(
        remaining_windows=25,
        anchors_per_episode=7,
        parallel_workers=4,
        remaining_episode_budget=None,
    ) == (7, 7, 7, 4)
    assert _pair_window_allocations(
        remaining_windows=25,
        anchors_per_episode=7,
        parallel_workers=4,
        remaining_episode_budget=5,
    ) == (7, 7)
    assert _pair_window_allocations(
        remaining_windows=25,
        anchors_per_episode=7,
        parallel_workers=4,
        remaining_episode_budget=1,
    ) == ()


def test_pair_batch_runs_concurrently_but_returns_episode_index_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barrier = threading.Barrier(4, timeout=2.0)

    def fake_collect(config, writer, family, **kwargs):
        del config, writer, family
        barrier.wait()
        return (
            int(kwargs["pair_index"]),
            int(kwargs["episode_index"]),
            int(kwargs["pair_windows"]),
        )

    monkeypatch.setattr(
        "expert_dataset.joint_risk_bundle_v2_formal._collect_matched_pair_for_quota",
        fake_collect,
    )
    results = _collect_pair_batch(
        _config(Path("/tmp/unused-performance-config")),
        object(),
        "adjacent_lane_cut_in",
        pair_index_base=10,
        episode_index_base=20,
        allocations=(7, 7, 7, 4),
        parallel_workers=4,
    )
    assert results == (
        (10, 20, 7),
        (11, 22, 7),
        (12, 24, 7),
        (13, 26, 4),
    )


def test_parallel_workers_is_runtime_only_and_strictly_positive(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert "parallel_workers" not in config.frozen_payload()
    with pytest.raises(FormalV2CollectionError, match="parallel_workers"):
        run_formal_v2_collection(config, parallel_workers=0)


def test_performance_family_is_pilot_only_and_not_frozen(tmp_path: Path) -> None:
    pilot = _config(tmp_path / "pilot")
    assert "performance_family" not in pilot.frozen_payload()
    with pytest.raises(FormalV2CollectionError, match="unknown performance_family"):
        run_formal_v2_collection(
            pilot,
            performance_family="not_a_scenario",
        )

    formal = _config(tmp_path / "formal", mode="formal")
    with pytest.raises(FormalV2CollectionError, match="restricted to formal_pilot"):
        run_formal_v2_collection(
            formal,
            performance_family="adjacent_lane_cut_in",
        )


def test_commit_batch_refreshes_metadata_once(monkeypatch: pytest.MonkeyPatch) -> None:
    writer = object.__new__(FormalV2PartitionWriter)
    refresh_flags = []
    refresh_count = 0

    def fake_commit(**kwargs):
        refresh_flags.append(bool(kwargs["refresh_metadata"]))
        return {
            "episode_commit_s": 0.25,
            "metadata_refresh_s": 0.0,
            "total_s": 0.25,
        }

    def fake_materialize():
        nonlocal refresh_count
        refresh_count += 1

    monkeypatch.setattr(writer, "commit", fake_commit)
    monkeypatch.setattr(writer, "_materialize_metadata", fake_materialize)
    dummy = (object(), {}, {}, (), ())
    timing = writer.commit_batch((dummy, dummy, dummy, dummy))

    assert refresh_flags == [False, False, False, False]
    assert refresh_count == 1
    assert timing["episode_count"] == 4.0
    assert timing["episode_commit_s"] == pytest.approx(1.0)


def test_performance_pilot_configs_differ_only_in_output_identity() -> None:
    serial = load_formal_v2_config(
        REPO_ROOT
        / "configs/dataset/data_collect_bundle_v2_performance_pilot_serial.yaml"
    )
    parallel = load_formal_v2_config(
        REPO_ROOT
        / "configs/dataset/data_collect_bundle_v2_performance_pilot_parallel.yaml"
    )
    assert serial.run_mode == parallel.run_mode == "formal_pilot"
    assert serial.anchor_steps == parallel.anchor_steps == (35, 40, 45, 50, 55, 60, 65)
    assert serial.max_episode_steps == parallel.max_episode_steps == 120
    assert serial.target_windows == parallel.target_windows == {
        "id": 224,
        "compositional_ood": 224,
        "topology_ood": 224,
    }
    assert serial.output_root != parallel.output_root
