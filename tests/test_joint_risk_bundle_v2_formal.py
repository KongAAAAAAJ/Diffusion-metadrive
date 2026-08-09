from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from expert_dataset.joint_risk_bundle_v2_formal import (
    CELL_ORDER,
    FORMAL_TARGETS,
    FormalV2CollectionError,
    FormalV2Config,
    FormalV2PartitionWriter,
    _episode_spec,
    load_formal_v2_config,
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
    assert pilot.max_episode_steps == formal.max_episode_steps == 160
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
    assert first.split == "test"
    assert first.matched_pair_id.endswith("pair_000000")
    assert first.spawn_seed != _episode_spec(
        config,
        "compositional_ood",
        "adjacent_lane_cut_in",
        "control",
        episode_index=0,
        cell_episode_index=0,
        attempt=0,
    ).spawn_seed
