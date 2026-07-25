from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from expert_dataset.collect_joint_bev import (
    AgentRole,
    JOINT_SAMPLE_DTYPES,
    JOINT_SAMPLE_SHAPES,
    JointBEVSample,
)
from expert_dataset.joint_bev_storage import (
    EpisodeSplitAssigner,
    EpisodeSplitConfig,
    JointBEVDatasetStore,
    JointStorageError,
    PACKED_BEV_FIELD,
    SPLIT_NAMES,
    STORAGE_SCHEMA_VERSION,
    fingerprint_payload,
)
from expert_dataset.semantic_bev_codec import (
    BEV_COMPRESSION_RATIO,
    PACKED_BEV_SHAPE,
    unpack_semantic_bev,
)
from models.bev_planner.mode_contract import ModeIndex


def _sample(marker: int = 0) -> JointBEVSample:
    values = {
        name: np.zeros(shape, dtype=JOINT_SAMPLE_DTYPES[name])
        for name, shape in JOINT_SAMPLE_SHAPES.items()
    }
    values["bev"][:, 0, 0, 0] = np.uint8(255 if marker % 2 else 0)
    values["bev"][:, 3, 0, 1] = np.asarray(
        (0, 85, 170, 255), dtype=np.uint8
    )[marker % 4]
    values["agent_role"] = np.asarray(list(AgentRole), dtype=np.int64)
    values["mode_valid_mask"][:, ModeIndex.STOP] = True
    values["gt_mode"][:] = int(ModeIndex.STOP)
    return JointBEVSample(**values)


def _fingerprint(name: str = "test") -> str:
    return fingerprint_payload({"dataset": name, "schema": "joint-first"})


def _open(
    root: Path,
    *,
    resume: bool,
    split: EpisodeSplitConfig | None = None,
    fingerprint: str | None = None,
) -> JointBEVDatasetStore:
    return JointBEVDatasetStore(
        root,
        split_config=split or EpisodeSplitConfig(0.6, 0.2, 0.2, seed=17),
        dataset_fingerprint=fingerprint or _fingerprint(),
        resume=resume,
    )


def test_episode_split_assignment_is_deterministic_and_whole_episode() -> None:
    config = EpisodeSplitConfig(0.8, 0.1, 0.1, seed=13)
    first = EpisodeSplitAssigner(config)
    second = EpisodeSplitAssigner(config)
    assert [first.split_for_episode(index) for index in range(100)] == [
        second.split_for_episode(index) for index in range(100)
    ]
    assert set(first.split_for_episode(index) for index in range(1000)) == set(
        SPLIT_NAMES
    )


def test_independent_split_writers_store_mmap_joint_first_episodes(tmp_path: Path) -> None:
    split_config = EpisodeSplitConfig(1.0, 1.0, 1.0, seed=5)
    seen = set()
    with _open(tmp_path / "dataset", resume=False, split=split_config) as store:
        while seen != set(SPLIT_NAMES):
            episode_index = store.next_episode_index
            expected_split = store.assigner.split_for_episode(episode_index)
            episode = store.commit_episode(
                episode_index,
                [_sample(episode_index), _sample(episode_index + 1)],
                {
                    "scenario_id": "S5_hard_brake_lead",
                    "local_route": "R3_mainline_straight",
                },
            )
            assert episode.split == expected_split
            seen.add(expected_split)
        summary = store.summary()

    root = tmp_path / "dataset"
    all_episode_paths = []
    for split in SPLIT_NAMES:
        manifest = json.loads((root / split / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["split"] == split
        assert manifest["episode_count"] == summary["splits"][split]["episodes"]
        for entry in manifest["episodes"]:
            episode_path = root / split / "episodes" / entry["directory"]
            all_episode_paths.append(episode_path)
            packed = np.load(
                episode_path / f"{PACKED_BEV_FIELD}.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            assert isinstance(packed, np.memmap)
            assert packed.shape == (2, 3, *PACKED_BEV_SHAPE)
            assert packed.dtype == np.uint8
            assert not (episode_path / "bev.npy").exists()
            assert (
                np.prod(JOINT_SAMPLE_SHAPES["bev"])
                / np.prod(packed.shape[1:])
                == BEV_COMPRESSION_RATIO
            )
            expected = np.stack(
                [
                    _sample(entry["episode_index"]).bev,
                    _sample(entry["episode_index"] + 1).bev,
                ]
            )
            np.testing.assert_array_equal(unpack_semantic_bev(packed), expected)
    assert len(all_episode_paths) == len({path.name for path in all_episode_paths})
    assert all(path.parent.parent.name in SPLIT_NAMES for path in all_episode_paths)


def test_resume_continues_without_overwriting_and_reconciles_stale_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    with _open(root, resume=False) as store:
        stored = store.commit_episode(0, [_sample(7)], {"scenario_id": "S5"})
        assert stored.episode_index == 0

    state_path = root / "collection_state.json"
    stale = json.loads(state_path.read_text(encoding="utf-8"))
    stale.update(
        {
            "next_episode_index": 0,
            "attempted_episodes": 0,
            "stored_episodes": 0,
            "total_joint_samples": 0,
        }
    )
    state_path.write_text(json.dumps(stale), encoding="utf-8")

    with _open(root, resume=True) as resumed:
        assert resumed.next_episode_index == 1
        assert resumed.total_joint_samples == 1
        resumed.record_rejected_episode(1, "no_joint_samples")
        assert resumed.next_episode_index == 2
        resumed.commit_episode(2, [_sample(9)], {"scenario_id": "S6"})
        assert resumed.next_episode_index == 3

    with _open(root, resume=True) as final:
        assert final.next_episode_index == 3
        assert final.summary()["stored_episodes"] == 2
        assert final.summary()["rejected_episodes"] == 1


def test_expert_labels_do_not_change_packed_bev_input(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    base = _sample(1)
    changed_values = {
        name: np.array(value, copy=True) for name, value in base.as_dict().items()
    }
    changed_values["expert_trajectory"][:] = np.float32(12.5)
    changed = JointBEVSample(**changed_values)
    with _open(root, resume=False) as store:
        episode = store.commit_episode(0, [base, changed], {"scenario_id": "S5"})
    packed = np.load(
        root
        / episode.split
        / "episodes"
        / episode.directory
        / f"{PACKED_BEV_FIELD}.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    np.testing.assert_array_equal(packed[0], packed[1])


def test_resume_rejects_changed_contract(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    with _open(root, resume=False):
        pass
    with pytest.raises(JointStorageError, match="contract mismatch"):
        _open(
            root,
            resume=True,
            split=EpisodeSplitConfig(0.7, 0.2, 0.1, seed=17),
        )
    with pytest.raises(JointStorageError, match="contract mismatch"):
        _open(root, resume=True, fingerprint=_fingerprint("different"))

    contract_path = root / "dataset_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["schema_version"] = STORAGE_SCHEMA_VERSION - 1
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(JointStorageError, match="contract mismatch"):
        _open(root, resume=True)


def test_resume_rejects_corrupt_committed_array(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    with _open(root, resume=False) as store:
        episode = store.commit_episode(0, [_sample()], {})
        array_path = (
            root
            / episode.split
            / "episodes"
            / episode.directory
            / f"{PACKED_BEV_FIELD}.npy"
        )
    with array_path.open("r+b") as stream:
        stream.truncate(64)
    with pytest.raises(JointStorageError, match="unable to mmap episode array"):
        _open(root, resume=True)


def test_resume_rejects_invalid_state_json(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    with _open(root, resume=False):
        pass
    (root / "collection_state.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(JointStorageError, match="invalid JSON state"):
        _open(root, resume=True)


def test_non_resume_rejects_existing_dataset_and_lock_rejects_second_writer(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    first = _open(root, resume=False)
    try:
        with pytest.raises(JointStorageError, match="another writer"):
            _open(root, resume=True)
    finally:
        first.close()
    with pytest.raises(JointStorageError, match="resume is disabled"):
        _open(root, resume=False)


def test_known_interrupted_temporary_episode_is_ignored(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    with _open(root, resume=False) as store:
        split = store.assigner.split_for_episode(0)
    temporary = root / split / "episodes" / ".episode_00000000.tmp-deadbeef"
    temporary.mkdir()
    (temporary / "partial.npy").write_bytes(b"partial")
    with _open(root, resume=True) as resumed:
        ignored = resumed.summary()["splits"][split]["ignored_temporary_directories"]
        assert temporary.name in ignored
        assert resumed.next_episode_index == 0
