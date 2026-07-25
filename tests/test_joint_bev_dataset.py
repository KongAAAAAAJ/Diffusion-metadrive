from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from expert_dataset.collect_joint_bev import (
    AgentRole,
    JOINT_SAMPLE_DTYPES,
    JOINT_SAMPLE_SHAPES,
    JointBEVSample,
)
from expert_dataset.joint_bev_dataset import (
    EpisodeChunkBatchSampler,
    JointBEVDataset,
    JointBEVDatasetConfig,
    JointBEVDatasetError,
    build_joint_bev_dataloader,
)
from expert_dataset.joint_bev_storage import (
    EpisodeSplitConfig,
    JointBEVDatasetStore,
    fingerprint_payload,
)
from models.bev_planner.mode_contract import ModeIndex


def _sample(marker: int) -> JointBEVSample:
    values = {
        name: np.zeros(shape, dtype=JOINT_SAMPLE_DTYPES[name])
        for name, shape in JOINT_SAMPLE_SHAPES.items()
    }
    values["bev"][:, 0, marker % 256, marker % 256] = 255
    values["bev"][:, 1, 0, 0] = np.asarray((0, 128, 255), dtype=np.uint8)[
        marker % 3
    ]
    values["bev"][:, 3, 0, 1] = np.asarray(
        (0, 85, 170, 255), dtype=np.uint8
    )[marker % 4]
    values["ego_state"][:] = np.float32(marker)
    values["ego_pose_global"][:, 0] = np.float32(marker)
    values["formation_relation_state"][:] = np.float32(marker + 0.5)
    values["relation_valid_mask"][:] = True
    values["agent_role"] = np.asarray(list(AgentRole), dtype=np.int64)
    values["mode_valid_mask"][:, ModeIndex.STOP] = True
    values["gt_mode"][:] = int(ModeIndex.STOP)
    values["expert_trajectory"][:, :, 0] = np.float32(marker)
    return JointBEVSample(**values)


def _create_dataset(
    root: Path,
    *,
    episode_count: int = 3,
    samples_per_episode: int = 2,
) -> list[JointBEVSample]:
    all_samples = []
    with JointBEVDatasetStore(
        root,
        split_config=EpisodeSplitConfig(1.0, 0.0, 0.0, seed=5),
        dataset_fingerprint=fingerprint_payload({"test": "joint-dataset"}),
        resume=False,
    ) as store:
        marker = 0
        for episode_index in range(episode_count):
            samples = []
            for _ in range(samples_per_episode):
                sample = _sample(marker)
                samples.append(sample)
                all_samples.append(sample)
                marker += 1
            store.commit_episode(
                episode_index,
                samples,
                {
                    "scenario_id": "S5_hard_brake_lead",
                    "local_route": "R3_mainline_straight",
                },
            )
    return all_samples


def test_dataset_restores_exact_joint_first_tensors(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    expected = _create_dataset(root)
    dataset = JointBEVDataset(
        JointBEVDatasetConfig(root, "train", mmap_cache_episodes=2)
    )
    try:
        assert len(dataset) == len(expected)
        item = dataset[3]
        assert set(item) == set(JOINT_SAMPLE_SHAPES)
        expected_torch_dtypes = {
            "bev": torch.uint8,
            "ego_state": torch.float32,
            "ego_pose_global": torch.float32,
            "formation_relation_state": torch.float32,
            "relation_valid_mask": torch.bool,
            "agent_role": torch.int64,
            "coarse_trajectories": torch.float32,
            "mode_valid_mask": torch.bool,
            "gt_mode": torch.int64,
            "expert_trajectory": torch.float32,
        }
        for name, tensor in item.items():
            assert tensor.shape == JOINT_SAMPLE_SHAPES[name]
            assert tensor.dtype == expected_torch_dtypes[name]
            np.testing.assert_array_equal(tensor.numpy(), expected[3].as_dict()[name])
        np.testing.assert_array_equal(dataset[-1]["bev"], expected[-1].bev)
    finally:
        dataset.close()


def test_dataloader_default_collate_adds_only_batch_dimension(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _create_dataset(root, episode_count=2, samples_per_episode=3)
    loader = build_joint_bev_dataloader(
        root,
        "train",
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))
    assert batch["bev"].shape == (2, *JOINT_SAMPLE_SHAPES["bev"])
    assert batch["ego_state"].shape == (2, *JOINT_SAMPLE_SHAPES["ego_state"])
    assert batch["coarse_trajectories"].shape == (
        2,
        *JOINT_SAMPLE_SHAPES["coarse_trajectories"],
    )
    assert batch["mode_valid_mask"].dtype == torch.bool
    assert batch["gt_mode"].dtype == torch.int64
    loader.dataset.close()


def test_episode_chunk_sampler_is_deterministic_complete_and_epoch_aware(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    _create_dataset(root, episode_count=3, samples_per_episode=5)
    dataset = JointBEVDataset(JointBEVDatasetConfig(root, "train"))
    sampler = EpisodeChunkBatchSampler(
        dataset,
        batch_size=4,
        shuffle=True,
        seed=11,
        chunk_size=8,
    )
    first = list(iter(sampler))
    second = list(iter(sampler))
    assert first == second
    assert sorted(value for batch in first for value in batch) == list(
        range(len(dataset))
    )
    assert len(first) == len(sampler)
    sampler.set_epoch(1)
    assert list(iter(sampler)) != first
    dataset.close()


def test_mmap_lru_bounds_open_episode_files(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _create_dataset(root, episode_count=4, samples_per_episode=1)
    dataset = JointBEVDataset(
        JointBEVDatasetConfig(root, "train", mmap_cache_episodes=2)
    )
    before = len(os.listdir("/proc/self/fd"))
    for index in range(len(dataset)):
        dataset[index]
    after = len(os.listdir("/proc/self/fd"))
    assert len(dataset._episode_cache) == 2
    assert after - before <= 2 * len(JOINT_SAMPLE_SHAPES) + 2
    dataset.close()
    assert len(os.listdir("/proc/self/fd")) <= before + 2


def test_multi_worker_loader_does_not_share_parent_mmap_cache(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    _create_dataset(root, episode_count=2, samples_per_episode=2)
    loader = build_joint_bev_dataloader(
        root,
        "train",
        batch_size=2,
        shuffle=True,
        num_workers=2,
        seed=3,
    )
    loader.dataset[0]
    assert len(loader.dataset._episode_cache) == 1
    batches = list(loader)
    assert sum(int(batch["bev"].shape[0]) for batch in batches) == 4
    assert len(loader.dataset._episode_cache) == 1


def test_dataset_rejects_split_or_contract_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _create_dataset(root, episode_count=1, samples_per_episode=1)
    with pytest.raises(JointBEVDatasetError, match="split must be"):
        JointBEVDatasetConfig(root, "all")

    contract_path = root / "dataset_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["packed_semantic_bev"]["bitorder"] = "big"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(JointBEVDatasetError, match="packed semantic BEV"):
        JointBEVDataset(JointBEVDatasetConfig(root, "train"))
