from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_abstract_anchors_loads_selected_trajectory_key(tmp_path: Path):
    module = _load_module("abstract_anchors_test", REPO_ROOT / "expert_dataset/abstract_anchors.py")
    shard_path = tmp_path / "shard_000000.npz"
    np.savez(
        shard_path,
        trajectory=np.zeros((2, 8, 3), dtype=np.float32),
        trajectory_raw=np.ones((2, 8, 3), dtype=np.float32),
        trajectory_mode=np.zeros((2,), dtype=np.int64),
        ego_speed_km_h=np.zeros((2,), dtype=np.float32),
        front_object_distance=np.zeros((2,), dtype=np.float32),
        front_object_speed_km_h=np.zeros((2,), dtype=np.float32),
        lane_index=np.zeros((2,), dtype=np.int64),
        reference_lane_index=np.zeros((2,), dtype=np.int64),
        reference_longitudinal=np.zeros((2,), dtype=np.float32),
        reference_lateral=np.zeros((2,), dtype=np.float32),
        lane_width=np.ones((2,), dtype=np.float32),
        current_ref_lane_count=np.ones((2,), dtype=np.int64),
        next_ref_lane_count=np.zeros((2,), dtype=np.int64),
        reference_pose_world=np.zeros((2, 3), dtype=np.float32),
    )

    data = module._load_anchor_dataset(
        shard_paths=[shard_path],
        trajectory_key="trajectory_raw",
        max_trajectories=None,
        rng=np.random.RandomState(0),
    )

    assert data["trajectory_output"].shape == (2, 8, 3)
    assert np.allclose(data["trajectory_output"], 1.0)
    assert np.allclose(data["trajectory"], 1.0)


def test_abstract_anchors_reports_available_keys_for_missing_trajectory_key(tmp_path: Path):
    module = _load_module("abstract_anchors_test_missing", REPO_ROOT / "expert_dataset/abstract_anchors.py")
    shard_path = tmp_path / "shard_000000.npz"
    np.savez(
        shard_path,
        trajectory=np.zeros((1, 8, 3), dtype=np.float32),
        trajectory_mode=np.zeros((1,), dtype=np.int64),
        ego_speed_km_h=np.zeros((1,), dtype=np.float32),
        front_object_distance=np.zeros((1,), dtype=np.float32),
        front_object_speed_km_h=np.zeros((1,), dtype=np.float32),
        lane_index=np.zeros((1,), dtype=np.int64),
        reference_lane_index=np.zeros((1,), dtype=np.int64),
        reference_longitudinal=np.zeros((1,), dtype=np.float32),
        reference_lateral=np.zeros((1,), dtype=np.float32),
        lane_width=np.ones((1,), dtype=np.float32),
        current_ref_lane_count=np.ones((1,), dtype=np.int64),
        next_ref_lane_count=np.zeros((1,), dtype=np.int64),
        reference_pose_world=np.zeros((1, 3), dtype=np.float32),
    )

    with pytest.raises(RuntimeError, match="missing required semantic-anchor fields"):
        module._load_anchor_dataset(
            shard_paths=[shard_path],
            trajectory_key="trajectory_raw",
            max_trajectories=None,
            rng=np.random.RandomState(0),
        )


def test_generate_plan_anchors_exports_xy_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load_module("abstract_anchors_generate_xy", REPO_ROOT / "expert_dataset/abstract_anchors.py")
    output_path = tmp_path / "anchors.npy"
    trajectory = np.stack(
        [
            np.linspace(0.0, 7.0, 8, dtype=np.float32),
            np.linspace(-1.0, 1.0, 8, dtype=np.float32),
            np.linspace(0.1, 0.8, 8, dtype=np.float32),
        ],
        axis=1,
    )
    dataset = {
        "trajectory_output": trajectory[None, ...],
        "trajectory": trajectory[None, ...],
        "trajectory_mode": np.asarray([0], dtype=np.int64),
        "ego_speed_km_h": np.asarray([30.0], dtype=np.float32),
        "front_object_distance": np.asarray([50.0], dtype=np.float32),
        "front_object_speed_km_h": np.asarray([20.0], dtype=np.float32),
        "lane_index": np.asarray([0], dtype=np.int64),
        "reference_lane_index": np.asarray([0], dtype=np.int64),
        "reference_longitudinal": np.asarray([0.0], dtype=np.float32),
        "reference_lateral": np.asarray([0.0], dtype=np.float32),
        "lane_width": np.asarray([4.0], dtype=np.float32),
        "current_ref_lane_count": np.asarray([1], dtype=np.int64),
        "next_ref_lane_count": np.asarray([0], dtype=np.int64),
        "reference_pose_world": np.zeros((1, 3), dtype=np.float32),
        "source_shard": np.asarray(["shard_000000.npz"]),
        "source_sample_index": np.asarray([0], dtype=np.int64),
        "sample_index": np.asarray([0], dtype=np.int64),
    }

    monkeypatch.setattr(module, "resolve_shard_paths", lambda dataset_root, split: [tmp_path / "shard_000000.npz"])
    monkeypatch.setattr(module, "_load_anchor_dataset", lambda shard_paths, trajectory_key, max_trajectories, rng: dataset)
    monkeypatch.setattr(module, "label_dataset", lambda shard_paths: {"labels": np.asarray([int(module.BehaviorMode.CRUISE)])})
    monkeypatch.setattr(
        module,
        "_cluster_bucket",
        lambda dataset, labels, mode, max_clusters, seed: (
            [
                {
                    "behavior_mode": module.BehaviorMode.CRUISE,
                    "subtype_id": 0,
                    "global_index": 0,
                    "cluster_size": 1,
                    "cluster_inertia": 0.0,
                    "semantic_purity": 1.0,
                    "trajectory_mode_histogram": {"0": 1},
                    "feature_points": np.zeros((1, 2), dtype=np.float32),
                    "medoid_feature": np.zeros(2, dtype=np.float32),
                }
            ]
            if mode == module.BehaviorMode.CRUISE
            else []
        ),
    )

    anchors = module.generate_plan_anchors(
        dataset_root=tmp_path,
        output_path=output_path,
        split="all",
        trajectory_key="trajectory",
        num_anchors=1,
        max_trajectories=None,
        seed=0,
        max_iters=10,
        tolerance=1e-4,
    )

    saved = np.load(output_path)
    assert anchors.shape == (1, 8, 2)
    assert saved.shape == (1, 8, 2)
    assert np.allclose(saved[0], trajectory[:, :2])


def test_generate_plan_anchors_empty_output_is_xy_shaped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load_module("abstract_anchors_generate_empty", REPO_ROOT / "expert_dataset/abstract_anchors.py")
    output_path = tmp_path / "anchors.npy"
    dataset = {
        "trajectory_output": np.zeros((0, 8, 3), dtype=np.float32),
        "trajectory": np.zeros((0, 8, 3), dtype=np.float32),
        "trajectory_mode": np.zeros((0,), dtype=np.int64),
        "ego_speed_km_h": np.zeros((0,), dtype=np.float32),
        "front_object_distance": np.zeros((0,), dtype=np.float32),
        "front_object_speed_km_h": np.zeros((0,), dtype=np.float32),
        "lane_index": np.zeros((0,), dtype=np.int64),
        "reference_lane_index": np.zeros((0,), dtype=np.int64),
        "reference_longitudinal": np.zeros((0,), dtype=np.float32),
        "reference_lateral": np.zeros((0,), dtype=np.float32),
        "lane_width": np.zeros((0,), dtype=np.float32),
        "current_ref_lane_count": np.zeros((0,), dtype=np.int64),
        "next_ref_lane_count": np.zeros((0,), dtype=np.int64),
        "reference_pose_world": np.zeros((0, 3), dtype=np.float32),
        "source_shard": np.asarray([], dtype="U128"),
        "source_sample_index": np.zeros((0,), dtype=np.int64),
        "sample_index": np.zeros((0,), dtype=np.int64),
    }

    monkeypatch.setattr(module, "resolve_shard_paths", lambda dataset_root, split: [tmp_path / "shard_000000.npz"])
    monkeypatch.setattr(module, "_load_anchor_dataset", lambda shard_paths, trajectory_key, max_trajectories, rng: dataset)
    monkeypatch.setattr(module, "label_dataset", lambda shard_paths: {"labels": np.zeros((0,), dtype=np.int16)})
    monkeypatch.setattr(module, "_cluster_bucket", lambda dataset, labels, mode, max_clusters, seed: [])

    anchors = module.generate_plan_anchors(
        dataset_root=tmp_path,
        output_path=output_path,
        split="all",
        trajectory_key="trajectory",
        num_anchors=1,
        max_trajectories=None,
        seed=0,
        max_iters=10,
        tolerance=1e-4,
    )

    saved = np.load(output_path)
    assert anchors.shape == (0, 8, 2)
    assert saved.shape == (0, 8, 2)


def test_generate_plan_anchors_temp_flattens_selected_modes_to_zero_y(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load_module(
        "abstract_anchors_default_temp_test",
        REPO_ROOT / "expert_dataset/abstract_anchors_default_temp.py",
    )
    output_path = tmp_path / "anchors.npy"
    base_anchors = np.stack(
        [
            np.stack(
                [
                    np.linspace(float(mode_idx), float(mode_idx) + 7.0, 8, dtype=np.float64),
                    np.linspace(float(mode_idx) + 0.1, float(mode_idx) + 0.8, 8, dtype=np.float64),
                ],
                axis=1,
            )
            for mode_idx in range(8)
        ],
        axis=0,
    )
    flattened_centers = base_anchors.reshape(8, -1)

    monkeypatch.setattr(module, "resolve_shard_paths", lambda dataset_root, split: [tmp_path / "shard_000000.npz"])
    monkeypatch.setattr(
        module,
        "load_trajectory_matrix",
        lambda shard_paths, trajectory_key, max_trajectories, rng: np.zeros((8, 16), dtype=np.float64),
    )
    monkeypatch.setattr(
        module,
        "run_kmeans",
        lambda data, num_clusters, seed, max_iters, tolerance: (
            flattened_centers.copy(),
            np.arange(8, dtype=np.int64),
            0.0,
        ),
    )

    anchors = module.generate_plan_anchors(
        dataset_root=tmp_path,
        output_path=output_path,
        split="all",
        trajectory_key="trajectory",
        num_anchors=8,
        max_trajectories=None,
        seed=0,
        max_iters=10,
        tolerance=1e-4,
    )

    saved = np.load(output_path)
    flattened_modes = {0, 1, 4, 7}
    for mode_idx in range(8):
        if mode_idx in flattened_modes:
            assert np.allclose(anchors[mode_idx, :, 1], 0.0)
            assert np.allclose(saved[mode_idx, :, 1], 0.0)
            assert np.allclose(anchors[mode_idx, :, 0], base_anchors[mode_idx, :, 0])
        else:
            assert np.allclose(anchors[mode_idx], base_anchors[mode_idx])
            assert np.allclose(saved[mode_idx], base_anchors[mode_idx])


def test_run_expert_parse_args_supports_expert_type_and_debug_flag():
    module = _load_module("run_expert_test", REPO_ROOT / "expert_dataset/run_expert.py")

    args = module.parse_args([
        "--expert-type", "idm",
        "--episodes", "3",
        "--render", "0",
        "--print-trajectory-debug", "1",
    ])

    assert args.expert_type == "idm"
    assert args.episodes == 3
    assert args.render == 0
    assert args.print_trajectory_debug == 1
