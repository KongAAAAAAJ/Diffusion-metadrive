from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "analysis" / "verify_phase1.py"
SPEC = importlib.util.spec_from_file_location("verify_phase1_under_test", MODULE_PATH)
verify_phase1 = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = verify_phase1
SPEC.loader.exec_module(verify_phase1)


def _encode_strings(values):
    return np.asarray([value.encode("utf-8") for value in values], dtype="S64")


def _write_shard(
    dataset_root: Path,
    name: str,
    include_local_route: bool,
    episode_ids: np.ndarray | None = None,
) -> None:
    shard_dir = dataset_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ego_pose_world": np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 0.1]], dtype=np.float32),
        "future_ego_pose_world": np.asarray(
            [
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[2.0, 1.0, 0.1], [3.0, 1.5, 0.2]],
            ],
            dtype=np.float32,
        ),
        "trajectory_mode": np.asarray([0, 1], dtype=np.int16),
        "route_id": _encode_strings(["mainline", "ramp_merge"]),
    }
    if include_local_route:
        payload["local_route"] = _encode_strings(["R1_entry_straight", "R7_merge_core"])
    if episode_ids is not None:
        payload["episode_id"] = np.asarray(episode_ids, dtype=np.int32)
    np.savez_compressed(shard_dir / name, **payload)


def test_load_dataset_prefers_local_route_and_falls_back_to_route_id(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    _write_shard(dataset_root, "shard_000000.npz", include_local_route=True)

    loaded = verify_phase1.load_dataset(dataset_root)

    assert "local_route" in loaded
    assert loaded["local_route"].tolist() == ["R1_entry_straight", "R7_merge_core"]

    fallback_root = tmp_path / "fallback_dataset"
    _write_shard(fallback_root, "shard_000000.npz", include_local_route=False)

    fallback = verify_phase1.load_dataset(fallback_root)

    assert fallback["local_route"].tolist() == ["mainline", "ramp_merge"]


def test_verify_phase1_outputs_heatmap_counts_and_per_route_plots(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    _write_shard(dataset_root, "shard_000000.npz", include_local_route=True)

    data = verify_phase1.load_dataset(dataset_root)
    route_counts = verify_phase1.save_route_counts(data, manifest=None, output_dir=output_dir)
    verify_phase1.save_heatmap(data, output_dir, bins=20)
    saved_counts = verify_phase1.save_per_route_trajectories(data, output_dir, max_traj_plots=10)

    assert route_counts == {"R1_entry_straight": 1, "R7_merge_core": 1}
    assert saved_counts == {"R1_entry_straight": 1, "R7_merge_core": 1}
    assert (output_dir / "fig1_heatmap.png").is_file()
    assert (output_dir / "fig2_route_counts.png").is_file()
    assert (output_dir / "per_route" / "R1_entry_straight" / "traj_000000.png").is_file()
    assert (output_dir / "per_route" / "R7_merge_core" / "traj_000001.png").is_file()


def test_save_route_counts_prefers_episode_counts_when_episode_id_is_available(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    shard_dir = dataset_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        shard_dir / "shard_000000.npz",
        ego_pose_world=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        future_ego_pose_world=np.asarray(
            [
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                [[4.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        ),
        trajectory_mode=np.asarray([0, 0, 1, 1], dtype=np.int16),
        route_id=_encode_strings(["mainline", "mainline", "ramp_merge", "ramp_merge"]),
        local_route=_encode_strings(
            ["R1_entry_straight", "R1_entry_straight", "R7_merge_core", "R7_merge_core"]
        ),
        episode_id=np.asarray([10, 10, 20, 21], dtype=np.int32),
    )

    data = verify_phase1.load_dataset(dataset_root)
    route_counts = verify_phase1.save_route_counts(data, manifest=None, output_dir=output_dir)

    assert route_counts == {"R1_entry_straight": 1, "R7_merge_core": 2}


def test_save_route_counts_falls_back_to_sample_counts_without_episode_id(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    _write_shard(dataset_root, "shard_000000.npz", include_local_route=True, episode_ids=None)

    data = verify_phase1.load_dataset(dataset_root)
    route_counts = verify_phase1.save_route_counts(data, manifest=None, output_dir=output_dir)

    assert route_counts == {"R1_entry_straight": 1, "R7_merge_core": 1}
