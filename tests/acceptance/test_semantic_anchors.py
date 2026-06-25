from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from expert_dataset.abstract_anchors import generate_plan_anchors, resolve_shard_paths
from expert_dataset.semantic_labeler import BehaviorMode, label_dataset


def _trajectory(end_x: float, end_y: float, delta_heading: float = 0.0) -> np.ndarray:
    x = np.linspace(0.5, end_x, 8, dtype=np.float32)
    y = np.linspace(0.0, end_y, 8, dtype=np.float32)
    heading = np.linspace(0.0, delta_heading, 8, dtype=np.float32)
    return np.stack([x, y, heading], axis=-1)


def _append_samples(storage: dict[str, list], *, trajectory_mode: int, end_x: float, end_y: float, delta_heading: float,
                    ego_speed_km_h: float, front_distance: float, next_ref_lane_count: int, current_ref_lane_count: int = 2,
                    lane_width: float = 4.0, count: int = 3) -> None:
    for idx in range(count):
        traj = _trajectory(
            end_x=end_x + 0.2 * idx,
            end_y=end_y + 0.05 * idx,
            delta_heading=delta_heading + 0.02 * idx,
        )
        storage["trajectory"].append(traj)
        storage["trajectory_raw"].append(traj.copy())
        storage["trajectory_mode"].append(np.int8(trajectory_mode))
        storage["ego_speed_km_h"].append(np.float32(ego_speed_km_h))
        storage["front_object_distance"].append(np.float32(front_distance))
        storage["front_object_speed_km_h"].append(np.float32(12.0 if front_distance > 0 else -1.0))
        storage["lane_index"].append(np.int16(1))
        storage["reference_lane_index"].append(np.int16(1))
        storage["reference_longitudinal"].append(np.float32(0.0))
        storage["reference_lateral"].append(np.float32(end_y))
        storage["lane_width"].append(np.float32(lane_width))
        storage["current_ref_lane_count"].append(np.int16(current_ref_lane_count))
        storage["next_ref_lane_count"].append(np.int16(next_ref_lane_count))
        storage["reference_pose_world"].append(np.asarray([0.0, 0.0, 0.0], dtype=np.float32))


def _build_dataset_root(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "semantic_dataset"
    shard_dir = dataset_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    storage = {key: [] for key in (
        "trajectory",
        "trajectory_raw",
        "trajectory_mode",
        "ego_speed_km_h",
        "front_object_distance",
        "front_object_speed_km_h",
        "lane_index",
        "reference_lane_index",
        "reference_longitudinal",
        "reference_lateral",
        "lane_width",
        "current_ref_lane_count",
        "next_ref_lane_count",
        "reference_pose_world",
    )}

    _append_samples(storage, trajectory_mode=0, end_x=28.0, end_y=0.1, delta_heading=0.0, ego_speed_km_h=25.0, front_distance=-1.0, next_ref_lane_count=2)
    _append_samples(storage, trajectory_mode=0, end_x=18.0, end_y=0.2, delta_heading=0.0, ego_speed_km_h=18.0, front_distance=-1.0, next_ref_lane_count=2)
    _append_samples(storage, trajectory_mode=1, end_x=8.5, end_y=0.0, delta_heading=0.0, ego_speed_km_h=20.0, front_distance=10.0, next_ref_lane_count=2)
    _append_samples(storage, trajectory_mode=1, end_x=3.0, end_y=0.0, delta_heading=0.0, ego_speed_km_h=10.0, front_distance=6.0, next_ref_lane_count=2)
    _append_samples(storage, trajectory_mode=5, end_x=1.5, end_y=0.2, delta_heading=0.0, ego_speed_km_h=15.0, front_distance=-1.0, next_ref_lane_count=1)
    _append_samples(storage, trajectory_mode=2, end_x=18.0, end_y=4.5, delta_heading=0.1, ego_speed_km_h=22.0, front_distance=-1.0, next_ref_lane_count=2)
    _append_samples(storage, trajectory_mode=3, end_x=18.0, end_y=-4.5, delta_heading=-0.1, ego_speed_km_h=22.0, front_distance=-1.0, next_ref_lane_count=2)
    _append_samples(storage, trajectory_mode=0, end_x=12.0, end_y=3.2, delta_heading=-0.8, ego_speed_km_h=16.0, front_distance=-1.0, next_ref_lane_count=-1)
    _append_samples(storage, trajectory_mode=0, end_x=12.0, end_y=-3.2, delta_heading=0.8, ego_speed_km_h=16.0, front_distance=-1.0, next_ref_lane_count=-1)

    shard_path = shard_dir / "shard_000000.npz"
    np.savez_compressed(
        shard_path,
        **{
            key: np.asarray(value, dtype=np.float32 if key not in {"trajectory_mode", "lane_index", "reference_lane_index", "current_ref_lane_count", "next_ref_lane_count"} else None)
            for key, value in storage.items()
        },
    )
    return dataset_root


def test_label_distribution(tmp_path: Path):
    dataset_root = _build_dataset_root(tmp_path)
    shard_paths = resolve_shard_paths(dataset_root, "all")
    labels = label_dataset(shard_paths)["labels"]
    counts = {mode: int(np.sum(labels == int(mode))) for mode in BehaviorMode}
    assert all(count > 0 for count in counts.values())


def test_anchor_count(tmp_path: Path):
    dataset_root = _build_dataset_root(tmp_path)
    output_path = tmp_path / "anchors.npy"
    anchors = generate_plan_anchors(dataset_root, output_path, "all", "trajectory", 9, None, 0, 100, 1e-4)
    assert 0 < anchors.shape[0] <= 10


def test_medoid_is_real_sample(tmp_path: Path):
    dataset_root = _build_dataset_root(tmp_path)
    output_path = tmp_path / "anchors.npy"
    anchors = generate_plan_anchors(dataset_root, output_path, "all", "trajectory", 9, None, 0, 100, 1e-4)
    meta = json.loads(output_path.with_name("anchors_meta.json").read_text(encoding="utf-8"))
    for item in meta:
        with np.load(dataset_root / "shards" / item["source_shard"], allow_pickle=False) as shard:
            source = shard["trajectory"][item["source_sample_index"]]
        assert np.allclose(source, anchors[item["anchor_id"]])


def test_anchor_shape_compatible(tmp_path: Path):
    dataset_root = _build_dataset_root(tmp_path)
    output_path = tmp_path / "anchors.npy"
    anchors = generate_plan_anchors(dataset_root, output_path, "all", "trajectory", 9, None, 0, 100, 1e-4)
    assert anchors.ndim == 3
    assert anchors.shape[1] == 8
    assert anchors.shape[2] in {2, 3}


def test_semantic_purity(tmp_path: Path):
    dataset_root = _build_dataset_root(tmp_path)
    output_path = tmp_path / "anchors.npy"
    generate_plan_anchors(dataset_root, output_path, "all", "trajectory", 9, None, 0, 100, 1e-4)
    meta = json.loads(output_path.with_name("anchors_meta.json").read_text(encoding="utf-8"))
    assert all(item["semantic_purity"] >= 0.8 for item in meta)


def test_geometric_sanity(tmp_path: Path):
    dataset_root = _build_dataset_root(tmp_path)
    output_path = tmp_path / "anchors.npy"
    anchors = generate_plan_anchors(dataset_root, output_path, "all", "trajectory", 9, None, 0, 100, 1e-4)
    meta = json.loads(output_path.with_name("anchors_meta.json").read_text(encoding="utf-8"))
    for item in meta:
        traj = anchors[item["anchor_id"]]
        behavior = item["behavior_mode"]
        if behavior == BehaviorMode.CRUISE.name:
            assert abs(float(traj[-1, 1])) < 2.0
        if behavior in {BehaviorMode.LANE_CHANGE_LEFT.name, BehaviorMode.LANE_CHANGE_RIGHT.name}:
            assert 1.6 <= abs(float(traj[-1, 1])) <= 6.0
        if behavior in {BehaviorMode.TURN_LEFT.name, BehaviorMode.TURN_RIGHT.name}:
            assert abs(float(traj[-1, 2] - traj[0, 2])) > 0.5
        assert np.all(np.diff(traj[:, 0]) >= -1e-5)
