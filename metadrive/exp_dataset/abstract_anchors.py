"""
【难以基于规则对语义行为进行精确划分，各个行为模式的边界不清晰，存在重叠，需要再想别的办法解决】
生成MetaDrive语义计划锚点的脚本，使用基于行为模式的分桶K-Medoids聚类方法，从原始轨迹数据中提取具有代表性的锚点轨迹。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from metadrive.exp_dataset.semantic_labeler import BehaviorMode, REQUIRED_LABEL_FIELDS, label_dataset, label_sample


DEFAULT_DATASET_ROOT = Path("/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metaData_strong_idm_single")
DEFAULT_NUM_ANCHORS = 9
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "metadrive_anchors.npy"
DEFAULT_FIGURE_PATH = Path(__file__).resolve().parent / "metadrive_anchors.png"
DEFAULT_TRAJECTORY_KEY = "trajectory"
TARGET_SUBTYPES = {         # 各行为模式的目标子类数量
    BehaviorMode.CRUISE: 2,
    BehaviorMode.FOLLOW: 2,
    BehaviorMode.DECELERATE: 1,
    BehaviorMode.LANE_CHANGE_LEFT: 1,
    BehaviorMode.LANE_CHANGE_RIGHT: 1,
    BehaviorMode.TURN_LEFT: 1,
    BehaviorMode.TURN_RIGHT: 1,
}
BEHAVIOR_COLORS = {
    BehaviorMode.CRUISE: "#1f77b4",
    BehaviorMode.FOLLOW: "#ff7f0e",
    BehaviorMode.DECELERATE: "#9467bd",
    BehaviorMode.LANE_CHANGE_LEFT: "#2ca02c",
    BehaviorMode.LANE_CHANGE_RIGHT: "#17becf",
    BehaviorMode.TURN_LEFT: "#8c564b",
    BehaviorMode.TURN_RIGHT: "#e377c2",
}
BEHAVIOR_PRIORITY = [
    BehaviorMode.CRUISE,
    BehaviorMode.FOLLOW,
    BehaviorMode.DECELERATE,
    BehaviorMode.LANE_CHANGE_LEFT,
    BehaviorMode.LANE_CHANGE_RIGHT,
    BehaviorMode.TURN_LEFT,
    BehaviorMode.TURN_RIGHT,
]
_LAST_CLUSTER_CANDIDATES: list[dict] = []


def resolve_shard_paths(dataset_root: Path, split: str) -> List[Path]:
    """加载数据集分片路径列表"""
    shard_dir = dataset_root / "shards"
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"Shard directory does not exist: {shard_dir}")

    shard_paths = sorted(shard_dir.glob("*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found under {shard_dir}")

    normalized_split = split.lower()
    if normalized_split == "all":
        return shard_paths

    split_file = dataset_root / "splits" / f"{normalized_split}.txt"
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file does not exist: {split_file}")

    shard_names = {line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()}
    resolved = [path for path in shard_paths if path.name in shard_names]
    if not resolved:
        raise FileNotFoundError(f"Split {normalized_split!r} did not match any shard under {shard_dir}")
    return resolved


def _load_anchor_dataset(
    shard_paths: Sequence[Path],
    trajectory_key: str,
    max_trajectories: int | None,
    rng: np.random.RandomState,
) -> dict[str, np.ndarray]:
    """加载包含轨迹和语义标注的原始数据集，返回一个合并后的字典，包含所有样本的轨迹和相关特征。"""
    required_keys = set(REQUIRED_LABEL_FIELDS)
    required_keys.add(trajectory_key)
    merged: dict[str, list[np.ndarray]] = {key: [] for key in required_keys}
    source_shards: list[np.ndarray] = []
    source_sample_indices: list[np.ndarray] = []
    sample_indices: list[np.ndarray] = []
    global_offset = 0

    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as shard:
            missing = [key for key in required_keys if key not in shard]
            if missing:
                raise RuntimeError(
                    f"Shard {shard_path.name} is missing required semantic-anchor fields: {', '.join(sorted(missing))}"
                )
            sample_count = int(shard[trajectory_key].shape[0])
            for key in required_keys:
                merged[key].append(np.asarray(shard[key]))
            source_shards.append(np.asarray([shard_path.name] * sample_count, dtype="U128"))
            source_sample_indices.append(np.arange(sample_count, dtype=np.int64))
            sample_indices.append(np.arange(global_offset, global_offset + sample_count, dtype=np.int64))
            global_offset += sample_count

    dataset = {key: np.concatenate(chunks, axis=0) for key, chunks in merged.items()}
    dataset["source_shard"] = np.concatenate(source_shards, axis=0)
    dataset["source_sample_index"] = np.concatenate(source_sample_indices, axis=0)
    dataset["sample_index"] = np.concatenate(sample_indices, axis=0)
    dataset["trajectory_output"] = np.asarray(dataset[trajectory_key], dtype=np.float32)
    if trajectory_key != "trajectory":
        dataset["trajectory"] = np.asarray(dataset[trajectory_key], dtype=np.float32)

    if max_trajectories is not None and max_trajectories > 0 and dataset["trajectory_output"].shape[0] > max_trajectories:
        selected = np.sort(rng.choice(dataset["trajectory_output"].shape[0], size=max_trajectories, replace=False))
        dataset = {key: value[selected] for key, value in dataset.items()}
    return dataset


def _compute_curvature(trajectory: np.ndarray) -> float:
    if trajectory.shape[0] < 2:
        return 0.0
    heading_diff = np.diff(trajectory[:, 2])
    return float(np.mean(np.abs(heading_diff)))


def _feature_vector(mode: BehaviorMode, sample: Mapping[str, np.ndarray | float | int]) -> np.ndarray:
    trajectory = np.asarray(sample["trajectory"], dtype=np.float32)
    end_x = float(trajectory[-1, 0])
    end_y = float(trajectory[-1, 1])
    max_abs_y = float(np.max(np.abs(trajectory[:, 1])))
    delta_heading = float(trajectory[-1, 2] - trajectory[0, 2])

    if mode == BehaviorMode.CRUISE:
        return np.asarray([end_x, float(np.mean(trajectory[:, 1])), max_abs_y], dtype=np.float64)
    if mode == BehaviorMode.FOLLOW:
        longitudinal_var = float(np.var(np.diff(trajectory[:, 0]))) if trajectory.shape[0] > 1 else 0.0
        front_distance = float(np.asarray(sample["front_object_distance"]).reshape(-1)[0])
        return np.asarray([end_x, longitudinal_var, front_distance], dtype=np.float64)
    if mode == BehaviorMode.DECELERATE:
        max_heading_change = float(np.max(np.abs(np.diff(trajectory[:, 2])))) if trajectory.shape[0] > 1 else 0.0
        return np.asarray([end_x, max_heading_change], dtype=np.float64)
    if mode in {BehaviorMode.LANE_CHANGE_LEFT, BehaviorMode.LANE_CHANGE_RIGHT}:
        return np.asarray([end_y, max_abs_y, end_x], dtype=np.float64)
    return np.asarray([delta_heading, _compute_curvature(trajectory), end_x], dtype=np.float64)


def _pairwise_distance(data: np.ndarray) -> np.ndarray:
    diff = data[:, None, :] - data[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def _init_medoids(data: np.ndarray, num_clusters: int, rng: np.random.RandomState) -> np.ndarray:
    num_samples = data.shape[0]
    medoids = [int(rng.randint(num_samples))]
    min_dist = np.linalg.norm(data - data[medoids[0]], axis=1)
    for _ in range(1, num_clusters):
        total = float(np.sum(min_dist))
        if total <= 0.0:
            candidate = int(rng.randint(num_samples))
        else:
            probs = min_dist / total
            candidate = int(rng.choice(num_samples, p=probs))
        if candidate in medoids:
            candidate = int(rng.choice([idx for idx in range(num_samples) if idx not in medoids]))
        medoids.append(candidate)
        min_dist = np.minimum(min_dist, np.linalg.norm(data - data[candidate], axis=1))
    return np.asarray(medoids, dtype=np.int64)


def run_kmedoids(data: np.ndarray, num_clusters: int, seed: int, max_iters: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    if num_clusters <= 0:
        raise ValueError("num_clusters must be > 0")
    if data.shape[0] < num_clusters:
        raise ValueError("num_clusters exceeds available samples")
    if data.shape[0] == num_clusters:
        medoids = np.arange(data.shape[0], dtype=np.int64)
        assignments = np.arange(data.shape[0], dtype=np.int64)
        return medoids, assignments, 0.0

    rng = np.random.RandomState(seed)
    pairwise = _pairwise_distance(data)
    medoids = _init_medoids(data, num_clusters, rng)

    for _ in range(max_iters):
        dist_to_medoids = pairwise[:, medoids]
        assignments = np.argmin(dist_to_medoids, axis=1)
        updated = medoids.copy()
        for cluster_idx in range(num_clusters):
            member_indices = np.where(assignments == cluster_idx)[0]
            if member_indices.size == 0:
                remaining = [idx for idx in range(data.shape[0]) if idx not in updated]
                updated[cluster_idx] = int(rng.choice(remaining))
                continue
            intra = pairwise[np.ix_(member_indices, member_indices)]
            costs = intra.sum(axis=1)
            updated[cluster_idx] = int(member_indices[int(np.argmin(costs))])
        if np.array_equal(np.sort(updated), np.sort(medoids)):
            medoids = updated
            break
        medoids = updated

    dist_to_medoids = pairwise[:, medoids]
    assignments = np.argmin(dist_to_medoids, axis=1)
    inertia = float(np.sum(dist_to_medoids[np.arange(data.shape[0]), assignments]))
    return medoids, assignments, inertia


def _zscore(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (features - mean) / std


def _bucket_candidates(dataset: dict[str, np.ndarray], labels: np.ndarray, mode: BehaviorMode) -> np.ndarray:
    return np.where(labels == int(mode))[0]


def _label_loaded_dataset(dataset: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(
        [
            int(
                label_sample(
                    {
                        "trajectory": dataset["trajectory_output"][idx],
                        "trajectory_mode": dataset["trajectory_mode"][idx],
                        "ego_speed_km_h": dataset["ego_speed_km_h"][idx],
                        "front_object_distance": dataset["front_object_distance"][idx],
                        "front_object_speed_km_h": dataset["front_object_speed_km_h"][idx],
                        "lane_index": dataset["lane_index"][idx],
                        "reference_lane_index": dataset["reference_lane_index"][idx],
                        "reference_longitudinal": dataset["reference_longitudinal"][idx],
                        "reference_lateral": dataset["reference_lateral"][idx],
                        "lane_width": dataset["lane_width"][idx],
                        "current_ref_lane_count": dataset["current_ref_lane_count"][idx],
                        "next_ref_lane_count": dataset["next_ref_lane_count"][idx],
                        "reference_pose_world": dataset["reference_pose_world"][idx],
                    }
                )
            )
            for idx in range(dataset["trajectory_output"].shape[0])
        ],
        dtype=np.int16,
    )


def _cluster_bucket(
    dataset: dict[str, np.ndarray],
    labels: np.ndarray,
    mode: BehaviorMode,
    max_clusters: int,
    seed: int,
) -> list[dict]:
    candidate_indices = _bucket_candidates(dataset, labels, mode)
    if candidate_indices.size == 0:
        return []

    cluster_count = min(int(max_clusters), int(candidate_indices.size))
    features = np.stack(
        [
            _feature_vector(
                mode,
                {key: value[idx] for key, value in dataset.items() if isinstance(value, np.ndarray)},
            )
            for idx in candidate_indices
        ],
        axis=0,
    )
    scaled = _zscore(features)
    medoid_local, assignments, _ = run_kmedoids(scaled, cluster_count, seed=seed + int(mode))

    results = []
    for subtype_id, local_medoid_idx in enumerate(medoid_local.tolist()):
        cluster_member_local = np.where(assignments == subtype_id)[0]
        cluster_member_global = candidate_indices[cluster_member_local]
        medoid_global = int(candidate_indices[local_medoid_idx])
        cluster_dist = np.linalg.norm(scaled[cluster_member_local] - scaled[local_medoid_idx], axis=1)
        trajectory_modes = dataset["trajectory_mode"][cluster_member_global]
        unique_modes, counts = np.unique(trajectory_modes, return_counts=True)
        mode_hist = {str(int(mode_id)): int(count) for mode_id, count in zip(unique_modes.tolist(), counts.tolist())}
        results.append(
            {
                "behavior_mode": mode,
                "subtype_id": subtype_id,
                "global_index": medoid_global,
                "cluster_size": int(cluster_member_global.size),
                "cluster_inertia": float(np.sum(cluster_dist)),
                "semantic_purity": 1.0,
                "trajectory_mode_histogram": mode_hist,
                "feature_points": features[cluster_member_local, :2].astype(np.float32),
                "medoid_feature": features[local_medoid_idx, :2].astype(np.float32),
            }
        )
    return results


def _trim_to_max_anchors(candidates: list[dict], max_anchors: int) -> list[dict]:
    ordered = sorted(candidates, key=lambda item: (BEHAVIOR_PRIORITY.index(item["behavior_mode"]), item["subtype_id"]))
    if len(ordered) <= max_anchors:
        return ordered
    return ordered[:max_anchors]


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _meta_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.stem + "_meta.json")


def _stats_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.stem + "_stats.json")


def _distribution_figure_path(figure_path: Path) -> Path:
    return figure_path.with_name(figure_path.stem + "_distribution.png")


def _scatter_dir(figure_path: Path) -> Path:
    return figure_path.with_name(figure_path.stem + "_buckets")


def plot_plan_anchors(anchors: np.ndarray, meta: list[dict] | None = None, figure_path: Path | None = None, show_figure: bool = True) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    for anchor_idx in range(anchors.shape[0]):
        traj = anchors[anchor_idx]
        if meta is not None:
            behavior = BehaviorMode[meta[anchor_idx]["behavior_mode"]]
            color = BEHAVIOR_COLORS[behavior]
            label = f"{meta[anchor_idx]['behavior_mode']}-{meta[anchor_idx]['subtype_id']}"
        else:
            color = "#1f77b4"
            label = f"Mode {anchor_idx}"
        plt.plot(traj[:, 0], traj[:, 1], marker="o", color=color, label=label)
    plt.title(f"Semantic Plan Anchors ({anchors.shape[0]} modes)")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.legend()
    plt.grid()
    plt.axis("equal")

    if figure_path is not None:
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        print(f"Saved anchor figure to {figure_path}")

    if show_figure:
        plt.show()
    else:
        plt.close()


def _plot_label_distribution(label_distribution: dict[str, int], figure_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [key for key, value in label_distribution.items() if value > 0]
    sizes = [label_distribution[key] for key in labels]
    if not labels:
        return
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%")
    plt.title("Semantic Behavior Distribution")
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_bucket_scatters(cluster_candidates: list[dict], figure_path: Path) -> None:
    import matplotlib.pyplot as plt

    scatter_dir = _scatter_dir(figure_path)
    scatter_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict]] = {}
    for item in cluster_candidates:
        grouped.setdefault(item["behavior_mode"].name, []).append(item)
    for behavior_name, items in grouped.items():
        plt.figure(figsize=(6, 5))
        for item in items:
            points = item["feature_points"]
            plt.scatter(points[:, 0], points[:, 1], alpha=0.3, label=f"cluster-{item['subtype_id']}")
            medoid = item["medoid_feature"]
            plt.scatter([medoid[0]], [medoid[1]], marker="x", s=100, color="black")
        plt.title(f"{behavior_name} feature scatter")
        plt.xlabel("feature_0")
        plt.ylabel("feature_1")
        plt.legend()
        plt.savefig(scatter_dir / f"{behavior_name.lower()}.png", dpi=300, bbox_inches="tight")
        plt.close()


def generate_plan_anchors(
    dataset_root: Path,
    output_path: Path,
    split: str,
    trajectory_key: str,
    num_anchors: int,
    max_trajectories: int | None,
    seed: int,
    max_iters: int,
    tolerance: float,
) -> np.ndarray:
    global _LAST_CLUSTER_CANDIDATES
    del tolerance
    rng = np.random.RandomState(seed)

    # 1. 加载数据集分片路径列表
    shard_paths = resolve_shard_paths(dataset_root, split)
    dataset = _load_anchor_dataset(shard_paths, trajectory_key=trajectory_key, max_trajectories=max_trajectories, rng=rng)

    # 2. 标注数据集样本的行为模式
    # 如果数据集中已经包含 trajectory_mode 字段，则直接使用；否则调用 label_sample 进行标注
    labels_payload = label_dataset(list(shard_paths))
    if trajectory_key == "trajectory" and dataset["trajectory_output"].shape[0] == labels_payload["labels"].shape[0]:
        labels = labels_payload["labels"]
    else:
        labels = _label_loaded_dataset(dataset)

    # 3. 对每个行为模式进行 K-Medoids 聚类，生成候选锚点
    cluster_candidates: list[dict] = []
    for mode in BEHAVIOR_PRIORITY:
        cluster_candidates.extend(
            _cluster_bucket(
                dataset,
                labels,
                mode,
                max_clusters=TARGET_SUBTYPES[mode],
                seed=seed,
            )
        )
    cluster_candidates = _trim_to_max_anchors(cluster_candidates, num_anchors)  # 上限截断，最终锚点数量不超过 num_anchors
    _LAST_CLUSTER_CANDIDATES = cluster_candidates

    # 4. 从候选锚点中提取轨迹输出，并保存最终的锚点数组和相关元信息
    anchors = []
    meta = []
    anchors_per_class = {mode.name: 0 for mode in BEHAVIOR_PRIORITY}
    label_distribution = {mode.name: int(np.sum(labels == int(mode))) for mode in BEHAVIOR_PRIORITY}
    for anchor_id, item in enumerate(cluster_candidates):
        global_index = item["global_index"]
        behavior_mode = item["behavior_mode"]
        anchors.append(np.asarray(dataset["trajectory_output"][global_index][..., :2], dtype=np.float32))
        anchors_per_class[behavior_mode.name] += 1
        meta.append(
            {
                "anchor_id": anchor_id,
                "behavior_mode": behavior_mode.name,
                "subtype_id": int(item["subtype_id"]),
                "source_sample_index": int(dataset["source_sample_index"][global_index]),
                "sample_index": int(dataset["sample_index"][global_index]),
                "source_shard": str(dataset["source_shard"][global_index]),
                "cluster_size": int(item["cluster_size"]),
                "cluster_inertia": float(item["cluster_inertia"]),
                "semantic_purity": float(item["semantic_purity"]),
                "trajectory_mode_histogram": item["trajectory_mode_histogram"],
            }
        )

    # 将锚点轨迹保存为 NumPy 数组，并将元信息和统计信息保存为 JSON 文件
    anchors_array = np.stack(anchors, axis=0) if anchors else np.zeros((0, 8, 2), dtype=np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, anchors_array)
    _write_json(_meta_path(output_path), meta)
    _write_json(
        _stats_path(output_path),
        {
            "total_samples": int(dataset["trajectory_output"].shape[0]),
            "label_distribution": label_distribution,
            "anchor_count": int(anchors_array.shape[0]),
            "anchors_per_class": anchors_per_class,
        },
    )
    print(
        f"Saved {anchors_array.shape[0]} semantic plan anchors to {output_path} "
        f"with shape={anchors_array.shape} dtype={anchors_array.dtype}"
    )
    return anchors_array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute MetaDrive semantic plan anchors with bucketed K-Medoids clustering.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--split", type=str, default="all")
    parser.add_argument("--trajectory-key", type=str, default=DEFAULT_TRAJECTORY_KEY)
    parser.add_argument("--num-anchors", type=int, default=DEFAULT_NUM_ANCHORS)
    parser.add_argument("--max-trajectories", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iters", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--visualize", default=True)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_FIGURE_PATH)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    anchors = generate_plan_anchors(
        dataset_root=args.dataset_root,
        output_path=args.output_path,
        split=args.split,
        trajectory_key=args.trajectory_key,
        num_anchors=args.num_anchors,
        max_trajectories=args.max_trajectories,
        seed=args.seed,
        max_iters=args.max_iters,
        tolerance=args.tolerance,
    )
    if args.visualize:
        figure_path = args.figure_path
        meta = json.loads(_meta_path(args.output_path).read_text(encoding="utf-8"))
        plot_plan_anchors(anchors, meta=meta, figure_path=figure_path, show_figure=not args.no_show)
        _plot_label_distribution(
            json.loads(_stats_path(args.output_path).read_text(encoding="utf-8"))["label_distribution"],
            _distribution_figure_path(figure_path),
        )
        _plot_bucket_scatters(_LAST_CLUSTER_CANDIDATES, figure_path)


if __name__ == "__main__":
    main()
