from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


DEFAULT_DATASET_ROOT = Path("/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/1_metaData_idm_single")
DEFAULT_NUM_ANCHORS = 8
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / f"metadrive_anchors.npy"
DEFAULT_FIGURE_PATH = Path(__file__).resolve().parent / f"metadrive_anchors.png"
DEFAULT_TRAJECTORY_KEY = "trajectory"


def resolve_shard_paths(dataset_root: Path, split: str) -> List[Path]:
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


def load_trajectory_matrix(
	shard_paths: Sequence[Path],
	trajectory_key: str,
	max_trajectories: int | None,
	rng: np.random.RandomState,
) -> np.ndarray:
	chunks: List[np.ndarray] = []
	total_samples = 0
	for shard_path in shard_paths:
		with np.load(shard_path, allow_pickle=False) as shard:
			if trajectory_key not in shard:
				raise KeyError(
					f"Trajectory key {trajectory_key!r} not found in {shard_path.name}; "
					f"available_keys={sorted(shard.files)}"
				)
			trajectory_xy = shard[trajectory_key][..., :2].astype(np.float64)
		chunks.append(trajectory_xy)
		total_samples += int(trajectory_xy.shape[0])

	if not chunks:
		raise ValueError("No trajectories were loaded from the selected shards")

	trajectories = np.concatenate(chunks, axis=0)
	if max_trajectories is not None and max_trajectories > 0 and trajectories.shape[0] > max_trajectories:
		selected_indices = rng.choice(trajectories.shape[0], size=max_trajectories, replace=False)
		trajectories = trajectories[selected_indices]

	return trajectories.reshape(trajectories.shape[0], -1)


def pairwise_squared_l2(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
	data_norm = np.sum(data * data, axis=1, keepdims=True)
	center_norm = np.sum(centers * centers, axis=1, keepdims=True).T
	dist2 = data_norm + center_norm - 2.0 * data @ centers.T
	return np.maximum(dist2, 0.0)


def init_kmeans_plus_plus(data: np.ndarray, num_clusters: int, rng: np.random.RandomState) -> np.ndarray:
	num_samples = data.shape[0]
	if num_clusters > num_samples:
		raise ValueError(f"num_clusters={num_clusters} exceeds available trajectories={num_samples}")

	centers = np.empty((num_clusters, data.shape[1]), dtype=np.float64)
	first_idx = int(rng.randint(num_samples))
	centers[0] = data[first_idx]
	closest_dist2 = np.sum((data - centers[0]) ** 2, axis=1)

	for center_idx in range(1, num_clusters):
		total = float(closest_dist2.sum())
		if total <= 0.0:
			candidate_idx = int(rng.randint(num_samples))
		else:
			probs = closest_dist2 / total
			candidate_idx = int(rng.choice(num_samples, p=probs))
		centers[center_idx] = data[candidate_idx]
		new_dist2 = np.sum((data - centers[center_idx]) ** 2, axis=1)
		closest_dist2 = np.minimum(closest_dist2, new_dist2)

	return centers


def run_kmeans(
	data: np.ndarray,
	num_clusters: int,
	seed: int,
	max_iters: int,
	tolerance: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
	rng = np.random.RandomState(seed)
	centers = init_kmeans_plus_plus(data, num_clusters, rng)
	labels = np.zeros(data.shape[0], dtype=np.int64)
	inertia = float("inf")

	for _ in range(max_iters):
		dist2 = pairwise_squared_l2(data, centers)
		new_labels = np.argmin(dist2, axis=1)
		new_inertia = float(dist2[np.arange(data.shape[0]), new_labels].sum())

		new_centers = centers.copy()
		for cluster_idx in range(num_clusters):
			members = data[new_labels == cluster_idx]
			if len(members) == 0:
				new_centers[cluster_idx] = data[int(rng.randint(data.shape[0]))]
			else:
				new_centers[cluster_idx] = members.mean(axis=0)

		center_shift = float(np.linalg.norm(new_centers - centers))
		centers = new_centers
		labels = new_labels
		inertia = new_inertia
		if center_shift <= tolerance:
			break

	return centers, labels, inertia


def sort_anchor_modes(anchors: np.ndarray) -> np.ndarray:
	end_points = anchors[:, -1, :]
	order = np.lexsort((end_points[:, 1], end_points[:, 0]))
	return anchors[order]

def plot_plan_anchors(anchors: np.ndarray, figure_path: Path | None = None, show_figure: bool = True) -> None:
	import matplotlib.pyplot as plt

	plt.figure(figsize=(8, 8))
	for mode_idx in range(anchors.shape[0]):
		traj = anchors[mode_idx]
		plt.plot(traj[:, 0], traj[:, 1], marker="o", label=f"Mode {mode_idx}")
	plt.title(f"Plan Anchor Trajectories ({anchors.shape[0]} modes, {anchors.shape[1]} poses each)")
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
	rng = np.random.RandomState(seed)
	shard_paths = resolve_shard_paths(dataset_root, split)
	data = load_trajectory_matrix(
		shard_paths,
		trajectory_key=trajectory_key,
		max_trajectories=max_trajectories,
		rng=rng,
	)
	centers, _, inertia = run_kmeans(
		data=data,
		num_clusters=num_anchors,
		seed=seed,
		max_iters=max_iters,
		tolerance=tolerance,
	)

	anchors = centers.reshape(num_anchors, -1, 2)
	anchors = sort_anchor_modes(anchors).astype(np.float64)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	np.save(output_path, anchors)
	print(
		f"Saved {num_anchors} plan anchors to {output_path} "
		f"with shape={anchors.shape} dtype={anchors.dtype} inertia={inertia:.3f}"
	)
	return anchors


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Compute MetaDrive plan anchors with K-Means clustering.")
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
		if figure_path is None:
			figure_path = args.output_path.with_suffix(".png") if args.no_show else None
		plot_plan_anchors(anchors, figure_path=figure_path, show_figure=not args.no_show)

if __name__ == "__main__":
	main()
