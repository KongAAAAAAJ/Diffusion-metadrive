"""Analyze the distribution of trajectory samples in the dataset.

This script loads all trajectories from the requested dataset split, translates
each trajectory so that its first point becomes the origin, and overlays the
resulting xy traces into a single figure for quick distribution inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np


DEFAULT_DATASET_ROOT = Path("/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metaIDM")
DEFAULT_DISTRIBUTION_FIGURE_PATH = Path(__file__).resolve().parent / "trajectory_distribution.png"
DEFAULT_TRAJECTORY_KEY = "trajectory"


def resolve_shard_paths(dataset_root: Path, split: str) -> List[Path]:
    """Resolve dataset shards for a split."""
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


def load_trajectory_dataset(
    shard_paths: Sequence[Path],
    trajectory_key: str,
    max_trajectories: int | None,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Load trajectory tensors only."""
    chunks: list[np.ndarray] = []
    total_samples = 0
    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as shard:
            if trajectory_key not in shard:
                raise RuntimeError(
                    f"Shard {shard_path.name} is missing trajectory key {trajectory_key!r}"
                )
            trajectories = np.asarray(shard[trajectory_key], dtype=np.float32)
            chunks.append(trajectories)
            total_samples += int(trajectories.shape[0])

    if not chunks:
        return np.zeros((0, 0, 0), dtype=np.float32)

    merged = np.concatenate(chunks, axis=0)
    if max_trajectories is not None and max_trajectories > 0 and total_samples > max_trajectories:
        selected = np.sort(rng.choice(total_samples, size=max_trajectories, replace=False))
        merged = merged[selected]
    return merged


def normalize_trajectory_xy_to_origin(trajectories_xy: np.ndarray) -> np.ndarray:
    """Translate each trajectory so that its first point becomes the origin."""
    trajectories_xy = np.asarray(trajectories_xy, dtype=np.float32)
    if trajectories_xy.ndim != 3 or trajectories_xy.shape[-1] != 2:
        raise ValueError(f"Expected trajectory xy tensor with shape [N, T, 2], got {trajectories_xy.shape}")
    if trajectories_xy.shape[1] == 0:
        return trajectories_xy.copy()
    return trajectories_xy - trajectories_xy[:, :1, :]


def summarize_trajectory_distribution(trajectories: np.ndarray) -> dict[str, float | int | list[int] | list[float]]:
    """Compute a small set of stats for quick terminal inspection."""
    trajectories = np.asarray(trajectories, dtype=np.float32)
    if trajectories.ndim != 3:
        raise ValueError(f"Expected trajectory tensor with shape [N, T, D], got {trajectories.shape}")

    count = int(trajectories.shape[0])
    shape = [int(trajectories.shape[1]), int(trajectories.shape[2])]
    if count == 0:
        return {
            "trajectory_count": 0,
            "trajectory_shape": shape,
            "origin_max_abs": 0.0,
            "endpoint_xy_min": [0.0, 0.0],
            "endpoint_xy_max": [0.0, 0.0],
            "endpoint_xy_mean": [0.0, 0.0],
            "max_abs_lateral": 0.0,
            "max_forward_extent": 0.0,
        }

    xy = np.asarray(trajectories[..., :2], dtype=np.float32)
    endpoints = xy[:, -1, :]
    return {
        "trajectory_count": count,
        "trajectory_shape": shape,
        "origin_max_abs": float(np.max(np.abs(xy[:, 0, :]))),
        "endpoint_xy_min": endpoints.min(axis=0).astype(np.float32).tolist(),
        "endpoint_xy_max": endpoints.max(axis=0).astype(np.float32).tolist(),
        "endpoint_xy_mean": endpoints.mean(axis=0).astype(np.float32).tolist(),
        "max_abs_lateral": float(np.max(np.abs(xy[..., 1]))),
        "max_forward_extent": float(np.max(xy[..., 0])),
    }


def plot_trajectory_distribution(
    trajectories_xy: np.ndarray,
    figure_path: Path | None = None,
    show_figure: bool = True,
) -> None:
    """Overlay all normalized trajectories into a single figure."""
    import matplotlib.pyplot as plt

    trajectories_xy = np.asarray(trajectories_xy, dtype=np.float32)
    if trajectories_xy.ndim != 3 or trajectories_xy.shape[-1] != 2:
        raise ValueError(f"Expected trajectory xy tensor with shape [N, T, 2], got {trajectories_xy.shape}")

    fig, ax = plt.subplots(figsize=(10, 10))
    if trajectories_xy.shape[0] > 0:
        for traj in trajectories_xy:
            ax.plot(traj[:, 0], traj[:, 1], color="#1f77b4", alpha=0.03, linewidth=0.6)
        ax.scatter([0.0], [0.0], color="#d62728", s=20, zorder=3)
    ax.set_title(f"Trajectory Distribution ({trajectories_xy.shape[0]} samples)")
    ax.set_xlabel("Local X (meters)")
    ax.set_ylabel("Local Y (meters)")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    if figure_path is not None:
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        print(f"Saved trajectory distribution figure to {figure_path}")

    if show_figure:
        plt.show()
    else:
        plt.close(fig)


def analyze_trajectory_distribution(
    dataset_root: Path,
    split: str,
    trajectory_key: str,
    max_trajectories: int | None,
    seed: int,
    figure_path: Path | None,
    show_figure: bool,
) -> dict[str, float | int | list[int] | list[float]]:
    """Run the end-to-end trajectory distribution analysis."""
    rng = np.random.RandomState(seed)
    shard_paths = resolve_shard_paths(dataset_root, split)
    trajectories = load_trajectory_dataset(
        shard_paths,
        trajectory_key=trajectory_key,
        max_trajectories=max_trajectories,
        rng=rng,
    )
    trajectories_xy = normalize_trajectory_xy_to_origin(np.asarray(trajectories[..., :2], dtype=np.float32))
    stats = summarize_trajectory_distribution(trajectories_xy)
    print(json.dumps(stats, indent=2))
    plot_trajectory_distribution(trajectories_xy, figure_path=figure_path, show_figure=show_figure)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all dataset trajectories in one normalized distribution figure."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", type=str, default="all")
    parser.add_argument("--trajectory-key", type=str, default=DEFAULT_TRAJECTORY_KEY)
    parser.add_argument("--max-trajectories", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--figure-path", type=Path, default=DEFAULT_DISTRIBUTION_FIGURE_PATH)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_trajectory_distribution(
        dataset_root=args.dataset_root,
        split=args.split,
        trajectory_key=args.trajectory_key,
        max_trajectories=args.max_trajectories,
        seed=args.seed,
        figure_path=args.figure_path,
        show_figure=not args.no_show,
    )


if __name__ == "__main__":
    main()
