from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute basic statistics for a collected MetaDrive dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True, default=Path("/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metadrive_ppo_test"), help="Path to the root of the dataset.")
    return parser.parse_args(argv)


def _empty_summary() -> dict:
    return {
        "total_samples": 0,
        "scenario_coverage": {},
        "trajectory_stats": {
            axis: {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
            for axis in ("x", "y", "heading")
        },
    }


def _shard_files(dataset_root: Path) -> list[Path]:
    shard_dir = dataset_root / "shards"
    if not shard_dir.exists():
        return []
    return sorted(shard_dir.glob("*.npz"))


def _trajectory_axis_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def build_dataset_stats(dataset_root: Path) -> dict:
    shard_paths = _shard_files(dataset_root)
    if not shard_paths:
        return _empty_summary()

    total_samples = 0
    scenario_coverage: dict[str, int] = {}
    traj_x: list[np.ndarray] = []
    traj_y: list[np.ndarray] = []
    traj_heading: list[np.ndarray] = []

    for shard_path in shard_paths:
        with np.load(shard_path, allow_pickle=False) as shard:
            if "trajectory" not in shard:
                continue
            trajectory = np.asarray(shard["trajectory"], dtype=np.float32)
            shard_samples = int(trajectory.shape[0])
            total_samples += shard_samples
            traj_x.append(trajectory[..., 0].reshape(-1))
            traj_y.append(trajectory[..., 1].reshape(-1))
            traj_heading.append(trajectory[..., 2].reshape(-1))

            if "trajectory_mode" in shard:
                modes = np.asarray(shard["trajectory_mode"]).reshape(-1)
                unique, counts = np.unique(modes, return_counts=True)
                for mode_value, count in zip(unique.tolist(), counts.tolist()):
                    key = str(int(mode_value))
                    scenario_coverage[key] = scenario_coverage.get(key, 0) + int(count)
            else:
                scenario_coverage["unknown"] = scenario_coverage.get("unknown", 0) + shard_samples

    summary = {
        "total_samples": int(total_samples),
        "scenario_coverage": dict(sorted(scenario_coverage.items())),
        "trajectory_stats": {
            "x": _trajectory_axis_stats(np.concatenate(traj_x) if traj_x else np.asarray([], dtype=np.float32)),
            "y": _trajectory_axis_stats(np.concatenate(traj_y) if traj_y else np.asarray([], dtype=np.float32)),
            "heading": _trajectory_axis_stats(
                np.concatenate(traj_heading) if traj_heading else np.asarray([], dtype=np.float32)
            ),
        },
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stats = build_dataset_stats(args.dataset_root)
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
