from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute basic statistics for a collected MetaDrive dataset.")
    parser.add_argument("--dataset-root", type=Path, required=True, default=Path("/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metadrive_ppo_test"), help="Path to the root of the dataset.")
    parser.add_argument("--check-integrity", action="store_true", help="Validate sample count, skill coverage, value range, and finite values.")
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


def _discover_dataset_roots(dataset_root: Path) -> list[Path]:
    if (dataset_root / "shards").exists():
        return [dataset_root]
    if not dataset_root.exists():
        return []
    return sorted(path for path in dataset_root.iterdir() if path.is_dir() and (path / "shards").exists())


def _trajectory_axis_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _empty_integrity() -> dict:
    return {
        "dataset_roots": [],
        "sample_count_ok": False,
        "trajectory_range_ok": False,
        "finite_ok": False,
        "passed": False,
        "skill_coverage": {
            "straight": 0,
            "turn": 0,
            "lane_change": 0,
            "obstacle_avoidance": 0,
        },
        "non_finite_samples": 0,
    }


def build_dataset_stats(dataset_root: Path, check_integrity: bool = False) -> dict:
    dataset_roots = _discover_dataset_roots(dataset_root)
    shard_paths: list[Path] = []
    for root in dataset_roots:
        shard_paths.extend(_shard_files(root))
    if not shard_paths:
        return _empty_summary()

    total_samples = 0
    scenario_coverage: dict[str, int] = {}
    traj_x: list[np.ndarray] = []
    traj_y: list[np.ndarray] = []
    traj_heading: list[np.ndarray] = []
    straight_count = 0
    turn_count = 0
    lane_change_count = 0
    obstacle_avoidance_count = 0
    non_finite_samples = 0

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
            non_finite_samples += int((~np.isfinite(trajectory)).any(axis=(1, 2)).sum())
            turn_count += int((np.abs(trajectory[:, -1, 2] - trajectory[:, 0, 2]) > 0.25).sum())

            if "trajectory_mode" in shard:
                modes = np.asarray(shard["trajectory_mode"]).reshape(-1)
                unique, counts = np.unique(modes, return_counts=True)
                for mode_value, count in zip(unique.tolist(), counts.tolist()):
                    key = str(int(mode_value))
                    scenario_coverage[key] = scenario_coverage.get(key, 0) + int(count)
                straight_count += int(((modes == 0) | (modes == 1)).sum())
                lane_change_count += int(((modes == 2) | (modes == 3)).sum())
                obstacle_avoidance_count += int(((modes == 4) | (modes == 5)).sum())
            else:
                scenario_coverage["unknown"] = scenario_coverage.get("unknown", 0) + shard_samples

    x_stats = _trajectory_axis_stats(np.concatenate(traj_x) if traj_x else np.asarray([], dtype=np.float32))
    y_stats = _trajectory_axis_stats(np.concatenate(traj_y) if traj_y else np.asarray([], dtype=np.float32))
    heading_stats = _trajectory_axis_stats(np.concatenate(traj_heading) if traj_heading else np.asarray([], dtype=np.float32))
    summary = {
        "total_samples": int(total_samples),
        "dataset_roots": [str(path) for path in dataset_roots],
        "scenario_coverage": dict(sorted(scenario_coverage.items())),
        "trajectory_stats": {
            "x": x_stats,
            "y": y_stats,
            "heading": heading_stats,
        },
    }
    if check_integrity:
        trajectory_range_ok = (
            x_stats["min"] >= -5.0
            and x_stats["max"] <= 100.0
            and y_stats["min"] >= -30.0
            and y_stats["max"] <= 30.0
            and heading_stats["min"] >= -math.pi
            and heading_stats["max"] <= math.pi
        )
        integrity = {
            "dataset_roots": [str(path) for path in dataset_roots],
            "sample_count_ok": total_samples >= 500,
            "trajectory_range_ok": bool(trajectory_range_ok),
            "finite_ok": non_finite_samples == 0,
            "skill_coverage": {
                "straight": int(straight_count),
                "turn": int(turn_count),
                "lane_change": int(lane_change_count),
                "obstacle_avoidance": int(obstacle_avoidance_count),
            },
            "non_finite_samples": int(non_finite_samples),
        }
        integrity["passed"] = bool(
            integrity["sample_count_ok"]
            and integrity["trajectory_range_ok"]
            and integrity["finite_ok"]
            and integrity["skill_coverage"]["straight"] >= 100
            and integrity["skill_coverage"]["turn"] >= 50
            and integrity["skill_coverage"]["lane_change"] >= 30
            and integrity["skill_coverage"]["obstacle_avoidance"] >= 20
        )
        summary["integrity"] = integrity
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stats = build_dataset_stats(args.dataset_root, check_integrity=bool(args.check_integrity))
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
