from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "analysis" / "analysis_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("analysis_dataset_under_test", MODULE_PATH)
analysis_dataset = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(analysis_dataset)


def _write_shard(dataset_root: Path, name: str, trajectory: np.ndarray) -> None:
    shard_dir = dataset_root / "shards"
    split_dir = dataset_root / "splits"
    shard_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    np.savez(shard_dir / name, trajectory=trajectory.astype(np.float32))
    (split_dir / "train.txt").write_text(f"{name}\n", encoding="utf-8")


def test_normalize_trajectories_to_first_point_origin():
    trajectories = np.asarray(
        [
            [[1.0, 2.0], [3.0, 5.0], [6.0, 9.0]],
            [[-2.0, 1.0], [0.0, 1.5], [2.0, 3.0]],
        ],
        dtype=np.float32,
    )

    normalized = analysis_dataset.normalize_trajectory_xy_to_origin(trajectories)

    np.testing.assert_allclose(normalized[:, 0, :], 0.0)
    np.testing.assert_allclose(normalized[0], np.asarray([[0.0, 0.0], [2.0, 3.0], [5.0, 7.0]], dtype=np.float32))
    np.testing.assert_allclose(normalized[1], np.asarray([[0.0, 0.0], [2.0, 0.5], [4.0, 2.0]], dtype=np.float32))


def test_load_trajectory_dataset_and_stats_limit_max_trajectories(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    trajectories = np.asarray(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.2, 0.1]],
            [[0.0, 0.0, 0.0], [2.0, 1.0, 0.2]],
            [[0.0, 0.0, 0.0], [3.0, -1.5, -0.3]],
        ],
        dtype=np.float32,
    )
    _write_shard(dataset_root, "shard_000.npz", trajectories)

    loaded = analysis_dataset.load_trajectory_dataset(
        analysis_dataset.resolve_shard_paths(dataset_root, "all"),
        trajectory_key="trajectory",
        max_trajectories=2,
        rng=np.random.RandomState(0),
    )
    stats = analysis_dataset.summarize_trajectory_distribution(loaded)

    assert loaded.shape == (2, 2, 3)
    assert stats["trajectory_count"] == 2
    assert stats["trajectory_shape"] == [2, 3]
    assert stats["origin_max_abs"] == 0.0
    assert len(stats["endpoint_xy_min"]) == 2
    assert len(stats["endpoint_xy_max"]) == 2


def test_plot_trajectory_distribution_saves_png(tmp_path: Path):
    figure_path = tmp_path / "distribution.png"
    trajectories = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.5]],
            [[0.0, 0.0], [1.5, -0.2], [3.0, -0.4]],
        ],
        dtype=np.float32,
    )

    analysis_dataset.plot_trajectory_distribution(trajectories, figure_path=figure_path, show_figure=False)

    assert figure_path.is_file()
    assert figure_path.stat().st_size > 0


def test_parse_args_defaults_to_trajectory_distribution(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analysis_dataset.py",
            "--trajectory-key",
            "trajectory",
        ],
    )

    args = analysis_dataset.parse_args()

    assert args.trajectory_key == "trajectory"
    assert args.figure_path == analysis_dataset.DEFAULT_DISTRIBUTION_FIGURE_PATH
