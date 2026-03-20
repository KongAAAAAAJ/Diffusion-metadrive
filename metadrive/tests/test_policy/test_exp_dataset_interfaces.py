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
    module = _load_module("abstract_anchors_test", REPO_ROOT / "metadrive/exp_dataset/abstract_anchors.py")
    shard_path = tmp_path / "shard_000000.npz"
    np.savez(
        shard_path,
        trajectory=np.zeros((2, 8, 3), dtype=np.float32),
        trajectory_raw=np.ones((2, 8, 3), dtype=np.float32),
    )

    data = module.load_trajectory_matrix(
        shard_paths=[shard_path],
        trajectory_key="trajectory_raw",
        max_trajectories=None,
        rng=np.random.RandomState(0),
    )

    assert data.shape == (2, 16)
    assert np.allclose(data, 1.0)


def test_abstract_anchors_reports_available_keys_for_missing_trajectory_key(tmp_path: Path):
    module = _load_module("abstract_anchors_test_missing", REPO_ROOT / "metadrive/exp_dataset/abstract_anchors.py")
    shard_path = tmp_path / "shard_000000.npz"
    np.savez(shard_path, trajectory=np.zeros((1, 8, 3), dtype=np.float32))

    with pytest.raises(KeyError, match="available_keys"):
        module.load_trajectory_matrix(
            shard_paths=[shard_path],
            trajectory_key="trajectory_raw",
            max_trajectories=None,
            rng=np.random.RandomState(0),
        )


def test_run_expert_parse_args_supports_expert_type_and_debug_flag():
    module = _load_module("run_expert_test", REPO_ROOT / "metadrive/exp_dataset/run_expert.py")

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
