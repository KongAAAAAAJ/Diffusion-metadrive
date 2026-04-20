from __future__ import annotations

from pathlib import Path

from metadrive.policy.diffusion_policy.train_transfuser import create_next_run_dir, validate_runtime_paths
from metadrive.policy.diffusion_policy.transfuser_config import build_transfuser_config
from metadrive.policy.diffusion_policy.transfuser_model_v2 import TrajectoryHead


def test_create_next_run_dir_uses_existing_run_count(tmp_path: Path):
    output_root = tmp_path / "diffusion"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "checkpoints").mkdir()
    (output_root / "tb").mkdir()

    run_1 = create_next_run_dir(output_root)
    assert run_1.name == "run_1"
    assert run_1.exists()

    run_2 = create_next_run_dir(output_root)
    assert run_2.name == "run_2"
    assert run_2.exists()


def test_create_next_run_dir_ignores_non_run_directories(tmp_path: Path):
    output_root = tmp_path / "diffusion"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_1").mkdir()
    (output_root / "open_loop_eval").mkdir()
    (output_root / "closed_loop").mkdir()

    run_dir = create_next_run_dir(output_root)
    assert run_dir.name == "run_2"


def test_validate_runtime_paths_skips_anchor_requirement_in_dynamic_mode(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    shard_dir = dataset_root / "shards"
    split_dir = dataset_root / "splits"
    shard_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / "shard_000000.npz").write_bytes(b"placeholder")
    (split_dir / "train.txt").write_text("shard_000000\n", encoding="utf-8")
    (split_dir / "val.txt").write_text("shard_000000\n", encoding="utf-8")

    config = build_transfuser_config(
        "small",
        dataset_root=str(dataset_root),
        plan_anchor_path=str(tmp_path / "missing_anchors.npy"),
        use_dynamic_anchors=True,
    )

    validate_runtime_paths(config)


def test_trajectory_head_allows_missing_anchor_file_in_dynamic_mode(tmp_path: Path):
    config = build_transfuser_config(
        "small",
        plan_anchor_path=str(tmp_path / "missing_anchors.npy"),
        use_dynamic_anchors=True,
    )

    head = TrajectoryHead(
        num_poses=config.trajectory_sampling.num_poses,
        d_ffn=config.tf_d_ffn,
        d_model=config.tf_d_model,
        plan_anchor_path=config.plan_anchor_path,
        config=config,
    )

    assert tuple(head.plan_anchor.shape) == (
        config.ego_fut_mode,
        config.trajectory_sampling.num_poses,
        2,
    )
