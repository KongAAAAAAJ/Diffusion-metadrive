from __future__ import annotations

from pathlib import Path

import pytest

from metadrive.policy.diffusion_policy.train_transfuser import (
    build_checkpoint_callback,
    create_next_run_dir,
    resolve_resume_max_epochs,
    resolve_training_run_dir,
    validate_runtime_paths,
)
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


def test_resolve_training_run_dir_uses_checkpoint_parent_run_dir(tmp_path: Path):
    output_root = tmp_path / "diffusion"
    run_dir = output_root / "run_7"
    ckpt_path = run_dir / "checkpoints" / "diffusion-epoch=05.ckpt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path.write_bytes(b"placeholder")

    resolved_run_dir = resolve_training_run_dir(output_root, str(ckpt_path))

    assert resolved_run_dir == run_dir


def test_resolve_training_run_dir_rejects_checkpoint_outside_run_dir(tmp_path: Path):
    output_root = tmp_path / "diffusion"
    ckpt_path = tmp_path / "manual.ckpt"
    ckpt_path.write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="run_"):
        resolve_training_run_dir(output_root, str(ckpt_path))


def test_resolve_resume_max_epochs_uses_additional_epoch_semantics():
    assert resolve_resume_max_epochs(checkpoint_epoch=5, additional_epochs=10) == 16


def test_build_checkpoint_callback_keeps_best_three_and_latest(tmp_path: Path):
    callback = build_checkpoint_callback(tmp_path)

    assert callback.monitor == "val/loss"
    assert callback.mode == "min"
    assert callback.save_top_k == 3
    assert callback.save_last is True
    assert Path(callback.dirpath) == tmp_path


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
