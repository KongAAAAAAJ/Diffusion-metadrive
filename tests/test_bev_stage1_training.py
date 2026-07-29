from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from expert_dataset.collect_joint_bev import (
    AgentRole,
    JOINT_SAMPLE_DTYPES,
    JOINT_SAMPLE_SHAPES,
    JointBEVSample,
)
from expert_dataset.joint_bev_dataset import JointBEVDataset, JointBEVDatasetConfig
from expert_dataset.joint_bev_storage import (
    EpisodeSplitConfig,
    JointBEVDatasetStore,
    fingerprint_payload,
)
from models.bev_planner import (
    BEVOnlyDiffusionPlanner,
    BEVOnlyDiffusionPlannerConfig,
    JointStage1Loss,
)
from models.bev_planner.mode_contract import ModeIndex
from train.train_bev_diffusion_stage1 import (
    Stage1TrainingError,
    build_stage1_optimizer,
    checkpoint_payload,
    deterministic_overfit_noise,
    evaluate_overfit_fixed,
    load_stage1_checkpoint,
    loss_from_batch,
    module_gradient_norms,
    planner_forward_from_batch,
    role_gradient_diagnostics,
    save_stage1_checkpoint,
    train_one_epoch,
    validate_stage1_config,
)


def _config() -> dict[str, object]:
    return {
        "experiment": {
            "variant": "A",
            "joint_update": "joint_mean",
            "predecessor_condition": "none",
        },
        "dataset": {
            "root": "/tmp/data",
            "train_split": "train",
            "val_split": "val",
        },
        "training": {
            "device": "cpu",
            "mixed_precision": False,
            "grad_scaler_init_scale": 1024.0,
            "seed": 17,
            "batch_size": 8,
            "gradient_accumulation": 2,
            "max_epochs": 2,
            "num_workers": 0,
            "planner_lr": 1e-4,
            "backbone_lr": 1e-5,
            "weight_decay": 1e-4,
            "gradient_diagnostics_interval": 100,
        },
        "loss": {
            "xy_weight": 1.0,
            "heading_weight": 0.2,
            "mode_weight": 1.0,
            "motion_weight": 0.05,
            "pair_weight": 0.25,
            "xy_beta_m": 1.0,
            "heading_beta_rad": 0.1,
            "motion_beta": 1.0,
            "pair_beta_m": 1.0,
            "dt_s": 0.5,
        },
        "overfit": {
            "max_optimizer_steps": 500,
            "evaluation_interval_steps": 20,
            "timestep": 8,
            "min_loss_reduction": 0.70,
            "min_mode_accuracy": 0.95,
            "max_gt_mode_ade_m": 1.0,
        },
    }


def _sample(marker: int) -> JointBEVSample:
    values = {
        name: np.zeros(shape, dtype=JOINT_SAMPLE_DTYPES[name])
        for name, shape in JOINT_SAMPLE_SHAPES.items()
    }
    values["bev"][:, 0] = 255
    values["ego_state"][:, 0] = 6.0
    values["ego_pose_global"][:, 0] = np.asarray((20.0, 10.0, 0.0))
    values["formation_relation_state"][:] = np.float32(marker * 0.01)
    values["relation_valid_mask"][:] = True
    values["agent_role"] = np.asarray(list(AgentRole), dtype=np.int64)
    times = np.arange(1, 9, dtype=np.float32) * np.float32(0.5)
    for mode in range(10):
        values["coarse_trajectories"][:, mode, :, 0] = times * np.float32(
            6.0
        ) + np.float32(mode * 0.05)
    values["mode_valid_mask"][:] = True
    values["gt_mode"][:] = int(ModeIndex.KEEP_MEDIUM)
    values["expert_trajectory"][:] = values["coarse_trajectories"][
        :, int(ModeIndex.KEEP_MEDIUM)
    ]
    return JointBEVSample(**values)


def _packed_batch(tmp_path: Path, samples: int = 2) -> dict[str, torch.Tensor]:
    root = tmp_path / "dataset"
    with JointBEVDatasetStore(
        root,
        split_config=EpisodeSplitConfig(1.0, 0.0, 0.0, seed=7),
        dataset_fingerprint=fingerprint_payload({"round": 8}),
        resume=False,
    ) as store:
        store.commit_episode(
            0,
            [_sample(index) for index in range(samples)],
            {"scenario_id": "round8_test", "local_route": "straight"},
        )
    dataset = JointBEVDataset(JointBEVDatasetConfig(root, "train"))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    dataset.close()
    return batch


@pytest.fixture(scope="module")
def planner() -> BEVOnlyDiffusionPlanner:
    return BEVOnlyDiffusionPlanner(
        BEVOnlyDiffusionPlannerConfig(
            d_model=32,
            num_heads=4,
            ffn_dim=64,
            decoder_layers=1,
        )
    )


def test_config_rejects_non_a_and_non_fixed_overfit_timestep() -> None:
    config = _config()
    validate_stage1_config(config)
    changed = copy.deepcopy(config)
    changed["experiment"]["joint_update"] = "sequential_role"
    with pytest.raises(Stage1TrainingError, match="joint_mean"):
        validate_stage1_config(changed)
    changed = copy.deepcopy(config)
    changed["overfit"]["timestep"] = 7
    with pytest.raises(Stage1TrainingError, match="fixed at 8"):
        validate_stage1_config(changed)


def test_optimizer_groups_are_disjoint_exhaustive_and_use_fixed_lrs(
    planner: BEVOnlyDiffusionPlanner,
) -> None:
    optimizer = build_stage1_optimizer(
        planner,
        planner_lr=1e-4,
        backbone_lr=1e-5,
        weight_decay=1e-4,
    )
    assert [group["name"] for group in optimizer.param_groups] == [
        "backbone",
        "planner",
    ]
    assert [group["lr"] for group in optimizer.param_groups] == [1e-5, 1e-4]
    parameter_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert set(parameter_ids) == {id(parameter) for parameter in planner.parameters()}


def test_real_packed_batch_forward_loss_backward_and_gradient_diagnostics(
    tmp_path: Path,
    planner: BEVOnlyDiffusionPlanner,
) -> None:
    planner.zero_grad(set_to_none=True)
    batch = _packed_batch(tmp_path)
    noise, timesteps = deterministic_overfit_noise(
        batch,
        batch_index=0,
        seed=17,
        timestep=8,
    )
    output = planner_forward_from_batch(
        planner,
        batch,
        diffusion_noise=noise,
        diffusion_timesteps=timesteps,
    )
    result = loss_from_batch(JointStage1Loss(), output, batch)
    role_diagnostics = role_gradient_diagnostics(result, planner)
    result.total.backward()
    gradient_norms = module_gradient_norms(planner)

    assert torch.isfinite(result.total)
    assert set(role_diagnostics) >= {
        "gradient_probe/mode_head/cosine_leader_middle",
        "gradient_probe/mode_head/cancellation_ratio",
        "gradient_probe/trajectory_head/cosine_leader_middle",
        "gradient_probe/trajectory_head/cancellation_ratio",
    }
    assert all(np.isfinite(value) for value in role_diagnostics.values())
    assert all(np.isfinite(value) for value in gradient_norms.values())
    assert all(value > 0.0 for value in gradient_norms.values())


def test_gradient_accumulation_executes_one_joint_optimizer_step(
    tmp_path: Path,
    planner: BEVOnlyDiffusionPlanner,
) -> None:
    batch = _packed_batch(tmp_path, samples=2)
    micro_batches = [
        {name: value.clone() for name, value in batch.items()},
        {name: value.clone() for name, value in batch.items()},
    ]
    optimizer = build_stage1_optimizer(
        planner,
        planner_lr=1e-4,
        backbone_lr=1e-5,
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    before = planner.mode_head.weight.detach().clone()
    metrics, optimizer_steps, diagnostics = train_one_epoch(
        planner=planner,
        loss_module=JointStage1Loss(),
        dataloader=micro_batches,
        optimizer=optimizer,
        scaler=scaler,
        device=torch.device("cpu"),
        mixed_precision=False,
        gradient_accumulation=2,
        optimizer_step=0,
        max_optimizer_steps=None,
        diagnostics_interval=100,
        overfit_seed=17,
        overfit_timestep=8,
    )
    assert optimizer_steps == 1
    assert np.isfinite(metrics["loss/total"])
    assert any("gradient/mode_head" in record for record in diagnostics)
    assert not torch.equal(before, planner.mode_head.weight)


def test_fixed_overfit_evaluation_restores_batch_norm_statistics(
    tmp_path: Path,
    planner: BEVOnlyDiffusionPlanner,
) -> None:
    batch = _packed_batch(tmp_path)
    batch_norm = next(
        module
        for module in planner.backbone.modules()
        if isinstance(module, torch.nn.BatchNorm2d)
    )
    running_mean = batch_norm.running_mean.detach().clone()
    running_var = batch_norm.running_var.detach().clone()
    batches = batch_norm.num_batches_tracked.detach().clone()
    planner.eval()
    metrics = evaluate_overfit_fixed(
        planner=planner,
        loss_module=JointStage1Loss(),
        dataloader=[batch],
        device=torch.device("cpu"),
        mixed_precision=False,
        seed=17,
        timestep=8,
    )
    assert np.isfinite(metrics["loss/total"])
    assert planner.training is False
    torch.testing.assert_close(batch_norm.running_mean, running_mean)
    torch.testing.assert_close(batch_norm.running_var, running_var)
    torch.testing.assert_close(batch_norm.num_batches_tracked, batches)


def test_checkpoint_round_trip_is_strict_and_marks_diagnostic(
    tmp_path: Path,
    planner: BEVOnlyDiffusionPlanner,
) -> None:
    optimizer = build_stage1_optimizer(
        planner,
        planner_lr=1e-4,
        backbone_lr=1e-5,
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    payload = checkpoint_payload(
        planner=planner,
        optimizer=optimizer,
        scaler=scaler,
        config=_config(),
        dataset_fingerprint="a" * 64,
        epoch=3,
        optimizer_step=11,
        metrics={"loss/total": 1.0},
        diagnostic_only=True,
    )
    path = save_stage1_checkpoint(tmp_path / "diagnostic.pt", payload)
    expected = planner.mode_head.weight.detach().clone()
    with torch.no_grad():
        planner.mode_head.weight.add_(1.0)
    loaded = load_stage1_checkpoint(path, planner)
    torch.testing.assert_close(planner.mode_head.weight, expected)
    assert loaded["diagnostic_only"] is True
    assert loaded["cross_split_overfit"] is True
    assert loaded["eligible_for_formal_training"] is False
    loaded["format"] = "legacy"
    torch.save(loaded, tmp_path / "legacy.pt")
    with pytest.raises(Stage1TrainingError, match="format mismatch"):
        load_stage1_checkpoint(tmp_path / "legacy.pt", planner)
