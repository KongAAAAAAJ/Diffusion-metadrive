from __future__ import annotations

import pytest
import torch

from models.diffusion.transfuser_config import build_transfuser_config
from models.diffusion.transfuser_model_v2 import DiffMotionPlanningRefinementModule, V2TransfuserModel


def _build_inputs(
    batch_size: int = 2,
    ego_fut_mode: int = 4,
    ego_fut_ts: int = 8,
    embed_dims: int = 16,
    target_point_dim: int = 6,
):
    traj_feature = torch.randn((batch_size, ego_fut_mode, embed_dims), dtype=torch.float32)
    target_point_embed = torch.randn((batch_size, target_point_dim), dtype=torch.float32)
    noisy_traj_points = torch.randn((batch_size, ego_fut_mode, ego_fut_ts, 2), dtype=torch.float32)
    return traj_feature, target_point_embed, noisy_traj_points


def test_refinement_module_mlp_mode_keeps_output_shape():
    traj_feature, target_point_embed, _ = _build_inputs()
    module = DiffMotionPlanningRefinementModule(
        embed_dims=16,
        ego_fut_ts=8,
        ego_fut_mode=4,
        target_point_dim=6,
        trajectory_reg_decoder_type="mlp",
    )

    plan_reg, plan_cls = module(traj_feature, target_point_embed=target_point_embed)

    assert tuple(plan_reg.shape) == (2, 4, 8, 3)
    assert tuple(plan_cls.shape) == (2, 4)


def test_refinement_module_gru_mode_keeps_output_shape():
    traj_feature, target_point_embed, noisy_traj_points = _build_inputs()
    module = DiffMotionPlanningRefinementModule(
        embed_dims=16,
        ego_fut_ts=8,
        ego_fut_mode=4,
        target_point_dim=6,
        trajectory_reg_decoder_type="gru",
        trajectory_gru_hidden_dim=12,
    )

    plan_reg, plan_cls = module(
        traj_feature,
        target_point_embed=target_point_embed,
        noisy_traj_points=noisy_traj_points,
    )

    assert tuple(plan_reg.shape) == (2, 4, 8, 3)
    assert tuple(plan_cls.shape) == (2, 4)


def test_refinement_module_gru_mode_requires_noisy_traj_points():
    traj_feature, target_point_embed, _ = _build_inputs()
    module = DiffMotionPlanningRefinementModule(
        embed_dims=16,
        ego_fut_ts=8,
        ego_fut_mode=4,
        target_point_dim=6,
        trajectory_reg_decoder_type="gru",
    )

    with pytest.raises(ValueError, match="noisy_traj_points"):
        module(traj_feature, target_point_embed=target_point_embed, noisy_traj_points=None)


def test_refinement_module_gru_changes_when_anchor_changes():
    traj_feature, target_point_embed, noisy_traj_points = _build_inputs(batch_size=1, ego_fut_mode=2)
    module = DiffMotionPlanningRefinementModule(
        embed_dims=16,
        ego_fut_ts=8,
        ego_fut_mode=2,
        target_point_dim=6,
        trajectory_reg_decoder_type="gru",
        trajectory_gru_hidden_dim=10,
    )

    out_a, _ = module(
        traj_feature,
        target_point_embed=target_point_embed,
        noisy_traj_points=noisy_traj_points,
    )
    out_b, _ = module(
        traj_feature,
        target_point_embed=target_point_embed,
        noisy_traj_points=noisy_traj_points + 0.5,
    )

    assert not torch.allclose(out_a, out_b)


def test_build_config_supports_gru_decoder_overrides():
    config = build_transfuser_config(
        "small",
        trajectory_reg_decoder_type="gru",
        trajectory_gru_hidden_dim=96,
    )

    assert config.trajectory_reg_decoder_type == "gru"
    assert config.trajectory_gru_hidden_dim == 96


def test_v2_transfuser_model_supports_gru_forward_and_infer():
    config = build_transfuser_config(
        "small",
        trajectory_reg_decoder_type="gru",
        trajectory_gru_hidden_dim=64,
    )
    model = V2TransfuserModel(config)
    batch_size = 1
    num_poses = config.trajectory_sampling.num_poses

    features = {
        "camera_feature": torch.zeros((batch_size, 3, config.camera_height, config.camera_width), dtype=torch.float32),
        "lidar_feature": torch.zeros((batch_size, 1, config.lidar_resolution_height, config.lidar_resolution_width), dtype=torch.float32),
        "status_feature": torch.zeros((batch_size, config.status_feature_dim), dtype=torch.float32),
        "target_point": torch.zeros((batch_size, 2), dtype=torch.float32),
        "coarse_trajectories": torch.zeros((batch_size, config.ego_fut_mode, num_poses, 2), dtype=torch.float32),
        "mode_valid_mask": torch.ones((batch_size, config.ego_fut_mode), dtype=torch.bool),
    }
    targets = {
        "trajectory": torch.zeros((batch_size, num_poses, 3), dtype=torch.float32),
        "agent_states": torch.zeros((batch_size, config.num_bounding_boxes, 5), dtype=torch.float32),
        "agent_labels": torch.zeros((batch_size, config.num_bounding_boxes), dtype=torch.bool),
        "bev_semantic_map": torch.zeros((batch_size, config.bev_pixel_height, config.bev_pixel_width), dtype=torch.int64),
    }

    model.train()
    with torch.no_grad():
        train_out = model(features, targets=targets)
    model.eval()
    with torch.no_grad():
        infer_out = model.infer_multimodal(features)

    assert tuple(train_out["trajectory"].shape) == (batch_size, num_poses, 3)
    assert "trajectory_loss" in train_out
    assert tuple(infer_out["trajectory"].shape) == (batch_size, num_poses, 3)
    assert tuple(infer_out["trajectory_candidates"].shape) == (batch_size, config.ego_fut_mode, num_poses, 3)
