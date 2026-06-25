import torch
import torch.nn as nn

from models.diffusion.transfuser_config import build_transfuser_config
from models.diffusion.transfuser_model_v2 import (
    DiffMotionPlanningRefinementModule,
    V2TransfuserModel,
    compute_preference_bias,
)


def test_preference_bias_prefers_anchor_endpoint_closer_to_preference_point() -> None:
    coarse = torch.tensor(
        [[
            [[0.0, 0.0], [2.0, 0.0]],
            [[0.0, 0.0], [8.0, 0.0]],
        ]],
        dtype=torch.float32,
    )
    preference = torch.tensor([[7.0, 0.0]], dtype=torch.float32)

    bias = compute_preference_bias(
        preference_point=preference,
        coarse_trajectories=coarse,
        temperature=1.0,
    )

    assert bias.shape == (1, 2)
    assert bias[0, 1] > bias[0, 0]


def test_preference_bias_masks_invalid_modes() -> None:
    coarse = torch.tensor(
        [[
            [[0.0, 0.0], [2.0, 0.0]],
            [[0.0, 0.0], [8.0, 0.0]],
        ]],
        dtype=torch.float32,
    )
    preference = torch.tensor([[8.0, 0.0]], dtype=torch.float32)
    mask = torch.tensor([[True, False]])

    bias = compute_preference_bias(
        preference_point=preference,
        coarse_trajectories=coarse,
        temperature=1.0,
        mode_valid_mask=mask,
    )

    assert bias[0, 1] <= -1e4


def test_preference_bias_normalizes_valid_mode_distances() -> None:
    coarse = torch.tensor(
        [[
            [[0.0, 0.0], [2.0, 0.0]],
            [[0.0, 0.0], [8.0, 0.0]],
            [[0.0, 0.0], [14.0, 0.0]],
        ]],
        dtype=torch.float32,
    )
    preference = torch.tensor([[8.0, 0.0]], dtype=torch.float32)

    bias = compute_preference_bias(
        preference_point=preference,
        coarse_trajectories=coarse,
        temperature=2.0,
    )

    expected = torch.tensor([[-0.5, 0.0, -0.5]], dtype=torch.float32)
    torch.testing.assert_close(bias, expected)


def test_preference_bias_changes_class_logits_without_changing_regression() -> None:
    torch.manual_seed(3)
    module = DiffMotionPlanningRefinementModule(
        embed_dims=8,
        ego_fut_ts=2,
        ego_fut_mode=2,
        target_point_dim=0,
        trajectory_reg_decoder_type="mlp",
    )
    traj_feature = torch.randn(1, 2, 8)

    reg_without_bias, cls_without_bias = module(traj_feature)
    preference_bias = torch.tensor([[0.0, 2.5]], dtype=torch.float32)
    reg_with_bias, cls_with_bias = module(traj_feature, preference_bias=preference_bias)

    torch.testing.assert_close(reg_with_bias, reg_without_bias)
    torch.testing.assert_close(cls_with_bias - cls_without_bias, preference_bias)


def test_refinement_module_decouples_classification_and_regression_guidance() -> None:
    torch.manual_seed(7)
    module = DiffMotionPlanningRefinementModule(
        embed_dims=8,
        ego_fut_ts=2,
        ego_fut_mode=3,
        target_point_dim=4,
        trajectory_reg_decoder_type="mlp",
    )
    traj_feature = torch.randn(1, 3, 8)
    reg_target_embed_a = torch.randn(1, 3, 4)
    reg_target_embed_b = reg_target_embed_a.clone()
    reg_target_embed_b[:, 1, :] += 2.0
    cls_preference_embed = torch.randn(1, 4)

    reg_a, cls_a = module(
        traj_feature,
        reg_target_embed=reg_target_embed_a,
        cls_preference_embed=cls_preference_embed,
    )
    reg_b, cls_b = module(
        traj_feature,
        reg_target_embed=reg_target_embed_b,
        cls_preference_embed=cls_preference_embed,
    )

    assert not torch.allclose(reg_a, reg_b)
    torch.testing.assert_close(cls_a, cls_b)


def test_v2_model_cls_preference_guidance_prefers_preference_point() -> None:
    config = build_transfuser_config("small", target_guidance_type="multi_point")
    model = V2TransfuserModel(config)
    model._target_point_mlp = nn.Identity()
    features = {
        "status_feature": torch.zeros((1, config.status_feature_dim), dtype=torch.float32),
        "target_point": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "preference_point": torch.tensor([[7.0, 8.0]], dtype=torch.float32),
    }

    cls_preference_embed = model._encode_cls_preference_guidance(features)

    torch.testing.assert_close(cls_preference_embed, features["preference_point"])


def test_v2_model_cls_preference_guidance_falls_back_to_target_point() -> None:
    config = build_transfuser_config("small", target_guidance_type="multi_point")
    model = V2TransfuserModel(config)
    model._target_point_mlp = nn.Identity()
    features = {
        "status_feature": torch.zeros((1, config.status_feature_dim), dtype=torch.float32),
        "target_point": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    }

    cls_preference_embed = model._encode_cls_preference_guidance(features)

    torch.testing.assert_close(cls_preference_embed, features["target_point"])
