import torch

from metadrive.policy.diffusion_policy.transfuser_model_v2 import (
    DiffMotionPlanningRefinementModule,
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

    assert bias[0, 1] < -1e8


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
