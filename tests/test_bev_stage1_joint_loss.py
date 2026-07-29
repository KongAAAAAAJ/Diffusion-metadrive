from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from models.bev_planner.stage1_joint_loss import (
    JointStage1Loss,
    Stage1LossConfig,
    Stage1LossError,
    local_trajectory_to_world_xy,
    trajectory_speed_acceleration,
    wrapped_heading_error,
)


def _loss_batch(
    batch_size: int = 2,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    expert = torch.zeros((batch_size, 3, 8, 3), dtype=torch.float32)
    times = torch.arange(1, 9, dtype=torch.float32) * 0.5
    for role in range(3):
        expert[:, role, :, 0] = times * (6.0 + role)
        expert[:, role, :, 1] = 0.2 * role
        expert[:, role, :, 2] = 0.01 * role
    gt_mode = torch.tensor([[0, 4, 7]], dtype=torch.int64).repeat(batch_size, 1)
    candidates = expert.unsqueeze(2).repeat(1, 1, 10, 1, 1)
    mode_logits = torch.full((batch_size, 3, 10), -10.0, dtype=torch.float32)
    mode_logits.scatter_(-1, gt_mode.unsqueeze(-1), 10.0)
    selected_index = gt_mode[..., None, None, None].expand(-1, -1, 1, 8, 3)
    output = {
        "trajectory_candidates": candidates,
        "mode_logits": mode_logits,
        "selected_mode": gt_mode.clone(),
        "selected_trajectory": candidates.gather(2, selected_index).squeeze(2),
    }
    batch = {
        "expert_trajectory": expert,
        "gt_mode": gt_mode,
        "mode_valid_mask": torch.ones((batch_size, 3, 10), dtype=torch.bool),
        "ego_state": torch.zeros((batch_size, 3, 8), dtype=torch.float32),
        "ego_pose_global": torch.tensor(
            [[[20.0, 3.0, 0.0], [10.0, 3.0, 0.0], [0.0, 3.0, 0.0]]],
            dtype=torch.float32,
        ).repeat(batch_size, 1, 1),
    }
    batch["ego_state"][..., 0] = torch.tensor([6.0, 7.0, 8.0])
    return output, batch


def _compute(
    loss: JointStage1Loss,
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
):
    return loss(
        output,
        expert_trajectory=batch["expert_trajectory"],
        gt_mode=batch["gt_mode"],
        mode_valid_mask=batch["mode_valid_mask"],
        ego_state=batch["ego_state"],
        ego_pose_global=batch["ego_pose_global"],
    )


def test_perfect_trajectory_has_only_near_zero_mode_ce() -> None:
    output, batch = _loss_batch()
    result = _compute(JointStage1Loss(), output, batch)
    assert result.xy == 0
    assert result.heading == 0
    assert result.velocity == 0
    assert result.acceleration == 0
    assert result.pair_joint == 0
    assert result.mode_ce < 1e-6
    assert result.total < 1e-6
    assert result.gt_mode_ade == 0
    assert result.selected_ade == 0
    assert result.mode_accuracy == 1
    assert result.role_total.shape == (2, 3)
    assert set(result.scalar_metrics()) >= {
        "loss/total",
        "loss/pair_joint",
        "metric/gt_mode_ade",
        "role/leader_total",
        "role/middle_total",
        "role/rear_total",
    }


def test_only_gt_mode_is_used_for_trajectory_regression() -> None:
    output, batch = _loss_batch(batch_size=1)
    baseline = _compute(JointStage1Loss(), output, batch)
    output["trajectory_candidates"][:, :, 9, :, :2] += 1000.0
    changed = _compute(JointStage1Loss(), output, batch)
    torch.testing.assert_close(changed.xy, baseline.xy)
    torch.testing.assert_close(changed.heading, baseline.heading)
    torch.testing.assert_close(changed.pair_joint, baseline.pair_joint)


def test_loss_weights_match_frozen_formula() -> None:
    output, batch = _loss_batch(batch_size=1)
    gt = batch["gt_mode"]
    for role in range(3):
        output["trajectory_candidates"][0, role, gt[0, role], :, 0] += 2.0
    selected_index = gt[..., None, None, None].expand(-1, -1, 1, 8, 3)
    output["selected_trajectory"] = (
        output["trajectory_candidates"].gather(2, selected_index).squeeze(2)
    )
    config = Stage1LossConfig()
    result = _compute(JointStage1Loss(config), output, batch)
    expected_local = (
        config.xy_weight * result.xy
        + config.heading_weight * result.heading
        + config.mode_weight * result.mode_ce
        + config.motion_weight * (result.velocity + result.acceleration)
    )
    torch.testing.assert_close(result.local_joint, expected_local)
    torch.testing.assert_close(
        result.total,
        result.local_joint + config.pair_weight * result.pair_joint,
    )


def test_wrapped_heading_treats_opposite_pi_boundaries_as_close() -> None:
    predicted = torch.tensor([math.pi - 0.01], requires_grad=True)
    target = torch.tensor([-math.pi + 0.01])
    difference = wrapped_heading_error(predicted, target)
    torch.testing.assert_close(difference, torch.tensor([-0.02]), atol=1e-5, rtol=0)
    difference.square().sum().backward()
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()


def test_speed_and_acceleration_include_origin_and_current_speed() -> None:
    trajectory = torch.tensor(
        [
            [
                [
                    [1.0, 0.0],
                    [3.0, 0.0],
                    [6.0, 0.0],
                    [10.0, 0.0],
                    [15.0, 0.0],
                    [21.0, 0.0],
                    [28.0, 0.0],
                    [36.0, 0.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    speed, acceleration = trajectory_speed_acceleration(
        trajectory,
        torch.tensor([[2.0]]),
        dt_s=0.5,
    )
    torch.testing.assert_close(
        speed, torch.tensor([[[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]]])
    )
    torch.testing.assert_close(
        acceleration,
        torch.tensor([[[0.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]]]),
    )


def test_local_to_world_rotation_is_exact() -> None:
    local = torch.tensor(
        [[[[1.0, 0.0], [0.0, 2.0]]]],
        dtype=torch.float32,
    )
    pose = torch.tensor([[[10.0, 20.0, math.pi / 2]]], dtype=torch.float32)
    world = local_trajectory_to_world_xy(local, pose)
    torch.testing.assert_close(
        world,
        torch.tensor([[[[10.0, 21.0], [8.0, 20.0]]]]),
        atol=1e-5,
        rtol=0,
    )


def test_pair_loss_couples_role_gradients() -> None:
    output, batch = _loss_batch(batch_size=1)
    candidates = output["trajectory_candidates"].clone().requires_grad_(True)
    gt = batch["gt_mode"]
    with torch.no_grad():
        for role in range(3):
            candidates[0, role, gt[0, role], :, 0] += float(role)
    output["trajectory_candidates"] = candidates
    selected_index = gt[..., None, None, None].expand(-1, -1, 1, 8, 3)
    output["selected_trajectory"] = candidates.gather(2, selected_index).squeeze(2)
    result = _compute(JointStage1Loss(), output, batch)
    result.pair_joint.backward()
    assert candidates.grad is not None
    gradients = [
        candidates.grad[0, role, gt[0, role], :, 0].mean() for role in range(3)
    ]
    assert gradients[0] < 0
    assert gradients[2] > 0
    torch.testing.assert_close(sum(gradients), torch.tensor(0.0), atol=1e-6, rtol=0)


def test_masked_cross_entropy_is_finite_in_half_precision() -> None:
    output, batch = _loss_batch(batch_size=1)
    batch["mode_valid_mask"][:, :, 8] = False
    output["mode_logits"] = output["mode_logits"].half()
    output["mode_logits"][~batch["mode_valid_mask"]] = float("-inf")
    output["mode_logits"].requires_grad_(True)
    result = _compute(JointStage1Loss(), output, batch)
    assert result.mode_ce.dtype == torch.float32
    assert torch.isfinite(result.total)
    result.total.backward()
    assert output["mode_logits"].grad is not None
    assert torch.isfinite(output["mode_logits"].grad).all()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda output, batch: batch["gt_mode"].fill_(10),
            "outside",
        ),
        (
            lambda output, batch: batch["mode_valid_mask"].scatter_(
                -1, batch["gt_mode"].unsqueeze(-1), False
            ),
            "gt_mode must be enabled",
        ),
        (
            lambda output, batch: batch["expert_trajectory"].fill_(float("nan")),
            "non-finite",
        ),
        (
            lambda output, batch: output["mode_logits"].__setitem__(
                (slice(None), slice(None), 9), float("-inf")
            ),
            "valid mode logits",
        ),
    ],
)
def test_invalid_contracts_are_rejected(mutation, match: str) -> None:
    output, batch = _loss_batch(batch_size=1)
    mutation(output, batch)
    with pytest.raises(Stage1LossError, match=match):
        _compute(JointStage1Loss(), output, batch)


def test_invalid_mode_logit_must_be_negative_infinity() -> None:
    output, batch = _loss_batch(batch_size=1)
    batch["mode_valid_mask"][:, :, 9] = False
    with pytest.raises(Stage1LossError, match="negative infinity"):
        _compute(JointStage1Loss(), output, batch)


def test_config_rejects_invalid_weight_or_interval() -> None:
    with pytest.raises(Stage1LossError, match="xy_weight"):
        Stage1LossConfig(xy_weight=-1.0)
    with pytest.raises(Stage1LossError, match="dt_s"):
        Stage1LossConfig(dt_s=0.0)
