from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from models.bev_planner import (
    BEVOnlyDiffusionPlanner,
    BEVOnlyDiffusionPlannerConfig,
    JointGRPOConfig,
    JointGRPOError,
    JointGRPORollout,
    JointGRPORolloutB,
    JointGRPOTrainerA,
    JointGRPOTrainerB,
    ModeResidualTrajectoryHead,
    fixed_scale_safe_advantages,
    joint_grpo_optimizer_contract,
)
from models.bev_planner.bev_only_diffusion_planner import MAX_BACKGROUND_ACTORS
from models.bev_planner.joint_grpo import _hierarchical_active_mean


def _planner(condition: str = "none") -> BEVOnlyDiffusionPlanner:
    planner = BEVOnlyDiffusionPlanner(
        BEVOnlyDiffusionPlannerConfig(
            d_model=32,
            num_heads=4,
            ffn_dim=64,
            decoder_layers=1,
            predecessor_condition=condition,
        )
    )
    with torch.no_grad():
        torch.nn.init.normal_(
            planner.diffusion_decoder.trajectory_head[-1].weight,
            mean=0.0,
            std=0.01,
        )
        torch.nn.init.normal_(
            planner.diffusion_decoder.trajectory_head[-1].bias,
            mean=0.0,
            std=0.01,
        )
    return planner


def _inputs(batch_size: int = 1) -> dict[str, torch.Tensor]:
    coarse = torch.zeros((batch_size, 3, 10, 8, 3), dtype=torch.float32)
    coarse[..., 0] = torch.arange(1, 9, dtype=torch.float32) * 0.5
    return {
        "bev": torch.zeros((batch_size, 3, 8, 256, 256), dtype=torch.uint8),
        "ego_state": torch.zeros((batch_size, 3, 8), dtype=torch.float32),
        "formation_relation_state": torch.zeros(
            (batch_size, 3, 12), dtype=torch.float32
        ),
        "relation_valid_mask": torch.ones((batch_size, 3, 2), dtype=torch.bool),
        "agent_role": torch.arange(3, dtype=torch.int64)
        .unsqueeze(0)
        .repeat(batch_size, 1),
        "coarse_trajectories": coarse,
        "mode_valid_mask": torch.ones((batch_size, 3, 10), dtype=torch.bool),
        "background_actor_state": torch.zeros(
            (batch_size, 3, MAX_BACKGROUND_ACTORS, 8), dtype=torch.float32
        ),
        "background_actor_valid_mask": torch.zeros(
            (batch_size, 3, MAX_BACKGROUND_ACTORS), dtype=torch.bool
        ),
        "scenario_code": torch.ones((batch_size,), dtype=torch.int64),
        "rule_formation_state": torch.zeros((batch_size,), dtype=torch.int64),
        "rule_action_condition": torch.zeros(
            (batch_size, 3), dtype=torch.int64
        ),
    }


def _bind_single_mode_signal(
    rollout: JointGRPORollout,
    *,
    role: int = 1,
    mode: int = 4,
) -> JointGRPORollout:
    shape = rollout.candidate_trajectories.shape[:4]
    current = torch.zeros(shape, dtype=torch.float32)
    frozen = torch.zeros_like(current)
    current[0, role, mode] = torch.tensor([-1.0, 1.0])
    frozen[0, role, mode] = torch.tensor([-2.0, 0.0])
    return rollout.with_reward_signals(
        current_rewards=current,
        frozen_rewards=frozen,
        collision_mask=torch.zeros_like(current, dtype=torch.bool),
        out_of_drivable_mask=torch.zeros_like(current, dtype=torch.bool),
        valid_executable_mode_mask=rollout.mode_valid_mask.clone(),
    )


def _module_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in module.state_dict().items()
    }


def _assert_nested_equal(first: object, second: object) -> None:
    if isinstance(first, torch.Tensor):
        assert isinstance(second, torch.Tensor)
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert isinstance(second, dict) and first.keys() == second.keys()
        for key in first:
            _assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert isinstance(second, type(first)) and len(first) == len(second)
        for left, right in zip(first, second):
            _assert_nested_equal(left, right)
    else:
        assert first == second


def test_fixed_scale_advantage_and_optimizer_contract() -> None:
    contract = joint_grpo_optimizer_contract()
    assert contract["version"] == "stage2_joint_grpo_optimizer_v8"
    assert contract["ppo_ratio_or_clipping"] is False
    assert contract["weight_decay"] == 0.0

    current = torch.tensor(
        [[[[0.0, 2.0, 3.0, 4.0]] * 10] * 3], dtype=torch.float32
    )
    frozen = torch.tensor(
        [[[[0.0, 2.0, 4.0, 0.0]] * 10] * 3], dtype=torch.float32
    )
    collision = torch.zeros_like(current, dtype=torch.bool)
    out = torch.zeros_like(current, dtype=torch.bool)
    collision[0, 0, 0, 0] = True
    out[0, 0, 0, 1] = True
    valid = torch.ones((1, 3, 10), dtype=torch.bool)
    valid[0, 2, 9] = False

    centered, advantage, signal = fixed_scale_safe_advantages(
        current,
        frozen,
        collision,
        out,
        valid,
        trajectories_per_mode=4,
    )

    torch.testing.assert_close(
        centered[0, 0, 0], torch.tensor([-2.25, -0.25, 0.75, 1.75])
    )
    torch.testing.assert_close(
        advantage[0, 0, 0], torch.tensor([-1.0, -1.0, 0.0, 1.75])
    )
    assert signal[0, 0, 0]
    assert torch.count_nonzero(advantage[0, 2, 9]) == 0


def test_fixed_trailing_reduction_equalizes_roles() -> None:
    values = torch.zeros((1, 3, 10, 2, 3), dtype=torch.float32)
    mask = torch.zeros((1, 3, 10), dtype=torch.bool)
    values[:, 0, 0] = 1.0
    mask[:, 0, 0] = True
    values[:, 1, :5] = 3.0
    mask[:, 1, :5] = True
    torch.testing.assert_close(
        _hierarchical_active_mean(values, mask),
        torch.tensor((1.0 + 3.0) / 3.0),
    )


def test_mode_residual_changes_only_its_own_mode() -> None:
    base = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 24))
    head = ModeResidualTrajectoryHead(base)
    features = torch.randn(
        (2, 3, 10, 128), generator=torch.Generator().manual_seed(7)
    )
    before = head(features).detach()
    with torch.no_grad():
        head.mode_residual_weight[6].fill_(0.01)
        head.mode_residual_bias[6].fill_(0.02)
    after = head(features).detach()
    assert head.mode_residual_weight.shape == (10, 24, 128)
    assert head.mode_residual_bias.shape == (10, 24)
    torch.testing.assert_close(before[..., :6, :], after[..., :6, :], rtol=0, atol=0)
    torch.testing.assert_close(before[..., 7:, :], after[..., 7:, :], rtol=0, atol=0)
    assert not torch.equal(before[..., 6, :], after[..., 6, :])


@pytest.fixture(scope="module")
def rollout_a() -> tuple[JointGRPOTrainerA, JointGRPORollout]:
    trainer = JointGRPOTrainerA(
        _planner(), JointGRPOConfig(trajectories_per_mode=2)
    )
    rollout = trainer.sample_groups(
        _inputs(),
        generator=torch.Generator().manual_seed(17),
        transition_generator=torch.Generator().manual_seed(18),
        noise_bundle_identity=(17, 0, 0),
    )
    return trainer, rollout


def test_paired_rollout_shapes_and_zero_residual_identity(rollout_a) -> None:
    _, rollout = rollout_a
    assert rollout.chains_normalized.shape == (1, 2, 5, 3, 10, 8, 2)
    assert rollout.candidate_trajectories.shape == (1, 3, 10, 2, 8, 3)
    assert rollout.frozen_candidate_trajectories.shape == (1, 3, 10, 2, 8, 3)
    assert rollout.noise_bundle_identity == (17, 0, 0)
    torch.testing.assert_close(
        rollout.candidate_trajectories,
        rollout.frozen_candidate_trajectories,
        rtol=0,
        atol=0,
    )
    assert not hasattr(rollout, "old_trajectory_log_prob")


def test_loss_shape_and_pg_gradient_are_mode_isolated() -> None:
    trainer = JointGRPOTrainerA(
        _planner(),
        JointGRPOConfig(
            trajectories_per_mode=2,
            bc_weight=0.0,
            reference_kl_weight=0.0,
        ),
    )
    rollout = _bind_single_mode_signal(
        trainer.sample_groups(
            _inputs(), generator=torch.Generator().manual_seed(23)
        )
    )
    loss = trainer.compute_loss(rollout)
    assert loss.new_trajectory_log_prob.shape == (1, 3, 10, 2, 3)
    assert loss.advantages.shape == (1, 3, 10, 2)
    assert loss.trajectory_pg_by_mode.shape == (10,)
    loss.total.backward()
    head = trainer.planner.diffusion_decoder.trajectory_head
    assert isinstance(head, ModeResidualTrajectoryHead)
    nonzero_modes = {
        mode
        for mode in range(10)
        if torch.count_nonzero(head.mode_residual_weight.grad[mode])
        or torch.count_nonzero(head.mode_residual_bias.grad[mode])
    }
    assert nonzero_modes == {4}


def test_full_valid_anchor_reaches_inactive_mode_but_not_invalid_mode() -> None:
    inputs = _inputs()
    inputs["mode_valid_mask"][:, :, 2] = False
    trainer = JointGRPOTrainerA(
        _planner(), JointGRPOConfig(trajectories_per_mode=2)
    )
    head = trainer.planner.diffusion_decoder.trajectory_head
    assert isinstance(head, ModeResidualTrajectoryHead)
    with torch.no_grad():
        head.mode_residual_bias[2].fill_(0.03)
        head.mode_residual_bias[3].fill_(0.03)
    rollout = _bind_single_mode_signal(
        trainer.sample_groups(
            inputs, generator=torch.Generator().manual_seed(29)
        )
    )
    loss = trainer.compute_loss(rollout)
    loss.behavior_cloning.backward()
    assert torch.count_nonzero(head.mode_residual_bias.grad[2]) == 0
    assert torch.count_nonzero(head.mode_residual_bias.grad[3]) > 0


def test_one_update_changes_only_residual_and_forbids_reuse() -> None:
    trainer = JointGRPOTrainerA(
        _planner(),
        JointGRPOConfig(
            trajectories_per_mode=2,
            post_update_reference_kl_max=1e6,
            max_adapter_relative_drift=1e6,
        ),
    )
    rollout = _bind_single_mode_signal(
        trainer.sample_groups(
            _inputs(), generator=torch.Generator().manual_seed(31)
        )
    )
    head = trainer.planner.diffusion_decoder.trajectory_head
    assert isinstance(head, ModeResidualTrajectoryHead)
    base_before = _module_state(head.base)
    mode_before = _module_state(trainer.planner.mode_head)
    result = trainer.update(rollout)
    assert result.optimizer_step == 1
    assert not result.stability_guard_rejected
    assert result.gradient_norms["mode_4"] > 0.0
    assert all(
        result.gradient_norms[f"mode_{mode}"] == 0.0
        for mode in range(10)
        if mode != 4
    )
    base_after = _module_state(head.base)
    assert base_after.keys() == base_before.keys()
    for name, value in base_before.items():
        assert torch.equal(base_after[name], value)
    for name, value in mode_before.items():
        assert torch.equal(trainer.planner.mode_head.state_dict()[name], value)
    with pytest.raises(JointGRPOError, match="exactly once"):
        trainer.update(rollout)


def test_zero_signal_skips_optimizer_step() -> None:
    trainer = JointGRPOTrainerA(
        _planner(), JointGRPOConfig(trajectories_per_mode=2)
    )
    rollout = trainer.sample_groups(
        _inputs(), generator=torch.Generator().manual_seed(37)
    )
    zeros = torch.zeros((1, 3, 10, 2), dtype=torch.float32)
    rollout = rollout.with_reward_signals(
        current_rewards=zeros,
        frozen_rewards=zeros,
        collision_mask=torch.zeros_like(zeros, dtype=torch.bool),
        out_of_drivable_mask=torch.zeros_like(zeros, dtype=torch.bool),
        valid_executable_mode_mask=rollout.mode_valid_mask,
    )
    result = trainer.update(rollout)
    assert result.zero_signal
    assert result.optimizer_step == 0
    assert not result.stability_guard_rejected


def test_guard_restores_residual_parameters_and_adam_state() -> None:
    trainer = JointGRPOTrainerA(
        _planner(),
        JointGRPOConfig(
            trajectories_per_mode=2,
            post_update_reference_kl_max=1e-30,
            max_adapter_relative_drift=1e6,
        ),
    )
    head = trainer.planner.diffusion_decoder.trajectory_head
    assert isinstance(head, ModeResidualTrajectoryHead)
    for parameter in head.residual_parameters():
        parameter.grad = torch.zeros_like(parameter)
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    rollout = _bind_single_mode_signal(
        trainer.sample_groups(
            _inputs(), generator=torch.Generator().manual_seed(41)
        )
    )
    parameter_before = tuple(
        parameter.detach().clone() for parameter in head.residual_parameters()
    )
    optimizer_before = copy.deepcopy(trainer.optimizer.state_dict())
    result = trainer.update(rollout)
    assert result.stability_guard_rejected
    assert result.optimizer_step == 0
    for parameter, expected in zip(head.residual_parameters(), parameter_before):
        assert torch.equal(parameter, expected)
    _assert_nested_equal(trainer.optimizer.state_dict(), optimizer_before)


def test_variant_b_retains_paired_all_mode_contract() -> None:
    trainer = JointGRPOTrainerB(
        _planner("predicted_detached"),
        JointGRPOConfig(trajectories_per_mode=2),
    )
    rollout = trainer.sample_groups(
        _inputs(), generator=torch.Generator().manual_seed(43)
    )
    assert isinstance(rollout, JointGRPORolloutB)
    assert rollout.candidate_trajectories.shape == (1, 3, 10, 2, 8, 3)
    assert rollout.frozen_candidate_trajectories.shape == (1, 3, 10, 2, 8, 3)
    assert rollout.predecessor_action_history_normalized is not None
    assert rollout.predecessor_action_history_normalized.shape == (1, 2, 4, 2, 8, 3)
