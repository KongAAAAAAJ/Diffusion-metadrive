from __future__ import annotations

import copy

import pytest
import torch

from models.bev_planner import (
    BEVOnlyDiffusionPlanner,
    BEVOnlyDiffusionPlannerConfig,
    JointGRPOConfig,
    JointGRPOError,
    JointGRPOPolicyUpdateConfig,
    JointGRPORolloutB,
    JointGRPOTrainerA,
    JointGRPOTrainerB,
    joint_grpo_optimizer_contract,
    normalize_same_mode_advantages,
    same_mode_active_mask,
)
from models.bev_planner.bev_only_diffusion_planner import MAX_BACKGROUND_ACTORS
from models.bev_planner.joint_grpo import _hierarchical_active_mean


def _planner(condition: str = "none") -> BEVOnlyDiffusionPlanner:
    return BEVOnlyDiffusionPlanner(
        BEVOnlyDiffusionPlannerConfig(
            d_model=32,
            num_heads=4,
            ffn_dim=64,
            decoder_layers=1,
            predecessor_condition=condition,
        )
    )


def _inputs(batch_size: int = 1) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(91)
    times = torch.arange(1, 9, dtype=torch.float32) * 0.5
    coarse = torch.zeros((batch_size, 3, 10, 8, 3), dtype=torch.float32)
    for role in range(3):
        for mode in range(10):
            coarse[:, role, mode, :, 0] = times * (6.0 + 0.2 * role)
            coarse[:, role, mode, :, 0] += 0.1 * mode
            coarse[:, role, mode, :, 1] = 0.05 * (mode - 4)
            coarse[:, role, mode, :, 2] = 0.01 * (mode - 4)
    return {
        "bev": torch.randint(
            0,
            256,
            (batch_size, 3, 8, 256, 256),
            dtype=torch.uint8,
            generator=generator,
        ),
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
        "rule_action_condition": torch.zeros((batch_size, 3), dtype=torch.int64),
    }


def _state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _same_state(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]
) -> bool:
    return first.keys() == second.keys() and all(
        torch.equal(first[name], second[name]) for name in first
    )


def test_same_mode_config_contract_advantage_and_strict_gate() -> None:
    config = JointGRPOConfig()
    assert config.trajectories_per_mode == 48
    policy = JointGRPOPolicyUpdateConfig()
    assert (policy.update_epochs, policy.clip_epsilon_low, policy.clip_epsilon_high) == (
        10,
        0.1,
        0.2,
    )
    contract = joint_grpo_optimizer_contract(policy)
    assert contract["version"] == "stage2_joint_grpo_optimizer_v5"
    assert contract["trainable_modules"] == ["diffusion_decoder.trajectory_head"]
    assert "mode_ratio_factorization" not in contract

    rewards = torch.zeros((1, 3, 10, 4), dtype=torch.float32)
    rewards[0, 0, 2] = torch.tensor([-3.0, -1.0, 1.0, 3.0])
    rewards[0, 1, 7] = 5.0
    active = torch.zeros((1, 3, 10), dtype=torch.bool)
    active[0, 0, 2] = True
    active[0, 1, 7] = True
    advantages = normalize_same_mode_advantages(
        rewards,
        active,
        trajectories_per_mode=4,
    )
    torch.testing.assert_close(advantages[0, 0, 2].mean(), torch.zeros(()))
    torch.testing.assert_close(
        advantages[0, 0, 2].square().mean(),
        torch.tensor(1.0),
        rtol=1e-5,
        atol=1e-5,
    )
    assert torch.count_nonzero(advantages[0, 1, 7]) == 0
    assert torch.count_nonzero(advantages[~active]) == 0

    changed = rewards.clone()
    changed[0, 2, 9] = torch.tensor([-100.0, 0.0, 100.0, 200.0])
    changed_advantages = normalize_same_mode_advantages(changed, active)
    torch.testing.assert_close(
        advantages[0, 0, 2], changed_advantages[0, 0, 2]
    )

    pretrain = torch.zeros((1, 3, 10), dtype=torch.float32)
    valid = torch.ones_like(pretrain, dtype=torch.bool)
    below = torch.full_like(rewards, -0.01)
    assert not bool(same_mode_active_mask(below, pretrain, valid_mode_mask=valid).any())
    below[0, 2, 4, 0] = 0.0
    gated = same_mode_active_mask(below, pretrain, valid_mode_mask=valid)
    assert gated.sum().item() == 1
    assert gated[0, 2, 4]
    valid[0, 2, 4] = False
    assert not bool(
        same_mode_active_mask(below, pretrain, valid_mode_mask=valid).any()
    )


def test_hierarchical_reduction_equalizes_vehicle_weight() -> None:
    values = torch.zeros((1, 3, 10, 2, 3), dtype=torch.float32)
    mask = torch.zeros((1, 3, 10), dtype=torch.bool)
    values[:, 0, 0] = 1.0
    mask[:, 0, 0] = True
    values[:, 1, :5] = 3.0
    mask[:, 1, :5] = True

    reduced = _hierarchical_active_mean(values, mask)

    torch.testing.assert_close(reduced, torch.tensor((1.0 + 3.0 + 0.0) / 3.0))


@pytest.fixture(scope="module")
def rollout_a():
    trainer = JointGRPOTrainerA(
        _planner(), JointGRPOConfig(trajectories_per_mode=2)
    )
    rollout = trainer.sample_groups(
        _inputs(), generator=torch.Generator().manual_seed(17)
    )
    return trainer, rollout


def test_all_mode_rollout_shapes_and_frozen_outputs(rollout_a) -> None:
    trainer, rollout = rollout_a
    assert rollout.chains_normalized.shape == (1, 2, 5, 3, 10, 8, 2)
    assert rollout.candidate_trajectories.shape == (1, 3, 10, 2, 8, 3)
    assert rollout.old_trajectory_log_prob.shape == (1, 3, 10, 2, 3)
    assert not hasattr(rollout, "sampled_modes")
    assert not hasattr(rollout, "old_mode_log_prob")
    assert not hasattr(rollout, "selected_trajectories")

    frozen = trainer.infer_frozen_pretrain(rollout)
    assert set(frozen) == {
        "selected_trajectory",
        "selected_mode",
        "all_mode_trajectories",
        "mode_logits",
    }
    assert frozen["all_mode_trajectories"].shape == (1, 3, 10, 8, 3)
    assert frozen["mode_logits"].shape == (1, 3, 10)
    assert frozen["selected_trajectory"].shape == (1, 3, 8, 3)
    assert frozen["selected_mode"].shape == (1, 3)


def test_loss_keeps_vehicle_mode_log_prob_groups_isolated(rollout_a) -> None:
    trainer, rollout = rollout_a
    rewards = torch.zeros((1, 3, 10, 2), dtype=torch.float32)
    rewards[..., 0] = -1.0
    rewards[..., 1] = 1.0
    active = rollout.mode_valid_mask.clone()

    loss = trainer.compute_loss(rollout, rewards, active)

    assert loss.advantages.shape == rewards.shape
    assert loss.new_trajectory_log_prob.shape == (1, 3, 10, 2, 3)
    torch.testing.assert_close(
        loss.new_trajectory_log_prob,
        rollout.old_trajectory_log_prob,
        rtol=0,
        atol=2e-5,
    )
    torch.testing.assert_close(
        loss.trajectory_importance_ratio,
        torch.ones_like(loss.trajectory_importance_ratio),
        rtol=0,
        atol=2e-5,
    )
    assert not hasattr(loss, "mode_pg")
    assert not hasattr(loss, "mode_reference_kl")

    invalid_active = active.clone()
    invalid_rollout = copy.copy(rollout)
    object.__setattr__(
        invalid_rollout,
        "mode_valid_mask",
        torch.zeros_like(rollout.mode_valid_mask),
    )
    with pytest.raises(JointGRPOError, match="subset of hard-valid"):
        trainer.compute_loss(invalid_rollout, rewards, invalid_active)


@pytest.mark.parametrize(
    "loss_name",
    ("trajectory_pg", "behavior_cloning", "trajectory_reference_kl"),
)
def test_inactive_modes_have_zero_gradient_for_every_loss_term(
    loss_name: str,
) -> None:
    trainer = JointGRPOTrainerA(
        _planner(), JointGRPOConfig(trajectories_per_mode=2)
    )
    rollout = trainer.sample_groups(
        _inputs(), generator=torch.Generator().manual_seed(23)
    )
    with torch.no_grad():
        trainer.planner.diffusion_decoder.trajectory_head[-1].bias.add_(0.02)

    rewards = torch.zeros((1, 3, 10, 2), dtype=torch.float32)
    rewards[..., 0] = -1.0
    rewards[..., 1] = 1.0
    active = torch.zeros((1, 3, 10), dtype=torch.bool)
    active[0, 1, 4] = True
    head_outputs: list[torch.Tensor] = []

    def capture_head_output(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        output.retain_grad()
        head_outputs.append(output)

    handle = trainer.planner.diffusion_decoder.trajectory_head.register_forward_hook(
        capture_head_output
    )
    try:
        loss = trainer.compute_loss(rollout, rewards, active)
        getattr(loss, loss_name).backward()
    finally:
        handle.remove()

    saw_active_gradient = False
    active_outputs = active[:, None, :, :, None]
    for output in head_outputs:
        if output.grad is None:
            continue
        gradient = output.grad.reshape(1, 2, 3, 10, -1)
        assert torch.count_nonzero(gradient.masked_select(~active_outputs)) == 0
        saw_active_gradient |= bool(
            torch.count_nonzero(gradient.masked_select(active_outputs))
        )
    assert saw_active_gradient


def test_only_trajectory_head_is_trainable_and_zero_signal_skips_step() -> None:
    trainer = JointGRPOTrainerA(
        _planner(), JointGRPOConfig(trajectories_per_mode=2)
    )
    expected = {
        id(parameter)
        for parameter in trainer.planner.diffusion_decoder.trajectory_head.parameters()
    }
    assert {
        id(parameter)
        for parameter in trainer.planner.parameters()
        if parameter.requires_grad
    } == expected
    assert {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    } == expected

    rollout = trainer.sample_groups(
        _inputs(), generator=torch.Generator().manual_seed(29)
    )
    before = _state(trainer.planner.diffusion_decoder.trajectory_head)
    rewards = torch.ones((1, 3, 10, 2), dtype=torch.float32)
    active = rollout.mode_valid_mask.clone()
    result = trainer.update(
        rollout,
        rewards,
        active,
        policy_update=JointGRPOPolicyUpdateConfig(update_epochs=2),
    )

    assert result.optimizer_step == 0
    assert result.zero_signal_epochs == 2
    assert all(epoch.zero_signal_epoch for epoch in result.epoch_results)
    assert _same_state(
        before, _state(trainer.planner.diffusion_decoder.trajectory_head)
    )


def test_nonzero_update_changes_head_but_not_shared_decoder_or_mode_head() -> None:
    trainer = JointGRPOTrainerA(
        _planner(), JointGRPOConfig(trajectories_per_mode=2)
    )
    rollout = trainer.sample_groups(
        _inputs(), generator=torch.Generator().manual_seed(31)
    )
    head_before = _state(trainer.planner.diffusion_decoder.trajectory_head)
    shared_before = {
        name: value.detach().clone()
        for name, value in trainer.planner.diffusion_decoder.state_dict().items()
        if not name.startswith("trajectory_head.")
    }
    mode_before = _state(trainer.planner.mode_head)
    rewards = torch.zeros((1, 3, 10, 2), dtype=torch.float32)
    rewards[..., 0] = -1.0
    rewards[..., 1] = 1.0

    result = trainer.update(
        rollout,
        rewards,
        rollout.mode_valid_mask,
        policy_update=JointGRPOPolicyUpdateConfig(update_epochs=1),
    )

    assert result.optimizer_step == 1
    assert not _same_state(
        head_before, _state(trainer.planner.diffusion_decoder.trajectory_head)
    )
    shared_after = trainer.planner.diffusion_decoder.state_dict()
    assert all(torch.equal(value, shared_after[name]) for name, value in shared_before.items())
    assert _same_state(mode_before, _state(trainer.planner.mode_head))


def test_variant_b_uses_the_same_all_mode_tensor_contract() -> None:
    planner = _planner("predicted_detached")
    trainer = JointGRPOTrainerB(
        planner, JointGRPOConfig(trajectories_per_mode=2)
    )
    rollout = trainer.sample_groups(
        _inputs(), generator=torch.Generator().manual_seed(37)
    )

    assert isinstance(rollout, JointGRPORolloutB)
    assert rollout.candidate_trajectories.shape == (1, 3, 10, 2, 8, 3)
    assert rollout.old_trajectory_log_prob.shape == (1, 3, 10, 2, 3)
    assert rollout.predecessor_action_history_normalized.shape == (
        1,
        2,
        4,
        2,
        8,
        3,
    )
