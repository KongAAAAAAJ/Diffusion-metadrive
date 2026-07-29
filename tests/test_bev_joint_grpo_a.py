from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from models.bev_planner import (
    BEVOnlyDiffusionPlanner,
    BEVOnlyDiffusionPlannerConfig,
    JointGRPOConfig,
    JointGRPOError,
    JointGRPOTrainerA,
    StandardGaussianDDIM,
    normalize_signed_advantages,
)
from models.bev_planner.joint_grpo import (
    _gather_modes,
    _repeat_context,
    _repeat_groups,
    _trajectory_bc,
)
from train.bev_joint_grpo import (
    GRPO_CHECKPOINT_FORMAT,
    GRPO_CHECKPOINT_SCHEMA_VERSION,
    grpo_checkpoint_payload,
    load_grpo_checkpoint,
    save_grpo_checkpoint,
    validate_stage1_a_source_metadata,
)
from train.train_bev_diffusion_stage1 import (
    CHECKPOINT_FORMAT as STAGE1_CHECKPOINT_FORMAT,
    CHECKPOINT_SCHEMA_VERSION as STAGE1_CHECKPOINT_SCHEMA_VERSION,
)


def _planner() -> BEVOnlyDiffusionPlanner:
    return BEVOnlyDiffusionPlanner(
        BEVOnlyDiffusionPlannerConfig(
            d_model=32,
            num_heads=4,
            ffn_dim=64,
            decoder_layers=1,
            predecessor_condition="none",
        )
    )


def _model_inputs(batch_size: int = 1) -> dict[str, torch.Tensor]:
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
        "relation_valid_mask": torch.ones(
            (batch_size, 3, 2), dtype=torch.bool
        ),
        "agent_role": torch.arange(3, dtype=torch.int64)
        .unsqueeze(0)
        .repeat(batch_size, 1),
        "coarse_trajectories": coarse,
        "mode_valid_mask": torch.ones(
            (batch_size, 3, 10), dtype=torch.bool
        ),
    }


def _state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _changed(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> bool:
    return any(not torch.equal(before[name], after[name]) for name in before)


def _state_equal(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]
) -> bool:
    return first.keys() == second.keys() and all(
        torch.equal(first[name], second[name]) for name in first
    )


def _source_metadata(*, diagnostic: bool = True) -> dict[str, object]:
    return {
        "schema_version": STAGE1_CHECKPOINT_SCHEMA_VERSION,
        "format": STAGE1_CHECKPOINT_FORMAT,
        "variant": "A",
        "predecessor_condition": "none",
        "diagnostic_only": diagnostic,
        "eligible_for_formal_training": not diagnostic,
        "dataset_fingerprint": "b" * 64,
    }


def test_config_and_signed_advantage_contract() -> None:
    config = JointGRPOConfig()
    assert config.group_size == 4
    assert config.initial_noise_timestep == 8
    assert config.denoise_steps == 4
    assert config.roll_timesteps == (15, 10, 5, 0)
    assert config.stochastic_timesteps == (15, 10, 5)
    assert config.eta == 1.0
    assert config.mode_pg_weight == config.trajectory_pg_weight == 1.0
    assert config.bc_weight == 0.1
    assert config.reference_kl_weight == 0.02
    with pytest.raises(JointGRPOError, match="group_size"):
        JointGRPOConfig(group_size=3)
    with pytest.raises(JointGRPOError, match="initial_noise_timestep"):
        JointGRPOConfig(initial_noise_timestep=9)
    with pytest.raises(JointGRPOError, match="non-negative"):
        JointGRPOConfig(reference_kl_weight=-0.1)

    rewards = torch.tensor(
        [[-1.5, -0.5, 0.5, 1.5], [2.0, 2.0, 2.0, 2.0]],
        dtype=torch.float32,
    )
    advantages = normalize_signed_advantages(rewards)
    assert torch.any(advantages[0] < 0)
    assert torch.any(advantages[0] > 0)
    torch.testing.assert_close(advantages.mean(dim=1), torch.zeros(2))
    torch.testing.assert_close(advantages[1], torch.zeros(4))
    with pytest.raises(JointGRPOError, match="dtype"):
        normalize_signed_advantages(rewards.double())
    with pytest.raises(JointGRPOError, match="shape"):
        normalize_signed_advantages(rewards[:, :3])
    bad = rewards.clone()
    bad[0, 0] = torch.nan
    with pytest.raises(JointGRPOError, match="finite"):
        normalize_signed_advantages(bad)


def test_standard_gaussian_ddim_sampling_replay_and_schedule() -> None:
    transition = StandardGaussianDDIM()
    sample = torch.zeros((2, 3, 10, 8, 2), dtype=torch.float32)
    prediction = torch.full_like(sample, 0.25)
    first = transition.step(
        model_output=prediction,
        timestep=15,
        previous_timestep=10,
        sample=sample,
        eta=1.0,
        generator=torch.Generator().manual_seed(7),
    )
    assert first.log_prob is not None
    assert first.log_prob.shape == (2, 3, 10)
    replay = transition.step(
        model_output=prediction,
        timestep=15,
        previous_timestep=10,
        sample=sample,
        eta=1.0,
        prev_sample=first.prev_sample,
    )
    torch.testing.assert_close(replay.mean, first.mean)
    torch.testing.assert_close(replay.log_prob, first.log_prob)
    standardized = (first.prev_sample - first.mean) / first.std
    expected = (
        -0.5 * standardized.square()
        - torch.log(first.std)
        - 0.5 * torch.log(torch.tensor(2.0 * torch.pi))
    ).sum(dim=(-2, -1))
    torch.testing.assert_close(first.log_prob, expected)

    terminal = transition.step(
        model_output=prediction,
        timestep=0,
        previous_timestep=-1,
        sample=sample,
        eta=1.0,
    )
    assert terminal.log_prob is None
    assert terminal.std.item() == 0.0
    with pytest.raises(JointGRPOError, match="previous_timestep"):
        transition.step(
            model_output=prediction,
            timestep=0,
            previous_timestep=-0,
            sample=sample,
            eta=1.0,
        )


@pytest.fixture(scope="module")
def rollout_pair():
    trainer = JointGRPOTrainerA(_planner())
    model_inputs = _model_inputs()
    first = trainer.sample_groups(
        model_inputs,
        generator=torch.Generator().manual_seed(17),
    )
    second = trainer.sample_groups(
        model_inputs,
        generator=torch.Generator().manual_seed(17),
    )
    return trainer, first, second


def test_joint_rollout_shapes_seed_and_hard_mask(rollout_pair) -> None:
    _, first, second = rollout_pair
    assert first.chains_normalized.shape == (1, 4, 5, 3, 10, 8, 2)
    assert first.sampled_modes.shape == (1, 4, 3)
    assert first.sampled_modes.dtype == torch.int64
    assert first.selected_trajectories.shape == (1, 4, 3, 8, 3)
    assert first.old_mode_log_prob.shape == (1, 4)
    assert first.old_trajectory_log_prob.shape == (1, 4, 3)
    for name in (
        "chains_normalized",
        "sampled_modes",
        "selected_trajectories",
        "old_mode_log_prob",
        "old_trajectory_log_prob",
    ):
        torch.testing.assert_close(getattr(first, name), getattr(second, name))

    stop_inputs = _model_inputs()
    stop_inputs["mode_valid_mask"][..., :9] = False
    stop_rollout = JointGRPOTrainerA(_planner()).sample_groups(
        stop_inputs,
        generator=torch.Generator().manual_seed(23),
    )
    assert torch.equal(
        stop_rollout.sampled_modes,
        torch.full_like(stop_rollout.sampled_modes, 9),
    )


def test_joint_probability_replay_zero_kl_and_weighted_loss(
    rollout_pair,
) -> None:
    trainer, rollout, _ = rollout_pair
    rewards = torch.tensor([[-1.5, -0.5, 0.5, 1.5]], dtype=torch.float32)
    result = trainer.compute_loss(rollout, rewards)
    torch.testing.assert_close(
        result.new_mode_log_prob,
        rollout.old_mode_log_prob,
        rtol=0,
        atol=2e-6,
    )
    torch.testing.assert_close(
        result.new_trajectory_log_prob,
        rollout.old_trajectory_log_prob,
        rtol=0,
        atol=2e-5,
    )

    groups = rollout.group_size
    flat_count = rollout.batch_size * groups
    context = _repeat_context(rollout.context, groups)
    coarse = _repeat_groups(rollout.coarse_trajectories, groups)
    valid = _repeat_groups(rollout.mode_valid_mask, groups)
    modes = rollout.sampled_modes.reshape(flat_count, 3)
    chains = rollout.chains_normalized.reshape(
        flat_count, 5, 3, 10, 8, 2
    )
    role_transition_log_probs = []
    final_logits = None
    with torch.no_grad():
        for index, (timestep, previous_timestep) in enumerate(
            ((15, 10), (10, 5), (5, 0), (0, -1))
        ):
            timesteps = torch.full(
                (flat_count, 3), timestep, dtype=torch.int64
            )
            candidates, final_logits = (
                trainer.planner.predict_denoised_candidates(
                    chains[:, index],
                    timesteps,
                    context,
                    coarse,
                    valid,
                )
            )
            transition = trainer.transition.step(
                model_output=trainer.planner._normalize_xy(
                    candidates[..., :2]
                ).float(),
                timestep=timestep,
                previous_timestep=previous_timestep,
                sample=chains[:, index].float(),
                eta=1.0,
                prev_sample=chains[:, index + 1].float(),
            )
            if transition.log_prob is not None:
                role_transition_log_probs.append(
                    _gather_modes(
                        transition.log_prob.unsqueeze(-1), modes
                    ).squeeze(-1)
                )
    assert final_logits is not None
    role_mode_log_prob = torch.log_softmax(
        final_logits.float().masked_fill(~valid, float("-inf")),
        dim=-1,
    ).gather(-1, modes.unsqueeze(-1)).squeeze(-1)
    manual_mode_joint = role_mode_log_prob.sum(dim=-1).reshape(1, 4)
    manual_trajectory_joint = torch.stack(
        [value.sum(dim=-1) for value in role_transition_log_probs],
        dim=-1,
    ).reshape(1, 4, 3)
    torch.testing.assert_close(
        manual_mode_joint, rollout.old_mode_log_prob
    )
    torch.testing.assert_close(
        manual_trajectory_joint, rollout.old_trajectory_log_prob
    )
    torch.testing.assert_close(result.behavior_cloning, torch.zeros(()))
    torch.testing.assert_close(result.mode_reference_kl, torch.zeros(()))
    torch.testing.assert_close(
        result.trajectory_reference_kl, torch.zeros(())
    )
    expected = (
        result.mode_pg
        + result.trajectory_pg
        + 0.1 * result.behavior_cloning
        + 0.02 * result.reference_kl
    )
    torch.testing.assert_close(result.total, expected)

    current = torch.zeros((1, 8, 3), dtype=torch.float32)
    reference = current.clone()
    current[..., 2] = -torch.pi + 0.01
    reference[..., 2] = torch.pi - 0.01
    wrapped = _trajectory_bc(current, reference, JointGRPOConfig())
    assert wrapped.item() < 0.01


def test_reference_kl_becomes_positive_after_policy_perturbation(
    rollout_pair,
) -> None:
    trainer, rollout, _ = rollout_pair
    decoder_original = _state(trainer.planner.diffusion_decoder)
    mode_original = _state(trainer.planner.mode_head)
    with torch.no_grad():
        trainer.planner.diffusion_decoder.trajectory_head[-1].bias.add_(0.01)
        trainer.planner.mode_head.weight[0, 0].add_(0.05)
    result = trainer.compute_loss(
        rollout,
        torch.tensor([[-1.5, -0.5, 0.5, 1.5]], dtype=torch.float32),
    )
    assert result.reference_kl.item() >= 0.0
    assert result.mode_reference_kl.item() > 0.0
    assert result.trajectory_reference_kl.item() > 0.0
    trainer.planner.diffusion_decoder.load_state_dict(decoder_original)
    trainer.planner.mode_head.load_state_dict(mode_original)


def test_exactly_one_update_changes_only_trainable_policy() -> None:
    trainer = JointGRPOTrainerA(_planner())
    optimizer_parameters = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    expected_trainable = {
        id(parameter)
        for parameter in (
            *trainer.planner.diffusion_decoder.parameters(),
            *trainer.planner.mode_head.parameters(),
        )
    }
    assert optimizer_parameters == expected_trainable
    assert {
        id(parameter)
        for parameter in trainer.planner.parameters()
        if parameter.requires_grad
    } == expected_trainable
    assert not any(
        parameter.requires_grad for parameter in trainer.reference.parameters()
    )
    inputs = _model_inputs()
    inputs["mode_valid_mask"][..., 1:4] = False
    inputs["mode_valid_mask"][..., 6:9] = False
    rollout = trainer.sample_groups(
        inputs,
        generator=torch.Generator().manual_seed(29),
    )
    decoder_before = _state(trainer.planner.diffusion_decoder)
    mode_before = _state(trainer.planner.mode_head)
    backbone_before = _state(trainer.planner.backbone)
    fusion_before = _state(trainer.planner.bev_fusion)
    context_before = _state(trainer.planner.context_encoder)
    reference_before = _state(trainer.reference)
    result = trainer.update(
        rollout,
        torch.tensor([[-1.5, -0.5, 0.5, 1.5]], dtype=torch.float32),
    )
    assert result.optimizer_step == trainer.optimizer_step == 1
    assert result.gradient_norms["diffusion_decoder"] > 0.0
    assert result.gradient_norms["mode_head"] > 0.0
    assert _changed(decoder_before, _state(trainer.planner.diffusion_decoder))
    assert _changed(mode_before, _state(trainer.planner.mode_head))
    assert _state_equal(backbone_before, _state(trainer.planner.backbone))
    assert _state_equal(fusion_before, _state(trainer.planner.bev_fusion))
    assert _state_equal(context_before, _state(trainer.planner.context_encoder))
    assert _state_equal(reference_before, _state(trainer.reference))
    assert all(
        torch.isfinite(parameter).all() for parameter in trainer.planner.parameters()
    )
    with pytest.raises(JointGRPOError, match="exactly once"):
        trainer.update(
            rollout,
            torch.tensor([[-1.5, -0.5, 0.5, 1.5]], dtype=torch.float32),
        )


def test_source_metadata_and_grpo_checkpoint_round_trip(tmp_path: Path) -> None:
    source = _source_metadata(diagnostic=True)
    validate_stage1_a_source_metadata(
        source,
        allow_diagnostic_source=True,
    )
    with pytest.raises(JointGRPOError, match="explicit opt-in"):
        validate_stage1_a_source_metadata(
            source,
            allow_diagnostic_source=False,
        )
    bad_b = copy.deepcopy(source)
    bad_b["variant"] = "B"
    bad_b["predecessor_condition"] = "predicted_detached"
    with pytest.raises(JointGRPOError, match="variant mismatch"):
        validate_stage1_a_source_metadata(
            bad_b,
            allow_diagnostic_source=True,
        )
    legacy = copy.deepcopy(source)
    legacy["schema_version"] = 1
    with pytest.raises(JointGRPOError, match="schema_version mismatch"):
        validate_stage1_a_source_metadata(
            legacy,
            allow_diagnostic_source=True,
        )

    trainer = JointGRPOTrainerA(_planner())
    payload = grpo_checkpoint_payload(
        trainer=trainer,
        source_stage1_sha256="a" * 64,
        source_stage1_payload=source,
        metrics={"loss/total": 0.0},
        diagnostic_only=True,
    )
    assert payload["schema_version"] == GRPO_CHECKPOINT_SCHEMA_VERSION == 1
    assert payload["format"] == GRPO_CHECKPOINT_FORMAT
    assert payload["diagnostic_only"] is True
    assert payload["eligible_for_formal_training"] is False
    path = save_grpo_checkpoint(tmp_path / "diagnostic.pt", payload)
    restored = JointGRPOTrainerA(_planner())
    loaded = load_grpo_checkpoint(
        path,
        restored,
        expected_source_stage1_sha256="a" * 64,
    )
    assert loaded["optimizer_step"] == 0
    assert _state_equal(_state(restored.planner), _state(trainer.planner))
    assert _state_equal(_state(restored.reference), _state(trainer.reference))

    bad_checkpoint = copy.deepcopy(payload)
    bad_checkpoint["variant"] = "B"
    bad_path = save_grpo_checkpoint(tmp_path / "variant_b.pt", bad_checkpoint)
    with pytest.raises(JointGRPOError, match="variant mismatch"):
        load_grpo_checkpoint(
            bad_path,
            JointGRPOTrainerA(_planner()),
            expected_source_stage1_sha256="a" * 64,
        )

    with pytest.raises(JointGRPOError, match="diagnostic Stage 1"):
        grpo_checkpoint_payload(
            trainer=trainer,
            source_stage1_sha256="a" * 64,
            source_stage1_payload=source,
            metrics={},
            diagnostic_only=False,
        )
