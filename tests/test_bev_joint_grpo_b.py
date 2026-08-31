from __future__ import annotations

import copy
from pathlib import Path
from unittest import mock

import pytest
import torch

from models.bev_planner import (
    BEVOnlyDiffusionPlanner,
    BEVOnlyDiffusionPlannerConfig,
    JointGRPOError,
    JointGRPORolloutB,
    JointGRPOTrainerA,
    JointGRPOTrainerB,
)
from models.bev_planner.bev_only_diffusion_planner import MAX_BACKGROUND_ACTORS
from models.bev_planner.joint_grpo import (
    JointGRPOPolicyUpdateConfig,
    _repeat_context,
    _repeat_groups,
)
from train.bev_joint_grpo import (
    GRPO_B_CHECKPOINT_FORMAT,
    grpo_b_checkpoint_payload,
    grpo_checkpoint_payload,
    load_grpo_b_checkpoint,
    save_grpo_checkpoint,
    validate_stage1_b_source_metadata,
)
from train.train_bev_diffusion_stage1 import (
    CHECKPOINT_FORMAT as STAGE1_CHECKPOINT_FORMAT,
    CHECKPOINT_SCHEMA_VERSION as STAGE1_CHECKPOINT_SCHEMA_VERSION,
)


def _planner(condition: str = "predicted_detached") -> BEVOnlyDiffusionPlanner:
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
    generator = torch.Generator().manual_seed(101)
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


def _state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _state_equal(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]
) -> bool:
    return first.keys() == second.keys() and all(
        torch.equal(first[name], second[name]) for name in first
    )


def _changed(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> bool:
    return any(not torch.equal(before[name], after[name]) for name in before)


def _source_metadata(*, diagnostic: bool = True) -> dict[str, object]:
    return {
        "schema_version": STAGE1_CHECKPOINT_SCHEMA_VERSION,
        "format": STAGE1_CHECKPOINT_FORMAT,
        "variant": "B",
        "predecessor_condition": "predicted_detached",
        "diagnostic_only": diagnostic,
        "eligible_for_formal_training": not diagnostic,
        "dataset_fingerprint": "c" * 64,
    }


@pytest.fixture(scope="module")
def rollout_pair():
    planner = _planner()
    with torch.no_grad():
        planner.diffusion_decoder.predecessor_residual_gate.fill_(0.25)
        torch.nn.init.normal_(
            planner.diffusion_decoder.trajectory_head[-1].weight,
            std=0.01,
        )
    trainer = JointGRPOTrainerB(planner)
    first = trainer.sample_groups(
        _inputs(),
        generator=torch.Generator().manual_seed(17),
    )
    second = trainer.sample_groups(
        _inputs(),
        generator=torch.Generator().manual_seed(17),
    )
    return trainer, first, second


def test_b_frozen_pretrain_matches_stage1_and_generates_own_history() -> None:
    planner = _planner()
    with torch.no_grad():
        planner.diffusion_decoder.predecessor_residual_gate.fill_(0.25)
    trainer = JointGRPOTrainerB(planner)
    inputs = _inputs()
    training_generator = torch.Generator().manual_seed(18)
    fixed_histories: list[torch.Tensor | None] = []
    original = trainer._decode_roles

    def traced(*args, **kwargs):
        if kwargs["decoder"] is trainer.reference.diffusion_decoder:
            fixed_histories.append(kwargs["fixed_history"])
        return original(*args, **kwargs)

    with mock.patch.object(
        trainer,
        "_decode_roles",
        side_effect=traced,
    ), mock.patch.object(
        trainer,
        "_infer_frozen_pretrain_from_state",
        wraps=trainer._infer_frozen_pretrain_from_state,
    ) as frozen_inference:
        rollout = trainer.sample_groups(inputs, generator=training_generator)
        generator_state = training_generator.get_state().clone()
        first = trainer.infer_frozen_pretrain(rollout)
        second = trainer.infer_frozen_pretrain(rollout)
        assert frozen_inference.call_count == 1
    with torch.inference_mode():
        stage1 = trainer.planner(**inputs)

    assert fixed_histories
    assert all(history is None for history in fixed_histories)
    assert len(fixed_histories) == planner.config.inference_denoise_steps
    assert first["trajectory_candidates"].shape == (1, 3, 10, 8, 3)
    assert first["trajectory_candidates"].dtype == torch.float32
    assert torch.isfinite(first["trajectory_candidates"]).all()
    assert first["trajectory_candidates"].requires_grad is False
    assert first["trajectory_candidates"].grad_fn is None
    assert first["trajectory_candidates"].is_inference() is False
    assert first["selected_trajectory"].shape == (1, 3, 8, 3)
    assert first["selected_trajectory"].dtype == torch.float32
    assert first["selected_mode"].shape == (1, 3)
    assert first["selected_mode"].dtype == torch.int64
    assert torch.isfinite(first["selected_trajectory"]).all()
    assert first["selected_trajectory"].requires_grad is False
    assert first["selected_trajectory"].grad_fn is None
    assert torch.equal(training_generator.get_state(), generator_state)
    torch.testing.assert_close(
        first["trajectory_candidates"],
        second["trajectory_candidates"],
    )
    torch.testing.assert_close(
        first["trajectory_candidates"],
        stage1["trajectory_candidates"],
    )
    torch.testing.assert_close(
        first["selected_trajectory"],
        second["selected_trajectory"],
    )
    torch.testing.assert_close(
        first["selected_trajectory"],
        stage1["selected_trajectory"],
    )
    assert torch.equal(first["selected_mode"], second["selected_mode"])
    assert torch.equal(first["selected_mode"], stage1["selected_mode"])


def test_b_contract_rollout_order_history_and_seed(rollout_pair) -> None:
    trainer, first, second = rollout_pair
    assert isinstance(first, JointGRPORolloutB)
    assert first.frozen_pretrain_trajectory_candidates.shape == (
        1,
        3,
        10,
        8,
        3,
    )
    assert first.frozen_pretrain_selected_trajectory.shape == (1, 3, 8, 3)
    assert first.frozen_pretrain_selected_mode.shape == (1, 3)
    assert first.frozen_pretrain_trajectory_candidates.is_inference() is False
    assert first.chains_normalized.shape == (1, 4, 5, 3, 10, 8, 2)
    assert first.sampled_modes.shape == (1, 4, 3)
    assert first.selected_trajectories.shape == (1, 4, 3, 8, 3)
    assert first.old_mode_log_prob.shape == (1, 4)
    assert first.old_trajectory_log_prob.shape == (1, 4, 3)
    history = first.predecessor_action_history_normalized
    assert history.shape == (1, 4, 4, 2, 8, 3)
    assert history.dtype == torch.float32
    assert history.requires_grad is False
    assert torch.isfinite(history).all()
    assert bool((history[..., :2].abs() <= 1.0).all())
    for name in (
        "frozen_pretrain_trajectory_candidates",
        "frozen_pretrain_selected_trajectory",
        "frozen_pretrain_selected_mode",
        "chains_normalized",
        "sampled_modes",
        "selected_trajectories",
        "old_mode_log_prob",
        "old_trajectory_log_prob",
        "predecessor_action_history_normalized",
    ):
        torch.testing.assert_close(getattr(first, name), getattr(second, name))

    order: list[int] = []
    original = trainer.planner.diffusion_decoder.forward_role

    def traced(*args, **kwargs):
        order.append(int(kwargs["role_index"]))
        return original(*args, **kwargs)

    with mock.patch.object(
        trainer.planner.diffusion_decoder,
        "forward_role",
        side_effect=traced,
    ):
        trainer.sample_groups(
            _inputs(),
            generator=torch.Generator().manual_seed(23),
        )
    assert order == [0, 1, 2] * 4


def test_stop_only_mask_and_recorded_argmax_history() -> None:
    planner = _planner()
    with torch.no_grad():
        planner.diffusion_decoder.predecessor_residual_gate.fill_(0.3)
    trainer = JointGRPOTrainerB(planner)
    inputs = _inputs()
    inputs["mode_valid_mask"][..., :9] = False
    rollout = trainer.sample_groups(
        inputs,
        generator=torch.Generator().manual_seed(29),
    )
    assert torch.equal(
        rollout.sampled_modes,
        torch.full_like(rollout.sampled_modes, 9),
    )

    groups = rollout.group_size
    context = _repeat_context(rollout.context, groups)
    coarse = _repeat_groups(rollout.coarse_trajectories, groups)
    valid = _repeat_groups(rollout.mode_valid_mask, groups)
    sample = rollout.chains_normalized.reshape(4, 5, 3, 10, 8, 2)[:, 0]
    timesteps = torch.full((4, 3), 15, dtype=torch.int64)
    _, _, expected_history = trainer._decode_roles(
        decoder=trainer.planner.diffusion_decoder,
        mode_head=trainer.planner.mode_head,
        sample=sample,
        timesteps=timesteps,
        context=context,
        coarse=coarse,
        valid_mask=valid,
        fixed_history=None,
    )
    torch.testing.assert_close(
        expected_history,
        rollout.predecessor_action_history_normalized[:, :, 0].reshape(
            4, 2, 8, 3
        ),
    )


def test_fixed_history_replay_and_reference_share_exact_condition(
    rollout_pair,
) -> None:
    trainer, rollout, _ = rollout_pair
    rewards = torch.tensor([[-1.5, -0.5, 0.5, 1.5]], dtype=torch.float32)
    seen: list[torch.Tensor] = []
    original = trainer._decode_roles

    def traced(*args, **kwargs):
        history = kwargs["fixed_history"]
        if history is not None:
            seen.append(history)
        return original(*args, **kwargs)

    with mock.patch.object(trainer, "_decode_roles", side_effect=traced):
        result = trainer.compute_loss(rollout, rewards)
    assert len(seen) == 8
    for index in range(0, len(seen), 2):
        assert seen[index].data_ptr() == seen[index + 1].data_ptr()
        torch.testing.assert_close(seen[index], seen[index + 1])
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
    torch.testing.assert_close(result.mode_reference_kl, torch.zeros(()))
    torch.testing.assert_close(
        result.trajectory_reference_kl,
        torch.zeros(()),
        atol=1e-10,
        rtol=0,
    )
    bad_history = seen[0].detach().clone().requires_grad_(True)
    with pytest.raises(JointGRPOError, match="detached"):
        trainer._decode_roles(
            decoder=trainer.planner.diffusion_decoder,
            mode_head=trainer.planner.mode_head,
            sample=rollout.chains_normalized.reshape(
                4, 5, 3, 10, 8, 2
            )[:, 0],
            timesteps=torch.full((4, 3), 15, dtype=torch.int64),
            context=_repeat_context(rollout.context, 4),
            coarse=_repeat_groups(rollout.coarse_trajectories, 4),
            valid_mask=_repeat_groups(rollout.mode_valid_mask, 4),
            fixed_history=bad_history,
        )


def test_b_reference_kl_is_positive_after_policy_perturbation(
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
    assert result.mode_reference_kl.item() > 0.0
    assert result.trajectory_reference_kl.item() > 0.0
    trainer.planner.diffusion_decoder.load_state_dict(decoder_original)
    trainer.planner.mode_head.load_state_dict(mode_original)


def test_condition_history_has_leader_middle_rear_directionality(
    rollout_pair,
) -> None:
    trainer, rollout, _ = rollout_pair
    context = _repeat_context(rollout.context, 4)
    coarse = _repeat_groups(rollout.coarse_trajectories, 4)
    valid = _repeat_groups(rollout.mode_valid_mask, 4)
    sample = rollout.chains_normalized.reshape(4, 5, 3, 10, 8, 2)[:, 0]
    timesteps = torch.full((4, 3), 15, dtype=torch.int64)
    baseline_history = (
        rollout.predecessor_action_history_normalized[:, :, 0]
        .reshape(4, 2, 8, 3)
        .clone()
    )
    baseline, _, _ = trainer._decode_roles(
        decoder=trainer.planner.diffusion_decoder,
        mode_head=trainer.planner.mode_head,
        sample=sample,
        timesteps=timesteps,
        context=context,
        coarse=coarse,
        valid_mask=valid,
        fixed_history=baseline_history,
    )

    changed_leader = baseline_history.clone()
    changed_leader[:, 0, :, 0] += 0.4
    middle_candidates, middle_features = (
        trainer.planner.diffusion_decoder.forward_role(
            trainer.planner._denormalize_xy(sample.clamp(-1.0, 1.0)),
            coarse,
            timesteps,
            context,
            role_index=1,
            predecessor_action=changed_leader[:, 0],
        )
    )
    middle_logits = trainer.planner.mode_head(middle_features).squeeze(-1)
    middle_selected = trainer.planner._select_predecessor_trajectory(
        middle_candidates,
        middle_logits,
        valid[:, 1],
    )
    changed_leader[:, 1] = trainer.planner._normalize_predecessor_action(
        middle_selected
    )
    propagated, _, _ = trainer._decode_roles(
        decoder=trainer.planner.diffusion_decoder,
        mode_head=trainer.planner.mode_head,
        sample=sample,
        timesteps=timesteps,
        context=context,
        coarse=coarse,
        valid_mask=valid,
        fixed_history=changed_leader,
    )
    torch.testing.assert_close(propagated[:, 0], baseline[:, 0])
    assert not torch.allclose(propagated[:, 1], baseline[:, 1])
    assert not torch.allclose(propagated[:, 2], baseline[:, 2])

    changed_middle = baseline_history.clone()
    changed_middle[:, 1, :, 1] -= 0.4
    rear_only, _, _ = trainer._decode_roles(
        decoder=trainer.planner.diffusion_decoder,
        mode_head=trainer.planner.mode_head,
        sample=sample,
        timesteps=timesteps,
        context=context,
        coarse=coarse,
        valid_mask=valid,
        fixed_history=changed_middle,
    )
    torch.testing.assert_close(rear_only[:, :2], baseline[:, :2])
    assert not torch.allclose(rear_only[:, 2], baseline[:, 2])


def test_zero_gate_and_nonzero_gate_gradient_phases() -> None:
    zero_planner = _planner()
    zero_trainer = JointGRPOTrainerB(zero_planner)
    zero_rollout = zero_trainer.sample_groups(
        _inputs(),
        generator=torch.Generator().manual_seed(31),
    )
    rewards = torch.tensor([[-1.5, -0.5, 0.5, 1.5]], dtype=torch.float32)
    zero_trainer.optimizer.zero_grad(set_to_none=True)
    zero_trainer.compute_loss(zero_rollout, rewards).total.backward()
    encoder = zero_planner.diffusion_decoder.predecessor_action_encoder
    gate = zero_planner.diffusion_decoder.predecessor_residual_gate
    assert encoder is not None and gate is not None
    assert gate.grad is not None and torch.isfinite(gate.grad)
    assert gate.grad.abs() > 0
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in encoder.parameters()
    )

    planner = _planner()
    with torch.no_grad():
        planner.diffusion_decoder.predecessor_residual_gate.fill_(0.25)
    trainer = JointGRPOTrainerB(planner)
    rollout = trainer.sample_groups(
        _inputs(),
        generator=torch.Generator().manual_seed(37),
    )
    frozen_before = trainer.infer_frozen_pretrain(rollout)
    decoder_before = _state(planner.diffusion_decoder)
    mode_before = _state(planner.mode_head)
    encoder = planner.diffusion_decoder.predecessor_action_encoder
    assert encoder is not None
    encoder_before = _state(encoder)
    backbone_before = _state(planner.backbone)
    fusion_before = _state(planner.bev_fusion)
    context_before = _state(planner.context_encoder)
    reference_before = _state(trainer.reference)
    update = trainer.update(
        rollout,
        rewards,
        policy_update=JointGRPOPolicyUpdateConfig(update_epochs=2),
    )
    assert update.optimizer_step == 2
    assert len(update.epoch_results) == 2
    assert all(
        epoch.gradient_norms["predecessor_action_encoder"] > 0
        for epoch in update.epoch_results
    )
    assert all(
        epoch.gradient_norms["predecessor_residual_gate"] > 0
        for epoch in update.epoch_results
    )
    assert _changed(decoder_before, _state(planner.diffusion_decoder))
    assert _changed(mode_before, _state(planner.mode_head))
    assert encoder is not None
    assert _changed(encoder_before, _state(encoder))
    assert _state_equal(backbone_before, _state(planner.backbone))
    assert _state_equal(fusion_before, _state(planner.bev_fusion))
    assert _state_equal(context_before, _state(planner.context_encoder))
    assert _state_equal(reference_before, _state(trainer.reference))
    frozen_after = trainer.infer_frozen_pretrain(rollout)
    torch.testing.assert_close(
        frozen_after["selected_trajectory"],
        frozen_before["selected_trajectory"],
    )
    assert torch.equal(
        frozen_after["selected_mode"],
        frozen_before["selected_mode"],
    )


def test_b_source_checkpoint_and_cross_variant_rejection(tmp_path: Path) -> None:
    source = _source_metadata(diagnostic=True)
    validate_stage1_b_source_metadata(
        source,
        allow_diagnostic_source=True,
    )
    with pytest.raises(JointGRPOError, match="explicit opt-in"):
        validate_stage1_b_source_metadata(
            source,
            allow_diagnostic_source=False,
        )
    with pytest.raises(JointGRPOError, match="cannot produce a formal"):
        grpo_b_checkpoint_payload(
            trainer=JointGRPOTrainerB(_planner()),
            source_stage1_sha256="d" * 64,
            source_stage1_payload=source,
            metrics={"loss/total": 0.0},
            diagnostic_only=False,
        )
    bad_a = copy.deepcopy(source)
    bad_a["variant"] = "A"
    bad_a["predecessor_condition"] = "none"
    with pytest.raises(JointGRPOError, match="variant mismatch"):
        validate_stage1_b_source_metadata(
            bad_a,
            allow_diagnostic_source=True,
        )

    trainer = JointGRPOTrainerB(_planner())
    rollout = trainer.sample_groups(
        _inputs(),
        generator=torch.Generator().manual_seed(73),
    )
    frozen_before = trainer.infer_frozen_pretrain(rollout)
    payload = grpo_b_checkpoint_payload(
        trainer=trainer,
        source_stage1_sha256="d" * 64,
        source_stage1_payload=source,
        metrics={"loss/total": 0.0},
        diagnostic_only=True,
    )
    assert payload["format"] == GRPO_B_CHECKPOINT_FORMAT
    assert payload["variant"] == "B"
    assert payload["predecessor_condition"] == "predicted_detached"
    path = save_grpo_checkpoint(tmp_path / "b.pt", payload)
    restored = JointGRPOTrainerB(_planner())
    load_grpo_b_checkpoint(
        path,
        restored,
        expected_source_stage1_sha256="d" * 64,
    )
    assert _state_equal(_state(restored.planner), _state(trainer.planner))
    assert _state_equal(_state(restored.reference), _state(trainer.reference))
    frozen_after = restored.infer_frozen_pretrain(rollout)
    torch.testing.assert_close(
        frozen_after["selected_trajectory"],
        frozen_before["selected_trajectory"],
    )
    assert torch.equal(
        frozen_after["selected_mode"],
        frozen_before["selected_mode"],
    )

    with pytest.raises(JointGRPOError, match="trainer variant"):
        load_grpo_b_checkpoint(
            path,
            JointGRPOTrainerA(_planner("none")),  # type: ignore[arg-type]
            expected_source_stage1_sha256="d" * 64,
        )

    formal_source = _source_metadata(diagnostic=False)
    formal_payload = grpo_b_checkpoint_payload(
        trainer=trainer,
        source_stage1_sha256="d" * 64,
        source_stage1_payload=formal_source,
        metrics={"loss/total": 0.0},
        diagnostic_only=False,
    )
    assert formal_payload["diagnostic_only"] is False
    assert formal_payload["eligible_for_formal_training"] is True

    a_source = dict(formal_source)
    a_source["variant"] = "A"
    a_source["predecessor_condition"] = "none"
    a_payload = grpo_checkpoint_payload(
        trainer=JointGRPOTrainerA(_planner("none")),
        source_stage1_sha256="d" * 64,
        source_stage1_payload=a_source,
        metrics={"loss/total": 0.0},
        diagnostic_only=False,
    )
    a_path = save_grpo_checkpoint(tmp_path / "a.pt", a_payload)
    with pytest.raises(JointGRPOError, match="format mismatch"):
        load_grpo_b_checkpoint(
            a_path,
            JointGRPOTrainerB(_planner()),
            expected_source_stage1_sha256="d" * 64,
        )
