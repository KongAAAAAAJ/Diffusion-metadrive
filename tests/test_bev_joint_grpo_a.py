from __future__ import annotations

import copy
import dataclasses
import gc
import weakref
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
)
from models.bev_planner.bev_only_diffusion_planner import MAX_BACKGROUND_ACTORS
from train.bev_joint_grpo import (
    GRPO_CHECKPOINT_FORMAT,
    GRPO_CHECKPOINT_SCHEMA_VERSION,
    GRPO_REWARD_SOURCE,
    ALIGNED_GRPO_CHECKPOINT_FORMAT,
    ALIGNED_GRPO_CHECKPOINT_SCHEMA_VERSION,
    LEGACY_GRPO_CHECKPOINT_FORMAT,
    LEGACY_GRPO_CHECKPOINT_SCHEMA_VERSION,
    PREVIOUS_GRPO_CHECKPOINT_FORMAT,
    PREVIOUS_GRPO_CHECKPOINT_SCHEMA_VERSION,
    SAME_MODE_GRPO_CHECKPOINT_FORMAT,
    SAME_MODE_GRPO_CHECKPOINT_SCHEMA_VERSION,
    grpo_checkpoint_payload,
    load_grpo_a_checkpoint_for_evaluation,
    load_grpo_a_config_for_evaluation,
    load_grpo_checkpoint,
    load_grpo_config_from_checkpoint,
    save_grpo_checkpoint,
    validate_stage1_a_source_metadata,
)
from train.train_bev_diffusion_stage1 import (
    CHECKPOINT_FORMAT as STAGE1_CHECKPOINT_FORMAT,
    CHECKPOINT_SCHEMA_VERSION as STAGE1_CHECKPOINT_SCHEMA_VERSION,
)


def _planner() -> BEVOnlyDiffusionPlanner:
    planner = BEVOnlyDiffusionPlanner(
        BEVOnlyDiffusionPlannerConfig(
            d_model=32,
            num_heads=4,
            ffn_dim=64,
            decoder_layers=1,
            predecessor_condition="none",
        )
    )
    with torch.no_grad():
        torch.nn.init.normal_(
            planner.diffusion_decoder.trajectory_head[-1].weight, std=0.01
        )
    return planner


def _inputs() -> dict[str, torch.Tensor]:
    coarse = torch.zeros((1, 3, 10, 8, 3), dtype=torch.float32)
    coarse[..., 0] = torch.arange(1, 9, dtype=torch.float32) * 0.5
    return {
        "bev": torch.zeros((1, 3, 8, 256, 256), dtype=torch.uint8),
        "ego_state": torch.zeros((1, 3, 8), dtype=torch.float32),
        "formation_relation_state": torch.zeros((1, 3, 12), dtype=torch.float32),
        "relation_valid_mask": torch.ones((1, 3, 2), dtype=torch.bool),
        "agent_role": torch.arange(3, dtype=torch.int64).unsqueeze(0),
        "coarse_trajectories": coarse,
        "mode_valid_mask": torch.ones((1, 3, 10), dtype=torch.bool),
        "background_actor_state": torch.zeros(
            (1, 3, MAX_BACKGROUND_ACTORS, 8), dtype=torch.float32
        ),
        "background_actor_valid_mask": torch.zeros(
            (1, 3, MAX_BACKGROUND_ACTORS), dtype=torch.bool
        ),
        "scenario_code": torch.ones((1,), dtype=torch.int64),
        "rule_formation_state": torch.zeros((1,), dtype=torch.int64),
        "rule_action_condition": torch.zeros((1, 3), dtype=torch.int64),
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


def test_standard_gaussian_ddim_sampling_and_exact_replay() -> None:
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
    replay = transition.step(
        model_output=prediction,
        timestep=15,
        previous_timestep=10,
        sample=sample,
        eta=1.0,
        prev_sample=first.prev_sample,
    )
    torch.testing.assert_close(first.mean, replay.mean)
    torch.testing.assert_close(first.log_prob, replay.log_prob)
    terminal = transition.step(
        model_output=prediction,
        timestep=0,
        previous_timestep=-1,
        sample=sample,
        eta=1.0,
    )
    assert terminal.log_prob is None


def test_frozen_pretrain_returns_all_modes_and_matches_stage1() -> None:
    trainer = JointGRPOTrainerA(
        _planner(),
        JointGRPOConfig(
            trajectories_per_mode=2,
            post_update_reference_kl_max=1e6,
            max_adapter_relative_drift=1e6,
        ),
    )
    inputs = _inputs()
    rollout = trainer.sample_groups(
        inputs, generator=torch.Generator().manual_seed(17)
    )
    first = trainer.infer_frozen_pretrain(rollout)
    second = trainer.infer_frozen_pretrain(rollout)
    direct = trainer.infer_frozen_pretrain_from_inputs(inputs)
    with torch.inference_mode():
        stage1 = trainer.planner(**inputs)
    assert first["all_mode_trajectories"].shape == (1, 3, 10, 8, 3)
    assert first["mode_logits"].shape == (1, 3, 10)
    torch.testing.assert_close(
        first["all_mode_trajectories"], second["all_mode_trajectories"]
    )
    torch.testing.assert_close(
        first["all_mode_trajectories"], direct["all_mode_trajectories"]
    )
    torch.testing.assert_close(
        first["selected_trajectory"], stage1["selected_trajectory"]
    )
    assert torch.equal(first["selected_mode"], stage1["selected_mode"])


@pytest.mark.parametrize("separate_transition_generator", [False, True])
def test_current_only_sampling_matches_paired_current_path_and_rng_progress(
    separate_transition_generator: bool,
) -> None:
    trainer = JointGRPOTrainerA(
        _planner(),
        JointGRPOConfig(
            trajectories_per_mode=2,
            post_update_reference_kl_max=1e6,
            max_adapter_relative_drift=1e6,
        ),
    )
    paired_generator = torch.Generator().manual_seed(29)
    current_generator = torch.Generator().manual_seed(29)
    paired_transition_generator = (
        torch.Generator().manual_seed(31)
        if separate_transition_generator
        else None
    )
    current_transition_generator = (
        torch.Generator().manual_seed(31)
        if separate_transition_generator
        else None
    )

    paired = trainer.sample_groups(
        _inputs(),
        generator=paired_generator,
        transition_generator=paired_transition_generator,
    )
    current = trainer.sample_current_groups(
        _inputs(),
        generator=current_generator,
        transition_generator=current_transition_generator,
    )

    assert current.shape == (1, 3, 10, 2, 8, 3)
    assert torch.equal(current, paired.candidate_trajectories)
    assert torch.equal(
        torch.rand((7,), generator=paired_generator),
        torch.rand((7,), generator=current_generator),
    )
    if separate_transition_generator:
        assert paired_transition_generator is not None
        assert current_transition_generator is not None
        assert torch.equal(
            torch.rand((7,), generator=paired_transition_generator),
            torch.rand((7,), generator=current_transition_generator),
        )


def test_rollout_can_enter_update_only_once() -> None:
    trainer = JointGRPOTrainerA(
        _planner(),
        JointGRPOConfig(
            trajectories_per_mode=2,
            post_update_reference_kl_max=1e6,
            max_adapter_relative_drift=1e6,
        ),
    )
    rollout = trainer.sample_groups(
        _inputs(), generator=torch.Generator().manual_seed(19)
    )
    rewards = torch.zeros((1, 3, 10, 2), dtype=torch.float32)
    rewards[..., 0] = -1.0
    rewards[..., 1] = 1.0
    rollout = rollout.with_reward_signals(
        current_rewards=rewards,
        frozen_rewards=rewards - 1.0,
        collision_mask=torch.zeros_like(rewards, dtype=torch.bool),
        out_of_drivable_mask=torch.zeros_like(rewards, dtype=torch.bool),
        valid_executable_mode_mask=rollout.mode_valid_mask,
    )
    trainer.update(rollout)
    with pytest.raises(JointGRPOError, match="exactly once"):
        trainer.update(rollout)
    reference = weakref.ref(rollout)
    del rollout
    gc.collect()
    assert reference() is None
    assert len(trainer._consumed_rollouts) == 0


def test_stage1_source_metadata_requires_explicit_diagnostic_opt_in() -> None:
    source = _source_metadata(diagnostic=True)
    validate_stage1_a_source_metadata(source, allow_diagnostic_source=True)
    with pytest.raises(JointGRPOError, match="explicit opt-in"):
        validate_stage1_a_source_metadata(source, allow_diagnostic_source=False)
    wrong = dict(source)
    wrong["variant"] = "B"
    with pytest.raises(JointGRPOError, match="variant mismatch"):
        validate_stage1_a_source_metadata(wrong, allow_diagnostic_source=True)


def test_schema6_strict_resume_restores_optimizer_and_reference(tmp_path: Path) -> None:
    trainer = JointGRPOTrainerA(
        _planner(), JointGRPOConfig(trajectories_per_mode=2)
    )
    payload = grpo_checkpoint_payload(
        trainer=trainer,
        source_stage1_sha256="a" * 64,
        source_stage1_payload=_source_metadata(),
        metrics={"loss/total": 0.0},
        diagnostic_only=True,
    )
    assert payload["schema_version"] == GRPO_CHECKPOINT_SCHEMA_VERSION == 6
    assert payload["format"] == GRPO_CHECKPOINT_FORMAT
    assert payload["optimizer_contract_version"] == "stage2_joint_grpo_optimizer_v8"
    path = save_grpo_checkpoint(tmp_path / "current.pt", payload)

    restored = JointGRPOTrainerA(
        _planner(), JointGRPOConfig(trajectories_per_mode=2)
    )
    loaded = load_grpo_checkpoint(
        path, restored, expected_source_stage1_sha256="a" * 64
    )
    assert loaded["optimizer_step"] == 0
    assert _same_state(_state(restored.planner), _state(trainer.planner))
    assert _same_state(_state(restored.reference), _state(trainer.reference))
    assert load_grpo_config_from_checkpoint(path) == trainer.config


@pytest.mark.parametrize(
    "generation", ["legacy", "previous", "same_mode_v3", "aligned_v5"]
)
def test_old_schema_is_planner_only_for_historical_evaluation(
    tmp_path: Path, generation: str
) -> None:
    trainer = JointGRPOTrainerA(
        _planner(), JointGRPOConfig(trajectories_per_mode=2)
    )
    payload = grpo_checkpoint_payload(
        trainer=trainer,
        source_stage1_sha256="a" * 64,
        source_stage1_payload=_source_metadata(),
        metrics={"loss/total": 0.0},
        diagnostic_only=True,
    )
    historical = copy.deepcopy(payload)
    historical["planner_state"] = {
        name.replace(
            "diffusion_decoder.trajectory_head.base.",
            "diffusion_decoder.trajectory_head.",
        ): value
        for name, value in historical["planner_state"].items()
        if "trajectory_head.mode_residual_" not in name
    }
    raw_config = dataclasses.asdict(trainer.config)
    raw_config["group_size"] = raw_config.pop("trajectories_per_mode")
    if generation == "legacy":
        historical["schema_version"] = LEGACY_GRPO_CHECKPOINT_SCHEMA_VERSION
        historical["format"] = LEGACY_GRPO_CHECKPOINT_FORMAT
        historical["reward_source"] = "external"
        historical.pop("optimizer_contract_version")
    elif generation == "previous":
        historical["schema_version"] = PREVIOUS_GRPO_CHECKPOINT_SCHEMA_VERSION
        historical["format"] = PREVIOUS_GRPO_CHECKPOINT_FORMAT
        historical["reward_source"] = "external_with_explicit_pretrain_baseline"
        historical["optimizer_contract_version"] = "stage2_joint_grpo_optimizer_v4"
    elif generation == "same_mode_v3":
        historical["schema_version"] = SAME_MODE_GRPO_CHECKPOINT_SCHEMA_VERSION
        historical["format"] = SAME_MODE_GRPO_CHECKPOINT_FORMAT
        historical["reward_source"] = GRPO_REWARD_SOURCE
        historical["optimizer_contract_version"] = "stage2_joint_grpo_optimizer_v5"
    else:
        historical["schema_version"] = ALIGNED_GRPO_CHECKPOINT_SCHEMA_VERSION
        historical["format"] = ALIGNED_GRPO_CHECKPOINT_FORMAT
        historical["reward_source"] = GRPO_REWARD_SOURCE
        historical["optimizer_contract_version"] = "stage2_joint_grpo_optimizer_v7"
    historical["grpo_config"] = raw_config
    historical["optimizer_step"] = 27
    path = save_grpo_checkpoint(tmp_path / f"{generation}.pt", historical)

    evaluation = JointGRPOTrainerA(
        _planner(), JointGRPOConfig(trajectories_per_mode=2)
    )
    reference_before = _state(evaluation.reference)
    loaded = load_grpo_a_checkpoint_for_evaluation(
        path, evaluation, expected_source_stage1_sha256="a" * 64
    )
    assert loaded["optimizer_step"] == 27
    assert _same_state(_state(evaluation.planner), _state(trainer.planner))
    assert _same_state(_state(evaluation.reference), reference_before)
    assert evaluation.optimizer.state == {}
    assert evaluation.optimizer_step == 0
    assert load_grpo_a_config_for_evaluation(path) == trainer.config
    with pytest.raises(JointGRPOError, match="schema_version mismatch"):
        load_grpo_config_from_checkpoint(path)
    with pytest.raises(JointGRPOError, match="schema_version mismatch"):
        load_grpo_checkpoint(
            path,
            JointGRPOTrainerA(
                _planner(), JointGRPOConfig(trajectories_per_mode=2)
            ),
            expected_source_stage1_sha256="a" * 64,
        )
