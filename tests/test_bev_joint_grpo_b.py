from __future__ import annotations

from pathlib import Path
from unittest import mock

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
)
from models.bev_planner.bev_only_diffusion_planner import MAX_BACKGROUND_ACTORS
from train.bev_joint_grpo import (
    GRPO_B_CHECKPOINT_FORMAT,
    GRPO_CHECKPOINT_SCHEMA_VERSION,
    grpo_b_checkpoint_payload,
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
        "variant": "B",
        "predecessor_condition": "predicted_detached",
        "diagnostic_only": diagnostic,
        "eligible_for_formal_training": not diagnostic,
        "dataset_fingerprint": "c" * 64,
    }


def test_b_frozen_pretrain_matches_stage1_and_builds_own_history() -> None:
    planner = _planner()
    with torch.no_grad():
        planner.diffusion_decoder.predecessor_residual_gate.fill_(0.25)
    trainer = JointGRPOTrainerB(
        planner, JointGRPOConfig(trajectories_per_mode=2)
    )
    inputs = _inputs()
    rollout = trainer.sample_groups(
        inputs, generator=torch.Generator().manual_seed(17)
    )
    fixed_histories: list[torch.Tensor | None] = []
    original = trainer._decode_roles

    def traced(*args, **kwargs):
        if kwargs["decoder"] is trainer.reference.diffusion_decoder:
            fixed_histories.append(kwargs["fixed_history"])
        return original(*args, **kwargs)

    with mock.patch.object(trainer, "_decode_roles", side_effect=traced):
        frozen = trainer.infer_frozen_pretrain(rollout)
    with torch.inference_mode():
        stage1 = trainer.planner(**inputs)

    assert fixed_histories
    assert all(history is None for history in fixed_histories)
    torch.testing.assert_close(
        frozen["all_mode_trajectories"], stage1["trajectory_candidates"]
    )
    torch.testing.assert_close(
        frozen["selected_trajectory"], stage1["selected_trajectory"]
    )
    assert torch.equal(frozen["selected_mode"], stage1["selected_mode"])


def test_b_rollout_and_replay_use_all_mode_contract_and_fixed_history() -> None:
    planner = _planner()
    with torch.no_grad():
        planner.diffusion_decoder.predecessor_residual_gate.fill_(0.25)
        torch.nn.init.normal_(
            planner.diffusion_decoder.trajectory_head[-1].weight, std=0.01
        )
    trainer = JointGRPOTrainerB(
        planner, JointGRPOConfig(trajectories_per_mode=2)
    )
    rollout = trainer.sample_groups(
        _inputs(), generator=torch.Generator().manual_seed(19)
    )
    assert isinstance(rollout, JointGRPORolloutB)
    assert rollout.chains_normalized.shape == (1, 2, 5, 3, 10, 8, 2)
    assert rollout.candidate_trajectories.shape == (1, 3, 10, 2, 8, 3)
    assert rollout.old_trajectory_log_prob.shape == (1, 3, 10, 2, 3)
    assert rollout.predecessor_action_history_normalized.shape == (1, 2, 4, 2, 8, 3)

    seen: list[torch.Tensor] = []
    original = trainer._decode_roles

    def traced(*args, **kwargs):
        history = kwargs["fixed_history"]
        if history is not None:
            seen.append(history)
        return original(*args, **kwargs)

    rewards = torch.zeros((1, 3, 10, 2), dtype=torch.float32)
    rewards[..., 0] = -1.0
    rewards[..., 1] = 1.0
    with mock.patch.object(trainer, "_decode_roles", side_effect=traced):
        loss = trainer.compute_loss(
            rollout, rewards, rollout.mode_valid_mask
        )
    assert len(seen) == 8
    for index in range(0, len(seen), 2):
        assert seen[index].data_ptr() == seen[index + 1].data_ptr()
    torch.testing.assert_close(
        loss.new_trajectory_log_prob,
        rollout.old_trajectory_log_prob,
        rtol=0,
        atol=2e-5,
    )


def test_b_only_trajectory_head_updates() -> None:
    planner = _planner()
    with torch.no_grad():
        planner.diffusion_decoder.predecessor_residual_gate.fill_(0.25)
    trainer = JointGRPOTrainerB(
        planner, JointGRPOConfig(trajectories_per_mode=2)
    )
    rollout = trainer.sample_groups(
        _inputs(), generator=torch.Generator().manual_seed(23)
    )
    head_before = _state(planner.diffusion_decoder.trajectory_head)
    mode_before = _state(planner.mode_head)
    encoder = planner.diffusion_decoder.predecessor_action_encoder
    assert encoder is not None
    encoder_before = _state(encoder)
    gate_before = planner.diffusion_decoder.predecessor_residual_gate.detach().clone()
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
    assert not _same_state(head_before, _state(planner.diffusion_decoder.trajectory_head))
    assert _same_state(mode_before, _state(planner.mode_head))
    assert _same_state(encoder_before, _state(encoder))
    torch.testing.assert_close(
        planner.diffusion_decoder.predecessor_residual_gate, gate_before
    )


def test_b_source_metadata_and_schema3_strict_resume(tmp_path: Path) -> None:
    source = _source_metadata(diagnostic=True)
    validate_stage1_b_source_metadata(source, allow_diagnostic_source=True)
    with pytest.raises(JointGRPOError, match="explicit opt-in"):
        validate_stage1_b_source_metadata(source, allow_diagnostic_source=False)

    trainer = JointGRPOTrainerB(
        _planner(), JointGRPOConfig(trajectories_per_mode=2)
    )
    payload = grpo_b_checkpoint_payload(
        trainer=trainer,
        source_stage1_sha256="d" * 64,
        source_stage1_payload=source,
        metrics={"loss/total": 0.0},
        diagnostic_only=True,
    )
    assert payload["schema_version"] == GRPO_CHECKPOINT_SCHEMA_VERSION == 3
    assert payload["format"] == GRPO_B_CHECKPOINT_FORMAT
    path = save_grpo_checkpoint(tmp_path / "b.pt", payload)
    restored = JointGRPOTrainerB(
        _planner(), JointGRPOConfig(trajectories_per_mode=2)
    )
    load_grpo_b_checkpoint(
        path, restored, expected_source_stage1_sha256="d" * 64
    )
    assert _same_state(_state(restored.planner), _state(trainer.planner))
    assert _same_state(_state(restored.reference), _state(trainer.reference))

    with pytest.raises(JointGRPOError, match="trainer variant"):
        load_grpo_b_checkpoint(
            path,
            JointGRPOTrainerA(
                _planner("none"), JointGRPOConfig(trajectories_per_mode=2)
            ),  # type: ignore[arg-type]
            expected_source_stage1_sha256="d" * 64,
        )
