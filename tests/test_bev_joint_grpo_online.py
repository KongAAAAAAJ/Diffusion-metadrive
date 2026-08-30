from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.bev_planner.joint_grpo import (
    JointGRPOConfig,
    JointGRPOPolicyUpdateConfig,
    joint_grpo_optimizer_contract,
    joint_grpo_optimizer_contract_sha256,
    normalize_signed_advantages,
)
from models.bev_planner.joint_reward import (
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256,
    JOINT_REWARD_CONTRACT,
    JOINT_REWARD_CONTRACT_SHA256,
    JointRewardConfig,
    joint_reward_config_sha256,
)
from models.bev_planner.mode_contract import ModeIndex
from models.bev_planner.trajectory_optimizer import (
    KinematicTrajectoryOptimizer,
    KinematicTrajectoryOptimizerConfig,
    TrajectoryOptimizationError,
)
from models.decisioner.rule_decisioner import (
    JointActionProposal,
    RuleMakerProposalBatch,
)
from scenarios.bev_round13_contract import primary_scenario_contract
from scenarios.definitions import SCENARIO_BY_ID
from train.train_bev_joint_grpo_online import (
    AGENT_IDS,
    PRIMARY_S5_S9_SCENARIOS,
    ROLLOUT_COLLECTION_CONTRACT_VERSION,
    JointGRPOOnlineConfig,
    JointGRPOTrainingConfig,
    OnlineGRPOError,
    _OnlineRuleCondition,
    _balanced_bucket_targets,
    _bucket_visit_is_complete,
    _checkpoint_file_sha256,
    _config_from_yaml,
    _condition_online_model_inputs,
    _cuda_peak_memory_bytes,
    _finalize_online_rule_action,
    _fixed_raw_proxy_and_simulator_validation,
    _grpo_config_artifact_payload,
    _joint_rewards_are_informative,
    _load_trainer,
    _new_online_rule_maker,
    _next_run_directory,
    _next_unfinished_bucket_index,
    _resume_best_checkpoint_anchor,
    _reject_exhausted_uninformative_budget,
    _reset_cuda_peak_memory,
    _rollout_start_offset_upper_bound,
    _round_robin_training_buckets,
    _sample_rollout_start_offset,
    _sampler_state,
    _scenario_ready_for_primary_sampling,
    _scenario_sampling_window_closed_from_summary,
    _score_select_and_optimize_raw_candidates,
    _should_record_advantage_vector,
    _split_loss_metrics_by_step_axis,
    _validate_online_checkpoint_metadata,
    _validate_raw_reward_config,
    _validate_sampler_state,
    _validation_raw_proxy_reward,
    _validation_reward_comparison_metrics,
    _write_advantage_vector_summary,
    constant_velocity_actions,
    episode_has_ended,
    execution_mode_valid_mask,
    joint_trajectory_action,
    model_inputs_to_batch,
    main as online_main,
    optimize_selected_model_trajectories,
    rollout_collection_contract,
    run_joint_grpo_training,
)


RUN_YAML_HEADER = (
    "run:\n"
    "  variant: A\n"
    "  run_mode: smoke\n"
    "  source_checkpoint: /tmp/stage1.pt\n"
)


def test_next_run_directory_continues_after_migrated_run4(tmp_path: Path) -> None:
    output_root = tmp_path / "run_3" / "grpo_open"
    (output_root / "run_4").mkdir(parents=True)

    next_run = _next_run_directory(output_root)

    assert next_run == output_root / "run_5"
    assert (next_run / "checkpoints").is_dir()


def _binding() -> dict[str, object]:
    reward_config = JointRewardConfig()
    optimizer_config = KinematicTrajectoryOptimizerConfig()
    policy_update = JointGRPOPolicyUpdateConfig()
    online_config = JointGRPOOnlineConfig(device="cpu")
    generator_state = torch.Generator().manual_seed(17).get_state()
    return {
        "schema_version": 1,
        "format": "bev_joint_grpo_a_v1",
        "variant": "A",
        "predecessor_condition": "A",
        "source_stage1_sha256": "a" * 64,
        "grpo_config": dataclasses.asdict(JointGRPOConfig()),
        "optimizer_step": 4,
        "run_mode": "smoke",
        "diagnostic_only": True,
        "eligible_for_formal_training": False,
        "reward_contract_version": JOINT_REWARD_CONTRACT["version"],
        "reward_contract_sha256": JOINT_REWARD_CONTRACT_SHA256,
        "reward_config": dataclasses.asdict(reward_config),
        "reward_config_sha256": joint_reward_config_sha256(reward_config),
        "reward_application_contract": GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
        "reward_application_contract_sha256": (
            GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256
        ),
        "reward_input_domain": "tau_d",
        "candidate_selection_domain": "tau_d",
        "execution_input_domain": "tau_cmd",
        "best_checkpoint_metric": "validation/raw_proxy_reward_mean",
        "tracking_expansion_enabled": False,
        "calibration_required": False,
        "policy_update_contract": joint_grpo_optimizer_contract(policy_update),
        "policy_update_contract_sha256": (
            joint_grpo_optimizer_contract_sha256(policy_update)
        ),
        "rollout_collection_contract": rollout_collection_contract(
            online_config
        ),
        "scenario_contract_sha256": primary_scenario_contract()["sha256"],
        "scenario_seeds": [17, 23],
        "trajectory_optimizer_config": dataclasses.asdict(optimizer_config),
        "trajectory_optimizer_sha256": optimizer_config.sha256(),
        "environment_steps": 1,
        "sampler_state": {
            "sampled_rollouts": 1,
            "uninformative_rollouts": 0,
            "bucket_target_counts": [10] * 10,
            "bucket_sample_counts": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "bucket_optimizer_step_counts": [4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "bucket_episode_counts": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "next_bucket_index": 0,
            "current_visit_progress": 1,
            "generator_state": generator_state,
            "last_validated_rollout": 1,
        },
    }


def test_online_config_and_run_mode_are_strict(tmp_path: Path) -> None:
    with pytest.raises(OnlineGRPOError):
        JointGRPOOnlineConfig(device="auto")
    with pytest.raises(OnlineGRPOError):
        JointGRPOOnlineConfig(total_rollout_groups=0)
    assert JointGRPOOnlineConfig().group_size == 24
    assert JointGRPOOnlineConfig(group_size=3).group_size == 3
    assert JointGRPOOnlineConfig(group_size=5).group_size == 5
    assert JointGRPOOnlineConfig().update_epochs == 4
    assert JointGRPOOnlineConfig().clip_epsilon == pytest.approx(0.2)
    assert JointGRPOOnlineConfig().rollout_groups_per_bucket_visit == 10
    assert JointGRPOOnlineConfig().rollout_start_offset_max_steps == 200
    assert JointGRPOOnlineConfig().rollout_start_min_remaining_steps == 10
    assert (
        JointGRPOOnlineConfig(rollout_start_offset_max_steps=0)
        .rollout_start_offset_max_steps
        == 0
    )
    assert JointGRPOOnlineConfig().validation_interval_rollouts == 20
    assert JointGRPOOnlineConfig().advantage_vector_log_interval_rollouts == 20
    with pytest.raises(OnlineGRPOError, match="complete ordered S5--S9"):
        JointGRPOOnlineConfig(
            scenarios=(("S1_free_cruise_straight", "R3_mainline_straight"),)
        )
    online = JointGRPOOnlineConfig(device="cpu")
    training = JointGRPOTrainingConfig(
        variant="A",
        run_mode="smoke",
        source_checkpoint=tmp_path / "stage1.pt",
        online=online,
    )
    assert training.online is online
    with pytest.raises(OnlineGRPOError, match="training_config"):
        run_joint_grpo_training(
            online,  # type: ignore[arg-type]
            output_root=tmp_path / "output",
        )


def test_training_config_wrapper_is_strict(tmp_path: Path) -> None:
    online = JointGRPOOnlineConfig(device="cpu")
    with pytest.raises(OnlineGRPOError, match="variant"):
        JointGRPOTrainingConfig(
            variant="C",  # type: ignore[arg-type]
            run_mode="smoke",
            source_checkpoint=tmp_path / "stage1.pt",
            online=online,
        )
    with pytest.raises(OnlineGRPOError, match="run_mode"):
        JointGRPOTrainingConfig(
            variant="A",
            run_mode="preview",  # type: ignore[arg-type]
            source_checkpoint=tmp_path / "stage1.pt",
            online=online,
        )
    with pytest.raises(OnlineGRPOError, match="source_checkpoint"):
        JointGRPOTrainingConfig(
            variant="A",
            run_mode="smoke",
            source_checkpoint="stage1.pt",  # type: ignore[arg-type]
            online=online,
        )
    with pytest.raises(OnlineGRPOError, match="online"):
        JointGRPOTrainingConfig(
            variant="A",
            run_mode="smoke",
            source_checkpoint=tmp_path / "stage1.pt",
            online=object(),  # type: ignore[arg-type]
        )


def test_cuda_peak_memory_is_reset_and_reported_only_for_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_devices: list[torch.device] = []
    queried_devices: list[torch.device] = []
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device: reset_devices.append(device),
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        lambda device: queried_devices.append(device) or 123456,
    )

    cpu = torch.device("cpu")
    _reset_cuda_peak_memory(cpu)
    assert _cuda_peak_memory_bytes(cpu) is None
    assert reset_devices == []
    assert queried_devices == []

    cuda = torch.device("cuda")
    _reset_cuda_peak_memory(cuda)
    assert _cuda_peak_memory_bytes(cuda) == 123456
    assert reset_devices == [cuda]
    assert queried_devices == [cuda]


@pytest.mark.parametrize("invalid_group_size", [True, False, 1, 0, -1])
def test_online_config_rejects_invalid_group_size(
    invalid_group_size: object,
) -> None:
    with pytest.raises(OnlineGRPOError, match="group_size"):
        JointGRPOOnlineConfig(
            group_size=invalid_group_size  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_interval",
    [0, -1, True, False],
)
def test_online_config_rejects_invalid_advantage_vector_log_interval(
    invalid_interval: object,
) -> None:
    with pytest.raises(
        OnlineGRPOError, match="advantage_vector_log_interval_rollouts"
    ):
        JointGRPOOnlineConfig(
            advantage_vector_log_interval_rollouts=invalid_interval  # type: ignore[arg-type]
        )


def test_online_config_loads_clipped_rollout_contract_from_yaml(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "online.yaml"
    config_path.write_text(
        RUN_YAML_HEADER
        + "online:\n"
        "  device: cpu\n"
        "  group_size: 5\n"
        "  total_rollout_groups: 41\n"
        "  update_epochs: 3\n"
        "  clip_epsilon: 0.15\n"
        "  rollout_groups_per_bucket_visit: 7\n"
        "  rollout_start_offset_max_steps: 29\n"
        "  rollout_start_min_remaining_steps: 8\n"
        "  validation_interval_rollouts: 11\n"
        "  advantage_vector_log_interval_rollouts: 37\n",
        encoding="utf-8",
    )

    training = _config_from_yaml(config_path)
    config = training.online

    assert training.variant == "A"
    assert training.run_mode == "smoke"
    assert training.source_checkpoint == Path("/tmp/stage1.pt")
    assert config.group_size == 5
    assert config.total_rollout_groups == 41
    assert config.update_epochs == 3
    assert config.clip_epsilon == pytest.approx(0.15)
    assert config.rollout_groups_per_bucket_visit == 7
    assert config.rollout_start_offset_max_steps == 29
    assert config.rollout_start_min_remaining_steps == 8
    assert config.validation_interval_rollouts == 11
    assert config.advantage_vector_log_interval_rollouts == 37


def test_yaml_requires_run_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "missing-run.yaml"
    config_path.write_text("online:\n  device: cpu\n", encoding="utf-8")

    with pytest.raises(OnlineGRPOError, match="requires a run mapping"):
        _config_from_yaml(config_path)


def test_cli_rejects_legacy_max_rollout_groups(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_bev_joint_grpo_online",
            "--config",
            str(tmp_path / "config.yaml"),
            "--output-root",
            str(tmp_path / "output"),
            "--max-rollout-groups",
            "1",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        online_main()

    assert exc_info.value.code == 2
    assert "unrecognized arguments: --max-rollout-groups 1" in capsys.readouterr().err


@pytest.mark.parametrize(
    "run_yaml",
    [
        "run: {}\n",
        "run:\n  run_mode: smoke\n  source_checkpoint: /tmp/stage1.pt\n",
        "run:\n  variant: A\n  source_checkpoint: /tmp/stage1.pt\n",
        "run:\n  variant: A\n  run_mode: smoke\n",
        (
            "run:\n"
            "  variant: A\n"
            "  run_mode: smoke\n"
            "  source_checkpoint: /tmp/stage1.pt\n"
            "  total_rollout_groups: 1\n"
        ),
    ],
)
def test_yaml_run_mapping_requires_exact_three_fields(
    tmp_path: Path,
    run_yaml: str,
) -> None:
    config_path = tmp_path / "invalid-run.yaml"
    config_path.write_text(
        run_yaml + "online:\n  device: cpu\n", encoding="utf-8"
    )

    with pytest.raises(OnlineGRPOError, match="must contain exactly"):
        _config_from_yaml(config_path)


@pytest.mark.parametrize("yaml_value", ["true", "false", "1", "0", "-1"])
def test_online_config_rejects_invalid_yaml_group_size(
    tmp_path: Path,
    yaml_value: str,
) -> None:
    config_path = tmp_path / f"invalid_group_{yaml_value}.yaml"
    config_path.write_text(
        RUN_YAML_HEADER
        + f"online:\n  device: cpu\n  group_size: {yaml_value}\n",
        encoding="utf-8",
    )

    with pytest.raises(OnlineGRPOError, match="group_size"):
        _config_from_yaml(config_path)


@pytest.mark.parametrize("invalid_update_epochs", [True, False, 0, -1, 1.5])
def test_online_config_rejects_invalid_update_epochs(
    invalid_update_epochs: object,
) -> None:
    with pytest.raises(OnlineGRPOError, match="update_epochs"):
        JointGRPOOnlineConfig(
            update_epochs=invalid_update_epochs  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid_quota", [True, False, 0, -1, 1.5])
def test_online_config_rejects_invalid_bucket_visit_quota(
    invalid_quota: object,
) -> None:
    with pytest.raises(OnlineGRPOError, match="rollout_groups_per_bucket_visit"):
        JointGRPOOnlineConfig(
            rollout_groups_per_bucket_visit=invalid_quota  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid_max_offset", [True, False, -1, 1.5])
def test_online_config_rejects_invalid_rollout_start_max_offset(
    invalid_max_offset: object,
) -> None:
    with pytest.raises(OnlineGRPOError, match="rollout_start_offset_max_steps"):
        JointGRPOOnlineConfig(
            rollout_start_offset_max_steps=invalid_max_offset  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid_min_remaining", [True, False, -1, 1.5])
def test_online_config_rejects_invalid_rollout_start_min_remaining(
    invalid_min_remaining: object,
) -> None:
    with pytest.raises(
        OnlineGRPOError, match="rollout_start_min_remaining_steps"
    ):
        JointGRPOOnlineConfig(
            rollout_start_min_remaining_steps=invalid_min_remaining  # type: ignore[arg-type]
        )


def test_online_config_rejects_start_reserve_smaller_than_visit_or_episode(
) -> None:
    with pytest.raises(
        OnlineGRPOError, match="rollout_start_min_remaining_steps"
    ):
        JointGRPOOnlineConfig(
            rollout_groups_per_bucket_visit=1,
            rollout_start_min_remaining_steps=0,
        )
    with pytest.raises(OnlineGRPOError, match="rollout_groups_per_bucket_visit"):
        JointGRPOOnlineConfig(
            rollout_groups_per_bucket_visit=11,
            rollout_start_min_remaining_steps=10,
        )
    with pytest.raises(OnlineGRPOError, match="environment_steps_per_episode"):
        JointGRPOOnlineConfig(
            environment_steps_per_episode=9,
            rollout_groups_per_bucket_visit=3,
            rollout_start_min_remaining_steps=10,
        )


@pytest.mark.parametrize(
    "invalid_clip_epsilon", [True, False, 0.0, 1.0, -0.1, float("nan")]
)
def test_online_config_rejects_invalid_clip_epsilon(
    invalid_clip_epsilon: object,
) -> None:
    with pytest.raises(OnlineGRPOError, match="clip_epsilon"):
        JointGRPOOnlineConfig(
            clip_epsilon=invalid_clip_epsilon  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "legacy_field",
    [
        "total_optimizer_steps",
        "validation_interval_steps",
        "advantage_vector_log_interval_steps",
    ],
)
def test_online_config_rejects_legacy_optimizer_step_fields(
    tmp_path: Path,
    legacy_field: str,
) -> None:
    config_path = tmp_path / f"legacy_{legacy_field}.yaml"
    config_path.write_text(
        RUN_YAML_HEADER + f"online:\n  device: cpu\n  {legacy_field}: 1\n",
        encoding="utf-8",
    )

    with pytest.raises(OnlineGRPOError, match="legacy optimizer-step fields"):
        _config_from_yaml(config_path)


def test_online_loader_forwards_exact_core_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_config = JointGRPOConfig(group_size=5)
    expected = (object(), {"source": "stage1"}, "a" * 64)
    observed: dict[str, object] = {}

    def fake_loader(
        path: Path,
        *,
        device: torch.device,
        config: JointGRPOConfig,
        allow_diagnostic_source: bool,
    ) -> tuple[object, dict[str, str], str]:
        observed.update(
            {
                "path": path,
                "device": device,
                "config": config,
                "allow_diagnostic_source": allow_diagnostic_source,
            }
        )
        return expected

    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online.load_stage1_a_for_grpo",
        fake_loader,
    )
    source = tmp_path / "stage1.pt"

    result = _load_trainer(
        "A",
        source,
        torch.device("cpu"),
        grpo_config=core_config,
        allow_diagnostic_source=True,
    )

    assert result == expected
    assert observed == {
        "path": source,
        "device": torch.device("cpu"),
        "config": core_config,
        "allow_diagnostic_source": True,
    }
    assert observed["config"] is core_config


def test_grpo_artifact_payload_preserves_fairness_hyperparameters() -> None:
    config = JointGRPOConfig(group_size=24)

    payload = _grpo_config_artifact_payload(config)

    assert payload == dataclasses.asdict(config)
    assert payload["group_size"] == 24
    for name in (
        "learning_rate",
        "mode_pg_weight",
        "trajectory_pg_weight",
        "bc_weight",
        "reference_kl_weight",
    ):
        assert name in payload


def test_advantage_vector_recording_schedule_uses_rollout_interval_or_final() -> None:
    assert not _should_record_advantage_vector(
        rollout_group=1,
        target_rollout_groups=350,
        interval_rollouts=100,
    )
    assert _should_record_advantage_vector(
        rollout_group=100,
        target_rollout_groups=350,
        interval_rollouts=100,
    )
    assert _should_record_advantage_vector(
        rollout_group=350,
        target_rollout_groups=350,
        interval_rollouts=100,
    )


def test_loss_metrics_use_rollout_and_optimizer_axes_without_duplication() -> None:
    rollout_metrics, optimizer_metrics = _split_loss_metrics_by_step_axis(
        {
            "advantage/mean": 0.0,
            "advantage/std": 1.0,
            "loss/total": 2.0,
            "policy/mode_ratio_mean": 1.1,
        }
    )

    assert rollout_metrics == {"advantage/mean": 0.0, "advantage/std": 1.0}
    assert optimizer_metrics == {
        "loss/total": 2.0,
        "policy/mode_ratio_mean": 1.1,
    }


@pytest.mark.parametrize("group_size", [3, 4, 5])
def test_advantage_vector_summary_is_sparse_and_preserves_tensor_values(
    group_size: int,
) -> None:
    class RecordingWriter:
        def __init__(self) -> None:
            self.calls: list[tuple[str, torch.Tensor, int]] = []

        def add_tensor(
            self, tag: str, tensor: torch.Tensor, step: int
        ) -> None:
            self.calls.append((tag, tensor, step))

    writer = RecordingWriter()
    advantages = torch.linspace(
        -1.0,
        1.0,
        steps=group_size,
        dtype=torch.float32,
        requires_grad=True,
    ).unsqueeze(0)

    assert not _write_advantage_vector_summary(
        writer,
        advantages,
        rollout_group=99,
        target_rollout_groups=350,
        interval_rollouts=100,
        group_size=group_size,
    )
    assert writer.calls == []

    assert _write_advantage_vector_summary(
        writer,
        advantages,
        rollout_group=100,
        target_rollout_groups=350,
        interval_rollouts=100,
        group_size=group_size,
    )
    assert len(writer.calls) == 1
    tag, recorded, step = writer.calls[0]
    assert tag == "advantage/vector"
    assert step == 100
    assert recorded.shape == (1, group_size)
    assert recorded.dtype == torch.float32
    assert recorded.device.type == "cpu"
    assert not recorded.requires_grad
    torch.testing.assert_close(recorded, advantages.detach().cpu())

    assert _write_advantage_vector_summary(
        writer,
        advantages,
        rollout_group=350,
        target_rollout_groups=350,
        interval_rollouts=100,
        group_size=group_size,
    )
    assert [call[2] for call in writer.calls] == [100, 350]


def test_scheduled_uninformative_rollout_writes_diagnostic_advantage() -> None:
    class RecordingWriter:
        def __init__(self) -> None:
            self.calls: list[tuple[str, torch.Tensor, int]] = []

        def add_tensor(
            self, tag: str, tensor: torch.Tensor, step: int
        ) -> None:
            self.calls.append((tag, tensor.clone(), step))

    writer = RecordingWriter()
    rewards = torch.full((1, 24), -1.0, dtype=torch.float32)
    advantages = normalize_signed_advantages(rewards, group_size=24)

    assert _write_advantage_vector_summary(
        writer,  # type: ignore[arg-type]
        advantages,
        rollout_group=20,
        target_rollout_groups=100,
        interval_rollouts=20,
        group_size=24,
    )
    assert len(writer.calls) == 1
    tag, persisted, step = writer.calls[0]
    assert tag == "advantage/vector"
    assert step == 20
    torch.testing.assert_close(persisted, torch.zeros_like(persisted))


@pytest.mark.parametrize(
    "invalid_advantages",
    [
        torch.zeros(4, dtype=torch.float32),
        torch.zeros((2, 4), dtype=torch.float32),
        torch.zeros((1, 3), dtype=torch.float32),
        torch.zeros((1, 1), dtype=torch.float32),
        torch.zeros((1, 0), dtype=torch.float32),
        torch.zeros((1, 4), dtype=torch.float64),
        torch.zeros((1, 4), dtype=torch.int64),
        torch.tensor([[0.0, 1.0, float("nan"), -1.0]], dtype=torch.float32),
        torch.tensor([[0.0, 1.0, float("inf"), -1.0]], dtype=torch.float32),
    ],
)
def test_advantage_vector_summary_rejects_invalid_tensor_contract(
    invalid_advantages: torch.Tensor,
) -> None:
    class FailingWriter:
        def add_tensor(self, *args: object) -> None:
            raise AssertionError("invalid advantages must not reach writer")

    with pytest.raises(OnlineGRPOError):
        _write_advantage_vector_summary(
            FailingWriter(),
            invalid_advantages,
            rollout_group=100,
            target_rollout_groups=350,
            interval_rollouts=100,
            group_size=4,
        )


def test_sampler_state_round_trip_preserves_counters_and_generator_sequence() -> None:
    generator = torch.Generator().manual_seed(17)
    torch.randn((5,), generator=generator)
    state = _sampler_state(
        sampled_rollouts=3,
        uninformative_rollouts=1,
        bucket_target_counts=[4, 4],
        bucket_sample_counts=[2, 1],
        bucket_optimizer_step_counts=[4, 4],
        bucket_episode_counts=[1, 1],
        next_bucket_index=1,
        current_visit_progress=1,
        generator_state=generator.get_state(),
        last_validated_rollout=2,
        rollout_groups_per_bucket_visit=2,
        update_epochs=4,
        optimizer_step=8,
    )

    restored = _validate_sampler_state(
        state,
        bucket_count=2,
        expected_bucket_target_counts=[4, 4],
        rollout_groups_per_bucket_visit=2,
        update_epochs=4,
        optimizer_step=8,
    )
    resumed_generator = torch.Generator()
    resumed_generator.set_state(restored["generator_state"])

    assert restored["sampled_rollouts"] == 3
    assert restored["uninformative_rollouts"] == 1
    assert restored["next_bucket_index"] == 1
    assert restored["current_visit_progress"] == 1
    assert restored["last_validated_rollout"] == 2
    assert restored["bucket_sample_counts"] == [2, 1]
    assert restored["bucket_optimizer_step_counts"] == [4, 4]
    assert restored["bucket_target_counts"] == [4, 4]
    assert restored["bucket_episode_counts"] == [1, 1]
    torch.testing.assert_close(
        torch.randn((8,), generator=resumed_generator),
        torch.randn((8,), generator=generator),
    )


def test_exhausted_all_uninformative_budget_is_not_a_successful_run() -> None:
    _reject_exhausted_uninformative_budget(
        sampled_rollouts=3,
        target_rollout_groups=4,
        optimizer_step=0,
        run_start_optimizer_step=0,
    )
    _reject_exhausted_uninformative_budget(
        sampled_rollouts=4,
        target_rollout_groups=4,
        optimizer_step=4,
        run_start_optimizer_step=0,
    )
    with pytest.raises(OnlineGRPOError, match="every fresh group was uninformative"):
        _reject_exhausted_uninformative_budget(
            sampled_rollouts=4,
            target_rollout_groups=4,
            optimizer_step=8,
            run_start_optimizer_step=8,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("next_bucket_index", 0, "current visit progress"),
        ("current_visit_progress", 0, "current visit progress"),
        ("bucket_target_counts", [5, 3], "bucket targets"),
        ("bucket_sample_counts", [1, 1], "bucket samples"),
        ("bucket_optimizer_step_counts", [4, 0], "bucket updates"),
        ("bucket_episode_counts", [1, 0], "bucket episodes"),
        ("uninformative_rollouts", 2, "rollout and optimizer counters"),
        ("generator_state", torch.zeros(2), "generator_state"),
    ],
)
def test_sampler_state_rejects_counter_and_generator_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    generator = torch.Generator().manual_seed(17)
    state: dict[str, object] = {
        "sampled_rollouts": 3,
        "uninformative_rollouts": 1,
        "bucket_target_counts": [4, 4],
        "bucket_sample_counts": [2, 1],
        "bucket_optimizer_step_counts": [4, 4],
        "bucket_episode_counts": [1, 1],
        "next_bucket_index": 1,
        "current_visit_progress": 1,
        "generator_state": generator.get_state(),
        "last_validated_rollout": 2,
    }
    state[field] = value

    with pytest.raises(OnlineGRPOError, match=message):
        _validate_sampler_state(
            state,
            bucket_count=2,
            expected_bucket_target_counts=[4, 4],
            rollout_groups_per_bucket_visit=2,
            update_epochs=4,
            optimizer_step=8,
        )


def test_tracking_expansion_is_rejected_and_base_config_is_frozen() -> None:
    _validate_raw_reward_config(JointRewardConfig())
    with pytest.raises(OnlineGRPOError, match="zero tracking margins"):
        _validate_raw_reward_config(
            JointRewardConfig(tracking_lateral_margin_m=0.01)
        )
    with pytest.raises(OnlineGRPOError, match="frozen base config"):
        _validate_raw_reward_config(JointRewardConfig(gap_weight=1.0))


def test_raw_proxy_receives_exact_tau_d_and_only_argmax_is_optimized() -> None:
    raw = np.arange(4 * 3 * 8 * 3, dtype=np.float32).reshape(4, 3, 8, 3)
    modes = np.arange(12, dtype=np.int64).reshape(4, 3) % 10
    seen: dict[str, np.ndarray] = {}

    class Proxy:
        def score(self, env, model_inputs, trajectories):
            seen["proxy"] = np.array(trajectories, copy=True)
            return SimpleNamespace(
                rewards=np.asarray([-3.0, 8.0, 2.0, 1.0], dtype=np.float32),
                unsafe=np.zeros(4, dtype=np.bool_),
            )

    class Optimizer:
        def optimize(self, trajectories, coarse, speeds, selected_modes):
            seen["optimizer"] = np.array(trajectories, copy=True)
            seen["modes"] = np.array(selected_modes, copy=True)
            return SimpleNamespace(
                optimized_trajectories=np.asarray(trajectories) + 1000.0
            )

    values = SimpleNamespace(
        coarse_trajectories=np.zeros((3, 10, 8, 3), dtype=np.float32),
        ego_state=np.zeros((3, 8), dtype=np.float32),
    )
    proxy, selected, optimization = _score_select_and_optimize_raw_candidates(
        object(),
        values,
        raw,
        modes,
        proxy_backend=Proxy(),
        trajectory_optimizer=Optimizer(),
    )

    assert np.array_equal(seen["proxy"], raw)
    assert selected == 1
    assert np.array_equal(seen["optimizer"], raw[1:2])
    assert np.array_equal(seen["modes"], modes[1:2])
    assert np.array_equal(
        optimization.optimized_trajectories[0], raw[1] + 1000.0
    )
    assert proxy.rewards[selected] == 8.0


def test_optimizer_failure_has_no_fallback_candidate() -> None:
    raw = np.zeros((4, 3, 8, 3), dtype=np.float32)
    modes = np.zeros((4, 3), dtype=np.int64)
    calls = []

    class Proxy:
        def score(self, env, model_inputs, trajectories):
            return SimpleNamespace(
                rewards=np.asarray([0.0, 1.0, 9.0, 2.0], dtype=np.float32)
            )

    class FailingOptimizer:
        def optimize(self, trajectories, coarse, speeds, selected_modes):
            calls.append(np.array(trajectories, copy=True))
            raise TrajectoryOptimizationError("selected candidate failed")

    values = SimpleNamespace(
        coarse_trajectories=np.zeros((3, 10, 8, 3), dtype=np.float32),
        ego_state=np.zeros((3, 8), dtype=np.float32),
    )
    with pytest.raises(TrajectoryOptimizationError, match="selected candidate"):
        _score_select_and_optimize_raw_candidates(
            object(),
            values,
            raw,
            modes,
            proxy_backend=Proxy(),
            trajectory_optimizer=FailingOptimizer(),
        )
    assert len(calls) == 1
    assert np.array_equal(calls[0], raw[2:3])


def test_best_checkpoint_objective_uses_only_raw_proxy_reward() -> None:
    earlier = {
        "validation/raw_proxy_reward_mean": -8.0,
        "validation/simulator_reward_mean": 100.0,
        "validation/unsafe_count": 0.0,
    }
    better_raw_worse_simulator = {
        "validation/raw_proxy_reward_mean": -7.0,
        "validation/simulator_reward_mean": -100.0,
        "validation/unsafe_count": 10.0,
    }
    exact_tie = {
        "validation/raw_proxy_reward_mean": -7.0,
        "validation/simulator_reward_mean": 1000.0,
    }
    best = _validation_raw_proxy_reward(earlier)
    candidate = _validation_raw_proxy_reward(better_raw_worse_simulator)
    assert candidate > best
    best = candidate
    assert not _validation_raw_proxy_reward(exact_tie) > best


def test_resume_inherits_verified_historical_raw_best(tmp_path: Path) -> None:
    binding = _binding()
    best_payload = {
        **binding,
        "metrics": {"validation/raw_proxy_reward_mean": -7.0},
        "best_validation_reward": -7.0,
        "best_checkpoint_sha256": None,
    }
    best_path = tmp_path / "best.pt"
    torch.save(best_payload, best_path)
    last_payload = {
        **binding,
        "metrics": {"validation/raw_proxy_reward_mean": -8.0},
        "best_validation_reward": -7.0,
        "best_checkpoint_sha256": _checkpoint_file_sha256(best_path),
    }
    resolved, reward = _resume_best_checkpoint_anchor(
        tmp_path / "last.pt", last_payload
    )
    assert resolved == best_path
    assert reward == -7.0
    mismatched_group_size = {
        **last_payload,
        "grpo_config": dataclasses.asdict(JointGRPOConfig(group_size=5)),
    }
    with pytest.raises(OnlineGRPOError, match="contract binding mismatch"):
        _resume_best_checkpoint_anchor(
            tmp_path / "last.pt", mismatched_group_size
        )
    mismatched_policy = {
        **last_payload,
        "policy_update_contract": joint_grpo_optimizer_contract(
            JointGRPOPolicyUpdateConfig(update_epochs=2)
        ),
        "policy_update_contract_sha256": (
            joint_grpo_optimizer_contract_sha256(
                JointGRPOPolicyUpdateConfig(update_epochs=2)
            )
        ),
    }
    with pytest.raises(OnlineGRPOError, match="contract binding mismatch"):
        _resume_best_checkpoint_anchor(tmp_path / "last.pt", mismatched_policy)
    mismatched_collection = {
        **last_payload,
        "rollout_collection_contract": {
            **dict(last_payload["rollout_collection_contract"]),
            "rollout_groups_per_bucket_visit": 5,
        },
    }
    with pytest.raises(OnlineGRPOError, match="contract binding mismatch"):
        _resume_best_checkpoint_anchor(
            tmp_path / "last.pt", mismatched_collection
        )
    last_payload["best_checkpoint_sha256"] = "0" * 64
    with pytest.raises(OnlineGRPOError, match="SHA256 mismatch"):
        _resume_best_checkpoint_anchor(tmp_path / "last.pt", last_payload)


def test_checkpoint_metadata_rejects_legacy_and_domain_drift() -> None:
    online_config = JointGRPOOnlineConfig(device="cpu")
    collection = rollout_collection_contract(online_config)
    bucket_targets = [10] * 10
    payload = {
        **_binding(),
        "best_validation_reward": -7.0,
        "best_checkpoint_sha256": None,
    }
    _validate_online_checkpoint_metadata(
        payload,
        run_mode="smoke",
        reward_config=JointRewardConfig(),
        scenario_contract_sha=primary_scenario_contract()["sha256"],
        scenario_seeds=(17, 23),
        policy_update=JointGRPOPolicyUpdateConfig(),
        collection_contract=collection,
        bucket_count=10,
        bucket_target_counts=bucket_targets,
        rollout_groups_per_bucket_visit=10,
        optimizer_step=4,
    )
    pre_persistent = _binding()
    pre_persistent.pop("rollout_collection_contract")
    with pytest.raises(
        OnlineGRPOError, match="rollout_collection_contract mismatch"
    ):
        _validate_online_checkpoint_metadata(
            pre_persistent,
            run_mode="smoke",
            reward_config=JointRewardConfig(),
            scenario_contract_sha=primary_scenario_contract()["sha256"],
            scenario_seeds=(17, 23),
            policy_update=JointGRPOPolicyUpdateConfig(),
            collection_contract=collection,
            bucket_count=10,
            bucket_target_counts=bucket_targets,
            rollout_groups_per_bucket_visit=10,
            optimizer_step=4,
        )
    legacy_v1 = {
        **_binding(),
        "rollout_collection_contract": {
            **collection,
            "version": "stage2_joint_grpo_persistent_episode_v1",
        },
    }
    with pytest.raises(
        OnlineGRPOError, match="rollout_collection_contract mismatch"
    ):
        _validate_online_checkpoint_metadata(
            legacy_v1,
            run_mode="smoke",
            reward_config=JointRewardConfig(),
            scenario_contract_sha=primary_scenario_contract()["sha256"],
            scenario_seeds=(17, 23),
            policy_update=JointGRPOPolicyUpdateConfig(),
            collection_contract=collection,
            bucket_count=10,
            bucket_target_counts=bucket_targets,
            rollout_groups_per_bucket_visit=10,
            optimizer_step=4,
        )
    payload["reward_input_domain"] = "tau_cmd"
    with pytest.raises(OnlineGRPOError, match="reward_input_domain mismatch"):
        _validate_online_checkpoint_metadata(
            payload,
            run_mode="smoke",
            reward_config=JointRewardConfig(),
            scenario_contract_sha=primary_scenario_contract()["sha256"],
            scenario_seeds=(17, 23),
            policy_update=JointGRPOPolicyUpdateConfig(),
            collection_contract=collection,
            bucket_count=10,
            bucket_target_counts=bucket_targets,
            rollout_groups_per_bucket_visit=10,
            optimizer_step=4,
        )
    legacy_policy = _binding()
    legacy_policy.pop("policy_update_contract")
    legacy_policy.pop("policy_update_contract_sha256")
    with pytest.raises(OnlineGRPOError, match="policy_update_contract"):
        _validate_online_checkpoint_metadata(
            legacy_policy,
            run_mode="smoke",
            reward_config=JointRewardConfig(),
            scenario_contract_sha=primary_scenario_contract()["sha256"],
            scenario_seeds=(17, 23),
            policy_update=JointGRPOPolicyUpdateConfig(),
            collection_contract=collection,
            bucket_count=10,
            bucket_target_counts=bucket_targets,
            rollout_groups_per_bucket_visit=10,
            optimizer_step=4,
        )
    with pytest.raises(OnlineGRPOError, match="policy_update_contract"):
        _validate_online_checkpoint_metadata(
            _binding(),
            run_mode="smoke",
            reward_config=JointRewardConfig(),
            scenario_contract_sha=primary_scenario_contract()["sha256"],
            scenario_seeds=(17, 23),
            policy_update=JointGRPOPolicyUpdateConfig(update_epochs=2),
            collection_contract=collection,
            bucket_count=10,
            bucket_target_counts=bucket_targets,
            rollout_groups_per_bucket_visit=10,
            optimizer_step=4,
        )
    with pytest.raises(OnlineGRPOError, match="policy_update_contract"):
        _validate_online_checkpoint_metadata(
            _binding(),
            run_mode="smoke",
            reward_config=JointRewardConfig(),
            scenario_contract_sha=primary_scenario_contract()["sha256"],
            scenario_seeds=(17, 23),
            policy_update=JointGRPOPolicyUpdateConfig(clip_epsilon=0.1),
            collection_contract=collection,
            bucket_count=10,
            bucket_target_counts=bucket_targets,
            rollout_groups_per_bucket_visit=10,
            optimizer_step=4,
        )
    payload = {**_binding(), "calibration_report_sha256": "b" * 64}
    with pytest.raises(OnlineGRPOError, match="legacy calibration semantics"):
        _validate_online_checkpoint_metadata(
            payload,
            run_mode="smoke",
            reward_config=JointRewardConfig(),
            scenario_contract_sha=primary_scenario_contract()["sha256"],
            scenario_seeds=(17, 23),
            policy_update=JointGRPOPolicyUpdateConfig(),
            collection_contract=collection,
            bucket_count=10,
            bucket_target_counts=bucket_targets,
            rollout_groups_per_bucket_visit=10,
            optimizer_step=4,
        )


def test_validation_reward_comparison_uses_raw_tag() -> None:
    result = _validation_reward_comparison_metrics(
        {"validation/raw_proxy_reward_mean": -8.0}, -10.5
    )
    assert result == {
        "validation/pretrain_reward": -10.5,
        "validation/reward_gain": 2.5,
    }
    with pytest.raises(OnlineGRPOError, match="raw_proxy_reward_mean"):
        _validation_reward_comparison_metrics(
            {"validation/simulator_reward_mean": -8.0}, -10.5
        )


def test_simulator_failure_is_non_gating_for_raw_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = SimpleNamespace(
        coarse_trajectories=np.zeros((3, 10, 8, 3), dtype=np.float32),
        ego_state=np.zeros((3, 8), dtype=np.float32),
    )

    class Env:
        def __init__(self):
            self.step_calls = 0

        def step(self, action):
            self.step_calls += 1
            return {}, {}, {"__all__": False}, {"__all__": False}, {}

        def close(self):
            return None

    env = Env()

    class Builder:
        def __init__(self, agent_ids):
            self.captures = 0

        def reset(self):
            self.captures = 0

        def capture_state(self, env, timestamp):
            self.captures += 1

        def history_ready(self):
            return self.captures >= 3

        def build_model_inputs(self, env):
            return values

    class Proxy:
        def __init__(self, config):
            pass

        def score(self, env, model_inputs, trajectories):
            assert env.step_calls == 2
            return SimpleNamespace(
                rewards=np.asarray([3.5], dtype=np.float32),
                unsafe=np.asarray([False]),
                collision=np.asarray([False]),
                out_of_drivable=np.asarray([False]),
            )

    class Optimizer:
        def optimize(self, trajectories, coarse, speeds, modes):
            return SimpleNamespace(optimized_trajectories=trajectories + 1.0)

    class Evaluator:
        def __init__(self, config):
            pass

        def evaluate(self, spec, prefix, candidates):
            raise RuntimeError("diagnostic backend unavailable")

    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online._new_env", lambda *a: env
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online.JointBEVSampleBuilder", Builder
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online.simulator_decision_dt_s", lambda env: 0.1
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online._scenario_ready_for_primary_sampling",
        lambda env: True,
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online.route_following_warmup_actions",
        lambda env, builder: {
            agent_id: np.zeros((8, 3), dtype=np.float32)
            for agent_id in AGENT_IDS
        },
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online._sample_rollout_start_offset",
        lambda *args, **kwargs: pytest.fail(
            "fixed validation must not sample a rollout start offset"
        ),
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online.execution_mode_valid_mask",
        lambda values, optimizer: np.ones((3, 10), dtype=np.bool_),
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online.model_inputs_to_batch",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online.planner_forward_from_batch",
        lambda *args, **kwargs: {
            "selected_trajectory": torch.zeros((1, 3, 8, 3)),
            "selected_mode": torch.zeros((1, 3), dtype=torch.int64),
        },
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online.JointTrajectoryProxyReward", Proxy
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online.KinematicTrajectoryOptimizer",
        Optimizer,
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online.JointSimulatorBranchEvaluator",
        Evaluator,
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online.capture_joint_pose_global",
        lambda env: np.zeros((3, 3), dtype=np.float32),
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online._new_online_rule_maker",
        lambda planner, env: object(),
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online._condition_online_model_inputs",
        lambda rule_maker, env, builder, values, **kwargs: _OnlineRuleCondition(
            model_inputs=values,
            proposal_batch=object(),
            rule_actions={agent_id: 0 for agent_id in AGENT_IDS},
            is_commitment=False,
            committed_execution_id=None,
            committed_plan_actions=None,
        ),
    )

    metrics, errors = _fixed_raw_proxy_and_simulator_validation(
        SimpleNamespace(config=SimpleNamespace(model_version="v2")),
        device=torch.device("cpu"),
        reward_config=JointRewardConfig(),
        scenarios=(PRIMARY_S5_S9_SCENARIOS[0],),
        seeds=(31,),
    )
    assert metrics["validation/raw_proxy_reward_mean"] == pytest.approx(3.5)
    assert metrics["validation/simulator_available"] == 0.0
    assert metrics["validation/simulator_failure_count"] == 1.0
    assert "validation/simulator_reward_mean" not in metrics
    assert errors[0]["error_type"] == "RuntimeError"
    assert env.step_calls == 2


def test_pretrain_raw_baseline_runs_once_before_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_planner = object()
    trainer = SimpleNamespace(planner=source_planner)
    validation_planners = []
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online._load_trainer",
        lambda *args, **kwargs: (trainer, {}, "a" * 64),
    )

    def fixed_validation(planner, **kwargs):
        validation_planners.append(planner)
        return ({"validation/raw_proxy_reward_mean": -10.0}, ())

    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online._fixed_raw_proxy_and_simulator_validation",
        fixed_validation,
    )

    def resume_loader(*args, **kwargs):
        assert validation_planners == [source_planner]
        raise OnlineGRPOError("resume loader reached")

    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online.load_grpo_checkpoint", resume_loader
    )
    training_config = JointGRPOTrainingConfig(
        variant="A",
        run_mode="smoke",
        source_checkpoint=tmp_path / "stage1.pt",
        online=JointGRPOOnlineConfig(
            device="cpu",
            total_rollout_groups=1,
            resume_checkpoint=tmp_path / "resume.pt",
        ),
    )
    with pytest.raises(OnlineGRPOError, match="resume loader reached"):
        run_joint_grpo_training(
            training_config,
            output_root=tmp_path / "output",
        )
    assert validation_planners == [source_planner]


def test_constant_rewards_and_training_buckets_are_strict() -> None:
    assert not _joint_rewards_are_informative(
        np.asarray([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
        group_size=4,
    )
    assert _joint_rewards_are_informative(
        np.asarray([-1.0, 0.0, -1.0, -1.0], dtype=np.float32),
        group_size=4,
    )
    buckets = _round_robin_training_buckets(PRIMARY_S5_S9_SCENARIOS, (17, 23))
    assert len(buckets) == len(set(buckets)) == 10


def test_persistent_collection_contract_and_balanced_bucket_targets() -> None:
    config = JointGRPOOnlineConfig(
        device="cpu",
        environment_steps_per_episode=200,
        rollout_groups_per_bucket_visit=7,
        validation_interval_rollouts=13,
    )
    contract = rollout_collection_contract(config)

    assert ROLLOUT_COLLECTION_CONTRACT_VERSION == (
        "stage2_joint_grpo_persistent_episode_v2"
    )
    assert contract["version"] == ROLLOUT_COLLECTION_CONTRACT_VERSION
    assert contract["rollout_groups_per_bucket_visit"] == 7
    assert contract["environment_steps_per_episode"] == 200
    assert contract["rollout_start_offset_max_steps"] == 200
    assert contract["rollout_start_min_remaining_steps"] == 10
    assert contract["rollout_start_offset_distribution"] == (
        "inclusive_uniform_integer"
    )
    assert contract["rollout_start_generator"] == "shared_training_torch_generator"
    assert contract["validation_start"] == "earliest_ready_without_random_offset"
    assert contract["validation_interval_rollouts"] == 13
    assert contract["checkpoint_boundary"] == "closed_environment_only"
    assert _balanced_bucket_targets(100, 10) == [10] * 10
    assert _balanced_bucket_targets(103, 10) == [11, 11, 11] + [10] * 7
    assert _balanced_bucket_targets(3, 5) == [1, 1, 1, 0, 0]


def test_rollout_start_offset_upper_bound_is_inclusive_and_reserves_ten_steps(
) -> None:
    config = JointGRPOOnlineConfig(
        device="cpu",
        environment_steps_per_episode=200,
        rollout_start_offset_max_steps=200,
        rollout_start_min_remaining_steps=10,
    )

    assert _rollout_start_offset_upper_bound(config, 0) == 190
    assert _rollout_start_offset_upper_bound(config, 170) == 20
    assert _rollout_start_offset_upper_bound(config, 170, 7) == 7
    assert _rollout_start_offset_upper_bound(config, 190) == 0
    with pytest.raises(OnlineGRPOError, match="remaining"):
        _rollout_start_offset_upper_bound(config, 191)


def test_zero_upper_bound_still_consumes_exactly_one_generator_draw() -> None:
    config = JointGRPOOnlineConfig(
        device="cpu", environment_steps_per_episode=200
    )
    generator = torch.Generator().manual_seed(17)
    before = generator.get_state().clone()

    offset, upper = _sample_rollout_start_offset(config, 190, generator)

    assert (offset, upper) == (0, 0)
    assert not torch.equal(before, generator.get_state())
    reference = torch.Generator().manual_seed(17)
    torch.randint(0, 1, (1,), generator=reference)
    torch.testing.assert_close(
        torch.randn((8,), generator=generator),
        torch.randn((8,), generator=reference),
    )


def test_rollout_start_offset_and_following_noise_resume_exactly() -> None:
    config = JointGRPOOnlineConfig(
        device="cpu",
        environment_steps_per_episode=200,
        rollout_start_offset_max_steps=17,
    )
    uninterrupted = torch.Generator().manual_seed(23)
    first = _sample_rollout_start_offset(config, 20, uninterrupted)
    state_at_checkpoint = uninterrupted.get_state()
    expected_next = _sample_rollout_start_offset(config, 20, uninterrupted)
    expected_noise = torch.randn((12,), generator=uninterrupted)

    resumed = torch.Generator()
    resumed.set_state(state_at_checkpoint)
    assert _sample_rollout_start_offset(config, 20, resumed) == expected_next
    torch.testing.assert_close(
        torch.randn((12,), generator=resumed), expected_noise
    )

    same_seed = torch.Generator().manual_seed(23)
    assert _sample_rollout_start_offset(config, 20, same_seed) == first


@pytest.mark.parametrize(
    ("scenario_id", "conflict_evidence", "route_completion"),
    [
        (
            "S5_hard_brake_lead",
            {"formation_recovered_after_hazard": True},
            {},
        ),
        (
            "S6_background_merge_in",
            {"formation_recovered_after_merge": True},
            {},
        ),
        (
            "S7_ego_merge_from_ramp",
            {"formation_recovered_after_merge": True},
            {"all_agents_entered_mainline": True},
        ),
        (
            "S8_ego_exit_to_ramp",
            {"formation_recovered_on_ramp": True},
            {"all_agents_continued_on_exit_ramp": True},
        ),
        (
            "S9_narrow_channel_negotiation",
            {"formation_recovered_after_return": True},
            {"all_agents_returned_to_original_lane": True},
        ),
    ],
)
def test_scenario_sampling_window_closes_only_at_frozen_completion_predicate(
    scenario_id: str,
    conflict_evidence: dict[str, bool],
    route_completion: dict[str, bool],
) -> None:
    closed = {
        "scenario_id": scenario_id,
        "conflict_evidence": conflict_evidence,
        "route_completion": route_completion,
    }
    assert _scenario_sampling_window_closed_from_summary(closed)

    for section_name in ("conflict_evidence", "route_completion"):
        for field_name in closed[section_name]:
            still_open = {
                "scenario_id": scenario_id,
                "conflict_evidence": dict(conflict_evidence),
                "route_completion": dict(route_completion),
            }
            still_open[section_name][field_name] = False
            assert not _scenario_sampling_window_closed_from_summary(still_open)


def test_bucket_visit_progress_keeps_or_advances_the_fair_cursor() -> None:
    targets = [5, 4, 4]
    samples = [2, 0, 0]
    assert not _bucket_visit_is_complete(
        bucket_index=0,
        bucket_sample_counts=samples,
        bucket_target_counts=targets,
        current_visit_progress=2,
        rollout_groups_per_bucket_visit=3,
    )
    assert _next_unfinished_bucket_index(
        samples, targets, start_index=0
    ) == 0

    samples[0] = 3
    assert _bucket_visit_is_complete(
        bucket_index=0,
        bucket_sample_counts=samples,
        bucket_target_counts=targets,
        current_visit_progress=3,
        rollout_groups_per_bucket_visit=3,
    )
    assert _next_unfinished_bucket_index(
        samples, targets, start_index=1
    ) == 1

    assert _next_unfinished_bucket_index(
        targets, targets, start_index=2
    ) is None


def _run_fake_persistent_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    validation_interval_rollouts: int,
    terminal_after_environment_steps: int | None,
    max_rollout_groups: int = 21,
    rollout_groups_per_bucket_visit: int = 3,
    rollout_start_offset_max_steps: int = 0,
    force_start_offset_to_upper: bool = False,
    window_close_steps_by_episode: tuple[int | None, ...] = (),
    single_training_bucket: bool = False,
    first_rollout_informative: bool = False,
    environment_steps_per_episode: int = 50,
    history_ready_step: int = 1,
    external_envs: list[object] | None = None,
) -> tuple[dict[str, object], list[object], list[int], object]:
    group_size = 3
    update_epochs = 2
    envs: list[object] = [] if external_envs is None else external_envs
    validation_rollouts: list[int] = []

    class Writer:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def add_scalar(self, *args, **kwargs) -> None:
            pass

        def add_tensor(self, *args, **kwargs) -> None:
            pass

        def add_text(self, *args, **kwargs) -> None:
            pass

        def close(self) -> None:
            pass

    class Env:
        def __init__(self, scenario, seed, episode_index) -> None:
            self.scenario = scenario
            self.seed = seed
            self.episode_index = episode_index
            self.step_calls = 0
            self.summary_calls = 0
            self.closed = False
            self._last_planner_batch = {}
            self._scenario_orchestrator = SimpleNamespace(
                get_episode_summary=self.get_episode_summary
            )

        def get_episode_summary(self):
            self.summary_calls += 1
            close_step = (
                window_close_steps_by_episode[self.episode_index]
                if self.episode_index < len(window_close_steps_by_episode)
                else None
            )
            closed = close_step is not None and self.step_calls >= close_step
            return {
                "scenario_id": self.scenario[0],
                "scenario_realized": True,
                "scenario_triggered": True,
                "scenario_recipes_complete": True,
                "scenario_notes": ["lead_brake_profile"],
                "conflict_evidence": {
                    "formation_recovered_after_hazard": closed,
                    "formation_recovered_after_merge": closed,
                    "formation_recovered_on_ramp": closed,
                    "formation_recovered_after_return": closed,
                },
                "route_completion": {
                    "all_agents_entered_mainline": closed,
                    "all_agents_continued_on_exit_ramp": closed,
                    "all_agents_returned_to_original_lane": closed,
                },
            }

        def step(self, action):
            self.step_calls += 1
            ended = (
                terminal_after_environment_steps is not None
                and self.step_calls >= terminal_after_environment_steps
            )
            return (
                {},
                {},
                {"__all__": ended},
                {"__all__": False},
                {},
            )

        def close(self) -> None:
            self.closed = True

    class Builder:
        def __init__(self, agent_ids) -> None:
            self.captures = 0

        def reset(self) -> None:
            self.captures = 0

        def capture_state(self, env, timestamp) -> None:
            self.captures += 1

        def history_ready(self) -> bool:
            return self.captures >= history_ready_step + 1

        def build_model_inputs(self, env):
            fields = {
                "mode_valid_mask": np.ones((3, 10), dtype=np.bool_),
                "coarse_trajectories": np.zeros(
                    (3, 10, 8, 3), dtype=np.float32
                ),
                "ego_state": np.zeros((3, 8), dtype=np.float32),
            }
            return SimpleNamespace(**fields, as_dict=lambda: fields)

        def augment_v2_model_inputs(self, env, values, **kwargs):
            return values

    proposal = JointActionProposal(
        proposal_id=1,
        rank=0,
        rule_score=1.0,
        decisions={agent_id: {"action": 0} for agent_id in AGENT_IDS},
    )

    class RuleMaker:
        is_formation_locked = False
        has_active_lane_change_commitments = False

        def propose_joint_actions(
            self, env, agent_ids, planner_batch, *, hard_valid_modes_by_action
        ):
            return RuleMakerProposalBatch(batch_id=1, proposals=(proposal,))

    class ScalarLoss:
        def scalar_metrics(self):
            return {
                "loss/total": 1.0,
                "advantage/mean": 0.0,
            }

    class Trainer:
        def __init__(self) -> None:
            self.config = JointGRPOConfig(group_size=group_size)
            self.planner = object()
            self.optimizer_step = 0
            self.sample_calls = 0
            self.update_calls = 0
            self.policy_updates: list[JointGRPOPolicyUpdateConfig] = []
            self.sample_environment_steps: list[tuple[int, int]] = []
            self.start_offset_calls: list[
                tuple[int, int | None, int, int]
            ] = []
            self.warmup_calls_by_episode: list[int] = []

        def sample_groups(self, batch, *, generator):
            torch.randn((1,), generator=generator)
            self.sample_calls += 1
            active_env = envs[-1]
            self.sample_environment_steps.append(
                (active_env.episode_index, active_env.step_calls)
            )
            return SimpleNamespace(
                selected_trajectories=torch.zeros(
                    (1, group_size, 3, 8, 3), dtype=torch.float32
                ),
                sampled_modes=torch.full(
                    (1, group_size, 3),
                    int(ModeIndex.KEEP_HIGH),
                    dtype=torch.int64,
                ),
            )

        def update(self, rollout, rewards, *, policy_update):
            self.update_calls += 1
            self.policy_updates.append(policy_update)
            advantages = normalize_signed_advantages(
                rewards,
                group_size=group_size,
                eps=self.config.advantage_eps,
            ).detach()
            epochs = []
            for epoch_index in range(policy_update.update_epochs):
                self.optimizer_step += 1
                epochs.append(
                    SimpleNamespace(
                        optimizer_step=self.optimizer_step,
                        epoch_in_rollout=epoch_index + 1,
                        total_gradient_norm=1.0,
                        gradient_norms={},
                        loss=ScalarLoss(),
                    )
                )
            return SimpleNamespace(
                epoch_results=tuple(epochs),
                loss=SimpleNamespace(advantages=advantages),
            )

    trainer = Trainer()

    def new_env(scenario, seed):
        env = Env(scenario, seed, len(envs))
        envs.append(env)
        trainer.warmup_calls_by_episode.append(0)
        return env

    def sample_start_offset(
        config,
        ready_step,
        generator,
        temporary_max_offset_steps=None,
    ):
        sampled_offset, upper = _sample_rollout_start_offset(
            config,
            ready_step,
            generator,
            temporary_max_offset_steps,
        )
        offset = upper if force_start_offset_to_upper else sampled_offset
        trainer.start_offset_calls.append(
            (ready_step, temporary_max_offset_steps, offset, upper)
        )
        return offset, upper

    def warmup_actions(env, builder):
        trainer.warmup_calls_by_episode[env.episode_index] += 1
        return {
            agent_id: np.zeros((8, 3), dtype=np.float32)
            for agent_id in AGENT_IDS
        }

    def fixed_validation(planner, **kwargs):
        validation_rollouts.append(trainer.sample_calls)
        return (
            {
                "validation/raw_proxy_reward_mean": float(
                    trainer.sample_calls
                )
            },
            (),
        )

    def score_candidates(
        env,
        values,
        raw_candidates,
        sampled_modes,
        **kwargs,
    ):
        rewards = (
            np.zeros(group_size, dtype=np.float32)
            if trainer.sample_calls == 1 and not first_rollout_informative
            else np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
        )
        proxy = SimpleNamespace(
            rewards=rewards,
            unsafe=np.zeros(group_size, dtype=np.bool_),
        )
        optimization = SimpleNamespace(
            optimized_trajectories=np.zeros(
                (1, 3, 8, 3), dtype=np.float32
            ),
            elapsed_ms=0.1,
            intervention_ade_m=np.zeros((1, 3), dtype=np.float32),
            intervention_fde_m=np.zeros((1, 3), dtype=np.float32),
            raw_valid=np.ones((1, 3), dtype=np.bool_),
            retained_raw_fraction=np.ones((1, 3), dtype=np.float32),
            optimized_valid=np.ones((1, 3), dtype=np.bool_),
        )
        return proxy, 0, optimization

    def finalize(rule_maker, condition, **kwargs):
        return (
            kwargs["optimization"],
            {
                "conditioned_rollouts": 1,
                "proposal_match_attempts": 1,
                "proposal_matches": 1,
                "condition_failures": 0,
                "forced_safe_stops": 0,
                "s7_feedback_exception_hits": 0,
                "commitment_conditioned_rollouts": 0,
                "commitment_feedback_incompatible": 0,
            },
            condition.committed_execution_id,
            condition.committed_plan_actions,
        )

    def checkpoint_payload(**kwargs):
        return {
            "metrics": dict(kwargs["metrics"]),
            "optimizer_step": trainer.optimizer_step,
            "environment_steps": kwargs["environment_steps"],
            "best_validation_reward": kwargs["best_validation_reward"],
            "best_checkpoint_sha256": kwargs["best_checkpoint_sha256"],
            "rollout_collection_contract": dict(
                kwargs["collection_contract"]
            ),
            "sampler_state": dict(kwargs["sampler_state"]),
        }

    def checkpoint_loader(path, restored, **kwargs):
        return torch.load(path, map_location="cpu", weights_only=False)

    def validate_checkpoint(payload, **kwargs):
        assert payload["rollout_collection_contract"] == dict(
            kwargs["collection_contract"]
        )
        return _validate_sampler_state(
            payload["sampler_state"],
            bucket_count=kwargs["bucket_count"],
            expected_bucket_target_counts=kwargs["bucket_target_counts"],
            rollout_groups_per_bucket_visit=kwargs[
                "rollout_groups_per_bucket_visit"
            ],
            update_epochs=kwargs["policy_update"].update_epochs,
            optimizer_step=kwargs["optimizer_step"],
        )

    plot_names = (
        "advantage_heatmap",
        "reward_curve",
        "validation_reward_curve",
        "grpo_loss_curve",
        "kl_loss_curve",
        "policy_stability_curve",
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online.SummaryWriter", Writer
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online._load_trainer",
        lambda *args, **kwargs: (trainer, {}, "a" * 64),
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online._fixed_raw_proxy_and_simulator_validation",
        fixed_validation,
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online._new_env", new_env
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online.JointBEVSampleBuilder", Builder
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online._new_online_rule_maker",
        lambda planner, env: RuleMaker(),
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online.simulator_decision_dt_s",
        lambda env: 0.1,
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online._sample_rollout_start_offset",
        sample_start_offset,
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online.route_following_warmup_actions",
        warmup_actions,
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online.execution_mode_valid_mask",
        lambda values, optimizer: np.ones((3, 10), dtype=np.bool_),
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online.model_inputs_to_batch",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online._score_select_and_optimize_raw_candidates",
        score_candidates,
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online._finalize_online_rule_action",
        finalize,
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online._checkpoint_payload",
        checkpoint_payload,
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online.load_grpo_checkpoint",
        checkpoint_loader,
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online._validate_online_checkpoint_metadata",
        validate_checkpoint,
    )
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online.generate_grpo_plots",
        lambda tb, plots: {
            name: tmp_path / f"{name}.png" for name in plot_names
        },
    )
    if single_training_bucket:
        monkeypatch.setattr(
            "train.train_bev_joint_grpo_online._round_robin_training_buckets",
            lambda scenarios, seeds: ((scenarios[0], seeds[0]),),
        )

    training_config = JointGRPOTrainingConfig(
        variant="A",
        run_mode="smoke",
        source_checkpoint=tmp_path / "stage1.pt",
        online=JointGRPOOnlineConfig(
            device="cpu",
            group_size=group_size,
            total_rollout_groups=max_rollout_groups,
            update_epochs=update_epochs,
            environment_steps_per_episode=environment_steps_per_episode,
            rollout_groups_per_bucket_visit=rollout_groups_per_bucket_visit,
            rollout_start_offset_max_steps=rollout_start_offset_max_steps,
            validation_interval_rollouts=validation_interval_rollouts,
            advantage_vector_log_interval_rollouts=max_rollout_groups,
        ),
    )
    report = run_joint_grpo_training(
        training_config,
        output_root=tmp_path / "output",
    )
    return report, envs, validation_rollouts, trainer


def test_main_loop_reuses_live_env_for_three_groups_and_steps_uninformative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, envs, validation_rollouts, trainer = (
        _run_fake_persistent_collection(
            tmp_path,
            monkeypatch,
            validation_interval_rollouts=21,
            terminal_after_environment_steps=None,
        )
    )

    assert report["format"] == "bev_joint_grpo_online_report_v6"
    assert report["sampled_rollouts"] == 21
    assert report["uninformative_rollouts"] == 1
    assert report["optimizer_steps"] == 40
    assert trainer.update_calls == 20
    assert all(update.update_epochs == 2 for update in trainer.policy_updates)
    assert len(envs) == report["environment_episode_count"] == 10
    assert envs[0].step_calls == 4  # one warm-up plus three fresh groups
    assert len(trainer.start_offset_calls) == len(envs)
    assert all(call == (1, None, 0, 0) for call in trainer.start_offset_calls)
    assert all(env.closed for env in envs)
    assert report["environment_steps"] == 31
    assert report["warmup_environment_steps"] == 10
    assert validation_rollouts == [0, 21]
    counters = report["training_bucket_counters"]
    assert [item["target_rollouts"] for item in counters] == [3] + [2] * 9
    assert [item["sampled_rollouts"] for item in counters] == [3] + [2] * 9
    assert report["rollout_collection_contract"]["version"] == (
        ROLLOUT_COLLECTION_CONTRACT_VERSION
    )
    run_dir = Path(report["last_checkpoint"]).parent.parent
    frozen_config = json.loads((run_dir / "config.json").read_text())
    assert frozen_config["format"] == "bev_joint_grpo_online_config_v6"
    assert frozen_config["rollout_collection_contract"] == report[
        "rollout_collection_contract"
    ]
    checkpoint = torch.load(
        report["last_checkpoint"], map_location="cpu", weights_only=False
    )
    assert checkpoint["rollout_collection_contract"] == report[
        "rollout_collection_contract"
    ]
    sampler = checkpoint["sampler_state"]
    assert sampler["bucket_target_counts"] == [3] + [2] * 9
    assert sampler["bucket_sample_counts"] == [3] + [2] * 9
    assert sampler["bucket_episode_counts"] == [1] * 10
    assert sampler["current_visit_progress"] == 0
    assert sampler["next_bucket_index"] == 0
    assert report["rollout_start_diagnostics_this_run"] == {
        "attempt_count": 10,
        "accepted_count": 10,
        "rejected_count": 0,
        "offset_min_steps": 0,
        "offset_mean_steps": 0.0,
        "offset_max_steps": 0,
    }
    start_events = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "rollout_start"
    ]
    assert len(start_events) == 10
    assert {event["status"] for event in start_events} == {"accepted"}
    assert all(
        {
            "optimizer_step",
            "rollout_group",
            "training_bucket_index",
            "scenario",
            "route",
            "seed",
            "bucket_episode",
            "t_ready",
            "feasible_upper_bound",
            "sampled_offset",
            "target_step",
        }.issubset(event)
        for event in start_events
    )


def test_random_start_hits_exact_target_then_collects_ten_consecutive_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, envs, _, trainer = _run_fake_persistent_collection(
        tmp_path,
        monkeypatch,
        validation_interval_rollouts=10,
        terminal_after_environment_steps=None,
        max_rollout_groups=10,
        rollout_groups_per_bucket_visit=10,
        rollout_start_offset_max_steps=3,
        single_training_bucket=True,
    )

    assert trainer.start_offset_calls == [(1, None, 3, 3)]
    assert trainer.sample_environment_steps == [
        (0, step) for step in range(4, 14)
    ]
    assert len(envs) == 1
    assert envs[0].step_calls == 14
    assert envs[0].summary_calls == 13
    assert trainer.warmup_calls_by_episode == [4]
    assert trainer.sample_calls == 10
    assert trainer.update_calls == 9  # first group is intentionally uninformative
    assert report["sampled_rollouts"] == 10
    assert report["environment_steps"] == 14
    assert report["warmup_environment_steps"] == 4


def test_training_queries_scenario_summary_only_once_per_history_ready_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, envs, _, trainer = _run_fake_persistent_collection(
        tmp_path,
        monkeypatch,
        validation_interval_rollouts=1,
        terminal_after_environment_steps=None,
        max_rollout_groups=1,
        rollout_groups_per_bucket_visit=1,
        single_training_bucket=True,
        first_rollout_informative=True,
        history_ready_step=3,
    )

    assert trainer.sample_environment_steps == [(0, 3)]
    assert trainer.warmup_calls_by_episode == [3]
    assert envs[0].step_calls == 4
    assert envs[0].summary_calls == 1


def test_main_loop_fails_immediately_when_ready_step_has_no_ten_step_reserve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed_envs: list[object] = []
    with pytest.raises(OnlineGRPOError, match="no feasible rollout start offset"):
        _run_fake_persistent_collection(
            tmp_path,
            monkeypatch,
            validation_interval_rollouts=1,
            terminal_after_environment_steps=None,
            max_rollout_groups=1,
            rollout_groups_per_bucket_visit=1,
            rollout_start_offset_max_steps=200,
            single_training_bucket=True,
            first_rollout_informative=True,
            environment_steps_per_episode=200,
            history_ready_step=191,
            external_envs=observed_envs,
        )

    assert len(observed_envs) == 1
    assert observed_envs[0].step_calls == 191
    assert observed_envs[0].summary_calls == 1
    assert observed_envs[0].closed


def test_episode_end_before_random_target_tightens_then_retries_same_visit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, envs, _, trainer = _run_fake_persistent_collection(
        tmp_path,
        monkeypatch,
        validation_interval_rollouts=1,
        terminal_after_environment_steps=3,
        max_rollout_groups=1,
        rollout_groups_per_bucket_visit=1,
        rollout_start_offset_max_steps=3,
        single_training_bucket=True,
        first_rollout_informative=True,
    )

    assert trainer.start_offset_calls == [
        (1, None, 3, 3),
        (1, 1, 1, 1),
    ]
    assert trainer.sample_environment_steps == [(1, 2)]
    assert [env.step_calls for env in envs] == [3, 3]
    run_dir = Path(report["last_checkpoint"]).parent.parent
    statuses = [
        json.loads(line)["status"]
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "rollout_start"
    ]
    assert statuses == [
        "rejected_episode_ended_before_target",
        "accepted",
    ]


def test_late_random_targets_tighten_and_three_rejections_are_not_empty_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, envs, _, trainer = _run_fake_persistent_collection(
        tmp_path,
        monkeypatch,
        validation_interval_rollouts=1,
        terminal_after_environment_steps=None,
        max_rollout_groups=1,
        rollout_groups_per_bucket_visit=1,
        rollout_start_offset_max_steps=4,
        force_start_offset_to_upper=True,
        window_close_steps_by_episode=(4, 3, 2, None),
        single_training_bucket=True,
        first_rollout_informative=True,
    )

    assert [call[1:] for call in trainer.start_offset_calls] == [
        (None, 4, 4),
        (2, 2, 2),
        (1, 1, 1),
        (0, 0, 0),
    ]
    assert len(envs) == 4
    assert trainer.sample_environment_steps == [(3, 1)]
    assert report["sampled_rollouts"] == 1
    assert report["rollout_start_diagnostics_this_run"] == {
        "attempt_count": 4,
        "accepted_count": 1,
        "rejected_count": 3,
        "offset_min_steps": 0,
        "offset_mean_steps": 1.75,
        "offset_max_steps": 4,
    }
    run_dir = Path(report["last_checkpoint"]).parent.parent
    statuses = [
        json.loads(line)["status"]
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if json.loads(line)["event"] == "rollout_start"
    ]
    assert statuses == [
        "rejected_window_closed_before_target",
        "rejected_window_closed_before_target",
        "rejected_window_closed_before_target",
        "accepted",
    ]


def test_window_close_mid_segment_keeps_partial_visit_and_never_warms_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, envs, _, trainer = _run_fake_persistent_collection(
        tmp_path,
        monkeypatch,
        validation_interval_rollouts=3,
        terminal_after_environment_steps=None,
        max_rollout_groups=3,
        rollout_groups_per_bucket_visit=3,
        window_close_steps_by_episode=(2, None),
        single_training_bucket=True,
    )

    assert trainer.sample_environment_steps == [(0, 1), (1, 1), (1, 2)]
    assert trainer.warmup_calls_by_episode == [1, 1]
    assert [env.step_calls for env in envs] == [2, 3]
    assert report["sampled_rollouts"] == 3
    assert report["training_bucket_counters"][0]["environment_episodes"] == 2
    assert trainer.start_offset_calls == [
        (1, None, 0, 0),
        (1, None, 0, 0),
    ]


def test_validation_and_natural_end_resume_same_bucket_visit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report, envs, validation_rollouts, _ = _run_fake_persistent_collection(
        tmp_path / "validation",
        monkeypatch,
        validation_interval_rollouts=2,
        terminal_after_environment_steps=None,
    )
    assert validation_rollouts == [0] + list(range(2, 21, 2)) + [21]
    assert report["training_bucket_counters"][0]["environment_episodes"] == 2
    assert envs[0].step_calls == 3  # warm-up + rollouts 1 and 2
    assert envs[1].step_calls == 2  # warm-up + rollout 3, same bucket
    assert envs[0].scenario == envs[1].scenario
    assert envs[0].seed == envs[1].seed

    report, terminal_envs, _, _ = _run_fake_persistent_collection(
        tmp_path / "terminal",
        monkeypatch,
        validation_interval_rollouts=21,
        terminal_after_environment_steps=2,
    )
    assert report["training_bucket_counters"][0]["environment_episodes"] == 3
    assert len(terminal_envs) == 21
    assert all(env.step_calls == 2 for env in terminal_envs)
    assert report["warmup_environment_steps"] == 21


@pytest.mark.parametrize("group_size", [3, 5])
def test_online_informative_reward_detection_follows_group_size(
    group_size: int,
) -> None:
    constant = np.full(group_size, -1.0, dtype=np.float32)
    informative = constant.copy()
    informative[-1] = 0.0

    assert not _joint_rewards_are_informative(constant, group_size=group_size)
    assert _joint_rewards_are_informative(informative, group_size=group_size)


def test_scenario_routes_and_primary_sampling_contract() -> None:
    for scenario_id, route in PRIMARY_S5_S9_SCENARIOS:
        assert route in SCENARIO_BY_ID[scenario_id].allowed_local_routes
    assert JointGRPOOnlineConfig(device="cpu").scenarios == PRIMARY_S5_S9_SCENARIOS

    class Orchestrator:
        def get_episode_summary(self):
            return {
                "scenario_id": "S8_ego_exit_to_ramp",
                "scenario_realized": True,
                "scenario_recipes_complete": False,
            }

    env = SimpleNamespace(_scenario_orchestrator=Orchestrator())
    assert not _scenario_ready_for_primary_sampling(env)


def test_execution_helpers_preserve_raw_policy_action() -> None:
    coarse = np.zeros((3, 10, 8, 3), dtype=np.float32)
    coarse[..., 0] = 4.0 * np.arange(1, 9, dtype=np.float32)
    raw = np.broadcast_to(coarse[:, 0], (1, 3, 8, 3)).copy()
    raw[..., 0] *= -1.0
    original = raw.copy()
    values = SimpleNamespace(
        coarse_trajectories=coarse,
        ego_state=np.pad(
            np.full((3, 1), 8.0, dtype=np.float32), ((0, 0), (0, 7))
        ),
    )
    result = optimize_selected_model_trajectories(
        values, raw, np.zeros((1, 3), dtype=np.int64)
    )
    assert np.array_equal(raw, original)
    assert np.array_equal(result.raw_trajectories, original)
    assert result.optimized_valid.all()


def test_online_action_and_batch_helpers_are_strict() -> None:
    env = SimpleNamespace(
        agents={
            f"agent{role}": SimpleNamespace(speed_km_h=18.0)
            for role in range(3)
        }
    )
    assert constant_velocity_actions(env)["agent0"][1, 0] == pytest.approx(5.0)
    trajectories = np.zeros((3, 8, 3), dtype=np.float32)
    action = joint_trajectory_action(trajectories)
    trajectories[0, 0, 0] = 99.0
    assert action["agent0"][0, 0] == 0.0
    assert episode_has_ended(
        {"__all__": False},
        {"__all__": False},
        {"agent1": {"out_of_road": True}},
    )

    coarse = np.zeros((3, 10, 8, 3), dtype=np.float32)
    coarse[..., 0] = (
        4.0 * np.arange(1, 9, dtype=np.float32)[None, None, :] * 0.5
    )
    coarse[:, 9, :, 0] = np.asarray(
        [1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32
    )
    fields = {
        "bev": np.zeros((3, 8, 256, 256), dtype=np.uint8),
        "ego_state": np.zeros((3, 8), dtype=np.float32),
        "formation_relation_state": np.zeros((3, 12), dtype=np.float32),
        "relation_valid_mask": np.ones((3, 2), dtype=np.bool_),
        "agent_role": np.arange(3, dtype=np.int64),
        "coarse_trajectories": coarse,
        "mode_valid_mask": np.ones((3, 10), dtype=np.bool_),
    }
    fields["ego_state"][:, 0] = 4.0
    values = SimpleNamespace(**fields, as_dict=lambda: fields)
    mask = execution_mode_valid_mask(values)
    batch = model_inputs_to_batch(values, torch.device("cpu"), mode_valid_mask=mask)
    assert torch.equal(batch["mode_valid_mask"][0], torch.from_numpy(mask.copy()))


def test_v2_online_inputs_use_and_accept_exact_rule_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposal = JointActionProposal(
        proposal_id=4,
        rank=0,
        rule_score=1.0,
        decisions={
            agent_id: {"action": 0}
            for agent_id in ("agent0", "agent1", "agent2")
        },
    )
    proposal_batch = RuleMakerProposalBatch(batch_id=9, proposals=(proposal,))

    class RuleMaker:
        is_formation_locked = True
        has_active_lane_change_commitments = False

        def __init__(self):
            self.reset_args = None
            self.proposal_args = None
            self.accepted = None

        def reset(self, env, agent_ids):
            self.reset_args = (env, tuple(agent_ids))

        def propose_joint_actions(
            self, env, agent_ids, planner_batch, *, hard_valid_modes_by_action
        ):
            self.proposal_args = (
                env,
                tuple(agent_ids),
                planner_batch,
                hard_valid_modes_by_action,
            )
            return proposal_batch

        def accept_joint_action(self, batch_id, proposal_id):
            self.accepted = (batch_id, proposal_id)

    class Builder:
        def __init__(self):
            self.augmentation = None

        def augment_v2_model_inputs(self, env, values, **kwargs):
            self.augmentation = (env, values, kwargs)
            return SimpleNamespace(
                conditioned=True, mode_valid_mask=values.mode_valid_mask
            )

    rule_maker = RuleMaker()
    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online.make_rule_maker",
        lambda config: rule_maker,
    )
    env = SimpleNamespace(
        config={"scenario_id": "S5_hard_brake_lead"},
        _last_planner_batch={"tick": 3},
        trajectory_to_control=lambda agent_id, trajectory: np.zeros(2),
    )
    planner = SimpleNamespace(config=SimpleNamespace(model_version="v2"))
    values = SimpleNamespace(
        mode_valid_mask=np.ones((3, 10), dtype=np.bool_)
    )
    builder = Builder()

    resolved_rule_maker = _new_online_rule_maker(planner, env)
    condition = _condition_online_model_inputs(
        resolved_rule_maker,
        env,
        builder,
        values,
        committed_execution_id=None,
        committed_plan_actions=None,
    )

    assert resolved_rule_maker is rule_maker
    assert rule_maker.reset_args == (env, ("agent0", "agent1", "agent2"))
    assert condition.model_inputs.conditioned
    assert condition.proposal_batch is proposal_batch
    assert not condition.is_commitment
    assert builder.augmentation[2] == {
        "rule_action_condition": {"agent0": 0, "agent1": 0, "agent2": 0},
        "rule_formation_state": True,
    }

    optimization = SimpleNamespace(
        optimized_trajectories=np.zeros((1, 3, 8, 3), dtype=np.float32)
    )
    resolved, diagnostics, committed_id, committed_actions = (
        _finalize_online_rule_action(
        rule_maker,
        condition,
        env=env,
        scenario=("S5_hard_brake_lead", "R1_entry_straight"),
        values=condition.model_inputs,
        selected_modes=np.full(3, int(ModeIndex.KEEP_HIGH), dtype=np.int64),
        optimization=optimization,
        optimizer=object(),
        )
    )

    assert resolved is optimization
    assert committed_id is None
    assert committed_actions is None
    assert rule_maker.accepted == (9, 4)
    assert diagnostics["conditioned_rollouts"] == 1
    assert diagnostics["proposal_matches"] == 1
    assert diagnostics["condition_failures"] == 0


def test_online_rule_maker_commitment_persists_and_resumes_proposals() -> None:
    proposal = JointActionProposal(
        proposal_id=4,
        rank=0,
        rule_score=1.0,
        decisions={agent_id: {"action": 0} for agent_id in AGENT_IDS},
    )
    proposal_batch = RuleMakerProposalBatch(batch_id=9, proposals=(proposal,))

    class RuleMaker:
        is_formation_locked = True

        def __init__(self) -> None:
            self.active = False
            self.proposal_calls = 0
            self.advance_calls: list[int] = []

        @property
        def has_active_lane_change_commitments(self) -> bool:
            return self.active

        def propose_joint_actions(
            self, env, agent_ids, planner_batch, *, hard_valid_modes_by_action
        ):
            self.proposal_calls += 1
            return proposal_batch

        def accept_joint_action(self, batch_id, proposal_id) -> None:
            assert (batch_id, proposal_id) == (9, 4)
            self.active = True

        def advance_committed_execution(
            self, env, agent_ids, execution_id
        ) -> None:
            self.advance_calls.append(int(execution_id))
            if len(self.advance_calls) == 3:
                self.active = False

        def committed_execution_rule_actions(self, env, plan_actions):
            return dict(plan_actions)

    class Builder:
        def augment_v2_model_inputs(self, env, values, **kwargs):
            return values

    rule_maker = RuleMaker()
    env = SimpleNamespace(
        _last_planner_batch={},
        trajectory_to_control=lambda agent_id, trajectory: np.zeros(2),
    )
    values = SimpleNamespace(
        mode_valid_mask=np.ones((3, 10), dtype=np.bool_),
        coarse_trajectories=np.zeros((3, 10, 8, 3), dtype=np.float32),
        ego_state=np.zeros((3, 8), dtype=np.float32),
    )
    optimization = SimpleNamespace(
        optimized_trajectories=np.zeros((1, 3, 8, 3), dtype=np.float32)
    )

    proposal_condition = _condition_online_model_inputs(
        rule_maker,
        env,
        Builder(),
        values,
        committed_execution_id=None,
        committed_plan_actions=None,
    )
    _, _, committed_id, committed_actions = _finalize_online_rule_action(
        rule_maker,
        proposal_condition,
        env=env,
        scenario=("S5_hard_brake_lead", "R1_entry_straight"),
        values=values,
        selected_modes=np.full(3, int(ModeIndex.KEEP_HIGH), dtype=np.int64),
        optimization=optimization,
        optimizer=KinematicTrajectoryOptimizer(),
    )
    assert committed_id == 9
    assert committed_actions == {agent_id: 0 for agent_id in AGENT_IDS}

    committed_condition = _condition_online_model_inputs(
        rule_maker,
        env,
        Builder(),
        values,
        committed_execution_id=committed_id,
        committed_plan_actions=committed_actions,
    )
    assert committed_condition.is_commitment
    assert committed_condition.proposal_batch is None
    assert rule_maker.proposal_calls == 1
    resolved, diagnostics, resumed_id, resumed_actions = (
        _finalize_online_rule_action(
            rule_maker,
            committed_condition,
            env=env,
            scenario=("S5_hard_brake_lead", "R1_entry_straight"),
            values=values,
            selected_modes=np.full(
                3, int(ModeIndex.KEEP_HIGH), dtype=np.int64
            ),
            optimization=optimization,
            optimizer=KinematicTrajectoryOptimizer(),
        )
    )
    assert resolved is optimization
    assert diagnostics["proposal_match_attempts"] == 0
    assert diagnostics["proposal_matches"] == 0
    assert diagnostics["commitment_conditioned_rollouts"] == 1
    assert diagnostics["commitment_feedback_incompatible"] == 0
    assert resumed_id == committed_id
    assert resumed_actions == committed_actions

    incompatible_condition = _condition_online_model_inputs(
        rule_maker,
        env,
        Builder(),
        values,
        committed_execution_id=resumed_id,
        committed_plan_actions=resumed_actions,
    )
    resolved, diagnostics, resumed_id, resumed_actions = (
        _finalize_online_rule_action(
            rule_maker,
            incompatible_condition,
            env=env,
            scenario=("S5_hard_brake_lead", "R1_entry_straight"),
            values=values,
            selected_modes=np.full(
                3, int(ModeIndex.LEFT_HIGH), dtype=np.int64
            ),
            optimization=optimization,
            optimizer=KinematicTrajectoryOptimizer(),
        )
    )
    assert resolved.optimized_trajectories.shape == (1, 3, 8, 3)
    assert np.all(resolved.selected_modes == int(ModeIndex.STOP))
    assert diagnostics["proposal_match_attempts"] == 0
    assert diagnostics["proposal_matches"] == 0
    assert diagnostics["commitment_conditioned_rollouts"] == 1
    assert diagnostics["commitment_feedback_incompatible"] == 1
    assert resumed_id == committed_id
    assert resumed_actions == committed_actions

    next_proposal = _condition_online_model_inputs(
        rule_maker,
        env,
        Builder(),
        values,
        committed_execution_id=resumed_id,
        committed_plan_actions=resumed_actions,
    )
    assert not next_proposal.is_commitment
    assert next_proposal.proposal_batch is proposal_batch
    assert next_proposal.committed_execution_id is None
    assert next_proposal.committed_plan_actions is None
    assert rule_maker.proposal_calls == 2
    assert rule_maker.advance_calls == [9, 9, 9]


@pytest.mark.parametrize("model_version", [None, "v1"])
def test_online_grpo_rejects_non_v2_planner(model_version: object) -> None:
    planner = SimpleNamespace(config=SimpleNamespace(model_version=model_version))
    with pytest.raises(OnlineGRPOError, match="requires a v2 planner"):
        _new_online_rule_maker(planner, SimpleNamespace())


def test_v2_unmatched_rule_action_executes_batched_safe_stop() -> None:
    proposal = JointActionProposal(
        proposal_id=4,
        rank=0,
        rule_score=1.0,
        decisions={
            agent_id: {"action": 0}
            for agent_id in ("agent0", "agent1", "agent2")
        },
    )
    proposal_batch = RuleMakerProposalBatch(batch_id=9, proposals=(proposal,))

    class RuleMaker:
        accepted = None
        has_active_lane_change_commitments = False

        def accept_joint_action(self, batch_id, proposal_id):
            self.accepted = (batch_id, proposal_id)

    rule_maker = RuleMaker()
    env = SimpleNamespace(
        trajectory_to_control=lambda agent_id, trajectory: np.zeros(2)
    )
    optimization = SimpleNamespace(
        optimized_trajectories=np.zeros((1, 3, 8, 3), dtype=np.float32)
    )
    values = SimpleNamespace(
        coarse_trajectories=np.zeros((3, 10, 8, 3), dtype=np.float32),
        ego_state=np.zeros((3, 8), dtype=np.float32),
    )

    condition = _OnlineRuleCondition(
        model_inputs=values,
        proposal_batch=proposal_batch,
        rule_actions={agent_id: 0 for agent_id in ("agent0", "agent1", "agent2")},
        is_commitment=False,
        committed_execution_id=None,
        committed_plan_actions=None,
    )
    resolved, diagnostics, committed_id, committed_actions = (
        _finalize_online_rule_action(
        rule_maker,
        condition,
        env=env,
        scenario=("S5_hard_brake_lead", "R1_entry_straight"),
        values=values,
        selected_modes=np.full(3, int(ModeIndex.LEFT_HIGH), dtype=np.int64),
        optimization=optimization,
        optimizer=KinematicTrajectoryOptimizer(),
        )
    )

    assert resolved.optimized_trajectories.shape == (1, 3, 8, 3)
    assert resolved.selected_modes.shape == (1, 3)
    assert np.all(resolved.selected_modes == int(ModeIndex.STOP))
    assert np.isfinite(resolved.optimized_trajectories).all()
    assert len(joint_trajectory_action(resolved.optimized_trajectories[0])) == 3
    assert rule_maker.accepted is None
    assert committed_id is None
    assert committed_actions is None
    assert diagnostics["condition_failures"] == 1
    assert diagnostics["forced_safe_stops"] == 1
