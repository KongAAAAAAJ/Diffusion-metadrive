from __future__ import annotations

import dataclasses
import json
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
    _round_robin_training_buckets,
    _sampler_state,
    _scenario_ready_for_primary_sampling,
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
    optimize_selected_model_trajectories,
    rollout_collection_contract,
    run_joint_grpo_training,
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
    assert JointGRPOOnlineConfig().validation_interval_rollouts == 20
    assert JointGRPOOnlineConfig().advantage_vector_log_interval_rollouts == 20
    with pytest.raises(OnlineGRPOError, match="complete ordered S5--S9"):
        JointGRPOOnlineConfig(
            scenarios=(("S1_free_cruise_straight", "R3_mainline_straight"),)
        )
    config = JointGRPOOnlineConfig(device="cpu")
    with pytest.raises(OnlineGRPOError, match="positive"):
        run_joint_grpo_training(
            config,
            variant="A",
            run_mode="smoke",
            source_checkpoint=tmp_path / "missing.pt",
            output_root=tmp_path / "output",
            max_rollout_groups=0,
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
        "online:\n"
        "  device: cpu\n"
        "  group_size: 5\n"
        "  total_rollout_groups: 41\n"
        "  update_epochs: 3\n"
        "  clip_epsilon: 0.15\n"
        "  rollout_groups_per_bucket_visit: 7\n"
        "  validation_interval_rollouts: 11\n"
        "  advantage_vector_log_interval_rollouts: 37\n",
        encoding="utf-8",
    )

    config = _config_from_yaml(config_path)

    assert config.group_size == 5
    assert config.total_rollout_groups == 41
    assert config.update_epochs == 3
    assert config.clip_epsilon == pytest.approx(0.15)
    assert config.rollout_groups_per_bucket_visit == 7
    assert config.validation_interval_rollouts == 11
    assert config.advantage_vector_log_interval_rollouts == 37


@pytest.mark.parametrize("yaml_value", ["true", "false", "1", "0", "-1"])
def test_online_config_rejects_invalid_yaml_group_size(
    tmp_path: Path,
    yaml_value: str,
) -> None:
    config_path = tmp_path / f"invalid_group_{yaml_value}.yaml"
    config_path.write_text(
        f"online:\n  device: cpu\n  group_size: {yaml_value}\n",
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
        f"online:\n  device: cpu\n  {legacy_field}: 1\n",
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
        def close(self):
            return None

    class Builder:
        def __init__(self, agent_ids):
            pass

        def reset(self):
            pass

        def capture_state(self, env, timestamp):
            pass

        def history_ready(self):
            return True

        def build_model_inputs(self, env):
            return values

    class Proxy:
        def __init__(self, config):
            pass

        def score(self, env, model_inputs, trajectories):
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

    monkeypatch.setattr("train.train_bev_joint_grpo_online._new_env", lambda *a: Env())
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
    config = JointGRPOOnlineConfig(
        device="cpu", resume_checkpoint=tmp_path / "resume.pt"
    )
    with pytest.raises(OnlineGRPOError, match="resume loader reached"):
        run_joint_grpo_training(
            config,
            variant="A",
            run_mode="smoke",
            source_checkpoint=tmp_path / "stage1.pt",
            output_root=tmp_path / "output",
            max_rollout_groups=1,
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

    assert contract["version"] == ROLLOUT_COLLECTION_CONTRACT_VERSION
    assert contract["rollout_groups_per_bucket_visit"] == 7
    assert contract["environment_steps_per_episode"] == 200
    assert contract["validation_interval_rollouts"] == 13
    assert contract["checkpoint_boundary"] == "closed_environment_only"
    assert _balanced_bucket_targets(100, 10) == [10] * 10
    assert _balanced_bucket_targets(103, 10) == [11, 11, 11] + [10] * 7
    assert _balanced_bucket_targets(3, 5) == [1, 1, 1, 0, 0]


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
) -> tuple[dict[str, object], list[object], list[int], object]:
    group_size = 3
    update_epochs = 2
    envs: list[object] = []
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
        def __init__(self, scenario, seed) -> None:
            self.scenario = scenario
            self.seed = seed
            self.step_calls = 0
            self.closed = False
            self._last_planner_batch = {}

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
            return self.captures >= 2

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

        def sample_groups(self, batch, *, generator):
            torch.randn((1,), generator=generator)
            self.sample_calls += 1
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
        env = Env(scenario, seed)
        envs.append(env)
        return env

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
            if trainer.sample_calls == 1
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

    config = JointGRPOOnlineConfig(
        device="cpu",
        group_size=group_size,
        update_epochs=update_epochs,
        environment_steps_per_episode=50,
        rollout_groups_per_bucket_visit=3,
        validation_interval_rollouts=validation_interval_rollouts,
        advantage_vector_log_interval_rollouts=21,
    )
    report = run_joint_grpo_training(
        config,
        variant="A",
        run_mode="smoke",
        source_checkpoint=tmp_path / "stage1.pt",
        output_root=tmp_path / "output",
        max_rollout_groups=21,
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

    assert report["format"] == "bev_joint_grpo_online_report_v5"
    assert report["sampled_rollouts"] == 21
    assert report["uninformative_rollouts"] == 1
    assert report["optimizer_steps"] == 40
    assert trainer.update_calls == 20
    assert all(update.update_epochs == 2 for update in trainer.policy_updates)
    assert len(envs) == report["environment_episode_count"] == 10
    assert envs[0].step_calls == 4  # one warm-up plus three fresh groups
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
    assert frozen_config["format"] == "bev_joint_grpo_online_config_v5"
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
