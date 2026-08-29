from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.bev_planner.joint_grpo import JointGRPOConfig
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
    PRIMARY_S5_S9_SCENARIOS,
    JointGRPOOnlineConfig,
    OnlineGRPOError,
    _checkpoint_file_sha256,
    _config_from_yaml,
    _condition_online_model_inputs,
    _finalize_online_rule_action,
    _fixed_raw_proxy_and_simulator_validation,
    _joint_rewards_are_informative,
    _load_trainer,
    _new_online_rule_maker,
    _next_run_directory,
    _resume_best_checkpoint_anchor,
    _round_robin_training_buckets,
    _scenario_ready_for_primary_sampling,
    _score_select_and_optimize_raw_candidates,
    _should_record_advantage_vector,
    _validate_online_checkpoint_metadata,
    _validate_raw_reward_config,
    _validation_raw_proxy_reward,
    _validation_reward_comparison_metrics,
    _write_advantage_vector_summary,
    constant_velocity_actions,
    episode_has_ended,
    execution_mode_valid_mask,
    joint_trajectory_action,
    model_inputs_to_batch,
    optimize_selected_model_trajectories,
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
    return {
        "schema_version": 1,
        "format": "bev_joint_grpo_a_v1",
        "variant": "A",
        "predecessor_condition": "A",
        "source_stage1_sha256": "a" * 64,
        "grpo_config": dataclasses.asdict(JointGRPOConfig()),
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
        "scenario_contract_sha256": primary_scenario_contract()["sha256"],
        "scenario_seeds": [17, 23],
        "trajectory_optimizer_config": dataclasses.asdict(optimizer_config),
        "trajectory_optimizer_sha256": optimizer_config.sha256(),
        "environment_steps": 1,
    }


def test_online_config_and_run_mode_are_strict(tmp_path: Path) -> None:
    with pytest.raises(OnlineGRPOError):
        JointGRPOOnlineConfig(device="auto")
    with pytest.raises(OnlineGRPOError):
        JointGRPOOnlineConfig(total_optimizer_steps=0)
    assert JointGRPOOnlineConfig().group_size == 4
    assert JointGRPOOnlineConfig(group_size=3).group_size == 3
    assert JointGRPOOnlineConfig(group_size=5).group_size == 5
    assert JointGRPOOnlineConfig().advantage_vector_log_interval_steps == 100
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
            max_optimizer_steps=0,
        )


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
    with pytest.raises(OnlineGRPOError, match="advantage_vector_log_interval_steps"):
        JointGRPOOnlineConfig(
            advantage_vector_log_interval_steps=invalid_interval  # type: ignore[arg-type]
        )


def test_online_config_loads_group_size_and_advantage_interval_from_yaml(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "online.yaml"
    config_path.write_text(
        "online:\n"
        "  device: cpu\n"
        "  group_size: 5\n"
        "  advantage_vector_log_interval_steps: 37\n",
        encoding="utf-8",
    )

    config = _config_from_yaml(config_path)

    assert config.group_size == 5
    assert config.advantage_vector_log_interval_steps == 37


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


def test_advantage_vector_recording_schedule_uses_interval_or_final_step() -> None:
    assert not _should_record_advantage_vector(
        optimizer_step=1,
        target_steps=350,
        interval_steps=100,
    )
    assert _should_record_advantage_vector(
        optimizer_step=100,
        target_steps=350,
        interval_steps=100,
    )
    assert _should_record_advantage_vector(
        optimizer_step=350,
        target_steps=350,
        interval_steps=100,
    )


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
        optimizer_step=99,
        target_steps=350,
        interval_steps=100,
        group_size=group_size,
    )
    assert writer.calls == []

    assert _write_advantage_vector_summary(
        writer,
        advantages,
        optimizer_step=100,
        target_steps=350,
        interval_steps=100,
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
        optimizer_step=350,
        target_steps=350,
        interval_steps=100,
        group_size=group_size,
    )
    assert [call[2] for call in writer.calls] == [100, 350]


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
            optimizer_step=100,
            target_steps=350,
            interval_steps=100,
            group_size=4,
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
    last_payload["best_checkpoint_sha256"] = "0" * 64
    with pytest.raises(OnlineGRPOError, match="SHA256 mismatch"):
        _resume_best_checkpoint_anchor(tmp_path / "last.pt", last_payload)


def test_checkpoint_metadata_rejects_legacy_and_domain_drift() -> None:
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
    )
    payload["reward_input_domain"] = "tau_cmd"
    with pytest.raises(OnlineGRPOError, match="reward_input_domain mismatch"):
        _validate_online_checkpoint_metadata(
            payload,
            run_mode="smoke",
            reward_config=JointRewardConfig(),
            scenario_contract_sha=primary_scenario_contract()["sha256"],
            scenario_seeds=(17, 23),
        )
    payload = {**_binding(), "calibration_report_sha256": "b" * 64}
    with pytest.raises(OnlineGRPOError, match="legacy calibration semantics"):
        _validate_online_checkpoint_metadata(
            payload,
            run_mode="smoke",
            reward_config=JointRewardConfig(),
            scenario_contract_sha=primary_scenario_contract()["sha256"],
            scenario_seeds=(17, 23),
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
        lambda rule_maker, env, builder, values: (values, object()),
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
            max_optimizer_steps=1,
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
    conditioned, resolved_batch = _condition_online_model_inputs(
        resolved_rule_maker, env, builder, values
    )

    assert resolved_rule_maker is rule_maker
    assert rule_maker.reset_args == (env, ("agent0", "agent1", "agent2"))
    assert conditioned.conditioned
    assert resolved_batch is proposal_batch
    assert builder.augmentation[2] == {
        "rule_action_condition": {"agent0": 0, "agent1": 0, "agent2": 0},
        "rule_formation_state": True,
    }

    optimization = SimpleNamespace(
        optimized_trajectories=np.zeros((1, 3, 8, 3), dtype=np.float32)
    )
    resolved, diagnostics = _finalize_online_rule_action(
        rule_maker,
        proposal_batch,
        env=env,
        scenario=("S5_hard_brake_lead", "R1_entry_straight"),
        values=conditioned,
        selected_modes=np.full(3, int(ModeIndex.KEEP_HIGH), dtype=np.int64),
        optimization=optimization,
        optimizer=object(),
    )

    assert resolved is optimization
    assert rule_maker.accepted == (9, 4)
    assert diagnostics["conditioned_rollouts"] == 1
    assert diagnostics["proposal_matches"] == 1
    assert diagnostics["condition_failures"] == 0


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

    resolved, diagnostics = _finalize_online_rule_action(
        rule_maker,
        proposal_batch,
        env=env,
        scenario=("S5_hard_brake_lead", "R1_entry_straight"),
        values=values,
        selected_modes=np.full(3, int(ModeIndex.LEFT_HIGH), dtype=np.int64),
        optimization=optimization,
        optimizer=KinematicTrajectoryOptimizer(),
    )

    assert resolved.optimized_trajectories.shape == (1, 3, 8, 3)
    assert resolved.selected_modes.shape == (1, 3)
    assert np.all(resolved.selected_modes == int(ModeIndex.STOP))
    assert np.isfinite(resolved.optimized_trajectories).all()
    assert len(joint_trajectory_action(resolved.optimized_trajectories[0])) == 3
    assert rule_maker.accepted is None
    assert diagnostics["condition_failures"] == 1
    assert diagnostics["forced_safe_stops"] == 1
