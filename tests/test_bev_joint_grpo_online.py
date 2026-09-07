from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import train.train_bev_joint_grpo_online as online
from train.train_bev_joint_grpo_online import (
    JointGRPOOnlineConfig,
    OnlineGRPOError,
    _append_validation_selection_event,
    _config_from_yaml,
    _diffusion_attempt_generators,
    _fixed_scale_reward_signals,
    _sampler_state,
    _validate_sampler_state,
    _write_dynamic_sampling_attempt_event,
    execute_cached_frozen_baseline,
    rollout_collection_contract,
)
from models.bev_planner.joint_reward import (
    VehicleModeRewardConfig,
    VehicleModeRewardResult,
)


def _valid_sampler_state() -> tuple[dict[str, object], dict[str, object]]:
    generator = torch.Generator().manual_seed(17)
    kwargs = {
        "accepted_update_states": 2,
        "sampling_attempts": 5,
        "rejected_sampling_attempts": 3,
        "exhausted_states": 1,
        "baseline_execution_steps": 3,
        "zero_signal_epochs": 4,
        "warmup_environment_steps": 7,
        "bucket_target_counts": [4],
        "bucket_accepted_update_counts": [2],
        "bucket_sampling_attempt_counts": [5],
        "bucket_rejected_sampling_attempt_counts": [3],
        "bucket_exhausted_state_counts": [1],
        "bucket_baseline_execution_step_counts": [3],
        "bucket_zero_signal_epoch_counts": [4],
        "bucket_optimizer_step_counts": [16],
        "bucket_episode_counts": [1],
        "next_bucket_index": 0,
        "current_visit_progress": 0,
        "rollout_start_generator_state": generator.get_state(),
        "last_validated_update_state": 2,
        "rollout_groups_per_bucket_visit": 2,
        "optimizer_step": 16,
        "environment_steps": 10,
        "max_sampling_attempts": 12,
    }
    state = _sampler_state(**kwargs)
    return state, kwargs


def test_online_config_defaults_match_formal_same_mode_contract() -> None:
    config = JointGRPOOnlineConfig()
    assert config.trajectories_per_mode == 48
    assert not hasattr(config, "update_epochs")
    assert not hasattr(config, "clip_epsilon_low")
    assert not hasattr(config, "clip_epsilon_high")
    assert config.max_sampling_attempts_per_state == 3


def test_checked_in_yaml_uses_only_new_sampling_fields() -> None:
    config = _config_from_yaml(Path("configs/train/bev_joint_grpo.yaml"))
    assert config.online.trajectories_per_mode == 48
    assert config.online.max_sampling_attempts_per_state == 3
    assert not hasattr(config.online, "update_epochs")
    payload = Path("configs/train/bev_joint_grpo.yaml").read_text(encoding="utf-8")
    for removed in (
        "group_size:",
        "pretrain_improvement_margin:",
        "max_candidate_groups_per_state:",
        "max_attempted_groups_multiplier:",
    ):
        assert removed not in payload


def _stub_git_status(monkeypatch: pytest.MonkeyPatch, status: str) -> None:
    def run(arguments, **kwargs):
        del kwargs
        if arguments[1:3] == ("rev-parse", "HEAD"):
            stdout = "a" * 40 + "\n"
        else:
            stdout = status
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(online.subprocess, "run", run)


@pytest.mark.parametrize(
    "status",
    (
        "",
        " M configs/train/bev_joint_grpo.yaml\n",
        "M  configs/train/bev_joint_grpo.yaml\n",
        "MM configs/train/bev_joint_grpo.yaml\n",
    ),
)
def test_implementation_commit_allows_only_runtime_config_edits(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    _stub_git_status(monkeypatch, status)
    assert online._implementation_commit() == "a" * 40


@pytest.mark.parametrize(
    "status",
    (
        " M train/train_bev_joint_grpo_online.py\n",
        " D configs/train/bev_joint_grpo.yaml\n",
        "R  configs/train/bev_joint_grpo.yaml -> configs/train/moved.yaml\n",
    ),
)
def test_implementation_commit_rejects_code_and_config_removal(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    _stub_git_status(monkeypatch, status)
    with pytest.raises(OnlineGRPOError, match="disallowed tracked changes"):
        online._implementation_commit()


def test_yaml_rejects_obsolete_group_size(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
run:
  variant: A
  run_mode: smoke
  source_checkpoint: stage1.pt
online:
  group_size: 48
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(OnlineGRPOError, match="legacy fields: group_size"):
        _config_from_yaml(path)


def test_fixed_scale_signal_is_independent_per_vehicle_mode_and_validity() -> None:
    rewards = np.zeros((3, 10, 4), dtype=np.float32)
    frozen = np.ones((3, 10, 4), dtype=np.float32)
    valid = np.ones((3, 10), dtype=np.bool_)
    rewards[1, 7] = np.asarray([0.0, 1.0, 2.0, 3.0])
    frozen[1, 7] = rewards[1, 7]
    rewards[2, 4] = np.asarray([0.0, 1.0, 2.0, 3.0])
    frozen[2, 4] = rewards[2, 4]
    valid[2, 4] = False
    _, advantages, signal = _fixed_scale_reward_signals(
        rewards,
        frozen,
        np.zeros_like(rewards, dtype=np.bool_),
        np.zeros_like(rewards, dtype=np.bool_),
        valid,
    )
    assert signal.sum() == 1 and signal[1, 7]
    assert not signal[2, 4]
    assert advantages[1, 7].max() == pytest.approx(1.5)


def test_dynamic_attempt_metrics_exclude_invalid_modes(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    rewards = np.full((3, 10, 48), 1000.0, dtype=np.float32)
    pretrain = np.full((3, 10), 1000.0, dtype=np.float32)
    valid = np.zeros((3, 10), dtype=np.bool_)
    valid[0, 0] = True
    rewards[0, 0] = np.arange(48, dtype=np.float32)
    pretrain[0, 0] = 23.5
    paired = np.broadcast_to(pretrain[..., None], rewards.shape).copy()
    centered, advantages, signal = _fixed_scale_reward_signals(
        rewards,
        paired,
        np.zeros_like(rewards, dtype=np.bool_),
        np.zeros_like(rewards, dtype=np.bool_),
        valid,
    )

    shape = rewards.shape
    components = {
        "progress_score": rewards / 0.47,
        "gap_penalty": np.zeros(shape, dtype=np.float32),
        "ttc_penalty": np.zeros(shape, dtype=np.float32),
        "road_penalty": np.zeros(shape, dtype=np.float32),
        "comfort_penalty": np.zeros(shape, dtype=np.float32),
        "minimum_background_gap_m": np.ones(shape, dtype=np.float32),
        "minimum_teammate_gap_m": np.ones(shape, dtype=np.float32),
        "minimum_road_margin_m": np.ones(shape, dtype=np.float32),
        "minimum_ttc_s": np.ones(shape, dtype=np.float32),
    }
    result = VehicleModeRewardResult(
        rewards=rewards,
        pretrain_rewards=pretrain,
        valid_mode_mask=valid,
        unsafe=np.zeros(shape, dtype=np.bool_),
        collision=np.zeros(shape, dtype=np.bool_),
        out_of_drivable=np.zeros(shape, dtype=np.bool_),
        clearance_violation=np.zeros(shape, dtype=np.bool_),
        pretrain_unsafe=np.zeros((3, 10), dtype=np.bool_),
        pretrain_collision=np.zeros((3, 10), dtype=np.bool_),
        pretrain_out_of_drivable=np.zeros((3, 10), dtype=np.bool_),
        pretrain_clearance_violation=np.zeros((3, 10), dtype=np.bool_),
        components=components,
        pretrain_components={},
    )
    metrics = _write_dynamic_sampling_attempt_event(
        path,
        optimizer_step=0,
        accepted_update_state=1,
        sampling_attempt=1,
        retry_index=1,
        bucket_index=0,
        scenario=("S5_hard_brake_lead", "R1_entry_straight"),
        seed=17,
        reward_result=result,
        paired_frozen_rewards=paired,
        centered_rewards=centered,
        advantages=advantages,
        hard_valid_mode_mask=valid,
        signal_mode_mask=signal,
        reward_config=VehicleModeRewardConfig(trajectories_per_mode=48),
    )

    assert metrics["train/valid_all/vehicle_reward_mean"] == pytest.approx(23.5 / 3.0)
    assert metrics["train/valid_all/same_mode_pretrain_reward_mean"] == pytest.approx(23.5 / 3.0)
    event = json.loads(path.read_text(encoding="utf-8"))
    assert event["accepted"] is True
    assert event["signal_mode_count"] == 1.0
    assert len(event["vehicle_mode_groups"]) == 30
    assert sum(group["valid"] for group in event["vehicle_mode_groups"]) == 1
    assert (
        metrics[
            "train/active_only/positive_advantage_below_pretrain_fraction"
        ]
        == 0.0
    )


def test_retry_noise_does_not_shift_later_live_state_streams() -> None:
    first_initial, first_transition = _diffusion_attempt_generators(
        device=torch.device("cpu"),
        training_seed=17,
        live_state_index=9,
        retry_index=0,
    )
    expected_initial = torch.randn((16,), generator=first_initial)
    expected_transition = torch.randn((16,), generator=first_transition)
    for retry_index in range(3):
        retry_initial, retry_transition = _diffusion_attempt_generators(
            device=torch.device("cpu"),
            training_seed=17,
            live_state_index=8,
            retry_index=retry_index,
        )
        torch.randn((16,), generator=retry_initial)
        torch.randn((16,), generator=retry_transition)
    actual_initial, actual_transition = _diffusion_attempt_generators(
        device=torch.device("cpu"),
        training_seed=17,
        live_state_index=9,
        retry_index=0,
    )
    torch.testing.assert_close(
        torch.randn((16,), generator=actual_initial), expected_initial
    )
    torch.testing.assert_close(
        torch.randn((16,), generator=actual_transition), expected_transition
    )
    assert not torch.equal(expected_initial, expected_transition)


def test_sampler_state_round_trip_allows_zero_signal_epochs() -> None:
    state, kwargs = _valid_sampler_state()
    restored = _validate_sampler_state(
        state,
        bucket_count=1,
        expected_bucket_target_counts=[4],
        rollout_groups_per_bucket_visit=2,
        optimizer_step=kwargs["optimizer_step"],
        environment_steps=kwargs["environment_steps"],
        max_sampling_attempts=kwargs["max_sampling_attempts"],
    )
    assert restored["accepted_update_states"] == 2
    assert restored["rejected_sampling_attempts"] == 3
    assert restored["zero_signal_epochs"] == 4
    assert restored["baseline_execution_steps"] == 3


def test_sampler_state_rejects_baseline_execution_drift() -> None:
    state, kwargs = _valid_sampler_state()
    state["baseline_execution_steps"] = 2
    with pytest.raises(OnlineGRPOError, match="baseline execution counters"):
        _validate_sampler_state(
            state,
            bucket_count=1,
            expected_bucket_target_counts=[4],
            rollout_groups_per_bucket_visit=2,
            optimizer_step=kwargs["optimizer_step"],
            environment_steps=kwargs["environment_steps"],
            max_sampling_attempts=kwargs["max_sampling_attempts"],
        )


def test_collection_contract_forbids_sampled_candidate_execution() -> None:
    contract = rollout_collection_contract(JointGRPOOnlineConfig())
    assert contract["version"] == "stage2_joint_grpo_persistent_episode_v7"
    assert contract["comparison_unit"] == "vehicle_mode"
    assert contract["trajectories_per_mode"] == 48
    assert contract["retry_condition"] == "all_vehicle_modes_have_zero_signal"
    assert contract["sampled_candidate_execution"] is False
    assert contract["environment_action"] == "cached_frozen_stage1_argmax_only"


def test_checkpoint_selection_uses_three_validation_mean_and_safety_gate() -> None:
    baseline = {
        "validation/collision_count": 0.0,
        "validation/out_of_road_count": 0.0,
        "validation/S7/simulator_out_count": 0.0,
    }
    history: list[dict[str, float]] = []
    for state, simulator_gain in ((20, 0.1), (40, 0.2), (60, 0.3)):
        validation = {
            "validation/simulator_available": 1.0,
            "validation/simulator_reward_gain": simulator_gain,
            "validation/selected_reward_gain": simulator_gain + 0.1,
            "validation/S7/out_delta": 0.0,
            **baseline,
        }
        eligible, score = _append_validation_selection_event(
            history,
            accepted_update_state=state,
            validation=validation,
            pretrain_validation=baseline,
        )
    assert eligible is True
    assert score == pytest.approx((0.2, 0.3))

    unsafe = {
        "validation/simulator_available": 1.0,
        "validation/simulator_reward_gain": 1.0,
        "validation/selected_reward_gain": 1.0,
        "validation/S7/out_delta": 1.0,
        "validation/collision_count": 0.0,
        "validation/out_of_road_count": 0.0,
        "validation/S7/simulator_out_count": 1.0,
    }
    eligible, _ = _append_validation_selection_event(
        history,
        accepted_update_state=80,
        validation=unsafe,
        pretrain_validation=baseline,
    )
    assert eligible is False


class _OneStepEnv:
    def __init__(self) -> None:
        self.actions: list[dict[str, np.ndarray]] = []

    def step(self, action: dict[str, np.ndarray]):
        self.actions.append(action)
        return "obs", "reward", {"__all__": False}, {"__all__": False}, {}


def test_cached_baseline_helper_optimizes_finalizes_and_steps_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = np.arange(1 * 3 * 8 * 3, dtype=np.float32).reshape(1, 3, 8, 3)
    modes = np.asarray([[1, 2, 3]], dtype=np.int64)
    optimized = raw + 10.0
    optimization = SimpleNamespace(optimized_trajectories=optimized)
    calls: list[tuple[str, np.ndarray]] = []

    def fake_optimize(model_inputs, trajectories, selected_modes, *, optimizer):
        del model_inputs, optimizer
        calls.append(("optimize", np.array(trajectories, copy=True)))
        np.testing.assert_array_equal(selected_modes, modes)
        return optimization

    def fake_finalize(
        rule_maker,
        condition,
        *,
        env,
        scenario,
        values,
        selected_modes,
        optimization,
        optimizer,
    ):
        del rule_maker, condition, env, scenario, values, optimizer
        calls.append(("finalize", np.array(selected_modes, copy=True)))
        return optimization, {"conditioned_rollouts": 1}, None, None

    monkeypatch.setattr(online, "optimize_selected_model_trajectories", fake_optimize)
    monkeypatch.setattr(online, "_finalize_online_rule_action", fake_finalize)
    env = _OneStepEnv()

    step_result, result, diagnostics, _, _ = execute_cached_frozen_baseline(
        env=env,
        rule_maker=object(),
        condition=SimpleNamespace(),
        scenario=("S5_hard_brake_lead", "R1_entry_straight"),
        model_inputs=object(),
        frozen_raw_trajectories=raw,
        frozen_selected_modes=modes,
        optimizer=object(),
    )

    assert step_result[0] == "obs"
    assert result is optimization
    assert diagnostics == {"conditioned_rollouts": 1}
    assert [name for name, _ in calls] == ["optimize", "finalize"]
    assert len(env.actions) == 1
    for role, agent_id in enumerate(online.AGENT_IDS):
        np.testing.assert_array_equal(env.actions[0][agent_id], optimized[0, role])


def test_cached_baseline_helper_has_no_candidate_input_and_rejects_bad_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_optimize(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("must validate cache before optimizer")

    monkeypatch.setattr(online, "optimize_selected_model_trajectories", fake_optimize)
    with pytest.raises(OnlineGRPOError, match="cached frozen baseline"):
        execute_cached_frozen_baseline(
            env=_OneStepEnv(),
            rule_maker=object(),
            condition=SimpleNamespace(),
            scenario=("S5_hard_brake_lead", "R1_entry_straight"),
            model_inputs=object(),
            frozen_raw_trajectories=np.zeros((3, 8, 3), dtype=np.float32),
            frozen_selected_modes=np.zeros((1, 3), dtype=np.int64),
            optimizer=object(),
        )
    assert not called


@pytest.mark.parametrize("has_active_mode", [True, False])
def test_one_update_state_retries_only_if_all_modes_inactive_and_executes_frozen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    has_active_mode: bool,
) -> None:
    executed: list[np.ndarray] = []

    class Writer:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_tensor(self, *args, **kwargs):
            pass

        def add_text(self, *args, **kwargs):
            pass

        def close(self):
            pass

    class Env:
        _last_planner_batch = {}

        def __init__(self):
            self.step_calls = 0

        def step(self, action):
            self.step_calls += 1
            return {}, {}, {"__all__": False}, {"__all__": False}, {}

        def close(self):
            pass

    fields = {
        "mode_valid_mask": np.ones((3, 10), dtype=np.bool_),
        "coarse_trajectories": np.zeros((3, 10, 8, 3), dtype=np.float32),
        "ego_state": np.zeros((3, 8), dtype=np.float32),
    }
    values = SimpleNamespace(**fields, as_dict=lambda: fields)

    class Builder:
        def __init__(self, agent_ids):
            del agent_ids

        def reset(self):
            pass

        def capture_state(self, env, timestamp):
            del env, timestamp

        def history_ready(self):
            return True

        def build_model_inputs(self, env):
            del env
            return values

    class Loss:
        advantages = torch.zeros((1, 3, 10, 2), dtype=torch.float32)

        def scalar_metrics(self):
            return {"loss/total": 0.5, "advantage/mean": 0.0}

    class Rollout:
        def __init__(self):
            self.candidate_trajectories = torch.full(
                (1, 3, 10, 2, 8, 3), 99.0, dtype=torch.float32
            )
            self.frozen_candidate_trajectories = torch.full(
                (1, 3, 10, 2, 8, 3), 98.0, dtype=torch.float32
            )
            self.noise_bundle_identity = (17, 0, 0)
            self.signal_mode_mask = np.zeros((3, 10), dtype=np.bool_)

        def with_reward_signals(self, **kwargs):
            _, advantages, signal = _fixed_scale_reward_signals(
                kwargs["current_rewards"].squeeze(0).cpu().numpy(),
                kwargs["frozen_rewards"].squeeze(0).cpu().numpy(),
                kwargs["collision_mask"].squeeze(0).cpu().numpy(),
                kwargs["out_of_drivable_mask"].squeeze(0).cpu().numpy(),
                kwargs["valid_executable_mode_mask"].squeeze(0).cpu().numpy(),
            )
            self.signal_mode_mask = signal
            Loss.advantages = torch.from_numpy(advantages).unsqueeze(0)
            return self

    class Trainer:
        def __init__(self):
            self.config = SimpleNamespace(trajectories_per_mode=2)
            self.planner = SimpleNamespace(
                config=SimpleNamespace(
                    model_version="v2",
                    inference_seed=0,
                )
            )
            self.optimizer_step = 0
            self.sample_calls = 0
            self.frozen_calls = 0
            self.update_masks: list[np.ndarray] = []

        def sample_groups(
            self,
            batch,
            *,
            generator,
            transition_generator,
            noise_bundle_identity,
        ):
            del batch
            assert len(noise_bundle_identity) == 3
            torch.randn((1,), generator=generator)
            torch.randn((1,), generator=transition_generator)
            self.sample_calls += 1
            return Rollout()

        def infer_frozen_pretrain(self, rollout):
            del rollout
            self.frozen_calls += 1
            return {
                "selected_trajectory": torch.full(
                    (1, 3, 8, 3), -7.0, dtype=torch.float32
                ),
                "selected_mode": torch.zeros((1, 3), dtype=torch.int64),
                "all_mode_trajectories": torch.full(
                    (1, 3, 10, 8, 3), -7.0, dtype=torch.float32
                ),
            }

        def update(self, rollout):
            self.update_masks.append(rollout.signal_mode_mask.copy())
            self.optimizer_step += 1
            return SimpleNamespace(
                loss=Loss(),
                optimizer_step=self.optimizer_step,
                total_gradient_norm=1.0,
                gradient_norms={"mode_residual": 1.0},
                clipped_gradient_norms={"mode_residual": 1.0},
                adapter_relative_drifts=(0.0,) * 10,
                post_update_reference_kl=0.0,
                zero_signal=False,
                stability_guard_rejected=False,
                stability_guard_trigger_modes=(),
            )

    trainer = Trainer()

    class RewardBackend:
        def __init__(self, config):
            del config

        def score_pretrain(self, *args):
            return SimpleNamespace(
                rewards=np.zeros((3, 10), dtype=np.float32),
                valid_mode_mask=np.ones((3, 10), dtype=np.bool_),
            )

        def score_candidates(self, env, model_inputs, candidates, *args):
            del env, model_inputs, args
            rewards = np.full((3, 10, 2), -1.0, dtype=np.float32)
            if np.all(candidates == 99.0):
                if has_active_mode:
                    rewards[1, 4] = np.asarray([-1.0, 1.0], dtype=np.float32)
            elif np.all(candidates == 98.0):
                rewards.fill(0.0)
                if has_active_mode:
                    rewards[1, 4] = np.asarray([-2.0, 0.0], dtype=np.float32)
            else:
                raise AssertionError("unexpected candidate source")
            shape = rewards.shape
            components = {
                "progress_score": rewards / np.float32(0.47),
                "gap_penalty": np.zeros(shape, dtype=np.float32),
                "ttc_penalty": np.zeros(shape, dtype=np.float32),
                "road_penalty": np.zeros(shape, dtype=np.float32),
                "comfort_penalty": np.zeros(shape, dtype=np.float32),
                "minimum_background_gap_m": np.ones(shape, dtype=np.float32),
                "minimum_teammate_gap_m": np.ones(shape, dtype=np.float32),
                "minimum_road_margin_m": np.ones(shape, dtype=np.float32),
                "minimum_ttc_s": np.ones(shape, dtype=np.float32),
            }
            return VehicleModeRewardResult(
                rewards=rewards,
                pretrain_rewards=np.zeros((3, 10), dtype=np.float32),
                valid_mode_mask=np.ones((3, 10), dtype=np.bool_),
                unsafe=np.zeros(shape, dtype=np.bool_),
                collision=np.zeros(shape, dtype=np.bool_),
                out_of_drivable=np.zeros(shape, dtype=np.bool_),
                clearance_violation=np.zeros(shape, dtype=np.bool_),
                pretrain_unsafe=np.zeros((3, 10), dtype=np.bool_),
                pretrain_collision=np.zeros((3, 10), dtype=np.bool_),
                pretrain_out_of_drivable=np.zeros((3, 10), dtype=np.bool_),
                pretrain_clearance_violation=np.zeros((3, 10), dtype=np.bool_),
                components=components,
                pretrain_components={},
            )

    env = Env()

    def execute_baseline(**kwargs):
        raw = np.asarray(kwargs["frozen_raw_trajectories"])
        executed.append(raw.copy())
        result = kwargs["env"].step({})
        optimization = SimpleNamespace(optimized_trajectories=raw)
        diagnostics = {
            "conditioned_rollouts": 1,
            "proposal_match_attempts": 0,
            "proposal_matches": 0,
            "condition_failures": 0,
            "forced_safe_stops": 0,
            "s7_feedback_exception_hits": 0,
            "commitment_conditioned_rollouts": 0,
            "commitment_feedback_incompatible": 0,
        }
        return result, optimization, diagnostics, None, None

    def checkpoint_payload(**kwargs):
        return {
            "metrics": dict(kwargs["metrics"]),
            "optimizer_step": trainer.optimizer_step,
            "environment_steps": kwargs["environment_steps"],
            "best_validation_reward": kwargs["best_validation_reward"],
            "best_selected_reward_gain": kwargs["best_selected_reward_gain"],
            "best_checkpoint_sha256": kwargs["best_checkpoint_sha256"],
            "validation_selection_history": list(
                kwargs["validation_selection_history"]
            ),
            "sampler_state": dict(kwargs["sampler_state"]),
        }

    def save_checkpoint(path, payload):
        torch.save(payload, path)
        return path

    monkeypatch.setattr(online, "SummaryWriter", Writer)
    monkeypatch.setattr(online, "_implementation_commit", lambda: "a" * 40)
    monkeypatch.setattr(online, "VehicleModeCounterfactualReward", RewardBackend)
    monkeypatch.setattr(
        online, "_load_trainer", lambda *args, **kwargs: (trainer, {}, "a" * 64)
    )
    monkeypatch.setattr(
        online,
        "_load_grpo_validation_state_bank",
        lambda *args, **kwargs: ({}, "b" * 64),
    )
    monkeypatch.setattr(
        online,
        "_fixed_raw_proxy_and_simulator_validation",
        lambda *args, **kwargs: (
            {
                "validation/vehicle_reward_mean": 1.0,
                "validation/simulator_available": 1.0,
                "validation/simulator_reward_gain": 0.1,
                "validation/selected_reward_gain": 0.1,
                "validation/simulator_collision_count": 0.0,
                "validation/frozen_simulator_collision_count": 0.0,
                "validation/simulator_out_count": 0.0,
                "validation/frozen_simulator_out_count": 0.0,
                "validation/S7/out_count": 0.0,
                "validation/S7/frozen_out_count": 0.0,
                "validation/S7/out_delta": 0.0,
            },
            (),
        ),
    )
    monkeypatch.setattr(online, "_new_env", lambda *args, **kwargs: env)
    monkeypatch.setattr(online, "JointBEVSampleBuilder", Builder)
    monkeypatch.setattr(online, "_new_online_rule_maker", lambda *args: object())
    monkeypatch.setattr(
        online,
        "_condition_online_model_inputs",
        lambda *args, **kwargs: SimpleNamespace(
            model_inputs=values,
            committed_execution_id=None,
            committed_plan_actions=None,
        ),
    )
    monkeypatch.setattr(
        online,
        "_scenario_summary",
        lambda env: {
            "scenario_id": "S5_hard_brake_lead",
            "scenario_realized": True,
            "scenario_triggered": True,
            "scenario_recipes_complete": True,
            "scenario_notes": ["lead_brake_profile"],
            "conflict_evidence": {},
            "route_completion": {},
        },
    )
    monkeypatch.setattr(online, "_sample_rollout_start_offset", lambda *a, **k: (0, 0))
    monkeypatch.setattr(
        online,
        "execution_mode_valid_mask",
        lambda *args, **kwargs: np.ones((3, 10), dtype=np.bool_),
    )
    monkeypatch.setattr(online, "model_inputs_to_batch", lambda *a, **k: {})
    monkeypatch.setattr(online, "execute_cached_frozen_baseline", execute_baseline)
    monkeypatch.setattr(online, "_checkpoint_payload", checkpoint_payload)
    monkeypatch.setattr(online, "save_grpo_checkpoint", save_checkpoint)
    monkeypatch.setattr(
        online,
        "load_grpo_checkpoint",
        lambda path, restored, **kwargs: torch.load(
            path, map_location="cpu", weights_only=False
        ),
    )
    monkeypatch.setattr(
        online,
        "_validate_online_checkpoint_metadata",
        lambda payload, **kwargs: payload["sampler_state"],
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
        online,
        "generate_grpo_plots",
        lambda *args, **kwargs: {name: tmp_path / f"{name}.png" for name in plot_names},
    )
    monkeypatch.setattr(
        online,
        "_round_robin_training_buckets",
        lambda *args: ((online.PRIMARY_S5_S9_SCENARIOS[0], 17),),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = online.JointGRPOTrainingConfig(
        variant="A",
        run_mode="smoke",
        source_checkpoint=tmp_path / "stage1.pt",
        online=JointGRPOOnlineConfig(
            device="cpu",
            trajectories_per_mode=2,
            total_rollout_groups=1,
            environment_steps_per_episode=10,
            rollout_groups_per_bucket_visit=1,
            rollout_start_offset_max_steps=0,
            rollout_start_min_remaining_steps=1,
            validation_interval_rollouts=1,
            advantage_vector_log_interval_rollouts=1,
            max_sampling_attempts_per_state=3,
            max_sampling_attempts_multiplier=3,
        ),
    )

    if has_active_mode:
        report = online.run_joint_grpo_training(config, run_dir=run_dir)
    else:
        with pytest.raises(OnlineGRPOError, match="max_sampling_attempts"):
            online.run_joint_grpo_training(config, run_dir=run_dir)
        report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))

    assert report["accepted_update_states"] == int(has_active_mode)
    assert report["sampling_attempts"] == (1 if has_active_mode else 3)
    assert report["rejected_sampling_attempts"] == (0 if has_active_mode else 3)
    assert report["exhausted_states"] == int(not has_active_mode)
    assert report["baseline_execution_steps"] == 1
    assert trainer.sample_calls == (1 if has_active_mode else 3)
    assert trainer.frozen_calls == 1
    if has_active_mode:
        assert trainer.update_masks[0].sum() == 1
    else:
        assert not trainer.update_masks
    assert len(executed) == env.step_calls == 1
    np.testing.assert_array_equal(executed[0], -7.0)
    if has_active_mode:
        assert report["checkpoint_selection_status"] == "no_eligible_checkpoint"
        assert report["best_checkpoint"] is None
        assert not (run_dir / "checkpoints" / "best.pt").exists()
