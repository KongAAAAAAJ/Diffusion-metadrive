from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from train.train_bev_joint_grpo_online import (
    PRIMARY_S5_S9_SCENARIOS,
    JointGRPOOnlineConfig,
    OnlineGRPOError,
    constant_velocity_actions,
    execution_mode_valid_mask,
    episode_has_ended,
    joint_trajectory_action,
    model_inputs_to_batch,
    optimize_selected_model_trajectories,
    run_joint_grpo_training,
    _joint_rewards_are_informative,
    _round_robin_training_buckets,
    _summarize_calibration_tracking,
    _validate_calibration_trajectory_optimizer_contract,
    _scenario_ready_for_primary_sampling,
)
from models.bev_planner.trajectory_optimizer import (
    KinematicTrajectoryOptimizerConfig,
)
from scenarios.definitions import SCENARIO_BY_ID
from scenarios.bev_round13_contract import HOLDOUT_SEEDS, primary_scenario_contract


def _calibration(path: Path, *, variant: str = "A", passed: bool = False) -> Path:
    optimizer_config = KinematicTrajectoryOptimizerConfig()
    path.write_text(
        json.dumps(
            {
                "format": "bev_joint_reward_calibration_v2",
                "calibration_phase": "holdout",
                "variant": variant,
                "passed": passed,
                "blockers": [] if passed else ["stop_terminal_speed"],
                "reward_config": {},
                "scenario_contract": primary_scenario_contract(),
                "scenario_contract_sha256": primary_scenario_contract()["sha256"],
                "scenarios": [list(value) for value in PRIMARY_S5_S9_SCENARIOS],
                "seeds": list(HOLDOUT_SEEDS),
                "trajectory_optimizer_config": optimizer_config.__dict__,
                "trajectory_optimizer_sha256": optimizer_config.sha256(),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_online_config_and_run_mode_are_strict(tmp_path: Path) -> None:
    with pytest.raises(OnlineGRPOError):
        JointGRPOOnlineConfig(device="auto")
    with pytest.raises(OnlineGRPOError):
        JointGRPOOnlineConfig(total_optimizer_steps=0)
    with pytest.raises(OnlineGRPOError, match="complete ordered S5--S9"):
        JointGRPOOnlineConfig(
            scenarios=(("S1_free_cruise_straight", "R3_mainline_straight"),)
        )

    report = _calibration(tmp_path / "failed.json")
    config = JointGRPOOnlineConfig(device="cpu", calibration_report=report)
    with pytest.raises(OnlineGRPOError, match="calibration failed"):
        run_joint_grpo_training(
            config,
            variant="A",
            run_mode="smoke",
            source_checkpoint=tmp_path / "does-not-exist.pt",
            output_root=tmp_path / "output",
            max_optimizer_steps=20,
        )
    with pytest.raises(OnlineGRPOError, match="positive"):
        run_joint_grpo_training(
            config,
            variant="A",
            run_mode="smoke",
            source_checkpoint=tmp_path / "does-not-exist.pt",
            output_root=tmp_path / "output",
            max_optimizer_steps=0,
        )
    with pytest.raises(OnlineGRPOError, match="forbids"):
        run_joint_grpo_training(
            config,
            variant="A",
            run_mode="formal",
            source_checkpoint=tmp_path / "does-not-exist.pt",
            output_root=tmp_path / "output",
            max_optimizer_steps=20,
        )


def test_failed_calibration_bypass_is_explicit_and_smoke_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _calibration(tmp_path / "failed.json")
    config = JointGRPOOnlineConfig(device="cpu", calibration_report=report)

    def reached_source_loader(*args, **kwargs):
        raise OnlineGRPOError("source loader reached")

    monkeypatch.setattr(
        "train.train_bev_joint_grpo_online._load_trainer",
        reached_source_loader,
    )
    with pytest.raises(OnlineGRPOError, match="source loader reached"):
        run_joint_grpo_training(
            config,
            variant="A",
            run_mode="smoke",
            source_checkpoint=tmp_path / "does-not-exist.pt",
            output_root=tmp_path / "output",
            max_optimizer_steps=1,
            allow_failed_calibration_diagnostic=True,
        )
    with pytest.raises(OnlineGRPOError, match="restricted to Variant A smoke"):
        run_joint_grpo_training(
            config,
            variant="A",
            run_mode="formal",
            source_checkpoint=tmp_path / "does-not-exist.pt",
            output_root=tmp_path / "output",
            allow_failed_calibration_diagnostic=True,
        )

    passed_report = _calibration(tmp_path / "passed.json", passed=True)
    passed_config = JointGRPOOnlineConfig(
        device="cpu", calibration_report=passed_report
    )
    with pytest.raises(OnlineGRPOError, match="requires a failed"):
        run_joint_grpo_training(
            passed_config,
            variant="A",
            run_mode="smoke",
            source_checkpoint=tmp_path / "does-not-exist.pt",
            output_root=tmp_path / "output",
            max_optimizer_steps=1,
            allow_failed_calibration_diagnostic=True,
        )


def test_empty_safe_group_tracking_becomes_an_explicit_failed_gate() -> None:
    tracking, lateral_p95, heading_p95, passed = (
        _summarize_calibration_tracking([])
    )
    assert tracking["row_count"] == 0
    assert tracking["blocked_reason"] == "no_closed_loop_safe_group"
    assert tracking["overall"] is None
    assert tracking["recommended_envelope"]["within_controller_limits"] is False
    assert lateral_p95 is None
    assert heading_p95 is None
    assert passed is False

    safe_row = {
        "scenario": "S5_hard_brake_lead",
        "role": 0,
        "longitudinal_errors_m": [0.1, 0.2],
        "lateral_errors_m": [0.05, 0.1],
        "heading_errors_rad": [0.01, 0.02],
    }
    tracking, lateral_p95, heading_p95, passed = (
        _summarize_calibration_tracking([safe_row])
    )
    assert tracking["row_count"] == 1
    assert lateral_p95 == pytest.approx(0.0975)
    assert heading_p95 == pytest.approx(0.0195)
    assert passed is True


def test_calibration_scenario_routes_match_runtime_contract() -> None:
    for scenario_id, route in PRIMARY_S5_S9_SCENARIOS:
        assert route in SCENARIO_BY_ID[scenario_id].allowed_local_routes
        assert route in SCENARIO_BY_ID[scenario_id].trigger_by_local_route
    assert JointGRPOOnlineConfig(device="cpu").scenarios == PRIMARY_S5_S9_SCENARIOS


def test_execution_mask_override_reaches_policy_batch_without_mutating_inputs() -> None:
    coarse = np.zeros((3, 10, 8, 3), dtype=np.float32)
    coarse[..., 0] = (
        4.0 * np.arange(1, 9, dtype=np.float32)[None, None, :] * 0.5
    )
    coarse[:, 9, :, 0] = np.asarray(
        [1.5, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32
    )
    physical_mask = np.ones((3, 10), dtype=np.bool_)
    fields = {
        "bev": np.zeros((3, 8, 256, 256), dtype=np.uint8),
        "ego_state": np.zeros((3, 8), dtype=np.float32),
        "formation_relation_state": np.zeros((3, 12), dtype=np.float32),
        "relation_valid_mask": np.ones((3, 2), dtype=np.bool_),
        "agent_role": np.arange(3, dtype=np.int64),
        "coarse_trajectories": coarse,
        "mode_valid_mask": physical_mask,
    }
    fields["ego_state"][:, 0] = 4.0
    values = SimpleNamespace(
        **fields,
        as_dict=lambda: fields,
    )

    execution_mask = execution_mode_valid_mask(values)
    batch = model_inputs_to_batch(
        values,
        torch.device("cpu"),
        mode_valid_mask=execution_mask,
    )

    assert np.array_equal(values.mode_valid_mask, physical_mask)
    assert torch.equal(
        batch["mode_valid_mask"][0], torch.from_numpy(execution_mask.copy())
    )
    assert bool(batch["mode_valid_mask"][0, :, 9].all())


def test_primary_sampling_waits_for_every_scenario_recipe() -> None:
    class Orchestrator:
        def __init__(self, complete: bool) -> None:
            self.complete = complete

        def get_episode_summary(self):
            return {
                "scenario_id": "S8_ego_exit_to_ramp",
                "scenario_realized": True,
                "scenario_recipes_complete": self.complete,
            }

    env = SimpleNamespace(_scenario_orchestrator=Orchestrator(False))
    assert not _scenario_ready_for_primary_sampling(env)
    env._scenario_orchestrator.complete = True
    assert _scenario_ready_for_primary_sampling(env)


def test_s5_primary_sampling_waits_for_actual_hard_brake() -> None:
    class Orchestrator:
        def __init__(self) -> None:
            self.triggered = False
            self.notes = ["adjacent_spawned:left_side"]

        def get_episode_summary(self):
            return {
                "scenario_id": "S5_hard_brake_lead",
                "scenario_realized": True,
                "scenario_triggered": self.triggered,
                "scenario_notes": list(self.notes),
                "scenario_recipes_complete": False,
            }

    orchestrator = Orchestrator()
    env = SimpleNamespace(_scenario_orchestrator=orchestrator)
    assert not _scenario_ready_for_primary_sampling(env)

    orchestrator.triggered = True
    assert not _scenario_ready_for_primary_sampling(env)

    orchestrator.notes.append("lead_brake_profile")
    assert _scenario_ready_for_primary_sampling(env)


def test_calibration_variant_is_checked_before_source_load(tmp_path: Path) -> None:
    report = _calibration(tmp_path / "wrong.json", variant="B", passed=True)
    config = JointGRPOOnlineConfig(device="cpu", calibration_report=report)
    with pytest.raises(OnlineGRPOError, match="variant mismatch"):
        run_joint_grpo_training(
            config,
            variant="A",
            run_mode="smoke",
            source_checkpoint=tmp_path / "does-not-exist.pt",
            output_root=tmp_path / "output",
            max_optimizer_steps=1,
        )


def test_json_optimizer_contract_is_canonicalized_without_relaxing_hash() -> None:
    config = KinematicTrajectoryOptimizerConfig()
    serialized = json.loads(
        json.dumps(
            {
                "trajectory_optimizer_config": config.__dict__,
                "trajectory_optimizer_sha256": config.sha256(),
            }
        )
    )
    assert _validate_calibration_trajectory_optimizer_contract(serialized) == config

    serialized["trajectory_optimizer_sha256"] = "0" * 64
    with pytest.raises(OnlineGRPOError, match="optimizer contract mismatch"):
        _validate_calibration_trajectory_optimizer_contract(serialized)


def test_constant_joint_rewards_are_skipped_instead_of_faking_an_update() -> None:
    assert not _joint_rewards_are_informative(
        np.asarray([-21.0, -21.0, -21.0, -21.0], dtype=np.float32)
    )
    assert _joint_rewards_are_informative(
        np.asarray([-21.0, -20.0, -21.0, -21.0], dtype=np.float32)
    )
    with pytest.raises(OnlineGRPOError, match=r"float \[4\]"):
        _joint_rewards_are_informative(np.zeros(3, dtype=np.float32))


def test_online_training_buckets_cover_every_scenario_seed_pair() -> None:
    buckets = _round_robin_training_buckets(
        PRIMARY_S5_S9_SCENARIOS, (17, 23)
    )
    assert len(buckets) == 10
    assert len(set(buckets)) == 10
    counts = [0 for _ in buckets]
    for optimizer_step in range(20):
        counts[optimizer_step % len(buckets)] += 1
    assert counts == [2] * 10


def test_online_action_helpers_are_label_free_and_strict() -> None:
    env = SimpleNamespace(
        agents={
            f"agent{role}": SimpleNamespace(speed_km_h=18.0)
            for role in range(3)
        }
    )
    actions = constant_velocity_actions(env)
    assert set(actions) == {"agent0", "agent1", "agent2"}
    assert actions["agent0"].shape == (8, 3)
    assert actions["agent0"][1, 0] == pytest.approx(5.0)

    trajectories = np.zeros((3, 8, 3), dtype=np.float32)
    converted = joint_trajectory_action(trajectories)
    trajectories[0, 0, 0] = 99.0
    assert converted["agent0"][0, 0] == 0.0
    with pytest.raises(OnlineGRPOError):
        joint_trajectory_action(np.zeros((3, 7, 3), dtype=np.float32))

    flags = {"__all__": False}
    assert not episode_has_ended(flags, flags, {})
    assert episode_has_ended(
        flags,
        flags,
        {"agent1": {"out_of_road": True}},
    )


def test_execution_optimizer_keeps_raw_policy_action_immutable() -> None:
    coarse = np.zeros((3, 10, 8, 3), dtype=np.float32)
    coarse[..., 0] = 4.0 * np.arange(1, 9, dtype=np.float32)
    raw = np.broadcast_to(coarse[:, 0], (4, 3, 8, 3)).copy()
    raw[..., 0] *= -1.0
    original = raw.copy()
    values = SimpleNamespace(
        coarse_trajectories=coarse,
        ego_state=np.pad(
            np.full((3, 1), 8.0, dtype=np.float32), ((0, 0), (0, 7))
        ),
    )
    modes = np.zeros((4, 3), dtype=np.int64)

    result = optimize_selected_model_trajectories(values, raw, modes)

    assert np.array_equal(raw, original)
    assert np.array_equal(result.raw_trajectories, original)
    assert result.optimized_valid.all()
    assert not np.array_equal(result.optimized_trajectories, original)
