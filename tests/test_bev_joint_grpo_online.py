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
    episode_has_ended,
    joint_trajectory_action,
    optimize_selected_model_trajectories,
    run_joint_grpo_training,
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


def test_calibration_scenario_routes_match_runtime_contract() -> None:
    for scenario_id, route in PRIMARY_S5_S9_SCENARIOS:
        assert route in SCENARIO_BY_ID[scenario_id].allowed_local_routes
        assert route in SCENARIO_BY_ID[scenario_id].trigger_by_local_route
    assert JointGRPOOnlineConfig(device="cpu").scenarios == PRIMARY_S5_S9_SCENARIOS


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
