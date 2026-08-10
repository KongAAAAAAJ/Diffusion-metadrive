from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from chassis_execution.contracts import ChassisExecutionPrediction
from chassis_execution.fixture import synthetic_command
from chassis_execution.reward import (
    ChassisExecutionRewardConfig,
    ChassisExecutionRewardError,
    ChassisExecutionRewardEvaluator,
)


ROOT = Path(__file__).resolve().parents[1]


def _model_inputs() -> SimpleNamespace:
    relation = np.zeros((3, 12), dtype=np.float32)
    relation[0, 4] = -15.74
    relation[0, 10] = -31.48
    relation[1, 4] = 15.74
    relation[1, 10] = -15.74
    relation[2, 4] = 31.48
    relation[2, 10] = 15.74
    bev = np.zeros((3, 8, 256, 256), dtype=np.uint8)
    bev[:, 0] = 255
    return SimpleNamespace(bev=bev, formation_relation_state=relation)


class _NoStepEnvironment(SimpleNamespace):
    def __init__(self) -> None:
        agents = {}
        for role, x in enumerate((0.0, -15.74, -31.48)):
            agents[f"agent{role}"] = SimpleNamespace(
                name=f"agent{role}",
                position=np.asarray([x, 0.0], dtype=np.float32),
                heading_theta=0.0,
                velocity=np.asarray([4.0, 0.0], dtype=np.float32),
                LENGTH=5.74,
                WIDTH=2.3,
            )
        super().__init__(
            agents=agents,
            step_calls=0,
            _desired_center_spacing_m=lambda *_: 15.74,
        )

    def step(self, *_args, **_kwargs):
        self.step_calls += 1
        raise AssertionError("CF-5 must not execute candidate MetaDrive branches")


def _prediction(
    *,
    out_of_road_group: int | None = None,
    rollover_group: int | None = None,
    uncertain_group: int | None = None,
    soft_uncertainty: tuple[float, float, float, float] | None = None,
) -> ChassisExecutionPrediction:
    times = torch.arange(1, 41, dtype=torch.float32) * 0.1
    trajectory = torch.zeros((1, 4, 3, 40, 3), dtype=torch.float32)
    trajectory[..., 0] = 4.0 * times
    if out_of_road_group is not None:
        trajectory[:, out_of_road_group, :, :, 1] = 40.0
    chassis = torch.zeros((1, 4, 3, 40, 8), dtype=torch.float32)
    chassis[..., 0] = 4.0
    if rollover_group is not None:
        chassis[:, rollover_group, :, :, 7] = 0.9
    control = torch.zeros((1, 4, 3, 40, 3), dtype=torch.float32)
    control[..., 1] = 0.2
    trajectory_aleatoric = torch.full_like(trajectory, 0.01)
    trajectory_epistemic = torch.zeros_like(trajectory)
    if uncertain_group is not None:
        trajectory_epistemic[:, uncertain_group, ..., :2] = 8.99
    if soft_uncertainty is not None:
        for group, variance in enumerate(soft_uncertainty):
            trajectory_epistemic[:, group, ..., :2] = variance
    chassis_aleatoric = torch.full_like(chassis, 0.01)
    chassis_epistemic = torch.zeros_like(chassis)
    control_aleatoric = torch.full_like(control, 0.01)
    control_epistemic = torch.zeros_like(control)
    return ChassisExecutionPrediction(
        executed_trajectory_mean=trajectory,
        chassis_state_mean=chassis,
        control_mean=control,
        trajectory_aleatoric_variance=trajectory_aleatoric,
        trajectory_epistemic_variance=trajectory_epistemic,
        trajectory_total_variance=trajectory_aleatoric + trajectory_epistemic,
        chassis_aleatoric_variance=chassis_aleatoric,
        chassis_epistemic_variance=chassis_epistemic,
        chassis_total_variance=chassis_aleatoric + chassis_epistemic,
        control_aleatoric_variance=control_aleatoric,
        control_epistemic_variance=control_epistemic,
        control_total_variance=control_aleatoric + control_epistemic,
    )


class _FrozenSurrogate(nn.Module):
    def __init__(self, prediction: ChassisExecutionPrediction) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.prediction = prediction
        self.calls = 0

    def predict(self, _command):
        self.calls += 1
        return self.prediction


class _FailingSurrogate(_FrozenSurrogate):
    def predict(self, _command):
        raise RuntimeError("surrogate failure")


def _evaluator(prediction: ChassisExecutionPrediction) -> ChassisExecutionRewardEvaluator:
    evaluator = ChassisExecutionRewardEvaluator(_FrozenSurrogate(prediction))
    evaluator.geometric._prediction_planner._predicted_obstacles = lambda *args, **kwargs: []
    return evaluator


def test_cf5_protocol_freezes_tau_cmd_tau_a_and_no_branch_semantics() -> None:
    payload = json.loads(
        (ROOT / "schemas" / "chassis_execution_reward_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["input"]["trajectory"] == "tau_cmd only"
    assert payload["input"]["raw_tau_d_allowed"] is False
    assert payload["execution"]["geometric_reward_trajectory"].startswith("predicted tau_a")
    assert payload["execution"]["metadrive_candidate_branches"] == 0
    assert payload["output"]["credit_assignment_action"] == "raw tau_d rollout log probability in CF-6"


def test_execution_geometry_chassis_and_uncertainty_are_hard_gated() -> None:
    prediction = _prediction(out_of_road_group=1, rollover_group=2, uncertain_group=3)
    evaluator = _evaluator(prediction)
    env = _NoStepEnvironment()
    result = evaluator.score(env, _model_inputs(), synthetic_command(batch_size=1))
    assert result.unsafe.tolist() == [[False, True, True, True]]
    assert result.geometric_unsafe.tolist() == [[False, True, False, False]]
    assert result.chassis_unsafe.tolist() == [[False, False, True, False]]
    assert result.uncertainty_unsafe.tolist() == [[False, False, False, True]]
    assert -5.0 <= float(result.rewards[0, 0]) <= 5.0
    assert bool((result.rewards[0, 1:] < -20.0).all())
    assert result.rewards.requires_grad is False
    assert evaluator.surrogate.calls == 1
    assert evaluator.metadrive_candidate_branch_count == 0
    assert env.step_calls == 0


def test_reward_uses_predicted_tau_a_not_safe_tau_cmd() -> None:
    evaluator = _evaluator(_prediction(out_of_road_group=0))
    command = synthetic_command(batch_size=1)
    assert float(command.tau_cmd[..., 1].abs().max()) == 0.0
    result = evaluator.score(_NoStepEnvironment(), _model_inputs(), command)
    assert result.geometric_unsafe[0, 0]
    assert result.rewards[0, 0] < -20.0


def test_soft_uncertainty_monotonically_reduces_safe_reward() -> None:
    evaluator = _evaluator(
        _prediction(soft_uncertainty=(0.0, 0.04, 0.16, 0.36))
    )
    result = evaluator.score(
        _NoStepEnvironment(), _model_inputs(), synthetic_command(batch_size=1)
    )
    assert not result.unsafe.any()
    rewards = result.rewards[0].tolist()
    assert rewards[0] > rewards[1] > rewards[2] > rewards[3]


def test_surrogate_must_be_frozen_and_failure_has_no_fallback() -> None:
    trainable = _FrozenSurrogate(_prediction())
    trainable.anchor.requires_grad_(True)
    with pytest.raises(ChassisExecutionRewardError, match="frozen"):
        ChassisExecutionRewardEvaluator(trainable)
    failing = ChassisExecutionRewardEvaluator(_FailingSurrogate(_prediction()))
    failing.geometric._prediction_planner._predicted_obstacles = lambda *args, **kwargs: []
    with pytest.raises(RuntimeError, match="surrogate failure"):
        failing.score(
            _NoStepEnvironment(), _model_inputs(), synthetic_command(batch_size=1)
        )


def test_only_one_online_physical_state_is_accepted() -> None:
    evaluator = _evaluator(_prediction())
    with pytest.raises(ChassisExecutionRewardError, match="one physical state"):
        evaluator.score(
            _NoStepEnvironment(), _model_inputs(), synthetic_command(batch_size=2)
        )


def test_reward_config_is_strict() -> None:
    with pytest.raises(ChassisExecutionRewardError, match="rollover"):
        ChassisExecutionRewardConfig(rollover_index_limit=1.1)
    with pytest.raises(ChassisExecutionRewardError, match="below -5"):
        ChassisExecutionRewardConfig(unsafe_base_reward=-5.0)
