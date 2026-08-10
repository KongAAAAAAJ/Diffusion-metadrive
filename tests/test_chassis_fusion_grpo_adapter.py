from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from chassis_execution.contracts import (
    CHASSIS_STATE_FIELDS,
    CONTROL_FIELDS,
    ChassisExecutionPrediction,
)
from chassis_execution.fixture import synthetic_command
from chassis_execution.grpo_adapter import (
    ChassisExecutionContext,
    ChassisFusionGRPOAdapter,
    ChassisFusionGRPOError,
)
from chassis_execution.reward import ChassisExecutionRewardEvaluator
from chassis_execution.surrogate import command_to_dense
from models.bev_planner.bev_only_diffusion_planner import BEVPlannerContext
from models.bev_planner.joint_grpo import (
    JointGRPOLossResult,
    JointGRPORollout,
    JointGRPOUpdateResult,
)
from models.bev_planner.trajectory_optimizer import (
    KinematicTrajectoryOptimizer,
    TrajectoryOptimizationError,
)


ROOT = Path(__file__).resolve().parents[1]


def _coarse_and_raw(speed: float = 4.0):
    times = torch.arange(1, 9, dtype=torch.float32) * 0.5
    coarse = torch.zeros((1, 3, 10, 8, 3), dtype=torch.float32)
    coarse[..., 0] = speed * times
    raw = coarse[:, None, :, 0].repeat(1, 4, 1, 1, 1).clone()
    raw[:, 1, ..., 0] *= 0.95
    raw[:, 2, ..., 0] *= 1.05
    raw[:, 3, ..., 1] = 0.05
    return coarse, raw


def _trainer_inputs() -> dict[str, torch.Tensor]:
    coarse, _ = _coarse_and_raw()
    return {
        "bev": torch.zeros((1, 3, 8, 256, 256), dtype=torch.uint8),
        "ego_state": torch.tensor([[[4.0] + [0.0] * 7] * 3], dtype=torch.float32),
        "formation_relation_state": torch.zeros((1, 3, 12), dtype=torch.float32),
        "relation_valid_mask": torch.ones((1, 3, 2), dtype=torch.bool),
        "agent_role": torch.arange(3, dtype=torch.int64).unsqueeze(0),
        "coarse_trajectories": coarse,
        "mode_valid_mask": torch.ones((1, 3, 10), dtype=torch.bool),
    }


def _context() -> ChassisExecutionContext:
    command = synthetic_command(batch_size=1)
    return ChassisExecutionContext(
        initial_state=command.initial_state,
        vehicle_condition=command.vehicle_condition,
        controller_context=command.controller_context,
        controller_mode=command.controller_mode,
        agent_role=command.agent_role,
    )


def _reward_inputs() -> SimpleNamespace:
    bev = np.zeros((3, 8, 256, 256), dtype=np.uint8)
    bev[:, 0] = 255
    relation = np.zeros((3, 12), dtype=np.float32)
    relation[0, 4], relation[0, 10] = -15.74, -31.48
    relation[1, 4], relation[1, 10] = 15.74, -15.74
    relation[2, 4], relation[2, 10] = 31.48, 15.74
    return SimpleNamespace(bev=bev, formation_relation_state=relation)


class _Environment(SimpleNamespace):
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
            actions=None,
            _desired_center_spacing_m=lambda *_: 15.74,
        )

    def step(self, actions):
        self.step_calls += 1
        self.actions = actions
        return "one-physical-transition"


class _FailingStepEnvironment(_Environment):
    def step(self, actions):
        self.step_calls += 1
        self.actions = actions
        raise RuntimeError("physical step failed")


class _CommandFollowingSurrogate(nn.Module):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.fail = fail

    def predict(self, command):
        if self.fail:
            raise RuntimeError("surrogate exploded")
        batch_size, groups = command.tau_cmd.shape[:2]
        trajectory = command_to_dense(
            command.tau_cmd.reshape(-1, 3, 8, 3)
        ).reshape(batch_size, groups, 3, 40, 3)
        prefix = trajectory.shape[:-1]
        chassis = torch.zeros(
            prefix + (len(CHASSIS_STATE_FIELDS),),
            dtype=torch.float32,
            device=trajectory.device,
        )
        chassis[..., 0] = 4.0
        control = torch.zeros(
            prefix + (len(CONTROL_FIELDS),),
            dtype=torch.float32,
            device=trajectory.device,
        )
        variance_t = torch.full_like(trajectory, 0.01)
        variance_c = torch.full_like(chassis, 0.01)
        variance_u = torch.full_like(control, 0.01)
        zeros_t = torch.zeros_like(trajectory)
        zeros_c = torch.zeros_like(chassis)
        zeros_u = torch.zeros_like(control)
        return ChassisExecutionPrediction(
            executed_trajectory_mean=trajectory,
            chassis_state_mean=chassis,
            control_mean=control,
            trajectory_aleatoric_variance=variance_t,
            trajectory_epistemic_variance=zeros_t,
            trajectory_total_variance=variance_t,
            chassis_aleatoric_variance=variance_c,
            chassis_epistemic_variance=zeros_c,
            chassis_total_variance=variance_c,
            control_aleatoric_variance=variance_u,
            control_epistemic_variance=zeros_u,
            control_total_variance=variance_u,
        )


class _Planner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.diffusion_decoder = nn.Linear(1, 1, bias=False)
        self.mode_head = nn.Linear(1, 1, bias=False)


class _Trainer:
    def __init__(self) -> None:
        self.planner = _Planner()
        self.optimizer_step = 0
        self.sample_calls = 0
        self.update_calls = 0
        self.updated_rollout = None
        self.updated_rewards = None

    def sample_groups(self, model_inputs, *, generator):
        del generator
        self.sample_calls += 1
        coarse, raw = _coarse_and_raw()
        return JointGRPORollout(
            context=BEVPlannerContext(
                bev_feature=torch.zeros((1, 1, 1, 1)),
                role_tokens=torch.zeros((1, 3, 1)),
            ),
            coarse_trajectories=coarse,
            mode_valid_mask=model_inputs["mode_valid_mask"],
            chains_normalized=torch.zeros((1, 4, 5, 3, 10, 8, 2)),
            sampled_modes=torch.zeros((1, 4, 3), dtype=torch.int64),
            selected_trajectories=raw,
            old_mode_log_prob=torch.zeros((1, 4)),
            old_trajectory_log_prob=torch.zeros((1, 4, 3)),
        )

    def update(self, rollout, rewards):
        self.update_calls += 1
        self.updated_rollout = rollout
        self.updated_rewards = rewards
        with torch.no_grad():
            self.planner.diffusion_decoder.weight.add_(0.01)
            self.planner.mode_head.weight.add_(0.02)
        self.optimizer_step += 1
        scalar = rewards.mean()
        loss = JointGRPOLossResult(
            total=scalar,
            mode_pg=scalar,
            trajectory_pg=scalar,
            behavior_cloning=scalar,
            mode_reference_kl=scalar,
            trajectory_reference_kl=scalar,
            reference_kl=scalar,
            advantages=rewards,
            new_mode_log_prob=torch.zeros_like(rewards),
            new_trajectory_log_prob=torch.zeros((1, 4, 3)),
        )
        return JointGRPOUpdateResult(
            loss=loss,
            gradient_norms={"diffusion_decoder": 1.0, "mode_head": 1.0},
            total_gradient_norm=1.0,
            optimizer_step=self.optimizer_step,
        )


class _FailingOptimizer(KinematicTrajectoryOptimizer):
    def optimize(self, *_args, **_kwargs):
        raise TrajectoryOptimizationError("optimizer failed")


def _adapter(*, fail_surrogate: bool = False, fail_optimizer: bool = False):
    trainer = _Trainer()
    evaluator = ChassisExecutionRewardEvaluator(
        _CommandFollowingSurrogate(fail=fail_surrogate)
    )
    evaluator.geometric._prediction_planner._predicted_obstacles = lambda *a, **k: []
    optimizer = _FailingOptimizer() if fail_optimizer else KinematicTrajectoryOptimizer()
    return trainer, evaluator, ChassisFusionGRPOAdapter(trainer, optimizer, evaluator)


def test_machine_protocol_freezes_action_ownership_and_one_step() -> None:
    payload = json.loads(
        (ROOT / "schemas" / "chassis_fusion_grpo_adapter_v1.json").read_text()
    )
    assert payload["probability_action"]["name"] == "tau_d"
    assert payload["probability_action"]["allowed_as_surrogate_input"] is False
    assert payload["execution_action"]["name"] == "tau_cmd"
    assert payload["environment_contract"]["candidate_metadrive_branches"] == 0
    assert payload["environment_contract"]["selected_action_env_steps"] == 1
    assert payload["diagnostic_provenance"]["eligible_for_formal_training"] is False


def test_raw_rollout_is_updated_once_and_only_tau_cmd_enters_surrogate() -> None:
    trainer, evaluator, adapter = _adapter()
    env = _Environment()
    result = adapter.run_update(
        env=env,
        trainer_model_inputs=_trainer_inputs(),
        reward_model_inputs=_reward_inputs(),
        chassis_context=_context(),
        generator=torch.Generator().manual_seed(7),
    )
    assert trainer.sample_calls == trainer.update_calls == 1
    assert trainer.updated_rollout is result.rollout
    assert trainer.updated_rewards is result.reward.rewards
    torch.testing.assert_close(result.tau_d, result.rollout.selected_trajectories)
    torch.testing.assert_close(
        result.tau_cmd,
        torch.from_numpy(np.array(result.optimization.optimized_trajectories)),
    )
    assert result.tau_cmd.requires_grad is False
    assert result.reward.rewards.requires_grad is False
    assert result.policy_sha256_before != result.policy_sha256_after
    assert result.metadrive_candidate_branches == 0
    assert evaluator.evaluation_count == 1
    assert env.step_calls == 0


def test_inference_only_candidate_evaluation_does_not_update_policy() -> None:
    trainer, evaluator, adapter = _adapter()
    env = _Environment()
    before = {
        name: value.detach().clone()
        for name, value in trainer.planner.state_dict().items()
    }
    result = adapter.evaluate_candidates(
        env=env,
        trainer_model_inputs=_trainer_inputs(),
        reward_model_inputs=_reward_inputs(),
        chassis_context=_context(),
        generator=torch.Generator().manual_seed(8),
    )
    assert trainer.update_calls == trainer.optimizer_step == 0
    assert evaluator.evaluation_count == 1
    assert result.metadrive_candidate_branches == 0
    assert result.sampling_ms >= 0.0
    assert result.reward_ms >= 0.0
    assert result.total_ms >= result.reward_ms
    for name, value in trainer.planner.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0.0, atol=0.0)
    assert adapter.execute_selected_once(env, result) == "one-physical-transition"
    assert env.step_calls == 1


def test_only_selected_tau_cmd_is_executed_once() -> None:
    _, _, adapter = _adapter()
    env = _Environment()
    result = adapter.run_update(
        env=env,
        trainer_model_inputs=_trainer_inputs(),
        reward_model_inputs=_reward_inputs(),
        chassis_context=_context(),
        generator=torch.Generator().manual_seed(9),
    )
    assert adapter.execute_selected_once(env, result) == "one-physical-transition"
    assert env.step_calls == 1
    assert set(env.actions) == {"agent0", "agent1", "agent2"}
    expected = result.selected_tau_cmd[0].cpu().numpy()
    for role in range(3):
        np.testing.assert_array_equal(env.actions[f"agent{role}"], expected[role])
    with pytest.raises(ChassisFusionGRPOError, match="exactly once"):
        adapter.execute_selected_once(env, result)
    assert env.step_calls == 1


def test_failed_physical_step_is_still_consumed_and_cannot_be_retried() -> None:
    _, _, adapter = _adapter()
    reward_env = _Environment()
    result = adapter.run_update(
        env=reward_env,
        trainer_model_inputs=_trainer_inputs(),
        reward_model_inputs=_reward_inputs(),
        chassis_context=_context(),
        generator=torch.Generator().manual_seed(10),
    )
    failing_env = _FailingStepEnvironment()
    with pytest.raises(RuntimeError, match="physical step failed"):
        adapter.execute_selected_once(failing_env, result)
    with pytest.raises(ChassisFusionGRPOError, match="exactly once"):
        adapter.execute_selected_once(failing_env, result)
    assert failing_env.step_calls == 1


@pytest.mark.parametrize("failure", ["optimizer", "surrogate"])
def test_failure_has_no_reward_fallback_update_or_environment_step(failure: str) -> None:
    trainer, _, adapter = _adapter(
        fail_optimizer=failure == "optimizer",
        fail_surrogate=failure == "surrogate",
    )
    env = _Environment()
    expected = TrajectoryOptimizationError if failure == "optimizer" else RuntimeError
    with pytest.raises(expected):
        adapter.run_update(
            env=env,
            trainer_model_inputs=_trainer_inputs(),
            reward_model_inputs=_reward_inputs(),
            chassis_context=_context(),
            generator=torch.Generator().manual_seed(11),
        )
    assert trainer.update_calls == 0
    assert trainer.optimizer_step == 0
    assert env.step_calls == 0


def test_context_and_online_batch_are_strict() -> None:
    command = synthetic_command(batch_size=1)
    with pytest.raises(ChassisFusionGRPOError, match="agent_role"):
        ChassisExecutionContext(
            initial_state=command.initial_state,
            vehicle_condition=command.vehicle_condition,
            controller_context=command.controller_context,
            controller_mode=command.controller_mode,
            agent_role=torch.tensor([[1, 0, 2]], dtype=torch.int64),
        )
    _, _, adapter = _adapter()
    bad = _trainer_inputs()
    bad["ego_state"] = bad["ego_state"].repeat(2, 1, 1)
    with pytest.raises(ChassisFusionGRPOError, match="ego_state"):
        adapter.run_update(
            env=_Environment(),
            trainer_model_inputs=bad,
            reward_model_inputs=_reward_inputs(),
            chassis_context=_context(),
            generator=torch.Generator().manual_seed(13),
        )
