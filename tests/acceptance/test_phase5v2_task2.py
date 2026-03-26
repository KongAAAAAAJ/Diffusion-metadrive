from __future__ import annotations

import copy
import inspect

import pytest
import torch
from torch import nn

from train.ma_grpo_trainer import MultiAgentGRPOTrainer


class StubTrajectoryHead(nn.Module):
    def __init__(self, ego_fut_mode: int = 2, num_poses: int = 8):
        super().__init__()
        self.ego_fut_mode = ego_fut_mode
        self.plan_anchor = nn.Parameter(torch.zeros(ego_fut_mode, num_poses, 3), requires_grad=False)

    def norm_odo(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def denorm_odo(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def bezier_xyyaw(self, xy: torch.Tensor) -> torch.Tensor:
        return torch.cat([xy, torch.zeros_like(xy[..., :1])], dim=-1)


class StubPlanner(nn.Module):
    def __init__(self, ego_fut_mode: int = 2):
        super().__init__()
        self.model = nn.Module()
        self.model._trajectory_head = StubTrajectoryHead(ego_fut_mode=ego_fut_mode)
        self.scale = nn.Parameter(torch.tensor(0.9))
        self.bias = nn.Parameter(torch.tensor(0.01))

    def extract_rl_context(self, batch: dict):
        contexts = {}
        for agent_id in batch.keys():
            device = batch[agent_id]["status"].device
            contexts[agent_id] = {"_stub": torch.zeros(1, device=device)}
        return contexts, list(batch.keys())

    def predict_denoised_traj(self, noisy_traj_norm: torch.Tensor, timestep: torch.Tensor, context: dict) -> torch.Tensor:
        del context
        t_scale = timestep.float().view(-1, 1, 1, 1) / 1000.0
        return torch.tanh(self.scale * noisy_traj_norm + self.bias + t_scale)


class ToyEnv:
    def __init__(self, num_agents: int = 2):
        self.num_agents = num_agents
        self.last_actions = None

    def reset(self):
        return {
            f"agent{i}": {
                "camera": torch.zeros(3, 256, 1024),
                "lidar": torch.zeros(1, 256, 256),
                "status": torch.zeros(8),
                "formation_relation_state": torch.zeros(12),
            }
            for i in range(self.num_agents)
        }

    def step(self, actions):
        self.last_actions = actions
        obs = self.reset()
        reward = {agent_id: 0.0 for agent_id in obs}
        terminated = {agent_id: False for agent_id in obs}
        truncated = {agent_id: False for agent_id in obs}
        terminated["__all__"] = False
        truncated["__all__"] = False
        info = {agent_id: {} for agent_id in obs}
        return obs, reward, terminated, truncated, info

    def evaluate_trajectory_group(self, agent_id: str, trajectories):
        del agent_id
        step_infos = []
        crash_flags = []
        out_flags = []
        for sample_idx, traj in enumerate(trajectories):
            progress = float(traj[-1, 0].item()) + float(abs(traj[:, 1]).mean().item())
            crash = sample_idx == 0
            info = {
                "progress": progress,
                "formation_error": float(sample_idx) + 0.5,
                "min_gap": 10.0 - sample_idx,
                "jerk": 0.0,
                "delta_steering": 0.0,
                "crash": crash,
                "out_of_road": False,
            }
            step_infos.append([info for _ in range(8)])
            crash_flags.append(crash)
            out_flags.append(False)
        return {"step_infos": step_infos, "crash_flags": crash_flags, "out_of_road_flags": out_flags}


TRAINER_CONFIG = {
    "group_size": 4,
    "lr": 1e-3,
    "advantage_discount_gamma": 0.8,
    "reward_config": {"delta_s_max": 10.0, "d_norm": 10.0, "d_safe": 8.0},
    "ddim_steps": 4,
    "ddim_eta": 0.5,
}


def _build_trainer():
    return MultiAgentGRPOTrainer(model=StubPlanner(), ref_model=StubPlanner(), env=ToyEnv(), config=TRAINER_CONFIG)


def _synthetic_rollouts(trainer: MultiAgentGRPOTrainer, reward: torch.Tensor, crash: torch.Tensor | None = None) -> dict:
    g, m = reward.shape
    if crash is None:
        crash = torch.zeros(g, m, dtype=torch.bool)
    out = torch.zeros_like(crash)
    chain = [torch.zeros(g, m, 8, 2) for _ in range(trainer.scheduler.step_num + 1)]
    return {
        "agent_ids": ["agent0"],
        "batch": {"agent0": {"status": torch.zeros(8)}},
        "agent0": {
            "reward_per_anchor": reward.clone(),
            "crash_per_anchor": crash.clone(),
            "out_per_anchor": out,
            "diffusion_chain": chain,
            "trajectory_all": torch.zeros(g, m, 8, 3),
            "best_g": 0,
            "best_k": 0,
        },
    }


def test_collect_group_samples_returns_multi_anchor_fields_and_calls_eval_per_anchor(monkeypatch):
    trainer = _build_trainer()
    calls = []

    def wrapped(agent_id, trajs):
        calls.append((agent_id, trajs.shape))
        return trainer.env.evaluate_trajectory_group(agent_id, trajs)

    monkeypatch.setattr(trainer, "_evaluate_trajectory_group", wrapped)
    rollouts = trainer.collect_group_samples(group_size=4)
    agent_rollout = rollouts["agent0"]
    m = agent_rollout["trajectory_all"].shape[1]

    assert agent_rollout["reward_per_anchor"].shape == (4, m)
    assert len(calls) == len(rollouts["agent_ids"]) * m


def test_compute_advantages_is_independent_per_anchor_and_detects_cross_anchor_case():
    trainer = _build_trainer()
    reward = torch.tensor(
        [
            [10.0, -1.0],
            [12.0, -1.0],
            [14.0, -1.0],
            [16.0, -1.0],
        ]
    )
    rollouts = _synthetic_rollouts(trainer, reward)
    advantages = trainer.compute_advantages(rollouts)["agent0"]

    anchor0 = advantages[:, 0, 0]
    anchor1 = advantages[:, 1, 0]
    assert abs(float(anchor0.mean())) < 0.55
    assert torch.allclose(anchor1, torch.zeros_like(anchor1), atol=1e-6)
    assert not torch.allclose(anchor0, anchor1)


def test_crash_advantage_is_minus_one_for_all_steps():
    trainer = _build_trainer()
    reward = torch.tensor([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0], [4.0, 6.0]])
    crash = torch.tensor([[True, False], [False, False], [False, True], [False, False]])
    rollouts = _synthetic_rollouts(trainer, reward, crash)
    advantages = trainer.compute_advantages(rollouts)["agent0"]
    assert torch.all(advantages[0, 0] == -1.0)
    assert torch.all(advantages[2, 1] == -1.0)


def test_best_k_picks_nonzero_anchor_when_it_dominates(monkeypatch):
    trainer = MultiAgentGRPOTrainer(model=StubPlanner(), ref_model=StubPlanner(), env=ToyEnv(num_agents=1), config=TRAINER_CONFIG)

    sample = {
        agent_id: {
            "trajectory": torch.zeros(2, 2, 8, 3),
            "log_prob": torch.zeros(2, 2, trainer.scheduler.step_num),
            "diffusion_chain": [torch.zeros(2, 2, 8, 2) for _ in range(trainer.scheduler.step_num + 1)],
        }
        for agent_id in trainer.env.reset().keys()
    }
    monkeypatch.setattr(trainer.scheduler, "sample_with_log_prob", lambda *args, **kwargs: sample)
    scores = [
        {"step_infos": [[{"progress": 1.0, "formation_error": 0.0, "min_gap": 10.0, "jerk": 0.0, "delta_steering": 0.0, "crash": False, "out_of_road": False}] * 8 for _ in range(2)], "crash_flags": [False, False], "out_of_road_flags": [False, False]},
        {"step_infos": [[{"progress": 5.0, "formation_error": 0.0, "min_gap": 10.0, "jerk": 0.0, "delta_steering": 0.0, "crash": False, "out_of_road": False}] * 8 for _ in range(2)], "crash_flags": [False, False], "out_of_road_flags": [False, False]},
    ]
    monkeypatch.setattr(trainer, "_evaluate_trajectory_group", lambda agent_id, trajs: scores.pop(0))

    rollouts = trainer.collect_group_samples(group_size=2, obs=trainer.env.reset())
    assert rollouts["agent0"]["best_k"] == 1


def test_compute_rl_loss_source_has_no_mode_index_slice_and_backwardable():
    trainer = _build_trainer()
    src = inspect.getsource(MultiAgentGRPOTrainer.compute_rl_loss)
    assert "mode_index" not in src
    assert "replay_all[torch.arange" not in src

    env = ToyEnv()
    model = StubPlanner()
    ref_model = copy.deepcopy(model)
    ref_model.bias.data.add_(0.05)
    trainer = MultiAgentGRPOTrainer(model=model, ref_model=ref_model, env=env, config=TRAINER_CONFIG)
    rollouts = trainer.collect_group_samples(group_size=4)
    advantages = trainer.compute_advantages(rollouts)
    rl_loss = trainer.compute_rl_loss(rollouts, advantages)
    assert torch.isfinite(rl_loss)
    rl_loss.backward()
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad).all()


def test_five_updates_remain_finite():
    env = ToyEnv()
    model = StubPlanner()
    ref_model = copy.deepcopy(model)
    ref_model.bias.data.add_(0.05)
    trainer = MultiAgentGRPOTrainer(model=model, ref_model=ref_model, env=env, config=TRAINER_CONFIG)
    obs = env.reset()
    for _ in range(5):
        rollouts = trainer.collect_group_samples(group_size=4, obs=obs)
        metrics = trainer.update(rollouts)
        obs = trainer.step_env_with_best(rollouts)
        assert torch.isfinite(torch.tensor(metrics["loss"]))
