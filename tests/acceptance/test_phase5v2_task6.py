from __future__ import annotations

import copy

import torch
from torch import nn

from train.ma_grpo_trainer import MultiAgentGRPOTrainer
from train.closedloop_executor import ClosedLoopExecutor
from train.joint_group import build_joint_groups, extract_joint_trajectories, select_top_k_candidates
from evaluation.reward_terms import compute_team_reward


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
        self._tick = 0
        self._last_state = {"tick": 0}

    def reset(self):
        self._tick = 0
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
        self._tick += 1
        obs = self.reset()
        reward = {agent_id: 0.0 for agent_id in obs}
        terminated = {agent_id: False for agent_id in obs}
        truncated = {agent_id: False for agent_id in obs}
        terminated["__all__"] = self._tick >= 2
        truncated["__all__"] = False
        info = {}
        for agent_id, traj in actions.items():
            traj_t = torch.as_tensor(traj)
            info[agent_id] = {
                "progress": float(traj_t[0, 0].item()) if traj_t.numel() else 0.0,
                "formation_error": float(abs(traj_t[0, 1].item())) if traj_t.numel() else 0.0,
                "min_gap": 10.0,
                "jerk": 0.0,
                "delta_steering": 0.0,
                "crash": False,
                "out_of_road": False,
            }
        return obs, reward, terminated, truncated, info

    def evaluate_trajectory_group(self, agent_id: str, trajectories):
        del agent_id
        step_infos = []
        crash_flags = []
        out_flags = []
        for sample_idx, traj in enumerate(trajectories):
            info = {
                "progress": float(traj[-1, 0].item()),
                "formation_error": float(sample_idx) * 0.1,
                "min_gap": 10.0,
                "jerk": 0.0,
                "delta_steering": 0.0,
                "crash": sample_idx == 0,
                "out_of_road": False,
            }
            step_infos.append([info for _ in range(8)])
            crash_flags.append(sample_idx == 0)
            out_flags.append(False)
        return {"step_infos": step_infos, "crash_flags": crash_flags, "out_of_road_flags": out_flags}

    def get_state(self):
        return {"tick": self._tick}

    def set_state(self, state):
        self._tick = int(state["tick"])


CONFIG = {
    "group_size": 4,
    "lr": 1e-3,
    "advantage_discount_gamma": 0.8,
    "reward_config": {"delta_s_max": 10.0, "d_norm": 10.0, "d_safe": 8.0},
    "ddim_steps": 4,
    "ddim_eta": 0.5,
    "lambda_local": 0.7,
    "lambda_team": 0.3,
    "joint_top_k": 2,
    "num_joint_groups": 2,
    "use_closedloop": True,
}


def _trainer():
    model = StubPlanner()
    ref_model = copy.deepcopy(model)
    ref_model.bias.data.add_(0.05)
    return MultiAgentGRPOTrainer(model=model, ref_model=ref_model, env=ToyEnv(), config=CONFIG)


def test_compute_team_advantages_shape_and_sparse_assignment():
    trainer = _trainer()
    rollouts = trainer.collect_group_samples(group_size=4)
    joint_groups = [{"agent0": (1, 1), "agent1": (2, 1)}]
    team_rewards = [2.0]

    team_adv = trainer.compute_team_advantages(rollouts, joint_groups, team_rewards)

    assert team_adv["agent0"].shape == (4, 2, trainer.scheduler.step_num)
    assert torch.all(team_adv["agent0"][0, 0] == 0)
    assert torch.any(team_adv["agent0"][1, 1] != 0)


def test_compute_team_advantages_marks_crash_group_negative():
    trainer = _trainer()
    rollouts = trainer.collect_group_samples(group_size=4)
    joint_groups = [{"agent0": (0, 0), "agent1": (0, 0)}]
    team_adv = trainer.compute_team_advantages(rollouts, joint_groups, [1.0])
    assert torch.all(team_adv["agent0"][0, 0] == -1.0)


def test_compute_combined_advantages_uses_lambda_weights():
    trainer = _trainer()
    local = {"agent0": torch.ones(2, 2, trainer.scheduler.step_num)}
    team = {"agent0": torch.full((2, 2, trainer.scheduler.step_num), 2.0)}

    combined = trainer.compute_combined_advantages(local, team)

    expected = 0.7 * local["agent0"] + 0.3 * team["agent0"]
    assert torch.allclose(combined["agent0"], expected)


def test_update_supports_joint_groups_none():
    trainer = _trainer()
    rollouts = trainer.collect_group_samples(group_size=4)
    plain_metrics = trainer.update(rollouts, joint_groups=None, team_rewards=None)
    assert torch.isfinite(torch.tensor(plain_metrics["loss"]))


def test_real_closedloop_pipeline_and_three_updates():
    trainer = _trainer()
    obs = trainer.env.reset()
    executor_results = None
    team_rewards = None

    for step_idx in range(3):
        rollouts = trainer.collect_group_samples(group_size=4, obs=obs)
        per_agent_candidates = {}
        for agent_id in rollouts["agent_ids"]:
            per_agent_candidates[agent_id] = select_top_k_candidates(
                rollouts[agent_id]["reward_per_anchor"],
                rollouts[agent_id]["crash_per_anchor"],
                top_k=trainer.config.get("joint_top_k", 2),
            )
        joint_groups = build_joint_groups(per_agent_candidates, num_groups=trainer.config.get("num_joint_groups", 2), seed=step_idx)
        joint_trajectories = [extract_joint_trajectories(group, rollouts) for group in joint_groups]
        executor = ClosedLoopExecutor(trainer.env, trainer.reward_config)
        executor_results = executor.execute_joint_groups(joint_trajectories)
        team_rewards = [compute_team_reward(result["step_infos"], trainer.reward_config) for result in executor_results]
        metrics = trainer.update(rollouts, joint_groups=joint_groups, team_rewards=team_rewards)
        obs = trainer.step_env_with_best(rollouts)
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert metrics["loss"] == metrics["loss"]

    assert executor_results is not None and len(executor_results) == trainer.config.get("num_joint_groups", 2)
    assert all("step_infos" in result and "crash_flags" in result for result in executor_results)
    assert isinstance(team_rewards, list)
    assert len(team_rewards) == trainer.config.get("num_joint_groups", 2)
    assert all(isinstance(value, float) for value in team_rewards)
