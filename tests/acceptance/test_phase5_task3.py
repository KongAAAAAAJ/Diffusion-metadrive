from __future__ import annotations

import copy

import torch
from torch import nn

from train.ma_grpo_trainer import MultiAgentGRPOTrainer


class StubTrajectoryHead(nn.Module):
    def __init__(self, ego_fut_mode: int = 3, num_poses: int = 8):
        super().__init__()
        self.ego_fut_mode = ego_fut_mode
        self._num_poses = num_poses
        self.plan_anchor = nn.Parameter(torch.zeros(ego_fut_mode, num_poses, 3), requires_grad=False)

    def norm_odo(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def denorm_odo(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def bezier_xyyaw(self, xy: torch.Tensor) -> torch.Tensor:
        return torch.cat([xy, torch.zeros_like(xy[..., :1])], dim=-1)


class StubPlanner(nn.Module):
    def __init__(self, ego_fut_mode: int = 3, num_poses: int = 8, num_agents: int = 3):
        super().__init__()
        self.num_agents = num_agents
        self.model = nn.Module()
        self.model._trajectory_head = StubTrajectoryHead(ego_fut_mode=ego_fut_mode, num_poses=num_poses)
        self.scale = nn.Parameter(torch.tensor(0.9))
        self.bias = nn.Parameter(torch.tensor(0.01))

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        outputs = {}
        for agent_id in batch.keys():
            device = batch[agent_id]["status"].device
            base = torch.linspace(0.0, 7.0, 8, device=device).unsqueeze(-1)
            xy = torch.cat([base, torch.zeros_like(base)], dim=-1) * self.scale + self.bias
            outputs[agent_id] = torch.cat([xy, torch.zeros_like(base)], dim=-1)
        return outputs

    def extract_rl_context(self, batch: dict):
        contexts = {}
        for agent_id in batch.keys():
            device = batch[agent_id]["status"].device
            contexts[agent_id] = {"_stub": torch.zeros(1, device=device)}
        return contexts, list(batch.keys())

    def predict_denoised_traj(self, noisy_traj_norm: torch.Tensor, timestep: torch.Tensor, context: dict) -> torch.Tensor:
        t_scale = timestep.float().view(-1, 1, 1, 1) / 1000.0
        return torch.tanh(self.scale * noisy_traj_norm + self.bias + t_scale)


class ToyEnv:
    def __init__(self, num_agents: int = 3, negative_rewards: bool = False):
        self.num_agents = num_agents
        self.negative_rewards = negative_rewards

    def reset(self):
        obs = {}
        for i in range(self.num_agents):
            obs[f"agent{i}"] = {
                "camera": torch.zeros(3, 256, 1024, dtype=torch.float32),
                "lidar": torch.zeros(1, 256, 256, dtype=torch.float32),
                "status": torch.zeros(8, dtype=torch.float32),
                "formation_relation_state": torch.zeros(12, dtype=torch.float32),
            }
        return obs

    def evaluate_trajectory_group(self, agent_id: str, trajectories: torch.Tensor):
        rewards = []
        crash_flags = []
        out_flags = []
        step_infos = []
        for idx, traj in enumerate(trajectories):
            progress = float(traj[-1, 0].item())
            if self.negative_rewards:
                progress = -abs(progress)
            formation_error = float(idx + 1)
            crash = idx == 0
            info = {
                "progress": progress,
                "formation_error": formation_error,
                "min_gap": 10.0 - idx,
                "jerk": 0.05 * idx,
                "delta_steering": 0.02 * idx,
                "crash": crash,
                "out_of_road": False,
            }
            rewards.append(info)
            crash_flags.append(crash)
            out_flags.append(False)
            step_infos.append([info for _ in range(8)])
        return {"step_infos": step_infos, "crash_flags": crash_flags, "out_of_road_flags": out_flags}


def _trainer_config():
    return {
        "group_size": 4,
        "lr": 1e-3,
        "kl_threshold": 5.0,
        "il_weight_default": 0.1,
        "il_weight_no_positive": 1.0,
        "advantage_discount_gamma": 0.8,
        "reward_config": {
            "delta_s_max": 10.0,
            "d_norm": 10.0,
            "d_safe": 8.0,
        },
        "ddim_steps": 4,
        "ddim_eta": 0.5,
    }


def test_ma_grpo_trainer_contract():
    model = StubPlanner()
    ref_model = copy.deepcopy(model)
    env = ToyEnv()
    trainer = MultiAgentGRPOTrainer(model=model, ref_model=ref_model, env=env, config=_trainer_config())
    assert isinstance(trainer, MultiAgentGRPOTrainer)

    rollouts = trainer.collect_group_samples(4)
    assert "agent0" in rollouts
    agent_rollout = rollouts["agent0"]
    assert tuple(agent_rollout["trajectory"].shape) == (4, 8, 3)
    assert agent_rollout["log_prob"].ndim == 2
    assert agent_rollout["reward"].shape == (4,)
    assert isinstance(agent_rollout["diffusion_chain"], list)

    advantages = trainer.compute_advantages(rollouts)
    adv0 = advantages["agent0"]
    assert adv0.ndim == 2
    non_crash_adv = adv0[1:]
    std_ok = torch.allclose(non_crash_adv, torch.zeros_like(non_crash_adv)) or abs(float(non_crash_adv.std().item()) - 1.0) < 0.5
    assert std_ok
    assert torch.all(adv0[0] == -1.0)

    update1 = trainer.update(rollouts)
    assert {"loss", "kl", "mean_reward", "il_weight", "grad_norm", "rl_loss", "il_loss"} <= set(update1.keys())
    assert torch.isfinite(torch.as_tensor(update1["loss"]))
    assert 0.0 <= float(update1["kl"]) < 10.0

    rollouts2 = trainer.collect_group_samples(4)
    update2 = trainer.update(rollouts2)
    assert float(update1["loss"]) != float(update2["loss"])

    neg_env = ToyEnv(negative_rewards=True)
    neg_trainer = MultiAgentGRPOTrainer(model=StubPlanner(), ref_model=StubPlanner(), env=neg_env, config=_trainer_config())
    neg_rollouts = neg_trainer.collect_group_samples(4)
    neg_update = neg_trainer.update(neg_rollouts)
    assert float(neg_update["il_weight"]) == 1.0
    assert torch.isfinite(torch.as_tensor(neg_update["rl_loss"]))
    assert torch.isfinite(torch.as_tensor(neg_update["il_loss"]))
