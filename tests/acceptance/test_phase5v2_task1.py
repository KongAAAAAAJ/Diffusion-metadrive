from __future__ import annotations

import copy

import torch
from torch import nn

from models.diffusion.diffusion_rl_scheduler import DiffusionRLScheduler
from train.ma_grpo_trainer import MultiAgentGRPOTrainer


class StubTrajectoryHead(nn.Module):
    def __init__(self, ego_fut_mode: int = 3, num_poses: int = 8):
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
    def __init__(self, ego_fut_mode: int = 3, scale: float = 0.9, bias: float = 0.01):
        super().__init__()
        self.model = nn.Module()
        self.model._trajectory_head = StubTrajectoryHead(ego_fut_mode=ego_fut_mode)
        self.scale = nn.Parameter(torch.tensor(scale))
        self.bias = nn.Parameter(torch.tensor(bias))

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
        obs = self.reset()
        reward = {agent_id: 0.0 for agent_id in obs}
        terminated = {agent_id: False for agent_id in obs}
        truncated = {agent_id: False for agent_id in obs}
        terminated["__all__"] = False
        truncated["__all__"] = False
        info = {agent_id: {} for agent_id in obs}
        self.last_actions = actions
        return obs, reward, terminated, truncated, info

    def evaluate_trajectory_group(self, agent_id: str, trajectories: torch.Tensor):
        del agent_id
        step_infos = []
        crash_flags = []
        out_flags = []
        for idx, traj in enumerate(trajectories):
            info = {
                "progress": float(traj[-1, 0].item()),
                "formation_error": float(idx) * 0.2,
                "min_gap": 10.0,
                "jerk": 0.0,
                "delta_steering": 0.0,
                "crash": False,
                "out_of_road": False,
            }
            step_infos.append([info for _ in range(8)])
            crash_flags.append(False)
            out_flags.append(False)
        return {"step_infos": step_infos, "crash_flags": crash_flags, "out_of_road_flags": out_flags}


CONFIG = {
    "num_train_timesteps": 1000,
    "num_inference_steps": 4,
    "eta": 0.5,
    "prediction_type": "sample",
    "trunc_timestep": 8,
}

TRAINER_CONFIG = {
    "group_size": 3,
    "lr": 1e-3,
    "advantage_discount_gamma": 0.8,
    "reward_config": {"delta_s_max": 10.0, "d_norm": 10.0, "d_safe": 8.0},
    "ddim_steps": 4,
    "ddim_eta": 0.5,
    "beta_reg_max": 1.0,
    "beta_reg_min": 0.2,
    "beta_reg_warmup_frac": 0.3,
    "beta_reg_decay_frac": 0.4,
    "total_steps": 10,
}


def test_compute_ref_predictions_returns_stepwise_predictions():
    scheduler = DiffusionRLScheduler(CONFIG)
    planner = StubPlanner()
    batch = ToyEnv().reset()
    outputs = scheduler.sample_with_log_prob(planner, batch, num_groups=2)
    chain = outputs["agent0"]["diffusion_chain"]

    ref_preds = scheduler.compute_ref_predictions(planner, batch, "agent0", chain)

    assert isinstance(ref_preds, list)
    assert len(ref_preds) == scheduler.step_num
    assert ref_preds[0].shape == chain[0].shape
    assert torch.isfinite(ref_preds[0]).all()


def test_compute_ref_reg_loss_positive_and_backwardable():
    model = StubPlanner(scale=0.9, bias=0.01)
    ref_model = StubPlanner(scale=0.7, bias=-0.02)
    trainer = MultiAgentGRPOTrainer(model=model, ref_model=ref_model, env=ToyEnv(), config=TRAINER_CONFIG)
    rollouts = trainer.collect_group_samples(group_size=3)

    loss = trainer.compute_ref_reg_loss(rollouts)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0

    loss.backward()
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad).all()


def test_beta_reg_schedule_and_update_metrics():
    model = StubPlanner(scale=0.9)
    ref_model = copy.deepcopy(model)
    ref_model.bias.data.add_(0.1)
    trainer = MultiAgentGRPOTrainer(model=model, ref_model=ref_model, env=ToyEnv(), config=TRAINER_CONFIG)

    assert trainer._get_beta_reg() == TRAINER_CONFIG["beta_reg_max"]
    trainer.global_step = TRAINER_CONFIG["total_steps"]
    assert trainer._get_beta_reg() == TRAINER_CONFIG["beta_reg_min"]
    trainer.global_step = 0

    rollouts = trainer.collect_group_samples(group_size=3)
    metrics = trainer.update(rollouts)
    assert "ref_reg_loss" in metrics
    assert "beta_reg" in metrics
    assert "il_loss" not in metrics
    assert torch.isfinite(torch.as_tensor(metrics["ref_reg_loss"]))
