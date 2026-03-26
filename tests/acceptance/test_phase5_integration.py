from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch

from envs.platoon_env import PlatoonEnv, obs_to_tensor
from evaluation.reward_terms import compute_step_reward
from metadrive.policy.diffusion_policy.transfuser_config import build_transfuser_config
from models.platoon.platoon_diffusion_planner import PlatoonDiffusionPlanner
from models.platoon.weight_migration import migrate_single_to_platoon
from train.ma_grpo_trainer import MultiAgentGRPOTrainer


CKPT_PATH = Path("/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_3/checkpoints/diffusion-epoch=97.ckpt")


def _planner_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_env() -> PlatoonEnv:
    return PlatoonEnv(
        {
            "observation_mode": "multimodal",
            "use_render": False,
            "num_agents": 3,
            "horizon": 100,
            "traffic_density": 0.04,
            "num_scenarios": 1,
            "use_hybrid_map": True,
            "hybrid_map_sequence": "SSXCOCSS",
        }
    )


def _make_planner() -> PlatoonDiffusionPlanner:
    assert CKPT_PATH.exists(), CKPT_PATH
    config = build_transfuser_config("small", plan_anchor_path="metadrive/exp_dataset/metadrive_anchors_ppo.npy")
    planner = PlatoonDiffusionPlanner(config, num_vehicles=3).eval()
    planner = migrate_single_to_platoon(str(CKPT_PATH), planner)
    return planner.to(_planner_device())


def _to_numpy_actions(outputs: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {
        agent_id: traj.detach().cpu().numpy().astype(np.float32, copy=False)
        for agent_id, traj in outputs.items()
    }


def test_phase5_full_integration_smoke():
    env = _make_env()
    planner = _make_planner()
    reward_config = {
        "delta_s_max": 10.0,
        "d_norm": 10.0,
        "d_safe": 8.0,
        "w_progress": 1.0,
        "w_formation": 0.5,
        "w_safety": 0.3,
        "w_collision": 10.0,
        "w_road": 5.0,
        "w_comfort": 0.1,
    }
    trainer_config = {
        "group_size": 2,
        "lr": 1e-4,
        "kl_threshold": 5.0,
        "il_weight_default": 0.1,
        "il_weight_no_positive": 1.0,
        "reward_config": reward_config,
        "ddim_steps": 4,
        "ddim_eta": 0.05,
        "advantage_discount_gamma": 0.8,
        "max_grad_norm": 8.0,
    }

    try:
        obs = env.reset()
        batch = {agent_id: obs_to_tensor(sample, device=_planner_device()) for agent_id, sample in obs.items()}

        with torch.no_grad():
            outputs = planner(batch)
        assert set(outputs.keys()) == {"agent0", "agent1", "agent2"}

        actions = _to_numpy_actions(outputs)
        next_obs, _, terminated, truncated, info = env.step(actions)
        assert next_obs
        assert terminated is not None and truncated is not None

        reward = compute_step_reward(info["agent0"], reward_config)
        assert isinstance(reward, float)
        assert math.isfinite(reward)

        current_obs = next_obs
        for _ in range(2):
            batch = {
                agent_id: obs_to_tensor(sample, device=_planner_device())
                for agent_id, sample in current_obs.items()
            }
            with torch.no_grad():
                outputs = planner(batch)
            current_obs, _, terminated, truncated, info = env.step(_to_numpy_actions(outputs))
            assert info
            if terminated.get("__all__", False) or truncated.get("__all__", False):
                break

        trainer = MultiAgentGRPOTrainer(
            model=planner,
            ref_model=copy.deepcopy(planner).to(_planner_device()),
            env=env,
            config=trainer_config,
        )
        rollouts = trainer.collect_group_samples(2)
        assert "agent0" in rollouts
        assert tuple(rollouts["agent0"]["trajectory"].shape[-2:]) == (8, 3)

        advantages = trainer.compute_advantages(rollouts)
        assert "agent0" in advantages

        metrics = trainer.update(rollouts)
        assert math.isfinite(float(metrics["loss"]))
    finally:
        env.close()
