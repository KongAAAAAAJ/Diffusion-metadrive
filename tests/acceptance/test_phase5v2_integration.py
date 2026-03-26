from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import pytest
import torch
import yaml

from evaluation.reward_terms import compute_team_reward
from train.closedloop_executor import ClosedLoopExecutor
from train.joint_group import build_joint_groups, extract_joint_trajectories, select_top_k_candidates
from train.ma_grpo_trainer import MultiAgentGRPOTrainer
import train.train_platoon_rl as entry

try:
    from envs.platoon_env import PlatoonEnv  # noqa: F401
    METADRIVE_AVAILABLE = True
except Exception:
    METADRIVE_AVAILABLE = False


def _make_toy_trainer(env=None, config=None):
    base_config = {
        "group_size": 2,
        "lr": 5e-5,
        "ddim_steps": 4,
        "ddim_eta": 0.02,
        "advantage_discount_gamma": 0.8,
        "max_grad_norm": 5.0,
        "total_steps": 10,
        "beta_reg_max": 1.0,
        "beta_reg_min": 0.1,
        "beta_reg_warmup_frac": 0.3,
        "beta_reg_decay_frac": 0.4,
        "reward_config": {
            "delta_s_max": 1.0,
            "d_norm": 10.0,
            "d_safe": 8.0,
            "w_progress": 1.0,
            "w_formation": 0.0,
            "w_safety": 0.0,
            "w_collision": 10.0,
            "w_road": 5.0,
            "w_comfort": 0.0,
            "w_team_formation": 0.5,
            "w_team_safety": 1.0,
            "w_team_efficiency": 0.3,
            "w_team_collision": 10.0,
        },
    }
    if config:
        base_config.update(config)
    model = entry.ToyPlanner(num_agents=1)
    ref_model = copy.deepcopy(model)
    env = env or entry.ToyEnv(num_agents=1, mode="toy-single")
    return MultiAgentGRPOTrainer(model=model, ref_model=ref_model, env=env, config=base_config)


def _assert_numeric_metrics_are_finite(summary: dict, keys: list[str]):
    for key in keys:
        assert key in summary
        value = summary[key]
        if isinstance(value, list):
            assert value, f"{key} should not be empty"
            for item in value:
                assert math.isfinite(float(item)), f"{key} contains non-finite value {item}"
        else:
            assert math.isfinite(float(value)), f"{key} is non-finite: {value}"


def test_ref_reg_replaces_il():
    trainer = _make_toy_trainer()
    with torch.no_grad():
        trainer.model.scale.add_(0.05)
    obs = trainer.env.reset()
    rollouts = trainer.collect_group_samples(group_size=2, obs=obs)
    metrics = trainer.update(rollouts)
    assert "ref_reg_loss" in metrics
    assert "il_loss" not in metrics
    assert 0.0 < metrics["ref_reg_loss"] < 100.0
    assert trainer.config["beta_reg_min"] <= metrics["beta_reg"] <= trainer.config["beta_reg_max"]


class ControlledToyEnv(entry.ToyEnv):
    def __init__(self):
        super().__init__(num_agents=1, mode="toy-single")
        self.eval_calls = 0

    def evaluate_trajectory_group(self, agent_id: str, trajectories: torch.Tensor):
        del agent_id, trajectories
        reward_table = [
            [1.0, 3.0],
            [10.0, 11.0],
            [-5.0, -5.0],
        ]
        crash_table = [
            [False, False],
            [False, False],
            [True, True],
        ]
        mode_idx = self.eval_calls
        self.eval_calls += 1
        rewards = reward_table[mode_idx]
        crashes = crash_table[mode_idx]
        step_infos = []
        for reward in rewards:
            step_infos.append([
                {
                    "progress": reward,
                    "formation_error": 0.0,
                    "min_gap": 20.0,
                    "jerk": 0.0,
                    "delta_steering": 0.0,
                    "crash": False,
                    "out_of_road": False,
                }
            ])
        return {
            "step_infos": step_infos,
            "crash_flags": crashes,
            "out_of_road_flags": [False, False],
        }


def test_intra_anchor_advantage():
    trainer = _make_toy_trainer(env=ControlledToyEnv())
    obs = trainer.env.reset()
    rollouts = trainer.collect_group_samples(group_size=2, obs=obs)
    advantages = trainer.compute_advantages(rollouts)["agent0"]
    assert torch.allclose(advantages[:, 0, :], advantages[:, 1, :], atol=1e-5)
    assert torch.all(advantages[:, 2, :] == -1.0)


def test_env_state_roundtrip_toy():
    env = entry.ToyEnv(num_agents=1, mode="toy-single")
    env.reset()
    saved_state = env.get_state()
    action = {"agent0": torch.zeros(8, 3, dtype=torch.float32).numpy()}
    env.step(action)
    env.set_state(saved_state)


def test_closedloop_executor():
    env = entry.ToyEnv(num_agents=2, mode="platoon")
    executor = ClosedLoopExecutor(env, reward_config={})
    before = env.get_state()
    joint_actions = {
        "agent0": torch.zeros(8, 3, dtype=torch.float32).numpy(),
        "agent1": torch.ones(8, 3, dtype=torch.float32).numpy(),
    }
    result = executor.execute_joint_trajectory(joint_actions)
    assert set(result.keys()) == {"step_infos", "crash_flags", "out_of_road_flags", "terminated", "profile"}
    assert "restore_reset" in result["profile"]
    assert "restore_set_state" in result["profile"]
    assert "step_execution" in result["profile"]
    assert env.get_state() == before


def test_joint_group_building():
    per_agent_candidates = {
        "agent0": [(0, 0), (1, 1)],
        "agent1": [(0, 1), (1, 0)],
        "agent2": [(0, 0), (0, 1)],
    }
    groups = build_joint_groups(per_agent_candidates, num_groups=4, seed=7)
    assert 1 <= len(groups) <= 4
    for group in groups:
        assert set(group.keys()) == set(per_agent_candidates.keys())


def test_team_reward():
    safe_infos = {
        "agent0": [{"progress": 5.0, "formation_error": 0.2, "crash": False}],
        "agent1": [{"progress": 5.2, "formation_error": 0.1, "crash": False}],
    }
    crash_infos = {
        "agent0": [{"progress": 5.0, "formation_error": 0.2, "crash": True}],
        "agent1": [{"progress": 5.2, "formation_error": 0.1, "crash": False}],
    }
    config = yaml.safe_load(Path("configs/train/platoon_grpo_v2.yaml").read_text(encoding="utf-8"))["reward_config"]
    safe_reward = compute_team_reward(safe_infos, config)
    crash_reward = compute_team_reward(crash_infos, config)
    assert safe_reward > crash_reward


def test_layered_advantage():
    trainer = _make_toy_trainer(config={"lambda_local": 0.7, "lambda_team": 0.3})
    local = {"agent0": torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])}
    team = {"agent0": torch.tensor([[[0.0, 1.0], [2.0, 0.0]]])}
    combined = trainer.compute_combined_advantages(local, team)["agent0"]
    expected = 0.7 * local["agent0"] + 0.3 * team["agent0"]
    assert torch.allclose(combined, expected)


def test_full_train_loop_toy(tmp_path):
    t0 = time.time()
    summary = entry.build_runtime(
        mode="toy-single",
        config_path="configs/train/platoon_grpo_v2.yaml",
        steps=5,
        render=False,
        checkpoint_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "logs"),
        ckpt_path=entry.DEFAULT_SINGLE_CKPT,
        run_training=True,
    )
    elapsed = time.time() - t0
    _assert_numeric_metrics_are_finite(summary, ["loss", "rl_loss", "ref_reg_loss", "beta_reg", "mean_reward", "kl"])
    assert elapsed < 300, f"toy-single 5 steps took {elapsed:.1f}s, exceeds 5-minute limit"


@pytest.mark.skipif(not METADRIVE_AVAILABLE, reason="MetaDrive environment is unavailable")
def test_full_train_loop_platoon_closedloop(tmp_path):
    summary = entry.build_runtime(
        mode="platoon-closedloop",
        config_path="configs/train/platoon_grpo_v2.yaml",
        steps=3,
        render=False,
        checkpoint_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "logs"),
        ckpt_path=entry.DEFAULT_SINGLE_CKPT,
        num_agents=3,
        run_training=True,
    )
    _assert_numeric_metrics_are_finite(summary, ["loss", "rl_loss", "ref_reg_loss", "beta_reg", "kl", "team_reward_mean"])


def test_config_v2_completeness():
    data = yaml.safe_load(Path("configs/train/platoon_grpo_v2.yaml").read_text(encoding="utf-8"))
    for key in [
        "beta_reg_max",
        "beta_reg_min",
        "lambda_local",
        "lambda_team",
        "joint_top_k",
        "num_joint_groups",
        "use_closedloop",
    ]:
        assert key in data
    reward = data["reward_config"]
    for key in ["w_team_formation", "w_team_safety", "w_team_efficiency", "w_team_collision"]:
        assert key in reward
