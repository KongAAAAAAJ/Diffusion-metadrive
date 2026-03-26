from __future__ import annotations

import math
from pathlib import Path

from envs.platoon_env import PlatoonEnv
from models.platoon.platoon_diffusion_planner import PlatoonDiffusionPlanner
from train.train_platoon_rl import build_runtime


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "train" / "platoon_grpo_v2.yaml"
CKPT = Path("/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_3/checkpoints/diffusion-epoch=97.ckpt")


def test_phase5_task8_platoon_mode_runs_two_steps():
    summary = build_runtime(
        mode="platoon",
        config_path=str(CONFIG),
        steps=2,
        render=False,
        checkpoint_dir="checkpoints/platoon_rl_real_test",
        log_dir="logs/platoon_rl_real_test",
        ckpt_path=str(CKPT),
        num_agents=3,
        run_training=True,
    )
    assert int(summary["steps"]) == 2


def test_phase5_task8_uses_real_env():
    runtime = build_runtime(
        mode="platoon",
        config_path=str(CONFIG),
        steps=2,
        render=False,
        checkpoint_dir="checkpoints/platoon_rl_real_test",
        log_dir="logs/platoon_rl_real_test",
        ckpt_path=str(CKPT),
        num_agents=3,
        run_training=False,
    )
    assert isinstance(runtime["trainer"].env, PlatoonEnv)
    runtime["trainer"].env.close()


def test_phase5_task8_uses_real_planner():
    runtime = build_runtime(
        mode="platoon",
        config_path=str(CONFIG),
        steps=2,
        render=False,
        checkpoint_dir="checkpoints/platoon_rl_real_test",
        log_dir="logs/platoon_rl_real_test",
        ckpt_path=str(CKPT),
        num_agents=3,
        run_training=False,
    )
    assert isinstance(runtime["trainer"].model, PlatoonDiffusionPlanner)
    runtime["trainer"].env.close()


def test_phase5_task8_update_loss_finite():
    runtime = build_runtime(
        mode="platoon",
        config_path=str(CONFIG),
        steps=2,
        render=False,
        checkpoint_dir="checkpoints/platoon_rl_real_test",
        log_dir="logs/platoon_rl_real_test",
        ckpt_path=str(CKPT),
        num_agents=3,
        run_training=False,
    )
    trainer = runtime["trainer"]
    try:
        obs = trainer.env.reset()
        rollouts = trainer.collect_group_samples(group_size=2, obs=obs)
        metrics = trainer.update(rollouts)
        assert math.isfinite(float(metrics["loss"]))
    finally:
        trainer.env.close()


def test_phase5_task8_collect_accepts_obs():
    runtime = build_runtime(
        mode="platoon",
        config_path=str(CONFIG),
        steps=2,
        render=False,
        checkpoint_dir="checkpoints/platoon_rl_real_test",
        log_dir="logs/platoon_rl_real_test",
        ckpt_path=str(CKPT),
        num_agents=3,
        run_training=False,
    )
    trainer = runtime["trainer"]
    try:
        obs = trainer.env.reset()
        rollouts = trainer.collect_group_samples(group_size=2, obs=obs)
        assert "agent0" in rollouts
    finally:
        trainer.env.close()


def test_phase5_task8_step_env_with_best_advances_state():
    runtime = build_runtime(
        mode="platoon",
        config_path=str(CONFIG),
        steps=2,
        render=False,
        checkpoint_dir="checkpoints/platoon_rl_real_test",
        log_dir="logs/platoon_rl_real_test",
        ckpt_path=str(CKPT),
        num_agents=3,
        run_training=False,
    )
    trainer = runtime["trainer"]
    try:
        obs = trainer.env.reset()
        before = trainer.env.get_formation_relation_state("agent0").copy()
        rollouts = trainer.collect_group_samples(group_size=2, obs=obs)
        next_obs = trainer.step_env_with_best(rollouts)
        after = trainer.env.get_formation_relation_state("agent0")
        assert next_obs
        assert not (before == after).all()
    finally:
        trainer.env.close()


def test_phase5_task8_toy_mode_backward_compatible():
    summary = build_runtime(
        mode="toy-single",
        config_path=str(CONFIG),
        steps=2,
        render=False,
        checkpoint_dir="checkpoints/platoon_rl_real_test",
        log_dir="logs/toy_single_rl_real_test",
        ckpt_path=str(CKPT),
        num_agents=1,
        run_training=True,
    )
    assert int(summary["steps"]) == 2
