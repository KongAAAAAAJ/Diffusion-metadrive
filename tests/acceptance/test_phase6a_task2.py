from __future__ import annotations

import inspect

import numpy as np
import pytest

from train.closedloop_executor import ClosedLoopExecutor
from train.train_platoon_rl import ToyEnv

try:
    from envs.platoon_env import PlatoonEnv

    METADRIVE_AVAILABLE = True
except Exception:
    METADRIVE_AVAILABLE = False


def test_restore_state_no_reset():
    source = inspect.getsource(ClosedLoopExecutor._restore_state)
    assert "reset()" not in source


def test_execute_joint_groups_no_reset():
    source = inspect.getsource(ClosedLoopExecutor.execute_joint_groups)
    assert "reset()" not in source


def test_toy_env_joint_groups_functional():
    env = ToyEnv(num_agents=2, mode="platoon")
    executor = ClosedLoopExecutor(env, reward_config={})
    env.reset()
    group1 = {
        "agent0": np.zeros((8, 3), dtype=np.float32),
        "agent1": np.ones((8, 3), dtype=np.float32) * 0.5,
    }
    group2 = {
        "agent0": np.ones((8, 3), dtype=np.float32),
        "agent1": np.zeros((8, 3), dtype=np.float32),
    }
    results = executor.execute_joint_groups([group1, group2])
    assert len(results) == 2
    for result in results:
        assert "step_infos" in result
        assert "crash_flags" in result
        assert "out_of_road_flags" in result
        assert "terminated" in result
        assert "profile" in result


def test_state_restored_after_joint_groups():
    env = ToyEnv(num_agents=2, mode="platoon")
    executor = ClosedLoopExecutor(env, reward_config={})
    env.reset()
    before = env.get_state()
    groups = [
        {"agent0": np.zeros((8, 3), dtype=np.float32), "agent1": np.zeros((8, 3), dtype=np.float32)},
        {"agent0": np.ones((8, 3), dtype=np.float32), "agent1": np.ones((8, 3), dtype=np.float32)},
    ]
    executor.execute_joint_groups(groups)
    after = env.get_state()
    assert before == after


def test_groups_independence():
    env = ToyEnv(num_agents=2, mode="platoon")
    executor = ClosedLoopExecutor(env, reward_config={})
    env.reset()
    safe_group = {
        "agent0": np.ones((8, 3), dtype=np.float32) * 0.5,
        "agent1": np.ones((8, 3), dtype=np.float32) * 0.5,
    }
    crash_group = {
        "agent0": np.zeros((8, 3), dtype=np.float32),
        "agent1": np.ones((8, 3), dtype=np.float32),
    }
    results = executor.execute_joint_groups([safe_group, crash_group])
    assert len(results) == 2
    assert results[0]["crash_flags"].keys() == results[1]["crash_flags"].keys()
    assert results[0]["step_infos"] is not results[1]["step_infos"]


@pytest.mark.skipif(not METADRIVE_AVAILABLE, reason="MetaDrive unavailable")
def test_real_env_performance():
    import train.train_platoon_rl as entry

    summary = entry.build_runtime(
        mode="platoon-closedloop",
        config_path="configs/train/platoon_grpo_v2.yaml",
        steps=3,
        render=False,
        checkpoint_dir="/tmp/phase6a_task2_perf/ckpt",
        log_dir="/tmp/phase6a_task2_perf/logs",
        ckpt_path=entry.DEFAULT_SINGLE_CKPT,
        num_agents=3,
        run_training=True,
    )
    joint_times = summary.get("profile_joint", [])
    assert joint_times
    avg_joint = sum(float(value) for value in joint_times) / len(joint_times)
    assert avg_joint < 10.0, f"joint_mean={avg_joint:.3f}s, should be < 10s"

    for reward in summary.get("team_reward_mean", []):
        assert np.isfinite(float(reward))
