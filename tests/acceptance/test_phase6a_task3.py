from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from evaluation.reward_terms import compute_team_reward
from train.closedloop_executor import ClosedLoopExecutor, _execute_single_group_standalone
from train.train_platoon_rl import _build_joint_training_inputs

try:
    from train.closedloop_executor import ParallelClosedLoopExecutor
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False

try:
    from envs.platoon_env import PlatoonEnv
    import train.train_platoon_rl as entry
    METADRIVE_AVAILABLE = True
except Exception:
    METADRIVE_AVAILABLE = False


def _write_temp_config(tmp_path: Path, closedloop_workers: int) -> Path:
    data = yaml.safe_load(Path("configs/train/platoon_grpo_v2.yaml").read_text(encoding="utf-8"))
    data["closedloop_workers"] = int(closedloop_workers)
    path = tmp_path / f"platoon_grpo_v2_workers_{closedloop_workers}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_parallel_executor_importable():
    from train.closedloop_executor import ParallelClosedLoopExecutor

    assert ParallelClosedLoopExecutor is not None


@pytest.mark.skipif(not METADRIVE_AVAILABLE or not PARALLEL_AVAILABLE, reason="MetaDrive or parallel executor unavailable")
def test_parallel_lifecycle():
    env = PlatoonEnv({"num_agents": 3, "traffic_density": 0.04, "num_scenarios": 1, "use_render": False})
    try:
        env.reset()
        executor = ParallelClosedLoopExecutor(
            env=env,
            env_config={
                "observation_mode": "multimodal",
                "use_render": False,
                "num_agents": 3,
                "traffic_density": 0.04,
                "num_scenarios": 1,
                "horizon": 100,
                "use_hybrid_map": True,
                "hybrid_map_sequence": "SSXCOCSS",
            },
            reward_config={},
            num_workers=2,
        )
        executor.start()
        assert len(executor._workers) == 2
        assert all(worker.is_alive() for worker in executor._workers)
        executor.stop()
        assert len(executor._workers) == 0
    finally:
        env.close()


@pytest.mark.skipif(not METADRIVE_AVAILABLE or not PARALLEL_AVAILABLE, reason="MetaDrive or parallel executor unavailable")
def test_closedloop_workers_zero_uses_serial_path(tmp_path):
    config_path = _write_temp_config(tmp_path, closedloop_workers=0)
    runtime = entry.build_runtime(
        mode="platoon-closedloop",
        config_path=str(config_path),
        steps=1,
        render=False,
        checkpoint_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "logs"),
        ckpt_path=entry.DEFAULT_SINGLE_CKPT,
        num_agents=3,
        run_training=False,
    )
    trainer = runtime["trainer"]
    obs = trainer.env.reset()
    trainer.current_obs = obs
    try:
        rollouts = trainer.collect_group_samples(group_size=int(runtime["config"].get("group_size", 2)), obs=obs)
        joint_groups, team_rewards, joint_profile, closedloop_crash_flags = _build_joint_training_inputs(trainer, rollouts)
        assert joint_groups is not None and team_rewards is not None
        assert np.isfinite(float(sum(team_rewards) / len(team_rewards)))
        assert joint_profile.get("group_count", 0.0) >= 1.0
        assert closedloop_crash_flags is not None
        assert not hasattr(trainer, "_parallel_executor") or trainer._parallel_executor is None
    finally:
        runtime["env"].close()


@pytest.mark.skipif(not METADRIVE_AVAILABLE or not PARALLEL_AVAILABLE, reason="MetaDrive or parallel executor unavailable")
def test_parallel_execution_functional():
    env_config = {
        "observation_mode": "multimodal",
        "use_render": False,
        "num_agents": 3,
        "traffic_density": 0.04,
        "num_scenarios": 1,
        "horizon": 100,
        "use_hybrid_map": True,
        "hybrid_map_sequence": "SSXCOCSS",
    }
    env = PlatoonEnv(env_config)
    try:
        env.reset()
        executor = ParallelClosedLoopExecutor(env=env, env_config=env_config, reward_config={}, num_workers=2)
        groups = []
        rng = np.random.default_rng(7)
        for _ in range(4):
            groups.append({f"agent{i}": rng.normal(size=(8, 3)).astype(np.float32) for i in range(3)})
        results = executor.execute_joint_groups(groups)
        assert len(results) == 4
        for result in results:
            assert "step_infos" in result
            assert "crash_flags" in result
            assert "out_of_road_flags" in result
            assert "terminated" in result
            assert "profile" in result
        executor.stop()
    finally:
        env.close()


@pytest.mark.skipif(not METADRIVE_AVAILABLE or not PARALLEL_AVAILABLE, reason="MetaDrive or parallel executor unavailable")
def test_parallel_vs_serial_team_reward_close():
    env_config = {
        "observation_mode": "multimodal",
        "use_render": False,
        "num_agents": 3,
        "traffic_density": 0.04,
        "num_scenarios": 1,
        "horizon": 100,
        "use_hybrid_map": True,
        "hybrid_map_sequence": "SSXCOCSS",
    }
    reward_config = yaml.safe_load(Path("configs/train/platoon_grpo_v2.yaml").read_text(encoding="utf-8"))["reward_config"]
    env = PlatoonEnv(env_config)
    try:
        env.reset()
        rng = np.random.default_rng(11)
        groups = []
        for _ in range(4):
            groups.append({f"agent{i}": rng.normal(size=(8, 3)).astype(np.float32) for i in range(3)})
        serial_results = ClosedLoopExecutor(env=env, reward_config=reward_config).execute_joint_groups(groups)
        parallel_executor = ParallelClosedLoopExecutor(env=env, env_config=env_config, reward_config=reward_config, num_workers=2)
        parallel_results = parallel_executor.execute_joint_groups(groups)
        serial_rewards = [compute_team_reward(result["step_infos"], reward_config) for result in serial_results]
        parallel_rewards = [compute_team_reward(result["step_infos"], reward_config) for result in parallel_results]
        assert len(serial_rewards) == len(parallel_rewards)
        for serial_reward, parallel_reward in zip(serial_rewards, parallel_rewards):
            baseline = max(abs(float(serial_reward)), 1e-6)
            rel_err = abs(float(parallel_reward) - float(serial_reward)) / baseline
            assert rel_err < 0.2, f"serial={serial_reward}, parallel={parallel_reward}, rel_err={rel_err:.3f}"
        parallel_executor.stop()
    finally:
        env.close()


def test_config_has_closedloop_workers():
    data = yaml.safe_load(Path("configs/train/platoon_grpo_v2.yaml").read_text(encoding="utf-8"))
    assert "closedloop_workers" in data


@pytest.mark.skipif(not METADRIVE_AVAILABLE or not PARALLEL_AVAILABLE, reason="MetaDrive or parallel executor unavailable")
def test_parallel_runtime_under_threshold(tmp_path):
    config_path = _write_temp_config(tmp_path, closedloop_workers=4)
    summary = entry.build_runtime(
        mode="platoon-closedloop",
        config_path=str(config_path),
        steps=3,
        render=False,
        checkpoint_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "logs"),
        ckpt_path=entry.DEFAULT_SINGLE_CKPT,
        num_agents=3,
        run_training=True,
    )
    joint_times = [float(value) for value in summary.get("profile_joint", [])]
    assert joint_times
    assert sum(joint_times) / len(joint_times) < 2.0
    assert all(np.isfinite(float(value)) for value in summary.get("team_reward_mean", []))
    assert all(abs(float(value)) < 1e-6 for value in summary.get("profile_joint_restore_reset", []))


def test_execute_single_group_handles_all_agents_detached_assertion():
    class DummyAgentManager:
        def __init__(self):
            self._active_objects = {}

    class DummyEnv:
        def __init__(self):
            self.agent_manager = DummyAgentManager()
            self._last_info = {
                'agent0': {'progress': 0.0, 'formation_error': 9.0, 'min_gap': 0.0},
                'agent1': {'progress': 0.0, 'formation_error': 9.0, 'min_gap': 0.0},
            }

        def step(self, actions):
            del actions
            raise AssertionError('Not enough objects exist!')

    env = DummyEnv()
    result = _execute_single_group_standalone(
        env,
        {
            'agent0': np.zeros((8, 3), dtype=np.float32),
            'agent1': np.zeros((8, 3), dtype=np.float32),
        },
        horizon=8,
    )
    assert result['terminated'] is True
    assert result['crash_flags']['agent0'] is True
    assert result['crash_flags']['agent1'] is True
    assert len(result['step_infos']['agent0']) == 1
