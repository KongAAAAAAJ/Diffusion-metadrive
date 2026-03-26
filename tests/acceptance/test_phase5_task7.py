from __future__ import annotations

import copy

import numpy as np

from envs.platoon_env import PlatoonEnv
from evaluation.reward_terms import compute_trajectory_reward


def _build_env() -> PlatoonEnv:
    return PlatoonEnv(
        {
            "observation_mode": "multimodal",
            "use_render": False,
            "num_agents": 3,
            "horizon": 100,
            "traffic_density": 0.04,
            "num_scenarios": 1,
        }
    )


def _candidate_trajectories() -> np.ndarray:
    base_x = np.linspace(0.0, 8.0, 8, dtype=np.float32)
    safe = np.stack([base_x, np.zeros_like(base_x), np.zeros_like(base_x)], axis=-1)
    collide = np.stack([np.linspace(0.0, -10.0, 8, dtype=np.float32), np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32)], axis=-1)
    offroad = np.stack([base_x, np.full(8, 5.0, dtype=np.float32), np.zeros(8, dtype=np.float32)], axis=-1)
    return np.stack([safe, collide, offroad], axis=0)


def test_phase5_task7_method_exists():
    env = _build_env()
    try:
        env.reset()
        assert hasattr(env, "evaluate_trajectory_group")
    finally:
        env.close()


def test_phase5_task7_return_structure():
    env = _build_env()
    try:
        env.reset()
        result = env.evaluate_trajectory_group("agent0", _candidate_trajectories())
        assert set(result.keys()) == {"step_infos", "crash_flags", "out_of_road_flags"}
        assert len(result["step_infos"]) == 3
        assert len(result["crash_flags"]) == 3
        assert len(result["out_of_road_flags"]) == 3
        assert all(len(step_infos) == 8 for step_infos in result["step_infos"])
    finally:
        env.close()


def test_phase5_task7_step_info_keys():
    env = _build_env()
    try:
        env.reset()
        result = env.evaluate_trajectory_group("agent0", _candidate_trajectories())
        sample = result["step_infos"][0][0]
        required = {"progress", "formation_error", "min_gap", "jerk", "delta_steering", "crash", "out_of_road"}
        assert required.issubset(sample.keys())
    finally:
        env.close()


def test_phase5_task7_collision_logic():
    env = _build_env()
    try:
        env.reset()
        result = env.evaluate_trajectory_group("agent0", _candidate_trajectories())
        assert any(bool(flag) for flag in result["crash_flags"])
    finally:
        env.close()


def test_phase5_task7_reward_differs_across_trajectories():
    env = _build_env()
    try:
        env.reset()
        result = env.evaluate_trajectory_group("agent0", _candidate_trajectories())
        rewards = [compute_trajectory_reward(step_infos, {}) for step_infos in result["step_infos"]]
        assert len(set(round(value, 4) for value in rewards)) > 1
    finally:
        env.close()


def test_phase5_task7_env_state_unchanged():
    env = _build_env()
    try:
        env.reset()
        before = copy.deepcopy(env.get_formation_relation_state("agent0"))
        env.evaluate_trajectory_group("agent0", _candidate_trajectories())
        after = env.get_formation_relation_state("agent0")
        assert np.allclose(before, after)
    finally:
        env.close()
