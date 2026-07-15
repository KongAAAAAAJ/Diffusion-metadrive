from __future__ import annotations

import numpy as np

from envs.platoon_env import PlatoonEnv
from envs.diffusion_envs.base_multi_env import BaseMultiEnv


def _build_env() -> PlatoonEnv:
    return PlatoonEnv({"use_render": False})


def test_platoon_env_inherits_base_multi_env():
    assert issubclass(PlatoonEnv, BaseMultiEnv)


def test_reset_returns_three_agent_keys():
    env = _build_env()
    try:
        obs = env.reset()
        assert isinstance(obs, dict)
        assert list(obs.keys()) == ["agent0", "agent1", "agent2"]
    finally:
        env.close()


def test_reset_observations_are_non_none():
    env = _build_env()
    try:
        obs = env.reset()
        for agent_id in ["agent0", "agent1", "agent2"]:
            assert obs[agent_id] is not None
            assert isinstance(obs[agent_id], (dict, np.ndarray))
    finally:
        env.close()


def test_relation_state_shape_is_12():
    env = _build_env()
    try:
        env.reset()
        relation = env.get_formation_relation_state("agent0")
        assert relation.shape == (12,)
    finally:
        env.close()


def test_step_low_level_returns_five_tuple():
    env = _build_env()
    try:
        env.reset()
        actions = {agent_id: np.zeros((2,), dtype=np.float32) for agent_id in env._agent_ids}
        result = env.step(actions)
        assert isinstance(result, tuple)
        assert len(result) == 5
    finally:
        env.close()


def test_step_trajectory_returns_five_tuple():
    env = _build_env()
    try:
        env.reset()
        actions = {agent_id: np.zeros((8, 3), dtype=np.float32) for agent_id in env._agent_ids}
        result = env.step(actions)
        assert isinstance(result, tuple)
        assert len(result) == 5
    finally:
        env.close()


def test_info_contains_control_mode():
    env = _build_env()
    try:
        env.reset()
        actions = {agent_id: np.zeros((2,), dtype=np.float32) for agent_id in env._agent_ids}
        _, _, _, _, info = env.step(actions)
        assert "control_mode" in info["agent0"]
    finally:
        env.close()


def test_info_contains_formation_error_and_min_gap_as_float():
    env = _build_env()
    try:
        env.reset()
        actions = {agent_id: np.zeros((8, 3), dtype=np.float32) for agent_id in env._agent_ids}
        _, _, _, _, info = env.step(actions)
        assert isinstance(info["agent0"]["formation_error"], float)
        assert isinstance(info["agent0"]["min_gap"], float)
    finally:
        env.close()


def test_terminated_and_truncated_contain_all():
    env = _build_env()
    try:
        env.reset()
        actions = {agent_id: np.zeros((2,), dtype=np.float32) for agent_id in env._agent_ids}
        _, _, terminated, truncated, _ = env.step(actions)
        assert "__all__" in terminated
        assert "__all__" in truncated
    finally:
        env.close()


def test_initial_spacing_matches_vehicle_length_plus_time_headway():
    env = _build_env()
    try:
        env.reset()
        target_gap = env._desired_center_spacing_m("agent0", "agent1")
        actual_gap = float(env.agents["agent0"].position[0] - env.agents["agent1"].position[0])
        assert abs(actual_gap - target_gap) < 1.0
    finally:
        env.close()


def test_metadrive_config_spawn_gap_matches_desired_center_spacing():
    env = _build_env()
    try:
        config = env._build_metadrive_config()
        target_gap = env._desired_center_spacing_m()
        agent_configs = config["agent_configs"]

        assert config["platoon_spawn_gap_m"] == target_gap
        assert agent_configs["agent0"]["spawn_longitude"] - agent_configs["agent1"]["spawn_longitude"] == target_gap
        assert agent_configs["agent1"]["spawn_longitude"] - agent_configs["agent2"]["spawn_longitude"] == target_gap
    finally:
        env.close()
