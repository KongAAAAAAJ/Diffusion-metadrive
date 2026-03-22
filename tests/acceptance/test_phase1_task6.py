from __future__ import annotations

import math

import numpy as np

from envs.platoon_env import PlatoonEnv
from metadrive.policy.idm_policy import IDMPolicy


def _build_env() -> PlatoonEnv:
    return PlatoonEnv({"use_render": False, "enable_idm_lane_change": False})


def _build_idm_policy(vehicle, target_speed_km_h: float) -> IDMPolicy:
    policy = IDMPolicy(vehicle, random_seed=0)
    policy.TIME_WANTED = 0.05
    policy.DISTANCE_WANTED = 0.5
    policy.NORMAL_SPEED = 18.0
    policy.target_speed = 18.0
    policy.ACC_FACTOR = 0.8
    policy.DEACC_FACTOR = -5.0
    return policy


def _step_idm_rollout(env: PlatoonEnv, num_steps: int = 3) -> dict[str, dict]:
    env.reset()
    env.config["enable_idm_lane_change"] = False
    env.engine.global_config["enable_idm_lane_change"] = False
    policies = {
        agent_id: _build_idm_policy(env.agents[agent_id], env.platoon_config.target_speed_km_h)
        for agent_id in env._agent_ids
        if agent_id in env.agents
    }
    info: dict[str, dict] = {}
    for _ in range(num_steps):
        actions = {}
        for agent_id in env._agent_ids:
            if agent_id in env.agents and agent_id not in policies:
                policies[agent_id] = _build_idm_policy(env.agents[agent_id], env.platoon_config.target_speed_km_h)
            policy = policies.get(agent_id)
            actions[agent_id] = np.asarray(policy.act(), dtype=np.float32) if policy is not None else np.zeros((2,), dtype=np.float32)
        _, _, _, _, info = env.step(actions)
    return info


def test_info_contains_required_phase5_fields():
    env = _build_env()
    try:
        info = _step_idm_rollout(env, num_steps=1)
        required = {
            "formation_error": float,
            "min_gap": float,
            "crash": bool,
            "arrive_dest": bool,
            "out_of_road": bool,
            "progress": float,
            "jerk": float,
            "delta_steering": float,
            "speed_km_h": float,
        }
        for agent_id, agent_info in info.items():
            for key, expected_type in required.items():
                assert key in agent_info, (agent_id, key, agent_info)
                assert isinstance(agent_info[key], expected_type), (agent_id, key, type(agent_info[key]))
    finally:
        env.close()


def test_progress_is_positive_after_three_idm_steps():
    env = _build_env()
    try:
        info = _step_idm_rollout(env, num_steps=3)
        for agent_info in info.values():
            assert agent_info["progress"] > 0.0, agent_info
    finally:
        env.close()


def test_jerk_is_finite_after_three_idm_steps():
    env = _build_env()
    try:
        info = _step_idm_rollout(env, num_steps=3)
        for agent_info in info.values():
            assert math.isfinite(agent_info["jerk"]), agent_info
    finally:
        env.close()


def test_delta_steering_is_finite_after_three_idm_steps():
    env = _build_env()
    try:
        info = _step_idm_rollout(env, num_steps=3)
        for agent_info in info.values():
            assert math.isfinite(agent_info["delta_steering"]), agent_info
    finally:
        env.close()
