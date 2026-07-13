from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from envs.platoon_env import PlatoonEnv


class _FakePlatoonEnv(PlatoonEnv):
    @property
    def agents(self):
        return self._fake_agents


class _FakeLane:
    def local_coordinates(self, position):
        return float(position[0]), float(position[1])


def _fake_env() -> PlatoonEnv:
    env = object.__new__(_FakePlatoonEnv)
    lane = _FakeLane()
    env.config = {
        "platoon_reward_enabled": True,
        "platoon_w_safety": 2.0,
        "platoon_w_formation": 2.0,
        "platoon_w_efficiency": 0.5,
        "platoon_w_comfort": 0.1,
        "platoon_collision_penalty": 10.0,
        "platoon_out_of_road_penalty": 5.0,
        "platoon_d_safe": 8.0,
        "platoon_d_norm": 10.0,
        "platoon_delta_s_max": 5.0,
        "platoon_reward_clip": 20.0,
        "initial_speed_km_h": 25.0,
        "headway_time_s": 0.5,
        "vehicle_length_m": 5.0,
    }
    env._agent_ids = ["agent0", "agent1"]
    env._fake_agents = {
        "agent0": SimpleNamespace(
            id="agent0",
            position=np.asarray([2.0, 0.0], dtype=np.float32),
            lane=lane,
            heading_theta=0.0,
            speed_km_h=25.0,
            LENGTH=5.0,
            crash_vehicle=False,
            crash_object=False,
            crash_building=False,
            crash_human=False,
            crash_sidewalk=False,
        ),
        "agent1": SimpleNamespace(
            id="agent1",
            position=np.asarray([-6.0, 0.0], dtype=np.float32),
            lane=lane,
            heading_theta=0.0,
            speed_km_h=25.0,
            LENGTH=5.0,
            crash_vehicle=False,
            crash_object=False,
            crash_building=False,
            crash_human=False,
            crash_sidewalk=False,
        ),
    }
    env._last_progress_refs = {
        "agent0": (lane, 0.0, np.asarray([0.0, 0.0], dtype=np.float32)),
        "agent1": (lane, -8.0, np.asarray([-8.0, 0.0], dtype=np.float32)),
    }
    env._last_actions = {
        "agent0": np.asarray([0.1, 0.2], dtype=np.float32),
        "agent1": np.asarray([0.0, 0.2], dtype=np.float32),
    }
    env._pending_low_level_actions = {
        "agent0": np.asarray([0.2, 0.3], dtype=np.float32),
        "agent1": np.asarray([0.1, 0.3], dtype=np.float32),
    }
    env._platoon_reward_cache = None
    return env


def test_platoon_reward_function_returns_shared_team_reward_and_components():
    env = _fake_env()

    reward0, info0 = env.reward_function("agent0")
    reward1, info1 = env.reward_function("agent1")

    assert reward0 == reward1
    assert info0["platoon_reward"] == reward0
    assert info1["platoon_reward"] == reward1
    for key in (
        "reward_safety",
        "reward_formation",
        "reward_efficiency",
        "reward_comfort",
        "team_min_gap",
        "team_mean_formation_error",
        "team_mean_progress",
        "team_crash_count",
        "team_out_of_road_count",
    ):
        assert key in info0
        assert key in info1


def test_build_info_reuses_reward_cache_without_recomputing_progress():
    env = _fake_env()
    reward, reward_info = env.reward_function("agent0")
    progress_after_reward = {
        agent_id: float(ref[1]) for agent_id, ref in env._last_progress_refs.items()
    }

    info = env._build_info_dict(
        "low_level",
        actions=env._pending_low_level_actions,
        base_info={agent_id: dict(reward_info) for agent_id in env._agent_ids},
    )

    assert {agent_id: float(ref[1]) for agent_id, ref in env._last_progress_refs.items()} == progress_after_reward
    assert info["agent0"]["platoon_reward"] == reward
    assert info["agent1"]["platoon_reward"] == reward
    assert info["agent0"]["progress"] == 2.0
    assert info["agent1"]["progress"] == 2.0


def test_build_info_includes_scenario_orchestrator_summary():
    env = _fake_env()
    env._scenario_orchestrator = SimpleNamespace(
        get_episode_summary=lambda: {
            "scenario_id": "S7_ego_merge_from_ramp",
            "scenario_triggered": True,
            "scenario_realized": False,
            "scenario_trigger_step": 1,
            "scenario_realized_step": None,
            "scenario_notes": ["recipe_not_realized"],
        }
    )

    info = env._build_info_dict(
        "low_level",
        actions=env._pending_low_level_actions,
        base_info={},
    )

    assert info["agent0"]["scenario_id"] == "S7_ego_merge_from_ramp"
    assert info["agent0"]["scenario_triggered"] is True
    assert info["agent0"]["scenario_realized"] is False
    assert info["agent0"]["scenario_notes"] == ["recipe_not_realized"]


def test_build_info_preserves_terminal_info_for_removed_agent():
    env = _fake_env()
    env._agent_roles = {"agent0": "leader", "agent1": "follower"}
    env._fake_agents.pop("agent0")

    info = env._build_info_dict(
        "low_level",
        actions={"agent1": env._pending_low_level_actions["agent1"]},
        base_info={
            "agent0": {
                "crash_vehicle": True,
                "out_of_road": False,
                "episode_length": 17,
            },
            "agent1": {"crash_vehicle": False},
        },
    )

    assert info["agent0"] == {
        "crash_vehicle": True,
        "out_of_road": False,
        "episode_length": 17,
    }
    assert "formation_relation_state" in info["agent1"]
