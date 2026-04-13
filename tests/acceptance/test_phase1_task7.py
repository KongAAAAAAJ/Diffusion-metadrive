from __future__ import annotations

from envs.platoon_env import PlatoonEnv
from scenarios.hazard_scenarios import get_hazard_scenario_configs


def test_dynamic_cut_in_scenario_can_reset():
    env = PlatoonEnv({"use_render": False, "hazard_scenario": "dynamic_cut_in"})
    try:
        obs = env.reset()
        assert list(obs.keys()) == ["agent0", "agent1", "agent2"]
    finally:
        env.close()


def test_static_obstacle_detour_sets_traffic_density_to_point_fifteen():
    env = PlatoonEnv({"use_render": False, "hazard_scenario": "static_obstacle_detour"})
    try:
        assert abs(float(env.config["traffic_density"]) - 0.15) < 1e-6
    finally:
        env.close()


def test_none_hazard_scenario_keeps_default_behavior():
    env = PlatoonEnv({"use_render": False, "hazard_scenario": None})
    try:
        assert abs(float(env.config["traffic_density"]) - 0.04) < 1e-6
        assert env.config["hybrid_map_blocks_config"]
    finally:
        env.close()


def test_all_registered_hazard_scenarios_can_be_passed_into_env():
    for scenario in get_hazard_scenario_configs():
        env = PlatoonEnv({"use_render": False, "hazard_scenario": scenario["name"]})
        try:
            obs = env.reset()
            assert isinstance(obs, dict)
        finally:
            env.close()
