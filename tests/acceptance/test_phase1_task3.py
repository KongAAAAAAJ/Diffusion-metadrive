from __future__ import annotations

from scenarios.hazard_scenarios import get_hazard_scenario_configs


def test_hazard_scenario_configs_returns_list_with_at_least_three_entries():
    configs = get_hazard_scenario_configs()
    assert isinstance(configs, list)
    assert len(configs) >= 3


def test_each_config_has_name_description_and_env_overrides():
    configs = get_hazard_scenario_configs()
    for config in configs:
        assert isinstance(config["name"], str)
        assert isinstance(config["description"], str)
        assert isinstance(config["env_overrides"], dict)


def test_required_scenario_names_are_present():
    names = {config["name"] for config in get_hazard_scenario_configs()}
    assert {"static_obstacle_detour", "dynamic_cut_in", "bottleneck_narrow_bridge"} <= names


def test_each_env_override_has_at_least_one_key():
    for config in get_hazard_scenario_configs():
        assert len(config["env_overrides"]) >= 1


def test_each_description_is_meaningful_length():
    for config in get_hazard_scenario_configs():
        assert len(config["description"]) >= 10
