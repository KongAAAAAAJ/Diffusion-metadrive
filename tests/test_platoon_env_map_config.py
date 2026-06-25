from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from envs import platoon_env as platoon_env_module
from envs.platoon_env import PlatoonEnv, PlatoonEnvConfig
from envs.diffusion_envs.base_multi_env import DEFAULT_HYBRID_MAP_CONFIG


def test_platoon_env_uses_base_multi_env_default_hybrid_map_config() -> None:
    config = PlatoonEnvConfig()

    assert config.hybrid_map_blocks_config == DEFAULT_HYBRID_MAP_CONFIG
    assert config.hybrid_map_blocks_config is not DEFAULT_HYBRID_MAP_CONFIG


def test_platoon_env_default_map_config_is_deep_copied() -> None:
    original = deepcopy(DEFAULT_HYBRID_MAP_CONFIG)
    config = PlatoonEnvConfig()

    config.hybrid_map_blocks_config[0]["length"] = -1.0

    assert DEFAULT_HYBRID_MAP_CONFIG == original


def test_scenario_definition_can_supply_initial_speed_when_not_explicit(monkeypatch) -> None:
    monkeypatch.setattr(
        platoon_env_module,
        "SCENARIO_BY_ID",
        {"S_speed": SimpleNamespace(ego_initial_speed_km_h=18.0)},
        raising=False,
    )
    config = PlatoonEnv._apply_scenario_definition_defaults(
        {"scenario_id": "S_speed", "initial_speed_km_h": 25.0},
        explicit_config_keys={"scenario_id"},
    )

    assert config["initial_speed_km_h"] == 18.0


def test_explicit_initial_speed_overrides_scenario_definition(monkeypatch) -> None:
    monkeypatch.setattr(
        platoon_env_module,
        "SCENARIO_BY_ID",
        {"S_speed": SimpleNamespace(ego_initial_speed_km_h=18.0)},
        raising=False,
    )
    config = PlatoonEnv._apply_scenario_definition_defaults(
        {"scenario_id": "S_speed", "initial_speed_km_h": 31.0},
        explicit_config_keys={"scenario_id", "initial_speed_km_h"},
    )

    assert config["initial_speed_km_h"] == 31.0
