from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


class _StubConfig(dict):
    def update(self, other=None, *args, **kwargs):
        if other is not None:
            super().update(other)
        return self


def _load_base_multi_env_module():
    module_name = "base_multi_env_route_spawn_test"
    path = REPO_ROOT / "metadrive/envs/diffusion_envs/base_multi_env.py"
    stubbed = {}

    def register(name: str, module: types.ModuleType) -> None:
        stubbed[name] = sys.modules.get(name)
        sys.modules[name] = module

    multi_agent_module = types.ModuleType("metadrive.envs.marl_envs.multi_agent_metadrive")

    class _MultiAgentMetaDrive:
        def __init__(self, config=None):
            self.config = config or _StubConfig()

        @staticmethod
        def default_config():
            return _StubConfig({"spawn_roads": ["default_spawn_road"]})

        def setup_engine(self):
            self.engine = type(
                "Engine",
                (),
                {"update_manager": lambda self, name, manager: updates.append((name, manager.__class__.__name__))},
            )()

    updates = []
    multi_agent_module.MultiAgentMetaDrive = _MultiAgentMetaDrive
    register("metadrive.envs.marl_envs.multi_agent_metadrive", multi_agent_module)

    hybrid_module = types.ModuleType("metadrive.envs.diffusion_envs.custom_hybrid_map")
    hybrid_module.MAHybridPGMapManager = type("MAHybridPGMapManager", (), {})
    register("metadrive.envs.diffusion_envs.custom_hybrid_map", hybrid_module)

    state_obs_module = types.ModuleType("metadrive.obs.state_obs")
    state_obs_module.LidarStateObservation = object
    register("metadrive.obs.state_obs", state_obs_module)

    diff_obs_module = types.ModuleType("metadrive.obs.diff_obs.top_down_state_obs_multi_channel")
    diff_obs_module.TopDownLidarStateObservation = object
    diff_obs_module.DatasetCollectObservation = object
    register("metadrive.obs.diff_obs.top_down_state_obs_multi_channel", diff_obs_module)

    rgb_module = types.ModuleType("metadrive.component.sensors.rgb_camera")
    rgb_module.RGBCamera = object
    register("metadrive.component.sensors.rgb_camera", rgb_module)

    utils_module = types.ModuleType("metadrive.utils")
    utils_module.Config = _StubConfig
    register("metadrive.utils", utils_module)

    traffic_module = types.ModuleType("metadrive.manager.traffic_manager")
    traffic_module.TrafficMode = type("TrafficMode", (), {"Respawn": "Respawn", "Hybrid": "Hybrid"})
    register("metadrive.manager.traffic_manager", traffic_module)

    engine_utils_module = types.ModuleType("metadrive.engine.engine_utils")
    engine_utils_module.initialize_global_config = lambda config: None
    register("metadrive.engine.engine_utils", engine_utils_module)

    transfuser_module = types.ModuleType("metadrive.policy.diffusion_policy.transfuser_config")
    transfuser_module.build_transfuser_config = lambda size: {"size": size}
    transfuser_module.transfuser_config_to_dict = lambda config: dict(config)
    register("metadrive.policy.diffusion_policy.transfuser_config", transfuser_module)

    route_spawn_module = types.ModuleType("metadrive.envs.diffusion_envs.route_spawn_manager")
    route_spawn_module.RouteAwareSpawnManager = type("RouteAwareSpawnManager", (), {})
    register("metadrive.envs.diffusion_envs.route_spawn_manager", route_spawn_module)

    custom_traffic_module = types.ModuleType("metadrive.envs.diffusion_envs.route_traffic_manager")
    custom_traffic_module.RouteAwareTrafficManager = type("RouteAwareTrafficManager", (), {})
    register("metadrive.envs.diffusion_envs.route_traffic_manager", custom_traffic_module)

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    module._test_stubbed_modules = stubbed
    return module, updates


def test_default_config_uses_main_route_spawn_settings():
    module, _ = _load_base_multi_env_module()

    config = module.BaseMultiEnv.default_config()

    assert config["spawn_roads"] == ["default_spawn_road"]
    assert config["ego_spawn_mode"] == "main_route_only"
    assert config["route_preset"] == "mainline"
    route_block_ids = tuple(config["ego_main_route_block_ids"])
    assert route_block_ids == module.DEFAULT_MAIN_ROUTE_BLOCK_IDS
    assert "ego_spawn_route_start" not in config
    assert "ego_spawn_route_start_block_id" not in config
    assert "ego_destination_block_id" not in config
    assert config["ego_spawn_buffer_mode"] == "traffic_gap"
    assert config["ego_spawn_buffer_scale"] == 1.0
    assert config["traffic_spawn_min_gap_ahead"] == 12.0
    assert config["traffic_spawn_min_gap_behind"] == 8.0
    assert config["traffic_spawn_lane_relaxation"] is True


def test_setup_engine_replaces_spawn_and_traffic_managers_with_route_aware_versions():
    module, updates = _load_base_multi_env_module()
    env = module.BaseMultiEnv()

    env.setup_engine()

    assert ("map_manager", "MAHybridPGMapManager") in updates
    assert ("spawn_manager", "RouteAwareSpawnManager") in updates
    assert ("traffic_manager", "RouteAwareTrafficManager") in updates


def test_route_preset_can_switch_to_ramp_merge_route():
    module, _ = _load_base_multi_env_module()

    config = module.BaseMultiEnv.default_config()
    config["route_preset"] = "ramp_merge"
    resolved = module.BaseMultiEnv._resolve_route_block_ids(config["route_preset"])

    assert resolved == module.DEFAULT_RAMP_MERGE_ROUTE_BLOCK_IDS
    assert resolved[0] == "s0"
    assert resolved[-1] == "c4"
