from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]


class _StubConfig(dict):
    def update(self, other=None, *args, **kwargs):
        if other is not None:
            super().update(other)
        return self


def _load_base_multi_env_module():
    module_name = "base_multi_env_spawn_strategy_test"
    path = REPO_ROOT / "metadrive/envs/diffusion_envs/base_multi_env.py"
    stubbed = {}

    def register(name: str, module: types.ModuleType) -> None:
        stubbed[name] = sys.modules.get(name)
        sys.modules[name] = module

    multi_agent_module = types.ModuleType("metadrive.envs.marl_envs.multi_agent_metadrive")

    class _MultiAgentMetaDrive:
        @staticmethod
        def default_config():
            return _StubConfig({"spawn_roads": ["default_spawn_road"]})

        def setup_engine(self):
            if not hasattr(self, "engine"):
                self.engine = SimpleNamespace(update_manager=lambda *args, **kwargs: None)

    multi_agent_module.MultiAgentMetaDrive = _MultiAgentMetaDrive
    register("metadrive.envs.marl_envs.multi_agent_metadrive", multi_agent_module)

    hybrid_module = types.ModuleType("metadrive.envs.diffusion_envs.custom_hybrid_map")
    hybrid_module.MAHybridPGMapManager = object
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
    utils_module.merge_dicts = lambda left, right, **kwargs: {**right, **left}
    register("metadrive.utils", utils_module)

    traffic_module = types.ModuleType("metadrive.manager.traffic_manager")
    traffic_module.TrafficMode = SimpleNamespace(Trigger="Trigger", Respawn="Respawn")
    register("metadrive.manager.traffic_manager", traffic_module)

    engine_utils_module = types.ModuleType("metadrive.engine.engine_utils")
    engine_utils_module.initialize_global_config = lambda config: None
    register("metadrive.engine.engine_utils", engine_utils_module)

    transfuser_module = types.ModuleType("metadrive.policy.diffusion_policy.transfuser_config")
    transfuser_module.build_transfuser_config = lambda size: {"size": size}
    transfuser_module.transfuser_config_to_dict = lambda config: dict(config)
    register("metadrive.policy.diffusion_policy.transfuser_config", transfuser_module)

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        for name, previous in stubbed.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


def test_default_config_enables_map_respawn_roads_strategy():
    module = _load_base_multi_env_module()

    config = module.BaseMultiEnv.default_config()

    assert config["spawn_strategy"] == "map_respawn_roads"
    assert config["spawn_diversify_roads"] is True
    assert config["spawn_roads"] is None
    assert "traffic_target_speed" in config
    assert config["traffic_target_speed"] is None


def test_collect_map_respawn_roads_merges_unique_roads_from_all_blocks():
    module = _load_base_multi_env_module()
    env = module.BaseMultiEnv.__new__(module.BaseMultiEnv)
    road_a = object()
    road_b = object()
    env.current_map = SimpleNamespace(
        blocks=[
            SimpleNamespace(get_respawn_roads=lambda: [road_a, road_b]),
            SimpleNamespace(get_respawn_roads=lambda: [road_b]),
        ]
    )

    resolved = env._collect_map_respawn_roads()

    assert resolved == [road_a, road_b]


def test_refresh_spawn_roads_if_needed_skips_when_user_explicitly_configured_spawn_roads():
    module = _load_base_multi_env_module()
    env = module.BaseMultiEnv.__new__(module.BaseMultiEnv)
    env.config = {
        "spawn_strategy": "map_respawn_roads",
        "spawn_roads": ["user_defined_road"],
    }
    env.current_map = SimpleNamespace(blocks=[])
    refreshed = []
    env.engine = SimpleNamespace(spawn_manager=SimpleNamespace(refresh_spawn_roads=lambda roads: refreshed.append(roads)))

    env._refresh_map_spawn_roads_if_needed()

    assert refreshed == []


def test_setup_engine_replaces_default_traffic_manager_with_custom_manager():
    module = _load_base_multi_env_module()
    env = module.BaseMultiEnv.__new__(module.BaseMultiEnv)
    updates = []
    env.engine = SimpleNamespace(update_manager=lambda name, manager: updates.append((name, manager.__class__.__name__)))

    custom_manager_module = types.ModuleType("envs.traffic_manager")

    class CustomTrafficManager:
        pass

    custom_manager_module.CustomTrafficManager = CustomTrafficManager
    previous = sys.modules.get("envs.traffic_manager")
    sys.modules["envs.traffic_manager"] = custom_manager_module
    try:
        env.setup_engine()
    finally:
        if previous is None:
            sys.modules.pop("envs.traffic_manager", None)
        else:
            sys.modules["envs.traffic_manager"] = previous

    assert updates == [
        ("map_manager", "object"),
        ("traffic_manager", "CustomTrafficManager"),
    ]
