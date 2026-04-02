from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_route_spawn_manager_module():
    module_name = "route_spawn_manager_test"
    path = REPO_ROOT / "metadrive/envs/diffusion_envs/route_spawn_manager.py"
    stubbed = {}

    def register(name: str, module: types.ModuleType) -> None:
        stubbed[name] = sys.modules.get(name)
        sys.modules[name] = module

    spawn_manager_module = types.ModuleType("metadrive.manager.spawn_manager")

    class _SpawnManager:
        RESPAWN_REGION_LONGITUDE = 8.0
        RESPAWN_REGION_LATERAL = 3.0
        MAX_VEHICLE_LENGTH = 5.0
        MAX_VEHICLE_WIDTH = 2.0

        def __init__(self):
            self.engine = SimpleNamespace(global_config={})
            self.available_agent_configs = []
            self.safe_spawn_places = {}
            self.spawn_roads = []
            self.need_update_spawn_places = True
            self.spawn_places_used = []
            self.np_random = np.random.RandomState(0)
            self.num_agents = 1
            self.lane_num = 2
            self.exit_length = 24.0
            self._init_agent_configs = {"agent0": {}}
            self._episode_spawn_seed = None

        def reset(self):
            if self._episode_spawn_seed is not None:
                rng = np.random.RandomState(self._episode_spawn_seed)
            else:
                rng = self.np_random
            selected = int(rng.choice(list(range(len(self.available_agent_configs))), 1, replace=False)[0])
            config = dict(self.available_agent_configs[selected]["config"])
            self.engine.global_config["agent_configs"] = {"agent0": config}
            self._episode_spawn_seed = None

        def set_episode_spawn_seed(self, seed):
            self._episode_spawn_seed = int(seed)

        def _auto_fill_spawn_roads_randomly(self, spawn_roads):
            configs = []
            safe = []
            for road in spawn_roads:
                for lane_idx in range(self.lane_num):
                    configs.append(
                        {
                            "identifier": f"{road.start_node}|{road.end_node}|{lane_idx}",
                            "config": {
                                "spawn_lane_index": (road.start_node, road.end_node, lane_idx),
                                "spawn_longitude": 4.0 + lane_idx,
                                "spawn_lateral": 0.0,
                            },
                        }
                    )
                    safe.append(configs[-1])
            return configs, safe

    spawn_manager_module.SpawnManager = _SpawnManager
    register("metadrive.manager.spawn_manager", spawn_manager_module)

    road_module = types.ModuleType("metadrive.component.road_network")

    class _Road:
        def __init__(self, start, end, negative=False):
            self.start_node = start
            self.end_node = end
            self._negative = negative

        def __repr__(self):
            return f"Road({self.start_node}->{self.end_node})"

        def is_negative_road(self):
            return self._negative

        def lane_index(self, lane_idx):
            return (self.start_node, self.end_node, lane_idx)

    road_module.Road = _Road
    register("metadrive.component.road_network", road_module)

    first_block_module = types.ModuleType("metadrive.component.pgblock.first_block")
    first_block_module.FirstPGBlock = type("FirstPGBlock", (), {"NODE_2": ">>", "NODE_3": ">>>"})
    register("metadrive.component.pgblock.first_block", first_block_module)

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


def test_main_route_spawn_roads_follow_forward_chain_only():
    module = _load_route_spawn_manager_module()

    class Road:
        def __init__(self, start, end, negative=False):
            self.start_node = start
            self.end_node = end
            self._negative = negative

        def is_negative_road(self):
            return self._negative

    manager = module.RouteAwareSpawnManager()

    start_road = Road(">>", ">>>")
    road_ab = Road(">>>", "A")
    branch = Road(">>>", "SIDE")
    road_bc = Road("A", "B")
    negative = Road("A", "NEG", negative=True)
    manager.engine = SimpleNamespace(global_config={"ego_spawn_route_start": (">>", ">>>")})
    manager.current_map = SimpleNamespace(
        blocks=[
            SimpleNamespace(get_respawn_roads=lambda: [start_road]),
            SimpleNamespace(pre_block_socket=SimpleNamespace(positive_road=start_road), get_respawn_roads=lambda: [road_ab, branch]),
            SimpleNamespace(pre_block_socket=SimpleNamespace(positive_road=road_ab), get_respawn_roads=lambda: [road_bc, negative]),
        ]
    )

    roads = manager.get_main_route_spawn_roads(manager.current_map)

    assert [(road.start_node, road.end_node) for road in roads] == [
        (">>", ">>>"),
        (">>>", "A"),
        ("A", "B"),
    ]


def test_reset_refreshes_spawn_roads_to_main_route_and_caches_spawn_zone():
    module = _load_route_spawn_manager_module()

    class Road:
        def __init__(self, start, end, negative=False):
            self.start_node = start
            self.end_node = end
            self._negative = negative

        def is_negative_road(self):
            return self._negative

        def lane_index(self, lane_idx):
            return (self.start_node, self.end_node, lane_idx)

    manager = module.RouteAwareSpawnManager()
    manager.engine = SimpleNamespace(global_config={"ego_spawn_route_start": (">>", ">>>")})
    start_road = Road(">>", ">>>")
    road_ab = Road(">>>", "A")
    manager.current_map = SimpleNamespace(
        blocks=[
            SimpleNamespace(get_respawn_roads=lambda: [start_road]),
            SimpleNamespace(pre_block_socket=SimpleNamespace(positive_road=start_road), get_respawn_roads=lambda: [road_ab]),
        ]
    )

    manager.set_episode_spawn_seed(0)
    manager.reset()

    assert [(road.start_node, road.end_node) for road in manager.spawn_roads] == [
        (">>", ">>>"),
        (">>>", "A"),
    ]
    cached = manager.ego_spawn_zones
    assert len(cached) == 1
    assert cached[0]["road"] in {(">>", ">>>"), (">>>", "A")}
