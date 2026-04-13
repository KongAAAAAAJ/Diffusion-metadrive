from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


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

        def update_destination_for(self, agent_id, vehicle_config):
            return vehicle_config

    spawn_manager_module.SpawnManager = _SpawnManager
    register("metadrive.manager.spawn_manager", spawn_manager_module)

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


class _Road:
    def __init__(self, start, end, negative=False):
        self.start_node = start
        self.end_node = end
        self._negative = negative

    def is_negative_road(self):
        return self._negative

    def lane_index(self, lane_idx):
        return (self.start_node, self.end_node, lane_idx)


def _make_block(graph_block_id, positive_roads, negative_roads=()):
    roads = list(positive_roads) + [_Road(r.start_node, r.end_node, negative=True) for r in negative_roads]
    return SimpleNamespace(
        graph_block_id=graph_block_id,
        get_respawn_roads=lambda: roads,
    )


def _make_map(blocks):
    return SimpleNamespace(blocks=blocks)


def test_get_main_route_spawn_roads_uses_manual_block_chain():
    module = _load_route_spawn_manager_module()
    manager = module.RouteAwareSpawnManager()

    road_s0 = _Road("ROOT_END", "S0_END")
    road_c0 = _Road("S0_END", "C0_END")
    road_c4 = _Road("C3_END", "C4_END")
    current_map = _make_map([
        _make_block("root", [_Road(">>", "ROOT_END")]),
        _make_block("s0", [road_s0]),
        _make_block("c0", [road_c0]),
        _make_block("c4", [road_c4]),
    ])
    manager.engine = SimpleNamespace(
        global_config={"ego_main_route_block_ids": ("s0", "c0", "c4")},
        current_map=current_map,
    )
    manager.current_map = current_map

    roads = manager.get_main_route_spawn_roads(current_map)

    assert [(r.start_node, r.end_node) for r in roads] == [
        ("ROOT_END", "S0_END"),
        ("S0_END", "C0_END"),
        ("C3_END", "C4_END"),
    ]


def test_get_main_route_spawn_roads_picks_sorted_positive_road_and_skips_negative():
    module = _load_route_spawn_manager_module()
    manager = module.RouteAwareSpawnManager()

    pos_a = _Road("X", "A1")
    pos_b = _Road("X", "A0")  # sorted ascending -> A0 is picked
    neg = _Road("X", "NEG")
    block = _make_block("s0", [pos_a, pos_b], negative_roads=[neg])
    current_map = _make_map([block])
    manager.engine = SimpleNamespace(
        global_config={"ego_main_route_block_ids": ("s0",)},
        current_map=current_map,
    )

    roads = manager.get_main_route_spawn_roads(current_map)

    assert [(r.start_node, r.end_node) for r in roads] == [("X", "A0")]


def test_get_main_route_spawn_roads_raises_on_unknown_block_id():
    module = _load_route_spawn_manager_module()
    manager = module.RouteAwareSpawnManager()

    current_map = _make_map([_make_block("s0", [_Road("A", "B")])])
    manager.engine = SimpleNamespace(
        global_config={"ego_main_route_block_ids": ("s0", "ghost")},
        current_map=current_map,
    )

    with pytest.raises(ValueError, match="ghost"):
        manager.get_main_route_spawn_roads(current_map)


def test_get_main_route_spawn_roads_raises_on_empty_route_config():
    module = _load_route_spawn_manager_module()
    manager = module.RouteAwareSpawnManager()
    manager.engine = SimpleNamespace(
        global_config={"ego_main_route_block_ids": []},
        current_map=_make_map([]),
    )

    with pytest.raises(ValueError, match="ego_main_route_block_ids"):
        manager.get_main_route_spawn_roads(manager.engine.current_map)


def test_reset_only_spawns_on_first_block_in_manual_chain():
    module = _load_route_spawn_manager_module()
    manager = module.RouteAwareSpawnManager()

    road_s0 = _Road("ROOT_END", "S0_END")
    road_c0 = _Road("S0_END", "C0_END")
    current_map = _make_map([
        _make_block("s0", [road_s0]),
        _make_block("c0", [road_c0]),
    ])
    manager.engine = SimpleNamespace(
        global_config={
            "ego_main_route_block_ids": ("s0", "c0"),
            "ego_spawn_buffer_scale": 1.0,
        },
    )
    manager.current_map = current_map

    manager.set_episode_spawn_seed(0)
    manager.reset()

    # spawn_roads 只包含链里的第一段（s0）
    assert [(r.start_node, r.end_node) for r in manager.spawn_roads] == [("ROOT_END", "S0_END")]
    agent_configs = manager.engine.global_config["agent_configs"]
    assert tuple(agent_configs["agent0"]["spawn_lane_index"][:2]) == ("ROOT_END", "S0_END")
    assert manager.ego_spawn_zones
    assert manager.ego_spawn_zones[0]["road"] == ("ROOT_END", "S0_END")


def test_update_destination_for_uses_last_block_in_manual_chain():
    module = _load_route_spawn_manager_module()
    manager = module.RouteAwareSpawnManager()

    road_c4 = _Road("C3_END", "C4_END")
    current_map = _make_map([
        _make_block("s0", [_Road("ROOT_END", "S0_END")]),
        _make_block("c4", [road_c4]),
    ])
    manager.engine = SimpleNamespace(
        global_config={"ego_main_route_block_ids": ("s0", "c4")},
        current_map=current_map,
    )

    updated = manager.update_destination_for("agent0", {"spawn_lane_index": ("A", "B", 0)})

    assert updated["destination"] == "C4_END"


def test_get_main_route_spawn_roads_falls_back_to_positive_socket_when_respawn_road_missing():
    module = _load_route_spawn_manager_module()
    manager = module.RouteAwareSpawnManager()

    road_s0 = _Road("ROOT_END", "S0_END")
    road_c4 = _Road("C3_END", "C4_END")
    current_map = _make_map([
        SimpleNamespace(
            graph_block_id="s0",
            get_respawn_roads=lambda: [],
            get_socket_list=lambda: [SimpleNamespace(positive_road=road_s0)],
        ),
        SimpleNamespace(
            graph_block_id="c4",
            get_respawn_roads=lambda: [],
            get_socket_list=lambda: [SimpleNamespace(positive_road=road_c4)],
        ),
    ])
    manager.engine = SimpleNamespace(
        global_config={"ego_main_route_block_ids": ("s0", "c4")},
        current_map=current_map,
    )

    roads = manager.get_main_route_spawn_roads(current_map)

    assert [(r.start_node, r.end_node) for r in roads] == [
        ("ROOT_END", "S0_END"),
        ("C3_END", "C4_END"),
    ]


def test_update_destination_for_falls_back_to_positive_socket_when_respawn_road_missing():
    module = _load_route_spawn_manager_module()
    manager = module.RouteAwareSpawnManager()

    road_c4 = _Road("C3_END", "C4_END")
    current_map = _make_map([
        SimpleNamespace(
            graph_block_id="s0",
            get_respawn_roads=lambda: [],
            get_socket_list=lambda: [SimpleNamespace(positive_road=_Road("ROOT_END", "S0_END"))],
        ),
        SimpleNamespace(
            graph_block_id="c4",
            get_respawn_roads=lambda: [],
            get_socket_list=lambda: [SimpleNamespace(positive_road=road_c4)],
        ),
    ])
    manager.engine = SimpleNamespace(
        global_config={"ego_main_route_block_ids": ("s0", "c4")},
        current_map=current_map,
    )

    updated = manager.update_destination_for("agent0", {"spawn_lane_index": ("A", "B", 0)})

    assert updated["destination"] == "C4_END"
