from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_spawn_manager_module():
    module_name = "spawn_manager_episode_seed_test"
    path = REPO_ROOT / "metadrive/manager/spawn_manager.py"
    stubbed = {}

    def register(name: str, module: types.ModuleType) -> None:
        stubbed[name] = sys.modules.get(name)
        sys.modules[name] = module

    straight_lane_module = types.ModuleType("metadrive.component.lane.straight_lane")
    straight_lane_module.StraightLane = type("StraightLane", (), {})
    register("metadrive.component.lane.straight_lane", straight_lane_module)

    first_block_module = types.ModuleType("metadrive.component.pgblock.first_block")
    first_block_module.FirstPGBlock = type("FirstPGBlock", (), {"ENTRANCE_LENGTH": 0.0})
    register("metadrive.component.pgblock.first_block", first_block_module)

    base_vehicle_module = types.ModuleType("metadrive.component.vehicle.base_vehicle")
    base_vehicle_module.BaseVehicle = type("BaseVehicle", (), {"MAX_LENGTH": 10.0, "MAX_WIDTH": 2.5})
    register("metadrive.component.vehicle.base_vehicle", base_vehicle_module)

    constants_module = types.ModuleType("metadrive.constants")
    constants_module.MetaDriveType = SimpleNamespace(VEHICLE="vehicle")
    constants_module.CollisionGroup = SimpleNamespace(Vehicle=1)
    register("metadrive.constants", constants_module)

    engine_utils_module = types.ModuleType("metadrive.engine.engine_utils")
    engine_utils_module.get_engine = lambda: None
    register("metadrive.engine.engine_utils", engine_utils_module)

    base_manager_module = types.ModuleType("metadrive.manager.base_manager")

    class _BaseManager:
        def __init__(self):
            self.np_random = np.random.RandomState(0)

        def seed(self, random_seed):
            self.np_random = np.random.RandomState(random_seed)

    base_manager_module.BaseManager = _BaseManager
    register("metadrive.manager.base_manager", base_manager_module)

    class _Config(dict):
        def get_dict(self):
            return dict(self)

        def force_update(self, update):
            self.update(update)

        def force_set(self, key, value):
            self[key] = value

    utils_module = types.ModuleType("metadrive.utils")
    utils_module.Config = _Config
    register("metadrive.utils", utils_module)

    coords_module = types.ModuleType("metadrive.utils.coordinates_shift")
    coords_module.panda_vector = lambda *args, **kwargs: None
    coords_module.panda_heading = lambda heading: heading
    register("metadrive.utils.coordinates_shift", coords_module)

    pg_utils_module = types.ModuleType("metadrive.utils.pg.utils")
    pg_utils_module.rect_region_detection = lambda *args, **kwargs: SimpleNamespace(hasHit=lambda: False)
    register("metadrive.utils.pg.utils", pg_utils_module)

    panda_bullet_module = types.ModuleType("panda3d.bullet")
    panda_bullet_module.BulletBoxShape = object
    panda_bullet_module.BulletGhostNode = object
    register("panda3d.bullet", panda_bullet_module)

    panda_core_module = types.ModuleType("panda3d.core")
    panda_core_module.Vec3 = object
    register("panda3d.core", panda_core_module)

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


def _make_manager(module):
    manager = module.SpawnManager.__new__(module.SpawnManager)
    manager.num_agents = 1
    manager.np_random = np.random.RandomState(123)
    manager._episode_spawn_seed = None
    manager._use_map_respawn_roads = False
    manager.engine = SimpleNamespace(global_config={"spawn_diversify_roads": True})
    manager.lane_num = 3
    manager.exit_length = 24.0
    manager.safe_spawn_places = {}
    manager.spawn_roads = []
    manager.need_update_spawn_places = True
    manager.spawn_places_used = []
    manager.available_agent_configs = [
        {"config": {"spawn_lane_index": ("A", "B", 0), "spawn_longitude": 4.0, "spawn_lateral": 0.0}},
        {"config": {"spawn_lane_index": ("A", "B", 1), "spawn_longitude": 12.0, "spawn_lateral": 0.0}},
        {"config": {"spawn_lane_index": ("A", "B", 2), "spawn_longitude": 20.0, "spawn_lateral": 0.0}},
    ]
    manager._init_agent_configs = {"agent0": {}}
    return manager


def test_spawn_manager_same_episode_seed_reproduces_single_agent_spawn_config():
    module = _load_spawn_manager_module()
    manager_a = _make_manager(module)
    manager_b = _make_manager(module)

    manager_a.set_episode_spawn_seed(77)
    manager_a.reset()
    config_a = dict(manager_a.engine.global_config["agent_configs"]["agent0"])

    manager_b.set_episode_spawn_seed(77)
    manager_b.reset()
    config_b = dict(manager_b.engine.global_config["agent_configs"]["agent0"])

    assert config_a == config_b


def test_spawn_manager_different_episode_seed_changes_single_agent_spawn_config():
    module = _load_spawn_manager_module()
    manager_a = _make_manager(module)
    manager_b = _make_manager(module)

    manager_a.set_episode_spawn_seed(77)
    manager_a.reset()
    config_a = dict(manager_a.engine.global_config["agent_configs"]["agent0"])

    manager_b.set_episode_spawn_seed(78)
    manager_b.reset()
    config_b = dict(manager_b.engine.global_config["agent_configs"]["agent0"])

    assert config_a != config_b


def test_spawn_manager_single_agent_reset_uses_sampled_target_slot_not_first_slot():
    module = _load_spawn_manager_module()
    manager = _make_manager(module)

    selected_seed = next(
        seed for seed in range(256)
        if int(np.random.RandomState(seed).choice([0, 1, 2], 1, replace=False)[0]) != 0
    )

    manager.set_episode_spawn_seed(selected_seed)
    manager.reset()
    config = manager.engine.global_config["agent_configs"]["agent0"]

    assert config["spawn_lane_index"] != ("A", "B", 0)


def test_spawn_manager_refresh_spawn_roads_rebuilds_available_slots():
    module = _load_spawn_manager_module()
    manager = _make_manager(module)

    class StubRoad:
        def __init__(self, start, end):
            self.start_node = start
            self.end_node = end

        def lane_index(self, lane_idx):
            return (self.start_node, self.end_node, lane_idx)

    road = StubRoad("R0", "R1")

    manager.refresh_spawn_roads([road])

    lane_indices = [entry["config"]["spawn_lane_index"] for entry in manager.available_agent_configs]
    assert lane_indices[0] == ("R0", "R1", 0)
    assert lane_indices[-1] == ("R0", "R1", 2)
    assert len(manager.available_agent_configs) == 9
    assert manager.engine.global_config["spawn_roads"] == [road]


def test_spawn_manager_multi_agent_reset_prefers_distinct_roads_when_enabled():
    module = _load_spawn_manager_module()
    manager = _make_manager(module)
    manager.num_agents = 2
    manager.available_agent_configs = [
        {"config": {"spawn_lane_index": ("A", "B", 0), "spawn_longitude": 4.0, "spawn_lateral": 0.0}},
        {"config": {"spawn_lane_index": ("A", "B", 1), "spawn_longitude": 12.0, "spawn_lateral": 0.0}},
        {"config": {"spawn_lane_index": ("C", "D", 0), "spawn_longitude": 4.0, "spawn_lateral": 0.0}},
        {"config": {"spawn_lane_index": ("C", "D", 1), "spawn_longitude": 12.0, "spawn_lateral": 0.0}},
    ]
    manager._init_agent_configs = {"agent0": {}, "agent1": {}}

    manager.set_episode_spawn_seed(3)
    manager.reset()

    agent_configs = manager.engine.global_config["agent_configs"]
    spawn_roads = {
        tuple(agent_configs["agent0"]["spawn_lane_index"][:2]),
        tuple(agent_configs["agent1"]["spawn_lane_index"][:2]),
    }
    assert len(spawn_roads) == 2
