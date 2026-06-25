from __future__ import annotations

import importlib.util
import sys
import types
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_route_traffic_manager_module():
    module_name = "route_traffic_manager_test"
    path = REPO_ROOT / "envs/diffusion_envs/route_traffic_manager.py"
    stubbed = {}

    def register(name: str, module: types.ModuleType) -> None:
        stubbed[name] = sys.modules.get(name)
        sys.modules[name] = module

    custom_module = types.ModuleType("envs.diffusion_envs.traffic_manager")

    class _CustomTrafficManager:
        VEHICLE_GAP = 10.0

        def __init__(self):
            self.engine = SimpleNamespace(spawn_manager=None, global_config={})

    custom_module.CustomTrafficManager = _CustomTrafficManager
    register("envs.diffusion_envs.traffic_manager", custom_module)

    traffic_module = types.ModuleType("metadrive.manager.traffic_manager")
    traffic_module.BlockVehicles = namedtuple("block_vehicles", "trigger_road vehicles")
    register("metadrive.manager.traffic_manager", traffic_module)

    policy_package = types.ModuleType("metadrive.policy")
    register("metadrive.policy", policy_package)

    idm_policy_module = types.ModuleType("metadrive.policy.idm_policy")
    idm_policy_module.IDMPolicy = type("IDMPolicy", (), {})
    register("metadrive.policy.idm_policy", idm_policy_module)

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


def test_conflict_filter_rejects_background_vehicle_within_same_road_buffer():
    module = _load_route_traffic_manager_module()
    manager = module.RouteAwareTrafficManager()
    manager.engine = SimpleNamespace(
        spawn_manager=SimpleNamespace(
            ego_spawn_zones=[
                {
                    "road": ("A", "B"),
                    "spawn_longitude": 20.0,
                    "buffer_longitudinal": 10.0,
                }
            ]
        ),
        global_config={"ego_spawn_buffer_scale": 1.0},
    )

    assert manager._conflicts_with_ego_spawn({"spawn_lane_index": ("A", "B", 0), "spawn_longitude": 25.0}) is True


def test_conflict_filter_keeps_background_vehicle_outside_lane_buffer_or_on_other_lane():
    module = _load_route_traffic_manager_module()
    manager = module.RouteAwareTrafficManager()
    manager.engine = SimpleNamespace(
        spawn_manager=SimpleNamespace(
            ego_spawn_zones=[
                {
                    "road": ("A", "B"),
                    "spawn_lane_index": ("A", "B", 0),
                    "spawn_longitude": 20.0,
                    "buffer_longitudinal": 10.0,
                }
            ]
        ),
        global_config={
            "ego_spawn_buffer_scale": 1.0,
            "traffic_spawn_lane_relaxation": True,
            "traffic_spawn_min_gap_ahead": 12.0,
            "traffic_spawn_min_gap_behind": 8.0,
        },
    )

    assert manager._conflicts_with_ego_spawn({"spawn_lane_index": ("A", "B", 0), "spawn_longitude": 35.5}) is False
    assert manager._conflicts_with_ego_spawn({"spawn_lane_index": ("A", "B", 1), "spawn_longitude": 22.0}) is False
    assert manager._conflicts_with_ego_spawn({"spawn_lane_index": ("C", "D", 0), "spawn_longitude": 22.0}) is False


def test_conflict_filter_uses_asymmetric_gaps_for_same_lane():
    module = _load_route_traffic_manager_module()
    manager = module.RouteAwareTrafficManager()
    manager.engine = SimpleNamespace(
        spawn_manager=SimpleNamespace(
            ego_spawn_zones=[
                {
                    "road": ("A", "B"),
                    "spawn_lane_index": ("A", "B", 0),
                    "spawn_longitude": 20.0,
                    "buffer_longitudinal": 10.0,
                }
            ]
        ),
        global_config={
            "ego_spawn_buffer_scale": 1.0,
            "traffic_spawn_lane_relaxation": True,
            "traffic_spawn_min_gap_ahead": 12.0,
            "traffic_spawn_min_gap_behind": 8.0,
        },
    )

    assert manager._conflicts_with_ego_spawn({"spawn_lane_index": ("A", "B", 0), "spawn_longitude": 31.0}) is True
    assert manager._conflicts_with_ego_spawn({"spawn_lane_index": ("A", "B", 0), "spawn_longitude": 32.5}) is False
    assert manager._conflicts_with_ego_spawn({"spawn_lane_index": ("A", "B", 0), "spawn_longitude": 13.5}) is True
    assert manager._conflicts_with_ego_spawn({"spawn_lane_index": ("A", "B", 0), "spawn_longitude": 11.5}) is False


def test_final_spawn_guard_blocks_candidate_that_overlaps_ego_zone():
    module = _load_route_traffic_manager_module()
    manager = module.RouteAwareTrafficManager()
    manager.engine = SimpleNamespace(
        current_map=SimpleNamespace(
            road_network=SimpleNamespace(
                get_lane=lambda lane_index: SimpleNamespace(
                    width=3.5,
                    position=lambda longitudinal, lateral: (float(longitudinal), float(lateral)),
                )
            )
        ),
        spawn_manager=SimpleNamespace(
            ego_spawn_zones=[
                {
                    "road": ("A", "B"),
                    "spawn_lane_index": ("A", "B", 0),
                    "spawn_longitude": 20.0,
                    "spawn_lateral": 0.0,
                }
            ]
        ),
        global_config={
            "vehicle_config": {"vehicle_model": "xl"},
            "traffic_vehicle_config": {"vehicle_model": "default"},
        },
    )

    assert manager._passes_final_spawn_guard({"spawn_lane_index": ("A", "B", 0), "spawn_longitude": 22.0}) is False
    assert manager._passes_final_spawn_guard({"spawn_lane_index": ("A", "B", 0), "spawn_longitude": 40.0}) is True


def test_spawn_traffic_vehicle_if_safe_can_skip_adding_vehicle_to_active_traffic_list():
    module = _load_route_traffic_manager_module()
    manager = module.RouteAwareTrafficManager()
    manager.generate_seed = lambda: 7
    manager._traffic_vehicles = []
    manager._conflicts_with_ego_spawn = lambda config: False
    manager._passes_final_spawn_guard = lambda config: True
    spawned_vehicle = SimpleNamespace(id="veh_0", name="veh_0")
    manager.spawn_object = lambda vehicle_type, vehicle_config: spawned_vehicle
    manager.add_policy = lambda *args, **kwargs: None
    manager.engine = SimpleNamespace(
        spawn_manager=SimpleNamespace(ego_spawn_zones=[]),
        global_config={"traffic_vehicle_config": {}},
    )
    vehicle = manager._spawn_traffic_vehicle_if_safe(
        "stub_vehicle_type",
        {"spawn_lane_index": ("A", "B", 0), "spawn_longitude": 0.0},
        add_to_active_traffic=False,
        policy_class=object,
    )

    assert vehicle is spawned_vehicle
    assert manager._traffic_vehicles == []


def test_spawn_traffic_vehicle_if_safe_injects_fixed_destination_from_spawn_manager():
    module = _load_route_traffic_manager_module()
    manager = module.RouteAwareTrafficManager()
    manager.generate_seed = lambda: 7
    manager._traffic_vehicles = []
    manager._conflicts_with_ego_spawn = lambda config: False
    manager._passes_final_spawn_guard = lambda config: True

    captured = {}

    def _spawn_object(vehicle_type, vehicle_config):
        captured["config"] = dict(vehicle_config)
        return SimpleNamespace(id="veh_0", name="veh_0")

    manager.spawn_object = _spawn_object
    manager.add_policy = lambda *args, **kwargs: None
    manager.engine = SimpleNamespace(
        spawn_manager=SimpleNamespace(
            ego_spawn_zones=[],
            update_destination_for=lambda agent_id, vehicle_config: dict(vehicle_config, destination="C4_END"),
        ),
        global_config={"traffic_vehicle_config": {}},
    )

    manager._spawn_traffic_vehicle_if_safe(
        "stub_vehicle_type",
        {"spawn_lane_index": ("A", "B", 0), "spawn_longitude": 0.0},
        add_to_active_traffic=False,
        policy_class=object,
    )

    assert captured["config"]["destination"] == "C4_END"


def test_resolve_trigger_road_uses_first_route_block_own_positive_route_road():
    module = _load_route_traffic_manager_module()
    manager = module.RouteAwareTrafficManager()
    manager.engine = SimpleNamespace(
        global_config={"ego_main_route_block_ids": ("s0", "c0", "c4")},
    )

    route_road = SimpleNamespace(start_node=">>>", end_node="S0_END")
    block = SimpleNamespace(
        graph_block_id="s0",
        get_respawn_roads=lambda: [route_road],
        get_socket_list=lambda: [],
        pre_block_socket=SimpleNamespace(positive_road=SimpleNamespace(start_node=">>", end_node=">>>")),
    )

    resolved = manager._resolve_trigger_road(block)

    assert resolved is route_road


def test_resolve_trigger_road_uses_pre_block_socket_for_non_first_route_blocks():
    module = _load_route_traffic_manager_module()
    manager = module.RouteAwareTrafficManager()
    manager.engine = SimpleNamespace(
        global_config={"ego_main_route_block_ids": ("s0", "c0", "c4")},
    )

    pre_socket_road = SimpleNamespace(start_node="S0_END", end_node="C0_ENTRY")
    block = SimpleNamespace(
        graph_block_id="c0",
        get_respawn_roads=lambda: [],
        get_socket_list=lambda: [],
        pre_block_socket=SimpleNamespace(positive_road=pre_socket_road),
    )

    resolved = manager._resolve_trigger_road(block)

    assert resolved is pre_socket_road


def test_after_reset_force_activates_all_pending_trigger_vehicles():
    module = _load_route_traffic_manager_module()
    manager = module.RouteAwareTrafficManager()
    manager._traffic_vehicles = []
    manager.block_triggered_vehicles = [
        SimpleNamespace(vehicles=["veh_a", "veh_b"]),
        SimpleNamespace(vehicles=["veh_c"]),
    ]
    manager.get_objects = lambda names: {name: SimpleNamespace(name=name) for name in names}
    manager.engine = SimpleNamespace(global_config={})

    original_after_reset = module.CustomTrafficManager.after_reset if hasattr(module.CustomTrafficManager, "after_reset") else None
    called = {"super_after_reset": 0}

    def _fake_super_after_reset(self):
        called["super_after_reset"] += 1

    module.CustomTrafficManager.after_reset = _fake_super_after_reset
    try:
        manager.after_reset()
    finally:
        if original_after_reset is None:
            delattr(module.CustomTrafficManager, "after_reset")
        else:
            module.CustomTrafficManager.after_reset = original_after_reset

    assert called["super_after_reset"] == 1
    assert manager.block_triggered_vehicles == []
    assert [vehicle.name for vehicle in manager._traffic_vehicles] == ["veh_c", "veh_a", "veh_b"]


def test_after_step_respawn_prefers_all_route_lanes_when_available():
    module = _load_route_traffic_manager_module()
    manager = module.RouteAwareTrafficManager()

    route_lane = SimpleNamespace(index=("R", "S", 0), length=100.0)
    fallback_lane = SimpleNamespace(index=("A", "B", 0), length=50.0)
    removed = SimpleNamespace(on_lane=False, id="veh_0", after_step=lambda: None)
    manager._traffic_vehicles = [removed]
    manager.respawn_lanes = [fallback_lane]
    manager.mode = "hybrid"
    manager.np_random = SimpleNamespace(randint=lambda low, high: 0, rand=lambda: 0.2)
    manager.clear_objects = lambda ids: None

    captured = {}

    def _spawn(vehicle_type, config):
        captured["vehicle_type"] = vehicle_type
        captured["config"] = dict(config)

    manager._spawn_traffic_vehicle_if_safe = _spawn
    manager._get_all_route_lanes = lambda: [route_lane]
    manager.engine = SimpleNamespace(global_config={}, current_map=None)

    manager.after_step()

    assert captured["vehicle_type"] is SimpleNamespace
    assert captured["config"]["spawn_lane_index"] == ("R", "S", 0)


def test_get_all_route_lanes_flattens_respawn_lane_groups():
    module = _load_route_traffic_manager_module()
    manager = module.RouteAwareTrafficManager()
    lane_a = SimpleNamespace(index=("A", "B", 0))
    lane_b = SimpleNamespace(index=("A", "B", 1))
    route_block = SimpleNamespace(
        graph_block_id="r0",
        get_respawn_lanes=lambda: [[lane_a, lane_b]],
    )
    other_block = SimpleNamespace(
        graph_block_id="x0",
        get_respawn_lanes=lambda: [[SimpleNamespace(index=("X", "Y", 0))]],
    )
    manager.engine = SimpleNamespace(
        global_config={"ego_main_route_block_ids": ("r0",)},
        current_map=SimpleNamespace(blocks=[route_block, other_block]),
    )

    lanes = manager._get_all_route_lanes()

    assert [lane.index for lane in lanes] == [("A", "B", 0), ("A", "B", 1)]


def test_final_spawn_guard_blocks_candidate_that_overlaps_existing_traffic_vehicle():
    module = _load_route_traffic_manager_module()
    manager = module.RouteAwareTrafficManager()
    lane = SimpleNamespace(
        width=3.5,
        position=lambda longitudinal, lateral: (float(longitudinal), float(lateral)),
    )
    manager._traffic_vehicles = [
        SimpleNamespace(
            position=(22.0, 0.0),
            LENGTH=4.5,
            WIDTH=1.85,
        )
    ]
    manager.engine = SimpleNamespace(
        current_map=SimpleNamespace(
            road_network=SimpleNamespace(
                get_lane=lambda lane_index: lane,
            )
        ),
        spawn_manager=SimpleNamespace(ego_spawn_zones=[]),
        global_config={
            "vehicle_config": {"vehicle_model": "default"},
            "traffic_vehicle_config": {"vehicle_model": "default"},
        },
    )

    assert manager._passes_final_spawn_guard({"spawn_lane_index": ("A", "B", 0), "spawn_longitude": 22.0}) is False
