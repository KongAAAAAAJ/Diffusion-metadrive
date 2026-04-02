from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_route_traffic_manager_module():
    module_name = "route_traffic_manager_test"
    path = REPO_ROOT / "metadrive/envs/diffusion_envs/route_traffic_manager.py"
    stubbed = {}

    def register(name: str, module: types.ModuleType) -> None:
        stubbed[name] = sys.modules.get(name)
        sys.modules[name] = module

    custom_module = types.ModuleType("metadrive.envs.diffusion_envs.traffic_manager")

    class _CustomTrafficManager:
        VEHICLE_GAP = 10.0

        def __init__(self):
            self.engine = SimpleNamespace(spawn_manager=None, global_config={})

    custom_module.CustomTrafficManager = _CustomTrafficManager
    register("metadrive.envs.diffusion_envs.traffic_manager", custom_module)

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
