from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_traffic_manager_module():
    module_name = "traffic_manager_destroy_test"
    path = REPO_ROOT / "metadrive/manager/traffic_manager.py"
    stubbed = {}

    def register(name: str, module: types.ModuleType) -> None:
        stubbed[name] = sys.modules.get(name)
        sys.modules[name] = module

    abs_lane_module = types.ModuleType("metadrive.component.lane.abs_lane")
    abs_lane_module.AbstractLane = type("AbstractLane", (), {})
    register("metadrive.component.lane.abs_lane", abs_lane_module)

    base_map_module = types.ModuleType("metadrive.component.map.base_map")
    base_map_module.BaseMap = type("BaseMap", (), {})
    register("metadrive.component.map.base_map", base_map_module)

    road_network_module = types.ModuleType("metadrive.component.road_network")
    road_network_module.Road = type("Road", (), {})
    register("metadrive.component.road_network", road_network_module)

    base_vehicle_module = types.ModuleType("metadrive.component.vehicle.base_vehicle")
    base_vehicle_module.BaseVehicle = type("BaseVehicle", (), {})
    register("metadrive.component.vehicle.base_vehicle", base_vehicle_module)

    constants_module = types.ModuleType("metadrive.constants")
    constants_module.TARGET_VEHICLES = "target_vehicles"
    constants_module.TRAFFIC_VEHICLES = "traffic_vehicles"
    constants_module.OBJECT_TO_AGENT = "object_to_agent"
    constants_module.AGENT_TO_OBJECT = "agent_to_object"
    register("metadrive.constants", constants_module)

    base_manager_module = types.ModuleType("metadrive.manager.base_manager")

    class _BaseManager:
        def __init__(self):
            self.engine = SimpleNamespace(
                global_config={
                    "traffic_mode": "basic",
                    "random_traffic": False,
                    "traffic_density": 0.1,
                }
            )

        def clear_objects(self, ids, *args, **kwargs):
            return self.engine.clear_objects(ids, *args, **kwargs)

    base_manager_module.BaseManager = _BaseManager
    register("metadrive.manager.base_manager", base_manager_module)

    utils_module = types.ModuleType("metadrive.utils")
    utils_module.merge_dicts = lambda a, b, allow_new_keys=True: dict(a, **b)
    register("metadrive.utils", utils_module)

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


def test_destroy_skips_stale_traffic_vehicle_ids():
    module = _load_traffic_manager_module()
    manager = module.PGTrafficManager()

    active_vehicle = SimpleNamespace(id="active-id")
    stale_vehicle = SimpleNamespace(id="stale-id")
    cleared_ids = []

    def _clear_objects(ids, *args, **kwargs):
        cleared_ids.append(list(ids))
        return ids

    manager.engine = SimpleNamespace(
        global_config={
            "traffic_mode": "basic",
            "random_traffic": False,
            "traffic_density": 0.1,
        },
        _spawned_objects={"active-id": object()},
        clear_objects=_clear_objects,
    )
    manager._traffic_vehicles = [active_vehicle, stale_vehicle]

    manager.destroy()

    assert cleared_ids == [["active-id"]]
    assert manager._traffic_vehicles == []
