from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "local_traffic_spawner.py"
    spec = importlib.util.spec_from_file_location("local_traffic_spawner_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_spawn_base_traffic_places_route_conditioned_vehicles_and_sets_speed():
    module = _load_module()
    spawned = []
    policy = SimpleNamespace(NORMAL_SPEED=0.0, target_speed=0.0)

    class _Lane:
        def __init__(self, index, length=120.0):
            self.index = index
            self.length = float(length)

        def local_coordinates(self, position):
            return (float(position[0]), 0.0)

    def _spawn_if_safe(vehicle_type, config):
        vehicle = SimpleNamespace(name=f"veh_{len(spawned)}")
        spawned.append((vehicle_type, dict(config), vehicle))
        return vehicle

    lanes = [_Lane(("A", "B", 0)), _Lane(("A", "B", 1))]
    road_network = SimpleNamespace(
        graph={"A": {"B": lanes}},
        get_lane=lambda lane_index: lanes[lane_index[2]] if lane_index[:2] == ("A", "B") else None,
    )
    env = SimpleNamespace(
        engine=SimpleNamespace(
            current_map=SimpleNamespace(road_network=road_network),
            traffic_manager=SimpleNamespace(
                _spawn_traffic_vehicle_if_safe=_spawn_if_safe,
                random_vehicle_type=lambda: "sedan",
            ),
            get_policy=lambda name: policy,
        )
    )
    ego_vehicle = SimpleNamespace(lane_index=("A", "B", 0), position=np.asarray([40.0, 0.0], dtype=np.float32))

    count = module.LocalTrafficSpawner().spawn_base_traffic(
        env,
        ego_vehicle,
        local_route="R1_entry_straight",
        traffic_density=0.10,
        rng=np.random.RandomState(0),
    )

    assert count >= 1
    assert len(spawned) == count
    assert all(config["spawn_lane_index"][:2] == ("A", "B") for _, config, _ in spawned)
    assert policy.NORMAL_SPEED > 0.0
    assert policy.target_speed > 0.0


def test_transition_route_uses_reduced_density_and_low_density_template():
    module = _load_module()
    spawner = module.LocalTrafficSpawner()

    assert spawner.get_effective_traffic_density("R4_mainline_transition", 0.12) == 0.03
    template = module.ROUTE_TEMPLATES["R4_mainline_transition"]
    assert len(template) <= 2
    assert all(slot.lane_offset == 0 for slot in template)


def test_merge_approach_route_uses_reduced_density_and_capped_template():
    module = _load_module()
    spawner = module.LocalTrafficSpawner()

    assert spawner.get_effective_traffic_density("R6_mainline_merge_approach", 0.10) == pytest.approx(0.04)
    template = module.ROUTE_TEMPLATES["R6_mainline_merge_approach"]
    assert len(template) <= 2
    assert all(slot.lane_offset == 0 for slot in template)
    assert max(slot.p for slot in template) <= 0.45
