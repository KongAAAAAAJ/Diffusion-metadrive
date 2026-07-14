"""Deterministic traffic recipe parameters for the S7 ego-ramp-merge scenario."""

from __future__ import annotations

from typing import Callable


_MAINLINE_VEHICLES = (
    # lane_id, name, spawn_longitude, target_speed_kmh
    (0, "main_lane_0_vehicle_1", 15.0, 22.0),
    (0, "main_lane_0_vehicle_2", 45.0, 25.0),
    (0, "main_lane_0_vehicle_3", 75.0, 27.0),
    (1, "main_lane_1_vehicle_1", 10.0, 22.0),
    (1, "main_lane_1_vehicle_2", 40.0, 24.0),
    (1, "main_lane_1_vehicle_3", 70.0, 26.0),
    (1, "main_lane_1_vehicle_4", 100.0, 28.0),
    (2, "main_lane_2_vehicle_1", 20.0, 23.0),
    (2, "main_lane_2_vehicle_2", 55.0, 26.0),
    (2, "main_lane_2_vehicle_3", 90.0, 29.0),
)


def build_s7_traffic_recipes(recipe_factory: Callable[..., object]) -> tuple[object, ...]:
    """Build S7 mainline traffic recipes without importing ``RecipeSpec`` here."""
    return tuple(
        recipe_factory(
            "inject_background_vehicle",
            {
                "name": name,
                "reference_kind": "block_route_road",
                "block_id": "g1",
                "lane_id": lane_id,
                "spawn_longitude": spawn_longitude,
                "target_speed_kmh": target_speed,
                "trigger_on_start": True,
            },
        )
        for lane_id, name, spawn_longitude, target_speed in _MAINLINE_VEHICLES
    )
