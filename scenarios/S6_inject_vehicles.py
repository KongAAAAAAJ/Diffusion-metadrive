"""Deterministic traffic recipe parameters for the S6 merge scenario."""

from __future__ import annotations

from typing import Callable


_SAME_LANE_VEHICLES = (
    ("ego_lane_front_1", 20.0, 20.0),
    ("ego_lane_front_2", 35.0, 21.0),
    ("ego_lane_front_3", 50.0, 22.0),
    ("ego_lane_front_4", 65.0, 23.0),
    ("ego_lane_rear_1", -40.0, 24.0),
    ("ego_lane_rear_2", -55.0, 23.0),
    ("ego_lane_rear_3", -70.0, 22.0),
    ("ego_lane_rear_4", -85.0, 21.0),
)

_ADJACENT_LANE_OFFSETS = (-30.0, -10.0, 15.0, 35.0)
_ADJACENT_LANE_SPEEDS = (20.0, 22.0, 24.0, 21.0)


def build_s6_traffic_recipes(recipe_factory: Callable[..., object]) -> tuple[object, ...]:
    """Build S6 recipes without importing ``RecipeSpec`` back into this module."""
    merge_vehicle = recipe_factory(
        "inject_background_vehicle",
        {
            "reference_kind": "block_socket_road",
            "block_id": "g1",
            "socket_index": 1,
            "spawn_longitude": 30.0,
            "target_speed_kmh": 24.0,
            "policy": "idm_merge",
            "merge_front_gap_m": 25.0,
            "merge_rear_gap_m": 15.0,
            "merge_creep_speed_kmh": 5.0,
        },
    )

    same_lane_vehicles = tuple(
        recipe_factory(
            "inject_background_vehicle",
            {
                "name": name,
                "reference_kind": "ego_lane",
                "spawn_longitude": 0.0,
                "spawn_longitude_offset": offset,
                "target_speed_kmh": target_speed,
                "trigger_on_start": True,
            },
        )
        for name, offset, target_speed in _SAME_LANE_VEHICLES
    )

    adjacent_vehicles = tuple(
        {
            "name": f"{lane_side}_lane_{index}",
            "lane_side": lane_side,
            "spawn_longitude_offset_m": offset,
            "target_speed_kmh": target_speed,
        }
        for lane_side in ("left", "right")
        for index, (offset, target_speed) in enumerate(
            zip(_ADJACENT_LANE_OFFSETS, _ADJACENT_LANE_SPEEDS),
            start=1,
        )
    )
    adjacent_lane_traffic = recipe_factory(
        "inject_adjacent_lane_vehicles",
        {
            "trigger_on_start": True,
            "clearance_scope": "same_lane",
            "vehicles": adjacent_vehicles,
        },
    )

    return (merge_vehicle, *same_lane_vehicles, adjacent_lane_traffic)
