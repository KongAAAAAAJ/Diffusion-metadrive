"""Controlled traffic recipe for the S6 merge scenario."""

from __future__ import annotations

from typing import Callable


def build_s6_traffic_recipes(recipe_factory: Callable[..., object]) -> tuple[object, ...]:
    """Build one arrival-aligned merge vehicle and no incidental traffic."""
    merge_vehicle = recipe_factory(
        "inject_background_vehicle",
        {
            "reference_kind": "block_socket_road",
            "block_id": "g1",
            "socket_index": 1,
            "target_speed_kmh": 24.0,
            "policy": "idm_merge",
            "merge_front_gap_m": 10.0,
            "merge_rear_gap_m": 10.0,
            "merge_creep_speed_kmh": 20.0,
            "merge_arrival_offset_s": 1.2,
            "trigger_on_start": True,
        },
    )
    return (merge_vehicle,)
