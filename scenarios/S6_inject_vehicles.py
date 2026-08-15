"""Controlled traffic recipe for the S6 merge scenario."""

from __future__ import annotations

from typing import Callable


def build_s6_traffic_recipes(recipe_factory: Callable[..., object]) -> tuple[object, ...]:
    """Build one gap-conditioned merge actor and no incidental traffic."""
    merge_vehicle = recipe_factory(
        "inject_background_vehicle",
        {
            "reference_kind": "block_socket_road",
            "block_id": "g1",
            "socket_index": 1,
            "target_speed_range_kmh": (18.0, 27.0),
            "policy": "idm_merge",
            "target_gap_choices": ("agent0-agent1", "agent1-agent2"),
            "merge_front_gap_range_m": (6.0, 10.0),
            "merge_rear_gap_range_m": (6.0, 10.0),
            "merge_creep_speed_kmh": 18.0,
            "merge_rear_ttc_min_s": 0.5,
            "merge_arrival_delta_range_s": (-0.5, 0.5),
            "predicted_conflict_ttc_range_s": (1.5, 3.5),
            "trigger_time_range_s": (0.0, 1.0),
            "trigger_on_start": True,
            "scenario_vehicle_role": "s6_gap_intruder",
        },
    )
    return (merge_vehicle,)
