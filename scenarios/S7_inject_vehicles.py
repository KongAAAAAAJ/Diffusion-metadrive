"""Arrival-aligned actor recipe for the S7 ego-ramp-merge scenario."""

from __future__ import annotations

from typing import Callable


def build_s7_traffic_recipes(recipe_factory: Callable[..., object]) -> tuple[object, ...]:
    """Build one atomic 4--6 actor mainline-gap recipe."""
    return (
        recipe_factory(
            "inject_s7_merge_traffic",
            {
                "block_id": "g1",
                "target_lane_id": 2,
                "required_roles": (
                    "critical_gap_front",
                    "critical_gap_rear",
                    "next_gap_front",
                    "next_gap_rear",
                ),
                "optional_adjacent_actor_range": (0, 2),
                "trigger_on_start": True,
            },
        ),
    )
