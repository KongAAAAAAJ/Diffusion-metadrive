"""Arrival-aligned actor recipe for the S7 ego-ramp-merge scenario."""

from __future__ import annotations

from typing import Callable


def build_s7_traffic_recipes(recipe_factory: Callable[..., object]) -> tuple[object, ...]:
    """Build one atomic timed-stream plus parallel-constraint recipe."""
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
                "parallel_constraint_actor_range": (1, 3),
                "parallel_constraint_role_prefix": "parallel_merge_constraint_",
                "trigger_on_start": True,
            },
        ),
    )
