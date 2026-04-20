from __future__ import annotations

from typing import List, Optional


def get_hazard_scenario_configs() -> List[dict]:
    """Return the Phase 1 minimum hazard scenario registry.

    Each entry may optionally include ``scenario_id`` and ``local_route`` fields.
    When both are present, PlatoonEnv will activate the corresponding
    ScenarioOrchestrator in addition to applying ``env_overrides``.
    """
    return [
        {
            "name": "static_obstacle_detour",
            "description": "Static lane blockage requiring platoon detour or compression.",
            # scenario_id / local_route omitted: pure env_overrides only (accident_prob)
            "env_overrides": {
                "traffic_density": 0.15,
                "accident_prob": 0.15,
            },
        },
        {
            "name": "dynamic_cut_in",
            "description": "Dynamic vehicle cuts into platoon path and forces coordinated response.",
            # S6: background vehicle merges into the mainline from the on-ramp
            "scenario_id": "S6_background_merge_in",
            "local_route": "R6_mainline_merge_approach",
            "env_overrides": {
                "traffic_density": 0.08,
                "random_traffic": True,
            },
        },
        {
            "name": "bottleneck_narrow_bridge",
            "description": "Bottleneck passage that compresses inter-vehicle spacing and slows progress.",
            # S9: merge-split narrow channel with two injected background vehicles
            "scenario_id": "S9_narrow_channel_negotiation",
            "local_route": "R8_narrow_channel",
            "env_overrides": {
                "traffic_density": 0.02,
                "use_hybrid_map": True,
            },
        },
    ]


def get_hazard_scenario_config(name: Optional[str]) -> Optional[dict]:
    if name is None:
        return None
    for config in get_hazard_scenario_configs():
        if config["name"] == name:
            return config
    raise ValueError(f"Unknown hazard scenario: {name}")
