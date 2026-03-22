from __future__ import annotations

from typing import List, Optional


def get_hazard_scenario_configs() -> List[dict]:
    """Return the Phase 1 minimum hazard scenario registry."""
    return [
        {
            "name": "static_obstacle_detour",
            "description": "Static lane blockage requiring platoon detour or compression.",
            "env_overrides": {
                "traffic_density": 0.15,
                "accident_prob": 0.15,
            },
        },
        {
            "name": "dynamic_cut_in",
            "description": "Dynamic vehicle cuts into platoon path and forces coordinated response.",
            "env_overrides": {
                "traffic_density": 0.08,
                "random_traffic": True,
            },
        },
        {
            "name": "bottleneck_narrow_bridge",
            "description": "Bottleneck passage that compresses inter-vehicle spacing and slows progress.",
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
