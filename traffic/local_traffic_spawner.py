"""
Route-conditioned local traffic spawner for expert data collection.

Called once per episode after env.reset() to place a small set of
structurally correct background vehicles around the ego spawn position.
Uses lane-topology slot resolution - no Euclidean random scatter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class TrafficSlot:
    """One candidate spawn slot relative to ego's lane position."""

    lane_offset: int
    long_offset: float
    speed_kmh: float
    p: float


ROUTE_TEMPLATES: Dict[str, List[TrafficSlot]] = {
    "R1_entry_straight": [
        TrafficSlot(0, +20, 24, 0.85),
        TrafficSlot(0, +45, 26, 0.55),
        TrafficSlot(0, -18, 22, 0.70),
        TrafficSlot(+1, +15, 25, 0.55),
    ],
    "R2_entry_curve": [
        TrafficSlot(0, +28, 20, 0.80),
        TrafficSlot(0, -22, 18, 0.65),
        TrafficSlot(+1, +22, 20, 0.45),
    ],
    "R3_mainline_straight": [
        TrafficSlot(0, +20, 26, 0.85),
        TrafficSlot(0, +45, 28, 0.60),
        TrafficSlot(0, -15, 24, 0.70),
        TrafficSlot(+1, +18, 26, 0.65),
        TrafficSlot(-1, +18, 26, 0.50),
    ],
    "R3_post_transition_straight": [
        TrafficSlot(0, +18, 24, 0.85),
        TrafficSlot(0, +40, 26, 0.55),
        TrafficSlot(0, -15, 22, 0.65),
        TrafficSlot(+1, +16, 24, 0.55),
        TrafficSlot(-1, +16, 24, 0.45),
    ],
    "R4_mainline_transition": [
        TrafficSlot(0, +18, 18, 0.35),
        TrafficSlot(0, -14, 16, 0.25),
    ],
    "R5_ramp_curve": [
        TrafficSlot(0, +28, 18, 0.75),
        TrafficSlot(0, -22, 16, 0.60),
    ],
    "R6_mainline_merge_approach": [
        TrafficSlot(0, +20, 24, 0.45),
        TrafficSlot(0, -18, 22, 0.30),
    ],
    "R6_exit_to_ramp": [
        TrafficSlot(0, +20, 22, 0.80),
        TrafficSlot(0, -15, 20, 0.65),
        TrafficSlot(+1, +18, 24, 0.50),
    ],
    "R7_merge_core": [
        TrafficSlot(0, +22, 18, 0.70),
        TrafficSlot(0, -18, 16, 0.55),
    ],
    "R8_narrow_channel": [
        TrafficSlot(0, +25, 14, 0.80),
        TrafficSlot(0, -20, 14, 0.65),
        TrafficSlot(+1, +15, 12, 0.50),
    ],
    "R9_post_split_curve": [
        TrafficSlot(0, +28, 20, 0.75),
        TrafficSlot(0, -22, 18, 0.60),
    ],
}

_DEFAULT_TEMPLATE: List[TrafficSlot] = [
    TrafficSlot(0, +22, 22, 0.75),
    TrafficSlot(0, -18, 20, 0.60),
]


class LocalTrafficSpawner:
    """Spawn a small set of structurally correct background vehicles at reset time."""

    _DENSITY_REF = 0.10
    _ROUTE_DENSITY_SCALE: Dict[str, float] = {
        "R4_mainline_transition": 0.25,
        "R6_mainline_merge_approach": 0.6,
    }

    @classmethod
    def get_effective_traffic_density(cls, local_route: str, traffic_density: float) -> float:
        scale = float(cls._ROUTE_DENSITY_SCALE.get(str(local_route), 1.0))
        return float(traffic_density) * scale

    def spawn_base_traffic(
        self,
        env,
        ego_vehicle,
        local_route: str,
        traffic_density: float,
        rng: np.random.RandomState,
    ) -> int:
        ego_lane_index = getattr(ego_vehicle, "lane_index", None)
        if ego_lane_index is None:
            return 0

        current_map = getattr(getattr(env, "engine", None), "current_map", None)
        if current_map is None:
            return 0

        traffic_manager = getattr(getattr(env, "engine", None), "traffic_manager", None)
        if traffic_manager is None or not hasattr(traffic_manager, "_spawn_traffic_vehicle_if_safe"):
            return 0

        road = (ego_lane_index[0], ego_lane_index[1])
        try:
            lanes_on_road = current_map.road_network.graph[road[0]][road[1]]
        except (KeyError, AttributeError):
            return 0

        ego_lane_obj = current_map.road_network.get_lane(ego_lane_index)
        if ego_lane_obj is None:
            return 0
        ego_long = float(ego_lane_obj.local_coordinates(ego_vehicle.position)[0])

        template = ROUTE_TEMPLATES.get(local_route, _DEFAULT_TEMPLATE)
        density_scale = self.get_effective_traffic_density(local_route, float(traffic_density)) / self._DENSITY_REF
        density_scale = max(0.2, min(density_scale, 2.0))

        spawned = 0
        for slot in template:
            effective_p = slot.p * density_scale
            if rng.rand() > effective_p:
                continue

            lane_tuple = self._resolve_lane_index(road, ego_lane_index[2], slot.lane_offset, lanes_on_road)
            if lane_tuple is None:
                continue

            lane_obj = current_map.road_network.get_lane(lane_tuple)
            if lane_obj is None:
                continue

            spawn_long = ego_long + slot.long_offset
            if not (2.0 <= spawn_long <= lane_obj.length - 2.0):
                continue

            v_type = traffic_manager.random_vehicle_type()
            v = traffic_manager._spawn_traffic_vehicle_if_safe(
                v_type,
                {"spawn_lane_index": lane_tuple, "spawn_longitude": float(spawn_long)},
            )
            if v is not None:
                self._set_vehicle_speed(env, v, slot.speed_kmh)
                spawned += 1

        return spawned

    @staticmethod
    def _resolve_lane_index(
        road: tuple,
        ego_lane_idx: int,
        lane_offset: int,
        lanes_on_road,
    ) -> Optional[tuple]:
        target_idx = ego_lane_idx + lane_offset
        if 0 <= target_idx < len(lanes_on_road):
            return (road[0], road[1], target_idx)
        return None

    @staticmethod
    def _set_vehicle_speed(env, vehicle, speed_kmh: float) -> None:
        policy = getattr(getattr(env, "engine", None), "get_policy", lambda *_: None)(vehicle.name)
        if policy is None:
            return
        if hasattr(policy, "NORMAL_SPEED"):
            policy.NORMAL_SPEED = float(speed_kmh)
        if hasattr(policy, "target_speed"):
            policy.target_speed = float(speed_kmh)
