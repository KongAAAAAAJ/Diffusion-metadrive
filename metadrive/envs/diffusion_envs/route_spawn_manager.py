from __future__ import annotations

from typing import Iterable

from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.manager.spawn_manager import SpawnManager


def _road_key(road) -> tuple[str, str]:
    return (road.start_node, road.end_node)


class RouteAwareSpawnManager(SpawnManager):
    """Spawn ego only on the forward main-route roads of the current map."""

    DEFAULT_ROUTE_START = (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3)
    DEFAULT_TRAFFIC_GAP = 10.0

    def __init__(self):
        super().__init__()
        self.ego_spawn_zones = []

    def _iter_positive_respawn_roads(self, current_map) -> Iterable:
        for block in getattr(current_map, "blocks", []) or []:
            for road in getattr(block, "get_respawn_roads", lambda: [])() or []:
                if hasattr(road, "is_negative_road") and road.is_negative_road():
                    continue
                yield road

    def _select_next_route_road(self, current_map, current_road):
        for block in getattr(current_map, "blocks", [])[1:]:
            pre_socket = getattr(block, "pre_block_socket", None)
            pre_road = getattr(pre_socket, "positive_road", None)
            if pre_road is None or _road_key(pre_road) != _road_key(current_road):
                continue
            block_candidates = [
                road for road in getattr(block, "get_respawn_roads", lambda: [])() or []
                if not (hasattr(road, "is_negative_road") and road.is_negative_road())
                and road.start_node == current_road.end_node
            ]
            if block_candidates:
                return sorted(block_candidates, key=_road_key)[0]
        return None

    def get_main_route_spawn_roads(self, current_map):
        if current_map is None:
            return []

        start_node, end_node = tuple(
            self.engine.global_config.get("ego_spawn_route_start", self.DEFAULT_ROUTE_START)
        )
        start_candidates = [
            road for road in self._iter_positive_respawn_roads(current_map)
            if _road_key(road) == (start_node, end_node)
        ]
        if not start_candidates:
            return []

        current_road = sorted(start_candidates, key=_road_key)[0]
        route_roads = [current_road]
        seen_road_keys = {_road_key(current_road)}

        while True:
            next_road = self._select_next_route_road(current_map, current_road)
            if next_road is None:
                break
            next_key = _road_key(next_road)
            if next_key in seen_road_keys:
                break
            route_roads.append(next_road)
            seen_road_keys.add(next_key)
            current_road = next_road

        return route_roads

    def _refresh_main_route_spawn_roads(self):
        current_map = getattr(self.engine, "current_map", None)
        if current_map is None and hasattr(self, "current_map"):
            current_map = self.current_map
        route_roads = self.get_main_route_spawn_roads(current_map)
        if not route_roads:
            return
        self.refresh_spawn_roads(route_roads)

    def refresh_spawn_roads(self, spawn_roads):
        agent_configs, safe_spawn_places = self._auto_fill_spawn_roads_randomly(spawn_roads)
        self.available_agent_configs = agent_configs
        self.safe_spawn_places = {
            place["identifier"]: place for place in safe_spawn_places
        }
        self.spawn_roads = list(spawn_roads)
        self.need_update_spawn_places = True
        self.engine.global_config["spawn_roads"] = list(spawn_roads)

    def _cache_ego_spawn_zones(self):
        scale = float(self.engine.global_config.get("ego_spawn_buffer_scale", 1.0))
        buffer_longitudinal = self.DEFAULT_TRAFFIC_GAP * scale
        self.ego_spawn_zones = []
        for config in (self.engine.global_config.get("agent_configs", {}) or {}).values():
            lane_index = tuple(config["spawn_lane_index"])
            self.ego_spawn_zones.append(
                {
                    "road": tuple(lane_index[:2]),
                    "spawn_lane_index": lane_index,
                    "spawn_longitude": float(config["spawn_longitude"]),
                    "spawn_lateral": float(config.get("spawn_lateral", 0.0)),
                    "buffer_longitudinal": buffer_longitudinal,
                }
            )

    def reset(self):
        self._refresh_main_route_spawn_roads()
        super().reset()
        self._cache_ego_spawn_zones()
