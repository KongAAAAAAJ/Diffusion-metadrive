from __future__ import annotations

from typing import List, Optional

from metadrive.manager.spawn_manager import SpawnManager


def _road_key(road) -> tuple[str, str]:
    return (road.start_node, road.end_node)


class RouteAwareSpawnManager(SpawnManager):
    """Spawn ego only on a **manually** specified main-route block chain.

    与之前的实现不同，这里不再做任何 socket 自动搜索 / auto-walk；
    全部路由信息来自 global_config["ego_main_route_block_ids"]：一个
    按顺序写死的 graph_block_id 列表（例如 s0 -> c0 -> ... -> c4）。
    """

    DEFAULT_TRAFFIC_GAP = 10.0

    def __init__(self):
        super().__init__()
        self.ego_spawn_zones = []

    # ------------------------------------------------------------------
    # 工具：block 查找 + 正向 respawn road 提取
    # ------------------------------------------------------------------
    @staticmethod
    def _get_block_by_graph_id(current_map, block_id):
        if current_map is None or block_id is None:
            return None
        for block in getattr(current_map, "blocks", []) or []:
            if getattr(block, "graph_block_id", None) == block_id:
                return block
        return None

    @classmethod
    def _get_first_positive_respawn_road(cls, block):
        if block is None:
            return None
        roads = [
            road for road in getattr(block, "get_respawn_roads", lambda: [])() or []
            if not (hasattr(road, "is_negative_road") and road.is_negative_road())
        ]
        if not roads:
            return None
        return sorted(roads, key=_road_key)[0]

    @classmethod
    def _get_first_positive_socket_road(cls, block):
        if block is None:
            return None
        sockets = getattr(block, "get_socket_list", lambda: [])() or []
        roads = [socket.positive_road for socket in sockets if getattr(socket, "positive_road", None) is not None]
        if not roads:
            sockets_dict = getattr(block, "_sockets", {}) or {}
            roads = [
                socket.positive_road
                for socket in sockets_dict.values()
                if getattr(socket, "positive_road", None) is not None
            ]
        if not roads:
            return None
        return sorted(roads, key=_road_key)[0]

    @classmethod
    def _get_first_positive_route_road(cls, block):
        road = cls._get_first_positive_respawn_road(block)
        if road is not None:
            return road
        return cls._get_first_positive_socket_road(block)

    # ------------------------------------------------------------------
    # 主线路由构造
    # ------------------------------------------------------------------
    def _get_configured_route_block_ids(self) -> List[str]:
        raw = self.engine.global_config.get("ego_main_route_block_ids") or []
        block_ids = [str(bid) for bid in raw if bid]
        if not block_ids:
            raise ValueError(
                "ego_main_route_block_ids must be a non-empty list of graph_block_id strings."
            )
        return block_ids

    def get_main_route_spawn_roads(self, current_map):
        """根据手动 block 链返回正向 route roads 列表。"""
        if current_map is None:
            return []

        block_ids = self._get_configured_route_block_ids()
        route_roads = []
        for block_id in block_ids:
            block = self._get_block_by_graph_id(current_map, block_id)
            if block is None:
                raise ValueError(
                    f"ego_main_route_block_ids references unknown graph_block_id: {block_id!r}"
                )
            road = self._get_first_positive_route_road(block)
            if road is None:
                raise ValueError(
                    f"Block {block_id!r} has no positive route road; "
                    "cannot be used in ego_main_route_block_ids."
                )
            route_roads.append(road)
        return route_roads

    def _resolve_destination_node(self, current_map) -> Optional[str]:
        """终点 = 手动 block 链最后一个 block 的首条正向 respawn road 的 end_node。"""
        block_ids = self._get_configured_route_block_ids()
        last_block = self._get_block_by_graph_id(current_map, block_ids[-1])
        if last_block is None:
            raise ValueError(
                f"Destination block {block_ids[-1]!r} not found in current map."
            )
        road = self._get_first_positive_route_road(last_block)
        if road is None:
            raise ValueError(
                f"Destination block {block_ids[-1]!r} has no positive route road."
            )
        return road.end_node

    # ------------------------------------------------------------------
    # SpawnManager overrides
    # ------------------------------------------------------------------
    def _refresh_main_route_spawn_roads(self):
        current_map = getattr(self.engine, "current_map", None)
        if current_map is None and hasattr(self, "current_map"):
            current_map = self.current_map
        route_roads = self.get_main_route_spawn_roads(current_map)
        if not route_roads:
            return
        # 只用链里的第一段作为 ego 的出生路段，后续路段仅作为路由参考，
        # 但保留原行为（把整条链作为 spawn_roads 传入）以便 traffic manager
        # 等消费者可以读取完整主线。
        self.refresh_spawn_roads(route_roads[:1])

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

    def update_destination_for(self, agent_id, vehicle_config):
        current_map = getattr(self.engine, "current_map", None)
        destination = self._resolve_destination_node(current_map)
        if destination is None:
            return super().update_destination_for(agent_id, vehicle_config)
        updated = dict(vehicle_config)
        updated["destination"] = destination
        return updated
