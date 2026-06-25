from __future__ import annotations

from typing import List, Optional

import numpy as np

from metadrive.manager.spawn_manager import SpawnManager
from scenarios.definitions import SCENARIO_BY_ID


class _RouteRoadRef:
    def __init__(self, start_node: str, end_node: str):
        self.start_node = start_node
        self.end_node = end_node

    def lane_index(self, lane_idx: int):
        return (self.start_node, self.end_node, lane_idx)


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
    def _get_positive_block_network_road_by_index(cls, block, road_index):
        if block is None or road_index is None:
            return None
        lane_groups = getattr(getattr(block, "block_network", None), "get_positive_lanes", lambda: [])() or []
        try:
            lane_group = lane_groups[int(road_index)]
        except (IndexError, TypeError, ValueError):
            return None
        if not lane_group:
            return None
        lane_index = getattr(lane_group[0], "index", None)
        if lane_index is None:
            return None
        return _RouteRoadRef(lane_index[0], lane_index[1])

    @classmethod
    def _get_first_positive_block_network_road(cls, block):
        if block is None:
            return None
        lane_groups = getattr(getattr(block, "block_network", None), "get_positive_lanes", lambda: [])() or []
        for lane_group in lane_groups:
            if not lane_group:
                continue
            lane = lane_group[0]
            lane_index = getattr(lane, "index", None)
            if lane_index is None:
                continue
            return _RouteRoadRef(lane_index[0], lane_index[1])
        return None

    @classmethod
    def _get_last_positive_block_network_road(cls, block):
        if block is None:
            return None
        lane_groups = getattr(getattr(block, "block_network", None), "get_positive_lanes", lambda: [])() or []
        for lane_group in reversed(lane_groups):
            if not lane_group:
                continue
            lane = lane_group[0]
            lane_index = getattr(lane, "index", None)
            if lane_index is None:
                continue
            return _RouteRoadRef(lane_index[0], lane_index[1])
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
        road = cls._get_first_positive_block_network_road(block)
        if road is not None:
            return road
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
        road = self._get_last_positive_block_network_road(last_block)
        if road is None:
            road = self._get_first_positive_respawn_road(last_block)
        if road is None:
            road = self._get_first_positive_socket_road(last_block)
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

    def _get_road_lane_num(self, road) -> int:
        """Return the actual number of lanes for *road* from the current map.

        `self.lane_num` is fixed at the value of ``map_config["lane_num"]`` (the
        mainline lane count) and is therefore wrong for single-lane ramp roads
        (OneWayStraight / OneWayCurve blocks).  Looking up the actual count from
        the road-network graph prevents IndexError when ego is spawned on a
        one-lane ramp.
        """
        current_map = getattr(self.engine, "current_map", None)
        if current_map is not None:
            try:
                lanes = current_map.road_network.graph[road.start_node][road.end_node]
                return len(lanes)
            except (KeyError, AttributeError, TypeError):
                pass
        return self.lane_num

    def _auto_fill_spawn_roads_randomly(self, spawn_roads):
        """Like the base implementation but uses the road's actual lane count."""
        import math
        from metadrive.component.pgblock.first_block import FirstPGBlock
        from metadrive.utils import Config

        num_slots = int(math.floor(self.exit_length / self.RESPAWN_REGION_LONGITUDE))
        interval = self.exit_length / num_slots
        self._longitude_spawn_interval = interval

        agent_configs = []
        safe_spawn_places = []
        for road in spawn_roads:
            road_lane_num = self._get_road_lane_num(road)
            for lane_idx in range(road_lane_num):
                for j in range(num_slots):
                    long = 1 / 2 * self.RESPAWN_REGION_LONGITUDE + j * self.RESPAWN_REGION_LONGITUDE
                    lane_tuple = road.lane_index(lane_idx)
                    agent_configs.append(
                        Config(
                            dict(
                                identifier="|".join((str(s) for s in lane_tuple + (j,))),
                                config={
                                    "spawn_lane_index": lane_tuple,
                                    "spawn_longitude": long,
                                    "spawn_lateral": 0,
                                },
                            ),
                            unchangeable=True,
                        )
                    )
                    if j == 0:
                        safe_spawn_places.append(agent_configs[-1])
        return agent_configs, safe_spawn_places

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

    def _fixed_route_spawn_enabled(self) -> bool:
        return bool(self.engine.global_config.get("platoon_fixed_route_spawn", False))

    def _fixed_route_spawn_lane_idx(self, lane_count: int) -> int:
        scenario = self._get_spawn_scenario()
        lane_id = getattr(scenario, "ego_spawn_lane_id", None)
        if lane_id is not None:
            return max(0, min(int(lane_id), max(lane_count - 1, 0)))
        probabilities = self._get_spawn_lane_probabilities()
        if probabilities:
            positive = {
                str(lane_label): float(weight)
                for lane_label, weight in probabilities.items()
                if float(weight) > 0.0
            }
            if positive:
                # Fixed ego spawn should not depend on RNG.  When a scenario
                # supplies probabilities, use its most likely lane deterministically.
                lane_label = max(sorted(positive), key=lambda key: positive[key])
                return self._resolve_named_lane_index(lane_count, lane_label)
        preference = self._get_spawn_lane_preference()
        if preference in {"rightmost", "leftmost", "middle"}:
            return self._resolve_named_lane_index(lane_count, str(preference))
        return 0

    def _fixed_route_spawn_lead_longitude(self, lane_length: float, gap_m: float) -> float:
        tail_buffer = float(self.engine.global_config.get("platoon_spawn_tail_buffer_m", 6.0))
        front_buffer = float(self.engine.global_config.get("platoon_spawn_front_buffer_m", 8.0))
        num_agents = int(self.engine.global_config.get("num_agents", 1))
        min_lead = gap_m * max(num_agents - 1, 0) + tail_buffer
        max_lead = max(float(lane_length) - front_buffer, 2.0)
        scenario = self._get_spawn_scenario()
        longitude_from_start = getattr(scenario, "ego_spawn_longitude_m", None)
        if longitude_from_start is not None:
            requested = float(longitude_from_start)
            return float(np.clip(requested, min_lead, max_lead))
        distance_to_route_end = getattr(scenario, "ego_spawn_distance_to_route_end_m", None)
        if distance_to_route_end is not None:
            requested = float(lane_length) - float(distance_to_route_end)
            return float(np.clip(requested, min_lead, max_lead))
        return float(min(min_lead, max_lead))

    def _select_fixed_route_spawn_road(self, route_roads):
        if not route_roads:
            return None
        scenario = self._get_spawn_scenario()
        reference_block_id = getattr(scenario, "ego_spawn_reference_block_id", None)
        if not reference_block_id:
            return route_roads[0]
        reference_kind = getattr(scenario, "ego_spawn_reference_kind", None)
        if reference_kind == "block_internal_road":
            current_map = getattr(self.engine, "current_map", None)
            block = self._get_block_by_graph_id(current_map, str(reference_block_id))
            road = self._get_positive_block_network_road_by_index(
                block,
                getattr(scenario, "ego_spawn_internal_road_index", None),
            )
            if road is not None:
                return road
        try:
            block_ids = self._get_configured_route_block_ids()
            block_index = block_ids.index(str(reference_block_id))
        except (ValueError, TypeError):
            current_map = getattr(self.engine, "current_map", None)
            road = self._get_first_positive_route_road(
                self._get_block_by_graph_id(current_map, str(reference_block_id))
            )
            if road is not None:
                return road
            return route_roads[0]
        if block_index >= len(route_roads):
            return route_roads[0]
        return route_roads[block_index]

    def _apply_fixed_route_spawn_configs(self) -> bool:
        """Place all ego agents at deterministic positions on the first route road.

        This makes ``ego_spawn_zones`` represent the final platoon start before
        the traffic manager generates background vehicles, so traffic can avoid
        the real start positions instead of the temporary default spawn slots.
        """
        if not self._fixed_route_spawn_enabled():
            return False
        current_map = getattr(self.engine, "current_map", None)
        if current_map is None:
            return False
        route_roads = self.get_main_route_spawn_roads(current_map)
        if not route_roads:
            return False
        road = self._select_fixed_route_spawn_road(route_roads)
        if road is None:
            return False
        try:
            lanes = current_map.road_network.graph[road.start_node][road.end_node]
        except (AttributeError, KeyError, TypeError):
            return False
        if not lanes:
            return False

        lane_idx = max(0, min(self._fixed_route_spawn_lane_idx(len(lanes)), len(lanes) - 1))
        lane = lanes[lane_idx]
        lane_index = tuple(road.lane_index(lane_idx))
        num_agents = int(self.engine.global_config.get("num_agents", 1))
        gap_m = float(self.engine.global_config.get("platoon_spawn_gap_m", self.DEFAULT_TRAFFIC_GAP))
        lead_long = self._fixed_route_spawn_lead_longitude(float(getattr(lane, "length", 0.0)), gap_m)
        speed_m_s = float(self.engine.global_config.get("initial_speed_km_h", 25.0)) / 3.6

        existing_configs = self.engine.global_config.get("agent_configs", {}) or {}
        agent_configs = {}
        for idx in range(num_agents):
            agent_id = f"agent{idx}"
            spawn_longitude = max(float(lead_long) - idx * gap_m, 2.0)
            config = dict(existing_configs.get(agent_id, {}))
            config.update(
                {
                    "spawn_lane_index": lane_index,
                    "spawn_longitude": float(spawn_longitude),
                    "spawn_lateral": 0.0,
                    "spawn_velocity": (speed_m_s, 0.0),
                    "spawn_velocity_car_frame": True,
                    "_specified_spawn_lane": True,
                }
            )
            if not config.get("destination", None):
                config = self.update_destination_for(agent_id, config)
            agent_configs[agent_id] = config

        self.engine.global_config["agent_configs"] = agent_configs
        return True

    def reset(self):
        self._refresh_main_route_spawn_roads()
        super().reset()
        self._apply_spawn_lane_preference()
        self._apply_fixed_route_spawn_configs()
        self._cache_ego_spawn_zones()

    def _get_spawn_lane_preference(self) -> str | None:
        scenario = self._get_spawn_scenario()
        if scenario is None:
            return None
        return scenario.ego_spawn_lane_preference

    def _get_spawn_scenario(self):
        scenario_id = self.engine.global_config.get("scenario_id")
        if not scenario_id:
            return None
        return SCENARIO_BY_ID.get(str(scenario_id))

    def _get_spawn_lane_probabilities(self) -> dict[str, float] | None:
        scenario = self._get_spawn_scenario()
        if scenario is None:
            return None
        return scenario.ego_spawn_lane_probabilities

    @staticmethod
    def _resolve_named_lane_index(lane_count: int, lane_label: str) -> int:
        if lane_count <= 0:
            return 0
        if lane_label == "rightmost":
            return lane_count - 1
        if lane_label == "leftmost":
            return 0
        if lane_label == "middle":
            return min(max(int(round((lane_count - 1) / 2.0)), 0), lane_count - 1)
        return lane_count - 1

    def _apply_spawn_lane_preference(self) -> None:
        preference = self._get_spawn_lane_preference()
        probabilities = self._get_spawn_lane_probabilities()
        current_map = getattr(self.engine, "current_map", None)
        road_network = getattr(current_map, "road_network", None)
        if road_network is None:
            return
        agent_configs = self.engine.global_config.get("agent_configs", {}) or {}
        for config in agent_configs.values():
            lane_index = tuple(config.get("spawn_lane_index", ()))
            if len(lane_index) != 3:
                continue
            try:
                lanes = road_network.graph[lane_index[0]][lane_index[1]]
            except (AttributeError, KeyError, TypeError):
                continue
            if not lanes:
                continue
            if probabilities:
                lane_labels = []
                weights = []
                for lane_label, weight in probabilities.items():
                    if float(weight) <= 0.0:
                        continue
                    lane_labels.append(str(lane_label))
                    weights.append(float(weight))
                if lane_labels:
                    probs = np.asarray(weights, dtype=np.float64)
                    probs = probs / np.sum(probs)
                    sampled_label = str(self.np_random.choice(lane_labels, p=probs))
                    preferred_lane_idx = self._resolve_named_lane_index(len(lanes), sampled_label)
                    config["spawn_lane_index"] = (lane_index[0], lane_index[1], preferred_lane_idx)
                    continue
            if preference not in {"rightmost", "leftmost"}:
                continue
            preferred_lane_idx = len(lanes) - 1 if preference == "rightmost" else 0
            config["spawn_lane_index"] = (lane_index[0], lane_index[1], preferred_lane_idx)

    def update_destination_for(self, agent_id, vehicle_config):
        current_map = getattr(self.engine, "current_map", None)
        destination = self._resolve_destination_node(current_map)
        if destination is None:
            return super().update_destination_for(agent_id, vehicle_config)
        updated = dict(vehicle_config)
        updated["destination"] = destination
        return updated
