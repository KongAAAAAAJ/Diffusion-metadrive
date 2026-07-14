"""S6-only IDM policy and navigation for a ramp-to-mainline merge."""

from __future__ import annotations

from math import inf

from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.component.road_network import Road
from metadrive.policy.idm_policy import FrontBackObjects, IDMPolicy


class StartEdgeNodeNavigation(NodeNetworkNavigation):
    """Keep the physical spawn edge when it is not on the shortest node path."""

    @staticmethod
    def _build_start_edge_checkpoints(road_network, current_lane_index, destination):
        start_node, end_node = current_lane_index[:2]
        try:
            tail = road_network.shortest_path((end_node,), destination)
        except (AssertionError, KeyError, StopIteration, TypeError):
            return []
        if not tail or tail[0] != end_node:
            return []
        return [start_node, *tail]

    def set_route(self, current_lane_index, destination):
        # Let the parent initialize markers and destination information first.
        super().set_route(current_lane_index, destination)
        self.merge_branch_roads = set()
        self.merge_target_lane = None
        checkpoints = self._build_start_edge_checkpoints(
            self.map.road_network,
            current_lane_index,
            destination,
        )
        if len(checkpoints) < 2:
            return

        graph = self.map.road_network.graph
        if any(end not in graph.get(start, {}) for start, end in zip(checkpoints[:-1], checkpoints[1:])):
            return

        if checkpoints != self.checkpoints:
            self.spawn_road = current_lane_index[:-1]
            self.checkpoints = checkpoints
        else:
            checkpoints = self.checkpoints
        self._target_checkpoints_index = [0, 1] if len(checkpoints) > 2 else [0, 0]
        self.current_road = Road(checkpoints[0], checkpoints[1])
        self.current_ref_lanes = graph[checkpoints[0]][checkpoints[1]]
        if len(checkpoints) > 2:
            self.next_road = Road(checkpoints[1], checkpoints[2])
            self.next_ref_lanes = graph[checkpoints[1]][checkpoints[2]]
        else:
            self.next_road = None
            self.next_ref_lanes = None

        self.final_road = Road(checkpoints[-2], checkpoints[-1])
        self.final_lane = graph[checkpoints[-2]][checkpoints[-1]][-1]
        self.total_length = sum(
            float(graph[start][end][0].length)
            for start, end in zip(checkpoints[:-1], checkpoints[1:])
        )
        self.travelled_length = 0.0
        self._last_long_in_ref_lane = 0.0

        if len(checkpoints) > 2:
            start, branch, merge = checkpoints[:3]
            mainline_lanes = graph.get(start, {}).get(merge)
            if mainline_lanes:
                self.merge_branch_roads = {(start, branch), (branch, merge)}
                self.merge_target_lane = mainline_lanes[-1]


class IDMMergePolicy(IDMPolicy):
    """IDM variant that waits for a safe mainline gap on a natural merge."""

    def __init__(
        self,
        control_object,
        random_seed,
        *,
        merge_front_gap_m: float = 25.0,
        merge_rear_gap_m: float = 15.0,
        merge_creep_speed_kmh: float = 5.0,
        merge_cruise_speed_kmh: float = 24.0,
    ):
        super().__init__(control_object, random_seed)
        self.merge_front_gap_m = float(merge_front_gap_m)
        self.merge_rear_gap_m = float(merge_rear_gap_m)
        self.merge_creep_speed_kmh = float(merge_creep_speed_kmh)
        self.merge_cruise_speed_kmh = float(merge_cruise_speed_kmh)
        self.NORMAL_SPEED = self.merge_cruise_speed_kmh
        self.target_speed = self.merge_cruise_speed_kmh

    def lane_change_policy(self, all_objects):
        parent_result = super().lane_change_policy(all_objects)
        self._record_merge_state(active=False)

        target_lane = self._find_merge_target_lane()
        if target_lane is None:
            return parent_result

        search_distance = max(
            float(self.MAX_LONG_DIST),
            self.merge_front_gap_m,
            self.merge_rear_gap_m,
        )
        surrounding = FrontBackObjects.get_find_front_back_objs(
            all_objects,
            target_lane,
            self.control_object.position,
            max_distance=search_distance,
        )
        front_gap = float(surrounding.front_min_distance()) if surrounding.has_front_object() else inf
        rear_gap = float(surrounding.back_min_distance()) if surrounding.has_back_object() else inf
        gap_accepted = front_gap >= self.merge_front_gap_m and rear_gap >= self.merge_rear_gap_m
        self.target_speed = self.merge_cruise_speed_kmh if gap_accepted else self.merge_creep_speed_kmh
        self._record_merge_state(
            active=True,
            front_gap=front_gap,
            rear_gap=rear_gap,
            accepted=gap_accepted,
        )
        return parent_result

    def _find_merge_target_lane(self):
        vehicle = self.control_object
        navigation = getattr(vehicle, "navigation", None)
        current_lane = getattr(vehicle, "lane", None)
        current_index = getattr(current_lane, "index", None)
        if current_index is None:
            return None
        current_road = tuple(current_index[:2])
        branch_roads = set(getattr(navigation, "merge_branch_roads", ()) or ())
        if current_road not in branch_roads:
            return None
        return getattr(navigation, "merge_target_lane", None)

    def _record_merge_state(
        self,
        *,
        active: bool,
        front_gap: float = inf,
        rear_gap: float = inf,
        accepted: bool = False,
    ) -> None:
        self.action_info["merge_active"] = bool(active)
        self.action_info["merge_front_gap"] = float(front_gap)
        self.action_info["merge_rear_gap"] = float(rear_gap)
        self.action_info["merge_gap_accepted"] = bool(accepted)
