"""S6-only IDM policy and navigation for a ramp-to-mainline merge."""

from __future__ import annotations

from math import inf

from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.component.road_network import Road
from metadrive.policy.idm_policy import FrontBackObjects, IDMPolicy

from envs.diffusion_envs.ground_truth_idm_policy import GroundTruthIDMMixin


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
        self.merge_force_road = None
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
                self.merge_force_road = (branch, merge)
                self.merge_target_lane = mainline_lanes[-1]


class IDMMergePolicy(GroundTruthIDMMixin, IDMPolicy):
    """IDM variant that waits for a safe mainline gap on a natural merge."""

    def __init__(
        self,
        control_object,
        random_seed,
        *,
        merge_front_gap_m: float = 5.0,
        merge_rear_gap_m: float = 5.0,
        merge_creep_speed_kmh: float = 20.0,
        merge_cruise_speed_kmh: float = 24.0,
        merge_activation_step: int = 2,
    ):
        super().__init__(control_object, random_seed)
        self.merge_front_gap_m = float(merge_front_gap_m)
        self.merge_rear_gap_m = float(merge_rear_gap_m)
        self.merge_creep_speed_kmh = float(merge_creep_speed_kmh)
        self.merge_cruise_speed_kmh = float(merge_cruise_speed_kmh)
        self.merge_activation_step = int(merge_activation_step)
        if self.merge_activation_step < 0:
            raise ValueError("merge_activation_step must be non-negative")
        self.NORMAL_SPEED = self.merge_cruise_speed_kmh
        self.target_speed = self.merge_cruise_speed_kmh
        self.merge_completed = False
        self.merge_policy_step = 0

    def reset(self):
        super().reset()
        self.merge_completed = False
        self.merge_policy_step = 0

    def lane_change_policy(self, all_objects):
        current_step = self.merge_policy_step
        self.merge_policy_step += 1
        parent_result = super().lane_change_policy(all_objects)
        self._update_merge_completed()
        self._record_merge_state(
            active=False,
            force_active=False,
            completed=self.merge_completed,
        )
        if self.merge_completed:
            return parent_result[0], parent_result[1], self.control_object.lane

        target_lane = self._find_merge_target_lane()
        if target_lane is None:
            return parent_result

        force_active = current_step >= self.merge_activation_step

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
        rear_ttc = self._rear_ttc_s(surrounding.back_object(), rear_gap)
        gap_accepted = front_gap >= self.merge_front_gap_m and rear_gap >= self.merge_rear_gap_m and rear_ttc >= 4.0  # *前后间距 10m 10m, 后车TTC >= 2s

        global_config = getattr(getattr(self.control_object, "engine", None), "global_config", {})
        if bool(global_config.get("merge_policy_debug", False)):
            print(
                f"Front gap: {front_gap:.2f} m, Rear gap: {rear_gap:.2f} m, "
                f"Rear TTC: {rear_ttc:.2f} s, Gap accepted: {gap_accepted}, "
                f"Force active: {force_active}"
            )

        if gap_accepted:
            rear_speed_kmh = self._vehicle_speed_mps(surrounding.back_object()) * 3.6
            self.target_speed = max(0.6 * rear_speed_kmh, self.merge_cruise_speed_kmh)
        else:
            self.target_speed = self.merge_creep_speed_kmh
        self._record_merge_state(
            active=True,
            force_active=force_active,
            front_gap=front_gap,
            rear_gap=rear_gap,
            rear_ttc=rear_ttc,
            accepted=gap_accepted,
            completed=self.merge_completed,
        )
        if force_active and gap_accepted:
            return surrounding.front_object(), surrounding.front_min_distance(), target_lane
        if force_active:
            return parent_result[0], parent_result[1], self.control_object.lane
        return parent_result

    def _is_force_merge_road(self) -> bool:
        vehicle = self.control_object
        navigation = getattr(vehicle, "navigation", None)
        current_lane = getattr(vehicle, "lane", None)
        current_index = getattr(current_lane, "index", None)
        if current_index is None:
            return False
        force_road = getattr(navigation, "merge_force_road", None)
        return force_road is not None and tuple(current_index[:2]) == tuple(force_road)

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

    def _rear_ttc_s(self, rear_object, rear_gap: float) -> float:
        if rear_object is None or rear_gap == inf:
            return inf
        ego_speed = self._vehicle_speed_mps(self.control_object)
        rear_speed = self._vehicle_speed_mps(rear_object)
        closing_speed = rear_speed - ego_speed
        if closing_speed <= 1e-6:
            return inf
        return float(rear_gap) / float(closing_speed)

    @staticmethod
    def _vehicle_speed_mps(vehicle) -> float:
        speed_km_h = getattr(vehicle, "speed_km_h", None)
        if speed_km_h is not None:
            return float(speed_km_h) / 3.6
        speed = getattr(vehicle, "speed", None)
        if speed is not None:
            return float(speed)
        velocity = getattr(vehicle, "velocity", None)
        if velocity is not None:
            return float((float(velocity[0]) ** 2 + float(velocity[1]) ** 2) ** 0.5)
        return 0.0

    def _update_merge_completed(self) -> None:
        if self.merge_completed:
            return
        vehicle = self.control_object
        navigation = getattr(vehicle, "navigation", None)
        target_lane = getattr(navigation, "merge_target_lane", None)
        target_index = getattr(target_lane, "index", None)
        current_lane = getattr(vehicle, "lane", None)
        current_index = getattr(current_lane, "index", None)
        if target_index is None or current_index is None:
            return
        self.merge_completed = tuple(current_index[:2]) == tuple(target_index[:2])

    def _record_merge_state(
        self,
        *,
        active: bool,
        force_active: bool = False,
        front_gap: float = inf,
        rear_gap: float = inf,
        rear_ttc: float = inf,
        accepted: bool = False,
        completed: bool = False,
    ) -> None:
        self.action_info["merge_active"] = bool(active)
        self.action_info["merge_force_active"] = bool(force_active)
        self.action_info["merge_front_gap"] = float(front_gap)
        self.action_info["merge_rear_gap"] = float(rear_gap)
        self.action_info["merge_rear_ttc_s"] = float(rear_ttc)
        self.action_info["merge_gap_accepted"] = bool(accepted)
        self.action_info["merge_completed"] = bool(completed)
