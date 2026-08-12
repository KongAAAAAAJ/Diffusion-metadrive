"""S6-only IDM policy and navigation for a ramp-to-mainline merge."""

from __future__ import annotations

from math import inf, sqrt

import numpy as np

from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from metadrive.component.road_network import Road
from metadrive.component.vehicle.PID_controller import PIDController
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
                downstream_lanes = (
                    graph.get(merge, {}).get(checkpoints[3])
                    if len(checkpoints) > 3
                    else None
                )
                target_lanes = downstream_lanes or mainline_lanes
                # S6 uses the continuous middle lane on the road after the
                # conflict point.  The connector naturally lands on the
                # outermost lane, so the merge policy must keep steering left
                # until this downstream target is actually reached.
                self.merge_target_lane = (
                    target_lanes[-2]
                    if len(target_lanes) >= 2
                    else target_lanes[-1]
                )


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
        merge_rear_ttc_min_s: float = 4.0,
        merge_lateral_kp: float = 0.7,
    ):
        super().__init__(control_object, random_seed)
        self.merge_front_gap_m = float(merge_front_gap_m)
        self.merge_rear_gap_m = float(merge_rear_gap_m)
        self.merge_creep_speed_kmh = float(merge_creep_speed_kmh)
        self.merge_cruise_speed_kmh = float(merge_cruise_speed_kmh)
        self.merge_activation_step = int(merge_activation_step)
        self.merge_rear_ttc_min_s = float(merge_rear_ttc_min_s)
        if self.merge_activation_step < 0:
            raise ValueError("merge_activation_step must be non-negative")
        self.NORMAL_SPEED = self.merge_cruise_speed_kmh
        self.target_speed = self.merge_cruise_speed_kmh
        self.merge_completed = False
        self.merge_sweep_committed = False
        self.merge_policy_step = 0
        # The connector reaches the outside lane at a shallow angle, then S6
        # must cross one additional lane width into the continuous middle
        # mainline.  A 0.4 proportional gain starts the sweep but can remain
        # on lane 2 until the sampled 6--10 m corridor has already closed for
        # slower actors.  Use a responsive merge-only controller so an
        # accepted sweep physically reaches lane 1 inside that corridor. A
        # slow actor spends longer on the curved connector and needs a faster
        # lateral convergence before its longitudinal 6--10 m window closes.
        self.lateral_pid = PIDController(float(merge_lateral_kp), 0.002, 0.05)

    def reset(self):
        super().reset()
        self.merge_completed = False
        self.merge_sweep_committed = False
        self.merge_policy_step = 0

    def lane_change_policy(self, all_objects):
        current_step = self.merge_policy_step
        self.merge_policy_step += 1
        parent_result = super().lane_change_policy(all_objects)
        self._update_merge_completed()
        designated_gap_completed = bool(
            getattr(
                self.control_object,
                "scenario_designated_gap_completed",
                False,
            )
        )
        if designated_gap_completed:
            target_gap_id = str(
                getattr(self.control_object, "scenario_target_gap_id", "")
            )
            post_merge_speed_kmh = min(float(self.merge_cruise_speed_kmh), 20.0)
            self.NORMAL_SPEED = post_merge_speed_kmh
            self.target_speed = post_merge_speed_kmh
        self._record_merge_state(
            active=False,
            force_active=False,
            completed=self.merge_completed,
        )
        if self.merge_completed:
            designated_front = getattr(
                self.control_object, "scenario_target_front_vehicle", None
            )
            actor_index = tuple(
                getattr(self.control_object, "lane_index", ()) or ()
            )
            front_index = tuple(
                getattr(designated_front, "lane_index", ()) or ()
            ) if designated_front is not None else ()
            target_gap_id = str(
                getattr(self.control_object, "scenario_target_gap_id", "")
            )
            escape_completed = bool(
                getattr(
                    self.control_object,
                    "scenario_ego_escape_completed",
                    False,
                )
            )
            if (
                designated_front is not None
                and len(actor_index) >= 3
                and len(front_index) >= 3
                and (
                    int(actor_index[2]) == int(front_index[2])
                    or (
                        target_gap_id == "agent0-agent1"
                        and not escape_completed
                    )
                )
            ):
                front_distance = float(
                    np.linalg.norm(
                        np.asarray(designated_front.position[:2], dtype=float)
                        - np.asarray(self.control_object.position[:2], dtype=float)
                    )
                )
                return designated_front, front_distance, self.control_object.lane
            return parent_result[0], parent_result[1], self.control_object.lane

        target_lane = self._find_merge_target_lane()
        if designated_gap_completed and target_lane is not None:
            # The OBB sweep can prove gap occupation one or two lane-label
            # updates before the actor centre is on the continuous lane.
            # Keep steering to the explicit target lane while applying the
            # post-merge speed; do not let stock IDM select the terminating
            # outside lane during this short label-transition interval.
            return parent_result[0], parent_result[1], target_lane
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
        front_object = surrounding.front_object()
        rear_object = surrounding.back_object()
        designated_front = getattr(
            self.control_object, "scenario_target_front_vehicle", None
        )
        designated_rear = getattr(
            self.control_object, "scenario_target_rear_vehicle", None
        )
        designated_gap_bound = bool(
            designated_front is not None and designated_rear is not None
        )
        if designated_gap_bound:
            # The connector overlaps several mainline lane families. Generic
            # nearest-front lookup can consequently select a parallel-lane
            # vehicle (or no rear vehicle) even though the recipe's declared
            # front/actor/rear corridor is geometrically valid. Project the
            # three bound roles onto the explicit destination lane instead.
            front_gap = inf
            rear_gap = inf
            front_object = None
            rear_object = None
            try:
                front_s = float(
                    target_lane.local_coordinates(designated_front.position)[0]
                )
                actor_s = float(
                    target_lane.local_coordinates(self.control_object.position)[0]
                )
                rear_s = float(
                    target_lane.local_coordinates(designated_rear.position)[0]
                )
                if front_s > actor_s:
                    front_gap = front_s - actor_s
                    front_object = designated_front
                if actor_s > rear_s:
                    rear_gap = actor_s - rear_s
                    rear_object = designated_rear
            except (AttributeError, TypeError, ValueError):
                pass
        rear_ttc = self._rear_ttc_s(rear_object, rear_gap)
        gap_accepted = (
            (
                not designated_gap_bound
                or (front_object is designated_front and rear_object is designated_rear)
            )
            and front_gap >= self.merge_front_gap_m
            and rear_gap >= self.merge_rear_gap_m
            and rear_ttc >= self.merge_rear_ttc_min_s
        )
        if force_active and gap_accepted:
            self.merge_sweep_committed = True

        global_config = getattr(getattr(self.control_object, "engine", None), "global_config", {})
        if bool(global_config.get("merge_policy_debug", False)):
            print(
                f"Front gap: {front_gap:.2f} m, Rear gap: {rear_gap:.2f} m, "
                f"Rear TTC: {rear_ttc:.2f} s, Gap accepted: {gap_accepted}, "
                f"Force active: {force_active}"
            )

        if gap_accepted:
            rear_speed_kmh = self._vehicle_speed_mps(rear_object) * 3.6
            self.target_speed = max(0.6 * rear_speed_kmh, self.merge_cruise_speed_kmh)
        else:
            self.target_speed = self.merge_creep_speed_kmh
        if self.merge_sweep_committed:
            # Once the lateral sweep is atomic, do not re-enter creep on the
            # next transient gap sample.  That abrupt slowdown invalidates the
            # ego planner's constant-speed actor prediction and lets the
            # designated rear ego close below the unchanged 5 m hard gate.
            target_gap_id = str(
                getattr(self.control_object, "scenario_target_gap_id", "")
            )
            self.target_speed = (
                min(float(self.merge_cruise_speed_kmh) + 3.0, 27.0)
                if target_gap_id == "agent1-agent2"
                else float(self.merge_cruise_speed_kmh)
            )
        self._record_merge_state(
            active=True,
            force_active=force_active,
            front_gap=front_gap,
            rear_gap=rear_gap,
            rear_ttc=rear_ttc,
            accepted=gap_accepted,
            completed=self.merge_completed,
        )
        self.action_info["merge_sweep_committed"] = bool(
            self.merge_sweep_committed
        )
        if force_active and self.merge_sweep_committed:
            return front_object, front_gap, target_lane
        if force_active:
            if designated_gap_bound:
                return front_object, front_gap, self.control_object.lane
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
        target_lane = getattr(navigation, "merge_target_lane", None)
        target_index = tuple(getattr(target_lane, "index", ()) or ())
        if current_road in branch_roads:
            return target_lane
        # The physical connector first lands on the outer lane.  Continue the
        # same route-required lateral move on the mainline until the actor is
        # actually centred on the ego formation's middle lane.
        if (
            len(target_index) >= 3
            and tuple(current_index[:2]) == tuple(target_index[:2])
            and tuple(current_index) != target_index
        ):
            return target_lane
        if len(target_index) >= 3 and len(current_index) >= 3:
            # The accepted cut-in can span a MetaDrive graph seam before the
            # actor has fully converged from outer lane 2 to the platoon's
            # lane slot 1.  Rebind that semantic slot on the current road;
            # otherwise a static upstream lane object makes the merge policy
            # stop steering exactly at the seam.
            road_network = getattr(
                getattr(navigation, "map", None), "road_network", None
            )
            graph = getattr(road_network, "graph", {}) or {}
            current_lanes = list(
                graph.get(current_index[0], {}).get(current_index[1], ()) or ()
            )
            target_slot = int(target_index[2])
            downstream_target = next(
                (
                    lane
                    for lane in current_lanes
                    if int(tuple(getattr(lane, "index", ()) or (0, 0, -1))[2])
                    == target_slot
                ),
                None,
            )
            if downstream_target is not None and int(current_index[2]) != target_slot:
                navigation.merge_target_lane = downstream_target
                return downstream_target
        return None

    def _rear_ttc_s(self, rear_object, rear_gap: float) -> float:
        if rear_object is None or rear_gap == inf:
            return inf
        ego_speed = self._vehicle_speed_mps(self.control_object)
        rear_speed = self._vehicle_speed_mps(rear_object)
        closing_speed = rear_speed - ego_speed
        if closing_speed <= 1e-6:
            return inf
        return float(rear_gap) / float(closing_speed)

    def desired_gap(self, ego_vehicle, front_obj, projected: bool = True) -> float:
        """Return an S6 merge-following gap in metres with consistent units.

        MetaDrive 0.4.3's base implementation multiplies ``speed_km_h`` by a
        time headway and consequently asks a 24 km/h merge actor for roughly
        40 m.  That makes the actor stop at the conflict point after entering
        a deliberately sampled 6--10 m gap.  Use the same IDM shape with
        m/s quantities, a 7 m jam distance and 0.8 s merge headway; the
        planner's unchanged 5 m dense background-vehicle gate remains the
        final safety authority.
        """
        del projected
        ego_speed = self._vehicle_speed_mps(ego_vehicle)
        front_speed = self._vehicle_speed_mps(front_obj)
        closing_speed = max(ego_speed - front_speed, 0.0)
        braking_scale = sqrt(max(self.ACC_FACTOR * -self.DEACC_FACTOR, 1.0e-6))
        target_gap_id = str(
            getattr(ego_vehicle, "scenario_target_gap_id", "")
        )
        if self.merge_sweep_committed and not self.merge_completed:
            jam_distance_m = 5.0
            time_headway_s = 0.5
        elif not self.merge_completed:
            jam_distance_m = 7.0
            time_headway_s = 0.8
        else:
            jam_distance_m = 5.0 if target_gap_id == "agent0-agent1" else 7.0
            time_headway_s = 0.5 if target_gap_id == "agent0-agent1" else 0.8
        return float(
            jam_distance_m
            + time_headway_s * ego_speed
            + ego_speed * closing_speed / (2.0 * braking_scale)
        )

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
        self.merge_completed = tuple(current_index) == tuple(target_index)
        if self.merge_completed:
            # The cut-in actor is not a platoon member, but it remains a
            # normal traffic participant after entering the mainline.  Keep
            # the sampled 18--27 km/h cruise speed; the former 10 km/h value
            # for the second gap violated the scenario range and caused the
            # designated rear ego to close into the unchanged 5 m safety
            # envelope before its lane-change response could complete.
            post_merge_speed_kmh = min(float(self.merge_cruise_speed_kmh), 20.0)
            self.NORMAL_SPEED = post_merge_speed_kmh
            self.target_speed = post_merge_speed_kmh

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
