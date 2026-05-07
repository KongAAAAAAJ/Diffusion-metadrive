"""Scenario orchestrator for local traffic control during episode execution.

This module is the canonical location for ScenarioOrchestrator and the
TriggerEvaluator abstraction. The old path
(metadrive.exp_dataset.scenario_orchestrator) is a backward-compatibility shim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Tuple

from scenarios.definitions import ScenarioDefinition
from metadrive.policy.idm_policy import FrontBackObjects, IDMPolicy

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Trigger evaluator protocol + built-in implementations
# ---------------------------------------------------------------------------

class TriggerEvaluator:
    """Protocol: returns True when the scenario trigger condition is met.

    Subclass or replace to customise when a scenario fires (e.g. platoon lead
    vehicle vs formation centroid vs any agent).
    """

    def is_triggered(self, env, agent_id: str, orchestrator: "ScenarioOrchestrator") -> bool:
        raise NotImplementedError


class SingleVehicleTrigger(TriggerEvaluator):
    """Default trigger: fires when the named agent enters the block/longitudinal window."""

    def is_triggered(self, env, agent_id: str, orchestrator: "ScenarioOrchestrator") -> bool:
        ego_vehicle = (getattr(env, "agents", {}) or {}).get(agent_id)
        if ego_vehicle is None:
            return False
        return orchestrator._is_in_trigger_window(ego_vehicle)


class LeadVehicleTrigger(TriggerEvaluator):
    """Platoon trigger: always evaluates against a fixed lead agent, regardless
    of which agent_id is passed to before_step().
    """

    def __init__(self, lead_agent_id: str = "agent0") -> None:
        self.lead_agent_id = lead_agent_id

    def is_triggered(self, env, agent_id: str, orchestrator: "ScenarioOrchestrator") -> bool:
        lead_vehicle = (getattr(env, "agents", {}) or {}).get(self.lead_agent_id)
        if lead_vehicle is None:
            return False
        return orchestrator._is_in_trigger_window(lead_vehicle)


# ---------------------------------------------------------------------------
# Episode tracking dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScenarioEpisodeSummary:
    scenario_id: str
    scenario_triggered: bool = False
    scenario_realized: bool = False
    trigger_step: int | None = None
    realized_step: int | None = None
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ScenarioOrchestrator
# ---------------------------------------------------------------------------

class ScenarioOrchestrator:
    def __init__(
        self,
        definition: ScenarioDefinition,
        local_route: str,
        trigger_evaluator: TriggerEvaluator | None = None,
    ) -> None:
        self.definition = definition
        self.local_route = local_route
        self.trigger_spec = definition.get_trigger_spec(local_route)
        self.summary = ScenarioEpisodeSummary(scenario_id=definition.scenario_id)
        self._trigger_evaluator: TriggerEvaluator = (
            trigger_evaluator if trigger_evaluator is not None else SingleVehicleTrigger()
        )
        self._road_to_block_id: Dict[Tuple[str, str], str] = {}
        self._speed_profiles: Dict[str, Dict[str, float]] = {}
        self._lead_vehicle_name: str | None = None

    def reset(self, env, agent_id: str) -> None:
        self.summary = ScenarioEpisodeSummary(scenario_id=self.definition.scenario_id)
        self._road_to_block_id = self._build_road_to_block_id(env)
        self._speed_profiles = {}
        self._lead_vehicle_name = None

    def before_step(self, env, agent_id: str, step_count: int) -> None:
        self._apply_speed_profiles(env)
        if self.summary.scenario_triggered:
            return
        if not self._trigger_evaluator.is_triggered(env, agent_id, self):
            return
        self.summary.scenario_triggered = True
        self.summary.trigger_step = int(step_count)
        ego_vehicle = (getattr(env, "agents", {}) or {}).get(agent_id)
        if ego_vehicle is None:
            return
        self._execute_recipe(env, ego_vehicle, step_count)

    def get_episode_summary(self) -> Dict[str, object]:
        return {
            "scenario_id": self.summary.scenario_id,
            "scenario_triggered": bool(self.summary.scenario_triggered),
            "scenario_realized": bool(self.summary.scenario_realized),
            "scenario_trigger_step": self.summary.trigger_step,
            "scenario_realized_step": self.summary.realized_step,
            "scenario_notes": list(self.summary.notes),
        }

    def _execute_recipe(self, env, ego_vehicle, step_count: int) -> None:
        if not self.definition.traffic_recipes:
            self._mark_realized(step_count, "no-op")
            return
        realized = False
        for recipe in self.definition.traffic_recipes:
            handler = getattr(self, f"_handle_{recipe.operation}", None)
            if handler is None:
                self.summary.notes.append(f"unsupported_recipe:{recipe.operation}")
                continue
            if handler(env, ego_vehicle, recipe.params, step_count):
                realized = True
        if not realized:
            self.summary.notes.append("recipe_not_realized")

    def _handle_ensure_lead_vehicle(self, env, ego_vehicle, params: Dict[str, object], step_count: int) -> bool:
        front_vehicle = self._find_front_vehicle(ego_vehicle)
        if front_vehicle is not None:
            setattr(front_vehicle, "scenario_managed_vehicle", True)
            self._lead_vehicle_name = getattr(front_vehicle, "name", None)
            self._mark_realized(step_count, "lead_present")
        else:
            distance_m = float(params.get("distance_m", 20.0))
            target_speed_kmh = float(params.get("target_speed_kmh", getattr(ego_vehicle, "speed_km_h", 20.0)))
            spawned = self._spawn_on_reference(
                env,
                ego_vehicle,
                reference_kind="ego_lane",
                spawn_longitude_offset=distance_m,
                target_speed_kmh=target_speed_kmh,
            )
            if spawned is None:
                self.summary.notes.append("lead_spawn_failed")
                return False
            self._lead_vehicle_name = getattr(spawned, "name", None)
            self._mark_realized(step_count, "lead_spawned")
        if self._lead_vehicle_name is not None:
            target_speed_kmh = float(params.get("target_speed_kmh", getattr(ego_vehicle, "speed_km_h", 20.0)))
            self._speed_profiles[self._lead_vehicle_name] = {
                "remaining_steps": float("inf"),
                "target_speed_kmh": target_speed_kmh,
            }
        return True

    def _handle_hard_brake_lead(self, env, ego_vehicle, params: Dict[str, object], step_count: int) -> bool:
        front_vehicle, front_distance = self._find_front_vehicle_with_distance(ego_vehicle)
        min_distance = float(params.get("front_distance_min_m", 10.0))
        max_distance = float(params.get("front_distance_max_m", float(params.get("lead_distance_m", 20.0)) + 10.0))
        front_vehicle_in_range = (
            front_vehicle is not None and front_distance is not None and min_distance <= front_distance <= max_distance
        )
        if not front_vehicle_in_range:
            if front_vehicle is not None:
                self.summary.notes.append("lead_out_of_range")
            spawned = self._spawn_lead_vehicle(env, ego_vehicle, params)
            if spawned is None:
                self.summary.notes.append("lead_brake_failed")
                return False
            self._lead_vehicle_name = getattr(spawned, "name", None)
            self.summary.notes.append("lead_spawned")
        elif not self._has_controllable_policy(env, front_vehicle):
            self.summary.notes.append("lead_policy_missing")
            spawned = self._spawn_lead_vehicle(env, ego_vehicle, params)
            if spawned is None:
                self.summary.notes.append("lead_brake_failed")
                return False
            self._lead_vehicle_name = getattr(spawned, "name", None)
            self.summary.notes.append("lead_spawned")
        else:
            setattr(front_vehicle, "scenario_managed_vehicle", True)
            self._lead_vehicle_name = getattr(front_vehicle, "name", None)
            self.summary.notes.append("lead_present")

        if self._lead_vehicle_name is None:
            self.summary.notes.append("lead_brake_failed")
            return False
        self._speed_profiles[self._lead_vehicle_name] = {
            "remaining_steps": float(params.get("brake_duration_steps", 30)),
            "target_speed_kmh": float(params.get("brake_target_speed_kmh", 3.0)),
        }
        self._mark_realized(step_count, "lead_brake_profile")
        return True

    def _handle_inject_background_vehicle(self, env, ego_vehicle, params: Dict[str, object], step_count: int) -> bool:
        reference_kind = str(params.get("reference_kind", "ego_lane"))
        spawned = self._spawn_on_reference(
            env,
            ego_vehicle,
            reference_kind=reference_kind,
            block_id=params.get("block_id"),
            socket_index=params.get("socket_index"),
            lane_index=int(params.get("lane_index", 0)),
            spawn_longitude=float(params.get("spawn_longitude", 15.0)),
            spawn_longitude_offset=float(params.get("spawn_longitude_offset", 0.0)),
            target_speed_kmh=float(params.get("target_speed_kmh", getattr(ego_vehicle, "speed_km_h", 20.0))),
        )
        if spawned is None:
            self.summary.notes.append(f"inject_failed:{reference_kind}")
            return False
        self._mark_realized(step_count, f"injected:{reference_kind}")
        return True

    def _apply_speed_profiles(self, env) -> None:
        if not self._speed_profiles:
            return
        finished = []
        for vehicle_name, profile in list(self._speed_profiles.items()):
            vehicle = self._find_traffic_vehicle(env, vehicle_name)
            if vehicle is None:
                finished.append(vehicle_name)
                continue
            setattr(vehicle, "scenario_managed_vehicle", True)
            target_speed_kmh = float(profile["target_speed_kmh"])
            self._set_vehicle_target_speed(env, vehicle, target_speed_kmh)
            self._force_vehicle_speed(vehicle, target_speed_kmh)
            setattr(vehicle, "scenario_warning_marker", "!")
            profile["remaining_steps"] = float(profile["remaining_steps"]) - 1.0
            if profile["remaining_steps"] <= 0:
                finished.append(vehicle_name)
        for vehicle_name in finished:
            vehicle = self._find_traffic_vehicle(env, vehicle_name)
            if vehicle is not None and hasattr(vehicle, "scenario_warning_marker"):
                delattr(vehicle, "scenario_warning_marker")
            self._speed_profiles.pop(vehicle_name, None)

    def _find_front_vehicle(self, ego_vehicle):
        front_vehicle, _ = self._find_front_vehicle_with_distance(ego_vehicle)
        return front_vehicle

    def _find_front_vehicle_with_distance(self, ego_vehicle):
        ref_lane = _select_reference_lane(ego_vehicle)
        if ref_lane is None or not hasattr(ego_vehicle, "lidar"):
            return None, None
        current_ref_lanes = getattr(getattr(ego_vehicle, "navigation", None), "current_ref_lanes", None)
        all_objects = ego_vehicle.lidar.get_surrounding_objects(ego_vehicle)
        surrounding_objects = FrontBackObjects.get_find_front_back_objs(
            all_objects,
            ref_lane,
            ego_vehicle.position,
            max_distance=IDMPolicy.MAX_LONG_DIST,
            ref_lanes=current_ref_lanes if current_ref_lanes and ref_lane in current_ref_lanes else None,
        )
        front_vehicle = surrounding_objects.front_object()
        if front_vehicle is None:
            return None, None
        try:
            ego_long = ref_lane.local_coordinates(ego_vehicle.position)[0]
            front_long = ref_lane.local_coordinates(front_vehicle.position)[0]
            return front_vehicle, max(float(front_long - ego_long), 0.0)
        except Exception:
            try:
                delta = front_vehicle.position - ego_vehicle.position
                return front_vehicle, float((delta[0] ** 2 + delta[1] ** 2) ** 0.5)
            except Exception:
                return front_vehicle, None

    def _spawn_on_reference(
        self,
        env,
        ego_vehicle,
        reference_kind: str,
        *,
        block_id=None,
        socket_index=None,
        lane_index: int = 0,
        spawn_longitude: float = 10.0,
        spawn_longitude_offset: float = 0.0,
        target_speed_kmh: float = 20.0,
    ):
        lane_tuple = self._resolve_lane_index(
            env,
            ego_vehicle,
            reference_kind=reference_kind,
            block_id=block_id,
            socket_index=socket_index,
            lane_index=lane_index,
        )
        if lane_tuple is None:
            return None
        current_map = getattr(getattr(env, "engine", None), "current_map", None)
        if current_map is None:
            return None
        lane = current_map.road_network.get_lane(lane_tuple)
        ego_long = lane.local_coordinates(ego_vehicle.position)[0] if reference_kind == "ego_lane" else 0.0
        spawn_long = min(max(spawn_longitude + spawn_longitude_offset + ego_long, 2.0), max(lane.length - 2.0, 2.0))
        spawn_position = lane.position(float(spawn_long), 0.0)
        min_clearance = float(getattr(env, "config", {}).get("scenario_spawn_min_agent_clearance_m", 10.0))
        if not self._spawn_position_clear_of_agents(env, spawn_position, min_clearance):
            self.summary.notes.append("spawn_blocked:agent_clearance")
            return None
        traffic_manager = getattr(getattr(env, "engine", None), "traffic_manager", None)
        if traffic_manager is None or not hasattr(traffic_manager, "_spawn_traffic_vehicle_if_safe"):
            return None
        vehicle_type = traffic_manager.random_vehicle_type()
        spawned = traffic_manager._spawn_traffic_vehicle_if_safe(
            vehicle_type,
            {"spawn_lane_index": lane_tuple, "spawn_longitude": float(spawn_long)},
        )
        if spawned is not None:
            setattr(spawned, "scenario_managed_vehicle", True)
            self._set_vehicle_target_speed(env, spawned, target_speed_kmh)
        return spawned

    @staticmethod
    def _spawn_position_clear_of_agents(env, spawn_position, min_clearance_m: float) -> bool:
        agents = getattr(env, "agents", {}) or {}
        try:
            spawn_xy = (float(spawn_position[0]), float(spawn_position[1]))
        except Exception:
            return False
        min_clearance_m = max(float(min_clearance_m), 0.0)
        for agent in agents.values():
            try:
                agent_pos = getattr(agent, "position", None)
                if agent_pos is None:
                    continue
                dx = float(spawn_xy[0] - float(agent_pos[0]))
                dy = float(spawn_xy[1] - float(agent_pos[1]))
                if (dx * dx + dy * dy) ** 0.5 < min_clearance_m:
                    return False
            except Exception:
                continue
        return True

    def _spawn_lead_vehicle(self, env, ego_vehicle, params: Dict[str, object]):
        return self._spawn_on_reference(
            env,
            ego_vehicle,
            reference_kind="ego_lane",
            spawn_longitude_offset=float(params.get("lead_distance_m", 20.0)),
            target_speed_kmh=float(params.get("lead_target_speed_kmh", 20.0)),
        )

    def _resolve_lane_index(self, env, ego_vehicle, *, reference_kind: str, block_id=None, socket_index=None, lane_index: int = 0):
        if reference_kind == "ego_lane":
            lane_idx = getattr(ego_vehicle, "lane_index", None)
            return tuple(lane_idx) if lane_idx is not None else None
        current_map = getattr(getattr(env, "engine", None), "current_map", None)
        if current_map is None or block_id is None:
            return None
        block = self._get_block_by_graph_id(current_map, str(block_id))
        if block is None:
            return None
        if reference_kind == "block_route_road":
            road = self._get_first_positive_route_road(block)
            if road is None:
                return None
            lanes = current_map.road_network.graph[road.start_node][road.end_node]
            resolved_lane = min(max(int(lane_index), 0), len(lanes) - 1)
            return (road.start_node, road.end_node, resolved_lane)
        if reference_kind == "block_socket_road":
            socket = block.get_socket(int(socket_index or 0))
            road = getattr(socket, "positive_road", None)
            if road is None:
                return None
            lanes = current_map.road_network.graph[road.start_node][road.end_node]
            resolved_lane = min(max(int(lane_index), 0), len(lanes) - 1)
            return (road.start_node, road.end_node, resolved_lane)
        return None

    def _find_traffic_vehicle(self, env, vehicle_name: str):
        traffic_manager = getattr(getattr(env, "engine", None), "traffic_manager", None)
        vehicles = getattr(traffic_manager, "_traffic_vehicles", []) or []
        for vehicle in vehicles:
            if getattr(vehicle, "name", None) == vehicle_name:
                return vehicle
        return None

    @staticmethod
    def _has_controllable_policy(env, vehicle) -> bool:
        policy = getattr(getattr(env, "engine", None), "get_policy", lambda *_: None)(getattr(vehicle, "name", None))
        return policy is not None

    @staticmethod
    def _force_vehicle_speed(vehicle, target_speed_kmh: float) -> None:
        if hasattr(vehicle, "set_throttle_brake"):
            vehicle.set_throttle_brake(-1.0 if target_speed_kmh <= 3.0 else -0.6)
        if hasattr(vehicle, "set_velocity"):
            target_speed_mps = max(float(target_speed_kmh) / 3.6, 0.0)
            if target_speed_kmh <= 3.0:
                vehicle.set_velocity([0.0, 0.0], in_local_frame=True)
            else:
                vehicle.set_velocity([target_speed_mps, 0.0], in_local_frame=True)

    def _set_vehicle_target_speed(self, env, vehicle, speed_kmh: float) -> None:
        policy = getattr(getattr(env, "engine", None), "get_policy", lambda *_: None)(vehicle.name)
        if policy is None:
            return
        if hasattr(policy, "NORMAL_SPEED"):
            policy.NORMAL_SPEED = float(speed_kmh)
        if hasattr(policy, "target_speed"):
            policy.target_speed = float(speed_kmh)

    def _is_in_trigger_window(self, ego_vehicle) -> bool:
        current_block_id, longitudinal = self._resolve_ego_position(ego_vehicle)
        if current_block_id != self.trigger_spec.block_id or longitudinal is None:
            return False
        return self.trigger_spec.longitudinal_min <= longitudinal <= self.trigger_spec.longitudinal_max

    def _resolve_ego_position(self, ego_vehicle):
        lane = _select_reference_lane(ego_vehicle)
        if lane is None or not hasattr(lane, "index"):
            return None, None
        road_key = tuple(lane.index[:2])
        block_id = self._road_to_block_id.get(road_key)
        longitudinal = lane.local_coordinates(ego_vehicle.position)[0]
        return block_id, float(longitudinal)

    def _build_road_to_block_id(self, env) -> Dict[Tuple[str, str], str]:
        current_map = getattr(getattr(env, "engine", None), "current_map", None)
        if current_map is None:
            return {}
        mapping: Dict[Tuple[str, str], str] = {}
        for block in getattr(current_map, "blocks", []) or []:
            block_id = getattr(block, "graph_block_id", None)
            if block_id is None:
                continue
            for roads in getattr(getattr(block, "block_network", None), "get_positive_lanes", lambda: [])() or []:
                for lane in roads:
                    mapping[tuple(lane.index[:2])] = str(block_id)
            sockets = getattr(block, "get_socket_list", lambda: [])() or []
            for socket in sockets:
                for road in (getattr(socket, "positive_road", None), getattr(socket, "negative_road", None)):
                    if road is not None:
                        mapping[(road.start_node, road.end_node)] = str(block_id)
            for road in getattr(block, "get_respawn_roads", lambda: [])() or []:
                mapping[(road.start_node, road.end_node)] = str(block_id)
        return mapping

    @staticmethod
    def _get_block_by_graph_id(current_map, block_id):
        for block in getattr(current_map, "blocks", []) or []:
            if getattr(block, "graph_block_id", None) == block_id:
                return block
        return None

    @staticmethod
    def _get_first_positive_route_road(block):
        roads = [
            road for road in getattr(block, "get_respawn_roads", lambda: [])() or []
            if not (hasattr(road, "is_negative_road") and road.is_negative_road())
        ]
        if roads:
            return sorted(roads, key=lambda road: (road.start_node, road.end_node))[0]
        sockets = getattr(block, "get_socket_list", lambda: [])() or []
        positive = [socket.positive_road for socket in sockets if getattr(socket, "positive_road", None) is not None]
        if not positive:
            return None
        return sorted(positive, key=lambda road: (road.start_node, road.end_node))[0]

    def _mark_realized(self, step_count: int, note: str) -> None:
        self.summary.scenario_realized = True
        if self.summary.realized_step is None:
            self.summary.realized_step = int(step_count)
        self.summary.notes.append(note)


def _select_reference_lane(vehicle):
    lane = getattr(vehicle, "lane", None)
    if lane is not None:
        return lane
    navigation = getattr(vehicle, "navigation", None)
    current_ref_lanes = getattr(navigation, "current_ref_lanes", None)
    if current_ref_lanes:
        return current_ref_lanes[0]
    lane_index = getattr(vehicle, "lane_index", None)
    current_map = getattr(getattr(vehicle, "engine", None), "current_map", None)
    road_network = getattr(current_map, "road_network", None)
    if lane_index is not None and road_network is not None:
        try:
            return road_network.get_lane(lane_index)
        except Exception:
            return None
    return None
