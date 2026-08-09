"""Scenario orchestrator for local traffic control during episode execution.

This module is the canonical location for ScenarioOrchestrator and the
TriggerEvaluator abstraction. The old path
(metadrive.exp_dataset.scenario_orchestrator) is a backward-compatibility shim.
"""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from scenarios.definitions import ScenarioDefinition, TriggerSpec
from metadrive.policy.idm_policy import IDMPolicy

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class _RouteRoadRef:
    start_node: str
    end_node: str


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
        self._completed_recipe_keys: set[str] = set()
        self._spawned_adjacent_vehicle_keys: set[str] = set()
        self._scenario_random_seed: int | None = None
        self._scenario_rng = None
        self._resolved_recipe_parameters: Dict[str, Dict[str, float]] = {}

    def reset(self, env, agent_id: str) -> None:
        self.summary = ScenarioEpisodeSummary(scenario_id=self.definition.scenario_id)
        self._road_to_block_id = self._build_road_to_block_id(env)
        self._speed_profiles = {}
        self._lead_vehicle_name = None
        self._completed_recipe_keys = set()
        self._spawned_adjacent_vehicle_keys = set()
        self._scenario_random_seed = self._derive_scenario_random_seed(env)
        self._scenario_rng = np.random.RandomState(self._scenario_random_seed)
        self._resolved_recipe_parameters = {}

    def before_step(self, env, agent_id: str, step_count: int) -> None:
        self._apply_speed_profiles(env)
        ego_vehicle = (getattr(env, "agents", {}) or {}).get(agent_id)
        if ego_vehicle is None:
            return
        if self.definition.scenario_id != "S5_hard_brake_lead":
            triggered_now = self._trigger_evaluator.is_triggered(env, agent_id, self)
            if triggered_now and not self.summary.scenario_triggered:
                self.summary.scenario_triggered = True
                self.summary.trigger_step = int(step_count)
        self._execute_recipe(env, ego_vehicle, step_count)

    def get_episode_summary(self) -> Dict[str, object]:
        recipe_count = len(self.definition.traffic_recipes)
        completed_recipe_count = len(self._completed_recipe_keys)
        return {
            "scenario_id": self.summary.scenario_id,
            "scenario_triggered": bool(self.summary.scenario_triggered),
            "scenario_realized": bool(self.summary.scenario_realized),
            "scenario_trigger_step": self.summary.trigger_step,
            "scenario_realized_step": self.summary.realized_step,
            "scenario_recipe_count": int(recipe_count),
            "scenario_completed_recipe_count": int(completed_recipe_count),
            "scenario_recipes_complete": bool(
                completed_recipe_count == recipe_count
            ),
            "scenario_notes": list(self.summary.notes),
            "scenario_random_seed": self._scenario_random_seed,
            "resolved_recipe_parameters": {
                key: dict(value)
                for key, value in self._resolved_recipe_parameters.items()
            },
        }

    def _execute_recipe(self, env, ego_vehicle, step_count: int) -> None:
        if not self.definition.traffic_recipes:
            self._mark_realized(step_count, "no-op")
            return
        realized = False
        considered = False
        for recipe_index, recipe in enumerate(self.definition.traffic_recipes):
            recipe_key = f"{recipe_index}:{recipe.operation}"
            if recipe_key in self._completed_recipe_keys:
                continue
            if not self._recipe_triggered(env, ego_vehicle, recipe.params, step_count):
                continue
            considered = True
            if not self.summary.scenario_triggered and not self._is_s5_startup_support_recipe(recipe):
                self.summary.scenario_triggered = True
                self.summary.trigger_step = int(step_count)
            handler = getattr(self, f"_handle_{recipe.operation}", None)
            if handler is None:
                self.summary.notes.append(f"unsupported_recipe:{recipe.operation}")
                self._completed_recipe_keys.add(recipe_key)
                continue
            if handler(env, ego_vehicle, recipe.params, step_count):
                realized = True
                self._completed_recipe_keys.add(recipe_key)
        if considered and not realized:
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
        params = self._resolve_hard_brake_params(env, params)
        self._resolved_recipe_parameters["hard_brake_lead"] = {
            "lead_bumper_gap_m": float(params["lead_bumper_gap_m"]),
            "lead_target_speed_kmh": float(params["lead_target_speed_kmh"]),
            "brake_target_speed_kmh": float(params["brake_target_speed_kmh"]),
            "brake_deceleration_mps2": float(params["brake_deceleration_mps2"]),
        }
        front_vehicle, front_distance = self._find_front_vehicle_with_distance(ego_vehicle)
        gap_range = params["lead_bumper_gap_range_m"]
        min_distance = float(gap_range[0])
        max_distance = float(gap_range[1])
        front_bumper_gap = (
            self._bumper_gap_m(ego_vehicle, front_vehicle, front_distance)
            if front_vehicle is not None and front_distance is not None
            else None
        )
        front_vehicle_in_range = (
            front_vehicle is not None
            and front_bumper_gap is not None
            and min_distance <= front_bumper_gap <= max_distance
        )
        if not front_vehicle_in_range:
            if front_vehicle is not None:
                self.summary.notes.append("lead_out_of_range")
            spawned = self._spawn_lead_vehicle(env, ego_vehicle, params)
            if spawned is None:
                self.summary.notes.append("lead_brake_failed")
                return False
            self._mark_hard_brake_lead_vehicle(spawned)
            self._lead_vehicle_name = getattr(spawned, "name", None)
            self.summary.notes.append("lead_spawned")
        elif not self._has_controllable_policy(env, front_vehicle):
            self.summary.notes.append("lead_policy_missing")
            spawned = self._spawn_lead_vehicle(env, ego_vehicle, params)
            if spawned is None:
                self.summary.notes.append("lead_brake_failed")
                return False
            self._mark_hard_brake_lead_vehicle(spawned)
            self._lead_vehicle_name = getattr(spawned, "name", None)
            self.summary.notes.append("lead_spawned")
        else:
            self._mark_hard_brake_lead_vehicle(front_vehicle)
            self._lead_vehicle_name = getattr(front_vehicle, "name", None)
            self.summary.notes.append("lead_present")

        if self._lead_vehicle_name is None:
            self.summary.notes.append("lead_brake_failed")
            return False
        lead_vehicle = self._find_traffic_vehicle(env, self._lead_vehicle_name)
        if lead_vehicle is None:
            self.summary.notes.append("lead_brake_failed")
            return False
        target_speed_kmh = float(params["brake_target_speed_kmh"])
        deceleration_mps2 = float(params["brake_deceleration_mps2"])
        self._speed_profiles[self._lead_vehicle_name] = {
            "remaining_steps": float("inf"),
            "target_speed_kmh": target_speed_kmh,
            "max_deceleration_mps2": deceleration_mps2,
        }
        setattr(
            lead_vehicle,
            "scenario_brake_initial_speed_kmh",
            float(getattr(lead_vehicle, "speed_km_h", 0.0) or 0.0),
        )
        setattr(lead_vehicle, "scenario_brake_target_speed_kmh", target_speed_kmh)
        setattr(lead_vehicle, "scenario_brake_deceleration_mps2", deceleration_mps2)
        setattr(lead_vehicle, "scenario_brake_trigger_step", int(step_count))
        realized_gap = self._bumper_gap_m(
            ego_vehicle,
            lead_vehicle,
            self._longitudinal_distance_m(ego_vehicle, lead_vehicle),
        )
        if realized_gap is not None:
            setattr(lead_vehicle, "scenario_brake_trigger_bumper_gap_m", realized_gap)
        self._mark_realized(step_count, "lead_brake_profile")
        return True

    def _mark_hard_brake_lead_vehicle(self, vehicle) -> None:
        setattr(vehicle, "scenario_managed_vehicle", True)
        setattr(vehicle, "scenario_warning_marker", "!")
        setattr(vehicle, "scenario_id", self.definition.scenario_id)
        setattr(vehicle, "scenario_role", "hard_brake_lead")

    def _handle_inject_background_vehicle(self, env, ego_vehicle, params: Dict[str, object], step_count: int) -> bool:
        params = self._resolve_background_vehicle_params(env, params)
        merge_alignment = None
        if (
            str(params.get("policy", "")) == "idm_merge"
            and "merge_arrival_offset_s" in params
        ):
            merge_alignment = self._resolve_s6_merge_alignment(
                env,
                ego_vehicle,
                params,
            )
            if merge_alignment is None:
                self.summary.notes.append("s6_merge_alignment_failed")
                return False
            params["spawn_longitude"] = merge_alignment["spawn_longitude_m"]
        reference_kind = str(params.get("reference_kind", "ego_lane"))
        policy_class, policy_kwargs, vehicle_config_overrides = self._resolve_injected_background_policy(params)
        lane_index = params.get("lane_index", params.get("lane_id", 0))
        spawned = self._spawn_on_reference(
            env,
            ego_vehicle,
            reference_kind=reference_kind,
            block_id=params.get("block_id"),
            socket_index=params.get("socket_index"),
            internal_road_index=params.get("internal_road_index"),
            lane_index=int(lane_index),
            spawn_longitude=float(params.get("spawn_longitude", 15.0)),
            spawn_longitude_offset=float(params.get("spawn_longitude_offset", 0.0)),
            target_speed_kmh=float(params.get("target_speed_kmh", getattr(ego_vehicle, "speed_km_h", 20.0))),
            min_clearance_m=0.0 if merge_alignment is not None else None,
            policy_class=policy_class,
            policy_kwargs=policy_kwargs,
            vehicle_config_overrides=vehicle_config_overrides,
            vehicle_type=self._scenario_vehicle_type(),
        )
        if spawned is None:
            self.summary.notes.append(f"inject_failed:{reference_kind}")
            return False
        setattr(spawned, "scenario_warning_marker", "!")
        setattr(
            spawned,
            "scenario_vehicle_role",
            (
                str(params.get("scenario_vehicle_role"))
                if params.get("scenario_vehicle_role") is not None
                else (
                    "s6_merge_vehicle"
                    if merge_alignment is not None
                    else "injected_background"
                )
            ),
        )
        if merge_alignment is not None:
            for key, value in merge_alignment.items():
                setattr(spawned, f"scenario_{key}", value)
        self._mark_realized(step_count, f"injected:{reference_kind}")
        return True

    @staticmethod
    def _resolve_injected_background_policy(params: Dict[str, object]):
        policy_name = params.get("policy")
        if policy_name is None:
            return None, None, None
        cruise_speed = float(params.get("target_speed_kmh", 24.0))
        if str(policy_name) == "idm_merge":
            from envs.diffusion_envs.idm_merge_policy import IDMMergePolicy, StartEdgeNodeNavigation

            policy_kwargs = {
                "merge_front_gap_m": float(params.get("merge_front_gap_m", 25.0)),
                "merge_rear_gap_m": float(params.get("merge_rear_gap_m", 15.0)),
                "merge_creep_speed_kmh": float(params.get("merge_creep_speed_kmh", 5.0)),
                "merge_cruise_speed_kmh": cruise_speed,
            }
            if "merge_activation_step" in params:
                policy_kwargs["merge_activation_step"] = int(
                    params["merge_activation_step"]
                )
            return (
                IDMMergePolicy,
                policy_kwargs,
                {"navigation_module": StartEdgeNodeNavigation},
            )
        if str(policy_name) == "forced_cut_in":
            from envs.diffusion_envs.forced_cut_in_policy import ForcedCutInPolicy

            return (
                ForcedCutInPolicy,
                {
                    "activation_step": int(params.get("activation_step", 80)),
                    "target_lane_offset": int(params.get("target_lane_offset", 1)),
                    "front_gap_m": float(params.get("front_gap_m", 5.0)),
                    "rear_gap_m": float(params.get("rear_gap_m", 5.0)),
                    "cruise_speed_kmh": cruise_speed,
                },
                None,
            )
        raise ValueError(f"Unknown injected background policy: {policy_name!r}")

    def _handle_inject_adjacent_lane_vehicles(self, env, ego_vehicle, params: Dict[str, object], step_count: int) -> bool:
        vehicles = params.get("vehicles", ())
        if not isinstance(vehicles, (list, tuple)):
            self.summary.notes.append("adjacent_config_invalid")
            return False
        realized = False
        for vehicle_index, vehicle_params in enumerate(vehicles):
            if not isinstance(vehicle_params, dict):
                self.summary.notes.append(f"adjacent_vehicle_invalid:{vehicle_index}")
                continue
            vehicle_name = str(vehicle_params.get("name", f"vehicle_{vehicle_index}"))
            spawn_key = f"{self.local_route}:{vehicle_name}"
            if spawn_key in self._spawned_adjacent_vehicle_keys:
                continue
            vehicle_params = self._resolve_adjacent_vehicle_params(env, vehicle_params)
            lane_side = str(vehicle_params.get("lane_side", "left"))
            lane_tuple = self._resolve_adjacent_lane_index(env, ego_vehicle, lane_side=lane_side)
            if lane_tuple is None:
                self.summary.notes.append(f"adjacent_lane_missing:{vehicle_name}")
                self._spawned_adjacent_vehicle_keys.add(spawn_key)
                continue
            policy_class, policy_kwargs, vehicle_config_overrides = (
                self._resolve_injected_background_policy(vehicle_params)
            )
            spawned = self._spawn_on_lane_tuple(
                env,
                ego_vehicle,
                lane_tuple=lane_tuple,
                spawn_longitude_offset=float(vehicle_params.get("spawn_longitude_offset_m", 12.0)),
                target_speed_kmh=float(vehicle_params.get("target_speed_kmh", getattr(ego_vehicle, "speed_km_h", 20.0))),
                min_clearance_m=vehicle_params.get("min_agent_clearance_m"),
                clearance_scope=str(vehicle_params.get("clearance_scope", params.get("clearance_scope", "all_agents"))),
                policy_class=policy_class,
                policy_kwargs=policy_kwargs,
                vehicle_config_overrides=vehicle_config_overrides,
                vehicle_type=self._scenario_vehicle_type(),
            )
            if spawned is None:
                self.summary.notes.append(f"adjacent_spawn_failed:{vehicle_name}")
                continue
            if vehicle_params.get("scenario_vehicle_role") is not None:
                setattr(
                    spawned,
                    "scenario_vehicle_role",
                    str(vehicle_params["scenario_vehicle_role"]),
                )
            elif self.definition.scenario_id == "S5_hard_brake_lead":
                setattr(
                    spawned,
                    "scenario_vehicle_role",
                    f"s5_adjacent_{lane_side}",
                )
            self._spawned_adjacent_vehicle_keys.add(spawn_key)
            spawned_name = getattr(spawned, "name", None)
            if spawned_name is not None:
                self._speed_profiles[spawned_name] = {
                    "remaining_steps": float(vehicle_params.get("speed_profile_duration_steps", float("inf"))),
                    "target_speed_kmh": float(vehicle_params.get("target_speed_kmh", getattr(ego_vehicle, "speed_km_h", 20.0))),
                }
            self._mark_realized(step_count, f"adjacent_spawned:{vehicle_name}")
            realized = True
        return realized

    def _resolve_hard_brake_params(self, env, params: Dict[str, object]) -> Dict[str, object]:
        resolved = dict(params)
        gap_range = params.get("lead_bumper_gap_range_m")
        if not self._is_float_range(gap_range):
            raise ValueError("hard_brake_lead requires lead_bumper_gap_range_m")
        resolved["lead_bumper_gap_m"] = self._sample_float_range(env, gap_range)
        if "lead_target_speed_range_kmh" in params:
            resolved["lead_target_speed_kmh"] = self._sample_float_range(env, params["lead_target_speed_range_kmh"])
        if "brake_target_speed_range_kmh" in params:
            resolved["brake_target_speed_kmh"] = self._sample_float_range(env, params["brake_target_speed_range_kmh"])
        deceleration_range = params.get("brake_deceleration_range_mps2")
        if not self._is_float_range(deceleration_range):
            raise ValueError("hard_brake_lead requires brake_deceleration_range_mps2")
        resolved["brake_deceleration_mps2"] = self._sample_float_range(
            env, deceleration_range
        )
        return resolved

    def _resolve_adjacent_vehicle_params(self, env, params: Dict[str, object]) -> Dict[str, object]:
        resolved = dict(params)
        if "spawn_longitude_offset_range_m" in params:
            resolved["spawn_longitude_offset_m"] = self._sample_float_range(env, params["spawn_longitude_offset_range_m"])
        if "target_speed_range_kmh" in params:
            resolved["target_speed_kmh"] = self._sample_float_range(env, params["target_speed_range_kmh"])
        return resolved

    def _resolve_background_vehicle_params(self, env, params: Dict[str, object]) -> Dict[str, object]:
        resolved = dict(params)
        spawn_longitude = params.get("spawn_longitude")
        if self._is_float_range(spawn_longitude):
            resolved["spawn_longitude"] = self._sample_float_range(env, spawn_longitude)
        return resolved

    def _resolve_s6_merge_alignment(
        self,
        env,
        leader,
        params: Dict[str, object],
    ) -> Dict[str, object] | None:
        if str(params.get("policy")) != "idm_merge":
            self.summary.notes.append("s6_merge_policy_invalid")
            return None
        if "spawn_longitude" in params:
            self.summary.notes.append("s6_absolute_spawn_forbidden")
            return None
        try:
            arrival_offset_s = float(params["merge_arrival_offset_s"])
            merge_speed_mps = float(params["target_speed_kmh"]) / 3.6
        except (KeyError, TypeError, ValueError):
            self.summary.notes.append("s6_merge_contract_invalid")
            return None
        if arrival_offset_s <= 0.0 or merge_speed_mps <= 0.0:
            self.summary.notes.append("s6_merge_contract_invalid")
            return None

        branch_lane_index = self._resolve_lane_index(
            env,
            leader,
            reference_kind=str(params.get("reference_kind", "")),
            block_id=params.get("block_id"),
            socket_index=params.get("socket_index"),
            internal_road_index=params.get("internal_road_index"),
            lane_index=int(params.get("lane_index", params.get("lane_id", 0))),
        )
        current_map = getattr(getattr(env, "engine", None), "current_map", None)
        road_network = getattr(current_map, "road_network", None)
        graph = getattr(road_network, "graph", None)
        leader_lane = getattr(leader, "lane", None)
        leader_lane_index = getattr(leader_lane, "index", None)
        if (
            branch_lane_index is None
            or graph is None
            or leader_lane is None
            or leader_lane_index is None
        ):
            self.summary.notes.append("s6_merge_topology_missing")
            return None

        branch_start, branch_node = tuple(branch_lane_index[:2])
        leader_start, merge_node = tuple(leader_lane_index[:2])
        if leader_start != branch_start:
            self.summary.notes.append("s6_merge_start_mismatch")
            return None
        connector_lanes = graph.get(branch_node, {}).get(merge_node)
        if not connector_lanes:
            self.summary.notes.append("s6_merge_connector_missing")
            return None

        branch_lane = road_network.get_lane(tuple(branch_lane_index))
        connector_lane = connector_lanes[0]
        try:
            leader_longitudinal = float(
                leader_lane.local_coordinates(leader.position)[0]
            )
            leader_remaining_m = float(leader_lane.length) - leader_longitudinal
            leader_speed_mps = float(getattr(leader, "speed_km_h", 0.0)) / 3.6
        except (AttributeError, TypeError, ValueError):
            self.summary.notes.append("s6_merge_geometry_invalid")
            return None
        if leader_remaining_m <= 0.0 or leader_speed_mps <= 0.0:
            self.summary.notes.append("s6_merge_geometry_invalid")
            return None

        leader_ttc_s = leader_remaining_m / leader_speed_mps
        merge_ttc_s = leader_ttc_s + arrival_offset_s
        branch_route_length_m = float(branch_lane.length) + float(
            connector_lane.length
        )
        spawn_longitude_m = branch_route_length_m - merge_speed_mps * merge_ttc_s
        minimum_spawn_m = 2.0
        maximum_spawn_m = float(branch_lane.length) - 2.0
        if not minimum_spawn_m <= spawn_longitude_m <= maximum_spawn_m:
            self.summary.notes.append("s6_merge_spawn_out_of_range")
            return None

        middle = (getattr(env, "agents", {}) or {}).get("agent1")
        middle_lane = getattr(middle, "lane", None)
        middle_lane_index = getattr(middle_lane, "index", None)
        if middle is None or tuple((middle_lane_index or ())[:2]) != (
            leader_start,
            merge_node,
        ):
            self.summary.notes.append("s6_middle_lane_mismatch")
            return None
        try:
            middle_longitudinal = float(
                middle_lane.local_coordinates(middle.position)[0]
            )
            middle_remaining_m = float(middle_lane.length) - middle_longitudinal
            middle_speed_mps = float(getattr(middle, "speed_km_h", 0.0)) / 3.6
        except (AttributeError, TypeError, ValueError):
            self.summary.notes.append("s6_middle_geometry_invalid")
            return None
        if middle_speed_mps <= 0.0:
            self.summary.notes.append("s6_middle_geometry_invalid")
            return None
        middle_ttc_s = middle_remaining_m / middle_speed_mps
        if not leader_ttc_s < merge_ttc_s < middle_ttc_s:
            self.summary.notes.append("s6_merge_not_between_leader_middle")
            return None

        conflict_point = leader_lane.position(float(leader_lane.length), 0.0)
        spawn_position = branch_lane.position(float(spawn_longitude_m), 0.0)
        spawn_heading = float(branch_lane.heading_theta_at(spawn_longitude_m))
        for agent in (getattr(env, "agents", {}) or {}).values():
            if self._oriented_boxes_overlap(
                spawn_position,
                spawn_heading,
                (5.74, 2.3),
                getattr(agent, "position", (0.0, 0.0)),
                float(getattr(agent, "heading_theta", 0.0) or 0.0),
                (
                    self._vehicle_length_m(agent),
                    float(
                        getattr(
                            agent,
                            "WIDTH",
                            getattr(agent, "width", 2.3),
                        )
                        or 2.3
                    ),
                ),
            ):
                self.summary.notes.append("s6_merge_initial_obb_overlap")
                return None
        return {
            "spawn_longitude_m": float(spawn_longitude_m),
            "merge_arrival_offset_s": float(arrival_offset_s),
            "leader_ttc_s": float(leader_ttc_s),
            "merge_ttc_s": float(merge_ttc_s),
            "middle_ttc_s": float(middle_ttc_s),
            "leader_conflict_distance_m": float(leader_remaining_m),
            "merge_conflict_distance_m": float(
                branch_route_length_m - spawn_longitude_m
            ),
            "conflict_point_xy": tuple(
                float(value) for value in conflict_point[:2]
            ),
        }

    @staticmethod
    def _oriented_boxes_overlap(
        first_position,
        first_heading: float,
        first_dimensions,
        second_position,
        second_heading: float,
        second_dimensions,
    ) -> bool:
        axes = (
            (math.cos(first_heading), math.sin(first_heading)),
            (-math.sin(first_heading), math.cos(first_heading)),
            (math.cos(second_heading), math.sin(second_heading)),
            (-math.sin(second_heading), math.cos(second_heading)),
        )
        delta = (
            float(second_position[0]) - float(first_position[0]),
            float(second_position[1]) - float(first_position[1]),
        )
        boxes = (
            (
                0.5 * float(first_dimensions[0]),
                0.5 * float(first_dimensions[1]),
                float(first_heading),
            ),
            (
                0.5 * float(second_dimensions[0]),
                0.5 * float(second_dimensions[1]),
                float(second_heading),
            ),
        )
        for axis_x, axis_y in axes:
            projected_centers = abs(delta[0] * axis_x + delta[1] * axis_y)
            projected_radii = []
            for half_length, half_width, heading in boxes:
                longitudinal = (math.cos(heading), math.sin(heading))
                lateral = (-math.sin(heading), math.cos(heading))
                projected_radii.append(
                    half_length
                    * abs(axis_x * longitudinal[0] + axis_y * longitudinal[1])
                    + half_width
                    * abs(axis_x * lateral[0] + axis_y * lateral[1])
                )
            if projected_centers > projected_radii[0] + projected_radii[1]:
                return False
        return True

    @staticmethod
    def _rng_from_env(env):
        traffic_manager = getattr(getattr(env, "engine", None), "traffic_manager", None)
        rng = getattr(traffic_manager, "np_random", None)
        if rng is not None:
            return rng
        rng = getattr(env, "np_random", None)
        if rng is not None:
            return rng
        return getattr(getattr(env, "engine", None), "np_random", None)

    def _derive_scenario_random_seed(self, env) -> int:
        """Build an episode-local seed independent of traffic-manager RNG use.

        The traffic manager also samples vehicle types, object seeds and policy
        seeds.  Using that shared stream for scenario parameters made S5's
        braking strength depend on unrelated object lifecycle details.  A
        stable digest avoids Python's process-randomized ``hash()`` and makes
        scenario parameters a pure function of the episode contract.
        """

        raw_seed = getattr(env, "current_seed", None)
        if raw_seed is None:
            raw_seed = getattr(getattr(env, "engine", None), "global_random_seed", 0)
        payload = (
            f"{int(raw_seed or 0)}\0{self.definition.scenario_id}"
            f"\0{self.local_route}"
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")

    @staticmethod
    def _is_float_range(value) -> bool:
        return isinstance(value, (tuple, list)) and len(value) == 2

    def _sample_float_range(self, env, value_range) -> float:
        low, high = value_range
        rng = (
            self._scenario_rng
            if self.definition.scenario_id == "S5_hard_brake_lead"
            else self._rng_from_env(env)
        )
        if rng is None:
            rng = self._scenario_rng
        if rng is not None and hasattr(rng, "uniform"):
            return float(rng.uniform(float(low), float(high)))
        return float((float(low) + float(high)) * 0.5)

    def _sample_int_range(self, env, value_range) -> int:
        low, high = value_range
        low_i = int(low)
        high_i = int(high)
        rng = (
            self._scenario_rng
            if self.definition.scenario_id == "S5_hard_brake_lead"
            else self._rng_from_env(env)
        )
        if rng is None:
            rng = self._scenario_rng
        if rng is not None and hasattr(rng, "randint"):
            return int(rng.randint(low_i, high_i + 1))
        if rng is not None and hasattr(rng, "integers"):
            return int(rng.integers(low_i, high_i + 1))
        return int(round((low_i + high_i) * 0.5))

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
            max_deceleration_mps2 = profile.get("max_deceleration_mps2")
            if max_deceleration_mps2 is None:
                self._set_vehicle_target_speed(env, vehicle, target_speed_kmh)
                self._force_vehicle_speed(vehicle, target_speed_kmh)
            else:
                next_speed_kmh = self._bounded_brake_speed_kmh(
                    vehicle,
                    target_speed_kmh=target_speed_kmh,
                    max_deceleration_mps2=float(max_deceleration_mps2),
                    dt_s=self._scenario_step_dt_s(env),
                )
                self._set_vehicle_target_speed(env, vehicle, next_speed_kmh)
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
        if ref_lane is None:
            return None, None
        try:
            ego_long, _ = ref_lane.local_coordinates(ego_vehicle.position)
        except Exception:
            return None, None

        engine = getattr(ego_vehicle, "engine", None)
        traffic_manager = getattr(engine, "traffic_manager", None)
        raw_vehicles = getattr(traffic_manager, "_traffic_vehicles", ()) or ()
        vehicles = raw_vehicles.values() if isinstance(raw_vehicles, dict) else raw_vehicles
        lane_half_width = 0.5 * float(getattr(ref_lane, "width", 3.5) or 3.5)
        best_vehicle = None
        best_distance = float("inf")
        for other in vehicles:
            if other is ego_vehicle:
                continue
            try:
                other_long, other_lateral = ref_lane.local_coordinates(other.position)
            except Exception:
                continue
            other_width = float(
                getattr(other, "WIDTH", getattr(other, "width", 2.0)) or 2.0
            )
            if abs(float(other_lateral)) > lane_half_width + 0.5 * other_width:
                continue
            distance = float(other_long) - float(ego_long)
            if 0.0 < distance <= IDMPolicy.MAX_LONG_DIST and distance < best_distance:
                best_vehicle = other
                best_distance = distance
        if best_vehicle is None:
            return None, None
        return best_vehicle, best_distance

    def _spawn_on_reference(
        self,
        env,
        ego_vehicle,
        reference_kind: str,
        *,
        block_id=None,
        socket_index=None,
        internal_road_index=None,
        lane_index: int = 0,
        spawn_longitude: float = 10.0,
        spawn_longitude_offset: float = 0.0,
        target_speed_kmh: float = 20.0,
        min_clearance_m=None,
        policy_class=None,
        policy_kwargs=None,
        vehicle_config_overrides=None,
        vehicle_type=None,
    ):
        lane_tuple = self._resolve_lane_index(
            env,
            ego_vehicle,
            reference_kind=reference_kind,
            block_id=block_id,
            socket_index=socket_index,
            internal_road_index=internal_road_index,
            lane_index=lane_index,
        )
        if lane_tuple is None:
            return None
        return self._spawn_on_lane_tuple(
            env,
            ego_vehicle,
            lane_tuple=lane_tuple,
            spawn_longitude=spawn_longitude,
            spawn_longitude_offset=spawn_longitude_offset,
            target_speed_kmh=target_speed_kmh,
            reference_kind=reference_kind,
            min_clearance_m=min_clearance_m,
            policy_class=policy_class,
            policy_kwargs=policy_kwargs,
            vehicle_config_overrides=vehicle_config_overrides,
            vehicle_type=vehicle_type,
        )

    def _spawn_on_lane_tuple(
        self,
        env,
        ego_vehicle,
        *,
        lane_tuple,
        spawn_longitude: float = 0.0,
        spawn_longitude_offset: float = 0.0,
        target_speed_kmh: float = 20.0,
        reference_kind: str = "ego_lane",
        min_clearance_m=None,
        clearance_scope: str = "all_agents",
        policy_class=None,
        policy_kwargs=None,
        vehicle_config_overrides=None,
        vehicle_type=None,
    ):
        current_map = getattr(getattr(env, "engine", None), "current_map", None)
        if current_map is None:
            return None
        lane = current_map.road_network.get_lane(lane_tuple)
        ego_long = lane.local_coordinates(ego_vehicle.position)[0] if reference_kind == "ego_lane" else 0.0
        spawn_long = min(max(spawn_longitude + spawn_longitude_offset + ego_long, 2.0), max(lane.length - 2.0, 2.0))
        spawn_position = lane.position(float(spawn_long), 0.0)
        min_clearance = (
            float(min_clearance_m)
            if min_clearance_m is not None
            else float(getattr(env, "config", {}).get("scenario_spawn_min_agent_clearance_m", 10.0))
        )
        if not self._spawn_position_clear_of_agents(
            env,
            spawn_position,
            min_clearance,
            spawn_lane_index=lane_tuple,
            clearance_scope=clearance_scope,
        ):
            self.summary.notes.append("spawn_blocked:agent_clearance")
            return None
        traffic_manager = getattr(getattr(env, "engine", None), "traffic_manager", None)
        if traffic_manager is None or not hasattr(traffic_manager, "_spawn_traffic_vehicle_if_safe"):
            return None
        if vehicle_type is None:
            vehicle_type = traffic_manager.random_vehicle_type()
        spawn_speed_mps = max(float(target_speed_kmh), 0.0) / 3.6
        spawned = traffic_manager._spawn_traffic_vehicle_if_safe(
            vehicle_type,
            {
                "spawn_lane_index": lane_tuple,
                "spawn_longitude": float(spawn_long),
                "spawn_velocity": (spawn_speed_mps, 0.0),
                "spawn_velocity_car_frame": True,
            },
            policy_class=policy_class,
            policy_kwargs=policy_kwargs,
            vehicle_config_overrides=vehicle_config_overrides,
        )
        if spawned is not None:
            setattr(spawned, "scenario_managed_vehicle", True)
            self._set_vehicle_target_speed(env, spawned, target_speed_kmh)
        return spawned

    def _resolve_adjacent_lane_index(self, env, ego_vehicle, *, lane_side: str):
        ego_lane_index = getattr(ego_vehicle, "lane_index", None)
        if ego_lane_index is None:
            lane = _select_reference_lane(ego_vehicle)
            ego_lane_index = getattr(lane, "index", None)
        if ego_lane_index is None:
            return None
        current_map = getattr(getattr(env, "engine", None), "current_map", None)
        if current_map is None:
            return None
        road_network = getattr(current_map, "road_network", None)
        graph = getattr(road_network, "graph", {}) if road_network is not None else {}
        start_node, end_node, lane_id = tuple(ego_lane_index)
        lanes = graph.get(start_node, {}).get(end_node)
        if not lanes:
            return None
        if lane_side == "left":
            adjacent_lane_id = int(lane_id) - 1
        elif lane_side == "right":
            adjacent_lane_id = int(lane_id) + 1
        else:
            return None
        if adjacent_lane_id < 0 or adjacent_lane_id >= len(lanes):
            return None
        return (start_node, end_node, adjacent_lane_id)

    @staticmethod
    def _spawn_position_clear_of_agents(
        env,
        spawn_position,
        min_clearance_m: float,
        *,
        spawn_lane_index=None,
        clearance_scope: str = "all_agents",
    ) -> bool:
        agents = getattr(env, "agents", {}) or {}
        try:
            spawn_xy = (float(spawn_position[0]), float(spawn_position[1]))
        except Exception:
            return False
        min_clearance_m = max(float(min_clearance_m), 0.0)
        target_lane = tuple(spawn_lane_index) if spawn_lane_index is not None else None
        for agent in agents.values():
            try:
                if clearance_scope == "same_lane":
                    agent_lane = getattr(agent, "lane_index", None)
                    if agent_lane is None:
                        lane = _select_reference_lane(agent)
                        agent_lane = getattr(lane, "index", None)
                    if target_lane is None or agent_lane is None or tuple(agent_lane) != target_lane:
                        continue
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
        bumper_gap_m = float(params["lead_bumper_gap_m"])
        ego_length = self._vehicle_length_m(ego_vehicle)
        spawned = self._spawn_on_reference(
            env,
            ego_vehicle,
            reference_kind="ego_lane",
            spawn_longitude_offset=bumper_gap_m + ego_length,
            target_speed_kmh=float(params["lead_target_speed_kmh"]),
            vehicle_type=self._scenario_vehicle_type(),
        )
        if spawned is None:
            return None
        lane = _select_reference_lane(ego_vehicle)
        if lane is None:
            return spawned
        try:
            ego_long = float(lane.local_coordinates(ego_vehicle.position)[0])
            center_distance = (
                bumper_gap_m
                + 0.5 * ego_length
                + 0.5 * self._vehicle_length_m(spawned)
            )
            target_long = float(
                min(
                    max(ego_long + center_distance, 2.0),
                    max(float(lane.length) - 2.0, 2.0),
                )
            )
            spawned.set_position(lane.position(target_long, 0.0))
            spawned.set_heading_theta(lane.heading_theta_at(target_long))
        except Exception:
            return spawned
        return spawned

    def _scenario_vehicle_type(self):
        """Return the fixed physical type for reproducible scenario actors."""

        if self.definition.scenario_id not in {
            "S5_hard_brake_lead",
            "S8_ego_exit_to_ramp",
        }:
            return None
        from metadrive.component.vehicle.vehicle_type import TrafficDefaultVehicle

        return TrafficDefaultVehicle

    def _resolve_lane_index(
        self,
        env,
        ego_vehicle,
        *,
        reference_kind: str,
        block_id=None,
        socket_index=None,
        internal_road_index=None,
        lane_index: int = 0,
    ):
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
        if reference_kind == "block_internal_road":
            lane_groups = getattr(getattr(block, "block_network", None), "get_positive_lanes", lambda: [])() or []
            try:
                internal_group = lane_groups[int(internal_road_index or 0)]
            except (IndexError, TypeError, ValueError):
                return None
            if not internal_group:
                return None
            road_lane_index = getattr(internal_group[0], "index", None)
            if road_lane_index is None:
                return None
            start_node, end_node = road_lane_index[:2]
            lanes = current_map.road_network.graph[start_node][end_node]
            resolved_lane = min(max(int(lane_index), 0), len(lanes) - 1)
            return (start_node, end_node, resolved_lane)
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

    @staticmethod
    def _bounded_brake_speed_kmh(
        vehicle,
        *,
        target_speed_kmh: float,
        max_deceleration_mps2: float,
        dt_s: float,
    ) -> float:
        current_speed_kmh = max(
            float(getattr(vehicle, "speed_km_h", 0.0) or 0.0),
            0.0,
        )
        target_speed_kmh = max(float(target_speed_kmh), 0.0)
        deceleration_mps2 = max(float(max_deceleration_mps2), 0.0)
        next_speed_kmh = max(
            target_speed_kmh,
            current_speed_kmh - deceleration_mps2 * max(float(dt_s), 0.0) * 3.6,
        )
        if hasattr(vehicle, "set_throttle_brake"):
            vehicle.set_throttle_brake(0.0)
        if hasattr(vehicle, "set_velocity"):
            vehicle.set_velocity(
                [next_speed_kmh / 3.6, 0.0],
                in_local_frame=True,
            )
        setattr(
            vehicle,
            "scenario_commanded_deceleration_mps2",
            (
                (current_speed_kmh - next_speed_kmh)
                / 3.6
                / max(float(dt_s), 1e-6)
            ),
        )
        return float(next_speed_kmh)

    @staticmethod
    def _vehicle_length_m(vehicle) -> float:
        return float(
            getattr(
                vehicle,
                "LENGTH",
                getattr(vehicle, "length", 5.74),
            )
            or 5.74
        )

    @classmethod
    def _bumper_gap_m(cls, ego_vehicle, other_vehicle, center_distance):
        if other_vehicle is None or center_distance is None:
            return None
        return max(
            float(center_distance)
            - 0.5
            * (
                cls._vehicle_length_m(ego_vehicle)
                + cls._vehicle_length_m(other_vehicle)
            ),
            0.0,
        )

    @staticmethod
    def _longitudinal_distance_m(ego_vehicle, other_vehicle):
        lane = _select_reference_lane(ego_vehicle)
        if lane is None or other_vehicle is None:
            return None
        try:
            ego_long = float(lane.local_coordinates(ego_vehicle.position)[0])
            other_long = float(lane.local_coordinates(other_vehicle.position)[0])
        except Exception:
            return None
        return other_long - ego_long

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

    def _recipe_triggered(self, env, ego_vehicle, params: Dict[str, object], step_count: int | None = None) -> bool:
        if "trigger_after_s" in params:
            return self._trigger_after_seconds_reached(env, params["trigger_after_s"], step_count)
        if bool(params.get("trigger_on_start", False)):
            return True
        trigger_by_local_route = params.get("trigger_by_local_route")
        if not trigger_by_local_route:
            return self._is_in_trigger_window(ego_vehicle)
        trigger_spec = self._parse_recipe_trigger(trigger_by_local_route)
        if trigger_spec is None:
            return False
        current_block_id, longitudinal = self._resolve_ego_position(ego_vehicle)
        if current_block_id != trigger_spec.block_id or longitudinal is None:
            return False
        return trigger_spec.longitudinal_min <= longitudinal <= trigger_spec.longitudinal_max

    def _trigger_after_seconds_reached(self, env, trigger_after_s: object, step_count: int | None) -> bool:
        if step_count is None:
            return False
        dt = self._scenario_step_dt_s(env)
        return float(step_count) * dt >= float(trigger_after_s)

    @staticmethod
    def _scenario_step_dt_s(env) -> float:
        engine = getattr(env, "engine", None)
        config = getattr(engine, "global_config", None)
        if config is None:
            config = getattr(env, "config", None)

        def _get_config_value(key: str, default: float) -> float:
            if config is None:
                return default
            try:
                return float(config.get(key, default))
            except Exception:
                pass
            try:
                return float(config[key])
            except Exception:
                pass
            return float(getattr(config, key, default))

        return _get_config_value("physics_world_step_size", 0.02) * _get_config_value("decision_repeat", 5.0)

    def _is_s5_startup_support_recipe(self, recipe: RecipeSpec) -> bool:
        return (
            self.definition.scenario_id == "S5_hard_brake_lead"
            and recipe.operation == "inject_adjacent_lane_vehicles"
            and bool(recipe.params.get("trigger_on_start", False))
        )

    def _parse_recipe_trigger(self, trigger_by_local_route) -> TriggerSpec | None:
        if not isinstance(trigger_by_local_route, dict):
            return None
        raw_spec = trigger_by_local_route.get(self.local_route)
        if raw_spec is None:
            return None
        if isinstance(raw_spec, TriggerSpec):
            return raw_spec
        if isinstance(raw_spec, dict):
            return TriggerSpec(
                str(raw_spec["block_id"]),
                float(raw_spec["longitudinal_min"]),
                float(raw_spec["longitudinal_max"]),
            )
        return None

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
        lane_groups = getattr(getattr(block, "block_network", None), "get_positive_lanes", lambda: [])() or []
        for lane_group in lane_groups:
            if not lane_group:
                continue
            lane_index = getattr(lane_group[0], "index", None)
            if lane_index is not None:
                return _RouteRoadRef(lane_index[0], lane_index[1])
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
