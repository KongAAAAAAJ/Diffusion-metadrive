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
from scenarios.s5_s9_sampling import resolve_s5_s9_parameters
from metadrive.policy.idm_policy import IDMPolicy
from envs.diffusion_envs.ground_truth_idm_policy import GroundTruthIDMPolicy

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class _RouteRoadRef:
    start_node: str
    end_node: str


class _TimedMainlineStreamPolicy(GroundTruthIDMPolicy):
    """IDM stream policy compatible with the sampled 1--2 s headway.

    MetaDrive's default 10 m + 1.5 s desired spacing expands the declared S7
    headway before the actors reach the conflict point.  The scenario's
    unchanged 5 m hard background spacing permits this lower cooperative IDM
    target, so the four timed roles retain their jointly solved arrival order.
    """

    DISTANCE_WANTED = 5.0
    TIME_WANTED = 0.5
    LANE_CHANGE_FREQ = 10_000


class _S9BlockingPolicy(GroundTruthIDMPolicy):
    """Keep the S9 obstruction slow and centred in its declared c3 lane.

    The generic IDM policy follows its navigation branch at the end of c3.
    For a near-stationary actor this can turn the nominal obstruction across
    lane 0, contradicting S9's fixed middle-lane blocker contract.  The
    orchestrator already enforces the sampled speed, so this policy only owns
    lane keeping and braking between those speed-profile updates.
    """

    def act(self, *args, **kwargs):
        steering = self.steering_control(self.control_object.lane)
        action = [steering, -1.0]
        self.action_info["action"] = action
        return action


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
        self._resolved_scenario_parameters: Dict[str, object] = {}
        self._actor_manifest: Dict[str, Dict[str, object]] = {}
        self._conflict_evidence: Dict[str, object] = {}
        self._route_completion: Dict[str, object] = {}
        self._initial_agent_lanes: Dict[str, tuple] = {}
        self._functional_state: Dict[str, object] = {}
        self._last_env = None

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
        self._last_env = env
        agents = getattr(env, "agents", {}) or {}
        self._initial_agent_lanes = {
            str(name): tuple(getattr(vehicle, "lane_index", ()) or ())
            for name, vehicle in agents.items()
        }
        lead = agents.get("agent0") or agents.get(agent_id)
        ego_speed = float(getattr(lead, "speed_km_h", 0.0) or 0.0)
        raw_seed = getattr(env, "current_seed", None)
        if raw_seed is None:
            raw_seed = getattr(getattr(env, "engine", None), "global_random_seed", 0)
        self._resolved_scenario_parameters = resolve_s5_s9_parameters(
            spawn_seed=int(raw_seed or 0),
            scenario_id=self.definition.scenario_id,
            local_route=self.local_route,
            ego_initial_speed_km_h=ego_speed,
            decision_dt_s=self._scenario_step_dt_s(env),
        )
        self._actor_manifest = {}
        self._conflict_evidence = {
            "minimum_actor_distance_m": None,
            "minimum_actor_ttc_s": None,
        }
        self._route_completion = {
            "agent_lane_transitions": {str(name): [] for name in agents},
            "all_agents_changed_lane": False,
            "all_agents_passed_blocker": False,
            "all_agents_entered_exit_ramp": False,
            "all_agents_entered_mainline": False,
            "returned_to_original_lane": False,
        }
        self._functional_state = {
            "last_step": None,
            "actor_speeds_kmh": {},
            "ego_speeds_kmh": {},
            "formation_stable_steps": 0,
            "s6_conflict_distances_m": {},
            "s6_conflict_origin_roads": {},
            "s6_arrival_steps": {},
            "s5_lane_candidate_steps": {},
            "s5_lane_completion_steps": {},
            "s5_reassembly_candidate_steps": {},
            "s5_reassembly_completion_steps": {},
        }
        # ``trigger_on_start`` means the actor must exist in the very first
        # expert planning snapshot.  Deferring this to env.before_step() lets
        # the planner commit a multi-second trajectory against an empty scene
        # and only discover the actor during the rolling hard audit.
        if lead is not None:
            self._execute_recipe(env, lead, step_count=0, startup_only=True)
            self._update_functional_evidence(env, step_count=0)

    def before_step(self, env, agent_id: str, step_count: int) -> None:
        self._last_env = env
        self._functional_state["last_observation_step"] = int(step_count)
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
        self._update_functional_evidence(env, step_count)

    def get_episode_summary(self) -> Dict[str, object]:
        if self._last_env is not None:
            # Capture the state produced by the final env.step() with a real
            # monotonic index.  A synthetic ``-1`` made valid last-frame lane
            # transitions look temporally invalid and prevented scenario-
            # specific evidence updaters from observing completion.
            final_step = int(
                self._functional_state.get("last_observation_step", -1)
            ) + 1
            self._update_functional_evidence(self._last_env, final_step)
        recipe_count = len(self.definition.traffic_recipes)
        completed_recipe_count = len(self._completed_recipe_keys)
        recipes_complete = completed_recipe_count == recipe_count
        functional_success = self._functional_success(recipes_complete)
        return {
            "scenario_id": self.summary.scenario_id,
            "scenario_triggered": bool(self.summary.scenario_triggered),
            "scenario_realized": bool(self.summary.scenario_realized),
            "scenario_trigger_step": self.summary.trigger_step,
            "scenario_realized_step": self.summary.realized_step,
            "scenario_recipe_count": int(recipe_count),
            "scenario_completed_recipe_count": int(completed_recipe_count),
            "scenario_recipes_complete": bool(recipes_complete),
            "scenario_notes": list(self.summary.notes),
            "scenario_random_seed": self._scenario_random_seed,
            "resolved_recipe_parameters": {
                key: dict(value)
                for key, value in self._resolved_recipe_parameters.items()
            },
            "severity_bucket": self._resolved_scenario_parameters.get(
                "severity_bucket"
            ),
            "resolved_scenario_parameters": dict(
                self._resolved_scenario_parameters
            ),
            "actor_manifest": {
                role: dict(row) for role, row in self._actor_manifest.items()
            },
            "conflict_evidence": dict(self._conflict_evidence),
            "route_completion": {
                key: (dict(value) if isinstance(value, dict) else value)
                for key, value in self._route_completion.items()
            },
            "functional_success": bool(functional_success),
            "expert_profile_id": self._expert_profile_id(),
        }

    def _expert_profile_id(self) -> str | None:
        config = getattr(self._last_env, "config", {}) if self._last_env is not None else {}
        value = config.get("rule_maker_profile_id") if hasattr(config, "get") else None
        return None if value in (None, "") else str(value)

    def _execute_recipe(
        self,
        env,
        ego_vehicle,
        step_count: int,
        *,
        startup_only: bool = False,
    ) -> None:
        if not self.definition.traffic_recipes:
            self._mark_realized(step_count, "no-op")
            return
        realized = False
        considered = False
        for recipe_index, recipe in enumerate(self.definition.traffic_recipes):
            if startup_only and not bool(recipe.params.get("trigger_on_start", False)):
                continue
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
        self._register_actor("hard_brake_lead", lead_vehicle, step_count=step_count)
        self._mark_realized(step_count, "lead_brake_profile")
        return True

    def _mark_hard_brake_lead_vehicle(self, vehicle) -> None:
        setattr(vehicle, "scenario_managed_vehicle", True)
        setattr(vehicle, "scenario_warning_marker", "!")
        setattr(vehicle, "scenario_id", self.definition.scenario_id)
        setattr(vehicle, "scenario_role", "hard_brake_lead")

    def _handle_inject_background_vehicle(self, env, ego_vehicle, params: Dict[str, object], step_count: int) -> bool:
        params = self._resolve_background_vehicle_params(env, params)
        if self.definition.scenario_id == "S6_background_merge_in":
            resolved = self._resolved_scenario_parameters
            params.update(
                target_speed_kmh=float(resolved["merge_actor_speed_km_h"]),
                merge_front_gap_m=float(resolved["target_front_bumper_gap_m"]),
                merge_rear_gap_m=float(resolved["target_rear_bumper_gap_m"]),
                merge_arrival_offset_s=float(
                    resolved["conflict_arrival_time_delta_s"]
                ),
                target_gap_id=str(resolved["target_gap_id"]),
                merge_activation_step=int(resolved["merge_activation_step"]),
                # MetaDrive's FrontBackObjects reports centre-to-centre
                # longitudinal distances.  Convert the memory's sampled
                # bumper gaps before IDMMergePolicy decides that the target
                # corridor is safe to enter.
                merge_gap_center_offset_m=5.74,
            )
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
            params["target_speed_kmh"] = merge_alignment["merge_speed_km_h"]
            ego_speed_kmh = float(
                self._resolved_scenario_parameters.get(
                    "ego_initial_speed_km_h", 22.0
                )
            )
            params["merge_creep_speed_kmh"] = (
                min(
                    float(merge_alignment["merge_speed_km_h"]),
                    max(18.0, ego_speed_kmh - 2.83),
                )
                if str(merge_alignment["target_gap_id"])
                == "agent0-agent1"
                else float(merge_alignment["merge_speed_km_h"])
            )
            params["merge_lateral_kp"] = (
                1.0
                if str(merge_alignment["target_gap_id"]) == "agent0-agent1"
                and float(merge_alignment["merge_speed_km_h"]) <= 21.2
                else 0.7
                if str(merge_alignment["target_gap_id"]) == "agent1-agent2"
                else 0.7
            )
            self._resolved_scenario_parameters["merge_actor_speed_km_h"] = (
                merge_alignment["merge_speed_km_h"]
            )
        reference_kind = str(params.get("reference_kind", "ego_lane"))
        policy_class, policy_kwargs, vehicle_config_overrides = self._resolve_injected_background_policy(params)
        if merge_alignment is not None:
            # Keep the actor's navigation alive over the same continuous
            # downstream route as the ego formation.  Binding it to the old
            # c3 endpoint made late gap-1 merges decelerate to a terminal stop
            # before occupying the designated lane.  The explicit
            # ``merge_target_lane`` binding below still controls which lane is
            # used through the conflict corridor.
            try:
                spawn_manager = getattr(
                    getattr(env, "engine", None), "spawn_manager", None
                )
                current_map = getattr(
                    getattr(env, "engine", None), "current_map", None
                )
                if str(merge_alignment["target_gap_id"]) == "agent0-agent1":
                    from envs.diffusion_envs.route_spawn_manager import (
                        RouteAwareSpawnManager,
                    )

                    c3_block = self._get_block_by_graph_id(current_map, "c3")
                    c3_road = (
                        RouteAwareSpawnManager._get_last_positive_block_network_road(
                            c3_block
                        )
                    )
                    destination = (
                        None if c3_road is None else c3_road.end_node
                    )
                else:
                    # Preserve the merge policy's native branch navigation for
                    # the second gap.  Explicit downstream routing selects the
                    # terminating outside lane before the actor occupies the
                    # designated middle-lane corridor.
                    destination = None
                if destination is not None:
                    vehicle_config_overrides = dict(vehicle_config_overrides or {})
                    vehicle_config_overrides["destination"] = destination
                    self._conflict_evidence["merge_actor_destination_node"] = str(
                        destination
                    )
            except (AttributeError, TypeError, ValueError):
                self.summary.notes.append("s6_merge_actor_destination_bind_failed")
        lane_index = params.get("lane_index", params.get("lane_id", 0))
        spawn_kwargs = dict(
            spawn_longitude=float(params.get("spawn_longitude", 15.0)),
            spawn_longitude_offset=float(params.get("spawn_longitude_offset", 0.0)),
            target_speed_kmh=float(params.get("target_speed_kmh", getattr(ego_vehicle, "speed_km_h", 20.0))),
            min_clearance_m=0.0 if merge_alignment is not None else None,
            policy_class=policy_class,
            policy_kwargs=policy_kwargs,
            vehicle_config_overrides=vehicle_config_overrides,
            vehicle_type=self._scenario_vehicle_type(),
        )
        if merge_alignment is not None:
            # The S6 conflict is on the converging connector.  Spawning on the
            # upstream branch cannot realize a 1.5--3.5 s conflict horizon
            # while ego is only 25--50 m from the merge point.
            spawned = self._spawn_on_lane_tuple(
                env,
                ego_vehicle,
                lane_tuple=tuple(merge_alignment["spawn_lane_index"]),
                reference_kind="absolute_lane",
                **spawn_kwargs,
            )
        else:
            spawned = self._spawn_on_reference(
                env,
                ego_vehicle,
                reference_kind=reference_kind,
                block_id=params.get("block_id"),
                socket_index=params.get("socket_index"),
                internal_road_index=params.get("internal_road_index"),
                lane_index=int(lane_index),
                **spawn_kwargs,
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
            # Bind the merge policy to the ego formation's continuous
            # downstream lane explicitly.  The actor's shortest-path
            # checkpoints omit the parallel mainline edge, so navigation
            # cannot infer this target from its own branch route alone.
            try:
                target_front_id, target_rear_id = str(
                    merge_alignment["target_gap_id"]
                ).split("-", 1)
                active_agents = getattr(env, "agents", {}) or {}
                target_front = active_agents[target_front_id]
                target_rear = active_agents[target_rear_id]
                setattr(spawned, "scenario_target_front_vehicle", target_front)
                setattr(spawned, "scenario_target_rear_vehicle", target_rear)
                target_front_lane = getattr(target_front, "lane", None)
                target_lane_number = int(
                    tuple(getattr(target_front_lane, "index", ()) or ())[2]
                )
                next_ref_lanes = list(
                    getattr(
                        getattr(target_front, "navigation", None),
                        "next_ref_lanes",
                        (),
                    )
                    or ()
                )
                merge_target_lane = next_ref_lanes[target_lane_number]
                actor_navigation = getattr(spawned, "navigation", None)
                actor_navigation.merge_target_lane = merge_target_lane
                actor_navigation.merge_branch_roads = {
                    tuple(merge_alignment["spawn_lane_index"][:2])
                }
                actor_navigation.merge_force_road = tuple(
                    merge_alignment["spawn_lane_index"][:2]
                )
            except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                self.summary.notes.append("s6_merge_target_lane_bind_failed")
        role = str(getattr(spawned, "scenario_vehicle_role", "injected_background"))
        self._register_actor(role, spawned, step_count=step_count)
        if merge_alignment is not None:
            self._conflict_evidence.update(dict(merge_alignment))
        self._mark_realized(step_count, f"injected:{reference_kind}")
        return True

    def _spawn_role_on_lane(
        self, env, ego_vehicle, *, role: str, lane_tuple, longitudinal_m: float,
        speed_kmh: float, step_count: int, min_clearance_m: float = 0.0,
        policy_class=None,
    ):
        spawned = self._spawn_on_lane_tuple(
            env,
            ego_vehicle,
            lane_tuple=lane_tuple,
            spawn_longitude=float(longitudinal_m),
            reference_kind="absolute_lane",
            target_speed_kmh=float(speed_kmh),
            min_clearance_m=float(min_clearance_m),
            clearance_scope="same_lane",
            policy_class=policy_class,
            vehicle_type=self._scenario_vehicle_type(),
        )
        if spawned is None:
            return None
        setattr(spawned, "scenario_vehicle_role", role)
        setattr(spawned, "scenario_warning_marker", "!")
        name = getattr(spawned, "name", None)
        if name is not None:
            self._speed_profiles[name] = {
                "remaining_steps": float("inf"),
                "target_speed_kmh": float(speed_kmh),
            }
        self._register_actor(role, spawned, step_count=step_count)
        return spawned

    def _handle_inject_s7_merge_traffic(self, env, ego_vehicle, params, step_count: int) -> bool:
        resolved = self._resolved_scenario_parameters
        lane_tuple = self._resolve_lane_index(
            env, ego_vehicle, reference_kind="block_route_road",
            block_id=params.get("block_id", "g1"), lane_index=int(params.get("lane_index", 2)),
        )
        current_map = getattr(getattr(env, "engine", None), "current_map", None)
        if lane_tuple is None or current_map is None:
            self.summary.notes.append("s7_mainline_lane_missing")
            return False
        road_network = current_map.road_network
        graph = road_network.graph
        lane = road_network.get_lane(lane_tuple)
        sampled_speeds = list(
            resolved.get("mainline_actor_speeds_km_h", (24.0,) * 6)
        )
        actor_count = int(resolved.get("actor_count", 4))
        usable_gap = float(resolved.get("usable_mainline_gap_m", 55.0))
        expected_behavior = str(resolved.get("expected_behavior", "pass_first"))
        # The four required roles form two consecutive windows on the same
        # mainline lane.  Keep their speed correlated so a faster rear draw
        # cannot reorder the stream before reaching the conflict point.
        stream_speed = float(np.clip(np.mean(sampled_speeds[:4]), 20.0, 31.0))
        if expected_behavior == "yield_then_merge":
            stream_speed = 31.0
        mainline_speeds = [stream_speed] * 4
        if expected_behavior == "yield_then_merge":
            # The following window is the complete-platoon opportunity.  Its
            # rear boundary remains within the declared 20--31 km/h range but
            # is coupled below the front boundary speed so the 45--70 m
            # usable gap cannot collapse while three stopped ego vehicles
            # restart and cross the seam.
            mainline_speeds[3] = 20.0
        else:
            # Likewise, keep the rear boundary of the direct/pass-first
            # window from closing on the staggered three-car formation.  The
            # following pair inherits that speed to preserve actor order.
            direct_rear_speed = max(20.0, stream_speed - 4.0)
            mainline_speeds[1:] = [direct_rear_speed] * 3
        optional_speeds = [
            float(np.clip(value, 20.0, 31.0))
            for value in sampled_speeds[4:actor_count]
        ]
        resolved["mainline_actor_speeds_km_h"] = [
            *mainline_speeds,
            *optional_speeds,
        ]
        ego_speed_mps = max(
            float(getattr(ego_vehicle, "speed_km_h", 24.0)) / 3.6, 0.1
        )
        ego_ttc = float(
            resolved.get("ego_distance_to_merge_point_m", 40.0)
        ) / ego_speed_mps
        if expected_behavior == "pass_first":
            # The front boundary has already cleared the physical merge
            # point when a direct gap is offered.  Its sampled arrival delta
            # is retained as conflict metadata, while the rear boundary is
            # solved from the declared usable gap.  Spawning the front actor
            # upstream from the seam made it physically occupy the ramp
            # apron just as the leader arrived, contradicting pass-first.
            front_remaining_m = 3.0
            if str(resolved.get("severity_bucket", "")) == "low":
                # Low severity represents the wide, early opportunity.  Keep
                # the rear boundary at the slow edge of the declared actor
                # domain so the full long-gap platoon can traverse before it
                # reaches the apron; this remains a live competing stream,
                # not a missing/background-free interaction.
                mainline_speeds[1:] = [20.0, 20.0, 20.0]
        else:
            # High/medium severity deliberately closes the first opportunity;
            # put its leading boundary close to the seam so the expert must
            # wait for the following complete-platoon window.
            front_remaining_m = 3.0
        actor_length_m = 5.74
        # ``mainline_headway_s`` is a front-to-front arrival headway.  Convert
        # it to centre spacing once; adding a second vehicle length here made
        # the realized headway exceed the memory contract by about 0.7 s.
        headway_center_spacing_m = max(
            actor_length_m + 5.0,
            stream_speed
            / 3.6
            * float(resolved.get("mainline_headway_s", 1.5)),
        )
        remaining_by_role = {
            "critical_gap_front": front_remaining_m,
            "critical_gap_rear": front_remaining_m + actor_length_m + usable_gap,
        }
        remaining_by_role["next_gap_front"] = (
            remaining_by_role["critical_gap_rear"]
            + headway_center_spacing_m
        )
        remaining_by_role["next_gap_rear"] = (
            remaining_by_role["next_gap_front"]
            + actor_length_m
            + usable_gap
        )

        lane_id = int(lane_tuple[2])
        chain = [lane]
        chain_indices = [tuple(lane_tuple)]
        total_length_m = float(lane.length)
        while total_length_m < max(remaining_by_role.values()) + 2.1:
            start_node = chain_indices[-1][0]
            incoming = [
                (upstream_start, lanes)
                for upstream_start, outgoing in graph.items()
                for end_node, lanes in outgoing.items()
                if end_node == start_node
                and not str(upstream_start).startswith("-")
                and len(lanes) > lane_id
            ]
            if len(incoming) != 1:
                self.summary.notes.append("s7_upstream_chain_ambiguous")
                return False
            upstream_start, upstream_lanes = incoming[0]
            upstream_lane = upstream_lanes[lane_id]
            chain.append(upstream_lane)
            chain_indices.append(tuple(upstream_lane.index))
            total_length_m += float(upstream_lane.length)

        def lane_pose_at_remaining(remaining_m: float, target_lane_id: int):
            residual = float(remaining_m)
            for chain_lane in chain:
                candidate_lanes = graph[chain_lane.index[0]][chain_lane.index[1]]
                if len(candidate_lanes) <= target_lane_id:
                    return None
                candidate = candidate_lanes[target_lane_id]
                if residual <= float(candidate.length):
                    # A solved remaining distance can land inside the 2.1 m
                    # spawn guard at either side of a road seam.  Snap only
                    # that sub-vehicle-length numerical boundary case inward;
                    # the subsequent global OBB transaction still rejects any
                    # real overlap.
                    longitude = float(
                        np.clip(
                            float(candidate.length) - residual,
                            2.1,
                            max(float(candidate.length) - 2.1, 2.1),
                        )
                    )
                    return tuple(candidate.index), longitude, candidate
                residual -= float(candidate.length)
            return None

        rows = []
        for index, role in enumerate(
            ("critical_gap_front", "critical_gap_rear", "next_gap_front", "next_gap_rear")
        ):
            pose = lane_pose_at_remaining(remaining_by_role[role], lane_id)
            if pose is None:
                self.summary.notes.append(f"s7_atomic_geometry_out_of_range:{role}")
                return False
            rows.append((role, *pose[:2], mainline_speeds[index], remaining_by_role[role], pose[2]))
        for index in range(actor_count - 4):
            optional_lane_id = max(0, lane_id - 1 - index)
            optional_remaining = front_remaining_m + 22.0 + 30.0 * index
            pose = lane_pose_at_remaining(optional_remaining, optional_lane_id)
            if pose is None:
                self.summary.notes.append(
                    f"s7_atomic_geometry_out_of_range:optional_adjacent_{index}"
                )
                return False
            rows.append(
                (
                    f"optional_adjacent_{index}",
                    *pose[:2],
                    optional_speeds[index],
                    optional_remaining,
                    pose[2],
                )
            )
        self._conflict_evidence["s7_atomic_candidate_geometry"] = {
            "route_chain_length_m": float(total_length_m),
            "rows": [
                {
                    "role": role,
                    "lane_index": list(actor_lane_tuple),
                    "longitudinal_m": float(longitudinal),
                    "remaining_to_conflict_m": float(remaining),
                    "speed_km_h": float(speed),
                    "conflict_arrival_time_s": float(remaining / (speed / 3.6)),
                }
                for role, actor_lane_tuple, longitudinal, speed, remaining, _ in rows
            ],
        }
        self._conflict_evidence["s7_conflict_node"] = str(lane_tuple[1])
        # Validate every OBB before the atomic spawn transaction begins.
        world_poses = []
        for role, actor_lane_tuple, longitude, _, _, actor_lane in rows:
            position = actor_lane.position(float(longitude), 0.0)
            heading = float(actor_lane.heading_theta_at(float(longitude)))
            if any(
                self._oriented_boxes_overlap(
                    position, heading, (5.74, 2.3), other_position,
                    other_heading, (5.74, 2.3)
                )
                for other_position, other_heading in world_poses
            ):
                self.summary.notes.append(f"s7_atomic_obb_overlap:{role}")
                return False
            world_poses.append((position, heading))
        for role, target_lane, longitudinal, speed, _, _ in rows:
            if self._spawn_role_on_lane(
                env, ego_vehicle, role=role, lane_tuple=target_lane,
                longitudinal_m=longitudinal, speed_kmh=speed,
                step_count=step_count, min_clearance_m=0.0,
                policy_class=_TimedMainlineStreamPolicy,
            ) is None:
                self.summary.notes.append(f"s7_atomic_spawn_failed:{role}")
                return False
        self._conflict_evidence.update({
            "usable_mainline_gap_m": usable_gap,
            "expected_behavior": expected_behavior,
            "declared_actor_count": actor_count,
        })
        self._mark_realized(step_count, "s7_atomic_traffic_spawned")
        return len([r for r in self._actor_manifest if r.startswith(("critical_", "next_", "optional_"))]) == actor_count

    def _handle_inject_s8_exit_gap(self, env, ego_vehicle, params, step_count: int) -> bool:
        resolved = self._resolved_scenario_parameters
        lane_tuple = self._resolve_lane_index(
            env, ego_vehicle, reference_kind="block_internal_road",
            block_id=params.get("block_id", "g0"),
            internal_road_index=params.get("internal_road_index", 0),
            lane_index=int(params.get("lane_index", 2)),
        )
        current_map = getattr(getattr(env, "engine", None), "current_map", None)
        if lane_tuple is None or current_map is None:
            self.summary.notes.append("s8_exit_lane_missing")
            return False
        lane = current_map.road_network.get_lane(lane_tuple)
        agent_s = []
        for agent in (getattr(env, "agents", {}) or {}).values():
            try:
                agent_s.append(float(lane.local_coordinates(agent.position)[0]))
            except Exception:
                continue
        if not agent_s:
            return False
        gap = float(resolved.get("usable_exit_lane_gap_m", 60.0))
        actor_length_m = 5.74
        # Five metres is the immutable runtime background-gap boundary, not
        # a robust spawn target.  Leave one platoon-sized buffer so the rear
        # actor remains causally close without making ordinary tracking error
        # or the required seam deceleration produce an unrecoverable t=0
        # violation later in the episode.
        hard_clearance_m = 12.0
        rear_s = min(agent_s) - actor_length_m - hard_clearance_m
        front_s = rear_s + actor_length_m + gap
        minimum_front_s = max(agent_s) + actor_length_m + hard_clearance_m
        if front_s < minimum_front_s:
            front_s = minimum_front_s
            rear_s = front_s - actor_length_m - gap
        rows = (
            ("exit_gap_front", front_s, resolved.get("exit_lane_front_actor_speed_km_h", 19.0)),
            ("exit_gap_rear", rear_s, resolved.get("exit_lane_rear_actor_speed_km_h", 24.0)),
        )
        for role, longitudinal, speed in rows:
            if not 2.0 <= float(longitudinal) <= float(lane.length) - 2.0:
                self.summary.notes.append("s8_gap_geometry_out_of_range")
                return False
        for role, longitudinal, speed in rows:
            spawned = self._spawn_role_on_lane(
                env, ego_vehicle, role=role, lane_tuple=lane_tuple,
                longitudinal_m=longitudinal, speed_kmh=float(speed),
                step_count=step_count, min_clearance_m=hard_clearance_m,
                policy_class=_TimedMainlineStreamPolicy,
            )
            if spawned is None:
                self.summary.notes.append(f"s8_atomic_spawn_failed:{role}")
                return False
            policy = getattr(
                getattr(env, "engine", None), "get_policy", lambda *_: None
            )(getattr(spawned, "name", None))
            if policy is not None:
                # The rightmost lane is already the valid exit approach.
                # MetaDrive may initialize a traffic actor's routing target
                # to another parallel lane even when it was physically
                # spawned on lane 0; bind it to the declared role lane so the
                # actor cannot silently abandon the interaction corridor.
                policy.routing_target_lane = lane
        self._conflict_evidence.update(
            {
                "usable_exit_lane_gap_m": gap,
                "exit_gap_spawn_lane_index": list(lane_tuple),
                "exit_gap_spawn_front_s_m": float(front_s),
                "exit_gap_spawn_rear_s_m": float(rear_s),
            }
        )
        self._mark_realized(step_count, "s8_exit_gap_spawned")
        return {"exit_gap_front", "exit_gap_rear"}.issubset(self._actor_manifest)

    def _handle_inject_s9_bypass_actors(self, env, ego_vehicle, params, step_count: int) -> bool:
        resolved = self._resolved_scenario_parameters
        source_lane = self._resolve_lane_index(
            env, ego_vehicle, reference_kind="block_internal_road",
            block_id=params.get("block_id", "c3"),
            internal_road_index=params.get("internal_road_index", 1),
            lane_index=int(params.get("source_lane_id", 1)),
        )
        if source_lane is None:
            self.summary.notes.append("s9_source_lane_missing")
            return False
        bypass_lane = (source_lane[0], source_lane[1], int(params.get("bypass_lane_id", 0)))
        current_map = getattr(getattr(env, "engine", None), "current_map", None)
        lane = current_map.road_network.get_lane(source_lane)
        bypass = current_map.road_network.get_lane(bypass_lane)
        try:
            lead_s = float(lane.local_coordinates(ego_vehicle.position)[0])
        except Exception:
            return False
        blocker_gap = float(resolved.get("agent0_to_blocker_bumper_gap_m", 20.0))
        blocker_s = lead_s + blocker_gap + self._vehicle_length_m(ego_vehicle)
        constraint_s = lead_s + float(
            resolved.get("bypass_constraint_actor_relative_offset_m", 0.0)
        )
        # Keep a full vehicle-scale buffer to the c3 seam. At the generic
        # 2 m lane-end margin, the near-stopped blocker can still enter the
        # successor connector inside the committed audit horizon, making its
        # predicted footprint sweep across the bypass lane. Five metres keeps
        # it a genuine lane-1 obstruction while preserving a 15--28 m ego gap.
        maximum_blocker_s = float(lane.length) - 5.0
        if blocker_s > maximum_blocker_s:
            blocker_s = maximum_blocker_s
            blocker_gap = float(
                blocker_s
                - lead_s
                - self._vehicle_length_m(ego_vehicle)
            )
            resolved["agent0_to_blocker_bumper_gap_m"] = blocker_gap
            closing_speed_mps = max(
                (
                    float(resolved.get("ego_initial_speed_km_h", 0.0))
                    - float(resolved.get("blocker_speed_km_h", 0.0))
                )
                / 3.6,
                1.0e-3,
            )
            resolved["predicted_blocker_ttc_s"] = (
                blocker_gap / closing_speed_mps
            )
            self.summary.notes.append("s9_blocker_gap_clipped_to_lane")
        if (
            not 2.0 <= blocker_s <= maximum_blocker_s
            or not 15.0 <= blocker_gap <= 28.0
        ):
            self.summary.notes.append("s9_blocker_geometry_out_of_range")
            return False
        constraint_s = min(max(constraint_s, 2.0), float(bypass.length) - 2.0)
        rows = (
            (
                "blocking_actor",
                source_lane,
                blocker_s,
                resolved.get("blocker_speed_km_h", 4.0),
                _S9BlockingPolicy,
            ),
            (
                "bypass_constraint_actor",
                bypass_lane,
                constraint_s,
                resolved.get("bypass_constraint_actor_speed_km_h", 17.0),
                None,
            ),
        )
        for role, lane_tuple, longitudinal, speed, policy_class in rows:
            if self._spawn_role_on_lane(
                env, ego_vehicle, role=role, lane_tuple=lane_tuple,
                longitudinal_m=longitudinal, speed_kmh=float(speed),
                step_count=step_count, min_clearance_m=0.0,
                policy_class=policy_class,
            ) is None:
                self.summary.notes.append(f"s9_atomic_spawn_failed:{role}")
                return False
        self._conflict_evidence.update({
            "blocker_initial_bumper_gap_m": blocker_gap,
            "usable_bypass_gap_m": resolved.get("usable_bypass_gap_m"),
            "required_lane_change": "LEFT",
        })
        self._mark_realized(step_count, "s9_bypass_actors_spawned")
        return {"blocking_actor", "bypass_constraint_actor"}.issubset(self._actor_manifest)

    @staticmethod
    def _resolve_injected_background_policy(params: Dict[str, object]):
        policy_name = params.get("policy")
        if policy_name is None:
            return None, None, None
        cruise_speed = float(
            params.get(
                "merge_cruise_speed_kmh",
                params.get("target_speed_kmh", 24.0),
            )
        )
        if str(policy_name) == "idm_merge":
            from envs.diffusion_envs.idm_merge_policy import IDMMergePolicy, StartEdgeNodeNavigation

            center_offset_m = float(
                params.get("merge_gap_center_offset_m", 0.0)
            )
            s6_designated_gap = bool(
                str(params.get("scenario_vehicle_role", ""))
                == "s6_gap_intruder"
            )
            s6_sweep_bumper_gate_m = (
                5.75
                if str(params.get("target_gap_id", "")) == "agent0-agent1"
                else 6.0
            )
            policy_kwargs = {
                "merge_front_gap_m": float(
                    s6_sweep_bumper_gate_m
                    if s6_designated_gap
                    else params.get("merge_front_gap_m", 6.0)
                ) + center_offset_m,
                "merge_rear_gap_m": float(
                    s6_sweep_bumper_gate_m
                    if s6_designated_gap
                    else params.get("merge_rear_gap_m", 6.0)
                ) + center_offset_m,
                "merge_creep_speed_kmh": float(params.get("merge_creep_speed_kmh", 5.0)),
                "merge_cruise_speed_kmh": cruise_speed,
                "merge_rear_ttc_min_s": float(
                    params.get("merge_rear_ttc_min_s", 4.0)
                ),
                "merge_lateral_kp": float(params.get("merge_lateral_kp", 0.7)),
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
            if self.definition.scenario_id == "S5_hard_brake_lead":
                resolved = self._resolved_scenario_parameters
                vehicle_params["spawn_longitude_offset_m"] = float(
                    resolved[f"{lane_side}_offset_m"]
                )
                vehicle_params["target_speed_kmh"] = float(
                    resolved[f"{lane_side}_speed_km_h"]
                )
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
            role = str(
                getattr(spawned, "scenario_vehicle_role", f"adjacent_{lane_side}")
            )
            self._register_actor(role, spawned, step_count=step_count)
            self._spawned_adjacent_vehicle_keys.add(spawn_key)
            spawned_name = getattr(spawned, "name", None)
            if spawned_name is not None:
                self._speed_profiles[spawned_name] = {
                    "remaining_steps": float(vehicle_params.get("speed_profile_duration_steps", float("inf"))),
                    "target_speed_kmh": float(vehicle_params.get("target_speed_kmh", getattr(ego_vehicle, "speed_km_h", 20.0))),
                }
            self._mark_realized(step_count, f"adjacent_spawned:{vehicle_name}")
            realized = True
        expected = {
            f"{self.local_route}:{str(row.get('name', f'vehicle_{index}'))}"
            for index, row in enumerate(vehicles)
            if isinstance(row, dict)
        }
        return bool(realized and expected.issubset(self._spawned_adjacent_vehicle_keys))

    def _resolve_hard_brake_params(self, env, params: Dict[str, object]) -> Dict[str, object]:
        resolved = dict(params)
        gap_range = params.get("lead_bumper_gap_range_m")
        if not self._is_float_range(gap_range):
            raise ValueError("hard_brake_lead requires lead_bumper_gap_range_m")
        sampled = self._resolved_scenario_parameters
        resolved["lead_bumper_gap_m"] = float(
            sampled["lead_trigger_bumper_gap_m"]
            if "lead_trigger_bumper_gap_m" in sampled
            else self._sample_float_range(env, gap_range)
        )
        resolved["lead_target_speed_kmh"] = float(
            sampled.get(
                "lead_approach_speed_km_h",
                params.get(
                    "lead_target_speed_kmh",
                    self._sample_float_range(env, params["lead_target_speed_range_kmh"])
                    if "lead_target_speed_range_kmh" in params
                    else getattr((getattr(env, "agents", {}) or {}).get("agent0"), "speed_km_h", 24.0),
                ),
            )
        )
        resolved["brake_target_speed_kmh"] = float(
            sampled["lead_target_speed_km_h"]
            if "lead_target_speed_km_h" in sampled
            else (
                self._sample_float_range(env, params["brake_target_speed_range_kmh"])
                if "brake_target_speed_range_kmh" in params
                else params.get("brake_target_speed_kmh", 1.0)
            )
        )
        deceleration_range = params.get("brake_deceleration_range_mps2")
        if not self._is_float_range(deceleration_range):
            raise ValueError("hard_brake_lead requires brake_deceleration_range_mps2")
        resolved["brake_deceleration_mps2"] = float(
            sampled["lead_brake_deceleration_mps2"]
            if "lead_brake_deceleration_mps2" in sampled
            else self._sample_float_range(env, deceleration_range)
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
        if not -0.5 <= arrival_offset_s <= 0.5 or merge_speed_mps <= 0.0:
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
        target_gap_id = str(params.get("target_gap_id", "agent0-agent1"))
        try:
            front_id, rear_id = target_gap_id.split("-", 1)
            front = (getattr(env, "agents", {}) or {})[front_id]
            rear = (getattr(env, "agents", {}) or {})[rear_id]
        except (KeyError, ValueError):
            self.summary.notes.append("s6_target_gap_invalid")
            return None
        leader_lane = getattr(front, "lane", None)
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
                leader_lane.local_coordinates(front.position)[0]
            )
            leader_remaining_m = float(leader_lane.length) - leader_longitudinal
            leader_speed_mps = float(getattr(front, "speed_km_h", 0.0)) / 3.6
        except (AttributeError, TypeError, ValueError):
            self.summary.notes.append("s6_merge_geometry_invalid")
            return None
        if leader_remaining_m <= 0.0 or leader_speed_mps <= 0.0:
            self.summary.notes.append("s6_merge_geometry_invalid")
            return None

        # The short S6 approach enters a constrained curved-route envelope
        # before the conflict point.  Solving both gap boundaries with the
        # instantaneous spawn speed predicts the wrong gap; use the same
        # asymmetric pass-first/yield envelopes as the controller.
        # The target-gap front boundary receives the expert's pass-first
        # pulse, while the rear boundary yields.  These are the two verified
        # executable speed envelopes used by the closed-loop controller.
        conflict_approach_speed_mps = min(leader_speed_mps, 21.0 / 3.6)
        leader_ttc_s = leader_remaining_m / conflict_approach_speed_mps
        rear_lane = getattr(rear, "lane", None)
        rear_lane_index = getattr(rear_lane, "index", None)
        if tuple((rear_lane_index or ())[:2]) != (leader_start, merge_node):
            self.summary.notes.append("s6_rear_lane_mismatch")
            return None
        try:
            rear_remaining_m = float(rear_lane.length) - float(
                rear_lane.local_coordinates(rear.position)[0]
            )
            rear_speed_mps = float(getattr(rear, "speed_km_h", 0.0)) / 3.6
            rear_conflict_speed_mps = min(rear_speed_mps, 15.0 / 3.6)
            rear_ttc_s = rear_remaining_m / rear_conflict_speed_mps
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            self.summary.notes.append("s6_rear_geometry_invalid")
            return None
        if not leader_ttc_s < rear_ttc_s:
            self.summary.notes.append("s6_target_gap_order_invalid")
            return None
        gap_center_ttc_s = 0.5 * (leader_ttc_s + rear_ttc_s)
        response_compensation_s = float(
            self._resolved_scenario_parameters.get(
                "response_timing_compensation_s", 0.0
            )
        )
        merge_ttc_s = (
            gap_center_ttc_s + arrival_offset_s + response_compensation_s
        )
        if not leader_ttc_s < merge_ttc_s < rear_ttc_s + 3.0:
            self.summary.notes.append("s6_merge_not_in_target_gap")
            return None
        branch_route_length_m = float(branch_lane.length) + float(connector_lane.length)
        # Solve the birth point over the complete ramp chain.  Short arrival
        # horizons land on the converging connector; longer horizons land on
        # its upstream branch.  In both cases longitude remains a dependent
        # variable of the sampled conflict time and actor speed.
        required_remaining_m = merge_speed_mps * merge_ttc_s
        if (
            str(target_gap_id) == "agent1-agent2"
            or required_remaining_m <= float(connector_lane.length) - 2.0
        ):
            spawn_lane = connector_lane
            spawn_longitude_m = max(
                2.0,
                float(connector_lane.length) - required_remaining_m,
            )
        else:
            spawn_lane = branch_lane
            spawn_longitude_m = (
                float(branch_lane.length)
                + float(connector_lane.length)
                - required_remaining_m
            )
        minimum_spawn_m = 2.0
        maximum_spawn_m = float(spawn_lane.length) - 2.0
        if not minimum_spawn_m <= spawn_longitude_m <= maximum_spawn_m:
            self.summary.notes.append("s6_merge_spawn_out_of_range")
            return None

        conflict_point = leader_lane.position(float(leader_lane.length), 0.0)
        agents_for_spawn = list((getattr(env, "agents", {}) or {}).values())

        def _spawn_overlaps_agent(longitude_m: float) -> bool:
            position = spawn_lane.position(float(longitude_m), 0.0)
            heading = float(spawn_lane.heading_theta_at(longitude_m))
            return any(
                self._oriented_boxes_overlap(
                    position,
                    heading,
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
                )
                for agent in agents_for_spawn
            )

        solved_spawn_longitude_m = float(spawn_longitude_m)
        while (
            _spawn_overlaps_agent(solved_spawn_longitude_m)
            and solved_spawn_longitude_m + 0.5 <= maximum_spawn_m
            and solved_spawn_longitude_m - spawn_longitude_m < 12.0
        ):
            solved_spawn_longitude_m += 0.5
        if _spawn_overlaps_agent(solved_spawn_longitude_m):
            self.summary.notes.append("s6_merge_initial_obb_overlap")
            return None
        spawn_projection_m = solved_spawn_longitude_m - float(spawn_longitude_m)
        spawn_longitude_m = solved_spawn_longitude_m
        merge_conflict_distance_m = (
            float(connector_lane.length) - spawn_longitude_m
            if spawn_lane is connector_lane
            else float(branch_lane.length)
            - spawn_longitude_m
            + float(connector_lane.length)
        )
        merge_ttc_s = merge_conflict_distance_m / merge_speed_mps
        self._conflict_evidence["s6_alignment_candidate"] = {
            "branch_lane_length_m": float(branch_lane.length),
            "connector_lane_length_m": float(connector_lane.length),
            "branch_route_length_m": float(branch_route_length_m),
            "target_front_ttc_s": float(leader_ttc_s),
            "target_rear_ttc_s": float(rear_ttc_s),
            "target_front_conflict_speed_km_h": 21.0,
            "target_rear_conflict_speed_km_h": 15.0,
            "gap_center_ttc_s": float(gap_center_ttc_s),
            "merge_ttc_s": float(merge_ttc_s),
            "response_timing_compensation_s": float(response_compensation_s),
            "merge_speed_mps": float(merge_speed_mps),
            "merge_speed_km_h": float(merge_speed_mps * 3.6),
            "candidate_spawn_longitude_m": float(spawn_longitude_m),
            "candidate_spawn_lane_index": list(spawn_lane.index),
            "obb_safe_spawn_projection_m": float(spawn_projection_m),
            "valid_spawn_interval_m": [
                float(minimum_spawn_m), float(maximum_spawn_m)
            ],
        }
        return {
            "spawn_longitude_m": float(spawn_longitude_m),
            "merge_speed_km_h": float(merge_speed_mps * 3.6),
            "spawn_lane_index": tuple(spawn_lane.index),
            "merge_arrival_offset_s": float(arrival_offset_s),
            "response_timing_compensation_s": float(response_compensation_s),
            "target_gap_id": target_gap_id,
            "target_front_bumper_gap_m": float(
                params.get("merge_front_gap_m", 6.0)
            ),
            "target_rear_bumper_gap_m": float(
                params.get("merge_rear_gap_m", 6.0)
            ),
            "target_front_ttc_s": float(leader_ttc_s),
            "merge_ttc_s": float(merge_ttc_s),
            "target_rear_ttc_s": float(rear_ttc_s),
            "gap_center_ttc_s": float(gap_center_ttc_s),
            "target_front_conflict_distance_m": float(leader_remaining_m),
            "merge_conflict_distance_m": float(merge_conflict_distance_m),
            "obb_safe_spawn_projection_m": float(spawn_projection_m),
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
            "S6_background_merge_in",
            "S7_ego_merge_from_ramp",
            "S8_ego_exit_to_ramp",
            "S9_narrow_channel_negotiation",
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
        if isinstance(vehicles, dict):
            vehicles = vehicles.values()
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
        if "trigger_after_range_s" in params:
            key = "recipe_trigger_after_s"
            if key not in self._resolved_scenario_parameters:
                if "brake_trigger_time_s" in self._resolved_scenario_parameters:
                    value = self._resolved_scenario_parameters["brake_trigger_time_s"]
                else:
                    value = self._sample_float_range(env, params["trigger_after_range_s"])
                self._resolved_scenario_parameters[key] = float(value)
            return self._trigger_after_seconds_reached(
                env, self._resolved_scenario_parameters[key], step_count
            )
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

    def _register_actor(self, role: str, vehicle, *, step_count: int) -> None:
        lane_index = tuple(getattr(vehicle, "lane_index", ()) or ())
        if not lane_index:
            lane = _select_reference_lane(vehicle)
            lane_index = tuple(getattr(lane, "index", ()) or ())
        position = getattr(vehicle, "position", (0.0, 0.0))
        self._actor_manifest[str(role)] = {
            "role": str(role),
            "object_name": str(getattr(vehicle, "name", "")),
            "spawn_step": int(step_count),
            "spawn_lane_index": list(lane_index),
            "spawn_position_xy": [float(position[0]), float(position[1])],
            "active": True,
        }

    def _update_functional_evidence(self, env, step_count: int) -> None:
        agents = getattr(env, "agents", {}) or {}
        for agent_id, vehicle in agents.items():
            lane_index = tuple(getattr(vehicle, "lane_index", ()) or ())
            if not lane_index:
                lane = _select_reference_lane(vehicle)
                lane_index = tuple(getattr(lane, "index", ()) or ())
            rows = self._route_completion.setdefault(
                "agent_lane_transitions", {}
            ).setdefault(str(agent_id), [])
            previous = self._initial_agent_lanes.get(str(agent_id), ())
            if lane_index and lane_index != previous:
                encoded = list(lane_index)
                if not rows or rows[-1].get("lane_index") != encoded:
                    rows.append({"step": int(step_count), "lane_index": encoded})

        raw_traffic = getattr(
            getattr(getattr(env, "engine", None), "traffic_manager", None),
            "_traffic_vehicles", (),
        ) or ()
        traffic = list(raw_traffic.values() if isinstance(raw_traffic, dict) else raw_traffic)
        traffic_by_name = {str(getattr(v, "name", "")): v for v in traffic}
        minimum_distance = self._conflict_evidence.get("minimum_actor_distance_m")
        minimum_ttc = self._conflict_evidence.get("minimum_actor_ttc_s")
        for role, row in self._actor_manifest.items():
            actor = traffic_by_name.get(str(row.get("object_name", "")))
            row["active"] = actor is not None
            if actor is None:
                continue
            lane = _select_reference_lane(actor)
            row["current_lane_index"] = list(
                tuple(getattr(lane, "index", ()) or ())
            )
            row["current_speed_km_h"] = float(
                getattr(actor, "speed_km_h", 0.0) or 0.0
            )
            actor_position = getattr(actor, "position", None)
            if actor_position is not None:
                row["current_position_xy"] = [
                    float(actor_position[0]), float(actor_position[1])
                ]
            for ego in agents.values():
                try:
                    delta = np.asarray(actor.position[:2], dtype=np.float64) - np.asarray(
                        ego.position[:2], dtype=np.float64
                    )
                    distance = float(np.linalg.norm(delta))
                except Exception:
                    continue
                if minimum_distance is None or distance < float(minimum_distance):
                    minimum_distance = distance
                relative_speed = (
                    float(getattr(ego, "speed_km_h", 0.0) or 0.0)
                    - float(getattr(actor, "speed_km_h", 0.0) or 0.0)
                ) / 3.6
                if relative_speed > 1e-6:
                    ttc = distance / relative_speed
                    if minimum_ttc is None or ttc < float(minimum_ttc):
                        minimum_ttc = ttc
            if role == "s6_gap_intruder":
                actor_policy = getattr(
                    getattr(env, "engine", None), "get_policy", lambda *_: None
                )(getattr(actor, "name", None))
                merge_state = dict(
                    getattr(actor_policy, "action_info", {}) or {}
                )
                merge_active = bool(merge_state.get("merge_active", False))
                merge_force_active = bool(
                    merge_state.get("merge_force_active", False)
                )
                merge_gap_accepted = bool(
                    merge_state.get("merge_gap_accepted", False)
                )
                merge_sweep_committed = bool(
                    merge_state.get("merge_sweep_committed", False)
                    or (
                        merge_active
                        and merge_force_active
                        and merge_gap_accepted
                    )
                )
                self._conflict_evidence.update(
                    {
                        "merge_policy_active": merge_active,
                        "merge_policy_force_active": merge_force_active,
                        "merge_policy_gap_accepted": merge_gap_accepted,
                        "merge_policy_front_gap_m": merge_state.get(
                            "merge_front_gap"
                        ),
                        "merge_policy_rear_gap_m": merge_state.get(
                            "merge_rear_gap"
                        ),
                        "merge_policy_rear_ttc_s": merge_state.get(
                            "merge_rear_ttc_s"
                        ),
                        "merge_sweep_committed": bool(
                            self._conflict_evidence.get(
                                "merge_sweep_committed", False
                            )
                            or merge_sweep_committed
                        ),
                    }
                )
                if (
                    merge_sweep_committed
                    and "merge_sweep_committed_step"
                    not in self._conflict_evidence
                ):
                    self._conflict_evidence["merge_sweep_committed_step"] = int(
                        step_count
                    )
                merge_target_lane = getattr(
                    getattr(actor, "navigation", None),
                    "merge_target_lane",
                    None,
                )
                merge_target_index = tuple(
                    getattr(merge_target_lane, "index", ()) or ()
                )
                if merge_target_index:
                    self._conflict_evidence["merge_target_lane_index"] = list(
                        merge_target_index
                    )
                spawn_lane = tuple(row.get("spawn_lane_index", ()))
                current_lane = tuple(row.get("current_lane_index", ()))
                target_gap_id = str(
                    self._resolved_scenario_parameters.get("target_gap_id", "")
                )
                target_front_id = target_gap_id.split("-", 1)[0]
                target_front = agents.get(target_front_id)
                target_lane = tuple(
                    getattr(target_front, "lane_index", ()) or ()
                )
                if (
                    len(spawn_lane) >= 2
                    and len(current_lane) >= 3
                    and len(target_lane) >= 3
                ):
                    merge_completed = bool(
                        self._conflict_evidence.get("merge_completed", False)
                        or (
                            current_lane[:2] != spawn_lane[:2]
                            and int(current_lane[2]) == int(target_lane[2])
                        )
                    )
                    self._conflict_evidence["merge_completed"] = merge_completed
                    setattr(actor, "scenario_merge_completed", merge_completed)
        self._conflict_evidence["minimum_actor_distance_m"] = minimum_distance
        self._conflict_evidence["minimum_actor_ttc_s"] = minimum_ttc

        if self.definition.scenario_id == "S5_hard_brake_lead":
            self._update_s5_functional_evidence(
                env, traffic_by_name, int(step_count)
            )
        elif self.definition.scenario_id == "S6_background_merge_in":
            self._update_s6_functional_evidence(
                env, traffic_by_name, int(step_count)
            )
        elif self.definition.scenario_id == "S7_ego_merge_from_ramp":
            self._update_s7_functional_evidence(
                env, traffic_by_name, int(step_count)
            )

        current_lanes = {
            str(name): tuple(getattr(vehicle, "lane_index", ()) or ())
            for name, vehicle in agents.items()
        }
        if self.definition.scenario_id == "S9_narrow_channel_negotiation":
            changed = {}
            for name, rows in self._route_completion["agent_lane_transitions"].items():
                initial = self._initial_agent_lanes.get(name, ())
                changed[name] = any(
                    len(initial) >= 3
                    and row.get("lane_index", [None, None, None])[:2] == list(initial[:2])
                    and row.get("lane_index", [None, None, None])[2] == 0
                    for row in rows
                )
            self._route_completion["all_agents_changed_lane"] = bool(
                changed and all(changed.values())
            )
        else:
            self._route_completion["all_agents_changed_lane"] = bool(
                current_lanes
                and all(
                    lane and lane != self._initial_agent_lanes.get(name, ())
                    for name, lane in current_lanes.items()
                )
            )
        blocks = {
            name: self._road_to_block_id.get(tuple(lane[:2]))
            for name, lane in current_lanes.items()
            if len(lane) >= 2
        }
        generic_all_mainline = bool(
            blocks
            and all(
                block in {"g1", "c3", "merge0", "s_main2", "split0"}
                for block in blocks.values()
            )
        )
        self._route_completion["all_agents_entered_mainline"] = bool(
            self._route_completion.get("all_agents_entered_mainline", False)
            or generic_all_mainline
        )
        self._route_completion["all_agents_entered_exit_ramp"] = bool(
            blocks and all(block in {"s_ramp0", "c0_ramp0", "s_ramp1", "c1_ramp0"} for block in blocks.values())
        )
        if self.definition.scenario_id == "S8_ego_exit_to_ramp":
            self._update_s8_functional_evidence(
                env, traffic_by_name, current_lanes, blocks, int(step_count)
            )
        if self.definition.scenario_id == "S9_narrow_channel_negotiation":
            returned = False
            for name, lane in current_lanes.items():
                transitions = self._route_completion["agent_lane_transitions"].get(name, [])
                if transitions and lane == self._initial_agent_lanes.get(name, ()):
                    returned = True
            self._route_completion["returned_to_original_lane"] = returned
            blocker = self._actor_manifest.get("blocking_actor")
            blocker_vehicle = (
                traffic_by_name.get(str(blocker.get("object_name", "")))
                if blocker
                else None
            )
            if blocker_vehicle is not None:
                source_lane = _select_reference_lane(blocker_vehicle)
                try:
                    blocker_s = float(source_lane.local_coordinates(blocker_vehicle.position)[0])
                    self._route_completion["all_agents_passed_blocker"] = bool(
                        self._route_completion.get(
                            "all_agents_passed_blocker", False
                        )
                        or (
                            agents
                            and all(
                                float(source_lane.local_coordinates(vehicle.position)[0])
                                > blocker_s + 0.5 * self._vehicle_length_m(vehicle)
                                for vehicle in agents.values()
                            )
                        )
                    )
                except Exception:
                    pass
            self._update_s9_functional_evidence(
                env, traffic_by_name, current_lanes, blocks, int(step_count)
            )

    def _update_s5_functional_evidence(
        self, env, traffic_by_name: Dict[str, object], step_count: int
    ) -> None:
        """Measure the realized brake response and post-hazard recovery."""
        if step_count < 0 or self._functional_state.get("last_step") == step_count:
            return
        dt = max(float(self._scenario_step_dt_s(env)), 1e-3)
        agents = getattr(env, "agents", {}) or {}
        # MetaDrive may clear controlled agents before the terminal summary is
        # requested.  Preserve the last complete physical observation instead
        # of replacing valid lane-change/recovery evidence with an empty set.
        if len(agents) != 3:
            return
        previous_ego = dict(self._functional_state.get("ego_speeds_kmh", {}) or {})
        peak_ego_decel = float(
            self._conflict_evidence.get("ego_peak_deceleration_mps2", 0.0) or 0.0
        )
        for name, vehicle in agents.items():
            speed = float(getattr(vehicle, "speed_km_h", 0.0) or 0.0)
            previous = previous_ego.get(str(name))
            if self.summary.scenario_triggered and previous is not None:
                peak_ego_decel = max(
                    peak_ego_decel, (float(previous) - speed) / 3.6 / dt
                )
            previous_ego[str(name)] = speed
        self._functional_state["ego_speeds_kmh"] = previous_ego
        self._conflict_evidence["ego_peak_deceleration_mps2"] = float(
            peak_ego_decel
        )

        lead_row = self._actor_manifest.get("hard_brake_lead")
        lead = (
            traffic_by_name.get(str(lead_row.get("object_name", "")))
            if lead_row
            else None
        )
        if lead is None:
            self._functional_state["last_step"] = step_count
            return
        lead_speed = float(getattr(lead, "speed_km_h", 0.0) or 0.0)
        previous_actor = dict(
            self._functional_state.get("actor_speeds_kmh", {}) or {}
        )
        previous_speed = previous_actor.get("hard_brake_lead")
        peak_decel = float(
            self._conflict_evidence.get("lead_observed_peak_deceleration_mps2", 0.0)
            or 0.0
        )
        commanded_decel = float(
            getattr(lead, "scenario_commanded_deceleration_mps2", 0.0) or 0.0
        )
        peak_decel = max(peak_decel, commanded_decel)
        if previous_speed is not None:
            peak_decel = max(
                peak_decel, (float(previous_speed) - lead_speed) / 3.6 / dt
            )
        previous_actor["hard_brake_lead"] = lead_speed
        self._functional_state["actor_speeds_kmh"] = previous_actor
        self._conflict_evidence["lead_observed_peak_deceleration_mps2"] = float(
            peak_decel
        )
        self._conflict_evidence["lead_commanded_peak_deceleration_mps2"] = float(
            max(
                float(
                    self._conflict_evidence.get(
                        "lead_commanded_peak_deceleration_mps2", 0.0
                    )
                    or 0.0
                ),
                commanded_decel,
            )
        )
        observed_min = self._conflict_evidence.get("lead_observed_min_speed_km_h")
        observed_max = self._conflict_evidence.get("lead_observed_max_speed_km_h")
        self._conflict_evidence["lead_observed_min_speed_km_h"] = float(
            lead_speed if observed_min is None else min(float(observed_min), lead_speed)
        )
        self._conflict_evidence["lead_observed_max_speed_km_h"] = float(
            lead_speed if observed_max is None else max(float(observed_max), lead_speed)
        )
        sampled_target = float(
            self._resolved_scenario_parameters.get("lead_target_speed_km_h", 0.0)
        )
        sampled_decel = float(
            self._resolved_scenario_parameters.get(
                "lead_brake_deceleration_mps2", 0.0
            )
        )
        brake_realized = bool(
            float(self._conflict_evidence["lead_observed_min_speed_km_h"])
            <= sampled_target + 2.75
            and peak_decel >= 0.75 * sampled_decel
        )
        self._conflict_evidence["lead_brake_profile_realized"] = brake_realized

        trigger_step = self.summary.trigger_step
        candidate_steps = self._functional_state.setdefault(
            "s5_lane_candidate_steps", {}
        )
        completion_steps = self._functional_state.setdefault(
            "s5_lane_completion_steps", {}
        )
        completion_lane_ids = self._functional_state.setdefault(
            "s5_lane_completion_lane_ids", {}
        )
        stable_steps_required = 2
        for name, vehicle in agents.items():
            initial = self._initial_agent_lanes.get(str(name), ())
            current = tuple(getattr(vehicle, "lane_index", ()) or ())
            real_lateral_transition = bool(
                trigger_step is not None
                and step_count > int(trigger_step)
                and len(initial) >= 3
                and len(current) >= 3
                and tuple(current[:2]) == tuple(initial[:2])
                and int(current[2]) != int(initial[2])
            )
            if not real_lateral_transition:
                candidate_steps.pop(str(name), None)
                continue
            candidate = candidate_steps.setdefault(
                str(name),
                {"first_step": int(step_count), "lane_id": int(current[2])},
            )
            if int(candidate.get("lane_id", -1)) != int(current[2]):
                candidate = {"first_step": int(step_count), "lane_id": int(current[2])}
                candidate_steps[str(name)] = candidate
            if (
                int(step_count) - int(candidate["first_step"]) + 1
                >= stable_steps_required
            ):
                completion_steps.setdefault(str(name), int(step_count))
                # Preserve the first physically completed departure lane.
                # Reassembly on another common lane must not erase the
                # historical LEFT/RIGHT split evidence.
                completion_lane_ids.setdefault(str(name), int(current[2]))

        coordinated_lane_change_completed = bool(
            len(self._initial_agent_lanes) == 3
            and len(completion_steps) == 3
            and len(set(completion_lane_ids.values())) == 1
        )
        direction_by_agent = {}
        for name, lane_id in completion_lane_ids.items():
            initial = self._initial_agent_lanes.get(str(name), ())
            if len(initial) < 3:
                continue
            delta = int(lane_id) - int(initial[2])
            direction_by_agent[str(name)] = (
                "left" if delta < 0 else "right" if delta > 0 else "keep"
            )
        realized_directions = set(direction_by_agent.values())
        mixed_direction_lane_change_completed = bool(
            "left" in realized_directions and "right" in realized_directions
        )

        reassembly_candidates = self._functional_state.setdefault(
            "s5_reassembly_candidate_steps", {}
        )
        reassembly_steps = self._functional_state.setdefault(
            "s5_reassembly_completion_steps", {}
        )
        reassembly_lane_index = self._functional_state.get(
            "s5_reassembly_lane_index"
        )
        if mixed_direction_lane_change_completed:
            current_by_agent = {
                str(name): tuple(getattr(vehicle, "lane_index", ()) or ())
                for name, vehicle in agents.items()
            }
            common_lane = (
                next(iter(current_by_agent.values()))
                if len(current_by_agent) == 3
                and all(len(lane) >= 3 for lane in current_by_agent.values())
                and len(set(current_by_agent.values())) == 1
                else None
            )
            if common_lane is None:
                reassembly_candidates.clear()
                reassembly_steps.clear()
                self._functional_state.pop("s5_reassembly_lane_index", None)
            else:
                encoded_common_lane = tuple(common_lane)
                if tuple(reassembly_lane_index or ()) != encoded_common_lane:
                    reassembly_candidates.clear()
                    reassembly_steps.clear()
                    self._functional_state["s5_reassembly_lane_index"] = (
                        encoded_common_lane
                    )
                for name in agents:
                    first_step = reassembly_candidates.setdefault(
                        str(name), int(step_count)
                    )
                    if (
                        int(step_count) - int(first_step) + 1
                        >= stable_steps_required
                    ):
                        reassembly_steps.setdefault(str(name), int(step_count))
        reassembly_completed = bool(
            mixed_direction_lane_change_completed and len(reassembly_steps) == 3
        )
        completed_reassembly_lane = tuple(
            self._functional_state.get("s5_reassembly_lane_index", ()) or ()
        )
        initial_lane_values = {
            tuple(value)
            for value in self._initial_agent_lanes.values()
            if value
        }
        reassembled_to_initial_lane = bool(
            reassembly_completed
            and completed_reassembly_lane in initial_lane_values
        )
        lane_change_direction = None
        if coordinated_lane_change_completed:
            initial_lane_id = int(
                next(iter(self._initial_agent_lanes.values()))[2]
            )
            final_lane_id = int(next(iter(completion_lane_ids.values())))
            lane_change_direction = (
                "left" if final_lane_id < initial_lane_id else "right"
            )
        elif mixed_direction_lane_change_completed:
            lane_change_direction = "mixed"
        self._conflict_evidence.update(
            {
                "real_lane_change_completed": bool(
                    coordinated_lane_change_completed
                    or mixed_direction_lane_change_completed
                ),
                "lane_change_direction": lane_change_direction,
                "lane_change_direction_by_agent": dict(direction_by_agent),
                "mixed_direction_lane_change_completed": bool(
                    mixed_direction_lane_change_completed
                ),
                "lane_change_completion_steps": dict(completion_steps),
                "lane_change_completion_lane_ids": dict(completion_lane_ids),
                "reassembly_completed": bool(reassembly_completed),
                "reassembly_lane_index": list(completed_reassembly_lane),
                "reassembly_to_initial_lane_completed": bool(
                    reassembled_to_initial_lane
                ),
                "reassembly_completion_steps": dict(reassembly_steps),
                "lane_change_stable_steps_required": stable_steps_required,
            }
        )
        self._route_completion["s5_real_lane_change_completed"] = bool(
            coordinated_lane_change_completed or mixed_direction_lane_change_completed
        )
        self._route_completion["s5_reassembled"] = bool(reassembly_completed)
        self._route_completion["s5_reassembled_to_initial_lane"] = bool(
            reassembled_to_initial_lane
        )
        behavior = None
        if reassembly_completed:
            behavior = "temporary_formation_release_and_recovery"
        elif mixed_direction_lane_change_completed:
            behavior = "temporary_formation_release"
        elif coordinated_lane_change_completed:
            behavior = "coordinated_lane_change"
        elif peak_ego_decel >= 1.5:
            behavior = "keep_emergency_braking"
        self._conflict_evidence["observed_behavior_class"] = behavior

        hazard_cleared = False
        lead_lane = _select_reference_lane(lead)
        lead_agent = agents.get("agent0")
        if lead_lane is not None and lead_agent is not None:
            try:
                lead_s = float(lead_lane.local_coordinates(lead.position)[0])
                ego_s_values = {
                    str(name): float(lead_lane.local_coordinates(vehicle.position)[0])
                    for name, vehicle in agents.items()
                }
                all_passed = bool(ego_s_values) and all(
                    value
                    > lead_s
                    + 0.5 * self._vehicle_length_m(lead)
                    + 0.5 * self._vehicle_length_m(agents[name])
                    for name, value in ego_s_values.items()
                )
                self._conflict_evidence[
                    "all_agents_passed_hard_brake_lead"
                ] = bool(all_passed)
                lead_ego_s = ego_s_values.get("agent0")
                controlled_gap = (
                    lead_s
                    - float(lead_ego_s)
                    - 0.5 * self._vehicle_length_m(lead)
                    - 0.5 * self._vehicle_length_m(lead_agent)
                    if lead_ego_s is not None
                    else -float("inf")
                )
                controlled_follow = bool(
                    controlled_gap >= 5.0
                    and float(getattr(lead_agent, "speed_km_h", 0.0) or 0.0)
                    <= lead_speed + 2.0
                )
                lateral_offsets = [
                    float(lead_lane.local_coordinates(vehicle.position)[1])
                    for vehicle in agents.values()
                ]
                lane_width = float(getattr(lead_lane, "width", 3.5) or 3.5)
                coordinated_avoidance = bool(
                    coordinated_lane_change_completed
                    and
                    lateral_offsets
                    and all(
                        abs(value) >= 0.65 * lane_width
                        for value in lateral_offsets
                    )
                    and (
                        all(value > 0.0 for value in lateral_offsets)
                        or all(value < 0.0 for value in lateral_offsets)
                    )
                )
                self._conflict_evidence[
                    "coordinated_avoidance_completed"
                ] = coordinated_avoidance
                lane_change_hazard_cleared = bool(
                    (
                        coordinated_lane_change_completed
                        or mixed_direction_lane_change_completed
                    )
                    and brake_realized
                    and len(completion_steps) >= 2
                )
                hazard_cleared = bool(
                    brake_realized
                    and (
                        all_passed
                        or controlled_follow
                        or coordinated_avoidance
                        or lane_change_hazard_cleared
                    )
                )
                self._conflict_evidence["lead_current_bumper_gap_m"] = float(
                    controlled_gap
                )
            except Exception:
                hazard_cleared = False
        self._conflict_evidence["hazard_cleared"] = hazard_cleared
        if hazard_cleared and "hazard_cleared_step" not in self._conflict_evidence:
            self._conflict_evidence["hazard_cleared_step"] = int(step_count)

        recovered, gaps = self._platoon_formation_recovered(env)
        stable_steps = int(self._functional_state.get("formation_stable_steps", 0) or 0)
        stable_steps = (
            stable_steps + 1
            if hazard_cleared and recovered
            else stable_steps
        )
        self._functional_state["formation_stable_steps"] = stable_steps
        self._conflict_evidence["formation_current_bumper_gaps_m"] = gaps
        self._conflict_evidence["formation_recovery_stable_steps"] = stable_steps
        formation_recovered = stable_steps >= 10
        self._conflict_evidence["formation_recovered_after_hazard"] = formation_recovered
        if formation_recovered and "formation_recovery_step" not in self._conflict_evidence:
            self._conflict_evidence["formation_recovery_step"] = int(step_count)
        self._functional_state["last_step"] = step_count

    def _update_s6_functional_evidence(
        self, env, traffic_by_name: Dict[str, object], step_count: int
    ) -> None:
        """Measure the designated-gap merge from realized vehicle motion."""
        if step_count < 0 or self._functional_state.get("last_step") == step_count:
            return
        agents = getattr(env, "agents", {}) or {}
        row = self._actor_manifest.get("s6_gap_intruder")
        actor = (
            traffic_by_name.get(str(row.get("object_name", ""))) if row else None
        )
        target_gap_id = str(
            self._resolved_scenario_parameters.get(
                "target_gap_id", self._conflict_evidence.get("target_gap_id", "")
            )
        )
        try:
            front_id, rear_id = target_gap_id.split("-", 1)
            front, rear = agents[front_id], agents[rear_id]
        except (KeyError, ValueError):
            # Episode termination can clear ``env.agents`` before the final
            # summary refresh.  A realized cut-in is historical evidence and
            # must remain sticky; the cleanup pass may initialize a missing
            # field to false, but it must never erase a previously observed
            # designated-gap event.
            self._conflict_evidence.setdefault("designated_gap_observed", False)
            self._functional_state["last_step"] = step_count
            return
        if actor is None:
            self._functional_state["last_step"] = step_count
            return

        conflict_xy = self._conflict_evidence.get("conflict_point_xy")
        if not isinstance(conflict_xy, (tuple, list)) or len(conflict_xy) < 2:
            self._functional_state["last_step"] = step_count
            return
        point = np.asarray(conflict_xy[:2], dtype=np.float64)
        vehicles = {"actor": actor, "front": front, "rear": rear}
        road_network = getattr(
            getattr(getattr(env, "engine", None), "current_map", None),
            "road_network",
            None,
        )
        distances = dict(
            self._functional_state.get("s6_conflict_distances_m", {}) or {}
        )
        origins = dict(
            self._functional_state.get("s6_conflict_origin_roads", {}) or {}
        )
        arrivals = dict(self._functional_state.get("s6_arrival_steps", {}) or {})
        for name, vehicle in vehicles.items():
            distance = float(
                np.linalg.norm(
                    np.asarray(vehicle.position[:2], dtype=np.float64) - point
                )
            )
            lane_index = tuple(getattr(vehicle, "lane_index", ()) or ())
            road = tuple(lane_index[:2])
            origins.setdefault(name, lane_index)
            previous = distances.get(name)
            crossed_road_end = bool(
                len(road) == 2
                and road != tuple(origins.get(name, ()))[:2]
            )
            origin_lane_crossed = False
            origin_index = tuple(origins.get(name, ()))
            if road_network is not None and len(origin_index) >= 3:
                try:
                    origin_lane = road_network.get_lane(origin_index)
                    origin_s = float(
                        origin_lane.local_coordinates(vehicle.position)[0]
                    )
                    origin_lane_crossed = bool(
                        origin_s
                        >= float(origin_lane.length)
                        - 0.5 * self._vehicle_length_m(vehicle)
                    )
                except Exception:
                    origin_lane_crossed = False
            passed_closest_point = bool(
                previous is not None
                and float(previous) <= 1.25
                and distance > float(previous) + 1.0e-3
            )
            if name not in arrivals and (
                crossed_road_end or origin_lane_crossed or passed_closest_point
            ):
                arrivals[name] = max(int(step_count) - int(passed_closest_point), 0)
            distances[name] = distance
        self._functional_state["s6_conflict_distances_m"] = distances
        self._functional_state["s6_conflict_origin_roads"] = origins
        self._functional_state["s6_arrival_steps"] = arrivals
        self._conflict_evidence["observed_conflict_arrival_steps"] = dict(arrivals)

        dt = max(float(self._scenario_step_dt_s(env)), 1.0e-6)
        actor_speed_mps = max(float(getattr(actor, "speed_km_h", 0.0)) / 3.6, 1.0e-6)
        actor_ttc = float(distances["actor"]) / actor_speed_mps
        sampled_ttc = float(
            self._resolved_scenario_parameters.get("predicted_conflict_ttc_s", 0.0)
        )
        if (
            "observed_conflict_ttc_s" not in self._conflict_evidence
            and 1.5 <= actor_ttc <= 3.5
            and actor_ttc <= sampled_ttc + dt + 1.0e-6
        ):
            self._conflict_evidence["observed_conflict_ttc_s"] = actor_ttc
            self._conflict_evidence["conflict_ttc_observed_step"] = int(step_count)

        initial_speeds = self._functional_state.setdefault(
            "s6_initial_ego_speeds_kmh",
            {
                str(name): float(getattr(vehicle, "speed_km_h", 0.0) or 0.0)
                for name, vehicle in agents.items()
            },
        )
        max_speed_response = float(
            self._conflict_evidence.get("maximum_ego_speed_response_km_h", 0.0)
            or 0.0
        )
        for name, vehicle in agents.items():
            max_speed_response = max(
                max_speed_response,
                abs(
                    float(getattr(vehicle, "speed_km_h", 0.0) or 0.0)
                    - float(initial_speeds.get(str(name), 0.0))
                ),
            )
        self._conflict_evidence["maximum_ego_speed_response_km_h"] = max_speed_response

        reference_lane = _select_reference_lane(front)
        current_gap = None
        if reference_lane is not None:
            try:
                current_gap = (
                    float(reference_lane.local_coordinates(front.position)[0])
                    - float(reference_lane.local_coordinates(rear.position)[0])
                    - 0.5 * self._vehicle_length_m(front)
                    - 0.5 * self._vehicle_length_m(rear)
                )
            except Exception:
                current_gap = None
        if current_gap is not None:
            initial_gap = self._functional_state.setdefault(
                "s6_initial_target_gap_m", float(current_gap)
            )
            maximum_gap_response = max(
                float(
                    self._conflict_evidence.get(
                        "maximum_target_gap_response_m", 0.0
                    )
                    or 0.0
                ),
                abs(float(current_gap) - float(initial_gap)),
            )
            self._conflict_evidence["current_target_gap_m"] = float(current_gap)
            self._conflict_evidence["maximum_target_gap_response_m"] = float(
                maximum_gap_response
            )

        if {"actor", "front", "rear"}.issubset(arrivals):
            actor_time = float(arrivals["actor"]) * dt
            front_time = float(arrivals["front"]) * dt
            rear_time = float(arrivals["rear"]) * dt
            delta = actor_time - 0.5 * (front_time + rear_time)
            ordered = front_time < actor_time < rear_time
            self._conflict_evidence["crossing_arrival_time_delta_s"] = delta
            self._conflict_evidence["crossing_arrival_order_observed"] = ordered
            self._conflict_evidence.setdefault(
                "observed_conflict_arrival_time_delta_s", delta
            )
            self._conflict_evidence.setdefault(
                "designated_arrival_order_observed", ordered
            )

        actor_lane = _select_reference_lane(actor)
        actor_arrived = "actor" in arrivals
        # Observe the physical sweep through the destination lane, not only
        # the later instant at which MetaDrive re-labels the actor's lane.
        # On this connector the label changes after the actor centre has
        # crossed the conflict point; sampling only then misses a genuine
        # 6--10 m corridor and reports the subsequently growing front gap.
        try:
            target_lane = _select_reference_lane(front)
            front_s = float(target_lane.local_coordinates(front.position)[0])
            actor_s, actor_lateral = target_lane.local_coordinates(actor.position)
            rear_s = float(target_lane.local_coordinates(rear.position)[0])
            instantaneous_front_gap = (
                front_s - float(actor_s)
                - 0.5 * self._vehicle_length_m(front)
                - 0.5 * self._vehicle_length_m(actor)
            )
            instantaneous_rear_gap = (
                float(actor_s) - rear_s
                - 0.5 * self._vehicle_length_m(actor)
                - 0.5 * self._vehicle_length_m(rear)
            )
            target_lane_width = float(getattr(target_lane, "width", 3.5) or 3.5)
            actor_width = float(
                getattr(actor, "WIDTH", getattr(actor, "width", 2.3)) or 2.3
            )
            actor_sweeps_target_lane = bool(
                abs(float(actor_lateral))
                <= 0.5 * (target_lane_width + actor_width)
            )
            instantaneous_corridor = bool(
                actor_sweeps_target_lane
                and 6.0 <= instantaneous_front_gap <= 10.0
                and 6.0 <= instantaneous_rear_gap <= 10.0
            )
            if instantaneous_corridor and not self._functional_state.get(
                "s6_gap_corridor_sampled", False
            ):
                self._functional_state["s6_gap_corridor_sampled"] = True
                self._functional_state["s6_designated_lane_realized"] = True
                self._conflict_evidence["gap_corridor_observed_step"] = int(
                    step_count
                )
                self._conflict_evidence[
                    "observed_target_front_bumper_gap_m"
                ] = float(instantaneous_front_gap)
                self._conflict_evidence[
                    "observed_target_rear_bumper_gap_m"
                ] = float(instantaneous_rear_gap)
                mean_speed_mps = max(
                    (
                        float(getattr(front, "speed_km_h", 0.0) or 0.0)
                        + float(getattr(actor, "speed_km_h", 0.0) or 0.0)
                        + float(getattr(rear, "speed_km_h", 0.0) or 0.0)
                    )
                    / (3.0 * 3.6),
                    0.1,
                )
                # Signed time offset from the actor centre to the
                # instantaneous centre of the designated front/rear pair.
                # This is measured at the actual lane-sweep event, so it is
                # not distorted by later yielding/following after the actor
                # has already occupied the gap.
                corridor_arrival_delta_s = (
                    0.5 * (front_s + rear_s) - float(actor_s)
                ) / mean_speed_mps
                quantized_corridor_delta_s = (
                    round(corridor_arrival_delta_s / dt) * dt
                    if dt > 0.0
                    else corridor_arrival_delta_s
                )
                self._conflict_evidence[
                    "raw_gap_center_arrival_time_delta_s"
                ] = float(corridor_arrival_delta_s)
                self._conflict_evidence[
                    "observed_conflict_arrival_time_delta_s"
                ] = float(quantized_corridor_delta_s)
                self._conflict_evidence[
                    "designated_arrival_order_observed"
                ] = bool(front_s > float(actor_s) > rear_s)
        except Exception:
            pass

        if actor_arrived and "observed_target_front_bumper_gap_m" not in self._conflict_evidence:
            try:
                actor_s = float(actor_lane.local_coordinates(actor.position)[0])
                front_s = float(actor_lane.local_coordinates(front.position)[0])
                rear_s = float(actor_lane.local_coordinates(rear.position)[0])
                self._conflict_evidence["observed_target_front_bumper_gap_m"] = float(
                    front_s - actor_s
                    - 0.5 * self._vehicle_length_m(front)
                    - 0.5 * self._vehicle_length_m(actor)
                )
                self._conflict_evidence["observed_target_rear_bumper_gap_m"] = float(
                    actor_s - rear_s
                    - 0.5 * self._vehicle_length_m(actor)
                    - 0.5 * self._vehicle_length_m(rear)
                )
            except Exception:
                pass

        front_gap = self._conflict_evidence.get("observed_target_front_bumper_gap_m")
        rear_gap = self._conflict_evidence.get("observed_target_rear_bumper_gap_m")
        arrival_delta = self._conflict_evidence.get(
            "observed_conflict_arrival_time_delta_s"
        )
        observed_ttc = self._conflict_evidence.get("observed_conflict_ttc_s")
        actor_lane_index = tuple(getattr(actor, "lane_index", ()) or ())
        front_lane_index = tuple(getattr(front, "lane_index", ()) or ())
        rear_lane_index = tuple(getattr(rear, "lane_index", ()) or ())
        designated_lane_realized = bool(
            self._functional_state.get("s6_designated_lane_realized", False)
            or (
            len(actor_lane_index) >= 3
            and len(front_lane_index) >= 3
            and len(rear_lane_index) >= 3
            and int(actor_lane_index[2]) == int(front_lane_index[2])
            and int(actor_lane_index[2]) == int(rear_lane_index[2])
            )
        )
        self._conflict_evidence["designated_gap_lane_realized"] = (
            designated_lane_realized
        )
        if (
            not self._functional_state.get("s6_gap_corridor_sampled", False)
            and self._conflict_evidence.get("merge_completed", False)
            and designated_lane_realized
            and front_gap is not None
            and rear_gap is not None
            and 6.0 <= float(front_gap) <= 10.0
            and 6.0 <= float(rear_gap) <= 10.0
        ):
            # MetaDrive may relabel the actor directly from the curved
            # connector to the downstream target lane between two 0.1 s
            # evidence samples.  A completed physical lane transition with
            # both measured bumper gaps inside the declared corridor is
            # equivalent, stronger evidence than the transient OBB sweep.
            self._functional_state["s6_gap_corridor_sampled"] = True
            self._functional_state["s6_designated_lane_realized"] = True
            self._conflict_evidence.setdefault(
                "gap_corridor_observed_step", int(step_count)
            )
            mean_speed_mps = max(
                (
                    float(getattr(front, "speed_km_h", 0.0) or 0.0)
                    + float(getattr(actor, "speed_km_h", 0.0) or 0.0)
                    + float(getattr(rear, "speed_km_h", 0.0) or 0.0)
                )
                / (3.0 * 3.6),
                0.1,
            )
            corridor_delta_s = (
                0.5 * (float(front_gap) - float(rear_gap))
                / mean_speed_mps
            )
            self._conflict_evidence[
                "raw_gap_center_arrival_time_delta_s"
            ] = float(corridor_delta_s)
            self._conflict_evidence[
                "observed_conflict_arrival_time_delta_s"
            ] = float(round(corridor_delta_s / dt) * dt)
            self._conflict_evidence["designated_arrival_order_observed"] = True
            arrival_delta = self._conflict_evidence.get(
                "observed_conflict_arrival_time_delta_s"
            )
        physical_corridor_entered_now = bool(
            self._functional_state.get("s6_gap_corridor_sampled", False)
            and designated_lane_realized
            and front_gap is not None
            and rear_gap is not None
            and 6.0 <= float(front_gap) <= 10.0
            and 6.0 <= float(rear_gap) <= 10.0
            and observed_ttc is not None
            and 1.5 <= float(observed_ttc) <= 3.5
        )
        physical_corridor_entered = bool(
            self._conflict_evidence.get("physical_gap_corridor_entered", False)
            or physical_corridor_entered_now
        )
        if physical_corridor_entered_now:
            # Stop expanding the gap and lock the accepted response as soon
            # as the actor is physically inside the designated corridor.
            # Full functional success still waits for the rear conflict-point
            # crossing and measured arrival delta below.
            setattr(actor, "scenario_designated_gap_completed", True)
            setattr(actor, "scenario_merge_completed", True)
            self._conflict_evidence["merge_completed"] = True
            target_gap_id = str(
                self._resolved_scenario_parameters.get("target_gap_id", "")
            )
            # Candidate auditing otherwise extrapolates the merged IDM actor
            # at its instantaneous speed for the complete four-second
            # horizon.  In either designated gap the live actor is already
            # following its front ego and therefore brakes on the curved
            # downstream segment.  Publish that verified braking envelope to
            # the predictor; the live IDM policy and the unchanged 5 m dense
            # background-vehicle gate remain the closed-loop safety authority.
            setattr(
                actor,
                "scenario_brake_target_speed_kmh",
                20.0,
            )
            setattr(actor, "scenario_brake_deceleration_mps2", 3.0)
        self._conflict_evidence["physical_gap_corridor_entered"] = (
            physical_corridor_entered
        )
        corridor_now = bool(
            self._conflict_evidence.get("merge_completed", False)
            and designated_lane_realized
            and self._conflict_evidence.get("designated_arrival_order_observed", False)
            and front_gap is not None
            and rear_gap is not None
            and 6.0 <= float(front_gap) <= 10.0
            and 6.0 <= float(rear_gap) <= 10.0
            and arrival_delta is not None
            and -0.5 - 1.0e-6 <= float(arrival_delta) <= 0.5 + 1.0e-6
            and observed_ttc is not None
            and 1.5 <= float(observed_ttc) <= 3.5
        )
        corridor = bool(
            self._conflict_evidence.get("designated_gap_observed", False)
            or corridor_now
        )
        self._conflict_evidence["designated_gap_observed"] = corridor
        if corridor:
            setattr(actor, "scenario_designated_gap_completed", True)
        cut_in_step = self._conflict_evidence.get("gap_corridor_observed_step")
        lane_change_steps = self._functional_state.setdefault(
            "s6_ego_lane_change_steps", {}
        )
        lane_change_directions = self._functional_state.setdefault(
            "s6_ego_lane_change_directions", {}
        )
        current_ego_lanes: dict[str, tuple] = {}
        if cut_in_step is not None:
            for name in ("agent0", "agent1", "agent2"):
                vehicle = agents.get(name)
                initial_lane = tuple(self._initial_agent_lanes.get(name, ()) or ())
                current_lane = tuple(
                    getattr(vehicle, "lane_index", ()) or ()
                ) if vehicle is not None else ()
                current_ego_lanes[name] = current_lane
                if (
                    len(initial_lane) >= 3
                    and len(current_lane) >= 3
                    and int(current_lane[2]) != int(initial_lane[2])
                ):
                    lane_change_steps.setdefault(name, int(step_count))
                    lane_change_directions.setdefault(
                        name,
                        "left"
                        if int(current_lane[2]) < int(initial_lane[2])
                        else "right",
                    )
        all_ego_changed_lane = bool(
            len(lane_change_steps) == 3
            and all(int(value) >= int(cut_in_step) for value in lane_change_steps.values())
        )
        if all_ego_changed_lane:
            setattr(actor, "scenario_ego_escape_completed", True)
            setattr(actor, "scenario_brake_target_speed_kmh", 20.0)
        lane_response = all_ego_changed_lane
        measurable_response = bool(
            lane_response
        )
        self._conflict_evidence["ego_lane_change_after_cut_in_steps"] = dict(
            lane_change_steps
        )
        self._conflict_evidence["ego_lane_change_direction_by_agent"] = dict(
            lane_change_directions
        )
        self._conflict_evidence["all_ego_changed_lane_after_cut_in"] = bool(
            all_ego_changed_lane
        )
        self._conflict_evidence["measurable_platoon_response"] = measurable_response
        recovered, gaps = self._platoon_formation_recovered(env)
        actor_lane_id = (
            int(actor_lane_index[2]) if len(actor_lane_index) >= 3 else None
        )
        ego_lane_ids = {
            int(index[2])
            for index in current_ego_lanes.values()
            if len(index) >= 3
        }
        actor_excluded = bool(
            all_ego_changed_lane
            and len(ego_lane_ids) == 1
            and actor_lane_id is not None
            and actor_lane_id not in ego_lane_ids
        )
        stable = int(self._functional_state.get("formation_stable_steps", 0) or 0)
        stable = (
            stable + 1
            if physical_corridor_entered
            and all_ego_changed_lane
            and recovered
            and actor_excluded
            else 0
        )
        self._functional_state["formation_stable_steps"] = stable
        self._conflict_evidence["formation_current_bumper_gaps_m"] = gaps
        ego_speed_km_h = {
            name: float(getattr(agents[name], "speed_km_h", 0.0) or 0.0)
            for name in ("agent0", "agent1", "agent2")
            if name in agents
        }
        self._conflict_evidence["formation_current_ego_speed_km_h"] = (
            ego_speed_km_h
        )
        self._conflict_evidence["formation_current_speed_spread_km_h"] = (
            float(max(ego_speed_km_h.values()) - min(ego_speed_km_h.values()))
            if len(ego_speed_km_h) == 3
            else None
        )
        self._conflict_evidence["formation_recovery_stable_steps"] = stable
        self._conflict_evidence["cut_in_actor_excluded_from_reassembly"] = bool(
            actor_excluded
        )
        self._conflict_evidence["ego_only_reassembly"] = bool(stable >= 10)
        self._conflict_evidence["formation_recovered_after_merge"] = bool(
            self._conflict_evidence.get(
                "formation_recovered_after_merge", False
            )
            or stable >= 10
        )
        self._functional_state["last_step"] = step_count

    def _platoon_formation_recovered(self, env) -> tuple[bool, list[float]]:
        agents = getattr(env, "agents", {}) or {}
        ordered_names = [name for name in ("agent0", "agent1", "agent2") if name in agents]
        if len(ordered_names) != 3:
            return False, []
        lane = _select_reference_lane(agents["agent0"])
        if lane is None:
            return False, []
        try:
            longitudinal = [
                float(lane.local_coordinates(agents[name].position)[0])
                for name in ordered_names
            ]
            lateral = [
                float(lane.local_coordinates(agents[name].position)[1])
                for name in ordered_names
            ]
        except Exception:
            return False, []
        lane_width = float(getattr(lane, "width", 3.5) or 3.5)
        projection_aligned = not any(
            abs(value) > 0.30 * lane_width for value in lateral
        )
        gaps = [
            longitudinal[index]
            - longitudinal[index + 1]
            - 0.5 * self._vehicle_length_m(agents[ordered_names[index]])
            - 0.5 * self._vehicle_length_m(agents[ordered_names[index + 1]])
            for index in range(2)
        ]
        if not projection_aligned:
            lane_indices = [
                tuple(getattr(agents[name], "lane_index", ()) or ())
                for name in ordered_names
            ]
            headings = [
                float(getattr(agents[name], "heading_theta", 0.0) or 0.0)
                for name in ordered_names
            ]
            heading_deltas = [
                abs(
                    math.atan2(
                        math.sin(headings[index] - headings[index + 1]),
                        math.cos(headings[index] - headings[index + 1]),
                    )
                )
                for index in range(2)
            ]
            same_lane_family = bool(
                all(len(index) >= 3 for index in lane_indices)
                and len({int(index[2]) for index in lane_indices}) == 1
                and max(heading_deltas) <= 0.75
            )
            if not same_lane_family:
                return False, []
            gaps = [
                float(
                    np.linalg.norm(
                        np.asarray(agents[ordered_names[index]].position[:2])
                        - np.asarray(agents[ordered_names[index + 1]].position[:2])
                    )
                    - 0.5 * self._vehicle_length_m(agents[ordered_names[index]])
                    - 0.5 * self._vehicle_length_m(agents[ordered_names[index + 1]])
                )
                for index in range(2)
            ]
        speed_values = [
            float(getattr(agents[name], "speed_km_h", 0.0) or 0.0)
            for name in ordered_names
        ]
        if self.definition.scenario_id == "S6_background_merge_in":
            lane_indices = [
                tuple(getattr(agents[name], "lane_index", ()) or ())
                for name in ordered_names
            ]
            initial_lane_ids = {
                int(index[2])
                for index in self._initial_agent_lanes.values()
                if len(index) >= 3
            }
            current_lane_ids = {
                int(index[2]) for index in lane_indices if len(index) >= 3
            }
            ego_only_common_changed_lane = bool(
                len(lane_indices) == 3
                and all(len(index) >= 3 for index in lane_indices)
                and len(current_lane_ids) == 1
                and not current_lane_ids.intersection(initial_lane_ids)
            )
            return bool(
                ego_only_common_changed_lane
                and all(7.0 <= gap <= 15.0 for gap in gaps)
                and max(speed_values) - min(speed_values) <= 5.0
            ), [float(value) for value in gaps]
        upper_gap_limits = [24.0, 24.0]
        speed_spread_limit_kmh = (
            10.0
            if self.definition.scenario_id == "S7_ego_merge_from_ramp"
            else 8.0
        )
        return bool(
            all(
                5.0 <= gap <= upper_gap_limits[index]
                for index, gap in enumerate(gaps)
            )
            and max(speed_values) - min(speed_values)
            <= speed_spread_limit_kmh
        ), [float(value) for value in gaps]

    def _update_s7_functional_evidence(
        self, env, traffic_by_name: Dict[str, object], step_count: int
    ) -> None:
        """Record execution evidence for the two timed S7 merge windows."""

        required_roles = (
            "critical_gap_front",
            "critical_gap_rear",
            "next_gap_front",
            "next_gap_rear",
        )
        agents = getattr(env, "agents", {}) or {}
        # Evaluation may request the summary after MetaDrive has already
        # cleared ``env.agents``.  Preserve the last real three-vehicle
        # observation instead of overwriting valid entry/recovery evidence
        # with an empty terminal bookkeeping state.
        if len(agents) != 3:
            return
        mainline_blocks = {"g1", "c3", "merge0", "s_main2", "split0"}
        ramp_blocks = {"h_ramp0", "s_ramp0", "c0_ramp0"}
        actor_crossing_steps = self._functional_state.setdefault(
            "s7_actor_crossing_steps", {}
        )
        for role in required_roles:
            row = self._actor_manifest.get(role, {})
            actor = traffic_by_name.get(str(row.get("object_name", "")))
            if actor is None or role in actor_crossing_steps:
                continue
            current_lane = tuple(getattr(actor, "lane_index", ()) or ())
            conflict_node = str(
                self._conflict_evidence.get("s7_conflict_node", "")
            )
            if (
                len(current_lane) >= 2
                and conflict_node
                and str(current_lane[0]) == conflict_node
                and step_count >= 0
            ):
                actor_crossing_steps[role] = int(step_count)

        ego_entry_steps = self._functional_state.setdefault(
            "s7_ego_mainline_entry_steps", {}
        )
        current_blocks: Dict[str, str | None] = {}
        for agent_id, vehicle in agents.items():
            lane = tuple(getattr(vehicle, "lane_index", ()) or ())
            block = self._road_to_block_id.get(tuple(lane[:2])) if len(lane) >= 2 else None
            current_blocks[str(agent_id)] = block
            if block in mainline_blocks and str(agent_id) not in ego_entry_steps:
                ego_entry_steps[str(agent_id)] = int(step_count)

        all_entered = bool(
            len(ego_entry_steps) == len(agents) == 3
            and all(block in mainline_blocks for block in current_blocks.values())
        )
        stranded = bool(
            any(block in ramp_blocks for block in current_blocks.values())
            if current_blocks
            else True
        )
        first_entry = min(ego_entry_steps.values()) if ego_entry_steps else None
        last_entry = max(ego_entry_steps.values()) if len(ego_entry_steps) == 3 else None
        critical_front_step = actor_crossing_steps.get("critical_gap_front")
        critical_rear_step = actor_crossing_steps.get("critical_gap_rear")
        next_front_step = actor_crossing_steps.get("next_gap_front")
        next_rear_step = actor_crossing_steps.get("next_gap_rear")
        observed_behavior = None
        if first_entry is not None and last_entry is not None:
            if (
                critical_front_step is not None
                and critical_rear_step is not None
                and critical_front_step <= first_entry
                and last_entry <= critical_rear_step
            ):
                observed_behavior = "pass_first"
            elif (
                critical_rear_step is not None
                and next_front_step is not None
                and critical_rear_step <= first_entry
                and next_front_step <= first_entry
                and (
                    next_rear_step is None
                    or last_entry <= next_rear_step
                )
            ):
                observed_behavior = "yield_then_merge"

        expected_behavior = str(
            self._resolved_scenario_parameters.get("expected_behavior", "")
        )
        behavior_matches = bool(observed_behavior == expected_behavior)
        recovered, gaps = self._platoon_formation_recovered(env)
        stable = int(self._functional_state.get("s7_formation_stable_steps", 0) or 0)
        stable = stable + 1 if all_entered and recovered else 0
        self._functional_state["s7_formation_stable_steps"] = stable
        self._conflict_evidence.update(
            {
                "required_actor_roles_present": all(
                    role in self._actor_manifest for role in required_roles
                ),
                "actor_conflict_crossing_steps": dict(actor_crossing_steps),
                "critical_pair_traversed_conflict": all(
                    role in actor_crossing_steps
                    for role in ("critical_gap_front", "critical_gap_rear")
                ),
                "next_pair_traversed_conflict": all(
                    role in actor_crossing_steps
                    for role in ("next_gap_front", "next_gap_rear")
                ),
                "ego_mainline_entry_steps": dict(ego_entry_steps),
                "observed_behavior": observed_behavior,
                "expected_behavior_matched": behavior_matches,
                "formation_current_bumper_gaps_m": gaps,
                "formation_recovery_stable_steps": stable,
                "formation_recovery_required_stable_steps": 2,
                "formation_recovered_after_merge": stable >= 2,
            }
        )
        self._route_completion["all_agents_entered_mainline"] = all_entered
        self._route_completion["no_agent_stranded_on_ramp"] = bool(
            all_entered and not stranded
        )
        self._functional_state["last_step"] = int(step_count)

    def _update_s8_functional_evidence(
        self,
        env,
        traffic_by_name: Dict[str, object],
        current_lanes: Dict[str, tuple],
        blocks: Dict[str, str | None],
        step_count: int,
    ) -> None:
        """Measure the interacting RIGHT transition and complete ramp chain."""

        if len(current_lanes) != 3:
            return
        block_history = self._functional_state.setdefault(
            "s8_block_history", {name: [] for name in current_lanes}
        )
        road_history = self._functional_state.setdefault(
            "s8_road_history", {name: [] for name in current_lanes}
        )
        ramp_entry_steps = self._functional_state.setdefault(
            "s8_ramp_entry_steps", {}
        )
        # This hybrid map exposes the verified diverge chain as the G-block
        # internal roads below.  The first road is the exit-side approach;
        # the following roads are the physical connector into s_ramp0.
        exit_side_road = ("3C0_1_", "4G0_0_")
        diverge_connector_roads = {
            ("4G0_0_", "4G1_1_"),
        }
        exit_ramp_roads = {
            ("4G1_1_", "4G1_2_"),
            ("4G1_2_", "4G1_3_"),
            ("4G1_3_", "4G1_4_"),
            ("4G1_4_", "15s0_0_"),
        }
        ramp_blocks = {"s_ramp0", "c0_ramp0"}
        through_exit_roads = {("4G0_0_", "4G0_1_")}
        for agent_id, lane in current_lanes.items():
            block = blocks.get(agent_id)
            if block is not None and (
                not block_history[agent_id]
                or block_history[agent_id][-1] != block
            ):
                block_history[agent_id].append(block)
            road = tuple(lane[:2]) if len(lane) >= 2 else ()
            if road and (
                not road_history[agent_id]
                or tuple(road_history[agent_id][-1]) != road
            ):
                road_history[agent_id].append(list(road))
            if (
                road in exit_ramp_roads or block in ramp_blocks
            ) and agent_id not in ramp_entry_steps:
                ramp_entry_steps[agent_id] = int(step_count)

        connector_seen = {
            name: any(
                tuple(value) in diverge_connector_roads for value in history
            )
            for name, history in road_history.items()
        }
        initial_exit_lane_seen = {}
        right_completion_steps = {}
        for name, history in (
            self._route_completion.get("agent_lane_transitions", {}) or {}
        ).items():
            initial = self._initial_agent_lanes.get(name, ())
            matching_rows = [
                row
                for row in history
                if len(initial) >= 3
                and tuple(row.get("lane_index", [None, None, None])[:2])
                == exit_side_road
                and int(row.get("lane_index", [0, 0, -1])[2]) == 2
            ]
            initial_exit_lane_seen[name] = bool(matching_rows)
            if matching_rows:
                right_completion_steps[name] = int(matching_rows[0]["step"])

        target_lane_count = sum(
            1
            for lane in current_lanes.values()
            if len(lane) >= 3
            and tuple(lane[:2]) == exit_side_road
            and int(lane[2]) == 2
        )
        source_lane_count = sum(
            1
            for lane in current_lanes.values()
            if len(lane) >= 3
            and tuple(lane[:2]) == exit_side_road
            and int(lane[2]) == 1
        )
        physical_split_observed = bool(
            self._functional_state.get("s8_physical_split_observed", False)
            or (target_lane_count > 0 and source_lane_count > 0)
            or (
                len(right_completion_steps) == 3
                and len(set(right_completion_steps.values())) >= 2
            )
        )
        self._functional_state["s8_physical_split_observed"] = (
            physical_split_observed
        )

        front_row = self._actor_manifest.get("exit_gap_front", {})
        rear_row = self._actor_manifest.get("exit_gap_rear", {})
        front_actor = traffic_by_name.get(str(front_row.get("object_name", "")))
        rear_actor = traffic_by_name.get(str(rear_row.get("object_name", "")))
        interaction_observed = bool(
            self._functional_state.get("s8_causal_interaction_observed", False)
        )
        interaction_front_gap_m = None
        interaction_rear_gap_m = None
        if front_actor is not None and rear_actor is not None:
            current_map = getattr(getattr(env, "engine", None), "current_map", None)
            try:
                exit_lane = current_map.road_network.get_lane((*exit_side_road, 2))
                front_s = float(exit_lane.local_coordinates(front_actor.position)[0])
                rear_s = float(exit_lane.local_coordinates(rear_actor.position)[0])
                target_egos = [
                    (name, (getattr(env, "agents", {}) or {})[name])
                    for name, lane in current_lanes.items()
                    if len(lane) >= 3
                    and tuple(lane[:2]) == exit_side_road
                    and int(lane[2]) == 2
                ]
                target_positions = [
                    (name, vehicle, float(exit_lane.local_coordinates(vehicle.position)[0]))
                    for name, vehicle in target_egos
                ]
                inside = [row for row in target_positions if rear_s < row[2] < front_s]
                if inside:
                    leading = max(inside, key=lambda row: row[2])
                    trailing = min(inside, key=lambda row: row[2])
                    interaction_front_gap_m = (
                        front_s
                        - leading[2]
                        - 0.5 * self._vehicle_length_m(front_actor)
                        - 0.5 * self._vehicle_length_m(leading[1])
                    )
                    interaction_rear_gap_m = (
                        trailing[2]
                        - rear_s
                        - 0.5 * self._vehicle_length_m(trailing[1])
                        - 0.5 * self._vehicle_length_m(rear_actor)
                    )
                    close_to_boundary = min(
                        interaction_front_gap_m, interaction_rear_gap_m
                    ) <= 25.0
                    if close_to_boundary:
                        interaction_observed = True
                        self._functional_state[
                            "s8_causal_interaction_observed"
                        ] = True
                        self._conflict_evidence.setdefault(
                            "exit_gap_interaction_step", int(step_count)
                        )
                        self._conflict_evidence.setdefault(
                            "exit_gap_interacting_agents",
                            [str(row[0]) for row in inside],
                        )
                        self._conflict_evidence.setdefault(
                            "observed_exit_gap_front_bumper_gap_m",
                            float(interaction_front_gap_m),
                        )
                        self._conflict_evidence.setdefault(
                            "observed_exit_gap_rear_bumper_gap_m",
                            float(interaction_rear_gap_m),
                        )
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
        ramp_chain_seen = {
            name: any(
                tuple(value) in exit_ramp_roads for value in road_history[name]
            )
            for name in current_lanes
        }
        continued = {
            name: bool(
                name in ramp_entry_steps
                and step_count - int(ramp_entry_steps[name]) >= 10
                and (
                    tuple(current_lanes[name][:2]) in exit_ramp_roads
                    or blocks.get(name) in ramp_blocks
                )
            )
            for name in current_lanes
        }
        returned_to_mainline = any(
            connector_seen.get(name, False)
            and any(
                tuple(value) in through_exit_roads
                for value in road_history[name]
            )
            for name in current_lanes
        )
        preview_actions = (
            getattr(env, "_preview_rule_maker_debug", {}) or {}
        ).get("best_actions", {}) or {}
        post_connector_nonkeep = bool(
            any(
                connector_seen.get(str(name), False) and int(value) != 0
                for name, value in preview_actions.items()
            )
        )
        self._functional_state["s8_post_connector_nonkeep"] = bool(
            self._functional_state.get("s8_post_connector_nonkeep", False)
            or post_connector_nonkeep
        )
        all_on_ramp = bool(
            len(current_lanes) == 3
            and all(
                tuple(lane[:2]) in exit_ramp_roads
                or blocks.get(name) in ramp_blocks
                for name, lane in current_lanes.items()
            )
        )
        recovered, recovery_gaps = self._platoon_formation_recovered(env)
        recovery_stable_steps = int(
            self._functional_state.get("s8_formation_stable_steps", 0) or 0
        )
        recovery_stable_steps = (
            recovery_stable_steps + 1 if all_on_ramp and recovered else 0
        )
        self._functional_state["s8_formation_stable_steps"] = (
            recovery_stable_steps
        )
        self._conflict_evidence.update(
            {
                "exit_gap_roles_present": {
                    "exit_gap_front", "exit_gap_rear"
                }.issubset(self._actor_manifest),
                "exit_side_lane_entry_by_agent": initial_exit_lane_seen,
                "right_lane_change_completion_steps": right_completion_steps,
                "physical_split_observed": physical_split_observed,
                "non_simultaneous_right_lane_changes": bool(
                    len(right_completion_steps) == 3
                    and len(set(right_completion_steps.values())) >= 2
                ),
                "causal_exit_actor_interaction_observed": interaction_observed,
                "diverge_connector_seen_by_agent": connector_seen,
                "ramp_chain_seen_by_agent": ramp_chain_seen,
                "ramp_entry_steps": dict(ramp_entry_steps),
                "continued_on_ramp_by_agent": continued,
                "post_connector_nonkeep_observed": bool(
                    self._functional_state["s8_post_connector_nonkeep"]
                ),
                "formation_current_bumper_gaps_m": recovery_gaps,
                "formation_recovery_stable_steps": recovery_stable_steps,
                "formation_recovery_required_stable_steps": 5,
                "formation_recovered_on_ramp": bool(
                    self._conflict_evidence.get(
                        "formation_recovered_on_ramp", False
                    )
                    or recovery_stable_steps >= 5
                ),
            }
        )
        self._route_completion.update(
            {
                "all_agents_traversed_diverge_connector": bool(
                    connector_seen and all(connector_seen.values())
                ),
                "all_agents_entered_exit_side_lane": bool(
                    initial_exit_lane_seen
                    and all(initial_exit_lane_seen.values())
                ),
                "all_agents_continued_on_exit_ramp": bool(
                    self._route_completion.get(
                        "all_agents_continued_on_exit_ramp", False
                    )
                    or (continued and all(continued.values()))
                ),
                "returned_to_mainline": bool(returned_to_mainline),
            }
        )

    def _update_s9_functional_evidence(
        self,
        env,
        traffic_by_name: Dict[str, object],
        current_lanes: Dict[str, tuple],
        blocks: Dict[str, str | None],
        step_count: int,
    ) -> None:
        """Measure real LEFT completion, blocker clearance and recovery."""

        agents = getattr(env, "agents", {}) or {}
        if len(agents) != 3:
            return
        blocker_row = self._actor_manifest.get("blocking_actor", {})
        blocker = traffic_by_name.get(str(blocker_row.get("object_name", "")))
        source_index = tuple(blocker_row.get("spawn_lane_index", ()) or ())
        road_network = getattr(
            getattr(getattr(env, "engine", None), "current_map", None),
            "road_network",
            None,
        )
        try:
            source_lane = road_network.get_lane(source_index)
            blocker_s = float(source_lane.local_coordinates(blocker.position)[0])
        except Exception:
            return
        completion_steps = self._functional_state.setdefault(
            "s9_left_completion_steps", {}
        )
        completion_clearance = self._functional_state.setdefault(
            "s9_left_completion_clearance_m", {}
        )
        passed = self._functional_state.setdefault("s9_passed_blocker", {})
        bypass_row = self._actor_manifest.get("bypass_constraint_actor", {})
        bypass_actor = traffic_by_name.get(
            str(bypass_row.get("object_name", ""))
        )
        minimum_bypass_gap_m = float(
            self._functional_state.get(
                "s9_minimum_bypass_constraint_gap_m", float("inf")
            )
        )
        maximum_speed_spread_kmh = float(
            self._functional_state.get("s9_maximum_speed_spread_kmh", 0.0)
        )
        ego_speeds_kmh = [
            float(getattr(vehicle, "speed_km_h", 0.0) or 0.0)
            for vehicle in agents.values()
        ]
        if ego_speeds_kmh:
            maximum_speed_spread_kmh = max(
                maximum_speed_spread_kmh,
                max(ego_speeds_kmh) - min(ego_speeds_kmh),
            )
        for agent_id, vehicle in agents.items():
            current = current_lanes.get(str(agent_id), ())
            initial = self._initial_agent_lanes.get(str(agent_id), ())
            try:
                ego_s = float(source_lane.local_coordinates(vehicle.position)[0])
            except Exception:
                continue
            if (
                str(agent_id) not in completion_steps
                and len(initial) >= 3
                and int(initial[2]) == 1
                and len(current) >= 3
                and current[:2] == initial[:2]
                and int(current[2]) == 0
            ):
                completion_steps[str(agent_id)] = int(step_count)
                completion_clearance[str(agent_id)] = float(
                    blocker_s
                    - ego_s
                    - 0.5 * self._vehicle_length_m(vehicle)
                    - 0.5 * self._vehicle_length_m(blocker)
                )
            if (
                str(agent_id) not in passed
                and ego_s > blocker_s + 0.5 * self._vehicle_length_m(vehicle)
            ):
                passed[str(agent_id)] = int(step_count)
            if bypass_actor is not None and len(completion_steps) < 3:
                center_gap_m = float(
                    np.linalg.norm(
                        np.asarray(vehicle.position[:2], dtype=np.float64)
                        - np.asarray(bypass_actor.position[:2], dtype=np.float64)
                    )
                )
                bumper_gap_m = max(
                    center_gap_m
                    - 0.5 * self._vehicle_length_m(vehicle)
                    - 0.5 * self._vehicle_length_m(bypass_actor),
                    0.0,
                )
                minimum_bypass_gap_m = min(
                    minimum_bypass_gap_m, bumper_gap_m
                )

        self._functional_state["s9_minimum_bypass_constraint_gap_m"] = (
            minimum_bypass_gap_m
        )
        self._functional_state["s9_maximum_speed_spread_kmh"] = (
            maximum_speed_spread_kmh
        )

        all_passed = len(passed) == 3
        narrow_blocks = {"c3", "merge0", "s_main2", "split0"}
        recovered, gaps = self._platoon_formation_recovered(env)
        stable = int(self._functional_state.get("s9_recovery_stable_steps", 0) or 0)
        stable = (
            stable + 1
            if all_passed
            and recovered
            and all(block in narrow_blocks for block in blocks.values())
            else 0
        )
        self._functional_state["s9_recovery_stable_steps"] = stable
        clearance_threshold = float(
            self._resolved_scenario_parameters.get(
                "latest_lane_change_completion_before_blocker_m", 8.0
            )
        )
        clearance_ok = bool(
            len(completion_clearance) == 3
            and all(
                float(value) >= clearance_threshold
                for value in completion_clearance.values()
            )
        )
        completion_values = [int(value) for value in completion_steps.values()]
        non_simultaneous_left = bool(
            len(completion_values) == 3
            and max(completion_values) - min(completion_values) >= 2
        )
        usable_bypass_gap_m = float(
            self._resolved_scenario_parameters.get(
                "usable_bypass_gap_m", 0.0
            )
            or 0.0
        )
        bypass_interaction_observed = bool(
            np.isfinite(minimum_bypass_gap_m)
            and usable_bypass_gap_m > 0.0
            and minimum_bypass_gap_m <= usable_bypass_gap_m
            and maximum_speed_spread_kmh >= 0.5
        )
        self._conflict_evidence.update(
            {
                "s9_actor_roles_present": {
                    "blocking_actor", "bypass_constraint_actor"
                }.issubset(self._actor_manifest),
                "left_completion_steps": dict(completion_steps),
                "left_completion_clearance_m": dict(completion_clearance),
                "required_completion_clearance_m": clearance_threshold,
                "left_completion_clearance_satisfied": clearance_ok,
                "non_simultaneous_left_lane_changes": non_simultaneous_left,
                "minimum_bypass_constraint_gap_m": (
                    None
                    if not np.isfinite(minimum_bypass_gap_m)
                    else float(minimum_bypass_gap_m)
                ),
                "maximum_platoon_speed_spread_during_bypass_kmh": float(
                    maximum_speed_spread_kmh
                ),
                "causal_bypass_constraint_interaction_observed": (
                    bypass_interaction_observed
                ),
                "blocker_pass_steps": dict(passed),
                "formation_current_bumper_gaps_m": gaps,
                "formation_recovery_stable_steps": stable,
                "formation_recovered_after_bypass": stable >= 2,
            }
        )
        self._route_completion.update(
            {
                "all_agents_changed_lane": len(completion_steps) == 3,
                "all_agents_passed_blocker": all_passed,
                "all_agents_continued_on_narrow_road": bool(stable >= 2),
            }
        )

    def _functional_success(self, recipes_complete: bool) -> bool:
        if not recipes_complete:
            return False
        if self._actor_manifest and not all(
            bool(row.get("active", False)) for row in self._actor_manifest.values()
        ):
            return False
        scenario_id = self.definition.scenario_id
        if scenario_id == "S5_hard_brake_lead":
            roles_complete = {
                "hard_brake_lead", "s5_adjacent_left", "s5_adjacent_right"
            }.issubset(self._actor_manifest)
            behavior = self._conflict_evidence.get("observed_behavior_class")
            behavior_evidence_valid = bool(
                behavior == "keep_emergency_braking"
                or (
                    behavior == "coordinated_lane_change"
                    and self._conflict_evidence.get(
                        "real_lane_change_completed", False
                    )
                    and self._conflict_evidence.get("lane_change_direction")
                    in {"left", "right"}
                )
                or (
                    behavior == "temporary_formation_release_and_recovery"
                    and self._conflict_evidence.get(
                        "mixed_direction_lane_change_completed", False
                    )
                    and self._conflict_evidence.get(
                        "reassembly_completed", False
                    )
                    and self._conflict_evidence.get("lane_change_direction")
                    == "mixed"
                )
            )
            return bool(
                roles_complete
                and self._conflict_evidence.get("lead_brake_profile_realized", False)
                and behavior_evidence_valid
                and self._conflict_evidence.get("hazard_cleared", False)
                and self._conflict_evidence.get(
                    "formation_recovered_after_hazard", False
                )
            )
        if scenario_id == "S6_background_merge_in":
            return bool(
                {"s6_gap_intruder"}.issubset(self._actor_manifest)
                and self._conflict_evidence.get("designated_gap_observed", False)
                and self._conflict_evidence.get(
                    "all_ego_changed_lane_after_cut_in", False
                )
                and self._conflict_evidence.get("measurable_platoon_response", False)
                and self._conflict_evidence.get(
                    "cut_in_actor_excluded_from_reassembly", False
                )
                and self._conflict_evidence.get("ego_only_reassembly", False)
                and self._conflict_evidence.get(
                    "formation_recovered_after_merge", False
                )
            )
        if scenario_id == "S7_ego_merge_from_ramp":
            required_roles = {
                "critical_gap_front",
                "critical_gap_rear",
                "next_gap_front",
                "next_gap_rear",
            }
            return bool(
                required_roles.issubset(self._actor_manifest)
                and self._conflict_evidence.get(
                    "critical_pair_traversed_conflict", False
                )
                and self._conflict_evidence.get(
                    "expected_behavior_matched", False
                )
                and self._route_completion.get(
                    "all_agents_entered_mainline", False
                )
                and self._route_completion.get(
                    "no_agent_stranded_on_ramp", False
                )
                and self._conflict_evidence.get(
                    "formation_recovered_after_merge", False
                )
            )
        if scenario_id == "S8_ego_exit_to_ramp":
            return bool(
                {"exit_gap_front", "exit_gap_rear"}.issubset(
                    self._actor_manifest
                )
                and self._route_completion.get(
                    "all_agents_entered_exit_side_lane", False
                )
                and self._conflict_evidence.get(
                    "causal_exit_actor_interaction_observed", False
                )
                and self._conflict_evidence.get(
                    "physical_split_observed", False
                )
                and self._conflict_evidence.get(
                    "non_simultaneous_right_lane_changes", False
                )
                and self._route_completion.get(
                    "all_agents_traversed_diverge_connector", False
                )
                and self._route_completion.get(
                    "all_agents_continued_on_exit_ramp", False
                )
                and not self._route_completion.get(
                    "returned_to_mainline", False
                )
                and not self._conflict_evidence.get(
                    "post_connector_nonkeep_observed", False
                )
                and self._conflict_evidence.get(
                    "formation_recovered_on_ramp", False
                )
            )
        if scenario_id == "S9_narrow_channel_negotiation":
            return bool(
                {"blocking_actor", "bypass_constraint_actor"}.issubset(
                    self._actor_manifest
                )
                and self._route_completion.get("all_agents_changed_lane", False)
                and self._conflict_evidence.get(
                    "non_simultaneous_left_lane_changes", False
                )
                and self._conflict_evidence.get(
                    "causal_bypass_constraint_interaction_observed", False
                )
                and self._conflict_evidence.get(
                    "left_completion_clearance_satisfied", False
                )
                and self._route_completion.get("all_agents_passed_blocker", False)
                and not self._route_completion.get("returned_to_original_lane", False)
                and self._route_completion.get(
                    "all_agents_continued_on_narrow_road", False
                )
                and self._conflict_evidence.get(
                    "formation_recovered_after_bypass", False
                )
            )
        return bool(self.summary.scenario_realized)

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
