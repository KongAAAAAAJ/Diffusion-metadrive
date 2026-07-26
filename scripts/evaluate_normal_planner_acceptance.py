#!/usr/bin/env python3
"""Evaluate native Normal planner acceptance without persisting a dataset."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from expert_dataset.collect_joint_bev import (
    JointBEVSampleBuilder,
    JointCollectionError,
    JointStepRejected,
    RulePlannerExpert,
    SensorlessJointBEVPlatoonEnv,
    simulator_decision_dt_s,
)
from expert_dataset.run_joint_bev_collection import (
    _configure_episode,
    load_run_config,
    sample_episode_spec,
)


AGENT_IDS = ("agent0", "agent1", "agent2")
DEFAULT_SCENARIOS = ("S5_hard_brake_lead", "S6_background_merge_in")


def _episode_specs(config, scenario_id: str, count: int):
    scenario_config = replace(config, scenario_weights={scenario_id: 1.0})
    return [sample_episode_spec(scenario_config, index) for index in range(count)]


def _info_failure(info) -> str | None:
    for agent_id in AGENT_IDS:
        item = info.get(agent_id, {}) if isinstance(info, dict) else {}
        if any(
            bool(item.get(key, False))
            for key in (
                "crash",
                "crash_vehicle",
                "crash_object",
                "crash_building",
                "crash_human",
            )
        ):
            return f"crash:{agent_id}"
        if any(
            bool(item.get(key, False))
            for key in ("out_of_road", "out_of_route")
        ):
            return f"out_of_road:{agent_id}"
    return None


def _traffic_vehicles(env) -> list[object]:
    traffic_manager = getattr(getattr(env, "engine", None), "traffic_manager", None)
    raw = getattr(traffic_manager, "_traffic_vehicles", ()) or ()
    return list(raw.values()) if isinstance(raw, dict) else list(raw)


def _vehicle_length_m(vehicle) -> float:
    return float(
        getattr(vehicle, "LENGTH", getattr(vehicle, "length", 5.74)) or 5.74
    )


def _vehicle_label(env, vehicle) -> str:
    for agent_id, agent in (getattr(env, "agents", {}) or {}).items():
        if agent is vehicle:
            return str(agent_id)
    role = getattr(vehicle, "scenario_vehicle_role", None)
    if role:
        return str(role)
    return str(getattr(vehicle, "name", ""))


def _bumper_gap_m(first, second) -> float:
    center_distance = float(
        np.linalg.norm(
            np.asarray(first.position[:2], dtype=np.float64)
            - np.asarray(second.position[:2], dtype=np.float64)
        )
    )
    return max(
        center_distance
        - 0.5 * (_vehicle_length_m(first) + _vehicle_length_m(second)),
        0.0,
    )


def _obb_clearance_m(first, second) -> float:
    first_heading = float(getattr(first, "heading_theta", 0.0) or 0.0)
    second_heading = float(getattr(second, "heading_theta", 0.0) or 0.0)
    axes = [
        np.asarray([np.cos(first_heading), np.sin(first_heading)]),
        np.asarray([-np.sin(first_heading), np.cos(first_heading)]),
        np.asarray([np.cos(second_heading), np.sin(second_heading)]),
        np.asarray([-np.sin(second_heading), np.cos(second_heading)]),
    ]
    delta = np.asarray(second.position[:2], dtype=np.float64) - np.asarray(
        first.position[:2], dtype=np.float64
    )
    dimensions = []
    for vehicle in (first, second):
        dimensions.append(
            (
                0.5 * _vehicle_length_m(vehicle),
                0.5
                * float(
                    getattr(
                        vehicle,
                        "WIDTH",
                        getattr(vehicle, "width", 2.3),
                    )
                    or 2.3
                ),
                float(getattr(vehicle, "heading_theta", 0.0) or 0.0),
            )
        )
    separations = []
    for axis in axes:
        projected_radii = []
        for half_length, half_width, heading in dimensions:
            longitudinal = np.asarray([np.cos(heading), np.sin(heading)])
            lateral = np.asarray([-np.sin(heading), np.cos(heading)])
            projected_radii.append(
                half_length * abs(float(np.dot(axis, longitudinal)))
                + half_width * abs(float(np.dot(axis, lateral)))
            )
        separations.append(
            abs(float(np.dot(delta, axis)))
            - projected_radii[0]
            - projected_radii[1]
        )
    return max(float(max(separations)), 0.0)


def _minimum_background_gap_m(env) -> float | None:
    agents = list((getattr(env, "agents", {}) or {}).values())
    traffic = _traffic_vehicles(env)
    if not agents or not traffic:
        return None
    return float(
        min(_bumper_gap_m(agent, vehicle) for agent in agents for vehicle in traffic)
    )


def _engine_policy(env, vehicle):
    getter = getattr(getattr(env, "engine", None), "get_policy", None)
    if not callable(getter):
        return None
    return getter(getattr(vehicle, "name", None))


def _merge_vehicle_states(env) -> list[dict]:
    states = []
    for vehicle in _traffic_vehicles(env):
        policy = _engine_policy(env, vehicle)
        if policy is None or type(policy).__name__ != "IDMMergePolicy":
            continue
        action_info = dict(getattr(policy, "action_info", {}) or {})
        states.append(
            {
                "name": str(getattr(vehicle, "name", "")),
                "role": getattr(vehicle, "scenario_vehicle_role", None),
                "position": np.asarray(vehicle.position[:2], dtype=np.float64).tolist(),
                "speed_kmh": float(getattr(vehicle, "speed_km_h", 0.0) or 0.0),
                "alignment": {
                    key: getattr(vehicle, f"scenario_{key}", None)
                    for key in (
                        "merge_arrival_offset_s",
                        "leader_ttc_s",
                        "merge_ttc_s",
                        "middle_ttc_s",
                        "leader_conflict_distance_m",
                        "merge_conflict_distance_m",
                        "conflict_point_xy",
                    )
                },
                "policy_state": {
                    key: action_info.get(key)
                    for key in (
                        "merge_active",
                        "merge_force_active",
                        "merge_front_gap",
                        "merge_rear_gap",
                        "merge_rear_ttc_s",
                        "merge_gap_accepted",
                        "merge_completed",
                    )
                },
            }
        )
    return states


def _s5_trigger_snapshot(env) -> dict | None:
    agents = getattr(env, "agents", {}) or {}
    leader = agents.get("agent0")
    if leader is None:
        return None
    traffic = _traffic_vehicles(env)
    lead = next(
        (
            vehicle
            for vehicle in traffic
            if getattr(vehicle, "scenario_role", None) == "hard_brake_lead"
            and hasattr(vehicle, "scenario_brake_trigger_step")
        ),
        None,
    )
    if lead is None:
        return None
    snapshot = {
        "trigger_step": int(getattr(lead, "scenario_brake_trigger_step")),
        "lead_bumper_gap_m": float(
            getattr(
                lead,
                "scenario_brake_trigger_bumper_gap_m",
                _bumper_gap_m(leader, lead),
            )
        ),
        "lead_initial_speed_kmh": float(
            getattr(lead, "scenario_brake_initial_speed_kmh", 0.0)
        ),
        "lead_target_speed_kmh": float(
            getattr(lead, "scenario_brake_target_speed_kmh", 0.0)
        ),
        "lead_deceleration_mps2": float(
            getattr(lead, "scenario_brake_deceleration_mps2", 0.0)
        ),
        "adjacent": {},
    }
    lane = getattr(leader, "lane", None)
    for vehicle in traffic:
        role = str(getattr(vehicle, "scenario_vehicle_role", ""))
        if not role.startswith("s5_adjacent_"):
            continue
        signed_center_offset = None
        if lane is not None:
            try:
                leader_s = float(lane.local_coordinates(leader.position)[0])
                vehicle_s = float(lane.local_coordinates(vehicle.position)[0])
                signed_center_offset = vehicle_s - leader_s
            except Exception:
                pass
        snapshot["adjacent"][role] = {
            "signed_center_offset_m": signed_center_offset,
            "bumper_gap_m": _bumper_gap_m(leader, vehicle),
            "speed_kmh": float(getattr(vehicle, "speed_km_h", 0.0) or 0.0),
        }
    return snapshot


def _crash_context(env, failure_reason: str) -> dict | None:
    if not failure_reason.startswith("crash:"):
        return None
    agent_id = failure_reason.split(":", 1)[1]
    agent = (getattr(env, "agents", {}) or {}).get(agent_id)
    if agent is None:
        return None
    neighbors = []
    crashed_objects = []
    for vehicle in [
        *list((getattr(env, "agents", {}) or {}).values()),
        *_traffic_vehicles(env),
    ]:
        if vehicle is agent:
            continue
        policy = _engine_policy(env, vehicle)
        neighbors.append(
            {
                "name": _vehicle_label(env, vehicle),
                "bumper_gap_m": _bumper_gap_m(agent, vehicle),
                "obb_clearance_m": _obb_clearance_m(agent, vehicle),
                "policy": None if policy is None else type(policy).__name__,
                "scenario_role": getattr(vehicle, "scenario_vehicle_role", None),
            }
        )
        if any(
            bool(getattr(vehicle, key, False))
            for key in (
                "crash",
                "crash_vehicle",
                "crash_object",
                "crash_building",
                "crash_human",
            )
        ):
            crashed_objects.append(_vehicle_label(env, vehicle))
    neighbors.sort(key=lambda value: value["bumper_gap_m"])
    return {
        "agent_id": agent_id,
        "nearest_objects": neighbors[:3],
        "crashed_objects": crashed_objects,
        "merge_vehicles": _merge_vehicle_states(env),
    }


def _run_episode(env, spec, max_steps: int, verify_samples: bool) -> dict:
    _configure_episode(env, spec)
    env.reset()
    expert = RulePlannerExpert(env, AGENT_IDS)
    builder = JointBEVSampleBuilder(AGENT_IDS) if verify_samples else None
    decision_dt_s = simulator_decision_dt_s(env)
    planning_times_ms = []
    role_raw = Counter()
    role_valid = Counter()
    collision_rejections = Counter()
    braking_steps = 0
    stop_steps = 0
    label_rejections = Counter()
    samples_checked = 0
    failure_reason = None
    failure_debug = None
    simulator_steps = 0
    minimum_background_gap_m = float("inf")
    s5_snapshot = None
    lead_speed_history_kmh = []
    merge_prediction_errors_m = []
    agent_tracking_errors_m: dict[str, list[float]] = defaultdict(list)
    pending_merge_predictions: dict[int, list[tuple[str, np.ndarray]]] = defaultdict(
        list
    )
    pending_agent_predictions: dict[int, list[tuple[str, np.ndarray]]] = defaultdict(
        list
    )
    minimum_pairwise_gaps_m: dict[str, float] = {}
    minimum_merge_agent_obb_clearance_m = {
        agent_id: float("inf") for agent_id in AGENT_IDS
    }
    merge_prediction_steps = max(int(round(0.5 / decision_dt_s)), 1)

    for step in range(max_steps):
        traffic_by_name = {
            str(getattr(vehicle, "name", "")): vehicle
            for vehicle in _traffic_vehicles(env)
        }
        for name, predicted_xy in pending_merge_predictions.pop(step, []):
            vehicle = traffic_by_name.get(name)
            if vehicle is not None:
                merge_prediction_errors_m.append(
                    float(
                        np.linalg.norm(
                            np.asarray(vehicle.position[:2], dtype=np.float64)
                            - predicted_xy
                        )
                    )
                )
        for agent_id, predicted_xy in pending_agent_predictions.pop(step, []):
            agent = (getattr(env, "agents", {}) or {}).get(agent_id)
            if agent is not None:
                agent_tracking_errors_m[agent_id].append(
                    float(
                        np.linalg.norm(
                            np.asarray(agent.position[:2], dtype=np.float64)
                            - predicted_xy
                        )
                    )
                )
        all_objects = [
            *list((getattr(env, "agents", {}) or {}).values()),
            *_traffic_vehicles(env),
        ]
        for first_index, first in enumerate(all_objects):
            for second in all_objects[first_index + 1 :]:
                first_name = str(getattr(first, "name", ""))
                second_name = str(getattr(second, "name", ""))
                pair_name = "|".join(sorted((first_name, second_name)))
                gap = _bumper_gap_m(first, second)
                minimum_pairwise_gaps_m[pair_name] = min(
                    minimum_pairwise_gaps_m.get(pair_name, float("inf")),
                    gap,
                )
        merge_vehicles = [
            vehicle
            for vehicle in _traffic_vehicles(env)
            if getattr(vehicle, "scenario_vehicle_role", None)
            == "s6_merge_vehicle"
        ]
        for merge_vehicle in merge_vehicles:
            for agent_id, agent in (getattr(env, "agents", {}) or {}).items():
                if agent_id in minimum_merge_agent_obb_clearance_m:
                    minimum_merge_agent_obb_clearance_m[agent_id] = min(
                        minimum_merge_agent_obb_clearance_m[agent_id],
                        _obb_clearance_m(agent, merge_vehicle),
                    )
        observed_gap = _minimum_background_gap_m(env)
        if observed_gap is not None:
            minimum_background_gap_m = min(minimum_background_gap_m, observed_gap)
        if s5_snapshot is None:
            s5_snapshot = _s5_trigger_snapshot(env)
        if s5_snapshot is not None:
            lead = next(
                (
                    vehicle
                    for vehicle in _traffic_vehicles(env)
                    if getattr(vehicle, "scenario_role", None)
                    == "hard_brake_lead"
                ),
                None,
            )
            if lead is not None:
                lead_speed_history_kmh.append(
                    float(getattr(lead, "speed_km_h", 0.0) or 0.0)
                )
        if builder is not None:
            builder.capture_state(env, timestamp_s=step * decision_dt_s)
        try:
            expert_step = expert.plan(env)
        except JointCollectionError as exc:
            failure_reason = exc.reason_code
            planner_debug = expert.planner.get_last_debug() or {}
            failure_debug = {
                "joint": planner_debug.get("_joint", {}),
                "agents": {
                    agent_id: {
                        key: planner_debug.get(agent_id, {}).get(key)
                        for key in (
                            "rule_action",
                            "fallback_reason",
                            "raw_candidate_count",
                            "generated_valid_candidate_count",
                            "kinematic_rejection_count",
                            "corridor_rejection_count",
                            "road_rejection_count",
                            "background_collision_rejection_count",
                            "collision_rejections_by_object",
                            "source_envelope",
                            "target_envelope",
                            "safe_corridor_by_duration_m",
                        )
                    }
                    for agent_id in AGENT_IDS
                },
            }
            break

        planner_debug = expert.planner.get_last_debug() or {}
        joint_debug = planner_debug.get("_joint", {})
        planning_times_ms.append(float(joint_debug.get("planning_time_ms", np.nan)))
        for agent_id in AGENT_IDS:
            trajectory = np.asarray(
                expert_step.trajectories_world[agent_id],
                dtype=np.float64,
            )
            pending_agent_predictions[step + merge_prediction_steps].append(
                (agent_id, trajectory[0, :2].copy())
            )
        for merge_state in _merge_vehicle_states(env):
            vehicle = traffic_by_name.get(merge_state["name"])
            if vehicle is None:
                continue
            predicted = expert.planner._predict_vehicle_trajectory(
                env,
                vehicle,
                np.asarray([0.0, 0.5], dtype=np.float64),
            )
            pending_merge_predictions[step + merge_prediction_steps].append(
                (
                    merge_state["name"],
                    np.asarray(predicted[1, :2], dtype=np.float64),
                )
            )
        for agent_id in AGENT_IDS:
            item = planner_debug.get(agent_id, {})
            role_raw[agent_id] += int(item.get("raw_candidate_count", 0))
            role_valid[agent_id] += int(
                item.get("generated_valid_candidate_count", 0)
            )
            collision_rejections.update(
                item.get("collision_rejections_by_object", {})
            )
            if float(item.get("selected_acceleration_mps2", 0.0)) < -0.25:
                braking_steps += 1
            if item.get("selected_stop_time_s") is not None:
                stop_steps += 1

        if builder is not None and builder.history_ready():
            try:
                sample = builder.build_sample(env, expert_step)
            except JointStepRejected as exc:
                label_rejections[exc.reason_code] += 1
            else:
                if not np.all(
                    sample.mode_valid_mask[
                        np.arange(len(AGENT_IDS)),
                        sample.gt_mode,
                    ]
                ):
                    raise RuntimeError("mode_valid_mask[gt_mode] invariant failed")
                samples_checked += 1

        _, _, terminated, truncated, info = env.low_level_step(
            dict(expert_step.controls)
        )
        simulator_steps += 1
        failure_reason = _info_failure(info)
        if failure_reason is not None and failure_reason.startswith("crash:"):
            failure_debug = {
                **(failure_debug or {}),
                "crash_context": _crash_context(env, failure_reason),
            }
        if (
            failure_reason is not None
            or bool(terminated.get("__all__", False))
            or bool(truncated.get("__all__", False))
        ):
            break

    finite_times = np.asarray(planning_times_ms, dtype=np.float64)
    finite_times = finite_times[np.isfinite(finite_times)]
    observed_decelerations = (
        -np.diff(np.asarray(lead_speed_history_kmh, dtype=np.float64))
        / 3.6
        / decision_dt_s
        if len(lead_speed_history_kmh) >= 2
        else np.empty((0,), dtype=np.float64)
    )
    scenario_orchestrator = getattr(env, "_scenario_orchestrator", None)
    scenario_summary = (
        scenario_orchestrator.get_episode_summary()
        if scenario_orchestrator is not None
        and hasattr(scenario_orchestrator, "get_episode_summary")
        else None
    )
    if (
        failure_reason is None
        and scenario_summary is not None
        and not bool(scenario_summary.get("scenario_realized", False))
    ):
        failure_reason = "scenario_not_realized"
    persistable = (
        failure_reason is None and samples_checked > 0
        if verify_samples
        else None
    )
    return {
        "scenario_id": spec.scenario_id,
        "local_route": spec.local_route,
        "spawn_seed": int(spec.spawn_seed),
        "planner_native": not (
            failure_reason is not None
            and failure_reason.startswith("normal_planner_")
        ),
        "failure_reason": failure_reason,
        "failure_debug": failure_debug,
        "persistable_episode": persistable,
        "simulator_steps": simulator_steps,
        "samples_checked": samples_checked,
        "label_rejections": dict(label_rejections),
        "planning_times_ms": finite_times.tolist(),
        "role_raw": dict(role_raw),
        "role_valid": dict(role_valid),
        "collision_rejections": dict(collision_rejections),
        "braking_steps": braking_steps,
        "stop_steps": stop_steps,
        "minimum_background_gap_m": (
            float(minimum_background_gap_m)
            if np.isfinite(minimum_background_gap_m)
            else None
        ),
        "s5_trigger_snapshot": s5_snapshot,
        "s5_max_observed_deceleration_mps2": (
            float(np.max(observed_decelerations))
            if observed_decelerations.size
            else None
        ),
        "merge_prediction_error_m": merge_prediction_errors_m,
        "agent_tracking_error_m": dict(agent_tracking_errors_m),
        "minimum_pairwise_gaps_m": dict(
            sorted(
                minimum_pairwise_gaps_m.items(),
                key=lambda item: item[1],
            )
        ),
        "minimum_merge_agent_obb_clearance_m": {
            agent_id: (
                float(value) if np.isfinite(value) else None
            )
            for agent_id, value in minimum_merge_agent_obb_clearance_m.items()
        },
        "final_merge_states": _merge_vehicle_states(env),
        "scenario_summary": scenario_summary,
    }


def evaluate(args: argparse.Namespace) -> dict:
    config = load_run_config(args.config)
    report = {
        "config": str(Path(args.config).resolve()),
        "episodes_per_scenario": args.episodes_per_scenario,
        "max_steps": args.max_steps,
        "verify_samples": not args.skip_sample_verification,
        "scenarios": {},
    }
    for scenario_id in args.scenarios:
        env = SensorlessJointBEVPlatoonEnv(dict(config.env_config))
        started_at = time.perf_counter()
        try:
            episodes = [
                _run_episode(
                    env,
                    spec,
                    args.max_steps,
                    not args.skip_sample_verification,
                )
                for spec in _episode_specs(
                    config,
                    scenario_id,
                    args.episodes_per_scenario,
                )
            ]
        finally:
            env.close()

        times = np.asarray(
            [
                value
                for episode in episodes
                for value in episode["planning_times_ms"]
            ],
            dtype=np.float64,
        )
        failure_counts = Counter(
            episode["failure_reason"] or "none" for episode in episodes
        )
        role_raw = Counter()
        role_valid = Counter()
        collision_rejections = Counter()
        label_rejections = Counter()
        for episode in episodes:
            role_raw.update(episode["role_raw"])
            role_valid.update(episode["role_valid"])
            collision_rejections.update(episode["collision_rejections"])
            label_rejections.update(episode["label_rejections"])
        native_count = sum(bool(episode["planner_native"]) for episode in episodes)
        persistable_values = [
            bool(episode["persistable_episode"])
            for episode in episodes
            if episode["persistable_episode"] is not None
        ]
        minimum_gaps = np.asarray(
            [
                episode["minimum_background_gap_m"]
                for episode in episodes
                if episode["minimum_background_gap_m"] is not None
            ],
            dtype=np.float64,
        )
        merge_prediction_errors = np.asarray(
            [
                value
                for episode in episodes
                for value in episode["merge_prediction_error_m"]
            ],
            dtype=np.float64,
        )
        tracking_errors = {
            agent_id: np.asarray(
                [
                    value
                    for episode in episodes
                    for value in episode["agent_tracking_error_m"].get(
                        agent_id, ()
                    )
                ],
                dtype=np.float64,
            )
            for agent_id in AGENT_IDS
        }
        final_merge_states = [
            state
            for episode in episodes
            for state in episode["final_merge_states"]
        ]
        arrival_offsets = np.asarray(
            [
                state["alignment"]["merge_arrival_offset_s"]
                for state in final_merge_states
                if state["alignment"]["merge_arrival_offset_s"] is not None
            ],
            dtype=np.float64,
        )
        merge_clearances = {
            agent_id: np.asarray(
                [
                    episode["minimum_merge_agent_obb_clearance_m"][agent_id]
                    for episode in episodes
                    if episode["minimum_merge_agent_obb_clearance_m"][agent_id]
                    is not None
                ],
                dtype=np.float64,
            )
            for agent_id in AGENT_IDS
        }
        total_agent_steps = max(
            3 * sum(int(episode["simulator_steps"]) for episode in episodes),
            1,
        )
        scenario_report = {
            "native_episodes": native_count,
            "total_episodes": len(episodes),
            "native_episode_rate": native_count / max(len(episodes), 1),
            "persistable_episodes": (
                sum(persistable_values) if persistable_values else None
            ),
            "persistable_episode_rate": (
                sum(persistable_values) / len(persistable_values)
                if persistable_values
                else None
            ),
            "failure_counts": dict(sorted(failure_counts.items())),
            "role_candidate_acceptance": {
                agent_id: (
                    role_valid[agent_id] / role_raw[agent_id]
                    if role_raw[agent_id]
                    else 0.0
                )
                for agent_id in AGENT_IDS
            },
            "top_collision_rejections": collision_rejections.most_common(10),
            "label_rejections": dict(sorted(label_rejections.items())),
            "samples_checked": sum(
                int(episode["samples_checked"]) for episode in episodes
            ),
            "braking_agent_step_rate": sum(
                int(episode["braking_steps"]) for episode in episodes
            )
            / total_agent_steps,
            "stop_agent_step_rate": sum(
                int(episode["stop_steps"]) for episode in episodes
            )
            / total_agent_steps,
            "minimum_background_gap_m": (
                float(np.min(minimum_gaps)) if minimum_gaps.size else None
            ),
            "merge_prediction_error_m": {
                "mean": (
                    float(np.mean(merge_prediction_errors))
                    if merge_prediction_errors.size
                    else None
                ),
                "p95": (
                    float(np.percentile(merge_prediction_errors, 95))
                    if merge_prediction_errors.size
                    else None
                ),
                "max": (
                    float(np.max(merge_prediction_errors))
                    if merge_prediction_errors.size
                    else None
                ),
            },
            "merge_arrival_offset_s": {
                "mean": (
                    float(np.mean(arrival_offsets))
                    if arrival_offsets.size
                    else None
                ),
                "min": (
                    float(np.min(arrival_offsets))
                    if arrival_offsets.size
                    else None
                ),
                "max": (
                    float(np.max(arrival_offsets))
                    if arrival_offsets.size
                    else None
                ),
            },
            "merge_completed_episodes": sum(
                any(
                    bool(
                        state["policy_state"].get("merge_completed", False)
                    )
                    for state in episode["final_merge_states"]
                )
                for episode in episodes
            ),
            "minimum_merge_agent_obb_clearance_m": {
                agent_id: (
                    float(np.min(values)) if values.size else None
                )
                for agent_id, values in merge_clearances.items()
            },
            "agent_tracking_error_m": {
                agent_id: {
                    "mean": (
                        float(np.mean(values)) if values.size else None
                    ),
                    "p95": (
                        float(np.percentile(values, 95))
                        if values.size
                        else None
                    ),
                    "max": (
                        float(np.max(values)) if values.size else None
                    ),
                }
                for agent_id, values in tracking_errors.items()
            },
            "s5_trigger_snapshots": [
                episode["s5_trigger_snapshot"]
                for episode in episodes
                if episode["s5_trigger_snapshot"] is not None
            ],
            "s5_max_observed_deceleration_mps2": [
                episode["s5_max_observed_deceleration_mps2"]
                for episode in episodes
                if episode["s5_max_observed_deceleration_mps2"] is not None
            ],
            "planner_latency_ms": {
                "mean": float(np.mean(times)) if times.size else None,
                "p95": float(np.percentile(times, 95)) if times.size else None,
                "max": float(np.max(times)) if times.size else None,
            },
            "wall_time_s": time.perf_counter() - started_at,
            "episode_outcomes": [
                {
                    "planner_native": bool(episode["planner_native"]),
                    "persistable_episode": episode["persistable_episode"],
                    "failure_reason": episode["failure_reason"],
                    "local_route": episode["local_route"],
                    "spawn_seed": int(episode["spawn_seed"]),
                    "simulator_steps": int(episode["simulator_steps"]),
                    "samples_checked": int(episode["samples_checked"]),
                    **(
                        {"failure_debug": episode["failure_debug"]}
                        if args.include_failure_debug
                        else {}
                    ),
                }
                for episode in episodes
            ],
        }
        if args.include_episode_details:
            scenario_report["episode_details"] = episodes
        report["scenarios"][scenario_id] = scenario_report
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset/data_collect.yaml"),
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=DEFAULT_SCENARIOS,
        default=list(DEFAULT_SCENARIOS),
    )
    parser.add_argument("--episodes-per-scenario", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--skip-sample-verification", action="store_true")
    parser.add_argument("--include-episode-details", action="store_true")
    parser.add_argument("--include-failure-debug", action="store_true")
    args = parser.parse_args()
    if args.episodes_per_scenario <= 0 or args.max_steps <= 0:
        parser.error("episode and step counts must be positive")
    return args


if __name__ == "__main__":
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))
