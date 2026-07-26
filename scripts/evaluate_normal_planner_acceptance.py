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

    for step in range(max_steps):
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
        if (
            failure_reason is not None
            or bool(terminated.get("__all__", False))
            or bool(truncated.get("__all__", False))
        ):
            break

    finite_times = np.asarray(planning_times_ms, dtype=np.float64)
    finite_times = finite_times[np.isfinite(finite_times)]
    return {
        "planner_native": not (
            failure_reason is not None
            and failure_reason.startswith("normal_planner_")
        ),
        "failure_reason": failure_reason,
        "failure_debug": failure_debug,
        "simulator_steps": simulator_steps,
        "samples_checked": samples_checked,
        "label_rejections": dict(label_rejections),
        "planning_times_ms": finite_times.tolist(),
        "role_raw": dict(role_raw),
        "role_valid": dict(role_valid),
        "collision_rejections": dict(collision_rejections),
        "braking_steps": braking_steps,
        "stop_steps": stop_steps,
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
        total_agent_steps = max(
            3 * sum(int(episode["simulator_steps"]) for episode in episodes),
            1,
        )
        scenario_report = {
            "native_episodes": native_count,
            "total_episodes": len(episodes),
            "native_episode_rate": native_count / max(len(episodes), 1),
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
            "planner_latency_ms": {
                "mean": float(np.mean(times)) if times.size else None,
                "p95": float(np.percentile(times, 95)) if times.size else None,
                "max": float(np.max(times)) if times.size else None,
            },
            "wall_time_s": time.perf_counter() - started_at,
            "episode_outcomes": [
                {
                    "planner_native": bool(episode["planner_native"]),
                    "failure_reason": episode["failure_reason"],
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
