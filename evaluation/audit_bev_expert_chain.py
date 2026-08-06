"""Audit the S5/S6/S7/S9 RuleMaker-to-execution expert chain."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from types import MethodType
from typing import Mapping, Sequence

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from expert_dataset.collect_joint_bev import (  # noqa: E402
    JointBEVSampleBuilder,
    JointCollectionError,
    JointStepRejected,
    RulePlannerExpert,
    SensorlessJointBEVPlatoonEnv,
    simulator_decision_dt_s,
)
from expert_dataset.run_joint_bev_collection import (  # noqa: E402
    JointEpisodeSpec,
    _configure_episode,
    load_run_config,
)
from models.platoon_planner.platoon_normal_planner import (  # noqa: E402
    PlatoonNormalPlanner,
)
from models.bev_planner.mode_contract import (  # noqa: E402
    KEEP_MODES,
    LEFT_MODES,
    RIGHT_MODES,
    ModeIndex,
    build_hard_mode_valid_mask,
    validate_trajectory_kinematics,
)
from scenarios.bev_round13_contract import (  # noqa: E402
    PRIMARY_S5_S9_SCENARIOS,
)
from scenarios.definitions import get_scenario_definition  # noqa: E402


AUDIT_SCENARIOS = tuple(
    value
    for value in PRIMARY_S5_S9_SCENARIOS
    if not value[0].startswith("S8_")
)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _episode_spec(
    scenario_id: str,
    local_route: str,
    seed: int,
) -> JointEpisodeSpec:
    definition = get_scenario_definition(scenario_id)
    rng = np.random.RandomState(int(seed))
    initial_speed = definition.ego_initial_speed_km_h
    if isinstance(initial_speed, (tuple, list)) and len(initial_speed) == 2:
        speed = float(rng.uniform(float(initial_speed[0]), float(initial_speed[1])))
    elif initial_speed is None:
        speed = 25.0
    else:
        speed = float(initial_speed)
    density = float(definition.override_traffic_density or 0.0)
    return JointEpisodeSpec(
        scenario_id=scenario_id,
        local_route=local_route,
        spawn_seed=int(seed),
        traffic_density=density,
        initial_speed_km_h=speed,
    )


def _crash_reason(agent_info: Mapping[str, object], agent_id: str) -> str | None:
    if bool(agent_info.get("crash_vehicle", False)):
        return f"crash_vehicle:{agent_id}"
    if bool(agent_info.get("crash_sidewalk", False)):
        return f"crash_sidewalk:{agent_id}"
    if any(
        bool(agent_info.get(key, False))
        for key in ("crash", "crash_object", "crash_building", "crash_human")
    ):
        return f"crash_object:{agent_id}"
    if any(
        bool(agent_info.get(key, False))
        for key in ("out_of_road", "out_of_route")
    ):
        return f"out_of_road:{agent_id}"
    return None


def _compact_rule_debug(debug: Mapping[str, object]) -> dict[str, object]:
    selected_candidates = {}
    candidates_by_agent = debug.get("candidates_by_agent", {}) or {}
    if isinstance(candidates_by_agent, Mapping):
        for agent_id, candidates in candidates_by_agent.items():
            if not isinstance(candidates, Sequence):
                continue
            selected = next(
                (
                    value
                    for value in candidates
                    if isinstance(value, Mapping)
                    and bool(value.get("selected", False))
                ),
                None,
            )
            if selected is not None:
                selected_candidates[str(agent_id)] = {
                    "action": selected.get("action"),
                    "target_point": selected.get("target_point"),
                    "source_lane_index": selected.get("source_lane_index"),
                    "target_lane_index": selected.get("target_lane_index"),
                }
    return {
        "best_actions": dict(debug.get("best_actions", {}) or {}),
        "formation_locked": bool(debug.get("formation_locked", False)),
        "risk_triggered": bool(debug.get("risk_triggered", False)),
        "risk_info": debug.get("risk_info", {}),
        "best_score": debug.get("best_score"),
        "selected_candidates": selected_candidates,
    }


def _compact_planner_debug(debug: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {
        "joint": debug.get("_joint", {}),
        "execution": debug.get("_execution", {}),
    }
    for agent_id in ("agent0", "agent1", "agent2"):
        value = debug.get(agent_id, {})
        if not isinstance(value, Mapping):
            continue
        result[agent_id] = {
            key: value.get(key)
            for key in (
                "rule_action",
                "rule_target_point",
                "source_lane_index",
                "target_lane_index",
                "target_lane_chain",
                "raw_candidate_count",
                "generated_valid_candidate_count",
                "candidate_count",
                "kinematic_rejection_count",
                "corridor_rejection_count",
                "road_rejection_count",
                "background_collision_rejection_count",
                "lane_end_rejection_count",
                "kinematic_rejections_by_reason",
                "collision_rejections_by_object",
                "lateral_targets_m",
                "source_lane_remaining_m",
                "source_lane_usable_progress_m",
                "lane_end_restricted",
                "lane_change_durations_s",
                "selected_acceleration_mps2",
                "selected_lane_change_duration_s",
                "selected_lane_change_start_delay_s",
            )
        }
    return result


def _agent_state(env, agent_id: str) -> dict[str, object]:
    vehicle = env.agents[agent_id]
    lane = getattr(vehicle, "lane", None)
    longitudinal = None
    lateral = None
    lane_index = None
    lane_length = None
    if lane is not None:
        lane_index = list(tuple(getattr(lane, "index", ()) or ()))
        lane_length = float(getattr(lane, "length", 0.0) or 0.0)
        try:
            longitudinal, lateral = lane.local_coordinates(vehicle.position)
        except Exception:
            longitudinal = None
            lateral = None
    return {
        "position": np.asarray(vehicle.position[:2], dtype=np.float64).tolist(),
        "heading": float(getattr(vehicle, "heading_theta", 0.0)),
        "speed_km_h": float(getattr(vehicle, "speed_km_h", 0.0) or 0.0),
        "lane_index": lane_index,
        "lane_length_m": lane_length,
        "lane_s_m": None if longitudinal is None else float(longitudinal),
        "lane_d_m": None if lateral is None else float(lateral),
    }


def _mode_mask_components(
    env,
    builder: JointBEVSampleBuilder,
    model_inputs,
    expert_step,
    agent_ids: Sequence[str],
) -> dict[str, object]:
    action_groups = {
        -1: LEFT_MODES,
        0: KEEP_MODES + (int(ModeIndex.STOP),),
        1: RIGHT_MODES,
    }
    component_masks: dict[str, object] = {}
    for role_index, agent_id in enumerate(agent_ids):
        anchors = builder.anchor_generator.generate(env, agent_id)
        result = build_hard_mode_valid_mask(
            model_inputs.bev[role_index],
            model_inputs.coarse_trajectories[role_index],
            float(model_inputs.ego_state[role_index, 0]),
            anchors.topology,
        )
        action = int(expert_step.rule_actions[agent_id])
        component_masks[agent_id] = {
            "topology": result.topology_mask.tolist(),
            "road": result.road_mask.tolist(),
            "kinematic": result.kinematic_mask.tolist(),
            "valid": result.valid_mask.tolist(),
            "action_group": list(action_groups[action]),
            "valid_in_action_group": [
                int(index)
                for index in action_groups[action]
                if bool(result.valid_mask[int(index)])
            ],
            "kinematic_violations": {
                str(mode_index): list(
                    validate_trajectory_kinematics(
                        model_inputs.coarse_trajectories[
                            role_index, mode_index
                        ],
                        float(model_inputs.ego_state[role_index, 0]),
                        np.zeros((3,), dtype=np.float64),
                    ).violations
                )
                for mode_index in range(
                    model_inputs.coarse_trajectories.shape[1]
                )
                if not bool(result.kinematic_mask[mode_index])
            },
        }
    return component_masks


def _legacy_keep_probe(
    env,
    decisions: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Compare the pre-13.72 KEEP lattice with the production lattice."""

    def legacy_targets(
        self,
        action,
        start_d,
        desired_end_d,
        source_lane,
        target_lane,
    ):
        del start_d
        if int(action) != 0:
            return PlatoonNormalPlanner._candidate_lateral_targets(
                self,
                action,
                0.0,
                desired_end_d,
                source_lane,
                target_lane,
            )
        margin = self.keep_lateral_margin_m
        lane_half_width = 0.5 * float(
            getattr(source_lane, "width", 3.5) or 3.5
        )
        values = (
            np.clip(
                desired_end_d - margin,
                -lane_half_width,
                lane_half_width,
            ),
            np.clip(desired_end_d, -lane_half_width, lane_half_width),
            np.clip(
                desired_end_d + margin,
                -lane_half_width,
                lane_half_width,
            ),
        )
        return tuple(
            dict.fromkeys(round(float(value), 4) for value in values)
        )

    current = PlatoonNormalPlanner()
    current.plan(env, decisions)
    current_debug = current.get_last_debug() or {}
    legacy = PlatoonNormalPlanner()
    legacy._candidate_lateral_targets = MethodType(legacy_targets, legacy)
    legacy.plan(env, decisions)
    legacy_debug = legacy.get_last_debug() or {}
    return {
        "current_joint_safe": not bool(
            (current_debug.get("_joint", {}) or {}).get(
                "fallback_used", True
            )
        ),
        "legacy_joint_safe": not bool(
            (legacy_debug.get("_joint", {}) or {}).get(
                "fallback_used", True
            )
        ),
        "current": _compact_planner_debug(current_debug),
        "legacy": _compact_planner_debug(legacy_debug),
    }


def audit_episode(
    env: SensorlessJointBEVPlatoonEnv,
    spec: JointEpisodeSpec,
    *,
    max_steps: int,
) -> dict[str, object]:
    _configure_episode(env, spec)
    env.reset(seed=int(spec.spawn_seed))
    agent_ids = ("agent0", "agent1", "agent2")
    expert = RulePlannerExpert(env, agent_ids)
    builder = JointBEVSampleBuilder(agent_ids)
    builder.reset()
    dt_s = simulator_decision_dt_s(env)
    rows: list[dict[str, object]] = []
    failure_reason = None
    s5_probe = None

    for step in range(int(max_steps)):
        builder.capture_state(env, step * dt_s)
        try:
            expert_step = expert.plan(env)
        except JointCollectionError as exc:
            failure_reason = exc.reason_code
            rows.append(
                {
                    "step": step,
                    "failure_reason": failure_reason,
                    "error": str(exc),
                    "rule": _compact_rule_debug(
                        expert.rule_maker.get_last_debug() or {}
                    ),
                    "planner": _compact_planner_debug(
                        expert.planner.get_last_debug() or {}
                    ),
                    "agents": {
                        agent_id: _agent_state(env, agent_id)
                        for agent_id in agent_ids
                        if agent_id in env.agents
                    },
                }
            )
            break

        rule_debug = expert.rule_maker.get_last_debug() or {}
        planner_debug = expert.planner.get_last_debug() or {}
        row: dict[str, object] = {
            "step": step,
            "rule": _compact_rule_debug(rule_debug),
            "planner": _compact_planner_debug(planner_debug),
            "controls": {
                name: np.asarray(value, dtype=np.float32).tolist()
                for name, value in expert_step.controls.items()
            },
            "agents": {
                agent_id: _agent_state(env, agent_id)
                for agent_id in agent_ids
            },
        }
        if builder.history_ready():
            try:
                model_inputs = builder.build_model_inputs(env)
                row["mode_valid_mask"] = model_inputs.mode_valid_mask.tolist()
                row["mode_mask_components"] = _mode_mask_components(
                    env,
                    builder,
                    model_inputs,
                    expert_step,
                    agent_ids,
                )
                sample = builder.build_sample(env, expert_step)
                row["gt_mode"] = sample.gt_mode.tolist()
            except (JointCollectionError, JointStepRejected) as exc:
                failure_reason = exc.reason_code
                row["failure_reason"] = failure_reason
                row["error"] = str(exc)
                rows.append(row)
                break

        if (
            spec.scenario_id == "S5_hard_brake_lead"
            and any(
                int(value) == 0
                for value in (rule_debug.get("best_actions", {}) or {}).values()
            )
        ):
            decisions = {
                agent_id: {
                    "action": int(expert_step.rule_actions[agent_id]),
                    "target_point": np.asarray(
                        (planner_debug.get(agent_id, {}) or {}).get(
                            "rule_target_point",
                            [15.0, 0.0],
                        ),
                        dtype=np.float32,
                    ),
                }
                for agent_id in agent_ids
            }
            probe = {"step": step, **_legacy_keep_probe(env, decisions)}
            if s5_probe is None:
                s5_probe = probe
            if (
                bool(probe["current_joint_safe"])
                and not bool(probe["legacy_joint_safe"])
            ):
                s5_probe = probe

        _, _, terminated, truncated, info = env.low_level_step(
            dict(expert_step.controls)
        )
        for agent_id in agent_ids:
            agent_info = info.get(agent_id, {}) if isinstance(info, Mapping) else {}
            if isinstance(agent_info, Mapping):
                reason = _crash_reason(agent_info, agent_id)
                if reason is not None:
                    failure_reason = reason
                    row["failure_reason"] = reason
                    row["failure_info"] = dict(agent_info)
                    break
        rows.append(row)
        if (
            failure_reason is not None
            or bool(terminated.get("__all__", False))
            or bool(truncated.get("__all__", False))
        ):
            break

    orchestrator = getattr(env, "_scenario_orchestrator", None)
    scenario_summary = (
        orchestrator.get_episode_summary()
        if orchestrator is not None
        and hasattr(orchestrator, "get_episode_summary")
        else {}
    )
    return {
        "spec": asdict(spec),
        "failure_reason": failure_reason,
        "scenario_summary": scenario_summary,
        "s5_feasibility_probe": s5_probe,
        "steps": rows,
    }


def run_audit(
    *,
    config_path: Path,
    seeds: Sequence[int],
    max_steps: int,
) -> dict[str, object]:
    config = load_run_config(config_path)
    episodes = []
    for scenario_id, local_route in AUDIT_SCENARIOS:
        for seed in seeds:
            env_config = dict(config.env_config)
            env_config.update(
                {"start_seed": int(seed), "num_scenarios": 1}
            )
            env = SensorlessJointBEVPlatoonEnv(env_config)
            try:
                episodes.append(
                    audit_episode(
                        env,
                        _episode_spec(scenario_id, local_route, int(seed)),
                        max_steps=max_steps,
                    )
                )
            finally:
                env.close()
    return {
        "format": "bev_expert_chain_audit_v1",
        "scenarios": [list(value) for value in AUDIT_SCENARIOS],
        "seeds": [int(value) for value in seeds],
        "max_steps": int(max_steps),
        "episodes": episodes,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset/data_collect_diagnostic64.yaml"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(17, 23))
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_audit(
        config_path=args.config,
        seeds=args.seeds,
        max_steps=args.max_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_value)
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "output": str(args.output),
        "episodes": len(report["episodes"]),
        "failures": [
            {
                "scenario": episode["spec"]["scenario_id"],
                "seed": episode["spec"]["spawn_seed"],
                "reason": episode["failure_reason"],
            }
            for episode in report["episodes"]
            if episode["failure_reason"] is not None
        ],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
