"""preview_and_evaluation.py

Run PlatoonEnv with a three-stage decision/planning/control pipeline, save top-down videos, and
(optionally) evaluate the tested algorithm's performance in the scenario via
evaluation/platoon_performance.py (PDMS).

Default pipeline:
  --decision-policy rule_maker
  --planning-policy lattice
  --control-policy pid

Rendering strategy (matches collect_expert.py):
  - use_render=False — no 3D Panda3D window
  - env.render(mode="top_down", window=False, ...) for frame capture
  - per-vehicle labels (ego1, ego2, ...) overlaid on each frame
  - videos written via mediapy

Evaluation (--evaluate):
  - The per-step selected planner trajectory of each agent is scored with
    evaluation/platoon_performance.compute_pdms_reward_batch
    (and compute_pairwise_formation_reward for follower formation terms),
    using a chain leader assignment: agent_i's leader is agent_{i-1}.
  - Per-episode JSON/PNG results are written under
    {output_root}/{scenario_id}/metrices/

Usage:
    # single scenario, with evaluation
    python -m evaluation.preview_and_evaluation \\
        --scenario-id S5_hard_brake_lead --local-route R3_mainline_straight \\
        --num-agents 3 --num-episodes 2 --evaluate \\
        --output-root /tmp/platoon_preview

    # all configured scenarios
    python -m evaluation.preview_and_evaluation \\
        --all-scenarios --num-agents 3 --num-episodes 2 --evaluate \\
        --output-root /tmp/platoon_preview
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from evaluation.evaluation_helper import (
    FormationUnlockTracker,
    _aggregate_pdms_across_episodes,
    _aggregate_pdms_records,
    _collect_episode_step_record,
    _extract_team_pdms_series,
    _plot_average_episode_pdms,
    _plot_episode_pdms,
    _plot_episode_results,
    _publication_agent_color,
    _save_episode_metrics_json,
    _save_publication_figure,
    summarize_formation_unlock_records,
)
from evaluation.platoon_performance import (
    build_platoon_metric_params,
    compute_pairwise_formation_reward,
    compute_pdms_reward_batch,
)
from models.controller.PIDController import _world_trajectory_to_ego_local
from models.decisioner.rule_decisioner import select_controller_by_formation
from expert_dataset.collect_joint_bev import (
    JointBEVSampleBuilder,
    JointCollectionError,
    JointStepRejected,
    RulePlannerExpert,
    SensorlessJointBEVPlatoonEnv,
    simulator_decision_dt_s,
)
from scenarios.bev_round13_contract import (
    PRIMARY_S5_S9_SCENARIOS,
    deterministic_initial_speed_km_h,
)
from tools.topdown_view import (
    capture_topdown_frame as _capture_topdown_frame,
    overlay_planning_debug as _overlay_planning_debug,
    overlay_platoon_labels as _overlay_platoon_labels,
    overlay_rule_maker_debug as _overlay_rule_maker_debug,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_SCENARIO_ID = "S7_ego_merge_from_ramp"
DEFAULT_NUM_AGENTS = 3
DEFAULT_NUM_EPISODES = 3
DEFAULT_OUTPUT_ROOT = Path("/media/kong/Elements_SE/Diffusion_Data/outputs/run_results")
DEFAULT_TRAFFIC_DENSITY = 0
DEFAULT_START_SEED = 11
DEFAULT_VIDEO_FPS = 10
DEFAULT_HORIZON = 600
DEFAULT_DECISION_POLICY = "rule_maker"
DEFAULT_PLANNING_POLICY = "lattice"
DEFAULT_CONTROL_POLICY = "adaptive"
DEFAULT_ROUND13_73_SEEDS = (17, 23, 31, 47, 59, 71, 83, 97, 109, 127)


# ---------------------------------------------------------------------------
# Rendering helpers  (implementations live in tools/topdown_view.py)
# ---------------------------------------------------------------------------


def _write_video(path: Path, frames: list, fps: int) -> None:
    if not frames:
        return
    import mediapy
    path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(path), frames, fps=int(fps))


def _semantic_bev_to_rgb(bev: np.ndarray) -> np.ndarray:
    """Render one strict eight-channel semantic BEV as an RGB diagnostic."""

    value = np.asarray(bev)
    if value.shape != (8, 256, 256) or value.dtype != np.uint8:
        raise ValueError("semantic BEV frame must be uint8 [8,256,256]")
    rgb = np.zeros((256, 256, 3), dtype=np.float32)

    def blend(channel: int, color: tuple[int, int, int], alpha: float) -> None:
        occupancy = value[channel].astype(np.float32) / 255.0
        weight = np.clip(alpha * occupancy, 0.0, 1.0)[..., None]
        rgb[:] = rgb * (1.0 - weight) + np.asarray(
            color, dtype=np.float32
        ) * weight

    blend(0, (70, 70, 70), 0.85)    # drivable
    blend(1, (235, 235, 235), 0.95) # lane geometry
    blend(2, (40, 210, 80), 0.90)   # navigation route
    blend(7, (255, 80, 20), 0.45)   # background t-1.0
    blend(6, (255, 135, 20), 0.60)  # background t-0.5
    blend(5, (255, 220, 30), 0.85)  # background t
    blend(4, (210, 40, 220), 0.95)  # platoon vehicles
    blend(3, (30, 170, 255), 1.00)  # ego history
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def _semantic_bev_mosaic(bev: np.ndarray | None) -> np.ndarray:
    if bev is None:
        return np.zeros((256, 256 * 3 + 8, 3), dtype=np.uint8)
    value = np.asarray(bev)
    if value.shape != (3, 8, 256, 256) or value.dtype != np.uint8:
        raise ValueError("joint semantic BEV must be uint8 [3,8,256,256]")
    separator = np.full((256, 4, 3), 20, dtype=np.uint8)
    return np.concatenate(
        [
            _semantic_bev_to_rgb(value[0]),
            separator,
            _semantic_bev_to_rgb(value[1]),
            separator,
            _semantic_bev_to_rgb(value[2]),
        ],
        axis=1,
    )


def _fatal_info_reason(
    info: Mapping[str, Mapping[str, object]] | None,
    agent_ids: Sequence[str],
) -> tuple[str | None, list[str], list[str]]:
    crash_agents: list[str] = []
    out_agents: list[str] = []
    crash_priority = (
        "crash_sidewalk",
        "crash_vehicle",
        "crash_object",
        "crash_building",
        "crash_human",
        "crash",
    )
    crash_reason: str | None = None
    for agent_id in agent_ids:
        agent_info = (info or {}).get(agent_id, {})
        if not isinstance(agent_info, Mapping):
            continue
        triggered = [
            key for key in crash_priority if bool(agent_info.get(key, False))
        ]
        if triggered:
            crash_agents.append(agent_id)
            key = triggered[0]
            if (
                crash_reason is None
                or crash_priority.index(key)
                < crash_priority.index(crash_reason)
            ):
                crash_reason = key
        if any(
            bool(agent_info.get(key, False))
            for key in ("out_of_road", "out_of_route")
        ):
            out_agents.append(agent_id)
    if crash_agents:
        return crash_reason or "crash", crash_agents, out_agents
    if out_agents:
        return "out_of_road", crash_agents, out_agents
    return None, crash_agents, out_agents


def _write_trajectory_npz(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(records)
    steps = np.asarray(
        [int(record["step"]) for record in records], dtype=np.int32
    )
    actions = np.full((count, 3), -9, dtype=np.int8)
    trajectories = np.full((count, 3, 8, 3), np.nan, dtype=np.float32)
    controls = np.full((count, 3, 2), np.nan, dtype=np.float32)
    gt_mode = np.full((count, 3), -1, dtype=np.int64)
    mode_valid_mask = np.zeros((count, 3, 10), dtype=np.bool_)
    collection_ready = np.zeros((count,), dtype=np.bool_)
    for row, record in enumerate(records):
        actions[row] = np.asarray(record["rule_actions"], dtype=np.int8)
        trajectories[row] = np.asarray(
            record["trajectories_world"], dtype=np.float32
        )
        controls[row] = np.asarray(record["controls"], dtype=np.float32)
        collection_ready[row] = bool(record.get("collection_ready", False))
        if record.get("gt_mode") is not None:
            gt_mode[row] = np.asarray(record["gt_mode"], dtype=np.int64)
        if record.get("mode_valid_mask") is not None:
            mode_valid_mask[row] = np.asarray(
                record["mode_valid_mask"], dtype=np.bool_
            )
    np.savez_compressed(
        path,
        step=steps,
        rule_actions=actions,
        trajectories_world=trajectories,
        controls=controls,
        collection_ready=collection_ready,
        gt_mode=gt_mode,
        mode_valid_mask=mode_valid_mask,
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.percentile(array, percentile))
        if array.size
        else 0.0
    )


def _aggregate_expert_episode_summaries(
    episodes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    count = len(episodes)
    attempted = sum(int(value.get("planning_attempt_steps", 0)) for value in episodes)
    planned = sum(int(value.get("native_joint_success_steps", 0)) for value in episodes)
    ready_attempted = sum(
        int(value.get("collection_ready_attempt_steps", 0))
        for value in episodes
    )
    ready = sum(
        int(value.get("collection_ready_success_steps", 0))
        for value in episodes
    )
    wall = [float(value.get("rollout_wall_time_s", 0.0)) for value in episodes]
    simulated = [
        float(value.get("simulated_duration_s", 0.0)) for value in episodes
    ]
    return {
        "episode_count": count,
        "collision_rate": (
            sum(bool(value.get("collision", False)) for value in episodes)
            / max(count, 1)
        ),
        "out_of_road_rate": (
            sum(bool(value.get("out_of_road", False)) for value in episodes)
            / max(count, 1)
        ),
        "persistable_episode_rate": (
            sum(bool(value.get("persistable", False)) for value in episodes)
            / max(count, 1)
        ),
        "native_joint_planning_success_rate": planned / max(attempted, 1),
        "collection_ready_step_success_rate": ready / max(ready_attempted, 1),
        "planning_attempt_steps": attempted,
        "native_joint_success_steps": planned,
        "collection_ready_attempt_steps": ready_attempted,
        "collection_ready_success_steps": ready,
        "rollout_wall_time_s": {
            "mean": float(np.mean(wall)) if wall else 0.0,
            "p50": _percentile(wall, 50),
            "p95": _percentile(wall, 95),
        },
        "simulated_duration_s": {
            "mean": float(np.mean(simulated)) if simulated else 0.0,
            "p50": _percentile(simulated, 50),
            "p95": _percentile(simulated, 95),
        },
        "failure_reasons": dict(
            Counter(
                str(value.get("failure_reason"))
                for value in episodes
                if value.get("failure_reason")
            )
        ),
    }


def _write_episode_summary_csv(
    path: Path, episodes: Sequence[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "episode",
        "seed",
        "simulator_steps",
        "simulated_duration_s",
        "rollout_wall_time_s",
        "planning_attempt_steps",
        "native_joint_success_steps",
        "native_joint_planning_success_rate",
        "collection_ready_attempt_steps",
        "collection_ready_success_steps",
        "collection_ready_step_success_rate",
        "collision",
        "collision_agents",
        "out_of_road",
        "out_of_road_agents",
        "persistable",
        "failure_reason",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for episode in episodes:
            row = {key: episode.get(key) for key in fieldnames}
            for key in ("collision_agents", "out_of_road_agents"):
                row[key] = ",".join(str(value) for value in row[key] or ())
            writer.writerow(row)


def _expert_debug_snapshot(
    env,
    agent_ids: Sequence[str],
) -> dict[str, object]:
    rule_debug = getattr(env, "_preview_rule_maker_debug", None) or {}
    planning = getattr(env, "_preview_planning_debug", None) or {}
    planner_debug = planning.get("planner_debug", {}) if isinstance(planning, Mapping) else {}
    rule_fields = (
        "formation_locked",
        "risk_triggered",
        "best_actions",
        "best_score",
        "dynamic_roles",
        "forced_lane_decision",
        "forced_lane_wait_info",
        "risk_info",
    )
    joint_fields = (
        "fallback_used",
        "fallback_reason",
        "missing_agents",
        "combination_count",
        "pairwise_conflict_count",
        "pairwise_conflict_pair_count",
        "pairwise_conflict_counts_by_pair",
        "selected_indices",
        "selected_score",
        "planning_time_ms",
    )
    agent_fields = (
        "action",
        "candidate_count",
        "generated_valid_candidate_count",
        "raw_candidate_count",
        "kinematic_rejection_count",
        "corridor_rejection_count",
        "road_rejection_count",
        "background_collision_rejection_count",
        "lane_end_rejection_count",
        "collision_rejections_by_object",
        "kinematic_rejections_by_reason",
        "lane_end_restricted",
        "source_lane_remaining_m",
        "lane_end_deadline_s",
        "safe_corridor_by_duration_m",
        "fallback_used",
        "fallback_reason",
    )
    agents = getattr(env, "agents", {}) or {}
    agent_states = {}
    for agent_id in agent_ids:
        vehicle = agents.get(agent_id)
        if vehicle is None:
            continue
        agent_states[agent_id] = {
            "position": np.asarray(
                getattr(vehicle, "position", (np.nan, np.nan)),
                dtype=np.float64,
            )[:2].tolist(),
            "heading": float(getattr(vehicle, "heading_theta", np.nan)),
            "speed_km_h": float(getattr(vehicle, "speed_km_h", np.nan)),
            "lane_index": list(getattr(vehicle, "lane_index", ()) or ()),
        }
    return {
        "rule_maker": {
            key: rule_debug.get(key)
            for key in rule_fields
            if isinstance(rule_debug, Mapping) and key in rule_debug
        },
        "planner_joint": {
            key: (planner_debug.get("_joint", {}) or {}).get(key)
            for key in joint_fields
            if isinstance(planner_debug, Mapping)
            and key in (planner_debug.get("_joint", {}) or {})
        },
        "planner_agents": {
            agent_id: {
                key: (planner_debug.get(agent_id, {}) or {}).get(key)
                for key in agent_fields
                if isinstance(planner_debug, Mapping)
                and key in (planner_debug.get(agent_id, {}) or {})
            }
            for agent_id in agent_ids
        },
        "agent_states": agent_states,
    }


# ---------------------------------------------------------------------------
# Pipeline factories
# ---------------------------------------------------------------------------

def _normalize_decisions(decisions: dict[str, object]) -> dict[str, dict]:
    normalized: dict[str, dict] = {}
    for agent_id, decision in (decisions or {}).items():
        if isinstance(decision, dict):
            normalized[agent_id] = {
                "action": int(decision.get("action", 0)),
                "target_point": np.asarray(decision.get("target_point", [15.0, 0.0]), dtype=np.float32).reshape(2),
            }
        else:
            normalized[agent_id] = {
                "action": 0,
                "target_point": np.asarray(decision, dtype=np.float32).reshape(2),
            }
    return normalized


def _build_planning_debug(
    *,
    planning_policy: str,
    agent_ids: list[str],
    trajectories: dict[str, np.ndarray],
    planner_debug: dict | None,
) -> dict:
    planner_debug = planner_debug or {}
    candidates_by_agent: dict[str, list[dict]] = {}
    for agent_id in agent_ids:
        agent_debug = planner_debug.get(agent_id, {}) if isinstance(planner_debug, dict) else {}
        candidates = list(agent_debug.get("candidates", []) or [])
        if not candidates and agent_id in trajectories:
            candidates = [
                {
                    "score": 0.0,
                    "selected": True,
                    "trajectory_world": np.asarray(trajectories[agent_id], dtype=np.float32).tolist(),
                }
            ]
        candidates_by_agent[agent_id] = candidates
    return {
        "planning_policy": planning_policy,
        "coordinate_frame": "world",
        "agent_ids": list(agent_ids),
        "trajectories_by_agent": {
            agent_id: np.asarray(trajectory, dtype=np.float32).tolist()
            for agent_id, trajectory in trajectories.items()
        },
        "candidates_by_agent": candidates_by_agent,
        "planner_debug": planner_debug,
    }


def build_rule_planner_pipeline_factory(
    decision_policy: str = DEFAULT_DECISION_POLICY,
    planning_policy: str = DEFAULT_PLANNING_POLICY,
    control_policy: str = DEFAULT_CONTROL_POLICY,
) -> Callable:
    if decision_policy != "rule_maker":
        raise ValueError(f"Unsupported decision_policy: {decision_policy!r}")
    if planning_policy != "lattice":
        raise ValueError(f"Unsupported planning_policy: {planning_policy!r}")
    if control_policy not in ("pid", "adaptive"):
        raise ValueError(f"Unsupported control_policy: {control_policy!r}")

    def factory(env, agent_ids: list[str], seed: int) -> Callable:
        del seed
        if tuple(agent_ids) != ("agent0", "agent1", "agent2"):
            raise ValueError(
                "strict rule-planner evaluation requires agent0/1/2"
            )
        expert = RulePlannerExpert(env, agent_ids)

        def action_fn(env) -> dict[str, np.ndarray]:
            active_agent_ids = [aid for aid in agent_ids if aid in getattr(env, "agents", {})]
            if not active_agent_ids:
                setattr(env, "_preview_rule_maker_debug", None)
                setattr(env, "_preview_planning_debug", None)
                return {}

            try:
                expert_step = expert.plan(env)
            except JointCollectionError:
                setattr(
                    env,
                    "_preview_rule_maker_debug",
                    expert.rule_maker.get_last_debug() or {},
                )
                setattr(
                    env,
                    "_preview_planning_debug",
                    _build_planning_debug(
                        planning_policy=planning_policy,
                        agent_ids=active_agent_ids,
                        trajectories={},
                        planner_debug=expert.planner.get_last_debug() or {},
                    ),
                )
                raise

            planning_debug = _build_planning_debug(
                planning_policy=planning_policy,
                agent_ids=active_agent_ids,
                trajectories=dict(expert_step.trajectories_world),
                planner_debug=expert.planner.get_last_debug() or {},
            )
            setattr(
                env,
                "_preview_rule_maker_debug",
                expert.rule_maker.get_last_debug() or {},
            )
            setattr(env, "_preview_planning_debug", planning_debug)
            setattr(env, "_preview_expert_step", expert_step)
            setattr(
                env,
                "_preview_control_debug",
                getattr(
                    (
                        select_controller_by_formation(
                            expert.rule_maker,
                            expert.pid_controller,
                            expert.lqr_controller,
                        )
                        if control_policy == "adaptive"
                        else expert.pid_controller
                    ),
                    "get_last_debug",
                    lambda: None,
                )(),
            )
            return dict(expert_step.controls)

        setattr(action_fn, "expert", expert)
        return action_fn

    return factory


# Backward-compatible private alias for callers/tests that used the original
# preview-only helper before it became the shared rule-planner expert API.
_build_pipeline_factory = build_rule_planner_pipeline_factory


# ---------------------------------------------------------------------------
# PDMS evaluation helpers (metrics/platoon_performance.py integration)
# ---------------------------------------------------------------------------

def _rebuild_heading_from_xy(traj: np.ndarray) -> np.ndarray:
    """Reconstruct heading channel from atan2(dy, dx), matching training PDMS computation."""
    xy = traj[:, :2]
    dxy = np.diff(xy, axis=0)
    h_tail = np.arctan2(dxy[:, 1], dxy[:, 0]).astype(np.float32)
    heading = np.concatenate([h_tail[:1], h_tail])
    result = traj.copy()
    result[:, 2] = heading
    return result


def _select_planned_trajectory(env, agent_id: str) -> Optional[np.ndarray]:
    """Return the world-frame trajectory selected by the planning stage."""
    debug = getattr(env, "_preview_planning_debug", None) or {}
    trajectory = (debug.get("trajectories_by_agent", {}) or {}).get(agent_id)
    if trajectory is not None:
        return np.asarray(trajectory, dtype=np.float32)
    for candidate in (debug.get("candidates_by_agent", {}) or {}).get(agent_id, []) or []:
        if bool(candidate.get("selected", False)) and candidate.get("trajectory_world") is not None:
            return np.asarray(candidate["trajectory_world"], dtype=np.float32)
    return None


def _compute_step_pdms(env, agent_ids: list[str], debug, info, pdms_params: dict) -> dict[str, dict[str, float]]:
    """
    Score each agent's selected planner trajectory with the PDMS reward.

    Leader assignment is chain-based: agent_ids[i]'s reference leader is
    agent_ids[i - 1] (agent_ids[0] is treated as the platoon head, is_leader=True).
    Agents without a valid selected trajectory this step (or whose leader lacks
    one) are skipped.
    """
    del debug
    results: dict[str, dict[str, float]] = {}
    traj_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for idx, agent_id in enumerate(agent_ids):
        vehicle = env.agents.get(agent_id)
        if vehicle is None:
            continue

        trajectory_world = _select_planned_trajectory(env, agent_id)
        if trajectory_world is None:
            continue
        trajectory_local = _world_trajectory_to_ego_local(vehicle, trajectory_world)
        if trajectory_local.shape[0] < 2:
            continue

        pose = np.array(
            [vehicle.position[0], vehicle.position[1], vehicle.heading_theta],
            dtype=np.float64,
        )
        trajectory_local3 = _rebuild_heading_from_xy(trajectory_local[:, :3])
        traj_cache[agent_id] = (trajectory_local3, pose)

        is_leader = idx == 0
        prev_traj3 = prev_pose = None
        formation_lon = formation_lat = np.zeros(1, dtype=np.float32)
        if not is_leader:
            leader_data = traj_cache.get(agent_ids[idx - 1])
            if leader_data is None:
                continue
            prev_traj3, prev_pose = leader_data
            formation_lon, formation_lat = compute_pairwise_formation_reward(
                trajectory_local3[None],
                pose,
                prev_traj3,
                prev_pose,
                desired_gap_m=pdms_params["desired_gap_m"],
                lon_decay_m=pdms_params["lon_decay_m"],
                lat_decay_m=pdms_params["lat_decay_m"],
                progress_s_max=pdms_params["progress_s_max"],
                waypoint_decay_gamma=pdms_params["waypoint_decay_gamma"],
            )

        agent_info = info.get(agent_id, {}) if isinstance(info, dict) else {}
        env_crashed = bool(agent_info.get("crash", False) or agent_info.get("crash_vehicle", False))
        env_out_of_road = bool(agent_info.get("out_of_road", False))

        _, batch_debug = compute_pdms_reward_batch(
            trajectory_local3[None],
            pose,
            prev_traj3,
            prev_pose,
            trajectory_local3,
            is_leader,
            formation_lon,
            formation_lat,
            pdms_params,
            env_crashed=env_crashed,
            env_out_of_road=env_out_of_road,
        )

        results[agent_id] = {
            "reward": float(batch_debug["reward"][0]),
            "progress": float(batch_debug["progress"][0]),
            "formation_lon": float(batch_debug["formation_lon"][0]),
            "formation_lat": float(batch_debug["formation_lat"][0]),
            "speed": float(batch_debug["speed"][0]),
            "comfort": float(batch_debug["comfort"][0]),
            "consistency": float(batch_debug["consistency"][0]),
            "gate": float(batch_debug["gate"][0]),
        }

    return results


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _episode_failed_immediately(frames, terminated, truncated, info) -> bool:
    if len(frames) > 1:
        return False
    if terminated is None and truncated is None:
        return False
    if truncated is not None and bool(truncated.get("__all__", False)):
        return False
    if terminated is not None and bool(terminated.get("__all__", False)):
        for agent_info in (info or {}).values():
            if bool(agent_info.get("crash", False) or agent_info.get("crash_vehicle", False)):
                return True
    return False


def _agent_has_crash(agent_info: Mapping[str, object]) -> bool:
    crash_keys = ("crash", "crash_vehicle", "crash_object", "crash_building", "crash_human")
    return any(bool(agent_info.get(key, False)) for key in crash_keys)


def _format_episode_stop_reason(
    step_idx: int,
    terminated: Optional[Mapping[str, bool]],
    truncated: Optional[Mapping[str, bool]],
    info: Optional[Mapping[str, Mapping[str, object]]],
) -> str:
    agent_info_by_id = info or {}
    crash_agents = [
        agent_id
        for agent_id, agent_info in agent_info_by_id.items()
        if _agent_has_crash(agent_info)
    ]
    out_of_road_agents = [
        agent_id
        for agent_id, agent_info in agent_info_by_id.items()
        if bool(agent_info.get("out_of_road", False))
    ]
    arrive_agents = [
        agent_id
        for agent_id, agent_info in agent_info_by_id.items()
        if bool(agent_info.get("arrive_dest", False))
    ]

    reasons: list[str] = []
    if crash_agents:
        reasons.append("crash")
    if out_of_road_agents:
        reasons.append("out_of_road")
    if arrive_agents:
        reasons.append("arrive_dest")
    if not reasons:
        if truncated is not None and bool(truncated.get("__all__", False)):
            reasons.append("truncated")
        elif terminated is not None and bool(terminated.get("__all__", False)):
            reasons.append("terminated")
        else:
            reasons.append("unknown")

    parts = [f"Episode stopped: step={int(step_idx)} reason={','.join(reasons)}"]
    if crash_agents:
        parts.append(f"crash_agents={','.join(crash_agents)}")
    if out_of_road_agents:
        parts.append(f"out_of_road_agents={','.join(out_of_road_agents)}")
    if arrive_agents:
        parts.append(f"arrive_agents={','.join(arrive_agents)}")
    return " ".join(parts)


def _run_single_episode(
    env, agent_ids, lead_id, heading_up, seed, action_fn_factory,
    pdms_params: Optional[dict] = None,
    platoon_metrics: Optional[object] = None,
    pdms_records: Optional[list] = None,
    episode_step_records: Optional[list] = None,
    unlock_tracker: FormationUnlockTracker | None = None,
    sample_builder: JointBEVSampleBuilder | None = None,
    semantic_bev_frames: Optional[list[np.ndarray]] = None,
    trajectory_records: Optional[list[dict[str, object]]] = None,
    episode_diagnostics: Optional[dict[str, object]] = None,
):
    spawn_manager = getattr(getattr(env, "engine", None), "spawn_manager", None)
    if spawn_manager is not None and hasattr(spawn_manager, "set_episode_spawn_seed"):
        spawn_manager.set_episode_spawn_seed(int(seed))
    obs = env.reset()
    action_fn = action_fn_factory(env, agent_ids, seed)
    strict_expert_chain = hasattr(action_fn, "expert")
    builder = sample_builder or JointBEVSampleBuilder(agent_ids)
    builder.reset()
    frames: list[np.ndarray] = []
    terminated = truncated = info = None
    dt_s = simulator_decision_dt_s(env)
    rollout_started = time.perf_counter()
    planning_times_s: list[float] = []
    planning_attempt_steps = 0
    native_joint_success_steps = 0
    collection_ready_attempt_steps = 0
    collection_ready_success_steps = 0
    rejection_counts: Counter[str] = Counter()
    rejection_details: Counter[str] = Counter()
    rule_action_counts: Counter[str] = Counter()
    failure_reason: str | None = None
    collision_agents: list[str] = []
    out_of_road_agents: list[str] = []
    simulator_steps = 0

    if platoon_metrics is not None:
        platoon_metrics.start_episode()
    if pdms_records is not None:
        pdms_records.clear()
    if episode_step_records is not None:
        episode_step_records.clear()

    env_config = dict(getattr(env, "config", {}) or {})
    physics_world_step_size = float(env_config.get("physics_world_step_size", 2e-2))
    decision_repeat = int(env_config.get("decision_repeat", 5))
    low_level_step_dt = physics_world_step_size * max(decision_repeat, 1)
    previous_velocity_mps: dict[str, np.ndarray] = {}

    for _step in range(1000):
        try:
            builder.capture_state(env, timestamp_s=_step * dt_s)
        except JointCollectionError as exc:
            failure_reason = exc.reason_code
            rejection_counts[exc.reason_code] += 1
            rejection_details[str(exc)] += 1
            break

        planning_attempt_steps += 1
        planning_started = time.perf_counter()
        try:
            actions = action_fn(env)
        except JointCollectionError as exc:
            planning_times_s.append(time.perf_counter() - planning_started)
            failure_reason = exc.reason_code
            rejection_counts[exc.reason_code] += 1
            rejection_details[str(exc)] += 1
            planning_debug = getattr(env, "_preview_planning_debug", None)
            rule_maker_debug = getattr(env, "_preview_rule_maker_debug", None)
            frame = _capture_topdown_frame(
                env,
                lead_id,
                heading_up,
                agent_ids,
                rule_maker_debug=rule_maker_debug,
                planning_debug=planning_debug,
            )
            if frame is not None:
                frames.append(frame)
            if semantic_bev_frames is not None:
                semantic_bev_frames.append(_semantic_bev_mosaic(None))
            break
        planning_times_s.append(time.perf_counter() - planning_started)
        if not actions:
            failure_reason = "rule_maker_no_action"
            break
        native_joint_success_steps += 1
        expert_step = getattr(env, "_preview_expert_step", None)
        if strict_expert_chain and expert_step is None:
            failure_reason = "preview_expert_step_missing"
            break

        sample = None
        model_inputs = None
        if strict_expert_chain and builder.history_ready():
            collection_ready_attempt_steps += 1
            try:
                sample = builder.build_sample(env, expert_step)
                model_inputs = sample
                collection_ready_success_steps += 1
            except JointStepRejected as exc:
                rejection_counts[exc.reason_code] += 1
                rejection_details[str(exc)] += 1
                try:
                    model_inputs = builder.build_model_inputs(env)
                except JointCollectionError:
                    model_inputs = None
        if semantic_bev_frames is not None:
            semantic_bev_frames.append(
                _semantic_bev_mosaic(
                    None if model_inputs is None else np.asarray(model_inputs.bev)
                )
            )

        if expert_step is not None:
            ordered_rule_actions = [
                int(expert_step.rule_actions[agent_id]) for agent_id in agent_ids
            ]
            for agent_id, action in zip(agent_ids, ordered_rule_actions):
                rule_action_counts[f"{agent_id}:{action:+d}"] += 1
        else:
            ordered_rule_actions = []
        if trajectory_records is not None and expert_step is not None:
            trajectory_records.append(
                {
                    "step": int(_step),
                    "rule_actions": np.asarray(
                        ordered_rule_actions, dtype=np.int8
                    ),
                    "trajectories_world": np.stack(
                        [
                            np.asarray(
                                expert_step.trajectories_world[agent_id],
                                dtype=np.float32,
                            )
                            for agent_id in agent_ids
                        ]
                    ),
                    "controls": np.stack(
                        [
                            np.asarray(
                                expert_step.controls[agent_id], dtype=np.float32
                            )
                            for agent_id in agent_ids
                        ]
                    ),
                    "collection_ready": sample is not None,
                    "gt_mode": None if sample is None else sample.gt_mode,
                    "mode_valid_mask": (
                        None if sample is None else sample.mode_valid_mask
                    ),
                }
            )
        planning_debug = getattr(env, "_preview_planning_debug", None)
        rule_maker_debug = getattr(env, "_preview_rule_maker_debug", None)
        if unlock_tracker is not None:
            unlock_tracker.update(
                step_idx=_step,
                formation_locked=bool((rule_maker_debug or {}).get("formation_locked", True)),
            )

        frame = _capture_topdown_frame(
            env, lead_id, heading_up, agent_ids,
            rule_maker_debug=rule_maker_debug,
            planning_debug=planning_debug,
        )
        if frame is not None:
            frames.append(frame)

        obs, reward, terminated, truncated, info = env.low_level_step(actions)
        simulator_steps += 1

        if platoon_metrics is not None:
            platoon_metrics.update(info)
        if pdms_params is not None and pdms_records is not None:
            debug = rule_maker_debug
            step_pdms = _compute_step_pdms(env, agent_ids, debug, info, pdms_params)
            if step_pdms:
                pdms_records.append(step_pdms)
        else:
            step_pdms = {}

        if episode_step_records is not None:
            episode_step_records.append(
                _collect_episode_step_record(
                    env=env,
                    agent_ids=agent_ids,
                    step_idx=_step,
                    actions=actions,
                    info=info,
                    pdms=step_pdms,
                    planning_debug=planning_debug,
                    previous_speed_mps=previous_velocity_mps,
                    dt=low_level_step_dt,
                )
            )

        fatal_reason, crash_now, out_now = _fatal_info_reason(info, agent_ids)
        if crash_now:
            collision_agents = sorted(set(collision_agents).union(crash_now))
        if out_now:
            out_of_road_agents = sorted(
                set(out_of_road_agents).union(out_now)
            )
        if fatal_reason is not None:
            failed_agents = (
                crash_now if fatal_reason.startswith("crash") else out_now
            )
            suffix = ",".join(failed_agents)
            failure_reason = f"{fatal_reason}:{suffix}" if suffix else fatal_reason
            print(_format_episode_stop_reason(_step, terminated, truncated, info))
            break

        if terminated.get("__all__", False) or truncated.get("__all__", False):
            print(_format_episode_stop_reason(_step, terminated, truncated, info))
            break

    if platoon_metrics is not None and not _episode_failed_immediately(frames, terminated, truncated, info):
        platoon_metrics.end_episode()

    scenario_summary: Mapping[str, object] = {}
    orchestrator = getattr(env, "_scenario_orchestrator", None)
    if orchestrator is not None and hasattr(orchestrator, "get_episode_summary"):
        scenario_summary = dict(orchestrator.get_episode_summary() or {})
    if failure_reason is None and scenario_summary:
        if not bool(scenario_summary.get("scenario_realized", False)):
            failure_reason = "scenario_not_realized"
    if failure_reason is None and rejection_counts:
        reason, _ = sorted(
            rejection_counts.items(), key=lambda item: (-item[1], item[0])
        )[0]
        failure_reason = f"joint_step_rejected:{reason}"

    rollout_wall_time_s = time.perf_counter() - rollout_started
    if episode_diagnostics is not None:
        episode_diagnostics.clear()
        episode_diagnostics.update(
            {
                "seed": int(seed),
                "initial_speed_km_h": float(
                    getattr(env, "config", {}).get(
                        "initial_speed_km_h", np.nan
                    )
                ),
                "simulator_steps": int(simulator_steps),
                "simulated_duration_s": float(simulator_steps * dt_s),
                "rollout_wall_time_s": float(rollout_wall_time_s),
                "planning_attempt_steps": int(planning_attempt_steps),
                "native_joint_success_steps": int(native_joint_success_steps),
                "native_joint_planning_success_rate": (
                    native_joint_success_steps / max(planning_attempt_steps, 1)
                ),
                "collection_ready_attempt_steps": int(
                    collection_ready_attempt_steps
                ),
                "collection_ready_success_steps": int(
                    collection_ready_success_steps
                ),
                "collection_ready_step_success_rate": (
                    collection_ready_success_steps
                    / max(collection_ready_attempt_steps, 1)
                ),
                "planning_time_ms": {
                    "mean": (
                        1000.0 * float(np.mean(planning_times_s))
                        if planning_times_s
                        else 0.0
                    ),
                    "p50": 1000.0 * _percentile(planning_times_s, 50),
                    "p95": 1000.0 * _percentile(planning_times_s, 95),
                },
                "collision": bool(collision_agents),
                "collision_agents": collision_agents,
                "out_of_road": bool(out_of_road_agents),
                "out_of_road_agents": out_of_road_agents,
                "failure_reason": failure_reason,
                "rejection_counts": dict(rejection_counts),
                "rejection_details": dict(rejection_details),
                "rule_action_counts": dict(rule_action_counts),
                "scenario_summary": dict(scenario_summary),
                "final_debug": _expert_debug_snapshot(env, agent_ids),
                "persistable": bool(
                    failure_reason is None
                    and not collision_agents
                    and not out_of_road_agents
                    and collection_ready_success_steps > 0
                    and native_joint_success_steps == planning_attempt_steps
                ),
            }
        )

    return frames, terminated, truncated, info


# ---------------------------------------------------------------------------
# Route / scenario helpers
# ---------------------------------------------------------------------------

def _pick_local_route(scenario_id: str) -> str:
    from scenarios.definitions import SCENARIO_BY_ID
    defn = SCENARIO_BY_ID.get(scenario_id)
    if defn is None:
        raise ValueError(f"Unknown scenario: {scenario_id}")
    routes = defn.allowed_local_routes
    if not routes:
        raise ValueError(f"Scenario {scenario_id} has no allowed_local_routes")
    return routes[0]


def _all_scenario_route_pairs() -> list[tuple[str, str]]:
    """Return [(scenario_id, first_route), ...] for all defined scenarios, sorted."""
    from scenarios.definitions import SCENARIO_BY_ID
    pairs = []
    for sid, defn in sorted(SCENARIO_BY_ID.items()):
        routes = defn.allowed_local_routes
        if routes:
            pairs.append((sid, routes[0]))
    return pairs


# ---------------------------------------------------------------------------
# Main preview runner
# ---------------------------------------------------------------------------

def run_scenario(
    scenario_id: str,
    local_route: Optional[str],
    num_agents: int,
    num_episodes: int,
    output_root: Path,
    heading_up: bool,
    traffic_density: float,
    start_seed: int,
    fps: int,
    decision_policy: str = DEFAULT_DECISION_POLICY,
    planning_policy: str = DEFAULT_PLANNING_POLICY,
    control_policy: str = DEFAULT_CONTROL_POLICY,
    max_episode_retries: int = 1,
    env_factory: Optional[Callable] = None,
    evaluate: bool = False,
    metric_params: Optional[dict] = None,
    horizon: int = DEFAULT_HORIZON,
    seeds: Sequence[int] | None = None,
    save_topdown_video: bool = True,
    save_semantic_bev_video: bool = True,
    save_trajectory_data: bool = True,
) -> Path:
    if local_route is None:
        local_route = _pick_local_route(scenario_id)
    print(
        f"  preview source={Path(__file__).resolve()} "
        f"scenario={scenario_id} local_route={local_route}"
    )

    scenario_root = output_root / scenario_id
    video_dir = scenario_root / "video"
    semantic_video_dir = scenario_root / "semantic_bev_video"
    trajectory_dir = scenario_root / "trajectories"
    if save_topdown_video:
        video_dir.mkdir(parents=True, exist_ok=True)
    if save_semantic_bev_video:
        semantic_video_dir.mkdir(parents=True, exist_ok=True)
    if save_trajectory_data:
        trajectory_dir.mkdir(parents=True, exist_ok=True)

    episode_seeds = (
        tuple(int(value) for value in seeds)
        if seeds is not None
        else tuple(int(start_seed + index) for index in range(num_episodes))
    )
    if len(episode_seeds) != int(num_episodes):
        raise ValueError(
            "num_episodes must exactly match the number of explicit seeds"
        )
    if len(set(episode_seeds)) != len(episode_seeds):
        raise ValueError("episode seeds must be unique")

    env_config = {
        "num_agents": num_agents,
        "use_render": False,
        "use_hybrid_map": True,
        "num_scenarios": 1,
        "traffic_density": traffic_density,
        "scenario_id": scenario_id,
        "local_route": local_route,
        "crash_done": False,
        "out_of_road_done": False,
        "horizon": int(horizon),
    }
    if env_factory is not None:
        env = env_factory(env_config)
    else:
        env = SensorlessJointBEVPlatoonEnv(env_config)
    agent_ids = [f"agent{i}" for i in range(num_agents)]
    lead_id = agent_ids[0]
    action_fn_factory = _build_pipeline_factory(decision_policy, planning_policy, control_policy)

    platoon_metrics = None
    pdms_params: Optional[dict] = None
    if evaluate:
        pdms_params = build_platoon_metric_params(
            {**dict(getattr(env, "config", {}) or {}), **(metric_params or {})}
        )
    metrics_dir = output_root / scenario_id / "metrices"
    all_episode_step_records: list[list[dict]] = []
    formation_unlock_records: list[dict] = []
    expert_episode_summaries: list[dict[str, object]] = []

    try:
        for ep_idx in range(num_episodes):
            frames: list[np.ndarray] = []
            semantic_frames: list[np.ndarray] = []
            trajectory_records: list[dict[str, object]] = []
            episode_diagnostics: dict[str, object] = {}
            used_seed = episode_seeds[ep_idx]
            pdms_records: Optional[list] = [] if pdms_params is not None else None
            episode_step_records: Optional[list] = [] if evaluate else None
            episode_unlock_record: dict | None = None
            for retry in range(max(1, int(max_episode_retries))):
                episode_seed = used_seed + retry
                initial_speed_km_h = deterministic_initial_speed_km_h(
                    scenario_id, episode_seed
                )
                runtime_updates = {
                    "traffic_density": float(traffic_density),
                    "initial_speed_km_h": float(initial_speed_km_h),
                }
                env.config.update(runtime_updates)
                platoon_config = getattr(env, "platoon_config", None)
                if platoon_config is not None:
                    platoon_config.traffic_density = float(traffic_density)
                    platoon_config.initial_speed_km_h = float(
                        initial_speed_km_h
                    )
                global_config = getattr(
                    getattr(env, "engine", None), "global_config", None
                )
                if global_config is not None:
                    global_config.update(runtime_updates)
                semantic_frames.clear()
                trajectory_records.clear()
                episode_diagnostics.clear()
                unlock_tracker = FormationUnlockTracker(ep_idx)
                frames, terminated, truncated, info = _run_single_episode(
                    env, agent_ids, lead_id, heading_up, episode_seed, action_fn_factory,
                    pdms_params=pdms_params,
                    platoon_metrics=platoon_metrics,
                    pdms_records=pdms_records,
                    episode_step_records=episode_step_records,
                    unlock_tracker=unlock_tracker,
                    sample_builder=JointBEVSampleBuilder(agent_ids),
                    semantic_bev_frames=semantic_frames,
                    trajectory_records=trajectory_records,
                    episode_diagnostics=episode_diagnostics,
                )
                episode_unlock_record = unlock_tracker.finalize(len(frames) - 1)
                if not _episode_failed_immediately(frames, terminated, truncated, info):
                    used_seed = episode_seed
                    break
                print(f"  retry ep {ep_idx + 1} seed={episode_seed + 1} (frames={len(frames)})")

            video_path = video_dir / f"episode_{ep_idx:04d}.mp4"
            if not frames:
                frames.append(_semantic_bev_mosaic(None))
            if not semantic_frames:
                semantic_frames.append(_semantic_bev_mosaic(None))
            if save_topdown_video:
                _write_video(video_path, frames, fps)
            semantic_video_path = (
                semantic_video_dir / f"episode_{ep_idx:04d}.mp4"
            )
            if save_semantic_bev_video:
                _write_video(semantic_video_path, semantic_frames, fps)
            trajectory_path = trajectory_dir / f"episode_{ep_idx:04d}.npz"
            if save_trajectory_data:
                _write_trajectory_npz(trajectory_path, trajectory_records)
            print(
                f"  [{scenario_id}] ep {ep_idx + 1}/{num_episodes} "
                f"frames={len(frames)} seed={used_seed} "
                f"planning={episode_diagnostics.get('native_joint_success_steps', 0)}/"
                f"{episode_diagnostics.get('planning_attempt_steps', 0)} "
                f"failure={episode_diagnostics.get('failure_reason')}"
            )
            if episode_unlock_record is None:
                episode_unlock_record = FormationUnlockTracker(ep_idx).finalize()
            formation_unlock_records.append(episode_unlock_record)
            episode_diagnostics.update(
                {
                    "episode": int(ep_idx),
                    "scenario_id": scenario_id,
                    "local_route": local_route,
                    "topdown_video_path": (
                        str(video_path) if save_topdown_video else None
                    ),
                    "semantic_bev_video_path": (
                        str(semantic_video_path)
                        if save_semantic_bev_video
                        else None
                    ),
                    "trajectory_path": (
                        str(trajectory_path) if save_trajectory_data else None
                    ),
                }
            )
            expert_episode_summaries.append(dict(episode_diagnostics))
            _save_episode_metrics_json(
                metrics_dir
                / f"episode_{ep_idx:04d}"
                / "expert_episode.json",
                episode_diagnostics,
            )

            if evaluate:
                episode_pdms = {}
                if pdms_records is not None:
                    episode_pdms = _aggregate_pdms_records(pdms_records)
                episode_name = f"episode_{ep_idx:04d}"
                episode_steps = list(episode_step_records or [])
                all_episode_step_records.append(episode_steps)
                episode_payload = {
                    "scenario_id": scenario_id,
                    "local_route": local_route,
                    "decision_policy": decision_policy,
                    "planning_policy": planning_policy,
                    "control_policy": control_policy,
                    "episode": ep_idx,
                    "seed": used_seed,
                    "video_path": str(video_path),
                    "semantic_bev_video_path": str(semantic_video_path),
                    "trajectory_path": str(trajectory_path),
                    "formation_unlock": episode_unlock_record,
                    "expert_collection": episode_diagnostics,
                    "pdms": episode_pdms,
                    "steps": episode_steps,
                }
                episode_json = metrics_dir / f"{episode_name}" / "metrice.json"
                episode_pdms_png = metrics_dir / f"{episode_name}" / "metrice.png"
                episode_results_dir = metrics_dir / f"{episode_name}" / "results"
                _save_episode_metrics_json(episode_json, episode_payload)
                _plot_episode_pdms(episode_pdms_png, episode_steps)
                _plot_episode_results(episode_results_dir, episode_steps, agent_ids)
    finally:
        try:
            env.close()
        except Exception:
            pass

    if evaluate:
        unlock_summary = summarize_formation_unlock_records(formation_unlock_records)
        _save_episode_metrics_json(metrics_dir / "formation_unlock_summary.json", unlock_summary)
        _plot_average_episode_pdms(metrics_dir / "ave_metrice.png", all_episode_step_records)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    expert_summary = _aggregate_expert_episode_summaries(
        expert_episode_summaries
    )
    expert_summary.update(
        {
            "scenario_id": scenario_id,
            "local_route": local_route,
            "seeds": list(episode_seeds),
            "horizon": int(horizon),
            "decision_policy": decision_policy,
            "planning_policy": planning_policy,
            "control_policy": control_policy,
        }
    )
    _save_episode_metrics_json(
        metrics_dir / "expert_collection_summary.json", expert_summary
    )
    _write_episode_summary_csv(
        metrics_dir / "episode_summary.csv", expert_episode_summaries
    )

    return video_dir


def run_all_scenarios(
    num_agents: int,
    num_episodes: int,
    output_root: Path,
    heading_up: bool,
    traffic_density: float,
    start_seed: int,
    fps: int,
    decision_policy: str = DEFAULT_DECISION_POLICY,
    planning_policy: str = DEFAULT_PLANNING_POLICY,
    control_policy: str = DEFAULT_CONTROL_POLICY,
    evaluate: bool = False,
    metric_params: Optional[dict] = None,
    horizon: int = DEFAULT_HORIZON,
    scenario_pairs: Sequence[tuple[str, str]] | None = None,
    seeds: Sequence[int] | None = None,
    save_topdown_video: bool = True,
    save_semantic_bev_video: bool = True,
    save_trajectory_data: bool = True,
) -> None:
    """Evaluate all defined scenarios (one env per scenario to avoid map conflicts)."""
    pairs = list(
        _all_scenario_route_pairs()
        if scenario_pairs is None
        else scenario_pairs
    )
    print(
        f"\n=== Evaluating {len(pairs)} scenarios with "
        f"{decision_policy}/{planning_policy}/{control_policy} ===\n"
    )

    results: list[tuple[str, str, str]] = []  # (scenario_id, route, status)
    for scenario_id, route in pairs:
        print(f"--- {scenario_id} / {route} ---")
        try:
            video_dir = run_scenario(
                scenario_id=scenario_id,
                local_route=route,
                num_agents=num_agents,
                num_episodes=num_episodes,
                output_root=output_root,
                heading_up=heading_up,
                traffic_density=traffic_density,
                start_seed=start_seed,
                fps=fps,
                decision_policy=decision_policy,
                planning_policy=planning_policy,
                control_policy=control_policy,
                evaluate=evaluate,
                metric_params=metric_params,
                horizon=horizon,
                seeds=seeds,
                save_topdown_video=save_topdown_video,
                save_semantic_bev_video=save_semantic_bev_video,
                save_trajectory_data=save_trajectory_data,
            )
            results.append((scenario_id, route, f"OK  -> {video_dir}"))
        except Exception as exc:
            results.append((scenario_id, route, f"FAIL: {exc}"))
            print(f"  ERROR: {exc}")

    print("\n" + "=" * 70)
    print(f"{'SCENARIO':<40} {'ROUTE':<30} STATUS")
    print("=" * 70)
    for sid, route, status in results:
        print(f"{sid:<40} {route:<30} {status}")
    print("=" * 70)
    ok = sum(1 for _, _, s in results if s.startswith("OK"))
    print(f"\n{ok}/{len(results)} scenarios completed successfully.")

    if evaluate:
        print(f"\nPer-episode metrics are saved under: {output_root}/<scenario_id>/metrices/")

    combined: dict[str, object] = {
        "scenario_count": len(pairs),
        "completed_scenario_count": ok,
        "scenarios": {},
    }
    combined_rows: list[dict[str, object]] = []
    for scenario_id, route in pairs:
        path = (
            output_root
            / scenario_id
            / "metrices"
            / "expert_collection_summary.json"
        )
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        combined["scenarios"][scenario_id] = summary
        row = {
            "scenario_id": scenario_id,
            "local_route": route,
            "episode_count": summary.get("episode_count", 0),
            "collision_rate": summary.get("collision_rate", 0.0),
            "out_of_road_rate": summary.get("out_of_road_rate", 0.0),
            "persistable_episode_rate": summary.get(
                "persistable_episode_rate", 0.0
            ),
            "native_joint_planning_success_rate": summary.get(
                "native_joint_planning_success_rate", 0.0
            ),
            "collection_ready_step_success_rate": summary.get(
                "collection_ready_step_success_rate", 0.0
            ),
            "mean_rollout_wall_time_s": (
                summary.get("rollout_wall_time_s", {}).get("mean", 0.0)
            ),
            "mean_simulated_duration_s": (
                summary.get("simulated_duration_s", {}).get("mean", 0.0)
            ),
            "failure_reasons": json.dumps(
                summary.get("failure_reasons", {}),
                sort_keys=True,
                ensure_ascii=False,
            ),
        }
        combined_rows.append(row)
    _save_episode_metrics_json(
        output_root / "expert_collection_summary.json", combined
    )
    if combined_rows:
        csv_path = output_root / "expert_collection_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=tuple(combined_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(combined_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Platoon scenario top-down video preview + evaluation")
    parser.add_argument("--scenario-id", default=DEFAULT_SCENARIO_ID,
                        help="Scenario ID (ignored when --all-scenarios is set)")
    parser.add_argument("--local-route", default=None)
    parser.add_argument("--all-scenarios", action="store_true",
                        help="Run all defined scenarios sequentially")
    parser.add_argument(
        "--primary-s5-s9",
        action="store_true",
        help="Run the frozen S5-S9 expert collection scenarios and routes",
    )
    parser.add_argument("--num-agents", type=int, default=DEFAULT_NUM_AGENTS)
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--heading-up", default="false")
    parser.add_argument("--traffic-density", type=float, default=DEFAULT_TRAFFIC_DENSITY)
    parser.add_argument("--start-seed", type=int, default=DEFAULT_START_SEED)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit unique episode seeds; count must equal --num-episodes",
    )
    parser.add_argument("--video-fps", type=int, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                        help=f"Override environment horizon / max steps per agent (default: {DEFAULT_HORIZON})")
    parser.add_argument("--decision-policy", default=DEFAULT_DECISION_POLICY, choices=["rule_maker"],
                        help=f"Decision policy (default: {DEFAULT_DECISION_POLICY})")
    parser.add_argument("--planning-policy", default=DEFAULT_PLANNING_POLICY, choices=["lattice"],
                        help=f"Planning policy (default: {DEFAULT_PLANNING_POLICY})")
    parser.add_argument("--control-policy", default=DEFAULT_CONTROL_POLICY, choices=["pid", "adaptive"],
                        help=f"Control policy (default: {DEFAULT_CONTROL_POLICY})")
    # Evaluation
    parser.add_argument("--evaluate", action="store_true",
                        help="Compute PDMS reward from planner trajectories, "
                             "saving per-episode metrices files")
    parser.add_argument(
        "--no-topdown-video",
        action="store_true",
        help="Disable top-down trajectory-overlay video output",
    )
    parser.add_argument(
        "--no-semantic-bev-video",
        action="store_true",
        help="Disable three-role semantic BEV mosaic video output",
    )
    parser.add_argument(
        "--no-trajectory-data",
        action="store_true",
        help="Disable exact expert trajectory NPZ output",
    )
    args = parser.parse_args()
    if args.all_scenarios and args.primary_s5_s9:
        parser.error("--all-scenarios and --primary-s5-s9 are mutually exclusive")

    heading_up = args.heading_up.lower() in ("true", "1", "yes")
    output_root = Path(args.output_root)

    explicit_seeds = args.seeds
    if args.primary_s5_s9 and explicit_seeds is None:
        if args.num_episodes > len(DEFAULT_ROUND13_73_SEEDS):
            parser.error(
                "primary S5-S9 defaults provide at most 10 fixed seeds; "
                "pass --seeds for a larger run"
            )
        explicit_seeds = list(
            DEFAULT_ROUND13_73_SEEDS[: args.num_episodes]
        )

    if args.all_scenarios or args.primary_s5_s9:
        run_all_scenarios(
            num_agents=args.num_agents,
            num_episodes=args.num_episodes,
            output_root=output_root,
            heading_up=heading_up,
            traffic_density=args.traffic_density,
            start_seed=args.start_seed,
            fps=args.video_fps,
            decision_policy=args.decision_policy,
            planning_policy=args.planning_policy,
            control_policy=args.control_policy,
            evaluate=args.evaluate,
            horizon=args.horizon,
            scenario_pairs=(
                PRIMARY_S5_S9_SCENARIOS if args.primary_s5_s9 else None
            ),
            seeds=explicit_seeds,
            save_topdown_video=not args.no_topdown_video,
            save_semantic_bev_video=not args.no_semantic_bev_video,
            save_trajectory_data=not args.no_trajectory_data,
        )
    else:
        if not args.scenario_id:
            parser.error("--scenario-id is required when --all-scenarios is not set")
        video_dir = run_scenario(
            scenario_id=args.scenario_id,
            local_route=args.local_route,
            num_agents=args.num_agents,
            num_episodes=args.num_episodes,
            output_root=output_root,
            heading_up=heading_up,
            traffic_density=args.traffic_density,
            start_seed=args.start_seed,
            fps=args.video_fps,
            decision_policy=args.decision_policy,
            planning_policy=args.planning_policy,
            control_policy=args.control_policy,
            evaluate=args.evaluate,
            horizon=args.horizon,
            seeds=explicit_seeds,
            save_topdown_video=not args.no_topdown_video,
            save_semantic_bev_video=not args.no_semantic_bev_video,
            save_trajectory_data=not args.no_trajectory_data,
        )
        print(f"\n=== Videos saved to: {video_dir} ===")


if __name__ == "__main__":
    main()
