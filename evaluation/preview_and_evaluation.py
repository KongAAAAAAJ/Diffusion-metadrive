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
import sys
from pathlib import Path
from typing import Callable, Mapping, Optional

import numpy as np

from evaluation.evaluation_helper import (
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
)
from evaluation.platoon_performance import (
    build_platoon_metric_params,
    compute_pairwise_formation_reward,
    compute_pdms_reward_batch,
)
from models.controller.LQRFollowerController import LQRFollowerController
from models.controller.PIDController import PIDTrajectoryController, _world_trajectory_to_ego_local
from models.decisioner.rule_decisioner import make_rule_maker, select_controller_by_formation
from models.platoon_planner.platoon_normal_planner import PlatoonNormalPlanner
from tools.topdown_view import (
    capture_topdown_frame as _capture_topdown_frame,
    overlay_planning_debug as _overlay_planning_debug,
    overlay_platoon_labels as _overlay_platoon_labels,
    overlay_rule_maker_debug as _overlay_rule_maker_debug,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_SCENARIO_ID = "S5_hard_brake_lead"
DEFAULT_NUM_AGENTS = 3
DEFAULT_NUM_EPISODES = 3
DEFAULT_OUTPUT_ROOT = Path("/media/kong/Elements_SE/Diffusion_Data/outputs/run_results")
DEFAULT_TRAFFIC_DENSITY = 0.10
DEFAULT_START_SEED = 59
DEFAULT_VIDEO_FPS = 10
DEFAULT_HORIZON = 600
DEFAULT_DECISION_POLICY = "rule_maker"
DEFAULT_PLANNING_POLICY = "lattice"
DEFAULT_CONTROL_POLICY = "adaptive"


# ---------------------------------------------------------------------------
# Rendering helpers  (implementations live in tools/topdown_view.py)
# ---------------------------------------------------------------------------


def _write_video(path: Path, frames: list, fps: int) -> None:
    if not frames:
        return
    import mediapy
    path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(path), frames, fps=int(fps))


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
        "agent_ids": list(agent_ids),
        "trajectories_by_agent": {
            agent_id: np.asarray(trajectory, dtype=np.float32).tolist()
            for agent_id, trajectory in trajectories.items()
        },
        "candidates_by_agent": candidates_by_agent,
        "planner_debug": planner_debug,
    }


def _build_pipeline_factory(
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
        env_config = dict(getattr(env, "config", {}) or {})
        rule_maker = make_rule_maker(env_config)
        rule_maker.reset(env, agent_ids)
        planner = PlatoonNormalPlanner()
        pid_ctrl = PIDTrajectoryController(env_config)
        pid_ctrl.reset()
        lqr_ctrl = LQRFollowerController(env_config)
        lqr_ctrl.reset()

        def action_fn(env) -> dict[str, np.ndarray]:
            active_agent_ids = [aid for aid in agent_ids if aid in getattr(env, "agents", {})]
            if not active_agent_ids:
                setattr(env, "_preview_rule_maker_debug", None)
                setattr(env, "_preview_planning_debug", None)
                return {}

            planner_batch = getattr(env, "_last_planner_batch", None) or {}
            raw_decisions = rule_maker.compute(env, active_agent_ids, planner_batch)
            decisions = _normalize_decisions(raw_decisions)
            rule_debug = getattr(rule_maker, "get_last_debug", lambda: None)()
            if rule_debug is None:
                rule_debug = getattr(env, "_preview_rule_maker_debug", None)
            setattr(env, "_preview_rule_maker_debug", rule_debug)

            dynamic_roles = (rule_debug or {}).get("dynamic_roles", {}) if rule_debug else {}
            if dynamic_roles:
                apply_roles = getattr(env, "apply_dynamic_roles", None)
                if callable(apply_roles):
                    apply_roles(dynamic_roles)
                else:
                    setattr(env, "_agent_roles", dict(dynamic_roles))

            trajectories = planner.plan(env, decisions)

            planning_debug = _build_planning_debug(
                planning_policy=planning_policy,
                agent_ids=active_agent_ids,
                trajectories=trajectories,
                planner_debug=getattr(planner, "get_last_debug", lambda: None)(),
            )
            setattr(env, "_preview_planning_debug", planning_debug)
            if control_policy == "adaptive":
                ctrl = select_controller_by_formation(rule_maker, pid_ctrl, lqr_ctrl)
            else:
                ctrl = pid_ctrl
            actions = ctrl.compute_actions(env, trajectories)
            control_debug = getattr(ctrl, "get_last_debug", lambda: None)()
            setattr(env, "_preview_control_debug", control_debug)
            return actions

        return action_fn

    return factory


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
):
    spawn_manager = getattr(getattr(env, "engine", None), "spawn_manager", None)
    if spawn_manager is not None and hasattr(spawn_manager, "set_episode_spawn_seed"):
        spawn_manager.set_episode_spawn_seed(int(seed))
    obs = env.reset()
    action_fn = action_fn_factory(env, agent_ids, seed)
    frames: list[np.ndarray] = []
    terminated = truncated = info = None

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
        actions = action_fn(env)
        if not actions:
            break
        planning_debug = getattr(env, "_preview_planning_debug", None)

        frame = _capture_topdown_frame(
            env, lead_id, heading_up, agent_ids,
            rule_maker_debug=getattr(env, "_preview_rule_maker_debug", None),
            planning_debug=planning_debug,
        )
        if frame is not None:
            frames.append(frame)

        obs, reward, terminated, truncated, info = env.low_level_step(actions)

        if platoon_metrics is not None:
            platoon_metrics.update(info)
        if pdms_params is not None and pdms_records is not None:
            debug = getattr(env, "_preview_rule_maker_debug", None)
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

        if terminated.get("__all__", False) or truncated.get("__all__", False):
            print(_format_episode_stop_reason(_step, terminated, truncated, info))
            break

    if platoon_metrics is not None and not _episode_failed_immediately(frames, terminated, truncated, info):
        platoon_metrics.end_episode()

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
    max_episode_retries: int = 3,
    env_factory: Optional[Callable] = None,
    evaluate: bool = False,
    metric_params: Optional[dict] = None,
    horizon: int = DEFAULT_HORIZON,
) -> Path:
    if local_route is None:
        local_route = _pick_local_route(scenario_id)
    print(
        f"  preview source={Path(__file__).resolve()} "
        f"scenario={scenario_id} local_route={local_route}"
    )

    video_dir = output_root / scenario_id / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

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
        from envs.platoon_env import PlatoonEnv
        env = PlatoonEnv(env_config)
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

    try:
        for ep_idx in range(num_episodes):
            frames: list[np.ndarray] = []
            used_seed = start_seed + ep_idx
            pdms_records: Optional[list] = [] if pdms_params is not None else None
            episode_step_records: Optional[list] = [] if evaluate else None
            for retry in range(max(1, int(max_episode_retries))):
                episode_seed = used_seed + retry
                frames, terminated, truncated, info = _run_single_episode(
                    env, agent_ids, lead_id, heading_up, episode_seed, action_fn_factory,
                    pdms_params=pdms_params,
                    platoon_metrics=platoon_metrics,
                    pdms_records=pdms_records,
                    episode_step_records=episode_step_records,
                )
                if not _episode_failed_immediately(frames, terminated, truncated, info):
                    used_seed = episode_seed
                    break
                print(f"  retry ep {ep_idx + 1} seed={episode_seed + 1} (frames={len(frames)})")

            video_path = video_dir / f"episode_{ep_idx:04d}.mp4"
            if not frames:
                agent0_info = (info or {}).get("agent0", {}) if isinstance(info, dict) else {}
                raise RuntimeError(
                    f"{scenario_id} local_route={local_route} seed={used_seed} captured 0 frames; "
                    f"terminated={terminated} truncated={truncated} agent0_info={agent0_info}"
                )
            _write_video(video_path, frames, fps)
            print(
                f"  [{scenario_id}] ep {ep_idx + 1}/{num_episodes} -> {video_path.name} "
                f"({len(frames)} frames, seed={used_seed})"
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
        _plot_average_episode_pdms(metrics_dir / "ave_metrice.png", all_episode_step_records)

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
) -> None:
    """Evaluate all defined scenarios (one env per scenario to avoid map conflicts)."""
    pairs = _all_scenario_route_pairs()
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
    parser.add_argument("--num-agents", type=int, default=DEFAULT_NUM_AGENTS)
    parser.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--heading-up", default="false")
    parser.add_argument("--traffic-density", type=float, default=DEFAULT_TRAFFIC_DENSITY)
    parser.add_argument("--start-seed", type=int, default=DEFAULT_START_SEED)
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
    args = parser.parse_args()
    args.evaluate = True

    heading_up = args.heading_up.lower() in ("true", "1", "yes")
    output_root = Path(args.output_root)

    if args.all_scenarios:
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
        )
        print(f"\n=== Videos saved to: {video_dir} ===")


if __name__ == "__main__":
    main()
