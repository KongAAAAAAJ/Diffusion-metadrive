"""
Test script for selected-mode local refinement GRPO checkpoints.

Execution flow is strictly aligned with train/train_refine_grpo.py:
  - Env:     ModeSelectionSB3Env  (identical to training)
  - Planner: build_planner_for_selected_refinement()  (identical to training)
  - Rollout: rollout_multimodal_refinement()  (identical to training)
  - Reward:  _compute_pdms_reward_batch() on refined_traj (heading already atan2-reconstructed
             inside rollout, matching training exactly)
  - Mode:    argmax(masked_cls_logits) with fallback  (identical to training on_training_mode)
  - Execute: _execute_trajectories()  (identical to training)

Saves (consistent with test_transfuser_policy.py):
  random_action_reward_summary.json
  random_action_step_records.jsonl
  combined_traj_frames/episode_XXX/step_XXXXX.png   (topdown + overlay)
  traj_plots/{agent_id}/episode_XXX/step_XXXXX.png  (matplotlib, per-vehicle, local-frame)
  trajectory_data/episode_XXX/step_XXXXX.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any, Mapping

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# ── Training helpers (strict alignment) ───────────────────────────────────────
from train.train_refine_grpo import (
    _execute_trajectories,
    build_planner_for_selected_refinement,
    fix_candidates_heading_inplace,
    mode_names_from_planner,
    rollout_multimodal_refinement,
    select_best_mode_group,
    _get_vehicle_pose,
)
from evaluation.platoon_performance import (
    build_pdms_params,
    compute_pairwise_formation_reward,
    compute_pdms_reward_batch as _compute_pdms_reward_batch,
)
from models.refine_grpo.ddim_with_logprob import DDIMSchedulerWithLogProb

# ─────────────────────────────────────────────────────────────────────────────

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = str(_REPO_ROOT / "configs" / "train" / "refine_grpo.yaml")
_DEFAULT_OUTPUT_ROOT = str(_REPO_ROOT / "outputs" / "refine_grpo_test")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Test selected-refine-GRPO checkpoint (training-aligned)")
    p.add_argument("--refine-train-config-path", type=str, default=_DEFAULT_CONFIG)
    return p.parse_args(argv)


# ── Config loading ─────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    if not path or not os.path.isfile(path):
        print(f"[test-refine-grpo] WARNING: config not found at {path!r}, using defaults.", flush=True)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    print(f"[test-refine-grpo] loaded config: {path}", flush=True)
    return cfg


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_test_config(config: Mapping[str, Any]) -> dict:
    raw = dict(config.get("test_config", {}) or {})
    return {
        "episodes": int(raw.get("episodes", 3)),
        "max_steps": int(raw.get("max_steps", 0)),
        "output_root": str(raw.get("output_root", _DEFAULT_OUTPUT_ROOT)),
        "run_tag": str(raw.get("run_tag", "test")),
        "save_combined_traj_frames": _as_bool(raw.get("save_combined_traj_frames", True)),
        "save_traj_plots": _as_bool(raw.get("save_traj_plots", False)),
        "save_trajectory_data": _as_bool(raw.get("save_trajectory_data", True)),
        "save_2d_video": _as_bool(raw.get("save_2d_video", True)),
        "save_3d_video": _as_bool(raw.get("save_3d_video", False)),
        "video_fps": int(raw.get("video_fps", 10)),
    }


def _create_next_run_dir(parent: pathlib.Path) -> pathlib.Path:
    parent.mkdir(parents=True, exist_ok=True)
    existing = []
    for item in parent.iterdir():
        if not item.is_dir() or not item.name.startswith("run_"):
            continue
        suffix = item.name[len("run_"):]
        if suffix.isdigit():
            existing.append(int(suffix))
    run_dir = parent / f"run_{max(existing) + 1 if existing else 1}"
    run_dir.mkdir()
    return run_dir


# ── Env config builder (mirrors run_training in train_refine_grpo.py) ─

def _build_env_config(config: dict, test_config: Mapping[str, Any], planner) -> dict:
    env_config = dict(config.get("env_config", {}))
    env_config.setdefault("num_agents",          int(config.get("num_agents", 3)))
    env_config.setdefault("observation_mode",    "multimodal")
    env_config.setdefault("use_render",          False)
    if bool(test_config.get("save_3d_video", False)):
        env_config["use_render"] = True
    env_config.setdefault("planner_device",      "cuda")
    env_config.setdefault("lookahead_index",     int(config.get("lookahead_index", 2)))
    env_config.setdefault("target_speed_km_h",   float(config.get("target_speed_km_h", 30.0)))
    env_config.setdefault("controller_type",     str(config.get("controller_type", "stabilized")))
    env_config["use_action_mask"]    = False
    env_config["trajectory_source"] = "diffusion"
    env_config["planner"]            = planner

    if env_config.get("scenario_id") and not env_config.get("local_route"):
        from scenarios.definitions import SCENARIO_BY_ID

        scenario_id = str(env_config["scenario_id"])
        if scenario_id not in SCENARIO_BY_ID:
            raise ValueError(f"Unknown scenario_id in selected-refine test config: {scenario_id}")
        allowed_routes = tuple(SCENARIO_BY_ID[scenario_id].allowed_local_routes)
        if not allowed_routes:
            raise ValueError(f"Scenario {scenario_id} has no allowed local routes.")
        env_config["local_route"] = allowed_routes[0]

    if "start_seed" not in env_config and "seed" not in env_config:
        top_start_seed = config.get("start_seed", None)
        if top_start_seed is not None:
            env_config["start_seed"] = int(top_start_seed)
    env_config.setdefault("num_scenarios",    1)
    env_config.setdefault("traffic_density",  0.04)
    env_config.setdefault("random_traffic",   False)
    return env_config


# ── PDMS param dict (mirrors run_training) ────────────────────────────────────


# ── Locked-follow helpers ─────────────────────────────────────────────────────

def _call_rule_maker_locked(rule_maker) -> bool:
    locked = getattr(rule_maker, "is_formation_locked", False)
    return bool(locked() if callable(locked) else locked)


def _inject_rule_maker_targets(planner_batch: dict, decisions: Mapping[str, Mapping]) -> None:
    for agent_id, decision in (decisions or {}).items():
        if agent_id not in planner_batch or not isinstance(decision, Mapping):
            continue
        if "target_point" not in decision:
            continue
        planner_batch[agent_id]["target_point"] = np.asarray(
            decision["target_point"], dtype=np.float32
        ).reshape(2)


def _apply_rule_maker_roles(base_env, rule_maker_debug: dict | None) -> None:
    dynamic_roles = (rule_maker_debug or {}).get("dynamic_roles", {}) if rule_maker_debug else {}
    if not dynamic_roles:
        return
    apply_roles = getattr(base_env, "apply_dynamic_roles", None)
    if callable(apply_roles):
        apply_roles(dynamic_roles)
    else:
        setattr(base_env, "_agent_roles", dict(dynamic_roles))


def _locked_follow_decision(rule_maker, env, agent_ids: list[str], planner_batch: dict) -> dict:
    """Update RuleMaker and return the locked-follow decision state."""
    if rule_maker is None:
        return {
            "use_locked_follow": False,
            "formation_locked": False,
            "decisions": {},
            "rule_maker_debug": None,
        }
    base_env = getattr(env, "base_env", env)
    decisions = rule_maker.compute(base_env, agent_ids, planner_batch)
    _inject_rule_maker_targets(planner_batch, decisions)
    get_debug = getattr(rule_maker, "get_last_debug", None)
    rule_maker_debug = get_debug() if callable(get_debug) else None
    _apply_rule_maker_roles(base_env, rule_maker_debug)
    formation_locked = _call_rule_maker_locked(rule_maker)
    return {
        "use_locked_follow": formation_locked,
        "formation_locked": formation_locked,
        "decisions": decisions,
        "rule_maker_debug": rule_maker_debug,
    }


def _execute_locked_follow_step(
    *,
    env,
    agent_ids: list[str],
    normal_planner,
    follow_controller,
    decisions: Mapping[str, Mapping],
    rule_maker_debug: dict | None,
) -> tuple[float, bool, dict, dict[str, np.ndarray]]:
    from models.controller.PIDController import _world_trajectory_to_ego_local

    base_env = env.base_env
    trajectories_world = normal_planner.plan(base_env, decisions)
    low_level_actions = follow_controller.compute_actions(base_env, trajectories_world)
    controller_debug = (
        follow_controller.get_last_debug()
        if callable(getattr(follow_controller, "get_last_debug", None))
        else {}
    )
    normal_planner_debug = (
        normal_planner.get_last_debug()
        if callable(getattr(normal_planner, "get_last_debug", None))
        else {}
    )

    trajectories_local: dict[str, np.ndarray] = {}
    for agent_id in agent_ids:
        vehicle = getattr(base_env, "agents", {}).get(agent_id)
        trajectory_world = trajectories_world.get(agent_id)
        trajectories_local[agent_id] = (
            _world_trajectory_to_ego_local(vehicle, trajectory_world)
            if vehicle is not None and trajectory_world is not None
            else np.zeros((0, 3), dtype=np.float32)
        )

    if hasattr(base_env, "_pending_step_trajectories"):
        base_env._pending_step_trajectories = dict(trajectories_local)

    vehicle_state_before = env._vehicle_states(agent_ids) if hasattr(env, "_vehicle_states") else {}
    result = base_env.step(low_level_actions)
    if len(result) == 5:
        raw_obs, env_reward, terminated, truncated, info = result
        done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
    else:
        raw_obs, env_reward, done_dict, info = result
        terminated = done_dict
        truncated = {agent_id: False for agent_id in done_dict}
        done = bool(done_dict.get("__all__", False))

    info = dict(info or {})
    missing_agent_ids = [agent_id for agent_id in agent_ids if agent_id not in (raw_obs or {})]
    if missing_agent_ids:
        terminated = dict(terminated)
        truncated = dict(truncated)
        for agent_id in agent_ids:
            terminated[agent_id] = True
        terminated["__all__"] = True
        truncated["__all__"] = False
        done = True
        info["missing_agent_ids"] = list(missing_agent_ids)

    env_reward = dict(env_reward or {})
    reward_values = [float(env_reward.get(agent_id, 0.0)) for agent_id in agent_ids]
    scalar_reward = float(np.mean(reward_values)) if reward_values else 0.0

    vehicle_state_after = env._vehicle_states(agent_ids) if hasattr(env, "_vehicle_states") else {}
    base_crash_flags = {}
    safety_flags = {}
    for agent_id in agent_ids:
        agent_info = dict(info.get(agent_id, {}))
        base_crash_flags[agent_id] = {
            "crash": bool(agent_info.get("crash", False)),
            "crash_vehicle": bool(agent_info.get("crash_vehicle", False)),
            "crash_human": bool(agent_info.get("crash_human", False)),
            "crash_object": bool(agent_info.get("crash_object", False)),
            "crash_building": bool(agent_info.get("crash_building", False)),
            "crash_sidewalk": bool(agent_info.get("crash_sidewalk", False)),
            "out_of_road": bool(agent_info.get("out_of_road", False)),
        }
        safety_flags[agent_id] = {
            "crash": bool(agent_info.get("crash", False)),
            "terminal_crash": bool(
                agent_info.get("crash_vehicle", False)
                or agent_info.get("crash_human", False)
                or agent_info.get("crash_object", False)
                or agent_info.get("crash_building", False)
            ),
            "out_of_road": bool(agent_info.get("out_of_road", False)),
        }

    env._last_raw_obs = env._normalize_raw_obs(raw_obs or {}, previous_obs=env._last_raw_obs)
    env._last_obs = env._refresh_mode_export() if env._last_raw_obs else env._last_obs
    info.update(
        {
            "step": int(getattr(env, "_step_count", 0)),
            "control_backend": "locked_lqr_follow",
            "formation_locked": True,
            "rule_maker_debug": rule_maker_debug,
            "normal_planner_debug": normal_planner_debug,
            "controller_debug": controller_debug,
            "selected_low_level_action": {
                agent_id: np.asarray(low_level_actions[agent_id], dtype=float).tolist()
                for agent_id in agent_ids
                if agent_id in low_level_actions
            },
            "reward": scalar_reward,
            "terminated": bool(terminated.get("__all__", False)),
            "truncated": bool(truncated.get("__all__", False)),
            "termination_flags": {
                **{agent_id: bool(terminated.get(agent_id, False)) for agent_id in agent_ids},
                "__all__": bool(terminated.get("__all__", False)),
            },
            "truncation_flags": {
                **{agent_id: bool(truncated.get(agent_id, False)) for agent_id in agent_ids},
                "__all__": bool(truncated.get("__all__", False)),
            },
            "safety_flags": safety_flags,
            "base_crash_flags": base_crash_flags,
            "vehicle_state_before": vehicle_state_before,
            "vehicle_state_after": vehicle_state_after,
        }
    )
    if hasattr(env, "_write_debug_log"):
        env._write_debug_log(info)
    if hasattr(env, "_step_count"):
        env._step_count += 1
    return scalar_reward, done, info, trajectories_local



# ── Per-vehicle matplotlib trajectory plot (local-frame, no topdown renderer) ──

_MODE_COLORS = [
    "#1D4ED8", "#DC2626", "#16A34A", "#D97706",
    "#7C3AED", "#0891B2", "#BE185D", "#059669",
]


def _save_agent_traj_plot(
    output_path: pathlib.Path,
    refined_np: np.ndarray,      # [G*M, T, 3] local-frame
    on_training_mode: int,
    num_modes: int,
    pdms_reward: float,
    episode_step: int,
    agent_id: str,
    mode_names: list[str],
):
    """Matplotlib trajectory plot for a single vehicle in ego-local coordinates."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    GxM = refined_np.shape[0]
    for gi in range(GxM):
        m = gi % num_modes
        color = _MODE_COLORS[m % len(_MODE_COLORS)]
        is_selected = (gi == on_training_mode)
        label = mode_names[m] if is_selected and mode_names and m < len(mode_names) else None
        ax.plot(refined_np[gi, :, 0], refined_np[gi, :, 1],
                color=color,
                linewidth=2.5 if is_selected else 0.7,
                alpha=1.0 if is_selected else 0.25,
                label=label)
        if is_selected:
            ax.plot(refined_np[gi, -1, 0], refined_np[gi, -1, 1],
                    "o", color=color, markersize=5)
    ax.set_xlabel("x (m, ego-local)")
    ax.set_ylabel("y (m, ego-local)")
    ax.set_aspect("equal", "datalim")
    ax.set_title(f"{agent_id}  step={episode_step}  pdms={pdms_reward:+.3f}", fontsize=9)
    if any(ax.get_legend_handles_labels()[1]):
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


# ── Combined traj frame (topdown + overlay) ───────────────────────────────────

def _save_combined_traj_frame_topdown(
    *,
    base_env,
    output_path: pathlib.Path,
    agent_ids: list[str],
    candidates: np.ndarray,          # [N, M, T, 3] coarse local-frame from export
    poses: dict[str, np.ndarray],    # agent_id → [x, y, heading]
    frame_data_by_agent: dict,       # agent_id → {refined_np:[G*M,T,3], on_training_mode, M}
    masks: np.ndarray,               # [N, M] bool
    mode_names: list[str],
    pdms_reward: float,
    episode_step: int,
    rule_maker_debug: dict | None = None,
):
    from models.diffusion.test_transfuser_policy import (
        _build_step_overlay_polylines,
        _capture_topdown_frame_with_overlay,
    )
    from tools.topdown_view import overlay_rule_maker_debug

    primary_id = agent_ids[0]
    primary_pose = poses.get(primary_id)
    if primary_pose is None:
        return

    pri_ego_xy = np.asarray(primary_pose[:2], dtype=np.float64)
    pri_heading = float(primary_pose[2])

    # Peer data (all agents except primary), drawn behind primary
    peers = []
    for idx, agent_id in enumerate(agent_ids):
        if idx == 0:
            continue
        pose = poses.get(agent_id)
        if pose is None:
            continue
        rd = frame_data_by_agent.get(agent_id, {})
        M = rd.get("M", candidates.shape[1])
        refined_np = rd.get("refined_np")  # [G*M, T, 3]
        peers.append({
            "xy":          np.asarray(pose[:2], dtype=np.float64),
            "heading_rad": float(pose[2]),
            "anchors":     candidates[idx, :, :, :2],           # [M, T, 2] coarse
            "candidates":  (refined_np[:M, :, :2] if refined_np is not None
                            else candidates[idx, :, :, :2]),    # [M, T, 2] group-0 refined
            "selected":    rd.get("on_training_mode", 0),
            "valid_mask":  masks[idx] if masks is not None else None,
        })

    rd_pri = frame_data_by_agent.get(primary_id, {})
    M_pri = rd_pri.get("M", candidates.shape[1])
    refined_np_pri = rd_pri.get("refined_np")  # [G*M, T, 3]

    overlay = _build_step_overlay_polylines(
        ego_xy=pri_ego_xy,
        ego_heading_rad=pri_heading,
        coarse_anchor_trajectories=candidates[0, :, :, :2],          # [M, T, 2]
        multimodal_trajectories=(refined_np_pri[:M_pri, :, :2]       # [M, T, 2] group-0 refined
                                 if refined_np_pri is not None
                                 else candidates[0, :, :, :2]),
        selected_mode_idx=rd_pri.get("on_training_mode", 0),
        topology_polyline_world=None,
        mode_valid_mask=masks[0],
        mode_slot_names=mode_names if mode_names else None,
        peer_ego_data=peers,
    )

    frame = _capture_topdown_frame_with_overlay(
        base_env, overlay,
        screen_size=800, film_size=10000,
        camera_position=tuple(pri_ego_xy),
        rule_maker_debug=None,
    )
    if frame is None:
        return
    frame = np.ascontiguousarray(overlay_rule_maker_debug(frame, base_env, rule_maker_debug))

    def _gc(v: float) -> str:
        return "pass" if v >= 1.0 else "FAIL"

    def _put(text: str, y: int, scale: float = 0.42) -> None:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (30, 30, 200), 1, cv2.LINE_AA)

    # step_text = f"step={episode_step}  pdms_mean={pdms_reward:+.3f}"
    # _put(step_text, 20, scale=0.48)

    # *标记reward信息
    # y_cursor = 44
    # line_h = 20
    # for agent_id in agent_ids:
    #     rd = frame_data_by_agent.get(agent_id, {})
    #     dbg = rd.get("reward_debug", {})
    #     gm = rd.get("on_training_mode", 0)

    #     def _gv(key, default=0.0):
    #         v = dbg.get(key, default)
    #         return float(v[gm]) if hasattr(v, "__len__") else float(v)

    #     label = agent_id.replace("agent", "A")
    #     header = f"{label}: pdms={_gv('reward'):+.3f}  qual={_gv('quality'):.3f}"
    #     gate_line = (
    #         f"  gate: col={_gc(_gv('collision_gate', 1))}"
    #         f" road={_gc(_gv('road_gate', 1))}"
    #         f" smth={_gc(_gv('smoothness_gate', 1))}"
    #     )
    #     qual_line = (
    #         f"  qual: prog={_gv('progress'):.2f}"
    #         f" form={_gv('formation'):.2f}"
    #         f" spd={_gv('speed'):.2f}"
    #         f" lane={_gv('lane'):.2f}"
    #         f" cmft={_gv('comfort'):.2f}"
    #         f" cons={_gv('consistency'):.2f}"
    #     )
    #     for text in (header, gate_line, qual_line):
    #         _put(text, y_cursor)
    #         y_cursor += line_h
    #     y_cursor += 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return frame  # RGB, for video writing


# ── Video writer helper ───────────────────────────────────────────────────────

def _open_video_writer(path: pathlib.Path, fps: int, width: int, height: int) -> "cv2.VideoWriter":
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    return writer


def _capture_plain_2d_frame(base_env) -> "np.ndarray | None":
    """Capture a top-down RGB frame without trajectory or debug overlays."""
    from models.diffusion.test_transfuser_policy import _capture_2d_topdown_frame

    return _capture_2d_topdown_frame(base_env)


def _format_episode_end_reason(
    *,
    episode_idx: int,
    step_idx: int,
    info: Mapping | None,
    fallback_reason: str,
) -> str:
    """Format one episode-end line with per-agent safety details."""
    info_dict = dict(info or {})
    base_flags = info_dict.get("base_crash_flags")
    agent_ids = {
        str(agent_id)
        for agent_id, agent_info in info_dict.items()
        if str(agent_id).startswith("agent") and isinstance(agent_info, Mapping)
    }
    if isinstance(base_flags, Mapping):
        agent_ids.update(str(agent_id) for agent_id in base_flags)

    crash_keys = ("crash", "crash_vehicle", "crash_human", "crash_object", "crash_building")
    crash_agents: list[str] = []
    out_of_road_agents: list[str] = []
    arrive_agents: list[str] = []
    for agent_id in sorted(agent_ids):
        direct = info_dict.get(agent_id)
        direct = dict(direct) if isinstance(direct, Mapping) else {}
        base = base_flags.get(agent_id) if isinstance(base_flags, Mapping) else None
        base = dict(base) if isinstance(base, Mapping) else {}
        merged = {**direct, **base}
        if any(bool(merged.get(key, False)) for key in crash_keys):
            crash_agents.append(agent_id)
        if bool(merged.get("out_of_road", False)):
            out_of_road_agents.append(agent_id)
        if bool(direct.get("arrive_dest", False)):
            arrive_agents.append(agent_id)

    reasons: list[str] = []
    if crash_agents:
        reasons.append("crash")
    if out_of_road_agents:
        reasons.append("out_of_road")
    if arrive_agents:
        reasons.append("arrive_dest")
    if not reasons:
        reasons.append(str(fallback_reason or "unknown"))

    parts = [
        f"[test-refine-grpo] episode={int(episode_idx)} ended: "
        f"step={int(step_idx)} reason={','.join(reasons)}"
    ]
    if crash_agents:
        parts.append(f"crash_agents={','.join(crash_agents)}")
    if out_of_road_agents:
        parts.append(f"out_of_road_agents={','.join(out_of_road_agents)}")
    if arrive_agents:
        parts.append(f"arrive_agents={','.join(arrive_agents)}")
    return " ".join(parts)


def _save_step_outputs(
    *,
    env,
    output_dir: pathlib.Path,
    records_path: pathlib.Path,
    all_records: list[dict],
    test_config: dict,
    video_fps: int,
    vid_2d,
    vid_3d,
    episode_idx: int,
    step_idx: int,
    agent_ids: list[str],
    control_backend: str,
    formation_locked: bool,
    env_reward: float,
    pdms_reward: float,
    pdms_by_agent: dict,
    selected_modes: list[int],
    gt_modes: list[int],
    mode_valid_mask,
    all_rewards: list,
    executed_trajs: dict[str, np.ndarray],
    diffusion_2d_frame: "np.ndarray | None",
):
    """Persist exactly one step's trajectory, videos, and JSONL record."""
    post_step_poses = {aid: _get_vehicle_pose(env, aid) for aid in agent_ids}

    if test_config["save_trajectory_data"]:
        trajectory_data = {
            "episode": episode_idx,
            "step": step_idx,
            "control_backend": control_backend,
            "formation_locked": bool(formation_locked),
            "agents": {},
        }
        for index, agent_id in enumerate(agent_ids):
            pose = post_step_poses.get(agent_id)
            selected_mode = selected_modes[index] if index < len(selected_modes) else None
            gt_mode = gt_modes[index] if index < len(gt_modes) else None
            trajectory = executed_trajs.get(agent_id)
            trajectory_data["agents"][agent_id] = {
                "pose": pose.tolist() if pose is not None else None,
                "selected_mode": int(selected_mode) if selected_mode is not None else None,
                "gt_mode": int(gt_mode) if gt_mode is not None else None,
                "executed_traj_local": (
                    np.asarray(trajectory, dtype=np.float32).tolist() if trajectory is not None else []
                ),
            }
        trajectory_path = (
            output_dir / "trajectory_data" / f"episode_{episode_idx:03d}" / f"step_{step_idx:05d}.json"
        )
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_path.write_text(json.dumps(trajectory_data), encoding="utf-8")

    if test_config["save_2d_video"]:
        frame_2d_rgb = (
            _capture_plain_2d_frame(env.base_env)
            if formation_locked
            else diffusion_2d_frame
        )
        if frame_2d_rgb is not None:
            try:
                bgr_2d = cv2.cvtColor(frame_2d_rgb, cv2.COLOR_RGB2BGR)
                if vid_2d is None:
                    h2d, w2d = bgr_2d.shape[:2]
                    path_2d = output_dir / "videos" / "2d" / f"episode_{episode_idx:03d}.mp4"
                    vid_2d = _open_video_writer(path_2d, video_fps, w2d, h2d)
                vid_2d.write(bgr_2d)
            except Exception as error:
                print(f"[test-refine-grpo] 2D video write failed: {error}", flush=True)

    if test_config["save_3d_video"]:
        try:
            frame_3d = env.render(mode="rgb_array")
            if frame_3d is not None and isinstance(frame_3d, np.ndarray):
                bgr_3d = (
                    cv2.cvtColor(frame_3d, cv2.COLOR_RGB2BGR)
                    if frame_3d.ndim == 3 and frame_3d.shape[2] == 3
                    else frame_3d
                )
                if vid_3d is None:
                    h3d, w3d = bgr_3d.shape[:2]
                    path_3d = output_dir / "videos" / "3d" / f"episode_{episode_idx:03d}.mp4"
                    vid_3d = _open_video_writer(path_3d, video_fps, w3d, h3d)
                vid_3d.write(bgr_3d)
        except Exception:
            pass

    record = {
        "episode": episode_idx,
        "step": step_idx,
        "control_backend": control_backend,
        "formation_locked": bool(formation_locked),
        "env_reward": float(env_reward),
        "pdms_reward": float(pdms_reward),
        "pdms_by_agent": pdms_by_agent,
        "selected_modes": selected_modes,
        "gt_modes": gt_modes,
        "mode_valid_mask": (
            mode_valid_mask.tolist() if hasattr(mode_valid_mask, "tolist") else mode_valid_mask
        ),
        "all_rewards": all_rewards,
    }
    all_records.append(record)
    with records_path.open("a", encoding="utf-8") as file_handle:
        file_handle.write(json.dumps(record) + "\n")

    return vid_2d, vid_3d


# ── Summary builder (same structure as _summarize_random_action_rewards) ──────

def _summarize_results(
    records: list[dict],
    *,
    episodes: int,
    success: int,
    crash: int,
    out_of_road: int,
    metadata: dict,
) -> dict:
    if not records:
        return {
            "metadata": metadata,
            "num_records": 0,
            "episode_pdms_reward_per_step_mean": 0.0,
            "episode_pdms_reward_per_step_std": 0.0,
            "episode_env_reward_per_step_mean": 0.0,
            "episode_env_reward_per_step_std": 0.0,
            "success_rate": 0.0,
            "crash_rate": 0.0,
            "out_of_road_rate": 0.0,
            "per_mode": {},
        }

    pdms_vals = [r["pdms_reward"] for r in records]
    env_vals  = [r["env_reward"]  for r in records]
    num_eps   = max(1, episodes)

    per_mode: dict[str, dict] = {}
    for r in records:
        for idx, m in enumerate(r.get("selected_modes", [])):
            key = str(m)
            per_mode.setdefault(key, {"selected_count": 0, "pdms_rewards": [], "env_rewards": []})
            per_mode[key]["selected_count"] += 1
            per_mode[key]["pdms_rewards"].append(r.get("pdms_by_agent", {}).get(f"agent{idx}", 0.0))
            per_mode[key]["env_rewards"].append(r["env_reward"])

    per_mode_out = {}
    for k, v in per_mode.items():
        per_mode_out[k] = {
            "selected_count": v["selected_count"],
            "pdms_reward_mean": float(np.mean(v["pdms_rewards"])),
            "pdms_reward_std":  float(np.std(v["pdms_rewards"])),
            "env_reward_mean":  float(np.mean(v["env_rewards"])),
            "env_reward_std":   float(np.std(v["env_rewards"])),
        }

    return {
        "metadata": metadata,
        "num_records": len(records),
        "episode_pdms_reward_per_step_mean": float(np.mean(pdms_vals)),
        "episode_pdms_reward_per_step_std":  float(np.std(pdms_vals)),
        "episode_env_reward_per_step_mean":  float(np.mean(env_vals)),
        "episode_env_reward_per_step_std":   float(np.std(env_vals)),
        "success_rate":     success / num_eps,
        "crash_rate":       crash   / num_eps,
        "out_of_road_rate": out_of_road / num_eps,
        "per_mode": per_mode_out,
    }


# ── Main test loop ─────────────────────────────────────────────────────────────

def run_test(args):
    config = _load_config(args.refine_train_config_path)
    test_config = _resolve_test_config(config)

    train_followers_only = bool(config.get("train_followers_only", False))

    # When train_followers_only: keep the original pretrained_ckpt for the leader planner.
    _original_pretrained_ckpt = str(config.get("pretrained_ckpt", "") or "")

    # Env config used to build the planner before the env wrapper owns it.
    config.setdefault("num_agents", 3)
    env_config_pre = dict(config.get("env_config", {}))
    planner_device = str(env_config_pre.get("planner_device", "cuda"))
    env_config_pre["planner_device"] = planner_device

    # Build planner (same as training) — used by followers and env mode selection
    print("[test-refine-grpo] building planner ...", flush=True)
    planner = build_planner_for_selected_refinement(config, {**env_config_pre, "planner_device": planner_device})
    planner.eval()
    for p in planner.parameters():
        p.requires_grad_(False)

    # When train_followers_only: build a separate frozen pretrained planner for the leader
    if train_followers_only:
        print("[test-refine-grpo] train_followers_only=True: building frozen pretrain_planner for leader ...", flush=True)
        _pretrain_config = dict(config)
        _pretrain_config["pretrained_ckpt"] = _original_pretrained_ckpt
        pretrain_planner = build_planner_for_selected_refinement(
            _pretrain_config, {**env_config_pre, "planner_device": planner_device}
        )
        pretrain_planner.eval()
        for p in pretrain_planner.parameters():
            p.requires_grad_(False)
        print("[test-refine-grpo] pretrain_planner ready.", flush=True)
    else:
        pretrain_planner = None

    # Build env (same as training)
    from envs.wrap_platoon_env import ModeSelectionSB3Env
    env_config = _build_env_config(config, test_config, planner)
    env = ModeSelectionSB3Env(env_config)
    print(f"[test-refine-grpo] env ready: {type(env).__name__}", flush=True)

    # External upper-level decision model (used when target_guidance_type='external_point')
    _target_guidance_type = str(
        planner._model_config.target_guidance_type
        if hasattr(planner, "_model_config")
        else config.get("target_guidance_type", "multi_point")
    )
    _rule_maker = None
    _normal_planner = None
    _follow_controller = None
    if _target_guidance_type == "external_point":
        try:
            from models.controller import LQRFollowerController
            from models.decisioner.rule_decisioner import make_rule_maker
            from models.platoon_planner import PlatoonNormalPlanner

            rule_config = dict(config)
            rule_config.update(dict(config.get("env_config") or {}))
            _rule_maker = make_rule_maker(rule_config)
            _normal_planner = PlatoonNormalPlanner()
            _follow_controller = LQRFollowerController(env_config)
            _follow_controller.reset()
            print(
                "[test-refine-grpo] RuleMaker locked-follow enabled: "
                f"rule_maker={_rule_maker.__class__.__name__} "
                "planner=PlatoonNormalPlanner controller=LQRFollowerController",
                flush=True,
            )
        except Exception as exc:
            print(f"[test-refine-grpo] WARNING: locked-follow backend disabled: {exc}", flush=True)

    # Test-time overrides: single group for memory/speed efficiency
    config["num_refine_groups"] = 1

    # Rollout / DDIM params
    num_groups        = int(config.get("num_refine_groups",   4))
    rollout_chunk_size= int(config.get("rollout_chunk_size",  80))
    noise_t           = int(config.get("refine_noise_t",      8))
    denoise_steps     = int(config.get("refine_denoise_steps",4))
    eta               = 0.0   # deterministic eval; aligned with train's post-update eval pass
    desired_gap_m        = float(config.get("desired_gap_m",        10.0))
    progress_s_max       = float(config.get("progress_s_max",       15.0))
    waypoint_decay_gamma = float(config.get("waypoint_decay_gamma",  0.9))

    pdms_params = build_pdms_params(config)
    gate_road_half_width_m = pdms_params["gate_road_half_width_m"]

    # DDIM scheduler (same as training)
    refine_scheduler = DDIMSchedulerWithLogProb(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="sample",
    )
    refine_scheduler.set_timesteps(1000, device=planner._device())

    # Mode names for road-half-width and visualization
    mode_names = mode_names_from_planner(planner, None)

    # Output paths
    output_base = pathlib.Path(test_config["output_root"]) / str(test_config["run_tag"])
    output_dir = _create_next_run_dir(output_base)
    records_path = output_dir / "random_action_step_records.jsonl"
    summary_path = output_dir / "random_action_reward_summary.json"

    all_records: list[dict] = []
    episode_idx = 0
    success = crash = out_of_road = 0

    episodes = int(test_config["episodes"])
    max_steps = int(test_config["max_steps"])
    video_fps = int(test_config["video_fps"])

    for episode_idx in range(episodes):
        print(f"[test-refine-grpo] episode {episode_idx}/{episodes}", flush=True)
        env.reset()
        if _rule_maker is not None:
            _rule_maker.reset(env.base_env, list(getattr(env, "_agent_ids", [])))
        if _follow_controller is not None:
            _follow_controller.reset()

        done = False
        episode_step = 0
        episode_env_reward = 0.0
        episode_pdms_reward = 0.0
        vid_2d: "cv2.VideoWriter | None" = None
        vid_3d: "cv2.VideoWriter | None" = None
        # Real collision/road flags from the previous env step, keyed by agent_id.
        # None at the start of an episode (no prior step executed).
        prev_env_crash_flags: dict[str, dict] = {}
        info: dict = {}
        stop_reason = "unknown"

        while not done:
            if max_steps > 0 and episode_step >= max_steps:
                stop_reason = "max_steps"
                break

            planner_batch = env._last_planner_batch
            export        = env._last_export
            if not planner_batch or export is None:
                stop_reason = "planner_data_missing"
                break

            agent_ids        = list(export["agent_ids"])
            candidates       = np.asarray(export["trajectory_candidates"], dtype=np.float32)  # [N, M, T, 3]
            fix_candidates_heading_inplace(candidates)   # tanh*π → atan2(Δy,Δx)
            masks            = np.asarray(export["lane_valid_mask"], dtype=bool)              # [N, M]
            masked_logits_np = np.asarray(export["masked_cls_logits"], dtype=np.float32)     # [N, M]
            on_training_modes = np.argmax(masked_logits_np, axis=-1).astype(np.int64)        # [N]

            # Inject external target_point before extract_rl_context so it flows
            # into model features automatically (external_point guidance mode).
            rule_maker_debug = None
            locked_follow = {
                "use_locked_follow": False,
                "formation_locked": False,
                "decisions": {},
                "rule_maker_debug": None,
            }
            if _rule_maker is not None:
                locked_follow = _locked_follow_decision(_rule_maker, env, agent_ids, planner_batch)
                rule_maker_debug = locked_follow.get("rule_maker_debug")

            '如果 LOCKED，直接FOLLOW CONTROL，不通过diffusion planner规划'
            if (
                bool(locked_follow.get("use_locked_follow", False))
                and _normal_planner is not None
                and _follow_controller is not None
            ):
                try:
                    t_start = time.time()
                    env_reward, done, info, locked_local_trajs = _execute_locked_follow_step(
                        env=env,
                        agent_ids=agent_ids,
                        normal_planner=_normal_planner,
                        follow_controller=_follow_controller,
                        decisions=locked_follow.get("decisions", {}),
                        rule_maker_debug=rule_maker_debug,
                    )
                    t_end = time.time()
                    print(
                        f"[test-refine-grpo] episode={episode_idx} step={episode_step} "
                        f"control_backend=locked_lqr_follow step_time={t_end - t_start:.4f}s",
                        flush=True,
                    )
                except AssertionError as e:
                    print(f"[test-refine-grpo] locked-follow env step error: {e}", flush=True)
                    done = True
                    stop_reason = "env_step_error"
                    env_reward = 0.0
                    info = {}
                    locked_local_trajs = {}

                prev_env_crash_flags = (info or {}).get("base_crash_flags", {})
                if not done and any(
                    flags.get("crash", False) or flags.get("out_of_road", False)
                    for flags in prev_env_crash_flags.values()
                ):
                    done = True
                if done and stop_reason == "unknown":
                    if bool((info or {}).get("truncated", False)):
                        stop_reason = "truncated"
                    elif bool((info or {}).get("terminated", False)):
                        stop_reason = "terminated"

                episode_env_reward += float(env_reward)
                episode_pdms_reward += 0.0
                episode_step += 1
                vid_2d, vid_3d = _save_step_outputs(
                    env=env,
                    output_dir=output_dir,
                    records_path=records_path,
                    all_records=all_records,
                    test_config=test_config,
                    video_fps=video_fps,
                    vid_2d=vid_2d,
                    vid_3d=vid_3d,
                    episode_idx=episode_idx,
                    step_idx=episode_step,
                    agent_ids=agent_ids,
                    control_backend="locked_lqr_follow",
                    formation_locked=True,
                    env_reward=env_reward,
                    pdms_reward=0.0,
                    pdms_by_agent={},
                    selected_modes=[],
                    gt_modes=[],
                    mode_valid_mask=masks,
                    all_rewards=[],
                    executed_trajs=locked_local_trajs,
                    diffusion_2d_frame=None,
                )
                continue

            # Context extraction (identical to training line 1861)
            with torch.no_grad():
                contexts, context_agent_ids = planner.extract_rl_context(planner_batch)

            # Coarse trajectories per agent
            coarse_traj_by_agent: dict[str, torch.Tensor] = {}
            for aid in agent_ids:
                _ct = (planner_batch.get(aid) or {}).get("coarse_trajectories")
                if _ct is not None:
                    coarse_traj_by_agent[aid] = torch.as_tensor(
                        _ct, dtype=torch.float32, device=planner._device()
                    )

            poses = {aid: _get_vehicle_pose(env, aid) for aid in agent_ids}

            executed:           dict[str, np.ndarray] = {}
            step_pdms_by_agent: dict[str, float]     = {}
            step_selected_modes: list[int]            = []
            step_gt_modes:       list[int]            = []
            step_all_rewards:    list[list[float]]    = []
            frame_data_by_agent: dict[str, dict]      = {}  # for combined_traj_frames

            for idx, agent_id in enumerate(agent_ids):

                # Mode selection (identical to training lines 1888-1891)
                on_training_mode = int(on_training_modes[idx])
                if on_training_mode < 0 or on_training_mode >= masks.shape[1] or not bool(masks[idx, on_training_mode]):
                    on_training_mode = int(np.argmax(masks[idx]))
                    on_training_modes[idx] = on_training_mode

                selected_traj = candidates[idx, on_training_mode]  # [T, 3] coarse

                # Multimodal rollout (identical to training line 1899)
                with torch.no_grad():
                    
                    # train_followers_only: leader uses frozen pretrained planner
                    _rollout_planner = (
                        pretrain_planner if (train_followers_only and idx == 0) else planner
                    )
                    rollout = rollout_multimodal_refinement(
                        _rollout_planner,
                        contexts[agent_id],
                        num_groups=num_groups,
                        noise_t=noise_t,
                        denoise_steps=denoise_steps,
                        eta=eta,
                        scheduler=refine_scheduler,
                        chunk_size=rollout_chunk_size,
                        coarse_trajectories=coarse_traj_by_agent.get(agent_id),
                    )

                M   = int(rollout["num_modes"])
                GxM = num_groups * M
                # refined_traj already has heading from reconstruct_heading_from_xy (inside rollout)
                refined_np = rollout["refined_traj"].detach().cpu().numpy().astype(np.float32)  # [G*M, T, 3]

                # PDMS reward (identical to training lines 1919-1996)
                is_ldr = (idx == 0)
                _prev_traj_arg = None
                _prev_pose_arg = None
                if not is_ldr and poses.get(agent_id) is not None and poses.get(agent_ids[idx - 1]) is not None:
                    prev_agent     = agent_ids[idx - 1]
                    _prev_traj_arg = executed.get(prev_agent, candidates[idx - 1, int(on_training_modes[idx - 1])])
                    _prev_pose_arg = poses[prev_agent]

                if is_ldr or _prev_traj_arg is None:
                    _formation_lon_scores = np.ones(GxM, dtype=np.float32)
                    _formation_lat_scores = np.ones(GxM, dtype=np.float32)
                else:
                    _formation_lon_scores, _formation_lat_scores = compute_pairwise_formation_reward(
                        refined_np, poses[agent_id], _prev_traj_arg, _prev_pose_arg,
                        desired_gap_m=desired_gap_m,
                        lon_decay_m=pdms_params.get("lon_decay_m", 5.0),
                        lat_decay_m=pdms_params.get("lat_decay_m", 0.5),
                        waypoint_decay_gamma=waypoint_decay_gamma,
                    )

                # Per-mode road half-width and lane-y target
                _coarse_np = coarse_traj_by_agent.get(agent_id)
                if _coarse_np is not None and torch.is_tensor(_coarse_np):
                    _coarse_np = _coarse_np.detach().cpu().numpy()
                if _coarse_np is not None:
                    _coarse_np = np.asarray(_coarse_np, dtype=np.float32)

                _road_hw_gm = np.empty(GxM, dtype=np.float32)
                _T_c = (_coarse_np.shape[1] if _coarse_np is not None and _coarse_np.ndim == 3 else 1)
                _anchor_trajs_gm = np.zeros((GxM, _T_c, 2), dtype=np.float32)  # [GxM, T, 2]
                for _m in range(M):
                    _mname = mode_names[_m] if mode_names and _m < len(mode_names) else ""
                    _is_lc = "LC" in _mname
                    _road_hw_gm[_m::M] = gate_road_half_width_m * (2.0 if _is_lc else 1.0)
                    if (_coarse_np is not None and _m < _coarse_np.shape[0]
                            and _coarse_np.ndim == 3 and _coarse_np.shape[2] >= 2):
                        _anchor_trajs_gm[_m::M] = _coarse_np[_m, :, :2]

                # Real collision/road flags from the previous env step for this agent.
                _prev_flags = prev_env_crash_flags.get(agent_id, {})
                _env_crashed = bool(_prev_flags.get("crash", False)) if _prev_flags else None
                _env_out_of_road = bool(_prev_flags.get("out_of_road", False)) if _prev_flags else None

                rewards, reward_debug = _compute_pdms_reward_batch(
                    refined_np, poses[agent_id],
                    _prev_traj_arg, _prev_pose_arg,
                    selected_traj, is_ldr,
                    _formation_lon_scores, _formation_lat_scores, pdms_params,
                    road_half_widths=_road_hw_gm,
                    anchor_trajs=_anchor_trajs_gm,
                    env_crashed=_env_crashed,
                    env_out_of_road=_env_out_of_road,
                )  # rewards: [G*M]; reward_debug arrays are [G*M]

                executed_traj = refined_np[on_training_mode]           # [T, 3]
                executed_reward = float(rewards[on_training_mode])

                # gt_mode: best mode from group 0 (for analysis only, not executed)
                reward_2d = rewards.reshape(num_groups, M)
                _, gt_mode, _ = select_best_mode_group(
                    torch.as_tensor(reward_2d, dtype=torch.float32)
                )

                executed[agent_id] = executed_traj
                step_pdms_by_agent[agent_id] = executed_reward
                step_selected_modes.append(on_training_mode)
                step_gt_modes.append(int(gt_mode))
                step_all_rewards.append(rewards.tolist())
                frame_data_by_agent[agent_id] = {
                    "refined_np":      refined_np,
                    "on_training_mode": on_training_mode,
                    "M":               M,
                    "reward_debug":    reward_debug,
                }

                # Per-vehicle matplotlib plot (local-frame, no renderer needed)
                if test_config["save_traj_plots"]:
                    plot_path = (output_dir / "traj_plots" / agent_id
                                 / f"episode_{episode_idx:03d}"
                                 / f"step_{episode_step:05d}.png")
                    try:
                        _save_agent_traj_plot(
                            plot_path, refined_np, on_training_mode, M,
                            executed_reward, episode_step, agent_id, mode_names or [],
                        )
                    except Exception as _pe:
                        print(f"[test-refine-grpo] traj_plot failed ({agent_id}): {_pe}", flush=True)

            step_pdms_mean = float(np.mean(list(step_pdms_by_agent.values()))) if step_pdms_by_agent else 0.0

            # Combined traj frame: topdown + overlay
            # Captured BEFORE execute so agents are still at current positions.
            _frame_2d_rgb: "np.ndarray | None" = None
            if test_config["save_combined_traj_frames"] or test_config["save_2d_video"]:
                frame_path = (output_dir / "combined_traj_frames"
                              / f"episode_{episode_idx:03d}"
                              / f"step_{episode_step:05d}.png")
                try:
                    _frame_2d_rgb = _save_combined_traj_frame_topdown(
                        base_env=env.base_env,
                        output_path=frame_path,
                        agent_ids=agent_ids,
                        candidates=candidates,
                        poses=poses,
                        frame_data_by_agent=frame_data_by_agent,
                        masks=masks,
                        mode_names=mode_names or [],
                        pdms_reward=step_pdms_mean,
                        episode_step=episode_step,
                        rule_maker_debug=rule_maker_debug,
                    )
                    if not test_config["save_combined_traj_frames"] and frame_path.exists():
                        frame_path.unlink(missing_ok=True)
                except Exception as _fe:
                    print(f"[test-refine-grpo] frame save failed: {_fe}", flush=True)

            # Execute selected trajectories (identical to training line 2298)
            try:
                _, env_reward, done, info = _execute_trajectories(
                    env, planner, agent_ids, executed, env_config
                )
            except AssertionError as e:
                print(f"[test-refine-grpo] env step error: {e}", flush=True)
                done = True
                stop_reason = "env_step_error"
                env_reward = 0.0
                info = {}

            # Store real collision/road flags for use as gates in the NEXT step's reward computation.
            prev_env_crash_flags = (info or {}).get("base_crash_flags", {})
            # Early termination: if any agent crashed or went out of road, end this episode immediately.
            if not done and any(
                flags.get("crash", False) or flags.get("out_of_road", False)
                for flags in prev_env_crash_flags.values()
            ):
                done = True
            if done and stop_reason == "unknown":
                if bool((info or {}).get("truncated", False)):
                    stop_reason = "truncated"
                elif bool((info or {}).get("terminated", False)):
                    stop_reason = "terminated"

            episode_env_reward  += float(env_reward)
            episode_pdms_reward += step_pdms_mean
            episode_step += 1
            vid_2d, vid_3d = _save_step_outputs(
                env=env,
                output_dir=output_dir,
                records_path=records_path,
                all_records=all_records,
                test_config=test_config,
                video_fps=video_fps,
                vid_2d=vid_2d,
                vid_3d=vid_3d,
                episode_idx=episode_idx,
                step_idx=episode_step,
                agent_ids=agent_ids,
                control_backend="diffusion_planner",
                formation_locked=False,
                env_reward=env_reward,
                pdms_reward=step_pdms_mean,
                pdms_by_agent=step_pdms_by_agent,
                selected_modes=step_selected_modes,
                gt_modes=step_gt_modes,
                mode_valid_mask=masks,
                all_rewards=step_all_rewards,
                executed_trajs=executed,
                diffusion_2d_frame=_frame_2d_rgb,
            )

        # Release per-episode video writers
        if vid_2d is not None:
            vid_2d.release()
            vid_2d = None
        if vid_3d is not None:
            vid_3d.release()
            vid_3d = None

        print(
            _format_episode_end_reason(
                episode_idx=episode_idx,
                step_idx=episode_step,
                info=info,
                fallback_reason=stop_reason,
            ),
            flush=True,
        )

        # Episode-end stats
        _info_flat = {k: v for k, v in info.items() if not isinstance(v, dict)}
        if _info_flat.get("arrive_dest"):
            success += 1
        elif _info_flat.get("crash"):
            crash += 1
        elif _info_flat.get("out_of_road"):
            out_of_road += 1

        ep_pdms_per_step = episode_pdms_reward / max(1, episode_step)
        ep_env_per_step  = episode_env_reward  / max(1, episode_step)
        print(f"[test-refine-grpo] episode={episode_idx} steps={episode_step} "
              f"pdms/step={ep_pdms_per_step:.4f} env/step={ep_env_per_step:.4f}", flush=True)

    # Write summary
    metadata = {
        "checkpoint":            str(config.get("pretrained_ckpt", "") or ""),
        "refine_train_config_path": args.refine_train_config_path,
        "scenario_id":           env_config.get("scenario_id"),
        "scenario_ids":          env_config.get("scenario_ids"),
        "local_route":           env_config.get("local_route"),
        "num_agents":            int(config.get("num_agents", 3)),
        "episodes":              episodes,
        "start_seed":            env_config.get("start_seed", env_config.get("seed")),
        "output_dir":            str(output_dir),
    }
    summary = _summarize_results(
        all_records,
        episodes=episodes,
        success=success,
        crash=crash,
        out_of_road=out_of_road,
        metadata=metadata,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[test-refine-grpo] summary saved → {summary_path}", flush=True)
    print(f"  pdms/step  = {summary['episode_pdms_reward_per_step_mean']:.4f}", flush=True)
    print(f"  env/step   = {summary['episode_env_reward_per_step_mean']:.4f}",  flush=True)
    print(f"  success    = {summary['success_rate']:.3f}", flush=True)
    print(f"  crash      = {summary['crash_rate']:.3f}",   flush=True)
    print(f"  out_of_road= {summary['out_of_road_rate']:.3f}", flush=True)

    env.close()
    return summary


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    t0 = time.time()
    run_test(args)
    print(f"[test-refine-grpo] done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
