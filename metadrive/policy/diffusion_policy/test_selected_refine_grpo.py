"""
Test script for selected-mode local refinement GRPO checkpoints.

Execution flow is strictly aligned with train/train_selected_refine_grpo.py:
  - Env:     ModeSelectionSB3Env  (identical to training)
  - Planner: build_planner_for_selected_refinement()  (identical to training)
  - Rollout: rollout_multimodal_refinement()  (identical to training)
  - Reward:  _compute_pdms_reward_batch() on refined_traj (heading already atan2-reconstructed
             inside rollout, matching training exactly)
  - Mode:    argmax(masked_cls_logits) with fallback  (identical to training on_training_mode)
  - Execute: _execute_trajectories()  (identical to training)

Saves (consistent with run_grpo_test / test_transfuser_policy.py):
  random_action_reward_summary.json
  random_action_step_records.jsonl
  combined_traj_frames/episode_XXX/step_XXXXX.png   (topdown + overlay, same as run_grpo_test)
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
from train.train_selected_refine_grpo import (
    CBFBarrierGuidanceConfig,
    _compute_pdms_reward_batch,
    _execute_trajectories,
    _resolve_barrier_guidance_config,
    _summarize_barrier_metrics,
    build_planner_for_selected_refinement,
    mode_names_from_planner,
    rollout_multimodal_refinement,
    select_best_mode_group,
)
from train.train_plan_cls_grpo import (
    _get_vehicle_pose,
    compute_pairwise_formation_reward,
)
from models.diffusion.ddim_with_logprob import DDIMSchedulerWithLogProb

# ─────────────────────────────────────────────────────────────────────────────

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = str(_REPO_ROOT / "configs" / "train" / "selected_refine_grpo.yaml")
_DEFAULT_OUTPUT = str(_REPO_ROOT / "outputs" / "selected_refine_grpo_test")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Test selected-refine-GRPO checkpoint (training-aligned)")
    p.add_argument("--checkpoint",               type=str, default="",
                   help="Path to full_platoon_refine_grpo.ckpt (overrides pretrained_ckpt in YAML)")
    p.add_argument("--refine-train-config-path", type=str, default=_DEFAULT_CONFIG)
    p.add_argument("--scenario-id",              type=str, default="S1_free_cruise_straight")
    p.add_argument("--local-route",              type=str, default="")
    p.add_argument("--episodes",                 type=int, default=3)
    p.add_argument("--start-seed",               type=int, default=0)
    p.add_argument("--num-scenarios",            type=int, default=1)
    p.add_argument("--traffic-density",          type=float, default=0.04)
    p.add_argument("--random-traffic",           type=int, default=0,   choices=[0, 1])
    p.add_argument("--max-steps",                type=int, default=0,
                   help="Max steps per episode (0 = unlimited)")
    p.add_argument("--device",                   type=str, default="cuda")
    p.add_argument("--output-dir",               type=str, default=_DEFAULT_OUTPUT)
    p.add_argument("--save-combined-traj-frames", type=int, default=1, choices=[0, 1])
    p.add_argument("--save-traj-plots",          type=int, default=1, choices=[0, 1],
                   help="Save per-vehicle matplotlib trajectory plots to traj_plots/{agent_id}/")
    p.add_argument("--save-trajectory-data",     type=int, default=1, choices=[0, 1])
    p.add_argument("--render",                   type=int, default=0, choices=[0, 1])
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


# ── Env config builder (mirrors run_training in train_selected_refine_grpo.py) ─

def _build_env_config(config: dict, args, planner) -> dict:
    env_config = dict(config.get("env_config", {}))
    env_config.setdefault("num_agents",          int(config.get("num_agents", 3)))
    env_config.setdefault("observation_mode",    "multimodal")
    env_config.setdefault("use_render",          bool(args.render))
    env_config.setdefault("planner_device",      args.device)
    env_config.setdefault("lookahead_index",     int(config.get("lookahead_index", 2)))
    env_config.setdefault("target_speed_km_h",   float(config.get("target_speed_km_h", 30.0)))
    env_config.setdefault("controller_type",     str(config.get("controller_type", "stabilized")))
    env_config["use_action_mask"]    = False
    env_config["trajectory_source"] = "diffusion"
    env_config["planner"]            = planner

    # Scenario / seed
    if args.scenario_id:
        env_config["scenario_id"] = args.scenario_id
    if args.local_route:
        env_config["local_route"] = args.local_route
    env_config.setdefault("start_seed",       args.start_seed)
    env_config.setdefault("num_scenarios",    args.num_scenarios)
    env_config.setdefault("traffic_density",  args.traffic_density)
    env_config.setdefault("random_traffic",   bool(args.random_traffic))
    return env_config


# ── PDMS param dict (mirrors run_training) ────────────────────────────────────

def _build_pdms_params(config: dict) -> dict:
    return {
        "gate_collision_dist_m":  float(config.get("gate_collision_dist_m",  1.0)),
        "gate_ttc_s":             float(config.get("gate_ttc_s",             0.5)),
        "gate_road_half_width_m": float(config.get("gate_road_half_width_m", 3.0)),
        "gate_max_dh_rad":        float(config.get("gate_max_dh_rad",        0.6)),
        "w_progress":             float(config.get("w_progress",             0.2)),
        "w_formation":            float(config.get("w_formation",            0.4)),
        "w_speed":                float(config.get("w_speed",                0.2)),
        "w_lane":                 float(config.get("w_lane",                 0.15)),
        "w_comfort":              float(config.get("w_comfort",              0.05)),
        "w_pretrain":             float(config.get("w_pretrain",             0.0)),
        "progress_s_max":         float(config.get("progress_s_max",         15.0)),
        "target_speed_kmh":       float(config.get("target_speed_km_h", config.get("target_speed_kmh", 30.0))),
        "lane_decay_m":           float(config.get("lane_decay_m",           1.0)),
        "comfort_decay_rad":      float(config.get("comfort_decay_rad",      0.1)),
        "consistency_decay_m":    float(config.get("consistency_decay_m",    1.0)),
        "desired_gap_m":          float(config.get("desired_gap_m",          10.0)),
        "vehicle_length_m":       float(config.get("vehicle_length_m",       4.8)),
        "gate_horizon_steps":     int(config.get("gate_horizon_steps",       2)),
    }


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


# ── Combined traj frame (topdown + overlay, same as run_grpo_test) ────────────

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
):
    from metadrive.policy.diffusion_policy.test_transfuser_policy import (
        _build_step_overlay_polylines,
        _capture_topdown_frame_with_overlay,
        _format_combined_frame_reward_text,
    )

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
    )
    if frame is None:
        return

    reward_text = _format_combined_frame_reward_text(step=episode_step, pdms_reward=pdms_reward)
    cv2.putText(frame, reward_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, reward_text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 200), 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


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

    # CLI checkpoint overrides YAML pretrained_ckpt
    if args.checkpoint:
        config["pretrained_ckpt"] = args.checkpoint

    # Env config device override from CLI
    config.setdefault("num_agents", 3)
    env_config_pre = dict(config.get("env_config", {}))
    env_config_pre["planner_device"] = args.device

    # Build planner (same as training)
    print("[test-refine-grpo] building planner ...", flush=True)
    planner = build_planner_for_selected_refinement(config, {**env_config_pre, "planner_device": args.device})
    planner.eval()
    for p in planner.parameters():
        p.requires_grad_(False)

    # Build env (same as training)
    from envs.mode_selection_sb3_env import ModeSelectionSB3Env
    env_config = _build_env_config(config, args, planner)
    env = ModeSelectionSB3Env(env_config)
    print(f"[test-refine-grpo] env ready: {type(env).__name__}", flush=True)

    # Rollout / DDIM params
    num_groups        = int(config.get("num_refine_groups",   4))
    rollout_chunk_size= int(config.get("rollout_chunk_size",  80))
    noise_t           = int(config.get("refine_noise_t",      8))
    denoise_steps     = int(config.get("refine_denoise_steps",4))
    eta               = float(config.get("refine_eta",        0.1))
    desired_gap_m     = float(config.get("desired_gap_m",     10.0))
    progress_s_max    = float(config.get("progress_s_max",    15.0))

    pdms_params = _build_pdms_params(config)
    gate_road_half_width_m = pdms_params["gate_road_half_width_m"]

    # DDIM scheduler + barrier config (same as training)
    refine_scheduler = DDIMSchedulerWithLogProb(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="sample",
    )
    refine_scheduler.set_timesteps(1000, device=planner._device())
    barrier_config = _resolve_barrier_guidance_config(config, refine_scheduler, noise_t)

    # Mode names for road-half-width and visualization
    mode_names = mode_names_from_planner(planner, None)

    # Output paths
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "random_action_step_records.jsonl"
    summary_path = output_dir / "random_action_reward_summary.json"

    all_records: list[dict] = []
    episode_idx = 0
    success = crash = out_of_road = 0

    for episode_idx in range(args.episodes):
        print(f"[test-refine-grpo] episode {episode_idx}/{args.episodes}", flush=True)
        env.reset()

        done = False
        episode_step = 0
        episode_env_reward = 0.0
        episode_pdms_reward = 0.0

        while not done:
            if args.max_steps > 0 and episode_step >= args.max_steps:
                break

            planner_batch = env._last_planner_batch
            export        = env._last_export
            if not planner_batch or export is None:
                break

            agent_ids        = list(export["agent_ids"])
            candidates       = np.asarray(export["trajectory_candidates"], dtype=np.float32)  # [N, M, T, 3]
            masks            = np.asarray(export["mode_valid_mask"], dtype=bool)              # [N, M]
            masked_logits_np = np.asarray(export["masked_cls_logits"], dtype=np.float32)     # [N, M]
            on_training_modes = np.argmax(masked_logits_np, axis=-1).astype(np.int64)        # [N]

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
                    rollout = rollout_multimodal_refinement(
                        planner,
                        contexts[agent_id],
                        num_groups=num_groups,
                        noise_t=noise_t,
                        denoise_steps=denoise_steps,
                        eta=eta,
                        scheduler=refine_scheduler,
                        chunk_size=rollout_chunk_size,
                        coarse_trajectories=coarse_traj_by_agent.get(agent_id),
                        barrier_config=barrier_config,
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
                    _formation_scores = np.ones(GxM, dtype=np.float32)
                else:
                    _formation_scores = compute_pairwise_formation_reward(
                        refined_np, poses[agent_id], _prev_traj_arg, _prev_pose_arg,
                        desired_gap_m=desired_gap_m, progress_s_max=progress_s_max,
                    )

                # Per-mode road half-width and lane-y target
                _coarse_np = coarse_traj_by_agent.get(agent_id)
                if _coarse_np is not None and torch.is_tensor(_coarse_np):
                    _coarse_np = _coarse_np.detach().cpu().numpy()
                if _coarse_np is not None:
                    _coarse_np = np.asarray(_coarse_np, dtype=np.float32)

                _road_hw_gm    = np.empty(GxM, dtype=np.float32)
                _lane_y_tgt_gm = np.zeros(GxM, dtype=np.float32)
                for _m in range(M):
                    _mname = mode_names[_m] if mode_names and _m < len(mode_names) else ""
                    _is_lc = "LC" in _mname
                    _road_hw_gm[_m::M] = gate_road_half_width_m * (2.0 if _is_lc else 1.0)
                    if (_coarse_np is not None and _m < _coarse_np.shape[0]
                            and _coarse_np.ndim == 3 and _coarse_np.shape[2] >= 2):
                        _lane_y_tgt_gm[_m::M] = float(_coarse_np[_m, -1, 1])

                rewards = _compute_pdms_reward_batch(
                    refined_np, poses[agent_id],
                    _prev_traj_arg, _prev_pose_arg,
                    selected_traj, is_ldr,
                    _formation_scores, pdms_params,
                    road_half_widths=_road_hw_gm,
                    lane_y_targets=_lane_y_tgt_gm,
                )  # [G*M]

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
                }

                # Per-vehicle matplotlib plot (local-frame, no renderer needed)
                if args.save_traj_plots:
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

            # Combined traj frame: topdown + overlay (same as run_grpo_test)
            # Captured BEFORE execute so agents are still at current positions.
            if args.save_combined_traj_frames:
                frame_path = (output_dir / "combined_traj_frames"
                              / f"episode_{episode_idx:03d}"
                              / f"step_{episode_step:05d}.png")
                try:
                    _save_combined_traj_frame_topdown(
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
                    )
                except Exception as _fe:
                    print(f"[test-refine-grpo] frame save failed: {_fe}", flush=True)

            # Trajectory data: all agents in one JSON
            if args.save_trajectory_data:
                tdata = {
                    "episode": episode_idx,
                    "step": episode_step,
                    "agents": {},
                }
                for _idx2, _aid2 in enumerate(agent_ids):
                    _pose = poses.get(_aid2)
                    tdata["agents"][_aid2] = {
                        "pose": _pose.tolist() if _pose is not None else None,
                        "selected_mode": int(step_selected_modes[_idx2]),
                        "gt_mode": int(step_gt_modes[_idx2]),
                        "executed_traj_local": executed[_aid2].tolist() if _aid2 in executed else [],
                    }
                tdata_path = (output_dir / "trajectory_data"
                              / f"episode_{episode_idx:03d}"
                              / f"step_{episode_step:05d}.json")
                tdata_path.parent.mkdir(parents=True, exist_ok=True)
                tdata_path.write_text(json.dumps(tdata), encoding="utf-8")

            # Execute selected trajectories (identical to training line 2298)
            try:
                _, env_reward, done, info = _execute_trajectories(
                    env, planner, agent_ids, executed, env_config
                )
            except AssertionError as e:
                print(f"[test-refine-grpo] env step error: {e}", flush=True)
                done = True
                env_reward = 0.0
                info = {}

            episode_env_reward  += float(env_reward)
            episode_pdms_reward += step_pdms_mean
            episode_step += 1

            record = {
                "episode":        episode_idx,
                "step":           episode_step,
                "env_reward":     float(env_reward),
                "pdms_reward":    step_pdms_mean,
                "pdms_by_agent":  step_pdms_by_agent,
                "selected_modes": step_selected_modes,
                "gt_modes":       step_gt_modes,
                "mode_valid_mask": masks.tolist(),
                "all_rewards":    step_all_rewards,
            }
            all_records.append(record)

            with records_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")

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
        "checkpoint":            args.checkpoint,
        "refine_train_config_path": args.refine_train_config_path,
        "scenario_id":           args.scenario_id,
        "local_route":           args.local_route,
        "num_agents":            int(config.get("num_agents", 3)),
        "episodes":              args.episodes,
        "start_seed":            args.start_seed,
    }
    summary = _summarize_results(
        all_records,
        episodes=args.episodes,
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
