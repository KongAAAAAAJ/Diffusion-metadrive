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

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = str(_REPO_ROOT / "configs" / "train" / "refine_grpo.yaml")
_DEFAULT_OUTPUT = str(_REPO_ROOT / "outputs" / "refine_grpo_test")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Test selected-refine-GRPO checkpoint (training-aligned)")
    p.add_argument("--checkpoint",               type=str, default="/media/kong/Elements_SE/Diffusion_Data/outputs/refine_grpo/run_25/checkpoints/step_0035000_score_0.6398/full_platoon_refine_grpo.ckpt",
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
    p.add_argument("--save-traj-plots",          type=int, default=0, choices=[0, 1],
                   help="Save per-vehicle matplotlib trajectory plots to traj_plots/{agent_id}/")
    p.add_argument("--save-trajectory-data",     type=int, default=1, choices=[0, 1])
    p.add_argument("--save-2d-video",            type=int, default=1, choices=[0, 1],
                   help="Compile topdown frames into per-episode MP4 (videos/2d/episode_XXX.mp4)")
    p.add_argument("--save-3d-video",            type=int, default=0, choices=[0, 1],
                   help="Capture MetaDrive 3D render into per-episode MP4 (videos/3d/episode_XXX.mp4). Forces use_render=True.")
    p.add_argument("--video-fps",                type=int, default=10,
                   help="FPS for output MP4 videos")
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


# ── Env config builder (mirrors run_training in train_refine_grpo.py) ─

def _build_env_config(config: dict, args, planner) -> dict:
    env_config = dict(config.get("env_config", {}))
    env_config.setdefault("num_agents",          int(config.get("num_agents", 3)))
    env_config.setdefault("observation_mode",    "multimodal")
    env_config.setdefault("use_render",          bool(args.render) or bool(args.save_3d_video))
    env_config.setdefault("planner_device",      args.device)
    env_config.setdefault("lookahead_index",     int(config.get("lookahead_index", 2)))
    env_config.setdefault("target_speed_km_h",   float(config.get("target_speed_km_h", 30.0)))
    env_config.setdefault("controller_type",     str(config.get("controller_type", "stabilized")))
    env_config["use_action_mask"]    = False
    env_config["trajectory_source"] = "diffusion"
    env_config["planner"]            = planner

    # Scenario / seed
    if args.scenario_id:
        from scenarios.definitions import SCENARIO_BY_ID

        env_config.pop("scenario_ids", None)
        env_config["scenario_id"] = args.scenario_id
        if not args.local_route:
            if args.scenario_id not in SCENARIO_BY_ID:
                raise ValueError(f"Unknown scenario_id in selected-refine test config: {args.scenario_id}")
            allowed_routes = tuple(SCENARIO_BY_ID[args.scenario_id].allowed_local_routes)
            if not allowed_routes:
                raise ValueError(f"Scenario {args.scenario_id} has no allowed local routes.")
            env_config["local_route"] = allowed_routes[0]
    if args.local_route:
        env_config["local_route"] = args.local_route
    env_config.setdefault("start_seed",       args.start_seed)
    env_config.setdefault("num_scenarios",    args.num_scenarios)
    env_config.setdefault("traffic_density",  args.traffic_density)
    env_config.setdefault("random_traffic",   bool(args.random_traffic))
    return env_config


# ── PDMS param dict (mirrors run_training) ────────────────────────────────────



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
):
    from models.diffusion.test_transfuser_policy import (
        _build_step_overlay_polylines,
        _capture_topdown_frame_with_overlay,
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

    def _gc(v: float) -> str:
        return "pass" if v >= 1.0 else "FAIL"

    def _put(text: str, y: int, scale: float = 0.42) -> None:
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (30, 30, 200), 1, cv2.LINE_AA)

    step_text = f"step={episode_step}  pdms_mean={pdms_reward:+.3f}"
    _put(step_text, 20, scale=0.48)

    y_cursor = 44
    line_h = 20
    for agent_id in agent_ids:
        rd = frame_data_by_agent.get(agent_id, {})
        dbg = rd.get("reward_debug", {})
        gm = rd.get("on_training_mode", 0)

        def _gv(key, default=0.0):
            v = dbg.get(key, default)
            return float(v[gm]) if hasattr(v, "__len__") else float(v)

        label = agent_id.replace("agent", "A")
        header = f"{label}: pdms={_gv('reward'):+.3f}  qual={_gv('quality'):.3f}"
        gate_line = (
            f"  gate: col={_gc(_gv('collision_gate', 1))}"
            f" road={_gc(_gv('road_gate', 1))}"
            f" smth={_gc(_gv('smoothness_gate', 1))}"
        )
        qual_line = (
            f"  qual: prog={_gv('progress'):.2f}"
            f" form={_gv('formation'):.2f}"
            f" spd={_gv('speed'):.2f}"
            f" lane={_gv('lane'):.2f}"
            f" cmft={_gv('comfort'):.2f}"
            f" cons={_gv('consistency'):.2f}"
        )
        for text in (header, gate_line, qual_line):
            _put(text, y_cursor)
            y_cursor += line_h
        y_cursor += 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return frame  # RGB, for video writing


# ── Video writer helper ───────────────────────────────────────────────────────

def _open_video_writer(path: pathlib.Path, fps: int, width: int, height: int) -> "cv2.VideoWriter":
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    return writer


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

    train_followers_only = bool(config.get("train_followers_only", False))

    # When train_followers_only: keep the original pretrained_ckpt for the leader planner,
    # then override config["pretrained_ckpt"] with the GRPO ckpt for followers.
    _original_pretrained_ckpt = str(config.get("pretrained_ckpt", "") or "")

    # CLI checkpoint overrides YAML pretrained_ckpt
    if args.checkpoint:
        config["pretrained_ckpt"] = args.checkpoint

    # Env config device override from CLI
    config.setdefault("num_agents", 3)
    env_config_pre = dict(config.get("env_config", {}))
    env_config_pre["planner_device"] = args.device

    # Build planner (same as training) — used by followers and env mode selection
    print("[test-refine-grpo] building planner ...", flush=True)
    planner = build_planner_for_selected_refinement(config, {**env_config_pre, "planner_device": args.device})
    planner.eval()
    for p in planner.parameters():
        p.requires_grad_(False)

    # When train_followers_only: build a separate frozen pretrained planner for the leader
    if train_followers_only:
        print("[test-refine-grpo] train_followers_only=True: building frozen pretrain_planner for leader ...", flush=True)
        _pretrain_config = dict(config)
        _pretrain_config["pretrained_ckpt"] = _original_pretrained_ckpt
        pretrain_planner = build_planner_for_selected_refinement(
            _pretrain_config, {**env_config_pre, "planner_device": args.device}
        )
        pretrain_planner.eval()
        for p in pretrain_planner.parameters():
            p.requires_grad_(False)
        print("[test-refine-grpo] pretrain_planner ready.", flush=True)
    else:
        pretrain_planner = None

    # Build env (same as training)
    from envs.wrap_platoon_env import ModeSelectionSB3Env
    env_config = _build_env_config(config, args, planner)
    env = ModeSelectionSB3Env(env_config)
    print(f"[test-refine-grpo] env ready: {type(env).__name__}", flush=True)

    # External upper-level decision model (used when target_guidance_type='external_point')
    _target_guidance_type = str(
        planner._model_config.target_guidance_type
        if hasattr(planner, "_model_config")
        else config.get("target_guidance_type", "multi_point")
    )
    _rule_maker = None
    if _target_guidance_type == "external_point":
        from models.decision.rule_decisioner import make_rule_maker
        _rule_maker = make_rule_maker(dict(config))
        print(f"[test-refine-grpo] RuleMaker enabled: {_rule_maker.__class__.__name__}", flush=True)

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
        if _rule_maker is not None:
            _rule_maker.reset(env, list(getattr(env, "_agent_ids", [])))

        done = False
        episode_step = 0
        episode_env_reward = 0.0
        episode_pdms_reward = 0.0
        vid_2d: "cv2.VideoWriter | None" = None
        vid_3d: "cv2.VideoWriter | None" = None
        # Real collision/road flags from the previous env step, keyed by agent_id.
        # None at the start of an episode (no prior step executed).
        prev_env_crash_flags: dict[str, dict] = {}

        while not done:
            if args.max_steps > 0 and episode_step >= args.max_steps:
                break

            planner_batch = env._last_planner_batch
            export        = env._last_export
            if not planner_batch or export is None:
                break

            agent_ids        = list(export["agent_ids"])
            candidates       = np.asarray(export["trajectory_candidates"], dtype=np.float32)  # [N, M, T, 3]
            fix_candidates_heading_inplace(candidates)   # tanh*π → atan2(Δy,Δx)
            masks            = np.asarray(export["lane_valid_mask"], dtype=bool)              # [N, M]
            masked_logits_np = np.asarray(export["masked_cls_logits"], dtype=np.float32)     # [N, M]
            on_training_modes = np.argmax(masked_logits_np, axis=-1).astype(np.int64)        # [N]

            # Inject external target_point before extract_rl_context so it flows
            # into model features automatically (external_point guidance mode).
            if _rule_maker is not None:
                from models.decision.rule_decisioner import inject_rule_maker_target_points
                inject_rule_maker_target_points(_rule_maker, env, agent_ids, planner_batch)

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

            # Combined traj frame: topdown + overlay
            # Captured BEFORE execute so agents are still at current positions.
            _frame_2d_rgb: "np.ndarray | None" = None
            if args.save_combined_traj_frames or args.save_2d_video:
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
                    )
                    if not args.save_combined_traj_frames and frame_path.exists():
                        frame_path.unlink(missing_ok=True)
                except Exception as _fe:
                    print(f"[test-refine-grpo] frame save failed: {_fe}", flush=True)

            # Write 2D frame to video
            if args.save_2d_video and _frame_2d_rgb is not None:
                try:
                    _bgr_2d = cv2.cvtColor(_frame_2d_rgb, cv2.COLOR_RGB2BGR)
                    if vid_2d is None:
                        h2d, w2d = _bgr_2d.shape[:2]
                        _vpath_2d = output_dir / "videos" / "2d" / f"episode_{episode_idx:03d}.mp4"
                        vid_2d = _open_video_writer(_vpath_2d, args.video_fps, w2d, h2d)
                    vid_2d.write(_bgr_2d)
                except Exception as _ve:
                    print(f"[test-refine-grpo] 2D video write failed: {_ve}", flush=True)

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

            # Store real collision/road flags for use as gates in the NEXT step's reward computation.
            prev_env_crash_flags = (info or {}).get("base_crash_flags", {})
            # Early termination: if any agent crashed or went out of road, end this episode immediately.
            if not done and any(
                flags.get("crash", False) or flags.get("out_of_road", False)
                for flags in prev_env_crash_flags.values()
            ):
                done = True

            # Capture 3D frame after env step (requires use_render=True)
            if args.save_3d_video:
                try:
                    _frame_3d = env.render(mode="rgb_array")
                    if _frame_3d is not None and isinstance(_frame_3d, np.ndarray):
                        _bgr_3d = (cv2.cvtColor(_frame_3d, cv2.COLOR_RGB2BGR)
                                   if _frame_3d.ndim == 3 and _frame_3d.shape[2] == 3
                                   else _frame_3d)
                        if vid_3d is None:
                            h3d, w3d = _bgr_3d.shape[:2]
                            _vpath_3d = output_dir / "videos" / "3d" / f"episode_{episode_idx:03d}.mp4"
                            vid_3d = _open_video_writer(_vpath_3d, args.video_fps, w3d, h3d)
                        vid_3d.write(_bgr_3d)
                except Exception as _ve3:
                    pass  # 3D render unavailable in this env config

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

        # Release per-episode video writers
        if vid_2d is not None:
            vid_2d.release()
            vid_2d = None
        if vid_3d is not None:
            vid_3d.release()
            vid_3d = None

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
