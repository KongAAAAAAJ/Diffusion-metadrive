"""GRPO fine-tuning of plan_cls_branch in PlatoonDiffusionPlanner.

Only the final decoder layer's classification branch is trainable.
Head vehicle always uses pretrained argmax; follower vehicles learn via
pairwise formation proxy rewards computed across all valid candidate modes.
"""
from __future__ import annotations

import argparse
import copy
import heapq
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml

DEFAULT_CONFIG_PATH = "configs/train/plan_cls_grpo.yaml"
DEFAULT_OUTPUT_ROOT = Path("/media/kong/Elements_SE/Diffusion_Data/outputs/plan_cls_grpo")
_RUN_DIR_RE = re.compile(r"^run_(\d+)$")


# ── config / output dir helpers ───────────────────────────────────────────────

def load_config(path: str | Path) -> dict[str, Any]:
    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def create_next_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    existing = [
        int(m.group(1))
        for item in output_root.iterdir()
        if item.is_dir() and (m := _RUN_DIR_RE.match(item.name))
    ]
    run_dir = output_root / f"run_{max(existing) + 1 if existing else 1}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir


class TopKCheckpointKeeper:
    def __init__(self, k: int):
        self.k = int(k)
        self.heap: list[tuple[float, str]] = []

    def update(self, score: float, path: Path) -> None:
        heapq.heappush(self.heap, (float(score), str(path)))
        if len(self.heap) > self.k:
            _, remove = heapq.heappop(self.heap)
            shutil.rmtree(remove, ignore_errors=True)


# ── model helpers ─────────────────────────────────────────────────────────────

def _get_cls_branch(planner):
    """Return the plan_cls_branch from the last diffusion decoder layer."""
    return planner.model._trajectory_head.diff_decoder.layers[-1].task_decoder.plan_cls_branch


def build_planner_for_grpo(
    config: Mapping[str, Any],
    env_config: Mapping[str, Any],
) -> tuple:
    """Build PlatoonDiffusionPlanner with only plan_cls_branch trainable.

    Returns
    -------
    planner       : PlatoonDiffusionPlanner  (cls_branch has requires_grad=True)
    ref_cls_branch: frozen copy of cls_branch for KL reference
    """
    from metadrive.policy.diffusion_policy.transfuser_config import (
        build_transfuser_config,
        diffusion_model_config_to_overrides,
        load_diffusion_model_config,
    )
    from models.platoon.platoon_diffusion_planner import PlatoonDiffusionPlanner
    from models.platoon.weight_migration import migrate_single_to_platoon

    model_cfg = load_diffusion_model_config(
        str(config.get("model_config_path", "configs/diffusion/model.yaml"))
    )
    tf_config = build_transfuser_config(
        str(model_cfg.get("model_size", "small")),
        **diffusion_model_config_to_overrides(model_cfg),
    )
    planner = PlatoonDiffusionPlanner(tf_config, num_vehicles=int(config.get("num_agents", 3)))

    pretrained_ckpt = str(config.get("pretrained_ckpt", "") or "")
    if pretrained_ckpt:
        planner = migrate_single_to_platoon(pretrained_ckpt, planner)

    device_name = str(env_config.get("planner_device", "cpu"))
    device = torch.device(device_name if torch.cuda.is_available() or "cuda" not in device_name else "cpu")
    planner = planner.to(device)
    planner.eval()

    # Freeze everything, then unfreeze cls_branch
    for p in planner.parameters():
        p.requires_grad_(False)

    cls_branch = _get_cls_branch(planner)
    for p in cls_branch.parameters():
        p.requires_grad_(True)

    # Frozen reference copy for KL regularization
    ref_cls_branch = copy.deepcopy(cls_branch).to(device)
    for p in ref_cls_branch.parameters():
        p.requires_grad_(False)

    n_trainable = sum(p.numel() for p in planner.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in planner.parameters())
    print(
        f"[grpo] trainable params: {n_trainable:,} / {n_total:,}  "
        f"(only plan_cls_branch)",
        flush=True,
    )
    return planner, ref_cls_branch


# ── reward & loss ─────────────────────────────────────────────────────────────

def _local_to_world_traj(pose: np.ndarray, local_traj: np.ndarray) -> np.ndarray:
    """Convert (T, 3) ego-local trajectory to (T, 3) world-frame trajectory."""
    cos_h = np.cos(pose[2])
    sin_h = np.sin(pose[2])
    world = np.empty_like(local_traj)
    world[:, 0] = pose[0] + cos_h * local_traj[:, 0] - sin_h * local_traj[:, 1]
    world[:, 1] = pose[1] + sin_h * local_traj[:, 0] + cos_h * local_traj[:, 1]
    world[:, 2] = pose[2] + local_traj[:, 2]
    return world


def compute_pairwise_formation_reward(
    follower_candidates: np.ndarray,   # [M, 8, 3] ego-local (x, y, heading)
    follower_pose: np.ndarray,         # [3] world (x, y, heading)
    leader_traj: np.ndarray,           # [8, 3] ego-local (leader selected)
    leader_pose: np.ndarray,           # [3] world (x, y, heading)
    desired_gap_m: float = 10.0,
) -> np.ndarray:                       # [M] proxy reward per mode
    """Pairwise formation reward for all follower candidate modes.

    For each candidate, measures how well the follower would maintain
    desired_gap_m behind the leader at each trajectory timestep, then
    applies: R = 0.5*exp(-mean_lon/5.0) + 0.5*exp(-mean_lat/1.0)
    """
    M, T = follower_candidates.shape[:2]
    leader_world = _local_to_world_traj(leader_pose, leader_traj)   # [T, 3]
    rewards = np.empty(M, dtype=np.float32)

    for m in range(M):
        fw = _local_to_world_traj(follower_pose, follower_candidates[m])  # [T, 3]
        lon_errs = np.empty(T)
        lat_errs = np.empty(T)
        for t in range(T):
            lx, ly, lh = leader_world[t]
            # desired position: desired_gap_m behind leader
            des_x = lx - desired_gap_m * np.cos(lh)
            des_y = ly - desired_gap_m * np.sin(lh)
            dx = float(fw[t, 0]) - des_x
            dy = float(fw[t, 1]) - des_y
            lon_errs[t] = abs(dx * np.cos(lh) + dy * np.sin(lh))
            lat_errs[t] = abs(-dx * np.sin(lh) + dy * np.cos(lh))
        rewards[m] = 0.5 * np.exp(-lon_errs.mean() / 5.0) + 0.5 * np.exp(-lat_errs.mean() / 1.0)

    return rewards


def grpo_step_loss(
    logits: torch.Tensor,            # [M] one follower's logits, WITH grad
    ref_logits: torch.Tensor,        # [M] reference logits, no grad
    mode_valid_mask: torch.Tensor,   # [M] bool
    proxy_rewards: np.ndarray,       # [M]
    beta_kl: float = 0.02,
    adv_eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """GRPO loss for a single follower agent.

    Returns (loss_tensor, metrics_dict).
    """
    device = logits.device
    rewards = torch.tensor(proxy_rewards, dtype=torch.float32, device=device)
    valid_idx = mode_valid_mask.nonzero(as_tuple=False).squeeze(-1)

    if valid_idx.numel() < 2:
        # Not enough valid modes to compute meaningful advantage
        zero = logits.sum() * 0.0
        return zero, {"pg_loss": 0.0, "kl_loss": 0.0, "entropy": 0.0, "n_valid": int(valid_idx.numel())}

    r_valid = rewards[valid_idx]
    adv_valid = (r_valid - r_valid.mean()) / (r_valid.std() + adv_eps)

    # Masked softmax — restrict everything to valid indices to avoid 0 * (-inf) = NaN.
    # F.log_softmax on all-neg-inf invalid positions produces -inf; multiplying by the
    # zero-advantage or zero-pi values at those positions gives NaN (IEEE 754).
    # Solution: index into valid_idx first, then compute all quantities.
    neg_inf = torch.full_like(logits, float("-inf"))
    masked_logits = torch.where(mode_valid_mask, logits, neg_inf)
    log_pi_all = F.log_softmax(masked_logits, dim=-1)

    log_pi_valid = log_pi_all[valid_idx]           # finite values only
    pi_new_valid = log_pi_valid.exp()              # equivalent to softmax over valid subset

    pg_loss = -(adv_valid * log_pi_valid).sum()

    # KL: pi_ref || pi_new, computed over valid modes only
    masked_ref = torch.where(mode_valid_mask, ref_logits.float(), neg_inf)
    log_pi_ref_all = F.log_softmax(masked_ref, dim=-1)
    pi_ref_valid = log_pi_ref_all[valid_idx].exp()
    kl_loss = (pi_ref_valid * (log_pi_ref_all[valid_idx] - log_pi_valid)).sum()

    entropy = -(pi_new_valid * log_pi_valid).sum()
    loss = pg_loss + beta_kl * kl_loss

    return loss, {
        "pg_loss": float(pg_loss.detach()),
        "kl_loss": float(kl_loss.detach()),
        "entropy": float(entropy.detach()),
        "n_valid": int(valid_idx.numel()),
    }


# ── vehicle helpers ───────────────────────────────────────────────────────────

def _get_vehicle_pose(env, agent_id: str) -> np.ndarray | None:
    """Return [x, y, heading] world-frame pose or None if vehicle not found."""
    vehicle = getattr(env, "agents", {}).get(agent_id) or getattr(
        getattr(env, "base_env", None), "agents", {}
    ).get(agent_id)
    if vehicle is None:
        return None
    pos = np.asarray(vehicle.position[:2], dtype=np.float64)
    hdg = float(getattr(vehicle, "heading_theta", 0.0))
    return np.asarray([pos[0], pos[1], hdg], dtype=np.float64)


# ── checkpoint saving ─────────────────────────────────────────────────────────

def save_grpo_checkpoint(
    planner,
    pretrained_ckpt_path: str,
    ckpt_dir: Path,
    config: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Save plan_cls_branch.pt, full platoon ckpt, and actor_config.json."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cls_branch = _get_cls_branch(planner)

    # 1. Standalone cls_branch weights
    torch.save(cls_branch.state_dict(), ckpt_dir / "plan_cls_branch.pt")

    # 2. Full PlatoonDiffusionPlanner state dict (compatible with direct load)
    torch.save(
        {"model_state": planner.state_dict(), "config": dict(config)},
        ckpt_dir / "full_platoon_grpo.ckpt",
    )

    # 3. Metadata
    meta = {
        "pretrained_ckpt": pretrained_ckpt_path,
        "num_agents": int(config.get("num_agents", 3)),
        "beta_kl": float(config.get("beta_kl", 0.02)),
        "desired_gap_m": float(config.get("desired_gap_m", 10.0)),
    }
    meta.update(dict(extra or {}))
    (ckpt_dir / "actor_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ── main training loop ────────────────────────────────────────────────────────

def run_training(
    config: Mapping[str, Any],
    output_root: Path,
    total_timesteps: int,
) -> Path:
    from envs.mode_selection_sb3_env import ModeSelectionSB3Env

    run_dir = create_next_run_dir(output_root)
    print(f"[grpo] run_dir: {run_dir}", flush=True)

    # ── env config ────────────────────────────────────────────────────────────
    env_config = dict(config.get("env_config", {}))
    env_config.setdefault("num_agents", int(config.get("num_agents", 3)))
    env_config.setdefault("observation_mode", "multimodal")
    env_config.setdefault("use_render", False)
    env_config.setdefault("planner_device", "cuda")
    env_config.setdefault("lookahead_index", int(config.get("lookahead_index", 2)))
    env_config.setdefault("target_speed_km_h", float(config.get("target_speed_km_h", 30.0)))
    env_config.setdefault("controller_type", str(config.get("controller_type", "stabilized")))
    # GRPO uses coarse for actual env execution (fast) while proxy rewards use diffusion
    env_config["trajectory_source"] = "coarse"
    env_config["use_action_mask"] = False

    # ── planner ───────────────────────────────────────────────────────────────
    planner, ref_cls_branch = build_planner_for_grpo(config, env_config)
    env_config["planner"] = planner

    # ── optimizer ─────────────────────────────────────────────────────────────
    lr = float(config.get("learning_rate", 1e-4))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    beta_kl = float(config.get("beta_kl", 0.02))
    desired_gap_m = float(config.get("desired_gap_m", 10.0))
    ckpt_interval = int(config.get("checkpoint_interval_steps", 5000))
    ckpt_top_k = int(config.get("ckpt_top_k", 3))
    pretrained_ckpt = str(config.get("pretrained_ckpt", ""))

    cls_branch = _get_cls_branch(planner)
    optimizer = torch.optim.Adam(cls_branch.parameters(), lr=lr)
    keeper = TopKCheckpointKeeper(ckpt_top_k)
    ckpt_root = run_dir / "checkpoints"

    # ── tensorboard ───────────────────────────────────────────────────────────
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    except Exception:
        writer = None

    # ── env ───────────────────────────────────────────────────────────────────
    env = ModeSelectionSB3Env(env_config)
    num_agents = int(env_config["num_agents"])

    # ── training state ────────────────────────────────────────────────────────
    global_step = 0
    episode_idx = 0
    episode_rewards: list[float] = []
    last_ckpt_step = 0

    print(
        f"[grpo] starting training: total_steps={total_timesteps}  "
        f"lr={lr}  beta_kl={beta_kl}  desired_gap_m={desired_gap_m}",
        flush=True,
    )

    while global_step < total_timesteps:
        obs, _ = env.reset()
        episode_step = 0
        episode_env_reward = 0.0
        done = False

        while not done and global_step < total_timesteps:
            # ── get trainable logits via GRPO export ──────────────────────────
            planner_batch = env._last_planner_batch
            if not planner_batch:
                break

            grpo = planner.export_for_grpo(planner_batch, ref_cls_branch)
            agent_ids: list[str] = grpo["agent_ids"]
            candidates: np.ndarray = grpo["trajectory_candidates"]   # [N, M, 8, 3]
            logits: torch.Tensor = grpo["logits"]                     # [N, M]
            ref_logits: torch.Tensor = grpo["ref_logits"]             # [N, M]
            mode_valid_mask: torch.Tensor = grpo["mode_valid_mask"]   # [N, M]
            mode_valid_mask_np: np.ndarray = grpo["mode_valid_mask_np"]

            N, M = logits.shape

            # ── get vehicle poses before step ─────────────────────────────────
            poses = {aid: _get_vehicle_pose(env, aid) for aid in agent_ids}

            # ── select modes ──────────────────────────────────────────────────
            mode_actions = np.empty(N, dtype=np.int64)
            for i, aid in enumerate(agent_ids):
                if i == 0:
                    # Head vehicle: pretrained argmax (no learning)
                    mode_actions[i] = int(grpo["pretrained_argmax_mode"][i])
                else:
                    # Follower: sample from masked distribution during training
                    with torch.no_grad():
                        masked = logits[i].masked_fill(~mode_valid_mask[i], float("-inf"))
                        probs = F.softmax(masked, dim=-1)
                        mode_actions[i] = int(torch.multinomial(probs, 1).item())

            # ── compute proxy rewards for each follower ───────────────────────
            follower_step_loss = torch.zeros(1, device=logits.device)
            step_metrics: list[dict] = []

            for i in range(1, N):
                aid = agent_ids[i]
                prev_aid = agent_ids[i - 1]
                pose_i = poses.get(aid)
                pose_prev = poses.get(prev_aid)
                if pose_i is None or pose_prev is None:
                    continue

                leader_selected_traj = candidates[i - 1, mode_actions[i - 1]]  # [8, 3]
                follower_cands = candidates[i]                                   # [M, 8, 3]

                proxy_rewards = compute_pairwise_formation_reward(
                    follower_cands, pose_i, leader_selected_traj, pose_prev, desired_gap_m
                )

                # Zero out invalid modes
                proxy_rewards = proxy_rewards * mode_valid_mask_np[i].astype(np.float32)

                loss_i, metrics_i = grpo_step_loss(
                    logits[i], ref_logits[i], mode_valid_mask[i],
                    proxy_rewards, beta_kl=beta_kl,
                )
                follower_step_loss = follower_step_loss + loss_i
                step_metrics.append(metrics_i)

            if len(step_metrics) > 0:
                follower_step_loss = follower_step_loss / len(step_metrics)
                optimizer.zero_grad()
                follower_step_loss.backward()
                torch.nn.utils.clip_grad_norm_(cls_branch.parameters(), max_grad_norm)
                optimizer.step()

            # ── step env ──────────────────────────────────────────────────────
            obs, env_reward, terminated, truncated, info = env.step(mode_actions)
            done = bool(terminated) or bool(truncated)
            episode_env_reward += float(env_reward)
            episode_step += 1
            global_step += 1

            # ── logging ───────────────────────────────────────────────────────
            if writer is not None and step_metrics:
                avg_pg = float(np.mean([m["pg_loss"] for m in step_metrics]))
                avg_kl = float(np.mean([m["kl_loss"] for m in step_metrics]))
                avg_ent = float(np.mean([m["entropy"] for m in step_metrics]))
                writer.add_scalar("train/pg_loss", avg_pg, global_step)
                writer.add_scalar("train/kl_loss", avg_kl, global_step)
                writer.add_scalar("train/entropy", avg_ent, global_step)
                writer.add_scalar("train/grpo_loss", float(follower_step_loss.detach()), global_step)
                writer.add_scalar("train/env_reward_step", float(env_reward), global_step)

            # ── periodic checkpoint ───────────────────────────────────────────
            if global_step - last_ckpt_step >= ckpt_interval:
                last_ckpt_step = global_step
                ep_reward_per_step = episode_env_reward / max(1, episode_step)
                ckpt_dir = ckpt_root / f"step_{global_step:07d}_score_{ep_reward_per_step:.4f}"
                save_grpo_checkpoint(
                    planner, pretrained_ckpt, ckpt_dir, config,
                    {"score": ep_reward_per_step, "global_step": global_step, "episode": episode_idx},
                )
                keeper.update(ep_reward_per_step, ckpt_dir)
                print(
                    f"[grpo] step={global_step}  ep={episode_idx}  "
                    f"ep_reward/step={ep_reward_per_step:.4f}  ckpt saved",
                    flush=True,
                )

        # ── end of episode ────────────────────────────────────────────────────
        ep_reward_per_step = episode_env_reward / max(1, episode_step)
        episode_rewards.append(ep_reward_per_step)
        if writer is not None:
            writer.add_scalar("train/episode_reward_per_step", ep_reward_per_step, episode_idx)
        print(
            f"[grpo] episode={episode_idx}  steps={episode_step}  "
            f"reward/step={ep_reward_per_step:.4f}  global_step={global_step}",
            flush=True,
        )
        episode_idx += 1

    # ── final checkpoint ──────────────────────────────────────────────────────
    final_dir = ckpt_root / "final"
    save_grpo_checkpoint(
        planner, pretrained_ckpt, final_dir, config,
        {"note": "final checkpoint", "total_steps": global_step, "total_episodes": episode_idx},
    )

    (run_dir / "summary.json").write_text(
        json.dumps({
            "total_steps": global_step,
            "total_episodes": episode_idx,
            "mean_episode_reward_per_step": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        }, indent=2),
        encoding="utf-8",
    )

    if writer is not None:
        writer.close()

    return run_dir


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO fine-tuning of plan_cls_branch.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--pretrained-ckpt",
        default="/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_20/checkpoints/diffusion-epoch=25.ckpt",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--total-env-steps", type=int, default=50000)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--planner-device", default="cuda")
    parser.add_argument("--use-render", type=int, choices=(0, 1), default=0)
    parser.add_argument("--scenario-ids", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.pretrained_ckpt:
        config["pretrained_ckpt"] = args.pretrained_ckpt
    if args.num_agents > 0:
        config["num_agents"] = int(args.num_agents)
        config.setdefault("env_config", {})["num_agents"] = int(args.num_agents)
    if args.planner_device:
        config.setdefault("env_config", {})["planner_device"] = args.planner_device
    config.setdefault("env_config", {})["use_render"] = bool(args.use_render)
    if args.scenario_ids:
        ids = [s.strip() for s in args.scenario_ids.split(",") if s.strip()]
        if ids:
            config.setdefault("env_config", {})["scenario_ids"] = ids

    total_steps = int(args.total_env_steps or config.get("total_timesteps", 50000))
    run_dir = run_training(config, Path(args.output_root), total_steps)
    print(f"[grpo] outputs: {run_dir}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
