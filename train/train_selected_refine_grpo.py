"""Selected-mode local refinement GRPO.

This trainer keeps the existing mode-selection planner path intact, then
fine-tunes only the last trajectory generation submodule on small local
refinements around the selected mode trajectory.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import heapq
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from models.diffusion.ddim_with_logprob import DDIMSchedulerWithLogProb
from metadrive.policy.diffusion_policy.modules.multimodal_loss import py_sigmoid_focal_loss
from metadrive.policy.diffusion_policy.transfuser_policy import compute_trajectory_control
from train.train_plan_cls_grpo import _get_cls_branch, _get_vehicle_pose, compute_pairwise_formation_reward, _traj_speed_kmh

DEFAULT_CONFIG_PATH = "configs/train/selected_refine_grpo.yaml"
DEFAULT_OUTPUT_ROOT = Path("/media/kong/Elements_SE/Diffusion_Data/outputs/selected_refine_grpo")
_RUN_DIR_RE = re.compile(r"^run_(\d+)$")


def _env_default_cls_grpo_ckpt_dir() -> str:
    explicit = str(os.environ.get("CLS_GRPO_CKPT_DIR", "") or "").strip()
    full = str(os.environ.get("CLS_GRPO_FULL_CKPT", "") or "").strip()
    run_dir = str(os.environ.get("PLAN_CLS_GRPO_RUN_DIR", "") or "").strip()
    if explicit:
        return explicit
    if not full and run_dir:
        return str(Path(run_dir) / "checkpoints" / "final")
    return ""


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


def _last_task_decoder(planner):
    return planner.model._trajectory_head.diff_decoder.layers[-1].task_decoder


def load_cls_grpo_weights_for_refinement(
    planner,
    config: Mapping[str, Any],
    device: torch.device,
) -> str:
    """Optionally load classification-head GRPO weights before refinement.

    Returns the concrete source path used, or an empty string when refinement
    intentionally falls back to the classifier already present in the planner.
    """
    from models.platoon.weight_migration import load_platoon_grpo_checkpoint

    cls_grpo_ckpt_dir = str(config.get("cls_grpo_ckpt_dir", "") or "").strip()
    cls_grpo_full_ckpt = str(config.get("cls_grpo_full_ckpt", "") or "").strip()
    if cls_grpo_ckpt_dir and cls_grpo_full_ckpt:
        raise ValueError("Only one of cls_grpo_ckpt_dir and cls_grpo_full_ckpt may be provided.")

    if cls_grpo_full_ckpt:
        full_path = Path(cls_grpo_full_ckpt)
        if not full_path.exists():
            raise FileNotFoundError(f"[refine-grpo] cls_grpo_full_ckpt not found: {full_path}")
        load_platoon_grpo_checkpoint(str(full_path), planner)
        print(f"[refine-grpo] loaded cls GRPO full planner checkpoint: {full_path}", flush=True)
        return str(full_path)

    if cls_grpo_ckpt_dir:
        cls_path = Path(cls_grpo_ckpt_dir) / "plan_cls_branch.pt"
        if not cls_path.exists():
            raise FileNotFoundError(f"[refine-grpo] plan_cls_branch.pt not found: {cls_path}")
        state = torch.load(str(cls_path), map_location=device)
        _get_cls_branch(planner).load_state_dict(state)
        print(f"[refine-grpo] loaded cls GRPO branch weights: {cls_path}", flush=True)
        return str(cls_path)

    print("[refine-grpo] WARNING: no cls GRPO weights provided; using classifier from pretrained_ckpt.", flush=True)
    return ""


def build_planner_for_selected_refinement(config: Mapping[str, Any], env_config: Mapping[str, Any]):
    from metadrive.policy.diffusion_policy.transfuser_config import (
        build_transfuser_config,
        diffusion_model_config_to_overrides,
        load_diffusion_model_config,
    )
    from models.platoon.platoon_diffusion_planner import PlatoonDiffusionPlanner
    from models.platoon.weight_migration import (
        is_platoon_grpo_checkpoint,
        load_platoon_grpo_checkpoint,
        migrate_single_to_platoon,
    )

    model_cfg = load_diffusion_model_config(str(config.get("model_config_path", "configs/diffusion/model.yaml")))
    tf_config = build_transfuser_config(
        str(model_cfg.get("model_size", "small")),
        **diffusion_model_config_to_overrides(model_cfg),
    )
    planner = PlatoonDiffusionPlanner(
        tf_config,
        num_vehicles=int(config.get("num_agents", 3)),
        use_relation_encoder=bool(config.get("use_relation_encoder", True)),
    )
    device_name = str(env_config.get("planner_device", "cpu"))
    device = torch.device(device_name if torch.cuda.is_available() or "cuda" not in device_name else "cpu")
    ckpt_path = str(config.get("pretrained_ckpt", "") or "")
    if ckpt_path:
        if is_platoon_grpo_checkpoint(ckpt_path):
            planner = load_platoon_grpo_checkpoint(ckpt_path, planner)
            print(f"[refine-grpo] loaded full platoon checkpoint: {ckpt_path}", flush=True)
        else:
            planner = migrate_single_to_platoon(ckpt_path, planner)
            print(f"[refine-grpo] migrated single diffusion checkpoint: {ckpt_path}", flush=True)
    planner = planner.to(device)
    load_cls_grpo_weights_for_refinement(planner, config, device)
    return planner.eval()


def freeze_for_selected_refinement(planner) -> list[str]:
    """Freeze all params except the trajectory head (GRU) and plan_cls_branch."""
    for param in planner.parameters():
        param.requires_grad_(False)

    decoder = _last_task_decoder(planner)

    # trajectory head
    trainable_modules = []
    has_gru_parts = all(hasattr(decoder, n) for n in ("hidden_init", "reg_gru_cell", "delta_head"))
    if getattr(decoder, "trajectory_reg_decoder_type", "gru" if has_gru_parts else "mlp") == "gru":
        for name in ("hidden_init", "reg_gru_cell", "delta_head"):
            module = getattr(decoder, name, None)
            if module is not None:
                trainable_modules.append((name, module))
    else:
        module = getattr(decoder, "plan_reg_branch", None)
        if module is not None:
            trainable_modules.append(("plan_reg_branch", module))

    # classification head
    cls_branch = getattr(decoder, "plan_cls_branch", None)
    if cls_branch is not None:
        trainable_modules.append(("plan_cls_branch", cls_branch))

    trainable_names: list[str] = []
    for module_name, module in trainable_modules:
        for param_name, param in module.named_parameters():
            param.requires_grad_(True)
            trainable_names.append(f"{module_name}.{param_name}")
    return trainable_names


def _refine_state_dict(planner) -> dict[str, torch.Tensor]:
    decoder = _last_task_decoder(planner)
    names = ("hidden_init", "reg_gru_cell", "delta_head")
    has_gru_parts = all(hasattr(decoder, n) for n in ("hidden_init", "reg_gru_cell", "delta_head"))
    if getattr(decoder, "trajectory_reg_decoder_type", "gru" if has_gru_parts else "mlp") != "gru":
        names = ("plan_reg_branch",)
    state: dict[str, torch.Tensor] = {}
    for name in names:
        module = getattr(decoder, name, None)
        if module is None:
            continue
        for key, value in module.state_dict().items():
            state[f"{name}.{key}"] = value.detach().cpu()
    cls_branch = getattr(decoder, "plan_cls_branch", None)
    if cls_branch is not None:
        for key, value in cls_branch.state_dict().items():
            state[f"plan_cls_branch.{key}"] = value.detach().cpu()
    return state


def _debug_save_noise_plot(
    base_np: np.ndarray,
    raw_noise_np: np.ndarray,
    clamped_noise_np: np.ndarray,
    noisy_trajs_np: np.ndarray,
    save_path: Path,
) -> None:
    """Save a figure showing base trajectory, raw/clamped noise, and noisy trajectories.

    Parameters
    ----------
    base_np        : [T, 2] normalized xy of the selected (base) trajectory.
    raw_noise_np   : [G, T, 2] raw Gaussian noise before clamping.
    clamped_noise_np: [G, T, 2] noise after max_delta_norm clamping.
    noisy_trajs_np : [G, T, 2] final noisy trajectories (base + clamped_noise).
    save_path      : full file path to save the PNG to.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    G, _, _ = noisy_trajs_np.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── left: base trajectory + noisy variants ─────────────────────────────────
    ax = axes[0]
    ax.plot(base_np[:, 0], base_np[:, 1], "k-o", lw=2, ms=4, label="base")
    colors = plt.cm.tab10(np.linspace(0, 1, G))
    for g in range(G):
        ax.plot(noisy_trajs_np[g, :, 0], noisy_trajs_np[g, :, 1],
                color=colors[g], lw=1, alpha=0.7, label=f"group {g}")
    ax.set_title("Base + Noisy Trajectories (normalized space)")
    ax.set_xlabel("x_norm"); ax.set_ylabel("y_norm")
    ax.legend(fontsize=7); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    # ── middle: raw noise scatter (before clamping) ──────────────────────────────
    ax = axes[1]
    for g in range(G):
        ax.scatter(raw_noise_np[g, :, 0], raw_noise_np[g, :, 1],
                   color=colors[g], alpha=0.6, s=20, label=f"group {g}")
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_title(f"Raw Gaussian Noise (std={raw_noise_np.std():.4f})")
    ax.set_xlabel("noise_x"); ax.set_ylabel("noise_y")
    ax.legend(fontsize=7); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    # ── right: clamped noise scatter (after max_delta_norm) ─────────────────────
    ax = axes[2]
    for g in range(G):
        ax.scatter(clamped_noise_np[g, :, 0], clamped_noise_np[g, :, 1],
                   color=colors[g], alpha=0.6, s=20, label=f"group {g}")
    max_val = np.abs(clamped_noise_np).max()
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    ax.set_title(f"Clamped Noise (max |δ|={max_val:.4f})")
    ax.set_xlabel("noise_x"); ax.set_ylabel("noise_y")
    ax.legend(fontsize=7); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)



def normalize_group_advantages(
    rewards: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-4,
    positive_only: bool = False,
    baseline_reward: torch.Tensor | float | None = None,
) -> torch.Tensor:
    rewards = rewards.float()
    if valid_mask is None:
        valid_mask = torch.ones_like(rewards, dtype=torch.bool)
    valid = valid_mask.bool()
    out = torch.zeros_like(rewards)
    if int(valid.sum().item()) < 2:
        return out
    vals = rewards[valid]
    if baseline_reward is None:
        baseline = vals.mean()
    else:
        baseline = torch.as_tensor(baseline_reward, dtype=vals.dtype, device=vals.device)
        if not bool(torch.isfinite(baseline).item()):
            baseline = vals.mean()
    out[valid] = (vals - baseline) / (vals.std(unbiased=False) + float(eps))
    if positive_only:
        out = out.clamp(min=0.0)
    return out


def normalize_multimodal_advantages(
    rewards: torch.Tensor,      # [G*M]
    num_groups: int,
    num_modes: int,
    positive_only: bool = True,
    eps: float = 1e-4,
    baseline_reward: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Per-mode group-normalized advantage, optionally centered on a pretrained reward baseline."""
    rewards = rewards.float()
    r2d = rewards.view(num_groups, num_modes)   # [G, M]
    out = torch.zeros_like(r2d)
    for m in range(num_modes):
        vals = r2d[:, m]
        if int(vals.numel()) < 2:
            continue
        if baseline_reward is None:
            baseline = vals.mean()
        else:
            baseline = torch.as_tensor(baseline_reward, dtype=vals.dtype, device=vals.device)
            if not bool(torch.isfinite(baseline).item()):
                baseline = vals.mean()
        out[:, m] = (vals - baseline) / (vals.std(unbiased=False) + eps)
    if positive_only:
        out = out.clamp(min=0.0)
    return out.view(num_groups * num_modes)     # [G*M]





def select_best_mode_group(
    rewards: torch.Tensor,
) -> tuple[int, int, float]:
    """Return (group_idx, mode_idx, reward) for the highest-reward trajectory."""
    if rewards.ndim != 2:
        raise ValueError(f"rewards must have shape [G, M], got {tuple(rewards.shape)}")
    scores = rewards.detach().float()
    if scores.numel() < 1:
        raise ValueError("No trajectories available for mode selection.")
    flat_idx = int(torch.argmax(scores).item())
    num_modes = int(scores.shape[1])
    group_idx = flat_idx // num_modes
    mode_idx = flat_idx % num_modes
    return group_idx, mode_idx, float(scores[group_idx, mode_idx].item())


def slice_gt_mode_refinement(
    *,
    chains_norm: torch.Tensor,
    refined_xy: torch.Tensor,
    ref_mean_xy: torch.Tensor,
    rewards: torch.Tensor,
    num_groups: int,
    num_modes: int,
    gt_mode: int,
) -> dict[str, torch.Tensor]:
    """Slice all multimodal rollout tensors down to the G groups of one GT mode."""
    mode = int(gt_mode)
    if mode < 0 or mode >= int(num_modes):
        raise ValueError(f"gt_mode={mode} outside [0, {num_modes})")
    return {
        "chains_norm": chains_norm.view(num_groups, num_modes, *chains_norm.shape[1:])[:, mode],
        "refined_xy": refined_xy.view(num_groups, num_modes, *refined_xy.shape[1:])[:, mode],
        "ref_mean_xy": ref_mean_xy.view(num_groups, num_modes, *ref_mean_xy.shape[1:])[:, mode],
        "rewards": rewards.view(num_groups, num_modes)[:, mode],
    }


def focal_gt_mode_loss(
    logits: torch.Tensor,
    gt_modes: torch.Tensor | int,
    valid_mask: torch.Tensor | None = None,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> torch.Tensor:
    """Sigmoid focal loss with one-hot GT mode supervision."""
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    gt = torch.as_tensor(gt_modes, dtype=torch.long, device=logits.device).view(-1)
    if gt.numel() == 1 and logits.shape[0] > 1:
        gt = gt.expand(logits.shape[0])
    target = torch.zeros_like(logits)
    target.scatter_(1, gt.unsqueeze(1), 1.0)
    loss = py_sigmoid_focal_loss(
        logits,
        target,
        gamma=float(gamma),
        alpha=float(alpha),
        reduction="none",
        avg_factor=None,
    )
    if valid_mask is not None:
        mask = valid_mask.to(device=logits.device, dtype=logits.dtype)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand_as(logits)
        loss = loss * mask
        denom = mask.sum().clamp_min(1.0)
        return loss.sum() / denom
    return loss.mean()


def _as_numpy_2d(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def format_all_mode_debug_matrix(
    *,
    global_step: int,
    episode: int,
    env_step: int,
    agent_id: str,
    gt_mode: int,
    gt_group: int,
    gt_reward: float,
    on_training_mode: int,
    rewards: torch.Tensor | np.ndarray,
    adv_raw: torch.Tensor | np.ndarray,
    adv_clipped: torch.Tensor | np.ndarray,
    mode_names: list[str] | None = None,
) -> str:
    """Format all-mode refinement reward/advantage matrices for terminal debug."""
    reward_np = _as_numpy_2d(rewards).astype(float)
    raw_np = _as_numpy_2d(adv_raw).astype(float)
    clipped_np = _as_numpy_2d(adv_clipped).astype(float)
    fmt = {"precision": 3, "suppress_small": True, "max_line_width": 160}
    names_line = ""
    if mode_names is not None:
        names_line = "mode_names[M]=\n" + ", ".join(
            f"{idx}:{name}" for idx, name in enumerate(mode_names)
        ) + "\n"
    return (
        "[refine-grpo][all-mode-debug] "
        f"global_step={global_step} episode={episode} env_step={env_step} agent={agent_id} "
        f"on_training_mode={on_training_mode} gt_mode={gt_mode} gt_group={gt_group} gt_reward={gt_reward:.4f}\n"
        f"{names_line}"
        f"reward[G,M]=\n{np.array2string(reward_np, **fmt)}\n"
        f"adv_raw[G,M]=\n{np.array2string(raw_np, **fmt)}\n"
        f"adv_clipped[G,M]=\n{np.array2string(clipped_np, **fmt)}"
    )


def mode_names_from_planner(planner, num_modes: int | None = None) -> list[str]:
    """Return semantic mode names aligned with the planner's runtime mode slots."""
    try:
        from metadrive.policy.diffusion_policy.mode_definitions import build_mode_slots

        cfg = getattr(planner, "config", None)
        if cfg is None:
            cfg = getattr(getattr(planner, "model", None), "config", None)
        slots = build_mode_slots(
            keep_lane_count=getattr(cfg, "mode_keep_lane_count"),
            lane_change_left_count=getattr(cfg, "mode_lane_change_left_count"),
            lane_change_right_count=getattr(cfg, "mode_lane_change_right_count"),
            emergency_stop_count=getattr(cfg, "mode_emergency_stop_count"),
        )
        names = [slot.name for slot in slots]
    except Exception:
        count = int(num_modes or 0)
        names = [f"MODE_{idx}" for idx in range(count)]
    if num_modes is not None and len(names) != int(num_modes):
        return [f"MODE_{idx}" for idx in range(int(num_modes))]
    return names


def _local_xy_to_world_xy(pose: np.ndarray, local_xy: np.ndarray) -> np.ndarray:
    pose_arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    pts = np.asarray(local_xy, dtype=np.float64)
    cos_h = float(np.cos(pose_arr[2]))
    sin_h = float(np.sin(pose_arr[2]))
    world = np.empty_like(pts, dtype=np.float64)
    world[..., 0] = pose_arr[0] + cos_h * pts[..., 0] - sin_h * pts[..., 1]
    world[..., 1] = pose_arr[1] + sin_h * pts[..., 0] + cos_h * pts[..., 1]
    return world


def vehicle_box_corners(pose: np.ndarray | list[float], length: float = 4.8, width: float = 2.0) -> np.ndarray:
    """Return four world-frame corners of a vehicle rectangle."""
    pose_arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    half_l = float(length) * 0.5
    half_w = float(width) * 0.5
    local = np.asarray(
        [[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]],
        dtype=np.float64,
    )
    return _local_xy_to_world_xy(pose_arr, local)


def _trajectory_groups_np(value: torch.Tensor | np.ndarray) -> np.ndarray:
    traj = np.asarray(_as_numpy_2d(value), dtype=np.float64)
    if traj.ndim == 2:
        traj = traj[None, ...]
    if traj.ndim != 3 or traj.shape[-1] < 2:
        raise ValueError(f"trajectory groups must have shape [G,T,2+] or [T,2+], got {traj.shape}")
    return traj


def build_platoon_gt_mode_group_plot_data(
    agent_ids: list[str],
    poses: Mapping[str, np.ndarray],
    trajectories_by_agent: Mapping[str, torch.Tensor | np.ndarray],
) -> dict[str, dict[str, object]]:
    """Build world-frame trajectories and vehicle boxes for all plotted agents."""
    world_trajs: dict[str, list[np.ndarray]] = {}
    boxes: dict[str, np.ndarray] = {}
    pose_np: dict[str, np.ndarray] = {}
    for agent_id in agent_ids:
        if agent_id not in poses or poses[agent_id] is None or agent_id not in trajectories_by_agent:
            continue
        pose = np.asarray(poses[agent_id], dtype=np.float64).reshape(-1)
        traj_groups = _trajectory_groups_np(trajectories_by_agent[agent_id])
        pose_np[agent_id] = pose
        boxes[agent_id] = vehicle_box_corners(pose)
        world_trajs[agent_id] = [
            _local_xy_to_world_xy(pose, traj[:, :2])
            for traj in traj_groups
        ]
    return {"poses": pose_np, "boxes": boxes, "trajectories": world_trajs}


def agent_debug_plot_color(agent_index: int) -> str:
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    return colors[int(agent_index) % len(colors)]


def agent_best_debug_plot_color(agent_index: int) -> str:
    colors = ["mediumblue", "darkorange", "forestgreen", "firebrick", "indigo"]
    return colors[int(agent_index) % len(colors)]


def refined_background_alpha() -> float:
    return 0.12


def save_platoon_gt_mode_group_trajectory_plot(
    *,
    output_path: Path,
    agent_ids: list[str],
    poses: Mapping[str, np.ndarray],
    trajectories_by_agent: Mapping[str, torch.Tensor | np.ndarray],
    gt_modes_by_agent: Mapping[str, int],
    gt_groups_by_agent: Mapping[str, int],
    rewards_by_agent: Mapping[str, torch.Tensor | np.ndarray] | None = None,
    on_training_traj_by_agent: Mapping[str, np.ndarray] | None = None,
    on_training_reward_by_agent: Mapping[str, float] | None = None,
    on_training_mode_by_agent: Mapping[str, int] | None = None,
    coarse_traj_by_agent: Mapping[str, torch.Tensor] | None = None,
    all_refined_by_agent: Mapping[str, dict] | None = None,
    noisy_init_by_agent: Mapping[str, np.ndarray] | None = None,
    global_step: int,
    episode: int,
    env_step: int,
) -> None:
    """Save a world-frame diagnostic plot for GT-mode groups of the whole platoon."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    plot_data = build_platoon_gt_mode_group_plot_data(agent_ids, poses, trajectories_by_agent)
    world_trajs = plot_data["trajectories"]
    boxes = plot_data["boxes"]
    pose_np = plot_data["poses"]
    rewards_by_agent = rewards_by_agent or {}
    on_training_traj_by_agent = on_training_traj_by_agent or {}
    on_training_reward_by_agent = on_training_reward_by_agent or {}
    on_training_mode_by_agent = on_training_mode_by_agent or {}
    coarse_traj_by_agent = coarse_traj_by_agent or {}
    all_refined_by_agent = all_refined_by_agent or {}
    noisy_init_by_agent = noisy_init_by_agent or {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    on_training_colors = ["navy", "darkred", "darkgreen", "firebrick", "indigo"]
    coarse_colors  = ["deepskyblue", "tomato", "limegreen", "salmon", "mediumpurple"]
    all_points: list[np.ndarray] = []

    # ── All-mode refined trajectories (thin lines, colored by agent, no legend) ──
    for agent_pos, agent_id in enumerate(agent_ids):
        if agent_id not in all_refined_by_agent or poses.get(agent_id) is None:
            continue
        color = agent_debug_plot_color(agent_pos)
        _d = all_refined_by_agent[agent_id]
        _traj_all = np.asarray(_d["traj"], dtype=np.float64)   # [G*M, T, 3]
        _G = int(_d["num_groups"])
        _M = int(_d["num_modes"])
        _pose = np.asarray(poses[agent_id], dtype=np.float64).reshape(-1)
        for m in range(_M):
            for g in range(_G):
                _gm = g * _M + m
                if _gm >= _traj_all.shape[0]:
                    continue
                _w = _local_xy_to_world_xy(_pose, _traj_all[_gm, :, :2])
                all_points.append(_w)
                ax.plot(_w[:, 0], _w[:, 1],
                        color=color, linewidth=0.7, alpha=refined_background_alpha(),
                        label="_nolegend_", zorder=1)

    for agent_pos, agent_id in enumerate(agent_ids):
        if agent_id not in world_trajs:
            continue
        color = agent_debug_plot_color(agent_pos)
        best_color = agent_best_debug_plot_color(agent_pos)
        on_training_color = on_training_colors[agent_pos % len(on_training_colors)]
        coarse_color   = coarse_colors[agent_pos % len(coarse_colors)]
        rewards_np = (
            np.asarray(_as_numpy_2d(rewards_by_agent[agent_id]), dtype=float).reshape(-1)
            if agent_id in rewards_by_agent else np.asarray([], dtype=float)
        )
        gt_group = int(gt_groups_by_agent.get(agent_id, -1))
        gt_mode = int(gt_modes_by_agent.get(agent_id, -1))
        for group_idx, world in enumerate(world_trajs[agent_id]):
            all_points.append(world)
            is_gt = group_idx == gt_group
            if is_gt:
                reward_part = f" r={rewards_np[group_idx]:.3f}" if group_idx < rewards_np.size else ""
                _legend_label = f"{agent_id} best m={gt_mode} g={gt_group}{reward_part}"
            else:
                _legend_label = "_nolegend_"
            ax.plot(
                world[:, 0],
                world[:, 1],
                color=best_color if is_gt else color,
                linewidth=4.0 if is_gt else 1.1,
                alpha=1.0 if is_gt else 0.28,
                marker="o" if is_gt else None,
                markersize=4,
                label=_legend_label,
                zorder=20 if is_gt else 2,
            )
        # ── on-training trajectory overlay ───────────────────────────────────
        if agent_id in on_training_traj_by_agent and poses.get(agent_id) is not None:
            _pt = np.asarray(on_training_traj_by_agent[agent_id], dtype=np.float64)
            if _pt.ndim == 1:
                _pt = _pt[None, :]
            _pose = np.asarray(poses[agent_id], dtype=np.float64).reshape(-1)
            _pt_world = _local_xy_to_world_xy(_pose, _pt[:, :2])
            all_points.append(_pt_world)
            _pr = on_training_reward_by_agent.get(agent_id)
            _pm = on_training_mode_by_agent.get(agent_id, -1)
            _reward_str = f" r={_pr:.3f}" if _pr is not None else ""
            ax.plot(
                _pt_world[:, 0],
                _pt_world[:, 1],
                color=on_training_color,
                linewidth=3.5,
                linestyle="--",
                marker="*",
                markersize=8,
                alpha=0.95,
                zorder=10,
                label=f"{agent_id} on_training m={_pm}{_reward_str}",
            )
        # ── GT mode coarse anchor (dynamic anchor that seeds this group's noise) ──
        if agent_id in coarse_traj_by_agent and poses.get(agent_id) is not None:
            _ct = coarse_traj_by_agent[agent_id]
            if torch.is_tensor(_ct):
                _ct = _ct.detach().cpu().numpy()
            _ct = np.asarray(_ct, dtype=np.float64)  # [M, 8, 2]
            if _ct.ndim == 3 and gt_mode >= 0 and gt_mode < _ct.shape[0]:
                _pose = np.asarray(poses[agent_id], dtype=np.float64).reshape(-1)
                _coarse_world = _local_xy_to_world_xy(_pose, _ct[gt_mode, :, :2])
                all_points.append(_coarse_world)
                ax.plot(
                    _coarse_world[:, 0],
                    _coarse_world[:, 1],
                    color=coarse_color,
                    linewidth=4.0,
                    linestyle="-.",
                    marker="D",
                    markersize=5,
                    alpha=0.85,
                    zorder=15,
                    label=f"{agent_id} coarse m={gt_mode}",
                )
        # ── Initial noisy trajectories (ALL G*M groups) as scatter, colored by mode ──
        if agent_id in noisy_init_by_agent and poses.get(agent_id) is not None:
            _ni_d = noisy_init_by_agent[agent_id]
            if isinstance(_ni_d, dict):
                _ni = np.asarray(_ni_d["traj"], dtype=np.float64)   # [G*M, T, 2]
                _ni_G = int(_ni_d["num_groups"])
                _ni_M = int(_ni_d["num_modes"])
            else:
                _ni = np.asarray(_ni_d, dtype=np.float64)
                _ni_G, _ni_M = _ni.shape[0], 1
            _pose = np.asarray(poses[agent_id], dtype=np.float64).reshape(-1)
            if _ni.ndim == 3 and _ni.shape[2] >= 2:
                import matplotlib.cm as _cm_ni
                _ni_cmap = _cm_ni.get_cmap("tab20", _ni_M)
                for _m in range(_ni_M):
                    _mc = _ni_cmap(_m)
                    _pts_list = []
                    for _g in range(_ni_G):
                        _gm = _g * _ni_M + _m
                        if _gm >= _ni.shape[0]:
                            continue
                        _pts_list.append(_local_xy_to_world_xy(_pose, _ni[_gm, :, :2]))
                    if _pts_list:
                        _all_pts = np.concatenate(_pts_list, axis=0)
                        all_points.append(_all_pts)
                        ax.scatter(
                            _all_pts[:, 0], _all_pts[:, 1],
                            color=_mc, s=3, alpha=0.3, marker=".", zorder=2,
                            label="_nolegend_",
                        )
        box = boxes[agent_id]
        pose = pose_np[agent_id]
        all_points.append(box)
        ax.add_patch(Polygon(box, closed=True, fill=False, edgecolor=color, linewidth=2.2, label="_nolegend_"))
        ax.scatter([pose[0]], [pose[1]], color=color, s=35, zorder=5)
        ax.text(pose[0], pose[1], f" {agent_id}", color=color, weight="bold")

    if all_points:
        pts = np.concatenate(all_points, axis=0)
        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)
        pad = max(5.0, float(np.max(max_xy - min_xy)) * 0.15)
        ax.set_xlim(float(min_xy[0] - pad), float(max_xy[0] + pad))
        ax.set_ylim(float(min_xy[1] - pad), float(max_xy[1] + pad))
    ax.set_title(f"Platoon GT mode group trajectories | episode={episode} step={env_step} global={global_step}")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def save_gt_mode_group_trajectory_plot(
    *,
    output_path: Path,
    agent_id: str,
    prev_agent_id: str,
    follower_pose: np.ndarray,
    leader_pose: np.ndarray,
    refined_xy: torch.Tensor | np.ndarray,
    rewards: torch.Tensor | np.ndarray,
    gt_mode: int,
    gt_group: int,
    global_step: int,
    episode: int,
    env_step: int,
) -> None:
    """Save a lightweight world-frame diagnostic plot for the GT mode's G groups."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    traj_np = _as_numpy_2d(refined_xy).astype(float)
    rewards_np = np.asarray(_as_numpy_2d(rewards), dtype=float).reshape(-1)
    follower_pose_np = np.asarray(follower_pose, dtype=np.float64).reshape(-1)
    leader_pose_np = np.asarray(leader_pose, dtype=np.float64).reshape(-1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    all_points = []
    for group_idx, traj in enumerate(traj_np):
        world = _local_xy_to_world_xy(follower_pose_np, traj[:, :2])
        all_points.append(world)
        is_gt = int(group_idx) == int(gt_group)
        ax.plot(
            world[:, 0],
            world[:, 1],
            linewidth=3.0 if is_gt else 1.4,
            alpha=1.0 if is_gt else 0.55,
            marker="o" if is_gt else None,
            markersize=3,
            label=f"group {group_idx} reward={rewards_np[group_idx]:.3f}" if group_idx < rewards_np.size else f"group {group_idx}",
        )

    follower_box = vehicle_box_corners(follower_pose_np)
    leader_box = vehicle_box_corners(leader_pose_np)
    ax.add_patch(Polygon(follower_box, closed=True, fill=False, edgecolor="tab:orange", linewidth=2.2, label=f"{agent_id} box"))
    ax.add_patch(Polygon(leader_box, closed=True, fill=False, edgecolor="tab:blue", linewidth=2.2, label=f"{prev_agent_id} box"))
    ax.scatter([follower_pose_np[0]], [follower_pose_np[1]], color="tab:orange", s=35)
    ax.scatter([leader_pose_np[0]], [leader_pose_np[1]], color="tab:blue", s=35)
    ax.text(follower_pose_np[0], follower_pose_np[1], f" {agent_id}", color="tab:orange", weight="bold")
    ax.text(leader_pose_np[0], leader_pose_np[1], f" {prev_agent_id}", color="tab:blue", weight="bold")

    if all_points:
        pts = np.concatenate([*all_points, follower_box, leader_box], axis=0)
        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)
        pad = max(5.0, float(np.max(max_xy - min_xy)) * 0.15)
        ax.set_xlim(float(min_xy[0] - pad), float(max_xy[0] + pad))
        ax.set_ylim(float(min_xy[1] - pad), float(max_xy[1] + pad))
    ax.set_title(
        f"GT mode group trajectories | episode={episode} step={env_step} global={global_step}\n"
        f"agent={agent_id} gt_mode={gt_mode} gt_group={gt_group}"
    )
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def reconstruct_heading_from_xy(xy: torch.Tensor) -> torch.Tensor:
    """Append heading from xy differences, preserving leading dimensions."""
    dx = xy[..., 1:, 0] - xy[..., :-1, 0]
    dy = xy[..., 1:, 1] - xy[..., :-1, 1]
    headings_tail = torch.atan2(dy, dx)
    first = headings_tail[..., :1]
    heading = torch.cat([first, headings_tail], dim=-1)
    return torch.cat([xy, heading.unsqueeze(-1)], dim=-1)


def ddv2_refine_grpo_loss(
    new_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    poses_reg_steps: torch.Tensor,
    selected_xy: torch.Tensor,
    bc_weight: float = 0.1,
    kl_weight: float = 0.02,
    ref_mean_xy: torch.Tensor | None = None,
    step_discount: torch.Tensor | None = None,
    no_positive_bc_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """GRPO loss for G*M refinement trajectories with per-step denoising discount.

    advantages:      [G*M]              — safety-masked (clamp≥0 applied, unsafe→-1)
    new_log_probs:   [G*M, steps]
    poses_reg_steps: [G*M, steps, T, 2] — physical x_0 predictions at each denoising step
                                          (WITH computation graph; used for BC/KL loss)
    step_discount:   [steps]            — discount[i]=γ^(steps-1-i); None→uniform 1
    """
    adv = advantages.to(device=new_log_probs.device, dtype=new_log_probs.dtype).detach()
    # Expand advantage to each denoising step, then apply discount
    adv_per_step = adv.unsqueeze(-1).expand_as(new_log_probs).clone()      # [G*M, steps]
    if step_discount is not None:
        disc = step_discount.to(device=adv.device, dtype=adv.dtype)        # [steps]
        adv_per_step = adv_per_step * disc
    # Ratio trick: exp(logp - stop_grad(logp)) = 1 in value, gradient = ∂logp/∂θ
    ratio = torch.exp(new_log_probs - new_log_probs.detach())              # [G*M, steps]
    per_token_loss = -(ratio * adv_per_step)                               # [G*M, steps]
    # Per-step non-zero mean over trajectories → [steps], then mean over steps
    mask_nz = (per_token_loss != 0)                                        # [G*M, steps]
    nz_count = mask_nz.sum(dim=0).clamp(min=1)                            # [steps]
    per_step_loss = (per_token_loss * mask_nz).sum(dim=0) / nz_count      # [steps]
    pg_loss = per_step_loss.mean()
    # BC/KL loss on per-step x_0 predictions — gradients flow through poses_reg_steps
    # ref_mean_xy: [G*M, T, 2] → expand to [G*M, steps, T, 2] to match poses_reg_steps
    if ref_mean_xy is not None:
        _ref = ref_mean_xy.to(poses_reg_steps.device)                      # [G*M, T, 2]
        _ref_exp = _ref.unsqueeze(1).expand_as(poses_reg_steps)            # [G*M, steps, T, 2]
        bc_loss = F.l1_loss(poses_reg_steps, _ref_exp)
        kl_loss = F.mse_loss(poses_reg_steps, _ref_exp)
    else:
        bc_loss = poses_reg_steps.new_tensor(0.0)
        kl_loss = poses_reg_steps.new_tensor(0.0)
    active_advantage_count = int((adv > 0).sum().detach().item())
    has_positive = active_advantage_count > 0
    effective_bc_weight  = float(bc_weight)  if has_positive else float(no_positive_bc_weight)
    effective_kl_weight  = float(kl_weight)  if has_positive else float(no_positive_bc_weight)
    loss = pg_loss + effective_bc_weight * bc_loss + effective_kl_weight * kl_loss
    return loss, {
        "pg_loss": float(pg_loss.detach()),
        "bc_loss": float(bc_loss.detach()),
        "kl_loss": float(kl_loss.detach()),
        "ratio_mean": float(ratio.detach().mean()),
        "clip_fraction": 0.0,
        "approx_kl": 0.0,
        "active_advantage_count": active_advantage_count,
        "effective_bc_weight": effective_bc_weight,
        "effective_kl_weight": effective_kl_weight,
        "total_loss": float(loss.detach()),
    }


def _repeat_context(context: dict[str, Any], groups: int) -> dict[str, Any]:
    repeated = {}
    for key, value in context.items():
        if torch.is_tensor(value):
            repeated[key] = value.repeat_interleave(groups, dim=0)
        else:
            repeated[key] = value
    return repeated


def _selected_norm_xy(planner, selected_traj_np: np.ndarray) -> torch.Tensor:
    th = planner.model._trajectory_head
    device = planner._device()
    selected = torch.as_tensor(selected_traj_np, dtype=torch.float32, device=device)
    norm = th.norm_odo(selected.view(1, 1, selected.shape[0], 3))[..., :2]
    return norm.view(1, selected.shape[0], 2)


def _ddv2_roll_timesteps(denoise_steps: int, device: torch.device) -> torch.Tensor:
    step_num = max(1, int(denoise_steps))
    step_ratio = 20.0 / float(step_num)
    values = (np.arange(0, step_num) * step_ratio).round()[::-1].copy().astype(np.int64)
    return torch.from_numpy(values).to(device=device, dtype=torch.long)


def _add_ddv2_truncated_noise(
    scheduler: DDIMSchedulerWithLogProb,
    selected_xy_norm: torch.Tensor,
    *,
    num_groups: int,
    noise_t: int,
    debug_save_path: Path | None = None,
) -> torch.Tensor:
    base = selected_xy_norm.unsqueeze(0).expand(int(num_groups), -1, -1, -1).contiguous()
    noise = torch.randn_like(base)
    timesteps = torch.full((int(num_groups),), int(noise_t), dtype=torch.long, device=base.device)
    result = scheduler.add_noise(original_samples=base, noise=noise, timesteps=timesteps).clamp(-1.0, 1.0)
    if debug_save_path is not None:
        # base: [G,1,T,2] → squeeze dim-1 → [G,T,2]; selected_xy_norm: [1,T,2] → squeeze dim-0 → [T,2]
        _debug_save_noise_plot(
            base_np=selected_xy_norm.squeeze(0).detach().cpu().numpy(),
            raw_noise_np=noise.squeeze(1).detach().cpu().numpy(),
            clamped_noise_np=(result - base).squeeze(1).detach().cpu().numpy(),
            noisy_trajs_np=result.squeeze(1).detach().cpu().numpy(),
            save_path=debug_save_path,
        )
    return result


def _denorm_xy(planner, norm_xy: torch.Tensor) -> torch.Tensor:
    th = planner.model._trajectory_head
    zeros = torch.zeros((*norm_xy.shape[:-1], 1), dtype=norm_xy.dtype, device=norm_xy.device)
    norm_3 = torch.cat([norm_xy, zeros], dim=-1)
    return th.denorm_odo(norm_3)[..., :2]


@dataclass(frozen=True)
class CBFBarrierGuidanceConfig:
    enabled: bool = False
    guidance_scale: float = 0.05
    sigma_multiplier: float = 3.0
    delta_x_m: float = 0.0
    delta_y_m: float = 0.0
    eps: float = 1.0e-4
    outside_weight: float = 1.0
    grad_clip_norm: float = 1.0


def barrier_noise_bounds_m(
    scheduler: DDIMSchedulerWithLogProb,
    *,
    noise_t: int,
    sigma_multiplier: float,
) -> tuple[float, float]:
    """Return physical x/y ellipse radii implied by initial diffusion noise."""
    timestep = int(noise_t)
    alpha_bar = float(scheduler.alphas_cumprod[timestep].detach().cpu().item())
    std_norm = float(max(0.0, 1.0 - alpha_bar) ** 0.5)
    return (
        float(sigma_multiplier) * std_norm * (56.9 / 2.0),
        float(sigma_multiplier) * std_norm * (46.0 / 2.0),
    )


def _resolve_barrier_guidance_config(
    config: Mapping[str, Any],
    scheduler: DDIMSchedulerWithLogProb,
    noise_t: int,
) -> CBFBarrierGuidanceConfig:
    sigma_multiplier = float(config.get("refine_barrier_sigma_multiplier", 3.0))
    auto_dx, auto_dy = barrier_noise_bounds_m(
        scheduler, noise_t=int(noise_t), sigma_multiplier=sigma_multiplier
    )
    delta_x = float(config.get("refine_barrier_delta_x_m", 0.0))
    delta_y = float(config.get("refine_barrier_delta_y_m", 0.0))
    return CBFBarrierGuidanceConfig(
        enabled=bool(config.get("refine_barrier_guidance_enabled", False)),
        guidance_scale=float(config.get("refine_barrier_guidance_scale", 0.05)),
        sigma_multiplier=sigma_multiplier,
        delta_x_m=delta_x if delta_x > 0.0 else auto_dx,
        delta_y_m=delta_y if delta_y > 0.0 else auto_dy,
        eps=float(config.get("refine_barrier_eps", 1.0e-4)),
        outside_weight=float(config.get("refine_barrier_outside_weight", 1.0)),
        grad_clip_norm=float(config.get("refine_barrier_grad_clip_norm", 1.0)),
    )


def apply_cbf_barrier_guidance(
    planner,
    *,
    prev_sample_mean_norm: torch.Tensor,
    anchor_xy: torch.Tensor,
    config: CBFBarrierGuidanceConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Guide DDIM prev_sample_mean with an ego-local physical-space CBF ellipse."""
    zero_metrics = {
        "h_min": 0.0,
        "h_mean": 0.0,
        "barrier_cost": 0.0,
        "guidance_grad_norm": 0.0,
    }
    if not config.enabled or config.guidance_scale <= 0.0:
        return prev_sample_mean_norm, zero_metrics

    dx = max(float(config.delta_x_m), 1.0e-6)
    dy = max(float(config.delta_y_m), 1.0e-6)
    eps = max(float(config.eps), 1.0e-12)
    anchor = anchor_xy.to(device=prev_sample_mean_norm.device, dtype=prev_sample_mean_norm.dtype)

    with torch.enable_grad():
        if prev_sample_mean_norm.requires_grad:
            work_mean = prev_sample_mean_norm
        else:
            work_mean = prev_sample_mean_norm.detach().clone().requires_grad_(True)
        prev_xy = _denorm_xy(planner, work_mean.squeeze(1))
        delta = prev_xy - anchor
        h = 1.0 - (delta[..., 0] / dx).pow(2) - (delta[..., 1] / dy).pow(2)
        h_log = h.clamp(min=eps)
        inside_cost = -torch.log(h_log).mean()
        outside_cost = F.relu(eps - h).pow(2).mean()
        cost = inside_cost + float(config.outside_weight) * outside_cost
        grad = torch.autograd.grad(cost, work_mean, retain_graph=True, create_graph=False)[0]
        grad_norm = grad.detach().norm()
        max_norm = float(config.grad_clip_norm)
        if max_norm > 0.0:
            grad = grad * (max_norm / grad_norm.clamp(min=max_norm))
        guided = work_mean - float(config.guidance_scale) * grad.detach()

    metrics = {
        "h_min": float(h.detach().min().cpu().item()),
        "h_mean": float(h.detach().mean().cpu().item()),
        "barrier_cost": float(cost.detach().cpu().item()),
        "guidance_grad_norm": float(grad_norm.cpu().item()),
    }
    return guided.to(dtype=prev_sample_mean_norm.dtype), metrics


def _make_cbf_mean_guidance_fn(
    planner,
    anchor_xy: torch.Tensor,
    config: CBFBarrierGuidanceConfig | None,
    metrics_sink: list[dict[str, float]] | None = None,
) -> Callable[[torch.Tensor, int | torch.Tensor], torch.Tensor] | None:
    if config is None or not config.enabled:
        return None

    def _guide(prev_sample_mean: torch.Tensor, timestep: int | torch.Tensor) -> torch.Tensor:
        del timestep
        guided, metrics = apply_cbf_barrier_guidance(
            planner,
            prev_sample_mean_norm=prev_sample_mean,
            anchor_xy=anchor_xy,
            config=config,
        )
        if metrics_sink is not None:
            metrics_sink.append(metrics)
        return guided

    return _guide


def _summarize_barrier_metrics(metrics: list[dict[str, float]] | None) -> dict[str, float]:
    if not metrics:
        return {
            "barrier_h_min": 0.0,
            "barrier_h_mean": 0.0,
            "barrier_cost_mean": 0.0,
            "barrier_grad_norm_mean": 0.0,
        }
    return {
        "barrier_h_min": min(float(m.get("h_min", 0.0)) for m in metrics),
        "barrier_h_mean": float(np.mean([float(m.get("h_mean", 0.0)) for m in metrics])),
        "barrier_cost_mean": float(np.mean([float(m.get("barrier_cost", 0.0)) for m in metrics])),
        "barrier_grad_norm_mean": float(np.mean([float(m.get("guidance_grad_norm", 0.0)) for m in metrics])),
    }


def rollout_selected_refinement(
    planner,
    context: dict[str, Any],
    selected_traj_np: np.ndarray,
    *,
    num_groups: int,
    noise_t: int,
    noise_std: float,
    max_delta_norm: float,
    denoise_steps: int,
    eta: float,
    scheduler: DDIMSchedulerWithLogProb | None = None,
    debug_save_path: Path | None = None,
    barrier_config: CBFBarrierGuidanceConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Generate local refined groups around one selected trajectory."""
    device = planner._device()
    if scheduler is None:
        scheduler = DDIMSchedulerWithLogProb(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )
    if scheduler.num_inference_steps is None:
        scheduler.set_timesteps(1000, device=device)
    selected_norm = _selected_norm_xy(planner, selected_traj_np)  # [1,T,2]
    selected_xy_phys = torch.as_tensor(selected_traj_np[:, :2], dtype=torch.float32, device=device)
    anchor_xy = selected_xy_phys.unsqueeze(0).expand(int(num_groups), -1, -1)
    del noise_std, max_delta_norm
    sample = _add_ddv2_truncated_noise(
        scheduler,
        selected_norm,
        num_groups=num_groups,
        noise_t=noise_t,
        debug_save_path=debug_save_path,
    ).to(device)
    repeated_context = _repeat_context(context, int(num_groups))
    log_probs = []
    chains = [sample.detach()]
    timesteps = []
    last_prev_sample_mean = None
    barrier_metrics: list[dict[str, float]] = []
    mean_guidance_fn = _make_cbf_mean_guidance_fn(
        planner, anchor_xy, barrier_config, barrier_metrics
    )
    roll_timesteps = _ddv2_roll_timesteps(denoise_steps, device)
    for timestep in roll_timesteps:
        timesteps.append(timestep)
        batch_timestep = timestep.expand(int(num_groups))
        model_output = planner.predict_denoised_traj(sample, batch_timestep, repeated_context)  # [G,1,T,2]
        next_sample, log_prob, prev_sample_mean = scheduler.step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            eta=float(eta),
            mean_guidance_fn=mean_guidance_fn,
        )
        last_prev_sample_mean = prev_sample_mean
        log_probs.append(log_prob.squeeze(-1))
        sample = next_sample.detach()
        chains.append(sample)
    sampled_xy = _denorm_xy(planner, sample.squeeze(1)).detach()  # [G,T,2], used for execution/reward
    refined_xy_for_loss = _denorm_xy(
        planner,
        last_prev_sample_mean.squeeze(1) if last_prev_sample_mean is not None else sample.squeeze(1),
    )
    ref_mean_xy = selected_xy_phys.unsqueeze(0).expand_as(refined_xy_for_loss)
    refined_traj = reconstruct_heading_from_xy(sampled_xy)
    log_probs_t = torch.stack(log_probs, dim=-1)
    return {
        "all_diffusion_output": torch.stack(chains, dim=1).detach(),
        "refined_xy": refined_xy_for_loss,
        "sampled_xy": sampled_xy,
        "refined_traj": refined_traj,
        "log_probs": log_probs_t,
        "chains_norm": torch.stack(chains, dim=1).detach(),
        "timesteps": torch.stack(timesteps).detach(),
        "ref_mean_xy": ref_mean_xy,
        "selected_xy": selected_xy_phys,
        "barrier_metrics": barrier_metrics,
    }


def rollout_multimodal_refinement(
    planner,
    context: dict[str, Any],
    *,
    num_groups: int,
    noise_t: int,
    denoise_steps: int,
    eta: float,
    scheduler: DDIMSchedulerWithLogProb | None = None,
    chunk_size: int = 80,
    coarse_trajectories: torch.Tensor | None = None,
    barrier_config: CBFBarrierGuidanceConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Multi-modal rollout: G groups × M anchors = G*M candidate trajectories.

    Aligned with DiffusionDriveV2 forward_train_rl:
      - all M anchors used as base (dynamic coarse_trajectories when provided, else frozen plan_anchor)
      - DDIM truncated additive noise at t=noise_t
      - multiplicative noise applied in scheduler.step (DDIMSchedulerWithLogProb, eta>0)

    coarse_trajectories: per-agent dynamic anchors [M, 8, 2] physical ego-local;
                         falls back to frozen plan_anchor when None.
    chunk_size: number of trajectories processed per forward pass to limit peak GPU memory.
                Reduce if OOM; increase for speed.
    """
    device = planner._device()
    th = planner.model._trajectory_head

    if scheduler is None:
        scheduler = DDIMSchedulerWithLogProb(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )
    if scheduler.num_inference_steps is None:
        scheduler.set_timesteps(1000, device=device)

    anchor = (
        coarse_trajectories.to(device=device, dtype=torch.float32)
        if coarse_trajectories is not None
        else th.plan_anchor.to(device)
    )  # [M, 8, 2] physical
    M = anchor.shape[0]

    # Normalize via th.norm_odo (must match _denorm_xy which uses th.denorm_odo)
    anchor_h = torch.zeros(*anchor.shape[:-1], 1, dtype=anchor.dtype, device=device)
    anchor_norm = th.norm_odo(torch.cat([anchor, anchor_h], dim=-1))[..., :2]  # [M,8,2]

    # Expand G groups → [G*M, 1, 8, 2]
    base = (anchor_norm.unsqueeze(0)
            .expand(num_groups, -1, -1, -1)
            .reshape(num_groups * M, 8, 2)
            .unsqueeze(1))                                                   # [G*M,1,8,2]
    anchor_ref = (anchor.unsqueeze(0)
                  .expand(num_groups, -1, -1, -1)
                  .reshape(num_groups * M, 8, 2))                            # [G*M,T,2]

    # DDIM truncated additive noise at t=noise_t: √ᾱ_t·x₀ + √(1-ᾱ_t)·ε
    noise = torch.randn_like(base)
    if int(noise_t) == 0:
        # noise_t=0: use anchor directly without any noise injection
        sample_full = base.clone()
    else:
        ts_full = torch.full((num_groups * M,), int(noise_t), dtype=torch.long, device=device)
        sample_full = scheduler.add_noise(original_samples=base, noise=noise, timesteps=ts_full).clamp(-1.0, 1.0)

    roll_ts = _ddv2_roll_timesteps(denoise_steps, device)  # shared timestep sequence

    # ── chunked denoising: process chunk_size trajectories at a time ──────────
    # Rollout needs no gradients — gradients flow via recompute_refine_log_probs.
    chunk_log_probs: list[torch.Tensor] = []   # each [chunk, denoise_steps]
    chunk_chains:    list[torch.Tensor] = []   # each [chunk, denoise_steps+1, 1, T, 2]
    chunk_last_mean: list[torch.Tensor] = []   # each [chunk, 1, T, 2]
    timesteps_list:  list[torch.Tensor] = []   # collected once from first chunk
    barrier_metrics: list[dict[str, float]] = []

    GxM = num_groups * M
    with torch.no_grad():
        for c_start in range(0, GxM, chunk_size):
            c_end = min(c_start + chunk_size, GxM)
            sample = sample_full[c_start:c_end].clone()
            chunk_ctx = _repeat_context(context, c_end - c_start)
            chunk_anchor = anchor_ref[c_start:c_end]
            mean_guidance_fn = _make_cbf_mean_guidance_fn(
                planner, chunk_anchor, barrier_config, barrier_metrics
            )

            lp_steps:     list[torch.Tensor] = []
            chain_steps:  list[torch.Tensor] = [sample]
            last_mean_c:  torch.Tensor | None = None
            first_chunk = (c_start == 0)

            for timestep in roll_ts:
                if first_chunk:
                    timesteps_list.append(timestep)
                batch_ts = timestep.expand(c_end - c_start)
                model_output = planner.predict_denoised_traj(sample, batch_ts, chunk_ctx)
                next_sample, log_prob, prev_mean = scheduler.step(
                    model_output=model_output,
                    timestep=timestep,
                    sample=sample,
                    eta=float(eta),
                    mean_guidance_fn=mean_guidance_fn,
                )
                last_mean_c = prev_mean
                lp_steps.append(log_prob.squeeze(-1))   # [chunk]
                sample = next_sample.detach()
                chain_steps.append(sample)

            chunk_log_probs.append(torch.stack(lp_steps, dim=-1))           # [chunk, steps]
            chunk_chains.append(torch.stack(chain_steps, dim=1))             # [chunk, steps+1, 1, T, 2]
            chunk_last_mean.append(
                last_mean_c if last_mean_c is not None else sample.unsqueeze(1)
            )

    # Reassemble full tensors
    log_probs_t      = torch.cat(chunk_log_probs, dim=0)          # [G*M, denoise_steps]
    chains_norm_full = torch.cat(chunk_chains, dim=0)              # [G*M, steps+1, 1, T, 2]
    last_mean_full   = torch.cat(chunk_last_mean, dim=0)           # [G*M, 1, T, 2]
    sample_full      = chains_norm_full[:, -1]                     # [G*M, 1, T, 2] final sample

    sampled_xy = _denorm_xy(planner, sample_full.squeeze(1)).detach()   # [G*M,T,2]
    refined_xy = _denorm_xy(planner, last_mean_full.squeeze(1))         # [G*M,T,2]

    return {
        "sampled_xy":   sampled_xy,
        "refined_xy":   refined_xy,
        "refined_traj": reconstruct_heading_from_xy(sampled_xy),    # [G*M,T,3]
        "log_probs":    log_probs_t,                                 # [G*M,steps]
        "chains_norm":  chains_norm_full.detach(),                   # [G*M,steps+1,1,T,2]
        "timesteps":    torch.stack(timesteps_list).detach(),       # [steps]
        "ref_mean_xy":  anchor_ref,                                 # [G*M,T,2]
        "num_modes":    M,
        "num_groups":   num_groups,
        "barrier_metrics": barrier_metrics,
    }


def recompute_refine_log_probs(
    planner,
    context: dict[str, Any],
    chains_norm: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: DDIMSchedulerWithLogProb,
    eta: float,
    ref_mean_xy: torch.Tensor | None = None,
    barrier_config: CBFBarrierGuidanceConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute log-probs and per-step x_0 predictions on a fixed rollout chain.

    Returns:
        log_probs:       [G*M, steps]        — for PG loss
        poses_reg_steps: [G*M, steps, T, 2]  — physical x_0 pred at each step (WITH grad,
                         for BC/KL loss)
    """
    device = planner._device()
    chains_norm = chains_norm.to(device=device)
    timesteps = timesteps.to(device=device)
    groups = int(chains_norm.shape[0])
    repeated_context = _repeat_context(context, groups)
    log_probs = []
    x0_preds = []
    anchor_xy = ref_mean_xy.to(device=device) if ref_mean_xy is not None else None
    mean_guidance_fn = _make_cbf_mean_guidance_fn(
        planner, anchor_xy, barrier_config, None
    ) if anchor_xy is not None else None
    for step_idx in range(int(timesteps.shape[0])):
        sample = chains_norm[:, step_idx]
        prev_sample = chains_norm[:, step_idx + 1]
        timestep = timesteps[step_idx]
        batch_timestep = timestep.expand(groups)
        model_output = planner.predict_denoised_traj(sample, batch_timestep, repeated_context)
        _, log_prob, _ = scheduler.step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            eta=float(eta),
            prev_sample=prev_sample,
            mean_guidance_fn=mean_guidance_fn,
        )
        log_probs.append(log_prob.squeeze(-1))
        # Physical x_0 prediction — keep computation graph for BC/KL loss
        x0_preds.append(_denorm_xy(planner, model_output.squeeze(1)))  # [G*M, T, 2]
    log_probs_t = torch.stack(log_probs, dim=-1)         # [G*M, steps]
    poses_reg_steps = torch.stack(x0_preds, dim=1)       # [G*M, steps, T, 2]
    return log_probs_t, poses_reg_steps



def _compute_speed_reward_batch(
    refined_np: np.ndarray,
    target_kmh: float,
) -> np.ndarray:
    """clip(speed / target_kmh, 0, 1) for each trajectory. [G*M] ∈ [0, 1]."""
    out = np.empty(refined_np.shape[0], dtype=np.float32)
    for i, traj in enumerate(refined_np):
        out[i] = float(np.clip(_traj_speed_kmh(traj) / max(float(target_kmh), 1e-3), 0.0, 1.0))
    return out


def _compute_road_penalty_batch(
    refined_np: np.ndarray,
    road_half_width_m: "float | np.ndarray",
    penalty_decay_m: float = 1.0,
) -> np.ndarray:
    """exp(-max(0, max|y| - half_width) / decay). [G*M] ∈ (0, 1]; 1.0 = within road.

    road_half_width_m can be a scalar (same limit for all) or a per-trajectory
    array of shape [G*M] to apply different limits per mode (e.g. lane-change
    trajectories are allowed wider lateral deviation than keep-lane ones).
    """
    max_lat = np.abs(refined_np[:, :, 1]).max(axis=-1)           # [G*M]
    half_w = np.asarray(road_half_width_m, dtype=np.float32)     # scalar or [G*M]
    excess = np.maximum(0.0, max_lat - half_w)
    return np.exp(-excess / max(float(penalty_decay_m), 1e-6)).astype(np.float32)


def _compute_comfort_reward_batch(
    refined_np: np.ndarray,
    heading_decay_rad: float = 0.1,
) -> np.ndarray:
    """exp(-mean(|Δheading|) / decay). [G*M] ∈ (0, 1]; smoother → closer to 1.0."""
    headings = refined_np[:, :, 2]                          # [G*M, T]
    dh = np.abs(np.diff(headings, axis=-1))                 # [G*M, T-1]
    dh = np.minimum(dh, 2 * np.pi - dh)                    # wrap to [0, π]
    mean_dh = dh.mean(axis=-1)
    return np.exp(-mean_dh / max(float(heading_decay_rad), 1e-6)).astype(np.float32)


def _compute_pdms_reward_batch(
    refined_np: np.ndarray,
    follower_pose: np.ndarray,
    prev_traj: np.ndarray | None,
    prev_pose: np.ndarray | None,
    selected_traj: np.ndarray,
    is_leader: bool,
    formation_scores: np.ndarray,
    params: dict,
    road_half_widths: np.ndarray | None = None,
    lane_y_targets: np.ndarray | None = None,
) -> np.ndarray:
    """Gate × Quality reward for each of the G*M candidate trajectories.

    gate    = collision_gate * road_gate * ttc_gate * smoothness_gate  ∈ {0,1}
    quality = weighted sum of progress/formation/speed/lane/comfort/consistency
    reward  = gate * quality
    """
    GxM = refined_np.shape[0]
    rewards = np.zeros(GxM, dtype=np.float32)

    # Only use first N steps for collision/TTC gates (only t=0 is executed per step)
    _gate_N = int(params.get("gate_horizon_steps", 3))

    # Pre-compute leader world trajectory for first N steps (same for all G*M, only for followers)
    leader_world: np.ndarray | None = None
    if not is_leader and prev_traj is not None and prev_pose is not None:
        leader_world = _local_xy_to_world_xy(prev_pose, prev_traj[:_gate_N, :2])  # [N, 2]

    # Speed target: leader uses configured target; followers match the preceding vehicle's speed
    speed_target_kmh = (params["target_speed_kmh"] if is_leader
                        else max(_traj_speed_kmh(prev_traj) if prev_traj is not None
                                 else params["target_speed_kmh"], 1e-3))

    for gm in range(GxM):
        traj = refined_np[gm]  # [T, 3]

        # ── Gate terms ────────────────────────────────────────────────────────
        collision_gate = 1.0
        ttc_gate       = 1.0
        if leader_world is not None:
            follower_world = _local_xy_to_world_xy(follower_pose, traj[:_gate_N, :2])  # [N, 2]
            dist_t = np.linalg.norm(follower_world - leader_world, axis=-1)  # [N] center-to-center
            bumper_dist_t = dist_t - params.get("vehicle_length_m", 4.8)          # [N] bumper-to-bumper
            collision_gate = float(bumper_dist_t.min() >= params["gate_collision_dist_m"])
            # Per-timestep TTC over first N steps
            veh_len = params.get("vehicle_length_m", 4.8)
            d0 = float(np.linalg.norm(follower_pose[:2] - prev_pose[:2])) - veh_len
            dist_seq = np.concatenate([[max(d0, 0.0)], bumper_dist_t])      # [N+1]
            approach_rate = -np.diff(dist_seq) / 0.5                        # [N], positive = closing
            ttc_per_t = np.where(
                approach_rate > 1e-3,
                bumper_dist_t / np.clip(approach_rate, 1e-3, None),
                np.inf,
            )
            ttc = float(ttc_per_t.min())                                     # worst-case TTC
            ttc_gate = float(ttc >= params["gate_ttc_s"])

        max_lat = float(np.abs(traj[:, 1]).max())
        _road_hw = (float(road_half_widths[gm]) if road_half_widths is not None
                    else params["gate_road_half_width_m"])
        road_gate = float(max_lat <= _road_hw)

        dh = np.abs(np.diff(traj[:, 2]))
        dh = np.minimum(dh, 2 * np.pi - dh)
        smoothness_gate = float(dh.max() <= params["gate_max_dh_rad"])

        gate = collision_gate * road_gate * ttc_gate * smoothness_gate
        if gate == 0.0:
            continue  # leave reward[gm] = 0

        # ── Quality terms ─────────────────────────────────────────────────────
        # progress only for leader; followers use formation to guide following behavior
        progress = (float(np.clip(traj[-1, 0] / max(params["progress_s_max"], 1e-3), 0.0, 1.0))
                    if is_leader else 0.0)
        formation = float(formation_scores[gm])
        speed = float(np.clip(_traj_speed_kmh(traj) / speed_target_kmh, 0.0, 1.0))
        # Lane: lateral deviation of last 2 waypoints from target lane center
        _y_tgt = float(lane_y_targets[gm]) if lane_y_targets is not None else 0.0
        _lat_err = float(np.abs(traj[-2:, 1] - _y_tgt).mean())
        lane = float(np.exp(-_lat_err / max(params["lane_decay_m"], 1e-6)))
        comfort = float(np.exp(-dh.mean() / max(params["comfort_decay_rad"], 1e-6)))
        diff_xy = traj[:, :2] - selected_traj[:, :2]
        consistency = float(np.exp(-np.linalg.norm(diff_xy, axis=-1).mean() / max(params["consistency_decay_m"], 1e-6)))

        quality = (
            params["w_progress"]   * progress
          + params["w_formation"]  * formation
          + params["w_speed"]      * speed
          + params["w_lane"]       * lane
          + params["w_comfort"]    * comfort
          + params["w_pretrain"]   * consistency
        )
        rewards[gm] = float(gate * quality)

    return rewards


def _compute_trajectory_safety_mask(
    refined_np: np.ndarray,
    dt: float = 0.5,
    max_accel_mps2: float = 5.0,
    max_heading_diff_rad: float = np.pi / 4,
    follower_pose: np.ndarray | None = None,
    prev_traj: np.ndarray | None = None,
    prev_pose: np.ndarray | None = None,
    collision_dist_m: float = 1.0,
    vehicle_length_m: float = 4.8,
    collision_check_steps: int = 3,
) -> np.ndarray:
    """Return [N] bool mask: True = safe, False = fails a safety criterion.

    Safety criteria (any failure → unsafe):
      1. First step backward: x[1] - x[0] <= 0 (forward direction = +x in ego-local)
      2. Any step backward: dx[t] < 0 for any t
      3. Longitudinal acceleration out of [-max_accel, +max_accel] m/s²
      4. Heading change between adjacent waypoints > max_heading_diff_rad
      5. Bumper-to-bumper distance to preceding vehicle < collision_dist_m in the
         first collision_check_steps waypoints (only checked when prev_traj/pose given)
    """
    N, T, _ = refined_np.shape
    if N == 0 or T < 2:
        return np.ones(N, dtype=bool)
    x = refined_np[:, :, 0]                          # [N, T]
    h = refined_np[:, :, 2]                          # [N, T]
    dx = np.diff(x, axis=1)                          # [N, T-1]
    dy = np.diff(refined_np[:, :, 1], axis=1)        # [N, T-1]
    dh = np.diff(h, axis=1)                          # [N, T-1]
    dh = (dh + np.pi) % (2 * np.pi) - np.pi         # wrap to [-π, π]
    step_dist = np.sqrt(dx ** 2 + dy ** 2)           # [N, T-1] m
    speed = step_dist / dt                           # [N, T-1] m/s
    safe = np.ones(N, dtype=bool)
    safe &= dx[:, 0] > 0                             # 1. first-step backward → collision-like
    safe &= (dx >= 0).all(axis=1)                    # 2. any backward step
    if T >= 3:
        accel = np.diff(speed, axis=1) / dt          # [N, T-2] m/s²
        safe &= (np.abs(accel) <= max_accel_mps2).all(axis=1)  # 3. accel range
    safe &= (np.abs(dh) <= max_heading_diff_rad).all(axis=1)   # 4. heading jump
    # 5. Collision with preceding vehicle in first collision_check_steps waypoints
    if follower_pose is not None and prev_traj is not None and prev_pose is not None:
        k = min(collision_check_steps, T)
        leader_world = _local_xy_to_world_xy(prev_pose, prev_traj[:k, :2])   # [k, 2]
        follower_world = _local_xy_to_world_xy(
            follower_pose, refined_np[:, :k, :2]                              # [N, k, 2]
        )
        dist = np.linalg.norm(
            follower_world - leader_world[np.newaxis, :, :], axis=-1         # [N, k]
        )
        bumper_dist = dist - float(vehicle_length_m)
        safe &= (bumper_dist >= float(collision_dist_m)).all(axis=1)          # 5. collision
    return safe


def _make_step_discount(
    step_num: int,
    gamma: float = 0.8,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Discount weights [step_num]: discount[i] = gamma^(step_num-1-i).

    Last denoising step → weight 1.0; earlier steps → decreasing weights.
    """
    exponents = np.arange(step_num - 1, -1, -1, dtype=np.float32)
    disc = torch.from_numpy(gamma ** exponents)
    return disc.to(device=device) if device is not None else disc


def save_ddim_chain_scatter_plot(
    *,
    output_path: Path,
    chains_norm: torch.Tensor,       # [G, steps+1, 1, T, 2] normalized (GT mode slice)
    planner,
    pose: np.ndarray,                # [3] world pose of the agent
    agent_id: str,
    gt_mode: int,
    gt_group: int,
    global_step: int,
    episode: int,
    env_step: int,
    barrier_config=None,             # CBFBarrierGuidanceConfig | None
    anchor_xy_phys: np.ndarray | None = None,  # [T, 2] physical ego-local anchor for ellipse
) -> None:
    """Save scatter plot of intermediate DDIM chain trajectories for GT mode.

    Each DDIM step's G groups of trajectories are drawn as scatter points
    with a distinct color, so the denoising progression is visible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    chains_norm = chains_norm.detach()
    G = chains_norm.shape[0]
    n_steps = chains_norm.shape[1]    # steps+1 (step 0 = initial, step K = final)
    T = chains_norm.shape[-2]         # timesteps per trajectory
    pose_np = np.asarray(pose, dtype=np.float64).reshape(-1)

    # ── Barrier in-ellipse rate for final step (step K) ──────────────────────
    in_ellipse_rate: float | None = None
    if (barrier_config is not None and getattr(barrier_config, "enabled", False)
            and anchor_xy_phys is not None):
        dx = max(float(barrier_config.delta_x_m), 1e-6)
        dy = max(float(barrier_config.delta_y_m), 1e-6)
        final_norm = chains_norm[:, -1].squeeze(1)                           # [G, T, 2]
        final_phys = _denorm_xy(planner, final_norm).detach().cpu().numpy()  # [G, T, 2]
        anchor = np.asarray(anchor_xy_phys, dtype=np.float64)                # [T, 2]
        diff = final_phys - anchor[None, :, :]                               # [G, T, 2]
        h = 1.0 - (diff[:, :, 0] / dx) ** 2 - (diff[:, :, 1] / dy) ** 2   # [G, T]
        in_ellipse_rate = float((h >= 0).sum()) / max(h.size, 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    cmap = cm.get_cmap("plasma", n_steps)
    all_pts: list[np.ndarray] = []

    for s in range(n_steps):
        # [G, T, 2] normalized → physical → world
        step_norm = chains_norm[:, s].squeeze(1)       # [G, T, 2]
        step_phys = _denorm_xy(planner, step_norm).detach().cpu().numpy()  # [G, T, 2]
        step_world = np.concatenate([
            _local_xy_to_world_xy(pose_np, step_phys[g, :, :2])
            for g in range(G)
        ], axis=0)  # [G*T, 2]
        all_pts.append(step_world)
        label = f"step {s} (init)" if s == 0 else (f"step {s} (final)" if s == n_steps - 1 else f"step {s}")
        ax.scatter(
            step_world[:, 0], step_world[:, 1],
            color=cmap(s), s=6, alpha=0.55,
            label=label, zorder=2 + s,
        )

    # Mark GT group's final trajectory as a red line
    final_norm = chains_norm[:, -1].squeeze(1)        # [G, T, 2]
    final_phys = _denorm_xy(planner, final_norm).detach().cpu().numpy()
    if gt_group >= 0 and gt_group < G:
        gt_world = _local_xy_to_world_xy(pose_np, final_phys[gt_group, :, :2])
        ax.plot(gt_world[:, 0], gt_world[:, 1],
                color="red", linewidth=2.5, zorder=20, label=f"gt_group={gt_group} (final)")

    if all_pts:
        pts = np.concatenate(all_pts, axis=0)
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        pad = max(3.0, float(np.max(mx - mn)) * 0.15)
        ax.set_xlim(float(mn[0] - pad), float(mx[0] + pad))
        ax.set_ylim(float(mn[1] - pad), float(mx[1] + pad))

    ax.set_title(
        f"DDIM chain scatter | episode={episode} env_step={env_step} global={global_step}\n"
        f"agent={agent_id}  gt_mode={gt_mode}  gt_group={gt_group}  G={G}  steps={n_steps-1}"
    )
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    # Colorbar for step progression
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, n_steps - 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="DDIM step (0=noisy init, K=final)")
    ax.legend(fontsize=7, loc="best")
    if in_ellipse_rate is not None:
        total_pts = G * T
        inside_pts = int(round(in_ellipse_rate * total_pts))
        info_text = (
            f"barrier: δx={barrier_config.delta_x_m:.2f}m  δy={barrier_config.delta_y_m:.2f}m\n"
            f"step{n_steps-1} in-ellipse: {in_ellipse_rate*100:.1f}%"
            f"  ({inside_pts}/{total_pts} pts)"
        )
        ax.text(
            0.01, 0.97, info_text,
            transform=ax.transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def _execute_trajectories(env, planner, agent_ids: list[str], trajectories: dict[str, np.ndarray], config: Mapping[str, Any]):
    lookahead_index = int(config.get("lookahead_index", 2))
    target_speed_km_h = float(config.get("target_speed_km_h", 30.0))
    controller_type = str(config.get("controller_type", "stabilized"))
    low_level_actions = {}
    for agent_id in agent_ids:
        trajectory = np.asarray(trajectories[agent_id], dtype=np.float32)
        vehicle = getattr(env.base_env, "agents", {}).get(agent_id)
        current_speed_km_h = float(getattr(vehicle, "speed_km_h", 0.0)) if vehicle is not None else 0.0
        action_2d, _ = compute_trajectory_control(
            trajectory=trajectory,
            lookahead_index=lookahead_index,
            current_speed_km_h=current_speed_km_h,
            target_speed_km_h=target_speed_km_h,
            controller_type=controller_type,
        )
        low_level_actions[agent_id] = action_2d
    if hasattr(env.base_env, "_pending_step_trajectories"):
        env.base_env._pending_step_trajectories = {
            agent_id: np.asarray(trajectories[agent_id], dtype=np.float32)
            for agent_id in agent_ids
        }
    result = env.base_env.step(low_level_actions)
    if len(result) == 5:
        raw_obs, reward, terminated, truncated, info = result
        done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
    else:
        raw_obs, reward, done_dict, info = result
        terminated = done_dict
        truncated = {agent_id: False for agent_id in done_dict}
        done = bool(done_dict.get("__all__", False))
    reward_values = [float((reward or {}).get(agent_id, 0.0)) for agent_id in agent_ids]
    scalar_reward = float(np.mean(reward_values)) if reward_values else 0.0
    env._last_raw_obs = env._normalize_raw_obs(raw_obs or {}, previous_obs=env._last_raw_obs)
    obs = env._refresh_mode_export() if env._last_raw_obs else env._last_obs
    return obs, scalar_reward, done, dict(info or {})


def _save_refine_checkpoint(planner, ckpt_dir: Path, config: Mapping[str, Any], extra: Mapping[str, Any] | None = None):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_refine_state_dict(planner), ckpt_dir / "refine_head.pt")
    torch.save({"model_state": planner.state_dict(), "config": dict(config)}, ckpt_dir / "full_platoon_refine_grpo.ckpt")
    meta = dict(extra or {})
    meta.update({"num_agents": int(config.get("num_agents", 3)), "type": "selected_refine_grpo"})
    (ckpt_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run_training(config: Mapping[str, Any], output_root: Path, total_timesteps: int) -> Path:
    import random as _random
    from envs.mode_selection_sb3_env import ModeSelectionSB3Env

    # Global random seed (None = non-deterministic)
    _seed = config.get("seed", None)
    if _seed is not None:
        _seed = int(_seed)
        _random.seed(_seed)
        np.random.seed(_seed)
        torch.manual_seed(_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_seed)
        print(f"[refine-grpo] random seed = {_seed}", flush=True)
    else:
        print("[refine-grpo] random seed = None (non-deterministic)", flush=True)

    run_dir = create_next_run_dir(output_root)
    env_config = dict(config.get("env_config", {}))
    env_config.setdefault("num_agents", int(config.get("num_agents", 3)))
    env_config.setdefault("observation_mode", "multimodal")
    env_config.setdefault("use_render", False)
    env_config.setdefault("planner_device", "cuda")
    env_config.setdefault("lookahead_index", int(config.get("lookahead_index", 2)))
    env_config.setdefault("target_speed_km_h", float(config.get("target_speed_km_h", 30.0)))
    env_config.setdefault("controller_type", str(config.get("controller_type", "stabilized")))
    env_config["use_action_mask"] = False  # no invalid action masking for refinement training
    env_config["trajectory_source"] = "diffusion"
    # Propagate scenario seed from top-level config if not set in env_config
    if "start_seed" not in env_config and "seed" not in env_config:
        _scenario_seed = config.get("start_seed", None)
        if _scenario_seed is not None:
            env_config["start_seed"] = int(_scenario_seed)

    planner = build_planner_for_selected_refinement(config, env_config)
    trainable_names = freeze_for_selected_refinement(planner)
    if not trainable_names:
        raise RuntimeError("No trainable refinement parameters were found.")
    print(f"[refine-grpo] trainable params: {trainable_names}", flush=True)
    env_config["planner"] = planner
    env = ModeSelectionSB3Env(env_config)

    params = [p for p in planner.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=float(config.get("refine_lr", 1e-5)))
    keeper = TopKCheckpointKeeper(int(config.get("ckpt_top_k", 3)))

    num_groups = int(config.get("num_refine_groups", 4))
    rollout_chunk_size = int(config.get("rollout_chunk_size", 80))
    noise_t = int(config.get("refine_noise_t", 8))
    denoise_steps = int(config.get("refine_denoise_steps", 4))
    eta = float(config.get("refine_eta", 0.1))
    noise_std = float(config.get("refine_noise_std", 0.02))
    max_delta_norm = float(config.get("refine_max_delta_norm", 0.05))
    # PDMS gate thresholds (gate_collision_dist_m 为保险杠间距，非中心距)
    gate_collision_dist_m = float(config.get("gate_collision_dist_m",  1.0))  # bumper-to-bumper
    gate_ttc_s            = float(config.get("gate_ttc_s",             0.5))
    gate_road_half_width_m= float(config.get("gate_road_half_width_m", 3.0))
    gate_max_dh_rad       = float(config.get("gate_max_dh_rad",        0.6))
    vehicle_length_m      = float(config.get("vehicle_length_m",       4.8))
    gate_horizon_steps    = int(config.get("gate_horizon_steps",       2))   # collision/TTC 只看前N步
    lane_change_width_m   = float(config.get("lane_change_width_m",    3.5))  # target lane offset for LC
    # PDMS quality weights
    w_progress  = float(config.get("w_progress",  0.2))
    w_formation = float(config.get("w_formation", 0.4))
    w_speed     = float(config.get("w_speed",     0.2))
    w_lane      = float(config.get("w_lane",      0.15))
    w_comfort   = float(config.get("w_comfort",   0.05))
    w_pretrain  = float(config.get("w_pretrain",  0.0))
    # PDMS quality decay params
    lane_decay_m          = float(config.get("lane_decay_m",           1.0))
    comfort_decay_rad     = float(config.get("comfort_decay_rad",      0.1))
    consistency_decay_m   = float(config.get("consistency_decay_m",    1.0))
    cls_weight = float(config.get("refine_cls_weight", 1.0))
    bc_weight = float(config.get("refine_bc_weight", 0.1))
    kl_weight = float(config.get("refine_kl_weight", 0.02))
    no_positive_bc_weight = float(config.get("refine_no_positive_bc_weight", 1.0))
    advantage_baseline_strategy = str(config.get("refine_advantage_baseline", "group_mean"))
    discount_gamma = float(config.get("refine_discount_gamma", 0.8))
    traj_dt_s = float(config.get("traj_dt_s", 0.5))
    safety_max_accel_mps2 = float(config.get("safety_max_accel_mps2", 5.0))
    safety_max_heading_diff_rad = float(config.get("safety_max_heading_diff_deg", 45.0)) * np.pi / 180.0
    print_all_mode_debug = bool(config.get("refine_print_all_mode_debug", False))
    print_all_mode_debug_interval = max(1, int(config.get("refine_print_all_mode_debug_interval", 1)))
    _debug_idx_raw = config.get("refine_debug_agent_indices", [])
    debug_agent_indices: set[int] = set(int(i) for i in _debug_idx_raw) if _debug_idx_raw else set()
    debug_ddim_chains = bool(config.get("refine_debug_ddim_chains", False))
    desired_gap_m = float(config.get("desired_gap_m", 10.0))
    progress_s_max = float(config.get("progress_s_max", 15.0))
    target_speed_km_h = float(config.get("target_speed_km_h", 30.0))
    ckpt_interval = int(config.get("checkpoint_interval_steps", 5000))
    ckpt_root = run_dir / "checkpoints"
    _debug_noise_plot_dir_str = str(config.get("debug_noise_plot_dir", "") or "")
    debug_noise_plot_dir = Path(_debug_noise_plot_dir_str) if _debug_noise_plot_dir_str else None
    refine_scheduler = DDIMSchedulerWithLogProb(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="sample",
    )
    refine_scheduler.set_timesteps(1000, device=planner._device())
    barrier_config = _resolve_barrier_guidance_config(config, refine_scheduler, noise_t)

    # Frozen pretrained planner — deep copy taken before any GRPO updates; used only when
    # refine_advantage_baseline == 'pretrain_reward' to compute the true pretrain trajectory reward.
    _use_pretrain_baseline = (
        str(advantage_baseline_strategy).strip().lower()
        in {"pretrain_reward", "pretrain", "pretrained"}
    )
    if _use_pretrain_baseline:
        import copy as _copy
        pretrain_planner = _copy.deepcopy(planner)
        pretrain_planner.eval()
        for _p in pretrain_planner.parameters():
            _p.requires_grad_(False)
        print("[refine-grpo] frozen pretrain_planner created for pretrain_reward baseline.", flush=True)
    else:
        pretrain_planner = None

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(run_dir / "tb"))
    except Exception:
        writer = None

    global_step = 0
    episode_idx = 0
    episode_scores: list[float] = []
    last_ckpt_step = 0
    debug_path = run_dir / "selected_refine_debug.jsonl"

    while global_step < int(total_timesteps):
        env.reset()
        done = False
        episode_reward = 0.0
        episode_step = 0
        while not done and global_step < int(total_timesteps):
            planner_batch = env._last_planner_batch
            export = env._last_export
            if not planner_batch or export is None:
                break
            agent_ids = list(export["agent_ids"])
            candidates = np.asarray(export["trajectory_candidates"], dtype=np.float32)
            masks = np.asarray(export["mode_valid_mask"], dtype=bool)
            mode_names_for_debug = mode_names_from_planner(planner, masks.shape[1])
            masked_logits_np = np.asarray(export["masked_cls_logits"], dtype=np.float32)
            on_training_modes = np.argmax(masked_logits_np, axis=-1).astype(np.int64)
            cls_feature = torch.as_tensor(export["cls_feature"], dtype=torch.float32, device=planner._device())
            contexts, context_agent_ids = planner.extract_rl_context(planner_batch)
            if list(context_agent_ids) != agent_ids:
                raise RuntimeError(f"context/export agent mismatch: {context_agent_ids} != {agent_ids}")
            coarse_traj_by_agent: dict[str, torch.Tensor] = {}
            for _aid in agent_ids:
                _ct = (planner_batch.get(_aid) or {}).get("coarse_trajectories")
                if _ct is not None:
                    coarse_traj_by_agent[_aid] = torch.as_tensor(
                        _ct, dtype=torch.float32, device=planner._device()
                    )
            poses = {aid: _get_vehicle_pose(env, aid) for aid in agent_ids}

            executed: dict[str, np.ndarray] = {}
            debug_rows = []
            step_denoising_rewards: list[float] = []
            plot_trajectories_by_agent: dict[str, torch.Tensor | np.ndarray] = {}
            plot_gt_modes_by_agent: dict[str, int] = {}
            plot_gt_groups_by_agent: dict[str, int] = {}
            plot_rewards_by_agent: dict[str, torch.Tensor | np.ndarray] = {}
            on_training_traj_by_agent: dict[str, np.ndarray] = {}
            on_training_reward_by_agent: dict[str, float] = {}
            on_training_mode_by_agent: dict[str, int] = {}
            plot_all_refined_by_agent: dict[str, dict] = {}
            plot_noisy_init_by_agent: dict[str, np.ndarray] = {}
            for idx, agent_id in enumerate(agent_ids):
                # Recompute cls logits with the current (possibly just-updated) policy
                agent_train_logit = _get_cls_branch(planner)(cls_feature[idx].unsqueeze(0)).squeeze(0).squeeze(-1)
                on_training_mode = int(on_training_modes[idx])
                if on_training_mode < 0 or on_training_mode >= masks.shape[1] or not bool(masks[idx, on_training_mode]):
                    on_training_mode = int(np.argmax(masks[idx]))
                    on_training_modes[idx] = on_training_mode
                selected_traj = candidates[idx, on_training_mode]
                plot_trajectories_by_agent[agent_id] = selected_traj[None, ...]
                plot_gt_modes_by_agent[agent_id] = on_training_mode
                plot_gt_groups_by_agent[agent_id] = -1
                on_training_traj_by_agent[agent_id] = selected_traj
                on_training_mode_by_agent[agent_id] = on_training_mode
                # ── multi-modal rollout: G groups × M anchors (dynamic coarse when available) ──
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
                barrier_summary = _summarize_barrier_metrics(rollout.get("barrier_metrics"))
                M   = rollout["num_modes"]
                GxM = num_groups * M

                M = int(M)
                mode_valid = torch.as_tensor(masks[idx], dtype=torch.bool, device=planner._device())
                refined_np = rollout["refined_traj"].detach().cpu().numpy().astype(np.float32)  # [G*M,T,3]
                # ── PDMS: Gate × Quality reward ──────────────────────────────
                is_ldr = (idx == 0)
                _prev_traj_arg = None
                _prev_pose_arg = None
                on_training_traj_reward: float = float("nan")
                if not is_ldr and poses.get(agent_id) is not None and poses.get(agent_ids[idx - 1]) is not None:
                    prev_agent = agent_ids[idx - 1]
                    _prev_traj_arg = executed.get(prev_agent, candidates[idx - 1, int(on_training_modes[idx - 1])])
                    _prev_pose_arg = poses[prev_agent]

                # formation scores: batch computation avoids per-trajectory loops
                if is_ldr or _prev_traj_arg is None:
                    _formation_scores = np.ones(GxM, dtype=np.float32)
                else:
                    _formation_scores = compute_pairwise_formation_reward(
                        refined_np, poses[agent_id], _prev_traj_arg, _prev_pose_arg,
                        desired_gap_m=desired_gap_m, progress_s_max=progress_s_max,
                    )

                _pdms_params = {
                    "gate_collision_dist_m":  gate_collision_dist_m,
                    "gate_ttc_s":             gate_ttc_s,
                    "gate_road_half_width_m": gate_road_half_width_m,
                    "gate_max_dh_rad":        gate_max_dh_rad,
                    "w_progress":    w_progress,  "w_formation": w_formation,
                    "w_speed":       w_speed,     "w_lane":      w_lane,
                    "w_comfort":     w_comfort,   "w_pretrain":  w_pretrain,
                    "progress_s_max":     progress_s_max,
                    "target_speed_kmh":   target_speed_km_h,
                    "lane_decay_m":       lane_decay_m,
                    "comfort_decay_rad":  comfort_decay_rad,
                    "consistency_decay_m":consistency_decay_m,
                    "desired_gap_m":      desired_gap_m,
                    "vehicle_length_m":   vehicle_length_m,
                    "gate_horizon_steps": gate_horizon_steps,
                }
                # Per-trajectory semantic road/lane arrays based on mode names
                _road_hw_gm    = np.empty(GxM, dtype=np.float32)
                # Lane target y: use coarse anchor's LAST WAYPOINT y in ego-local
                # (map-aware, recomputed each step → handles "already in target lane" correctly)
                _coarse_np = coarse_traj_by_agent.get(agent_id)
                if _coarse_np is not None:
                    if torch.is_tensor(_coarse_np):
                        _coarse_np = _coarse_np.detach().cpu().numpy()
                    _coarse_np = np.asarray(_coarse_np, dtype=np.float32)  # [M, T, 2]

                _lane_y_tgt_gm = np.zeros(GxM, dtype=np.float32)
                for _m in range(M):
                    _mname = (mode_names_for_debug[_m]
                              if mode_names_for_debug and _m < len(mode_names_for_debug) else "")
                    _is_lc = "LC" in _mname
                    _road_hw_gm[_m::M] = gate_road_half_width_m * (2.0 if _is_lc else 1.0)
                    # Use coarse endpoint y if available; fall back to 0 (current lane center)
                    if (_coarse_np is not None and _m < _coarse_np.shape[0]
                            and _coarse_np.ndim == 3 and _coarse_np.shape[2] >= 2):
                        _lane_y_tgt_gm[_m::M] = float(_coarse_np[_m, -1, 1])  # last waypoint y
                    # else: keep 0 (current lane center, no lateral drift encouraged)

                # On-training trajectory: use its own mode's coarse endpoint y.
                _on_training_mode_name_for_reward = (
                    mode_names_for_debug[on_training_mode]
                    if mode_names_for_debug and on_training_mode < len(mode_names_for_debug) else ""
                )
                _on_training_road_hw = np.array([gate_road_half_width_m * (2.0 if "LC" in _on_training_mode_name_for_reward else 1.0)],
                                             dtype=np.float32)
                if (_coarse_np is not None and on_training_mode < _coarse_np.shape[0]
                        and _coarse_np.ndim == 3 and _coarse_np.shape[2] >= 2):
                    _on_training_lane_y = np.array([float(_coarse_np[on_training_mode, -1, 1])], dtype=np.float32)
                else:
                    _on_training_lane_y = np.array([0.0], dtype=np.float32)  # fall back to current lane

                rewards = _compute_pdms_reward_batch(
                    refined_np, poses[agent_id],
                    _prev_traj_arg, _prev_pose_arg,
                    selected_traj, is_ldr,
                    _formation_scores, _pdms_params,
                    road_half_widths=_road_hw_gm,
                    lane_y_targets=_lane_y_tgt_gm,
                )
                on_training_traj_reward = float(_compute_pdms_reward_batch(
                    selected_traj[None, :], poses[agent_id],
                    _prev_traj_arg, _prev_pose_arg,
                    selected_traj, is_ldr,
                    _formation_scores[:1], _pdms_params,
                    road_half_widths=_on_training_road_hw,
                    lane_y_targets=_on_training_lane_y,
                )[0])
                on_training_reward_by_agent[agent_id] = on_training_traj_reward

                reward_t = torch.as_tensor(rewards, dtype=torch.float32, device=planner._device())
                reward_2d = reward_t.view(num_groups, M)
                all_adv_raw = normalize_multimodal_advantages(
                    reward_t,
                    num_groups,
                    M,
                    positive_only=False,
                    baseline_reward=None,   # always group_mean for normalisation
                ).view(num_groups, M)
                # Post-processing: zero out low-quality trajectories
                # group_mean      → zero advantages below group mean (adv < 0)
                # pretrain_reward → run frozen pretrain_planner (1 group, deterministic),
                #                   compute its reward, normalise to advantage space using the
                #                   same per-mode stats, zero out any adv below that threshold
                _strategy = str(advantage_baseline_strategy).strip().lower()
                if _strategy in {"pretrain_reward", "pretrain", "pretrained"}:
                    with torch.no_grad():
                        _pretrain_rollout = rollout_multimodal_refinement(
                            pretrain_planner,
                            contexts[agent_id],
                            num_groups=1,
                            noise_t=noise_t,
                            denoise_steps=denoise_steps,
                            eta=0.0,            # deterministic denoising for reference
                            scheduler=refine_scheduler,
                            chunk_size=rollout_chunk_size,
                            coarse_trajectories=coarse_traj_by_agent.get(agent_id),
                            barrier_config=None,
                        )
                    _pretrain_np = _pretrain_rollout["refined_traj"].detach().cpu().numpy()  # [M, T, 3]
                    _pretrain_traj_m = _pretrain_np[on_training_mode]           # [T, 3]
                    if is_ldr or _prev_traj_arg is None:
                        _pretrain_form = np.ones(1, dtype=np.float32)
                    else:
                        _pretrain_form = compute_pairwise_formation_reward(
                            _pretrain_traj_m[None, :], poses[agent_id],
                            _prev_traj_arg, _prev_pose_arg,
                            desired_gap_m=desired_gap_m, progress_s_max=progress_s_max,
                        )
                    _pretrain_reward = float(_compute_pdms_reward_batch(
                        _pretrain_traj_m[None, :], poses[agent_id],
                        _prev_traj_arg, _prev_pose_arg,
                        selected_traj, is_ldr,
                        _pretrain_form, _pdms_params,
                        road_half_widths=_on_training_road_hw,
                        lane_y_targets=_on_training_lane_y,
                    )[0])
                    _rewards_m = reward_2d[:, on_training_mode]                 # [G] rollout rewards
                    _mean_m = _rewards_m.mean()
                    _std_m = _rewards_m.std(unbiased=False)
                    _pretrain_adv = float(
                        (_pretrain_reward - _mean_m) / (_std_m + 1e-6)
                    )
                    all_adv_clipped = torch.where(
                        all_adv_raw < _pretrain_adv,
                        torch.zeros_like(all_adv_raw),
                        all_adv_raw,
                    )
                else:
                    all_adv_clipped = all_adv_raw.clamp(min=0.0)

                # Safety masking: clamp to 0, then override with -1 for unsafe trajectories
                _safety_ok = _compute_trajectory_safety_mask(
                    refined_np,
                    dt=traj_dt_s,
                    max_accel_mps2=safety_max_accel_mps2,
                    max_heading_diff_rad=safety_max_heading_diff_rad,
                    follower_pose=poses.get(agent_id) if not is_ldr else None,
                    prev_traj=_prev_traj_arg,
                    prev_pose=_prev_pose_arg,
                    collision_dist_m=gate_collision_dist_m,
                    vehicle_length_m=vehicle_length_m,
                    collision_check_steps=3,
                )  # [G*M] bool numpy
                _adv_masked = all_adv_clipped.view(-1).clone()          # [G*M] positive-clamped
                _unsafe = ~torch.from_numpy(_safety_ok).to(_adv_masked.device)
                _adv_masked[_unsafe] = -1.0                             # safety failure → -1

                # Per-step discount weights for the denoising chain
                _step_disc = _make_step_discount(
                    denoise_steps, gamma=discount_gamma, device=planner._device()
                )  # [denoise_steps]

                try:
                    gt_group, gt_mode, gt_score = select_best_mode_group(reward_2d)
                    fallback = ""
                except ValueError:
                    gt_group, gt_mode, gt_score = -1, on_training_mode, float("nan")
                    fallback = "no_trajectories"

                _debug_this_agent = (
                    print_all_mode_debug
                    and (global_step % print_all_mode_debug_interval == 0)
                    and (not debug_agent_indices or idx in debug_agent_indices)
                )
                if _debug_this_agent:

                    _on_training_mode_name = (
                        mode_names_for_debug[on_training_mode]
                        if mode_names_for_debug and on_training_mode < len(mode_names_for_debug)
                        else str(on_training_mode)
                    )
                    print(
                        f"[refine-grpo][on-training-baseline] "
                        f"global_step={global_step} episode={episode_idx} env_step={episode_step} agent={agent_id} "
                        f"on_training_mode={on_training_mode}({_on_training_mode_name}) "
                        f"on_training_traj_reward={on_training_traj_reward:.4f} "
                        f"advantage_baseline={advantage_baseline_strategy}",
                        flush=True,
                    )

                    print(
                        format_all_mode_debug_matrix(
                            global_step=global_step,
                            episode=episode_idx,
                            env_step=episode_step,
                            agent_id=agent_id,
                            gt_mode=gt_mode,
                            gt_group=gt_group,
                            gt_reward=gt_score,
                            on_training_mode=on_training_mode,
                            rewards=reward_2d,
                            adv_raw=all_adv_raw,
                            adv_clipped=all_adv_clipped,
                            mode_names=mode_names_for_debug,
                        ),
                        flush=True,
                    )

                    _bm = rollout.get("barrier_metrics") or []
                    if _bm:
                        _h_min = min(float(m.get("h_min", 0.0)) for m in _bm)
                        _h_mean = float(np.mean([float(m.get("h_mean", 0.0)) for m in _bm]))
                        _cost = float(np.mean([float(m.get("barrier_cost", 0.0)) for m in _bm]))
                        _grad = float(np.mean([float(m.get("guidance_grad_norm", 0.0)) for m in _bm]))
                        print(
                            f"[refine-grpo][barrier-guidance] "
                            f"global_step={global_step} episode={episode_idx} env_step={episode_step} agent={agent_id} "
                            f"enabled={barrier_config.enabled} dx={barrier_config.delta_x_m:.3f} dy={barrier_config.delta_y_m:.3f} "
                            f"h_min={_h_min:.4f} h_mean={_h_mean:.4f} "
                            f"cost_mean={_cost:.4f} grad_norm_mean={_grad:.4f}",
                            flush=True,
                        )

                actual_gm = int(on_training_mode)  # group 0, on-training mode
                actual_reward = float(rewards[actual_gm])

                cls_loss = focal_gt_mode_loss(
                    agent_train_logit,
                    torch.tensor(gt_mode, dtype=torch.long, device=planner._device()),
                    valid_mask=mode_valid,
                )
                agent_loss = float(cls_weight) * cls_loss

                mode_slice = slice_gt_mode_refinement(
                    chains_norm=rollout["chains_norm"],
                    refined_xy=rollout["refined_xy"],
                    ref_mean_xy=rollout["ref_mean_xy"],
                    rewards=reward_t,
                    num_groups=num_groups,
                    num_modes=M,
                    gt_mode=gt_mode,
                )
                plot_trajectories_by_agent[agent_id] = mode_slice["refined_xy"].detach()
                plot_gt_modes_by_agent[agent_id] = gt_mode
                plot_gt_groups_by_agent[agent_id] = gt_group
                plot_all_refined_by_agent[agent_id] = {
                    "traj": refined_np,   # [G*M, T, 3] all modes, all groups
                    "num_groups": num_groups,
                    "num_modes": M,
                }
                # Initial noisy trajectories for ALL G*M groups (chains_norm[:,0] = before denoising)
                # rollout["chains_norm"]: [G*M, steps+1, 1, T, 2] normalized
                _noisy_norm_all = rollout["chains_norm"][:, 0].squeeze(1)  # [G*M, T, 2] normalized
                plot_noisy_init_by_agent[agent_id] = {
                    "traj": _denorm_xy(planner, _noisy_norm_all).detach().cpu().numpy(),  # [G*M, T, 2]
                    "num_groups": num_groups,
                    "num_modes": M,
                }
                # ── DDIM chain scatter debug ──────────────────────────────────
                if debug_ddim_chains and _debug_this_agent and poses.get(agent_id) is not None:
                    _chain_path = (
                        run_dir
                        / "ddim_chain_debug"
                        / f"episode_{episode_idx:03d}"
                        / f"step_{global_step:07d}_{agent_id}_m{gt_mode}.png"
                    )
                    save_ddim_chain_scatter_plot(
                        output_path=_chain_path,
                        chains_norm=mode_slice["chains_norm"],   # [G, steps+1, 1, T, 2]
                        planner=planner,
                        pose=poses[agent_id],
                        agent_id=agent_id,
                        gt_mode=gt_mode,
                        gt_group=gt_group,
                        global_step=global_step,
                        episode=episode_idx,
                        env_step=episode_step,
                        barrier_config=barrier_config,
                        anchor_xy_phys=(
                            mode_slice["ref_mean_xy"][0].detach().cpu().numpy()
                            if barrier_config is not None and barrier_config.enabled
                            else None
                        ),
                    )

                plot_rewards_by_agent[agent_id] = mode_slice["rewards"].detach()

                new_logp_sum_debug: list[float] = []
                new_log_probs, poses_reg_steps = recompute_refine_log_probs(
                    planner,
                    contexts[agent_id],
                    chains_norm=rollout["chains_norm"],   # [G*M, steps+1, 1, T, 2]
                    timesteps=rollout["timesteps"],
                    scheduler=refine_scheduler,
                    eta=eta,
                    ref_mean_xy=rollout["ref_mean_xy"],  # [G*M, T, 2]
                    barrier_config=barrier_config,
                )  # new_log_probs: [G*M, steps]; poses_reg_steps: [G*M, steps, T, 2]
                new_logp_sum_debug = (
                    new_log_probs.sum(dim=-1).detach().cpu().numpy().astype(float).tolist()
                )
                refine_loss_i, metrics_i = ddv2_refine_grpo_loss(
                    new_log_probs=new_log_probs,
                    advantages=_adv_masked,
                    poses_reg_steps=poses_reg_steps,        # [G*M, steps, T, 2] with grad
                    selected_xy=rollout["ref_mean_xy"][0],  # API compat placeholder
                    bc_weight=bc_weight,
                    kl_weight=kl_weight,
                    ref_mean_xy=rollout["ref_mean_xy"],     # [G*M, T, 2]
                    step_discount=_step_disc,
                    no_positive_bc_weight=no_positive_bc_weight,
                )
                agent_loss = agent_loss + refine_loss_i
                metrics_i["cls_loss"] = float(cls_loss.detach())
                metrics_i["total_loss"] = float(agent_loss.detach())
                metrics_i.update(barrier_summary)
                optimizer.zero_grad()
                agent_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, float(config.get("max_grad_norm", 1.0)))
                optimizer.step()

                executed[agent_id] = refined_np[actual_gm]
                step_denoising_rewards.append(actual_reward)
                debug_rows.append({
                    "agent_id": agent_id,
                    "on_training_mode": on_training_mode,
                    "gt_mode": gt_mode,
                    "gt_group": gt_group,
                    "actual_reward": actual_reward,
                    "all_mode_group_rewards": reward_2d.detach().cpu().numpy().astype(float).tolist(),
                    "advantages": _adv_masked.detach().cpu().numpy().astype(float).tolist(),
                    "safety_ok": _safety_ok.tolist(),
                    "fallback": fallback,
                    "metrics": metrics_i,
                    "rollout_logp_sum": rollout["log_probs"].sum(dim=-1).detach().cpu().numpy().astype(float).tolist(),
                    "new_logp_sum": new_logp_sum_debug,
                })

            if print_all_mode_debug and (global_step % print_all_mode_debug_interval == 0) and plot_trajectories_by_agent:
                platoon_plot_path = (
                    run_dir
                    / "all_mode_debug_plots"
                    / f"episode_{episode_idx:03d}"
                    / f"step_{global_step:07d}_platoon_gt_modes.png"
                )
                _plot_agent_ids = (
                    [aid for i, aid in enumerate(agent_ids) if i in debug_agent_indices]
                    if debug_agent_indices else agent_ids
                )
                save_platoon_gt_mode_group_trajectory_plot(
                    output_path=platoon_plot_path,
                    agent_ids=_plot_agent_ids,
                    poses=poses,
                    trajectories_by_agent=plot_trajectories_by_agent,
                    gt_modes_by_agent=plot_gt_modes_by_agent,
                    gt_groups_by_agent=plot_gt_groups_by_agent,
                    rewards_by_agent=plot_rewards_by_agent,
                    on_training_traj_by_agent=on_training_traj_by_agent,
                    on_training_reward_by_agent=on_training_reward_by_agent,
                    on_training_mode_by_agent=on_training_mode_by_agent,
                    coarse_traj_by_agent=coarse_traj_by_agent,
                    all_refined_by_agent=plot_all_refined_by_agent,
                    noisy_init_by_agent=plot_noisy_init_by_agent,
                    global_step=global_step,
                    episode=episode_idx,
                    env_step=episode_step,
                )
                print(f"[refine-grpo][all-mode-debug] saved_platoon_plot={platoon_plot_path}", flush=True)

            try:
                _, env_reward, done, info = _execute_trajectories(env, planner, agent_ids, executed, env_config)
            except AssertionError as _panda3d_err:
                print(
                    f"[refine-grpo] Panda3D rendering crash at step={global_step}: {_panda3d_err}\n"
                    f"[refine-grpo] Saving emergency checkpoint and exiting with os._exit(0).",
                    flush=True,
                )
                _emergency_dir = ckpt_root / "final"
                _save_refine_checkpoint(planner, _emergency_dir, config,
                                        {"note": "panda3d crash exit", "total_steps": global_step})
                if writer is not None:
                    try:
                        writer.close()
                    except Exception:
                        pass
                os._exit(0)
            step_denoising_reward = float(np.mean(step_denoising_rewards)) if step_denoising_rewards else float("nan")
            episode_reward += step_denoising_reward if not np.isnan(step_denoising_reward) else 0.0
            episode_step += 1
            global_step += 1
            if writer is not None:
                if not np.isnan(step_denoising_reward):
                    writer.add_scalar("train/denoising_reward_step", step_denoising_reward, global_step)
                _step_losses = [row["metrics"].get("total_loss", 0.0) for row in debug_rows if row.get("metrics")]
                if _step_losses:
                    writer.add_scalar("train/refine_loss", float(np.mean(_step_losses)), global_step)
            with debug_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "global_step": global_step,
                    "episode": episode_idx,
                    "step": episode_step,
                    "env_reward": float(env_reward),
                    "denoising_reward": step_denoising_reward,
                    "agents": debug_rows,
                    "terminated": bool(done),
                }) + "\n")

            if global_step - last_ckpt_step >= ckpt_interval:
                last_ckpt_step = global_step
                score = episode_reward / max(1, episode_step)
                ckpt_dir = ckpt_root / f"step_{global_step:07d}_score_{score:.4f}"
                _save_refine_checkpoint(planner, ckpt_dir, config, {"score": score, "global_step": global_step})
                keeper.update(score, ckpt_dir)
                print(f"[refine-grpo] step={global_step} score={score:.4f} checkpoint={ckpt_dir}", flush=True)

        score = episode_reward / max(1, episode_step)
        episode_scores.append(score)
        if writer is not None:
            writer.add_scalar("train/episode_reward_per_step", score, episode_idx)
        print(f"[refine-grpo] episode={episode_idx} steps={episode_step} reward/step={score:.4f}", flush=True)
        episode_idx += 1

    final_dir = ckpt_root / "final"
    _save_refine_checkpoint(planner, final_dir, config, {"total_steps": global_step, "episodes": episode_idx})
    (run_dir / "summary.json").write_text(json.dumps({
        "total_steps": global_step,
        "episodes": episode_idx,
        "mean_episode_reward_per_step": float(np.mean(episode_scores)) if episode_scores else 0.0,
        "trainable_names": trainable_names,
    }, indent=2), encoding="utf-8")
    if writer is not None:
        writer.close()
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Selected-mode local refinement GRPO training.")
    parser.add_argument("--config", default=str(os.environ.get("CONFIG", DEFAULT_CONFIG_PATH)))
    parser.add_argument("--pretrained-ckpt", default="")
    parser.add_argument("--cls-grpo-ckpt-dir", default=_env_default_cls_grpo_ckpt_dir())
    parser.add_argument("--cls-grpo-full-ckpt", default=str(os.environ.get("CLS_GRPO_FULL_CKPT", "") or ""))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--total-env-steps", type=int, default=None)
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--planner-device", default="")
    parser.add_argument("--scenario-ids", default=str(os.environ.get("SCENARIO_IDS", "") or ""))
    parser.add_argument("--start-seed", type=int, default=None,
                        help="Scenario start seed (overrides yaml start_seed / env_config.start_seed).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Global random seed for torch/numpy (overrides yaml seed).")
    return parser.parse_args(argv)


def resolve_total_timesteps(cli_total_env_steps: int | None, config: Mapping[str, Any]) -> int:
    if cli_total_env_steps is not None:
        return int(cli_total_env_steps)
    return int(config.get("total_timesteps", 50000))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.pretrained_ckpt:
        config["pretrained_ckpt"] = args.pretrained_ckpt
    if args.cls_grpo_ckpt_dir:
        config["cls_grpo_ckpt_dir"] = args.cls_grpo_ckpt_dir
    if args.cls_grpo_full_ckpt:
        config["cls_grpo_full_ckpt"] = args.cls_grpo_full_ckpt
    if args.num_agents is not None and args.num_agents > 0:
        config["num_agents"] = int(args.num_agents)
        config.setdefault("env_config", {})["num_agents"] = int(args.num_agents)
    if args.planner_device:
        config.setdefault("env_config", {})["planner_device"] = args.planner_device
    if args.scenario_ids:
        ids = [s.strip() for s in args.scenario_ids.split(",") if s.strip()]
        if ids:
            config.setdefault("env_config", {})["scenario_ids"] = ids
    if args.start_seed is not None:
        config["start_seed"] = int(args.start_seed)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    total_steps = resolve_total_timesteps(args.total_env_steps, config)
    run_dir = run_training(config, Path(args.output_root), total_steps)
    print(f"[refine-grpo] outputs: {run_dir}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
