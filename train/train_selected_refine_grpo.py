"""Selected-mode local refinement GRPO.

This trainer keeps the existing mode-selection planner path intact, then
fine-tunes only the last trajectory generation submodule on small local
refinements around the selected mode trajectory.
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

from models.diffusion.ddim_with_logprob import DDIMSchedulerWithLogProb
from metadrive.policy.diffusion_policy.transfuser_policy import compute_trajectory_control
from train.train_plan_cls_grpo import _get_cls_branch, _get_vehicle_pose, compute_pairwise_formation_reward

DEFAULT_CONFIG_PATH = "configs/train/selected_refine_grpo.yaml"
DEFAULT_OUTPUT_ROOT = Path("/media/kong/Elements_SE/Diffusion_Data/outputs/selected_refine_grpo")
_RUN_DIR_RE = re.compile(r"^run_(\d+)$")


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
    """Freeze planner except the last trajectory generation submodule."""
    for param in planner.parameters():
        param.requires_grad_(False)

    decoder = _last_task_decoder(planner)
    trainable_modules = []
    has_gru_parts = all(hasattr(decoder, name) for name in ("hidden_init", "reg_gru_cell", "delta_head"))
    if getattr(decoder, "trajectory_reg_decoder_type", "gru" if has_gru_parts else "mlp") == "gru":
        for name in ("hidden_init", "reg_gru_cell", "delta_head"):
            module = getattr(decoder, name, None)
            if module is not None:
                trainable_modules.append((name, module))
    else:
        module = getattr(decoder, "plan_reg_branch", None)
        if module is not None:
            trainable_modules.append(("plan_reg_branch", module))

    trainable_names: list[str] = []
    for module_name, module in trainable_modules:
        for param_name, param in module.named_parameters():
            param.requires_grad_(True)
            trainable_names.append(f"{module_name}.{param_name}")
    return trainable_names


def _refine_state_dict(planner) -> dict[str, torch.Tensor]:
    decoder = _last_task_decoder(planner)
    names = ("hidden_init", "reg_gru_cell", "delta_head")
    if getattr(decoder, "trajectory_reg_decoder_type", "mlp") != "gru":
        names = ("plan_reg_branch",)
    state: dict[str, torch.Tensor] = {}
    for name in names:
        module = getattr(decoder, name, None)
        if module is None:
            continue
        for key, value in module.state_dict().items():
            state[f"{name}.{key}"] = value.detach().cpu()
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
    eps: float = 1e-6,
    positive_only: bool = False,
) -> torch.Tensor:
    rewards = rewards.float()
    if valid_mask is None:
        valid_mask = torch.ones_like(rewards, dtype=torch.bool)
    valid = valid_mask.bool()
    out = torch.zeros_like(rewards)
    if int(valid.sum().item()) < 2:
        return out
    vals = rewards[valid]
    out[valid] = (vals - vals.mean()) / (vals.std(unbiased=False) + float(eps))
    if positive_only:
        out = out.clamp(min=0.0)
    return out


def normalize_multimodal_advantages(
    rewards: torch.Tensor,      # [G*M]
    valid_mask: torch.Tensor | None,  # [G*M] bool
    num_groups: int,
    num_modes: int,
    positive_only: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-mode group-normalized advantage: for each mode, normalize across its G groups."""
    rewards = rewards.float()
    r2d = rewards.view(num_groups, num_modes)   # [G, M]
    v2d = (valid_mask.bool().view(num_groups, num_modes)
           if valid_mask is not None
           else torch.ones_like(r2d, dtype=torch.bool))
    out = torch.zeros_like(r2d)
    for m in range(num_modes):
        r_m, v_m = r2d[:, m], v2d[:, m]
        if int(v_m.sum().item()) < 2:
            continue
        vals = r_m[v_m]
        out[v_m, m] = (vals - vals.mean()) / (vals.std(unbiased=False) + eps)
    if positive_only:
        out = out.clamp(min=0.0)
    return out.view(num_groups * num_modes)     # [G*M]


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
    refined_xy: torch.Tensor,
    selected_xy: torch.Tensor,
    bc_weight: float = 0.1,
    kl_weight: float = 0.02,
    ref_mean_xy: torch.Tensor | None = None,
    positive_advantage_only: bool = True,
    no_positive_bc_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """DiffusionDriveV2-style GRPO loss for one agent's refinement groups."""
    adv = advantages.to(device=new_log_probs.device, dtype=new_log_probs.dtype).detach()
    if positive_advantage_only:
        adv = adv.clamp(min=0.0)
    logp_sum = new_log_probs.sum(dim=-1)
    ratio = torch.exp(logp_sum - logp_sum.detach())
    pg_loss = -(ratio * adv).mean()
    # BC loss: each candidate vs its own anchor (ref_mean_xy); fall back to selected_xy if not provided
    _bc_ref = ref_mean_xy.to(refined_xy.device) if ref_mean_xy is not None else selected_xy.to(refined_xy.device).unsqueeze(0).expand_as(refined_xy)
    bc_loss = F.l1_loss(refined_xy, _bc_ref)
    if ref_mean_xy is None:
        kl_loss = refined_xy.new_tensor(0.0)
    else:
        kl_loss = F.mse_loss(refined_xy, ref_mean_xy.to(refined_xy.device))
    active_advantage_count = int((adv > 0).sum().detach().item())
    effective_bc_weight = float(bc_weight) if active_advantage_count > 0 else float(no_positive_bc_weight)
    loss = pg_loss + effective_bc_weight * bc_loss + float(kl_weight) * kl_loss
    zero = new_log_probs.new_tensor(0.0)
    return loss, {
        "pg_loss": float(pg_loss.detach()),
        "bc_loss": float(bc_loss.detach()),
        "kl_loss": float(kl_loss.detach()),
        "ratio_mean": float(ratio.detach().mean()),
        "ratio_max": float(ratio.detach().max()),
        "clip_fraction": 0.0,
        "approx_kl": float(zero.detach()),
        "active_advantage_count": active_advantage_count,
        "effective_bc_weight": effective_bc_weight,
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
    selected_xy_phys = torch.as_tensor(selected_traj_np[:, :2], dtype=torch.float32, device=device)
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
) -> dict[str, torch.Tensor]:
    """Multi-modal rollout: G groups × M plan_anchors = G*M candidate trajectories.

    Aligned with DiffusionDriveV2 forward_train_rl:
      - all M plan_anchors used as base (instead of 1 selected trajectory)
      - DDIM truncated additive noise at t=noise_t
      - multiplicative noise applied in scheduler.step (DDIMSchedulerWithLogProb, eta>0)
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

    anchor = th.plan_anchor.to(device)                                       # [M, 8, 2] physical
    M = anchor.shape[0]

    # Normalize (x/50, y/20) — consistent with DiffusionDriveV2 norm_odo, xy-only
    anchor_norm = torch.stack([anchor[..., 0] / 50.0, anchor[..., 1] / 20.0], dim=-1)  # [M,8,2]

    # Expand G groups → [G*M, 1, 8, 2]
    base = (anchor_norm.unsqueeze(0)
            .expand(num_groups, -1, -1, -1)
            .reshape(num_groups * M, 8, 2)
            .unsqueeze(1))                                                   # [G*M,1,8,2]

    # DDIM truncated additive noise at t=noise_t: √ᾱ_t·x₀ + √(1-ᾱ_t)·ε
    noise = torch.randn_like(base)
    ts = torch.full((num_groups * M,), int(noise_t), dtype=torch.long, device=device)
    sample = scheduler.add_noise(original_samples=base, noise=noise, timesteps=ts).clamp(-1.0, 1.0)

    repeated_context = _repeat_context(context, num_groups * M)

    log_probs: list[torch.Tensor] = []
    chains: list[torch.Tensor] = [sample.detach()]
    timesteps_list: list[torch.Tensor] = []
    last_prev_sample_mean: torch.Tensor | None = None

    for timestep in _ddv2_roll_timesteps(denoise_steps, device):
        batch_ts = timestep.expand(num_groups * M)
        model_output = planner.predict_denoised_traj(sample, batch_ts, repeated_context)
        next_sample, log_prob, prev_mean = scheduler.step(
            model_output=model_output, timestep=timestep, sample=sample, eta=float(eta)
        )
        last_prev_sample_mean = prev_mean
        log_probs.append(log_prob.squeeze(-1))   # [G*M]
        sample = next_sample.detach()
        chains.append(sample)
        timesteps_list.append(timestep)

    sampled_xy = _denorm_xy(planner, sample.squeeze(1)).detach()   # [G*M,T,2]
    _mean_src = last_prev_sample_mean if last_prev_sample_mean is not None else sample
    refined_xy = _denorm_xy(planner, _mean_src.squeeze(1))         # [G*M,T,2]

    # BC reference: each candidate vs its own anchor (physical space)
    anchor_ref = (anchor.unsqueeze(0)
                  .expand(num_groups, -1, -1, -1)
                  .reshape(num_groups * M, 8, 2))                   # [G*M,T,2]

    return {
        "sampled_xy":   sampled_xy,
        "refined_xy":   refined_xy,
        "refined_traj": reconstruct_heading_from_xy(sampled_xy),    # [G*M,T,3]
        "log_probs":    torch.stack(log_probs, dim=-1),             # [G*M,steps]
        "chains_norm":  torch.stack(chains, dim=1).detach(),        # [G*M,steps+1,1,T,2]
        "timesteps":    torch.stack(timesteps_list).detach(),       # [steps]
        "ref_mean_xy":  anchor_ref,                                 # [G*M,T,2]
        "num_modes":    M,
        "num_groups":   num_groups,
    }


def recompute_refine_log_probs(
    planner,
    context: dict[str, Any],
    chains_norm: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler: DDIMSchedulerWithLogProb,
    eta: float,
) -> torch.Tensor:
    """Recompute log-probs on a fixed refinement chain without resampling."""
    device = planner._device()
    chains_norm = chains_norm.to(device=device)
    timesteps = timesteps.to(device=device)
    groups = int(chains_norm.shape[0])
    repeated_context = _repeat_context(context, groups)
    log_probs = []
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
        )
        log_probs.append(log_prob.squeeze(-1))
    return torch.stack(log_probs, dim=-1)


def _trajectory_delta_valid(refined_xy: torch.Tensor, selected_xy: torch.Tensor, max_delta_m: float) -> torch.Tensor:
    # Same shape: multi-modal [G*M,T,2] vs [G*M,T,2]; different shape: single [G,T,2] vs [T,2]
    ref = selected_xy if refined_xy.shape == selected_xy.shape else selected_xy.unsqueeze(0)
    delta = torch.norm(refined_xy - ref, dim=-1).amax(dim=-1)
    finite = torch.isfinite(refined_xy).all(dim=(-1, -2))
    return finite & (delta <= float(max_delta_m))


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
    from envs.mode_selection_sb3_env import ModeSelectionSB3Env

    run_dir = create_next_run_dir(output_root)
    env_config = dict(config.get("env_config", {}))
    env_config.setdefault("num_agents", int(config.get("num_agents", 3)))
    env_config.setdefault("observation_mode", "multimodal")
    env_config.setdefault("use_render", False)
    env_config.setdefault("planner_device", "cuda")
    env_config.setdefault("lookahead_index", int(config.get("lookahead_index", 2)))
    env_config.setdefault("target_speed_km_h", float(config.get("target_speed_km_h", 30.0)))
    env_config.setdefault("controller_type", str(config.get("controller_type", "stabilized")))
    env_config["use_action_mask"] = True
    env_config["trajectory_source"] = "diffusion"

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
    noise_t = int(config.get("refine_noise_t", 8))
    denoise_steps = int(config.get("refine_denoise_steps", 4))
    eta = float(config.get("refine_eta", 0.1))
    noise_std = float(config.get("refine_noise_std", 0.02))
    max_delta_norm = float(config.get("refine_max_delta_norm", 0.05))
    max_delta_m = float(config.get("refine_max_delta_m", 1.0))
    bc_weight = float(config.get("refine_bc_weight", 0.1))
    kl_weight = float(config.get("refine_kl_weight", 0.02))
    positive_advantage_only = bool(config.get("refine_positive_advantage_only", True))
    no_positive_bc_weight = float(config.get("refine_no_positive_bc_weight", 1.0))
    desired_gap_m = float(config.get("desired_gap_m", 10.0))
    progress_s_max = float(config.get("progress_s_max", 15.0))
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
            logits = np.asarray(export["masked_cls_logits"], dtype=np.float32)
            selected_modes = np.argmax(logits, axis=-1).astype(np.int64)
            contexts, context_agent_ids = planner.extract_rl_context(planner_batch)
            if list(context_agent_ids) != agent_ids:
                raise RuntimeError(f"context/export agent mismatch: {context_agent_ids} != {agent_ids}")
            poses = {aid: _get_vehicle_pose(env, aid) for aid in agent_ids}

            step_loss = None
            executed: dict[str, np.ndarray] = {}
            debug_rows = []
            for idx, agent_id in enumerate(agent_ids):
                mode = int(selected_modes[idx])
                if mode < 0 or mode >= masks.shape[1] or not bool(masks[idx, mode]):
                    mode = int(np.argmax(masks[idx]))
                selected_traj = candidates[idx, mode]
                if idx == 0:
                    executed[agent_id] = selected_traj
                    debug_rows.append({
                        "agent_id": agent_id,
                        "selected_mode": mode,
                        "group_rewards": [],
                        "advantages": [],
                        "chosen_group": -1,
                        "valid_groups": [],
                        "fallback": "leader_uses_selected_trajectory",
                            "metrics": {
                                "pg_loss": 0.0,
                                "bc_loss": 0.0,
                                "kl_loss": 0.0,
                                "ratio_mean": 1.0,
                                "ratio_max": 1.0,
                                "clip_fraction": 0.0,
                                "approx_kl": 0.0,
                                "active_advantage_count": 0,
                                "effective_bc_weight": 0.0,
                                "total_loss": 0.0,
                            },
                        })
                    continue
                # ── multi-modal rollout: G groups × M plan_anchors ────────────────
                rollout = rollout_multimodal_refinement(
                    planner,
                    contexts[agent_id],
                    num_groups=num_groups,
                    noise_t=noise_t,
                    denoise_steps=denoise_steps,
                    eta=eta,
                    scheduler=refine_scheduler,
                )
                M   = rollout["num_modes"]
                GxM = num_groups * M

                valid_groups = _trajectory_delta_valid(
                    rollout["sampled_xy"], rollout["ref_mean_xy"], max_delta_m=max_delta_m
                )  # [G*M]
                refined_np = rollout["refined_traj"].detach().cpu().numpy().astype(np.float32)  # [G*M,T,3]
                rewards = np.zeros(GxM, dtype=np.float32)
                if poses.get(agent_id) is None or poses.get(agent_ids[idx - 1]) is None:
                    rewards[:] = 0.0
                else:
                    prev_agent = agent_ids[idx - 1]
                    prev_traj = executed.get(prev_agent, candidates[idx - 1, int(selected_modes[idx - 1])])
                    rewards = compute_pairwise_formation_reward(
                        refined_np,
                        poses[agent_id],
                        prev_traj,
                        poses[prev_agent],
                        desired_gap_m=desired_gap_m,
                        progress_s_max=progress_s_max,
                    )  # [G*M]

                # per-mode group-normalized advantage [G*M]
                reward_t = torch.as_tensor(rewards, dtype=torch.float32, device=planner._device())
                advantages = normalize_multimodal_advantages(
                    reward_t, valid_groups, num_groups, M,
                    positive_only=positive_advantage_only,
                )  # [G*M]

                # ── extract selected mode's G slice for loss update ───────────────
                adv_sel     = advantages.view(num_groups, M)[:, mode]            # [G]
                chains_sel  = rollout["chains_norm"].view(num_groups, M, *rollout["chains_norm"].shape[1:])[:, mode]
                # chains_sel: [G, steps+1, 1, T, 2]
                valid_sel   = valid_groups.view(num_groups, M)[:, mode]          # [G]
                refined_sel = rollout["refined_xy"].view(num_groups, M, -1, 2)[:, mode]    # [G,T,2]
                ref_sel     = rollout["ref_mean_xy"].view(num_groups, M, -1, 2)[:, mode]   # [G,T,2]

                new_logp_sum_debug: list[float] = []
                if int(valid_sel.sum().item()) >= 2:
                    new_log_probs = recompute_refine_log_probs(
                        planner,
                        contexts[agent_id],
                        chains_norm=chains_sel,          # [G, steps+1, 1, T, 2]
                        timesteps=rollout["timesteps"],
                        scheduler=refine_scheduler,
                        eta=eta,
                    )  # [G, denoise_steps]
                    new_logp_sum_debug = (
                        new_log_probs.sum(dim=-1).detach().cpu().numpy().astype(float).tolist()
                    )
                    loss_i, metrics_i = ddv2_refine_grpo_loss(
                        new_log_probs=new_log_probs,
                        advantages=adv_sel,
                        refined_xy=refined_sel,
                        selected_xy=ref_sel[0],          # API compat placeholder
                        bc_weight=bc_weight,
                        kl_weight=kl_weight,
                        ref_mean_xy=ref_sel,
                        positive_advantage_only=positive_advantage_only,
                        no_positive_bc_weight=no_positive_bc_weight,
                    )
                    step_loss = loss_i if step_loss is None else step_loss + loss_i
                else:
                    metrics_i = {
                        "pg_loss": 0.0,
                        "bc_loss": 0.0,
                        "kl_loss": 0.0,
                        "ratio_mean": 1.0,
                        "ratio_max": 1.0,
                        "clip_fraction": 0.0,
                        "approx_kl": 0.0,
                        "active_advantage_count": 0,
                        "effective_bc_weight": 0.0,
                        "total_loss": 0.0,
                    }

                # ── execute best from selected mode's G groups ───────────────────
                rewards_sel  = rewards.reshape(num_groups, M)[:, mode]   # [G]
                valid_sel_np = valid_sel.detach().cpu().numpy()           # [G]
                chosen_g     = int(np.argmax(np.where(valid_sel_np, rewards_sel, -np.inf)))
                chosen_gm    = chosen_g * M + mode
                if valid_sel_np[chosen_g]:
                    executed[agent_id] = refined_np[chosen_gm]
                    fallback = ""
                else:
                    executed[agent_id] = selected_traj
                    fallback = "invalid_refined_groups"
                debug_rows.append({
                    "agent_id": agent_id,
                    "selected_mode": mode,
                    "group_rewards": rewards_sel.tolist(),
                    "advantages": adv_sel.detach().cpu().numpy().astype(float).tolist(),
                    "chosen_group": chosen_g,
                    "valid_groups": valid_sel_np.astype(bool).tolist(),
                    "fallback": fallback,
                    "metrics": metrics_i,
                    "rollout_logp_sum": rollout["log_probs"].view(num_groups, M, -1)[:, mode].sum(dim=-1).detach().cpu().numpy().astype(float).tolist(),
                    "new_logp_sum": new_logp_sum_debug,
                })

            if step_loss is not None:
                step_loss = step_loss / max(1, len(agent_ids) - 1)
                optimizer.zero_grad()
                step_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, float(config.get("max_grad_norm", 1.0)))
                optimizer.step()

            _, env_reward, done, info = _execute_trajectories(env, planner, agent_ids, executed, env_config)
            episode_reward += float(env_reward)
            episode_step += 1
            global_step += 1
            if writer is not None:
                writer.add_scalar("train/env_reward_step", float(env_reward), global_step)
                if step_loss is not None:
                    writer.add_scalar("train/refine_loss", float(step_loss.detach()), global_step)
            with debug_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "global_step": global_step,
                    "episode": episode_idx,
                    "step": episode_step,
                    "env_reward": float(env_reward),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Selected-mode local refinement GRPO training.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--pretrained-ckpt", default="")
    parser.add_argument("--cls-grpo-ckpt-dir", default="")
    parser.add_argument("--cls-grpo-full-ckpt", default="")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--total-env-steps", type=int, default=50000)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--planner-device", default="cuda")
    parser.add_argument("--scenario-ids", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.pretrained_ckpt:
        config["pretrained_ckpt"] = args.pretrained_ckpt
    if args.cls_grpo_ckpt_dir:
        config["cls_grpo_ckpt_dir"] = args.cls_grpo_ckpt_dir
    if args.cls_grpo_full_ckpt:
        config["cls_grpo_full_ckpt"] = args.cls_grpo_full_ckpt
    if args.num_agents > 0:
        config["num_agents"] = int(args.num_agents)
        config.setdefault("env_config", {})["num_agents"] = int(args.num_agents)
    if args.planner_device:
        config.setdefault("env_config", {})["planner_device"] = args.planner_device
    if args.scenario_ids:
        ids = [s.strip() for s in args.scenario_ids.split(",") if s.strip()]
        if ids:
            config.setdefault("env_config", {})["scenario_ids"] = ids
    total_steps = int(args.total_env_steps or config.get("total_timesteps", 50000))
    run_dir = run_training(config, Path(args.output_root), total_steps)
    print(f"[refine-grpo] outputs: {run_dir}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
