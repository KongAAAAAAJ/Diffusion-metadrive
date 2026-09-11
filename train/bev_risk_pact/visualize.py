"""Visualization helpers for ChassisFusion Risk-PACT pilot.

These utilities operate in the *current ego local frame* used by
``background_actor_state``:
    [x_local, y_local, sin(dheading), cos(dheading),
     dvx_local, dvy_local, length, width]

Typical usage from validation/debug code::

    from train.bev_risk_pact.visualize import save_risk_pact_debug_plots

    save_risk_pact_debug_plots(
        actor_state=batch["background_actor_state"],
        actor_valid_mask=batch["background_actor_valid_mask"],
        old_trajectory=old_candidates,
        teacher_trajectory=teacher.teacher_trajectory,
        output_dir=run_dir / "risk_debug",
        step=global_step,
        batch_index=0,
        role_index=0,
        mode_index=0,
    )

The function saves four figures:
  1) 2x4 spatio-temporal risk-field snapshots;
  2) overlaid R=R_th dynamic safe-tube boundaries;
  3) old-vs-teacher trajectory correction over risk contours;
  4) per-step risk curve along old and teacher trajectories.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import torch
from torch import Tensor

from .config import RiskPACTConfig
from .constraint import RiskLevelSetConstraint
from .risk_field import DynamicGaussianRiskField


def _to_cpu_float(x: Tensor) -> Tensor:
    return x.detach().to(device="cpu", dtype=torch.float32)


def _rectangle_corners(
    center_xy: Sequence[float], yaw_rad: float, length_m: float, width_m: float
) -> np.ndarray:
    """Return a 4x2 rectangle polygon in the ego-local x/y plane."""
    half_l = 0.5 * max(float(length_m), 0.1)
    half_w = 0.5 * max(float(width_m), 0.1)
    local = np.array(
        [[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]],
        dtype=np.float32,
    )
    c, s = math.cos(float(yaw_rad)), math.sin(float(yaw_rad))
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    center = np.asarray(center_xy, dtype=np.float32)
    return local @ rot.T + center[None, :]


def _default_plot_extent(actor_state_br: Tensor, valid_br: Tensor) -> tuple[tuple[float, float], tuple[float, float]]:
    """Choose a stable ego-local plotting window, enlarged if actors lie outside it."""
    valid_xy = actor_state_br[valid_br, :2]
    x_min, x_max = -10.0, 45.0
    y_min, y_max = -10.0, 10.0
    if valid_xy.numel() > 0:
        x_min = min(x_min, float(valid_xy[:, 0].min()) - 10.0)
        x_max = max(x_max, float(valid_xy[:, 0].max()) + 15.0)
        y_min = min(y_min, float(valid_xy[:, 1].min()) - 6.0)
        y_max = max(y_max, float(valid_xy[:, 1].max()) + 6.0)
    return (x_min, x_max), (y_min, y_max)


def evaluate_risk_grid_at_time(
    risk_field: DynamicGaussianRiskField,
    actor_state_br: Tensor,
    actor_valid_br: Tensor,
    *,
    time_index: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    nx: int = 260,
    ny: int = 140,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the pilot Gaussian union risk on a dense x/y grid.

    Parameters use one batch/role slice:
        actor_state_br: [A,8]
        actor_valid_br: [A]

    Returns X, Y, risk_grid, actor_future_xy_at_t (valid actors only).
    """
    if actor_state_br.ndim != 2 or actor_state_br.shape[-1] != 8:
        raise ValueError("actor_state_br must have shape [A,8]")
    if actor_valid_br.shape != actor_state_br.shape[:-1]:
        raise ValueError("actor_valid_br must have shape [A]")

    cfg = risk_field.config
    actor_state_br = _to_cpu_float(actor_state_br)
    actor_valid_br = actor_valid_br.detach().to(device="cpu", dtype=torch.bool)

    # Reuse the exact constant-relative-velocity forecast from the core field.
    future = risk_field.actor_future_xy(actor_state_br[None, None], time_index + 1)
    centers = future[0, 0, :, time_index, :]  # [A,2]

    xs = torch.linspace(float(xlim[0]), float(xlim[1]), int(nx))
    ys = torch.linspace(float(ylim[0]), float(ylim[1]), int(ny))
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack((xx, yy), dim=-1)  # [ny,nx,2]

    delta = points[..., None, :] - centers[None, None, :, :]  # [ny,nx,A,2]

    # Confirmed ChassisFusion token contract: index 2=sin(dheading), 3=cos(dheading).
    sin_yaw = actor_state_br[:, 2]
    cos_yaw = actor_state_br[:, 3]
    norm = torch.sqrt(sin_yaw.square() + cos_yaw.square()).clamp_min(1.0e-6)
    sin_yaw = sin_yaw / norm
    cos_yaw = cos_yaw / norm

    dx, dy = delta[..., 0], delta[..., 1]
    longitudinal = cos_yaw[None, None, :] * dx + sin_yaw[None, None, :] * dy
    lateral = -sin_yaw[None, None, :] * dx + cos_yaw[None, None, :] * dy

    length = actor_state_br[:, 6].abs().clamp_min(0.1)
    width = actor_state_br[:, 7].abs().clamp_min(0.1)
    sigma_x = torch.maximum(
        0.5 * length + float(cfg.longitudinal_margin_m),
        torch.full_like(length, float(cfg.minimum_sigma_x_m)),
    )
    sigma_y = torch.maximum(
        0.5 * width + float(cfg.lateral_margin_m),
        torch.full_like(width, float(cfg.minimum_sigma_y_m)),
    )

    mahalanobis = (
        (longitudinal / sigma_x[None, None, :]).square()
        + (lateral / sigma_y[None, None, :]).square()
    )
    per_actor = torch.exp(-0.5 * mahalanobis)
    per_actor = torch.where(
        actor_valid_br[None, None, :], per_actor, torch.zeros_like(per_actor)
    )
    aggregate = 1.0 - torch.prod((1.0 - per_actor).clamp(1.0e-6, 1.0), dim=-1)

    return (
        xx.numpy(),
        yy.numpy(),
        aggregate.clamp(0.0, 1.0).numpy(),
        centers[actor_valid_br].numpy(),
    )


def _draw_ego_and_actors(
    ax,
    actor_state_br: Tensor,
    actor_valid_br: Tensor,
    actor_centers: np.ndarray,
) -> None:
    # Ego is the origin of the local frame; use a generic footprint for visualization only.
    ego_poly = _rectangle_corners((0.0, 0.0), 0.0, 5.0, 2.0)
    ax.add_patch(Polygon(ego_poly, closed=True, fill=False, linewidth=1.8, label="ego"))

    states = _to_cpu_float(actor_state_br)
    valid_indices = torch.nonzero(actor_valid_br.detach().cpu(), as_tuple=False).flatten().tolist()
    for plot_idx, actor_idx in enumerate(valid_indices):
        st = states[actor_idx]
        yaw = math.atan2(float(st[2]), float(st[3]))
        poly = _rectangle_corners(
            actor_centers[plot_idx], yaw, float(st[6]), float(st[7])
        )
        ax.add_patch(Polygon(poly, closed=True, fill=False, linewidth=1.2))
        ax.text(
            float(actor_centers[plot_idx, 0]),
            float(actor_centers[plot_idx, 1]),
            f"A{actor_idx}",
            fontsize=7,
            ha="center",
            va="center",
        )


def plot_risk_field_snapshots(
    actor_state_br: Tensor,
    actor_valid_br: Tensor,
    *,
    risk_field: DynamicGaussianRiskField,
    output_path: str | Path,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    horizon_steps: int = 8,
) -> None:
    """Save a 2x4 panel of R_t(x,y) and R=R_th contours."""
    if xlim is None or ylim is None:
        auto_x, auto_y = _default_plot_extent(_to_cpu_float(actor_state_br), actor_valid_br.cpu())
        xlim = xlim or auto_x
        ylim = ylim or auto_y

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    axes = axes.ravel()
    mappable = None
    for t in range(horizon_steps):
        ax = axes[t]
        X, Y, R, centers = evaluate_risk_grid_at_time(
            risk_field,
            actor_state_br,
            actor_valid_br,
            time_index=t,
            xlim=xlim,
            ylim=ylim,
        )
        mappable = ax.contourf(X, Y, R, levels=np.linspace(0.0, 1.0, 31), cmap="magma")
        ax.contour(
            X,
            Y,
            R,
            levels=[float(risk_field.config.risk_threshold)],
            linewidths=1.8,
        )
        _draw_ego_and_actors(ax, actor_state_br, actor_valid_br, centers)
        ax.set_title(f"t={(t + 1) * risk_field.config.horizon_dt_s:.1f}s")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        ax.set_xlabel("x_local [m]")
        ax.set_ylabel("y_local [m]")
    if mappable is not None:
        fig.colorbar(mappable, ax=axes.tolist(), shrink=0.85, label="aggregate risk")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_dynamic_safe_tube(
    actor_state_br: Tensor,
    actor_valid_br: Tensor,
    *,
    risk_field: DynamicGaussianRiskField,
    output_path: str | Path,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    horizon_steps: int = 8,
) -> None:
    """Overlay R_t(x,y)=R_th contours for all future timesteps."""
    if xlim is None or ylim is None:
        auto_x, auto_y = _default_plot_extent(_to_cpu_float(actor_state_br), actor_valid_br.cpu())
        xlim = xlim or auto_x
        ylim = ylim or auto_y

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for t in range(horizon_steps):
        X, Y, R, centers = evaluate_risk_grid_at_time(
            risk_field,
            actor_state_br,
            actor_valid_br,
            time_index=t,
            xlim=xlim,
            ylim=ylim,
        )
        color = cmap(t / max(horizon_steps - 1, 1))
        ax.contour(
            X,
            Y,
            R,
            levels=[float(risk_field.config.risk_threshold)],
            colors=[color],
            linewidths=1.8,
        )
        # Mark predicted actor centers to make relative-velocity motion immediately visible.
        if centers.size:
            ax.scatter(centers[:, 0], centers[:, 1], s=12, color=[color])
        ax.plot([], [], color=color, label=f"{(t + 1) * risk_field.config.horizon_dt_s:.1f}s")

    ego_poly = _rectangle_corners((0.0, 0.0), 0.0, 5.0, 2.0)
    ax.add_patch(Polygon(ego_poly, closed=True, fill=False, linewidth=1.8))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x_local [m]")
    ax.set_ylabel("y_local [m]")
    ax.set_title(f"Dynamic safe-tube boundary: R = {risk_field.config.risk_threshold:.2f}")
    ax.grid(alpha=0.2)
    ax.legend(ncol=4, fontsize=8)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _select_traj(traj: Tensor | None, b: int, r: int, m: int) -> Tensor | None:
    if traj is None:
        return None
    traj = _to_cpu_float(traj)
    if traj.ndim == 5:  # [B,R,M,H,D]
        return traj[b, r, m]
    if traj.ndim == 4:  # [B,R,H,D]
        return traj[b, r]
    raise ValueError("trajectory must have shape [B,R,M,H,D] or [B,R,H,D]")


def plot_teacher_correction(
    actor_state_br: Tensor,
    actor_valid_br: Tensor,
    old_traj_h: Tensor,
    teacher_traj_h: Tensor,
    *,
    risk_field: DynamicGaussianRiskField,
    output_path: str | Path,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Overlay old/teacher trajectories on their time-matched risk boundaries."""
    old_traj_h = _to_cpu_float(old_traj_h)
    teacher_traj_h = _to_cpu_float(teacher_traj_h)
    horizon_steps = int(old_traj_h.shape[-2])
    if xlim is None or ylim is None:
        auto_x, auto_y = _default_plot_extent(_to_cpu_float(actor_state_br), actor_valid_br.cpu())
        all_xy = torch.cat((old_traj_h[..., :2], teacher_traj_h[..., :2]), dim=-2)
        xlim = xlim or (
            min(auto_x[0], float(all_xy[:, 0].min()) - 5.0),
            max(auto_x[1], float(all_xy[:, 0].max()) + 5.0),
        )
        ylim = ylim or (
            min(auto_y[0], float(all_xy[:, 1].min()) - 4.0),
            max(auto_y[1], float(all_xy[:, 1].max()) + 4.0),
        )

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for t in range(horizon_steps):
        X, Y, R, centers = evaluate_risk_grid_at_time(
            risk_field,
            actor_state_br,
            actor_valid_br,
            time_index=t,
            xlim=xlim,
            ylim=ylim,
        )
        color = cmap(t / max(horizon_steps - 1, 1))
        ax.contour(
            X,
            Y,
            R,
            levels=[float(risk_field.config.risk_threshold)],
            colors=[color],
            linewidths=0.9,
            alpha=0.75,
        )

    old_xy = old_traj_h[..., :2].numpy()
    teacher_xy = teacher_traj_h[..., :2].numpy()
    ax.plot(old_xy[:, 0], old_xy[:, 1], "o--", linewidth=2.0, label="old x0")
    ax.plot(teacher_xy[:, 0], teacher_xy[:, 1], "o-", linewidth=2.0, label="PACT teacher")
    # Draw point-wise correction vectors.
    for p0, p1 in zip(old_xy, teacher_xy):
        ax.annotate("", xy=p1, xytext=p0, arrowprops={"arrowstyle": "->", "alpha": 0.55})

    ego_poly = _rectangle_corners((0.0, 0.0), 0.0, 5.0, 2.0)
    ax.add_patch(Polygon(ego_poly, closed=True, fill=False, linewidth=1.8))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x_local [m]")
    ax.set_ylabel("y_local [m]")
    ax.set_title("Risk-PACT teacher correction")
    ax.grid(alpha=0.2)
    ax.legend()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_risk_curve(
    actor_state: Tensor,
    actor_valid_mask: Tensor,
    old_trajectory: Tensor,
    teacher_trajectory: Tensor,
    *,
    risk_field: DynamicGaussianRiskField,
    output_path: str | Path,
    batch_index: int,
    role_index: int,
    mode_index: int,
) -> None:
    """Plot time-matched risk sampled along old and teacher trajectories."""
    old = old_trajectory[..., :2]
    teacher = teacher_trajectory[..., :2]
    old_r = risk_field.query(old, actor_state, actor_valid_mask).risk
    teacher_r = risk_field.query(teacher, actor_state, actor_valid_mask).risk
    if old_r.ndim == 4:
        old_r = old_r[batch_index, role_index, mode_index]
        teacher_r = teacher_r[batch_index, role_index, mode_index]
    elif old_r.ndim == 3:
        old_r = old_r[batch_index, role_index]
        teacher_r = teacher_r[batch_index, role_index]
    else:
        raise ValueError(f"unexpected risk shape {tuple(old_r.shape)}")

    old_r = _to_cpu_float(old_r).numpy()
    teacher_r = _to_cpu_float(teacher_r).numpy()
    times = (np.arange(len(old_r)) + 1) * float(risk_field.config.horizon_dt_s)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(times, old_r, "o--", label="old x0")
    ax.plot(times, teacher_r, "o-", label="PACT teacher")
    ax.axhline(float(risk_field.config.risk_threshold), linestyle=":", label="R_th")
    ax.set_xlabel("future time [s]")
    ax.set_ylabel("risk")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("Per-step dynamic risk")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_risk_pact_debug_plots(
    *,
    actor_state: Tensor,
    actor_valid_mask: Tensor,
    old_trajectory: Tensor | None,
    teacher_trajectory: Tensor | None,
    output_dir: str | Path,
    step: int,
    batch_index: int = 0,
    role_index: int = 0,
    mode_index: int = 0,
    config: RiskPACTConfig | None = None,
) -> None:
    """One-call debug entry point intended for validation/smoke training."""
    cfg = config or RiskPACTConfig()
    field = DynamicGaussianRiskField(cfg)
    output_dir = Path(output_dir)
    prefix = output_dir / f"step_{int(step):06d}_b{batch_index}_r{role_index}_m{mode_index}"

    state_br = actor_state[batch_index, role_index]
    valid_br = actor_valid_mask[batch_index, role_index]
    xlim, ylim = _default_plot_extent(_to_cpu_float(state_br), valid_br.detach().cpu())

    plot_risk_field_snapshots(
        state_br,
        valid_br,
        risk_field=field,
        output_path=f"{prefix}_snapshots.png",
        xlim=xlim,
        ylim=ylim,
    )
    plot_dynamic_safe_tube(
        state_br,
        valid_br,
        risk_field=field,
        output_path=f"{prefix}_safe_tube.png",
        xlim=xlim,
        ylim=ylim,
    )

    old_h = _select_traj(old_trajectory, batch_index, role_index, mode_index)
    teacher_h = _select_traj(teacher_trajectory, batch_index, role_index, mode_index)
    if old_h is not None and teacher_h is not None:
        plot_teacher_correction(
            state_br,
            valid_br,
            old_h,
            teacher_h,
            risk_field=field,
            output_path=f"{prefix}_teacher.png",
            xlim=xlim,
            ylim=ylim,
        )
        plot_risk_curve(
            actor_state,
            actor_valid_mask,
            old_trajectory,
            teacher_trajectory,
            risk_field=field,
            output_path=f"{prefix}_risk_curve.png",
            batch_index=batch_index,
            role_index=role_index,
            mode_index=mode_index,
        )
