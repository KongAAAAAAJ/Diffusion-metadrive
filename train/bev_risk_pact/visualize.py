"""Step-6 Risk-PACT visual diagnostics for the calibrated multi-source field."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .config import RiskPACTConfig, RiskPACTVisualizationConfig
from .road_field import (
    BEV_HEIGHT,
    BEV_WIDTH,
    BEV_X_MAX_M,
    BEV_X_MIN_M,
    BEV_Y_MAX_M,
    BEV_Y_MIN_M,
)
from .teacher import PACTTeacherResult


def _cpu(value: Tensor) -> Tensor:
    return value.detach().float().cpu()


def _rectangle(center: tuple[float, float], heading: float, length: float, width: float) -> np.ndarray:
    cx, cy = center
    corners = np.asarray(
        [[length / 2, width / 2], [length / 2, -width / 2], [-length / 2, -width / 2], [-length / 2, width / 2]],
        dtype=np.float64,
    )
    c, s = math.cos(heading), math.sin(heading)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return corners @ rot.T + np.asarray([cx, cy], dtype=np.float64)


def _actor_heading(state: Tensor) -> float:
    return math.atan2(float(state[2]), float(state[3]))


def _future_center(state: Tensor, time_s: float) -> tuple[float, float]:
    return (
        float(state[0] + state[4] * time_s),
        float(state[1] + state[5] * time_s),
    )


def _select_candidate(
    old_trajectory: Tensor,
    teacher_result: PACTTeacherResult,
    valid_mode_mask: Tensor,
    cfg: RiskPACTVisualizationConfig,
) -> tuple[int, int, int, int, int]:
    if old_trajectory.ndim != 6:
        raise ValueError("old_trajectory must be [B,3,10,N,H,D]")
    bsz, roles, modes, groups = old_trajectory.shape[:4]
    b = min(int(cfg.batch_index), bsz - 1)
    r = min(int(cfg.role_index), roles - 1)
    valid = _cpu(valid_mode_mask[b, r]).bool()
    risk = _cpu(teacher_result.constraint.trajectory_risk).reshape(bsz, roles, modes, groups)[b, r]

    requested_mode = min(int(cfg.mode_index), modes - 1)
    if cfg.selection == "highest_risk" and bool(valid.any()):
        # Step-6 smoke diagnostics should expose the genuinely most critical
        # executable candidate, not merely the most critical sample in mode 0.
        valid_grid = valid.unsqueeze(-1).expand(modes, groups)
        masked = risk.masked_fill(~valid_grid, float("-inf"))
        flat_index = int(masked.reshape(-1).argmax())
        m, n = divmod(flat_index, groups)
    else:
        if bool(valid[requested_mode]):
            m = requested_mode
        elif bool(valid.any()):
            valid_scores = risk.mean(dim=-1).masked_fill(~valid, float("-inf"))
            m = int(valid_scores.argmax())
        else:
            m = requested_mode
        n = min(int(cfg.candidate_index), groups - 1)
    flat = m * groups + n
    return b, r, m, n, flat


def _plot_scene(
    *,
    output_path: Path,
    background_actor_state: Tensor,
    background_actor_valid_mask: Tensor,
    platoon_actor_state: Tensor,
    platoon_actor_valid_mask: Tensor,
    road_sdf: Tensor | None,
    old_traj: Tensor,
    teacher_traj: Tensor,
    critical_index: int,
    risk_before: float,
    risk_after: float,
    config: RiskPACTConfig,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    t_s = (critical_index + 1) * float(config.horizon_dt_s)
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)

    if road_sdf is not None:
        sdf = _cpu(road_sdf).numpy()
        x_values = np.linspace(BEV_X_MAX_M, BEV_X_MIN_M, BEV_HEIGHT)
        y_values = np.linspace(BEV_Y_MAX_M, BEV_Y_MIN_M, BEV_WIDTH)
        Y, X = np.meshgrid(y_values, x_values)
        required = 0.5 * float(config.ego_width_m) + float(config.road_safety_margin_m)
        ax.contour(Y, X, sdf, levels=[required], linestyles="--", linewidths=1.7)
        ax.contour(Y, X, sdf, levels=[0.0], linewidths=1.0)

    def draw_actors(state_br: Tensor, valid_br: Tensor, *, long_clear: float, lat_clear: float, label: str) -> None:
        state_br = _cpu(state_br)
        valid_br = valid_br.detach().cpu().bool()
        first = True
        for idx in range(state_br.shape[0]):
            if not bool(valid_br[idx]):
                continue
            state = state_br[idx]
            center = _future_center(state, t_s)
            heading = _actor_heading(state)
            length = max(float(abs(state[6])), 0.1)
            width = max(float(abs(state[7])), 0.1)
            physical = _rectangle(center, heading, length, width)
            safe_length = length + float(config.ego_length_m) + 2.0 * long_clear
            safe_width = width + float(config.ego_width_m) + 2.0 * lat_clear
            safety = _rectangle(center, heading, safe_length, safe_width)
            ax.add_patch(Polygon(physical, closed=True, fill=False, linewidth=1.1))
            ax.add_patch(Polygon(safety, closed=True, fill=False, linestyle="--", linewidth=0.8, alpha=0.55))
            if first:
                ax.plot([], [], linestyle="--", label=f"{label} safety box")
                first = False

    draw_actors(
        background_actor_state,
        background_actor_valid_mask,
        long_clear=float(config.background_longitudinal_clearance_m),
        lat_clear=float(config.background_lateral_clearance_m),
        label="background",
    )
    draw_actors(
        platoon_actor_state,
        platoon_actor_valid_mask,
        long_clear=float(config.platoon_longitudinal_clearance_m),
        lat_clear=float(config.platoon_lateral_clearance_m),
        label="platoon",
    )

    old_xy = _cpu(old_traj[..., :2]).numpy()
    teacher_xy = _cpu(teacher_traj[..., :2]).numpy()
    ax.plot(old_xy[:, 1], old_xy[:, 0], "o--", linewidth=2.0, label="old x0")
    ax.plot(teacher_xy[:, 1], teacher_xy[:, 0], "o-", linewidth=2.0, label="PACT teacher")
    ax.scatter([old_xy[critical_index, 1]], [old_xy[critical_index, 0]], s=70, marker="x", label="critical point")
    for p0, p1 in zip(old_xy, teacher_xy):
        ax.annotate("", xy=(p1[1], p1[0]), xytext=(p0[1], p0[0]), arrowprops={"arrowstyle": "->", "alpha": 0.5})

    ax.set_xlabel("y_local [m]")
    ax.set_ylabel("x_local [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    ax.set_title(
        f"Risk-PACT critical scene, t={t_s:.1f}s | trajectory risk {risk_before:.3f} -> {risk_after:.3f}"
    )
    ax.legend(fontsize=8, loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_risk_curve(
    *, output_path: Path, old_field, teacher_field, critical_index: int, config: RiskPACTConfig
) -> None:
    import matplotlib.pyplot as plt

    total_old = _cpu(old_field.risk).numpy()
    total_teacher = _cpu(teacher_field.risk).numpy()
    h = len(total_old)
    times = (np.arange(h) + 1) * float(config.horizon_dt_s)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.plot(times, total_old, "o--", linewidth=2.0, label="total old")
    ax.plot(times, total_teacher, "o-", linewidth=2.0, label="total teacher")
    for label, value in (
        ("background effective", old_field.background_risk),
        ("platoon effective", old_field.platoon_risk),
        ("road", old_field.road_risk),
    ):
        if value is not None:
            ax.plot(times, _cpu(value).numpy(), linestyle=":", label=label)
    for label, value in (
        ("background raw CRV", old_field.background_risk_raw),
        ("platoon raw CRV", old_field.platoon_risk_raw),
    ):
        if value is not None:
            ax.plot(times, _cpu(value).numpy(), linestyle="--", alpha=0.5, label=label)
    if old_field.actor_confidence is not None:
        ax.plot(
            times, _cpu(old_field.actor_confidence).numpy(),
            linestyle="-.", alpha=0.7, label="actor confidence",
        )
    ax.axhline(float(config.risk_threshold), linestyle="--", label="risk threshold")
    ax.axvline(times[critical_index], linestyle="--", alpha=0.6, label="critical timestep")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("future time [s]")
    ax.set_ylabel("risk")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("Per-step multi-source risk")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_clearance_curve(*, output_path: Path, field, config: RiskPACTConfig) -> None:
    import matplotlib.pyplot as plt

    h = int(field.risk.shape[-1])
    times = (np.arange(h) + 1) * float(config.horizon_dt_s)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    if field.background_signed_clearance_m is not None:
        value = _cpu(field.background_signed_clearance_m).amin(dim=-1).numpy()
        ax.plot(times, value, "o-", label="background min signed clearance")
    if field.platoon_signed_clearance_m is not None:
        value = _cpu(field.platoon_signed_clearance_m).amin(dim=-1).numpy()
        ax.plot(times, value, "o-", label="platoon min signed clearance")
    if field.road_signed_distance_m is not None:
        required = 0.5 * float(config.ego_width_m) + float(config.road_safety_margin_m)
        road_clearance = _cpu(field.road_signed_distance_m).numpy() - required
        ax.plot(times, road_clearance, "o-", label="road safety clearance")
    ax.axhline(0.0, linestyle="--", label="safety boundary")
    ax.set_xlabel("future time [s]")
    ax.set_ylabel("signed safety clearance [m]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("Physical safety clearances along old trajectory")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_risk_pact_debug_plots(
    *,
    background_actor_state: Tensor,
    background_actor_valid_mask: Tensor,
    platoon_actor_state: Tensor,
    platoon_actor_valid_mask: Tensor,
    road_sdf: Tensor | None,
    old_trajectory: Tensor,
    valid_executable_mode_mask: Tensor,
    teacher_result: PACTTeacherResult,
    output_dir: str | Path,
    step: int,
    visualization_config: RiskPACTVisualizationConfig,
    config: RiskPACTConfig,
) -> dict[str, object]:
    """Save one interpretable Step-6 diagnostic bundle."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    b, r, m, n, flat = _select_candidate(
        old_trajectory, teacher_result, valid_executable_mode_mask, visualization_config
    )
    teacher_field = teacher_result.teacher_field
    teacher_constraint = teacher_result.teacher_constraint
    if teacher_field is None or teacher_constraint is None:
        raise ValueError("Step-6 visualization requires teacher post-projection diagnostics")

    old_traj = old_trajectory[b, r, m, n]
    teacher_traj = teacher_result.teacher_trajectory[b, r, flat]
    old_field = type("SelectedField", (), {})()
    new_field = type("SelectedField", (), {})()
    for name in (
        "risk", "background_risk", "platoon_risk", "road_risk",
        "background_risk_raw", "platoon_risk_raw", "actor_confidence",
        "road_signed_distance_m", "background_signed_clearance_m", "platoon_signed_clearance_m",
    ):
        value = getattr(teacher_result.field, name)
        setattr(old_field, name, None if value is None else value[b, r, flat])
        value2 = getattr(teacher_field, name)
        setattr(new_field, name, None if value2 is None else value2[b, r, flat])

    critical = teacher_result.constraint.critical_timestep_index
    critical_index = int(
        teacher_result.field.risk[b, r, flat].argmax().item()
        if critical is None else critical[b, r, flat].item()
    )
    risk_before = float(teacher_result.constraint.trajectory_risk[b, r, flat].item())
    risk_after = float(teacher_constraint.trajectory_risk[b, r, flat].item())
    prefix = output_dir / f"step_{int(step):06d}_b{b}_r{r}_m{m}_n{n}"

    _plot_scene(
        output_path=Path(str(prefix) + "_scene.png"),
        background_actor_state=background_actor_state[b, r],
        background_actor_valid_mask=background_actor_valid_mask[b, r],
        platoon_actor_state=platoon_actor_state[b, r],
        platoon_actor_valid_mask=platoon_actor_valid_mask[b, r],
        road_sdf=None if road_sdf is None else road_sdf[b, r],
        old_traj=old_traj,
        teacher_traj=teacher_traj,
        critical_index=critical_index,
        risk_before=risk_before,
        risk_after=risk_after,
        config=config,
    )
    _plot_risk_curve(
        output_path=Path(str(prefix) + "_risk.png"),
        old_field=old_field,
        teacher_field=new_field,
        critical_index=critical_index,
        config=config,
    )
    _plot_clearance_curve(
        output_path=Path(str(prefix) + "_clearance.png"), field=old_field, config=config
    )

    summary = {
        "step": int(step), "batch": b, "role": r, "mode": m, "candidate": n,
        "critical_timestep_index": critical_index,
        "critical_time_s": (critical_index + 1) * float(config.horizon_dt_s),
        "trajectory_risk_before": risk_before,
        "trajectory_risk_after": risk_after,
        "trajectory_risk_reduction": risk_before - risk_after,
        "actor_confidence_last": (
            None if old_field.actor_confidence is None
            else float(old_field.actor_confidence[-1].item())
        ),
        "active": bool(teacher_result.constraint.near_or_unsafe_mask[b, r, flat].item()),
    }
    Path(str(prefix) + "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
