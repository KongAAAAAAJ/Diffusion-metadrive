"""Metric serialization and plotting helpers for preview evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _aggregate_pdms_records(pdms_records: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """Average per-agent PDMS components over an episode, plus a `__team__` cross-agent mean."""
    per_agent: dict[str, list[dict[str, float]]] = {}
    for step in pdms_records:
        for agent_id, vals in step.items():
            per_agent.setdefault(agent_id, []).append(vals)

    result: dict[str, dict[str, float]] = {}
    for agent_id, steps in per_agent.items():
        keys = steps[0].keys()
        result[agent_id] = {k: float(np.mean([s[k] for s in steps])) for k in keys}

    if result:
        agent_rows = list(result.values())
        keys = agent_rows[0].keys()
        result["__team__"] = {k: float(np.mean([row[k] for row in agent_rows])) for k in keys}

    return result


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _action_to_record(action) -> dict[str, float | None]:
    arr = np.asarray(action if action is not None else [np.nan, np.nan], dtype=np.float32).reshape(-1)
    steer = float(arr[0]) if arr.size > 0 else float("nan")
    throttle = float(arr[1]) if arr.size > 1 else float("nan")
    return _json_safe({"steer": steer, "throttle": throttle})


def _vehicle_lane_s(vehicle) -> float | None:
    lane = getattr(vehicle, "lane", None)
    if lane is None or not hasattr(lane, "local_coordinates"):
        return None
    try:
        longitudinal = lane.local_coordinates(getattr(vehicle, "position"))[0]
    except Exception:
        return None
    value = float(longitudinal)
    return value if np.isfinite(value) else None


def _vehicle_velocity_mps(vehicle) -> np.ndarray | None:
    velocity = getattr(vehicle, "velocity", None)
    if velocity is None:
        return None
    try:
        arr = np.asarray(velocity, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size < 2:
        return None
    xy = arr[:2].astype(np.float32, copy=True)
    return xy if np.all(np.isfinite(xy)) else None


def _collect_episode_step_record(
    *,
    env,
    agent_ids: list[str],
    step_idx: int,
    actions: dict[str, np.ndarray],
    info: dict,
    pdms: dict[str, dict[str, float]],
    planning_debug: dict | None,
    previous_speed_mps: dict[str, np.ndarray],
    dt: float,
) -> dict:
    vehicles: dict[str, dict] = {}
    actions_record: dict[str, dict] = {}
    agents = getattr(env, "agents", {}) or {}
    for agent_id in agent_ids:
        action = actions.get(agent_id) if isinstance(actions, dict) else None
        action_record = _action_to_record(action)
        actions_record[agent_id] = action_record

        vehicle = agents.get(agent_id)
        if vehicle is None:
            vehicles[agent_id] = {
                "x": None,
                "y": None,
                "speed_km_h": None,
                "accel_mps2": None,
                "accel_x_mps2": None,
                "accel_y_mps2": None,
                "accel_norm_mps2": None,
                "heading_theta": None,
                "lane_s": None,
                "steer": action_record["steer"],
                "throttle": action_record["throttle"],
            }
            continue

        position = np.asarray(getattr(vehicle, "position", [np.nan, np.nan])[:2], dtype=np.float32)
        speed_km_h = float(getattr(vehicle, "speed_km_h", np.nan))
        heading_theta = float(getattr(vehicle, "heading_theta", np.nan))
        velocity_mps = _vehicle_velocity_mps(vehicle)
        prev_velocity_mps = previous_speed_mps.get(agent_id)
        accel_vec = None
        if velocity_mps is not None and prev_velocity_mps is not None and np.isfinite(dt) and dt > 0.0:
            prev_velocity = np.asarray(prev_velocity_mps, dtype=np.float32).reshape(-1)[:2]
            if prev_velocity.size == 2 and np.all(np.isfinite(prev_velocity)):
                accel_vec = (velocity_mps - prev_velocity) / float(dt)
        if velocity_mps is not None:
            previous_speed_mps[agent_id] = velocity_mps

        if accel_vec is not None and np.isfinite(heading_theta):
            heading_unit = np.asarray([np.cos(heading_theta), np.sin(heading_theta)], dtype=np.float32)
            accel_mps2 = float(np.dot(accel_vec, heading_unit))
            accel_x_mps2 = float(accel_vec[0])
            accel_y_mps2 = float(accel_vec[1])
            accel_norm_mps2 = float(np.linalg.norm(accel_vec))
        else:
            accel_mps2 = float("nan")
            accel_x_mps2 = float("nan")
            accel_y_mps2 = float("nan")
            accel_norm_mps2 = float("nan")

        vehicles[agent_id] = _json_safe(
            {
                "x": float(position[0]) if position.size > 0 else float("nan"),
                "y": float(position[1]) if position.size > 1 else float("nan"),
                "speed_km_h": speed_km_h,
                "accel_mps2": accel_mps2,
                "accel_x_mps2": accel_x_mps2,
                "accel_y_mps2": accel_y_mps2,
                "accel_norm_mps2": accel_norm_mps2,
                "heading_theta": heading_theta,
                "lane_s": _vehicle_lane_s(vehicle),
                "steer": action_record["steer"],
                "throttle": action_record["throttle"],
            }
        )

    planning_debug = planning_debug or {}
    planning_record = {
        "planning_policy": planning_debug.get("planning_policy"),
        "agent_ids": planning_debug.get("agent_ids", list(agent_ids)),
        "trajectories_by_agent": planning_debug.get("trajectories_by_agent", {}),
        "candidates_by_agent": planning_debug.get("candidates_by_agent", {}),
    }
    control_debug = getattr(env, "_preview_control_debug", None) or {}
    return _json_safe(
        {
            "step_idx": step_idx,
            "actions": actions_record,
            "info": info or {},
            "pdms": pdms or {},
            "planning": planning_record,
            "control_debug": control_debug,
            "vehicles": vehicles,
        }
    )


def _save_episode_metrics_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")


def _write_plot_placeholder(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff"
        b"\xff?\x00\x05\xfe\x02\xfeA\xbd\xb1\x0f\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _get_pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


_PUBLICATION_AGENT_COLORS = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)
_PUBLICATION_TEAM_COLOR = "#333333"
_PUBLICATION_RC = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
_PDMS_LABELS = {
    "reward": "Reward",
    "progress": "Progress",
    "formation_lon": "Formation longitudinal",
    "formation_lat": "Formation lateral",
    "speed": "Speed",
    "comfort": "Comfort",
    "consistency": "Consistency",
    "gate": "Gate",
}
_PDMS_ORDER = (
    "reward",
    "progress",
    "formation_lon",
    "formation_lat",
    "speed",
    "comfort",
    "consistency",
    "gate",
)


def _publication_agent_color(agent_id: str) -> str:
    if agent_id == "__team__":
        return _PUBLICATION_TEAM_COLOR
    digits = "".join(ch for ch in str(agent_id) if ch.isdigit())
    index = int(digits) if digits else sum(ord(ch) for ch in str(agent_id))
    return _PUBLICATION_AGENT_COLORS[index % len(_PUBLICATION_AGENT_COLORS)]


def _save_publication_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")


def _style_publication_axis(ax, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(direction="out", length=3, width=0.8, pad=2)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color="#D0D0D0", alpha=0.45, linewidth=0.5)


def _publication_legend(ax, *, outside: bool = False) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if outside:
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
            ncol=min(max(len(labels), 1), 4),
            frameon=False,
            handlelength=1.8,
            columnspacing=1.0,
        )
    else:
        ax.legend(handles, labels, loc="best", frameon=False, handlelength=1.8)


def _ordered_pdms_keys(pdms_keys: set[str]) -> list[str]:
    ordered = [key for key in _PDMS_ORDER if key in pdms_keys]
    ordered.extend(sorted(key for key in pdms_keys if key not in _PDMS_ORDER))
    return ordered


def _pdms_team_mean(step_pdms: dict, key: str) -> float:
    values = [
        float(vals[key])
        for agent_id, vals in (step_pdms or {}).items()
        if agent_id != "__team__"
        and isinstance(vals, dict)
        and key in vals
        and vals[key] is not None
        and np.isfinite(float(vals[key]))
    ]
    return float(np.mean(values)) if values else float("nan")


def _plot_episode_pdms(path: Path, step_records: list[dict]) -> None:
    plt = _get_pyplot()
    if plt is None or not step_records:
        _write_plot_placeholder(path)
        return

    pdms_keys = sorted(
        {
            key
            for record in step_records
            for vals in (record.get("pdms", {}) or {}).values()
            if isinstance(vals, dict)
            for key in vals.keys()
        }
    )
    if not pdms_keys:
        _write_plot_placeholder(path)
        return

    times = [record.get("step_idx", idx) for idx, record in enumerate(step_records)]
    pdms_keys = _ordered_pdms_keys(pdms_keys)
    ncols = 2 if len(pdms_keys) > 1 else 1
    nrows = int(np.ceil(len(pdms_keys) / ncols))
    with plt.rc_context(_PUBLICATION_RC):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(7.2, max(2.4, 1.85 * nrows)),
            sharex=True,
            squeeze=False,
        )
        flat_axes = list(axes.reshape(-1))
        for ax, key in zip(flat_axes, pdms_keys):
            agent_ids = sorted(
                {
                    agent_id
                    for record in step_records
                    for agent_id, vals in (record.get("pdms", {}) or {}).items()
                    if agent_id != "__team__" and isinstance(vals, dict) and key in vals
                }
            )
            for agent_id in agent_ids:
                values = [
                    (record.get("pdms", {}) or {}).get(agent_id, {}).get(key, np.nan)
                    for record in step_records
                ]
                ax.plot(
                    times,
                    values,
                    color=_publication_agent_color(agent_id),
                    linewidth=1.4,
                    label=agent_id,
                )
            team_values = [
                (record.get("pdms", {}) or {}).get("__team__", {}).get(key, _pdms_team_mean(record.get("pdms", {}), key))
                for record in step_records
            ]
            if np.any(np.isfinite(np.asarray(team_values, dtype=np.float32))):
                ax.plot(
                    times,
                    team_values,
                    color=_publication_agent_color("__team__"),
                    linewidth=1.3,
                    linestyle=(0, (3, 2)),
                    label="team mean",
                )
            ax.set_ylabel(_PDMS_LABELS.get(key, key.replace("_", " ").title()))
            _style_publication_axis(ax)
        for ax in flat_axes[len(pdms_keys):]:
            ax.set_visible(False)
        for ax in flat_axes[-ncols:]:
            if ax.get_visible():
                ax.set_xlabel("Step")
        _publication_legend(flat_axes[0], outside=True)
        fig.tight_layout(pad=0.7)
        _save_publication_figure(fig, path)
        plt.close(fig)


def _extract_team_pdms_series(episode_step_records: list[dict], key: str) -> np.ndarray:
    values: list[float] = []
    for record in episode_step_records or []:
        step_pdms = record.get("pdms", {}) or {}
        team_vals = step_pdms.get("__team__", {}) if isinstance(step_pdms, dict) else {}
        if isinstance(team_vals, dict) and key in team_vals and team_vals[key] is not None:
            values.append(float(team_vals[key]))
            continue
        values.append(_pdms_team_mean(step_pdms, key))
    return np.asarray(values, dtype=np.float32)


def _aggregate_pdms_across_episodes(all_episode_step_records: list[list[dict]]) -> dict:
    episodes = [episode for episode in (all_episode_step_records or []) if episode]
    pdms_keys = {
        key
        for episode in episodes
        for record in episode
        for vals in (record.get("pdms", {}) or {}).values()
        if isinstance(vals, dict)
        for key in vals.keys()
    }
    pdms_keys = _ordered_pdms_keys(pdms_keys)
    max_len = max((len(episode) for episode in episodes), default=0)
    metrics: dict[str, dict[str, np.ndarray]] = {}
    if max_len <= 0 or not pdms_keys:
        return {"steps": [], "metrics": metrics}

    for key in pdms_keys:
        stacked = np.full((len(episodes), max_len), np.nan, dtype=np.float32)
        for ep_idx, episode in enumerate(episodes):
            series = _extract_team_pdms_series(episode, key)
            stacked[ep_idx, : min(max_len, series.size)] = series[:max_len]
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
        metrics[key] = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    return {"steps": list(range(max_len)), "metrics": metrics}


def _plot_average_episode_pdms(path: Path, all_episode_step_records: list[list[dict]]) -> None:
    plt = _get_pyplot()
    aggregated = _aggregate_pdms_across_episodes(all_episode_step_records)
    metrics = aggregated.get("metrics", {})
    steps = aggregated.get("steps", [])
    if plt is None or not steps or not metrics:
        _write_plot_placeholder(path)
        return

    metric_keys = _ordered_pdms_keys(set(metrics.keys()))
    ncols = 2 if len(metric_keys) > 1 else 1
    nrows = int(np.ceil(len(metric_keys) / ncols))
    x = np.asarray(steps, dtype=np.float32)
    with plt.rc_context(_PUBLICATION_RC):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(7.2, max(2.4, 1.85 * nrows)),
            sharex=True,
            squeeze=False,
        )
        flat_axes = list(axes.reshape(-1))
        for ax, key in zip(flat_axes, metric_keys):
            mean = np.asarray(metrics[key]["mean"], dtype=np.float32)
            std = np.asarray(metrics[key]["std"], dtype=np.float32)
            lower = mean - std
            upper = mean + std
            ax.fill_between(
                x,
                lower,
                upper,
                color=_PUBLICATION_TEAM_COLOR,
                alpha=0.16,
                linewidth=0.0,
                label="Mean ± SD",
            )
            ax.plot(
                x,
                mean,
                color=_PUBLICATION_TEAM_COLOR,
                linewidth=1.6,
                label="Mean",
            )
            ax.set_ylabel(_PDMS_LABELS.get(key, key.replace("_", " ").title()))
            _style_publication_axis(ax)
        for ax in flat_axes[len(metric_keys):]:
            ax.set_visible(False)
        for ax in flat_axes[-ncols:]:
            if ax.get_visible():
                ax.set_xlabel("Step")
        _publication_legend(flat_axes[0], outside=True)
        fig.tight_layout(pad=0.7)
        _save_publication_figure(fig, path)
        plt.close(fig)


def _plot_time_series(path: Path, step_records: list[dict], agent_ids: list[str], key: str, ylabel: str) -> None:
    plt = _get_pyplot()
    if plt is None or not step_records:
        _write_plot_placeholder(path)
        return
    times = [record.get("step_idx", idx) for idx, record in enumerate(step_records)]
    with plt.rc_context(_PUBLICATION_RC):
        fig, ax = plt.subplots(figsize=(5.2, 2.8))
        agent_ids = sorted(
            {
                agent_id for agent_id in agent_ids
                if any(agent_id in (record.get("vehicles", {}) or {}) for record in step_records)
            }
        )
        for agent_id in agent_ids:
            values = [
                (record.get("vehicles", {}) or {}).get(agent_id, {}).get(key, np.nan)
                for record in step_records
            ]
            ax.plot(times, values, color=_publication_agent_color(agent_id), linewidth=1.5, label=agent_id)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        _style_publication_axis(ax)
        _publication_legend(ax)
        fig.tight_layout(pad=0.7)
        _save_publication_figure(fig, path)
        plt.close(fig)


def _as_finite_float_or_nan(value) -> float:
    if value is None:
        return float("nan")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _following_distance_series(step_records: list[dict], agent_ids: list[str]) -> list[tuple[str, np.ndarray]]:
    pairs: list[tuple[str, np.ndarray]] = []
    for front_id, rear_id in zip(agent_ids[:-1], agent_ids[1:]):
        values: list[float] = []
        for record in step_records:
            vehicles = record.get("vehicles", {}) or {}
            front_s = _as_finite_float_or_nan((vehicles.get(front_id, {}) or {}).get("lane_s"))
            rear_s = _as_finite_float_or_nan((vehicles.get(rear_id, {}) or {}).get("lane_s"))
            values.append(front_s - rear_s if np.isfinite(front_s) and np.isfinite(rear_s) else float("nan"))
        pairs.append((f"{front_id}-{rear_id}", np.asarray(values, dtype=np.float32)))
    return pairs


def _plot_following_distance(path: Path, step_records: list[dict], agent_ids: list[str]) -> None:
    plt = _get_pyplot()
    pairs = _following_distance_series(step_records, agent_ids)
    if plt is None or not step_records or not pairs:
        _write_plot_placeholder(path)
        return
    times = [record.get("step_idx", idx) for idx, record in enumerate(step_records)]
    with plt.rc_context(_PUBLICATION_RC):
        fig, ax = plt.subplots(figsize=(5.2, 2.8))
        for pair_idx, (label, values) in enumerate(pairs):
            color_agent = agent_ids[pair_idx] if pair_idx < len(agent_ids) else label
            ax.plot(times, values, color=_publication_agent_color(color_agent), linewidth=1.5, label=label)
        ax.set_xlabel("Step")
        ax.set_ylabel("Following distance s (m)")
        _style_publication_axis(ax)
        _publication_legend(ax)
        fig.tight_layout(pad=0.7)
        _save_publication_figure(fig, path)
        plt.close(fig)


def _plot_episode_results(results_dir: Path, step_records: list[dict], agent_ids: list[str]) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    plt = _get_pyplot()
    if plt is None or not step_records:
        for name in [
            "planned_trajectories.png",
            "xy.png",
            "speed_time.png",
            "accel_time.png",
            "heading_time.png",
            "distance.png",
            "steer_time.png",
            "throttle_time.png",
        ]:
            _write_plot_placeholder(results_dir / name)
        return

    with plt.rc_context(_PUBLICATION_RC):
        fig, ax = plt.subplots(figsize=(4.8, 4.3))
        last_step_idx = step_records[-1].get("step_idx", len(step_records) - 1)
        for record in step_records:
            is_last = record.get("step_idx") == last_step_idx
            trajectories = (record.get("planning", {}) or {}).get("trajectories_by_agent", {}) or {}
            for agent_id in agent_ids:
                traj = np.asarray(trajectories.get(agent_id, []), dtype=np.float32)
                if traj.ndim != 2 or traj.shape[0] == 0 or traj.shape[1] < 2:
                    continue
                ax.plot(
                    traj[:, 0],
                    traj[:, 1],
                    color=_publication_agent_color(agent_id),
                    linewidth=2.2 if is_last else 0.65,
                    alpha=0.95 if is_last else 0.12,
                    label=agent_id if is_last else None,
                )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        _style_publication_axis(ax, grid_axis="both")
        ax.axis("equal")
        _publication_legend(ax)
        fig.tight_layout(pad=0.7)
        _save_publication_figure(fig, results_dir / "planned_trajectories.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(4.8, 4.3))
        for agent_id in agent_ids:
            xs = np.asarray(
                [(record.get("vehicles", {}) or {}).get(agent_id, {}).get("x", np.nan) for record in step_records],
                dtype=np.float32,
            )
            ys = np.asarray(
                [(record.get("vehicles", {}) or {}).get(agent_id, {}).get("y", np.nan) for record in step_records],
                dtype=np.float32,
            )
            color = _publication_agent_color(agent_id)
            ax.plot(xs, ys, color=color, linewidth=1.6, label=agent_id)
            finite = np.where(np.isfinite(xs) & np.isfinite(ys))[0]
            if finite.size:
                start_idx = int(finite[0])
                end_idx = int(finite[-1])
                ax.scatter(xs[start_idx], ys[start_idx], s=22, facecolors="white", edgecolors=color, linewidths=1.0, zorder=3)
                ax.scatter(xs[end_idx], ys[end_idx], s=24, facecolors=color, edgecolors=color, linewidths=1.0, zorder=3)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        _style_publication_axis(ax, grid_axis="both")
        ax.axis("equal")
        _publication_legend(ax)
        fig.tight_layout(pad=0.7)
        _save_publication_figure(fig, results_dir / "xy.png")
        plt.close(fig)

    _plot_time_series(results_dir / "speed_time.png", step_records, agent_ids, "speed_km_h", "Speed (km/h)")
    _plot_time_series(results_dir / "accel_time.png", step_records, agent_ids, "accel_mps2", "Acceleration (m/s$^2$)")
    _plot_time_series(results_dir / "heading_time.png", step_records, agent_ids, "heading_theta", "Heading (rad)")
    _plot_following_distance(results_dir / "distance.png", step_records, agent_ids)
    _plot_time_series(results_dir / "steer_time.png", step_records, agent_ids, "steer", "Steering")
    _plot_time_series(results_dir / "throttle_time.png", step_records, agent_ids, "throttle", "Throttle")
