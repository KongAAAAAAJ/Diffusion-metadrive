from __future__ import annotations

from pathlib import Path

import numpy as np


def save_candidate_debug_plot(
    vehicle,
    candidates: list[dict],
    counter: int,
    output_dir: str | Path = "models/decisioner/outputs",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path_dir = Path(output_dir)
    output_path_dir.mkdir(parents=True, exist_ok=True)

    vehicle_name = str(getattr(vehicle, "name", "vehicle"))
    output_path = output_path_dir / f"candidate_{vehicle_name}_{int(counter):06d}.png"

    fig, ax = plt.subplots(figsize=(6, 5))
    ego_pos = np.asarray(getattr(vehicle, "position", [0.0, 0.0])[:2], dtype=np.float32)
    ax.scatter([ego_pos[0]], [ego_pos[1]], c="black", s=40, label="ego")

    action_names = {-1: "left", 0: "keep", 1: "right"}
    action_colors = {-1: "tab:blue", 0: "tab:green", 1: "tab:red"}

    for candidate in candidates:
        traj = candidate.get("trajectory_world")
        if traj is None:
            continue
        traj = np.asarray(traj, dtype=np.float32)
        if traj.ndim != 2 or traj.shape[0] == 0:
            continue

        action = int(candidate.get("action", 0))
        label = f"{action_names.get(action, action)} ({action})"
        color = action_colors.get(action, "tab:gray")
        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.8, label=label)
        ax.scatter([traj[-1, 0]], [traj[-1, 1]], color=color, s=24)

    lane_index = tuple(getattr(getattr(vehicle, "lane", None), "index", ()) or ())
    ax.set_title(f"{vehicle_name} candidates lane={lane_index} n={len(candidates)}")
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def save_lane_pair_debug_plot(
    vehicle,
    source_lane,
    target_lane,
    action: int,
    counter: int,
    output_dir: str | Path = "models/decisioner/outputs",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path_dir = Path(output_dir)
    output_path_dir.mkdir(parents=True, exist_ok=True)

    vehicle_name = str(getattr(vehicle, "name", "vehicle"))
    output_path = output_path_dir / f"lane_pair_{vehicle_name}_action_{int(action)}_{int(counter):06d}.png"

    fig, ax = plt.subplots(figsize=(6, 5))
    ego_pos = np.asarray(getattr(vehicle, "position", [0.0, 0.0])[:2], dtype=np.float32)
    ax.scatter([ego_pos[0]], [ego_pos[1]], c="black", s=40, label="ego")

    _plot_lane_centerline(ax, source_lane, color="tab:blue", linestyle="-", label="source_lane")
    _plot_lane_centerline(ax, target_lane, color="tab:red", linestyle="--", label="target_lane")

    source_index = tuple(getattr(source_lane, "index", ()) or ())
    target_index = tuple(getattr(target_lane, "index", ()) or ())
    ax.set_title(f"{vehicle_name} action={int(action)} source={source_index} target={target_index}")
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def save_s8_route_lanes_debug_plot(
    vehicle,
    road_network,
    checkpoints,
    source_lane,
    counter: int,
    output_dir: str | Path = "models/decisioner/outputs",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path_dir = Path(output_dir)
    output_path_dir.mkdir(parents=True, exist_ok=True)

    vehicle_name = str(getattr(vehicle, "name", "vehicle"))
    output_path = output_path_dir / f"s8_route_lanes_{vehicle_name}_{int(counter):06d}.png"

    fig, ax = plt.subplots(figsize=(70, 60))
    ego_pos = np.asarray(getattr(vehicle, "position", [0.0, 0.0])[:2], dtype=np.float32)
    ax.scatter([ego_pos[0]], [ego_pos[1]], c="black", s=45, label="ego", zorder=4)

    del checkpoints
    source_index = tuple(getattr(source_lane, "index", ()) or ())
    graph = getattr(road_network, "graph", {}) or {}
    for from_node, to_map in graph.items():
        for to_node, lanes_for_road in (to_map or {}).items():
            try:
                lanes = list(lanes_for_road)
            except Exception:
                lanes = []
            for lane_id, lane in enumerate(lanes):
                lane_index = tuple(getattr(lane, "index", ()) or (from_node, to_node, lane_id))
                color = "tab:blue" if lane_index == source_index else "0.65"
                linewidth = 2.6 if lane_index == source_index else 1.3
                points = _lane_centerline_points(lane)
                if points.size == 0:
                    continue
                ax.plot(points[:, 0], points[:, 1], color=color, linewidth=linewidth)
                mid = points[len(points) // 2]
                ax.text(
                    float(mid[0]),
                    float(mid[1]),
                    str(lane_index),
                    fontsize=7,
                    color=color,
                    ha="center",
                    va="center",
                    bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.0},
                    zorder=5,
                )

    ax.set_title(f"{vehicle_name} S8 map lanes all road graph")
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def _plot_lane_centerline(ax, lane, *, color: str, linestyle: str, label: str) -> None:
    if lane is None:
        return
    points = _lane_centerline_points(lane)
    if points.size == 0:
        return
    ax.plot(points[:, 0], points[:, 1], color=color, linestyle=linestyle, linewidth=2.0, label=label)


def _lane_centerline_points(lane) -> np.ndarray:
    try:
        lane_length = float(getattr(lane, "length", 0.0) or 0.0)
        if lane_length <= 0.0:
            return np.zeros((0, 2), dtype=np.float32)
        longitudinals = np.linspace(0.0, lane_length, num=80, dtype=np.float32)
        points = np.asarray([lane.position(float(s), 0.0)[:2] for s in longitudinals], dtype=np.float32)
    except Exception:
        return np.zeros((0, 2), dtype=np.float32)
    if points.ndim != 2 or points.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return points
