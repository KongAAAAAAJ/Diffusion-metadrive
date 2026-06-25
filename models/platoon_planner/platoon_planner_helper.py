from __future__ import annotations

from pathlib import Path

import numpy as np


def save_platoon_candidates_debug_plot(
    vehicle,
    resampled_candidates,
    best_index: int,
    output_dir: str | Path = "models/platoon_planner/outputs",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path_dir = Path(output_dir)
    output_path_dir.mkdir(parents=True, exist_ok=True)

    vehicle_name = str(getattr(vehicle, "name", "vehicle"))
    output_path = output_path_dir / f"platoon_candidates_{vehicle_name}.png"

    fig, ax = plt.subplots(figsize=(12, 12))
    ego_pos = np.asarray(getattr(vehicle, "position", [0.0, 0.0])[:2], dtype=np.float32)
    ax.scatter([ego_pos[0]], [ego_pos[1]], c="black", s=42, label="ego", zorder=4)

    for idx, candidate in enumerate(resampled_candidates):
        traj = np.asarray(candidate, dtype=np.float32)
        if traj.ndim != 2 or traj.shape[0] == 0 or traj.shape[1] < 2:
            continue

        if int(idx) == int(best_index):
            ax.plot(traj[:, 0], traj[:, 1], color="tab:red", linewidth=3.0, label="best", zorder=3)
            ax.scatter([traj[-1, 0]], [traj[-1, 1]], color="tab:red", s=34, zorder=4)
        else:
            ax.plot(traj[:, 0], traj[:, 1], color="0.70", linewidth=1.0, alpha=0.8, zorder=1)

    lane_index = tuple(getattr(getattr(vehicle, "lane", None), "index", ()) or ())
    ax.set_title(f"{vehicle_name} platoon candidates lane={lane_index} n={len(resampled_candidates)}")
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path
