"""Compare baseline vs GRPO trajectory plots frame-by-frame.

For each (episode, step) present in both runs, draws all agents'
candidate trajectories and selected trajectory on a shared matplotlib
axes (world-coordinate space, no background image required).

Usage:
    python scripts/make_grpo_comparison_frames.py \
        --baseline-dir  /path/to/baseline/output \
        --grpo-dir      /path/to/grpo/output \
        --output-dir    /path/to/comparison_frames
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np


# ── colours ───────────────────────────────────────────────────────────────────
BASELINE_CAND_COLOR  = "#4E79A7"   # steel blue  – candidate (thin)
BASELINE_SEL_COLOR   = "#1A3A5C"   # dark blue   – selected (thick)
GRPO_CAND_COLOR      = "#F28E2B"   # orange      – candidate (thin)
GRPO_SEL_COLOR       = "#7B3F00"   # dark orange – selected (thick)
AGENT_COLOR          = "#555555"   # dark grey   – vehicle dot
PRIMARY_COLOR        = "#CC0000"   # red         – primary vehicle


def _draw_agent_trajectories(ax, agents: dict, cand_color: str, sel_color: str,
                              agent_color: str, primary_id: str,
                              label_prefix: str, draw_legend: bool = False):
    """Draw all agents' candidates + selected trajectory onto *ax*."""
    for agent_id, data in agents.items():
        ego_xy     = np.asarray(data["ego_world_xy"], dtype=np.float32)
        cands      = np.asarray(data["candidate_trajs_world"], dtype=np.float32)  # [M, T, 2]
        sel_mode   = int(data["selected_mode"])
        valid_mask = data.get("valid_mask", [])
        is_primary = (agent_id == primary_id)

        # vehicle position
        dot_color = PRIMARY_COLOR if is_primary else agent_color
        ax.plot(ego_xy[0], ego_xy[1],
                marker="*" if is_primary else "o",
                markersize=10 if is_primary else 6,
                color=dot_color, zorder=5)

        if cands.ndim != 3 or cands.shape[2] < 2:
            continue

        M = cands.shape[0]
        for m in range(M):
            is_valid = valid_mask[m] if m < len(valid_mask) else True
            if not is_valid:
                continue
            traj = cands[m]  # [T, 2]
            pts = np.vstack([[ego_xy], traj])
            if m == sel_mode:
                ax.plot(pts[:, 0], pts[:, 1], color=sel_color,
                        lw=2.0, alpha=0.95, zorder=4,
                        label=f"{label_prefix} selected" if draw_legend else None)
            else:
                ax.plot(pts[:, 0], pts[:, 1], color=cand_color,
                        lw=0.8, alpha=0.35, zorder=3,
                        label=f"{label_prefix} candidates" if (draw_legend and m == 0) else None)


def make_comparison_frame(b_data: dict, g_data: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal")

    primary_id = b_data.get("primary_agent_id", "agent0")

    # baseline trajectories
    _draw_agent_trajectories(ax, b_data["agents"], BASELINE_CAND_COLOR,
                              BASELINE_SEL_COLOR, AGENT_COLOR, primary_id,
                              "Baseline", draw_legend=True)
    # GRPO trajectories
    _draw_agent_trajectories(ax, g_data["agents"], GRPO_CAND_COLOR,
                              GRPO_SEL_COLOR, AGENT_COLOR, primary_id,
                              "GRPO", draw_legend=True)

    # legend
    legend_handles = [
        mlines.Line2D([], [], color=BASELINE_SEL_COLOR,  lw=2, label="Baseline selected"),
        mlines.Line2D([], [], color=BASELINE_CAND_COLOR, lw=0.8, alpha=0.6, label="Baseline candidates"),
        mlines.Line2D([], [], color=GRPO_SEL_COLOR,      lw=2, label="GRPO selected"),
        mlines.Line2D([], [], color=GRPO_CAND_COLOR,     lw=0.8, alpha=0.6, label="GRPO candidates"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    ep  = b_data["episode"]
    stp = b_data["step"]
    ax.set_title(f"Episode {ep:03d} · Step {stp:05d}", fontsize=10)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.25)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


_EPISODE_RE = re.compile(r"^episode_(\d+)$")
_STEP_RE    = re.compile(r"^step_(\d+)\.json$")


def _collect_tdata(root: Path) -> dict[tuple[int, int], Path]:
    """Return {(episode, step): json_path} for all trajectory_data files."""
    tdata_root = root / "trajectory_data"
    out: dict[tuple[int, int], Path] = {}
    if not tdata_root.exists():
        return out
    for ep_dir in sorted(tdata_root.iterdir()):
        m = _EPISODE_RE.match(ep_dir.name)
        if not m or not ep_dir.is_dir():
            continue
        ep = int(m.group(1))
        for f in sorted(ep_dir.iterdir()):
            sm = _STEP_RE.match(f.name)
            if sm:
                out[(ep, int(sm.group(1)))] = f
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-frame trajectory comparison plots (baseline vs GRPO).")
    parser.add_argument("--baseline-dir",  required=True,
                        help="Output directory of the baseline test run")
    parser.add_argument("--grpo-dir",      required=True,
                        help="Output directory of the GRPO test run")
    parser.add_argument("--output-dir",    required=True,
                        help="Directory to save comparison frames")
    args = parser.parse_args()

    b_root = Path(args.baseline_dir)
    g_root = Path(args.grpo_dir)
    out    = Path(args.output_dir)

    b_tdata = _collect_tdata(b_root)
    g_tdata = _collect_tdata(g_root)

    common = sorted(set(b_tdata) & set(g_tdata))
    if not common:
        print("[compare] No matching (episode, step) pairs found. "
              "Make sure both runs used --save-combined-traj-frames 1.")
        return

    print(f"[compare] {len(common)} matching frames found. Generating plots...")
    for i, key in enumerate(common):
        ep, stp = key
        b_data = json.loads(b_tdata[key].read_text(encoding="utf-8"))
        g_data = json.loads(g_tdata[key].read_text(encoding="utf-8"))
        out_path = out / f"episode_{ep:03d}" / f"step_{stp:05d}.png"
        make_comparison_frame(b_data, g_data, out_path)
        if (i + 1) % 50 == 0 or (i + 1) == len(common):
            print(f"[compare]   {i+1}/{len(common)} done")

    print(f"[compare] Saved to {out}")


if __name__ == "__main__":
    main()
