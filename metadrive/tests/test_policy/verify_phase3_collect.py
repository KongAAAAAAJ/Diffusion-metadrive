"""verify_phase3_collect.py

Manual verification script for Phase 3 changes to collect_expert.py.

Runs a short MetaDrive episode, calls build_frame() and build_episode_samples(),
then checks all new field shapes/dtypes and produces a visualisation PNG.

Usage:
    /home/kong/anaconda3/envs/meta_drive/bin/python \
        metadrive/tests/test_policy/verify_phase3_collect.py
"""
from __future__ import annotations

import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np

# ── headless rendering ──────────────────────────────────────────────────────
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("DISPLAY", ":0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project path ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── imports ───────────────────────────────────────────────────────────────────
from metadrive.envs.diffusion_envs.base_multi_env import (
    DatasetCollectEnv,
    DEFAULT_HYBRID_MAP_CONFIG,
    ROUTE_PRESET_BLOCK_IDS,
)
from metadrive.exp_dataset.collect_expert import (
    ExpertCollectorConfig,
    build_episode_samples,
    build_frame,
)
from metadrive.exp_dataset.expert_idm_policy import ExpertIDMPolicy as Expert
from metadrive.policy.diffusion_policy.mode_definitions import MODE_SLOTS

# ── helpers ───────────────────────────────────────────────────────────────────
PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(name: str, cond: bool, detail: str = "") -> bool:
    tag = PASS if cond else FAIL
    print(f"  [{tag}] {name}" + (f"  ({detail})" if detail else ""))
    return cond


# ── expected field specs ──────────────────────────────────────────────────────
POLYLINE_FIELDS = [
    "current_lane_polyline",
    "left_lane_polyline",
    "right_lane_polyline",
    "left_branch_polyline",
    "right_branch_polyline",
]
FLAG_FIELDS = ["has_left_adjacent", "has_right_adjacent", "has_left_branch", "has_right_branch"]
GAP_FIELDS = ["left_lane_gap", "right_lane_gap"]
SAMPLE_EXTRA = ["coarse_trajectories", "mode_valid_mask", "gt_mode_label"]


def verify_frame(frame: dict) -> int:
    n_fail = 0
    for fname in POLYLINE_FIELDS:
        ok = fname in frame and np.asarray(frame[fname]).shape == (20, 2)
        if not check(f"frame['{fname}'] shape==(20,2)", ok,
                     str(np.asarray(frame[fname]).shape) if fname in frame else "missing"):
            n_fail += 1
        if fname in frame:
            ok2 = np.asarray(frame[fname]).dtype == np.float32
            if not check(f"frame['{fname}'] dtype==float32", ok2, str(np.asarray(frame[fname]).dtype)):
                n_fail += 1
    for fname in FLAG_FIELDS:
        ok = fname in frame and np.asarray(frame[fname]).dtype == np.int8
        if not check(f"frame['{fname}'] dtype==int8", ok,
                     str(np.asarray(frame[fname]).dtype) if fname in frame else "missing"):
            n_fail += 1
    for fname in GAP_FIELDS:
        ok = fname in frame and np.asarray(frame[fname]).dtype == np.float32
        if not check(f"frame['{fname}'] dtype==float32", ok,
                     str(np.asarray(frame[fname]).dtype) if fname in frame else "missing"):
            n_fail += 1
    return n_fail


def verify_sample(sample: dict) -> int:
    n_fail = 0
    ok = "coarse_trajectories" in sample and np.asarray(sample["coarse_trajectories"]).shape == (10, 8, 2)
    if not check("sample['coarse_trajectories'] shape==(10,8,2)", ok,
                 str(np.asarray(sample["coarse_trajectories"]).shape) if "coarse_trajectories" in sample else "missing"):
        n_fail += 1
    ok = "mode_valid_mask" in sample and np.asarray(sample["mode_valid_mask"]).shape == (10,)
    if not check("sample['mode_valid_mask'] shape==(10,)", ok,
                 str(np.asarray(sample["mode_valid_mask"]).shape) if "mode_valid_mask" in sample else "missing"):
        n_fail += 1
    ok = "gt_mode_label" in sample and np.asarray(sample["gt_mode_label"]).ndim == 0
    if not check("sample['gt_mode_label'] is scalar", ok,
                 str(np.asarray(sample["gt_mode_label"]).shape) if "gt_mode_label" in sample else "missing"):
        n_fail += 1
    # emergency stop always valid
    if "mode_valid_mask" in sample:
        ok = bool(sample["mode_valid_mask"][9])
        if not check("mode_valid_mask[9] (EMERGENCY_STOP) is True", ok):
            n_fail += 1
    return n_fail


# ── visualisation ─────────────────────────────────────────────────────────────
COLORS = {
    "current_lane_polyline": ("blue", "-", "current lane"),
    "left_lane_polyline":    ("green", "-", "left lane"),
    "right_lane_polyline":   ("red", "-", "right lane"),
    "left_branch_polyline":  ("cyan", "--", "left branch"),
    "right_branch_polyline": ("magenta", "--", "right branch"),
}

SLOT_COLORS = [
    "#2196F3",  # 0 KL high (blue)
    "#1565C0",  # 1 KL medium
    "#0D47A1",  # 2 KL low
    "#4CAF50",  # 3 LC_L high (green)
    "#388E3C",  # 4 LC_L medium
    "#1B5E20",  # 5 LC_L low
    "#F44336",  # 6 LC_R high (red)
    "#C62828",  # 7 LC_R medium
    "#7F0000",  # 8 LC_R low
    "#FF9800",  # 9 EMERGENCY_STOP (orange)
]


def make_figure(frames: list, samples: list, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Phase 3 Verification", fontsize=14)

    # --- Panel 1: polylines for first 5 frames ---
    ax = axes[0]
    ax.set_title("Lane Polylines (first 5 frames, ego-local)")
    ax.set_xlabel("x (forward, m)")
    ax.set_ylabel("y (left, m)")
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax.axvline(0, color="black", linewidth=0.5, linestyle=":")
    ax.plot(0, 0, "ko", markersize=8, label="ego")
    n_shown = 0
    for i, frame in enumerate(frames[:5]):
        for fname, (color, ls, label) in COLORS.items():
            pl = np.asarray(frame.get(fname, np.zeros((20, 2))), dtype=np.float32)
            if np.any(pl != 0):
                lbl = label if i == 0 else None
                ax.plot(pl[:, 0], pl[:, 1], color=color, linestyle=ls,
                        linewidth=1.5, alpha=0.6 + 0.08 * i, label=lbl)
                n_shown += 1
    if n_shown == 0:
        ax.text(5, 0, "all polylines zero (single-lane?)", color="gray")
    # flag summary
    flags = {f: int(frames[0].get(f, 0)) for f in FLAG_FIELDS}
    gaps  = {f: float(frames[0].get(f, -1)) for f in GAP_FIELDS}
    info = "\n".join([f"{k}={v}" for k, v in {**flags, **gaps}.items()])
    ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=7,
            verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend(fontsize=7, loc="lower right")
    ax.set_aspect("equal")

    # --- Panel 2: coarse trajectories for first sample ---
    ax2 = axes[1]
    ax2.set_title("Coarse Trajectories (sample 0)")
    ax2.set_xlabel("x (forward, m)")
    ax2.set_ylabel("y (left, m)")
    ax2.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax2.axvline(0, color="black", linewidth=0.5, linestyle=":")
    ax2.plot(0, 0, "ko", markersize=8, label="ego")

    if samples:
        sample = samples[0]
        coarse = np.asarray(sample.get("coarse_trajectories", np.zeros((10, 8, 2))))
        mask   = np.asarray(sample.get("mode_valid_mask", np.zeros(10, dtype=bool)))
        label  = int(sample.get("gt_mode_label", 0))
        gt_xy  = np.asarray(sample.get("trajectory", np.zeros((8, 3))))[:, :2]

        for slot in MODE_SLOTS:
            traj = coarse[slot.index]
            valid = bool(mask[slot.index])
            color = SLOT_COLORS[slot.index]
            ls = "-" if valid else "--"
            alpha = 0.9 if valid else 0.25
            lw = 2.5 if slot.index == label else 1.2
            marker = "o" if slot.index == label else None
            ax2.plot(traj[:, 0], traj[:, 1], color=color, linestyle=ls,
                     linewidth=lw, alpha=alpha, marker=marker, markersize=4,
                     label=f"{slot.index}:{slot.name[:12]}" + (" ★" if slot.index == label else ""))

        # GT trajectory
        ax2.plot(gt_xy[:, 0], gt_xy[:, 1], "k-", linewidth=2, label="GT traj", zorder=10)

        valid_names = [s.name for s in MODE_SLOTS if mask[s.index]]
        info2 = f"label={label} ({MODE_SLOTS[label].name})\nvalid: {', '.join(valid_names)}"
        ax2.text(0.02, 0.98, info2, transform=ax2.transAxes, fontsize=7,
                 verticalalignment="top", family="monospace",
                 bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))
        ax2.legend(fontsize=6, loc="lower right", ncol=2)
    else:
        ax2.text(5, 0, "no samples generated", color="gray")

    ax2.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\nFigure saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Phase 3 verification — collect_expert polyline fields")
    print("=" * 60)

    config = ExpertCollectorConfig(
        expert_type="idm",
        target_samples=10,
        max_episode_steps=120,
        horizon_steps=20,       # need at least this many frames
        trajectory_num_poses=8,
        target_stride_steps=5,
        sample_stride_steps=5,
        save_videos=False,
        trajectory_correction_enabled=False,
        trajectory_filter_enabled=False,
        num_scenarios=1,
        map_block_num=5,
        use_hybrid_map=True,
    )

    env_config = {
        "use_render": False,
        "num_scenarios": 1,
        "use_hybrid_map": True,
        "hybrid_map_blocks_config": list(DEFAULT_HYBRID_MAP_CONFIG),
        "map": 5,
        "top_down_camera_initial_z": 30.0,
    }

    env = DatasetCollectEnv(env_config)
    frames = []
    total_fail = 0

    try:
        obs_dict, info = env.reset()
        agent_id = list(obs_dict.keys())[0]
        vehicle = env.agents.get(agent_id)

        expert = Expert(control_object=vehicle, random_seed=42)

        print(f"\nCollecting frames (agent_id={agent_id}) ...")
        for step in range(150):
            if vehicle is None or agent_id not in env.agents:
                break
            action = expert.act(None)
            obs_dict, reward_dict, term_dict, trunc_dict, _ = env.step({agent_id: action})

            if agent_id not in obs_dict:
                break
            obs = obs_dict[agent_id]

            try:
                frame = build_frame(
                    vehicle,
                    config,
                    obs["ego_state"],
                    obs["others_state"],
                    obs["lidar"],
                    obs["topdown"],
                    obs["rgb_left"],
                    obs["rgb_front"],
                    obs["rgb_right"],
                )
                frames.append(frame)
            except Exception as e:
                print(f"  [WARN] build_frame step {step}: {e}")
                traceback.print_exc()

            if term_dict.get(agent_id, False) or trunc_dict.get(agent_id, False):
                break

        print(f"\nCollected {len(frames)} frames.")

        # --- verify first frame ---
        print("\n[1] Frame field checks (frame 0):")
        if frames:
            total_fail += verify_frame(frames[0])
        else:
            print("  [FAIL] no frames collected")
            total_fail += 1

        # --- build samples ---
        print("\n[2] Building episode samples ...")
        samples = []
        if len(frames) > config.horizon_steps:
            try:
                samples = build_episode_samples(
                    frames, config,
                    traffic_density=0.1,
                    route_id="mainline",
                    local_route="straight",
                    scenario_id="verify",
                )
                print(f"  Built {len(samples)} samples.")
            except Exception as e:
                print(f"  [FAIL] build_episode_samples: {e}")
                traceback.print_exc()
                total_fail += 1
        else:
            print(f"  [SKIP] not enough frames ({len(frames)} < {config.horizon_steps})")

        # --- verify first sample ---
        print("\n[3] Sample field checks (sample 0):")
        if samples:
            total_fail += verify_sample(samples[0])
        else:
            print("  [SKIP] no samples to check")

        # --- mode label distribution ---
        if samples:
            label_counts: dict = defaultdict(int)
            for s in samples:
                lbl = int(s.get("gt_mode_label", -1))
                name = MODE_SLOTS[lbl].name if 0 <= lbl < 10 else str(lbl)
                label_counts[name] += 1
            print("\n[4] Mode label distribution:")
            for name, count in sorted(label_counts.items(), key=lambda x: -x[1]):
                bar = "#" * count
                print(f"  {name:<30} {count:3d}  {bar}")

        # --- visualise ---
        out_png = "/tmp/phase3_verify.png"
        print("\n[5] Generating visualisation ...")
        try:
            make_figure(frames, samples, out_png)
        except Exception as e:
            print(f"  [WARN] visualisation failed: {e}")
            traceback.print_exc()

    finally:
        env.close()

    # --- summary ---
    print("\n" + "=" * 60)
    if total_fail == 0:
        print(f"\033[32mAll checks PASSED.\033[0m  frames={len(frames)}, samples={len(samples)}")
    else:
        print(f"\033[31m{total_fail} check(s) FAILED.\033[0m  frames={len(frames)}, samples={len(samples)}")
    print("=" * 60)
    return total_fail


if __name__ == "__main__":
    sys.exit(main())
