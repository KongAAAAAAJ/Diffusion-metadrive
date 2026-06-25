"""Closed-loop test for PlatoonDiffusionPlanner using a full diffusion checkpoint.

Uses PlatoonEnv + PlatoonDiffusionPlanner directly — no SB3 wrapper.
The planner's native plan_cls_branch selects the best mode for each vehicle.
Compatible with both the original pretrained ckpt and the PPO-merged ckpt.

Usage
-----
    python -m train.test_platoon_diffusion_policy \\
        --checkpoint /path/to/diffusion-epochXX.ckpt \\
        --scenario-id S1_free_cruise_straight \\
        --episodes 2 --max-steps 200

Or via shell script:
    bash scripts/run_platoon_diffusion_test.sh
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg", force=True)
import numpy as np

DEFAULT_OUTPUT_ROOT = Path("/media/kong/Elements_SE/Diffusion_Data/outputs/platoon_diffusion_test")
DEFAULT_MODEL_CONFIG_PATH = "configs/diffusion/model.yaml"
DEFAULT_CHECKPOINT = "/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_20/checkpoints/diffusion-epoch=25.ckpt"


# ---------------------------------------------------------------------------
# Env and planner construction
# ---------------------------------------------------------------------------

def _build_env_config(
    config: Mapping[str, Any],
    scenario_id: str,
    local_route: str,
    start_seed: int,
    traffic_density: float,
    use_render: bool,
) -> dict:
    env_cfg: dict = {
        "num_agents": int(config.get("num_agents", 3)),
        "observation_mode": "multimodal",
        "use_render": use_render,
        "traffic_density": traffic_density,
        "start_seed": start_seed,
        "num_scenarios": 1,
        "horizon": int(config.get("horizon", 1000)),
        "allow_respawn": False,
    }
    if scenario_id:
        env_cfg["scenario_id"] = scenario_id
    if local_route:
        env_cfg["local_route"] = local_route
    return env_cfg


def build_platoon_env(env_config: dict):
    from envs.platoon_env import PlatoonEnv
    return PlatoonEnv(env_config)


def build_platoon_planner(ckpt_path: str, num_agents: int, model_config_path: str, device: str = "cpu"):
    import torch
    from models.diffusion.transfuser_config import (
        build_transfuser_config,
        diffusion_model_config_to_overrides,
        load_diffusion_model_config,
    )
    from models.platoon_planner.platoon_diffusion_planner import PlatoonDiffusionPlanner
    from models.platoon_planner._weight_migration import migrate_single_to_platoon

    model_cfg = load_diffusion_model_config(model_config_path)
    tf_config = build_transfuser_config(
        str(model_cfg.get("model_size", "small")),
        **diffusion_model_config_to_overrides(model_cfg),
    )
    planner = PlatoonDiffusionPlanner(tf_config, num_vehicles=num_agents)
    planner = migrate_single_to_platoon(ckpt_path, planner)
    planner.eval()
    for p in planner.parameters():
        p.requires_grad_(False)
    planner = planner.to(torch.device(device) if device != "auto" else
                         torch.device("cuda") if __import__("torch").cuda.is_available() else torch.device("cpu"))
    return planner


# ---------------------------------------------------------------------------
# Observation → planner batch
# ---------------------------------------------------------------------------

def _build_planner_batch(
    raw_obs: Mapping[str, Mapping[str, Any]],
    agent_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Extract camera/lidar/status/formation_relation_state for each agent."""
    required = ("camera", "lidar", "status", "formation_relation_state")
    batch: dict[str, dict[str, Any]] = {}
    for aid in agent_ids:
        if aid not in raw_obs:
            continue
        obs = raw_obs[aid]
        missing = [k for k in required if k not in obs]
        if missing:
            raise KeyError(f"Observation for {aid} missing keys: {missing}")
        batch[aid] = {k: obs[k] for k in required}
    return batch


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_platoon_episode(
    env,
    planner,
    agent_ids: list[str],
    max_steps: int,
    save_topdown: bool,
    output_dir: Path | None,
    episode_idx: int,
    video_fps: int,
) -> dict[str, Any]:
    """Run one episode and return per-episode stats."""
    import cv2

    try:
        from models.diffusion.test_transfuser_policy import (
            _build_topdown_world_to_screen_projector,
            _capture_2d_topdown_frame,
        )
        _topdown_available = True
    except Exception:
        _topdown_available = False

    result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        raw_obs, info = result
    else:
        raw_obs, info = result, {}

    total_reward = 0.0
    steps = 0
    done = False
    video_frames: list[np.ndarray] = []
    mode_counts: dict[str, dict[int, int]] = {aid: {} for aid in agent_ids}

    while not done:
        # Planner inference
        planner_batch = _build_planner_batch(raw_obs, agent_ids)
        if not planner_batch:
            break
        with __import__("torch").no_grad():
            export = planner.export_mode_selection(planner_batch)

        candidates = np.asarray(export["trajectory_candidates"], dtype=np.float32)  # (N, M, 8, 3)
        masks = np.asarray(export["lane_valid_mask"], dtype=bool)                   # (N, M)
        masked_logits = np.asarray(export["masked_cls_logits"], dtype=np.float32)   # (N, M)
        exported_ids: list[str] = list(export["agent_ids"])

        # Select argmax mode per agent (respecting mask)
        trajectories: dict[str, np.ndarray] = {}
        for agent_idx, aid in enumerate(exported_ids):
            mode_idx = int(np.argmax(masked_logits[agent_idx]))
            traj = candidates[agent_idx, mode_idx]          # (8, 3)
            trajectories[aid] = traj.astype(np.float32)
            mode_counts[aid][mode_idx] = mode_counts[aid].get(mode_idx, 0) + 1

        # Capture frame before step — save as PNG and collect for video
        if save_topdown and _topdown_available and output_dir is not None:
            frame = _capture_2d_topdown_frame(env)
            if frame is not None:
                video_frames.append(frame)
                frame_path = (
                    output_dir
                    / "topdown_frames"
                    / f"episode_{episode_idx:03d}"
                    / f"step_{steps:05d}.png"
                )
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(
                    str(frame_path),
                    cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR),
                )

        # Step env
        step_result = env.step(trajectories)
        if len(step_result) == 5:
            raw_obs, reward, terminated, truncated, step_info = step_result
            done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
        else:
            raw_obs, reward, done_dict, step_info = step_result
            done = bool(done_dict.get("__all__", False))

        # Accumulate reward (mean across agents)
        reward_vals = [float(v) for v in (reward or {}).values() if isinstance(v, (int, float))]
        total_reward += float(np.mean(reward_vals)) if reward_vals else 0.0
        steps += 1

        if max_steps > 0 and steps >= max_steps:
            done = True

    # Save video
    if output_dir is not None and save_topdown and video_frames:
        vid_path = output_dir / "videos" / f"episode_{episode_idx:03d}.mp4"
        vid_path.parent.mkdir(parents=True, exist_ok=True)
        h, w = video_frames[0].shape[:2]
        writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), float(video_fps), (w, h))
        for frame in video_frames:
            writer.write(cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"[platoon_test] video saved: {vid_path}", flush=True)

    return {
        "episode_idx": episode_idx,
        "total_reward": float(total_reward),
        "steps": steps,
        "mode_counts": {aid: dict(mc) for aid, mc in mode_counts.items()},
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_platoon_policy(
    env,
    planner,
    agent_ids: list[str],
    episodes: int,
    max_steps: int,
    output_dir: Path | None,
    save_topdown: bool,
    save_trajectory_plot: bool,
    video_fps: int,
) -> dict[str, Any]:
    episode_results: list[dict] = []
    for ep in range(episodes):
        print(f"[platoon_test] episode {ep + 1}/{episodes} ...", flush=True)
        ep_result = run_platoon_episode(
            env, planner, agent_ids,
            max_steps=max_steps,
            save_topdown=save_topdown,
            output_dir=output_dir,
            episode_idx=ep,
            video_fps=video_fps,
        )
        episode_results.append(ep_result)
        print(
            f"[platoon_test] episode {ep} reward={ep_result['total_reward']:.3f} "
            f"steps={ep_result['steps']}",
            flush=True,
        )

    rewards = [r["total_reward"] for r in episode_results]
    lengths = [r["steps"] for r in episode_results]
    summary = {
        "episodes": episodes,
        "episode_reward_mean": float(np.mean(rewards)),
        "episode_reward_std": float(np.std(rewards)),
        "episode_reward_min": float(np.min(rewards)),
        "episode_reward_max": float(np.max(rewards)),
        "episode_length_mean": float(np.mean(lengths)),
        "episode_results": episode_results,
    }

    if save_trajectory_plot and output_dir is not None:
        _save_reward_plot(output_dir, episode_results)

    return summary


def _save_reward_plot(output_dir: Path, results: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
        rewards = [r["total_reward"] for r in results]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(rewards)), rewards)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total reward")
        ax.set_title("Platoon diffusion policy — episode rewards")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = output_dir / "episode_rewards.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"[platoon_test] reward plot saved: {plot_path}", flush=True)
    except Exception as exc:
        print(f"[platoon_test] WARNING: could not save reward plot: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_test(
    config: dict[str, Any],
    checkpoint: str,
    output_dir: Path,
    scenario_id: str,
    local_route: str,
    episodes: int,
    max_steps: int,
    start_seed: int,
    traffic_density: float,
    use_render: bool,
    save_topdown: bool,
    save_trajectory_plot: bool,
    video_fps: int,
    planner_device: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    num_agents = int(config.get("num_agents", 3))
    agent_ids = [f"agent{i}" for i in range(num_agents)]
    model_config_path = str(config.get("model_config_path", DEFAULT_MODEL_CONFIG_PATH))

    print(f"[platoon_test] checkpoint     : {checkpoint}", flush=True)
    print(f"[platoon_test] scenario_id    : {scenario_id or '(config default)'}", flush=True)
    print(f"[platoon_test] local_route    : {local_route or '(config default)'}", flush=True)
    print(f"[platoon_test] num_agents     : {num_agents}", flush=True)
    print(f"[platoon_test] episodes       : {episodes}", flush=True)
    print(f"[platoon_test] max_steps      : {max_steps or '(env horizon)'}", flush=True)
    print(f"[platoon_test] planner_device : {planner_device}", flush=True)
    print(f"[platoon_test] output_dir     : {output_dir}", flush=True)

    env_config = _build_env_config(config, scenario_id, local_route, start_seed, traffic_density, use_render)
    env = build_platoon_env(env_config)
    planner = build_platoon_planner(checkpoint, num_agents, model_config_path, planner_device)

    summary = evaluate_platoon_policy(
        env, planner, agent_ids,
        episodes=episodes,
        max_steps=max_steps,
        output_dir=output_dir,
        save_topdown=save_topdown,
        save_trajectory_plot=save_trajectory_plot,
        video_fps=video_fps,
    )
    summary["checkpoint"] = checkpoint
    summary["scenario_id"] = scenario_id
    summary["local_route"] = local_route

    (output_dir / "test_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[platoon_test] summary saved  : {output_dir / 'test_summary.json'}", flush=True)
    env.close()
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop platoon test using native diffusion planner.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Full diffusion checkpoint (.ckpt).")
    parser.add_argument("--model-config-path", default=DEFAULT_MODEL_CONFIG_PATH)
    parser.add_argument("--scenario-id", default="S1_free_cruise_straight")
    parser.add_argument("--local-route", default="",
                        help="Fix to a specific local route; empty = use config/default.")
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Max steps per episode (0 = env horizon).")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--traffic-density", type=float, default=0.06)
    parser.add_argument("--use-render", type=int, choices=(0, 1), default=0)
    parser.add_argument("--save-topdown", type=int, choices=(0, 1), default=1)
    parser.add_argument("--save-trajectory-plot", type=int, choices=(0, 1), default=1)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--planner-device", default="auto",
                        help="Device for planner inference: cpu / cuda / auto.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = {
        "num_agents": args.num_agents,
        "model_config_path": args.model_config_path,
        "horizon": 1000,
    }
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT / "test_1"
    # Auto-increment test directory
    if not args.output_dir:
        import re
        _re = re.compile(r"^test_(\d+)$")
        parent = DEFAULT_OUTPUT_ROOT
        parent.mkdir(parents=True, exist_ok=True)
        existing = [int(m.group(1)) for p in parent.iterdir()
                    if p.is_dir() and (m := _re.match(p.name))]
        output_dir = parent / f"test_{max(existing) + 1 if existing else 1}"

    run_test(
        config=config,
        checkpoint=args.checkpoint,
        output_dir=output_dir,
        scenario_id=args.scenario_id,
        local_route=args.local_route,
        episodes=args.episodes,
        max_steps=args.max_steps,
        start_seed=args.start_seed,
        traffic_density=args.traffic_density,
        use_render=bool(args.use_render),
        save_topdown=bool(args.save_topdown),
        save_trajectory_plot=bool(args.save_trajectory_plot),
        video_fps=args.video_fps,
        planner_device=args.planner_device,
    )
    print(f"[platoon_test] done. outputs: {output_dir}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
