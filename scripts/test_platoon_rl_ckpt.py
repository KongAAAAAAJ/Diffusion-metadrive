from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.platoon_env import PlatoonEnv, obs_to_tensor
from metadrive.policy.diffusion_policy.transfuser_config import build_transfuser_config
from models.platoon.platoon_diffusion_planner import PlatoonDiffusionPlanner
from models.platoon.weight_migration import migrate_single_to_platoon
from train.train_platoon_rl import DEFAULT_CONFIG_PATH, DEFAULT_RL_OUTPUT_ROOT, DEFAULT_SINGLE_CKPT


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Visualize a trained platoon RL checkpoint in PlatoonEnv.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to RL checkpoint, or 'latest'.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--single-ckpt", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--render", type=int, choices=(0, 1), default=1)
    parser.add_argument("--save-video", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--hazard-scenario", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args(argv)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_rl_checkpoint(checkpoint: str, root: str | Path = DEFAULT_RL_OUTPUT_ROOT) -> Path:
    if checkpoint != "latest":
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"RL checkpoint not found: {path}")
        return path

    root = Path(root)
    run_dirs = sorted([child for child in root.iterdir() if child.is_dir() and child.name.startswith("run_")])
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found under: {root}")
    latest_run = run_dirs[-1]
    ckpts = sorted((latest_run / "checkpoints").glob("step_*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No RL checkpoints found under: {latest_run / 'checkpoints'}")
    return ckpts[-1]


def resolve_output_dir(requested: str | None, checkpoint_path: Path) -> Path:
    if requested:
        return Path(requested)
    run_name = checkpoint_path.parent.parent.name if checkpoint_path.parent.parent.name.startswith("run_") else checkpoint_path.parent.name
    return Path("outputs/rl_eval") / f"{run_name}_{checkpoint_path.stem}"


def load_eval_config(args) -> dict[str, Any]:
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}
    if args.num_agents is not None:
        config["num_agents"] = int(args.num_agents)
    if args.hazard_scenario is not None:
        config["hazard_scenario"] = str(args.hazard_scenario)
    return config


def resolve_single_ckpt(args, rl_payload: dict[str, Any]) -> Path:
    if args.single_ckpt:
        path = Path(args.single_ckpt)
    else:
        candidate = rl_payload.get("single_ckpt") or rl_payload.get("single_ckpt_path") or DEFAULT_SINGLE_CKPT
        path = Path(candidate)
    if not path.exists():
        raise FileNotFoundError(f"Single-vehicle checkpoint not found: {path}")
    return path


def build_planner(config: dict[str, Any], single_ckpt_path: Path, device: torch.device) -> PlatoonDiffusionPlanner:
    anchor_path = Path("metadrive/exp_dataset/metadrive_anchors_ppo.npy")
    if not anchor_path.exists():
        anchor_path = Path("metadrive/exp_dataset/metadrive_anchors.npy")
    transfuser_config = build_transfuser_config("small", plan_anchor_path=str(anchor_path))
    planner = PlatoonDiffusionPlanner(transfuser_config, num_vehicles=int(config.get("num_agents", 3)))
    planner = migrate_single_to_platoon(str(single_ckpt_path), planner)
    planner = planner.to(device)
    planner.eval()
    return planner


def load_rl_checkpoint(checkpoint_path: Path, model: PlatoonDiffusionPlanner) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload.get("model_state_dict", payload.get("state_dict", payload))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[eval] missing_keys={len(missing)}")
    if unexpected:
        print(f"[eval] unexpected_keys={len(unexpected)}")
    return payload


def build_env(config: dict[str, Any], render: bool) -> PlatoonEnv:
    env_config = {
        "observation_mode": "multimodal",
        "use_render": bool(render),
        "num_agents": int(config.get("num_agents", 3)),
        "horizon": int(config.get("horizon", 100)),
        "traffic_density": float(config.get("traffic_density", 0.04)),
        "num_scenarios": int(config.get("num_scenarios", 1)),
        "use_hybrid_map": bool(config.get("use_hybrid_map", True)),
        "hybrid_map_sequence": str(config.get("hybrid_map_sequence", "SSXCOCSS")),
    }
    if "hazard_scenario" in config:
        env_config["hazard_scenario"] = config["hazard_scenario"]
    return PlatoonEnv(env_config)


class VideoRecorder:
    def __init__(self, output_path: Path, fps: int = 10):
        self.output_path = output_path
        self.fps = int(fps)
        self._writer = None

    def write(self, frame: np.ndarray) -> None:
        frame = np.asarray(frame)
        if frame.ndim != 3:
            raise ValueError(f"Expected HWC frame, got shape {tuple(frame.shape)}")
        if self._writer is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(str(self.output_path), fourcc, float(self.fps), (width, height))
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None


def render_frame(env: PlatoonEnv, text: dict[str, Any]) -> np.ndarray:
    if getattr(env, "top_down_renderer", None) is None:
        env.render(text=text, mode="top_down", window=False)
    frame = env.render(text=text, mode="top_down", to_image=True)
    if frame is None:
        raise RuntimeError("Top-down render returned no frame")
    return np.asarray(frame)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_episode(env: PlatoonEnv, model: PlatoonDiffusionPlanner, device: torch.device, render: bool, recorder: VideoRecorder | None, episode_idx: int, max_steps: int | None = None) -> dict[str, Any]:
    obs = env.reset()
    if render and hasattr(env, "switch_to_third_person_view"):
        try:
            env.switch_to_third_person_view()
        except Exception:
            pass
    done = False
    episode_reward = 0.0
    episode_length = 0
    final_info: dict[str, Any] = {}
    start = time.time()

    while not done:
        batch = {agent_id: obs_to_tensor(agent_obs, device=device) for agent_id, agent_obs in obs.items()}
        with torch.no_grad():
            actions = model(batch)
        action_np = {agent_id: traj.detach().cpu().numpy() for agent_id, traj in actions.items()}
        obs, reward, terminated, truncated, info = env.step(action_np)
        agent_id = next(iter(reward.keys()))
        episode_reward += float(reward[agent_id])
        episode_length += 1
        final_info = info.get(agent_id, {})
        done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
        if max_steps is not None and episode_length >= int(max_steps):
            done = True

        overlay = {
            "episode": episode_idx,
            "step": episode_length,
            "reward": f"{episode_reward:.2f}",
            "arrive_dest": bool(final_info.get("arrive_dest", False)),
            "crash": bool(final_info.get("crash", False) or final_info.get("crash_vehicle", False)),
        }
        if render:
            env.render(text=overlay)
        if recorder is not None:
            recorder.write(render_frame(env, overlay))

    return {
        "episode": episode_idx,
        "reward": episode_reward,
        "length": episode_length,
        "success": bool(final_info.get("arrive_dest", False)),
        "crash": bool(final_info.get("crash", False) or final_info.get("crash_vehicle", False)),
        "out_of_road": bool(final_info.get("out_of_road", False)),
        "wall_time": time.time() - start,
    }


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    checkpoint_path = resolve_rl_checkpoint(args.checkpoint)
    output_dir = resolve_output_dir(args.output_dir, checkpoint_path)
    config = load_eval_config(args)

    print(f"[eval] checkpoint={checkpoint_path}")
    print(f"[eval] config={args.config}")
    print(f"[eval] device={device}")
    print(f"[eval] output_dir={output_dir}")

    dummy_config = build_transfuser_config("small", plan_anchor_path="metadrive/exp_dataset/anchors.npy")
    dummy_model = PlatoonDiffusionPlanner(dummy_config, num_vehicles=int(config.get("num_agents", 3)))
    rl_payload = load_rl_checkpoint(checkpoint_path, dummy_model)
    del dummy_model
    single_ckpt_path = resolve_single_ckpt(args, rl_payload)
    model = build_planner(config, single_ckpt_path, device)
    load_rl_checkpoint(checkpoint_path, model)

    env = build_env(config, render=bool(args.render))
    episode_results = []
    video_paths = []
    try:
        for episode_idx in range(int(args.episodes)):
            recorder = None
            if bool(args.save_video):
                video_path = output_dir / f"episode_{episode_idx:03d}.mp4"
                recorder = VideoRecorder(video_path)
                video_paths.append(str(video_path))
            try:
                result = run_episode(
                    env=env,
                    model=model,
                    device=device,
                    render=bool(args.render),
                    recorder=recorder,
                    episode_idx=episode_idx,
                    max_steps=args.max_steps,
                )
            finally:
                if recorder is not None:
                    recorder.close()
            episode_results.append(result)
            print(
                f"[episode={episode_idx}] reward={result['reward']:.2f} length={result['length']} "
                f"success={result['success']} crash={result['crash']} out_of_road={result['out_of_road']}"
            )
    finally:
        env.close()

    num_episodes = max(len(episode_results), 1)
    summary = {
        "checkpoint": str(checkpoint_path),
        "single_ckpt": str(single_ckpt_path),
        "config": str(Path(args.config)),
        "output_dir": str(output_dir),
        "env": {
            "num_agents": int(config.get("num_agents", 3)),
            "traffic_density": float(config.get("traffic_density", 0.04)),
            "horizon": int(config.get("horizon", 100)),
            "use_hybrid_map": bool(config.get("use_hybrid_map", True)),
            "hybrid_map_sequence": str(config.get("hybrid_map_sequence", "SSXCOCSS")),
            "hazard_scenario": config.get("hazard_scenario"),
        },
        "episodes": episode_results,
        "success_rate": float(sum(int(item["success"]) for item in episode_results) / num_episodes),
        "crash_rate": float(sum(int(item["crash"]) for item in episode_results) / num_episodes),
        "out_of_road_rate": float(sum(int(item["out_of_road"]) for item in episode_results) / num_episodes),
        "avg_reward": float(np.mean([item["reward"] for item in episode_results])) if episode_results else 0.0,
        "avg_length": float(np.mean([item["length"] for item in episode_results])) if episode_results else 0.0,
        "videos": video_paths,
    }
    write_summary(output_dir / "summary.json", summary)
    print(
        f"[summary] success_rate={summary['success_rate']:.3f} crash_rate={summary['crash_rate']:.3f} "
        f"out_of_road_rate={summary['out_of_road_rate']:.3f} avg_reward={summary['avg_reward']:.2f} avg_length={summary['avg_length']:.1f}"
    )
    return summary


if __name__ == "__main__":
    main()
