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
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.selector_platoon_env import SelectorPlatoonEnv
from train.train_selector import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SINGLE_CKPT,
    build_env_config,
    require_rllib,
)

try:  # pragma: no cover - optional dependency in local unit tests
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.models import ModelCatalog
    from ray.tune.registry import register_env

    _RAY_AVAILABLE = True
except Exception:  # pragma: no cover
    ray = None  # type: ignore
    Algorithm = None  # type: ignore
    ModelCatalog = None  # type: ignore
    register_env = None  # type: ignore
    _RAY_AVAILABLE = False


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a trained RLlib MAPPO selector checkpoint in SelectorPlatoonEnv.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to RLlib checkpoint dir, or 'latest'.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--single-ckpt", type=str, default=None, help="Optional override for frozen single-vehicle planner ckpt.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--render", type=int, choices=(0, 1), default=1)
    parser.add_argument("--save-video", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-agents", type=int, default=None)
    parser.add_argument("--hazard-scenario", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", help="Planner device override for env-side frozen planner.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--explore", type=int, choices=(0, 1), default=0)
    parser.add_argument("--checkpoint-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args(argv)


def resolve_planner_device(device: str) -> str:
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover
            return "cpu"
    return str(device)


def resolve_rl_checkpoint(checkpoint: str, root: str | Path = DEFAULT_OUTPUT_ROOT) -> Path:
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
    ckpts = sorted(
        [child for child in (latest_run / "checkpoints").iterdir() if child.is_dir() and child.name.startswith("checkpoint_")]
    )
    if not ckpts:
        raise FileNotFoundError(f"No RLlib checkpoints found under: {latest_run / 'checkpoints'}")
    return ckpts[-1]


def resolve_run_dir(checkpoint_path: Path) -> Path:
    if checkpoint_path.name.startswith("checkpoint_") and checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent
    return checkpoint_path.parent


def load_run_summary(checkpoint_path: Path) -> dict[str, Any]:
    summary_path = resolve_run_dir(checkpoint_path) / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def resolve_output_dir(requested: str | None, checkpoint_path: Path) -> Path:
    if requested:
        return Path(requested)
    run_name = resolve_run_dir(checkpoint_path).name
    return Path("outputs/rl_eval") / f"{run_name}_{checkpoint_path.name}"


def load_eval_config(args, run_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp) or {}
    if run_summary:
        config.update(dict(run_summary.get("env_config", {}) or {}))
    if args.single_ckpt is not None:
        config["pretrained_ckpt"] = str(args.single_ckpt)
    if args.num_agents is not None:
        config["num_agents"] = int(args.num_agents)
    if args.hazard_scenario is not None:
        config["hazard_scenario"] = str(args.hazard_scenario)
    config["use_render"] = bool(args.render)
    config["planner_device"] = resolve_planner_device(args.device)
    return config


def resolve_pretrained_ckpt(config: dict[str, Any]) -> Path:
    candidate = str(config.get("pretrained_ckpt") or DEFAULT_SINGLE_CKPT)
    path = Path(candidate)
    if not path.exists():
        raise FileNotFoundError(f"Single-vehicle checkpoint not found: {path}")
    return path


def restore_algorithm(checkpoint_path: Path):
    require_rllib()
    if not _RAY_AVAILABLE:
        raise RuntimeError("Ray/RLlib is not available in this environment.")
    from models.selector.rllib_selector_model import IntentSelectorRLlibModel

    register_env("selector_platoon_env", lambda config: SelectorPlatoonEnv(config))
    ModelCatalog.register_custom_model("intent_selector_model", IntentSelectorRLlibModel)
    return Algorithm.from_checkpoint(str(checkpoint_path))


def extract_action_value(action_result: Any) -> int:
    if isinstance(action_result, tuple):
        action_result = action_result[0]
    if isinstance(action_result, np.ndarray):
        if action_result.size != 1:
            raise ValueError(f"Expected scalar selector action, got array with shape {action_result.shape}")
        return int(action_result.reshape(-1)[0])
    return int(action_result)


def render_frame(env: SelectorPlatoonEnv, text: dict[str, Any]) -> np.ndarray:
    base_env = env.base_env
    if getattr(base_env, "top_down_renderer", None) is None:
        base_env.render(text=text, mode="top_down", window=False)
    frame = base_env.render(text=text, mode="top_down", to_image=True)
    if frame is None:
        raise RuntimeError("Top-down render returned no frame")
    return np.asarray(frame)


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


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_overlay(episode_idx: int, step_idx: int, episode_reward: float, last_infos: dict[str, dict[str, Any]]) -> dict[str, Any]:
    selected = {agent_id: info.get("selected_intent") for agent_id, info in last_infos.items()}
    any_crash = any(bool(info.get("crash", False)) for info in last_infos.values())
    team_reward = float(np.mean([float(info.get("team_reward", 0.0)) for info in last_infos.values()])) if last_infos else 0.0
    return {
        "episode": episode_idx,
        "step": step_idx,
        "reward": f"{episode_reward:.2f}",
        "team_reward": f"{team_reward:.2f}",
        "selected_intent": selected,
        "crash": any_crash,
    }


def run_episode(
    env: SelectorPlatoonEnv,
    algo,
    render: bool,
    recorder: VideoRecorder | None,
    episode_idx: int,
    max_steps: int | None = None,
    explore: bool = False,
) -> dict[str, Any]:
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, _ = reset_result
    else:
        obs = reset_result
    if render and hasattr(env.base_env, "switch_to_third_person_view"):
        try:
            env.base_env.switch_to_third_person_view()
        except Exception:
            pass

    done = False
    episode_reward = 0.0
    episode_length = 0
    step_team_rewards: list[float] = []
    selected_intents: list[dict[str, int]] = []
    crash = False
    out_of_road = False

    while not done:
        actions = {}
        for agent_id, agent_obs in obs.items():
            action_result = algo.compute_single_action(
                observation=agent_obs,
                policy_id="shared_selector",
                explore=bool(explore),
            )
            actions[agent_id] = extract_action_value(action_result)

        obs, reward, terminated, truncated, infos = env.step(actions)
        last_infos = env.get_last_step_infos()
        episode_reward += float(np.mean(list(reward.values()))) if reward else 0.0
        episode_length += 1
        step_team_rewards.append(
            float(np.mean([float(info.get("team_reward", 0.0)) for info in last_infos.values()])) if last_infos else 0.0
        )
        selected_intents.append({agent_id: int(info.get("selected_intent", -1)) for agent_id, info in last_infos.items()})
        crash = crash or any(bool(info.get("crash", False)) for info in last_infos.values())
        out_of_road = out_of_road or any(bool(info.get("out_of_road", False)) for info in last_infos.values())

        done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
        if max_steps is not None and episode_length >= int(max_steps):
            done = True

        overlay = build_overlay(episode_idx, episode_length, episode_reward, last_infos)
        if render:
            env.base_env.render(text=overlay)
        if recorder is not None:
            recorder.write(render_frame(env, overlay))

    return {
        "episode": episode_idx,
        "reward": float(episode_reward),
        "length": int(episode_length),
        "success": bool(not crash and not out_of_road),
        "crash": bool(crash),
        "out_of_road": bool(out_of_road),
        "avg_team_reward": float(np.mean(step_team_rewards)) if step_team_rewards else 0.0,
        "selected_intents": selected_intents,
    }


def main(argv=None):
    args = parse_args(argv)
    checkpoint_path = resolve_rl_checkpoint(args.checkpoint, args.checkpoint_root)
    run_summary = load_run_summary(checkpoint_path)
    config = load_eval_config(args, run_summary)
    pretrained_ckpt = resolve_pretrained_ckpt(config)
    output_dir = resolve_output_dir(args.output_dir, checkpoint_path)

    print(f"[eval] checkpoint={checkpoint_path}")
    print(f"[eval] config={args.config}")
    print(f"[eval] single_ckpt={pretrained_ckpt}")
    print(f"[eval] planner_device={config.get('planner_device')}")
    print(f"[eval] output_dir={output_dir}")

    env_cfg = build_env_config(config)
    env_cfg["use_render"] = bool(args.render)
    env = SelectorPlatoonEnv(env_cfg)

    episode_results = []
    video_paths = []
    algo = None
    if _RAY_AVAILABLE:
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=True)
    try:
        algo = restore_algorithm(checkpoint_path)
        for episode_idx in range(int(args.episodes)):
            recorder = None
            if bool(args.save_video):
                video_path = output_dir / f"episode_{episode_idx:03d}.mp4"
                recorder = VideoRecorder(video_path)
                video_paths.append(str(video_path))
            try:
                result = run_episode(
                    env=env,
                    algo=algo,
                    render=bool(args.render),
                    recorder=recorder,
                    episode_idx=episode_idx,
                    max_steps=args.max_steps,
                    explore=bool(args.explore),
                )
            finally:
                if recorder is not None:
                    recorder.close()
            episode_results.append(result)
            print(
                f"[episode={episode_idx}] reward={result['reward']:.2f} length={result['length']} "
                f"success={result['success']} crash={result['crash']} out_of_road={result['out_of_road']} "
                f"avg_team_reward={result['avg_team_reward']:.3f}"
            )
    finally:
        env.close()
        if algo is not None:
            algo.stop()
        if _RAY_AVAILABLE:
            ray.shutdown()

    num_episodes = max(len(episode_results), 1)
    summary = {
        "checkpoint": str(checkpoint_path),
        "single_ckpt": str(pretrained_ckpt),
        "config": str(Path(args.config)),
        "output_dir": str(output_dir),
        "env": {
            "num_agents": int(config.get("num_agents", 3)),
            "traffic_density": float(config.get("traffic_density", 0.04)),
            "horizon": int(config.get("horizon", 100)),
            "use_hybrid_map": bool(config.get("use_hybrid_map", True)),
            "hybrid_map_sequence": str(config.get("hybrid_map_sequence", "SSXCOCSS")),
            "hazard_scenario": config.get("hazard_scenario"),
            "planner_device": str(config.get("planner_device", "cpu")),
        },
        "episodes": episode_results,
        "success_rate": float(sum(int(item["success"]) for item in episode_results) / num_episodes),
        "crash_rate": float(sum(int(item["crash"]) for item in episode_results) / num_episodes),
        "out_of_road_rate": float(sum(int(item["out_of_road"]) for item in episode_results) / num_episodes),
        "avg_reward": float(np.mean([item["reward"] for item in episode_results])) if episode_results else 0.0,
        "avg_length": float(np.mean([item["length"] for item in episode_results])) if episode_results else 0.0,
        "avg_team_reward": float(np.mean([item["avg_team_reward"] for item in episode_results])) if episode_results else 0.0,
        "videos": video_paths,
    }
    write_summary(output_dir / "summary.json", summary)
    print(
        f"[summary] success_rate={summary['success_rate']:.3f} crash_rate={summary['crash_rate']:.3f} "
        f"out_of_road_rate={summary['out_of_road_rate']:.3f} avg_reward={summary['avg_reward']:.2f} "
        f"avg_team_reward={summary['avg_team_reward']:.3f} avg_length={summary['avg_length']:.1f}"
    )
    return summary


if __name__ == "__main__":
    main()
