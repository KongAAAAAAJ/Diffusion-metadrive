from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import matplotlib
matplotlib.use("Agg", force=True)
import numpy as np
import cv2

from envs.mode_selection_sb3_env import ModeSelectionSB3Env
from models.mode_selection.sb3_mode_cls_policy import load_plan_cls_delta, require_sb3
from train.train_mode_cls_sb3 import build_planner, load_config
from metadrive.policy.diffusion_policy.test_transfuser_policy import (
    _build_topdown_world_to_screen_projector,
    _capture_2d_topdown_frame,
    _local_xy_to_world_xy,
)


DEFAULT_CONFIG_PATH = "configs/train/mode_cls_ppo.yaml"
DEFAULT_OUTPUT_ROOT = Path("/media/kong/Elements_SE/Diffusion_Data/outputs/mode_cls_ppo")
_RUN_DIR_RE = re.compile(r"^run_(\d+)$")
_TEST_DIR_RE = re.compile(r"^test_(\d+)$")


def create_next_test_dir(parent_dir: Path) -> Path:
    """Create test_<N+1> under parent_dir, where N is the count of existing test_* dirs."""
    parent_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        int(m.group(1))
        for p in parent_dir.iterdir()
        if p.is_dir() and (m := _TEST_DIR_RE.match(p.name))
    ]
    test_dir = parent_dir / f"test_{max(existing) + 1 if existing else 1}"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir
_AGENT_COLORS = {
    "agent0": (244, 114, 36),
    "agent1": (42, 157, 143),
    "agent2": (67, 97, 238),
    "agent3": (155, 93, 229),
    "agent4": (230, 57, 70),
}
_OTHER_COLOR = (120, 150, 170)
_SELECTED_COLOR = (255, 188, 66)


def _lighten_color(color: tuple[int, int, int], alpha: float = 0.35) -> tuple[int, int, int]:
    """Blend color toward white (alpha=1 → original, alpha=0 → white)."""
    return tuple(int(c * alpha + 255 * (1.0 - alpha)) for c in color)


def find_latest_run_dir(output_root: str | Path) -> Path:
    output_root = Path(output_root)
    candidates = []
    for item in output_root.glob("run_*"):
        if item.is_dir() and (match := _RUN_DIR_RE.match(item.name)):
            candidates.append((int(match.group(1)), item))
    if not candidates:
        raise FileNotFoundError(f"No run_* directory found under {output_root}")
    return sorted(candidates, key=lambda pair: pair[0])[-1][1]


def resolve_ppo_checkpoint(ppo_ckpt: str | Path | None = None, ppo_run_dir: str | Path | None = None) -> Path:
    if ppo_ckpt:
        path = Path(ppo_ckpt)
        if not path.is_file():
            raise FileNotFoundError(f"PPO checkpoint does not exist: {path}")
        return path
    if not ppo_run_dir:
        ppo_run_dir = find_latest_run_dir(DEFAULT_OUTPUT_ROOT)
    path = Path(ppo_run_dir) / "checkpoints" / "final" / "sb3_model.zip"
    if not path.is_file():
        raise FileNotFoundError(f"Final SB3 checkpoint does not exist: {path}")
    return path


def resolve_plan_cls_delta(ppo_ckpt: str | Path | None = None, ppo_run_dir: str | Path | None = None) -> Path:
    if ppo_ckpt:
        path = Path(ppo_ckpt).with_name("plan_cls_branch_delta.pt")
    else:
        if not ppo_run_dir:
            ppo_run_dir = find_latest_run_dir(DEFAULT_OUTPUT_ROOT)
        path = Path(ppo_run_dir) / "checkpoints" / "final" / "plan_cls_branch_delta.pt"
    if not path.is_file():
        raise FileNotFoundError(
            "plan_cls_branch_delta.pt is required when testing with an agent count "
            f"different from the training checkpoint space, but it was not found: {path}"
        )
    return path


def _init_mode_counts(num_agents: int, num_modes: int) -> dict[str, dict[str, int]]:
    return {f"agent{agent_idx}": {str(mode_idx): 0 for mode_idx in range(num_modes)} for agent_idx in range(num_agents)}


def _vehicle_pose(base_env, agent_id: str) -> tuple[np.ndarray, float] | None:
    vehicle = getattr(base_env, "agents", {}).get(agent_id)
    if vehicle is None:
        return None
    position = np.asarray(getattr(vehicle, "position", [0.0, 0.0])[:2], dtype=np.float64)
    heading = getattr(vehicle, "heading_theta", getattr(vehicle, "heading", 0.0))
    return position, float(heading)


def _local_candidates_to_world(
    candidates: np.ndarray,
    ego_position: np.ndarray,
    ego_heading: float,
) -> np.ndarray:
    candidates = np.asarray(candidates, dtype=np.float64)
    world = []
    for candidate in candidates:
        world.append([_local_xy_to_world_xy(point[:2], ego_position, ego_heading) for point in np.asarray(candidate)])
    return np.asarray(world, dtype=np.float64)


def export_candidates_world(env: ModeSelectionSB3Env) -> dict[str, np.ndarray]:
    if env._last_export is None:
        return {}
    candidates = np.asarray(env._last_export["trajectory_candidates"], dtype=np.float64)
    agent_ids = list(env._last_export["agent_ids"])
    base_env = getattr(env, "base_env", env)
    result: dict[str, np.ndarray] = {}
    for agent_idx, agent_id in enumerate(agent_ids):
        pose = _vehicle_pose(base_env, agent_id)
        if pose is None:
            continue
        result[agent_id] = _local_candidates_to_world(candidates[agent_idx], pose[0], pose[1])
    return result


def draw_multivehicle_multimodal_overlay(
    frame: np.ndarray,
    candidates_world: Mapping[str, np.ndarray],
    selected_modes: Mapping[str, int],
    projector: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    canvas = np.asarray(frame, dtype=np.uint8).copy()
    for agent_id, candidates in candidates_world.items():
        candidates_array = np.asarray(candidates, dtype=np.float64)
        if candidates_array.ndim != 3:
            continue
        agent_color = _AGENT_COLORS.get(agent_id, _OTHER_COLOR)
        selected_mode = int(selected_modes.get(agent_id, -1))

        # Pass 1: draw non-selected trajectories first (underneath)
        for mode_idx, candidate in enumerate(candidates_array):
            if int(mode_idx) == selected_mode:
                continue
            points = np.asarray([projector(point[:2]) for point in candidate], dtype=np.int32)
            if points.shape[0] < 2:
                continue
            cv2.polylines(canvas, [points], False, agent_color, 1, cv2.LINE_AA)

        # Pass 2: draw selected trajectory on top
        if selected_mode >= 0 and selected_mode < len(candidates_array):
            points = np.asarray(
                [projector(point[:2]) for point in candidates_array[selected_mode]], dtype=np.int32
            )
            if points.shape[0] >= 2:
                cv2.polylines(canvas, [points], False, _SELECTED_COLOR, 3, cv2.LINE_AA)
                cv2.circle(canvas, tuple(points[-1]), 3, _SELECTED_COLOR, -1, cv2.LINE_AA)

        if selected_mode >= 0:
            cv2.putText(
                canvas,
                f"{agent_id}: mode {selected_mode}",
                (16, 28 + 24 * int(agent_id.replace("agent", "") or 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                agent_color,
                2,
                cv2.LINE_AA,
            )
    return canvas


def _vehicle_footprint_corners(vehicle) -> np.ndarray | None:
    position = np.asarray(getattr(vehicle, "position", [0.0, 0.0])[:2], dtype=np.float64)
    length = float(getattr(vehicle, "LENGTH", 0.0))
    width = float(getattr(vehicle, "WIDTH", 0.0))
    if length <= 0.0 or width <= 0.0:
        return None
    heading = float(getattr(vehicle, "heading_theta", getattr(vehicle, "heading", 0.0)))
    cos_h = float(np.cos(heading))
    sin_h = float(np.sin(heading))
    rotation = np.asarray([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float64)
    local = np.asarray(
        [
            [length / 2.0, width / 2.0],
            [length / 2.0, -width / 2.0],
            [-length / 2.0, -width / 2.0],
            [-length / 2.0, width / 2.0],
        ],
        dtype=np.float64,
    )
    return position[None, :] + local @ rotation.T


def draw_vehicle_footprint_overlay(
    frame: np.ndarray,
    env: ModeSelectionSB3Env,
    projector: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    canvas = np.asarray(frame, dtype=np.uint8).copy()
    for agent_id, vehicle in getattr(env.base_env, "agents", {}).items():
        corners = _vehicle_footprint_corners(vehicle)
        if corners is None:
            continue
        color = _AGENT_COLORS.get(str(agent_id), _OTHER_COLOR)
        points = np.asarray([projector(point) for point in corners], dtype=np.int32)
        cv2.polylines(canvas, [points], True, color, 3, cv2.LINE_AA)
        center = np.asarray(projector(np.asarray(getattr(vehicle, "position", [0.0, 0.0])[:2])), dtype=np.int32)
        cv2.circle(canvas, tuple(center), 4, color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{agent_id} {float(getattr(vehicle, 'LENGTH', 0.0)):.1f}x{float(getattr(vehicle, 'WIDTH', 0.0)):.1f}m",
            tuple(center + np.asarray([8, -8], dtype=np.int32)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def _write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    for frame in frames:
        frame_rgb = np.asarray(frame, dtype=np.uint8)
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    writer.release()


def _save_episode_trajectory_plot(
    output_path: Path,
    episode_records: list[dict[str, Any]],
) -> None:
    if not episode_records:
        return
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for record in episode_records:
        selected_modes = record["selected_modes"]
        for agent_id, candidates in record["candidates_world"].items():
            candidates_array = np.asarray(candidates, dtype=np.float64)
            selected_mode = int(selected_modes.get(agent_id, -1))
            for mode_idx, candidate in enumerate(candidates_array):
                is_selected = int(mode_idx) == selected_mode
                ax.plot(
                    candidate[:, 0],
                    candidate[:, 1],
                    color="#D97706" if is_selected else "#1F6F8B",
                    alpha=0.9 if is_selected else 0.14,
                    linewidth=2.4 if is_selected else 0.8,
                )
                if is_selected:
                    ax.scatter(candidate[-1, 0], candidate[-1, 1], s=18, color="#D97706")
    ax.set_title("SB3 PPO multimodal trajectories (selected modes highlighted)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _save_frame(path: Path, frame: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))


def _termination_reason_from_info(info: Mapping[str, Any]) -> str:
    if bool(info.get("truncated", False)):
        return "truncated"
    safety_flags = dict(info.get("safety_flags", {}) or {})
    reasons = []
    for agent_id in sorted(safety_flags):
        flags = dict(safety_flags.get(agent_id, {}) or {})
        if flags.get("crash", False):
            reasons.append(f"{agent_id}_crash")
        if flags.get("out_of_road", False):
            reasons.append(f"{agent_id}_out_of_road")
    if reasons:
        return "__".join(reasons)
    if bool(info.get("terminated", False)):
        return "terminated"
    return "done"


def evaluate_mode_cls_policy(
    model,
    env: ModeSelectionSB3Env,
    episodes: int,
    max_steps: int,
    deterministic: bool,
    output_dir: Path | None = None,
    save_topdown_frames: bool = False,
    save_trajectory_plot: bool = False,
    save_video: bool = False,
    save_terminal_frame: bool = True,
    video_fps: int = 10,
) -> dict[str, Any]:
    episode_rewards = []
    episode_lengths = []
    invalid_rates = []
    mismatch_count = 0
    action_count = 0
    mode_counts = _init_mode_counts(env.num_agents, env._num_modes)

    for episode_idx in range(int(episodes)):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        video_frames: list[np.ndarray] = []
        episode_records: list[dict[str, Any]] = []
        for _step in range(int(max_steps)):
            action_masks = env.action_masks()
            action, _ = model.predict(obs, deterministic=bool(deterministic), action_masks=action_masks)
            action_arr = np.asarray(action, dtype=np.int64).reshape(-1)
            selected_modes = {f"agent{agent_idx}": int(mode_idx) for agent_idx, mode_idx in enumerate(action_arr.tolist())}
            candidates_world = export_candidates_world(env)
            topdown_frame = None
            overlay_frame = None
            projector = None
            if output_dir is not None and (save_topdown_frames or save_video):
                topdown_frame = _capture_2d_topdown_frame(env.base_env)
                if topdown_frame is not None:
                    projector = _build_topdown_world_to_screen_projector(env.base_env, topdown_frame)
                    if projector is not None:
                        overlay_frame = draw_multivehicle_multimodal_overlay(
                            topdown_frame,
                            candidates_world,
                            selected_modes,
                            projector,
                        )
                    if save_topdown_frames:
                        marked_topdown = overlay_frame if overlay_frame is not None else topdown_frame
                        _save_frame(
                            output_dir / "topdown_frames" / f"episode_{episode_idx:03d}" / f"step_{steps:05d}.png",
                            marked_topdown,
                        )
                    if save_video and overlay_frame is not None:
                        video_frames.append(overlay_frame)
            if save_trajectory_plot and output_dir is not None and candidates_world:
                episode_records.append({"candidates_world": candidates_world, "selected_modes": selected_modes})

            obs, reward, terminated, truncated, info = env.step(action_arr)
            total_reward += float(reward)
            steps += 1

            pretrained = np.asarray(info.get("pretrained_argmax_mode", []), dtype=np.int64).reshape(-1)
            if pretrained.shape == action_arr.shape:
                mismatch_count += int(np.sum(action_arr != pretrained))
            action_count += int(action_arr.size)
            invalid_rates.append(float(info.get("invalid_mode_rate", 0.0)))
            for agent_idx, mode_idx in enumerate(action_arr.tolist()):
                mode_counts[f"agent{agent_idx}"][str(int(mode_idx))] = mode_counts[f"agent{agent_idx}"].get(str(int(mode_idx)), 0) + 1
            if bool(terminated) or bool(truncated):
                if output_dir is not None and save_terminal_frame:
                    terminal_frame = _capture_2d_topdown_frame(env.base_env)
                    if terminal_frame is not None:
                        reason = _termination_reason_from_info(info)
                        terminal_dir = output_dir / "topdown_frames" / f"episode_{episode_idx:03d}"
                        raw_terminal_path = terminal_dir / f"step_{steps:05d}_terminal_{reason}_raw.png"
                        _save_frame(raw_terminal_path, terminal_frame)
                        terminal_projector = _build_topdown_world_to_screen_projector(env.base_env, terminal_frame)
                        if terminal_projector is not None:
                            footprint_frame = draw_vehicle_footprint_overlay(
                                terminal_frame,
                                env,
                                terminal_projector,
                            )
                            footprint_path = terminal_dir / f"step_{steps:05d}_terminal_{reason}_footprint.png"
                            _save_frame(footprint_path, footprint_frame)
                            terminal_frame = draw_multivehicle_multimodal_overlay(
                                terminal_frame,
                                candidates_world,
                                selected_modes,
                                terminal_projector,
                            )
                        terminal_path = (
                            terminal_dir / f"step_{steps:05d}_terminal_{reason}_overlay.png"
                        )
                        _save_frame(terminal_path, terminal_frame)
                        if save_video:
                            video_frames.append(terminal_frame)
                break
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if output_dir is not None and save_video:
            _write_video(output_dir / "videos" / f"episode_{episode_idx:03d}.mp4", video_frames, video_fps)
        if output_dir is not None and save_trajectory_plot:
            _save_episode_trajectory_plot(
                output_dir / "trajectory_plots" / f"episode_{episode_idx:03d}.png",
                episode_records,
            )

    rewards = np.asarray(episode_rewards, dtype=np.float32)
    return {
        "episodes": int(episodes),
        "total_steps": int(np.sum(episode_lengths)),
        "episode_reward_mean": float(rewards.mean()) if rewards.size else 0.0,
        "episode_reward_std": float(rewards.std()) if rewards.size else 0.0,
        "episode_reward_min": float(rewards.min()) if rewards.size else 0.0,
        "episode_reward_max": float(rewards.max()) if rewards.size else 0.0,
        "episode_length_mean": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "invalid_mode_rate": float(np.mean(invalid_rates)) if invalid_rates else 0.0,
        "argmax_mismatch_rate": float(mismatch_count / action_count) if action_count else 0.0,
        "mode_selection_counts": mode_counts,
    }


def build_test_env(config: Mapping[str, Any], output_dir: Path) -> ModeSelectionSB3Env:
    env_config = dict(config.get("env_config", {}))
    planner = build_planner(config, env_config)
    env_config["planner"] = planner
    env_config["reward_config"] = dict(config.get("reward_config", {}))
    env_config.setdefault("num_agents", int(config.get("num_agents", 3)))
    env_config.setdefault("num_modes", int(config.get("num_modes", 16)))
    env_config.setdefault("cls_feature_dim", int(config.get("cls_feature_dim", 160)))
    env_config.setdefault("global_state_dim", int(config.get("global_state_dim", 60)))
    env_config["debug_log_path"] = str(output_dir / "ppo_test_debug.jsonl")
    return ModeSelectionSB3Env(env_config)


def _build_policy_kwargs(config: Mapping[str, Any], env: ModeSelectionSB3Env) -> dict[str, Any]:
    plan_cls_branch = copy.deepcopy(env.planner.model._trajectory_head.diff_decoder.layers[-1].task_decoder.plan_cls_branch)
    return {
        "plan_cls_branch": plan_cls_branch,
        "num_agents": int(env.num_agents),
        "num_modes": int(config.get("num_modes", env._num_modes)),
        "cls_feature_dim": int(config.get("cls_feature_dim", env._cls_feature_dim)),
        "global_state_dim": int(config.get("global_state_dim", env._global_state_dim)),
        "lambda_kl": float(config.get("lambda_kl", 0.02)),
    }


def _spaces_mismatch(exc: ValueError) -> bool:
    message = str(exc)
    return "Observation spaces do not match" in message or "Action spaces do not match" in message


def load_mode_cls_model_for_test(
    config: Mapping[str, Any],
    env: ModeSelectionSB3Env,
    ppo_ckpt: Path,
    ppo_run_dir: str | Path | None = None,
):
    MaskablePPO, Policy = require_sb3()
    policy_kwargs = _build_policy_kwargs(config, env)
    try:
        return MaskablePPO.load(
            str(ppo_ckpt),
            env=env,
            device="cpu",
            custom_objects={"policy_class": Policy, "policy_kwargs": policy_kwargs},
        )
    except ValueError as exc:
        if not _spaces_mismatch(exc):
            raise

        delta_path = resolve_plan_cls_delta(ppo_ckpt, ppo_run_dir)
        print(
            "[mode_cls_ppo_test] SB3 checkpoint space differs from current test env; "
            f"rebuilding a {env.num_agents}-agent MaskablePPO wrapper and loading shared actor delta: {delta_path}",
            flush=True,
        )
        model = MaskablePPO(
            Policy,
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=float(config.get("learning_rate", 3e-4)),
            n_steps=max(2, int(config.get("n_steps", 32))),
            batch_size=max(2, int(config.get("batch_size", 32))),
            n_epochs=max(1, int(config.get("n_epochs", 4))),
            gamma=float(config.get("gamma", 0.99)),
            verbose=0,
            device="cpu",
        )
        load_plan_cls_delta(model.policy.mode_cls_core.plan_cls_branch, delta_path)
        return model


def build_pretrained_mode_cls_model_for_test(config: Mapping[str, Any], env: ModeSelectionSB3Env):
    """Build an SB3-compatible policy wrapper using the frozen pretrained cls head."""
    MaskablePPO, Policy = require_sb3()
    policy_kwargs = _build_policy_kwargs(config, env)
    return MaskablePPO(
        Policy,
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=float(config.get("learning_rate", 3e-4)),
        n_steps=max(2, int(config.get("n_steps", 32))),
        batch_size=max(2, int(config.get("batch_size", 32))),
        n_epochs=max(1, int(config.get("n_epochs", 4))),
        gamma=float(config.get("gamma", 0.99)),
        verbose=0,
        device="cpu",
    )


def run_test(
    config: Mapping[str, Any],
    ppo_ckpt: Path | None,
    ppo_run_dir: str | Path | None,
    output_dir: Path,
    episodes: int,
    max_steps: int,
    deterministic: bool,
    save_topdown_frames: bool,
    save_trajectory_plot: bool,
    save_video: bool,
    save_terminal_frame: bool,
    video_fps: int,
    policy_source: str = "ppo",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = build_test_env(config, output_dir)
    policy_source = str(policy_source).lower()
    if policy_source == "pretrained":
        model = build_pretrained_mode_cls_model_for_test(config, env)
    elif policy_source == "ppo":
        if ppo_ckpt is None:
            raise ValueError("policy_source=ppo requires a PPO checkpoint.")
        model = load_mode_cls_model_for_test(config, env, ppo_ckpt, ppo_run_dir)
    else:
        raise ValueError(f"Unsupported policy_source: {policy_source}")
    summary = evaluate_mode_cls_policy(
        model,
        env,
        episodes=episodes,
        max_steps=max_steps,
        deterministic=deterministic,
        output_dir=output_dir,
        save_topdown_frames=save_topdown_frames,
        save_trajectory_plot=save_trajectory_plot,
        save_video=save_video,
        save_terminal_frame=save_terminal_frame,
        video_fps=video_fps,
    )
    summary.update(
        {
            "policy_source": policy_source,
            "ppo_checkpoint": str(ppo_ckpt) if ppo_ckpt is not None else "",
            "pretrained_ckpt": str(config.get("pretrained_ckpt", "")),
            "output_dir": str(output_dir),
        }
    )
    (output_dir / "test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Closed-loop test for SB3 MaskablePPO mode-selection policy.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--pretrained-ckpt", default="")
    parser.add_argument("--ppo-ckpt", default="")
    parser.add_argument("--ppo-run-dir", default="")
    parser.add_argument("--policy-source", choices=("ppo", "pretrained"), default="ppo")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--deterministic", type=int, choices=(0, 1), default=1)
    parser.add_argument("--save-topdown-frames", type=int, choices=(0, 1), default=1)
    parser.add_argument("--save-trajectory-plot", type=int, choices=(0, 1), default=1)
    parser.add_argument("--save-video", type=int, choices=(0, 1), default=1)
    parser.add_argument("--save-terminal-frame", type=int, choices=(0, 1), default=1)
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--num-agents", type=int, default=0)
    parser.add_argument("--planner-device", default="")
    parser.add_argument("--use-render", type=int, choices=(0, 1), default=-1)
    parser.add_argument("--scenario-ids", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.pretrained_ckpt:
        config["pretrained_ckpt"] = args.pretrained_ckpt
    if args.num_agents > 0:
        config["num_agents"] = int(args.num_agents)
        config.setdefault("env_config", {})["num_agents"] = int(args.num_agents)
    if args.planner_device:
        config.setdefault("env_config", {})["planner_device"] = args.planner_device
    if args.use_render >= 0:
        config.setdefault("env_config", {})["use_render"] = bool(args.use_render)
    if args.scenario_ids:
        scenario_ids = [item.strip() for item in args.scenario_ids.split(",") if item.strip()]
        if scenario_ids:
            config.setdefault("env_config", {})["scenario_ids"] = scenario_ids

    ppo_run_dir = args.ppo_run_dir or ""
    ppo_ckpt: Path | None = None
    if args.policy_source == "ppo":
        ppo_run_dir = ppo_run_dir or find_latest_run_dir(config.get("output_root", DEFAULT_OUTPUT_ROOT))
        ppo_ckpt = resolve_ppo_checkpoint(args.ppo_ckpt, ppo_run_dir)
        default_parent = Path(ppo_run_dir)
    else:
        default_parent = Path(config.get("output_root", DEFAULT_OUTPUT_ROOT)) / "pretrained"
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = create_next_test_dir(default_parent)
    run_test(
        config,
        ppo_ckpt,
        ppo_run_dir,
        output_dir,
        args.episodes,
        args.max_steps,
        bool(args.deterministic),
        bool(args.save_topdown_frames),
        bool(args.save_trajectory_plot),
        bool(args.save_video),
        bool(args.save_terminal_frame),
        int(args.video_fps),
        args.policy_source,
    )
    print(f"[mode_cls_ppo_test] outputs: {output_dir}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
