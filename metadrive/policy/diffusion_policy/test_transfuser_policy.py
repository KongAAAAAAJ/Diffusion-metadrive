from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Callable, Mapping
import cv2
import numpy as np
import torch, time
from routes.route_definitions import get_required_preset, get_route_blocks
from scenarios.definitions import SCENARIO_BY_ID, get_scenario_definition
from scenarios.orchestrator import ScenarioOrchestrator
from metadrive.envs.diffusion_envs.base_multi_env import DatasetCollectEnv
from metadrive.policy.diffusion_policy.run_dir_utils import create_numbered_run_dir
from metadrive.policy.diffusion_policy.transfuser_callback import render_closed_loop_prediction
from metadrive.policy.diffusion_policy.transfuser_config import (
    build_transfuser_config,
    diffusion_model_config_to_overrides,
    load_diffusion_model_config,
    resolve_model_config_value,
    transfuser_config_to_dict,
)
from metadrive.policy.diffusion_policy.mode_visualization import mode_color
from metadrive.policy.diffusion_policy.transfuser_policy import TransfuserPolicy, compute_trajectory_control

MULTIMODAL_SELECTED_COLOR = "#C76B00"
MULTIMODAL_OTHER_COLOR = "#1F6F8B"
ROAD_BOUNDARY_COLOR = "#7A7A7A"
ACTUAL_TRAJECTORY_COLOR = "#1D4ED8"
DEFAULT_MODEL_CONFIG_PATH = "configs/diffusion/model.yaml"
DEFAULT_CHECKPOINT_PATH = "/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/run_20/checkpoints/diffusion-epoch=25.ckpt"
DEFAULT_CAMERA_OUTPUT_DIR = "/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/eval/closed/cameras"
DEFAULT_OUTPUT_DIR = "/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/eval/closed"
DEFAULT_PPO_ACTOR_CKPT = "/media/kong/Elements_SE/Diffusion_Data/outputs/mode_cls_ppo/run_4/checkpoints/final/sb3_model.zip"

PAPER_SELECTED_TRAJ_COLOR = (0, 94, 213)  # Okabe-Ito vermillion in BGR
PAPER_OTHER_TRAJ_COLOR = (178, 114, 86)  # sky blue
PAPER_ANCHOR_COLOR = (226, 210, 190)  # desaturated blue-gray
PAPER_TARGET_POINT_COLOR = (86, 180, 230)  # orange/yellow
PAPER_TARGET_LINE_COLOR = (115, 158, 0)  # bluish green
PAPER_TOPOLOGY_COLOR = (204, 121, 167)  # reddish purple
PAPER_MULTI_POINT_COLOR = (204, 121, 167)
PAPER_MULTI_POINT_SELECTED_COLOR = (92, 61, 142)  # deep purple
PAPER_EGO_FILL_COLOR = (115, 158, 0)
PAPER_EGO_EDGE_COLOR = (70, 110, 0)
PAPER_INVALID_MODE_COLOR = (170, 170, 170)


def _level_fraction_from_name(mode_name: str, mode_slot_names: list[str]) -> float:
    prefix = mode_name.rsplit("_", 1)[0]
    group_names = [name for name in mode_slot_names if name.startswith(prefix + "_")]
    try:
        level_index = int(mode_name.rsplit("_", 1)[1])
    except Exception:
        return 1.0
    if len(group_names) <= 1:
        return 1.0
    return 1.0 - float(level_index) / float(len(group_names) - 1)


def _plot_color_for_mode_index(mode_index: int, mode_slot_names: list[str] | None = None) -> tuple[int, int, int]:
    names = mode_slot_names or []
    mode_name = names[int(mode_index)] if 0 <= int(mode_index) < len(names) else ""
    if mode_name:
        base = mode_color(mode_name)
        if base != (255, 255, 255):
            return base
        fraction = _level_fraction_from_name(mode_name, names)
        if mode_name.startswith("KEEP_LEVEL_"):
            return (245, int(120 + 80 * fraction), 11)
        if mode_name.startswith("LEFT_LC_LEVEL_"):
            return (22, int(120 + 100 * fraction), 74)
        if mode_name.startswith("RIGHT_LC_LEVEL_"):
            return (13, int(130 + 100 * fraction), 136)

    # Stable fallback for old records that do not carry mode names.
    palette = (
        (245, 158, 11),
        (234, 88, 12),
        (217, 70, 239),
        (22, 163, 74),
        (34, 197, 94),
        (134, 239, 172),
        (13, 148, 136),
        (16, 185, 129),
        (153, 246, 228),
        (239, 68, 68),
    )
    return palette[int(mode_index) % len(palette)]


@dataclass
class StepTrajectoryPlotRecord:
    step_idx: int
    ego_position: np.ndarray
    selected_trajectory: np.ndarray | None
    multimodal_trajectories: np.ndarray | None
    selected_mode_idx: int | None
    mode_slot_names: list[str] | None = None
    mode_valid_mask: np.ndarray | None = None
    dynamic_anchor_trajectories: np.ndarray | None = None
    multi_point_world: np.ndarray | None = None
    target_point_world: np.ndarray | None = None
    target_line_world: np.ndarray | None = None
    topology_polyline_world: np.ndarray | None = None
    topdown_frame: np.ndarray | None = None
    world_to_screen_projector: Callable[[np.ndarray], np.ndarray] | None = None
    ego_speed_km_h: float | None = None
    ego_acceleration: float | None = None
    env_reward: float | None = None
    agent_label: str | None = None
    peer_records: list["StepTrajectoryPlotRecord"] | None = None


@dataclass
class ControlErrorRecord:
    episode_idx: int
    step_idx: int
    actual_local_x: float
    actual_local_y: float
    reference_local_x: float
    reference_local_y: float
    actual_world_x: float
    actual_world_y: float
    reference_world_x: float
    reference_world_y: float
    longitudinal_error_m: float
    lateral_error_m: float
    abs_longitudinal_error_m: float
    abs_lateral_error_m: float
    speed_error_km_h: float | None = None
    target_speed_km_h: float | None = None
    trajectory_target_speed_km_h: float | None = None
    speed_km_h: float | None = None
    acceleration_mps2: float | None = None
    steering: float | None = None
    throttle: float | None = None
    lookahead_y: float | None = None
    lookahead_heading: float | None = None


@dataclass(frozen=True)
class ScenarioRouteSelection:
    scenario_id: str
    local_route: str
    route_preset: str
    ego_main_route_block_ids: tuple[str, ...]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Closed-loop evaluation for MetaDrive TransFuser.")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT_PATH, help="Path to the trained TransFuser checkpoint.")
    parser.add_argument("--model-config-path", type=str, default=DEFAULT_MODEL_CONFIG_PATH)
    parser.add_argument("--model-size", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--num-agents",
        type=int,
        default=3,
        help="Number of vehicles to evaluate. 1 keeps the original TransfuserPolicy path; >1 uses PlatoonEnv.",
    )
    parser.add_argument("--render", type=int, choices=(0, 1), default=0)
    parser.add_argument("--image-on-cuda", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--target-speed-km-h", type=float, default=30.0)
    parser.add_argument("--lookahead-index", type=int, default=2)
    parser.add_argument("--controller-type", type=str, default="stabilized")
    parser.add_argument("--print-trajectory-debug", type=int, choices=(0, 1), default=1)
    parser.add_argument("--save-camera-interval", type=int, default=0)  # 默认关闭相机图片保存
    parser.add_argument("--camera-output-dir", type=str, default=DEFAULT_CAMERA_OUTPUT_DIR)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-scenarios", type=int, default=1)
    parser.add_argument("--traffic-density", type=float, default=0.06)
    parser.add_argument("--plan-anchor-path", type=str, default=None)
    parser.add_argument("--trajectory-reg-decoder-type", type=str, choices=("mlp", "gru"), default=None)
    parser.add_argument("--target-guidance-type", type=str, choices=("point", "line", "multi_point"), default=None)
    parser.add_argument("--target-line-num-points", type=int, default=None)
    parser.add_argument("--save-3d-video", type=int, choices=(0, 1), default=0)
    parser.add_argument("--save-2d-video", type=int, choices=(0, 1), default=1)
    parser.add_argument("--save-trajectory-plot", type=int, choices=(0, 1), default=1)
    parser.add_argument("--save-step-images", type=int, choices=(0, 1), default=0)
    parser.add_argument("--step-image-interval", type=int, default=1)
    parser.add_argument(
        "--scenario-id",
        type=str,
        default="S2_free_cruise_curve",
        help="Scenario id to evaluate. The script samples one allowed local route per episode.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--topdown-camera-height", type=float, default=80.0)
    parser.add_argument("--show-topology-polyline", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--local-route",
        type=str,
        default="",
        help="Fix evaluation to a specific local route (e.g. R1_entry_straight). "
             "Empty string means the default/random route selection is used.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Hard limit on steps per episode (0 = use the env horizon).",
    )
    parser.add_argument(
        "--selection-policy",
        type=str,
        choices=("argmax", "random_valid"),
        default="argmax",
        help="Mode selection policy for platoon planner backend.",
    )
    parser.add_argument("--random-action-seed", type=int, default=None)
    parser.add_argument("--random-use-mode-endpoint-target", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--ppo-actor-ckpt",
        type=str,
        default="",
        help="SB3 MaskablePPO actor checkpoint (.zip). "
             "Empty string (default) = skip PPO and use the pretrained diffusion planner directly (argmax/random_valid).",
    )
    parser.add_argument("--ppo-run-dir", type=str, default="")
    parser.add_argument("--ppo-deterministic", type=int, choices=(0, 1), default=1)
    return parser.parse_args(argv)


def _normalize_visualization_args(args):
    if args.save_3d_video and not args.render:
        args.render = 1
    if args.random_action_seed is None:
        args.random_action_seed = int(args.start_seed)
    return args


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _choose_mode_indices(
    *,
    masked_logits: np.ndarray,
    mode_valid_mask: np.ndarray,
    policy: str,
    rng: np.random.RandomState,
) -> list[int]:
    """Choose one mode per agent, respecting valid masks."""
    masked_logits = np.asarray(masked_logits, dtype=np.float32)
    mode_valid_mask = np.asarray(mode_valid_mask, dtype=bool)
    if mode_valid_mask.ndim != 2:
        raise ValueError(f"mode_valid_mask must have shape [N,M], got {mode_valid_mask.shape}")
    if policy == "argmax":
        return [int(np.argmax(masked_logits[i])) for i in range(mode_valid_mask.shape[0])]
    if policy != "random_valid":
        raise ValueError(f"Unsupported selection policy: {policy}")
    selected: list[int] = []
    for agent_index, valid in enumerate(mode_valid_mask):
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size == 0:
            raise ValueError(f"No valid trajectory mode for agent index {agent_index}.")
        selected.append(int(rng.choice(valid_indices)))
    return selected


def _resolve_ppo_actor_checkpoint(actor_ckpt: str, run_dir: str) -> "Path | None":
    """Return the resolved PPO actor checkpoint path, or None when neither argument is provided.

    None → use the pretrained diffusion planner directly (argmax / random_valid selection).
    """
    if not actor_ckpt and not run_dir:
        return None
    if actor_ckpt:
        path = Path(actor_ckpt)
    else:
        path = Path(run_dir) / "checkpoints" / "final" / "sb3_model.zip"
    if not path.is_file():
        raise FileNotFoundError(f"PPO actor checkpoint not found: {path}")
    return path


def _build_ppo_agent_relation_states(
    obs: Mapping[str, Mapping],
    agent_ids: list[str],
    relation_dim: int = 12,
) -> np.ndarray:
    rows = []
    for agent_id in agent_ids:
        relation = np.asarray(obs.get(agent_id, {}).get("formation_relation_state", []), dtype=np.float32).reshape(-1)
        if relation.size < relation_dim:
            relation = np.pad(relation, (0, relation_dim - relation.size))
        rows.append(relation[:relation_dim])
    return np.stack(rows, axis=0).astype(np.float32)


def _build_ppo_global_state(obs: Mapping[str, Mapping], agent_ids: list[str], global_state_dim: int = 60) -> np.ndarray:
    parts = []
    for agent_id in sorted(agent_ids):
        item = obs.get(agent_id, {})
        parts.append(np.asarray(item.get("status", []), dtype=np.float32).reshape(-1))
        parts.append(np.asarray(item.get("formation_relation_state", []), dtype=np.float32).reshape(-1))
    state = np.concatenate(parts, axis=0) if parts else np.zeros((0,), dtype=np.float32)
    if state.size < global_state_dim:
        state = np.pad(state, (0, global_state_dim - state.size))
    return state[:global_state_dim].astype(np.float32)


def _missing_controlled_agents(obs: Mapping[str, Mapping] | None, expected_agent_ids: list[str]) -> list[str]:
    obs = obs or {}
    return [agent_id for agent_id in expected_agent_ids if agent_id not in obs]


def _build_ppo_actor_obs(
    *,
    obs: Mapping[str, Mapping],
    agent_ids: list[str],
    policy_agent_ids: list[str],
    candidates: np.ndarray,
    mode_valid_mask: np.ndarray,
    raw_logits: np.ndarray,
) -> dict[str, np.ndarray]:
    candidates = np.asarray(candidates, dtype=np.float32)
    mode_valid_mask = np.asarray(mode_valid_mask, dtype=bool)
    raw_logits = np.asarray(raw_logits, dtype=np.float32)
    if candidates.ndim != 4:
        raise ValueError(f"trajectory_candidates must have shape [N,M,8,3], got {candidates.shape}")
    num_modes = int(candidates.shape[1])
    full_candidates = np.zeros((len(policy_agent_ids), num_modes, 8, 3), dtype=np.float32)
    full_masks = np.zeros((len(policy_agent_ids), num_modes), dtype=bool)
    full_logits = np.zeros((len(policy_agent_ids), num_modes), dtype=np.float32)
    # Missing vehicles still need one valid dummy action so SB3's masked
    # distribution remains well-defined. Their sampled action is ignored.
    full_masks[:, 0] = True
    active_index_by_id = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}
    for policy_idx, agent_id in enumerate(policy_agent_ids):
        active_idx = active_index_by_id.get(agent_id)
        if active_idx is None:
            continue
        full_candidates[policy_idx] = candidates[active_idx]
        full_masks[policy_idx] = mode_valid_mask[active_idx]
        full_logits[policy_idx] = raw_logits[active_idx]
    return {
        "agent_relation_states": _build_ppo_agent_relation_states(obs, policy_agent_ids),
        "trajectory_candidates": full_candidates,
        "agent_mode_masks": full_masks,
        "pretrained_logits": full_logits,
        "global_state": _build_ppo_global_state(obs, policy_agent_ids),
    }


def _predict_ppo_modes(model, actor_obs: dict[str, np.ndarray], mode_valid_mask: np.ndarray, deterministic: bool) -> list[int]:
    action_masks = np.asarray(mode_valid_mask, dtype=bool).reshape(-1)
    action, _ = model.predict(actor_obs, deterministic=deterministic, action_masks=action_masks)
    action_arr = np.asarray(action, dtype=np.int64).reshape(-1)
    if action_arr.shape[0] != np.asarray(mode_valid_mask).shape[0]:
        raise ValueError(f"PPO actor returned {action_arr.shape[0]} actions, expected {np.asarray(mode_valid_mask).shape[0]}")
    return [int(value) for value in action_arr.tolist()]


def _coerce_optional_point(value) -> list[float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size < 2:
        return None
    return [float(array[0]), float(array[1])]


def _apply_selected_mode_target_overrides(
    planner_batch: dict[str, dict],
    *,
    agent_ids: list[str],
    selected_modes: list[int],
    coarse_by_agent: dict[str, np.ndarray],
) -> dict[str, dict]:
    """Write selected dynamic-anchor endpoints as target/preference points."""
    metadata: dict[str, dict] = {}
    for agent_id, mode_idx in zip(agent_ids, selected_modes):
        if agent_id not in planner_batch:
            continue
        coarse = np.asarray(coarse_by_agent[agent_id], dtype=np.float32)
        endpoint = np.asarray(coarse[int(mode_idx), -1, :2], dtype=np.float32)
        before = _coerce_optional_point(planner_batch[agent_id].get("target_point"))
        planner_batch[agent_id]["target_point"] = endpoint.copy()
        planner_batch[agent_id]["preference_point"] = endpoint.copy()
        metadata[agent_id] = {
            "target_point_before": before,
            "target_point_after": endpoint.astype(float).tolist(),
            "selected_coarse_endpoint": endpoint.astype(float).tolist(),
        }
    return metadata


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _summarize_random_action_rewards(
    records: list[dict],
    *,
    episodes: int,
    success: int,
    crash: int,
    out_of_road: int,
    metadata: dict,
) -> dict:
    by_mode: dict[int, dict[str, list[float] | int]] = {}
    episode_env_rewards: dict[int, list[float]] = {}
    for record in records:
        episode_idx = int(record.get("episode", 0))
        episode_env_rewards.setdefault(episode_idx, []).append(float(record.get("env_reward_mean", 0.0)))
        selected_modes = dict(record.get("selected_mode", {}))
        env_rewards = dict(record.get("env_reward", {}))
        for agent_id, mode_idx in selected_modes.items():
            mode_i = int(mode_idx)
            bucket = by_mode.setdefault(
                mode_i,
                {"selected_count": 0, "env_reward": []},
            )
            bucket["selected_count"] = int(bucket["selected_count"]) + 1
            bucket["env_reward"].append(float(env_rewards.get(agent_id, 0.0)))  # type: ignore[union-attr]

    per_mode = {}
    for mode_i, bucket in sorted(by_mode.items()):
        env_values = list(bucket["env_reward"])  # type: ignore[arg-type]
        per_mode[str(mode_i)] = {
            "selected_count": int(bucket["selected_count"]),
            "env_reward_mean": _safe_mean(env_values),
            "env_reward_std": _safe_std(env_values),
        }

    # Normalize episode reward by step count → reward_per_step
    episode_per_step = [
        float(sum(values)) / max(len(values), 1)
        for _, values in sorted(episode_env_rewards.items())
    ]
    num_episodes = max(int(episodes), 1)
    return {
        "metadata": dict(metadata),
        "num_records": len(records),
        "episode_env_reward_per_step_mean": _safe_mean(episode_per_step),
        "episode_env_reward_per_step_std": _safe_std(episode_per_step),
        "episode_env_reward_per_step_min": float(min(episode_per_step)) if episode_per_step else 0.0,
        "episode_env_reward_per_step_max": float(max(episode_per_step)) if episode_per_step else 0.0,
        "success_rate": float(success) / num_episodes,
        "crash_rate": float(crash) / num_episodes,
        "out_of_road_rate": float(out_of_road) / num_episodes,
        "per_mode": per_mode,
    }


def _extract_state_dict(checkpoint_obj):
    return checkpoint_obj.get("state_dict", checkpoint_obj)


def _to_uint8_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def save_triplet_cameras(observation: dict, output_dir: Path, episode_idx: int, step_idx: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for obs_key, suffix in (("rgb_left", "left"), ("rgb_front", "front"), ("rgb_right", "right")):
        if obs_key not in observation:
            continue
        image = _to_uint8_image(observation[obs_key])
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_path = output_dir / f"ep{episode_idx:03d}_step{step_idx:05d}_{suffix}.png"
        cv2.imwrite(str(image_path), image_bgr)


def _capture_3d_topdown_frame(env, camera_height: float) -> np.ndarray | None:
    engine = getattr(env, "engine", None)
    main_camera = getattr(engine, "main_camera", None)
    if main_camera is None:
        return None
    agent_id = next(iter(getattr(env, "agents", {}).keys()), None)
    if agent_id is None:
        return None
    ego_pos = env.agents[agent_id].position
    frame = main_camera.perceive(
        to_float=False,
        new_parent_node=engine.origin,
        position=(float(ego_pos[0]), float(ego_pos[1]), float(camera_height)),
        hpr=(0, -90, 0),
    )
    if frame is not None and frame.ndim == 3 and frame.shape[2] > 3:
        frame = frame[:, :, :3]
    return frame


def _capture_2d_topdown_frame(env, screen_size: int = 800, film_size: int = 3000) -> np.ndarray | None:
    agent_id = next(iter(getattr(env, "agents", {}).keys()), None)
    if agent_id is None:
        return None
    ego_pos = env.agents[agent_id].position
    renderer = getattr(env, "top_down_renderer", None)
    if renderer is not None:
        renderer.position = (float(ego_pos[0]), float(ego_pos[1]))
    main_camera = getattr(getattr(env, "engine", None), "main_camera", None)
    original_track_agent = getattr(main_camera, "current_track_agent", None)
    if main_camera is not None:
        # The native topdown renderer draws a single "EGO" callout for the
        # tracked camera agent.  Step plots add explicit EGO1/EGO2/... labels,
        # so suppress the native callout only while capturing this frame.
        main_camera.current_track_agent = None
    try:
        frame = env.render(
            mode="top_down",
            window=False,
            screen_size=(screen_size, screen_size),
            film_size=(film_size, film_size),
            target_agent_heading_up=False,
            camera_position=(float(ego_pos[0]), float(ego_pos[1])),
        )
    finally:
        if main_camera is not None:
            main_camera.current_track_agent = original_track_agent
    if frame is None:
        return None
    frame_array = np.asarray(frame)
    if frame_array.ndim == 3 and frame_array.shape[2] > 3:
        frame_array = frame_array[:, :, :3]
    return frame_array


def _get_primary_agent_id(env) -> str | None:
    agents = getattr(env, "agents", {}) or {}
    if "agent0" in agents:
        return "agent0"
    return next(iter(agents.keys()), None)


def _extract_road_topology(env) -> list[np.ndarray]:
    try:
        road_network = env.engine.current_map.road_network
        all_lanes = road_network.get_all_lanes()
    except Exception:
        return []

    boundaries = []
    for lane in all_lanes:
        try:
            lane_width = getattr(lane, "width", None)
            if lane_width is None:
                lane_width = lane.width_at(0)
            half_width = float(lane_width) / 2.0
            left = np.asarray(lane.get_polyline(interval=2, lateral=half_width), dtype=np.float64)
            right = np.asarray(lane.get_polyline(interval=2, lateral=-half_width), dtype=np.float64)
            if left.ndim == 2 and left.shape[0] >= 2:
                boundaries.append(left)
            if right.ndim == 2 and right.shape[0] >= 2:
                boundaries.append(right)
        except Exception:
            continue
    return boundaries


def _collect_plot_points(
    actual_positions: list[np.ndarray],
    planned_trajectories: list[tuple[int, list[np.ndarray]]],
    multimodal_trajectories: list[tuple[int, np.ndarray, int]],
) -> np.ndarray:
    points = []
    for pos in actual_positions:
        pos_array = np.asarray(pos, dtype=np.float64)
        if pos_array.ndim == 1 and pos_array.shape[0] == 2:
            points.append(pos_array.reshape(1, 2))
        elif pos_array.ndim == 2 and pos_array.shape[1] == 2:
            points.append(pos_array)
    for _, planned_world in planned_trajectories:
        planned_array = np.asarray(planned_world, dtype=np.float64)
        if planned_array.ndim == 2 and planned_array.shape[0] > 0:
            points.append(planned_array)
    for _, candidates_world, _ in multimodal_trajectories:
        candidates_array = np.asarray(candidates_world, dtype=np.float64)
        if candidates_array.ndim == 3 and candidates_array.shape[0] > 0:
            points.append(candidates_array.reshape(-1, 2))
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    return np.concatenate(points, axis=0)


def _filter_road_boundaries_near_points(
    road_boundaries: list[np.ndarray] | None,
    points: np.ndarray,
    road_margin: float = 6.0,
) -> list[np.ndarray]:
    if not road_boundaries:
        return []
    points_array = np.asarray(points, dtype=np.float64)
    if points_array.ndim != 2 or points_array.shape[0] == 0:
        return []

    search_min_xy = points_array.min(axis=0) - float(road_margin)
    search_max_xy = points_array.max(axis=0) + float(road_margin)
    filtered_boundaries = []
    for boundary in road_boundaries:
        boundary_array = np.asarray(boundary, dtype=np.float64)
        if boundary_array.ndim != 2 or boundary_array.shape[0] < 2:
            continue
        boundary_min_xy = boundary_array.min(axis=0)
        boundary_max_xy = boundary_array.max(axis=0)
        intersects = not (
            boundary_max_xy[0] < search_min_xy[0]
            or boundary_min_xy[0] > search_max_xy[0]
            or boundary_max_xy[1] < search_min_xy[1]
            or boundary_min_xy[1] > search_max_xy[1]
        )
        if intersects:
            filtered_boundaries.append(boundary_array)
    return filtered_boundaries


def _draw_road_boundaries(ax, road_boundaries: list[np.ndarray]) -> None:
    for boundary in road_boundaries:
        boundary_array = np.asarray(boundary, dtype=np.float64)
        if boundary_array.ndim == 2 and boundary_array.shape[0] >= 2:
            ax.plot(
                boundary_array[:, 0],
                boundary_array[:, 1],
                "-",
                color=ROAD_BOUNDARY_COLOR,
                linewidth=1.2,
                alpha=0.85,
                zorder=1,
            )


def _draw_multimodal_trajectories(
    ax,
    origin: np.ndarray,
    candidates_world: np.ndarray,
    selected_mode_idx: int | None,
    *,
    selected_label: str | None = None,
    other_label: str | None = None,
) -> None:
    candidates_array = np.asarray(candidates_world, dtype=np.float64)
    if candidates_array.ndim != 3:
        return
    origin_array = np.asarray(origin, dtype=np.float64)
    for mode_i in range(candidates_array.shape[0]):
        full_path = np.vstack([origin_array, candidates_array[mode_i]])
        if selected_mode_idx is not None and mode_i == int(selected_mode_idx):
            ax.plot(
                full_path[:, 0],
                full_path[:, 1],
                "--",
                color=MULTIMODAL_SELECTED_COLOR,
                linewidth=2.6,
                alpha=0.95,
                zorder=4,
                label=selected_label,
            )
            selected_label = None
        else:
            ax.plot(
                full_path[:, 0],
                full_path[:, 1],
                "--",
                color=MULTIMODAL_OTHER_COLOR,
                linewidth=1.9,
                alpha=0.65,
                zorder=3,
                label=other_label,
            )
            other_label = None


def _build_topdown_world_to_screen_projector(env, frame: np.ndarray) -> Callable[[np.ndarray], np.ndarray] | None:
    renderer = getattr(env, "top_down_renderer", None)
    if renderer is None or frame is None:
        return None

    screen_size = (int(frame.shape[1]), int(frame.shape[0]))
    reference_world = None
    current_track_agent = getattr(renderer, "current_track_agent", None)
    if current_track_agent is not None:
        reference_world = np.asarray(getattr(current_track_agent, "position", (0.0, 0.0))[:2], dtype=np.float32)
    elif getattr(renderer, "position", None) is not None:
        reference_world = np.asarray(renderer.position[:2], dtype=np.float32)
    else:
        reference_world = np.zeros(2, dtype=np.float32)

    def _project_live(world_point_xy: np.ndarray) -> np.ndarray:
        point = np.asarray(world_point_xy, dtype=np.float32)
        off = None
        if not bool(getattr(renderer, "target_agent_heading_up", False)):
            field = renderer._screen_canvas.get_size()
            if getattr(renderer, "position", None) is not None or getattr(renderer, "current_track_agent", None) is not None:
                if getattr(renderer, "center_on_map", False):
                    frame_canvas_size = renderer._frame_canvas.get_size()
                    position = (frame_canvas_size[0] / 2, frame_canvas_size[1] / 2)
                else:
                    cam_pos = getattr(renderer, "position", None) or tuple(
                        getattr(renderer.current_track_agent, "position", (0.0, 0.0))
                    )
                    position = renderer._frame_canvas.pos2pix(*cam_pos)
            else:
                position = (field[0] / 2, field[1] / 2)
            off = (position[0] - field[0] / 2, position[1] - field[1] / 2)
        projected = np.asarray(renderer._world_to_screen_position(point, off), dtype=np.float32)
        if screen_size != tuple(renderer._screen_canvas.get_size()):
            scale_x = float(screen_size[0]) / float(renderer._screen_canvas.get_size()[0])
            scale_y = float(screen_size[1]) / float(renderer._screen_canvas.get_size()[1])
            projected = np.asarray([projected[0] * scale_x, projected[1] * scale_y], dtype=np.float32)
        return projected

    origin_screen = _project_live(reference_world)
    unit_x_screen = _project_live(reference_world + np.asarray([1.0, 0.0], dtype=np.float32))
    unit_y_screen = _project_live(reference_world + np.asarray([0.0, 1.0], dtype=np.float32))
    basis_x = unit_x_screen - origin_screen
    basis_y = unit_y_screen - origin_screen

    def _project(world_point: np.ndarray) -> np.ndarray:
        point = np.asarray(world_point, dtype=np.float32)
        delta = point[:2] - reference_world
        return np.asarray(
            origin_screen + delta[0] * basis_x + delta[1] * basis_y,
            dtype=np.float32,
        )

    return _project


def _capture_step_plot_render_context(
    env,
    enabled: bool,
) -> tuple[np.ndarray | None, Callable[[np.ndarray], np.ndarray] | None]:
    if not enabled:
        return None, None
    frame = _capture_2d_topdown_frame(env)
    if frame is None:
        return None, None
    return frame, _build_topdown_world_to_screen_projector(env, frame)


def _build_step_trajectory_plot_path(output_dir: Path, episode_idx: int, step_idx: int) -> Path:
    return output_dir / "step_trajectory_plots" / f"episode_{episode_idx:03d}" / f"step_{step_idx:05d}.png"


def _format_step_reward_lines(
    reward_records: list[tuple[str, float | None]],
) -> list[str]:
    env_parts = [
        f"{label}={float(env_reward):+.2f}"
        for label, env_reward in reward_records
        if env_reward is not None
    ]
    lines: list[str] = []
    if env_parts:
        lines.append("env reward: " + " ".join(env_parts))
    return lines


def _local_xy_to_world_xy(
    local_xy: np.ndarray,
    ego_world_position: np.ndarray,
    ego_heading_rad: float,
) -> np.ndarray:
    local_xy = np.asarray(local_xy, dtype=np.float64).reshape(-1)
    ego_world_position = np.asarray(ego_world_position, dtype=np.float64).reshape(-1)
    cos_h = float(np.cos(float(ego_heading_rad)))
    sin_h = float(np.sin(float(ego_heading_rad)))
    return np.asarray(
        [
            ego_world_position[0] + cos_h * local_xy[0] - sin_h * local_xy[1],
            ego_world_position[1] + sin_h * local_xy[0] + cos_h * local_xy[1],
        ],
        dtype=np.float64,
    )


def _world_xy_to_local_xy(
    world_xy: np.ndarray,
    ego_world_position: np.ndarray,
    ego_heading_rad: float,
) -> np.ndarray:
    world_xy = np.asarray(world_xy, dtype=np.float64).reshape(-1)
    ego_world_position = np.asarray(ego_world_position, dtype=np.float64).reshape(-1)
    delta = world_xy[:2] - ego_world_position[:2]
    cos_h = float(np.cos(float(ego_heading_rad)))
    sin_h = float(np.sin(float(ego_heading_rad)))
    return np.asarray(
        [
            cos_h * delta[0] + sin_h * delta[1],
            -sin_h * delta[0] + cos_h * delta[1],
        ],
        dtype=np.float64,
    )


def _interpolate_reference_point_from_trajectory(
    trajectory: np.ndarray,
    elapsed_s: float,
    waypoint_interval_s: float = 0.5,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return the time-aligned reference point and local tangent for one control step."""
    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[0] == 0 or traj.shape[1] < 2:
        return None

    points = np.vstack([np.zeros((1, 2), dtype=np.float64), traj[:, :2]])
    waypoint_interval_s = max(float(waypoint_interval_s), 1e-6)
    elapsed_s = max(float(elapsed_s), 0.0)
    segment_float = elapsed_s / waypoint_interval_s
    segment_idx = int(np.floor(segment_float))
    alpha = float(segment_float - segment_idx)
    segment_idx = min(max(segment_idx, 0), points.shape[0] - 2)
    if segment_idx == points.shape[0] - 2:
        alpha = min(alpha, 1.0)

    start = points[segment_idx]
    end = points[segment_idx + 1]
    segment = end - start
    seg_norm = float(np.linalg.norm(segment))
    if seg_norm < 1e-6:
        tangent = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        tangent = segment / seg_norm
    reference = start + alpha * segment
    return reference.astype(np.float64, copy=False), tangent.astype(np.float64, copy=False)


def _compute_control_error_record(
    *,
    episode_idx: int,
    step_idx: int,
    ego_xy_before_step: np.ndarray | None,
    ego_heading_before_step: float | None,
    ego_xy_after_step: np.ndarray | None,
    final_info: dict,
    step_dt_s: float,
    ego_speed_km_h: float | None,
    ego_accel_mps2: float | None,
) -> ControlErrorRecord | None:
    if ego_xy_before_step is None or ego_heading_before_step is None or ego_xy_after_step is None:
        return None
    predicted_traj = final_info.get("predicted_trajectory")
    reference = _interpolate_reference_point_from_trajectory(predicted_traj, elapsed_s=step_dt_s)
    if reference is None:
        return None

    reference_xy, tangent = reference
    actual_local = _world_xy_to_local_xy(
        np.asarray(ego_xy_after_step, dtype=np.float64),
        np.asarray(ego_xy_before_step, dtype=np.float64),
        float(ego_heading_before_step),
    )
    reference_world = _local_xy_to_world_xy(
        reference_xy,
        np.asarray(ego_xy_before_step, dtype=np.float64),
        float(ego_heading_before_step),
    )
    error_vec = actual_local - reference_xy
    longitudinal_error = float(np.dot(error_vec, tangent))
    lateral_error = float(tangent[0] * error_vec[1] - tangent[1] * error_vec[0])
    controller_debug = final_info.get("controller_debug", {}) or {}
    return ControlErrorRecord(
        episode_idx=int(episode_idx),
        step_idx=int(step_idx),
        actual_local_x=float(actual_local[0]),
        actual_local_y=float(actual_local[1]),
        reference_local_x=float(reference_xy[0]),
        reference_local_y=float(reference_xy[1]),
        actual_world_x=float(np.asarray(ego_xy_after_step, dtype=np.float64).reshape(-1)[0]),
        actual_world_y=float(np.asarray(ego_xy_after_step, dtype=np.float64).reshape(-1)[1]),
        reference_world_x=float(reference_world[0]),
        reference_world_y=float(reference_world[1]),
        longitudinal_error_m=longitudinal_error,
        lateral_error_m=lateral_error,
        abs_longitudinal_error_m=abs(longitudinal_error),
        abs_lateral_error_m=abs(lateral_error),
        speed_error_km_h=(
            float(controller_debug["speed_error"])
            if "speed_error" in controller_debug and controller_debug.get("speed_error") is not None
            else None
        ),
        target_speed_km_h=(
            float(controller_debug["target_speed_km_h"])
            if "target_speed_km_h" in controller_debug and controller_debug.get("target_speed_km_h") is not None
            else None
        ),
        trajectory_target_speed_km_h=(
            float(controller_debug["trajectory_target_speed_km_h"])
            if (
                "trajectory_target_speed_km_h" in controller_debug
                and controller_debug.get("trajectory_target_speed_km_h") is not None
            )
            else None
        ),
        speed_km_h=(None if ego_speed_km_h is None else float(ego_speed_km_h)),
        acceleration_mps2=(None if ego_accel_mps2 is None else float(ego_accel_mps2)),
        steering=(
            float(controller_debug["steering"])
            if "steering" in controller_debug and controller_debug.get("steering") is not None
            else None
        ),
        throttle=(
            float(controller_debug["throttle"])
            if "throttle" in controller_debug and controller_debug.get("throttle") is not None
            else None
        ),
        lookahead_y=(
            float(controller_debug["waypoint_y"])
            if "waypoint_y" in controller_debug and controller_debug.get("waypoint_y") is not None
            else None
        ),
        lookahead_heading=(
            float(controller_debug["waypoint_heading"])
            if "waypoint_heading" in controller_debug and controller_debug.get("waypoint_heading") is not None
            else None
        ),
    )


def _record_step_visualization(
    ego_before_step,
    ego_xy_before_step: np.ndarray | None,
    ego_heading_before_step: float | None,
    final_info: dict,
    episode_length: int,
    save_trajectory_plot: bool,
    actual_positions: list[np.ndarray],
    planned_trajectories: list[tuple[int, list[np.ndarray]]],
    multimodal_trajectories: list[tuple[int, np.ndarray, int]],
    step_plot_records: list[StepTrajectoryPlotRecord] | None = None,
    topdown_frame: np.ndarray | None = None,
    world_to_screen_projector: Callable[[np.ndarray], np.ndarray] | None = None,
    ego_accel_mps2: float | None = None,
    agent_label: str | None = None,
) -> None:
    if step_plot_records is None:
        step_plot_records = []
    if ego_xy_before_step is not None:
        actual_positions.append(np.asarray(ego_xy_before_step, dtype=np.float64))

    if not save_trajectory_plot or ego_before_step is None:
        return

    predicted_traj = final_info.get("predicted_trajectory")
    selected_world_array = None
    if predicted_traj is not None:
        planned_world = []
        for wp in np.asarray(predicted_traj):
            planned_world.append(
                _local_xy_to_world_xy(
                    np.asarray([float(wp[0]), float(wp[1])], dtype=np.float64),
                    ego_xy_before_step,
                    float(ego_heading_before_step),
                )
            )
        planned_trajectories.append((episode_length - 1, planned_world))
        selected_world_array = np.asarray(planned_world, dtype=np.float64)

    trajectory_candidates = final_info.get("trajectory_candidates")
    mode_idx = final_info.get("trajectory_mode_idx")
    candidates_world_array = None
    if trajectory_candidates is not None:
        candidates_world = []
        for candidate in np.asarray(trajectory_candidates):
            candidate_world = []
            for wp in np.asarray(candidate):
                candidate_world.append(
                    _local_xy_to_world_xy(
                        np.asarray([float(wp[0]), float(wp[1])], dtype=np.float64),
                        ego_xy_before_step,
                        float(ego_heading_before_step),
                    )
                )
            candidates_world.append(candidate_world)
        candidates_world_array = np.asarray(candidates_world, dtype=np.float64)

    dynamic_anchor_candidates = final_info.get("coarse_trajectories")
    dynamic_anchor_world_array = None
    multi_point_world_array = None
    if dynamic_anchor_candidates is not None:
        anchor_world = []
        for candidate in np.asarray(dynamic_anchor_candidates):
            candidate_world = []
            for wp in np.asarray(candidate):
                candidate_world.append(
                    _local_xy_to_world_xy(
                        np.asarray([float(wp[0]), float(wp[1])], dtype=np.float64),
                        ego_xy_before_step,
                        float(ego_heading_before_step),
                    )
                )
            anchor_world.append(candidate_world)
        dynamic_anchor_world_array = np.asarray(anchor_world, dtype=np.float64)
        if dynamic_anchor_world_array.ndim == 3 and dynamic_anchor_world_array.shape[1] > 0:
            multi_point_world_array = dynamic_anchor_world_array[:, -1, :2].copy()

    target_point = final_info.get("target_point")
    target_point_world = None
    if target_point is not None:
        target_point_xy = np.asarray(target_point, dtype=np.float64).reshape(-1)
        if target_point_xy.size >= 2:
            target_point_world = _local_xy_to_world_xy(
                np.asarray([float(target_point_xy[0]), float(target_point_xy[1])], dtype=np.float64),
                ego_xy_before_step,
                float(ego_heading_before_step),
            )

    target_line = final_info.get("target_line")
    target_line_world = None
    if target_line is not None:
        target_line_xy = np.asarray(target_line, dtype=np.float64)
        if target_line_xy.ndim == 2 and target_line_xy.shape[0] > 0 and target_line_xy.shape[1] >= 2:
            target_line_world = np.asarray(
                [
                    _local_xy_to_world_xy(
                        np.asarray([float(point[0]), float(point[1])], dtype=np.float64),
                        ego_xy_before_step,
                        float(ego_heading_before_step),
                    )
                    for point in target_line_xy
                ],
                dtype=np.float64,
            )

    topology_polyline = final_info.get("topology_polyline")
    topology_polyline_world = None
    if topology_polyline is not None:
        topo_xy = np.asarray(topology_polyline, dtype=np.float64)
        if topo_xy.ndim == 2 and topo_xy.shape[0] > 0 and topo_xy.shape[1] >= 2:
            topology_polyline_world = np.asarray(
                [
                    _local_xy_to_world_xy(
                        np.asarray([float(point[0]), float(point[1])], dtype=np.float64),
                        ego_xy_before_step,
                        float(ego_heading_before_step),
                    )
                    for point in topo_xy
                ],
                dtype=np.float64,
            )

    if selected_world_array is not None or candidates_world_array is not None:
        _ego_speed = float(getattr(ego_before_step, "speed_km_h", 0.0)) if ego_before_step is not None else None
        step_plot_records.append(
            StepTrajectoryPlotRecord(
                step_idx=episode_length,
                ego_position=np.asarray(ego_xy_before_step, dtype=np.float64),
                selected_trajectory=selected_world_array,
                multimodal_trajectories=candidates_world_array,
                selected_mode_idx=(int(mode_idx) if mode_idx is not None else None),
                mode_slot_names=list(final_info.get("mode_slot_names", [])) if final_info.get("mode_slot_names") else None,
                mode_valid_mask=(
                    np.asarray(final_info.get("mode_valid_mask"), dtype=bool)
                    if final_info.get("mode_valid_mask") is not None
                    else None
                ),
                dynamic_anchor_trajectories=dynamic_anchor_world_array,
                multi_point_world=multi_point_world_array,
                target_point_world=target_point_world,
                target_line_world=target_line_world,
                topology_polyline_world=topology_polyline_world,
                topdown_frame=(None if topdown_frame is None else np.asarray(topdown_frame, dtype=np.uint8).copy()),
                world_to_screen_projector=world_to_screen_projector,
                ego_speed_km_h=_ego_speed,
                ego_acceleration=ego_accel_mps2,
                env_reward=(
                    float(final_info["env_reward"])
                    if final_info.get("env_reward") is not None
                    else None
                ),
                agent_label=agent_label,
            )
        )

    if candidates_world_array is None or mode_idx is None:
        return

    multimodal_trajectories.append(
        (episode_length - 1, candidates_world_array, int(mode_idx))
    )


def _write_video(video_path: str, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    import mediapy

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    mediapy.write_video(video_path, frames, fps=fps)


def _save_step_image(
    *,
    final_info: dict,
    config,
    output_dir: Path,
    episode_idx: int,
    step_idx: int,
    anchors: np.ndarray | None,
    overlay_all_anchors: bool,
    metadata_text: list[str] | None = None,
) -> Path | None:
    required_keys = ("camera_feature", "lidar_feature", "predicted_trajectory")
    if any(final_info.get(key) is None for key in required_keys):
        return None

    image = render_closed_loop_prediction(
        features={
            "camera_feature": final_info["camera_feature"],
            "lidar_feature": final_info["lidar_feature"],
            "status_feature": final_info.get("status_feature"),
            "ego_state": final_info.get("ego_state"),
        },
        predictions={
            "trajectory": final_info["predicted_trajectory"],
            "trajectory_mode_idx": final_info.get("trajectory_mode_idx"),
            "trajectory_candidates": final_info.get("trajectory_candidates"),
            "trajectory_mode_logits": final_info.get("trajectory_mode_logits"),
        },
        config=config,
        anchors=anchors,
        overlay_all_anchors=overlay_all_anchors,
        metadata_text=metadata_text,
    )
    episode_dir = output_dir / "step_images" / f"episode_{episode_idx:03d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    output_path = episode_dir / f"step_{step_idx:05d}.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return output_path


def _compute_plot_view_bounds(
    actual_positions: list[np.ndarray],
    planned_trajectories: list[tuple[int, list[np.ndarray]]],
    multimodal_trajectories: list[tuple[int, np.ndarray, int]],
    road_boundaries: list[np.ndarray] | None = None,
    padding: float = 5.0,
    road_margin: float = 4.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    stacked = _collect_plot_points(actual_positions, planned_trajectories, multimodal_trajectories)
    if stacked.size == 0:
        return ((-padding, padding), (-padding, padding))

    nearby_boundaries = _filter_road_boundaries_near_points(road_boundaries, stacked, road_margin=road_margin)
    if nearby_boundaries:
        boundary_stacked = np.concatenate(nearby_boundaries, axis=0)
        stacked = np.concatenate([stacked, boundary_stacked], axis=0)

    min_xy = stacked.min(axis=0) - float(padding)
    max_xy = stacked.max(axis=0) + float(padding)
    return (float(min_xy[0]), float(max_xy[0])), (float(min_xy[1]), float(max_xy[1]))


def _save_step_trajectory_plot(
    *,
    step_record: StepTrajectoryPlotRecord,
    road_boundaries: list[np.ndarray],
    output_path: Path,
    episode_idx: int,
    show_topology_polyline: bool = False,
) -> None:
    if step_record.topdown_frame is None or step_record.world_to_screen_projector is None:
        return

    canvas = np.asarray(step_record.topdown_frame, dtype=np.uint8).copy()
    ego = np.asarray(step_record.ego_position, dtype=np.float64)
    projector = step_record.world_to_screen_projector

    # ── 1. Zoom: crop the canvas to 1/4 area (1/2 side length) centred on ego ──
    ego_screen = np.round(projector(ego)).astype(np.int32)
    h_full, w_full = canvas.shape[:2]
    crop_w = w_full // 2
    crop_h = h_full // 2
    cx, cy = int(ego_screen[0]), int(ego_screen[1])
    x0 = max(0, min(cx - crop_w // 2, w_full - crop_w))
    y0 = max(0, min(cy - crop_h // 2, h_full - crop_h))
    canvas = canvas[y0:y0 + crop_h, x0:x0 + crop_w].copy()
    canvas = cv2.resize(canvas, (w_full, h_full), interpolation=cv2.INTER_LINEAR)
    scale_x = float(w_full) / float(crop_w)
    scale_y = float(h_full) / float(crop_h)
    agent_label_color = (245, 120, 11)  # RGB orange, matching the original topdown EGO label style.

    def _proj_zoomed(world_point: np.ndarray) -> np.ndarray:
        """Project world → zoomed canvas pixel."""
        sp = projector(np.asarray(world_point, dtype=np.float32))
        return np.asarray(
            [(sp[0] - x0) * scale_x, (sp[1] - y0) * scale_y],
            dtype=np.float32,
        )

    def _draw_world_polyline(world_points: np.ndarray, color: tuple[int, int, int], thickness: int, alpha: float = 1.0):
        world_points = np.asarray(world_points, dtype=np.float64)
        if world_points.ndim != 2 or world_points.shape[0] < 2:
            return
        screen_xy = np.asarray([_proj_zoomed(p) for p in world_points], dtype=np.float32)
        polyline = np.round(screen_xy).astype(np.int32).reshape(-1, 1, 2)
        if alpha >= 0.999:
            cv2.polylines(canvas, [polyline], False, color, thickness, lineType=cv2.LINE_AA)
            return
        overlay = canvas.copy()
        cv2.polylines(overlay, [polyline], False, color, thickness, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0.0, dst=canvas)

    def _draw_agent_record(record: StepTrajectoryPlotRecord, *, primary: bool) -> None:
        record_ego = np.asarray(record.ego_position, dtype=np.float64)
        valid_mask = (
            np.asarray(record.mode_valid_mask, dtype=bool)
            if record.mode_valid_mask is not None
            else None
        )
        if record.dynamic_anchor_trajectories is not None:
            for mode_i, anchor in enumerate(np.asarray(record.dynamic_anchor_trajectories, dtype=np.float64)):
                full_path = np.vstack([record_ego, anchor])
                is_valid = valid_mask is None or mode_i >= len(valid_mask) or bool(valid_mask[mode_i])
                anchor_color = (
                    _plot_color_for_mode_index(mode_i, record.mode_slot_names)
                    if is_valid
                    else PAPER_INVALID_MODE_COLOR
                )
                _draw_world_polyline(full_path, anchor_color, 1, alpha=0.24 if is_valid else 0.16)

        if record.multimodal_trajectories is not None:
            candidates_array = np.asarray(record.multimodal_trajectories, dtype=np.float64)
            selected_idx = record.selected_mode_idx
            for mode_i in range(candidates_array.shape[0]):
                if selected_idx is not None and mode_i == int(selected_idx):
                    continue
                full_path = np.vstack([record_ego, candidates_array[mode_i]])
                mode_color_i = _plot_color_for_mode_index(mode_i, record.mode_slot_names)
                _draw_world_polyline(full_path, mode_color_i, 2 if primary else 1, alpha=0.62 if primary else 0.38)
            if selected_idx is not None and selected_idx < candidates_array.shape[0]:
                full_path = np.vstack([record_ego, candidates_array[int(selected_idx)]])
                selected_color = _plot_color_for_mode_index(int(selected_idx), record.mode_slot_names)
                _draw_world_polyline(full_path, selected_color, 4 if primary else 3, alpha=0.98 if primary else 0.82)
        elif record.selected_trajectory is not None:
            selected_path = np.vstack([record_ego, np.asarray(record.selected_trajectory, dtype=np.float64)])
            _draw_world_polyline(selected_path, PAPER_SELECTED_TRAJ_COLOR, 4 if primary else 3, alpha=0.96 if primary else 0.78)

        if show_topology_polyline and record.topology_polyline_world is not None:
            topology_world = np.asarray(record.topology_polyline_world, dtype=np.float64)
            if topology_world.ndim == 2 and topology_world.shape[0] >= 2:
                _draw_world_polyline(topology_world, PAPER_TOPOLOGY_COLOR, 2, alpha=0.74 if primary else 0.48)

        if record.target_line_world is not None:
            target_line_world = np.asarray(record.target_line_world, dtype=np.float64)
            if target_line_world.ndim == 2 and target_line_world.shape[0] >= 1:
                for target_line_point in target_line_world:
                    line_xy = np.round(_proj_zoomed(target_line_point)).astype(np.int32)
                    cv2.circle(
                        canvas,
                        tuple(int(v) for v in line_xy),
                        3,
                        PAPER_TARGET_LINE_COLOR,
                        thickness=-1,
                        lineType=cv2.LINE_AA,
                    )

        if record.target_point_world is not None:
            target_xy = np.round(_proj_zoomed(np.asarray(record.target_point_world, dtype=np.float64))).astype(np.int32)
            cv2.circle(
                canvas,
                tuple(int(v) for v in target_xy),
                5,
                PAPER_TARGET_POINT_COLOR,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                canvas,
                tuple(int(v) for v in target_xy),
                5,
                (45, 90, 140),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        record_ego_zoomed = np.round(_proj_zoomed(record_ego)).astype(np.int32)
        cv2.circle(
            canvas,
            tuple(int(v) for v in record_ego_zoomed),
            7,
            PAPER_EGO_FILL_COLOR,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            canvas,
            tuple(int(v) for v in record_ego_zoomed),
            7,
            PAPER_EGO_EDGE_COLOR,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        label = record.agent_label or "EGO"
        try:
            label_index = max(int("".join(ch for ch in label if ch.isdigit())) - 1, 0)
        except Exception:
            label_index = 0
        cv2.putText(
            canvas,
            label,
            (int(record_ego_zoomed[0]) - 19, int(record_ego_zoomed[1]) - 15 - 14 * label_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (int(record_ego_zoomed[0]) - 18, int(record_ego_zoomed[1]) - 14 - 14 * label_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            agent_label_color,
            2,
            cv2.LINE_AA,
        )

    for peer_record in step_record.peer_records or []:
        _draw_agent_record(peer_record, primary=False)

    if step_record.dynamic_anchor_trajectories is not None:
        valid_mask = (
            np.asarray(step_record.mode_valid_mask, dtype=bool)
            if step_record.mode_valid_mask is not None
            else None
        )
        for mode_i, anchor in enumerate(np.asarray(step_record.dynamic_anchor_trajectories, dtype=np.float64)):
            full_path = np.vstack([ego, anchor])
            is_valid = valid_mask is None or mode_i >= len(valid_mask) or bool(valid_mask[mode_i])
            anchor_color = (
                _plot_color_for_mode_index(mode_i, step_record.mode_slot_names)
                if is_valid
                else PAPER_INVALID_MODE_COLOR
            )
            _draw_world_polyline(full_path, anchor_color, 1, alpha=0.24 if is_valid else 0.16)

    if show_topology_polyline and step_record.topology_polyline_world is not None:
        topology_world = np.asarray(step_record.topology_polyline_world, dtype=np.float64)
        if topology_world.ndim == 2 and topology_world.shape[0] >= 2:
            _draw_world_polyline(topology_world, PAPER_TOPOLOGY_COLOR, 2, alpha=0.74)

    # ── 2. Draw non-selected modes first, selected mode last (on top) ──────────
    selected_mode_idx = step_record.selected_mode_idx
    if step_record.multimodal_trajectories is not None:
        candidates_array = np.asarray(step_record.multimodal_trajectories, dtype=np.float64)
        # Pass 1: non-selected modes
        for mode_i in range(candidates_array.shape[0]):
            if selected_mode_idx is not None and mode_i == int(selected_mode_idx):
                continue
            full_path = np.vstack([ego, candidates_array[mode_i]])
            mode_color_i = _plot_color_for_mode_index(mode_i, step_record.mode_slot_names)
            _draw_world_polyline(full_path, mode_color_i, 2, alpha=0.62)
        # Pass 2: selected mode drawn last → always on top
        if selected_mode_idx is not None and selected_mode_idx < candidates_array.shape[0]:
            full_path = np.vstack([ego, candidates_array[int(selected_mode_idx)]])
            selected_color = _plot_color_for_mode_index(int(selected_mode_idx), step_record.mode_slot_names)
            _draw_world_polyline(full_path, selected_color, 4, alpha=0.98)
    elif step_record.selected_trajectory is not None:
        selected_path = np.vstack([ego, np.asarray(step_record.selected_trajectory, dtype=np.float64)])
        _draw_world_polyline(selected_path, PAPER_SELECTED_TRAJ_COLOR, 4, alpha=0.96)

    # Ego dot (draw after trajectories so it's always on top)
    ego_zoomed = np.round(_proj_zoomed(ego)).astype(np.int32)
    cv2.circle(canvas, tuple(int(v) for v in ego_zoomed), 7, PAPER_EGO_FILL_COLOR, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, tuple(int(v) for v in ego_zoomed), 7, PAPER_EGO_EDGE_COLOR, thickness=1, lineType=cv2.LINE_AA)
    cv2.putText(
        canvas,
        step_record.agent_label or "EGO",
        (int(ego_zoomed[0]) - 19, int(ego_zoomed[1]) - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        step_record.agent_label or "EGO",
        (int(ego_zoomed[0]) - 18, int(ego_zoomed[1]) - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        agent_label_color,
        2,
        cv2.LINE_AA,
    )

    if step_record.target_line_world is not None:
        target_line_world = np.asarray(step_record.target_line_world, dtype=np.float64)
        if target_line_world.ndim == 2 and target_line_world.shape[0] >= 1:
            for target_line_point in target_line_world:
                line_xy = np.round(_proj_zoomed(target_line_point)).astype(np.int32)
                cv2.circle(
                    canvas,
                    tuple(int(v) for v in line_xy),
                    3,
                    PAPER_TARGET_LINE_COLOR,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

    if step_record.target_point_world is not None:
        target_xy = np.round(_proj_zoomed(np.asarray(step_record.target_point_world, dtype=np.float64))).astype(np.int32)
        cv2.circle(
            canvas,
            tuple(int(v) for v in target_xy),
            5,
            PAPER_TARGET_POINT_COLOR,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            canvas,
            tuple(int(v) for v in target_xy),
            5,
            (45, 90, 140),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    # ── 3. Info box with selected mode names ───────────────────────────────────
    def _mode_name_for_record(record: StepTrajectoryPlotRecord) -> str:
        idx = record.selected_mode_idx
        if idx is None:
            return "N/A"
        if record.mode_slot_names and 0 <= int(idx) < len(record.mode_slot_names):
            return record.mode_slot_names[int(idx)]
        try:
            from metadrive.policy.diffusion_policy.mode_definitions import MODE_SLOTS
            return MODE_SLOTS[int(idx)].name
        except Exception:
            return f"mode_{idx}"

    all_agent_records = [step_record] + list(step_record.peer_records or [])

    def _record_sort_key(record: StepTrajectoryPlotRecord) -> int:
        label = record.agent_label or ""
        digits = "".join(ch for ch in label if ch.isdigit())
        return int(digits) if digits else 999

    all_agent_records = sorted(all_agent_records, key=_record_sort_key)

    reward_lines = _format_step_reward_lines(
        [
            (record.agent_label or "EGO", record.env_reward)
            for record in all_agent_records
        ]
    )
    _mode_count = min(len(all_agent_records), 4)
    _mode_text_end_y = 52 + 18 * _mode_count
    _reward_text_end_y = _mode_text_end_y + 6 + 18 * len(reward_lines)
    _speed_text_y = max(118, _reward_text_end_y + 6)
    _legend_y0 = _speed_text_y + 20
    box_x1, box_y1, box_x2, box_y2 = 10, 10, 500, max(180, _legend_y0 + 44)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (250, 250, 250), thickness=-1)
    cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0.0, dst=canvas)
    cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), (210, 214, 220), thickness=1)
    cv2.putText(
        canvas,
        f"Episode {episode_idx}  Step {step_record.step_idx}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1, cv2.LINE_AA,
    )
    # ── 3a. Selected mode labels for all platoon agents ───
    mode_text_y = 52
    for record in all_agent_records[:4]:
        mode_name = _mode_name_for_record(record)
        mode_color = (
            _plot_color_for_mode_index(int(record.selected_mode_idx), record.mode_slot_names)
            if record.selected_mode_idx is not None
            else PAPER_SELECTED_TRAJ_COLOR
        )
        cv2.putText(
            canvas,
            f"{record.agent_label or 'EGO'}: {mode_name}",
            (20, mode_text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, mode_color, 2, cv2.LINE_AA,
        )
        mode_text_y += 18
    reward_text_y = mode_text_y + 6
    for reward_line in reward_lines:
        cv2.putText(
            canvas,
            reward_line,
            (20, reward_text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 70, 30), 1, cv2.LINE_AA,
        )
        reward_text_y += 18

    # ── 3b. Speed and acceleration ───
    _speed_str = f"{step_record.ego_speed_km_h:.1f} km/h" if step_record.ego_speed_km_h is not None else "-- km/h"
    _accel_str = f"{step_record.ego_acceleration:+.2f} m/s\u00b2" if step_record.ego_acceleration is not None else "-- m/s\u00b2"
    speed_text_y = _speed_text_y
    cv2.putText(
        canvas,
        f"speed: {_speed_str}  accel: {_accel_str}",
        (20, speed_text_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (40, 40, 120), 1, cv2.LINE_AA,
    )
    legend_y0 = _legend_y0
    cv2.putText(canvas, "selected: mode color", (20, legend_y0), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, "others: mode colors", (20, legend_y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(canvas, "anchors: same colors, faint", (178, legend_y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(canvas, "target line: green", (20, legend_y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, PAPER_TARGET_LINE_COLOR, 1, cv2.LINE_AA)
    cv2.putText(canvas, "target point: blue", (178, legend_y0 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, PAPER_TARGET_POINT_COLOR, 1, cv2.LINE_AA)
    if show_topology_polyline:
        cv2.putText(canvas, "topology: purple", (20, legend_y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, PAPER_TOPOLOGY_COLOR, 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _save_trajectory_plot(
    actual_positions: list[np.ndarray],
    planned_trajectories: list[tuple[int, list[np.ndarray]]],
    multimodal_trajectories: list[tuple[int, np.ndarray, int]],
    road_boundaries: list[np.ndarray],
    output_path: str,
    episode_idx: int,
    plot_interval: int = 10,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not actual_positions:
        return

    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    trajectory_points = _collect_plot_points(actual_positions, planned_trajectories, multimodal_trajectories)
    nearby_boundaries = _filter_road_boundaries_near_points(road_boundaries, trajectory_points, road_margin=8.0)
    _draw_road_boundaries(ax, nearby_boundaries)

    actual = np.asarray(actual_positions, dtype=np.float64)
    ax.plot(actual[:, 0], actual[:, 1], "-", color=ACTUAL_TRAJECTORY_COLOR, linewidth=2.2, label="Actual", zorder=5)
    ax.scatter(actual[0, 0], actual[0, 1], c="green", s=100, zorder=6, label="Start")
    ax.scatter(actual[-1, 0], actual[-1, 1], c="red", s=100, zorder=6, label="End")

    first_other_modes = True
    first_selected_mode = True
    for step_idx, candidates_world, best_mode_idx in multimodal_trajectories:
        if step_idx % plot_interval != 0:
            continue
        origin_idx = min(max(int(step_idx), 0), len(actual_positions) - 1)
        origin = np.asarray(actual_positions[origin_idx], dtype=np.float64)
        candidates_array = np.asarray(candidates_world, dtype=np.float64)
        if candidates_array.ndim != 3:
            continue
        _draw_multimodal_trajectories(
            ax,
            origin=origin,
            candidates_world=candidates_array,
            selected_mode_idx=int(best_mode_idx),
            selected_label=("Selected mode" if first_selected_mode else None),
            other_label=("Other modes" if first_other_modes else None),
        )
        first_selected_mode = False
        first_other_modes = False

    if not multimodal_trajectories:
        for step_idx, planned_world in planned_trajectories:
            if step_idx % plot_interval != 0:
                continue
            planned = np.asarray(planned_world, dtype=np.float64)
            if planned.size == 0:
                continue
            origin_idx = min(max(int(step_idx), 0), len(actual_positions) - 1)
            origin = np.asarray(actual_positions[origin_idx], dtype=np.float64)
            full_path = np.vstack([origin, planned])
            label = "Planned" if step_idx == 0 else None
            ax.plot(full_path[:, 0], full_path[:, 1], "--", color=MULTIMODAL_SELECTED_COLOR, alpha=0.95, linewidth=2.2, label=label, zorder=4)

    ax.set_title(f"Episode {episode_idx}: Multimodal Trajectories vs Actual (with Road Topology)")
    ax.set_xlabel("X (world)")
    ax.set_ylabel("Y (world)")
    ax.set_aspect("equal", adjustable="box")
    xlim, ylim = _compute_plot_view_bounds(
        actual_positions,
        planned_trajectories,
        multimodal_trajectories,
        road_boundaries=nearby_boundaries,
        road_margin=8.0,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _control_error_records_to_dicts(records: list[ControlErrorRecord]) -> list[dict]:
    return [
        {
            key: value
            for key, value in record.__dict__.items()
            if value is not None
        }
        for record in records
    ]


def _summarize_control_errors(records: list[ControlErrorRecord]) -> dict:
    if not records:
        return {
            "num_steps": 0,
            "mean_abs_lateral_error_m": None,
            "max_abs_lateral_error_m": None,
            "mean_abs_longitudinal_error_m": None,
            "max_abs_longitudinal_error_m": None,
        }
    lateral_abs = np.asarray([r.abs_lateral_error_m for r in records], dtype=np.float64)
    longitudinal_abs = np.asarray([r.abs_longitudinal_error_m for r in records], dtype=np.float64)
    lateral_signed = np.asarray([r.lateral_error_m for r in records], dtype=np.float64)
    longitudinal_signed = np.asarray([r.longitudinal_error_m for r in records], dtype=np.float64)
    return {
        "num_steps": int(len(records)),
        "mean_lateral_error_m": float(lateral_signed.mean()),
        "mean_longitudinal_error_m": float(longitudinal_signed.mean()),
        "mean_abs_lateral_error_m": float(lateral_abs.mean()),
        "max_abs_lateral_error_m": float(lateral_abs.max()),
        "mean_abs_longitudinal_error_m": float(longitudinal_abs.mean()),
        "max_abs_longitudinal_error_m": float(longitudinal_abs.max()),
        "p95_abs_lateral_error_m": float(np.percentile(lateral_abs, 95)),
        "p95_abs_longitudinal_error_m": float(np.percentile(longitudinal_abs, 95)),
    }


def _save_control_error_plots(
    records: list[ControlErrorRecord],
    output_dir: Path,
    episode_idx: int,
) -> None:
    if not records:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    steps = np.asarray([r.step_idx for r in records], dtype=np.int32)
    lateral = np.asarray([r.lateral_error_m for r in records], dtype=np.float64)
    longitudinal = np.asarray([r.longitudinal_error_m for r in records], dtype=np.float64)
    speed = np.asarray(
        [np.nan if r.speed_km_h is None else r.speed_km_h for r in records],
        dtype=np.float64,
    )
    speed_error = np.asarray(
        [np.nan if r.speed_error_km_h is None else r.speed_error_km_h for r in records],
        dtype=np.float64,
    )
    target_speed = np.asarray(
        [np.nan if r.target_speed_km_h is None else r.target_speed_km_h for r in records],
        dtype=np.float64,
    )
    trajectory_target_speed = np.asarray(
        [np.nan if r.trajectory_target_speed_km_h is None else r.trajectory_target_speed_km_h for r in records],
        dtype=np.float64,
    )
    throttle = np.asarray(
        [np.nan if r.throttle is None else r.throttle for r in records],
        dtype=np.float64,
    )
    steering = np.asarray(
        [np.nan if r.steering is None else r.steering for r in records],
        dtype=np.float64,
    )
    acceleration = np.asarray(
        [np.nan if r.acceleration_mps2 is None else r.acceleration_mps2 for r in records],
        dtype=np.float64,
    )
    actual_world = np.asarray([[r.actual_world_x, r.actual_world_y] for r in records], dtype=np.float64)
    reference_world = np.asarray([[r.reference_world_x, r.reference_world_y] for r in records], dtype=np.float64)

    def _save_time_series(
        values: np.ndarray,
        *,
        name: str,
        ylabel: str,
        color: str,
        title: str,
        extra_series: list[tuple[np.ndarray, str, str]] | None = None,
    ) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.plot(steps, values, color=color, linewidth=1.8, label=name)
        if extra_series:
            for series, label, series_color in extra_series:
                ax.plot(steps, series, color=series_color, linewidth=1.2, alpha=0.85, label=label)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / f"episode_{episode_idx:03d}_{name}.png", dpi=150)
        plt.close(fig)

    _save_time_series(
        lateral,
        name="lat_error",
        ylabel="lateral error (m)",
        color="#D97706",
        title=f"Episode {episode_idx}: lateral tracking error",
    )
    _save_time_series(
        longitudinal,
        name="lon_error",
        ylabel="longitudinal error (m)",
        color="#2563EB",
        title=f"Episode {episode_idx}: longitudinal tracking error",
    )
    _save_time_series(
        speed_error,
        name="speed_error",
        ylabel="speed error (km/h)",
        color="#DC2626",
        title=f"Episode {episode_idx}: controller speed error",
        extra_series=[
            (speed, "ego speed km/h", "#059669"),
            (target_speed, "effective target speed km/h", "#7C3AED"),
            (trajectory_target_speed, "trajectory target speed km/h", "#0891B2"),
        ],
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(steps, throttle, color="#DC2626", linewidth=1.7, label="acc/throttle command")
    ax.plot(steps, steering, color="#7C3AED", linewidth=1.7, label="steer command")
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_xlabel("step")
    ax.set_ylabel("control command")
    ax.grid(True, alpha=0.3)
    ax_acc = ax.twinx()
    ax_acc.plot(steps, acceleration, color="#059669", linewidth=1.2, alpha=0.75, label="measured accel m/s^2")
    ax_acc.set_ylabel("measured acceleration (m/s^2)")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_acc.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    ax.set_title(f"Episode {episode_idx}: control commands and measured acceleration")
    fig.tight_layout()
    fig.savefig(output_dir / f"episode_{episode_idx:03d}_control.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(reference_world[:, 0], reference_world[:, 1], "-", color="#F97316", linewidth=2.0, label="time-aligned reference")
    ax.plot(actual_world[:, 0], actual_world[:, 1], "-", color="#2563EB", linewidth=2.0, label="actual after step")
    ax.scatter(reference_world[0, 0], reference_world[0, 1], color="#FDBA74", s=60, label="ref start", zorder=5)
    ax.scatter(actual_world[0, 0], actual_world[0, 1], color="#93C5FD", s=60, label="actual start", zorder=5)
    ax.set_title(f"Episode {episode_idx}: reference vs actual trajectory")
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"episode_{episode_idx:03d}_trajectory_compare.png", dpi=150)
    plt.close(fig)


def _write_control_error_summary(
    *,
    output_dir: Path,
    per_episode_records: dict[int, list[ControlErrorRecord]],
) -> Path:
    all_records = [
        record
        for episode_records in per_episode_records.values()
        for record in episode_records
    ]
    payload = {
        "definition": (
            "At each closed-loop step, the actual ego pose after one control interval "
            "is transformed into the pre-step ego-local frame and compared with the "
            "predicted trajectory reference interpolated at the same elapsed time."
        ),
        "control_step_s": 0.1,
        "waypoint_interval_s": 0.5,
        "overall": _summarize_control_errors(all_records),
        "episodes": {
            str(episode_idx): {
                "summary": _summarize_control_errors(records),
                "records": _control_error_records_to_dicts(records),
            }
            for episode_idx, records in sorted(per_episode_records.items())
        },
    }
    output_path = output_dir / "control_errors" / "control_error_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return output_path


def infer_model_size_from_checkpoint(checkpoint_path: Path) -> str:
# 自动推断模型规模（small/base）
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    for key in (
        "_transfuser_model._query_embedding.weight",
        "agent._transfuser_model._query_embedding.weight",
        "_query_embedding.weight",
    ):
        if key in state_dict:
            query_embedding = state_dict[key]
            if tuple(query_embedding.shape) == (31, 256):
                return "base"
            if tuple(query_embedding.shape) == (17, 128):
                return "small"
    raise ValueError(
        f"Unable to infer model size from checkpoint: {checkpoint_path}. "
        "Expected one of the known _query_embedding shapes for small/base."
    )


def _model_overrides_from_args(args, model_config: dict) -> dict:
    overrides = diffusion_model_config_to_overrides(model_config)
    if args.plan_anchor_path is not None:
        overrides["plan_anchor_path"] = args.plan_anchor_path
    if args.trajectory_reg_decoder_type is not None:
        overrides["trajectory_reg_decoder_type"] = args.trajectory_reg_decoder_type
    if args.target_guidance_type is not None:
        overrides["target_guidance_type"] = args.target_guidance_type
    if args.target_line_num_points is not None:
        overrides["target_line_num_points"] = args.target_line_num_points
    return overrides


def build_env_config(args, resolved_model_size: str, model_config: dict):
    transfuser_config = build_transfuser_config(
        resolved_model_size,
        **_model_overrides_from_args(args, model_config),
    )

    return {
        "use_render": bool(args.render),
        "num_agents": 1,
        "start_seed": args.start_seed,
        "num_scenarios": args.num_scenarios,
        "traffic_density": args.traffic_density,
        "agent_policy": TransfuserPolicy,
        "show_policy_mark": False,
        "image_on_cuda": bool(args.image_on_cuda),

        "transfuser_checkpoint_path": args.checkpoint,
        "transfuser_policy_device": resolve_device(args.device),
        "transfuser_target_speed_km_h": args.target_speed_km_h,
        "transfuser_lookahead_index": args.lookahead_index,
        "transfuser_controller_type": args.controller_type,
        "transfuser_config": transfuser_config_to_dict(transfuser_config),
    }


def build_platoon_env_config(args, scenario_id: str = "", local_route: str = "") -> dict:
    env_config = {
        "num_agents": int(args.num_agents),
        "observation_mode": "multimodal",
        "use_render": bool(args.render),
        "show_policy_mark": False,
        "start_seed": int(args.start_seed),
        "num_scenarios": int(args.num_scenarios),
        "traffic_density": float(args.traffic_density),
        "image_on_cuda": bool(args.image_on_cuda),
        "allow_respawn": False,
    }
    if scenario_id:
        env_config["scenario_id"] = scenario_id
    if local_route:
        env_config["local_route"] = local_route
        env_config["route_preset"] = get_required_preset(local_route)
        env_config["ego_main_route_block_ids"] = list(get_route_blocks(local_route))
    return env_config


def build_platoon_planner(checkpoint_path: str, args, resolved_model_size: str, model_config: dict):
    from models.platoon.platoon_diffusion_planner import PlatoonDiffusionPlanner
    from models.platoon.weight_migration import migrate_single_to_platoon

    transfuser_config = build_transfuser_config(
        resolved_model_size,
        **_model_overrides_from_args(args, model_config),
    )
    planner = PlatoonDiffusionPlanner(transfuser_config, num_vehicles=int(args.num_agents))
    planner = migrate_single_to_platoon(checkpoint_path, planner)
    planner.eval()
    for parameter in planner.parameters():
        parameter.requires_grad_(False)
    return planner.to(torch.device(resolve_device(args.device)))


def _build_platoon_planner_batch(obs: dict, agent_ids: list[str]) -> dict[str, dict[str, np.ndarray]]:
    required_keys = ("camera", "lidar", "status", "formation_relation_state")
    # Optional keys passed through when present so the model receives full guidance:
    # target_point / preference_point → preference_bias computation
    # coarse_trajectories / mode_valid_mask → dynamic anchor mode selection
    optional_keys = ("target_point", "preference_point", "coarse_trajectories", "mode_valid_mask")
    batch = {}
    for agent_id in agent_ids:
        agent_obs = obs.get(agent_id)
        if agent_obs is None:
            continue
        missing = [key for key in required_keys if key not in agent_obs]
        if missing:
            raise KeyError(f"Platoon observation for {agent_id} missing keys: {missing}")
        sample = {key: agent_obs[key] for key in required_keys}
        for key in optional_keys:
            if key in agent_obs:
                sample[key] = agent_obs[key]
        batch[agent_id] = sample
    return batch


def _build_dynamic_mode_features_for_vehicle(vehicle, config) -> dict[str, np.ndarray]:
    from metadrive.policy.diffusion_policy.mode_context import build_mode_context_from_vehicle
    from metadrive.policy.diffusion_policy.mode_definitions import build_mode_slots, mode_slot_count
    from metadrive.policy.diffusion_policy.mode_trajectory_generator import ModeTrajectoryGenerator

    mode_slots = build_mode_slots(
        keep_lane_count=config.mode_keep_lane_count,
        lane_change_left_count=config.mode_lane_change_left_count,
        lane_change_right_count=config.mode_lane_change_right_count,
        emergency_stop_count=config.mode_emergency_stop_count,
    )
    num_slots = mode_slot_count(
        config.mode_keep_lane_count,
        config.mode_lane_change_left_count,
        config.mode_lane_change_right_count,
        config.mode_emergency_stop_count,
    )
    try:
        current_map = getattr(getattr(vehicle, "engine", None), "current_map", None)
        ctx = build_mode_context_from_vehicle(vehicle, current_map=current_map)
        generator = ModeTrajectoryGenerator(
            keep_lane_high_speed_mps=config.mode_keep_high_speed_mps,
            keep_lane_medium_speed_mps=config.mode_keep_medium_speed_mps,
            keep_lane_low_speed_mps=config.mode_keep_low_speed_mps,
            emergency_decel_mps2=config.mode_emergency_decel_mps2,
            keep_lane_level_count=config.mode_keep_lane_count,
            lane_change_left_level_count=config.mode_lane_change_left_count,
            lane_change_right_level_count=config.mode_lane_change_right_count,
            emergency_stop_level_count=config.mode_emergency_stop_count,
            mode_slots=mode_slots,
        )
        output = generator.generate(ctx)
        return {
            "coarse_trajectories": np.asarray(output.coarse_trajectories, dtype=np.float32),
            "mode_valid_mask": np.asarray(output.mode_valid_mask, dtype=bool),
        }
    except Exception as exc:
        print(f"[platoon_dynamic_anchor] WARNING: failed to build dynamic anchors: {exc}")
        return {
            "coarse_trajectories": np.zeros((num_slots, 8, 2), dtype=np.float32),
            "mode_valid_mask": np.zeros((num_slots,), dtype=bool),
        }


def _attach_platoon_dynamic_mode_features(
    planner_batch: dict[str, dict[str, np.ndarray]],
    env,
    config,
) -> None:
    for agent_id, sample in planner_batch.items():
        vehicle = env.agents.get(agent_id)
        if vehicle is None:
            continue
        sample.update(_build_dynamic_mode_features_for_vehicle(vehicle, config))


def _mode_slot_names_from_config(config) -> list[str]:
    from metadrive.policy.diffusion_policy.mode_definitions import build_mode_slots

    return [
        slot.name
        for slot in build_mode_slots(
            keep_lane_count=config.mode_keep_lane_count,
            lane_change_left_count=config.mode_lane_change_left_count,
            lane_change_right_count=config.mode_lane_change_right_count,
            emergency_stop_count=config.mode_emergency_stop_count,
        )
    ]


def _platoon_agent_final_info(
    *,
    agent_id: str,
    agent_index: int,
    agent_obs: dict,
    base_info: dict,
    trajectory: np.ndarray,
    candidates: np.ndarray,
    masked_logits: np.ndarray,
    raw_logits: np.ndarray,
    mode_valid_mask: np.ndarray,
    coarse_trajectories: np.ndarray | None,
    mode_idx: int,
    mode_slot_names: list[str],
    controller_debug: dict,
) -> dict:
    final_info = dict(base_info or {})
    # Populate target_point from the observation's navigation guidance so that
    # _record_step_visualization always draws a target_point circle.
    # When RANDOM_USE_MODE_ENDPOINT_TARGET=1, this default is overwritten below
    # (line that checks target_point_after is not None) with the selected coarse
    # endpoint, making the visual difference between the two modes obvious.
    _obs_target_point = agent_obs.get("target_point")
    if _obs_target_point is not None:
        final_info["target_point"] = np.asarray(_obs_target_point, dtype=np.float32).reshape(-1)
    final_info.update(
        {
            "agent_id": agent_id,
            "camera_feature": agent_obs.get("camera"),
            "lidar_feature": agent_obs.get("lidar"),
            "status_feature": agent_obs.get("status"),
            "predicted_trajectory": np.asarray(trajectory, dtype=np.float32),
            "trajectory_candidates": np.asarray(candidates[agent_index], dtype=np.float32),
            "trajectory_mode_idx": int(mode_idx),
            "trajectory_mode_logits": np.asarray(raw_logits[agent_index], dtype=np.float32),
            "masked_trajectory_mode_logits": np.asarray(masked_logits[agent_index], dtype=np.float32),
            "mode_valid_mask": np.asarray(mode_valid_mask[agent_index], dtype=bool),
            "coarse_trajectories": (
                None if coarse_trajectories is None else np.asarray(coarse_trajectories, dtype=np.float32)
            ),
            "mode_slot_names": mode_slot_names,
            "controller_debug": controller_debug,
        }
    )
    return final_info


def _resolve_episode_scenario_route(
    scenario_id: str,
    rng: np.random.RandomState,
    fixed_route: str = "",
) -> ScenarioRouteSelection:
    if scenario_id not in SCENARIO_BY_ID:
        valid_ids = ", ".join(sorted(SCENARIO_BY_ID))
        raise ValueError(f"Unknown scenario_id '{scenario_id}'. Valid scenarios: {valid_ids}")

    scenario = get_scenario_definition(scenario_id)
    allowed_routes = tuple(scenario.allowed_local_routes)
    if not allowed_routes:
        raise ValueError(f"Scenario '{scenario_id}' has no allowed local routes.")

    if fixed_route:
        if fixed_route not in allowed_routes:
            raise ValueError(
                f"--local-route '{fixed_route}' is not in allowed routes for '{scenario_id}': "
                f"{list(allowed_routes)}"
            )
        local_route = fixed_route
    elif len(allowed_routes) == 1:
        local_route = allowed_routes[0]
    else:
        local_route = str(rng.choice(allowed_routes))

    route_blocks = tuple(get_route_blocks(local_route))
    if not route_blocks:
        raise ValueError(f"Local route '{local_route}' for scenario '{scenario_id}' resolved to no blocks.")

    return ScenarioRouteSelection(
        scenario_id=scenario_id,
        local_route=local_route,
        route_preset=get_required_preset(local_route),
        ego_main_route_block_ids=route_blocks,
    )


def _apply_episode_route_config(env, selection: ScenarioRouteSelection) -> None:
    updates = {
        "scenario_id": selection.scenario_id,
        "local_route": selection.local_route,
        "route_preset": selection.route_preset,
        "ego_main_route_block_ids": list(selection.ego_main_route_block_ids),
    }
    env.config.update(updates)
    engine = getattr(env, "engine", None)
    global_config = getattr(engine, "global_config", None)
    if global_config is not None:
        global_config.update(updates)


def run_platoon_planner_backend(
    *,
    args,
    checkpoint_path: Path,
    resolved_model_size: str,
    model_config: dict,
    transfuser_config,
    anchors: np.ndarray | None,
) -> None:
    from envs.platoon_env import PlatoonEnv

    env_config = build_platoon_env_config(args, scenario_id=args.scenario_id, local_route=args.local_route)
    env = PlatoonEnv(env_config)
    planner = build_platoon_planner(str(checkpoint_path), args, resolved_model_size, model_config)
    mode_slot_names = _mode_slot_names_from_config(transfuser_config)
    ppo_actor_ckpt = _resolve_ppo_actor_checkpoint(
        str(getattr(args, "ppo_actor_ckpt", "") or ""),
        str(getattr(args, "ppo_run_dir", "") or ""),
    )

    ppo_actor = None
    if ppo_actor_ckpt is not None:
        from models.mode_selection.sb3_mode_cls_policy import require_sb3
        MaskablePPO, _ = require_sb3()
        ppo_actor = MaskablePPO.load(
            str(ppo_actor_ckpt), device=resolve_device(str(getattr(args, "device", "auto")))
        )
        ppo_deterministic = bool(getattr(args, "ppo_deterministic", 1))
        selection_policy = "ppo_actor"
        target_override_enabled = False
        print(f"[test] mode=ppo_actor  ckpt={ppo_actor_ckpt}", flush=True)
        print(f"[test] ppo_deterministic={ppo_deterministic}", flush=True)
    else:
        ppo_deterministic = False
        selection_policy = str(getattr(args, "selection_policy", "argmax"))
        target_override_enabled = bool(getattr(args, "random_use_mode_endpoint_target", 0))
        print(f"[test] mode=pretrained_planner  selection_policy={selection_policy}", flush=True)
        print(f"[test] target_override_enabled={target_override_enabled}", flush=True)

    # random_rng used by pretrained-planner path (argmax does not consume it)
    random_rng = np.random.RandomState(
        int(getattr(args, "random_action_seed", None) or args.start_seed)
    )
    policy_agent_ids = [f"agent{i}" for i in range(int(args.num_agents))]
    random_action_records: list[dict] = []
    summary = {
        "success": 0,
        "crash": 0,
        "out_of_road": 0,
        "reward_per_step": [],
        "episode_length": [],
        "lookahead_y": [],
        "lookahead_heading": [],
        "steering": [],
        "mode_idx": [],
    }
    episode_route_rng = np.random.RandomState(args.start_seed)
    per_episode_control_error_records: dict[int, list[ControlErrorRecord]] = {}

    try:
        for episode_idx in range(args.episodes):
            selection = _resolve_episode_scenario_route(
                args.scenario_id, episode_route_rng, fixed_route=getattr(args, "local_route", "")
            )
            _apply_episode_route_config(env, selection)
            print(
                f"[scenario episode={episode_idx}] "
                f"scenario_id={selection.scenario_id} "
                f"local_route={selection.local_route} "
                f"route_preset={selection.route_preset} "
                f"ego_main_route_block_ids={list(selection.ego_main_route_block_ids)}"
            )
            reset_result = env.reset()
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                obs, info = reset_result
            else:
                obs, info = reset_result, {}
            primary_agent_id = _get_primary_agent_id(env)
            if bool(args.render) and hasattr(env, "switch_to_third_person_view"):
                env.switch_to_third_person_view()

            done = False
            episode_reward = 0.0
            episode_length = 0
            final_info = {}
            episode_3d_frames = []
            episode_2d_frames = []
            actual_positions = []
            planned_trajectories = []
            multimodal_trajectories = []
            step_plot_records = []
            control_error_records: list[ControlErrorRecord] = []
            road_boundaries = _extract_road_topology(env)
            _prev_ego_speed_km_h: float | None = None

            while not done:
                active_agent_ids = [agent_id for agent_id in env.agents.keys() if agent_id in obs]
                if not active_agent_ids:
                    break
                primary_agent_id = primary_agent_id if primary_agent_id in active_agent_ids else active_agent_ids[0]
                pre_step_agent_state = {}
                for agent_id in active_agent_ids:
                    vehicle = env.agents.get(agent_id)
                    if vehicle is None:
                        continue
                    pre_step_agent_state[agent_id] = {
                        "vehicle": vehicle,
                        "xy": np.asarray(vehicle.position[:2], dtype=np.float64),
                        "heading": float(getattr(vehicle, "heading_theta", 0.0)),
                        "speed_km_h": float(getattr(vehicle, "speed_km_h", 0.0)),
                    }
                ego_before_step = env.agents.get(primary_agent_id)
                ego_xy_before_step = (
                    np.asarray(ego_before_step.position[:2], dtype=np.float64)
                    if ego_before_step is not None else None
                )
                ego_heading_before_step = (
                    float(getattr(ego_before_step, "heading_theta", 0.0))
                    if ego_before_step is not None else None
                )
                _cur_ego_speed_km_h = (
                    float(getattr(ego_before_step, "speed_km_h", 0.0))
                    if ego_before_step is not None else None
                )
                _step_dt = 0.1
                _ego_accel_mps2 = (
                    (_cur_ego_speed_km_h - _prev_ego_speed_km_h) / 3.6 / _step_dt
                    if (_cur_ego_speed_km_h is not None and _prev_ego_speed_km_h is not None)
                    else None
                )
                _prev_ego_speed_km_h = _cur_ego_speed_km_h
                step_plot_frame, step_plot_projector = _capture_step_plot_render_context(
                    env,
                    bool(args.save_trajectory_plot),
                )

                planner_batch = _build_platoon_planner_batch(obs, active_agent_ids)
                _attach_platoon_dynamic_mode_features(planner_batch, env, transfuser_config)
                coarse_by_agent = {
                    agent_id: np.asarray(sample.get("coarse_trajectories"), dtype=np.float32)
                    for agent_id, sample in planner_batch.items()
                    if sample.get("coarse_trajectories") is not None
                }
                with torch.no_grad():
                    export = planner.export_mode_selection(planner_batch)
                exported_ids = list(export["agent_ids"])
                candidates = np.asarray(export["trajectory_candidates"], dtype=np.float32)
                masked_logits = np.asarray(export["masked_cls_logits"], dtype=np.float32)
                raw_logits = np.asarray(export["raw_cls_logits"], dtype=np.float32)
                mode_valid_mask = np.asarray(export["mode_valid_mask"], dtype=bool)

                if ppo_actor is not None:
                    actor_obs = _build_ppo_actor_obs(
                        obs=obs,
                        agent_ids=exported_ids,
                        policy_agent_ids=policy_agent_ids,
                        candidates=candidates,
                        mode_valid_mask=mode_valid_mask,
                        raw_logits=raw_logits,
                    )
                    all_selected_modes = _predict_ppo_modes(
                        ppo_actor,
                        actor_obs,
                        actor_obs["agent_mode_masks"],
                        deterministic=ppo_deterministic,
                    )
                    selected_modes = [
                        int(all_selected_modes[policy_agent_ids.index(agent_id)])
                        for agent_id in exported_ids
                    ]
                else:
                    selected_modes = _choose_mode_indices(
                        masked_logits=masked_logits,
                        mode_valid_mask=mode_valid_mask,
                        policy=selection_policy,
                        rng=random_rng,
                    )
                target_override_metadata: dict[str, dict] = {}

                low_level_actions: dict[str, np.ndarray] = {}
                planner_final_info: dict[str, dict] = {}
                for agent_index, agent_id in enumerate(exported_ids):
                    mode_idx = int(selected_modes[agent_index])
                    if mode_idx < 0 or mode_idx >= mode_valid_mask.shape[1] or not bool(mode_valid_mask[agent_index, mode_idx]):
                        raise ValueError(f"Selected invalid mode {mode_idx} for {agent_id}")
                    trajectory = np.asarray(candidates[agent_index, mode_idx], dtype=np.float32)
                    vehicle = env.agents.get(agent_id)
                    current_speed_km_h = float(getattr(vehicle, "speed_km_h", 0.0)) if vehicle is not None else 0.0
                    action, controller_debug = compute_trajectory_control(
                        trajectory=trajectory,
                        lookahead_index=int(args.lookahead_index),
                        current_speed_km_h=current_speed_km_h,
                        target_speed_km_h=float(args.target_speed_km_h),
                        controller_type=str(args.controller_type),
                    )
                    low_level_actions[agent_id] = action
                    planner_final_info[agent_id] = _platoon_agent_final_info(
                        agent_id=agent_id,
                        agent_index=agent_index,
                        agent_obs=obs[agent_id],
                        base_info={},
                        trajectory=trajectory,
                        candidates=candidates,
                        masked_logits=masked_logits,
                        raw_logits=raw_logits,
                        mode_valid_mask=mode_valid_mask,
                        coarse_trajectories=planner_batch[agent_id].get("coarse_trajectories"),
                        mode_idx=mode_idx,
                        mode_slot_names=mode_slot_names,
                        controller_debug=controller_debug,
                    )
                    planner_final_info[agent_id].update(
                        {
                            "selection_policy": selection_policy,
                            "ppo_selected_mode": int(mode_idx),
                            "selected_mode_valid": bool(mode_valid_mask[agent_index, mode_idx]),
                            "target_point_override_enabled": target_override_enabled,
                            **target_override_metadata.get(
                                agent_id,
                                {
                                    "target_point_before": _coerce_optional_point(planner_batch[agent_id].get("target_point")),
                                    "target_point_after": None,
                                    "selected_coarse_endpoint": (
                                        np.asarray(coarse_by_agent[agent_id], dtype=np.float32)[mode_idx, -1, :2]
                                        .astype(float)
                                        .tolist()
                                        if agent_id in coarse_by_agent
                                        else None
                                    ),
                                },
                            ),
                        }
                    )
                    if planner_final_info[agent_id].get("target_point_after") is not None:
                        planner_final_info[agent_id]["target_point"] = np.asarray(
                            planner_final_info[agent_id]["target_point_after"],
                            dtype=np.float32,
                        )

                t_start = time.time()
                obs, reward, terminated, truncated, info = env.step(low_level_actions)
                t_end = time.time()
                print(f"[episode={episode_idx} step={episode_length}] step_time={t_end - t_start:.4f}s")
                episode_length += 1
                missing_agent_ids = _missing_controlled_agents(obs, policy_agent_ids)
                if missing_agent_ids:
                    print(
                        f"[episode={episode_idx} step={episode_length}] terminating because controlled agents disappeared: "
                        f"{missing_agent_ids}",
                        flush=True,
                    )
                    terminated = dict(terminated)
                    truncated = dict(truncated)
                    for agent_id in policy_agent_ids:
                        terminated[agent_id] = True
                    terminated["__all__"] = True
                    truncated["__all__"] = False
                    info = dict(info or {})
                    info["missing_agent_ids"] = list(missing_agent_ids)

                reward_vals = [float(v) for v in reward.values() if isinstance(v, (int, float, np.floating))]
                episode_reward += float(np.mean(reward_vals)) if reward_vals else 0.0
                for agent_id, agent_info in info.items():
                    if agent_id in planner_final_info:
                        # Preserve target_point when it was explicitly overridden to the
                        # selected mode's coarse endpoint (target_point_after is not None).
                        # env.step() info carries MetaDrive's own navigation target_point
                        # which would otherwise silently overwrite the override.
                        _preserve_keys = (
                            {"target_point"}
                            if planner_final_info[agent_id].get("target_point_after") is not None
                            else set()
                        )
                        for k, v in agent_info.items():
                            if k not in _preserve_keys:
                                planner_final_info[agent_id][k] = v
                env_rewards = {
                    agent_id: float(reward.get(agent_id, 0.0))
                    for agent_id in exported_ids
                }
                for agent_id in exported_ids:
                    if agent_id in planner_final_info:
                        planner_final_info[agent_id]["env_reward"] = env_rewards.get(agent_id)
                random_action_records.append(
                    {
                        "episode": int(episode_idx),
                        "step": int(episode_length),
                        "selection_policy": selection_policy,
                        "selected_mode": {
                            agent_id: int(selected_modes[idx])
                            for idx, agent_id in enumerate(exported_ids)
                        },
                        "selected_mode_valid": {
                            agent_id: bool(mode_valid_mask[idx, int(selected_modes[idx])])
                            for idx, agent_id in enumerate(exported_ids)
                        },
                        "candidate_endpoints": {
                            agent_id: candidates[idx, :, -1, :2].astype(float).tolist()
                            for idx, agent_id in enumerate(exported_ids)
                        },
                        "selected_candidate_endpoint": {
                            agent_id: candidates[idx, int(selected_modes[idx]), -1, :2].astype(float).tolist()
                            for idx, agent_id in enumerate(exported_ids)
                        },
                        "target_point_override_enabled": target_override_enabled,
                        "target_point_before": {
                            agent_id: planner_final_info[agent_id].get("target_point_before")
                            for agent_id in exported_ids
                        },
                        "target_point_after": {
                            agent_id: planner_final_info[agent_id].get("target_point_after")
                            for agent_id in exported_ids
                        },
                        "selected_coarse_endpoint": {
                            agent_id: planner_final_info[agent_id].get("selected_coarse_endpoint")
                            for agent_id in exported_ids
                        },
                        "env_reward": env_rewards,
                        "env_reward_mean": float(np.mean(list(env_rewards.values()))) if env_rewards else 0.0,
                    }
                )
                final_info = planner_final_info.get(primary_agent_id, next(iter(planner_final_info.values()), {}))

                ego_after_step = env.agents.get(primary_agent_id) if primary_agent_id is not None else None
                ego_xy_after_step = (
                    np.asarray(ego_after_step.position[:2], dtype=np.float64)
                    if ego_after_step is not None else None
                )
                control_error_record = _compute_control_error_record(
                    episode_idx=episode_idx,
                    step_idx=episode_length,
                    ego_xy_before_step=ego_xy_before_step,
                    ego_heading_before_step=ego_heading_before_step,
                    ego_xy_after_step=ego_xy_after_step,
                    final_info=final_info,
                    step_dt_s=_step_dt,
                    ego_speed_km_h=_cur_ego_speed_km_h,
                    ego_accel_mps2=_ego_accel_mps2,
                )
                if control_error_record is not None:
                    control_error_records.append(control_error_record)

                done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
                _max_steps = int(getattr(args, "max_steps", 0))
                if _max_steps > 0 and episode_length >= _max_steps:
                    done = True

                if args.save_3d_video:
                    frame_3d = _capture_3d_topdown_frame(env, args.topdown_camera_height)
                    if frame_3d is not None:
                        episode_3d_frames.append(frame_3d)
                if args.save_2d_video:
                    frame_2d = _capture_2d_topdown_frame(env)
                    if frame_2d is not None:
                        episode_2d_frames.append(frame_2d)

                if bool(args.save_trajectory_plot):
                    per_agent_step_records: list[StepTrajectoryPlotRecord] = []
                    for agent_id in exported_ids:
                        agent_state = pre_step_agent_state.get(agent_id)
                        agent_final_info = planner_final_info.get(agent_id)
                        if agent_state is None or agent_final_info is None:
                            continue
                        record_actual_positions = actual_positions if agent_id == primary_agent_id else []
                        record_planned = planned_trajectories if agent_id == primary_agent_id else []
                        record_multimodal = multimodal_trajectories if agent_id == primary_agent_id else []
                        before_count = len(per_agent_step_records)
                        _record_step_visualization(
                            ego_before_step=agent_state["vehicle"],
                            ego_xy_before_step=agent_state["xy"],
                            ego_heading_before_step=agent_state["heading"],
                            final_info=agent_final_info,
                            episode_length=episode_length,
                            save_trajectory_plot=True,
                            actual_positions=record_actual_positions,
                            planned_trajectories=record_planned,
                            multimodal_trajectories=record_multimodal,
                            step_plot_records=per_agent_step_records,
                            topdown_frame=step_plot_frame,
                            world_to_screen_projector=step_plot_projector,
                            ego_accel_mps2=(_ego_accel_mps2 if agent_id == primary_agent_id else None),
                            agent_label=f"EGO{exported_ids.index(agent_id) + 1}",
                        )
                        if len(per_agent_step_records) == before_count:
                            continue
                    if per_agent_step_records:
                        primary_record_index = exported_ids.index(primary_agent_id) if primary_agent_id in exported_ids else 0
                        primary_record_index = min(primary_record_index, len(per_agent_step_records) - 1)
                        primary_record = per_agent_step_records[primary_record_index]
                        primary_record.peer_records = [
                            record for index, record in enumerate(per_agent_step_records)
                            if index != primary_record_index
                        ]
                        step_plot_records.append(primary_record)

                if bool(args.save_step_images) and episode_length % max(int(args.step_image_interval), 1) == 0:
                    controller_debug = final_info.get("controller_debug", {})
                    metadata_text = [
                        f"episode={episode_idx} step={episode_length}",
                        f"reward={episode_reward:.2f}",
                        f"agents={len(exported_ids)}",
                    ]
                    if "steering" in controller_debug or "throttle" in controller_debug:
                        metadata_text.append(
                            f"steer={float(controller_debug.get('steering', 0.0)):+.3f} "
                            f"throttle={float(controller_debug.get('throttle', 0.0)):+.3f}"
                        )
                    _save_step_image(
                        final_info=final_info,
                        config=transfuser_config,
                        output_dir=Path(args.output_dir),
                        episode_idx=episode_idx,
                        step_idx=episode_length,
                        anchors=anchors,
                        overlay_all_anchors=True,
                        metadata_text=metadata_text,
                    )

                controller_debug = final_info.get("controller_debug")
                if controller_debug:
                    summary["lookahead_y"].append(float(controller_debug.get("waypoint_y", 0.0)))
                    summary["lookahead_heading"].append(float(controller_debug.get("waypoint_heading", 0.0)))
                    summary["steering"].append(float(controller_debug.get("steering", 0.0)))
                    mode_idx = final_info.get("trajectory_mode_idx")
                    if mode_idx is not None:
                        summary["mode_idx"].append(int(mode_idx))
                    if bool(args.print_trajectory_debug):
                        mode_text = f" mode={mode_idx}" if mode_idx is not None else ""
                        print(
                            f"[analysis episode={episode_idx} step={episode_length}] "
                            f"lookahead_y={controller_debug.get('waypoint_y', 0.0):+.3f} "
                            f"heading={controller_debug.get('waypoint_heading', 0.0):+.3f} "
                            f"steering={controller_debug.get('steering', 0.0):+.3f}"
                            f"{mode_text}"
                        )

                if bool(args.render):
                    env.render(
                        text={
                            "episode": episode_idx,
                            "step": episode_length,
                            "reward": f"{episode_reward:.2f}",
                        }
                    )

            summary["success"] += int(bool(final_info.get("arrive_dest", False)))
            summary["crash"] += int(bool(final_info.get("crash_vehicle", False) or final_info.get("crash", False)))
            summary["out_of_road"] += int(bool(final_info.get("out_of_road", False)))
            _rps = episode_reward / max(episode_length, 1)
            summary["reward_per_step"].append(_rps)
            summary["episode_length"].append(episode_length)
            per_episode_control_error_records[episode_idx] = control_error_records
            print(
                f"[episode={episode_idx}] reward_per_step={_rps:.4f} "
                f"total_reward={episode_reward:.2f} length={episode_length} "
                f"success={final_info.get('arrive_dest', False)} crash={final_info.get('crash', False)} "
                f"out_of_road={final_info.get('out_of_road', False)}"
            )
            output_dir = Path(args.output_dir)
            if args.save_3d_video and episode_3d_frames:
                _write_video(
                    str(output_dir / "videos_3d" / f"episode_{episode_idx:03d}.mp4"),
                    episode_3d_frames,
                    fps=args.video_fps,
                )
            if args.save_2d_video and episode_2d_frames:
                _write_video(
                    str(output_dir / "videos_2d" / f"episode_{episode_idx:03d}.mp4"),
                    episode_2d_frames,
                    fps=args.video_fps,
                )
            if args.save_trajectory_plot and actual_positions:
                _save_trajectory_plot(
                    actual_positions,
                    planned_trajectories,
                    multimodal_trajectories,
                    road_boundaries,
                    str(output_dir / "trajectory_plots" / f"episode_{episode_idx:03d}.png"),
                    episode_idx,
                )
                for step_record in step_plot_records:
                    _save_step_trajectory_plot(
                        step_record=step_record,
                        road_boundaries=road_boundaries,
                        output_path=_build_step_trajectory_plot_path(output_dir, episode_idx, step_record.step_idx),
                        episode_idx=episode_idx,
                        show_topology_polyline=bool(args.show_topology_polyline),
                    )
            if control_error_records:
                _save_control_error_plots(
                    control_error_records,
                    output_dir / "control_errors",
                    episode_idx,
                )
    finally:
        env.close()

    control_error_json = _write_control_error_summary(
        output_dir=Path(args.output_dir),
        per_episode_records=per_episode_control_error_records,
    )
    print(f"[control_error] summary_json={control_error_json}")
    random_records_path = Path(args.output_dir) / "random_action_step_records.jsonl"
    with random_records_path.open("w", encoding="utf-8") as handle:
        for record in random_action_records:
            handle.write(json.dumps(record) + "\n")
    random_summary_path = Path(args.output_dir) / "random_action_reward_summary.json"
    random_summary = _summarize_random_action_rewards(
        random_action_records,
        episodes=args.episodes,
        success=summary["success"],
        crash=summary["crash"],
        out_of_road=summary["out_of_road"],
        metadata={
            "selection_policy": selection_policy,
            "target_point_override_enabled": target_override_enabled,
            "random_action_seed": int(getattr(args, "random_action_seed", args.start_seed)),
            "scenario_id": str(args.scenario_id),
            "local_route": str(getattr(args, "local_route", "")),
            "checkpoint": str(checkpoint_path),
            "ppo_actor_ckpt": str(ppo_actor_ckpt),
            "ppo_deterministic": bool(ppo_deterministic),
            "num_agents": int(args.num_agents),
        },
    )
    random_summary_path.write_text(json.dumps(random_summary, indent=2), encoding="utf-8")
    print(f"[random_action] records_jsonl={random_records_path}")
    print(f"[random_action] summary_json={random_summary_path}")
    _print_test_summary(summary, args.episodes)


def _print_test_summary(summary: dict, episodes: int) -> None:
    num_episodes = max(episodes, 1)
    mode_hist = dict(sorted(Counter(summary["mode_idx"]).items()))
    reward_values = summary.get("reward_per_step") or summary.get("episode_reward") or [0.0]
    reward_key = "avg_reward_per_step" if "reward_per_step" in summary else "avg_reward"
    print(
        "summary: "
        f"success_rate={summary['success'] / num_episodes:.3f} "
        f"crash_rate={summary['crash'] / num_episodes:.3f} "
        f"out_of_road_rate={summary['out_of_road'] / num_episodes:.3f} "
        f"{reward_key}={float(np.mean(reward_values)):.4f} "
        f"avg_length={float(np.mean(summary['episode_length'])):.1f}"
    )
    if summary["lookahead_y"]:
        lookahead_y = np.asarray(summary["lookahead_y"], dtype=np.float32)
        lookahead_heading = np.asarray(summary["lookahead_heading"], dtype=np.float32)
        steering = np.asarray(summary["steering"], dtype=np.float32)
        print(
            "[analysis_summary] "
            f"mean_lookahead_y={float(lookahead_y.mean()):+.4f} "
            f"rightward_fraction={float((lookahead_y > 0).mean()):.3f} "
            f"mean_heading={float(lookahead_heading.mean()):+.4f} "
            f"mean_steering={float(steering.mean()):+.4f} "
            f"mode_hist={mode_hist}"
        )


def main():
    args = _normalize_visualization_args(parse_args())
    output_root = Path(args.output_dir)
    run_output_dir = create_numbered_run_dir(output_root)
    args.output_dir = str(run_output_dir)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model_config = load_diffusion_model_config(args.model_config_path)
    requested_model_size = resolve_model_config_value(args.model_size, model_config, "model_size", "auto")
    resolved_model_size = infer_model_size_from_checkpoint(checkpoint_path) if requested_model_size == "auto" else requested_model_size
    resolved_device = resolve_device(args.device)
    print(f"[test] checkpoint={checkpoint_path}")
    print(f"[test] model_config_path={args.model_config_path}")
    print(f"[test] model_size={resolved_model_size}")
    print(f"[test] device={resolved_device}")
    print(f"[test] num_agents={int(args.num_agents)}")
    print(f"[test] controller_type={args.controller_type}")
    print(
        f"[test] requested_model_size={requested_model_size}"
    )
    print(f"[test] image_on_cuda={bool(args.image_on_cuda)}")
    print(
        f"[test] render={bool(args.render)} save_3d_video={bool(args.save_3d_video)} "
        f"save_2d_video={bool(args.save_2d_video)} save_trajectory_plot={bool(args.save_trajectory_plot)} "
        f"save_step_images={bool(args.save_step_images)}"
    )
    print(f"[test] output_dir={args.output_dir} video_fps={args.video_fps}")
    if args.save_camera_interval > 0:
        print(f"[test] save_camera_interval={args.save_camera_interval} camera_output_dir={args.camera_output_dir}")

    transfuser_config = build_transfuser_config(
        resolved_model_size,
        **_model_overrides_from_args(args, model_config),
    )
    print(
        "[test] mode_counts="
        f"{transfuser_config.mode_keep_lane_count}/"
        f"{transfuser_config.mode_lane_change_left_count}/"
        f"{transfuser_config.mode_lane_change_right_count}/"
        f"{transfuser_config.mode_emergency_stop_count} "
        f"ego_fut_mode={transfuser_config.ego_fut_mode}"
    )
    anchors = None
    anchor_path = Path(transfuser_config.plan_anchor_path)
    if anchor_path.exists():
        anchors = np.load(anchor_path)
    if int(args.num_agents) > 1:
        print("[test] backend=platoon_planner + external PPO actor")
        run_platoon_planner_backend(
            args=args,
            checkpoint_path=checkpoint_path,
            resolved_model_size=resolved_model_size,
            model_config=model_config,
            transfuser_config=transfuser_config,
            anchors=anchors,
        )
        return
    print("[test] backend=single_policy (DatasetCollectEnv + TransfuserPolicy)")
    env_config = build_env_config(args, resolved_model_size, model_config)
    env = DatasetCollectEnv(env_config)
    summary = {
        "success": 0,
        "crash": 0,
        "out_of_road": 0,
        "reward_per_step": [],
        "episode_length": [],
        "lookahead_y": [],
        "lookahead_heading": [],
        "steering": [],
        "mode_idx": [],
    }
    episode_route_rng = np.random.RandomState(args.start_seed)
    per_episode_control_error_records: dict[int, list[ControlErrorRecord]] = {}

    try:
        for episode_idx in range(args.episodes):
            selection = _resolve_episode_scenario_route(
                args.scenario_id, episode_route_rng, fixed_route=getattr(args, "local_route", "")
            )
            _apply_episode_route_config(env, selection)
            print(
                f"[scenario episode={episode_idx}] "
                f"scenario_id={selection.scenario_id} "
                f"local_route={selection.local_route} "
                f"route_preset={selection.route_preset} "
                f"ego_main_route_block_ids={list(selection.ego_main_route_block_ids)}"
            )
            obs, info = env.reset()
            primary_agent_id = _get_primary_agent_id(env)
            if bool(args.render) and hasattr(env, "switch_to_third_person_view"):
                env.switch_to_third_person_view()

            # Initialise ScenarioOrchestrator to replay traffic recipes (lead vehicles,
            # hard brakes, background vehicle injections) that were active during training.
            _scenario_orchestrator = None
            try:
                _scenario_def = get_scenario_definition(selection.scenario_id)
                if _scenario_def.traffic_recipes:
                    _scenario_orchestrator = ScenarioOrchestrator(_scenario_def, selection.local_route)
                    _scenario_orchestrator.reset(env, primary_agent_id or "default_agent")
            except Exception:
                _scenario_orchestrator = None

            done = False
            episode_reward = 0.0
            episode_length = 0
            final_info = {}
            episode_3d_frames = []
            episode_2d_frames = []
            actual_positions = []
            planned_trajectories = []
            multimodal_trajectories = []
            step_plot_records = []
            control_error_records: list[ControlErrorRecord] = []
            road_boundaries = _extract_road_topology(env)
            _prev_ego_speed_km_h: float | None = None

            while not done:
                # Fire scenario recipes (vehicle injection / speed override) each step,
                # matching the behaviour of the expert data collection loop.
                if _scenario_orchestrator is not None:
                    _scenario_orchestrator.before_step(env, primary_agent_id or "default_agent", episode_length)
                ego_before_step = env.agents.get(primary_agent_id) if primary_agent_id is not None else None
                ego_xy_before_step = (
                    np.asarray(ego_before_step.position[:2], dtype=np.float64)
                    if ego_before_step is not None else None
                )
                ego_heading_before_step = (
                    float(getattr(ego_before_step, "heading_theta", 0.0))
                    if ego_before_step is not None else None
                )
                _cur_ego_speed_km_h = (
                    float(getattr(ego_before_step, "speed_km_h", 0.0))
                    if ego_before_step is not None else None
                )
                _step_dt = 0.1  # physics_world_step_size(0.02) * decision_repeat(5)
                _ego_accel_mps2 = (
                    (_cur_ego_speed_km_h - _prev_ego_speed_km_h) / 3.6 / _step_dt
                    if (_cur_ego_speed_km_h is not None and _prev_ego_speed_km_h is not None)
                    else None
                )
                _prev_ego_speed_km_h = _cur_ego_speed_km_h
                step_plot_frame, step_plot_projector = _capture_step_plot_render_context(
                    env,
                    bool(args.save_trajectory_plot),
                )
                # External actions are ignored when agent_policy is a closed-loop policy.
                dummy_actions = {
                    agent_id: np.zeros(2, dtype=np.float32)
                    for agent_id in env.agents.keys()
                }
                t_start = time.time()
                obs, reward, terminated, truncated, info = env.step(dummy_actions)
                t_end = time.time()
                print(f"[episode={episode_idx} step={episode_length}] step_time={t_end - t_start:.4f}s")
                agent_id = next(iter(reward.keys()))
                episode_reward += float(reward[agent_id])
                episode_length += 1
                final_info = info.get(agent_id, {})
                final_info["env_reward"] = float(reward[agent_id])
                ego_after_step = env.agents.get(primary_agent_id) if primary_agent_id is not None else None
                ego_xy_after_step = (
                    np.asarray(ego_after_step.position[:2], dtype=np.float64)
                    if ego_after_step is not None else None
                )
                control_error_record = _compute_control_error_record(
                    episode_idx=episode_idx,
                    step_idx=episode_length,
                    ego_xy_before_step=ego_xy_before_step,
                    ego_heading_before_step=ego_heading_before_step,
                    ego_xy_after_step=ego_xy_after_step,
                    final_info=final_info,
                    step_dt_s=_step_dt,
                    ego_speed_km_h=_cur_ego_speed_km_h,
                    ego_accel_mps2=_ego_accel_mps2,
                )
                if control_error_record is not None:
                    control_error_records.append(control_error_record)
                done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
                _max_steps = int(getattr(args, "max_steps", 0))
                if _max_steps > 0 and episode_length >= _max_steps:
                    done = True

                if args.save_3d_video:
                    frame_3d = _capture_3d_topdown_frame(env, args.topdown_camera_height)
                    if frame_3d is not None:
                        episode_3d_frames.append(frame_3d)

                frame_2d = None
                if args.save_2d_video:
                    frame_2d = _capture_2d_topdown_frame(env)
                    if frame_2d is not None:
                        episode_2d_frames.append(frame_2d)

                _record_step_visualization(
                    ego_before_step=ego_before_step,
                    ego_xy_before_step=ego_xy_before_step,
                    ego_heading_before_step=ego_heading_before_step,
                    final_info=final_info,
                    episode_length=episode_length,
                    save_trajectory_plot=bool(args.save_trajectory_plot),
                    actual_positions=actual_positions,
                    planned_trajectories=planned_trajectories,
                    multimodal_trajectories=multimodal_trajectories,
                    step_plot_records=step_plot_records,
                    topdown_frame=step_plot_frame,
                    world_to_screen_projector=step_plot_projector,
                    ego_accel_mps2=_ego_accel_mps2,
                )

                if args.save_camera_interval > 0:
                    agent_obs = obs.get(agent_id)
                    if agent_obs is not None and episode_length % args.save_camera_interval == 0:
                        save_triplet_cameras(
                            observation=agent_obs,
                            output_dir=Path(args.camera_output_dir),
                            episode_idx=episode_idx,
                            step_idx=episode_length,
                        )

                if bool(args.save_step_images) and episode_length % max(int(args.step_image_interval), 1) == 0:
                    controller_debug = final_info.get("controller_debug", {})
                    metadata_text = [
                        f"episode={episode_idx} step={episode_length}",
                        f"reward={episode_reward:.2f}",
                    ]
                    if "steering" in controller_debug or "throttle" in controller_debug:
                        metadata_text.append(
                            f"steer={float(controller_debug.get('steering', 0.0)):+.3f} "
                            f"throttle={float(controller_debug.get('throttle', 0.0)):+.3f}"
                        )
                    _save_step_image(
                        final_info=final_info,
                        config=transfuser_config,
                        output_dir=Path(args.output_dir),
                        episode_idx=episode_idx,
                        step_idx=episode_length,
                        anchors=anchors,
                        overlay_all_anchors=True,
                        metadata_text=metadata_text,
                    )

                controller_debug = final_info.get("controller_debug")
                if controller_debug:
                    summary["lookahead_y"].append(float(controller_debug.get("waypoint_y", 0.0)))
                    summary["lookahead_heading"].append(float(controller_debug.get("waypoint_heading", 0.0)))
                    summary["steering"].append(float(controller_debug.get("steering", 0.0)))
                    mode_idx = final_info.get("trajectory_mode_idx")
                    if mode_idx is not None:
                        summary["mode_idx"].append(int(mode_idx))
                    if bool(args.print_trajectory_debug):
                        mode_text = f" mode={mode_idx}" if mode_idx is not None else ""
                        print(
                            f"[analysis episode={episode_idx} step={episode_length}] "
                            f"lookahead_y={controller_debug.get('waypoint_y', 0.0):+.3f} "
                            f"heading={controller_debug.get('waypoint_heading', 0.0):+.3f} "
                            f"steering={controller_debug.get('steering', 0.0):+.3f}"
                            f"{mode_text}"
                        )

                if bool(args.render):
                    env.render(
                        text={
                            "episode": episode_idx,
                            "step": episode_length,
                            "reward": f"{episode_reward:.2f}",
                        }
                    )

            summary["success"] += int(bool(final_info.get("arrive_dest", False)))
            summary["crash"] += int(bool(final_info.get("crash_vehicle", False) or final_info.get("crash", False)))
            summary["out_of_road"] += int(bool(final_info.get("out_of_road", False)))
            _rps = episode_reward / max(episode_length, 1)
            summary["reward_per_step"].append(_rps)
            summary["episode_length"].append(episode_length)
            per_episode_control_error_records[episode_idx] = control_error_records
            print(
                f"[episode={episode_idx}] reward_per_step={_rps:.4f} "
                f"total_reward={episode_reward:.2f} length={episode_length} "
                f"success={final_info.get('arrive_dest', False)} crash={final_info.get('crash', False)} "
                f"out_of_road={final_info.get('out_of_road', False)}"
            )
            output_dir = Path(args.output_dir)
            if args.save_3d_video and episode_3d_frames:
                _write_video(
                    str(output_dir / "videos_3d" / f"episode_{episode_idx:03d}.mp4"),
                    episode_3d_frames,
                    fps=args.video_fps,
                )
            if args.save_2d_video and episode_2d_frames:
                _write_video(
                    str(output_dir / "videos_2d" / f"episode_{episode_idx:03d}.mp4"),
                    episode_2d_frames,
                    fps=args.video_fps,
                )
            if args.save_trajectory_plot and actual_positions:
                _save_trajectory_plot(
                    actual_positions,
                    planned_trajectories,
                    multimodal_trajectories,
                    road_boundaries,
                    str(output_dir / "trajectory_plots" / f"episode_{episode_idx:03d}.png"),
                    episode_idx,
                )
                for step_record in step_plot_records:
                    _save_step_trajectory_plot(
                        step_record=step_record,
                        road_boundaries=road_boundaries,
                        output_path=_build_step_trajectory_plot_path(output_dir, episode_idx, step_record.step_idx),
                        episode_idx=episode_idx,
                        show_topology_polyline=bool(args.show_topology_polyline),
                    )
            if control_error_records:
                _save_control_error_plots(
                    control_error_records,
                    output_dir / "control_errors",
                    episode_idx,
                )
    finally:
        env.close()

    control_error_json = _write_control_error_summary(
        output_dir=Path(args.output_dir),
        per_episode_records=per_episode_control_error_records,
    )
    print(f"[control_error] summary_json={control_error_json}")

    _print_test_summary(summary, args.episodes)


if __name__ == "__main__":
    main()
