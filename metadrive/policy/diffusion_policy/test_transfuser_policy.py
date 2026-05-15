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
from metadrive.policy.diffusion_policy.selected_mode_guidance import apply_selected_mode_guidance
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
        fraction = _level_fraction_from_name(mode_name, names)
        # Keep lane: shades of green  (fraction=1 → light, fraction=0 → deep)
        if mode_name.startswith("KEEP_LEVEL_") or "KEEP" in mode_name and "LC" not in mode_name:
            return (int(30 + 40 * fraction), int(150 + 60 * fraction), int(30 + 40 * fraction))
        # Left lane change: shades of yellow
        if mode_name.startswith("LEFT_LC_LEVEL_") or "LEFT" in mode_name:
            return (int(200 + 55 * fraction), int(120 + 90 * fraction), int(20 * fraction))
        # Right lane change: shades of blue
        if mode_name.startswith("RIGHT_LC_LEVEL_") or "RIGHT" in mode_name:
            return (int(10 + 50 * fraction), int(80 + 70 * fraction), int(160 + 65 * fraction))
        # Emergency stop / other: grey tones
        return (int(160 + 80 * fraction), int(20 + 80 * fraction), int(20 + 80 * fraction))

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


def _build_step_overlay_polylines(
    *,
    ego_xy: "np.ndarray",
    ego_heading_rad: float,
    coarse_anchor_trajectories: "np.ndarray | None",
    multimodal_trajectories: "np.ndarray | None",
    selected_mode_idx: "int | None",
    topology_polyline_world: "np.ndarray | None",
    mode_valid_mask: "np.ndarray | None",
    mode_slot_names: "list[str] | None",
    target_point_world: "np.ndarray | None" = None,
    preference_point_world: "np.ndarray | None" = None,
    peer_ego_data: "list[dict] | None" = None,
) -> "list[dict]":
    """Build overlay polyline list for _capture_topdown_frame_with_overlay.

    coarse_anchor_trajectories and multimodal_trajectories are in EGO-LOCAL frame
    (metres ahead / to-side relative to the vehicle).  This function converts them
    to world coordinates via _local_xy_to_world_xy before adding to the overlay list.

    peer_ego_data dicts must include "heading_rad" alongside "xy".

    Returns list of dicts, each either:
      {"world_points": np.ndarray (N,2), "color": (R,G,B), "width": int}   — polyline
      {"world_point":  np.ndarray (2,),  "color": (R,G,B), "radius_px": int, "type": "circle"}  — filled dot
    """
    import numpy as np_local
    overlays: list = []

    def _add(world_points, color, width):
        pts = np_local.asarray(world_points, dtype=np_local.float64)
        if pts.ndim == 2 and pts.shape[0] >= 2 and pts.shape[1] >= 2:
            overlays.append({"world_points": pts[:, :2], "color": color, "width": width})

    def _local_traj_to_world(local_pts: "np.ndarray", origin_xy, heading_rad: float) -> "np.ndarray":
        """Convert (T, 2+) local-frame trajectory to (T, 2) world XY."""
        pts = np_local.asarray(local_pts, dtype=np_local.float64)
        cos_h = float(np_local.cos(float(heading_rad)))
        sin_h = float(np_local.sin(float(heading_rad)))
        ox, oy = float(origin_xy[0]), float(origin_xy[1])
        world = np_local.empty((pts.shape[0], 2), dtype=np_local.float64)
        world[:, 0] = ox + cos_h * pts[:, 0] - sin_h * pts[:, 1]
        world[:, 1] = oy + sin_h * pts[:, 0] + cos_h * pts[:, 1]
        return world

    def _draw_agent_overlays(a_ego, a_heading, a_coarse, a_candidates, a_selected, a_mask, primary, a_pref_world=None):
        ego_world = np_local.asarray(a_ego[:2], dtype=np_local.float64).reshape(1, 2)

        # topology (only for leader, already in world frame)
        if primary and topology_polyline_world is not None:
            t = np_local.asarray(topology_polyline_world, dtype=np_local.float64)
            if t.ndim == 2 and t.shape[0] >= 2:
                _add(t, PAPER_TOPOLOGY_COLOR, 2)

        # coarse anchors — local → world, all drawn in grey
        if a_coarse is not None:
            anchors = np_local.asarray(a_coarse, dtype=np_local.float64)
            for mode_i, anchor in enumerate(anchors):
                if anchor.ndim != 2 or anchor.shape[1] < 2:
                    continue
                world_pts = _local_traj_to_world(anchor, a_ego, a_heading)
                path = np_local.vstack([ego_world, world_pts])
                _add(path, (160, 160, 160), 1)

        # multi-mode candidates — local → world
        if a_candidates is not None:
            cands = np_local.asarray(a_candidates, dtype=np_local.float64)
            if cands.ndim == 3 and cands.shape[2] >= 2:
                # non-selected first
                for mode_i in range(cands.shape[0]):
                    if a_selected is not None and mode_i == int(a_selected):
                        continue
                    world_pts = _local_traj_to_world(cands[mode_i], a_ego, a_heading)
                    path = np_local.vstack([ego_world, world_pts])
                    color = _plot_color_for_mode_index(mode_i, mode_slot_names if primary else None)
                    _add(path, color, 3 if primary else 1)
                # selected last (on top)
                if a_selected is not None and int(a_selected) < cands.shape[0]:
                    world_pts = _local_traj_to_world(cands[int(a_selected)], a_ego, a_heading)
                    path = np_local.vstack([ego_world, world_pts])
                    color = _plot_color_for_mode_index(int(a_selected), mode_slot_names if primary else None)
                    _add(path, color, 6 if primary else 3)

        # preference_point — orange dot per agent
        if a_pref_world is not None:
            pp = np_local.asarray(a_pref_world, dtype=np_local.float64).reshape(-1)
            if pp.shape[0] >= 2:
                overlays.append({"world_point": pp[:2], "color": (255, 140, 0), "radius_px": 7, "type": "circle"})

    # peer agents first (behind primary)
    for peer in (peer_ego_data or []):
        _draw_agent_overlays(
            peer.get("xy", np_local.zeros(2)),
            float(peer.get("heading_rad", 0.0)),
            peer.get("anchors"),
            peer.get("candidates"),
            peer.get("selected"),
            peer.get("valid_mask"),
            primary=True,
            a_pref_world=peer.get("preference_point_world"),
        )

    # primary agent last (on top)
    _draw_agent_overlays(
        ego_xy, float(ego_heading_rad),
        coarse_anchor_trajectories, multimodal_trajectories,
        selected_mode_idx, mode_valid_mask, primary=True,
        a_pref_world=preference_point_world,
    )

    return overlays


def _capture_topdown_frame_with_overlay(
    env,
    overlay_polylines: "list[dict]",
    *,
    screen_size: int = 800,
    film_size: int = 3000,
    camera_position: "tuple[float, float] | None" = None,
) -> "np.ndarray | None":
    """Capture topdown frame and draw trajectory overlays using the renderer's
    own coordinate system (renderer._world_to_screen_position), avoiding the
    OpenCV post-processing projector mismatch.
    """
    import pygame
    import numpy as np_local

    agents = getattr(env, "agents", {})
    if not agents and camera_position is None:
        return None

    if camera_position is None:
        cam_pos = tuple(next(iter(agents.values())).position[:2])
    else:
        cam_pos = (float(camera_position[0]), float(camera_position[1]))

    renderer = getattr(env, "top_down_renderer", None)
    if renderer is not None:
        renderer.position = cam_pos

    main_camera = getattr(getattr(env, "engine", None), "main_camera", None)
    original_track = getattr(main_camera, "current_track_agent", None) if main_camera else None
    if main_camera is not None:
        main_camera.current_track_agent = None

    try:
        env.render(
            mode="top_down",
            window=False,
            screen_size=(screen_size, screen_size),
            film_size=(film_size, film_size),
            target_agent_heading_up=False,
            camera_position=cam_pos,
        )
    finally:
        if main_camera is not None:
            main_camera.current_track_agent = original_track

    if renderer is None:
        return None

    # Draw overlays on _screen_canvas using renderer's coordinate system
    try:
        field = renderer._screen_canvas.get_size()
        pos_px = renderer._frame_canvas.pos2pix(*cam_pos)
        off = (pos_px[0] - field[0] / 2, pos_px[1] - field[1] / 2)
    except Exception:
        off = (0, 0)

    for poly in overlay_polylines:
        color = tuple(int(c) for c in poly["color"])
        if poly.get("type") == "circle":
            try:
                wp = np_local.asarray(poly["world_point"], dtype=np_local.float32)
                sp = renderer._world_to_screen_position(wp[:2], off)
                if sp is not None:
                    pygame.draw.circle(
                        renderer._screen_canvas, color,
                        (int(sp[0]), int(sp[1])),
                        int(poly.get("radius_px", 7)),
                    )
            except Exception:
                pass
            continue
        world_pts = np_local.asarray(poly["world_points"], dtype=np_local.float32)
        width = max(2.5, int(poly.get("width", 6)))
        screen_pts = []
        for wp in world_pts:
            try:
                sp = renderer._world_to_screen_position(wp[:2], off)
                if sp is not None:
                    screen_pts.append((int(sp[0]), int(sp[1])))
            except Exception:
                pass
        if len(screen_pts) >= 2:
            try:
                pygame.draw.lines(renderer._screen_canvas, color, False, screen_pts, width)
            except Exception:
                pass

    # Re-convert _screen_canvas to numpy
    try:
        from metadrive.engine.top_down_renderer import WorldSurface
        frame = WorldSurface.to_cv2_image(renderer._screen_canvas)
    except Exception:
        try:
            frame = np_local.transpose(pygame.surfarray.array3d(renderer._screen_canvas), (1, 0, 2))
        except Exception:
            return None

    if frame is None:
        return None
    frame = np_local.asarray(frame)
    if frame.ndim == 3 and frame.shape[2] > 3:
        frame = frame[:, :, :3]
    return frame



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
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--num-scenarios", type=int, default=1)
    parser.add_argument("--traffic-density", type=float, default=0.06)
    parser.add_argument(
        "--random-traffic",
        type=int,
        choices=(0, 1),
        default=0,
        help="Use MetaDrive's non-deterministic traffic manager RNG. Keep 0 for fair checkpoint comparisons.",
    )
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
        "--save-combined-traj-frames",
        type=int, choices=(0, 1), default=1,
        help="Save per-step combined trajectory frames "
             "(coarse anchors + all mode candidates + selected trajectory + map topology). "
             "Output: <output_dir>/combined_traj_frames/episode_XXX/step_XXXXX.png",
    )
    parser.add_argument(
        "--debug-coordinate-audit",
        type=int, choices=(0, 1), default=0,
        help="Write per-agent local/world coordinate audit records for platoon trajectory debugging.",
    )
    parser.add_argument(
        "--ppo-actor-ckpt",
        type=str,
        default="/media/kong/Elements_SE/Diffusion_Data/outputs/ppo/run_9/checkpoints/final/sb3_model.zip",
        help="SB3 MaskablePPO actor checkpoint (.zip). "
             "Empty string (default) = skip PPO and use the pretrained diffusion planner directly (argmax/random_valid).",
    )
    parser.add_argument("--ppo-run-dir", type=str, default="")
    parser.add_argument("--ppo-deterministic", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--use-relation-encoder", type=int, choices=(0, 1), default=1,
        help="0: skip relation_encoder (pure single-vehicle diffusion per agent); 1: use relation_encoder (default).",
    )
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


def _inject_semantic_preference_point(
    planner_batch: dict[str, dict],
    coarse_by_agent: dict[str, np.ndarray],
    agent_ids: list[str],
    env=None,
    target_speed_km_h: float = 30.0,
    horizon_s: float = 4.0,
) -> None:
    """Inject preference_point as lane-following endpoint at target speed.

    All vehicles in the platoon compute their preference_point along the
    HEAD vehicle's (agent_ids[0]) current lane, so followers' preference_points
    are guaranteed to share the same lane as the leader.

    Each vehicle projects its own position onto the leader's lane to get its
    arc-length s, then advances delta_s = target_speed * horizon along that lane
    and converts the resulting world point to ego-local coordinates.

    Falls back to keep-lane coarse endpoint (slot 0) if lane info unavailable.
    """
    delta_s = (target_speed_km_h / 3.6) * horizon_s

    # Resolve the leader's lane (agent_ids[0] = head vehicle = agent0)
    leader_lane = None
    if env is not None and agent_ids:
        leader_vehicle = getattr(env, "agents", {}).get(agent_ids[0])
        if leader_vehicle is not None:
            leader_lane = getattr(leader_vehicle, "lane", None)

    for agent_id in agent_ids:
        pp = None

        vehicle = getattr(env, "agents", {}).get(agent_id) if env is not None else None
        if vehicle is not None and leader_lane is not None:
            try:
                pos = getattr(vehicle, "position", None)
                heading = float(getattr(vehicle, "heading_theta", 0.0))
                if pos is not None:
                    # Project this vehicle's position onto the leader's lane
                    s_ego, _ = leader_lane.local_coordinates(pos)
                    s_target = min(float(s_ego) + delta_s, float(leader_lane.length))
                    target_world = np.asarray(leader_lane.position(s_target, 0.0), dtype=np.float32)
                    dx = float(target_world[0]) - float(pos[0])
                    dy = float(target_world[1]) - float(pos[1])
                    cos_h, sin_h = np.cos(heading), np.sin(heading)
                    pp = np.asarray([cos_h * dx + sin_h * dy, -sin_h * dx + cos_h * dy], dtype=np.float32)
            except Exception:
                pass

        # Fallback: keep-lane coarse endpoint (slot 0)
        if pp is None:
            coarse = coarse_by_agent.get(agent_id)
            if coarse is not None and coarse.shape[0] > 0:
                pp = coarse[0, -1, :2].astype(np.float32)

        if pp is not None:
            planner_batch[agent_id]["preference_point"] = pp


def _apply_selected_mode_target_overrides(
    planner_batch: dict[str, dict],
    *,
    agent_ids: list[str],
    selected_modes: list[int],
    coarse_by_agent: dict[str, np.ndarray],
    config=None,
) -> dict[str, dict]:
    """Write selected dynamic-anchor endpoints as target/preference points."""
    if config is None:
        config = build_transfuser_config()
    return apply_selected_mode_guidance(
        planner_batch,
        agent_ids=agent_ids,
        selected_modes=selected_modes,
        coarse_by_agent=coarse_by_agent,
        config=config,
    )


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
    mode_slot_names: list[str] | None = None,
    selected_label: str | None = None,
    other_label: str | None = None,
) -> None:
    candidates_array = np.asarray(candidates_world, dtype=np.float64)
    if candidates_array.ndim != 3:
        return
    origin_array = np.asarray(origin, dtype=np.float64)
    for mode_i in range(candidates_array.shape[0]):
        full_path = np.vstack([origin_array, candidates_array[mode_i]])
        rgb = _plot_color_for_mode_index(mode_i, mode_slot_names)
        mpl_color = tuple(c / 255.0 for c in rgb)
        if selected_mode_idx is not None and mode_i == int(selected_mode_idx):
            ax.plot(
                full_path[:, 0],
                full_path[:, 1],
                "--",
                color=mpl_color,
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
                color=mpl_color,
                linewidth=1.9,
                alpha=0.55,
                zorder=3,
                label=other_label,
            )
            other_label = None


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


def _point_or_none(value) -> list[float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return None
    return array.astype(float).tolist()


def _local_xy_to_world_list(
    value,
    ego_world_position: np.ndarray | None,
    ego_heading_rad: float | None,
) -> list[float] | None:
    if value is None or ego_world_position is None or ego_heading_rad is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size < 2:
        return None
    world = _local_xy_to_world_xy(array[:2], ego_world_position, float(ego_heading_rad))
    return world.astype(float).tolist()


def _build_coordinate_audit_record(
    *,
    episode_idx: int,
    step_idx: int,
    agent_id: str,
    exported_index: int,
    ego_xy_before_step: np.ndarray | None,
    ego_heading_before_step: float | None,
    final_info: dict,
) -> dict:
    selected_mode = final_info.get("trajectory_mode_idx")
    selected_mode_int = int(selected_mode) if selected_mode is not None else None
    candidates = final_info.get("trajectory_candidates")
    selected_candidate = None
    if candidates is not None and selected_mode_int is not None:
        candidates_array = np.asarray(candidates, dtype=np.float64)
        if (
            candidates_array.ndim == 3
            and 0 <= selected_mode_int < candidates_array.shape[0]
            and candidates_array.shape[1] > 0
        ):
            selected_candidate = candidates_array[selected_mode_int]

    coarse = final_info.get("coarse_trajectories")
    selected_coarse_endpoint = None
    if coarse is not None and selected_mode_int is not None:
        coarse_array = np.asarray(coarse, dtype=np.float64)
        if (
            coarse_array.ndim == 3
            and 0 <= selected_mode_int < coarse_array.shape[0]
            and coarse_array.shape[1] > 0
        ):
            selected_coarse_endpoint = coarse_array[selected_mode_int, -1, :2]

    target_point = final_info.get("target_point")
    controller_debug = final_info.get("controller_debug", {}) or {}
    first_local = selected_candidate[0] if selected_candidate is not None and len(selected_candidate) else None
    endpoint_local = selected_candidate[-1] if selected_candidate is not None and len(selected_candidate) else None

    return {
        "episode": int(episode_idx),
        "step": int(step_idx),
        "agent_id": str(agent_id),
        "exported_index": int(exported_index),
        "ego_xy_before_step": _point_or_none(ego_xy_before_step),
        "ego_heading_before_step": None if ego_heading_before_step is None else float(ego_heading_before_step),
        "selected_mode": selected_mode_int,
        "selected_candidate_first_local": _point_or_none(first_local),
        "selected_candidate_endpoint_local": _point_or_none(endpoint_local),
        "selected_candidate_first_world": _local_xy_to_world_list(first_local, ego_xy_before_step, ego_heading_before_step),
        "selected_candidate_endpoint_world": _local_xy_to_world_list(endpoint_local, ego_xy_before_step, ego_heading_before_step),
        "initial_selected_candidate_endpoint_local": _point_or_none(
            final_info.get("initial_selected_candidate_endpoint")
        ),
        "initial_selected_candidate_endpoint_world": _local_xy_to_world_list(
            final_info.get("initial_selected_candidate_endpoint"), ego_xy_before_step, ego_heading_before_step
        ),
        "guided_selected_candidate_endpoint_local": _point_or_none(
            final_info.get("guided_selected_candidate_endpoint")
        ),
        "guided_selected_candidate_endpoint_world": _local_xy_to_world_list(
            final_info.get("guided_selected_candidate_endpoint"), ego_xy_before_step, ego_heading_before_step
        ),
        "selected_coarse_endpoint_local": _point_or_none(selected_coarse_endpoint),
        "selected_coarse_endpoint_world": _local_xy_to_world_list(
            selected_coarse_endpoint, ego_xy_before_step, ego_heading_before_step
        ),
        "target_point_local": _point_or_none(target_point),
        "target_point_world": _local_xy_to_world_list(target_point, ego_xy_before_step, ego_heading_before_step),
        "controller_waypoint_x": (
            float(controller_debug["waypoint_x"])
            if controller_debug.get("waypoint_x") is not None
            else None
        ),
        "controller_waypoint_y": (
            float(controller_debug["waypoint_y"])
            if controller_debug.get("waypoint_y") is not None
            else None
        ),
        "controller_waypoint_heading": (
            float(controller_debug["waypoint_heading"])
            if controller_debug.get("waypoint_heading") is not None
            else None
        ),
    }


def _assert_platoon_candidate_alignment(
    *,
    exported_ids: list[str],
    candidates: np.ndarray,
    selected_modes: list[int],
    planner_final_info: dict[str, dict],
) -> None:
    candidates_array = np.asarray(candidates, dtype=np.float32)
    if candidates_array.ndim != 4:
        raise AssertionError(f"candidates must have shape [N,M,8,3], got {candidates_array.shape}")
    if len(exported_ids) != candidates_array.shape[0]:
        raise AssertionError(
            f"exported_ids length {len(exported_ids)} does not match candidates batch {candidates_array.shape[0]}"
        )
    if len(selected_modes) != len(exported_ids):
        raise AssertionError(
            f"selected_modes length {len(selected_modes)} does not match exported_ids length {len(exported_ids)}"
        )
    for agent_index, agent_id in enumerate(exported_ids):
        if agent_id not in planner_final_info:
            raise AssertionError(f"missing planner_final_info for {agent_id}")
        final_info = planner_final_info[agent_id]
        agent_candidates = np.asarray(final_info.get("trajectory_candidates"), dtype=np.float32)
        if agent_candidates.shape != candidates_array[agent_index].shape or not np.allclose(
            agent_candidates, candidates_array[agent_index]
        ):
            raise AssertionError(f"trajectory_candidates mismatch for {agent_id}")
        mode_idx = int(selected_modes[agent_index])
        if int(final_info.get("trajectory_mode_idx", -1)) != mode_idx:
            raise AssertionError(f"trajectory_mode_idx mismatch for {agent_id}")
        selected_trajectory = np.asarray(final_info.get("predicted_trajectory"), dtype=np.float32)
        expected_trajectory = candidates_array[agent_index, mode_idx]
        if selected_trajectory.shape != expected_trajectory.shape or not np.allclose(
            selected_trajectory, expected_trajectory
        ):
            raise AssertionError(f"selected trajectory mismatch for {agent_id}")


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


def _save_trajectory_plot(
    actual_positions: list[np.ndarray],
    planned_trajectories: list[tuple[int, list[np.ndarray]]],
    multimodal_trajectories: list[tuple[int, np.ndarray, int]],
    road_boundaries: list[np.ndarray],
    output_path: str,
    episode_idx: int,
    plot_interval: int = 10,
    mode_slot_names: list[str] | None = None,
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
            mode_slot_names=mode_slot_names,
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

    # GRPO platoon ckpt: {"model_state": {model.*, relation_encoder.*}, "config": ...}
    # Keys are prefixed with "model." relative to PlatoonDiffusionPlanner.
    if "model_state" in checkpoint and any(k.startswith("model.") for k in checkpoint["model_state"]):
        state_dict = {
            k.removeprefix("model."): v
            for k, v in checkpoint["model_state"].items()
            if k.startswith("model.")
        }
    else:
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
        "random_traffic": bool(getattr(args, "random_traffic", 0)),
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
        "random_traffic": bool(getattr(args, "random_traffic", 0)),
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
    from models.platoon.weight_migration import (
        is_platoon_grpo_checkpoint,
        load_platoon_grpo_checkpoint,
        migrate_single_to_platoon,
    )

    transfuser_config = build_transfuser_config(
        resolved_model_size,
        **_model_overrides_from_args(args, model_config),
    )
    planner = PlatoonDiffusionPlanner(
        transfuser_config,
        num_vehicles=int(args.num_agents),
        use_relation_encoder=bool(getattr(args, "use_relation_encoder", 1)),
    )
    if is_platoon_grpo_checkpoint(checkpoint_path):
        print(f"[test] detected GRPO platoon checkpoint — loading directly: {checkpoint_path}", flush=True)
        planner = load_platoon_grpo_checkpoint(checkpoint_path, planner)
    else:
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

        # Ensure every polyline starts from [0, 0] (the vehicle's actual position in
        # ego-local frame).  build_mode_context_from_vehicle samples from the lane
        # centreline projection of the vehicle's longitudinal position, which has a
        # small lateral offset from the vehicle centre.  Without this correction the
        # generator produces trajectories whose first point is NOT at the vehicle,
        # causing a visual gap between ego and the start of the coarse anchor.
        _origin = np.zeros((1, 2), dtype=np.float32)

        def _fix_poly(p):
            if p is None:
                return None
            p = np.asarray(p, dtype=np.float32)
            if p.ndim != 2 or p.shape[1] < 2:
                return p
            # Replace (or prepend) so the first row is the ego origin.
            # If the polyline already has many points, keep them all but anchor
            # the start at [0, 0] to remove the lateral-offset artefact.
            return np.vstack([_origin, p[1:, :2]])

        ctx.current_lane_polyline = _fix_poly(ctx.current_lane_polyline)
        ctx.left_lane_polyline    = _fix_poly(ctx.left_lane_polyline)
        ctx.right_lane_polyline   = _fix_poly(ctx.right_lane_polyline)
        ctx.left_branch_polyline  = _fix_poly(ctx.left_branch_polyline)
        ctx.right_branch_polyline = _fix_poly(ctx.right_branch_polyline)

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
    from metadrive.policy.diffusion_policy.mode_definitions import mode_slot_count
    num_slots = mode_slot_count(
        config.mode_keep_lane_count,
        config.mode_lane_change_left_count,
        config.mode_lane_change_right_count,
        config.mode_emergency_stop_count,
    )
    _fallback = {
        "coarse_trajectories": np.zeros((num_slots, 8, 2), dtype=np.float32),
        "mode_valid_mask": np.ones((num_slots,), dtype=bool),   # all-True: any action passes mask check
    }
    for agent_id, sample in planner_batch.items():
        vehicle = env.agents.get(agent_id)
        if vehicle is None:
            # Vehicle removed (crash/out-of-road) but episode not yet terminated:
            # write fallback so downstream code never KeyErrors on coarse_trajectories.
            sample.update(_fallback)
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
    # Populate target_point from the observation's navigation guidance.
    # When RANDOM_USE_MODE_ENDPOINT_TARGET=1, this default is overwritten below
    # (line that checks target_point_after is not None) with the selected coarse
    # endpoint, making the visual difference between the two modes obvious.
    _obs_target_point = agent_obs.get("target_point")
    if _obs_target_point is not None:
        final_info["target_point"] = np.asarray(_obs_target_point, dtype=np.float32).reshape(-1)
    _obs_topo = agent_obs.get("topology_polyline")
    if _obs_topo is not None:
        final_info["topology_polyline"] = np.asarray(_obs_topo, dtype=np.float32)
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


def _episode_reset_seed(start_seed: int, num_scenarios: int, episode_idx: int) -> int:
    """Deterministically map an episode index to a MetaDrive scenario seed."""
    scenario_count = max(1, int(num_scenarios))
    return int(start_seed) + (int(episode_idx) % scenario_count)


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
        target_override_enabled = False
        print(f"[test] mode=pretrained_planner  selection_policy={selection_policy}", flush=True)
    print(f"[test] semantic_preference_point_enabled=True  target_override_enabled={target_override_enabled}", flush=True)

    # random_rng used by pretrained-planner path (argmax does not consume it)
    random_rng = np.random.RandomState(
        int(getattr(args, "random_action_seed", None) or args.start_seed)
    )
    policy_agent_ids = [f"agent{i}" for i in range(int(args.num_agents))]
    random_action_records: list[dict] = []
    coordinate_audit_records: list[dict] = []
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
                f"ego_main_route_block_ids={list(selection.ego_main_route_block_ids)} "
                f"reset_seed={_episode_reset_seed(args.start_seed, args.num_scenarios, episode_idx)}"
            )
            reset_result = env.reset(seed=_episode_reset_seed(args.start_seed, args.num_scenarios, episode_idx))
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
            _last_step_reward = 0.0   # reward from previous step, shown on next frame
            final_info = {}
            episode_3d_frames = []
            episode_2d_frames = []
            control_error_records: list[ControlErrorRecord] = []
            road_boundaries = _extract_road_topology(env)
            _prev_ego_speed_km_h: float | None = None

            while not done:
                active_agent_ids = [agent_id for agent_id in env.agents.keys() if agent_id in obs]
                if not active_agent_ids:
                    break
                primary_agent_id = primary_agent_id if primary_agent_id in active_agent_ids else active_agent_ids[0]
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

                planner_batch = _build_platoon_planner_batch(obs, active_agent_ids)
                _attach_platoon_dynamic_mode_features(planner_batch, env, transfuser_config)
                coarse_by_agent = {
                    agent_id: np.asarray(sample.get("coarse_trajectories"), dtype=np.float32)
                    for agent_id, sample in planner_batch.items()
                    if sample.get("coarse_trajectories") is not None
                }
                # Inject semantic preference_point (target-speed lane endpoint) before
                # inference so the diffusion model uses it as a soft semantic target.
                _inject_semantic_preference_point(
                    planner_batch, coarse_by_agent, list(planner_batch.keys()),
                    env=env,
                    target_speed_km_h=float(getattr(args, "target_speed_km_h", 30.0)),
                    horizon_s=float(transfuser_config.target_point_prediction_horizon_s),
                )
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
                for agent_index, agent_id in enumerate(exported_ids):
                    mode_idx = int(selected_modes[agent_index])
                    if mode_idx < 0 or mode_idx >= mode_valid_mask.shape[1] or not bool(mode_valid_mask[agent_index, mode_idx]):
                        raise ValueError(f"Selected invalid mode {mode_idx} for {agent_id}")

                low_level_actions: dict[str, np.ndarray] = {}
                _step_trajectories: dict[str, np.ndarray] = {}
                planner_final_info: dict[str, dict] = {}
                for agent_index, agent_id in enumerate(exported_ids):
                    mode_idx = int(selected_modes[agent_index])
                    if mode_idx < 0 or mode_idx >= mode_valid_mask.shape[1] or not bool(mode_valid_mask[agent_index, mode_idx]):
                        raise ValueError(f"Selected invalid mode {mode_idx} for {agent_id}")
                    trajectory = np.asarray(candidates[agent_index, mode_idx], dtype=np.float32)
                    _step_trajectories[agent_id] = trajectory
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
                            "target_point_override_enabled": False,
                            "selected_candidate_endpoint": (
                                candidates[agent_index, mode_idx, -1, :2].astype(float).tolist()
                            ),
                            "selected_coarse_endpoint": (
                                np.asarray(coarse_by_agent[agent_id], dtype=np.float32)[mode_idx, -1, :2]
                                .astype(float)
                                .tolist()
                                if agent_id in coarse_by_agent
                                else None
                            ),
                        }
                    )
                    for guidance_key in ("target_point", "preference_point", "target_line", "topology_polyline"):
                        if planner_batch[agent_id].get(guidance_key) is not None:
                            planner_final_info[agent_id][guidance_key] = np.asarray(
                                planner_batch[agent_id][guidance_key],
                                dtype=np.float32,
                            )
                # Pass selected trajectories to PlatoonEnv for road_topo_reward.
                if hasattr(env, "_pending_step_trajectories"):
                    env._pending_step_trajectories = _step_trajectories
                if hasattr(env, "_pending_step_all_candidates"):
                    env._pending_step_all_candidates = {
                        agent_id: np.asarray(candidates[ai], dtype=np.float32)
                        for ai, agent_id in enumerate(exported_ids)
                    }
                if hasattr(env, "_pending_step_mode_valid_masks"):
                    env._pending_step_mode_valid_masks = {
                        agent_id: np.asarray(mode_valid_mask[ai], dtype=bool)
                        for ai, agent_id in enumerate(exported_ids)
                    }
                if hasattr(env, "_pending_step_mode_groups"):
                    from metadrive.policy.diffusion_policy.mode_definitions import build_mode_slots as _bms
                    _slot_map = {s.index: s for s in _bms(
                        keep_lane_count=transfuser_config.mode_keep_lane_count,
                        lane_change_left_count=transfuser_config.mode_lane_change_left_count,
                        lane_change_right_count=transfuser_config.mode_lane_change_right_count,
                        emergency_stop_count=transfuser_config.mode_emergency_stop_count,
                    )}
                    _mode_groups: dict[str, str] = {}
                    for _ai, _aid in enumerate(exported_ids):
                        _slot = _slot_map.get(int(selected_modes[_ai]))
                        if _slot is None:
                            _mode_groups[_aid] = "keep"
                        elif _slot.semantic_group == "STOP":
                            _mode_groups[_aid] = "stop"
                        elif _slot.lateral_direction == "left":
                            _mode_groups[_aid] = "left"
                        elif _slot.lateral_direction == "right":
                            _mode_groups[_aid] = "right"
                        else:
                            _mode_groups[_aid] = "keep"
                    env._pending_step_mode_groups = _mode_groups
                _assert_platoon_candidate_alignment(
                    exported_ids=exported_ids,
                    candidates=candidates,
                    selected_modes=selected_modes,
                    planner_final_info=planner_final_info,
                )
                if bool(getattr(args, "debug_coordinate_audit", 0)):
                    for audit_index, agent_id in enumerate(exported_ids):
                        coordinate_audit_records.append(
                            _build_coordinate_audit_record(
                                episode_idx=episode_idx,
                                step_idx=episode_length + 1,
                                agent_id=agent_id,
                                exported_index=audit_index,
                                ego_xy_before_step=ego_xy_before_step if agent_id == primary_agent_id else None,
                                ego_heading_before_step=ego_heading_before_step if agent_id == primary_agent_id else None,
                                final_info=planner_final_info[agent_id],
                            )
                        )

                if bool(getattr(args, "save_combined_traj_frames", 0)) and primary_agent_id in exported_ids:
                    _pri_idx = exported_ids.index(primary_agent_id)
                    _peers = []
                    for _ai, _aid in enumerate(exported_ids):
                        if _aid == primary_agent_id:
                            continue
                        _peer_vehicle = env.agents.get(_aid)
                        _peer_ego = getattr(_peer_vehicle, "position", None)
                        if _peer_ego is not None:
                            _peer_ego = np.asarray(_peer_ego[:2], dtype=np.float64)
                            _peer_veh = env.agents.get(_aid)
                            _peer_heading = float(getattr(_peer_veh, "heading_theta", 0.0)) if _peer_veh else 0.0
                            _peer_pp_local = planner_final_info.get(_aid, {}).get("preference_point")
                            _peer_pp_world = (
                                _local_xy_to_world_list(_peer_pp_local, _peer_ego, _peer_heading)
                                if _peer_pp_local is not None else None
                            )
                            _peers.append({
                                "xy": _peer_ego,
                                "heading_rad": _peer_heading,
                                "anchors": coarse_by_agent.get(_aid),
                                "candidates": (
                                    np.asarray(candidates[_ai], dtype=np.float32)
                                    if candidates is not None else None
                                ),
                                "selected": int(selected_modes[_ai]),
                                "valid_mask": mode_valid_mask[_ai] if mode_valid_mask is not None else None,
                                "preference_point_world": (
                                    np.asarray(_peer_pp_world, dtype=np.float64) if _peer_pp_world is not None else None
                                ),
                            })
                    _pri_vehicle = env.agents.get(primary_agent_id)
                    _pri_ego_xy = np.asarray(_pri_vehicle.position[:2], dtype=np.float64) if _pri_vehicle else np.zeros(2)
                    _pri_heading = float(getattr(_pri_vehicle, "heading_theta", 0.0)) if _pri_vehicle else 0.0
                    _pri_tp_local = planner_final_info.get(primary_agent_id, {}).get("target_point")
                    _pri_tp_world = (
                        _local_xy_to_world_list(_pri_tp_local, _pri_ego_xy, _pri_heading)
                        if _pri_tp_local is not None else None
                    )
                    _pri_pp_local = planner_final_info.get(primary_agent_id, {}).get("preference_point")
                    _pri_pp_world = (
                        _local_xy_to_world_list(_pri_pp_local, _pri_ego_xy, _pri_heading)
                        if _pri_pp_local is not None else None
                    )
                    _overlay = _build_step_overlay_polylines(
                        ego_xy=_pri_ego_xy,
                        ego_heading_rad=_pri_heading,
                        coarse_anchor_trajectories=coarse_by_agent.get(primary_agent_id),
                        multimodal_trajectories=candidates[_pri_idx] if candidates is not None else None,
                        selected_mode_idx=int(selected_modes[_pri_idx]),
                        topology_polyline_world=None,
                        mode_valid_mask=mode_valid_mask[_pri_idx] if mode_valid_mask is not None else None,
                        mode_slot_names=mode_slot_names,
                        target_point_world=np.asarray(_pri_tp_world, dtype=np.float64) if _pri_tp_world is not None else None,
                        preference_point_world=np.asarray(_pri_pp_world, dtype=np.float64) if _pri_pp_world is not None else None,
                        peer_ego_data=_peers,
                    )
                    _combined = _capture_topdown_frame_with_overlay(
                        env, _overlay,
                        screen_size=800, film_size=10000,
                        camera_position=tuple(_pri_ego_xy),
                    )
                    if _combined is not None:
                        # Draw PPO reward (from previous step) in the top-left corner.
                        _reward_text = f"step={episode_length}  reward={_last_step_reward:+.3f}  total={episode_reward:.2f}"
                        cv2.putText(_combined, _reward_text, (10, 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 3, cv2.LINE_AA)
                        cv2.putText(_combined, _reward_text, (10, 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 200), 1, cv2.LINE_AA)
                        _comb_path = (
                            Path(args.output_dir)
                            / "combined_traj_frames"
                            / f"episode_{episode_idx:03d}"
                            / f"step_{episode_length:05d}.png"
                        )
                        _comb_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(_comb_path), cv2.cvtColor(_combined, cv2.COLOR_RGB2BGR))

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
                _last_step_reward = float(np.mean(reward_vals)) if reward_vals else 0.0
                episode_reward += _last_step_reward
                for agent_id, agent_info in info.items():
                    if agent_id in planner_final_info:
                        for k, v in agent_info.items():
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
                        "target_point_override_enabled": False,
                        "semantic_preference_point": {
                            agent_id: (
                                np.asarray(planner_batch[agent_id].get("preference_point", [0.0, 0.0]), dtype=np.float32)
                                .astype(float).tolist()
                            )
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
    if bool(getattr(args, "debug_coordinate_audit", 0)):
        coordinate_audit_path = Path(args.output_dir) / "coordinate_audit.jsonl"
        with coordinate_audit_path.open("w", encoding="utf-8") as handle:
            for record in coordinate_audit_records:
                handle.write(json.dumps(record) + "\n")
        print(f"[coordinate_audit] records_jsonl={coordinate_audit_path}")
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
                f"ego_main_route_block_ids={list(selection.ego_main_route_block_ids)} "
                f"reset_seed={_episode_reset_seed(args.start_seed, args.num_scenarios, episode_idx)}"
            )
            obs, info = env.reset(seed=_episode_reset_seed(args.start_seed, args.num_scenarios, episode_idx))
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

                if bool(getattr(args, "save_combined_traj_frames", 0)) and ego_xy_before_step is not None:
                    _coarse = final_info.get("coarse_trajectories")
                    _cands  = final_info.get("trajectory_candidates")
                    _sel    = final_info.get("trajectory_mode_idx")
                    _tp_local = final_info.get("target_point")
                    _tp_world = (
                        _local_xy_to_world_list(_tp_local, ego_xy_before_step, ego_heading_before_step)
                        if _tp_local is not None else None
                    )
                    _pp_local = final_info.get("preference_point")
                    _pp_world = (
                        _local_xy_to_world_list(_pp_local, ego_xy_before_step, ego_heading_before_step)
                        if _pp_local is not None else None
                    )
                    if _cands is not None:
                        _cands = np.asarray(_cands, dtype=np.float32)
                        if _cands.ndim == 3 and _cands.shape[2] >= 2:
                            _cands = _cands[:, :, :2]
                    _overlay = _build_step_overlay_polylines(
                        ego_xy=ego_xy_before_step,
                        ego_heading_rad=float(ego_heading_before_step or 0.0),
                        coarse_anchor_trajectories=np.asarray(_coarse, dtype=np.float32) if _coarse is not None else None,
                        multimodal_trajectories=_cands,
                        selected_mode_idx=int(_sel) if _sel is not None else None,
                        topology_polyline_world=None,
                        mode_valid_mask=None,
                        mode_slot_names=_mode_slot_names_from_config(transfuser_config),
                        target_point_world=np.asarray(_tp_world, dtype=np.float64) if _tp_world is not None else None,
                        preference_point_world=np.asarray(_pp_world, dtype=np.float64) if _pp_world is not None else None,
                        peer_ego_data=None,
                    )
                    _combined = _capture_topdown_frame_with_overlay(
                        env, _overlay,
                        screen_size=800, film_size=10000,
                        camera_position=tuple(ego_xy_before_step),
                    )
                    if _combined is not None:
                        _comb_path = (
                            Path(args.output_dir)
                            / "combined_traj_frames"
                            / f"episode_{episode_idx:03d}"
                            / f"step_{episode_length:05d}.png"
                        )
                        _comb_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(_comb_path), cv2.cvtColor(_combined, cv2.COLOR_RGB2BGR))

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
