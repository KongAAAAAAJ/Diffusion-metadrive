from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable
import cv2
import numpy as np
import torch, time
from metadrive.exp_dataset.route_definitions import get_required_preset, get_route_blocks
from metadrive.exp_dataset.scenario_definitions import SCENARIO_BY_ID, get_scenario_definition
from metadrive.exp_dataset.scenario_orchestrator import ScenarioOrchestrator
from metadrive.envs.diffusion_envs.base_multi_env import DatasetCollectEnv
from metadrive.policy.diffusion_policy.run_dir_utils import create_numbered_run_dir
from metadrive.policy.diffusion_policy.transfuser_callback import render_closed_loop_prediction
from metadrive.policy.diffusion_policy.transfuser_config import build_transfuser_config, transfuser_config_to_dict
from metadrive.policy.diffusion_policy.transfuser_policy import TransfuserPolicy

MULTIMODAL_SELECTED_COLOR = "#C76B00"
MULTIMODAL_OTHER_COLOR = "#1F6F8B"
ROAD_BOUNDARY_COLOR = "#7A7A7A"
ACTUAL_TRAJECTORY_COLOR = "#1D4ED8"


@dataclass
class StepTrajectoryPlotRecord:
    step_idx: int
    ego_position: np.ndarray
    selected_trajectory: np.ndarray | None
    multimodal_trajectories: np.ndarray | None
    selected_mode_idx: int | None
    dynamic_anchor_trajectories: np.ndarray | None = None
    target_point_world: np.ndarray | None = None
    topdown_frame: np.ndarray | None = None
    world_to_screen_projector: Callable[[np.ndarray], np.ndarray] | None = None
    ego_speed_km_h: float | None = None
    ego_acceleration: float | None = None


@dataclass(frozen=True)
class ScenarioRouteSelection:
    scenario_id: str
    local_route: str
    route_preset: str
    ego_main_route_block_ids: tuple[str, ...]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Closed-loop evaluation for MetaDrive TransFuser.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained TransFuser checkpoint.")
    parser.add_argument("--model-size", type=str, default="auto")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--render", type=int, choices=(0, 1), default=0)
    parser.add_argument("--image-on-cuda", type=int, choices=(0, 1), default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--target-speed-km-h", type=float, default=30.0)
    parser.add_argument("--lookahead-index", type=int, default=2)
    parser.add_argument("--controller-type", type=str, default="stabilized")
    parser.add_argument("--print-trajectory-debug", type=int, choices=(0, 1), default=1)
    parser.add_argument("--save-camera-interval", type=int, default=0)  # 默认关闭相机图片保存
    parser.add_argument("--camera-output-dir", type=str, default="/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/closed_loop/cameras")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-scenarios", type=int, default=1)
    parser.add_argument("--traffic-density", type=float, default=0.06)
    parser.add_argument("--plan-anchor-path", type=str, default="metadrive/exp_dataset/anchors.npy")
    parser.add_argument("--trajectory-reg-decoder-type", type=str, choices=("mlp", "gru"), default="mlp")
    parser.add_argument("--save-3d-video", type=int, choices=(0, 1), default=0)
    parser.add_argument("--save-2d-video", type=int, choices=(0, 1), default=1)
    parser.add_argument("--save-trajectory-plot", type=int, choices=(0, 1), default=1)
    parser.add_argument("--save-step-images", type=int, choices=(0, 1), default=0)
    parser.add_argument("--step-image-interval", type=int, default=1)
    parser.add_argument(
        "--scenario-id",
        type=str,
        default="S1_free_cruise_straight",
        help="Scenario id to evaluate. The script samples one allowed local route per episode.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/closed_loop",
    )
    parser.add_argument("--video-fps", type=int, default=10)
    parser.add_argument("--topdown-camera-height", type=float, default=80.0)
    return parser.parse_args(argv)


def _normalize_visualization_args(args):
    if args.save_3d_video and not args.render:
        args.render = 1
    return args


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


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
    frame = env.render(
        mode="top_down",
        window=False,
        screen_size=(screen_size, screen_size),
        film_size=(film_size, film_size),
        target_agent_heading_up=False,
        camera_position=(float(ego_pos[0]), float(ego_pos[1])),
    )
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

    if selected_world_array is not None or candidates_world_array is not None:
        _ego_speed = float(getattr(ego_before_step, "speed_km_h", 0.0)) if ego_before_step is not None else None
        step_plot_records.append(
            StepTrajectoryPlotRecord(
                step_idx=episode_length,
                ego_position=np.asarray(ego_xy_before_step, dtype=np.float64),
                selected_trajectory=selected_world_array,
                multimodal_trajectories=candidates_world_array,
                selected_mode_idx=(int(mode_idx) if mode_idx is not None else None),
                dynamic_anchor_trajectories=dynamic_anchor_world_array,
                target_point_world=target_point_world,
                topdown_frame=(None if topdown_frame is None else np.asarray(topdown_frame, dtype=np.uint8).copy()),
                world_to_screen_projector=world_to_screen_projector,
                ego_speed_km_h=_ego_speed,
                ego_acceleration=ego_accel_mps2,
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

    if step_record.dynamic_anchor_trajectories is not None:
        for anchor in np.asarray(step_record.dynamic_anchor_trajectories, dtype=np.float64):
            full_path = np.vstack([ego, anchor])
            _draw_world_polyline(full_path, (205, 214, 244), 2, alpha=0.7)

    # ── 2. Draw non-selected modes first, selected mode last (on top) ──────────
    selected_mode_idx = step_record.selected_mode_idx
    if step_record.multimodal_trajectories is not None:
        candidates_array = np.asarray(step_record.multimodal_trajectories, dtype=np.float64)
        # Pass 1: non-selected modes
        for mode_i in range(candidates_array.shape[0]):
            if selected_mode_idx is not None and mode_i == int(selected_mode_idx):
                continue
            full_path = np.vstack([ego, candidates_array[mode_i]])
            _draw_world_polyline(full_path, (111, 179, 206), 2, alpha=0.72)
        # Pass 2: selected mode drawn last → always on top
        if selected_mode_idx is not None and selected_mode_idx < candidates_array.shape[0]:
            full_path = np.vstack([ego, candidates_array[int(selected_mode_idx)]])
            _draw_world_polyline(full_path, (242, 153, 74), 4, alpha=1.0)
    elif step_record.selected_trajectory is not None:
        selected_path = np.vstack([ego, np.asarray(step_record.selected_trajectory, dtype=np.float64)])
        _draw_world_polyline(selected_path, (242, 153, 74), 4, alpha=1.0)

    # Ego dot (draw after trajectories so it's always on top)
    ego_zoomed = np.round(_proj_zoomed(ego)).astype(np.int32)
    cv2.circle(canvas, tuple(int(v) for v in ego_zoomed), 7, (70, 190, 90), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(canvas, tuple(int(v) for v in ego_zoomed), 7, (30, 140, 50), thickness=1, lineType=cv2.LINE_AA)

    if step_record.target_point_world is not None:
        target_xy = np.round(_proj_zoomed(np.asarray(step_record.target_point_world, dtype=np.float64))).astype(np.int32)
        cv2.circle(
            canvas,
            tuple(int(v) for v in target_xy),
            5,
            (0, 0, 220),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        cv2.circle(
            canvas,
            tuple(int(v) for v in target_xy),
            5,
            (0, 0, 160),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    # ── 3. Info box with selected mode name ────────────────────────────────────
    try:
        from metadrive.policy.diffusion_policy.mode_definitions import MODE_SLOTS
        mode_name = MODE_SLOTS[int(selected_mode_idx)].name if selected_mode_idx is not None else "N/A"
    except Exception:
        mode_name = f"mode_{selected_mode_idx}" if selected_mode_idx is not None else "N/A"

    box_x1, box_y1, box_x2, box_y2 = 10, 10, 420, 128
    cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), (247, 248, 250), thickness=-1)
    cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), (210, 214, 220), thickness=1)
    cv2.putText(
        canvas,
        f"Episode {episode_idx}  Step {step_record.step_idx}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1, cv2.LINE_AA,
    )
    # ── 3a. Selected mode label (orange, prominent) ───
    cv2.putText(
        canvas,
        f"mode: {mode_name}",
        (20, 52),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 100, 20), 2, cv2.LINE_AA,
    )
    # ── 3b. Speed and acceleration ───
    _speed_str = f"{step_record.ego_speed_km_h:.1f} km/h" if step_record.ego_speed_km_h is not None else "-- km/h"
    _accel_str = f"{step_record.ego_acceleration:+.2f} m/s\u00b2" if step_record.ego_acceleration is not None else "-- m/s\u00b2"
    cv2.putText(
        canvas,
        f"speed: {_speed_str}  accel: {_accel_str}",
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (40, 40, 120), 1, cv2.LINE_AA,
    )
    cv2.putText(canvas, "selected: orange", (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (242, 153, 74), 1, cv2.LINE_AA)
    cv2.putText(canvas, "others: teal", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (111, 179, 206), 1, cv2.LINE_AA)
    cv2.putText(canvas, "anchors: light blue", (160, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 150, 210), 1, cv2.LINE_AA)

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


def build_env_config(args, resolved_model_size: str):
    transfuser_config = build_transfuser_config(
        resolved_model_size,
        trajectory_reg_decoder_type=args.trajectory_reg_decoder_type,
    )
    if args.plan_anchor_path:
        transfuser_config.plan_anchor_path = args.plan_anchor_path

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


def _resolve_episode_scenario_route(scenario_id: str, rng: np.random.RandomState) -> ScenarioRouteSelection:
    if scenario_id not in SCENARIO_BY_ID:
        valid_ids = ", ".join(sorted(SCENARIO_BY_ID))
        raise ValueError(f"Unknown scenario_id '{scenario_id}'. Valid scenarios: {valid_ids}")

    scenario = get_scenario_definition(scenario_id)
    allowed_routes = tuple(scenario.allowed_local_routes)
    if not allowed_routes:
        raise ValueError(f"Scenario '{scenario_id}' has no allowed local routes.")

    if len(allowed_routes) == 1:
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


def main():
    args = _normalize_visualization_args(parse_args())
    output_root = Path(args.output_dir)
    run_output_dir = create_numbered_run_dir(output_root)
    args.output_dir = str(run_output_dir)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    resolved_model_size = infer_model_size_from_checkpoint(checkpoint_path) if args.model_size == "auto" else args.model_size
    resolved_device = resolve_device(args.device)
    print(f"[test] checkpoint={checkpoint_path}")
    print(f"[test] model_size={resolved_model_size}")
    print(f"[test] device={resolved_device}")
    print(f"[test] controller_type={args.controller_type}")
    print(f"[test] image_on_cuda={bool(args.image_on_cuda)}")
    print(
        f"[test] render={bool(args.render)} save_3d_video={bool(args.save_3d_video)} "
        f"save_2d_video={bool(args.save_2d_video)} save_trajectory_plot={bool(args.save_trajectory_plot)} "
        f"save_step_images={bool(args.save_step_images)}"
    )
    print(f"[test] output_dir={args.output_dir} video_fps={args.video_fps}")
    if args.save_camera_interval > 0:
        print(f"[test] save_camera_interval={args.save_camera_interval} camera_output_dir={args.camera_output_dir}")

    env_config = build_env_config(args, resolved_model_size)
    env = DatasetCollectEnv(env_config)
    transfuser_config = build_transfuser_config(
        resolved_model_size,
        trajectory_reg_decoder_type=args.trajectory_reg_decoder_type,
    )
    if args.plan_anchor_path:
        transfuser_config.plan_anchor_path = args.plan_anchor_path
    anchors = None
    anchor_path = Path(transfuser_config.plan_anchor_path)
    if anchor_path.exists():
        anchors = np.load(anchor_path)
    summary = {
        "success": 0,
        "crash": 0,
        "out_of_road": 0,
        "episode_reward": [],
        "episode_length": [],
        "lookahead_y": [],
        "lookahead_heading": [],
        "steering": [],
        "mode_idx": [],
    }
    episode_route_rng = np.random.RandomState(args.start_seed)

    try:
        for episode_idx in range(args.episodes):
            selection = _resolve_episode_scenario_route(args.scenario_id, episode_route_rng)
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
                done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))

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
            summary["episode_reward"].append(episode_reward)
            summary["episode_length"].append(episode_length)
            print(
                f"[episode={episode_idx}] reward={episode_reward:.2f} length={episode_length} "
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
                    )
    finally:
        env.close()

    num_episodes = max(args.episodes, 1)
    mode_hist = dict(sorted(Counter(summary["mode_idx"]).items()))
    print(
        "summary: "
        f"success_rate={summary['success'] / num_episodes:.3f} "
        f"crash_rate={summary['crash'] / num_episodes:.3f} "
        f"out_of_road_rate={summary['out_of_road'] / num_episodes:.3f} "
        f"avg_reward={float(np.mean(summary['episode_reward'])):.2f} "
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


if __name__ == "__main__":
    main()
