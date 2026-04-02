"""collect_expert.py

Collect expert demonstration data from DatasetCollectEnv.

Strategy: a single environment instance is created once with the default
hybrid map (hybrid_map_sequence="SSXCOCSS", num_scenarios=1).  The map
is held fixed across all episodes; only the background traffic density is
re-sampled in [traffic_density_min, traffic_density_max] before each reset,
producing diverse traffic flow on the same road layout.

Each training sample contains:
  ego_state        – ego vehicle state vector              float32
  others_state     – surrounding vehicle state vector      float32
  lidar            – lidar observation vector              float32
  bev_raster       – BEV multi-channel image CHW           uint8   (C,H,W)
  left_camera      – left RGB camera image HWC             uint8   (H,W,3)
  front_camera     – front RGB camera image HWC            uint8   (H,W,3)
  right_camera     – right RGB camera image HWC            uint8   (H,W,3)
  trajectory       – 8 future waypoints in ego-local frame float32 (8,3)
  ego_pose_world   – current [x, y, heading_theta]         float32 (3,)
  action           – last applied [steer, accel]           float32 (2,)
  traffic_density  – episode traffic density scalar        float32 ()

Usage:
    python -m metadrive.dataset.collect_expert \\
        --output-root /data/datasets \\
        --dataset-name metadrive_ppo_hybrid \\
        --expert-type ppo \\
        --target-samples 50000
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.envs.diffusion_envs.base_multi_env import DatasetCollectEnv
from metadrive.examples.ppo_expert import expert as ppo_expert
from metadrive.exp_dataset.trajectory_correction import (
    TrajectoryCorrectionContext,
    TrajectoryMode,
    classify_trajectory_mode,
    correct_trajectory_geometry,
)
from metadrive.policy.idm_policy import FrontBackObjects, IDMPolicy
from metadrive.exp_dataset.expert_idm_policy import ExpertIDMPolicy as Expert
from metadrive.policy.diffusion_policy.transfuser_features import BoundingBox2DIndex
from metadrive.utils import Config
from metadrive.exp_dataset.metadrive_dataset import split_shards





# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExpertCollectorConfig:
    # Output
    output_root: Path = Path("/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets")
    dataset_name: str = "metadrive_ppo_data"
    expert_type: str = "ppo"  # idm | ppo
    target_samples: int = 100
    start_seed: int = 0
    spawn_seed_offset: int = 100000
    samples_per_shard: int = 2048
    resume: bool = False  # 数据集续采

    # Episode
    max_episode_steps: int = 1000  # 每个episode的最大步数

    # Video
    save_videos: bool = False
    video_fps: int = 10
    topdown_camera_height: float = 180.0

    # Trajectory supervision
    horizon_steps: int = 100       # minimum future context required per sample
    target_stride_steps: int = 5  # 轨迹点之间的时间间隔（step）
    sample_stride_steps: int = 2  # 两个样本之间的时间间隔（step）
    trajectory_num_poses: int = 8 # 轨迹点数量
    trajectory_dt: float = 0.1  # 仿真时间步长（s）
    num_bounding_boxes: int = 16
    lidar_min_x: float = -32.0  # lidar的感知范围（相对于ego车坐标系）
    lidar_max_x: float = 32.0
    lidar_min_y: float = -32.0
    lidar_max_y: float = 32.0

    # Traffic (map is fixed; only density varies)
    traffic_density_min: float = 0.06
    traffic_density_max: float = 0.08
    use_hybrid_map: bool = True
    hybrid_map_sequence: str = "SSXCOCSS"
    map_block_num: int = 5
    num_scenarios: int = 1

    # Dataset split
    train_split_ratio: float = 0.8
    val_split_ratio: float = 0.1
    test_split_ratio: float = 0.1
    split_seed: int = 0

    # Trajectory correction
    trajectory_correction_enabled: bool = True
    save_raw_trajectory: bool = True
    mode_classifier_version: str = "v1"
    centerline_attraction_strength: float = 0.85
    smoothing_strength: float = 0.25
    trajectory_visualization_enabled: bool = False
    trajectory_visualization_window_mode: str = "adaptive"
    trajectory_visualization_front_margin: float = 25.0
    trajectory_visualization_rear_margin: float = 8.0
    trajectory_visualization_lateral_margin: float = 10.0
    trajectory_visualization_min_span_x: float = 35.0
    trajectory_visualization_min_span_y: float = 20.0
    trajectory_visualization_max_span_x: float = 90.0
    trajectory_visualization_max_span_y: float = 36.0


# ---------------------------------------------------------------------------
# Traffic sampling
# ---------------------------------------------------------------------------

def sample_traffic_density(rng: np.random.RandomState, config: ExpertCollectorConfig) -> float:
    return float(rng.uniform(config.traffic_density_min, config.traffic_density_max))


def sample_episode_spawn_seed(rng: np.random.RandomState) -> int:
    return int(rng.randint(0, 2**31 - 1))


def build_episode_rngs(config: ExpertCollectorConfig) -> tuple[np.random.RandomState, np.random.RandomState]:
    traffic_rng = np.random.RandomState(config.start_seed)
    spawn_rng = np.random.RandomState(config.start_seed + config.spawn_seed_offset)
    return traffic_rng, spawn_rng


def fast_forward_episode_rngs(
    traffic_rng: np.random.RandomState,
    spawn_rng: np.random.RandomState,
    config: ExpertCollectorConfig,
    completed_episodes: int,
) -> None:
    for _ in range(int(completed_episodes)):
        traffic_rng.uniform(config.traffic_density_min, config.traffic_density_max)
        spawn_rng.randint(0, 2**31 - 1)


def describe_map_config(config: ExpertCollectorConfig) -> str:
    if bool(config.use_hybrid_map):
        return f"hybrid_fixed ({config.hybrid_map_sequence})"
    return f"random_block_map (map={config.map_block_num}, num_scenarios={config.num_scenarios})"


def _to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Config):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, type):
        return value.__name__
    if callable(value):
        return getattr(value, "__name__", str(value))
    return str(value)


def build_expert_policy(
    vehicle,
    random_seed: int,
):
    return Expert(
        control_object=vehicle,
        random_seed=random_seed,
    )


DEFAULT_TOPDOWN_SCREEN_SIZE = 800
DEFAULT_TOPDOWN_FILM_SIZE = 3000
DEFAULT_TEXT_CORNER = "top_left"


def build_topdown_render_kwargs(camera_position: tuple[float, float] | None = None) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "mode": "top_down",
        "window": False,
        "screen_size": (DEFAULT_TOPDOWN_SCREEN_SIZE, DEFAULT_TOPDOWN_SCREEN_SIZE),
        "film_size": (DEFAULT_TOPDOWN_FILM_SIZE, DEFAULT_TOPDOWN_FILM_SIZE),
        "target_agent_heading_up": False,
    }
    if camera_position is not None:
        kwargs["camera_position"] = camera_position
    return kwargs


def get_primary_agent_id(env) -> str | None:
    agents = getattr(env, "agents", {}) or {}
    if "agent0" in agents:
        return "agent0"
    return next(iter(agents.keys()), None)


def sync_topdown_camera_with_agent(env, agent_id: str | None) -> tuple[float, float] | None:
    if agent_id is None:
        return None
    agents = getattr(env, "agents", {}) or {}
    vehicle = agents.get(agent_id)
    if vehicle is None:
        return None
    camera_position = (float(vehicle.position[0]), float(vehicle.position[1]))
    renderer = getattr(env, "top_down_renderer", None)
    if renderer is not None:
        renderer.position = camera_position
    return camera_position


def compute_heading_up_rotation_deg(heading_rad: float) -> float:
    return float(-np.rad2deg(float(heading_rad)) + 90.0)


def rotate_frame_heading_up(frame_array: np.ndarray, heading_rad: float) -> np.ndarray:
    import pygame

    height, width = frame_array.shape[:2]
    rotation_deg = compute_heading_up_rotation_deg(heading_rad)
    source = pygame.surfarray.make_surface(frame_array.swapaxes(0, 1))
    canvas_size = max(width, height) * 2
    canvas = pygame.Surface((canvas_size, canvas_size))
    canvas.fill((255, 255, 255))
    canvas.blit(source, ((canvas_size - width) // 2, (canvas_size - height) // 2))
    rotated = pygame.transform.rotozoom(canvas, rotation_deg, 1.0)
    crop_x = max(rotated.get_width() // 2 - width // 2, 0)
    crop_y = max(rotated.get_height() // 2 - height // 2, 0)
    cropped = pygame.Surface((width, height))
    cropped.fill((255, 255, 255))
    cropped.blit(rotated, (0, 0), (crop_x, crop_y, width, height))
    return pygame.surfarray.array3d(cropped).swapaxes(0, 1)


def build_overlay_lines(episode_index: int, step_count: int) -> list[str]:
    return [
        "expert: collection",
        f"episode: {episode_index}",
        f"step: {step_count}",
    ]


def overlay_text_on_frame(frame_array: np.ndarray, lines: list[str], corner: str = DEFAULT_TEXT_CORNER) -> np.ndarray:
    import pygame

    if not lines:
        return frame_array
    height, width = frame_array.shape[:2]
    surface = pygame.surfarray.make_surface(frame_array.swapaxes(0, 1))
    if not pygame.font.get_init():
        pygame.font.init()
    font = pygame.font.SysFont("Arial", 24)
    rendered = [font.render(line, True, (0, 0, 0)) for line in lines]
    max_width = max(text.get_width() for text in rendered)
    line_height = max(text.get_height() for text in rendered)
    padding = 14
    if corner == "top_right":
        x = max(width - max_width - padding, padding)
    else:
        x = padding
    y = padding
    for idx, text_surface in enumerate(rendered):
        surface.blit(text_surface, (x, y + idx * line_height))
    return pygame.surfarray.array3d(surface).swapaxes(0, 1)


# ---------------------------------------------------------------------------
# Per-frame helpers
# ---------------------------------------------------------------------------

def _to_numpy_array(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)

def pose_to_array(vehicle) -> np.ndarray:
    return np.asarray(
        [vehicle.position[0], vehicle.position[1], vehicle.heading_theta],
        dtype=np.float32,
    )


def world_future_to_local(current_pose: np.ndarray, future_pose: np.ndarray) -> np.ndarray:
    """Transform a future world pose into the current ego-local coordinate frame."""
    dx = future_pose[0] - current_pose[0]
    dy = future_pose[1] - current_pose[1]
    heading = current_pose[2]
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    return np.asarray(
        [
            cos_h * dx + sin_h * dy,
            -sin_h * dx + cos_h * dy,
            future_pose[2] - heading,
        ],
        dtype=np.float32,
    )


def extract_last_action(vehicle) -> np.ndarray:
    return np.asarray(
        getattr(vehicle, "last_current_action", [None, [0.0, 0.0]])[1],
        dtype=np.float32,
    )


def _to_chw_uint8(image_obs: np.ndarray) -> np.ndarray:
    image_obs = _to_numpy_array(image_obs)
    return np.clip(np.moveaxis(image_obs, -1, 0) * 255.0, 0.0, 255.0).astype(np.uint8)


def _to_hwc_uint8(image_obs: np.ndarray) -> np.ndarray:
    image_obs = _to_numpy_array(image_obs)
    return np.clip(image_obs * 255.0, 0.0, 255.0).astype(np.uint8)


def _wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _safe_lane_ordinal(lane_index) -> int | None:
    if lane_index is None:
        return None
    if isinstance(lane_index, (tuple, list)) and lane_index:
        candidate = lane_index[-1]
    else:
        candidate = lane_index
    if isinstance(candidate, (int, np.integer)):
        return int(candidate)
    return None


def _pose_world_to_array(position, heading_theta: float) -> np.ndarray:
    return np.asarray([position[0], position[1], heading_theta], dtype=np.float32)


def _select_reference_lane(vehicle):
    navigation = getattr(vehicle, "navigation", None)
    current_ref_lanes = getattr(navigation, "current_ref_lanes", None)
    if current_ref_lanes:
        lane_idx = _safe_lane_ordinal(getattr(vehicle, "lane_index", None))
        if lane_idx is not None and 0 <= lane_idx < len(current_ref_lanes):
            return current_ref_lanes[lane_idx]
        if getattr(vehicle, "lane", None) in current_ref_lanes:
            return vehicle.lane
        return current_ref_lanes[0]
    return getattr(vehicle, "lane", None)


def _extract_front_object_state(vehicle, ref_lane) -> tuple[float | None, float | None]:
    if ref_lane is None or not hasattr(vehicle, "lidar"):
        return None, None
    try:
        current_ref_lanes = getattr(vehicle.navigation, "current_ref_lanes", None)
        all_objects = vehicle.lidar.get_surrounding_objects(vehicle)
        surrounding_objects = FrontBackObjects.get_find_front_back_objs(
            all_objects,
            ref_lane,
            vehicle.position,
            max_distance=IDMPolicy.MAX_LONG_DIST,
            ref_lanes=current_ref_lanes if current_ref_lanes and ref_lane in current_ref_lanes else None,
        )
        front_object = surrounding_objects.front_object()
        front_distance = surrounding_objects.front_min_distance()
        front_speed = None if front_object is None else float(getattr(front_object, "speed_km_h", 0.0))
        return float(front_distance), front_speed
    except Exception:
        return None, None


def _world_to_ego_xy(vehicle, obj_position: np.ndarray) -> np.ndarray:
    dx = obj_position[0] - vehicle.position[0]
    dy = obj_position[1] - vehicle.position[1]
    heading = vehicle.heading_theta
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    return np.asarray(
        [cos_h * dx + sin_h * dy, -sin_h * dx + cos_h * dy],
        dtype=np.float32,
    )


def extract_agent_targets(vehicle, config: ExpertCollectorConfig) -> tuple[np.ndarray, np.ndarray]:
    # 捕获在lidar范围内的所有车
    candidates = []
    for obj in vehicle.engine.get_objects(lambda o: isinstance(o, BaseVehicle)).values():
        if obj is vehicle:
            continue
        local_xy = _world_to_ego_xy(vehicle, obj.position)
        x, y = float(local_xy[0]), float(local_xy[1])
        if not (config.lidar_min_x <= x <= config.lidar_max_x and config.lidar_min_y <= y <= config.lidar_max_y):  
            continue
        heading = _wrap_to_pi(float(obj.heading_theta - vehicle.heading_theta))
        candidates.append(np.asarray([x, y, heading, float(obj.LENGTH), float(obj.WIDTH)], dtype=np.float32))

    # 从 candidates 中选出的、距离 ego 车最近的前 num_bounding_boxes 个目标，并构建 agent_states 和 agent_labels
    agent_states = np.zeros((config.num_bounding_boxes, BoundingBox2DIndex.size()), dtype=np.float32)
    agent_labels = np.zeros((config.num_bounding_boxes,), dtype=bool)
    if candidates:
        candidates = np.stack(candidates, axis=0)
        distances = np.linalg.norm(candidates[:, BoundingBox2DIndex.POINT], axis=-1)
        order = np.argsort(distances)[:config.num_bounding_boxes]
        selected = candidates[order]
        agent_states[: len(selected)] = selected
        agent_labels[: len(selected)] = True
    return agent_states, agent_labels


# ?用于? 将 BEV 多通道图像转为单通道的语义分割标签图
# 返回一个二维整型数组，每个像素的值表示其语义类别（0=背景，1=道路，2=交通，3=自车轨迹）。
def build_bev_semantic_map(bev_raster: np.ndarray) -> np.ndarray:
    road = bev_raster[0] > 0
    ego_history = bev_raster[1] > 0 if bev_raster.shape[0] > 1 else np.zeros_like(road)
    traffic = bev_raster[2:].max(axis=0) > 0 if bev_raster.shape[0] > 2 else np.zeros_like(road)

    semantic = np.zeros(road.shape, dtype=np.int64)
    semantic[road] = 1
    semantic[traffic] = 2
    semantic[ego_history] = 3
    return semantic


def build_frame(
    vehicle,
    collector_config: ExpertCollectorConfig,
    ego_state_obs: np.ndarray,
    others_state_obs: np.ndarray,
    lidar_obs: np.ndarray,
    topdown_obs: np.ndarray,
    left_camera_obs: np.ndarray,
    front_camera_obs: np.ndarray,
    right_camera_obs: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Pack one timestep observation into numpy arrays.

    Args:
        vehicle:    MetaDrive vehicle instance (for pose and last action).
        ego_state_obs: ego state vector.
        others_state_obs: surrounding vehicle state vector.
        lidar_obs: pure lidar vector.
        topdown_obs: BEV image (H, W, C) float32 in [0, 1].
        left_camera_obs/front_camera_obs/right_camera_obs: RGB images (H, W, C) float32 in [0, 1].
    """
    ego_state = _to_numpy_array(ego_state_obs).astype(np.float32, copy=False)
    other_states = _to_numpy_array(others_state_obs).astype(np.float32, copy=False)
    lidar = _to_numpy_array(lidar_obs).astype(np.float32, copy=False)
    state_275 = np.concatenate([ego_state, other_states, lidar], axis=0).astype(np.float32)
    bev_raster = _to_chw_uint8(topdown_obs)
    agent_states, agent_labels = extract_agent_targets(vehicle, collector_config)
    reference_lane = _select_reference_lane(vehicle)
    if reference_lane is not None:
        ref_long, ref_lat = reference_lane.local_coordinates(vehicle.position)
        reference_pose_world = _pose_world_to_array(
            reference_lane.position(ref_long, 0.0),
            reference_lane.heading_theta_at(ref_long),
        )
        reference_lane_index = _safe_lane_ordinal(getattr(reference_lane, "index", None))
        lane_width = float(getattr(reference_lane, "width", getattr(vehicle.lane, "width", 4.0)))
    else:
        ref_long, ref_lat = 0.0, 0.0
        reference_pose_world = pose_to_array(vehicle)
        reference_lane_index = _safe_lane_ordinal(getattr(vehicle, "lane_index", None))
        lane_width = float(getattr(vehicle.lane, "width", 4.0))
    front_object_distance, front_object_speed_km_h = _extract_front_object_state(vehicle, reference_lane)
    current_ref_lanes = getattr(vehicle.navigation, "current_ref_lanes", None)
    next_ref_lanes = getattr(vehicle.navigation, "next_ref_lanes", None)
    return {
        "ego_state": ego_state,
        "other_states": other_states,
        "lidar": lidar,
        "state_275": state_275,
        "bev_raster": bev_raster,
        "bev_semantic_map": build_bev_semantic_map(bev_raster),
        "agent_states": agent_states,
        "agent_labels": agent_labels,
        "left_camera": _to_hwc_uint8(left_camera_obs),
        "front_camera": _to_hwc_uint8(front_camera_obs),
        "right_camera": _to_hwc_uint8(right_camera_obs),
        "camera": _to_hwc_uint8(front_camera_obs),
        "rgb": _to_hwc_uint8(front_camera_obs),
        "ego_pose_world": pose_to_array(vehicle),
        "reference_pose_world": reference_pose_world,
        "action": extract_last_action(vehicle),
        "lane_index": np.asarray(
            -1 if _safe_lane_ordinal(getattr(vehicle, "lane_index", None)) is None
            else _safe_lane_ordinal(getattr(vehicle, "lane_index", None)),
            dtype=np.int16,
        ),
        "reference_lane_index": np.asarray(-1 if reference_lane_index is None else reference_lane_index, dtype=np.int16),
        "reference_longitudinal": np.asarray(ref_long, dtype=np.float32),
        "reference_lateral": np.asarray(ref_lat, dtype=np.float32),
        "lane_width": np.asarray(lane_width, dtype=np.float32),
        "current_ref_lane_count": np.asarray(len(current_ref_lanes) if current_ref_lanes else 1, dtype=np.int16),
        "next_ref_lane_count": np.asarray(len(next_ref_lanes), dtype=np.int16) if next_ref_lanes is not None else np.asarray(-1, dtype=np.int16),
        "front_object_distance": np.asarray(-1.0 if front_object_distance is None else front_object_distance, dtype=np.float32),
        "front_object_speed_km_h": np.asarray(-1.0 if front_object_speed_km_h is None else front_object_speed_km_h, dtype=np.float32),
        "ego_speed_km_h": np.asarray(float(vehicle.speed_km_h), dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Sample construction (sliding window over one episode)
# ---------------------------------------------------------------------------

def build_episode_samples(
    frames: List[Dict[str, np.ndarray]],
    config: ExpertCollectorConfig,
    traffic_density: float,
    visualization_dir: Path | None = None,
    episode_index: int = 0,
    map_geometry: List[Dict[str, np.ndarray]] | None = None,
) -> List[Dict[str, np.ndarray]]:
    """Slide a window over episode frames to build labelled training samples."""
    frame_count = len(frames)
    if frame_count <= config.horizon_steps:
        return []

    future_offsets = tuple(
        config.target_stride_steps * (i + 1) for i in range(config.trajectory_num_poses)
    )
    last_offset = future_offsets[-1]
    td_arr = np.asarray(traffic_density, dtype=np.float32)

    samples: List[Dict[str, np.ndarray]] = []
    for start_idx in range(0, frame_count - config.horizon_steps, config.sample_stride_steps):
        if start_idx + last_offset >= frame_count:
            continue

        current_frame = frames[start_idx]
        current_pose = current_frame["ego_pose_world"]
        raw_trajectory = np.stack(
            [
                world_future_to_local(current_pose, frames[start_idx + offset]["ego_pose_world"])
                for offset in future_offsets
            ],
            axis=0,
        )
        reference_trajectory = np.stack(
            [
                world_future_to_local(current_pose, frames[start_idx + offset]["reference_pose_world"])
                for offset in future_offsets
            ],
            axis=0,
        )
        future_lane_indices = tuple(
            int(frames[start_idx + offset]["reference_lane_index"])
            if int(frames[start_idx + offset]["reference_lane_index"]) >= 0 else None
            for offset in future_offsets
        )
        current_lane_index = int(current_frame["reference_lane_index"]) if int(current_frame["reference_lane_index"]) >= 0 else None
        next_ref_lane_count = int(current_frame["next_ref_lane_count"])
        context = TrajectoryCorrectionContext(
            current_lane_index=current_lane_index,
            future_lane_indices=future_lane_indices,
            current_ref_lane_count=max(int(current_frame["current_ref_lane_count"]), 1),
            next_ref_lane_count=(next_ref_lane_count if next_ref_lane_count >= 0 else None),
            lane_width=max(float(current_frame["lane_width"]), 1.0),
            front_object_distance=(
                None if float(current_frame["front_object_distance"]) < 0.0 else float(current_frame["front_object_distance"])
            ),
            ego_speed_km_h=float(current_frame["ego_speed_km_h"]),
            front_object_speed_km_h=(
                None if float(current_frame["front_object_speed_km_h"]) < 0.0 else float(current_frame["front_object_speed_km_h"])
            ),
        )
        trajectory_mode = classify_trajectory_mode(raw_trajectory, context)
        trajectory = raw_trajectory
        correction_metrics = {
            "strong_correction": 0.0,
            "mean_abs_lateral_before": float(np.mean(np.abs(raw_trajectory[:, 1]))),
            "mean_abs_lateral_after": float(np.mean(np.abs(raw_trajectory[:, 1]))),
            "final_abs_lateral_before": float(np.abs(raw_trajectory[-1, 1])),
            "final_abs_lateral_after": float(np.abs(raw_trajectory[-1, 1])),
            "mean_point_shift": 0.0,
        }
        if bool(config.trajectory_correction_enabled) and config.expert_type == "ppo":
            trajectory, correction_metrics = correct_trajectory_geometry(
                raw_trajectory,
                trajectory_mode,
                reference_trajectory=reference_trajectory,
                attraction_strength=config.centerline_attraction_strength,
                smoothing_strength=config.smoothing_strength,
            )
        if (
            bool(config.trajectory_visualization_enabled)
            and config.expert_type == "ppo"
            and visualization_dir is not None
        ):
            plot_path = visualization_dir / (
                f"ep_{episode_index:05d}_sample_{start_idx:05d}_"
                f"{trajectory_mode.name.lower()}.png"
            )
            save_trajectory_visualization(
                output_path=plot_path,
                raw_trajectory=raw_trajectory,
                corrected_trajectory=trajectory,
                config=config,
                trajectory_mode=trajectory_mode,
                episode_index=episode_index,
                sample_index=start_idx,
                current_pose=current_pose,
                map_geometry=map_geometry,
            )

        samples.append(
            {
                "ego_state": current_frame["ego_state"],
                "other_states": current_frame["other_states"],
                "lidar": current_frame["lidar"],
                "bev_raster": current_frame["bev_raster"],
                "bev_semantic_map": current_frame["bev_semantic_map"],
                "agent_states": current_frame["agent_states"],
                "agent_labels": current_frame["agent_labels"],
                "left_camera": current_frame["left_camera"],
                "front_camera": current_frame["front_camera"],
                "right_camera": current_frame["right_camera"],
                "camera": current_frame["camera"],
                "rgb": current_frame["rgb"],
                "state_275": current_frame["state_275"],
                "trajectory": trajectory,
                "trajectory_mode": np.asarray(int(trajectory_mode), dtype=np.int8),
                "trajectory_correction_strength": np.asarray(correction_metrics["strong_correction"], dtype=np.float32),
                "trajectory_mean_abs_lateral_before": np.asarray(correction_metrics["mean_abs_lateral_before"], dtype=np.float32),
                "trajectory_mean_abs_lateral_after": np.asarray(correction_metrics["mean_abs_lateral_after"], dtype=np.float32),
                "trajectory_final_abs_lateral_before": np.asarray(correction_metrics["final_abs_lateral_before"], dtype=np.float32),
                "trajectory_final_abs_lateral_after": np.asarray(correction_metrics["final_abs_lateral_after"], dtype=np.float32),
                "ego_pose_world": current_frame["ego_pose_world"],
                "reference_pose_world": current_frame["reference_pose_world"],
                "action": current_frame["action"],
                "ego_speed_km_h": np.asarray(current_frame["ego_speed_km_h"], dtype=np.float32),
                "front_object_distance": np.asarray(current_frame["front_object_distance"], dtype=np.float32),
                "front_object_speed_km_h": np.asarray(current_frame["front_object_speed_km_h"], dtype=np.float32),
                "lane_index": np.asarray(current_frame["lane_index"], dtype=np.int16),
                "reference_lane_index": np.asarray(current_frame["reference_lane_index"], dtype=np.int16),
                "reference_longitudinal": np.asarray(current_frame["reference_longitudinal"], dtype=np.float32),
                "reference_lateral": np.asarray(current_frame["reference_lateral"], dtype=np.float32),
                "lane_width": np.asarray(current_frame["lane_width"], dtype=np.float32),
                "current_ref_lane_count": np.asarray(current_frame["current_ref_lane_count"], dtype=np.int16),
                "next_ref_lane_count": np.asarray(current_frame["next_ref_lane_count"], dtype=np.int16),
                "traffic_density": td_arr,
            }
        )
        if bool(config.save_raw_trajectory):
            samples[-1]["trajectory_raw"] = raw_trajectory

    return samples


def format_eta(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:02d}m{secs:02d}s"


def build_episode_video_path(video_dir: Path, episode_index: int) -> Path:
    return video_dir / f"episode_{int(episode_index):06d}.mp4"


def capture_episode_topdown_frame(env, episode_index: int, step_count: int) -> np.ndarray:
    primary_agent_id = get_primary_agent_id(env)
    camera_position = sync_topdown_camera_with_agent(env, primary_agent_id)
    render_kwargs = build_topdown_render_kwargs(camera_position=camera_position)
    if getattr(env, "top_down_renderer", None) is None:
        frame = env.render(**render_kwargs)
        sync_topdown_camera_with_agent(env, primary_agent_id)
    else:
        frame = env.render(**build_topdown_render_kwargs())

    frame_array = np.asarray(frame)
    if frame_array.ndim >= 2:
        frame_array = frame_array.swapaxes(0, 1)
    agents = getattr(env, "agents", {}) or {}
    primary_agent = agents.get(primary_agent_id) if primary_agent_id is not None else None
    if primary_agent is not None:
        frame_array = rotate_frame_heading_up(frame_array, float(primary_agent.heading_theta))
    return overlay_text_on_frame(frame_array, build_overlay_lines(episode_index, step_count))


def write_episode_video(video_path: Path, frames: List[np.ndarray], fps: int) -> None:
    if not frames:
        return

    import mediapy

    video_path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(video_path), frames, fps=int(fps))


def _iter_road_network_lanes(road_network) -> List:
    graph = getattr(road_network, "graph", None)
    if graph is None:
        return []

    lanes = []
    seen = set()

    def visit(node) -> None:
        if node is None:
            return
        if hasattr(node, "get_polyline") and hasattr(node, "position"):
            node_id = id(node)
            if node_id not in seen:
                seen.add(node_id)
                lanes.append(node)
            return
        if hasattr(node, "lane"):
            visit(node.lane)
            return
        if isinstance(node, dict):
            for value in node.values():
                visit(value)
            return
        if isinstance(node, (list, tuple, set)):
            for value in node:
                visit(value)

    visit(graph)
    return lanes


def _sample_lane_polyline_world(lane, lateral_scale: float, interval: float = 2.0) -> np.ndarray:
    longs = np.arange(0.0, max(float(lane.length), 0.0), interval, dtype=np.float32)
    longs = np.concatenate([longs, np.asarray([float(lane.length)], dtype=np.float32)])
    points = []
    for longitudinal in longs:
        lateral = float(lane.width_at(float(longitudinal))) * lateral_scale
        point = lane.position(float(longitudinal), lateral)
        points.append(np.asarray(point[:2], dtype=np.float32))
    return np.stack(points, axis=0)


def build_map_visualization_geometry(env: DatasetCollectEnv) -> List[Dict[str, np.ndarray]]:
    road_network = getattr(env.current_map, "road_network", None)
    if road_network is None:
        return []
    geometry = []
    for lane in _iter_road_network_lanes(road_network):
        geometry.append(
            {
                "center": _sample_lane_polyline_world(lane, 0.0),
                "left_boundary": _sample_lane_polyline_world(lane, -0.5),
                "right_boundary": _sample_lane_polyline_world(lane, 0.5),
            }
        )
    return geometry


def resolve_map_visualization_geometry(
    env: DatasetCollectEnv,
    visualization_enabled: bool,
    cached_geometry: List[Dict[str, np.ndarray]] | None,
) -> List[Dict[str, np.ndarray]] | None:
    if not visualization_enabled:
        return None
    if cached_geometry is not None:
        return cached_geometry
    engine = getattr(env, "engine", None)
    if engine is None or getattr(engine, "current_map", None) is None:
        return None
    return build_map_visualization_geometry(env)


def world_polyline_to_local(current_pose: np.ndarray, polyline_world: np.ndarray) -> np.ndarray:
    polyline_world = np.asarray(polyline_world, dtype=np.float32)
    dx = polyline_world[:, 0] - float(current_pose[0])
    dy = polyline_world[:, 1] - float(current_pose[1])
    heading = float(current_pose[2])
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    local_x = cos_h * dx + sin_h * dy
    local_y = -sin_h * dx + cos_h * dy
    return np.stack([local_x, local_y], axis=1).astype(np.float32, copy=False)


def compute_visualization_window(
    raw_xy: np.ndarray,
    corrected_xy: np.ndarray,
    config: ExpertCollectorConfig,
) -> tuple[float, float, float, float]:
    all_xy = np.concatenate([raw_xy[:, :2], corrected_xy[:, :2], np.zeros((1, 2), dtype=np.float32)], axis=0)
    x_min = float(np.min(all_xy[:, 0])) - float(config.trajectory_visualization_rear_margin)
    x_max = float(np.max(all_xy[:, 0])) + float(config.trajectory_visualization_front_margin)
    y_min = float(np.min(all_xy[:, 1])) - float(config.trajectory_visualization_lateral_margin)
    y_max = float(np.max(all_xy[:, 1])) + float(config.trajectory_visualization_lateral_margin)

    span_x = x_max - x_min
    span_y = y_max - y_min
    min_span_x = float(config.trajectory_visualization_min_span_x)
    min_span_y = float(config.trajectory_visualization_min_span_y)
    max_span_x = float(config.trajectory_visualization_max_span_x)
    max_span_y = float(config.trajectory_visualization_max_span_y)

    if span_x < min_span_x:
        center_x = 0.5 * (x_min + x_max)
        x_min = center_x - 0.5 * min_span_x
        x_max = center_x + 0.5 * min_span_x
    if span_y < min_span_y:
        center_y = 0.5 * (y_min + y_max)
        y_min = center_y - 0.5 * min_span_y
        y_max = center_y + 0.5 * min_span_y

    if (x_max - x_min) > max_span_x:
        center_x = 0.5 * (x_min + x_max)
        x_min = center_x - 0.5 * max_span_x
        x_max = center_x + 0.5 * max_span_x
    if (y_max - y_min) > max_span_y:
        center_y = 0.5 * (y_min + y_max)
        y_min = center_y - 0.5 * max_span_y
        y_max = center_y + 0.5 * max_span_y

    return x_min, x_max, y_min, y_max


def polyline_intersects_window(
    polyline_xy: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> bool:
    polyline_xy = np.asarray(polyline_xy, dtype=np.float32)
    if polyline_xy.size == 0:
        return False
    return not (
        float(np.max(polyline_xy[:, 0])) < x_min or
        float(np.min(polyline_xy[:, 0])) > x_max or
        float(np.max(polyline_xy[:, 1])) < y_min or
        float(np.min(polyline_xy[:, 1])) > y_max
    )


def save_trajectory_visualization(
    output_path: Path,
    raw_trajectory: np.ndarray,
    corrected_trajectory: np.ndarray,
    config: ExpertCollectorConfig,
    trajectory_mode: TrajectoryMode,
    episode_index: int,
    sample_index: int,
    current_pose: np.ndarray,
    map_geometry: List[Dict[str, np.ndarray]] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_xy = np.asarray(raw_trajectory[:, :2], dtype=np.float32)
    corrected_xy = np.asarray(corrected_trajectory[:, :2], dtype=np.float32)
    x_min, x_max, y_min, y_max = compute_visualization_window(raw_xy, corrected_xy, config)

    plt.figure(figsize=(6, 4))
    if map_geometry:
        for lane_geometry in map_geometry:
            center_xy = world_polyline_to_local(current_pose, lane_geometry["center"])
            left_xy = world_polyline_to_local(current_pose, lane_geometry["left_boundary"])
            right_xy = world_polyline_to_local(current_pose, lane_geometry["right_boundary"])
            if not (
                polyline_intersects_window(center_xy, x_min, x_max, y_min, y_max)
                or polyline_intersects_window(left_xy, x_min, x_max, y_min, y_max)
                or polyline_intersects_window(right_xy, x_min, x_max, y_min, y_max)
            ):
                continue
            plt.plot(left_xy[:, 0], left_xy[:, 1], color="#808080", linewidth=0.9, alpha=0.4, zorder=1)
            plt.plot(right_xy[:, 0], right_xy[:, 1], color="#808080", linewidth=0.9, alpha=0.4, zorder=1)
            plt.plot(
                center_xy[:, 0], center_xy[:, 1],
                color="#caa64b", linewidth=0.8, alpha=0.45, linestyle="--", zorder=1
            )
    plt.plot(raw_xy[:, 0], raw_xy[:, 1], marker="o", linewidth=2, label="ppo_raw")
    plt.plot(corrected_xy[:, 0], corrected_xy[:, 1], marker="o", linewidth=2, label="corrected")
    plt.scatter([0.0], [0.0], c="black", s=30, label="ego_start")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"ep={episode_index} sample={sample_index} mode={trajectory_mode.name.lower()}")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"Unable to parse boolean value: {value}")


def trajectory_mode_name(mode_value: int) -> str:
    return TrajectoryMode(int(mode_value)).name.lower()


def build_trajectory_mode_summary(mode_counts: Dict[str, int], total_samples: int) -> Dict[str, Dict[str, float]]:
    denom = max(int(total_samples), 1)
    return {
        mode_name: {
            "count": int(count),
            "ratio": float(count) / float(denom),
        }
        for mode_name, count in mode_counts.items()
    }


def detect_existing_state(shard_dir: Path, report_dir: Path) -> Dict[str, object]:
    shard_paths = sorted(shard_dir.glob("shard_*.npz"))
    manifest_path = report_dir / "manifest.json"

    total_samples = 0
    total_episodes = 0
    mode_counts: Dict[str, int] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        total_samples = int(manifest.get("collected_samples", 0))
        total_episodes = int(manifest.get("episodes", 0))
        correction_summary = manifest.get("trajectory_correction", {}) or {}
        raw_mode_counts = correction_summary.get("mode_counts", {}) or {}
        mode_counts = {str(name): int(count) for name, count in raw_mode_counts.items()}
    else:
        for shard_path in shard_paths:
            with np.load(shard_path, allow_pickle=False) as shard_data:
                first_key = next(iter(shard_data.files), None)
                if first_key is None:
                    continue
                total_samples += int(shard_data[first_key].shape[0])

    return {
        "shard_count": len(shard_paths),
        "total_samples": total_samples,
        "total_episodes": total_episodes,
        "mode_counts": mode_counts,
    }


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------

class ShardWriter:
    """Accumulates samples in memory and flushes to .npz shards once full."""

    def __init__(self, output_dir: Path, samples_per_shard: int, start_shard_index: int = 0):
        self.output_dir = output_dir
        self.samples_per_shard = samples_per_shard
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffers: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.buffer_size = 0
        self.shard_index = start_shard_index

    def add_samples(self, samples: List[Dict[str, np.ndarray]]) -> int:
        if not samples:
            return 0
        for sample in samples:
            for key, value in sample.items():
                self.buffers[key].append(value)
        self.buffer_size += len(samples)

        written = 0
        while self.buffer_size >= self.samples_per_shard:
            self._flush_one(self.samples_per_shard)
            written += self.samples_per_shard
        return written

    def close(self) -> int:
        if self.buffer_size == 0:
            return 0
        remaining = self.buffer_size
        self._flush_one(remaining)
        return remaining

    def _flush_one(self, count: int) -> None:
        stacked = {
            key: np.stack(values[:count], axis=0)
            for key, values in self.buffers.items()
        }
        for values in self.buffers.values():
            del values[:count]
        self.buffer_size -= count
        shard_path = self.output_dir / f"shard_{self.shard_index:06d}.npz"
        np.savez_compressed(shard_path, **stacked)
        self.shard_index += 1


# ---------------------------------------------------------------------------
# Episode rollout
# ---------------------------------------------------------------------------

def rollout_episode(
    env: DatasetCollectEnv,
    config: ExpertCollectorConfig,
    episode_spawn_seed: int,
    episode_index: int,
) -> tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]] | None, List[np.ndarray]]:
    """Drive one episode with the selected expert; return raw frame list."""
    if hasattr(env, "engine") and getattr(env.engine, "spawn_manager", None) is not None:
        env.engine.spawn_manager.set_episode_spawn_seed(episode_spawn_seed)
    obs_dict, _ = env.reset()
    map_geometry = resolve_map_visualization_geometry(
        env=env,
        visualization_enabled=bool(config.trajectory_visualization_enabled),
        cached_geometry=None,
    )
    agent_id = list(obs_dict.keys())[0]  # single-agent env → always one key

    frames: List[Dict[str, np.ndarray]] = []
    video_frames: List[np.ndarray] = []
    done = False
    step = 0
    idm_policy = None

    while not done and step <= config.max_episode_steps:
        vehicle = env.agents.get(agent_id)
        if vehicle is None:
            break  # agent was removed (crash / out-of-road)

        obs = obs_dict[agent_id]
        frames.append(
            build_frame(
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
        )
        if bool(config.save_videos):
            video_frames.append(
                capture_episode_topdown_frame(
                    env=env,
                    episode_index=episode_index,
                    step_count=step,
                )
            )

        if config.expert_type == "ppo":
            action = ppo_expert(vehicle, deterministic=True)
        elif config.expert_type == "idm":
            if idm_policy is None:
                idm_policy = build_expert_policy(
                    vehicle,
                    random_seed=config.start_seed,
                )
            action = idm_policy.act()
        else:
            raise ValueError(f"Unsupported expert_type: {config.expert_type}")
        obs_dict, _, terminated, truncated, _ = env.step({agent_id: action})

        # Episode ends when all agents are done or the controlled agent is gone
        done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
        if agent_id not in obs_dict:
            done = True
        step += 1

    return frames, map_geometry, video_frames


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(
    config: ExpertCollectorConfig,
    dataset_root: Path,
    report_dir: Path,
    total_samples: int,
    total_episodes: int,
    split_summary: Dict,
    correction_summary: Dict,
    collection_wall_time_sec: float,
    resume: bool = False,
) -> None:
    manifest_path = report_dir / "manifest.json"
    existing_manifest = {}
    if resume and manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous_samples = int(existing_manifest.get("collected_samples", 0))
    manifest = {
        "dataset_name": config.dataset_name,
        "target_samples": config.target_samples,
        "collected_samples": total_samples,
        "episodes": total_episodes,
        "output_root": str(dataset_root),
        "expert_type": config.expert_type,
        "map": describe_map_config(config),
        "traffic_density_range": [config.traffic_density_min, config.traffic_density_max],
        "collection_wall_time_sec": float(collection_wall_time_sec),
        "splits": {name: len(shards) for name, shards in split_summary.items()},
        "trajectory_correction": correction_summary,
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in asdict(config).items()
        },
        "resume_history": existing_manifest.get("resume_history", []) + ([
            {
                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                "added_samples": int(total_samples - previous_samples),
            }
        ] if resume else []),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def run_collection(config: ExpertCollectorConfig) -> None:

    # 1. 初始化输出目录结构
    dataset_root = config.output_root / config.dataset_name
    shard_dir = dataset_root / "shards"
    report_dir = dataset_root / "reports"
    visualization_dir = report_dir / "trajectory_visualizations"
    video_dir = report_dir / "videos"
    shard_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    if bool(config.trajectory_visualization_enabled):
        visualization_dir.mkdir(parents=True, exist_ok=True)
    if bool(config.save_videos):
        video_dir.mkdir(parents=True, exist_ok=True)

    if config.resume:
        existing = detect_existing_state(shard_dir, report_dir)
        start_shard_index = int(existing["shard_count"])
        total_samples = int(existing["total_samples"])
        total_episodes = int(existing["total_episodes"])
        mode_counts: Dict[str, int] = {mode.name.lower(): 0 for mode in TrajectoryMode}
        for mode_name, count in dict(existing["mode_counts"]).items():
            mode_counts[str(mode_name)] = int(count)
        print(
            f"[resume] detected {total_samples} samples, {total_episodes} episodes, {start_shard_index} shards"
        )
    else:
        start_shard_index = 0
        total_samples = 0
        total_episodes = 0
        mode_counts = {mode.name.lower(): 0 for mode in TrajectoryMode}

    traffic_rng, spawn_rng = build_episode_rngs(config)
    fast_forward_episode_rngs(traffic_rng, spawn_rng, config, total_episodes)
    writer = ShardWriter(shard_dir, config.samples_per_shard, start_shard_index=start_shard_index)

    # 2. 创建环境实例（单环境单线程采集）
    env_config = {
        "use_render": False,
        "num_scenarios": int(config.num_scenarios),
        "image_on_cuda": True,
        "use_hybrid_map": bool(config.use_hybrid_map),
        "hybrid_map_sequence": config.hybrid_map_sequence,
        "map": int(config.map_block_num),
        "top_down_camera_initial_z": float(config.topdown_camera_height),
    }
    env = DatasetCollectEnv(env_config)
    map_geometry = None

    # 3. 保存环境参数配置
    config_path = dataset_root / "env_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(env.config), f, indent=2, ensure_ascii=False)

    total_env_steps = 0
    wall_start = time.perf_counter()
    correction_accumulator = {
        "mean_abs_lateral_before": 0.0,
        "mean_abs_lateral_after": 0.0,
        "final_abs_lateral_before": 0.0,
        "final_abs_lateral_after": 0.0,
        "strong_correction_fraction": 0.0,
    }

    try:
        while total_samples < config.target_samples:
            # 4. 采集新一轮的交通密度并注入环境配置
            traffic_density = sample_traffic_density(traffic_rng, config)
            episode_spawn_seed = sample_episode_spawn_seed(spawn_rng)
            env.config["traffic_density"] = traffic_density

            # 5. 驾驶一轮新 episode，得到原始帧列表和地图几何信息
            episode_index = total_episodes + 1
            frames, episode_map_geometry, video_frames = rollout_episode(
                env,
                config,
                episode_spawn_seed,
                episode_index=episode_index,
            )
            map_geometry = resolve_map_visualization_geometry(
                env=env,
                visualization_enabled=bool(config.trajectory_visualization_enabled),
                cached_geometry=(map_geometry if map_geometry is not None else episode_map_geometry),
            )

            # 6. 从帧列表中构建训练样本（滑动窗口），并写入分片文件
            samples = build_episode_samples(
                frames,
                config,
                traffic_density,
                visualization_dir=visualization_dir if bool(config.trajectory_visualization_enabled) else None,
                episode_index=episode_index,
                map_geometry=map_geometry,
            )
            if bool(config.save_videos):
                write_episode_video(
                    build_episode_video_path(video_dir, episode_index),
                    video_frames,
                    fps=config.video_fps,
                )

            # 7. 更新统计信息并打印进度
            writer.add_samples(samples)
            total_samples += len(samples)
            total_episodes += 1
            total_env_steps += len(frames)
            for sample in samples:
                mode_name = trajectory_mode_name(int(sample["trajectory_mode"]))
                mode_counts[mode_name] += 1
                correction_accumulator["mean_abs_lateral_before"] += float(sample["trajectory_mean_abs_lateral_before"])
                correction_accumulator["mean_abs_lateral_after"] += float(sample["trajectory_mean_abs_lateral_after"])
                correction_accumulator["final_abs_lateral_before"] += float(sample["trajectory_final_abs_lateral_before"])
                correction_accumulator["final_abs_lateral_after"] += float(sample["trajectory_final_abs_lateral_after"])
                correction_accumulator["strong_correction_fraction"] += float(sample["trajectory_correction_strength"])

            elapsed = max(time.perf_counter() - wall_start, 1e-6)
            step_per_sec = total_env_steps / elapsed
            sample_per_sec = total_samples / elapsed
            remaining_samples = max(config.target_samples - total_samples, 0)
            eta_seconds = remaining_samples / sample_per_sec if sample_per_sec > 0 else float("inf")

            print(
                f"[ep={total_episodes}] total_samples={total_samples}/{config.target_samples} "
                f"density={traffic_density:.3f} frames={len(frames)} ep_samples={len(samples)} "
                f"step/s={step_per_sec:.2f} sample/s={sample_per_sec:.2f} ETA={format_eta(eta_seconds)}"
            )
    finally:
        env.close()

    writer.close()

    # 8. 最后写入 manifest 文件，包含数据集统计信息和配置参数
    collection_wall_time_sec = time.perf_counter() - wall_start

    split_summary = split_shards(
        dataset_root=dataset_root,
        train_ratio=config.train_split_ratio,
        val_ratio=config.val_split_ratio,
        test_ratio=config.test_split_ratio,
        seed=config.split_seed,
    )
    denom = max(total_samples, 1)
    trajectory_mode_summary = build_trajectory_mode_summary(mode_counts, total_samples)
    correction_summary = {
        "enabled": bool(config.trajectory_correction_enabled),
        "mode_counts": mode_counts,
        "mode_summary": trajectory_mode_summary,
        "mean_abs_lateral_before": correction_accumulator["mean_abs_lateral_before"] / denom,
        "mean_abs_lateral_after": correction_accumulator["mean_abs_lateral_after"] / denom,
        "final_abs_lateral_before": correction_accumulator["final_abs_lateral_before"] / denom,
        "final_abs_lateral_after": correction_accumulator["final_abs_lateral_after"] / denom,
        "strong_correction_fraction": correction_accumulator["strong_correction_fraction"] / denom,
    }
    write_manifest(
        config,
        dataset_root,
        report_dir,
        total_samples,
        total_episodes,
        split_summary,
        correction_summary,
        collection_wall_time_sec=collection_wall_time_sec,
        resume=bool(config.resume),
    )

    print("Splits: " + ", ".join(f"{k}={len(v)} shards" for k, v in split_summary.items()))
    mode_stats_text = ", ".join(
        f"{mode}={stats['count']} ({stats['ratio']:.1%})"
        for mode, stats in trajectory_mode_summary.items()
    )
    print(f"Trajectory modes: {mode_stats_text}")
    print(f"Done. samples={total_samples}  output={dataset_root}")

    # TODO: 添加提取anchors过程


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> ExpertCollectorConfig:
    config = ExpertCollectorConfig()
    parser = argparse.ArgumentParser(
        description="Collect fixed-map expert trajectories from DatasetCollectEnv."
    )
    for f in fields(ExpertCollectorConfig):
        default = getattr(config, f.name)
        opts: dict = {"default": default, "dest": f.name}
        if isinstance(default, bool):
            opts["type"] = _coerce_bool
        elif isinstance(default, Path):
            opts["type"] = Path
        else:
            opts["type"] = type(default)
        if f.name == "expert_type":
            opts["choices"] = ("idm", "ppo")
        parser.add_argument(f"--{f.name.replace('_', '-')}", **opts)
    parser.parse_args(namespace=config)
    return config


if __name__ == "__main__":
    run_collection(parse_args())
