from enum import IntEnum
import math
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import zipfile

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from metadrive.policy.diffusion_policy.transfuser_config import TransfuserConfig

PROCESSED_DIR_FIELDS = (
    "camera_feature",
    "lidar_feature",
    "status_feature",
    "ego_state",
    "target_point",
    "target_line",
    "preference_point",
    "topology_polyline",
    "coarse_trajectories",
    "mode_valid_mask",
    "trajectory",
    "agent_states",
    "agent_labels",
    "bev_semantic_map",
)

METADATA_PASSTHROUGH_FIELDS = (
    "scenario_id",
    "local_route",
    "gt_mode_label",
    "expert_lateral_decision",
    "reference_lane_index",
    "trajectory_mode",
)

# ego_state layout (19D): vehicle_state (9D) + navigation_info (10D)
# [0] lateral_to_left, [1] lateral_to_right, [2] heading_diff, [3] speed,
# [4] steering, [5] last_steering, [6] last_throttle, [7] yaw_rate,
# [8] lateral_lane_offset, [9-13] navi_checkpoint_1, [14-18] navi_checkpoint_2


# 枚举类
class LaneDecision(IntEnum):
    """High-level lane-change decision used to select the target-point lane."""
    KEEP         = 0
    CHANGE_LEFT  = 1
    CHANGE_RIGHT = 2


class BoundingBox2DIndex(IntEnum):
    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4

    @classmethod
    def size(cls) -> int:
        return 5

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def POINT(cls):
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        return slice(cls._X, cls._HEADING + 1)


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)


def _to_hwc_image(image: np.ndarray) -> np.ndarray:
    image = _to_numpy(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {tuple(image.shape)}")
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(
            "Expected raw camera image in HWC layout with channel-last shape (H, W, C), "
            f"got shape {tuple(image.shape)}. If you are using an older raw dataset saved in CHW layout, "
            "convert it to HWC first."
        )
    return image


def stitch_three_cameras(
    left_camera: np.ndarray,
    front_camera: np.ndarray,
    right_camera: np.ndarray,
    config: TransfuserConfig,
) -> torch.Tensor:
    left = _to_hwc_image(left_camera)
    front = _to_hwc_image(front_camera)
    right = _to_hwc_image(right_camera)
    stitched = np.concatenate([left, front, right], axis=1)
    resized = cv2.resize(stitched, (config.camera_width, config.camera_height), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float()
    if resized.dtype == np.uint8:
        tensor = tensor / 255.0
    return tensor


def build_status_feature(ego_state: np.ndarray, config: TransfuserConfig) -> torch.Tensor:
    ego_state = _to_numpy(ego_state).astype(np.float32, copy=False)
    status = np.zeros(config.status_feature_dim, dtype=np.float32)
    take = min(config.status_feature_dim, ego_state.shape[0])
    status[:take] = ego_state[:take]
    return torch.from_numpy(status)


def _world_pose_to_local_xy(current_pose: np.ndarray, future_pose: np.ndarray) -> np.ndarray:
    current_pose = _to_numpy(current_pose).astype(np.float32, copy=False)
    future_pose = _to_numpy(future_pose).astype(np.float32, copy=False)
    dx = float(future_pose[0] - current_pose[0])
    dy = float(future_pose[1] - current_pose[1])
    heading = float(current_pose[2])
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    return np.asarray(
        [
            cos_h * dx + sin_h * dy,
            -sin_h * dx + cos_h * dy,
        ],
        dtype=np.float32,
    )


def _vehicle_pose_to_array(vehicle) -> np.ndarray:
    return np.asarray(
        [float(vehicle.position[0]), float(vehicle.position[1]), float(vehicle.heading_theta)],
        dtype=np.float32,
    )


def _zero_target_point() -> torch.Tensor:
    return torch.zeros((2,), dtype=torch.float32)


def _target_line_num_points(config: Optional[TransfuserConfig] = None) -> int:
    if config is None:
        return int(TransfuserConfig().target_line_num_points)
    return max(1, int(getattr(config, "target_line_num_points", TransfuserConfig().target_line_num_points)))


def _linear_target_line(target_point: np.ndarray, config: Optional[TransfuserConfig] = None) -> np.ndarray:
    target_point = _to_numpy(target_point).astype(np.float32, copy=False).reshape(-1)
    if target_point.shape[0] < 2:
        target_point = np.zeros((2,), dtype=np.float32)
    else:
        target_point = target_point[:2]
    num_points = _target_line_num_points(config)
    if num_points == 1:
        return target_point.reshape(1, 2).astype(np.float32, copy=False)
    ratios = np.linspace(0.0, 1.0, num_points, dtype=np.float32)[:, None]
    return (ratios * target_point[None, :]).astype(np.float32, copy=False)


def _zero_target_line(config: Optional[TransfuserConfig] = None) -> torch.Tensor:
    return torch.zeros((_target_line_num_points(config), 2), dtype=torch.float32)


def _gt_trajectory_endpoint_from_sample(sample: Dict[str, np.ndarray]) -> torch.Tensor:
    trajectory = _to_numpy(sample.get("trajectory", np.zeros((0, 3), dtype=np.float32))).astype(np.float32, copy=False)
    if trajectory.ndim != 2 or trajectory.shape[0] == 0 or trajectory.shape[1] < 2:
        return _zero_target_point()
    return torch.from_numpy(trajectory[-1, :2].astype(np.float32, copy=False))


def _derive_lane_decision_from_mode_idx(mode_idx: int) -> "LaneDecision":
    """Map a selected mode slot index to a LaneDecision."""
    from metadrive.policy.diffusion_policy.mode_definitions import get_mode_slot

    try:
        slot = get_mode_slot(int(mode_idx))
    except Exception:
        return LaneDecision.KEEP
    if slot.lateral_direction == "left":
        return LaneDecision.CHANGE_LEFT
    if slot.lateral_direction == "right":
        return LaneDecision.CHANGE_RIGHT
    return LaneDecision.KEEP


def _derive_lane_decision_from_reference_transitions(
    prev_reference_lane_index: int,
    current_reference_lane_index: int,
) -> "LaneDecision":
    """Determine LaneDecision from consecutive reference_lane_index values.

    MetaDrive lane ordinals: index 0 = leftmost, higher index = further right.
    (IDMPolicy confirms: left change = index-1 = decrease, right change = index+1 = increase)
    Both indices must be valid (>= 0) to detect a change; -1 means unknown.
    """
    if prev_reference_lane_index < 0 or current_reference_lane_index < 0:
        return LaneDecision.KEEP
    if current_reference_lane_index < prev_reference_lane_index:
        return LaneDecision.CHANGE_LEFT
    if current_reference_lane_index > prev_reference_lane_index:
        return LaneDecision.CHANGE_RIGHT
    return LaneDecision.KEEP


def _target_progress_distance(config: TransfuserConfig, speed_mps: float) -> float:
    horizon_s = float(config.target_point_prediction_horizon_s)
    return float(max(speed_mps * horizon_s, config.target_point_min_forward_distance_m))


def _build_mode_context_for_target_point(vehicle):
    from metadrive.policy.diffusion_policy.mode_context import build_mode_context_from_vehicle

    current_map = getattr(getattr(vehicle, "engine", None), "current_map", None)
    return build_mode_context_from_vehicle(vehicle, current_map=current_map)


def _build_live_target_lane_polyline(
    vehicle,
    lane_decision: "LaneDecision",
) -> Optional[np.ndarray]:
    """Prefer the live lane geometry over ModeContext's cached/sampled polyline.

    ModeContext is still useful as a fallback, but closed-loop target-point
    guidance should follow the actual lane objects whenever available.  This
    keeps the target point on the real lane centerline even if a cached
    polyline becomes stale around lane transitions or curved connectors.
    """
    navigation = getattr(vehicle, "navigation", None)
    current_ref_lanes = getattr(navigation, "current_ref_lanes", None) if navigation is not None else None
    next_ref_lanes = getattr(navigation, "next_ref_lanes", None) if navigation is not None else None
    current_lane = getattr(vehicle, "lane", None)
    if current_lane is None and current_ref_lanes:
        current_lane = current_ref_lanes[0]
    if current_lane is None:
        return None

    target_lane = current_lane
    current_map = getattr(getattr(vehicle, "engine", None), "current_map", None)
    road_network = getattr(current_map, "road_network", None) if current_map is not None else None
    lane_index = getattr(current_lane, "index", None) or getattr(vehicle, "lane_index", None)
    if current_map is None or road_network is None or lane_index is None:
        return None
    if lane_decision != LaneDecision.KEEP and lane_index is not None and road_network is not None:
        try:
            road_lanes = road_network.graph[lane_index[0]][lane_index[1]]
            lane_id = int(lane_index[2])
            if lane_decision == LaneDecision.CHANGE_LEFT and lane_id - 1 >= 0:
                target_lane = road_lanes[lane_id - 1]
            elif lane_decision == LaneDecision.CHANGE_RIGHT and lane_id + 1 < len(road_lanes):
                target_lane = road_lanes[lane_id + 1]
            else:
                return None
        except Exception:
            return None

    try:
        from metadrive.policy.diffusion_policy.mode_context import (
            _get_route_roads,
            _resolve_candidate_next_lanes,
            _sample_route_ahead_polyline_from_vehicle_position,
        )
    except Exception:
        return None

    # --- Lane heading sanity check -------------------------------------------
    # If target_lane runs in the OPPOSITE direction to ego (e.g. ego has been
    # pushed onto the oncoming lane after a collision), the polyline will point
    # backward in ego-local frame.  Detect this before sampling.
    ego_heading = float(getattr(vehicle, "heading_theta", 0.0))
    try:
        from metadrive.policy.diffusion_policy.mode_context import _lane_start_longitudinal
        _start_s = _lane_start_longitudinal(target_lane, np.asarray(vehicle.position, dtype=np.float32))
        if hasattr(target_lane, "heading_theta_at"):
            lane_heading_at_ego = float(target_lane.heading_theta_at(_start_s))
            heading_diff = math.atan2(
                math.sin(lane_heading_at_ego - ego_heading),
                math.cos(lane_heading_at_ego - ego_heading),
            )
            if abs(heading_diff) > math.pi / 2.0:
                return None  # lane is pointing the wrong direction
    except Exception:
        pass

    route_roads = _get_route_roads(current_map, ())
    try:
        # Use topology-based successor resolution instead of blindly reusing
        # navigation.next_ref_lanes.  Around transitions/curves, the latter can
        # correspond to a different nearby lane and pull the target point away
        # from the actual lane centerline.
        candidate_next_lanes = _resolve_candidate_next_lanes(
            target_lane,
            current_map,
            next_ref_lanes if lane_decision == LaneDecision.KEEP else None,
            route_roads,
        )
        world_polyline = _sample_route_ahead_polyline_from_vehicle_position(
            target_lane,
            np.asarray(vehicle.position, dtype=np.float32),
            candidate_next_lanes,
            current_map=current_map,
        )
    except Exception:
        return None

    if world_polyline is None or np.asarray(world_polyline).size == 0:
        return None
    current_pose = _vehicle_pose_to_array(vehicle)
    local_polyline = np.asarray(
        [
            _world_pose_to_local_xy(
                current_pose,
                np.asarray([float(point[0]), float(point[1]), 0.0], dtype=np.float32),
            )
            for point in np.asarray(world_polyline, dtype=np.float32)
        ],
        dtype=np.float32,
    )

    # --- Polyline forward check -----------------------------------------------
    # Check that points further along the polyline (skipping the first, which
    # can sit right at ego) actually lie ahead of ego (+x in local frame).
    # If the sampled lane is going backwards (all local-x < 0), it means the
    # lane geometry or heading check above was insufficient — reject this polyline
    # so compute_target_point falls back to the ModeContext polyline.
    if local_polyline.shape[0] >= 2:
        check_slice = local_polyline[1:min(6, local_polyline.shape[0])]
        if float(check_slice[:, 0].mean()) < 0.0:
            return None

    return local_polyline


def _simulate_reachable_distance(
    speed_mps: float,
    horizon_s: float,
    max_accel_mps2: float,
    max_jerk_mps3: float,
    dt: float = 0.1,
) -> float:
    horizon_s = max(float(horizon_s), 0.0)
    speed = max(float(speed_mps), 0.0)
    accel = 0.0
    distance = 0.0
    elapsed = 0.0
    max_accel = max(float(max_accel_mps2), 0.0)
    max_jerk = max(float(max_jerk_mps3), 0.0)
    if horizon_s <= 0.0:
        return 0.0
    while elapsed < horizon_s - 1e-8:
        step = min(float(dt), horizon_s - elapsed)
        if max_jerk > 0.0:
            accel = min(max_accel, accel + max_jerk * step)
        else:
            accel = max_accel
        distance += speed * step + 0.5 * accel * step * step
        speed = max(0.0, speed + accel * step)
        elapsed += step
    return float(distance)


def _get_target_lane_polyline(ctx: Any, lane_decision: "LaneDecision") -> Optional[np.ndarray]:
    if ctx is None:
        return None
    if lane_decision == LaneDecision.CHANGE_LEFT and getattr(ctx, "has_left_adjacent", False):
        poly = getattr(ctx, "left_lane_polyline", None)
        if poly is not None:
            return np.asarray(poly, dtype=np.float32)
    if lane_decision == LaneDecision.CHANGE_RIGHT and getattr(ctx, "has_right_adjacent", False):
        poly = getattr(ctx, "right_lane_polyline", None)
        if poly is not None:
            return np.asarray(poly, dtype=np.float32)
    poly = getattr(ctx, "current_lane_polyline", None)
    if poly is None:
        return None
    return np.asarray(poly, dtype=np.float32)


def _predict_world_position(obj: Any, horizon_s: float, use_constant_accel: bool) -> np.ndarray:
    position = np.asarray(getattr(obj, "position", np.zeros((2,), dtype=np.float32)), dtype=np.float32)
    speed_mps = float(getattr(obj, "speed_km_h", 0.0)) / 3.6
    accel_mps2 = min(float(getattr(obj, "acceleration", 0.0)), 0.0) if use_constant_accel else 0.0
    heading = float(getattr(obj, "heading_theta", 0.0))
    direction = np.asarray([math.cos(heading), math.sin(heading)], dtype=np.float32)
    longitudinal = speed_mps * horizon_s + 0.5 * accel_mps2 * horizon_s * horizon_s
    return position + direction * float(longitudinal)


def _expert_idm_target_progress_distance(
    vehicle,
    front_obj: Any,
    front_distance_m: float,
    config: TransfuserConfig,
) -> float:
    """Estimate a forward target distance from the expert IDM longitudinal rule.

    This is a closed-loop/live-only guidance signal.  It uses the same front
    object and relative-motion ingredients as the expert IDM longitudinal
    controller, then converts the resulting acceleration tendency into a
    4-second forward progress estimate.
    """
    from metadrive.policy.idm_policy import IDMPolicy

    ego_speed_mps = max(float(getattr(vehicle, "speed_km_h", 0.0)) / 3.6, 0.0)
    target_speed_mps = float(IDMPolicy.NORMAL_SPEED) / 3.6
    if target_speed_mps <= 1e-3:
        return 0.0

    accel = float(IDMPolicy.ACC_FACTOR) * (
        1.0 - np.power(max(ego_speed_mps, 0.0) / target_speed_mps, float(IDMPolicy.DELTA))
    )
    if front_obj is not None and np.isfinite(float(front_distance_m)) and front_distance_m > 1e-3:
        front_speed_mps = max(float(getattr(front_obj, "speed_km_h", 0.0)) / 3.6, 0.0)
        desired_gap = float(IDMPolicy.DISTANCE_WANTED)
        desired_gap += ego_speed_mps * float(IDMPolicy.TIME_WANTED)
        ab = max(float(IDMPolicy.ACC_FACTOR) * float(-IDMPolicy.DEACC_FACTOR), 1e-3)
        closing_speed = ego_speed_mps - front_speed_mps
        desired_gap += ego_speed_mps * closing_speed / (2.0 * math.sqrt(ab))
        desired_gap = max(float(IDMPolicy.DISTANCE_WANTED), desired_gap)
        accel -= float(IDMPolicy.ACC_FACTOR) * np.square(desired_gap / max(float(front_distance_m), 1e-3))

    horizon_s = float(config.target_point_prediction_horizon_s)
    progress = ego_speed_mps * horizon_s + 0.5 * accel * horizon_s * horizon_s
    return max(0.0, float(progress))


def _get_object_extent_radius(obj: Any, buffer_m: float) -> float:
    length = float(getattr(obj, "LENGTH", getattr(obj, "length", 4.5)))
    width = float(getattr(obj, "WIDTH", getattr(obj, "width", 2.0)))
    return 0.5 * float(max(length, width)) + float(buffer_m)


def _polyline_arc_distance_to_point(local_polyline: Optional[np.ndarray], local_point: np.ndarray) -> float:
    if local_polyline is None:
        return float("inf")
    polyline = np.asarray(local_polyline, dtype=np.float32)
    if polyline.ndim != 2 or polyline.shape[0] == 0:
        return float("inf")
    point = np.asarray(local_point, dtype=np.float32).reshape(1, 2)
    cumulative = np.concatenate(
        [np.zeros((1,), dtype=np.float32), np.cumsum(np.linalg.norm(np.diff(polyline, axis=0), axis=1), dtype=np.float32)],
        axis=0,
    )
    closest_idx = int(np.argmin(np.linalg.norm(polyline - point, axis=1)))
    return float(cumulative[closest_idx])


def _find_front_back_objects_for_target_lane(
    vehicle,
    lane_decision: "LaneDecision",
    mode_context=None,
):
    try:
        from metadrive.policy.idm_policy import FrontBackObjects
    except Exception:
        return None

    lidar = getattr(vehicle, "lidar", None)
    current_lane = getattr(vehicle, "lane", None)
    navigation = getattr(vehicle, "navigation", None)
    ref_lanes = getattr(navigation, "current_ref_lanes", None) if navigation is not None else None
    if lidar is None or current_lane is None:
        return None
    try:
        objects = lidar.get_surrounding_objects(vehicle)
    except Exception:
        return None
    if not ref_lanes or current_lane not in ref_lanes:
        return FrontBackObjects.get_find_front_back_objs_single_lane(objects, current_lane, vehicle.position, max_distance=80)

    try:
        current_idx = ref_lanes.index(current_lane)
    except ValueError:
        current_idx = int(getattr(current_lane, "index", (None, None, 0))[-1])
    target_idx = current_idx
    if lane_decision == LaneDecision.CHANGE_LEFT:
        target_idx = max(0, current_idx - 1)
    elif lane_decision == LaneDecision.CHANGE_RIGHT:
        target_idx = min(len(ref_lanes) - 1, current_idx + 1)
    target_lane = ref_lanes[target_idx]
    return FrontBackObjects.get_find_front_back_objs(objects, target_lane, vehicle.position, max_distance=80, ref_lanes=ref_lanes)


def _front_safe_distance_limit(
    vehicle,
    front_obj: Any,
    config: TransfuserConfig,
    target_polyline: Optional[np.ndarray] = None,
) -> float:
    if front_obj is None:
        return float("inf")
    front_pos = np.asarray(
        getattr(front_obj, "position", np.zeros((2,), dtype=np.float32)),
        dtype=np.float32,
    )
    current_pose = _vehicle_pose_to_array(vehicle)
    front_local = _world_pose_to_local_xy(
        current_pose,
        np.asarray([front_pos[0], front_pos[1], 0.0], dtype=np.float32),
    )
    front_radius = _get_object_extent_radius(front_obj, buffer_m=0.0)
    front_distance = _polyline_arc_distance_to_point(target_polyline, front_local)
    if not np.isfinite(front_distance):
        front_distance = float(front_local[0])
    front_speed_mps = float(getattr(front_obj, "speed_km_h", 0.0)) / 3.6
    front_accel_mps2 = 0.0
    if bool(getattr(config, "target_point_vehicle_prediction_use_constant_accel", True)):
        front_accel_mps2 = min(float(getattr(front_obj, "acceleration", 0.0)), 0.0)
    horizon_s = float(config.target_point_prediction_horizon_s)
    future_progress = max(0.0, front_speed_mps * horizon_s + 0.5 * front_accel_mps2 * horizon_s * horizon_s)
    safe_front_distance = float(front_distance) + float(future_progress)
    return max(0.0, safe_front_distance - float(config.target_point_front_safe_gap_m) - front_radius)


def _non_front_overlap_distance_limit(
    vehicle,
    target_polyline: Optional[np.ndarray],
    config: TransfuserConfig,
    front_obj: Any = None,
) -> float:
    if target_polyline is None or target_polyline.ndim != 2 or target_polyline.shape[0] == 0:
        return float("inf")
    lidar = getattr(vehicle, "lidar", None)
    if lidar is None:
        return float("inf")
    try:
        surrounding_objects = lidar.get_surrounding_objects(vehicle)
    except Exception:
        return float("inf")
    if not surrounding_objects:
        return float("inf")

    ego_radius = _get_object_extent_radius(vehicle, buffer_m=float(config.target_point_non_front_overlap_buffer_m))
    current_pose = _vehicle_pose_to_array(vehicle)
    navigation = getattr(vehicle, "navigation", None)
    try:
        lane_half_width = 0.5 * float(navigation.get_current_lane_width())
    except Exception:
        current_lane = getattr(vehicle, "lane", None)
        lane_half_width = 0.5 * float(current_lane.width_at(0.0)) if current_lane is not None and hasattr(current_lane, "width_at") else 1.75
    lane_corridor_threshold = max(1.25, lane_half_width * 0.9)
    min_limit = float("inf")
    for obj in surrounding_objects:
        if obj is None or obj is front_obj or obj is vehicle:
            continue
        # Use current object position rather than straight-line predicted future
        # position.  In curved road sections, straight-line prediction places
        # objects off the curve, causing false polyline overlap detections that
        # incorrectly shrink overlap_safe to near-zero.
        obj_pos = np.asarray(
            getattr(obj, "position", np.zeros((2,), dtype=np.float32)),
            dtype=np.float32,
        )
        predicted_local = _world_pose_to_local_xy(
            current_pose,
            np.asarray([obj_pos[0], obj_pos[1], 0.0], dtype=np.float32),
        )
        # Skip objects that are at or behind ego in the local frame.  A vehicle
        # that is currently colliding with ego (predicted_local[0] ≈ 0 or < 0)
        # appears at the polyline origin, which would incorrectly force
        # overlap_safe → 0 and collapse the target_point onto ego.
        if float(predicted_local[0]) <= 0.0:
            continue
        extent_radius = ego_radius + _get_object_extent_radius(obj, buffer_m=0.0)
        distances = np.linalg.norm(target_polyline - predicted_local.reshape(1, 2), axis=1)
        if float(np.min(distances)) > lane_corridor_threshold:
            continue
        inside = np.nonzero(distances <= extent_radius)[0]
        if inside.size == 0:
            continue
        first_idx = max(int(inside[0]) - 1, 0)
        cumulative = np.concatenate(
            [np.zeros((1,), dtype=np.float32), np.cumsum(np.linalg.norm(np.diff(target_polyline, axis=0), axis=1), dtype=np.float32)],
            axis=0,
        )
        min_limit = min(min_limit, float(cumulative[first_idx]))
    return float(min_limit)


def _bounded_target_progress_distance(
    vehicle,
    config: TransfuserConfig,
    target_polyline: Optional[np.ndarray],
    lane_decision: "LaneDecision",
    mode_context=None,
) -> float:
    speed_mps = float(getattr(vehicle, "speed_km_h", 0.0)) / 3.6
    reachable = _simulate_reachable_distance(
        speed_mps=speed_mps,
        horizon_s=float(config.target_point_prediction_horizon_s),
        max_accel_mps2=float(config.target_point_max_reachable_accel_mps2),
        max_jerk_mps3=float(config.target_point_max_reachable_jerk_mps3),
    )
    front_back = _find_front_back_objects_for_target_lane(vehicle, lane_decision, mode_context=mode_context)
    front_obj = front_back.front_object() if front_back is not None and front_back.has_front_object() else None
    front_distance = (
        float(front_back.front_min_distance())
        if front_back is not None and front_back.has_front_object() and front_back.front_min_distance() is not None
        else float("inf")
    )
    longitudinal_progress = (
        _expert_idm_target_progress_distance(
            vehicle,
            front_obj,
            front_distance,
            config,
        )
    )
    front_safe = _front_safe_distance_limit(vehicle, front_obj, config, target_polyline=target_polyline)
    overlap_safe = _non_front_overlap_distance_limit(vehicle, target_polyline, config, front_obj=front_obj)
    target_distance = min(float(longitudinal_progress), float(reachable), float(front_safe), float(overlap_safe))
    min_dist = float(config.target_point_min_forward_distance_m)
    # Apply the minimum-forward-distance floor when the path ahead is not
    # physically blocked by an imminent obstacle:
    #   (a) no front vehicle at all, or
    #   (b) front vehicle is at or behind ego (collision / overlap: front_distance ≤ 0),
    #       meaning it cannot actually block forward progress from here.
    # In either case, collapsing the target_point onto ego is never useful.
    apply_floor = front_obj is None or float(front_distance) <= 0.0
    if apply_floor:
        return max(min_dist, float(target_distance))
    return max(0.0, float(target_distance))


def _interpolate_local_target(local_points_xy: np.ndarray, target_distance: float) -> np.ndarray:
    local_points_xy = np.asarray(local_points_xy, dtype=np.float32)
    if local_points_xy.size == 0:
        return np.zeros((2,), dtype=np.float32)
    if local_points_xy.shape[0] == 1:
        return local_points_xy[0].astype(np.float32, copy=False)

    segment_lengths = np.linalg.norm(np.diff(local_points_xy, axis=0), axis=1)
    cumulative = np.concatenate(
        [np.zeros((1,), dtype=np.float32), np.cumsum(segment_lengths, dtype=np.float32)],
        axis=0,
    )
    if target_distance <= 0.0:
        return local_points_xy[0].astype(np.float32, copy=False)
    if target_distance >= float(cumulative[-1]):
        return local_points_xy[-1].astype(np.float32, copy=False)

    segment_idx = int(np.searchsorted(cumulative, target_distance, side="right") - 1)
    segment_idx = max(0, min(segment_idx, local_points_xy.shape[0] - 2))
    segment_start = float(cumulative[segment_idx])
    segment_length = float(segment_lengths[segment_idx])
    if segment_length <= 1e-6:
        return local_points_xy[segment_idx + 1].astype(np.float32, copy=False)
    ratio = float((target_distance - segment_start) / segment_length)
    start = local_points_xy[segment_idx]
    end = local_points_xy[segment_idx + 1]
    return (start + ratio * (end - start)).astype(np.float32, copy=False)


def _polyline_arc_lengths(local_points_xy: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    local_points_xy = np.asarray(local_points_xy, dtype=np.float32)
    if local_points_xy.ndim != 2 or local_points_xy.shape[0] < 2 or local_points_xy.shape[1] < 2:
        return None
    segment_lengths = np.linalg.norm(np.diff(local_points_xy[:, :2], axis=0), axis=1).astype(np.float32)
    cumulative = np.concatenate(
        [np.zeros((1,), dtype=np.float32), np.cumsum(segment_lengths, dtype=np.float32)],
        axis=0,
    )
    if float(cumulative[-1]) <= 1e-6:
        return None
    return segment_lengths, cumulative


def _project_point_to_polyline_arc_length(point_xy: np.ndarray, local_points_xy: np.ndarray) -> Optional[float]:
    arc_data = _polyline_arc_lengths(local_points_xy)
    if arc_data is None:
        return None
    segment_lengths, cumulative = arc_data
    point_xy = _to_numpy(point_xy).astype(np.float32, copy=False).reshape(-1)
    if point_xy.shape[0] < 2:
        return None
    point_xy = point_xy[:2]

    best_s = 0.0
    best_dist_sq = float("inf")
    for idx, (start, end) in enumerate(zip(local_points_xy[:-1, :2], local_points_xy[1:, :2])):
        seg = end - start
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq <= 1e-12:
            continue
        ratio = float(np.dot(point_xy - start, seg) / seg_len_sq)
        ratio = max(0.0, min(1.0, ratio))
        projected = start + ratio * seg
        dist_sq = float(np.dot(point_xy - projected, point_xy - projected))
        if dist_sq < best_dist_sq:
            best_dist_sq = dist_sq
            best_s = float(cumulative[idx]) + ratio * float(segment_lengths[idx])
    if not np.isfinite(best_dist_sq):
        return None
    return best_s


def _build_target_line_from_polyline(
    target_polyline: np.ndarray,
    target_point: np.ndarray,
    config: Optional[TransfuserConfig] = None,
) -> np.ndarray:
    """Sample local guidance points on target lane from ego projection to target point projection."""
    target_polyline = _to_numpy(target_polyline).astype(np.float32, copy=False)
    target_point = _to_numpy(target_point).astype(np.float32, copy=False).reshape(-1)
    if target_point.shape[0] < 2:
        target_point = np.zeros((2,), dtype=np.float32)
    else:
        target_point = target_point[:2]

    arc_data = _polyline_arc_lengths(target_polyline)
    if arc_data is None:
        return _linear_target_line(target_point, config)

    s0 = _project_point_to_polyline_arc_length(np.zeros((2,), dtype=np.float32), target_polyline)
    s1 = _project_point_to_polyline_arc_length(target_point, target_polyline)
    if s0 is None or s1 is None:
        return _linear_target_line(target_point, config)
    s1 = max(float(s0), float(s1))

    sample_s = np.linspace(float(s0), float(s1), _target_line_num_points(config), dtype=np.float32)
    return np.stack([_interpolate_local_target(target_polyline[:, :2], float(s)) for s in sample_s], axis=0).astype(
        np.float32, copy=False
    )


def _build_target_line(
    target_polyline: np.ndarray,
    target_point: torch.Tensor,
    config: Optional[TransfuserConfig] = None,
) -> torch.Tensor:
    return torch.from_numpy(_build_target_line_from_polyline(target_polyline, _to_numpy(target_point), config))


def _densify_polyline(local_points_xy: np.ndarray, max_segment_length: float = 0.25) -> np.ndarray:
    local_points_xy = np.asarray(local_points_xy, dtype=np.float32)
    if local_points_xy.ndim != 2 or local_points_xy.shape[0] <= 1:
        return local_points_xy
    dense_points = [local_points_xy[0]]
    max_segment_length = max(float(max_segment_length), 1e-3)
    for start, end in zip(local_points_xy[:-1], local_points_xy[1:]):
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length <= max_segment_length:
            dense_points.append(end)
            continue
        num_subsegments = int(np.ceil(length / max_segment_length))
        for step_idx in range(1, num_subsegments + 1):
            ratio = float(step_idx) / float(num_subsegments)
            dense_points.append((start + ratio * segment).astype(np.float32, copy=False))
    return np.asarray(dense_points, dtype=np.float32)


def decide_lane_change_for_vehicle(vehicle, overtake_timer: int = 0) -> "LaneDecision":
    """Replicates IDMPolicy.lane_change_policy() to decide lane-change direction.

    Uses IDMPolicy class constants directly so the decision logic stays in sync with
    the expert model used during dataset collection.  An optional overtake_timer mirrors
    IDM's LANE_CHANGE_FREQ gate: pass a per-vehicle counter incremented each step and
    reset to 0 after a lane change; the function only considers overtaking when
    overtake_timer >= IDMPolicy.LANE_CHANGE_FREQ.

    target_point is external conditioning and must be determined before the model runs.
    MetaDrive lane ordinals: index 0 = leftmost, higher = further right; left overtake has priority.
    """
    try:
        from metadrive.policy.idm_policy import FrontBackObjects, IDMPolicy

        current_lanes = vehicle.navigation.current_ref_lanes
        routing_lane = vehicle.lane
        all_objects = vehicle.lidar.get_surrounding_objects(vehicle)

        surrounding = FrontBackObjects.get_find_front_back_objs(
            all_objects, routing_lane, vehicle.position,
            IDMPolicy.MAX_LONG_DIST, current_lanes,
        )

        # ── routing-forced lane change (matches IDMPolicy lines ~352-386) ──────
        next_lanes = vehicle.navigation.next_ref_lanes
        lane_num_diff = len(current_lanes) - len(next_lanes) if next_lanes is not None else 0
        if lane_num_diff > 0:
            if current_lanes[0].is_previous_lane_of(next_lanes[0]):
                index_range = list(range(len(next_lanes)))
            else:
                index_range = list(range(lane_num_diff, len(current_lanes)))
            current_idx = routing_lane.index[-1] if hasattr(routing_lane, "index") else 0
            if current_idx not in index_range:
                if current_idx > index_range[-1]:
                    if (surrounding.left_back_min_distance() >= IDMPolicy.SAFE_LANE_CHANGE_DISTANCE
                            and surrounding.left_front_min_distance() >= 5):
                        return LaneDecision.CHANGE_LEFT
                else:
                    if (surrounding.right_back_min_distance() >= IDMPolicy.SAFE_LANE_CHANGE_DISTANCE
                            and surrounding.right_front_min_distance() >= 5):
                        return LaneDecision.CHANGE_RIGHT

        # ── active overtake (matches IDMPolicy lines ~389-409) ─────────────────
        # Only attempt when both ego and front vehicle are significantly below NORMAL_SPEED
        # and the overtake timer has elapsed (prevents oscillation).
        ego_speed = float(getattr(vehicle, "speed_km_h", 0.0))
        if (
            surrounding.has_front_object()
            and abs(ego_speed - IDMPolicy.NORMAL_SPEED) > 3
            and abs(surrounding.front_object().speed_km_h - IDMPolicy.NORMAL_SPEED) > 3
            and int(overtake_timer) >= IDMPolicy.LANE_CHANGE_FREQ
        ):
            front_speed = surrounding.front_object().speed_km_h
            available_range = list(range(len(current_lanes)))

            left_front_speed = None
            if (
                surrounding.left_lane_exist()
                and surrounding.left_front_min_distance() > IDMPolicy.SAFE_LANE_CHANGE_DISTANCE
                and surrounding.left_back_min_distance() > IDMPolicy.SAFE_LANE_CHANGE_DISTANCE
            ):
                left_front_speed = (
                    surrounding.left_front_object().speed_km_h
                    if surrounding.has_left_front_object()
                    else IDMPolicy.MAX_SPEED
                )
            right_front_speed = None
            if (
                surrounding.right_lane_exist()
                and surrounding.right_front_min_distance() > IDMPolicy.SAFE_LANE_CHANGE_DISTANCE
                and surrounding.right_back_min_distance() > IDMPolicy.SAFE_LANE_CHANGE_DISTANCE
            ):
                right_front_speed = (
                    surrounding.right_front_object().speed_km_h
                    if surrounding.has_right_front_object()
                    else IDMPolicy.MAX_SPEED
                )

            current_idx = routing_lane.index[-1] if hasattr(routing_lane, "index") else 0
            if (
                left_front_speed is not None
                and left_front_speed - front_speed > IDMPolicy.LANE_CHANGE_SPEED_INCREASE
                and current_idx - 1 in available_range
            ):
                return LaneDecision.CHANGE_LEFT
            if (
                right_front_speed is not None
                and right_front_speed - front_speed > IDMPolicy.LANE_CHANGE_SPEED_INCREASE
                and current_idx + 1 in available_range
            ):
                return LaneDecision.CHANGE_RIGHT
    except Exception:
        pass
    return LaneDecision.KEEP


def compute_target_point(
    vehicle,
    config: TransfuserConfig,
    lane_decision: "LaneDecision" = LaneDecision.KEEP,
) -> torch.Tensor:
    if vehicle is None:
        return _zero_target_point()

    # Use ModeContext to get polylines that chain across lane boundaries.
    # This ensures the target point continues into the next road segment when
    # ego is near the end of the current lane.
    try:
        ctx = _build_mode_context_for_target_point(vehicle)
    except Exception:
        ctx = None

    target_polyline = _build_live_target_lane_polyline(vehicle, lane_decision)
    if target_polyline is None:
        target_polyline = _get_target_lane_polyline(ctx, lane_decision)
    # Validate that the polyline actually points ahead of ego in local frame.
    # A backward-facing polyline (mean x of points [1..5] < 0) means the lane
    # assignment is wrong (e.g. ego on opposite-direction lane after a collision).
    if target_polyline is not None and target_polyline.shape[0] >= 2:
        check_slice = target_polyline[1:min(6, target_polyline.shape[0])]
        if float(check_slice[:, 0].mean()) < 0.0:
            target_polyline = None
    if target_polyline is not None and target_polyline.shape[0] > 0:
        target_polyline = _densify_polyline(target_polyline, max_segment_length=0.25)
        delta_s = _bounded_target_progress_distance(vehicle, config, target_polyline, lane_decision, mode_context=ctx)
        return torch.from_numpy(_interpolate_local_target(target_polyline, delta_s))

    # Last-resort fallback: single-lane clamped computation.
    current_lane = getattr(vehicle, "lane", None)
    navigation = getattr(vehicle, "navigation", None)
    current_ref_lanes = getattr(navigation, "current_ref_lanes", None) if navigation is not None else None
    if current_lane is None:
        if not current_ref_lanes:
            return _zero_target_point()
        current_lane = current_ref_lanes[0]
    if current_lane is None:
        return _zero_target_point()
    # If vehicle.lane is wrong-facing (e.g. ego pushed onto oncoming lane),
    # prefer the navigation's ref lane which is always on the correct route.
    try:
        _ego_heading = float(getattr(vehicle, "heading_theta", 0.0))
        if hasattr(current_lane, "local_coordinates") and hasattr(current_lane, "heading_theta_at"):
            _s_chk = float(current_lane.local_coordinates(vehicle.position)[0])
            _lane_h = float(current_lane.heading_theta_at(_s_chk))
            _hdiff = math.atan2(math.sin(_lane_h - _ego_heading), math.cos(_lane_h - _ego_heading))
            if abs(_hdiff) > math.pi / 2.0 and current_ref_lanes:
                current_lane = current_ref_lanes[0]
    except Exception:
        pass
    s_ego, _ = current_lane.local_coordinates(vehicle.position)
    speed_mps = float(getattr(vehicle, "speed_km_h", 0.0)) / 3.6
    delta_s = _simulate_reachable_distance(
        speed_mps=speed_mps,
        horizon_s=float(config.target_point_prediction_horizon_s),
        max_accel_mps2=float(config.target_point_max_reachable_accel_mps2),
        max_jerk_mps3=float(config.target_point_max_reachable_jerk_mps3),
    )
    s_target = min(float(s_ego) + delta_s, float(current_lane.length))
    target_world = np.asarray(current_lane.position(s_target, 0.0), dtype=np.float32)
    current_pose = _vehicle_pose_to_array(vehicle)
    return torch.from_numpy(_world_pose_to_local_xy(current_pose, np.asarray([target_world[0], target_world[1], 0.0], dtype=np.float32)))


def compute_target_point_from_sample(
    sample: Dict[str, np.ndarray],
    config: TransfuserConfig,
    lane_decision: "LaneDecision" = LaneDecision.KEEP,
) -> torch.Tensor:
    trajectory = _to_numpy(sample.get("trajectory", np.zeros((0, 3), dtype=np.float32))).astype(np.float32, copy=False)
    if trajectory.ndim == 2 and trajectory.shape[0] > 0 and trajectory.shape[1] >= 2:
        return _gt_trajectory_endpoint_from_sample(sample)

    speed_mps = float(_to_numpy(sample.get("ego_speed_km_h", np.asarray(0.0, dtype=np.float32))).reshape(-1)[0]) / 3.6
    delta_s = _target_progress_distance(config, speed_mps)

    if lane_decision != LaneDecision.KEEP:
        # Use the adjacent-lane polyline stored in the sample (already ego-local XY).
        if lane_decision == LaneDecision.CHANGE_LEFT:
            has_adj = bool(_to_numpy(sample.get("has_left_adjacent", np.asarray(0, dtype=np.int8))).reshape(-1)[0])
            poly_key = "left_lane_polyline"
        else:
            has_adj = bool(_to_numpy(sample.get("has_right_adjacent", np.asarray(0, dtype=np.int8))).reshape(-1)[0])
            poly_key = "right_lane_polyline"

        if has_adj and poly_key in sample:
            adj_poly = _to_numpy(sample[poly_key]).astype(np.float32, copy=False)
            if adj_poly.ndim == 2 and adj_poly.shape[0] > 0 and np.any(adj_poly):
                # adj_poly starts at ego's longitudinal position on the adjacent lane —
                # interpolate directly (no (0,0) prepend) so the target is ahead on adjacent lane.
                return torch.from_numpy(_interpolate_local_target(adj_poly, delta_s))
        # Fallthrough to current-lane if adjacent lane is unavailable

    # Prefer stored current_lane_polyline (ego-local, chains across lane boundaries).
    if "current_lane_polyline" in sample:
        cur_poly = _to_numpy(sample["current_lane_polyline"]).astype(np.float32, copy=False)
        if cur_poly.ndim == 2 and cur_poly.shape[0] > 0 and np.any(cur_poly):
            return torch.from_numpy(_interpolate_local_target(cur_poly, delta_s))

    # Fallback: use future_reference_pose_world without lane-index filtering so the
    # target can extend into the next road segment when ego is near the lane end.
    if "reference_pose_world" not in sample or "future_reference_pose_world" not in sample:
        return _zero_target_point()
    current_pose = _to_numpy(sample["reference_pose_world"]).astype(np.float32, copy=False)
    future_reference = _to_numpy(sample["future_reference_pose_world"]).astype(np.float32, copy=False)
    if future_reference.ndim != 2 or future_reference.shape[0] == 0:
        return _zero_target_point()

    local_points_xy = np.stack(
        [_world_pose_to_local_xy(current_pose, pose) for pose in future_reference],
        axis=0,
    ).astype(np.float32, copy=False)
    local_points_xy = np.concatenate([np.zeros((1, 2), dtype=np.float32), local_points_xy], axis=0)
    return torch.from_numpy(_interpolate_local_target(local_points_xy, delta_s))


def _build_topology_polyline_from_sample(
    sample: Dict[str, np.ndarray],
    lane_decision: "LaneDecision",
) -> torch.Tensor:
    current_poly = _to_numpy(sample.get("current_lane_polyline", np.zeros((0, 2), dtype=np.float32))).astype(
        np.float32, copy=False
    )
    selected_poly = current_poly
    if lane_decision == LaneDecision.CHANGE_LEFT:
        left_poly = _to_numpy(sample.get("left_lane_polyline", np.zeros((0, 2), dtype=np.float32))).astype(
            np.float32, copy=False
        )
        if left_poly.ndim == 2 and left_poly.shape[0] > 0 and np.any(left_poly):
            selected_poly = left_poly
    elif lane_decision == LaneDecision.CHANGE_RIGHT:
        right_poly = _to_numpy(sample.get("right_lane_polyline", np.zeros((0, 2), dtype=np.float32))).astype(
            np.float32, copy=False
        )
        if right_poly.ndim == 2 and right_poly.shape[0] > 0 and np.any(right_poly):
            selected_poly = right_poly

    if selected_poly.ndim == 2 and selected_poly.shape[0] > 0 and selected_poly.shape[1] == 2:
        return torch.from_numpy(selected_poly.astype(np.float32, copy=False))
    if current_poly.ndim == 2 and current_poly.shape[0] > 0 and current_poly.shape[1] == 2:
        return torch.from_numpy(current_poly.astype(np.float32, copy=False))
    return torch.from_numpy(_build_fallback_current_lane_polyline(sample).astype(np.float32, copy=False))


def _build_topology_polyline_from_live_vehicle(
    vehicle,
    lane_decision: "LaneDecision",
) -> torch.Tensor:
    live_polyline = _build_live_target_lane_polyline(vehicle, lane_decision)
    if live_polyline is not None:
        return torch.from_numpy(np.asarray(live_polyline, dtype=np.float32))

    ctx = _build_mode_context_for_target_point(vehicle)
    selected_poly = _get_target_lane_polyline(ctx, lane_decision)
    if selected_poly is None:
        selected_poly = getattr(ctx, "current_lane_polyline", None)
    if selected_poly is None:
        return torch.zeros((0, 2), dtype=torch.float32)
    return torch.from_numpy(np.asarray(selected_poly, dtype=np.float32))


# !这里与 TransFuser 原论文中的 BEV 处理方式不同!
def lidar_to_histogram(
    lidar: np.ndarray,
    config: TransfuserConfig,
) -> torch.Tensor:
    lidar = _to_numpy(lidar).astype(np.float32, copy=False)
    if lidar.ndim != 1:
        raise ValueError(f"Expected 1D lidar vector, got shape {tuple(lidar.shape)}")

    distances = np.clip(lidar, 0.0, 1.0) * config.lidar_max_distance
    num_lasers = max(len(distances), 1)
    angles = np.linspace(0.0, 2.0 * np.pi, num_lasers, endpoint=False)

    x = distances * np.cos(angles)
    y = -distances * np.sin(angles)

    valid = (
        (config.lidar_min_x <= x) & (x <= config.lidar_max_x) &
        (config.lidar_min_y <= y) & (y <= config.lidar_max_y)
    )
    x = x[valid]
    y = y[valid]

    xbins = np.linspace(config.lidar_min_x, config.lidar_max_x, config.lidar_resolution_width + 1)
    ybins = np.linspace(config.lidar_min_y, config.lidar_max_y, config.lidar_resolution_height + 1)
    hist = np.histogramdd(np.stack([x, y], axis=1), bins=(xbins, ybins))[0]
    hist = np.clip(hist, 0, config.hist_max_per_pixel) / config.hist_max_per_pixel
    hist = hist.astype(np.float32)[None, ...]
    return torch.from_numpy(hist)


def bev_raster_to_target(bev_raster: np.ndarray, config: TransfuserConfig) -> torch.Tensor:
    bev_raster = _to_numpy(bev_raster)
    if bev_raster.ndim != 3:
        raise ValueError(f"Expected BEV raster as CHW, got shape {tuple(bev_raster.shape)}")

    road = bev_raster[0] > 0
    ego_history = bev_raster[1] > 0 if bev_raster.shape[0] > 1 else np.zeros_like(road)
    traffic = bev_raster[2:].max(axis=0) > 0 if bev_raster.shape[0] > 2 else np.zeros_like(road)

    semantic = np.zeros(road.shape, dtype=np.int64)
    semantic[road] = 1
    semantic[traffic] = 2
    semantic[ego_history] = 3

    return semantic_map_to_target(semantic, config)


def semantic_map_to_target(bev_semantic_map: np.ndarray, config: TransfuserConfig) -> torch.Tensor:
    semantic = _to_numpy(bev_semantic_map).astype(np.int64, copy=False)
    if semantic.ndim != 2:
        raise ValueError(f"Expected BEV semantic map as HW, got shape {tuple(semantic.shape)}")

    target_h, target_w = config.bev_semantic_frame
    if semantic.shape != (target_h, target_w):
        semantic = cv2.resize(
            semantic.astype(np.uint8),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int64)

    return torch.from_numpy(semantic)


def normalize_agent_targets(
    agent_states: np.ndarray,
    agent_labels: np.ndarray,
    config: TransfuserConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    agent_states = _to_numpy(agent_states).astype(np.float32, copy=False)
    agent_labels = _to_numpy(agent_labels).astype(bool, copy=False)

    if agent_states.shape != (config.num_bounding_boxes, BoundingBox2DIndex.size()):
        padded = np.zeros((config.num_bounding_boxes, BoundingBox2DIndex.size()), dtype=np.float32)
        valid = min(config.num_bounding_boxes, agent_states.shape[0])
        padded[:valid] = agent_states[:valid]
        agent_states = padded

    if agent_labels.shape != (config.num_bounding_boxes,):
        padded_labels = np.zeros((config.num_bounding_boxes,), dtype=bool)
        valid = min(config.num_bounding_boxes, agent_labels.shape[0])
        padded_labels[:valid] = agent_labels[:valid]
        agent_labels = padded_labels

    return torch.from_numpy(agent_states), torch.from_numpy(agent_labels)  # ?是否按照距离 ego 最近排序?


def _build_mode_features(sample: Dict[str, np.ndarray], config: TransfuserConfig) -> Dict[str, torch.Tensor]:
    """Generate coarse_trajectories and mode_valid_mask from sample polyline fields.

    Returns dynamic mode coarse trajectories and validity mask.
    Falls back to zeros/False if generation fails.
    """
    from metadrive.policy.diffusion_policy.mode_context import build_mode_context_from_sample
    from metadrive.policy.diffusion_policy.mode_trajectory_generator import ModeTrajectoryGenerator
    from metadrive.policy.diffusion_policy.mode_definitions import mode_slot_count
    num_slots = mode_slot_count(
        config.mode_keep_lane_count,
        config.mode_lane_change_left_count,
        config.mode_lane_change_right_count,
        config.mode_emergency_stop_count,
    )
    try:
        ctx = build_mode_context_from_sample(sample)
        gen = ModeTrajectoryGenerator(
            keep_lane_high_speed_mps=config.mode_keep_high_speed_mps,
            keep_lane_medium_speed_mps=config.mode_keep_medium_speed_mps,
            keep_lane_low_speed_mps=config.mode_keep_low_speed_mps,
            emergency_decel_mps2=config.mode_emergency_decel_mps2,
            keep_lane_level_count=config.mode_keep_lane_count,
            lane_change_left_level_count=config.mode_lane_change_left_count,
            lane_change_right_level_count=config.mode_lane_change_right_count,
            emergency_stop_level_count=config.mode_emergency_stop_count,
        )
        out = gen.generate(ctx)
        return {
            "coarse_trajectories": torch.from_numpy(out.coarse_trajectories),
            "mode_valid_mask": torch.from_numpy(out.mode_valid_mask),
        }
    except Exception:
        return {
            "coarse_trajectories": torch.zeros((num_slots, 8, 2), dtype=torch.float32),
            "mode_valid_mask": torch.zeros((num_slots,), dtype=torch.bool),
        }


def _derive_lateral_decision_for_mode_label(sample: Dict[str, np.ndarray]) -> int:
    """Prefer sample-level future reference-lane semantics over instant IDM state."""
    if "expert_lateral_decision" not in sample:
        raise ValueError(
            "Missing expert_lateral_decision while building gt_mode_label. "
            "Re-collect raw data with expert_lateral_decision enabled before preprocessing."
        )

    if "reference_lane_index" in sample and "future_reference_lane_index" in sample:
        current = int(_to_numpy(sample["reference_lane_index"]).reshape(-1)[0])
        if current >= 0:
            future_indices = _to_numpy(sample["future_reference_lane_index"]).reshape(-1)
            for lane_index in future_indices:
                future = int(lane_index)
                if future < 0:
                    continue
                if future < current:
                    return -1
                if future > current:
                    return 1
            return 0

    return int(_to_numpy(sample["expert_lateral_decision"]).reshape(-1)[0])


def _gt_mode_label_from_expert_decision(
    sample: Dict[str, np.ndarray],
    features: Dict[str, torch.Tensor],
    config: TransfuserConfig,
) -> torch.Tensor:
    """Assign GT mode label for eval display: lateral-group-first, then L2 within group.

    Step 1 — lateral group from GT endpoint net lateral displacement:
      Use gt_xy[-1, 1] - gt_xy[0, 1] relative to a threshold to determine LEFT /
      KEEP / RIGHT.  Robust to IDM decision timing mismatch and lane-index delays.

    Step 2 — speed tier within group by L2 (mask-agnostic):
      Among ALL slots in the preferred lateral group (ignoring mode_valid_mask),
      pick the one whose coarse trajectory minimises mean per-step L2 to GT xy.
      mode_valid_mask is intentionally NOT used here because the generator often
      marks lane-change slots as invalid even when the vehicle is actively changing
      lanes (the generator queries road topology at a slightly different time/pose).
      For training, LossComputer uses its own mask-aware distance assignment.

    Fallback: pure L2 over all slots if no slot found in the preferred group.
    """
    from metadrive.policy.diffusion_policy.mode_definitions import build_mode_slots

    if "coarse_trajectories" not in features or "mode_valid_mask" not in features:
        raise ValueError("Missing dynamic mode features while building gt_mode_label.")

    gt_xy = _to_numpy(sample["trajectory"]).astype(np.float32)[:, :2]
    coarse = _to_numpy(features["coarse_trajectories"]).astype(np.float32)  # (N, T, 2)

    # Step 1: lateral group from net endpoint displacement
    lateral_threshold_m = float(getattr(config, "gt_mode_lateral_threshold_m", 0.5))
    net_lateral = float(gt_xy[-1, 1] - gt_xy[0, 1])   # + = left, - = right in ego frame
    if net_lateral > lateral_threshold_m:
        lateral_decision = -1    # left lane change
    elif net_lateral < -lateral_threshold_m:
        lateral_decision = 1     # right lane change
    else:
        lateral_decision = 0     # keep lane

    mode_slots = build_mode_slots(
        keep_lane_count=config.mode_keep_lane_count,
        lane_change_left_count=config.mode_lane_change_left_count,
        lane_change_right_count=config.mode_lane_change_right_count,
        emergency_stop_count=config.mode_emergency_stop_count,
    )

    # Step 2: pick the closest slot in the preferred group, mask-agnostic
    if lateral_decision == 0:
        group_slots = [s.index for s in mode_slots if s.semantic_group == "KEEP"]
    elif lateral_decision < 0:
        group_slots = [s.index for s in mode_slots if s.lateral_direction == "left"]
    else:
        group_slots = [s.index for s in mode_slots if s.lateral_direction == "right"]

    best_slot, best_dist = -1, float("inf")
    for slot_idx in group_slots:
        if slot_idx >= coarse.shape[0]:
            continue
        dist = float(np.linalg.norm(gt_xy[None] - coarse[slot_idx], axis=-1).mean())
        if dist < best_dist:
            best_dist, best_slot = dist, slot_idx

    if best_slot >= 0:
        return torch.tensor(best_slot, dtype=torch.int8)

    # Fallback: pure L2 over all slots (coarse should always be non-empty)
    dists = np.linalg.norm(gt_xy[None] - coarse, axis=-1).mean(axis=-1)  # (N,)
    return torch.tensor(int(np.argmin(dists)), dtype=torch.int8)


def sample_to_features_targets(
    sample: Dict[str, np.ndarray],
    config: TransfuserConfig,
    prev_reference_lane_index: int = -1,
):
    if "camera_feature" in sample and "lidar_feature" in sample and "status_feature" in sample:
        return processed_sample_to_features_targets(sample, config)

    # Derive lane decision from the transition between the previous and current frame's
    # reference_lane_index.  This determines which lane's centerline is used for target_point.
    current_ref_lane_idx = int(
        _to_numpy(sample.get("reference_lane_index", np.asarray(-1, dtype=np.int16))).reshape(-1)[0]
    )
    lane_decision = _derive_lane_decision_from_reference_transitions(prev_reference_lane_index, current_ref_lane_idx)

    target_point = compute_target_point_from_sample(sample, config, lane_decision)
    topology_polyline = _build_topology_polyline_from_sample(sample, lane_decision)
    features = {
        "camera_feature": stitch_three_cameras(
            sample["left_camera"], sample["front_camera"], sample["right_camera"], config
        ),
        "lidar_feature": lidar_to_histogram(sample["lidar"], config),
        "status_feature": build_status_feature(sample["ego_state"], config),
        "ego_state": torch.from_numpy(_to_numpy(sample["ego_state"]).astype(np.float32, copy=False)),
        "target_point": target_point,
        "target_line": _build_target_line(topology_polyline.numpy(), target_point, config),
        "preference_point": _gt_trajectory_endpoint_from_sample(sample),
        "lane_decision": torch.tensor(int(lane_decision), dtype=torch.int8),
    }
    features.update(_build_mode_features(sample, config))
    agent_states, agent_labels = normalize_agent_targets(
        sample["agent_states"], sample["agent_labels"], config
    )
    targets = {
        "trajectory": torch.from_numpy(_to_numpy(sample["trajectory"]).astype(np.float32, copy=False)),
        "topology_polyline": topology_polyline,
        "agent_states": agent_states,
        "agent_labels": agent_labels,
        "bev_semantic_map": semantic_map_to_target(sample["bev_semantic_map"], config)
        if "bev_semantic_map" in sample
        else bev_raster_to_target(sample["bev_raster"], config),
    }
    targets["gt_mode_label"] = _gt_mode_label_from_expert_decision(sample, features, config)
    return features, targets


def processed_sample_to_features_targets(
    sample: Dict[str, np.ndarray],
    config: Optional[TransfuserConfig] = None,
):
    target_point = (
        torch.from_numpy(_to_numpy(sample["target_point"]).astype(np.float32, copy=True))
        if "target_point" in sample
        else _zero_target_point()
    )
    if "target_line" in sample:
        target_line = torch.from_numpy(_to_numpy(sample["target_line"]).astype(np.float32, copy=True))
    elif "topology_polyline" in sample:
        target_line = _build_target_line(_to_numpy(sample["topology_polyline"]), target_point, config)
    else:
        target_line = torch.from_numpy(_linear_target_line(_to_numpy(target_point), config))

    features = {
        "camera_feature": torch.from_numpy(_to_numpy(sample["camera_feature"]).astype(np.float32, copy=True)),
        "lidar_feature": torch.from_numpy(_to_numpy(sample["lidar_feature"]).astype(np.float32, copy=True)),
        "status_feature": torch.from_numpy(_to_numpy(sample["status_feature"]).astype(np.float32, copy=True)),
        "ego_state": torch.from_numpy(_to_numpy(sample["ego_state"]).astype(np.float32, copy=True)),
        "target_point": target_point,
        "target_line": target_line,
        "preference_point": torch.from_numpy(
            _to_numpy(sample.get("preference_point", target_point)).astype(np.float32, copy=True)
        ),
    }
    if "coarse_trajectories" in sample:
        features["coarse_trajectories"] = torch.from_numpy(
            _to_numpy(sample["coarse_trajectories"]).astype(np.float32, copy=True)
        )
    if "mode_valid_mask" in sample:
        features["mode_valid_mask"] = torch.from_numpy(
            _to_numpy(sample["mode_valid_mask"]).astype(bool, copy=True)
        )
    targets = {
        "trajectory": torch.from_numpy(_to_numpy(sample["trajectory"]).astype(np.float32, copy=True)),
        "agent_states": torch.from_numpy(_to_numpy(sample["agent_states"]).astype(np.float32, copy=True)),
        "agent_labels": torch.from_numpy(_to_numpy(sample["agent_labels"]).astype(bool, copy=True)),
        "bev_semantic_map": torch.from_numpy(_to_numpy(sample["bev_semantic_map"]).astype(np.int64, copy=True)),
    }
    if "topology_polyline" in sample:
        targets["topology_polyline"] = torch.from_numpy(
            _to_numpy(sample["topology_polyline"]).astype(np.float32, copy=True)
        )
    if "coarse_trajectories" in features and "mode_valid_mask" in features:
        targets["gt_mode_label"] = _gt_mode_label_from_expert_decision(sample, features, config)
    return features, targets


def observation_to_features(
    observation: Dict[str, np.ndarray],
    config: TransfuserConfig,
    vehicle=None,
    lane_decision: "LaneDecision" = LaneDecision.KEEP,
) -> Dict[str, torch.Tensor]:
    target_point = compute_target_point(vehicle, config, lane_decision) if vehicle is not None else _zero_target_point()
    topology_polyline = (
        _build_topology_polyline_from_live_vehicle(vehicle, lane_decision)
        if vehicle is not None
        else torch.zeros((0, 2), dtype=torch.float32)
    )
    return {
        "camera_feature": stitch_three_cameras(
            observation["rgb_left"], observation["rgb_front"], observation["rgb_right"], config
        ),
        "lidar_feature": lidar_to_histogram(observation["lidar"], config),
        "status_feature": build_status_feature(observation["ego_state"], config),
        "ego_state": torch.from_numpy(_to_numpy(observation["ego_state"]).astype(np.float32, copy=False)),
        "target_point": target_point,
        "preference_point": target_point.clone(),
        "target_line": _build_target_line(topology_polyline.numpy(), target_point, config),
        "topology_polyline": topology_polyline,
    }


class MetaDriveTransfuserDataset(Dataset):
    """Dataset adapter from MetaDrive npz shards to TransFuser feature/target dicts."""

    def __init__(
        self,
        dataset_root: Union[str, Path],
        config: TransfuserConfig,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.config = config
        self.split = split
        self.max_samples = max_samples
        self._cache_all_shards = config.cache_shards_in_memory
        self.shard_paths = self._resolve_shards()
        self._index = []
        self._cached_shard_idx = None
        self._cached_shard = None
        self._all_shards = {}
        self._build_index()
        if self._cache_all_shards:
            self._preload_shards()

    def _resolve_shards(self):
        shard_dir = self.dataset_root / "shards"
        if not shard_dir.exists():
            raise FileNotFoundError(
                f"Dataset shard directory does not exist: {shard_dir}. "
                f"Expected dataset_root to point to a dataset folder containing 'shards/'."
            )
        shard_paths = self._list_shard_entries(shard_dir)
        if self.split == "all":
            return shard_paths
        split_path = self.dataset_root / "splits" / f"{self.split}.txt"
        if not split_path.exists():
            return shard_paths
        names = {line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()}
        names = {Path(name).stem if name.endswith(".npz") else name for name in names}
        resolved = [path for path in shard_paths if path.stem in names or path.name in names]
        if not resolved:
            raise FileNotFoundError(
                f"No shards found for split '{self.split}' under {self.dataset_root}. "
                f"Checked split file: {split_path}"
            )
        return resolved

    def _list_shard_entries(self, shard_dir: Path):
        entries = {}
        for path in sorted(shard_dir.iterdir()):
            if path.is_dir() and path.name.startswith("shard_"):
                entries[path.name] = path
        for path in sorted(shard_dir.glob("*.npz")):
            entries.setdefault(path.stem, path)
        return [entries[key] for key in sorted(entries)]

    def _is_processed_dir_shard(self, shard_path: Path) -> bool:
        return shard_path.is_dir()

    def _load_processed_dir_meta(self, shard_path: Path) -> Dict:
        meta_path = shard_path / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Processed shard metadata not found: {meta_path}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def _build_index(self):
        total = 0
        for shard_idx, shard_path in enumerate(self.shard_paths):
            try:
                if self._is_processed_dir_shard(shard_path):
                    trajectory = np.load(shard_path / "trajectory.npy", mmap_mode="r", allow_pickle=False)
                    length = int(trajectory.shape[0])
                else:
                    with np.load(shard_path, allow_pickle=False) as shard:
                        length = int(shard["trajectory"].shape[0])
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to build dataset index from shard '{shard_path}'. "
                    "Run the dataset integrity verifier before training."
                ) from exc
            for sample_idx in range(length):
                self._index.append((shard_idx, sample_idx))
                total += 1
                if self.max_samples is not None and total >= self.max_samples:
                    return

    def __len__(self):
        return len(self._index)

    def get_sample_metadata(self, idx: int) -> Dict[str, Union[int, str]]:
        shard_idx, sample_idx = self._index[idx]
        shard_path = self.shard_paths[shard_idx]
        metadata = {
            "sample_index": int(idx),
            "shard_index": int(shard_idx),
            "shard_name": shard_path.name,
            "shard_stem": shard_path.stem,
            "local_index": int(sample_idx),
        }
        shard = self._load_shard(shard_idx)
        for field_name in METADATA_PASSTHROUGH_FIELDS:
            if field_name not in shard:
                continue
            value = np.asarray(shard[field_name][sample_idx]).reshape(-1)
            if value.size == 0:
                continue
            scalar = value[0]
            if field_name in ("trajectory_mode", "gt_mode_label"):
                metadata[field_name] = int(scalar)
                continue
            if isinstance(scalar, bytes):
                metadata[field_name] = scalar.decode("utf-8")
            else:
                metadata[field_name] = str(scalar)
        return metadata

    # 用于按索引获取一个样本的特征和目标
    def __getitem__(self, idx):
        shard_idx, sample_idx = self._index[idx]
        shard = self._load_shard(shard_idx)
        sample = {key: shard[key][sample_idx] for key in shard.keys()}
        # Provide the previous frame's reference_lane_index so sample_to_features_targets
        # can derive lane_decision from the transition.  Use -1 at episode boundaries.
        prev_ref_lane_idx = -1
        if sample_idx > 0 and "reference_lane_index" in shard:
            try:
                prev_ref_lane_idx = int(np.asarray(shard["reference_lane_index"][sample_idx - 1]).reshape(-1)[0])
            except Exception:
                prev_ref_lane_idx = -1
        return sample_to_features_targets(sample, self.config, prev_reference_lane_index=prev_ref_lane_idx)

    def _load_shard(self, shard_idx: int) -> Dict[str, np.ndarray]:
        if self._cache_all_shards:
            return self._all_shards[shard_idx]
        if self._cached_shard_idx == shard_idx and self._cached_shard is not None:
            return self._cached_shard
        shard_path = self.shard_paths[shard_idx]
        if self._is_processed_dir_shard(shard_path):
            shard = self._load_processed_dir_shard(shard_path, mmap_mode="r")
            self._cached_shard_idx = shard_idx
            self._cached_shard = shard
            return shard
        try:
            with np.load(shard_path, allow_pickle=False) as raw:
                shard = {}
                for key in raw.files:
                    try:
                        shard[key] = raw[key]
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to read key '{key}' from shard '{shard_path}'. "
                            "Run the dataset integrity verifier before training."
                        ) from exc
        except RuntimeError:
            raise
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise RuntimeError(
                f"Failed to load shard '{shard_path}'. Run the dataset integrity verifier before training."
            ) from exc
        self._cached_shard_idx = shard_idx
        self._cached_shard = shard
        return shard

    def _preload_shards(self) -> None:
        for shard_idx in range(len(self.shard_paths)):
            shard_path = self.shard_paths[shard_idx]
            if self._is_processed_dir_shard(shard_path):
                self._all_shards[shard_idx] = self._load_processed_dir_shard(shard_path, mmap_mode=None)
                continue
            try:
                with np.load(shard_path, allow_pickle=False) as raw:
                    preloaded = {}
                    for key in raw.files:
                        try:
                            preloaded[key] = raw[key]
                        except Exception as exc:
                            raise RuntimeError(
                                f"Failed to preload key '{key}' from shard '{shard_path}'. "
                                "Run the dataset integrity verifier before training."
                            ) from exc
                    self._all_shards[shard_idx] = preloaded
            except RuntimeError:
                raise
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                raise RuntimeError(
                    f"Failed to preload shard '{shard_path}'. Run the dataset integrity verifier before training."
                ) from exc

    def _load_processed_dir_shard(self, shard_path: Path, mmap_mode: Optional[str]) -> Dict[str, np.ndarray]:
        try:
            meta = self._load_processed_dir_meta(shard_path)
            fields = tuple(meta.get("fields", {}).keys()) if isinstance(meta.get("fields"), dict) else PROCESSED_DIR_FIELDS
            shard = {}
            for key in fields:
                field_path = shard_path / f"{key}.npy"
                if not field_path.exists():
                    if key in ("target_line", "preference_point"):
                        continue
                    raise FileNotFoundError(f"Processed shard field not found: {field_path}")
                array = np.load(field_path, mmap_mode=mmap_mode, allow_pickle=False)
                shard[key] = np.asarray(array) if mmap_mode is None else array
            return shard
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load processed directory shard '{shard_path}'. "
                "Run the dataset integrity verifier before training."
            ) from exc
