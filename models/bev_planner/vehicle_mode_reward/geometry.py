"""Trajectory interpolation, coordinate transforms, and road geometry helpers."""

from __future__ import annotations

import math
from typing import Mapping

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from envs.observations.semantic_bev import SemanticBEVConfig

from .config import JointRewardError, VehicleModeRewardConfig
from .constants import AGENT_IDS, NUM_ROLES, TRAJECTORY_SHAPE

def _as_numpy_model_field(model_inputs: object, name: str) -> np.ndarray:
    if isinstance(model_inputs, Mapping):
        value = model_inputs.get(name)
    else:
        value = getattr(model_inputs, name, None)
    if value is None:
        raise JointRewardError(f"model_inputs is missing {name}")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)

def _dense_local_trajectories(
    trajectories: np.ndarray, config: VehicleModeRewardConfig
) -> tuple[np.ndarray, np.ndarray]:
    source_times = np.arange(1, 9, dtype=np.float64) * config.trajectory_dt_s
    target_times = np.arange(
        0.0,
        source_times[-1] + 0.5 * config.interpolation_dt_s,
        config.interpolation_dt_s,
        dtype=np.float64,
    )
    source_with_origin = np.concatenate(([0.0], source_times))
    dense = np.empty(
        (*trajectories.shape[:2], len(target_times), 3), dtype=np.float64
    )
    for group in range(trajectories.shape[0]):
        for role in range(NUM_ROLES):
            trajectory = trajectories[group, role].astype(np.float64)
            xy_with_origin = np.concatenate(
                (np.zeros((1, 2), dtype=np.float64), trajectory[:, :2]), axis=0
            )
            heading = np.unwrap(
                np.concatenate(([0.0], trajectory[:, 2]), axis=0)
            )
            dense[group, role, :, 0] = np.interp(
                target_times, source_with_origin, xy_with_origin[:, 0]
            )
            dense[group, role, :, 1] = np.interp(
                target_times, source_with_origin, xy_with_origin[:, 1]
            )
            dense[group, role, :, 2] = np.arctan2(
                np.sin(np.interp(target_times, source_with_origin, heading)),
                np.cos(np.interp(target_times, source_with_origin, heading)),
            )
    return dense, target_times


def _agent_pose(env: object, agent_id: str) -> np.ndarray:
    vehicle = (getattr(env, "agents", {}) or {}).get(agent_id)
    if vehicle is None:
        raise JointRewardError(f"simulator is missing active {agent_id}")
    position = np.asarray(getattr(vehicle, "position", ()), dtype=np.float64)
    heading = float(getattr(vehicle, "heading_theta", float("nan")))
    if position.size < 2 or not np.isfinite(position[:2]).all() or not math.isfinite(
        heading
    ):
        raise JointRewardError(f"simulator pose for {agent_id} is invalid")
    return np.asarray([position[0], position[1], heading], dtype=np.float64)


def _local_to_world(local: np.ndarray, pose: np.ndarray) -> np.ndarray:
    cos_h = math.cos(float(pose[2]))
    sin_h = math.sin(float(pose[2]))
    world = np.empty_like(local, dtype=np.float64)
    world[..., 0] = pose[0] + cos_h * local[..., 0] - sin_h * local[..., 1]
    world[..., 1] = pose[1] + sin_h * local[..., 0] + cos_h * local[..., 1]
    world[..., 2] = np.arctan2(
        np.sin(local[..., 2] + pose[2]), np.cos(local[..., 2] + pose[2])
    )
    return world


def tracking_aware_half_extents(
    config: VehicleModeRewardConfig,
) -> tuple[float, float]:
    """Return conservative half extents under measured tracking error."""

    heading = config.tracking_heading_margin_rad
    half_length = (
        0.5 * config.vehicle_length_m
        + config.tracking_longitudinal_margin_m
        + 0.5 * config.vehicle_width_m * math.sin(heading)
    )
    half_width = (
        0.5 * config.vehicle_width_m
        + config.tracking_lateral_margin_m
        + 0.5 * config.vehicle_length_m * math.sin(heading)
    )
    return half_length, half_width


def _footprint_points(
    local: np.ndarray,
    config: VehicleModeRewardConfig,
    *,
    tracking_aware: bool,
) -> np.ndarray:
    poses = np.asarray(local, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3 or not np.isfinite(poses).all():
        raise JointRewardError("footprint poses must be finite [N,3]")
    if tracking_aware:
        half_length, half_width = tracking_aware_half_extents(config)
    else:
        half_length = 0.5 * config.vehicle_length_m
        half_width = 0.5 * config.vehicle_width_m
    bev_config = SemanticBEVConfig()
    longitudinal_resolution = (
        bev_config.x_max_m - bev_config.x_min_m
    ) / (bev_config.height - 1)
    lateral_resolution = (
        bev_config.y_max_m - bev_config.y_min_m
    ) / (bev_config.width - 1)
    longitudinal = np.linspace(
        -half_length,
        half_length,
        max(2, int(math.ceil(2.0 * half_length / longitudinal_resolution)) + 1),
        dtype=np.float64,
    )
    lateral = np.linspace(
        -half_width,
        half_width,
        max(2, int(math.ceil(2.0 * half_width / lateral_resolution)) + 1),
        dtype=np.float64,
    )
    longitudinal_grid, lateral_grid = np.meshgrid(
        longitudinal, lateral, indexing="ij"
    )
    offsets = np.column_stack(
        (longitudinal_grid.reshape(-1), lateral_grid.reshape(-1))
    )
    heading = poses[:, 2]
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    points = np.empty((len(local), len(offsets), 2), dtype=np.float64)
    for index, (longitudinal, lateral) in enumerate(offsets):
        points[:, index, 0] = (
            poses[:, 0] + cos_h * longitudinal - sin_h * lateral
        )
        points[:, index, 1] = (
            poses[:, 1] + sin_h * longitudinal + cos_h * lateral
        )
    return points


def _local_points_to_bev_coordinates(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bev_config = SemanticBEVConfig()
    col = (
        (bev_config.y_max_m - points[:, 1])
        / (bev_config.y_max_m - bev_config.y_min_m)
        * (bev_config.width - 1)
    )
    row = (
        (bev_config.x_max_m - points[:, 0])
        / (bev_config.x_max_m - bev_config.x_min_m)
        * (bev_config.height - 1)
    )
    return row, col


def footprint_outside_drivable_series(
    local_poses: np.ndarray,
    drivable: np.ndarray,
    config: VehicleModeRewardConfig,
) -> np.ndarray:
    """Return physical-footprint raster violations for every dense sample."""

    drivable_values = np.asarray(drivable)
    bev_config = SemanticBEVConfig()
    if drivable_values.shape != (bev_config.height, bev_config.width):
        raise JointRewardError("drivable BEV must match SemanticBEVConfig")
    poses = np.asarray(local_poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3 or not np.isfinite(poses).all():
        raise JointRewardError("footprint poses must be finite [N,3]")
    half_length = 0.5 * config.vehicle_length_m
    half_width = 0.5 * config.vehicle_width_m
    offsets = np.asarray(
        [
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ],
        dtype=np.float64,
    )
    result = np.zeros(len(poses), dtype=np.bool_)
    for index, pose in enumerate(poses):
        cosine = math.cos(float(pose[2]))
        sine = math.sin(float(pose[2]))
        corners = np.empty((4, 2), dtype=np.float64)
        corners[:, 0] = (
            pose[0] + cosine * offsets[:, 0] - sine * offsets[:, 1]
        )
        corners[:, 1] = (
            pose[1] + sine * offsets[:, 0] + cosine * offsets[:, 1]
        )
        row, col = _local_points_to_bev_coordinates(corners)
        if bool(
            np.any(row < 0.0)
            or np.any(row > bev_config.height - 1)
            or np.any(col < 0.0)
            or np.any(col > bev_config.width - 1)
        ):
            result[index] = True
            continue
        polygon = np.column_stack((np.rint(col), np.rint(row))).astype(
            np.int32
        )
        col_min = int(np.min(polygon[:, 0]))
        col_max = int(np.max(polygon[:, 0]))
        row_min = int(np.min(polygon[:, 1]))
        row_max = int(np.max(polygon[:, 1]))
        roi_mask = np.zeros(
            (row_max - row_min + 1, col_max - col_min + 1),
            dtype=np.uint8,
        )
        roi_polygon = polygon - np.asarray([col_min, row_min], dtype=np.int32)
        cv2.fillConvexPoly(roi_mask, roi_polygon, 1)
        roi_drivable = drivable_values[
            row_min : row_max + 1, col_min : col_max + 1
        ]
        result[index] = bool(np.any(roi_drivable[roi_mask != 0] == 0))
    return result


def drivable_signed_distance_m(drivable: np.ndarray) -> np.ndarray:
    """Build a metric signed-distance field from a semantic drivable raster."""

    values = np.asarray(drivable)
    bev_config = SemanticBEVConfig()
    if values.shape != (bev_config.height, bev_config.width):
        raise JointRewardError("drivable BEV must match SemanticBEVConfig")
    mask = values != 0
    row_resolution = (bev_config.x_max_m - bev_config.x_min_m) / (
        bev_config.height - 1
    )
    col_resolution = (bev_config.y_max_m - bev_config.y_min_m) / (
        bev_config.width - 1
    )
    sampling = (row_resolution, col_resolution)
    inside = distance_transform_edt(
        np.pad(mask, 1, mode="constant", constant_values=False),
        sampling=sampling,
    )[1:-1, 1:-1]
    if bool(np.any(mask)):
        outside = distance_transform_edt(~mask, sampling=sampling)
    else:
        outside = np.full(
            mask.shape,
            math.hypot(
                bev_config.x_max_m - bev_config.x_min_m,
                bev_config.y_max_m - bev_config.y_min_m,
            ),
            dtype=np.float64,
        )
    return np.ascontiguousarray(inside - outside, dtype=np.float64)


def footprint_road_margin_series(
    local_poses: np.ndarray,
    signed_distance_m: np.ndarray,
    config: VehicleModeRewardConfig,
    *,
    tracking_aware: bool,
) -> np.ndarray:
    """Return the minimum signed drivable margin across footprint samples."""

    field = np.asarray(signed_distance_m, dtype=np.float64)
    bev_config = SemanticBEVConfig()
    if (
        field.shape != (bev_config.height, bev_config.width)
        or not np.isfinite(field).all()
    ):
        raise JointRewardError("signed drivable distance must be finite BEV [H,W]")
    points_by_time = _footprint_points(
        local_poses, config, tracking_aware=tracking_aware
    )
    points = points_by_time.reshape(-1, 2)
    row, col = _local_points_to_bev_coordinates(points)
    inside = (
        (row >= 0.0)
        & (row <= bev_config.height - 1)
        & (col >= 0.0)
        & (col <= bev_config.width - 1)
    )
    sampled = np.empty(len(points), dtype=np.float64)
    if np.any(inside):
        indices = np.flatnonzero(inside)
        row_inside = row[indices]
        col_inside = col[indices]
        row_low = np.floor(row_inside).astype(np.int64)
        col_low = np.floor(col_inside).astype(np.int64)
        row_high = np.minimum(row_low + 1, bev_config.height - 1)
        col_high = np.minimum(col_low + 1, bev_config.width - 1)
        row_fraction = row_inside - row_low
        col_fraction = col_inside - col_low
        sampled[indices] = (
            field[row_low, col_low]
            * (1.0 - row_fraction)
            * (1.0 - col_fraction)
            + field[row_high, col_low]
            * row_fraction
            * (1.0 - col_fraction)
            + field[row_low, col_high]
            * (1.0 - row_fraction)
            * col_fraction
            + field[row_high, col_high] * row_fraction * col_fraction
        )
    if np.any(~inside):
        indices = np.flatnonzero(~inside)
        clamped_row = np.clip(row[indices], 0.0, bev_config.height - 1)
        clamped_col = np.clip(col[indices], 0.0, bev_config.width - 1)
        row_resolution = (bev_config.x_max_m - bev_config.x_min_m) / (
            bev_config.height - 1
        )
        col_resolution = (bev_config.y_max_m - bev_config.y_min_m) / (
            bev_config.width - 1
        )
        outside_distance = np.hypot(
            (row[indices] - clamped_row) * row_resolution,
            (col[indices] - clamped_col) * col_resolution,
        )
        sampled[indices] = -outside_distance
    return np.min(sampled.reshape(points_by_time.shape[:2]), axis=1)
