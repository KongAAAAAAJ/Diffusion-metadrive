"""Shared oriented-box collision geometry for joint planning."""

from __future__ import annotations

import numpy as np


def obb_overlap_series(
    first: np.ndarray,
    first_dimensions: tuple[float, float],
    second: np.ndarray,
    second_dimensions: tuple[float, float],
    margin_m,
) -> bool:
    """Return whether two synchronized ``[N,3]`` pose series overlap."""

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 3:
        raise ValueError("OBB trajectories must have matching [N,3] shapes")
    margins = np.asarray(margin_m, dtype=np.float64)
    if margins.ndim == 0:
        margins = np.full((len(first), 2), float(margins), dtype=np.float64)
    elif margins.shape == (len(first),):
        margins = np.repeat(margins[:, None], 2, axis=1)
    if margins.shape != (len(first), 2) or np.any(margins < 0.0):
        raise ValueError("OBB margin must be non-negative scalar, [N], or [N,2]")

    first_half = np.asarray(first_dimensions, dtype=np.float64)[None, :] * 0.5 + margins
    second_half = np.asarray(second_dimensions, dtype=np.float64)[None, :] * 0.5 + margins
    first_long = np.column_stack([np.cos(first[:, 2]), np.sin(first[:, 2])])
    first_lat = np.column_stack([-first_long[:, 1], first_long[:, 0]])
    second_long = np.column_stack([np.cos(second[:, 2]), np.sin(second[:, 2])])
    second_lat = np.column_stack([-second_long[:, 1], second_long[:, 0]])
    delta = second[:, :2] - first[:, :2]
    separated = np.zeros(len(first), dtype=np.bool_)
    for axis in (first_long, first_lat, second_long, second_lat):
        first_radius = (
            first_half[:, 0] * np.abs(np.einsum("ij,ij->i", first_long, axis))
            + first_half[:, 1] * np.abs(np.einsum("ij,ij->i", first_lat, axis))
        )
        second_radius = (
            second_half[:, 0] * np.abs(np.einsum("ij,ij->i", second_long, axis))
            + second_half[:, 1] * np.abs(np.einsum("ij,ij->i", second_lat, axis))
        )
        separated |= (
            np.abs(np.einsum("ij,ij->i", delta, axis))
            > first_radius + second_radius
        )
    return bool(np.any(~separated))


def world_trajectory_to_ego_local(
    trajectory_world: np.ndarray,
    origin_pose: np.ndarray,
) -> np.ndarray:
    """Convert world ``[N,3]`` poses to the exact float32 label frame."""

    trajectory = np.asarray(trajectory_world, dtype=np.float64)
    origin = np.asarray(origin_pose, dtype=np.float64).reshape(-1)
    if trajectory.ndim != 2 or trajectory.shape[1] != 3:
        raise ValueError("trajectory_world must have shape [N,3]")
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("origin_pose must be finite [3]")
    if not np.isfinite(trajectory).all():
        raise ValueError("trajectory_world must be finite")
    delta = trajectory[:, :2] - origin[None, :2]
    cos_h = float(np.cos(origin[2]))
    sin_h = float(np.sin(origin[2]))
    local_heading = (
        trajectory[:, 2] - origin[2] + np.pi
    ) % (2.0 * np.pi) - np.pi
    local = np.column_stack(
        [
            cos_h * delta[:, 0] + sin_h * delta[:, 1],
            -sin_h * delta[:, 0] + cos_h * delta[:, 1],
            local_heading,
        ]
    )
    return np.ascontiguousarray(local.astype(np.float32))


__all__ = ["obb_overlap_series", "world_trajectory_to_ego_local"]
