"""Drivable transition surfaces for offset but connected lane seams."""

from __future__ import annotations

import math

import numpy as np

from metadrive.component.lane.point_lane import PointLane
from metadrive.type import MetaDriveType


def build_lane_seam_transition_centerline(
    predecessor,
    successor,
    *,
    transition_m: float = 8.0,
    step_m: float = 0.25,
    predecessor_start_s: float | None = None,
    successor_end_s: float | None = None,
) -> np.ndarray:
    """Return a smooth centerline spanning an offset connected-lane seam.

    MetaDrive's G block connects the right-most mainline lane to the exit bend
    at the same graph node, but the two lane centers are one lane-width apart.
    The transition is a cubic Hermite curve using the real lane tangents.  The
    same helper is consumed by the planner path and the junction road surface
    so their geometry cannot drift apart.
    """

    values = np.asarray([transition_m, step_m], dtype=np.float64)
    if not np.isfinite(values).all() or transition_m <= 0.0:
        raise ValueError("lane seam transition distance must be positive and finite")
    if step_m <= 0.0 or step_m > 0.5:
        raise ValueError("lane seam sampling step must be in (0, 0.5]")
    predecessor_length = float(getattr(predecessor, "length", 0.0) or 0.0)
    successor_length = float(getattr(successor, "length", 0.0) or 0.0)
    if predecessor_length <= 0.0 or successor_length <= 0.0:
        raise ValueError("lane seam endpoints must have positive lengths")

    start_s = (
        predecessor_length - min(float(transition_m), predecessor_length)
        if predecessor_start_s is None
        else float(predecessor_start_s)
    )
    end_s = (
        min(float(transition_m), successor_length)
        if successor_end_s is None
        else float(successor_end_s)
    )
    if not np.isfinite([start_s, end_s]).all():
        raise ValueError("lane seam transition bounds must be finite")
    if not 0.0 <= start_s < predecessor_length:
        raise ValueError("lane seam predecessor start is outside the lane")
    if not 0.0 < end_s <= successor_length:
        raise ValueError("lane seam successor end is outside the lane")
    back = predecessor_length - start_s
    ahead = end_s
    if back < 1.0 or ahead < 1.0:
        raise ValueError("offset lane seam lacks transition length")
    start = np.asarray(predecessor.position(start_s, 0.0)[:2], dtype=np.float64)
    end = np.asarray(successor.position(end_s, 0.0)[:2], dtype=np.float64)
    start_heading = float(predecessor.heading_theta_at(start_s))
    end_heading = float(successor.heading_theta_at(end_s))
    if not np.isfinite(np.concatenate((start, end, [start_heading, end_heading]))).all():
        raise ValueError("lane seam endpoints must be finite")

    chord = float(np.linalg.norm(end - start))
    if chord <= 1.0e-8:
        return np.asarray([[start[0], start[1], start_heading]], dtype=np.float64)
    tangent_length = max(chord * 0.75, 1.0)
    tangent_start = tangent_length * np.asarray(
        [math.cos(start_heading), math.sin(start_heading)], dtype=np.float64
    )
    tangent_end = tangent_length * np.asarray(
        [math.cos(end_heading), math.sin(end_heading)], dtype=np.float64
    )
    count = max(int(math.ceil(chord / step_m)) * 2, 4)
    u = np.linspace(0.0, 1.0, count + 1)
    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
    h10 = u**3 - 2.0 * u**2 + u
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = u**3 - u**2
    xy = (
        h00[:, None] * start[None, :]
        + h10[:, None] * tangent_start[None, :]
        + h01[:, None] * end[None, :]
        + h11[:, None] * tangent_end[None, :]
    )
    delta = np.gradient(xy, axis=0)
    heading = np.arctan2(delta[:, 1], delta[:, 0])
    return np.ascontiguousarray(np.column_stack((xy, heading)), dtype=np.float64)


def build_lane_seam_drivable_surface(
    predecessor,
    successor,
    *,
    transition_m: float = 8.0,
    step_m: float = 0.25,
) -> PointLane:
    """Build a non-routing lane polygon for the junction's drivable map surface."""

    centerline = build_lane_seam_transition_centerline(
        predecessor,
        successor,
        transition_m=transition_m,
        step_m=step_m,
    )
    predecessor_width = float(predecessor.width_at(float(predecessor.length)))
    successor_width = float(successor.width_at(0.0))
    width = min(predecessor_width, successor_width)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("lane seam surface width must be positive and finite")
    surface = PointLane(
        centerline[:, :2],
        width=width,
        need_lane_localization=False,
        metadrive_type=MetaDriveType.LANE_SURFACE_UNSTRUCTURE,
    )
    surface.is_junction_drivable_surface = True
    return surface


__all__ = [
    "build_lane_seam_drivable_surface",
    "build_lane_seam_transition_centerline",
]
