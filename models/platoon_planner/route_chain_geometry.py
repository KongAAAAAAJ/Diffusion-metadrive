"""Strict continuous geometry for frozen navigation lane chains."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


class RouteChainGeometryError(ValueError):
    """Raised when a navigation lane chain cannot form a continuous path."""


def lane_chain_indices(lanes: Sequence[object]) -> tuple[tuple, ...]:
    return tuple(tuple(getattr(lane, "index", ()) or ()) for lane in lanes)


def validate_lane_chain_topology(lanes: Sequence[object]) -> None:
    if not lanes:
        raise RouteChainGeometryError("route lane chain is empty")
    for lane in lanes:
        index = tuple(getattr(lane, "index", ()) or ())
        if len(index) < 3 or float(getattr(lane, "length", 0.0) or 0.0) <= 0.0:
            raise RouteChainGeometryError("route lane has an invalid index or length")
    for previous, successor in zip(lanes, lanes[1:]):
        first = tuple(getattr(previous, "index", ()) or ())
        second = tuple(getattr(successor, "index", ()) or ())
        if first[1] != second[0]:
            raise RouteChainGeometryError(
                f"route lane chain is disconnected: {first!r} -> {second!r}"
            )


def _lane_samples(
    lane,
    start_s: float,
    end_s: float,
    step_m: float,
    lateral_m: float = 0.0,
) -> np.ndarray:
    distance = max(float(end_s) - float(start_s), 0.0)
    count = max(int(math.ceil(distance / step_m)), 1)
    longitudinal = np.linspace(float(start_s), float(end_s), count + 1)
    rows = []
    for value in longitudinal:
        point = np.asarray(lane.position(float(value), float(lateral_m))[:2], dtype=np.float64)
        heading = float(lane.heading_theta_at(float(value)))
        rows.append((float(point[0]), float(point[1]), heading))
    return np.asarray(rows, dtype=np.float64)


def _hermite_join(
    start: np.ndarray,
    start_heading: float,
    end: np.ndarray,
    end_heading: float,
    step_m: float,
) -> np.ndarray:
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
    return np.column_stack((xy, heading))


def build_continuous_lane_chain_path(
    lanes: Sequence[object],
    *,
    start_s: float,
    start_lateral_m: float = 0.0,
    step_m: float = 0.25,
    seam_transition_m: float = 8.0,
) -> np.ndarray:
    """Build a dense center path, smoothing connected offset road seams.

    Some MetaDrive exit connectors start one lane-width to the side of the
    predecessor centerline.  Their road surfaces are connected, but simply
    concatenating lane centers creates a position jump.  The join is therefore
    formed over the final/initial route segments while preserving both lane
    tangents.  This is route geometry; it does not alter the temporal speed
    profile or any hard road/safety boundary.
    """

    lane_values = list(lanes)
    validate_lane_chain_topology(lane_values)
    if not np.isfinite([start_s, start_lateral_m, step_m, seam_transition_m]).all():
        raise RouteChainGeometryError("route geometry inputs must be finite")
    if step_m <= 0.0 or step_m > 0.5 or seam_transition_m <= 0.0:
        raise RouteChainGeometryError("route sampling step/transition is invalid")
    first_length = float(getattr(lane_values[0], "length", 0.0) or 0.0)
    if start_s < -1.0e-6 or start_s > first_length + 1.0e-6:
        raise RouteChainGeometryError("route start_s is outside the first lane")

    pieces: list[np.ndarray] = []
    current_start = float(np.clip(start_s, 0.0, first_length))
    for index, lane in enumerate(lane_values):
        lane_length = float(getattr(lane, "length", 0.0) or 0.0)
        lateral = float(start_lateral_m) if index == 0 else 0.0
        if index == len(lane_values) - 1:
            pieces.append(_lane_samples(lane, current_start, lane_length, step_m, lateral))
            break

        successor = lane_values[index + 1]
        successor_length = float(getattr(successor, "length", 0.0) or 0.0)
        end_center = np.asarray(lane.position(lane_length, 0.0)[:2], dtype=np.float64)
        next_center = np.asarray(successor.position(0.0, 0.0)[:2], dtype=np.float64)
        seam_gap = float(np.linalg.norm(next_center - end_center))
        widths = float(getattr(lane, "width", 3.5) or 3.5) + float(
            getattr(successor, "width", 3.5) or 3.5
        )
        if seam_gap > 0.5 * widths + 1.0 + 1.0e-6:
            raise RouteChainGeometryError(
                f"connected route seam is too wide ({seam_gap:.3f}m)"
            )
        if seam_gap <= 0.5:
            pieces.append(_lane_samples(lane, current_start, lane_length, step_m, lateral))
            current_start = 0.0
            continue

        back = min(float(seam_transition_m), max(lane_length - current_start, 0.0))
        ahead = min(float(seam_transition_m), successor_length)
        if back < 1.0 or ahead < 1.0:
            raise RouteChainGeometryError("offset route seam lacks transition length")
        join_start_s = lane_length - back
        join_end_s = ahead
        pieces.append(_lane_samples(lane, current_start, join_start_s, step_m, lateral))
        join_start = np.asarray(lane.position(join_start_s, 0.0)[:2], dtype=np.float64)
        join_end = np.asarray(successor.position(join_end_s, 0.0)[:2], dtype=np.float64)
        pieces.append(
            _hermite_join(
                join_start,
                float(lane.heading_theta_at(join_start_s)),
                join_end,
                float(successor.heading_theta_at(join_end_s)),
                step_m,
            )
        )
        current_start = join_end_s

    result = np.concatenate(
        [piece if index == 0 else piece[1:] for index, piece in enumerate(pieces)],
        axis=0,
    )
    delta = np.linalg.norm(np.diff(result[:, :2], axis=0), axis=1)
    keep = np.concatenate(([True], delta > 1.0e-8))
    result = np.ascontiguousarray(result[keep], dtype=np.float64)
    if result.shape[0] < 2 or not np.isfinite(result).all():
        raise RouteChainGeometryError("route spatial path is empty or non-finite")
    raw_arc = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(result[:, :2], axis=0), axis=1)))
    )
    query_arc = np.arange(0.0, float(raw_arc[-1]), float(step_m), dtype=np.float64)
    query_arc = np.append(query_arc, float(raw_arc[-1]))
    unwrapped_heading = np.unwrap(result[:, 2])
    result = np.column_stack(
        (
            np.interp(query_arc, raw_arc, result[:, 0]),
            np.interp(query_arc, raw_arc, result[:, 1]),
            np.interp(query_arc, raw_arc, unwrapped_heading),
        )
    )
    result[:, 2] = np.arctan2(np.sin(result[:, 2]), np.cos(result[:, 2]))
    if float(np.max(np.linalg.norm(np.diff(result[:, :2], axis=0), axis=1))) > 0.5 + 1.0e-6:
        raise RouteChainGeometryError("route spatial path sampling exceeds 0.5m")
    return np.ascontiguousarray(result, dtype=np.float64)


__all__ = [
    "RouteChainGeometryError",
    "build_continuous_lane_chain_path",
    "lane_chain_indices",
    "validate_lane_chain_topology",
]
