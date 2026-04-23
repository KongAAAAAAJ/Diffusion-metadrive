from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

TOPOLOGY_CURRENT = 0
TOPOLOGY_LEFT = 1
TOPOLOGY_RIGHT = 2
TOPOLOGY_BRANCH = 3
TOPOLOGY_NAMES = ("current", "left", "right", "branch")


def _as_polyline(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] < 2:
        return None
    arr = arr[:, :2]
    segment_lengths = np.linalg.norm(np.diff(arr, axis=0), axis=-1)
    if float(segment_lengths.sum()) <= 1e-6:
        return None
    return arr


def _polyline_length(polyline: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(polyline, axis=0), axis=-1).sum())


def _interpolate_at_fraction(polyline: np.ndarray, s: float) -> np.ndarray:
    s = float(np.clip(s, 0.0, 1.0))
    target_length = s * _polyline_length(polyline)
    accumulated = 0.0
    for start, end in zip(polyline[:-1], polyline[1:]):
        segment = end - start
        segment_length = float(np.linalg.norm(segment))
        if segment_length <= 1e-6:
            continue
        if accumulated + segment_length >= target_length:
            ratio = (target_length - accumulated) / segment_length
            return (start + ratio * segment).astype(np.float32)
        accumulated += segment_length
    return polyline[-1].astype(np.float32)


def _project_to_polyline(point: np.ndarray, polyline: np.ndarray) -> tuple[np.ndarray, float, float]:
    point = np.asarray(point, dtype=np.float32)[:2]
    total_length = _polyline_length(polyline)
    best_point = polyline[0]
    best_distance = float("inf")
    best_arc = 0.0
    accumulated = 0.0
    for start, end in zip(polyline[:-1], polyline[1:]):
        segment = end - start
        segment_length = float(np.linalg.norm(segment))
        if segment_length <= 1e-6:
            continue
        ratio = float(np.clip(np.dot(point - start, segment) / (segment_length * segment_length), 0.0, 1.0))
        projected = start + ratio * segment
        distance = float(np.linalg.norm(point - projected))
        if distance < best_distance:
            best_distance = distance
            best_point = projected
            best_arc = accumulated + ratio * segment_length
        accumulated += segment_length
    normalized_s = 0.0 if total_length <= 1e-6 else float(np.clip(best_arc / total_length, 0.0, 1.0))
    return best_point.astype(np.float32), normalized_s, best_distance


def normalize_polyline_dict(polylines: Mapping[str, Optional[np.ndarray]]) -> dict[str, Optional[np.ndarray]]:
    return {name: _as_polyline(polylines.get(name)) for name in TOPOLOGY_NAMES}


def build_topology_mask(polylines: Mapping[str, Optional[np.ndarray]]) -> np.ndarray:
    normalized = normalize_polyline_dict(polylines)
    return np.asarray([normalized[name] is not None for name in TOPOLOGY_NAMES], dtype=bool)


def map_preference_to_target_point(
    topology_choice: int,
    s: float,
    polylines: Mapping[str, Optional[np.ndarray]],
) -> tuple[np.ndarray, dict[str, object]]:
    normalized = normalize_polyline_dict(polylines)
    mask = build_topology_mask(normalized)
    choice = int(topology_choice)
    if choice < 0 or choice >= len(TOPOLOGY_NAMES) or not bool(mask[choice]):
        choice = TOPOLOGY_CURRENT if bool(mask[TOPOLOGY_CURRENT]) else int(np.flatnonzero(mask)[0])
    polyline_name = TOPOLOGY_NAMES[choice]
    polyline = normalized[polyline_name]
    if polyline is None:
        raise ValueError("At least one legal topology polyline is required.")
    clipped_s = float(np.clip(s, 0.0, 1.0))
    point = _interpolate_at_fraction(polyline, clipped_s)
    debug = {
        "co_preference_topology_mask": mask.astype(np.float32),
        "co_preference_choice": choice,
        "co_preference_topology": polyline_name,
        "co_preference_s": clipped_s,
        "co_preference_target_point": point,
        "co_preference_target_valid": True,
    }
    return point, debug


def project_point_to_best_polyline(
    point: np.ndarray,
    polylines: Mapping[str, Optional[np.ndarray]],
) -> tuple[int, float, np.ndarray, float]:
    normalized = normalize_polyline_dict(polylines)
    best_choice = -1
    best_s = 0.0
    best_point = np.zeros(2, dtype=np.float32)
    best_distance = float("inf")
    for idx, name in enumerate(TOPOLOGY_NAMES):
        polyline = normalized[name]
        if polyline is None:
            continue
        projected, s, distance = _project_to_polyline(point, polyline)
        if distance < best_distance:
            best_choice = idx
            best_s = s
            best_point = projected
            best_distance = distance
    if best_choice < 0:
        raise ValueError("Cannot project teacher target point without a legal topology polyline.")
    return best_choice, best_s, best_point, best_distance


def point_to_polyline_distance(point: np.ndarray, polyline: np.ndarray) -> tuple[float, float]:
    normalized = _as_polyline(polyline)
    if normalized is None:
        raise ValueError("selected_polyline must contain at least two non-degenerate points.")
    _, s, distance = _project_to_polyline(point, normalized)
    return distance, s
