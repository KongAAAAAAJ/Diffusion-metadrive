from __future__ import annotations

import numpy as np


def pairwise_agent_safety_score(
    traj_a,
    traj_b,
    *,
    safe_distance_m: float = 7.0,
    repulsion_weight: float = 4.0,
    attraction_weight: float = 0.8,
    soft_distance_ratio: float = 0.5,
    longitudinal_soft_ratio: float = 0.7,
    lateral_soft_ratio: float = 0.35,
) -> float:
    """Soft pairwise safety potential for two agent trajectories.

    Returns a finite, continuous score:
    - nearby trajectories get a stronger negative repulsion
    - well-separated trajectories get a small positive clearance reward
    """
    traj_a = np.asarray(traj_a, dtype=np.float32)
    traj_b = np.asarray(traj_b, dtype=np.float32)
    if traj_a.ndim != 2 or traj_b.ndim != 2 or traj_a.shape[0] == 0 or traj_b.shape[0] == 0:
        return 0.0
    min_len = min(traj_a.shape[0], traj_b.shape[0])
    if min_len <= 0:
        return 0.0

    safe_distance = max(1e-3, float(safe_distance_m))
    soft_distance = max(1e-3, safe_distance * float(soft_distance_ratio))
    longitudinal_soft = max(1e-3, safe_distance * float(longitudinal_soft_ratio))
    lateral_soft = max(1e-3, safe_distance * float(lateral_soft_ratio))

    ref = traj_a[:min_len, :2]
    other = traj_b[:min_len, :2]
    deltas = other - ref
    headings = _reference_tangents(ref)
    normals = np.stack([-headings[:, 1], headings[:, 0]], axis=1)
    lon = np.sum(deltas * headings, axis=1)
    lat = np.sum(deltas * normals, axis=1)

    anisotropic_dist = np.sqrt((lon / longitudinal_soft) ** 2 + (lat / lateral_soft) ** 2)
    min_dist = float(np.min(anisotropic_dist))
    mean_dist = float(np.mean(anisotropic_dist))

    repulsion = float(repulsion_weight) * float(np.exp(-0.5 * min_dist**2))
    clearance = float(attraction_weight) * float(np.tanh(max(0.0, mean_dist - 1.0) * soft_distance / safe_distance))
    return float(clearance - repulsion)


def _reference_tangents(points_xy: np.ndarray) -> np.ndarray:
    if points_xy.shape[0] <= 1:
        return np.asarray([[1.0, 0.0]], dtype=np.float32)
    diffs = np.zeros_like(points_xy, dtype=np.float32)
    diffs[:-1] = points_xy[1:] - points_xy[:-1]
    diffs[-1] = diffs[-2]
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    safe = np.where(norms > 1e-6, norms, 1.0)
    tangents = diffs / safe
    zero_mask = (norms[:, 0] <= 1e-6)
    tangents[zero_mask] = np.asarray([1.0, 0.0], dtype=np.float32)
    return tangents.astype(np.float32, copy=False)
