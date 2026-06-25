from __future__ import annotations

from typing import Mapping

import numpy as np

from models.diffusion.transfuser_features import _build_target_line_from_polyline


def _point_or_none(value) -> list[float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size < 2:
        return None
    return [float(array[0]), float(array[1])]


def build_selected_coarse_guidance(coarse_trajectory, config) -> dict[str, np.ndarray]:
    """Build planner guidance from one selected dynamic coarse trajectory.

    All returned coordinates are ego-local. ``topology_polyline`` is kept for
    visualization/debugging; the model consumes target/preference/target_line.
    """
    coarse = np.asarray(coarse_trajectory, dtype=np.float32)
    if coarse.ndim != 2 or coarse.shape[0] == 0 or coarse.shape[1] < 2:
        raise ValueError(f"coarse_trajectory must have shape [T,2+], got {coarse.shape}")
    coarse_xy = coarse[:, :2].astype(np.float32, copy=False)
    target_point = coarse_xy[-1].copy()
    topology_polyline = np.concatenate(
        [np.zeros((1, 2), dtype=np.float32), coarse_xy],
        axis=0,
    )
    target_line = _build_target_line_from_polyline(topology_polyline, target_point, config)
    return {
        "target_point": target_point,
        "preference_point": target_point.copy(),
        "topology_polyline": topology_polyline.astype(np.float32, copy=False),
        "target_line": np.asarray(target_line, dtype=np.float32),
    }


def apply_selected_mode_guidance(
    planner_batch: dict[str, dict],
    *,
    agent_ids: list[str],
    selected_modes: list[int],
    coarse_by_agent: Mapping[str, np.ndarray],
    config,
) -> dict[str, dict]:
    """Write selected-mode coarse guidance into each agent sample."""
    metadata: dict[str, dict] = {}
    for agent_id, mode_idx in zip(agent_ids, selected_modes):
        if agent_id not in planner_batch:
            continue
        if agent_id not in coarse_by_agent:
            continue
        coarse = np.asarray(coarse_by_agent[agent_id], dtype=np.float32)
        if coarse.ndim != 3 or not (0 <= int(mode_idx) < coarse.shape[0]):
            raise ValueError(f"Invalid selected mode {mode_idx} for {agent_id} coarse shape {coarse.shape}")
        before = _point_or_none(planner_batch[agent_id].get("target_point"))
        guidance = build_selected_coarse_guidance(coarse[int(mode_idx)], config)
        planner_batch[agent_id].update(guidance)
        metadata[agent_id] = {
            "target_point_before": before,
            "target_point_after": guidance["target_point"].astype(float).tolist(),
            "preference_point_after": guidance["preference_point"].astype(float).tolist(),
            "selected_coarse_endpoint": guidance["target_point"].astype(float).tolist(),
            "target_line_after": guidance["target_line"].astype(float).tolist(),
            "topology_polyline_after": guidance["topology_polyline"].astype(float).tolist(),
        }
    return metadata
