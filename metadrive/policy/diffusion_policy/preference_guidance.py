from __future__ import annotations

from typing import Mapping

import numpy as np


def _point_or_none(value) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size < 2:
        return None
    return array[:2].astype(np.float32, copy=True)


def _vehicle_position_xy(vehicle) -> np.ndarray | None:
    position = getattr(vehicle, "position", None)
    if position is None:
        return None
    array = np.asarray(position, dtype=np.float32).reshape(-1)
    if array.size < 2:
        return None
    return array[:2]


def _world_xy_to_vehicle_local(vehicle, world_xy: np.ndarray) -> np.ndarray | None:
    position = _vehicle_position_xy(vehicle)
    if position is None:
        return None
    heading = float(getattr(vehicle, "heading_theta", 0.0))
    delta = np.asarray(world_xy, dtype=np.float32).reshape(-1)[:2] - position[:2]
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    return np.asarray(
        [cos_h * float(delta[0]) + sin_h * float(delta[1]), -sin_h * float(delta[0]) + cos_h * float(delta[1])],
        dtype=np.float32,
    )


def _fallback_keep_lane_endpoint(coarse_by_agent: Mapping[str, np.ndarray], agent_id: str) -> np.ndarray | None:
    coarse = coarse_by_agent.get(agent_id)
    if coarse is None:
        return None
    coarse_array = np.asarray(coarse, dtype=np.float32)
    if coarse_array.ndim != 3 or coarse_array.shape[0] == 0 or coarse_array.shape[1] == 0:
        return None
    return coarse_array[0, -1, :2].astype(np.float32, copy=True)


def inject_external_preference_point(
    planner_batch: dict[str, dict],
    coarse_by_agent: Mapping[str, np.ndarray],
    agent_ids: list[str],
    *,
    env=None,
    target_speed_km_h: float = 30.0,
    horizon_s: float = 4.0,
) -> dict[str, dict]:
    """Inject planner-before-export semantic ``preference_point`` for each agent.

    The first implementation treats the preference point as an external
    lane-following decision prior: every controlled car projects itself onto the
    platoon head car's current lane, advances by ``target_speed * horizon``, and
    converts the resulting world point into that car's ego-local frame.

    If lane/vehicle state is unavailable, fallback to the keep-lane coarse
    endpoint. That fallback still happens before planner export and is not tied
    to the selected mode.
    """
    delta_s = float(target_speed_km_h) / 3.6 * float(horizon_s)
    agents = getattr(env, "agents", {}) if env is not None else {}
    leader_lane = None
    if agent_ids:
        leader_vehicle = agents.get(agent_ids[0])
        if leader_vehicle is not None:
            leader_lane = getattr(leader_vehicle, "lane", None)

    metadata: dict[str, dict] = {}
    for agent_id in agent_ids:
        pp = None
        source = None
        vehicle = agents.get(agent_id)
        if vehicle is not None and leader_lane is not None:
            try:
                position = _vehicle_position_xy(vehicle)
                if position is not None:
                    s_ego, _ = leader_lane.local_coordinates(position)
                    s_target = min(float(s_ego) + delta_s, float(getattr(leader_lane, "length", s_ego + delta_s)))
                    target_world = np.asarray(leader_lane.position(s_target, 0.0), dtype=np.float32)
                    pp = _world_xy_to_vehicle_local(vehicle, target_world)
                    source = "leader_lane"
            except Exception:
                pp = None
                source = None

        if pp is None:
            pp = _fallback_keep_lane_endpoint(coarse_by_agent, agent_id)
            if pp is not None:
                source = "fallback_keep_lane_coarse"

        pp = _point_or_none(pp)
        if pp is None or agent_id not in planner_batch:
            continue
        planner_batch[agent_id]["preference_point"] = pp
        metadata[agent_id] = {
            "preference_point": pp.astype(float).tolist(),
            "preference_source": source or "unknown",
        }
    return metadata
