from __future__ import annotations

import numpy as np


def build_platoon_metric_params(config: dict) -> dict:
    """Build platoon performance metric parameters from a config mapping."""
    config = config or {}
    return {
        "gate_collision_dist_m": float(config.get("gate_collision_dist_m", 1.0)),
        "gate_ttc_s": float(config.get("gate_ttc_s", 0.5)),
        "gate_road_half_width_m": float(config.get("gate_road_half_width_m", 3.0)),
        "gate_max_dh_rad": float(config.get("gate_max_dh_rad", 0.6)),
        "gate_horizon_steps": int(config.get("gate_horizon_steps", 2)),
        "vehicle_length_m": float(config.get("vehicle_length_m", 4.8)),
        "w_progress": float(config.get("w_progress", 0.2)),
        "w_formation_lon": float(config.get("w_formation_lon", 0.2)),
        "w_formation_lat": float(config.get("w_formation_lat", 0.2)),
        "w_speed": float(config.get("w_speed", 0.2)),
        "w_anchor": float(config.get("w_anchor", 0.15)),
        "w_comfort": float(config.get("w_comfort", 0.05)),
        "w_consistency": float(config.get("w_consistency", 0.0)),
        "progress_s_max": float(config.get("progress_s_max", 15.0)),
        "target_speed_kmh": float(config.get("target_speed_km_h", config.get("target_speed_kmh", 30.0))),
        "anchor_decay_m": float(config.get("anchor_decay_m", 1.0)),
        "anchor_power": float(config.get("anchor_power", 2.0)),
        "comfort_decay_rad": float(config.get("comfort_decay_rad", 0.1)),
        "consistency_decay_m": float(config.get("consistency_decay_m", 1.0)),
        "desired_gap_m": float(config.get("desired_gap_m", 10.0)),
        "lon_decay_m": float(config.get("lon_decay_m", 5.0)),
        "lat_decay_m": float(config.get("lat_decay_m", 0.5)),
        "w_preference": float(config.get("w_preference", 0.0)),
        "preference_decay_m": float(config.get("preference_decay_m", 5.0)),
        "waypoint_decay_gamma": float(config.get("waypoint_decay_gamma", 0.9)),
    }


build_pdms_params = build_platoon_metric_params


def _traj_speed_kmh(traj_local: np.ndarray, dt: float = 0.5) -> float:
    """Estimate average speed (km/h) from an ego-local trajectory."""
    traj_local = np.asarray(traj_local, dtype=np.float32)
    if traj_local.shape[0] < 2:
        return 0.0
    total_m = float(np.linalg.norm(np.diff(traj_local[:, :2], axis=0), axis=1).sum())
    return total_m / max((traj_local.shape[0] - 1) * dt, 1e-6) * 3.6


def _local_xy_to_world_xy(pose: np.ndarray, local_xy: np.ndarray) -> np.ndarray:
    pose_arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    pts = np.asarray(local_xy, dtype=np.float64)
    cos_h = float(np.cos(pose_arr[2]))
    sin_h = float(np.sin(pose_arr[2]))
    world = np.empty_like(pts, dtype=np.float64)
    world[..., 0] = pose_arr[0] + cos_h * pts[..., 0] - sin_h * pts[..., 1]
    world[..., 1] = pose_arr[1] + sin_h * pts[..., 0] + cos_h * pts[..., 1]
    return world


def compute_pairwise_formation_reward(
    follower_candidates: np.ndarray,
    follower_pose: np.ndarray,
    leader_traj: np.ndarray,
    leader_pose: np.ndarray,
    desired_gap_m: float = 10.0,
    lon_decay_m: float = 5.0,
    lat_decay_m: float = 0.5,
    progress_s_max: float = 15.0,  # kept for API compatibility
    same_lane: bool = True,
    waypoint_decay_gamma: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise formation score split into longitudinal and lateral components."""
    del progress_s_max
    follower_candidates = np.asarray(follower_candidates, dtype=np.float32)
    M = follower_candidates.shape[0]
    if not same_lane:
        zeros = np.zeros(M, dtype=np.float32)
        return zeros, zeros.copy()

    T = follower_candidates.shape[1]
    wp_weights = float(waypoint_decay_gamma) ** np.arange(T, dtype=np.float32)
    wp_weights /= max(float(wp_weights.sum()), 1e-6)

    leader_traj = np.asarray(leader_traj, dtype=np.float32)
    leader_world_xy = _local_xy_to_world_xy(leader_pose, leader_traj[:, :2])
    leader_headings = (
        leader_traj[:, 2] if leader_traj.shape[1] > 2 else np.zeros(T, dtype=np.float32)
    )

    cos_h = np.cos(leader_headings).astype(np.float32)
    sin_h = np.sin(leader_headings).astype(np.float32)
    desired_world_xy = np.stack(
        [
            leader_world_xy[:, 0] - float(desired_gap_m) * cos_h,
            leader_world_xy[:, 1] - float(desired_gap_m) * sin_h,
        ],
        axis=-1,
    ).astype(np.float32)

    follower_world_xy = _local_xy_to_world_xy(follower_pose, follower_candidates[:, :, :2]).astype(np.float32)
    dx = follower_world_xy[:, :, 0] - desired_world_xy[np.newaxis, :, 0]
    dy = follower_world_xy[:, :, 1] - desired_world_xy[np.newaxis, :, 1]

    lon_errs = np.abs(dx * cos_h + dy * sin_h)
    lat_errs = np.abs(-dx * sin_h + dy * cos_h)
    r_lon = (
        np.exp(-np.abs(lon_errs - float(desired_gap_m)) / max(float(lon_decay_m), 1e-6))
        @ wp_weights
    ).astype(np.float32)
    r_lat = (np.exp(-lat_errs / max(float(lat_decay_m), 1e-6)) @ wp_weights).astype(np.float32)
    return r_lon, r_lat


def compute_pdms_reward_batch(
    refined_np: np.ndarray,
    follower_pose: np.ndarray | None,
    prev_traj: np.ndarray | None,
    prev_pose: np.ndarray | None,
    selected_traj: np.ndarray,
    is_leader: bool,
    formation_lon_scores: np.ndarray,
    formation_lat_scores: np.ndarray,
    params: dict,
    road_half_widths: np.ndarray | None = None,
    anchor_trajs: np.ndarray | None = None,
    env_crashed: bool | None = None,
    env_out_of_road: bool | None = None,
    preference_point: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Gate x quality reward for a batch of platoon candidate trajectories."""
    refined_np = np.asarray(refined_np, dtype=np.float32)
    params = params or {}
    GxM, T, _ = refined_np.shape

    collision_gate_env = 0.0 if env_crashed else 1.0
    road_gate_env = 0.0 if env_out_of_road else 1.0
    shared_gate_scalar = np.float32(collision_gate_env * road_gate_env)

    speed_target_kmh = (
        params["target_speed_kmh"]
        if is_leader
        else max(_traj_speed_kmh(prev_traj) if prev_traj is not None else params["target_speed_kmh"], 1e-3)
    )

    gamma = float(params.get("waypoint_decay_gamma", 0.9))
    wp_weights = gamma ** np.arange(T, dtype=np.float32)
    wp_weights /= max(float(wp_weights.sum()), 1e-6)
    step_weights = gamma ** np.arange(T - 1, dtype=np.float32)
    step_weights /= max(float(step_weights.sum()), 1e-6)

    dh_all = np.abs(np.diff(refined_np[:, :, 2], axis=1))
    dh_all = np.minimum(dh_all, 2 * np.pi - dh_all)
    smoothness_gate_vec = (dh_all.max(axis=1) <= params["gate_max_dh_rad"]).astype(np.float32)

    n_plan = min(6, T)
    half_w = (
        np.asarray(road_half_widths, dtype=np.float32)
        if road_half_widths is not None
        else np.full(GxM, params["gate_road_half_width_m"], dtype=np.float32)
    )
    plan_road_gate_vec = (np.abs(refined_np[:, :n_plan, 1]).max(axis=1) <= half_w).astype(np.float32)

    if prev_traj is not None and prev_pose is not None and follower_pose is not None:
        k = min(n_plan, T)
        leader_world = _local_xy_to_world_xy(prev_pose, np.asarray(prev_traj, dtype=np.float32)[:k, :2])
        follower_world = _local_xy_to_world_xy(follower_pose, refined_np[:, :k, :2])
        dist = np.linalg.norm(follower_world - leader_world[np.newaxis], axis=-1)
        bumper = dist - float(params["vehicle_length_m"])
        plan_collision_gate_vec = (bumper.min(axis=1) >= float(params["gate_collision_dist_m"])).astype(np.float32)
    else:
        plan_collision_gate_vec = np.ones(GxM, dtype=np.float32)

    gate_vec = shared_gate_scalar * smoothness_gate_vec * plan_road_gate_vec * plan_collision_gate_vec

    if is_leader:
        progress_vec = np.clip(refined_np[:, :, 0] / max(params["progress_s_max"], 1e-3), 0.0, 1.0) @ wp_weights
    else:
        progress_vec = np.zeros(GxM, dtype=np.float32)

    step_dist = np.linalg.norm(np.diff(refined_np[:, :, :2], axis=1), axis=-1)
    speed_vec = np.clip(step_dist / 0.5 * 3.6 / speed_target_kmh, 0.0, 1.0) @ step_weights

    if anchor_trajs is not None:
        anc = np.asarray(anchor_trajs, dtype=np.float32).reshape(GxM, T, 2)
        anc_dist = np.linalg.norm(refined_np[:, :, :2] - anc, axis=-1)
    else:
        anc_dist = np.zeros((GxM, T), dtype=np.float32)
    anchor_err_vec = anc_dist @ wp_weights
    anchor_decay = max(float(params["anchor_decay_m"]), 1e-6)
    anchor_power = max(float(params.get("anchor_power", 2.0)), 1e-6)
    anchor_vec = np.exp(-(anchor_err_vec / anchor_decay) ** anchor_power)

    comfort_vec = np.exp(-dh_all / max(float(params["comfort_decay_rad"]), 1e-6)) @ step_weights

    selected_traj = np.asarray(selected_traj, dtype=np.float32)
    diff_xy = refined_np[:, :, :2] - selected_traj[np.newaxis, :, :2]
    consistency_vec = (
        np.exp(-np.linalg.norm(diff_xy, axis=-1) / max(float(params["consistency_decay_m"]), 1e-6))
        @ wp_weights
    )

    formation_lon_vec = (
        np.zeros(GxM, dtype=np.float32)
        if is_leader
        else np.asarray(formation_lon_scores, dtype=np.float32)
    )
    formation_lat_vec = (
        np.zeros(GxM, dtype=np.float32)
        if is_leader
        else np.asarray(formation_lat_scores, dtype=np.float32)
    )

    if is_leader and preference_point is not None:
        pref_pt = np.asarray(preference_point, dtype=np.float32).reshape(2)
        pref_dist = np.linalg.norm(refined_np[:, :, :2] - pref_pt[np.newaxis, np.newaxis, :], axis=-1)
        pref_decay = max(float(params.get("preference_decay_m", 5.0)), 1e-6)
        preference_vec = np.exp(-pref_dist / pref_decay) @ wp_weights
    else:
        preference_vec = np.zeros(GxM, dtype=np.float32)

    quality_vec = (
        params["w_progress"] * progress_vec
        + params["w_formation_lon"] * formation_lon_vec
        + params["w_formation_lat"] * formation_lat_vec
        + params["w_speed"] * speed_vec
        + params["w_anchor"] * anchor_vec
        + params["w_comfort"] * comfort_vec
        + params["w_consistency"] * consistency_vec
        + params["w_preference"] * preference_vec
    )
    rewards = (gate_vec * quality_vec).astype(np.float32)

    batch_debug = {
        "collision_gate": collision_gate_env,
        "road_gate": road_gate_env,
        "smoothness_gate": smoothness_gate_vec,
        "plan_road_gate": plan_road_gate_vec,
        "plan_collision_gate": plan_collision_gate_vec,
        "gate": gate_vec,
        "progress": progress_vec,
        "formation_lon": formation_lon_vec,
        "formation_lat": formation_lat_vec,
        "speed": speed_vec,
        "anchor": anchor_vec,
        "comfort": comfort_vec,
        "consistency": consistency_vec,
        "preference": preference_vec,
        "quality": quality_vec,
        "reward": rewards,
    }
    return rewards, batch_debug
