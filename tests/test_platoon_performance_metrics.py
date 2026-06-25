from __future__ import annotations

import numpy as np

from evaluation.platoon_performance import (
    build_platoon_metric_params,
    compute_pairwise_formation_reward,
    compute_pdms_reward_batch,
)


def _straight_traj(xs: list[float], y: float = 0.0) -> np.ndarray:
    xs_arr = np.asarray(xs, dtype=np.float32)
    ys_arr = np.full_like(xs_arr, float(y), dtype=np.float32)
    headings = np.zeros_like(xs_arr, dtype=np.float32)
    return np.stack([xs_arr, ys_arr, headings], axis=-1)


def test_build_platoon_metric_params_applies_defaults_and_overrides():
    params = build_platoon_metric_params({"w_progress": 3.0, "target_speed_km_h": 42.0})

    assert params["w_progress"] == 3.0
    assert params["target_speed_kmh"] == 42.0
    assert params["gate_collision_dist_m"] == 1.0
    assert params["desired_gap_m"] == 10.0


def test_pdms_reward_gate_closes_on_env_collision():
    params = build_platoon_metric_params({"w_progress": 1.0, "w_speed": 1.0, "w_anchor": 0.0})
    traj = _straight_traj([0.0, 3.0, 6.0, 9.0])[None]

    rewards, debug = compute_pdms_reward_batch(
        traj,
        follower_pose=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        prev_traj=None,
        prev_pose=None,
        selected_traj=traj[0],
        is_leader=True,
        formation_lon_scores=np.ones(1, dtype=np.float32),
        formation_lat_scores=np.ones(1, dtype=np.float32),
        params=params,
        env_crashed=True,
    )

    assert rewards.tolist() == [0.0]
    assert debug["collision_gate"] == 0.0
    assert debug["gate"][0] == 0.0


def test_leader_progress_component_increases_with_forward_distance():
    params = build_platoon_metric_params({"w_progress": 1.0, "w_speed": 0.0, "w_anchor": 0.0, "w_comfort": 0.0})
    short = _straight_traj([0.0, 1.0, 2.0, 3.0])
    long = _straight_traj([0.0, 4.0, 8.0, 12.0])
    batch = np.stack([short, long], axis=0)

    _, debug = compute_pdms_reward_batch(
        batch,
        follower_pose=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        prev_traj=None,
        prev_pose=None,
        selected_traj=short,
        is_leader=True,
        formation_lon_scores=np.ones(2, dtype=np.float32),
        formation_lat_scores=np.ones(2, dtype=np.float32),
        params=params,
    )

    assert debug["progress"][1] > debug["progress"][0]


def test_pairwise_formation_reward_prefers_desired_gap():
    leader_pose = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    follower_pose = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    leader = _straight_traj([10.0, 12.0, 14.0, 16.0])
    ideal = _straight_traj([-10.0, -8.0, -6.0, -4.0])
    too_close = _straight_traj([8.0, 10.0, 12.0, 14.0])

    lon_scores, lat_scores = compute_pairwise_formation_reward(
        np.stack([ideal, too_close], axis=0),
        follower_pose,
        leader,
        leader_pose,
        desired_gap_m=10.0,
    )

    assert lon_scores[0] > lon_scores[1]
    assert lat_scores[0] == lat_scores[1]


def test_pdms_debug_contains_gate_quality_and_component_arrays():
    params = build_platoon_metric_params({})
    traj = _straight_traj([0.0, 2.0, 4.0, 6.0])[None]

    rewards, debug = compute_pdms_reward_batch(
        traj,
        follower_pose=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        prev_traj=None,
        prev_pose=None,
        selected_traj=traj[0],
        is_leader=True,
        formation_lon_scores=np.ones(1, dtype=np.float32),
        formation_lat_scores=np.ones(1, dtype=np.float32),
        params=params,
    )

    for key in (
        "collision_gate",
        "road_gate",
        "smoothness_gate",
        "plan_road_gate",
        "plan_collision_gate",
        "gate",
        "progress",
        "formation_lon",
        "formation_lat",
        "speed",
        "anchor",
        "comfort",
        "consistency",
        "preference",
        "quality",
        "reward",
    ):
        assert key in debug
    assert debug["reward"][0] == rewards[0]
