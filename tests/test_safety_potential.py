from __future__ import annotations

import numpy as np

from models.decision.safety_potential import pairwise_agent_safety_score


def test_pairwise_agent_safety_score_is_soft_and_finite_for_close_trajectories():
    traj_a = np.asarray([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    traj_b = np.asarray([[0.5, 0.0], [5.5, 0.0], [10.5, 0.0]], dtype=np.float32)

    score = pairwise_agent_safety_score(traj_a, traj_b, safe_distance_m=7.0)

    assert np.isfinite(score)
    assert score < 0.0
    assert score > -10.0


def test_pairwise_agent_safety_score_prefers_more_separated_trajectories():
    traj_a = np.asarray([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    near_traj_b = np.asarray([[1.0, 0.0], [6.0, 0.0], [11.0, 0.0]], dtype=np.float32)
    far_traj_b = np.asarray([[0.0, 10.0], [5.0, 10.0], [10.0, 10.0]], dtype=np.float32)

    near_score = pairwise_agent_safety_score(traj_a, near_traj_b, safe_distance_m=7.0)
    far_score = pairwise_agent_safety_score(traj_a, far_traj_b, safe_distance_m=7.0)

    assert far_score > near_score


def test_pairwise_agent_safety_score_penalizes_longitudinal_close_following_more_than_adjacent_lane_parallel():
    traj_a = np.asarray([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    longitudinal_close = np.asarray([[2.0, 0.0], [7.0, 0.0], [12.0, 0.0]], dtype=np.float32)
    lateral_parallel = np.asarray([[0.0, 2.0], [5.0, 2.0], [10.0, 2.0]], dtype=np.float32)

    longitudinal_score = pairwise_agent_safety_score(traj_a, longitudinal_close, safe_distance_m=7.0)
    lateral_score = pairwise_agent_safety_score(traj_a, lateral_parallel, safe_distance_m=7.0)

    assert lateral_score > longitudinal_score
