from __future__ import annotations

import numpy as np
import torch

from evaluation.reward_terms import compute_team_reward
from train.joint_group import build_joint_groups, extract_joint_trajectories, select_top_k_candidates


def test_select_top_k_candidates_sorts_by_reward_and_skips_crash():
    reward = torch.tensor([[1.0, 9.0], [8.0, 7.0]])
    crash = torch.tensor([[False, True], [False, False]])

    candidates = select_top_k_candidates(reward, crash, top_k=2)

    assert candidates == [(1, 0), (1, 1)]


def test_select_top_k_candidates_falls_back_when_all_candidates_crash():
    reward = torch.tensor([[1.0, 9.0], [8.0, 7.0]])
    crash = torch.ones_like(reward, dtype=torch.bool)

    candidates = select_top_k_candidates(reward, crash, top_k=2)

    assert len(candidates) == 2
    assert candidates == [(0, 1), (1, 0)]


def test_build_joint_groups_contains_all_agents_and_respects_limit():
    per_agent_candidates = {
        "agent0": [(0, 0), (1, 0)],
        "agent1": [(0, 1), (1, 1)],
    }

    groups = build_joint_groups(per_agent_candidates, num_groups=3, seed=7)

    assert len(groups) <= 3
    assert groups
    assert all(set(group.keys()) == {"agent0", "agent1"} for group in groups)


def test_extract_joint_trajectories_returns_numpy_trajectories():
    rollouts = {
        "agent0": {"trajectory_all": torch.arange(2 * 2 * 8 * 3, dtype=torch.float32).view(2, 2, 8, 3)},
        "agent1": {"trajectory_all": torch.ones(2, 2, 8, 3)},
    }
    group = {"agent0": (1, 0), "agent1": (0, 1)}

    trajectories = extract_joint_trajectories(group, rollouts)

    assert set(trajectories.keys()) == {"agent0", "agent1"}
    assert isinstance(trajectories["agent0"], np.ndarray)
    assert trajectories["agent0"].shape == (8, 3)


def test_compute_team_reward_penalizes_crash():
    safe_infos = {
        "agent0": [{"progress": 2.0, "formation_error": 1.0, "crash": False} for _ in range(3)],
        "agent1": [{"progress": 2.5, "formation_error": 1.5, "crash": False} for _ in range(3)],
    }
    crash_infos = {
        "agent0": [{"progress": 2.0, "formation_error": 1.0, "crash": False} for _ in range(3)],
        "agent1": [{"progress": 0.5, "formation_error": 3.0, "crash": True} for _ in range(3)],
    }

    safe_reward = compute_team_reward(safe_infos, {"delta_s_max": 5.0, "d_norm": 10.0})
    crash_reward = compute_team_reward(crash_infos, {"delta_s_max": 5.0, "d_norm": 10.0})

    assert isinstance(safe_reward, float)
    assert crash_reward < 0.0
    assert safe_reward > crash_reward
