from __future__ import annotations

import math

from train.train_mode_cls_sb3 import compute_episode_reward_per_step_mean


def test_compute_episode_reward_per_step_mean_averages_episode_reward_by_length():
    value = compute_episode_reward_per_step_mean([
        {"r": 10.0, "l": 5},
        {"r": 3.0, "l": 3},
    ])

    assert value == 1.5


def test_compute_episode_reward_per_step_mean_returns_none_for_empty_buffer():
    assert compute_episode_reward_per_step_mean([]) is None


def test_compute_episode_reward_per_step_mean_skips_zero_length_entries():
    value = compute_episode_reward_per_step_mean([
        {"r": 100.0, "l": 0},
        {"r": 4.0, "l": 2},
    ])

    assert value == 2.0


def test_compute_episode_reward_per_step_mean_returns_none_when_all_entries_invalid():
    value = compute_episode_reward_per_step_mean([
        {"r": 100.0, "l": 0},
        {"r": math.nan, "l": 2},
    ])

    assert value is None
