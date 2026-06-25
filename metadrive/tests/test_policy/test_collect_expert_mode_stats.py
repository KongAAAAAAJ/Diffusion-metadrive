from __future__ import annotations

from expert_dataset.collect_expert import build_trajectory_mode_summary, trajectory_mode_name


def test_trajectory_mode_name_matches_enum_encoding():
    assert trajectory_mode_name(0) == "keep_lane"
    assert trajectory_mode_name(4) == "overtake"


def test_build_trajectory_mode_summary_returns_counts_and_ratios():
    summary = build_trajectory_mode_summary(
        {
            "keep_lane": 6,
            "follow": 2,
            "lane_change_left": 1,
            "lane_change_right": 1,
            "overtake": 0,
            "merge": 0,
        },
        total_samples=10,
    )

    assert summary["keep_lane"]["count"] == 6
    assert summary["keep_lane"]["ratio"] == 0.6
    assert summary["follow"]["count"] == 2
    assert summary["follow"]["ratio"] == 0.2
