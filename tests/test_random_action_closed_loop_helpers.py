import numpy as np
from types import SimpleNamespace

from metadrive.policy.diffusion_policy.test_transfuser_policy import (
    _apply_selected_mode_target_overrides,
    _choose_mode_indices,
    _episode_reset_seed,
    _format_step_reward_lines,
    _summarize_random_action_rewards,
    build_platoon_env_config,
)


def test_choose_mode_indices_samples_only_valid_modes_reproducibly():
    masks = np.asarray(
        [
            [False, True, False, True],
            [True, False, False, False],
        ],
        dtype=bool,
    )

    first = _choose_mode_indices(
        masked_logits=np.zeros((2, 4), dtype=np.float32),
        mode_valid_mask=masks,
        policy="random_valid",
        rng=np.random.RandomState(7),
    )
    second = _choose_mode_indices(
        masked_logits=np.zeros((2, 4), dtype=np.float32),
        mode_valid_mask=masks,
        policy="random_valid",
        rng=np.random.RandomState(7),
    )

    assert first == second
    assert masks[0, first[0]]
    assert masks[1, first[1]]


def test_episode_reset_seed_cycles_inside_scenario_range():
    assert [_episode_reset_seed(7, 3, idx) for idx in range(5)] == [7, 8, 9, 7, 8]
    assert [_episode_reset_seed(7, 0, idx) for idx in range(3)] == [7, 7, 7]


def test_build_platoon_env_config_disables_random_traffic_by_default():
    args = SimpleNamespace(
        num_agents=3,
        render=0,
        start_seed=7,
        num_scenarios=1,
        traffic_density=0.04,
        random_traffic=0,
        image_on_cuda=0,
    )

    env_config = build_platoon_env_config(args)

    assert env_config["random_traffic"] is False


def test_apply_selected_mode_target_overrides_uses_selected_coarse_endpoints():
    planner_batch = {
        "agent0": {},
        "agent1": {"target_point": np.asarray([99.0, 99.0], dtype=np.float32)},
    }
    coarse_by_agent = {
        "agent0": np.asarray(
            [
                [[0.0, 0.0], [1.0, 1.0]],
                [[0.0, 0.0], [2.0, 2.0]],
            ],
            dtype=np.float32,
        ),
        "agent1": np.asarray(
            [
                [[0.0, 0.0], [3.0, 3.0]],
                [[0.0, 0.0], [4.0, 4.0]],
            ],
            dtype=np.float32,
        ),
    }

    metadata = _apply_selected_mode_target_overrides(
        planner_batch,
        agent_ids=["agent0", "agent1"],
        selected_modes=[1, 0],
        coarse_by_agent=coarse_by_agent,
    )

    assert np.allclose(planner_batch["agent0"]["target_point"], [2.0, 2.0])
    assert np.allclose(planner_batch["agent0"]["preference_point"], [2.0, 2.0])
    assert np.allclose(planner_batch["agent1"]["target_point"], [3.0, 3.0])
    assert metadata["agent1"]["target_point_before"] == [99.0, 99.0]
    assert metadata["agent1"]["selected_coarse_endpoint"] == [3.0, 3.0]


def test_summarize_random_action_rewards_groups_by_mode():
    records = [
        {"selected_mode": {"agent0": 1}, "env_reward": {"agent0": 2.0}},
        {"selected_mode": {"agent0": 1}, "env_reward": {"agent0": 4.0}},
        {"selected_mode": {"agent0": 2}, "env_reward": {"agent0": -1.0}},
    ]

    summary = _summarize_random_action_rewards(
        records,
        episodes=1,
        success=0,
        crash=1,
        out_of_road=0,
        metadata={"selection_policy": "random_valid"},
    )

    assert summary["metadata"]["selection_policy"] == "random_valid"
    assert summary["per_mode"]["1"]["selected_count"] == 2
    assert summary["per_mode"]["1"]["env_reward_mean"] == 3.0
    assert "selector_reward_mean" not in summary["per_mode"]["1"]
    assert summary["per_mode"]["2"]["selected_count"] == 1
    assert summary["crash_rate"] == 1.0


def test_format_step_reward_lines_includes_only_env_rewards():
    lines = _format_step_reward_lines(
        [
            ("EGO1", 1.25),
            ("EGO2", -0.75),
        ]
    )

    assert lines == ["env reward: EGO1=+1.25 EGO2=-0.75"]
