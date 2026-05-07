import numpy as np
import pytest

from train.single_step_test import (
    _apply_selected_endpoint_target_policy,
    parse_manual_modes,
    validate_manual_modes,
)


def test_parse_manual_modes_expands_single_mode_to_all_agents():
    assert parse_manual_modes("3", num_agents=3) == [3, 3, 3]


def test_parse_manual_modes_accepts_one_mode_per_agent():
    assert parse_manual_modes("0,4,10", num_agents=3) == [0, 4, 10]


def test_parse_manual_modes_rejects_wrong_length():
    with pytest.raises(ValueError, match="one value or exactly 3 values"):
        parse_manual_modes("0,1", num_agents=3)


def test_validate_manual_modes_rejects_invalid_slot():
    mask = np.asarray([[True, False, True]], dtype=bool)
    with pytest.raises(ValueError, match="invalid manual mode"):
        validate_manual_modes(["agent0"], [1], mask)


def test_apply_selected_endpoint_target_policy_sets_target_and_preference_points():
    planner_batch = {
        "agent0": {"target_point": np.asarray([99.0, 99.0], dtype=np.float32)},
        "agent1": {},
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

    metadata = _apply_selected_endpoint_target_policy(
        planner_batch,
        agent_ids=["agent0", "agent1"],
        manual_modes=[1, 0],
        coarse_by_agent=coarse_by_agent,
    )

    np.testing.assert_allclose(planner_batch["agent0"]["target_point"], [2.0, 2.0])
    np.testing.assert_allclose(planner_batch["agent0"]["preference_point"], [2.0, 2.0])
    np.testing.assert_allclose(planner_batch["agent1"]["target_point"], [3.0, 3.0])
    assert metadata["agent0"]["target_point_before"] == [99.0, 99.0]
    assert metadata["agent0"]["target_point_after"] == [2.0, 2.0]
    assert metadata["agent0"]["selected_coarse_endpoint"] == [2.0, 2.0]
