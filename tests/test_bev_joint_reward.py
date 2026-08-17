from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from models.bev_planner import (
    JointRewardConfig,
    JointRewardError,
    JointTrajectoryProxyReward,
    calibrate_joint_rewards,
    compose_joint_reward,
)


def _trajectory(speed_mps: float, lateral_m: float = 0.0) -> np.ndarray:
    times = np.arange(1, 9, dtype=np.float32) * 0.5
    value = np.zeros((8, 3), dtype=np.float32)
    value[:, 0] = float(speed_mps) * times
    value[:, 1] = np.linspace(0.0, lateral_m, 8, dtype=np.float32)
    return value


def _model_inputs() -> SimpleNamespace:
    relation = np.zeros((3, 12), dtype=np.float32)
    relation[0, 4] = -15.74
    relation[0, 10] = -31.48
    relation[1, 4] = 15.74
    relation[1, 10] = -15.74
    relation[2, 4] = 31.48
    relation[2, 10] = 15.74
    bev = np.zeros((3, 8, 256, 256), dtype=np.uint8)
    bev[:, 0] = 255
    return SimpleNamespace(bev=bev, formation_relation_state=relation)


def _env() -> SimpleNamespace:
    agents = {}
    for role, x in enumerate((0.0, -15.74, -31.48)):
        agents[f"agent{role}"] = SimpleNamespace(
            name=f"agent{role}",
            position=np.asarray([x, 0.0], dtype=np.float32),
            heading_theta=0.0,
            velocity=np.asarray([5.0, 0.0], dtype=np.float32),
            LENGTH=5.74,
            WIDTH=2.3,
        )
    return SimpleNamespace(
        agents=agents,
        _desired_center_spacing_m=lambda *_: 15.74,
    )


def _reward() -> JointTrajectoryProxyReward:
    reward = JointTrajectoryProxyReward()
    reward._prediction_planner._predicted_obstacles = lambda *args, **kwargs: []
    return reward


def test_proxy_contract_and_progress_monotonicity() -> None:
    slow = np.stack([_trajectory(3.0)] * 3)
    fast = np.stack([_trajectory(6.0)] * 3)
    result = _reward().score(
        _env(), _model_inputs(), np.stack((slow, fast))
    )
    assert result.rewards.shape == (2,)
    assert result.rewards.dtype == np.float32
    assert not result.unsafe.any()
    assert result.components["progress"][1] > result.components["progress"][0]
    assert result.rewards[1] > result.rewards[0]


def test_out_of_drivable_and_platoon_collision_are_hard_unsafe() -> None:
    safe = np.stack([_trajectory(4.0)] * 3)
    out = safe.copy()
    out[0] = _trajectory(4.0, lateral_m=40.0)
    collision = safe.copy()
    collision[1] = _trajectory(9.0)
    result = _reward().score(
        _env(), _model_inputs(), np.stack((safe, out, collision))
    )
    assert result.out_of_drivable.tolist() == [False, True, False]
    assert result.collision.tolist() == [False, False, True]
    assert result.unsafe.tolist() == [False, True, True]
    assert result.rewards[0] >= -5.0
    assert np.all(result.rewards[1:] < -20.0)


def test_platoon_clearance_violation_is_hard_unsafe() -> None:
    safe = np.stack([_trajectory(4.0)] * 3)
    closing = safe.copy()
    closing[1] = _trajectory(5.5)
    result = _reward().score(
        _env(), _model_inputs(), np.stack((safe, closing))
    )
    assert result.unsafe.tolist() == [False, True]
    assert result.clearance_violation.tolist() == [False, True]
    assert result.components["clearance"][0] == pytest.approx(0.0)
    assert result.components["clearance"][1] < 0.0
    assert result.rewards[1] < -20.0


def test_clearance_hard_gate_boundary_is_strict() -> None:
    common = {
        "progress": np.zeros(2),
        "formation": np.zeros(2),
        "clearance": np.zeros(2),
        "comfort": np.zeros(2),
        "collision": np.zeros(2, dtype=np.bool_),
        "out_of_drivable": np.zeros(2, dtype=np.bool_),
        "clearance_violation": np.asarray([False, True], dtype=np.bool_),
        "config": JointRewardConfig(),
    }
    result = compose_joint_reward(**common)
    assert result.unsafe.tolist() == [False, True]
    assert result.rewards[0] == pytest.approx(0.0)
    assert result.rewards[1] < -20.0


def test_background_prediction_participates_in_collision() -> None:
    reward = JointTrajectoryProxyReward()
    times = np.arange(0.1, 4.01, 0.1)
    predicted = np.zeros((len(times), 3), dtype=np.float64)
    predicted[:, 0] = 10.0
    reward._prediction_planner._predicted_obstacles = (
        lambda *args, **kwargs: [("background", predicted, (5.74, 2.3))]
    )
    group = np.stack([_trajectory(4.0)] * 3)[None]
    result = reward.score(_env(), _model_inputs(), group)
    assert result.collision.tolist() == [True]
    assert result.rewards[0] < -20.0


def test_background_gap_uses_shared_corridor_not_radial_center_distance() -> None:
    reward = JointTrajectoryProxyReward()
    times = np.arange(0.1, 4.01, 0.1)
    adjacent = np.zeros((len(times), 3), dtype=np.float64)
    adjacent[:, 0] = 4.0 * times
    adjacent[:, 1] = 3.5
    reward._prediction_planner._predicted_obstacles = (
        lambda *args, **kwargs: [("adjacent", adjacent, (5.74, 2.3))]
    )
    group = np.stack([_trajectory(4.0)] * 3)[None]

    adjacent_result = reward.score(_env(), _model_inputs(), group)

    assert adjacent_result.components["minimum_background_gap_m"][0] == pytest.approx(
        1.0e6
    )
    assert not adjacent_result.clearance_violation[0]

    same_corridor = adjacent.copy()
    same_corridor[:, 0] += 10.0
    same_corridor[:, 1] = 0.0
    reward._prediction_planner._predicted_obstacles = (
        lambda *args, **kwargs: [("ahead", same_corridor, (5.74, 2.3))]
    )

    close_result = reward.score(_env(), _model_inputs(), group)

    assert close_result.components["minimum_background_gap_m"][0] == pytest.approx(
        4.26, abs=1.0e-5
    )
    assert close_result.clearance_violation[0]


def test_reward_rejects_labels_and_invalid_contracts() -> None:
    values = _model_inputs()
    values.gt_mode = np.zeros(3, dtype=np.int64)
    # Extra expert fields are ignored; reward only consumes BEV and relation.
    result = _reward().score(
        _env(), values, np.stack([np.stack([_trajectory(4.0)] * 3)])
    )
    assert np.isfinite(result.rewards).all()
    with pytest.raises(JointRewardError):
        _reward().score(
            _env(), values, np.zeros((4, 3, 7, 3), dtype=np.float32)
        )
    bad = _model_inputs()
    bad.bev = bad.bev.astype(np.float32)
    with pytest.raises(JointRewardError):
        _reward().score(
            _env(), bad, np.stack([np.stack([_trajectory(4.0)] * 3)])
        )


def test_calibration_statistics_and_gate() -> None:
    proxy = np.tile(np.asarray([0.0, 1.0, 2.0, 3.0]), (12, 1))
    simulator = proxy * 2.0 + 0.25
    safe = np.zeros_like(proxy, dtype=np.bool_)
    passed = calibrate_joint_rewards(proxy, simulator, safe, safe)
    assert passed.passed is True
    assert passed.mean_spearman == pytest.approx(1.0)
    assert passed.pairwise_agreement == pytest.approx(1.0)
    assert passed.informative_groups == 12

    simulator_bad = safe.copy()
    simulator_bad[0, 0] = True
    failed = calibrate_joint_rewards(proxy, simulator, safe, simulator_bad)
    assert failed.passed is False
    assert failed.false_safe_count == 1


def test_calibration_rejects_uninformative_or_wrong_shapes() -> None:
    constant = np.ones((12, 4), dtype=np.float32)
    safe = np.zeros((12, 4), dtype=np.bool_)
    result = calibrate_joint_rewards(constant, constant, safe, safe)
    assert result.passed is False
    assert result.informative_groups == 0
    with pytest.raises(JointRewardError):
        calibrate_joint_rewards(
            np.zeros((2, 3)), np.zeros((2, 3)), safe[:2, :3], safe[:2, :3]
        )


def test_reward_config_is_strict() -> None:
    with pytest.raises(JointRewardError):
        JointRewardConfig(local_mix=0.8, team_mix=0.8)
    with pytest.raises(JointRewardError):
        JointRewardConfig(unsafe_base_reward=-4.0)
    with pytest.raises(JointRewardError):
        JointRewardConfig(tracking_lateral_margin_m=1.01)


def test_tracking_envelope_makes_proxy_footprint_more_conservative() -> None:
    inputs = _model_inputs()
    # A narrow drivable stripe that admits the nominal center footprint but
    # not the measured-error-expanded footprint.
    inputs.bev[:, 0] = 0
    inputs.bev[:, 0, 188:228, 120:136] = 255
    stationary = np.stack([_trajectory(0.0)] * 3)[None]
    nominal = JointTrajectoryProxyReward(
        JointRewardConfig()
    )
    expanded = JointTrajectoryProxyReward(
        JointRewardConfig(
            tracking_longitudinal_margin_m=1.0,
            tracking_lateral_margin_m=0.8,
            tracking_heading_margin_rad=0.1,
        )
    )
    nominal._prediction_planner._predicted_obstacles = lambda *args, **kwargs: []
    expanded._prediction_planner._predicted_obstacles = lambda *args, **kwargs: []
    nominal_result = nominal.score(_env(), inputs, stationary)
    expanded_result = expanded.score(_env(), inputs, stationary)
    assert not nominal_result.out_of_drivable[0]
    assert expanded_result.out_of_drivable[0]
