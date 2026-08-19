from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from envs.observations.semantic_bev import SemanticBEVConfig
from models.bev_planner import (
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256,
    JOINT_REWARD_CONTRACT,
    JOINT_REWARD_CONTRACT_SHA256,
    JointRewardConfig,
    JointRewardError,
    JointTrajectoryProxyReward,
    aggregate_temporal_risk,
    closing_ttc_from_gap_series,
    compose_joint_reward,
    drivable_signed_distance_m,
    footprint_outside_drivable_series,
    footprint_road_margin_series,
    joint_reward_config_sha256,
    shared_corridor_gap_series,
    soft_threshold_risk,
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
    assert (
        result.components["progress_score"][1]
        > result.components["progress_score"][0]
    )
    assert result.rewards[1] > result.rewards[0]


def test_out_of_drivable_and_collision_are_candidate_local_penalties() -> None:
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
    assert np.all(result.rewards > -10.0)
    assert result.rewards[1] < result.rewards[0]
    assert result.rewards[2] < result.rewards[0]


def test_platoon_gap_is_continuous_and_unsafe_is_diagnostic() -> None:
    safe = np.stack([_trajectory(4.0)] * 3)
    closing = safe.copy()
    closing[1] = _trajectory(5.5)
    result = _reward().score(
        _env(), _model_inputs(), np.stack((safe, closing))
    )
    assert result.unsafe.tolist() == [False, True]
    assert result.clearance_violation.tolist() == [False, True]
    assert (
        result.components["gap_penalty"][1]
        > result.components["gap_penalty"][0]
    )
    assert result.rewards[1] < result.rewards[0]
    assert result.rewards[1] > -5.0


def test_clearance_flag_does_not_override_reward() -> None:
    common = {
        "progress_score": np.zeros(2),
        "formation_penalty": np.zeros(2),
        "gap_penalty": np.zeros(2),
        "ttc_penalty": np.zeros(2),
        "road_penalty": np.zeros(2),
        "comfort_penalty": np.zeros(2),
        "collision": np.zeros(2, dtype=np.bool_),
        "out_of_drivable": np.zeros(2, dtype=np.bool_),
        "clearance_violation": np.asarray([False, True], dtype=np.bool_),
        "config": JointRewardConfig(),
    }
    result = compose_joint_reward(**common)
    assert result.unsafe.tolist() == [False, True]
    assert result.rewards.tolist() == pytest.approx([0.0, 0.0])


def test_background_prediction_participates_in_collision() -> None:
    reward = JointTrajectoryProxyReward()
    times = np.arange(0.0, 4.01, 0.1)
    predicted = np.zeros((len(times), 3), dtype=np.float64)
    predicted[:, 0] = 10.0
    reward._prediction_planner._predicted_obstacles = (
        lambda *args, **kwargs: [("background", predicted, (5.74, 2.3))]
    )
    group = np.stack([_trajectory(4.0)] * 3)[None]
    result = reward.score(_env(), _model_inputs(), group)
    assert result.collision.tolist() == [True]
    assert -10.0 < result.rewards[0] < -5.0


def test_proxy_continuous_risks_include_t0_to_point_one_ttc() -> None:
    reward = JointTrajectoryProxyReward()
    observed_times = None

    def predicted_obstacles(_env, _vehicle, times, **_kwargs):
        nonlocal observed_times
        observed_times = np.asarray(times, dtype=np.float64).copy()
        predicted = np.zeros((len(observed_times), 3), dtype=np.float64)
        predicted[:, 0] = 4.0 * observed_times + 10.34
        predicted[0, 0] = 10.74
        return [("background", predicted, (5.74, 2.3))]

    reward._prediction_planner._predicted_obstacles = predicted_obstacles
    group = np.stack([_trajectory(4.0)] * 3)[None]
    result = reward.score(_env(), _model_inputs(), group)

    assert observed_times is not None
    assert observed_times.shape == (41,)
    assert observed_times[0] == 0.0
    assert observed_times[-1] == pytest.approx(4.0)
    assert result.components["minimum_ttc_s"][0] == pytest.approx(1.15)
    ttc = np.full(41, 1.0e6, dtype=np.float64)
    ttc[1] = 1.15
    expected = aggregate_temporal_risk(
        soft_threshold_risk(
            ttc, warning_threshold=4.0, softness=0.5
        ),
        max_weight=0.7,
        mean_weight=0.3,
    )
    assert result.components["ttc_penalty"][0] == pytest.approx(expected)


def test_background_gap_uses_shared_corridor_not_radial_center_distance() -> None:
    reward = JointTrajectoryProxyReward()
    times = np.arange(0.0, 4.01, 0.1)
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


def test_alternative_background_branch_only_adds_continuous_risk() -> None:
    times = np.arange(0.0, 4.01, 0.1)
    nominal = np.zeros((len(times), 3), dtype=np.float64)
    nominal[:, 0] = 100.0
    alternative = np.zeros((len(times), 3), dtype=np.float64)
    alternative[:, 0] = 4.0 * times
    group = np.stack([_trajectory(4.0)] * 3)[None]

    one_branch = JointTrajectoryProxyReward()
    one_branch._prediction_planner._predicted_obstacles = (
        lambda *args, **kwargs: [
            ("background", nominal, (5.74, 2.3)),
            ("background:policy_branch_0", alternative, (5.74, 2.3)),
        ]
    )
    repeated_branch = JointTrajectoryProxyReward()
    repeated_branch._prediction_planner._predicted_obstacles = (
        lambda *args, **kwargs: [
            ("background", nominal, (5.74, 2.3)),
            ("background:policy_branch_0", alternative, (5.74, 2.3)),
            ("background:policy_branch_1", alternative.copy(), (5.74, 2.3)),
        ]
    )
    nominal_only = JointTrajectoryProxyReward()
    nominal_only._prediction_planner._predicted_obstacles = (
        lambda *args, **kwargs: [
            ("background", nominal, (5.74, 2.3)),
        ]
    )

    first = one_branch.score(_env(), _model_inputs(), group)
    repeated = repeated_branch.score(_env(), _model_inputs(), group)
    baseline = nominal_only.score(_env(), _model_inputs(), group)
    assert not first.collision[0]
    assert not first.clearance_violation[0]
    assert not first.unsafe[0]
    assert first.components["gap_penalty"][0] > baseline.components[
        "gap_penalty"
    ][0]
    assert repeated.rewards[0] == pytest.approx(first.rewards[0])
    assert repeated.components["gap_penalty"][0] == pytest.approx(
        first.components["gap_penalty"][0]
    )


def test_dangerous_candidates_keep_quality_ordering() -> None:
    result = compose_joint_reward(
        progress_score=np.asarray([0.1, 0.5, 0.9]),
        formation_penalty=np.asarray([0.8, 0.4, 0.1]),
        gap_penalty=np.asarray([1.0, 0.8, 0.6]),
        ttc_penalty=np.asarray([1.0, 0.7, 0.4]),
        road_penalty=np.asarray([0.8, 0.5, 0.2]),
        comfort_penalty=np.asarray([0.7, 0.4, 0.1]),
        collision=np.ones(3, dtype=np.bool_),
        out_of_drivable=np.zeros(3, dtype=np.bool_),
        clearance_violation=np.ones(3, dtype=np.bool_),
        config=JointRewardConfig(),
    )
    assert len(np.unique(result.rewards)) == 3
    assert np.all(np.diff(result.rewards) > 0.0)


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


def test_grpo_open_application_contract_freezes_tau_d_without_changing_reward() -> None:
    assert GRPO_OPEN_REWARD_APPLICATION_CONTRACT == {
        "version": "stage2_grpo_open_application_v1",
        "policy_sample_domain": "tau_d",
        "policy_probability_domain": "tau_d",
        "reward_input_domain": "tau_d",
        "candidate_selection_domain": "tau_d",
        "execution_input_domain": "tau_cmd",
        "execution_transform": "KinematicTrajectoryOptimizer(selected_tau_d)",
        "optimize_only_selected_candidate": True,
        "optimizer_must_succeed_before_policy_update": True,
        "tracking_expansion_enabled": False,
        "calibration_required": False,
        "best_checkpoint_metric": "validation/raw_proxy_reward_mean",
        "simulator_validation_role": "diagnostic_only",
    }
    assert GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256 == (
        "7498a5cb80388f7db3c6edfeb8108d3d38ea0415cb009e8c42929912c0a693b6"
    )


def test_reward_config_is_strict() -> None:
    with pytest.raises(TypeError):
        JointRewardConfig(local_mix=0.8, team_mix=0.8)
    with pytest.raises(TypeError):
        JointRewardConfig(unsafe_base_reward=-4.0)
    with pytest.raises(JointRewardError):
        JointRewardConfig(tracking_lateral_margin_m=1.01)
    with pytest.raises(JointRewardError):
        JointRewardConfig(temporal_max_weight=0.8)


def test_tracking_envelope_only_increases_continuous_road_risk() -> None:
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
    assert not expanded_result.out_of_drivable[0]
    assert (
        expanded_result.components["road_penalty"][0]
        > nominal_result.components["road_penalty"][0]
    )


def test_tracking_envelope_cannot_create_physical_collision() -> None:
    env = _env()
    for role, x in enumerate((0.0, -6.5, -13.0)):
        env.agents[f"agent{role}"].position = np.asarray(
            [x, 0.0], dtype=np.float32
        )
    group = np.stack([_trajectory(0.0)] * 3)[None]
    nominal = JointTrajectoryProxyReward(JointRewardConfig())
    expanded = JointTrajectoryProxyReward(
        JointRewardConfig(tracking_longitudinal_margin_m=0.6)
    )
    nominal._prediction_planner._predicted_obstacles = lambda *args, **kwargs: []
    expanded._prediction_planner._predicted_obstacles = lambda *args, **kwargs: []
    nominal_result = nominal.score(env, _model_inputs(), group)
    expanded_result = expanded.score(env, _model_inputs(), group)
    assert not nominal_result.collision[0]
    assert not expanded_result.collision[0]
    assert (
        expanded_result.components["gap_penalty"][0]
        > nominal_result.components["gap_penalty"][0]
    )


def test_compose_formula_is_exact_and_not_clipped() -> None:
    config = JointRewardConfig()
    result = compose_joint_reward(
        progress_score=np.asarray([1.0, 0.0]),
        formation_penalty=np.asarray([0.2, 1.0]),
        gap_penalty=np.asarray([0.3, 1.0]),
        ttc_penalty=np.asarray([0.4, 1.0]),
        road_penalty=np.asarray([0.5, 1.0]),
        comfort_penalty=np.asarray([0.6, 1.0]),
        collision=np.asarray([True, True]),
        out_of_drivable=np.asarray([True, True]),
        clearance_violation=np.asarray([True, True]),
        config=config,
    )
    expected = (
        0.47
        - 1.0 * 0.2
        - 1.185 * 0.3
        - 0.5 * 0.4
        - 0.5 * 0.5
        - 0.0225 * 0.6
        - 5.0
        - 4.0
    )
    assert result.rewards[0] == pytest.approx(expected)
    assert result.rewards[1] == pytest.approx(-12.2075)
    assert result.rewards[1] < -10.0  # no legacy total-reward clipping


def test_soft_gap_and_ttc_risks_are_continuous_and_monotonic() -> None:
    platoon_gap = soft_threshold_risk(
        np.asarray([7.01, 7.0, 6.99, 5.0]),
        warning_threshold=7.0,
        softness=0.5,
    )
    background_gap = soft_threshold_risk(
        np.asarray([5.01, 5.0, 4.99]),
        warning_threshold=5.0,
        softness=0.5,
    )
    assert np.all(np.diff(platoon_gap) > 0.0)
    assert np.all(np.diff(background_gap) > 0.0)
    assert abs(float(platoon_gap[2] - platoon_gap[1])) < 0.01
    assert abs(float(background_gap[2] - background_gap[1])) < 0.01

    receding = closing_ttc_from_gap_series(
        np.asarray([5.0, 5.1, 5.2]),
        dt_s=0.1,
        closing_speed_epsilon_mps=0.1,
        no_risk_gap_m=1.0e6,
        no_risk_ttc_s=1.0e6,
    )
    stationary = closing_ttc_from_gap_series(
        np.asarray([5.0, 5.0, 5.0]),
        dt_s=0.1,
        closing_speed_epsilon_mps=0.1,
        no_risk_gap_m=1.0e6,
        no_risk_ttc_s=1.0e6,
    )
    assert np.all(receding == 1.0e6)
    assert np.all(stationary == 1.0e6)
    slow_closing = closing_ttc_from_gap_series(
        np.asarray([5.0, 4.9, 4.8]),
        dt_s=0.1,
        closing_speed_epsilon_mps=0.1,
        no_risk_gap_m=1.0e6,
        no_risk_ttc_s=1.0e6,
    )
    fast_closing = closing_ttc_from_gap_series(
        np.asarray([5.0, 4.5, 4.0]),
        dt_s=0.1,
        closing_speed_epsilon_mps=0.1,
        no_risk_gap_m=1.0e6,
        no_risk_ttc_s=1.0e6,
    )
    assert np.min(fast_closing) < np.min(slow_closing)


def test_temporal_risk_uses_frozen_max_mean_weights() -> None:
    values = np.asarray([0.0, 0.5, 1.0])
    assert aggregate_temporal_risk(
        values, max_weight=0.7, mean_weight=0.3
    ) == pytest.approx(0.85)


def test_reward_contract_and_config_hashes_are_stable_and_sensitive() -> None:
    assert JOINT_REWARD_CONTRACT["version"] == "stage2_joint_reward_v2"
    assert len(JOINT_REWARD_CONTRACT_SHA256) == 64
    baseline = joint_reward_config_sha256(JointRewardConfig())
    changed = joint_reward_config_sha256(JointRewardConfig(gap_softness_m=0.6))
    assert len(baseline) == 64
    assert baseline != changed


def test_signed_road_margin_is_continuous_but_physical_out_is_discrete() -> None:
    drivable = np.zeros((256, 256), dtype=np.uint8)
    drivable[80:220, 90:166] = 255
    field = drivable_signed_distance_m(drivable)
    poses = np.asarray(
        [[10.0, 0.0, 0.0], [10.0, 8.0, 0.0], [10.0, 15.0, 0.0]],
        dtype=np.float64,
    )
    config = JointRewardConfig()
    margins = footprint_road_margin_series(
        poses, field, config, tracking_aware=True
    )
    outside = footprint_outside_drivable_series(poses, drivable, config)
    assert margins[0] > margins[1] > margins[2]
    assert outside.tolist() == [False, False, True]


def test_rotated_obb_support_participates_in_shared_corridor_gap() -> None:
    first = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
    perpendicular = np.asarray([[0.0, 3.0, np.pi / 2.0]], dtype=np.float64)
    gap = shared_corridor_gap_series(
        first,
        (5.74, 2.3),
        perpendicular,
        (5.74, 2.3),
        no_risk_gap_m=1.0e6,
    )
    assert gap[0] < 0.0


def test_physical_out_checks_full_raster_resolution_footprint() -> None:
    config = JointRewardConfig()
    bev_config = SemanticBEVConfig()
    drivable = np.full(
        (bev_config.height, bev_config.width), 255, dtype=np.uint8
    )

    def row_for_x(value: float) -> int:
        return int(
            round(
                (bev_config.x_max_m - value)
                / (bev_config.x_max_m - bev_config.x_min_m)
                * (bev_config.height - 1)
            )
        )

    def col_for_y(value: float) -> int:
        return int(
            round(
                (bev_config.y_max_m - value)
                / (bev_config.y_max_m - bev_config.y_min_m)
                * (bev_config.width - 1)
            )
        )

    rows = sorted((row_for_x(1.25), row_for_x(0.75)))
    columns = sorted((col_for_y(0.75), col_for_y(-0.75)))
    drivable[rows[0] : rows[1] + 1, columns[0] : columns[1] + 1] = 0
    outside = footprint_outside_drivable_series(
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        drivable,
        config,
    )
    assert outside.tolist() == [True]
