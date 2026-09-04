from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from models.bev_planner import (
    VEHICLE_MODE_REWARD_CONTRACT,
    VehicleModeCounterfactualReward,
    VehicleModeRewardConfig,
    vehicle_mode_reward_config_sha256,
)


def _trajectory(speed_mps: float, lateral_m: float = 0.0) -> np.ndarray:
    times = np.arange(1, 9, dtype=np.float32) * 0.5
    value = np.zeros((8, 3), dtype=np.float32)
    value[:, 0] = float(speed_mps) * times
    value[:, 1] = np.linspace(0.0, lateral_m, 8, dtype=np.float32)
    return value


def _env(positions: tuple[float, float, float] = (0.0, -15.74, -31.48)):
    agents = {}
    for role, x in enumerate(positions):
        agents[f"agent{role}"] = SimpleNamespace(
            name=f"agent{role}",
            position=np.asarray([x, 0.0], dtype=np.float32),
            heading_theta=0.0,
            velocity=np.asarray([4.0, 0.0], dtype=np.float32),
            LENGTH=5.74,
            WIDTH=2.3,
        )
    return SimpleNamespace(agents=agents)


def _model_inputs() -> SimpleNamespace:
    bev = np.zeros((3, 8, 256, 256), dtype=np.uint8)
    bev[:, 0] = 255
    # No formation_relation_state is intentionally provided.
    return SimpleNamespace(bev=bev)


def _inputs(sample_count: int = 2):
    selected = np.stack([_trajectory(4.0)] * 3)
    all_modes = np.broadcast_to(
        selected[:, None], (3, 10, 8, 3)
    ).copy()
    candidates = np.broadcast_to(
        all_modes[:, :, None], (3, 10, sample_count, 8, 3)
    ).copy()
    return candidates, all_modes, selected


def _scorer(sample_count: int = 2) -> VehicleModeCounterfactualReward:
    scorer = VehicleModeCounterfactualReward(
        VehicleModeRewardConfig(trajectories_per_mode=sample_count)
    )
    scorer._prediction_planner._predicted_obstacles = lambda *args, **kwargs: []
    return scorer


def test_contract_shape_baseline_equality_and_no_formation() -> None:
    candidates, all_modes, selected = _inputs()
    valid = np.zeros((3, 10), dtype=np.bool_)
    valid[0, 2] = True
    valid[2, 7] = True
    result = _scorer().score_counterfactuals(
        _env(),
        _model_inputs(),
        candidates[None],
        all_modes[None],
        selected[None],
        valid[None],
    )

    assert result.rewards.shape == (3, 10, 2)
    assert result.pretrain_rewards.shape == (3, 10)
    assert result.valid_mode_mask.tolist() == valid.tolist()
    assert result.rewards[0, 2] == pytest.approx(result.pretrain_rewards[0, 2])
    assert result.rewards[2, 7] == pytest.approx(result.pretrain_rewards[2, 7])
    assert "formation_penalty" not in result.components
    assert "formation_penalty" not in result.pretrain_components
    assert VEHICLE_MODE_REWARD_CONTRACT["formation_component"] is False
    assert len(vehicle_mode_reward_config_sha256(_scorer().config)) == 64


def test_invalid_modes_are_not_scored_and_are_zero_filled() -> None:
    candidates, all_modes, selected = _inputs()
    valid = np.zeros((3, 10), dtype=np.bool_)
    valid[1, 4] = True
    candidates[0, 0, :, :, 1] = 40.0
    result = _scorer().score_counterfactuals(
        _env(), _model_inputs(), candidates, all_modes, selected, valid
    )

    invalid = ~valid
    assert np.all(result.rewards[invalid] == 0.0)
    assert np.all(result.pretrain_rewards[invalid] == 0.0)
    assert not result.unsafe[invalid].any()
    assert not result.pretrain_unsafe[invalid].any()
    for values in result.components.values():
        assert np.all(values[invalid] == 0.0)


def test_other_vehicle_and_mode_candidates_do_not_change_target_group() -> None:
    candidates, all_modes, selected = _inputs()
    valid = np.zeros((3, 10), dtype=np.bool_)
    valid[0, 3] = True
    valid[1, 6] = True
    first = _scorer().score_counterfactuals(
        _env(), _model_inputs(), candidates, all_modes, selected, valid
    )

    changed = candidates.copy()
    changed[1:, :, :, :, 1] = 45.0
    changed[0, 4:, :, :, 1] = -45.0
    changed_all_modes = all_modes.copy()
    changed_all_modes[1:, :, :, 1] = 45.0
    second = _scorer().score_counterfactuals(
        _env(),
        _model_inputs(),
        changed,
        changed_all_modes,
        selected,
        valid,
    )

    assert second.rewards[0, 3] == pytest.approx(first.rewards[0, 3])
    assert second.pretrain_rewards[0, 3] == pytest.approx(
        first.pretrain_rewards[0, 3]
    )


def test_split_api_caches_same_mode_pretrain_across_noise_resamples() -> None:
    candidates, all_modes, selected = _inputs()
    valid = np.zeros((3, 10), dtype=np.bool_)
    valid[2, 5] = True
    scorer = _scorer()
    observed_group_sizes = []
    original = scorer._score_target_group

    def counted(**kwargs):
        observed_group_sizes.append(kwargs["target_trajectories"].shape[0])
        return original(**kwargs)

    scorer._score_target_group = counted
    pretrain = scorer.score_pretrain(
        _env(), _model_inputs(), all_modes, selected, valid
    )
    first = scorer.score_candidates(
        _env(), _model_inputs(), candidates, selected, valid, pretrain
    )
    candidates[2, 5, 0] = _trajectory(6.0)
    second = scorer.score_candidates(
        _env(), _model_inputs(), candidates, selected, valid, pretrain
    )

    assert observed_group_sizes == [1, 2, 2]
    assert first.pretrain_rewards[2, 5] == second.pretrain_rewards[2, 5]
    assert first.rewards[2, 5, 0] != second.rewards[2, 5, 0]


def test_target_reward_responds_to_target_teammate_interaction() -> None:
    candidates, all_modes, selected = _inputs()
    valid = np.zeros((3, 10), dtype=np.bool_)
    valid[1, 0] = True
    candidates[1, 0] = _trajectory(8.0)
    all_modes[1, 0] = _trajectory(8.0)

    stopped_leader = selected.copy()
    stopped_leader[0] = _trajectory(0.0)
    dangerous = _scorer().score_counterfactuals(
        _env(),
        _model_inputs(),
        candidates,
        all_modes,
        stopped_leader,
        valid,
    )
    moving_leader = selected.copy()
    moving_leader[0] = _trajectory(8.0)
    safe = _scorer().score_counterfactuals(
        _env(),
        _model_inputs(),
        candidates,
        all_modes,
        moving_leader,
        valid,
    )

    assert dangerous.collision[1, 0].all()
    assert not safe.collision[1, 0].any()
    assert np.all(dangerous.rewards[1, 0] < safe.rewards[1, 0])


def test_teammate_only_collision_does_not_pollute_target_reward() -> None:
    candidates, all_modes, selected = _inputs()
    valid = np.zeros((3, 10), dtype=np.bool_)
    valid[0, 0] = True
    distant_env = _env((100.0, 0.0, -15.74))
    quiet = _scorer().score_counterfactuals(
        distant_env,
        _model_inputs(),
        candidates,
        all_modes,
        selected,
        valid,
    )

    colliding_teammates = selected.copy()
    colliding_teammates[1] = _trajectory(0.0)
    colliding_teammates[2] = _trajectory(4.0)
    collision_elsewhere = _scorer().score_counterfactuals(
        distant_env,
        _model_inputs(),
        candidates,
        all_modes,
        colliding_teammates,
        valid,
    )

    assert collision_elsewhere.rewards[0, 0] == pytest.approx(
        quiet.rewards[0, 0]
    )
    assert not collision_elsewhere.collision[0, 0].any()
