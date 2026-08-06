from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.joint_simulator_branch import (
    JointEpisodeSpec,
    JointSimulatorBranchEvaluator,
    _local_reference_to_world,
    _reference_arc_kinematics,
    _tracking_error_against_reference,
    _trajectory_target_speed_mps,
    _world_reference_to_current_local,
)
from models.bev_planner import JointRewardError


def _trajectory(speed_mps: float) -> np.ndarray:
    value = np.zeros((8, 3), dtype=np.float32)
    value[:, 0] = speed_mps * np.arange(1, 9, dtype=np.float32) * 0.5
    return value


class _BranchEnv:
    def __init__(self, config) -> None:
        self.config = {
            "physics_world_step_size": 0.1,
            "decision_repeat": 5,
            **dict(config),
        }
        self._closed = False
        self.reset()

    def set_runtime_scenario_route(self, scenario_id: str, local_route: str) -> None:
        self.scenario_id = scenario_id
        self.local_route = local_route

    def reset(self):
        self._branch_trajectories = {}
        self.agents = {
            f"agent{role}": SimpleNamespace(
                name=f"agent{role}",
                position=np.asarray([x, 0.0], dtype=np.float64),
                heading_theta=0.0,
                speed_km_h=0.0,
                LENGTH=5.74,
                WIDTH=2.3,
            )
            for role, x in enumerate((0.0, -15.74, -31.48))
        }
        return {}

    def _desired_center_spacing_m(self, *_args) -> float:
        return 15.74

    def trajectory_to_control(self, *_args):
        raise AssertionError(
            "branch diagnostics must not invoke the stateful controller outside step()"
        )

    def trajectory_reference_to_control(self, agent_id, trajectory, reference):
        assert reference.source == "simulator_branch"
        self._branch_trajectories[agent_id] = np.asarray(trajectory).copy()
        return np.asarray([0.1, 0.2], dtype=np.float32)

    def step(self, actions):
        self._pending_low_level_actions = {
            agent_id: np.asarray([0.1, 0.2], dtype=np.float32)
            for agent_id in actions
        }
        for agent_id in actions:
            trajectory = self._branch_trajectories[agent_id]
            vehicle = self.agents[agent_id]
            point = np.asarray(trajectory, dtype=np.float64)[0]
            heading = float(vehicle.heading_theta)
            cos_h = np.cos(heading)
            sin_h = np.sin(heading)
            vehicle.position = vehicle.position + np.asarray(
                [
                    cos_h * point[0] - sin_h * point[1],
                    sin_h * point[0] + cos_h * point[1],
                ]
            )
            vehicle.heading_theta = float(heading + point[2])
            vehicle.speed_km_h = float(np.linalg.norm(point[:2]) / 0.5 * 3.6)
        flags = {agent_id: False for agent_id in self.agents}
        flags["__all__"] = False
        info = {agent_id: {} for agent_id in self.agents}
        return {}, {}, flags, flags.copy(), info

    def close(self) -> None:
        self._closed = True


def _spec(reference=None) -> JointEpisodeSpec:
    if reference is None:
        reference = np.asarray(
            [[0.0, 0.0, 0.0], [-15.74, 0.0, 0.0], [-31.48, 0.0, 0.0]],
            dtype=np.float64,
        )
    return JointEpisodeSpec(
        "S1_free_cruise_straight",
        "R3_mainline_straight",
        17,
        reference,
    )


def _evaluator() -> JointSimulatorBranchEvaluator:
    evaluator = JointSimulatorBranchEvaluator(env_factory=_BranchEnv)
    evaluator._vehicle_helper._surrounding_vehicles = lambda _env: []
    return evaluator


def test_branch_recreates_each_group_and_tracks_for_four_seconds() -> None:
    slow = np.stack([_trajectory(2.0)] * 3)
    fast = np.stack([_trajectory(4.0)] * 3)
    result = _evaluator().evaluate(
        _spec(),
        (),
        np.stack((slow, fast)),
    )
    assert result.executed_steps.tolist() == [8, 8]
    np.testing.assert_allclose(result.initial_speed_mps, 0.0)
    assert result.replay_position_error_m.tolist() == [0.0, 0.0]
    assert result.replay_heading_error_rad.tolist() == [0.0, 0.0]
    assert not result.reward.unsafe.any()
    assert result.reward.rewards[1] > result.reward.rewards[0]
    assert result.minimum_platoon_gap_m == pytest.approx([10.0, 10.0])
    assert result.tracking_longitudinal_error_m.shape == (2, 3)
    assert result.tracking_lateral_error_m.shape == (2, 3)
    assert result.tracking_heading_error_rad.shape == (2, 3)
    np.testing.assert_allclose(result.tracking_longitudinal_error_m, 0.0)
    np.testing.assert_allclose(result.tracking_lateral_error_m, 0.0)
    np.testing.assert_allclose(result.tracking_heading_error_rad, 0.0)
    assert len(result.tracking_traces) == 2
    assert len(result.tracking_traces[0]) == 3
    assert len(result.tracking_traces[0][0]["actual_world"]) == 8
    assert result.tracking_traces[0][0]["steering"] == pytest.approx([0.1] * 8)
    assert result.tracking_traces[0][0]["throttle"] == pytest.approx([0.2] * 8)
    trace = result.tracking_traces[0][0]
    assert len(trace["reference_arc_position_m"]) == 8
    assert len(trace["actual_projected_arc_position_m"]) == 8
    assert len(trace["reference_feedforward_speed_mps"]) == 8
    assert len(trace["reference_feedforward_acceleration_mps2"]) == 8
    assert len(trace["position_error_speed_increment_mps"]) == 8
    assert len(trace["formation_gap_error_m"]) == 8
    assert len(trace["formation_control_increment"]) == 8
    assert len(trace["actual_acceleration_mps2"]) == 8
    assert len(trace["lateral_heading_contaminated"]) == 8
    assert trace["maximum_continuous_saturation_s"] == pytest.approx(0.0)


def test_tracking_metric_does_not_turn_curve_timing_lag_into_lateral_error() -> None:
    reference = np.zeros((9, 3), dtype=np.float64)
    reference[:, 0] = np.arange(9, dtype=np.float64)
    reference[:, 1] = 0.1 * reference[:, 0] ** 2
    reference[:, 2] = np.arctan(0.2 * reference[:, 0])
    actual = reference[3].copy()
    longitudinal, lateral, heading, nearest = (
        _tracking_error_against_reference(actual, reference, elapsed_s=2.5)
    )
    assert longitudinal < 0.0
    assert lateral == pytest.approx(0.0, abs=1.0e-8)
    assert heading == pytest.approx(0.0, abs=1.0e-8)
    np.testing.assert_allclose(nearest, actual)


def test_current_target_speed_mixes_position_lag_into_speed_command() -> None:
    world = _local_reference_to_world(
        _trajectory(4.0), np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    )
    # At t=1 s the reference is x=4 m, while the vehicle is deliberately 2 m
    # behind.  The current target-speed extraction measures from the actual
    # vehicle to the next two fixed-time points and therefore adds that 2 m
    # position lag to the one-second speed command.
    local = _world_reference_to_current_local(
        world,
        np.asarray([2.0, 0.0, 0.0], dtype=np.float64),
        elapsed_s=1.0,
    )
    _, reference_speed, _ = _reference_arc_kinematics(world, elapsed_s=1.0)
    current_target = _trajectory_target_speed_mps(local)
    assert reference_speed == pytest.approx(4.0)
    assert current_target == pytest.approx(6.0)
    assert current_target - reference_speed == pytest.approx(2.0)


def test_branch_is_deterministic_and_does_not_mutate_inputs() -> None:
    candidates = np.stack([np.stack([_trajectory(3.0)] * 3)] * 2)
    before = candidates.copy()
    first = _evaluator().evaluate(_spec(), (), candidates)
    second = _evaluator().evaluate(_spec(), (), candidates)
    np.testing.assert_array_equal(candidates, before)
    np.testing.assert_array_equal(first.reward.rewards, second.reward.rewards)


def test_branch_rejects_replay_drift_and_invalid_prefix() -> None:
    candidates = np.stack([np.stack([_trajectory(3.0)] * 3)])
    wrong = _spec(
        np.asarray(
            [[1.0, 0.0, 0.0], [-15.74, 0.0, 0.0], [-31.48, 0.0, 0.0]]
        )
    )
    with pytest.raises(JointRewardError, match="did not reproduce"):
        _evaluator().evaluate(wrong, (), candidates)
    with pytest.raises(JointRewardError, match="exactly"):
        _evaluator().evaluate(_spec(), ({"agent0": _trajectory(3.0)},), candidates)
