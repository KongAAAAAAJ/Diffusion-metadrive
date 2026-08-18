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
from evaluation.joint_simulator_branch import _has_failure
from models.bev_planner import JointRewardConfig, JointRewardError


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
                heading=np.asarray([1.0, 0.0], dtype=np.float64),
                velocity=np.asarray([0.0, 0.0], dtype=np.float64),
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

    def trajectory_formation_constraint_enabled(self) -> bool:
        return True

    def trajectory_reference_to_control(
        self,
        agent_id,
        trajectory,
        reference,
        *,
        formation_constraint_enabled=True,
    ):
        assert isinstance(formation_constraint_enabled, bool)
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
            vehicle.heading = np.asarray(
                [np.cos(vehicle.heading_theta), np.sin(vehicle.heading_theta)],
                dtype=np.float64,
            )
            vehicle.velocity = vehicle.heading * (vehicle.speed_km_h / 3.6)
        flags = {agent_id: False for agent_id in self.agents}
        flags["__all__"] = False
        info = {agent_id: {} for agent_id in self.agents}
        return {}, {}, flags, flags.copy(), info

    def close(self) -> None:
        self._closed = True


class _CollisionBranchEnv(_BranchEnv):
    def step(self, actions):
        observation, reward, terminated, truncated, info = super().step(actions)
        info["agent0"]["crash_vehicle"] = True
        terminated["__all__"] = True
        return observation, reward, terminated, truncated, info


def _spec(reference=None) -> JointEpisodeSpec:
    if reference is None:
        reference = np.asarray(
            [[0.0, 0.0, 0.0], [-15.74, 0.0, 0.0], [-31.48, 0.0, 0.0]],
            dtype=np.float64,
        )
    return JointEpisodeSpec(
        "S5_hard_brake_lead",
        "R1_entry_straight",
        17,
        reference,
    )


def _evaluator() -> JointSimulatorBranchEvaluator:
    evaluator = JointSimulatorBranchEvaluator(env_factory=_BranchEnv)
    evaluator._vehicle_helper._surrounding_vehicles = lambda _env: []
    evaluator._reference_drivable_bev = lambda _env, _pose: np.full(
        (3, 256, 256), 255, dtype=np.uint8
    )
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
    assert set(result.reward.components) == {
        "progress_score",
        "formation_penalty",
        "gap_penalty",
        "ttc_penalty",
        "road_penalty",
        "comfort_penalty",
        "minimum_background_gap_m",
        "minimum_platoon_gap_m",
        "minimum_road_margin_m",
        "minimum_ttc_s",
    }
    for name in (
        "progress_score",
        "formation_penalty",
        "gap_penalty",
        "ttc_penalty",
        "road_penalty",
        "comfort_penalty",
    ):
        component = result.reward.components[name]
        assert np.isfinite(component).all()
        assert ((0.0 <= component) & (component <= 1.0)).all()
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


def test_closing_gap_produces_continuous_gap_and_ttc_penalties() -> None:
    steady = np.stack([_trajectory(2.0)] * 3)
    closing = np.stack(
        [_trajectory(1.0), _trajectory(3.0), _trajectory(5.0)]
    )
    result = _evaluator().evaluate(
        _spec(),
        (),
        np.stack((steady, closing)),
    )

    gap = result.reward.components["gap_penalty"]
    ttc = result.reward.components["ttc_penalty"]
    assert gap[1] > gap[0]
    assert ttc[1] > ttc[0]
    assert result.reward.components["minimum_ttc_s"][0] == pytest.approx(1.0e6)
    assert result.reward.components["minimum_ttc_s"][1] < 4.0
    assert result.reward.clearance_violation.tolist() == [False, True]
    assert result.reward.rewards[0] != result.reward.rewards[1]


def test_simulator_gap_transform_resamples_to_point_one_seconds() -> None:
    evaluator = _evaluator()
    snapshots = [
        {"background:agent0:vehicle": (5.0, 5.0)},
        {"background:agent0:vehicle": (4.6, 5.0)},
        {"background:agent0:vehicle": (4.6, 5.0)},
    ]
    _, ttc_penalty, minimum_ttc, _, minimum_background = (
        evaluator._continuous_gap_risks(snapshots, dt_s=0.1)
    )
    assert minimum_background == pytest.approx(4.6)
    assert minimum_ttc == pytest.approx(1.15)
    ttc = np.full(3, 1.0e6, dtype=np.float64)
    ttc[1] = 1.15
    from models.bev_planner import (
        aggregate_temporal_risk,
        soft_threshold_risk,
    )

    expected = aggregate_temporal_risk(
        soft_threshold_risk(
            ttc, warning_threshold=4.0, softness=0.5
        ),
        max_weight=0.7,
        mean_weight=0.3,
    )
    assert ttc_penalty == pytest.approx(expected)


def test_simulator_collision_is_additive_and_stops_at_the_real_event() -> None:
    evaluator = JointSimulatorBranchEvaluator(env_factory=_CollisionBranchEnv)
    evaluator._vehicle_helper._surrounding_vehicles = lambda _env: []
    evaluator._reference_drivable_bev = lambda _env, _pose: np.full(
        (3, 256, 256), 255, dtype=np.uint8
    )
    result = evaluator.evaluate(
        _spec(), (), np.stack([np.stack([_trajectory(2.0)] * 3)])
    )

    assert result.executed_steps.tolist() == [1]
    assert result.reward.collision.tolist() == [True]
    assert result.reward.out_of_drivable.tolist() == [False]
    components = result.reward.components
    expected = (
        evaluator.config.progress_weight * components["progress_score"][0]
        - evaluator.config.formation_weight
        * components["formation_penalty"][0]
        - evaluator.config.gap_weight * components["gap_penalty"][0]
        - evaluator.config.ttc_weight * components["ttc_penalty"][0]
        - evaluator.config.road_weight * components["road_penalty"][0]
        - evaluator.config.comfort_weight * components["comfort_penalty"][0]
        - evaluator.config.collision_penalty
    )
    assert result.reward.rewards[0] == pytest.approx(expected)
    assert np.isfinite(tuple(components.values())).all()


def test_tracking_footprint_only_increases_continuous_road_risk() -> None:
    drivable = np.zeros((3, 256, 256), dtype=np.uint8)
    drivable[:, :, 116:140] = 255

    def evaluate(config: JointRewardConfig):
        evaluator = JointSimulatorBranchEvaluator(
            config=config, env_factory=_BranchEnv
        )
        evaluator._vehicle_helper._surrounding_vehicles = lambda _env: []
        evaluator._reference_drivable_bev = (
            lambda _env, _pose: drivable.copy()
        )
        return evaluator.evaluate(
            _spec(), (), np.stack([np.stack([_trajectory(2.0)] * 3)])
        )

    physical = evaluate(JointRewardConfig())
    expanded = evaluate(
        JointRewardConfig(tracking_lateral_margin_m=0.75)
    )

    assert (
        expanded.reward.components["minimum_road_margin_m"][0]
        < physical.reward.components["minimum_road_margin_m"][0]
    )
    assert (
        expanded.reward.components["road_penalty"][0]
        > physical.reward.components["road_penalty"][0]
    )
    assert not physical.reward.out_of_drivable[0]
    assert not expanded.reward.out_of_drivable[0]


def test_simulator_clearance_diagnostics_use_tracking_gap_footprint() -> None:
    evaluator = JointSimulatorBranchEvaluator(
        config=JointRewardConfig(tracking_longitudinal_margin_m=0.5),
        env_factory=_BranchEnv,
    )
    evaluator._vehicle_helper._surrounding_vehicles = lambda _env: []
    evaluator._reference_drivable_bev = lambda _env, _pose: np.full(
        (3, 256, 256), 255, dtype=np.uint8
    )
    result = evaluator.evaluate(
        _spec(), (), np.stack([np.stack([_trajectory(2.0)] * 3)])
    )

    assert result.minimum_platoon_gap_m[0] == pytest.approx(9.0)
    assert result.reward.components["minimum_platoon_gap_m"][0] == pytest.approx(
        9.0
    )


def test_instant_gaps_ignore_adjacent_lane_radial_proximity() -> None:
    evaluator = JointSimulatorBranchEvaluator(env_factory=_BranchEnv)
    env = _BranchEnv({})
    adjacent = SimpleNamespace(
        position=np.asarray([0.0, 3.5], dtype=np.float64),
        heading_theta=0.0,
        LENGTH=5.74,
        WIDTH=2.3,
    )
    evaluator._vehicle_helper._surrounding_vehicles = (
        lambda _env: [("adjacent", adjacent)]
    )

    platoon_gap, adjacent_gap = evaluator._instant_gaps(env)

    assert platoon_gap == pytest.approx(10.0)
    assert adjacent_gap == pytest.approx(evaluator.config.no_risk_gap_m)

    env.agents["agent1"].position = np.asarray([0.0, 3.5])
    env.agents["agent2"].position = np.asarray([0.0, 7.0])
    adjacent_platoon_gap, _ = evaluator._instant_gaps(env)

    assert adjacent_platoon_gap == pytest.approx(
        evaluator.config.no_risk_gap_m
    )

    adjacent.position = np.asarray([10.0, 0.0], dtype=np.float64)
    _, close_gap = evaluator._instant_gaps(env)

    assert close_gap == pytest.approx(4.26)


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
def test_sidewalk_contact_is_a_road_termination_not_a_silent_safe_step():
    collision, out = _has_failure(
        {
            "agent0": {"crash_sidewalk": True},
            "agent1": {},
            "agent2": {},
        }
    )
    assert collision is False
    assert out is True
