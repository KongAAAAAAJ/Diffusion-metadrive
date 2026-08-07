from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from metadrive.component.lane.junction_lane import (
    build_lane_seam_drivable_surface,
    build_lane_seam_transition_centerline,
)
from models.platoon_planner.platoon_normal_planner import (
    CommittedTrajectoryError,
    JointTrajectoryExecutor,
    JointTrajectoryExecutionPlan,
    NormalPlannerNoFeasiblePlan,
    PlatoonNormalPlanner,
    TrajectoryExecutionSpec,
    audit_dense_footprint_on_lanes,
    _Neighbor,
    _TrafficEnvelope,
    _TrajectoryCandidate,
)
from models.platoon_planner.route_chain_geometry import (
    build_continuous_lane_chain_path,
)
from models.decisioner.rule_decisioner import JointActionProposal
from scenarios.orchestrator import ScenarioOrchestrator


class FakeLane:
    def __init__(self, lane_id: int, y: float, length: float = 120.0, width: float = 3.5):
        self.index = ("A", "B", lane_id)
        self.length = float(length)
        self.width = float(width)
        self.y = float(y)

    def local_coordinates(self, position):
        return float(position[0]), float(position[1] - self.y)

    def position(self, longitudinal: float, lateral: float):
        return np.asarray([float(longitudinal), self.y + float(lateral)], dtype=np.float32)

    def heading_theta_at(self, longitudinal: float):  # noqa: ARG002
        return 0.0


class AngledLane:
    def __init__(self, index, start, heading, length=10.0, width=3.5):
        self.index = tuple(index)
        self.start = np.asarray(start, dtype=np.float64)
        self.heading = float(heading)
        self.length = float(length)
        self.width = float(width)
        self.direction = np.asarray(
            [np.cos(self.heading), np.sin(self.heading)], dtype=np.float64
        )
        self.direction_lateral = np.asarray(
            [-np.sin(self.heading), np.cos(self.heading)], dtype=np.float64
        )

    def local_coordinates(self, position):
        delta = np.asarray(position, dtype=np.float64) - self.start
        return (
            float(np.dot(delta, self.direction)),
            float(np.dot(delta, self.direction_lateral)),
        )

    def position(self, longitudinal: float, lateral: float):
        return (
            self.start
            + float(longitudinal) * self.direction
            + float(lateral) * self.direction_lateral
        )

    def heading_theta_at(self, longitudinal: float):  # noqa: ARG002
        return self.heading

    def width_at(self, longitudinal: float):  # noqa: ARG002
        return self.width


class FakeRoadNetwork:
    def __init__(self):
        self._lanes = {
            ("A", "B", 0): FakeLane(0, 3.5),
            ("A", "B", 1): FakeLane(1, 0.0),
            ("A", "B", 2): FakeLane(2, -3.5),
        }

    def get_lane(self, lane_index):
        return self._lanes[tuple(lane_index)]


class S8HardcodedTargetFakeLane(FakeLane):
    def __init__(self, from_node: str, to_node: str, lane_id: int, y: float):
        super().__init__(lane_id, y)
        self.index = (from_node, to_node, lane_id)


class S8HardcodedTargetFakeRoadNetwork:
    def __init__(self):
        self._lanes = {
            ("3C0_1_", "4G0_0_", 2): S8HardcodedTargetFakeLane("3C0_1_", "4G0_0_", 2, -3.5),
            ("3C0_1_", "4G1_0_", 0): S8HardcodedTargetFakeLane("3C0_1_", "4G1_0_", 0, -7.0),
            ("4G0_0_", "4G1_1_", 0): S8HardcodedTargetFakeLane("4G0_0_", "4G1_1_", 0, -3.5),
        }

    def get_lane(self, lane_index):
        return self._lanes[tuple(lane_index)]


def _vehicle(
    name: str,
    x: float,
    y: float,
    lane,
    speed_km_h: float = 18.0,
    velocity: tuple[float, float] = (0.0, 0.0),
    heading_theta: float = 0.0,
):
    return SimpleNamespace(
        name=name,
        position=np.asarray([x, y], dtype=np.float32),
        heading_theta=float(heading_theta),
        speed_km_h=float(speed_km_h),
        velocity=np.asarray(velocity, dtype=np.float32),
        LENGTH=5.74,
        WIDTH=2.3,
        lane=lane,
        lane_index=lane.index,
    )


def _env(agent_lane_id: int = 1):
    road_network = FakeRoadNetwork()
    lane = road_network.get_lane(("A", "B", agent_lane_id))
    agents = {
        "agent0": _vehicle("agent0", 10.0, lane.y, lane),
    }
    return SimpleNamespace(
        agents=agents,
        engine=SimpleNamespace(current_map=SimpleNamespace(road_network=road_network)),
    )


def _env_s8_hardcoded_target():
    road_network = S8HardcodedTargetFakeRoadNetwork()
    return SimpleNamespace(
        engine=SimpleNamespace(current_map=SimpleNamespace(road_network=road_network)),
    )


def _execution_fixture():
    env = _env(agent_lane_id=1)
    lane = env.agents["agent0"].lane
    env.agents = {
        "agent0": _vehicle("agent0", 50.5, lane.y, lane, speed_km_h=18.0),
        "agent1": _vehicle("agent1", 30.5, lane.y, lane, speed_km_h=18.0),
        "agent2": _vehicle("agent2", 10.5, lane.y, lane, speed_km_h=18.0),
    }
    env.config = {"physics_world_step_size": 0.02, "decision_repeat": 5}
    env._scenario_step_count = 1
    times = np.arange(0.0, 8.2, 0.1, dtype=np.float64)
    specs = {}
    for agent_id, start_x in zip(env.agents, (50.0, 30.0, 10.0)):
        trajectory = np.column_stack(
            (
                start_x + 5.0 * times,
                np.zeros_like(times),
                np.zeros_like(times),
            )
        )
        specs[agent_id] = TrajectoryExecutionSpec(
            agent_id=agent_id,
            source_lane_index=lane.index,
            continuation_lane_index=(),
            target_lane_index=lane.index,
            start_s=start_x,
            start_d=0.0,
            end_d=0.0,
            initial_speed_mps=5.0,
            acceleration_mps2=0.0,
            acceleration_duration_s=4.0,
            recovery_acceleration_mps2=0.0,
            lane_change_duration_s=4.0,
            lane_change_start_delay_s=0.0,
            default_heading=0.0,
            selected_candidate_index=0,
            rule_target_point=(20.0, 0.0),
            sample_times_s=times,
            trajectory_world=trajectory,
        )
    plan = JointTrajectoryExecutionPlan(
        execution_id=7,
        proposal_id=2,
        proposal_rank=1,
        start_step=0,
        start_time_s=0.0,
        rule_actions={"agent0": -1, "agent1": 0, "agent2": 0},
        agent_specs=specs,
        committed_agents=("agent0",),
        completion_deadline_s=4.0,
    )
    return env, plan


def test_joint_trajectory_executor_rolls_absolute_time_without_restart():
    env, plan = _execution_fixture()
    executor = JointTrajectoryExecutor(PlatoonNormalPlanner())
    executor.start(env, plan)

    result = executor.roll(env)

    assert result.execution_id == 7
    assert result.elapsed_s == pytest.approx(0.1)
    assert result.trajectories_world["agent0"][0, 0] == pytest.approx(53.0)
    assert result.trajectories_world["agent0"][-1, 0] == pytest.approx(70.5)
    assert result.debug["trajectory_source"] == "committed_roll"
    assert result.debug["rolling_hard_audit"] == "passed"


def test_route_chain_geometry_smoothly_joins_offset_exit_connector():
    source = AngledLane(("A", "B", 2), (0.0, 0.0), 0.0, length=20.0)
    connector = AngledLane(("B", "C", 0), (20.0, -3.5), 0.0, length=10.0)
    target = AngledLane(("C", "D", 0), (30.0, -3.5), 0.0, length=15.0)

    path = build_continuous_lane_chain_path(
        [source, connector, target], start_s=0.0, step_m=0.25
    )

    spacing = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
    assert float(spacing.max()) <= 0.5 + 1e-6
    assert np.all(np.diff(path[:, 0]) >= -1e-6)
    assert path[0, :2] == pytest.approx([0.0, 0.0])
    assert path[-1, :2] == pytest.approx([45.0, -3.5])
    assert np.any((path[:, 1] < -0.1) & (path[:, 1] > -3.4))


def test_execution_spec_separates_stopped_reference_from_extended_spatial_path():
    times = np.arange(0.0, 8.1, 0.1, dtype=np.float64)
    stop_x = np.minimum(5.0 * times, 2.778)
    nominal = np.column_stack((stop_x, np.zeros_like(times), np.zeros_like(times)))
    spatial_x = np.arange(0.0, 30.25, 0.25)
    spatial = np.column_stack(
        (spatial_x, np.zeros_like(spatial_x), np.zeros_like(spatial_x))
    )

    spec = TrajectoryExecutionSpec(
        agent_id="agent0",
        source_lane_index=("A", "B", 0),
        continuation_lane_index=(),
        target_lane_index=("A", "B", 0),
        start_s=0.0,
        start_d=0.0,
        end_d=0.0,
        initial_speed_mps=5.0,
        acceleration_mps2=-4.5,
        acceleration_duration_s=4.0,
        recovery_acceleration_mps2=0.0,
        lane_change_duration_s=4.0,
        lane_change_start_delay_s=0.0,
        default_heading=0.0,
        selected_candidate_index=0,
        rule_target_point=(2.778, 0.0),
        sample_times_s=times,
        trajectory_world=nominal,
        spatial_path_world=spatial,
    )

    assert spec.reference_arc_m[-1] == pytest.approx(2.778)
    assert spec.path_arc_m[-1] == pytest.approx(30.0)


def test_execution_spec_sampling_uses_original_absolute_lateral_curve():
    env, plan = _execution_fixture()
    del env
    spec = plan.agent_specs["agent0"]
    curved = TrajectoryExecutionSpec(
        **{
            **{
                key: value
                for key, value in spec.__dict__.items()
                if key not in {"path_arc_m", "reference_arc_m"}
            },
            "trajectory_world": np.column_stack(
                (
                    spec.trajectory_world[:, 0],
                    spec.sample_times_s**2,
                    np.zeros_like(spec.sample_times_s),
                )
            ),
        }
    )

    shifted = JointTrajectoryExecutor._sample_spec(curved, np.asarray([0.6, 4.1]))

    assert shifted[:, 1] == pytest.approx([0.36, 16.81])
    assert shifted[0, 1] != pytest.approx(0.25)


def test_joint_trajectory_executor_rejects_tracking_deviation():
    env, plan = _execution_fixture()
    env.agents["agent2"].position[0] += 2.0
    executor = JointTrajectoryExecutor(PlatoonNormalPlanner())
    executor.start(env, plan)

    with pytest.raises(CommittedTrajectoryError) as error:
        executor.roll(env)

    assert error.value.reason_code == "committed_trajectory_tracking_deviation"


def test_default_hard_safety_gaps_match_collection_contract():
    planner = PlatoonNormalPlanner()

    assert planner.background_safe_gap_m == 5.0
    assert planner.platoon_safe_gap_m == 7.0


def test_candidate_pool_rejects_rotated_footprint_before_joint_search():
    env = _env(agent_lane_id=0)
    planner = PlatoonNormalPlanner()
    vehicle = env.agents["agent0"]

    pool, debug = planner._generate_candidate_pool(
        env,
        vehicle,
        1,
        np.asarray([20.0, -3.5], dtype=np.float32),
    )

    assert debug["road_rejection_count"] > 0
    assert debug["road_rejections_by_reason"][
        "footprint_point_outside_execution_lanes"
    ] > 0
    lanes = (
        env.engine.current_map.road_network.get_lane(("A", "B", 0)),
        env.engine.current_map.road_network.get_lane(("A", "B", 1)),
    )
    for candidate in pool:
        valid, detail = audit_dense_footprint_on_lanes(
            candidate.dense,
            lanes,
            planner._vehicle_dimensions(vehicle),
            dense_dt_s=planner.DENSE_DT_S,
        )
        assert valid, detail


def test_candidate_and_executor_share_identical_footprint_contract():
    env, plan = _execution_fixture()
    planner = PlatoonNormalPlanner()
    executor = JointTrajectoryExecutor(planner)
    spec = plan.agent_specs["agent0"]
    vehicle = env.agents["agent0"]
    trajectory = spec.trajectory_world[:41].copy()
    trajectory[5, 2] = -0.35
    trajectory[5, 1] = 1.65
    lane = env.engine.current_map.road_network.get_lane(spec.source_lane_index)

    candidate_result = audit_dense_footprint_on_lanes(
        trajectory,
        (lane,),
        planner._vehicle_dimensions(vehicle),
        dense_dt_s=planner.DENSE_DT_S,
    )
    executor_result = executor._footprint_road_audit(
        env, vehicle, trajectory, spec
    )

    assert candidate_result == executor_result
    assert candidate_result[0] is False


def test_dense_footprint_accepts_connected_lane_seam_without_boundary_relaxation():
    predecessor = AngledLane(("A", "B", 0), (0.0, 0.0), 0.0)
    successor = AngledLane(("B", "C", 0), (10.0, 0.0), 0.1)
    trajectory = np.asarray([[7.24, 0.0, 0.0]], dtype=np.float64)

    connected = audit_dense_footprint_on_lanes(
        trajectory,
        (predecessor, successor),
        (5.74, 2.3),
        dense_dt_s=0.1,
    )
    disconnected = audit_dense_footprint_on_lanes(
        trajectory,
        (
            predecessor,
            AngledLane(("X", "C", 0), (10.0, 0.0), 0.1),
        ),
        (5.74, 2.3),
        dense_dt_s=0.1,
    )

    assert connected == (True, {"reason": "passed"})
    assert disconnected[0] is False


def test_offset_g_block_seam_uses_explicit_drivable_surface_without_widening():
    predecessor = AngledLane(("A", "B", 2), (0.0, 0.0), 0.0, length=20.0)
    successor = AngledLane(("B", "C", 0), (20.0, -3.5), 0.0, length=30.0)
    transition = build_lane_seam_transition_centerline(
        predecessor, successor, transition_m=8.0, step_m=0.1
    )
    surface = build_lane_seam_drivable_surface(
        predecessor, successor, transition_m=8.0, step_m=0.1
    )
    surface.index = ("junction", "surface", 0)

    without_surface = audit_dense_footprint_on_lanes(
        transition,
        (predecessor, successor),
        (5.74, 2.3),
        dense_dt_s=0.1,
    )
    predecessor.junction_drivable_surfaces = (surface,)
    with_surface = audit_dense_footprint_on_lanes(
        transition,
        (predecessor, successor),
        (5.74, 2.3),
        dense_dt_s=0.1,
    )

    assert without_surface[0] is False
    assert with_surface == (True, {"reason": "passed"})
    assert surface.width == pytest.approx(3.5)
    assert surface.need_lane_localization is False


def test_offset_seam_path_does_not_backtrack_when_starting_inside_transition():
    predecessor = AngledLane(("A", "B", 2), (0.0, 0.0), 0.0, length=20.0)
    successor = AngledLane(("B", "C", 0), (20.0, -3.5), 0.0, length=30.0)

    path = build_continuous_lane_chain_path(
        (predecessor, successor),
        start_s=16.0,
        step_m=0.1,
        seam_transition_m=8.0,
    )

    np.testing.assert_allclose(path[0, :2], predecessor.position(16.0, 0.0), atol=1e-8)
    assert float(np.min(path[:, 0])) >= 16.0 - 1e-8


def test_ranked_planner_uses_first_rule_rank_with_native_trajectory(monkeypatch):
    env = _env(agent_lane_id=1)
    planner = PlatoonNormalPlanner()
    generated_actions = []
    output = np.column_stack(
        [
            np.arange(1, 9, dtype=np.float32) * 2.5 + 10.0,
            np.zeros(8, dtype=np.float32),
            np.zeros(8, dtype=np.float32),
        ]
    )
    dense = np.column_stack(
        [
            np.linspace(10.0, 30.0, 41),
            np.zeros(41),
            np.zeros(41),
        ]
    )
    candidate = _TrajectoryCandidate(
        dense=dense,
        output=output,
        score=0.0,
        acceleration_mps2=0.0,
        acceleration_duration_s=4.0,
        recovery_acceleration_mps2=0.0,
        lane_change_duration_s=4.0,
        lane_change_start_delay_s=0.0,
        stop_time_s=None,
        terminal_progress_m=20.0,
    )

    def fake_pool(_env, _vehicle, action, _target, **_kwargs):
        generated_actions.append(int(action))
        if int(action) == -1:
            return [], {"fallback_used": True, "fallback_reason": "no_safe_candidate"}
        return [candidate], {
            "fallback_used": False,
            "fallback_reason": None,
            "candidate_count": 1,
            "candidates": [{"selected": False}],
        }

    monkeypatch.setattr(planner, "_generate_candidate_pool", fake_pool)
    common = {
        "target_point": np.asarray([20.0, 0.0], dtype=np.float32),
        "source_lane_index": ("A", "B", 1),
        "target_lane_index": ("A", "B", 1),
        "formation_constraint_enabled": False,
        "coordination_mode": "EMERGENCY_INDEPENDENT",
    }
    proposals = (
        JointActionProposal(0, 0, 10.0, {"agent0": {**common, "action": -1}}),
        JointActionProposal(1, 1, 5.0, {"agent0": {**common, "action": 0}}),
    )

    result = planner.plan_ranked(env, proposals)

    assert result.proposal_id == 1
    assert result.proposal_rank == 1
    assert generated_actions == [-1, 0]
    assert planner.get_last_debug()["_ranked"]["proposal_attempts"][0][
        "native_feasible"
    ] is False


def test_ranked_planner_caches_repeated_infeasible_action_pool(monkeypatch):
    env = _env(agent_lane_id=1)
    planner = PlatoonNormalPlanner()
    calls = []

    def no_pool(_env, _vehicle, action, _target, **_kwargs):
        calls.append(int(action))
        return [], {"fallback_used": True, "fallback_reason": "no_safe_candidate"}

    monkeypatch.setattr(planner, "_generate_candidate_pool", no_pool)
    decision = {
        "agent0": {
            "action": -1,
            "target_point": np.asarray([20.0, 3.5], dtype=np.float32),
            "target_lane_index": ("A", "B", 0),
            "formation_constraint_enabled": False,
        }
    }
    proposals = (
        JointActionProposal(0, 0, 2.0, decision),
        JointActionProposal(1, 1, 1.0, decision),
    )

    with pytest.raises(NormalPlannerNoFeasiblePlan) as error:
        planner.plan_ranked(env, proposals)

    assert error.value.reason_code == "all_rule_proposals_infeasible"
    assert calls == [-1]
    assert error.value.debug["pool_cache_hit_count"] == 1


def test_keep_lateral_search_includes_current_and_desired_offsets():
    planner = PlatoonNormalPlanner()
    lane = FakeLane(1, 0.0)

    targets = planner._candidate_lateral_targets(
        0,
        -0.9,
        0.4,
        lane,
        lane,
    )

    assert targets == (-0.9, 0.4, 0.15, 0.65)


def test_lane_end_restriction_adds_urgent_zero_delay_durations():
    planner = PlatoonNormalPlanner()

    assert planner._lane_end_restricted(
        action=-1,
        ego_speed_mps=8.0,
        usable_source_progress_m=5.0,
    )
    assert planner._lane_change_durations(
        action=-1,
        lane_end_restricted=True,
    ) == (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)


def test_committed_lane_change_uses_absolute_remaining_deadline():
    planner = PlatoonNormalPlanner()

    first, first_remaining = planner._committed_lane_change_durations(
        planner.LANE_CHANGE_DURATIONS_S, 0.1
    )
    later, later_remaining = planner._committed_lane_change_durations(
        planner.LANE_CHANGE_DURATIONS_S, 3.5
    )

    assert first_remaining == pytest.approx(4.9)
    assert max(first) == pytest.approx(4.9)
    assert later_remaining == pytest.approx(1.5)
    assert later == (1.5,)
    assert planner._lane_change_durations(
        action=-1,
        lane_end_restricted=False,
    ) == planner.LANE_CHANGE_DURATIONS_S
    assert not planner._lane_end_restricted(
        action=0,
        ego_speed_mps=8.0,
        usable_source_progress_m=1.0,
    )


def test_lane_change_completion_progress_uses_candidate_progress():
    planner = PlatoonNormalPlanner()
    progress = planner._longitudinal_progress(
        1.0,
        -8.0,
        planner._dense_times,
    )

    completion = planner._lane_change_completion_progress(
        progress,
        duration_s=1.0,
        start_delay_s=0.0,
    )

    expected = float(np.interp(1.0, planner._dense_times, progress))
    assert completion == pytest.approx(expected)


def test_keep_returns_8x3_forward_trajectory():
    env = _env(agent_lane_id=1)
    planner = PlatoonNormalPlanner()

    result = planner.plan(
        env,
        {"agent0": {"action": 0, "target_point": np.asarray([25.0, 0.0], dtype=np.float32)}},
    )

    traj = result["agent0"]
    assert traj.shape == (8, 3)
    assert traj.dtype == np.float32
    assert traj[-1, 0] > traj[0, 0]
    assert np.allclose(traj[:, 1], 0.0, atol=0.6)
    assert np.max(np.abs(np.diff(traj[:, 2]))) < 0.2


def test_left_action_converges_toward_left_lane_center():
    env = _env(agent_lane_id=1)
    planner = PlatoonNormalPlanner()

    result = planner.plan(
        env,
        {"agent0": {"action": -1, "target_point": np.asarray([25.0, 0.0], dtype=np.float32)}},
    )

    traj = result["agent0"]
    assert traj[-1, 1] > 2.5


def test_right_action_converges_toward_right_lane_center():
    env = _env(agent_lane_id=1)
    planner = PlatoonNormalPlanner()

    result = planner.plan(
        env,
        {"agent0": {"action": 1, "target_point": np.asarray([25.0, 0.0], dtype=np.float32)}},
    )

    traj = result["agent0"]
    assert traj[-1, 1] < -2.5


def test_missing_adjacent_lane_falls_back_to_keep():
    env = _env(agent_lane_id=0)
    planner = PlatoonNormalPlanner()

    result = planner.plan(
        env,
        {"agent0": {"action": -1, "target_point": np.asarray([25.0, 0.0], dtype=np.float32)}},
    )

    traj = result["agent0"]
    assert np.allclose(traj[:, 1], 3.5, atol=0.6)


def test_resolve_target_lane_rejects_s8_wrong_upstream_branch():
    env = _env_s8_hardcoded_target()
    source_lane = env.engine.current_map.road_network.get_lane(("3C0_1_", "4G0_0_", 2))

    target_lane = PlatoonNormalPlanner._resolve_target_lane(env, source_lane, action=1)

    assert target_lane is None


def test_resolve_target_lane_rejects_negative_lane_id():
    env = _env(agent_lane_id=0)
    source_lane = env.engine.current_map.road_network.get_lane(("A", "B", 0))

    assert PlatoonNormalPlanner._resolve_target_lane(
        env, source_lane, action=-1
    ) is None


def test_target_point_changes_terminal_progress():
    env = _env(agent_lane_id=1)
    planner = PlatoonNormalPlanner()

    near = planner.plan(
        env,
        {"agent0": {"action": 0, "target_point": np.asarray([16.0, 0.0], dtype=np.float32)}},
    )["agent0"]
    far = planner.plan(
        env,
        {"agent0": {"action": 0, "target_point": np.asarray([40.0, 0.0], dtype=np.float32)}},
    )["agent0"]

    assert far[-1, 0] > near[-1, 0]


def test_candidate_failure_falls_back_to_keep_lane_trajectory():
    env = _env(agent_lane_id=1)
    lane = env.agents["agent0"].lane
    blocker = _vehicle("blocker", 30.0, 0.0, lane, speed_km_h=0.0)
    blocker.LENGTH = 200.0
    blocker.WIDTH = 20.0
    env.engine.traffic_manager = SimpleNamespace(_traffic_vehicles=[blocker])
    planner = PlatoonNormalPlanner()

    result = planner.plan(
        env,
        {"agent0": {"action": 0, "target_point": np.asarray([20.0, 0.0], dtype=np.float32)}},
    )
    debug = planner.get_last_debug()

    traj = result["agent0"]
    assert traj.shape == (8, 3)
    assert np.allclose(traj[:, 1], 0.0, atol=0.6)
    assert debug is not None
    assert debug["agent0"]["fallback_used"] is True


def test_debug_records_lattice_candidates_and_selected_trajectory():
    env = _env(agent_lane_id=1)
    planner = PlatoonNormalPlanner()

    result = planner.plan(
        env,
        {"agent0": {"action": 0, "target_point": np.asarray([25.0, 0.0], dtype=np.float32)}},
    )
    debug = planner.get_last_debug()

    assert debug is not None
    agent_debug = debug["agent0"]
    assert agent_debug["fallback_used"] is False
    assert agent_debug["candidate_count"] == len(agent_debug["candidates"])
    assert agent_debug["candidate_count"] > 1
    selected = [candidate for candidate in agent_debug["candidates"] if candidate["selected"]]
    assert len(selected) == 1
    assert np.allclose(np.asarray(selected[0]["trajectory_world"], dtype=np.float32), result["agent0"])
    assert all("score" in candidate for candidate in agent_debug["candidates"])


def test_score_candidate_adds_safety_distance_penalty_for_nearby_agent():
    env = _env(agent_lane_id=1)
    ego = env.agents["agent0"]
    lane = ego.lane
    env.agents["traffic"] = _vehicle("traffic", 24.0, 0.0, lane, speed_km_h=18.0)
    planner = PlatoonNormalPlanner(safety_distance_m=8.0, safety_weight=10.0, ttc_weight=0.0)
    candidate = np.asarray(
        [
            [10.0, 0.0, 0.0],
            [18.0, 0.0, 0.0],
            [24.0, 0.0, 0.0],
            [32.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    score_with_agent = planner._score_candidate(
        candidate,
        target_world=np.asarray([32.0, 0.0], dtype=np.float32),
        source_lane=lane,
        target_lane=lane,
        action=0,
        desired_end_d=0.0,
        env=env,
        vehicle=ego,
        duration=4.0,
    )
    env.agents.pop("traffic")
    score_without_agent = planner._score_candidate(
        candidate,
        target_world=np.asarray([32.0, 0.0], dtype=np.float32),
        source_lane=lane,
        target_lane=lane,
        action=0,
        desired_end_d=0.0,
        env=env,
        vehicle=ego,
        duration=4.0,
    )

    assert score_with_agent > score_without_agent + 9.0


def test_score_candidate_adds_ttc_penalty_for_slow_lead_vehicle():
    env = _env(agent_lane_id=1)
    ego = env.agents["agent0"]
    ego.speed_km_h = 36.0
    lane = ego.lane
    env.agents["slow_lead"] = _vehicle("slow_lead", 22.0, 0.0, lane, speed_km_h=0.0)
    planner = PlatoonNormalPlanner(safety_weight=0.0, ttc_threshold_s=3.0, ttc_weight=12.0)
    candidate = np.asarray(
        [
            [10.0, 0.0, 0.0],
            [18.0, 0.0, 0.0],
            [26.0, 0.0, 0.0],
            [34.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    slow_score = planner._score_candidate(
        candidate,
        target_world=np.asarray([34.0, 0.0], dtype=np.float32),
        source_lane=lane,
        target_lane=lane,
        action=0,
        desired_end_d=0.0,
        env=env,
        vehicle=ego,
        duration=4.0,
    )
    env.agents["slow_lead"].speed_km_h = 60.0
    fast_score = planner._score_candidate(
        candidate,
        target_world=np.asarray([34.0, 0.0], dtype=np.float32),
        source_lane=lane,
        target_lane=lane,
        action=0,
        desired_end_d=0.0,
        env=env,
        vehicle=ego,
        duration=4.0,
    )

    assert slow_score > fast_score + 3.0


def test_candidate_collision_detects_static_vehicle_obb_overlap():
    env = _env(agent_lane_id=1)
    ego = env.agents["agent0"]
    lane = ego.lane
    env.agents["stopped"] = _vehicle(
        "stopped",
        18.0,
        0.0,
        lane,
        speed_km_h=0.0,
        velocity=(0.0, 0.0),
    )
    planner = PlatoonNormalPlanner(collision_margin_m=0.0)
    candidate = np.asarray(
        [
            [10.0, 0.0, 0.0],
            [14.0, 0.0, 0.0],
            [18.0, 0.0, 0.0],
            [22.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    assert planner._candidate_collides_with_predicted_vehicles(candidate, env=env, vehicle=ego, duration=3.0)


def test_candidate_collision_uses_velocity_without_heading_fallback():
    env = _env(agent_lane_id=1)
    ego = env.agents["agent0"]
    lane = ego.lane
    env.agents["crossing"] = _vehicle(
        "crossing",
        18.0,
        -8.0,
        lane,
        velocity=(0.0, 4.0),
        heading_theta=np.pi,
    )
    env.agents["crossing"].lane = None
    env.agents["crossing"].lane_index = None
    planner = PlatoonNormalPlanner(collision_margin_m=0.0)
    candidate = np.asarray(
        [
            [10.0, 0.0, 0.0],
            [14.0, 0.0, 0.0],
            [18.0, 0.0, 0.0],
            [22.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    assert planner._candidate_collides_with_predicted_vehicles(candidate, env=env, vehicle=ego, duration=2.0)


def test_candidate_collision_ignores_vehicle_outside_aabb_width():
    env = _env(agent_lane_id=1)
    ego = env.agents["agent0"]
    lane = ego.lane
    env.agents["adjacent"] = _vehicle("adjacent", 18.0, 4.0, lane, velocity=(0.0, 0.0))
    planner = PlatoonNormalPlanner(collision_margin_m=0.0)
    candidate = np.asarray(
        [
            [10.0, 0.0, 0.0],
            [14.0, 0.0, 0.0],
            [18.0, 0.0, 0.0],
            [22.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    assert not planner._candidate_collides_with_predicted_vehicles(candidate, env=env, vehicle=ego, duration=3.0)


def test_candidate_collision_checks_traffic_manager_vehicles():
    env = _env(agent_lane_id=1)
    ego = env.agents["agent0"]
    lane = ego.lane
    traffic_vehicle = _vehicle(
        "traffic",
        18.0,
        0.0,
        lane,
        speed_km_h=0.0,
        velocity=(0.0, 0.0),
    )
    env.engine.traffic_manager = SimpleNamespace(_traffic_vehicles=[traffic_vehicle])
    planner = PlatoonNormalPlanner(collision_margin_m=0.0)
    candidate = np.asarray(
        [
            [10.0, 0.0, 0.0],
            [14.0, 0.0, 0.0],
            [18.0, 0.0, 0.0],
            [22.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    assert planner._candidate_collides_with_predicted_vehicles(candidate, env=env, vehicle=ego, duration=3.0)


def test_plan_filters_colliding_candidates_before_scoring_and_falls_back():
    env = _env(agent_lane_id=1)
    ego = env.agents["agent0"]
    lane = ego.lane
    blocker = _vehicle(
        "blocker",
        30.0,
        0.0,
        lane,
        speed_km_h=0.0,
        velocity=(0.0, 0.0),
    )
    blocker.LENGTH = 200.0
    blocker.WIDTH = 20.0
    env.engine.traffic_manager = SimpleNamespace(_traffic_vehicles=[blocker])
    planner = PlatoonNormalPlanner(collision_margin_m=0.0)

    result = planner.plan(
        env,
        {"agent0": {"action": 0, "target_point": np.asarray([25.0, 0.0], dtype=np.float32)}},
    )
    debug = planner.get_last_debug()

    assert result["agent0"].shape == (8, 3)
    assert debug is not None
    assert debug["agent0"]["fallback_used"] is True
    assert debug["agent0"]["fallback_reason"] == "no_safe_candidate"


def test_output_indices_are_exact_half_second_future_timestamps():
    planner = PlatoonNormalPlanner()

    sampled_times = planner._dense_times[planner._output_indices]

    np.testing.assert_allclose(sampled_times, np.arange(1, 9) * 0.5)


def test_emergency_braking_stops_without_reversing():
    planner = PlatoonNormalPlanner()

    progress = planner._longitudinal_progress(
        8.0,
        -8.0,
        planner._dense_times,
    )

    assert np.all(np.diff(progress) >= -1e-9)
    np.testing.assert_allclose(progress[planner._dense_times >= 1.0], 4.0)


def test_front_vehicle_shrinks_terminal_progress_corridor():
    planner = PlatoonNormalPlanner()
    env = _env()
    ego = env.agents["agent0"]
    lane = ego.lane
    near = _vehicle("near", 22.0, lane.y, lane, speed_km_h=0.0)
    far = _vehicle("far", 42.0, lane.y, lane, speed_km_h=0.0)

    near_upper = planner._safe_terminal_corridor(
        _TrafficEnvelope(
            front=_Neighbor("near", near, False, 12.0, 6.26, 0.0, 1.0),
            rear=None,
        ),
        ego,
        4.0,
    )[1]
    far_upper = planner._safe_terminal_corridor(
        _TrafficEnvelope(
            front=_Neighbor("far", far, False, 32.0, 26.26, 0.0, 1.0),
            rear=None,
        ),
        ego,
        4.0,
    )[1]

    assert near_upper < far_upper


def test_fast_rear_vehicle_excludes_unsafe_terminal_braking_progress():
    planner = PlatoonNormalPlanner()
    env = _env()
    ego = env.agents["agent0"]
    lane = ego.lane
    rear = _vehicle("rear", 0.0, lane.y, lane, speed_km_h=54.0)
    lower, _ = planner._safe_terminal_corridor(
        _TrafficEnvelope(
            front=None,
            rear=_Neighbor("rear", rear, False, -10.0, 4.26, 15.0, 0.5),
        ),
        ego,
        4.0,
    )
    emergency_progress = planner._longitudinal_progress(
        5.0,
        -8.0,
        np.asarray([4.0]),
    )[0]

    assert emergency_progress < lower


def test_search_corridor_allows_yielding_behind_a_closing_rear_vehicle():
    planner = PlatoonNormalPlanner()
    env = _env()
    ego = env.agents["agent0"]
    lane = ego.lane
    rear = _vehicle("rear", 0.0, lane.y, lane, speed_km_h=54.0)
    envelope = _TrafficEnvelope(
        front=None,
        rear=_Neighbor("rear", rear, False, -10.0, 4.26, 15.0, 0.5),
    )

    hard_lower, _ = planner._safe_terminal_corridor(envelope, ego, 4.0)
    search_lower, _ = planner._terminal_search_corridor(envelope, ego, 4.0)

    assert np.isfinite(hard_lower)
    assert search_lower == -float("inf")


def test_profile_selection_preserves_brake_wait_recover_shape():
    planner = PlatoonNormalPlanner()
    profiles = []
    for acceleration, duration, recovery in (
        (0.0, 4.0, 0.0),
        (-2.0, 4.0, 0.0),
        (-4.0, 1.0, 0.0),
        (-6.0, 1.0, 1.5),
        (-8.0, 2.0, 3.0),
        (-8.0, 4.0, 0.0),
        (2.0, 4.0, 0.0),
    ):
        progress = planner._longitudinal_progress(
            5.0,
            acceleration,
            planner._dense_times,
            acceleration_duration_s=duration,
            recovery_acceleration_mps2=recovery,
        )
        profiles.append((abs(acceleration), acceleration, duration, recovery, progress))
    profiles.sort(key=lambda value: value[0])

    selected = planner._select_longitudinal_profiles(profiles, maximum=6)

    assert any(
        acceleration == -8.0 and duration == 2.0 and recovery == 3.0
        for _, acceleration, duration, recovery, _ in selected
    )
    assert any(
        acceleration == -8.0 and duration == 4.0 and recovery == 0.0
        for _, acceleration, duration, recovery, _ in selected
    )


def test_tight_target_lane_gap_delays_lane_change_start():
    planner = PlatoonNormalPlanner()
    env = _env()
    lane = env.agents["agent0"].lane
    front = _vehicle("front", 18.0, lane.y, lane)
    tight = _TrafficEnvelope(
        front=_Neighbor("front", front, False, 8.0, 2.26, 5.0, 2.0),
        rear=None,
    )

    assert planner._lane_change_start_delays(1, 4.0, tight) == (
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
    )
    assert planner._lane_change_start_delays(
        1,
        4.0,
        _TrafficEnvelope(front=None, rear=None),
    )[0] == 0.0


def test_rotated_obb_avoids_axis_aligned_false_positive():
    angle = np.pi / 4.0
    first = np.asarray([[0.0, 0.0, angle]])
    lateral = np.asarray([-np.sin(angle), np.cos(angle)])
    second = np.asarray(
        [[*(lateral * 2.6), angle]],
        dtype=np.float64,
    )

    assert not PlatoonNormalPlanner._obb_overlap_series(
        first,
        (5.74, 2.3),
        second,
        (5.74, 2.3),
        0.0,
    )


def _candidate(x_values, score):
    dense = np.column_stack(
        [
            np.asarray(x_values, dtype=np.float64),
            np.zeros(len(x_values)),
            np.zeros(len(x_values)),
        ]
    )
    return _TrajectoryCandidate(
        dense=dense,
        output=dense[1:].astype(np.float32),
        score=float(score),
        acceleration_mps2=0.0,
        acceleration_duration_s=4.0,
        recovery_acceleration_mps2=0.0,
        lane_change_duration_s=4.0,
        lane_change_start_delay_s=0.0,
        stop_time_s=None,
        terminal_progress_m=float(x_values[-1] - x_values[0]),
    )


def test_joint_selection_replaces_conflicting_local_optimum():
    planner = PlatoonNormalPlanner(collision_margin_m=0.0)
    env = _env()
    lane = env.agents["agent0"].lane
    env.agents = {
        "agent0": _vehicle("agent0", 14.0, 0.0, lane),
        "agent1": _vehicle("agent1", 0.0, 0.0, lane),
    }
    front_fast = _candidate(np.linspace(14.0, 30.0, 9), 0.0)
    front_slow = _candidate(np.linspace(14.0, 22.0, 9), 1.0)
    rear_fast = _candidate(np.linspace(0.0, 24.0, 9), 0.0)
    rear_slow = _candidate(np.linspace(0.0, 8.0, 9), 1.0)

    selection, debug = planner._select_joint_candidates(
        env,
        ["agent0", "agent1"],
        env.agents,
        {
            "agent0": [front_fast, front_slow],
            "agent1": [rear_fast, rear_slow],
        },
    )

    assert selection is not None
    assert selection != (0, 0)
    assert debug["pairwise_conflict_count"] > 0
    assert debug["prefix_counts"] == [2, 2]


def test_independent_joint_selection_skips_formation_penalty(monkeypatch):
    planner = PlatoonNormalPlanner(collision_margin_m=0.0)
    env = _env()
    lane = env.agents["agent0"].lane
    env.agents = {
        "agent0": _vehicle("agent0", 22.0, 0.0, lane),
        "agent1": _vehicle("agent1", 8.0, 0.0, lane),
    }
    candidates = {
        "agent0": [_candidate(np.linspace(22.0, 30.0, 9), 0.0)],
        "agent1": [_candidate(np.linspace(8.0, 16.0, 9), 0.0)],
    }

    def forbidden(*_args, **_kwargs):
        raise AssertionError("formation penalty must be disabled")

    monkeypatch.setattr(planner, "_joint_formation_penalty", forbidden)
    selection, debug = planner._select_joint_candidates(
        env,
        ["agent0", "agent1"],
        env.agents,
        candidates,
        formation_constraint_enabled=False,
    )

    assert selection == (0, 0)
    assert debug["formation_constraint_enabled"] is False
    assert debug["formation_penalty_applied"] is False
    assert debug["prefix_counts"] == [1, 1]


def test_scenario_front_lookup_uses_simulator_state_without_lidar():
    road_network = FakeRoadNetwork()
    lane = road_network.get_lane(("A", "B", 1))
    ego = _vehicle("ego", 10.0, lane.y, lane)
    front = _vehicle("front", 30.0, lane.y, lane)
    adjacent = _vehicle("adjacent", 15.0, lane.y + 4.0, lane)
    engine = SimpleNamespace(
        traffic_manager=SimpleNamespace(
            _traffic_vehicles=[front, adjacent],
        )
    )
    ego.engine = engine

    found, distance = ScenarioOrchestrator._find_front_vehicle_with_distance(
        object(),
        ego,
    )

    assert found is front
    assert distance == pytest.approx(20.0)
