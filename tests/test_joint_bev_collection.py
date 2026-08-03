from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from envs.observations.semantic_bev import BEVChannel
from envs.platoon_env import PlatoonEnv
from expert_dataset.collect_joint_bev import (
    AgentRole,
    ExpertJointStep,
    JOINT_SAMPLE_DTYPES,
    JOINT_SAMPLE_SHAPES,
    MODEL_INPUT_FIELDS,
    JointBEVModelInputs,
    JointBEVSample,
    JointBEVSampleBuilder,
    JointCollectionError,
    RulePlannerExpert,
    SensorlessJointBEVPlatoonEnv,
)
from models.bev_planner.dynamic_anchors import (
    DynamicAnchorError,
    SimulatorDynamicAnchorGenerator,
)
from models.bev_planner.mode_contract import (
    LEFT_MODES,
    ModeIndex,
    validate_trajectory_kinematics,
)
from models.decisioner.rule_decisioner import (
    JointActionProposal,
    RuleMakerProposalBatch,
)
from models.platoon_planner.platoon_normal_planner import (
    JointTrajectoryExecutor,
    NormalPlannerNoFeasiblePlan,
    RankedJointPlan,
)


def _rectangle(x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    return np.asarray([[x0, y1], [x1, y1], [x1, y0], [x0, y0]], dtype=np.float32)


class _Lane:
    def __init__(self, lane_id: int, y: float) -> None:
        self.index = ("A", "B", lane_id)
        self.length = 200.0
        self.width = 3.5
        self.y = float(y)
        self.polygon = _rectangle(0.0, self.length, y - self.width / 2, y + self.width / 2)

    def position(self, longitudinal: float, lateral: float) -> np.ndarray:
        return np.asarray([longitudinal, self.y + lateral], dtype=np.float32)

    def local_coordinates(self, position: np.ndarray) -> tuple[float, float]:
        return float(position[0]), float(position[1] - self.y)

    def heading_theta_at(self, longitudinal: float) -> float:
        del longitudinal
        return 0.0

    def width_at(self, longitudinal: float) -> float:
        del longitudinal
        return self.width


class _RoadNetwork:
    def __init__(self) -> None:
        self.lanes = {
            ("A", "B", 0): _Lane(0, 3.5),
            ("A", "B", 1): _Lane(1, 0.0),
            ("A", "B", 2): _Lane(2, -3.5),
        }
        self.graph = {"A": {"B": list(self.lanes.values())}}

    def get_lane(self, index):
        return self.lanes[tuple(index)]

    def get_all_lanes(self):
        return list(self.lanes.values())


def _vehicle(agent_id: str, x: float, lane: _Lane) -> SimpleNamespace:
    navigation = SimpleNamespace(
        checkpoints=["A", "B"],
        _target_checkpoints_index=[0, 0],
        current_ref_lanes=[lane],
        next_ref_lanes=None,
    )
    return SimpleNamespace(
        name=agent_id,
        position=np.asarray([x, lane.y], dtype=np.float32),
        heading_theta=0.0,
        speed_km_h=18.0,
        top_down_length=5.74,
        top_down_width=2.3,
        LENGTH=5.74,
        WIDTH=2.3,
        steering=0.1,
        throttle_brake=0.2,
        lane=lane,
        navigation=navigation,
    )


class _Env:
    def __init__(self) -> None:
        network = _RoadNetwork()
        lane = network.get_lane(("A", "B", 1))
        self.agents = {
            "agent0": _vehicle("agent0", 30.0, lane),
            "agent1": _vehicle("agent1", 20.0, lane),
            "agent2": _vehicle("agent2", 10.0, lane),
        }
        self._agent_ids = list(self.agents)
        self.current_map = SimpleNamespace(road_network=network)
        self.engine = SimpleNamespace(current_map=self.current_map, get_objects=lambda: {})
        self.config = {"target_speed_km_h": 30.0}

    def get_formation_relation_state(self, agent_id: str) -> np.ndarray:
        ego_x = float(self.agents[agent_id].position[0])
        values = []
        for other_id in self._agent_ids:
            if other_id == agent_id:
                continue
            delta_x = float(self.agents[other_id].position[0]) - ego_x
            values.extend([delta_x, 0.0, 0.0, 0.0, -delta_x, 0.0])
        return np.asarray(values, dtype=np.float32)


def _local_to_world(vehicle: object, trajectory: np.ndarray) -> np.ndarray:
    value = np.asarray(trajectory, dtype=np.float32).copy()
    value[:, 0] += float(vehicle.position[0])
    value[:, 1] += float(vehicle.position[1])
    value[:, 2] += float(vehicle.heading_theta)
    return value


def _prime_builder(builder: JointBEVSampleBuilder, env: _Env) -> None:
    builder.capture_state(env, 0.0)
    builder.capture_state(env, 0.5)
    builder.capture_state(env, 1.0)
    assert builder.history_ready()


def _expert_step(
    env: _Env,
    generator: SimulatorDynamicAnchorGenerator,
    actions: dict[str, int],
) -> ExpertJointStep:
    trajectories = {}
    for agent_id, action in actions.items():
        anchors = generator.generate(env, agent_id).coarse_trajectories
        mode = {
            -1: ModeIndex.LEFT_MEDIUM,
            0: ModeIndex.KEEP_MEDIUM,
            1: ModeIndex.RIGHT_MEDIUM,
        }[action]
        trajectories[agent_id] = _local_to_world(env.agents[agent_id], anchors[mode])
    return ExpertJointStep(
        rule_actions=actions,
        trajectories_world=trajectories,
        controls={agent_id: np.zeros(2, dtype=np.float32) for agent_id in env.agents},
    )


def test_dynamic_anchors_have_fixed_modes_and_simulator_topology() -> None:
    env = _Env()
    output = SimulatorDynamicAnchorGenerator().generate(env, "agent0")
    assert output.coarse_trajectories.shape == (10, 8, 3)
    assert output.coarse_trajectories.dtype == np.float32
    assert output.topology.left_reachable
    assert output.topology.right_reachable
    assert output.coarse_trajectories[ModeIndex.KEEP_HIGH, -1, 0] > output.coarse_trajectories[
        ModeIndex.KEEP_LOW, -1, 0
    ]
    assert output.coarse_trajectories[ModeIndex.LEFT_MEDIUM, -1, 1] > 2.5
    assert output.coarse_trajectories[ModeIndex.RIGHT_MEDIUM, -1, 1] < -2.5
    stop = output.coarse_trajectories[ModeIndex.STOP]
    speed = env.agents["agent0"].speed_km_h / 3.6
    assert validate_trajectory_kinematics(
        stop, speed, np.zeros(3)
    ).valid
    increments = np.linalg.norm(
        np.diff(np.vstack([np.zeros((1, 2)), stop[:, :2]]), axis=0),
        axis=1,
    )
    stationary = np.flatnonzero(increments <= 1.0e-3)
    assert stationary.size
    first = int(stationary[0])
    assert np.all(stop[first:, :2] == stop[first, :2])
    assert np.all(stop[first:, 2] == stop[first, 2])


def test_stop_anchor_preserves_current_lane_offset_without_recentering() -> None:
    env = _Env()
    vehicle = env.agents["agent0"]
    vehicle.position[1] = vehicle.lane.y + 0.6

    output = SimulatorDynamicAnchorGenerator().generate(env, "agent0")
    stop = output.coarse_trajectories[ModeIndex.STOP]

    assert np.max(np.abs(stop[:, 1])) < 1.0e-5
    assert validate_trajectory_kinematics(
        stop,
        vehicle.speed_km_h / 3.6,
        np.zeros(3),
    ).valid


def test_low_speed_stop_limits_heading_change_to_kinematic_contract() -> None:
    env = _Env()
    vehicle = env.agents["agent0"]
    vehicle.position[1] = vehicle.lane.y - 0.4
    vehicle.heading_theta = -0.103
    vehicle.speed_km_h = 5.18

    output = SimulatorDynamicAnchorGenerator().generate(env, "agent0")
    stop = output.coarse_trajectories[ModeIndex.STOP]
    audit = validate_trajectory_kinematics(
        stop,
        vehicle.speed_km_h / 3.6,
        np.zeros(3),
    )

    assert audit.valid
    assert np.max(audit.curvature_per_m) <= 0.25 + 1.0e-6


def test_stop_anchor_uses_exact_stop_distance_inside_first_interval() -> None:
    env = _Env()
    vehicle = env.agents["agent0"]
    vehicle.speed_km_h = 1.8
    speed_mps = vehicle.speed_km_h / 3.6
    acceleration = -4.5
    stop_time_s = speed_mps / -acceleration
    expected_distance_m = (
        speed_mps * stop_time_s
        + 0.5 * acceleration * stop_time_s**2
    )

    output = SimulatorDynamicAnchorGenerator().generate(env, "agent0")
    stop = output.coarse_trajectories[ModeIndex.STOP]
    audit = validate_trajectory_kinematics(
        stop,
        speed_mps,
        np.zeros(3),
    )

    assert audit.valid
    np.testing.assert_allclose(
        audit.cumulative_distance_m,
        expected_distance_m,
        atol=1.0e-5,
    )


def test_moving_anchors_start_from_current_lane_offset() -> None:
    env = _Env()
    vehicle = env.agents["agent0"]
    vehicle.position[1] = vehicle.lane.y + 0.6
    vehicle.heading_theta = -0.2

    output = SimulatorDynamicAnchorGenerator().generate(env, "agent0")

    for mode in (
        ModeIndex.KEEP_HIGH,
        ModeIndex.KEEP_MEDIUM,
        ModeIndex.KEEP_LOW,
    ):
        trajectory = output.coarse_trajectories[mode]
        assert abs(float(trajectory[0, 1])) < 0.6
        assert validate_trajectory_kinematics(
            trajectory,
            vehicle.speed_km_h / 3.6,
            np.zeros(3),
        ).valid


def test_joint_sample_contract_is_exact_and_joint_first() -> None:
    env = _Env()
    generator = SimulatorDynamicAnchorGenerator()
    builder = JointBEVSampleBuilder(anchor_generator=generator)
    _prime_builder(builder, env)
    sample = builder.build_sample(
        env,
        _expert_step(env, generator, {"agent0": 0, "agent1": 0, "agent2": 0}),
    )
    assert tuple(sample.as_dict()) == tuple(JOINT_SAMPLE_SHAPES)
    for name, value in sample.as_dict().items():
        assert value.shape == JOINT_SAMPLE_SHAPES[name]
        assert value.dtype == JOINT_SAMPLE_DTYPES[name]
        assert value.flags.c_contiguous
        assert not value.flags.writeable
    np.testing.assert_array_equal(sample.agent_role, list(AgentRole))
    assert np.all(sample.mode_valid_mask[np.arange(3), sample.gt_mode])
    assert np.all(sample.relation_valid_mask)
    np.testing.assert_allclose(sample.ego_pose_global[:, 0], [30.0, 20.0, 10.0])


def test_rule_action_changes_only_label_not_bev_input() -> None:
    env = _Env()
    generator = SimulatorDynamicAnchorGenerator()
    builder = JointBEVSampleBuilder(anchor_generator=generator)
    _prime_builder(builder, env)
    online = builder.build_model_inputs(env)
    assert isinstance(online, JointBEVModelInputs)
    assert tuple(online.as_dict()) == MODEL_INPUT_FIELDS
    keep = builder.build_sample(
        env,
        _expert_step(env, generator, {"agent0": 0, "agent1": 0, "agent2": 0}),
    )
    left = builder.build_sample(
        env,
        _expert_step(env, generator, {"agent0": -1, "agent1": 0, "agent2": 0}),
    )
    np.testing.assert_array_equal(keep.bev, left.bev)
    for name, value in online.as_dict().items():
        np.testing.assert_array_equal(value, getattr(keep, name))
        assert not value.flags.writeable
    assert keep.gt_mode[0] != left.gt_mode[0]
    assert int(left.gt_mode[0]) in LEFT_MODES
    # Navigation is simulator route geometry, not the selected local manoeuvre.
    np.testing.assert_array_equal(
        keep.bev[0, BEVChannel.NAVIGATION_ROUTE],
        left.bev[0, BEVChannel.NAVIGATION_ROUTE],
    )


def test_one_agent_contract_failure_discards_whole_joint_step() -> None:
    env = _Env()
    # Remove the left lane from both direct lookup and the drivable map.
    network = env.current_map.road_network
    del network.lanes[("A", "B", 0)]
    network.graph["A"]["B"] = list(network.lanes.values())
    generator = SimulatorDynamicAnchorGenerator()
    builder = JointBEVSampleBuilder(anchor_generator=generator)
    _prime_builder(builder, env)
    expert = _expert_step(env, generator, {"agent0": 0, "agent1": 0, "agent2": 0})
    expert = ExpertJointStep(
        rule_actions={"agent0": -1, "agent1": 0, "agent2": 0},
        trajectories_world=expert.trajectories_world,
        controls=expert.controls,
    )
    with pytest.raises(JointCollectionError) as exc_info:
        builder.build_sample(env, expert)
    assert exc_info.value.reason_code == "gt_action_group_has_no_valid_mode"


def test_dynamic_anchor_failure_has_distinct_reason_code() -> None:
    class _BrokenAnchorGenerator:
        def generate(self, env, agent_id):  # noqa: ARG002
            raise DynamicAnchorError("synthetic STOP failure")

    env = _Env()
    builder = JointBEVSampleBuilder(
        anchor_generator=_BrokenAnchorGenerator()
    )
    _prime_builder(builder, env)

    with pytest.raises(JointCollectionError) as exc_info:
        builder.build_model_inputs(env)

    assert (
        exc_info.value.reason_code
        == "dynamic_anchor_kinematic_invalid"
    )


def test_joint_sample_rejects_partial_or_misaligned_data() -> None:
    values = {
        name: np.zeros(shape, dtype=JOINT_SAMPLE_DTYPES[name])
        for name, shape in JOINT_SAMPLE_SHAPES.items()
    }
    values["agent_role"] = np.asarray(list(AgentRole), dtype=np.int64)
    values["mode_valid_mask"][:, ModeIndex.STOP] = True
    values["gt_mode"][:] = int(ModeIndex.STOP)
    JointBEVSample(**values)
    values["bev"] = np.zeros((8, 256, 256), dtype=np.uint8)
    with pytest.raises(JointCollectionError, match="bev must have shape"):
        JointBEVSample(**values)


def test_rule_planner_expert_reports_joint_planner_failure_category() -> None:
    expert = RulePlannerExpert.__new__(RulePlannerExpert)
    expert.agent_ids = ("agent0", "agent1", "agent2")
    decision = {
        agent_id: {
            "action": 0,
            "target_point": np.asarray([20.0, 0.0], dtype=np.float32),
        }
        for agent_id in expert.agent_ids
    }
    proposal = JointActionProposal(0, 0, 0.0, decision)
    batch = RuleMakerProposalBatch(1, (proposal,))
    expert.rule_maker = SimpleNamespace(
        propose_joint_actions=lambda *args, **kwargs: batch,
        accept_joint_action=lambda *args, **kwargs: None,
        get_last_debug=lambda: {},
    )
    expert.planner = SimpleNamespace(
        plan_ranked=lambda *args, **kwargs: (_ for _ in ()).throw(
            NormalPlannerNoFeasiblePlan(
                "no ranked proposal",
                reason_code="all_rule_proposals_infeasible",
                debug={},
            )
        ),
        get_last_debug=lambda: {},
    )
    expert.trajectory_executor = JointTrajectoryExecutor(expert.planner)
    env = SimpleNamespace(
        agents={agent_id: object() for agent_id in expert.agent_ids},
        _last_planner_batch={},
    )

    with pytest.raises(JointCollectionError) as error:
        expert.plan(env)

    assert (
        error.value.reason_code
        == "all_rule_proposals_infeasible"
    )


def test_rule_planner_expert_uses_independent_pid_when_formation_is_unlocked() -> None:
    agent_ids = ("agent0", "agent1", "agent2")
    expert = RulePlannerExpert.__new__(RulePlannerExpert)
    expert.agent_ids = agent_ids
    expert._last_formation_locked = True
    decision = {
        agent_id: {
            "action": 0,
            "target_point": np.asarray([20.0, 0.0], dtype=np.float32),
            "formation_constraint_enabled": False,
            "coordination_mode": "EMERGENCY_INDEPENDENT",
        }
        for agent_id in agent_ids
    }
    proposal = JointActionProposal(0, 0, 0.0, decision)
    batch = RuleMakerProposalBatch(1, (proposal,))
    expert.rule_maker = SimpleNamespace(
        propose_joint_actions=lambda *args, **kwargs: batch,
        accept_joint_action=lambda *args, **kwargs: None,
        get_last_debug=lambda: {
            "dynamic_roles": {agent_id: "leader" for agent_id in agent_ids}
        },
        is_formation_locked=False,
    )
    trajectories = {
        agent_id: np.zeros((8, 3), dtype=np.float32)
        for agent_id in agent_ids
    }
    expert.planner = SimpleNamespace(
        plan_ranked=lambda *args, **kwargs: RankedJointPlan(
            proposal_id=0,
            proposal_rank=0,
            rule_score=0.0,
            decisions=decision,
            trajectories_world=trajectories,
            trajectories_local=trajectories,
            selected_candidate_indices={agent_id: 0 for agent_id in agent_ids},
        ),
        get_last_debug=lambda: {
            "_joint": {"fallback_used": False},
            **{
                agent_id: {"fallback_used": False}
                for agent_id in agent_ids
            },
        },
    )
    expert.trajectory_executor = JointTrajectoryExecutor(expert.planner)

    class _Controller:
        def __init__(self):
            self.reset_calls = 0
            self.compute_calls = 0

        def reset(self):
            self.reset_calls += 1

        def compute_actions(self, env, trajectories_world):  # noqa: ARG002
            self.compute_calls += 1
            return {
                agent_id: np.asarray([0.0, -0.2], dtype=np.float32)
                for agent_id in trajectories_world
            }

    expert.pid_controller = _Controller()
    expert.lqr_controller = _Controller()
    applied_roles = {}
    env = SimpleNamespace(
        agents={agent_id: object() for agent_id in agent_ids},
        _last_planner_batch={},
        apply_dynamic_roles=lambda roles: applied_roles.update(roles),
    )

    result = expert.plan(env)

    assert set(result.controls) == set(agent_ids)
    assert expert.pid_controller.compute_calls == 1
    assert expert.pid_controller.reset_calls == 1
    assert expert.lqr_controller.compute_calls == 0
    assert applied_roles == {agent_id: "leader" for agent_id in agent_ids}


def test_sensorless_environment_forces_no_rendering_observation_stack(monkeypatch) -> None:
    assert dict(SensorlessJointBEVPlatoonEnv.default_config()["sensors"]) == {}
    captured = {}

    def _capture_init(self, config):
        captured.update(config)

    monkeypatch.setattr(PlatoonEnv, "__init__", _capture_init)
    SensorlessJointBEVPlatoonEnv(
        {
            "num_agents": 3,
            "use_render": True,
            "image_observation": True,
            "sensors": {"forbidden": object()},
        }
    )
    assert captured["num_agents"] == 3
    assert captured["observation_mode"] == "bev_gt"
    assert captured["use_render"] is False
    assert captured["image_observation"] is False
    assert captured["image_on_cuda"] is False
    assert captured["sensors"] == {}
    assert captured["interface_panel"] == []
    assert captured["ground_truth_traffic_policy"] is True
    assert captured["agent_observation"].__name__ == "DummyObservation"


def test_builder_requires_exact_one_second_history() -> None:
    env = _Env()
    builder = JointBEVSampleBuilder()
    builder.capture_state(env, 0.0)
    builder.capture_state(env, 0.5)
    assert not builder.history_ready()
    with pytest.raises(JointCollectionError, match="history is not ready"):
        builder.build_sample(
            env,
            ExpertJointStep(rule_actions={}, trajectories_world={}, controls={}),
        )


def test_relation_mask_marks_missing_neighbors_before_joint_rejection() -> None:
    env = _Env()
    builder = JointBEVSampleBuilder()
    del env.agents["agent2"]
    np.testing.assert_array_equal(builder._relation_valid_mask(env, "agent0"), [True, False])
