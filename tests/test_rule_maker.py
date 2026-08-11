from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import models.decisioner.rule_decisioner as rule_maker_module
from models.decisioner.risk import RiskDetector, SimpleRuleRiskDetector
from models.decisioner.rule_decisioner_helper import (
    save_candidate_debug_plot,
    save_lane_pair_debug_plot,
    save_s7_route_lanes_debug_plot,
    save_s8_route_lanes_debug_plot,
)
from models.decisioner.rule_decisioner import (
    MultiAgentRuleMaker,
    load_rule_maker_config,
    make_rule_maker,
)


def test_rule_risk_detector_has_single_package_entrypoint():
    assert RiskDetector is not None
    assert SimpleRuleRiskDetector is not None
    assert importlib.util.find_spec("models.decisioner.risk_detect") is None


def test_rule_maker_factory_uses_multi_agent_type_without_legacy_alias():
    assert not hasattr(rule_maker_module, "KeepLaneFallbackRuleMaker")
    assert isinstance(make_rule_maker({}), MultiAgentRuleMaker)
    assert isinstance(make_rule_maker({"rule_maker_type": "multi_agent"}), MultiAgentRuleMaker)
    try:
        make_rule_maker({"rule_maker_type": "keep_lane_fallback"})
    except ValueError as exc:
        assert "keep_lane_fallback" in str(exc)
    else:
        raise AssertionError("legacy keep_lane_fallback rule_maker_type should be rejected")


def test_s5_profiles_share_safety_thresholds_and_change_preferences():
    profiles = {
        name: load_rule_maker_config("S5_hard_brake_lead", profile_id=name)
        for name in ("brake_first", "balanced", "evasive")
    }
    assert {p["traffic_safety_distance_m"] for p in profiles.values()} == {10.0}
    assert {p["agent_safety_distance_m"] for p in profiles.values()} == {9.0}
    assert profiles["brake_first"]["idm_time_headway_s"] == 2.2
    assert profiles["balanced"]["mobil_lane_change_threshold"] == 0.2
    assert profiles["evasive"]["lane_change_preference"] == 0.6
    assert make_rule_maker(
        {"scenario_id": "S5_hard_brake_lead"}, profile_id="evasive"
    ).profile_id == "evasive"


def test_unknown_s5_profile_is_rejected():
    with pytest.raises(ValueError, match="Unknown RuleMaker profile"):
        load_rule_maker_config("S5_hard_brake_lead", profile_id="missing")


def test_rule_maker_factory_configures_relock_ttc_threshold():
    default_rule_maker = make_rule_maker({})
    overridden_rule_maker = make_rule_maker({"rule_maker_relock_ttc_threshold_s": 7.5})

    assert default_rule_maker.relock_ttc_threshold_s == pytest.approx(5.0)
    assert overridden_rule_maker.relock_ttc_threshold_s == pytest.approx(7.5)


def test_rule_maker_factory_configures_forced_lane_unlock_wait_steps():
    rule_maker = make_rule_maker({"rule_maker_forced_lane_unlock_wait_steps": 3})

    assert rule_maker.forced_lane_unlock_wait_steps == 3


def test_rule_maker_factory_configures_relock_stable_steps():
    rule_maker = make_rule_maker({"rule_maker_relock_stable_steps": 7})

    assert rule_maker.relock_stable_steps == 7


class FakeLane:
    def __init__(self, lane_id: int, y: float, length: float = 200.0, width: float = 3.5):
        self.index = ("A", "B", lane_id)
        self.length = float(length)
        self.width = float(width)
        self.y = float(y)

    def local_coordinates(self, position):
        return float(position[0]), float(position[1] - self.y)

    def position(self, longitudinal: float, lateral: float):
        return np.asarray([float(longitudinal), self.y + float(lateral)], dtype=np.float32)


class FakeRoadNetwork:
    def __init__(self):
        self._lanes = {
            ("A", "B", 0): FakeLane(0, 3.5),
            ("A", "B", 1): FakeLane(1, 0.0),
            ("A", "B", 2): FakeLane(2, -3.5),
        }
        self.graph = {"A": {"B": [self._lanes[("A", "B", 0)], self._lanes[("A", "B", 1)], self._lanes[("A", "B", 2)]]}}

    def get_lane(self, lane_index):
        return self._lanes[tuple(lane_index)]


def _vehicle(name: str, x: float, y: float, lane_id: int, speed_km_h: float = 20.0, width: float = 1.8):
    return SimpleNamespace(
        name=name,
        position=np.asarray([x, y], dtype=np.float32),
        heading_theta=0.0,
        lane=FakeLane(lane_id, y),
        speed_km_h=float(speed_km_h),
        LENGTH=4.5,
        WIDTH=float(width),
    )


def _env(agents, traffic):
    return SimpleNamespace(
        agents=agents,
        engine=SimpleNamespace(
            current_map=SimpleNamespace(road_network=FakeRoadNetwork()),
            traffic_manager=SimpleNamespace(_traffic_vehicles=list(traffic)),
        ),
    )


def test_target_lane_rejects_negative_lane_id_instead_of_python_wraparound():
    env = _env(agents={}, traffic=[])
    source_lane = env.engine.current_map.road_network.get_lane(("A", "B", 0))
    vehicle = _vehicle("agent0", 0.0, 3.5, 0)

    target = MultiAgentRuleMaker()._target_lane(
        env, vehicle, source_lane, action=-1
    )

    assert target is None


def test_rule_maker_traffic_prediction_advances_actor_at_matching_times():
    traffic = _vehicle(
        "traffic",
        10.0,
        0.0,
        1,
        speed_km_h=36.0,
    )
    env = _env(agents={}, traffic=[traffic])
    rule_maker = MultiAgentRuleMaker(horizon_s=2.0, num_waypoints=3)

    predicted = rule_maker._predict_traffic_positions(
        env,
        traffic,
        count=3,
    )

    np.testing.assert_allclose(
        predicted,
        np.asarray(
            [[16.666666, 0.0], [23.333334, 0.0], [30.0, 0.0]],
            dtype=np.float32,
        ),
        atol=1e-5,
    )


def test_rule_maker_coarse_trajectory_contains_only_future_half_second_points():
    vehicle = _vehicle("agent0", 10.0, 0.0, 1, speed_km_h=18.0)
    env = _env(agents={"agent0": vehicle}, traffic=[])
    rule_maker = MultiAgentRuleMaker(horizon_s=4.0, num_waypoints=8)

    candidate = rule_maker._build_coarse_trajectory(
        env, vehicle, 0, other_vehicles=[]
    )

    trajectory = np.asarray(candidate["trajectory_world"])
    assert trajectory.shape == (8, 2)
    assert trajectory[0, 0] > vehicle.position[0]
    dense = rule_maker._dense_coarse_pose(trajectory, vehicle)
    np.testing.assert_allclose(dense[0, :2], vehicle.position, atol=1e-6)
    np.testing.assert_allclose(dense[-1, :2], trajectory[-1], atol=1e-6)


def test_lane_change_commitment_is_transactional_until_plan_acceptance():
    vehicle = _vehicle("agent0", 20.0, 0.0, 1, speed_km_h=20.0)
    env = _env(agents={"agent0": vehicle}, traffic=[])
    rule_maker = MultiAgentRuleMaker(
        locked_on_reset=False,
        lane_change_preference=20.0,
        lc_cost=0.0,
        w_mobil=0.0,
        w_keep_bias=0.0,
    )

    batch = rule_maker.propose_joint_actions(env, ["agent0"], {})
    proposal = next(
        value
        for value in batch.proposals
        if int(value.decisions["agent0"]["action"]) != 0
    )

    assert rule_maker._pending_lane_change_commitments == {}
    rule_maker.accept_joint_action(batch.batch_id, proposal.proposal_id)
    assert rule_maker._pending_lane_change_commitments["agent0"].action == int(
        proposal.decisions["agent0"]["action"]
    )


def test_committed_execution_advances_state_without_generating_proposals():
    vehicle = _vehicle("agent0", 20.0, 0.0, 1, speed_km_h=20.0)
    env = _env(agents={"agent0": vehicle}, traffic=[])
    rule_maker = MultiAgentRuleMaker(
        locked_on_reset=False,
        lane_change_preference=20.0,
        lc_cost=0.0,
        w_mobil=0.0,
        w_keep_bias=0.0,
    )
    batch = rule_maker.propose_joint_actions(env, ["agent0"], {})
    proposal = next(
        value
        for value in batch.proposals
        if int(value.decisions["agent0"]["action"]) != 0
    )
    action = int(proposal.decisions["agent0"]["action"])
    rule_maker.accept_joint_action(batch.batch_id, proposal.proposal_id)

    debug = rule_maker.advance_committed_execution(env, ["agent0"], 11)

    assert debug["proposal_count"] == 0
    assert debug["action_search"]["proposal_generation_skipped"] is True
    assert debug["active_execution_id"] == 11
    assert debug["lane_change_commitments"]["active"]["agent0"]["action"] == action
    assert rule_maker.active_lane_change_agent_ids == frozenset({"agent0"})

    target_lane_id = 1 + action
    target_y = {0: 3.5, 2: -3.5}[target_lane_id]
    env.agents["agent0"] = _vehicle(
        "agent0", 24.0, target_y, target_lane_id, speed_km_h=20.0
    )
    completed = rule_maker.advance_committed_execution(env, ["agent0"], 11)

    assert completed["lane_change_commitments"]["active"] == {}
    assert rule_maker.has_active_lane_change_commitments is False
    assert rule_maker.active_lane_change_agent_ids == frozenset()


def test_commitment_is_not_completed_at_lane_assignment_boundary():
    vehicle = _vehicle("agent0", 20.0, 0.0, 1, speed_km_h=20.0)
    env = _env(agents={"agent0": vehicle}, traffic=[])
    rule_maker = MultiAgentRuleMaker(
        locked_on_reset=False,
        lane_change_preference=20.0,
        lc_cost=0.0,
        w_mobil=0.0,
        w_keep_bias=0.0,
    )
    batch = rule_maker.propose_joint_actions(env, ["agent0"], {})
    proposal = next(
        value
        for value in batch.proposals
        if int(value.decisions["agent0"]["action"]) != 0
    )
    action = int(proposal.decisions["agent0"]["action"])
    rule_maker.accept_joint_action(batch.batch_id, proposal.proposal_id)
    rule_maker.advance_committed_execution(env, ["agent0"], 12)

    target_lane_id = 1 + action
    target_y = {0: 3.5, 2: -3.5}[target_lane_id]
    # The simulator may already assign the target lane at the half-lane
    # boundary.  The accepted maneuver must continue until its centreline is
    # actually reached.
    halfway_y = 0.5 * target_y
    env.agents["agent0"] = _vehicle(
        "agent0", 22.0, halfway_y, target_lane_id, speed_km_h=20.0
    )
    active = rule_maker.advance_committed_execution(env, ["agent0"], 12)
    assert "agent0" in active["lane_change_commitments"]["active"]
    assert rule_maker.committed_execution_rule_actions(
        env, {"agent0": action}
    ) == {"agent0": 0}

    # A footprint can be fully inside the target lane while its centre still
    # has a material lateral error.  That is not completion of the accepted
    # spatial manoeuvre.
    env.agents["agent0"] = _vehicle(
        "agent0", 23.0, target_y - np.sign(target_y) * 0.5, target_lane_id,
        speed_km_h=20.0,
    )
    almost = rule_maker.advance_committed_execution(env, ["agent0"], 12)
    assert "agent0" in almost["lane_change_commitments"]["active"]
    assert rule_maker.committed_execution_rule_actions(
        env, {"agent0": action}
    ) == {"agent0": 0}

    env.agents["agent0"] = _vehicle(
        "agent0", 24.0, target_y, target_lane_id, speed_km_h=20.0
    )
    env.agents["agent0"].heading_theta = 0.2
    heading_not_converged = rule_maker.advance_committed_execution(
        env, ["agent0"], 12
    )
    assert "agent0" in heading_not_converged[
        "lane_change_commitments"
    ]["active"]

    env.agents["agent0"].heading_theta = 0.0
    completed = rule_maker.advance_committed_execution(env, ["agent0"], 12)
    assert completed["lane_change_commitments"]["active"] == {}
    assert completed["lane_change_commitments"]["completed"]["agent0"][
        "completion_reason"
    ] == "converged_to_target_lane_pose"


def test_risk_detector_unlocks_when_leader_ttc_is_below_threshold():
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 10.0, 0.0, 1, speed_km_h=36.0),
            "agent1": _vehicle("agent1", 0.0, 0.0, 1, speed_km_h=36.0),
        },
        traffic=[_vehicle("front", 28.0, 0.0, 1, speed_km_h=18.0)],
    )
    detector = SimpleRuleRiskDetector(ttc_trigger_s=3.0)

    result = detector.detect(env, ["agent0", "agent1"], env.engine.traffic_manager._traffic_vehicles, "LOCKED")

    assert result["triggered"] is True
    assert result["transition"] == "LOCKED_TO_UNLOCKED"
    assert result["next_state"] == "UNLOCKED"
    assert result["leader"]["front_net_gap_m"] == pytest.approx(13.5)
    assert result["leader"]["closing_speed_mps"] == pytest.approx(5.0)
    assert result["leader"]["ttc_s"] == pytest.approx(2.7)


def test_risk_detector_unlocks_immediately_for_new_s5_hard_brake_marker():
    lead = _vehicle("hard_brake", 80.0, 0.0, 1, speed_km_h=24.0)
    lead.scenario_role = "hard_brake_lead"
    lead.scenario_brake_trigger_step = 30
    lead.scenario_brake_target_speed_kmh = 1.0
    lead.scenario_brake_deceleration_mps2 = 6.0
    env = _env(
        agents={"agent0": _vehicle("agent0", 20.0, 0.0, 1)},
        traffic=[lead],
    )
    detector = SimpleRuleRiskDetector(ttc_trigger_s=0.1)

    first = detector.detect(env, ["agent0"], [lead], "LOCKED")
    second = detector.detect(env, ["agent0"], [lead], "LOCKED")

    assert first["transition"] == "LOCKED_TO_UNLOCKED"
    assert first["reason"] == "hard_brake_lead_triggered"
    assert first["emergency_event"]["trigger_step"] == 30
    assert second["triggered"] is False


def test_s5_risk_detector_ignores_stale_previous_episode_brake_marker():
    lead = _vehicle("lead", 30.0, 0.0, 1, speed_km_h=18.0)
    lead.scenario_role = "hard_brake_lead"
    lead.scenario_id = "S5_hard_brake_lead"
    lead.scenario_brake_trigger_step = 30
    summary = {
        "scenario_id": "S5_hard_brake_lead",
        "scenario_triggered": False,
        "scenario_trigger_step": None,
    }
    env = _env(
        agents={"agent0": _vehicle("agent0", 10.0, 0.0, 1)},
        traffic=[lead],
    )
    env._scenario_orchestrator = SimpleNamespace(
        get_episode_summary=lambda: dict(summary)
    )
    detector = SimpleRuleRiskDetector(ttc_trigger_s=100.0)

    waiting = detector.detect(env, ["agent0"], [lead], "LOCKED")
    summary.update(scenario_triggered=True, scenario_trigger_step=30)
    triggered = detector.detect(env, ["agent0"], [lead], "LOCKED")

    assert waiting["next_state"] == "LOCKED"
    assert waiting["waiting_for_s5_hard_brake"] is True
    assert triggered["transition"] == "LOCKED_TO_UNLOCKED"
    assert triggered["reason"] == "hard_brake_lead_triggered"


def test_rule_maker_applies_emergency_independent_mode_before_action_selection():
    lead = _vehicle("hard_brake", 100.0, 0.0, 1, speed_km_h=24.0)
    lead.scenario_role = "hard_brake_lead"
    lead.scenario_brake_trigger_step = 30
    env = _env(
        agents={"agent0": _vehicle("agent0", 20.0, 0.0, 1)},
        traffic=[lead],
    )
    rule_maker = MultiAgentRuleMaker(locked_on_reset=True, risk_ttc_trigger_s=0.1)

    decisions = rule_maker.compute(env, ["agent0"], planner_batch={})
    debug = rule_maker.get_last_debug()

    assert decisions["agent0"]["formation_constraint_enabled"] is False
    assert decisions["agent0"]["coordination_mode"] == "EMERGENCY_INDEPENDENT"
    assert debug["state_transition"] == "LOCKED_TO_UNLOCKED"
    assert debug["dynamic_roles"] == {"agent0": "leader"}


def test_s5_stays_locked_and_keeps_lane_before_hard_brake_marker():
    env = _env(
        agents={"agent0": _vehicle("agent0", 20.0, 0.0, 1, speed_km_h=30.0)},
        traffic=[_vehicle("near_front", 28.0, 0.0, 1, speed_km_h=0.0)],
    )
    env.config = {"scenario_id": "S5_hard_brake_lead"}
    rule_maker = MultiAgentRuleMaker(
        locked_on_reset=True,
        lane_change_preference=100.0,
        lc_cost=0.0,
    )

    decisions = rule_maker.compute(env, ["agent0"], planner_batch={})
    debug = rule_maker.get_last_debug()

    assert decisions["agent0"]["action"] == 0
    assert debug["formation_locked"] is True
    assert debug["action_search"]["strategy"] == "s5_pre_brake_keep"
    assert debug["risk_info"]["waiting_for_s5_hard_brake"] is True


@pytest.mark.parametrize(
    ("front_x", "front_speed_km_h"),
    [
        (29.5, 18.0),  # TTC is exactly 3 seconds.
        (31.0, 18.0),  # TTC is greater than 3 seconds.
        (20.0, 36.0),  # No positive closing speed.
    ],
)
def test_risk_detector_keeps_locked_at_or_above_ttc_threshold(front_x, front_speed_km_h):
    env = _env(
        agents={"agent0": _vehicle("agent0", 10.0, 0.0, 1, speed_km_h=36.0)},
        traffic=[_vehicle("front", front_x, 0.0, 1, speed_km_h=front_speed_km_h)],
    )
    detector = SimpleRuleRiskDetector(ttc_trigger_s=3.0)

    result = detector.detect(env, ["agent0"], env.engine.traffic_manager._traffic_vehicles, "LOCKED")

    assert result["triggered"] is False
    assert result["next_state"] == "LOCKED"


def test_risk_detector_unlocks_when_partial_forced_lane_wait_reaches_threshold():
    env = _env(
        agents={"agent0": _vehicle("agent0", 10.0, 0.0, 1, speed_km_h=20.0)},
        traffic=[],
    )
    detector = SimpleRuleRiskDetector(ttc_trigger_s=3.0)

    result = detector.detect(
        env,
        ["agent0"],
        [],
        "LOCKED",
        forced_lane_wait_info={"partial_forced_wait_steps": 10, "threshold_steps": 10},
    )

    assert result["triggered"] is True
    assert result["next_state"] == "UNLOCKED"
    assert result["transition"] == "LOCKED_TO_UNLOCKED"
    assert result["reason"] == "partial_forced_lane_wait>=10_steps"


def test_risk_detector_blocks_relock_when_forced_wait_is_active():
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 30.0, 0.0, 1, speed_km_h=20.0),
            "agent1": _vehicle("agent1", 20.0, 0.0, 1, speed_km_h=20.0),
        },
        traffic=[],
    )
    detector = SimpleRuleRiskDetector(ttc_trigger_s=3.0)

    result = detector.detect(
        env,
        ["agent0", "agent1"],
        [],
        "UNLOCKED",
        forced_lane_wait_info={"forced_wait_active": True, "partial_forced_wait_steps": 10, "threshold_steps": 10},
    )

    assert result["triggered"] is False
    assert result["next_state"] == "UNLOCKED"
    assert result["forced_wait_clear"] is False


def test_risk_detector_ignores_close_vehicle_in_adjacent_lane_without_intrusion_for_ttc():
    env = _env(
        agents={"agent0": _vehicle("agent0", 10.0, 0.0, 1, speed_km_h=36.0)},
        traffic=[_vehicle("adjacent", 12.0, 3.5, 0, speed_km_h=0.0)],
    )
    detector = SimpleRuleRiskDetector(ttc_trigger_s=3.0)

    result = detector.detect(env, ["agent0"], env.engine.traffic_manager._traffic_vehicles, "LOCKED")

    assert result["triggered"] is False
    assert result["leader"]["front_vehicle_id"] is None


def test_risk_detector_counts_adjacent_vehicle_intruding_into_leader_lane_for_ttc():
    env = _env(
        agents={"agent0": _vehicle("agent0", 10.0, 0.0, 1, speed_km_h=36.0)},
        traffic=[_vehicle("intruder", 22.0, 2.5, 0, speed_km_h=18.0)],
    )
    detector = SimpleRuleRiskDetector(ttc_trigger_s=3.0)

    result = detector.detect(env, ["agent0"], env.engine.traffic_manager._traffic_vehicles, "LOCKED")

    assert result["triggered"] is True
    assert result["leader"]["front_vehicle_id"] == "intruder"
    assert result["leader"]["front_net_gap_m"] == pytest.approx(7.5)
    assert result["leader"]["ttc_s"] == pytest.approx(1.5)


def test_risk_detector_chooses_nearest_front_between_same_lane_and_intruding_vehicle():
    env = _env(
        agents={"agent0": _vehicle("agent0", 10.0, 0.0, 1, speed_km_h=36.0)},
        traffic=[
            _vehicle("same_lane_front", 28.0, 0.0, 1, speed_km_h=18.0),
            _vehicle("intruder", 24.0, 2.5, 0, speed_km_h=18.0),
        ],
    )
    detector = SimpleRuleRiskDetector(ttc_trigger_s=3.0)

    result = detector.detect(env, ["agent0"], env.engine.traffic_manager._traffic_vehicles, "LOCKED")

    assert result["leader"]["front_vehicle_id"] == "intruder"
    assert result["leader"]["front_net_gap_m"] == pytest.approx(9.5)


def test_risk_detector_intruding_vehicle_with_no_closing_speed_has_infinite_ttc():
    env = _env(
        agents={"agent0": _vehicle("agent0", 10.0, 0.0, 1, speed_km_h=36.0)},
        traffic=[_vehicle("intruder", 22.0, 2.5, 0, speed_km_h=40.0)],
    )
    detector = SimpleRuleRiskDetector(ttc_trigger_s=3.0)

    result = detector.detect(env, ["agent0"], env.engine.traffic_manager._traffic_vehicles, "LOCKED")

    assert result["triggered"] is False
    assert result["leader"]["front_vehicle_id"] == "intruder"
    assert result["leader"]["ttc_s"] == math.inf


def test_risk_detector_ignores_intruding_vehicle_behind_leader_for_ttc():
    env = _env(
        agents={"agent0": _vehicle("agent0", 10.0, 0.0, 1, speed_km_h=36.0)},
        traffic=[_vehicle("behind_intruder", 8.0, 2.5, 0, speed_km_h=0.0)],
    )
    detector = SimpleRuleRiskDetector(ttc_trigger_s=3.0)

    result = detector.detect(env, ["agent0"], env.engine.traffic_manager._traffic_vehicles, "LOCKED")

    assert result["triggered"] is False
    assert result["leader"]["front_vehicle_id"] is None


def test_risk_detector_relocks_when_follower_gap_is_below_threshold():
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 30.0, 0.0, 1),
            "agent1": _vehicle("agent1", 12.0, 0.0, 1),
            "agent2": _vehicle("agent2", -1.0, 0.0, 1),
        },
        traffic=[],
    )
    detector = SimpleRuleRiskDetector(
        ideal_following_distance_m=10.0,
        relock_gap_ratio=1.5,
        relock_stable_steps=1,
    )

    result = detector.detect(env, ["agent0", "agent1", "agent2"], [], "UNLOCKED")

    assert result["triggered"] is True
    assert result["transition"] == "UNLOCKED_TO_LOCKED"
    assert result["next_state"] == "LOCKED"
    assert result["min_follower_gap_m"] == pytest.approx(8.5)
    assert result["leader"]["ttc_s"] == math.inf
    assert result["follower_gap_condition_met"] is True
    assert result["leader_ttc_condition_met"] is True
    assert result["relock_ttc_threshold_s"] == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("front_x", "expected_ttc"),
    [
        (55.0, 4.1),
        (59.5, 5.0),
    ],
)
def test_risk_detector_keeps_unlocked_when_leader_ttc_is_not_above_relock_threshold(
    front_x,
    expected_ttc,
):
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 30.0, 0.0, 1, speed_km_h=36.0),
            "agent1": _vehicle("agent1", 12.0, 0.0, 1, speed_km_h=36.0),
        },
        traffic=[_vehicle("front", front_x, 0.0, 1, speed_km_h=18.0)],
    )
    detector = SimpleRuleRiskDetector(
        relock_ttc_threshold_s=5.0, relock_stable_steps=1
    )

    result = detector.detect(
        env,
        ["agent0", "agent1"],
        env.engine.traffic_manager._traffic_vehicles,
        "UNLOCKED",
    )

    assert result["triggered"] is False
    assert result["next_state"] == "UNLOCKED"
    assert result["leader"]["ttc_s"] == pytest.approx(expected_ttc)
    assert result["follower_gap_condition_met"] is True
    assert result["leader_ttc_condition_met"] is False


def test_risk_detector_relocks_when_follower_gap_and_leader_ttc_are_safe():
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 30.0, 0.0, 1, speed_km_h=36.0),
            "agent1": _vehicle("agent1", 12.0, 0.0, 1, speed_km_h=36.0),
        },
        traffic=[_vehicle("front", 60.0, 0.0, 1, speed_km_h=18.0)],
    )
    detector = SimpleRuleRiskDetector(
        relock_ttc_threshold_s=5.0, relock_stable_steps=1
    )

    result = detector.detect(
        env,
        ["agent0", "agent1"],
        env.engine.traffic_manager._traffic_vehicles,
        "UNLOCKED",
    )

    assert result["triggered"] is True
    assert result["leader"]["ttc_s"] == pytest.approx(5.1)
    assert result["leader_ttc_condition_met"] is True


def test_risk_detector_treats_non_closing_front_vehicle_as_safe_for_relock():
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 30.0, 0.0, 1, speed_km_h=36.0),
            "agent1": _vehicle("agent1", 12.0, 0.0, 1, speed_km_h=36.0),
        },
        traffic=[_vehicle("front", 40.0, 0.0, 1, speed_km_h=36.0)],
    )
    detector = SimpleRuleRiskDetector(
        relock_ttc_threshold_s=5.0, relock_stable_steps=1
    )

    result = detector.detect(
        env,
        ["agent0", "agent1"],
        env.engine.traffic_manager._traffic_vehicles,
        "UNLOCKED",
    )

    assert result["triggered"] is True
    assert result["leader"]["ttc_s"] == math.inf


def test_risk_detector_keeps_unlocked_at_exact_relock_threshold():
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 29.5, 0.0, 1),
            "agent1": _vehicle("agent1", 10.0, 0.0, 1),
        },
        traffic=[],
    )
    detector = SimpleRuleRiskDetector(ideal_following_distance_m=10.0, relock_gap_ratio=1.5)

    result = detector.detect(env, ["agent0", "agent1"], [], "UNLOCKED")

    assert result["triggered"] is False
    assert result["next_state"] == "UNLOCKED"
    assert result["min_follower_gap_m"] == pytest.approx(15.0)


@pytest.mark.parametrize("follower_x,follower_lane", [(20.0, 2), (40.0, 1)])
def test_risk_detector_keeps_unlocked_without_valid_follower_pair(follower_x, follower_lane):
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 30.0, 0.0, 1),
            "agent1": _vehicle("agent1", follower_x, -3.5 if follower_lane == 2 else 0.0, follower_lane),
        },
        traffic=[],
    )
    detector = SimpleRuleRiskDetector(ideal_following_distance_m=10.0, relock_gap_ratio=1.5)

    result = detector.detect(env, ["agent0", "agent1"], [], "UNLOCKED")

    assert result["triggered"] is False
    assert result["next_state"] == "UNLOCKED"
    assert result["min_follower_gap_m"] is None


class ConnectedFakeLane(FakeLane):
    def __init__(self, lane_id: int, y: float, x_offset: float, from_node: str, to_node: str, length: float = 30.0, width: float = 3.5):
        super().__init__(lane_id=lane_id, y=y, length=length, width=width)
        self.index = (from_node, to_node, lane_id)
        self.x_offset = float(x_offset)

    def local_coordinates(self, position):
        return float(position[0] - self.x_offset), float(position[1] - self.y)

    def position(self, longitudinal: float, lateral: float):
        return np.asarray([self.x_offset + float(longitudinal), self.y + float(lateral)], dtype=np.float32)


class ConnectedFakeRoadNetwork:
    def __init__(self):
        self.graph = {
            "A": {
                "B": [ConnectedFakeLane(1, 0.0, 0.0, "A", "B", length=30.0)],
            },
            "B": {
                "C": [ConnectedFakeLane(1, 0.0, 30.0, "B", "C", length=30.0)],
            },
        }

    def get_lane(self, lane_index):
        return self.graph[lane_index[0]][lane_index[1]][lane_index[2] if len(self.graph[lane_index[0]][lane_index[1]]) > 1 else 0]


class TwoLaneConnectedFakeRoadNetwork:
    def __init__(self):
        self.graph = {
            "A": {
                "B": [
                    ConnectedFakeLane(0, 3.5, 0.0, "A", "B", length=30.0),
                    ConnectedFakeLane(1, 0.0, 0.0, "A", "B", length=30.0),
                    ConnectedFakeLane(2, -3.5, 0.0, "A", "B", length=30.0),
                ],
            },
            "B": {
                "C": [
                    ConnectedFakeLane(0, 3.5, 30.0, "B", "C", length=30.0),
                    ConnectedFakeLane(1, 0.0, 30.0, "B", "C", length=30.0),
                    ConnectedFakeLane(2, -5.0, 30.0, "B", "C", length=30.0),
                ],
            },
        }

    def get_lane(self, lane_index):
        return self.graph[lane_index[0]][lane_index[1]][lane_index[2]]


class S8ExitFakeRoadNetwork:
    def __init__(self):
        self.graph = {
            "A": {
                "B": [
                    ConnectedFakeLane(0, 3.5, 0.0, "A", "B", length=30.0),
                    ConnectedFakeLane(1, 0.0, 0.0, "A", "B", length=30.0),
                    ConnectedFakeLane(2, -3.5, 0.0, "A", "B", length=30.0),
                ],
            },
            "B": {
                "C": [
                    ConnectedFakeLane(0, 3.5, 30.0, "B", "C", length=30.0),
                    ConnectedFakeLane(1, 0.0, 30.0, "B", "C", length=30.0),
                    ConnectedFakeLane(2, -3.5, 30.0, "B", "C", length=30.0),
                ],
            },
            "C": {
                "D": [
                    ConnectedFakeLane(0, -3.5, 60.0, "C", "D", length=30.0),
                ],
            },
        }

    def get_lane(self, lane_index):
        return self.graph[lane_index[0]][lane_index[1]][lane_index[2]]


class S8ImmediateRampFakeRoadNetwork:
    def __init__(self):
        self.graph = {
            "A": {
                "B": [
                    ConnectedFakeLane(0, 3.5, 0.0, "A", "B", length=30.0),
                    ConnectedFakeLane(1, 0.0, 0.0, "A", "B", length=30.0),
                    ConnectedFakeLane(2, -3.5, 0.0, "A", "B", length=30.0),
                ],
            },
            "B": {
                "C": [
                    ConnectedFakeLane(0, -3.5, 30.0, "B", "C", length=30.0),
                ],
            },
        }

    def get_lane(self, lane_index):
        return self.graph[lane_index[0]][lane_index[1]][lane_index[2]]


class S8HardcodedBranchFakeRoadNetwork:
    def __init__(self):
        self.graph = {
            "3C0_1_": {
                "4G0_0_": [
                    ConnectedFakeLane(0, 3.5, 0.0, "3C0_1_", "4G0_0_", length=30.0),
                    ConnectedFakeLane(1, 0.0, 0.0, "3C0_1_", "4G0_0_", length=30.0),
                    ConnectedFakeLane(2, -3.5, 0.0, "3C0_1_", "4G0_0_", length=30.0),
                ],
                "4G1_0_": [
                    ConnectedFakeLane(0, -7.0, 0.0, "3C0_1_", "4G1_0_", length=30.0),
                ],
            },
            "4G0_0_": {
                "4G1_1_": [
                    ConnectedFakeLane(0, -3.5, 30.0, "4G0_0_", "4G1_1_", length=30.0),
                ],
            },
        }

    def get_lane(self, lane_index):
        return self.graph[lane_index[0]][lane_index[1]][lane_index[2]]


class S7MergeFakeRoadNetwork:
    def __init__(self):
        self.graph = {
            "A": {
                "B": [
                    ConnectedFakeLane(0, 10.5, 0.0, "A", "B", length=30.0),
                    ConnectedFakeLane(1, 7.0, 0.0, "A", "B", length=30.0),
                    ConnectedFakeLane(2, 3.5, 0.0, "A", "B", length=30.0),
                    ConnectedFakeLane(3, 0.0, 0.0, "A", "B", length=30.0),
                ],
            },
            "B": {
                "C": [
                    ConnectedFakeLane(0, 10.5, 30.0, "B", "C", length=30.0),
                    ConnectedFakeLane(1, 7.0, 30.0, "B", "C", length=30.0),
                    ConnectedFakeLane(2, 3.5, 30.0, "B", "C", length=30.0),
                    ConnectedFakeLane(3, 0.0, 30.0, "B", "C", length=30.0),
                ],
            },
        }

    def get_lane(self, lane_index):
        return self.graph[lane_index[0]][lane_index[1]][lane_index[2]]


def _env_connected(vehicle):
    road_network = ConnectedFakeRoadNetwork()
    vehicle.lane = road_network.graph["A"]["B"][0]
    vehicle.position = np.asarray([25.0, 0.0], dtype=np.float32)
    vehicle.navigation = SimpleNamespace(checkpoints=["A", "B", "C"])
    return SimpleNamespace(
        agents={"agent0": vehicle},
        engine=SimpleNamespace(
            current_map=SimpleNamespace(road_network=road_network),
            traffic_manager=SimpleNamespace(_traffic_vehicles=[]),
        ),
    )


def _env_connected_two_lane(vehicle):
    road_network = TwoLaneConnectedFakeRoadNetwork()
    vehicle.lane = road_network.graph["A"]["B"][1]
    vehicle.position = np.asarray([25.0, 0.0], dtype=np.float32)
    vehicle.navigation = SimpleNamespace(checkpoints=["A", "B", "C"])
    return SimpleNamespace(
        agents={"agent0": vehicle},
        engine=SimpleNamespace(
            current_map=SimpleNamespace(road_network=road_network),
            traffic_manager=SimpleNamespace(_traffic_vehicles=[]),
        ),
    )


def _env_s8_exit(vehicle, scenario_id="S8_ego_exit_to_ramp", local_route="R6_exit_to_ramp"):
    road_network = S8ExitFakeRoadNetwork()
    vehicle.lane = road_network.graph["A"]["B"][1]
    vehicle.position = np.asarray([25.0, 0.0], dtype=np.float32)
    vehicle.navigation = SimpleNamespace(checkpoints=["A", "B", "C", "D"])
    return SimpleNamespace(
        config={"scenario_id": scenario_id, "local_route": local_route},
        agents={"agent0": vehicle},
        engine=SimpleNamespace(
            current_map=SimpleNamespace(road_network=road_network),
            traffic_manager=SimpleNamespace(_traffic_vehicles=[]),
        ),
    )


def _env_s8_immediate_ramp(vehicle, scenario_id="S8_ego_exit_to_ramp", local_route="R6_exit_to_ramp"):
    road_network = S8ImmediateRampFakeRoadNetwork()
    vehicle.lane = road_network.graph["A"]["B"][2]
    vehicle.position = np.asarray([25.0, -3.5], dtype=np.float32)
    vehicle.navigation = SimpleNamespace(checkpoints=["A", "B", "C"])
    return SimpleNamespace(
        config={"scenario_id": scenario_id, "local_route": local_route},
        agents={"agent0": vehicle},
        engine=SimpleNamespace(
            current_map=SimpleNamespace(road_network=road_network),
            traffic_manager=SimpleNamespace(_traffic_vehicles=[]),
        ),
    )


def _env_s8_hardcoded_branch(vehicle):
    road_network = S8HardcodedBranchFakeRoadNetwork()
    vehicle.lane = road_network.graph["3C0_1_"]["4G0_0_"][2]
    vehicle.position = np.asarray([25.0, -3.5], dtype=np.float32)
    vehicle.navigation = SimpleNamespace(checkpoints=["3C0_1_", "4G0_0_", "4G1_1_"])
    return SimpleNamespace(
        config={"scenario_id": "S8_ego_exit_to_ramp", "local_route": "R6_exit_to_ramp"},
        agents={"agent0": vehicle},
        engine=SimpleNamespace(
            current_map=SimpleNamespace(road_network=road_network),
            traffic_manager=SimpleNamespace(_traffic_vehicles=[]),
        ),
    )


def _env_s7_merge(vehicle, *, lane_id=3, scenario_id="S7_ego_merge_from_ramp", local_route="R7_merge_core"):
    road_network = S7MergeFakeRoadNetwork()
    vehicle.lane = road_network.graph["A"]["B"][lane_id]
    vehicle.position = np.asarray([25.0, vehicle.lane.y], dtype=np.float32)
    vehicle.navigation = SimpleNamespace(checkpoints=["A", "B", "C"])
    return SimpleNamespace(
        config={"scenario_id": scenario_id, "local_route": local_route},
        agents={"agent0": vehicle},
        engine=SimpleNamespace(
            current_map=SimpleNamespace(road_network=road_network),
            traffic_manager=SimpleNamespace(_traffic_vehicles=[]),
        ),
    )


def _env_s7_real_lane_contract(vehicle):
    road_network = SimpleNamespace()
    source_lane = ConnectedFakeLane(
        0,
        0.0,
        0.0,
        "9g0_0_",
        "9g1_4_",
        length=60.0,
    )
    mainline_lanes = [
        ConnectedFakeLane(
            index,
            10.5 - 3.5 * index,
            0.0,
            "9g0_0_",
            "9g0_1_",
            length=120.0,
        )
        for index in range(3)
    ]
    road_network.graph = {
        "9g0_0_": {
            "9g1_4_": [source_lane],
            "9g0_1_": mainline_lanes,
        }
    }
    road_network.get_lane = lambda lane_index: road_network.graph[
        lane_index[0]
    ][lane_index[1]][lane_index[2]]
    vehicle.lane = source_lane
    vehicle.position = np.asarray([25.0, 0.0], dtype=np.float32)
    vehicle.navigation = SimpleNamespace(
        checkpoints=["9g0_0_", "9g1_4_"],
    )
    return SimpleNamespace(
        config={
            "scenario_id": "S7_ego_merge_from_ramp",
            "local_route": "R7_merge_core",
        },
        agents={"agent0": vehicle},
        engine=SimpleNamespace(
            current_map=SimpleNamespace(road_network=road_network),
            traffic_manager=SimpleNamespace(_traffic_vehicles=[]),
        ),
    )


def test_rule_maker_returns_target_points_for_all_agents():
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 20.0, 0.0, 1),
            "agent1": _vehicle("agent1", 10.0, 0.0, 1),
        },
        traffic=[],
    )
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0)

    decisions = rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})

    assert set(decisions) == {"agent0", "agent1"}
    assert decisions["agent0"]["target_point"].shape == (2,)
    assert decisions["agent1"]["target_point"].shape == (2,)
    assert decisions["agent0"]["target_point"][0] > 0.0


def test_rule_maker_scores_joint_actions_and_avoids_blocked_left_lane():
    env = _env(
        agents={"agent0": _vehicle("agent0", 0.0, 0.0, 1)},
        traffic=[_vehicle("left_blocker", 16.0, 3.5, 0)],
    )
    rule_maker = MultiAgentRuleMaker(
        target_speed_km_h=30.0,
        horizon_s=2.0,
        lane_change_preference=2.0,
    )

    decisions = rule_maker.compute(env, ["agent0"], planner_batch={})

    assert decisions["agent0"]["target_point"][0] > 0.0
    assert decisions["agent0"]["target_point"][1] < -1.0


def test_rule_maker_marks_rear_agent_as_follower_when_same_decision_same_lane_and_clear():
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 10.0, 0.0, 1),
            "agent1": _vehicle("agent1", 0.0, 0.0, 1),
        },
        traffic=[],
    )
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0)

    rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})
    debug = rule_maker.get_last_debug()

    assert debug is not None
    assert debug["dynamic_roles"]["agent0"] == "leader"
    assert debug["dynamic_roles"]["agent1"] == "follower"


def test_rule_maker_marks_rear_agent_as_leader_when_blocked_between_agents():
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 10.0, 0.0, 1),
            "agent1": _vehicle("agent1", 0.0, 0.0, 1),
        },
        traffic=[_vehicle("blocker", 5.0, 0.0, 1)],
    )
    rule_maker = MultiAgentRuleMaker(
        target_speed_km_h=30.0,
        horizon_s=2.0,
        locked_on_reset=False,
        ideal_following_distance_m=0.0,
    )

    rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})
    debug = rule_maker.get_last_debug()

    assert debug is not None
    assert debug["dynamic_roles"]["agent0"] == "leader"
    assert debug["dynamic_roles"]["agent1"] == "leader"


def test_rule_maker_marks_rear_agent_as_leader_when_agents_are_not_in_same_lane():
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 10.0, 0.0, 1),
            "agent1": _vehicle("agent1", 0.0, -3.5, 2),
        },
        traffic=[],
    )
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, locked_on_reset=False)

    rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})
    debug = rule_maker.get_last_debug()

    assert debug is not None
    assert debug["dynamic_roles"]["agent0"] == "leader"
    assert debug["dynamic_roles"]["agent1"] == "leader"


def test_rule_maker_prefers_coherent_joint_combo_for_close_platoon_even_with_lane_change_bias():
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 10.0, 0.0, 1),
            "agent1": _vehicle("agent1", 2.0, 0.0, 1),
        },
        traffic=[],
    )
    rule_maker = MultiAgentRuleMaker(
        target_speed_km_h=30.0,
        horizon_s=2.0,
        lane_change_preference=2.0,
    )

    rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})
    debug = rule_maker.get_last_debug()

    assert debug is not None
    assert debug["best_actions"]["agent0"] == debug["best_actions"]["agent1"]


def test_debug_compute_selects_requested_action_candidate():
    env = _env(
        agents={"agent0": _vehicle("agent0", 10.0, 0.0, 1)},
        traffic=[],
    )
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, locked_on_reset=False)

    decisions = rule_maker.debug_compute(env, ["agent0"], planner_batch={}, manual_actions={"agent0": 1})
    debug = rule_maker.get_last_debug()

    assert decisions["agent0"]["action"] == 1
    assert decisions["agent0"]["target_point"].shape == (2,)
    assert debug is not None
    assert debug["manual_mode"] is True
    assert debug["requested_actions"] == {"agent0": 1}
    assert debug["best_actions"] == {"agent0": 1}
    assert debug["invalid_actions"] == {}


def test_rule_maker_helper_saves_candidate_debug_plot_png(tmp_path):
    env = _env(
        agents={"agent0": _vehicle("candidate_plot_agent", 10.0, 0.0, 1)},
        traffic=[],
    )
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, locked_on_reset=False)
    candidates = rule_maker._build_agent_candidates(env, env.agents["agent0"], [])

    output_path = save_candidate_debug_plot(
        env.agents["agent0"],
        candidates,
        counter=1,
        output_dir=tmp_path,
    )

    assert output_path == tmp_path / "candidate_candidate_plot_agent_000001.png"
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_rule_maker_helper_saves_lane_pair_debug_plot_png(tmp_path):
    vehicle = _vehicle("lane_pair_plot_agent", 10.0, 0.0, 1)
    source_lane = FakeLane(1, 0.0)
    target_lane = FakeLane(2, -3.5)

    output_path = save_lane_pair_debug_plot(
        vehicle,
        source_lane,
        target_lane,
        action=1,
        counter=1,
        output_dir=tmp_path,
    )

    assert output_path == tmp_path / "lane_pair_lane_pair_plot_agent_action_1_000001.png"
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_rule_maker_helper_saves_s8_route_lanes_debug_plot_png(tmp_path, monkeypatch):
    vehicle = _vehicle("s8_route_plot_agent", 25.0, 0.0, 1)
    env = _env_s8_exit(vehicle)
    seen_lane_indices = []
    original_lane_centerline_points = rule_maker_module.save_s8_route_lanes_debug_plot.__globals__["_lane_centerline_points"]

    def record_lane_centerline_points(lane):
        seen_lane_indices.append(tuple(getattr(lane, "index", ()) or ()))
        return original_lane_centerline_points(lane)

    monkeypatch.setitem(
        rule_maker_module.save_s8_route_lanes_debug_plot.__globals__,
        "_lane_centerline_points",
        record_lane_centerline_points,
    )

    output_path = save_s8_route_lanes_debug_plot(
        vehicle,
        env.engine.current_map.road_network,
        ["A", "B"],
        vehicle.lane,
        counter=1,
        output_dir=tmp_path,
    )

    assert output_path == tmp_path / "s8_route_lanes_s8_route_plot_agent_000001.png"
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert ("B", "C", 2) in seen_lane_indices
    assert ("C", "D", 0) in seen_lane_indices


def test_rule_maker_helper_saves_s7_route_lanes_debug_plot_png(tmp_path, monkeypatch):
    vehicle = _vehicle("s7_route_plot_agent", 25.0, 0.0, 3)
    env = _env_s7_merge(vehicle)
    seen_lane_indices = []
    original_lane_centerline_points = rule_maker_module.save_s7_route_lanes_debug_plot.__globals__["_lane_centerline_points"]

    def record_lane_centerline_points(lane):
        seen_lane_indices.append(tuple(getattr(lane, "index", ()) or ()))
        return original_lane_centerline_points(lane)

    monkeypatch.setitem(
        rule_maker_module.save_s7_route_lanes_debug_plot.__globals__,
        "_lane_centerline_points",
        record_lane_centerline_points,
    )

    output_path = save_s7_route_lanes_debug_plot(
        vehicle,
        env.engine.current_map.road_network,
        ["A", "B"],
        vehicle.lane,
        counter=1,
        output_dir=tmp_path,
    )

    assert output_path == tmp_path / "s7_route_lanes_s7_route_plot_agent_000001.png"
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert ("A", "B", 3) in seen_lane_indices
    assert ("B", "C", 2) in seen_lane_indices


def test_rule_maker_idm_profile_slows_terminal_speed_when_slow_front_vehicle_exists():
    env = _env(
        agents={"agent0": _vehicle("agent0", 0.0, 0.0, 1, speed_km_h=28.0)},
        traffic=[_vehicle("front_slow", 18.0, 0.0, 1, speed_km_h=10.0)],
    )
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0)

    candidates = rule_maker._build_agent_candidates(env, env.agents["agent0"], rule_maker._traffic_vehicles(env))
    keep_candidate = next(candidate for candidate in candidates if int(candidate["action"]) == 0)

    assert keep_candidate["terminal_speed_km_h"] < float(env.agents["agent0"].speed_km_h)
    assert keep_candidate["trajectory_world"][-1, 0] < 16.0


def test_rule_maker_mobil_gain_prefers_free_adjacent_lane_over_blocked_keep_lane():
    env = _env(
        agents={"agent0": _vehicle("agent0", 0.0, 0.0, 1, speed_km_h=28.0)},
        traffic=[_vehicle("front_slow", 18.0, 0.0, 1, speed_km_h=10.0)],
    )
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0)

    candidates = rule_maker._build_agent_candidates(env, env.agents["agent0"], rule_maker._traffic_vehicles(env))
    keep_candidate = next(candidate for candidate in candidates if int(candidate["action"]) == 0)
    right_candidate = next(candidate for candidate in candidates if int(candidate["action"]) == 1)

    assert right_candidate["mobil_gain"] > keep_candidate["mobil_gain"]


def test_rule_maker_mobil_penalizes_lane_change_when_target_lane_rear_vehicle_must_hard_brake():
    env = _env(
        agents={"agent0": _vehicle("agent0", 0.0, 0.0, 1, speed_km_h=20.0)},
        traffic=[
            _vehicle("front_slow", 18.0, 0.0, 1, speed_km_h=10.0),
            _vehicle("right_rear_fast", -5.0, -3.5, 2, speed_km_h=32.0),
        ],
    )
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0)

    candidates = rule_maker._build_agent_candidates(env, env.agents["agent0"], rule_maker._traffic_vehicles(env))
    keep_candidate = next(candidate for candidate in candidates if int(candidate["action"]) == 0)
    right_candidate = next(candidate for candidate in candidates if int(candidate["action"]) == 1)

    assert right_candidate["rear_vehicle_post_accel_mps2"] <= -rule_maker.idm_comfortable_brake_mps2
    assert right_candidate["mobil_gain"] < keep_candidate["mobil_gain"]


def test_rule_maker_lane_change_trajectory_uses_smooth_lateral_transition():
    env = _env(
        agents={"agent0": _vehicle("agent0", 0.0, 0.0, 1, speed_km_h=20.0)},
        traffic=[],
    )
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)

    candidates = rule_maker._build_agent_candidates(env, env.agents["agent0"], rule_maker._traffic_vehicles(env))
    right_candidate = next(candidate for candidate in candidates if int(candidate["action"]) == 1)
    traj = np.asarray(right_candidate["trajectory_world"], dtype=np.float32)

    lateral = traj[:, 1]
    assert lateral[0] > -0.2
    assert lateral[-1] < -3.0
    assert np.all(np.diff(lateral) <= 1e-4)
    assert np.any(np.abs(lateral[1:-1]) < 3.0)


def test_rule_maker_trajectory_can_continue_across_connected_lane_segments():
    vehicle = _vehicle("agent0", 25.0, 0.0, 1, speed_km_h=25.0)
    env = _env_connected(vehicle)
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)

    candidates = rule_maker._build_agent_candidates(env, env.agents["agent0"], [])
    keep_candidate = next(candidate for candidate in candidates if int(candidate["action"]) == 0)
    traj = np.asarray(keep_candidate["trajectory_world"], dtype=np.float32)

    assert traj[-1, 0] > 30.0


def test_rule_maker_lane_change_continues_on_target_lane_family_across_blocks():
    vehicle = _vehicle("agent0", 25.0, 0.0, 1, speed_km_h=25.0)
    env = _env_connected_two_lane(vehicle)
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)

    candidates = rule_maker._build_agent_candidates(env, env.agents["agent0"], [])
    right_candidate = next(candidate for candidate in candidates if int(candidate["action"]) == 1)
    traj = np.asarray(right_candidate["trajectory_world"], dtype=np.float32)

    assert traj[-1, 0] > 30.0
    assert traj[-1, 1] < -4.0


def test_s8_intermediate_lane_chain_preserves_slot_before_exit():
    vehicle = _vehicle("agent0", 25.0, 0.0, 1, speed_km_h=25.0)
    env = _env_s8_exit(vehicle)
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)

    lane_chain = rule_maker._reference_lane_chain(env, env.agents["agent0"], env.agents["agent0"].lane)
    lane_indices = [tuple(lane.index) for lane in lane_chain]

    assert lane_indices == [
        ("A", "B", 1),
        ("B", "C", 1),
    ]


def test_s7_reference_lane_chain_enters_rightmost_mainline_lane():
    vehicle = _vehicle("agent0", 25.0, 0.0, 1, speed_km_h=25.0)
    env = _env_s8_exit(vehicle, scenario_id="S7_ego_merge_from_ramp", local_route="R7_merge_core")
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)

    lane_chain = rule_maker._reference_lane_chain(env, env.agents["agent0"], env.agents["agent0"].lane)
    lane_indices = [tuple(lane.index) for lane in lane_chain]

    assert lane_indices == [
        ("A", "B", 1),
        ("B", "C", 2),
        ("C", "D", 0),
    ]


def test_s8_rightmost_lane_follows_exit_route_with_keep_not_fake_right():
    vehicle = _vehicle("agent0", 25.0, -3.5, 2, speed_km_h=25.0)
    env = _env_s8_immediate_ramp(vehicle)
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)

    right_candidate = rule_maker._build_coarse_trajectory(
        env, env.agents["agent0"], 1, []
    )
    keep_candidate = rule_maker._build_coarse_trajectory(
        env, env.agents["agent0"], 0, []
    )

    assert right_candidate is None
    assert keep_candidate is not None
    assert keep_candidate["target_lane_index"] == ("A", "B", 2)
    assert keep_candidate["target_lane_chain_indices"] == (
        ("A", "B", 2),
        ("B", "C", 0),
    )
    assert keep_candidate["forced_route_action"] is True


def test_s8_rightmost_lane_forces_keep_instead_of_returning_left():
    vehicle = _vehicle("agent0", 25.0, -3.5, 2, speed_km_h=25.0)
    env = _env_s8_immediate_ramp(vehicle)
    rule_maker = MultiAgentRuleMaker(
        target_speed_km_h=30.0,
        horizon_s=2.0,
        num_waypoints=8,
        locked_on_reset=True,
        lane_change_preference=100.0,
        lc_cost=0.0,
        w_mobil=0.0,
        w_keep_bias=-100.0,
    )

    batch = rule_maker.propose_joint_actions(
        env, ["agent0"], planner_batch={}
    )

    assert len(batch.proposals) == 1
    decision = batch.proposals[0].decisions["agent0"]
    assert int(decision["action"]) == 0
    assert decision["source_lane_index"] == ("A", "B", 2)
    assert decision["target_lane_index"] == ("A", "B", 2)
    assert decision["target_lane_chain"] == (
        ("A", "B", 2),
        ("B", "C", 0),
    )
    debug = rule_maker.get_last_debug()
    assert debug["forced_lane_decision"] is True
    selected = next(
        value
        for value in debug["candidates_by_agent"]["agent0"]
        if value["selected"]
    )
    assert selected["forced_route_action"] is True


def test_s8_rightmost_lane_rejects_wrong_upstream_branch():
    vehicle = _vehicle("agent0", 25.0, -3.5, 2, speed_km_h=25.0)
    env = _env_s8_hardcoded_branch(vehicle)
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)

    target_lane = rule_maker._target_lane(
        env, env.agents["agent0"], env.agents["agent0"].lane, 1
    )

    assert target_lane is None
    chain = rule_maker._reference_lane_chain(
        env, env.agents["agent0"], env.agents["agent0"].lane
    )
    assert ("3C0_1_", "4G1_0_", 0) not in {
        tuple(lane.index) for lane in chain
    }
    assert ("4G0_0_", "4G1_1_", 0) in {
        tuple(lane.index) for lane in chain
    }


def test_debug_compute_s8_rightmost_lane_does_not_emit_fake_right():
    vehicle = _vehicle("agent0", 25.0, -3.5, 2, speed_km_h=25.0)
    env = _env_s8_immediate_ramp(vehicle)
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)

    decisions = rule_maker.debug_compute(
        env, ["agent0"], planner_batch={}, manual_actions={"agent0": 0}
    )
    debug = rule_maker.get_last_debug()

    assert decisions["agent0"]["action"] == 0
    assert debug is not None
    assert debug["best_actions"] == {"agent0": 0}
    selected_candidates = [
        candidate
        for candidate in debug["candidates_by_agent"]["agent0"]
        if candidate["selected"]
    ]
    assert len(selected_candidates) == 1
    assert selected_candidates[0]["target_lane_index"] == ("A", "B", 2)
    assert selected_candidates[0]["target_lane_chain_indices"] == (
        ("A", "B", 2),
        ("B", "C", 0),
    )


def test_non_s8_action_one_still_requires_current_road_neighbor_lane():
    vehicle = _vehicle("agent0", 25.0, -3.5, 2, speed_km_h=25.0)
    env = _env_s8_immediate_ramp(vehicle, scenario_id="S7_ego_merge_from_ramp", local_route="R7_merge_core")
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)

    candidate = rule_maker._build_coarse_trajectory(env, env.agents["agent0"], 1, [])

    assert candidate is None


def test_debug_compute_records_invalid_manual_action_without_fallback():
    vehicle = _vehicle("agent0", 25.0, -3.5, 2, speed_km_h=25.0)
    env = _env_s8_immediate_ramp(vehicle, scenario_id="S7_ego_merge_from_ramp", local_route="R7_merge_core")
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)

    decisions = rule_maker.debug_compute(env, ["agent0"], planner_batch={}, manual_actions={"agent0": 1})
    debug = rule_maker.get_last_debug()

    assert "agent0" not in decisions
    assert debug is not None
    assert debug["manual_mode"] is True
    assert debug["requested_actions"] == {"agent0": 1}
    assert debug["best_actions"] == {}
    assert debug["invalid_actions"] == {"agent0": 1}


def test_s7_left_lane_change_to_mainline_third_lane_is_marked_forced():
    vehicle = _vehicle("agent0", 25.0, 0.0, 0, speed_km_h=25.0)
    env = _env_s7_real_lane_contract(vehicle)
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)

    candidate = rule_maker._build_coarse_trajectory(env, env.agents["agent0"], -1, [])

    assert candidate is not None
    assert candidate["target_lane_index"] == ("9g0_0_", "9g0_1_", 2)
    assert candidate["forced_lane_change"] is True


def test_s7_target_lane_uses_special_branch_for_left_action(monkeypatch):
    vehicle = _vehicle("agent0", 25.0, 0.0, 3, speed_km_h=25.0)
    env = _env_s7_merge(vehicle)
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)
    branch_target = env.engine.current_map.road_network.get_lane(("B", "C", 2))
    calls = []

    def fake_s7_target(env_arg, vehicle_arg, source_lane_arg):
        calls.append((env_arg, vehicle_arg, source_lane_arg))
        return branch_target

    monkeypatch.setattr(rule_maker, "_S7_downstream_target_lane", fake_s7_target)

    target_lane = rule_maker._target_lane(env, env.agents["agent0"], env.agents["agent0"].lane, -1)

    assert target_lane is branch_target
    assert calls == [(env, env.agents["agent0"], env.agents["agent0"].lane)]


def test_s7_downstream_target_lane_uses_real_route_contract():
    vehicle = _vehicle("agent0", 25.0, 0.0, 0, speed_km_h=25.0)
    env = _env_s7_real_lane_contract(vehicle)
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)

    target_lane = rule_maker._S7_downstream_target_lane(env, env.agents["agent0"], env.agents["agent0"].lane)

    assert target_lane is not None
    assert tuple(target_lane.index) == ("9g0_0_", "9g0_1_", 2)


def test_s7_current_hybrid_ramp_contract_is_forced_left() -> None:
    vehicle = _vehicle("agent0", 25.0, 0.0, 0, speed_km_h=25.0)
    target = ConnectedFakeLane(2, 0.0, 0.0, "9g0_0_", "9g0_1_", length=120.0)
    source = ConnectedFakeLane(0, 0.0, 0.0, "18c0_1_", "9g0_0_", length=60.0)
    network = SimpleNamespace(
        graph={"18c0_1_": {"9g0_0_": [source]}, "9g0_0_": {"9g0_1_": [target, target, target]}},
        get_lane=lambda index: target if tuple(index) == ("9g0_0_", "9g0_1_", 2) else source,
    )
    vehicle.lane = source
    vehicle.navigation = SimpleNamespace(checkpoints=["18c0_1_", "9g0_0_", "9g0_1_"])
    env = SimpleNamespace(
        config={"scenario_id": "S7_ego_merge_from_ramp", "local_route": "R7_merge_core"},
        agents={"agent0": vehicle},
        engine=SimpleNamespace(current_map=SimpleNamespace(road_network=network), traffic_manager=SimpleNamespace(_traffic_vehicles=[])),
    )
    maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)
    candidate = maker._build_coarse_trajectory(env, vehicle, -1, [])
    assert candidate is not None
    assert candidate["forced_lane_change"] is True
    assert tuple(candidate["target_lane_index"]) == ("9g0_0_", "9g0_1_", 2)


def test_s7_forced_lane_change_bypasses_joint_score_and_selects_left_action(monkeypatch):
    vehicle = _vehicle("agent0", 25.0, 0.0, 0, speed_km_h=25.0)
    env = _env_s7_real_lane_contract(vehicle)
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)

    def fail_if_scored(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("_score_joint_combo should not run for forced lane decisions")

    monkeypatch.setattr(rule_maker, "_score_joint_combo", fail_if_scored)
    decisions = rule_maker.compute(env, ["agent0"], planner_batch={})
    debug = rule_maker.get_last_debug()

    assert decisions["agent0"]["action"] == -1
    assert debug is not None
    assert debug["forced_lane_decision"] is True


def test_s7_left_lane_change_to_other_lane_is_not_marked_forced():
    vehicle = _vehicle("agent0", 25.0, 0.0, 2, speed_km_h=25.0)
    env = _env_s7_merge(vehicle, lane_id=2)
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)

    candidate = rule_maker._build_coarse_trajectory(env, env.agents["agent0"], -1, [])

    assert candidate is not None
    assert candidate["target_lane_index"] == ("A", "B", 1)
    assert "forced_lane_change" not in candidate


def test_non_s7_left_lane_change_to_mainline_third_lane_is_not_marked_forced():
    vehicle = _vehicle("agent0", 25.0, 0.0, 3, speed_km_h=25.0)
    env = _env_s7_merge(vehicle, scenario_id="S6_background_merge_in", local_route="R6_mainline_merge_approach")
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0, num_waypoints=8)

    candidate = rule_maker._build_coarse_trajectory(env, env.agents["agent0"], -1, [])

    assert candidate is not None
    assert candidate["target_lane_index"] == ("A", "B", 2)
    assert "forced_lane_change" not in candidate


def test_s6_locked_no_risk_keeps_lane_until_merge_conflict_is_detected(monkeypatch):
    vehicle = _vehicle("agent0", 25.0, 0.0, 1, speed_km_h=25.0)
    env = _env_s7_merge(
        vehicle,
        lane_id=1,
        scenario_id="S6_background_merge_in",
        local_route="R6_mainline_merge_approach",
    )
    rule_maker = MultiAgentRuleMaker(
        target_speed_km_h=30.0,
        horizon_s=2.0,
        num_waypoints=8,
        locked_on_reset=True,
    )

    def fail_if_generic_scored(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("no-risk S6 must not use generic MOBIL ranking")

    monkeypatch.setattr(rule_maker, "_best_locked_combo", fail_if_generic_scored)
    decisions = rule_maker.compute(env, ["agent0"], planner_batch={})
    debug = rule_maker.get_last_debug()

    assert decisions["agent0"]["action"] == 0
    assert debug["risk_triggered"] is False
    assert debug["action_search"]["strategy"] == "s6_no_risk_keep"
    assert debug["action_search"]["final_feasibility_authority"] == "normal_planner"


def test_s6_detected_risk_preserves_joint_action_search(monkeypatch):
    vehicle = _vehicle("agent0", 25.0, 0.0, 1, speed_km_h=25.0)
    env = _env_s7_merge(
        vehicle,
        lane_id=1,
        scenario_id="S6_background_merge_in",
        local_route="R6_mainline_merge_approach",
    )
    rule_maker = MultiAgentRuleMaker(
        target_speed_km_h=30.0,
        horizon_s=2.0,
        num_waypoints=8,
        locked_on_reset=True,
    )
    monkeypatch.setattr(
        rule_maker._risk_detector,
        "detect",
        lambda *args, **kwargs: {
            "triggered": True,
            "next_state": "EMERGENCY_INDEPENDENT",
            "transition": "LOCKED_TO_EMERGENCY_INDEPENDENT",
        },
    )

    def choose_right(*, candidate_sets, **kwargs):
        combo = tuple(
            next(value for value in candidates if int(value["action"]) == 1)
            for candidates in candidate_sets
        )
        return combo, 1.0, {"strategy": "risk_joint_search"}, [(combo, 1.0)]

    monkeypatch.setattr(rule_maker, "_best_conditional_combo", choose_right)
    decisions = rule_maker.compute(env, ["agent0"], planner_batch={})
    debug = rule_maker.get_last_debug()

    assert decisions["agent0"]["action"] == 1
    assert debug["risk_triggered"] is True
    assert debug["action_search"]["strategy"] == "risk_joint_search"


def test_s8_candidates_mark_right_lane_change_as_forced():
    vehicle = _vehicle("agent0", 25.0, 0.0, 1, speed_km_h=25.0)
    env = _env_s8_exit(vehicle)
    rule_maker = MultiAgentRuleMaker(
        target_speed_km_h=30.0,
        horizon_s=2.0,
        num_waypoints=8,
    )

    candidates = rule_maker._build_agent_candidates(env, env.agents["agent0"], [])

    assert candidates
    for candidate in candidates:
        assert "forced_lane_mean_lateral_distance_m" not in candidate
        assert "force_lane_score" not in candidate
        assert bool(candidate.get("forced_lane_change", False)) is (int(candidate["action"]) == 1)


def test_s8_forced_lane_change_bypasses_joint_score_and_selects_right_action(monkeypatch):
    vehicle = _vehicle("agent0", 25.0, 0.0, 1, speed_km_h=25.0)
    env = _env_s8_exit(vehicle)
    rule_maker = MultiAgentRuleMaker(
        target_speed_km_h=30.0,
        horizon_s=2.0,
        num_waypoints=8,
    )

    def fail_if_scored(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("_score_joint_combo should not run for forced lane decisions")

    monkeypatch.setattr(rule_maker, "_score_joint_combo", fail_if_scored)
    decisions = rule_maker.compute(env, ["agent0"], planner_batch={})
    debug = rule_maker.get_last_debug()

    assert decisions["agent0"]["action"] == 1
    assert debug is not None
    assert debug["forced_lane_decision"] is True


def test_non_s8_candidates_do_not_record_forced_lane_change():
    vehicle = _vehicle("agent0", 25.0, 0.0, 1, speed_km_h=25.0)
    env = _env_s8_exit(vehicle, scenario_id="S7_ego_merge_from_ramp", local_route="R7_merge_core")
    rule_maker = MultiAgentRuleMaker(
        target_speed_km_h=30.0,
        horizon_s=2.0,
        num_waypoints=8,
    )

    candidates = rule_maker._build_agent_candidates(env, env.agents["agent0"], [])

    assert candidates
    assert all("forced_lane_mean_lateral_distance_m" not in candidate for candidate in candidates)
    assert all("force_lane_score" not in candidate for candidate in candidates)
    assert all("forced_lane_change" not in candidate for candidate in candidates)


def test_rule_maker_joint_agent_safety_score_uses_soft_reward_not_hard_constraint():
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0)
    close_combo = (
        {"trajectory_world": np.asarray([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=np.float32)},
        {"trajectory_world": np.asarray([[0.5, 0.0], [5.5, 0.0], [10.5, 0.0]], dtype=np.float32)},
    )
    far_combo = (
        {"trajectory_world": np.asarray([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=np.float32)},
        {"trajectory_world": np.asarray([[0.0, 10.0], [5.0, 10.0], [10.0, 10.0]], dtype=np.float32)},
    )

    close_score = rule_maker._joint_agent_safety_score(close_combo)
    far_score = rule_maker._joint_agent_safety_score(far_combo)

    assert np.isfinite(close_score)
    assert close_score > -10.0
    assert far_score > close_score


def test_rule_maker_joint_agent_safety_score_is_more_tolerant_for_adjacent_lane_parallel_motion():
    rule_maker = MultiAgentRuleMaker(target_speed_km_h=30.0, horizon_s=2.0)
    longitudinal_close_combo = (
        {"trajectory_world": np.asarray([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=np.float32)},
        {"trajectory_world": np.asarray([[2.0, 0.0], [7.0, 0.0], [12.0, 0.0]], dtype=np.float32)},
    )
    lateral_parallel_combo = (
        {"trajectory_world": np.asarray([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]], dtype=np.float32)},
        {"trajectory_world": np.asarray([[0.0, 2.0], [5.0, 2.0], [10.0, 2.0]], dtype=np.float32)},
    )

    longitudinal_score = rule_maker._joint_agent_safety_score(longitudinal_close_combo)
    lateral_score = rule_maker._joint_agent_safety_score(lateral_parallel_combo)

    assert lateral_score > longitudinal_score


def test_rule_maker_starts_locked_with_shared_action_and_fixed_roles():
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 10.0, 0.0, 1),
            "agent1": _vehicle("agent1", 2.0, 0.0, 1),
        },
        traffic=[],
    )
    rule_maker = MultiAgentRuleMaker(
        target_speed_km_h=30.0,
        horizon_s=2.0,
        lane_change_preference=2.0,
        locked_on_reset=True,
        risk_ttc_trigger_s=3.0,
    )
    rule_maker.reset(env, ["agent0", "agent1"])

    rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})
    debug = rule_maker.get_last_debug()

    assert debug is not None
    assert debug["formation_locked"] is True
    assert debug["best_actions"]["agent0"] == debug["best_actions"]["agent1"]
    assert debug["dynamic_roles"] == {"agent0": "leader", "agent1": "follower"}


def test_rule_maker_unlocks_after_risk_trigger_and_recovers_dynamic_roles():
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 10.0, 0.0, 1),
            "agent1": _vehicle("agent1", 2.0, -3.5, 2),
        },
        traffic=[_vehicle("front_risk", 16.0, 0.0, 1, speed_km_h=0.0)],
    )
    rule_maker = MultiAgentRuleMaker(
        target_speed_km_h=30.0,
        horizon_s=2.0,
        locked_on_reset=True,
        risk_ttc_trigger_s=3.0,
    )
    rule_maker.reset(env, ["agent0", "agent1"])

    rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})
    debug = rule_maker.get_last_debug()

    assert debug is not None
    assert debug["formation_locked"] is False
    assert debug["risk_triggered"] is True
    assert debug["state_transition"] == "LOCKED_TO_UNLOCKED"
    assert debug["dynamic_roles"]["agent0"] == "leader"
    assert debug["dynamic_roles"]["agent1"] == "leader"


def test_rule_maker_relocks_only_after_twenty_stable_steps():
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 10.0, 0.0, 1, speed_km_h=36.0),
            "agent1": _vehicle("agent1", 0.0, -3.5, 2, speed_km_h=36.0),
        },
        traffic=[_vehicle("front_risk", 16.0, 0.0, 1, speed_km_h=0.0)],
    )
    rule_maker = MultiAgentRuleMaker(
        target_speed_km_h=30.0,
        horizon_s=2.0,
        locked_on_reset=True,
        risk_ttc_trigger_s=3.0,
        ideal_following_distance_m=10.0,
        relock_gap_ratio=1.5,
        relock_stable_steps=20,
    )
    rule_maker.reset(env, ["agent0", "agent1"])

    rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})
    assert rule_maker.is_formation_locked is False
    rule_maker._lane_change_commitments.clear()
    rule_maker._pending_lane_change_commitments.clear()

    env.agents["agent1"] = _vehicle("agent1", -2.0, 0.0, 1, speed_km_h=36.0)
    env.engine.traffic_manager._traffic_vehicles = []
    for _ in range(19):
        rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})
        debug = rule_maker.get_last_debug()
        assert debug is not None
        assert debug["formation_locked"] is False
    rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})
    debug = rule_maker.get_last_debug()

    assert debug is not None
    assert debug["formation_locked"] is True
    assert debug["risk_triggered"] is True
    assert debug["state_transition"] == "UNLOCKED_TO_LOCKED"


def _forced_wait_candidate(
    action: int,
    *,
    forced: bool = False,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
):
    candidate = {
        "action": int(action),
        "valid": True,
        "score": 0.0,
        "trajectory_world": np.asarray(
            [
                [origin_x, origin_y],
                [origin_x + 5.0, origin_y - 3.5 * float(action)],
            ],
            dtype=np.float32,
        ),
        "target_point": np.asarray([5.0, 0.0], dtype=np.float32),
        "source_lane_index": ("A", "B", 1),
        "target_lane_index": ("A", "B", 1 + int(action)),
        "mobil_gain": 0.0,
    }
    if forced:
        candidate["forced_lane_change"] = True
    return candidate


def _scored_forced_candidate(
    action: int,
    *,
    mobil_gain: float,
    forced: bool = False,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
):
    candidate = _forced_wait_candidate(
        action,
        forced=forced,
        origin_x=origin_x,
        origin_y=origin_y,
    )
    candidate["mobil_gain"] = float(mobil_gain)
    candidate["target_point"] = np.asarray([5.0, float(action)], dtype=np.float32)
    return candidate


def test_independent_joint_score_omits_formation_consistency_terms():
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 20.0, 0.0, 1),
            "agent1": _vehicle("agent1", 8.0, 0.0, 1),
        },
        traffic=[],
    )
    rule_maker = MultiAgentRuleMaker(
        w_formation_consistent=100.0,
        w_formation_inconsistent_cost=100.0,
        w_close_keep=100.0,
        w_close_lc_same_cost=100.0,
        w_close_lc_diff_cost=100.0,
    )
    combo = (
        _forced_wait_candidate(0, origin_x=20.0),
        _forced_wait_candidate(1, origin_x=8.0),
    )

    locked = rule_maker._score_joint_combo(
        env,
        ["agent0", "agent1"],
        combo,
        [],
        formation_constraint_enabled=True,
    )
    independent = rule_maker._score_joint_combo(
        env,
        ["agent0", "agent1"],
        combo,
        [],
        formation_constraint_enabled=False,
    )

    assert independent > locked


def test_unlocked_forced_agent_must_select_forced_candidate(monkeypatch):
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 10.0, 0.0, 1),
            "agent1": _vehicle("agent1", 2.0, -3.5, 2),
        },
        traffic=[],
    )
    rule_maker = MultiAgentRuleMaker(
        target_speed_km_h=30.0,
        horizon_s=2.0,
        locked_on_reset=False,
    )

    def fake_candidates(_env, vehicle, _traffic_vehicles, **_kwargs):
        origin = {
            "origin_x": float(vehicle.position[0]),
            "origin_y": float(vehicle.position[1]),
        }
        if vehicle.name == "agent0":
            return [
                _scored_forced_candidate(0, mobil_gain=10.0, forced=False, **origin),
                _scored_forced_candidate(1, mobil_gain=-10.0, forced=True, **origin),
            ]
        return [
            _scored_forced_candidate(0, mobil_gain=-1.0, forced=False, **origin),
            _scored_forced_candidate(1, mobil_gain=2.0, forced=False, **origin),
        ]

    monkeypatch.setattr(rule_maker, "_build_agent_candidates", fake_candidates)
    decisions = rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})
    debug = rule_maker.get_last_debug()

    assert decisions["agent0"]["action"] == 1
    assert decisions["agent1"]["action"] == 1
    assert debug is not None
    assert debug["formation_locked"] is False
    assert debug["forced_lane_decision"] is False
    selected_agent0 = [
        candidate
        for candidate in debug["candidates_by_agent"]["agent0"]
        if candidate["selected"]
    ]
    assert len(selected_agent0) == 1
    assert selected_agent0[0]["forced_lane_change"] is True


def test_rule_maker_unlocks_after_partial_forced_lane_wait_timeout_and_blocks_early_relock(monkeypatch):
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 10.0, 0.0, 1),
            "agent1": _vehicle("agent1", 2.0, 0.0, 1),
        },
        traffic=[],
    )
    rule_maker = MultiAgentRuleMaker(
        target_speed_km_h=30.0,
        horizon_s=2.0,
        locked_on_reset=True,
        forced_lane_unlock_wait_steps=10,
        relock_stable_steps=1,
    )

    forced_enabled = {"value": True}

    def fake_candidates(_env, vehicle, _traffic_vehicles, **_kwargs):
        origin = {
            "origin_x": float(vehicle.position[0]),
            "origin_y": float(vehicle.position[1]),
        }
        if vehicle.name == "agent0" and forced_enabled["value"]:
            return [_forced_wait_candidate(0, forced=False, **origin), _forced_wait_candidate(1, forced=True, **origin)]
        return [_forced_wait_candidate(0, forced=False, **origin), _forced_wait_candidate(1, forced=False, **origin)]

    monkeypatch.setattr(rule_maker, "_build_agent_candidates", fake_candidates)
    for _ in range(9):
        rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})
        debug = rule_maker.get_last_debug()
        assert debug["formation_locked"] is True
        assert debug["forced_lane_wait_info"]["partial_forced_wait_steps"] < 10

    rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})
    debug = rule_maker.get_last_debug()

    assert debug["formation_locked"] is False
    assert debug["state_transition"] == "LOCKED_TO_UNLOCKED"
    assert debug["risk_info"]["reason"] == "partial_forced_lane_wait>=10_steps"
    assert debug["forced_lane_wait_info"]["forced_agents"] == ["agent0"]
    assert debug["forced_lane_wait_info"]["all_forced_ready"] is False
    assert debug["forced_lane_wait_info"]["partial_forced_wait_steps"] == 10
    assert debug["forced_lane_wait_info"]["forced_wait_active"] is True
    rule_maker.accept_joint_action(debug["proposal_batch_id"], 0)

    env.agents["agent1"] = _vehicle("agent1", 2.0, 0.0, 1)
    rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})
    debug = rule_maker.get_last_debug()
    assert debug["formation_locked"] is False
    assert debug["state_transition"] is None
    assert debug["risk_info"]["forced_wait_clear"] is False
    assert debug["forced_lane_wait_info"]["forced_wait_frozen"] is True
    assert debug["forced_lane_wait_info"]["partial_forced_wait_steps"] == 10

    forced_enabled["value"] = False
    rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})
    debug = rule_maker.get_last_debug()
    assert debug["formation_locked"] is False
    assert debug["risk_info"]["active_lane_change_commitments"] is True

    env.agents["agent0"] = _vehicle("agent0", 10.0, -3.5, 2)
    env.agents["agent1"] = _vehicle("agent1", -2.0, -3.5, 2)
    rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})
    debug = rule_maker.get_last_debug()
    assert debug["formation_locked"] is True
    assert debug["state_transition"] == "UNLOCKED_TO_LOCKED"
    assert debug["risk_info"]["forced_wait_clear"] is True
    assert debug["forced_lane_wait_info"]["forced_wait_active"] is False


def test_rule_maker_clears_forced_lane_wait_when_signal_disappears(monkeypatch):
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 10.0, 0.0, 1),
            "agent1": _vehicle("agent1", 2.0, 0.0, 1),
        },
        traffic=[],
    )
    rule_maker = MultiAgentRuleMaker(
        target_speed_km_h=30.0,
        horizon_s=2.0,
        locked_on_reset=True,
        forced_lane_unlock_wait_steps=10,
        relock_stable_steps=1,
    )
    forced_enabled = {"value": True}

    def fake_candidates(_env, vehicle, _traffic_vehicles, **_kwargs):
        origin = {
            "origin_x": float(vehicle.position[0]),
            "origin_y": float(vehicle.position[1]),
        }
        if vehicle.name == "agent0" and forced_enabled["value"]:
            return [_forced_wait_candidate(0, forced=False, **origin), _forced_wait_candidate(1, forced=True, **origin)]
        return [_forced_wait_candidate(0, forced=False, **origin), _forced_wait_candidate(1, forced=False, **origin)]

    monkeypatch.setattr(rule_maker, "_build_agent_candidates", fake_candidates)
    rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})
    assert rule_maker.get_last_debug()["forced_lane_wait_info"]["partial_forced_wait_steps"] == 1

    forced_enabled["value"] = False
    rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})
    debug = rule_maker.get_last_debug()
    assert debug["forced_lane_wait_info"]["forced_agents"] == []
    assert debug["forced_lane_wait_info"]["forced_lane_first_step"] == {}
    assert debug["forced_lane_wait_info"]["partial_forced_wait_steps"] == 0

    forced_enabled["value"] = True
    rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})
    debug = rule_maker.get_last_debug()
    assert debug["forced_lane_wait_info"]["partial_forced_wait_steps"] == 1


def test_rule_maker_all_forced_ready_does_not_trigger_wait_unlock(monkeypatch):
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 10.0, 0.0, 1),
            "agent1": _vehicle("agent1", 2.0, 0.0, 1),
        },
        traffic=[],
    )
    rule_maker = MultiAgentRuleMaker(
        target_speed_km_h=30.0,
        horizon_s=2.0,
        locked_on_reset=True,
        forced_lane_unlock_wait_steps=1,
    )

    def fake_candidates(_env, vehicle, _traffic_vehicles, **_kwargs):
        return [
            _forced_wait_candidate(
                1,
                forced=True,
                origin_x=float(vehicle.position[0]),
                origin_y=float(vehicle.position[1]),
            )
        ]

    monkeypatch.setattr(rule_maker, "_build_agent_candidates", fake_candidates)
    rule_maker.compute(env, ["agent0", "agent1"], planner_batch={})
    debug = rule_maker.get_last_debug()

    assert debug["formation_locked"] is True
    assert debug["forced_lane_decision"] is True
    assert debug["forced_lane_wait_info"]["all_forced_ready"] is True
    assert debug["forced_lane_wait_info"]["partial_forced_wait_steps"] == 0


def test_route_required_combo_defers_coarse_conflict_to_normal_planner(
    monkeypatch,
):
    env = _env(
        agents={
            "agent0": _vehicle("agent0", 10.0, 0.0, 1),
            "agent1": _vehicle("agent1", 2.0, 0.0, 1),
        },
        traffic=[],
    )
    rule_maker = MultiAgentRuleMaker(locked_on_reset=True)

    def fake_candidates(_env, vehicle, _traffic_vehicles, **_kwargs):
        candidate = _forced_wait_candidate(
            0,
            origin_x=float(vehicle.position[0]),
            origin_y=float(vehicle.position[1]),
        )
        candidate["forced_route_action"] = True
        return [candidate]

    monkeypatch.setattr(rule_maker, "_build_agent_candidates", fake_candidates)
    monkeypatch.setattr(
        rule_maker,
        "_combo_has_hard_conflict",
        lambda *_args, **_kwargs: True,
    )

    batch = rule_maker.propose_joint_actions(
        env, ["agent0", "agent1"], planner_batch={}
    )

    assert len(batch.proposals) == 1
    assert {
        agent_id: int(decision["action"])
        for agent_id, decision in batch.proposals[0].decisions.items()
    } == {"agent0": 0, "agent1": 0}
    debug = rule_maker.get_last_debug()
    assert debug["action_search"]["coarse_conflict_detected"] is True
    assert (
        debug["action_search"]["final_feasibility_authority"]
        == "normal_planner"
    )


def test_lane_change_commitment_blocks_reversal_until_target_lane_entry():
    vehicle = _vehicle("agent0", 20.0, 0.0, 1, speed_km_h=20.0)
    env = _env(agents={"agent0": vehicle}, traffic=[])
    rule_maker = MultiAgentRuleMaker(
        target_speed_km_h=30.0,
        horizon_s=2.0,
        locked_on_reset=False,
        lane_change_preference=20.0,
        lc_cost=0.0,
        w_mobil=0.0,
        w_keep_bias=0.0,
    )

    first_batch = rule_maker.propose_joint_actions(
        env, ["agent0"], planner_batch={}
    )
    first = first_batch.proposals[0].decisions
    committed_action = int(first["agent0"]["action"])
    assert committed_action in (-1, 1)
    assert rule_maker.get_last_debug()["lane_change_commitments"]["pending"] == {}
    rule_maker.accept_joint_action(first_batch.batch_id, 0)
    assert rule_maker.get_last_debug()["lane_change_commitments"]["pending"]

    rule_maker.lane_change_preference = -100.0
    second = rule_maker.compute(env, ["agent0"], planner_batch={})
    assert int(second["agent0"]["action"]) == committed_action
    active = rule_maker.get_last_debug()["lane_change_commitments"]["active"]
    assert active["agent0"]["action"] == committed_action

    target_lane_id = 1 + committed_action
    target_y = {0: 3.5, 2: -3.5}[target_lane_id]
    env.agents["agent0"] = _vehicle(
        "agent0", 24.0, target_y, target_lane_id, speed_km_h=20.0
    )
    third = rule_maker.compute(env, ["agent0"], planner_batch={})
    debug = rule_maker.get_last_debug()

    assert debug["lane_change_commitments"]["active"] == {}
    assert debug["lane_change_commitments"]["completed"]["agent0"][
        "completion_reason"
    ] == "converged_to_target_lane_pose"
    assert int(third["agent0"]["action"]) != committed_action


def test_reset_clears_lane_change_commitment_state():
    env = _env(
        agents={"agent0": _vehicle("agent0", 20.0, 0.0, 1)}, traffic=[]
    )
    rule_maker = MultiAgentRuleMaker(
        locked_on_reset=False,
        lane_change_preference=20.0,
        lc_cost=0.0,
    )
    batch = rule_maker.propose_joint_actions(env, ["agent0"], planner_batch={})
    rule_maker.accept_joint_action(batch.batch_id, batch.proposals[0].proposal_id)
    assert rule_maker._pending_lane_change_commitments

    rule_maker.reset(env, ["agent0"])

    assert rule_maker._lane_change_commitments == {}
    assert rule_maker._pending_lane_change_commitments == {}


def test_proposals_exclude_actions_without_hard_valid_modes():
    env = _env(
        agents={"agent0": _vehicle("agent0", 20.0, 0.0, 1)},
        traffic=[],
    )
    rule_maker = MultiAgentRuleMaker(
        locked_on_reset=False,
        lane_change_preference=20.0,
        lc_cost=0.0,
    )

    batch = rule_maker.propose_joint_actions(
        env,
        ["agent0"],
        planner_batch={},
        hard_valid_modes_by_action={
            "agent0": {-1: (), 0: (0, 9), 1: ()}
        },
    )

    assert batch.proposals
    assert all(
        int(proposal.decisions["agent0"]["action"]) == 0
        for proposal in batch.proposals
    )
    assert all(
        proposal.decisions["agent0"]["hard_mode_action_valid"]
        for proposal in batch.proposals
    )
    debug = rule_maker.get_last_debug()
    assert debug["hard_mode_action_rejections"]


def test_s9_forces_left_for_every_ego_remaining_on_source_lane():
    env = SimpleNamespace(
        config={"scenario_id": "S9_narrow_channel_negotiation"},
        _scenario_orchestrator=SimpleNamespace(
            _actor_manifest={"blocking_actor": {}}
        ),
    )
    rule_maker = MultiAgentRuleMaker(locked_on_reset=False)
    rule_maker._decision_step = 6

    for agent_id in ("agent0", "agent1", "agent2"):
        left = {
            "agent_id": agent_id,
            "action": -1,
            "source_lane_index": ("10C0_0_", "10C0_1_", 1),
            "target_lane_index": ("10C0_0_", "10C0_1_", 0),
        }
        keep = dict(left, action=0, target_lane_index=left["source_lane_index"])

        assert rule_maker._is_s9_forced_route_candidate(env, left)
        assert not rule_maker._is_s9_forced_route_candidate(env, keep)


def test_s9_serial_fallback_moves_first_pending_ego_only():
    env = SimpleNamespace(config={"scenario_id": "S9_narrow_channel_negotiation"})
    agent_ids = ["agent0", "agent1", "agent2"]
    candidates = {
        agent_id: [
            {"agent_id": agent_id, "action": -1},
            {"agent_id": agent_id, "action": 0},
        ]
        for agent_id in agent_ids
    }
    forced = tuple(candidates[agent_id][0] for agent_id in agent_ids)

    fallback = MultiAgentRuleMaker._s9_serial_forced_fallback_combo(
        env, agent_ids, candidates, forced
    )

    assert fallback is not None
    assert [int(candidate["action"]) for candidate in fallback] == [-1, 0, 0]
