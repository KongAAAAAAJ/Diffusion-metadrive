from __future__ import annotations

import copy
import math
from types import SimpleNamespace
from types import MethodType

import numpy as np
import pytest
import torch

import evaluation.bev_four_model_evaluator as evaluator_module
from evaluation.bev_four_model_evaluator import (
    DIAGNOSTIC_EVAL_SCENARIOS,
    ModelEvaluationConfig,
    ModelEvaluationError,
    ReproducibilityToleranceConfig,
    _absolute_distribution,
    _configure_deterministic_inference,
    _behavior_sha256,
    _build_model_aggregates,
    _closing_risk_observations,
    _corridor_gap,
    _exact_binomial_ci,
    _empty_metrics,
    _finish_episode_metrics,
    _formal_conclusions_eligible,
    _formation_recovery,
    _execution_mask_or_record_rejection,
    _initial_state_signature,
    _initial_scene_sha256,
    _map_trajectory_controls,
    _validate_common_reward_binding,
    _validate_grpo_application_contract,
    _validate_grpo_evaluation_eligibility,
    _summarize,
    _aggregate_episode_metrics,
    _new_episode_state,
    _record_safe_stop_rejection,
    _step_precomputed_trajectory_controls,
    _timed_optimizer_call,
    _update_episode_state,
    compare_models,
    compare_repeated_reports,
    evaluate_models_repeated,
)
from envs.platoon_env import PlatoonEnv
from evaluation.bev_model_manifest import ComparisonSpec
from models.bev_planner.joint_reward import (
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256,
)
from models.bev_planner.trajectory_optimizer import TrajectoryOptimizationError
from scenarios.bev_round13_contract import HOLDOUT_SEEDS, PRIMARY_S5_S9_SCENARIOS


def test_evaluation_config_is_strict() -> None:
    with pytest.raises(ModelEvaluationError):
        ModelEvaluationConfig(device="auto")
    with pytest.raises(ModelEvaluationError):
        ModelEvaluationConfig(max_steps=0)
    with pytest.raises(ModelEvaluationError):
        ModelEvaluationConfig(device="cpu", save_visualizations=True)
    with pytest.raises(ModelEvaluationError):
        ModelEvaluationConfig(device="cpu", video_fps=0)
    with pytest.raises(ModelEvaluationError):
        ModelEvaluationConfig(device="cpu", topdown_screen_size=0)
    with pytest.raises(ModelEvaluationError):
        ModelEvaluationConfig(device="cpu", topdown_film_size=0)
    config = ModelEvaluationConfig(device="cpu")
    assert not hasattr(config, "inference_p95_limit_ms")
    assert config.v2_planning_tick_p95_limit_ms == 200.0
    assert config.scenarios == PRIMARY_S5_S9_SCENARIOS
    assert config.seeds == HOLDOUT_SEEDS
    assert DIAGNOSTIC_EVAL_SCENARIOS == PRIMARY_S5_S9_SCENARIOS
    with pytest.raises(ModelEvaluationError):
        ModelEvaluationConfig(
            device="cpu",
            scenarios=(("S1_free_cruise_straight", "R3_mainline_straight"),),
        )


def test_metric_aggregation_computes_rates_and_p95() -> None:
    raw = {
        "roles": {
            name: {
                "collision": 1,
                "out_of_road": 0,
                "progress": [1.0, 3.0],
                "speed_km_h": [10.0, 20.0],
                "minimum_gap_m": [6.0, 8.0],
                "selected_modes": [0, 9],
                "stop": 1,
                "acceleration_mps2": [1.0],
                "jerk": [0.2],
                "yaw_rate_rad_s": [0.1],
                "steering_change": [0.05],
            }
            for name in ("agent0", "agent1", "agent2")
        },
        "episode_collision": 1,
        "episode_out_of_road": 0,
        "episode_completed": 1,
        "execution_rejections": [
            {"scenario": "S8", "seed": 31, "step": 12, "reason": "infeasible"}
        ],
        "gap_5m_violation": 0,
        "gap_7m_violation": 1,
        "formation_error": [1.0, 2.0],
        "formation_spread": [2.0],
        "recovery_time_s": [1.0],
        "joint_reward": [0.5, 1.5],
        "timing": {
            "bev_build_ms": [1.0, 3.0],
            "model_inference_ms": [10.0, 20.0],
            "control_mapping_ms": [1.0, 2.0],
            "planning_tick_ms": [12.0, 25.0],
        },
    }
    result = _summarize(raw, 2)
    assert result["joint_safety"]["collision_rate"] == pytest.approx(0.5)
    assert result["roles"]["agent0"]["stop_rate"] == pytest.approx(0.5)
    assert result["execution"]["rejection_count"] == 1
    assert result["execution"]["rejection_episode_rate"] == pytest.approx(0.5)
    assert result["timing"]["model_inference_ms"]["p95_ms"] == pytest.approx(19.5)
    assert result["timing"]["planning_tick_ms"]["p99_ms"] == pytest.approx(24.87)


def test_shared_corridor_gap_excludes_adjacent_lane_and_keeps_obb_support() -> None:
    first = SimpleNamespace(
        position=(0.0, 0.0), heading_theta=0.0, LENGTH=5.74, WIDTH=2.3
    )
    same_lane = SimpleNamespace(
        position=(10.0, 0.0), heading_theta=0.0, LENGTH=5.74, WIDTH=2.3
    )
    adjacent_lane = SimpleNamespace(
        position=(2.0, 3.5), heading_theta=0.0, LENGTH=5.74, WIDTH=2.3
    )

    assert _corridor_gap(first, same_lane) == pytest.approx(4.26)
    assert _corridor_gap(first, adjacent_lane) is None


def test_closing_ttc_and_drac_ignore_non_closing_pairs() -> None:
    ttc, drac = _closing_risk_observations(
        {"pair": 4.0}, {"pair": 6.0}, dt_s=1.0
    )
    assert ttc == pytest.approx([2.0])
    assert drac == pytest.approx([0.5])
    assert _closing_risk_observations(
        {"pair": 6.0}, {"pair": 4.0}, dt_s=1.0
    ) == ([], [])
    slow_ttc, slow_drac = _closing_risk_observations(
        {"pair": 0.05}, {"pair": 0.10}, dt_s=1.0
    )
    assert slow_ttc == pytest.approx([1.0])
    assert slow_drac == pytest.approx([0.0125])


def test_exact_binomial_interval_and_recovery_semantics() -> None:
    assert _exact_binomial_ci(0, 10)[0] == 0.0
    assert _exact_binomial_ci(10, 10)[1] == 1.0
    recovered = _formation_recovery([True, True, False, False, True], dt_s=0.1)
    assert recovered == {
        "unlock_count": 1,
        "unlocked_duration_s": pytest.approx(0.2),
        "relock_recovery_success": True,
        "recovery_censored": False,
        "relock_recovery_time_s": pytest.approx(0.2),
    }
    assert _formation_recovery([True, False, False], dt_s=0.1)[
        "recovery_censored"
    ] is True
    assert _formation_recovery([True, True], dt_s=0.1)[
        "relock_recovery_success"
    ] is None


def test_empty_comfort_observation_is_null_but_observed_zero_stays_zero() -> None:
    assert _absolute_distribution([], "longitudinal_jerk", "mps3") == {
        "longitudinal_jerk_abs_mean_mps3": None,
        "longitudinal_jerk_abs_p95_mps3": None,
        "longitudinal_jerk_abs_max_mps3": None,
    }
    observed = _absolute_distribution([0.0], "longitudinal_jerk", "mps3")
    assert set(observed.values()) == {0.0}


def test_episode_aggregation_has_bounded_rejection_rates_and_binomial_ci() -> None:
    first = _episode_metric_row(
        model="stage1_a", gap_m=4.0, collision=True, progress_m=10.0
    )
    second = _episode_metric_row(
        model="stage1_a", gap_m=6.0, collision=False, progress_m=20.0, seed=47
    )
    first["safety"]["execution_rejected"] = True
    first["safety"]["execution_rejection_tick_fraction"] = 0.25

    aggregate = _aggregate_episode_metrics([first, second])

    assert aggregate["safety"]["collision_count"] == 1
    assert aggregate["safety"]["collision_rate"] == pytest.approx(0.5)
    assert len(aggregate["safety"]["collision_rate_ci95"]) == 2
    assert aggregate["safety"]["execution_rejection_episode_rate"] == pytest.approx(
        0.5
    )
    assert aggregate["safety"][
        "execution_rejection_tick_fraction_mean"
    ] == pytest.approx(0.125)
    assert 0.0 <= aggregate["safety"]["execution_rejection_episode_rate"] <= 1.0


def test_physical_dynamics_wrap_heading_and_preserve_per_agent_navigation() -> None:
    class Summary:
        calls = 0

        def get_episode_summary(self):
            self.calls += 1
            return {"scenario_realized": True, "functional_success": False}

    vehicles = {}
    for role, total in enumerate((100.0, 120.0, 140.0)):
        vehicles[f"agent{role}"] = SimpleNamespace(
            position=np.asarray([-10.0 * role, 0.0]),
            heading_theta=math.pi - 0.05,
            speed_km_h=3.6,
            LENGTH=5.74,
            WIDTH=2.3,
            navigation=SimpleNamespace(travelled_length=10.0 * role, total_length=total),
        )
    env = SimpleNamespace(
        agents=vehicles,
        engine=SimpleNamespace(
            traffic_manager=SimpleNamespace(_traffic_vehicles=[])
        ),
        _scenario_orchestrator=Summary(),
        _pending_low_level_actions={
            agent_id: np.asarray([0.1, 0.2]) for agent_id in vehicles
        },
    )
    rule_maker = SimpleNamespace(
        is_formation_locked=True, ideal_following_distance_m=7.0
    )
    state = _new_episode_state(env, rule_maker)
    pre_poses = np.asarray(
        [
            [*vehicles[agent_id].position, vehicles[agent_id].heading_theta]
            for agent_id in ("agent0", "agent1", "agent2")
        ]
    )
    for vehicle in vehicles.values():
        vehicle.position = np.asarray(vehicle.position) + np.asarray([-1.0, 0.0])
        vehicle.heading_theta = -math.pi + 0.05
        vehicle.navigation.travelled_length += 2.0
    _update_episode_state(
        state,
        env=env,
        rule_maker=rule_maker,
        step_info={
            agent_id: {
                "scenario_realized": True,
                "functional_success": False,
            }
            for agent_id in vehicles
        },
        pre_poses=pre_poses,
        dt_s=1.0,
    )
    assert env._scenario_orchestrator.calls == 0
    assert state["comfort"]["agent0"]["yaw_rate"] == pytest.approx([0.1])
    second_pre_poses = np.asarray(
        [
            [*vehicles[agent_id].position, vehicles[agent_id].heading_theta]
            for agent_id in ("agent0", "agent1", "agent2")
        ]
    )
    for vehicle in vehicles.values():
        vehicle.position = np.asarray(vehicle.position) + np.asarray([-2.0, 0.0])
        vehicle.heading_theta = -math.pi + 0.15
        vehicle.navigation.travelled_length += 3.0
    _update_episode_state(
        state,
        env=env,
        rule_maker=rule_maker,
        step_info={
            agent_id: {
                "scenario_realized": True,
                "functional_success": True,
            }
            for agent_id in vehicles
        },
        pre_poses=second_pre_poses,
        dt_s=1.0,
    )
    comfort = state["comfort"]["agent0"]
    assert comfort["longitudinal_jerk"][-1] == pytest.approx(
        comfort["longitudinal_acceleration"][-1]
        - comfort["longitudinal_acceleration"][-2]
    )
    assert comfort["yaw_acceleration"][-1] == pytest.approx(
        comfort["yaw_rate"][-1] - comfort["yaw_rate"][-2]
    )
    assert env._scenario_orchestrator.calls == 0
    episode = _finish_episode_metrics(
        state,
        model_id="stage1_a",
        scenario=("S9", "R8"),
        seed=31,
        dt_s=1.0,
        executed_steps=2,
        collision=False,
        out_of_road=False,
        role_collision={agent_id: False for agent_id in vehicles},
        role_out={agent_id: False for agent_id in vehicles},
        execution_rejected=False,
        rejection_steps=(),
    )
    assert episode["efficiency"]["per_agent"]["agent0"][
        "route_progress_m"
    ] == pytest.approx(5.0)
    assert episode["efficiency"]["per_agent"]["agent2"][
        "route_progress_fraction"
    ] == pytest.approx(5.0 / 120.0)
    assert episode["efficiency"]["first_stable_success_time_s"] == pytest.approx(
        2.0
    )


def test_terminal_metric_snapshot_preserves_final_progress_geometry_and_comfort() -> None:
    class Summary:
        calls = 0

        def get_episode_summary(self):
            self.calls += 1
            raise AssertionError("rollout metrics must consume the returned step info")

    vehicles = {
        f"agent{role}": SimpleNamespace(
            position=np.asarray([-10.0 * role, 0.0]),
            heading_theta=0.0,
            speed_km_h=3.6 * (role + 1),
            LENGTH=5.74,
            WIDTH=2.3,
            navigation=SimpleNamespace(
                travelled_length=10.0 * role, total_length=100.0 + 10.0 * role
            ),
        )
        for role in range(3)
    }
    env = SimpleNamespace(
        agents=dict(vehicles),
        engine=SimpleNamespace(
            traffic_manager=SimpleNamespace(_traffic_vehicles=[])
        ),
        _scenario_orchestrator=Summary(),
        _pending_low_level_actions={
            agent_id: np.asarray([0.1, 0.2]) for agent_id in vehicles
        },
    )
    rule_maker = SimpleNamespace(
        is_formation_locked=True, ideal_following_distance_m=7.0
    )
    state = _new_episode_state(env, rule_maker)
    pre_poses = np.asarray(
        [
            [*vehicles[agent_id].position, vehicles[agent_id].heading_theta]
            for agent_id in ("agent0", "agent1", "agent2")
        ]
    )
    terminal = vehicles["agent2"]
    terminal.position = terminal.position + np.asarray([-2.0, 0.0])
    terminal.heading_theta = 0.1
    terminal.navigation.travelled_length += 2.0
    terminal_snapshot = PlatoonEnv._evaluation_metric_snapshot(terminal)
    del env.agents["agent2"]
    step_info = {
        agent_id: {
            "scenario_realized": True,
            "functional_success": True,
            **(
                {"evaluation_metric_snapshot": terminal_snapshot}
                if agent_id == "agent2"
                else {}
            ),
        }
        for agent_id in vehicles
    }

    _update_episode_state(
        state,
        env=env,
        rule_maker=rule_maker,
        step_info=step_info,
        pre_poses=pre_poses,
        dt_s=1.0,
    )

    assert env._scenario_orchestrator.calls == 0
    assert state["last_navigation"]["agent2"][0] == pytest.approx(22.0)
    assert state["comfort"]["agent2"]["yaw_rate"] == pytest.approx([0.1])
    assert state["locked_speed_spreads_mps"] == pytest.approx([2.0])
    episode = _finish_episode_metrics(
        state,
        model_id="stage1_a",
        scenario=("S9", "R8"),
        seed=31,
        dt_s=1.0,
        executed_steps=1,
        collision=False,
        out_of_road=False,
        role_collision={agent_id: False for agent_id in vehicles},
        role_out={agent_id: False for agent_id in vehicles},
        execution_rejected=False,
        rejection_steps=(),
    )
    assert episode["efficiency"]["per_agent"]["agent2"][
        "route_progress_m"
    ] == pytest.approx(2.0)
    assert episode["efficiency"]["first_stable_success_time_s"] == pytest.approx(
        1.0
    )


def test_platoon_info_keeps_metric_snapshot_separate_from_scenario_summary() -> None:
    class Summary:
        def get_episode_summary(self):
            return {
                "scenario_realized": True,
                "functional_success": False,
                "route_completion": {"all_agents_changed_lane": True},
            }

    vehicle = SimpleNamespace(
        position=np.asarray([4.0, 5.0]),
        heading_theta=0.25,
        speed_km_h=18.0,
        LENGTH=5.74,
        WIDTH=2.3,
        navigation=SimpleNamespace(travelled_length=12.0, total_length=100.0),
    )
    env = SimpleNamespace(
        _agent_ids=["agent0"],
        agents={"agent0": vehicle},
        _evaluation_metric_snapshots={
            "agent0": PlatoonEnv._evaluation_metric_snapshot(vehicle)
        },
        _scenario_orchestrator=Summary(),
        _platoon_reward_cache=None,
        _last_actions={},
        _refresh_evaluation_metric_snapshots=None,
    )
    env._evaluation_metric_snapshot = PlatoonEnv._evaluation_metric_snapshot
    env._refresh_evaluation_metric_snapshots = MethodType(
        PlatoonEnv._refresh_evaluation_metric_snapshots, env
    )
    env._capture_evaluation_scenario_summary = MethodType(
        PlatoonEnv._capture_evaluation_scenario_summary, env
    )
    env._cfg_bool = lambda *_args: False
    env._compute_min_gap = lambda: 20.0
    env.get_agent_role = lambda _agent_id: "leader"
    env.get_formation_relation_state = lambda _agent_id: np.zeros((12,))
    env._compute_agent_formation_error = lambda _agent_id: 0.0
    env._compute_progress = lambda _agent_id: 1.0
    env._agent_speed_km_h = lambda _agent_id: 18.0

    info = PlatoonEnv._build_info_dict(
        env,
        "low_level",
        actions={"agent0": np.zeros((2,))},
        base_info={"agent0": {"route_completion": 0.12}},
    )

    assert info["agent0"]["route_completion"] == {
        "all_agents_changed_lane": True
    }
    assert info["agent0"]["evaluation_metric_snapshot"] == {
        "navigation_travelled_length_m": 12.0,
        "navigation_total_length_m": 100.0,
        "position_xy_m": [4.0, 5.0],
        "heading_rad": 0.25,
        "speed_km_h": 18.0,
        "length_m": 5.74,
        "width_m": 2.3,
    }


def test_terminal_summary_is_captured_before_agent_removal_and_not_recomputed() -> None:
    vehicles = {
        f"agent{role}": SimpleNamespace(
            position=np.asarray([-10.0 * role, 0.0]),
            heading_theta=0.0,
            speed_km_h=18.0,
            LENGTH=5.74,
            WIDTH=2.3,
            navigation=SimpleNamespace(
                travelled_length=float(role), total_length=100.0
            ),
        )
        for role in range(3)
    }
    observed_agent_counts = []
    env = SimpleNamespace(
        _agent_ids=list(vehicles),
        agents=dict(vehicles),
        _evaluation_metric_snapshots={},
        _evaluation_metric_snapshot_pending=True,
        _evaluation_scenario_summary=None,
        _platoon_reward_cache=None,
        _last_actions={},
    )
    env._scenario_orchestrator = SimpleNamespace(
        get_episode_summary=lambda: (
            observed_agent_counts.append(len(env.agents))
            or {
                "scenario_realized": True,
                "functional_success": True,
            }
        )
    )
    env._evaluation_metric_snapshot = PlatoonEnv._evaluation_metric_snapshot
    env._refresh_evaluation_metric_snapshots = MethodType(
        PlatoonEnv._refresh_evaluation_metric_snapshots, env
    )
    env._capture_evaluation_scenario_summary = MethodType(
        PlatoonEnv._capture_evaluation_scenario_summary, env
    )
    env._cfg_bool = lambda *_args: False
    env._compute_min_gap = lambda: 20.0
    env.get_agent_role = lambda agent_id: (
        "leader" if agent_id == "agent0" else "follower"
    )
    env.get_formation_relation_state = lambda _agent_id: np.zeros((12,))
    env._compute_agent_formation_error = lambda _agent_id: 0.0
    env._compute_progress = lambda _agent_id: 1.0
    env._agent_speed_km_h = lambda _agent_id: 18.0

    PlatoonEnv._refresh_evaluation_metric_snapshots_once(env)
    del env.agents["agent2"]
    env._scenario_orchestrator.get_episode_summary = lambda: (_ for _ in ()).throw(
        AssertionError("_build_info_dict must use the pre-removal summary")
    )
    info = PlatoonEnv._build_info_dict(
        env,
        "low_level",
        actions={agent_id: np.zeros((2,)) for agent_id in env.agents},
        base_info={agent_id: {} for agent_id in vehicles},
    )

    assert observed_agent_counts == [3]
    assert info["agent2"]["functional_success"] is True
    assert info["agent2"]["evaluation_metric_snapshot"][
        "navigation_travelled_length_m"
    ] == pytest.approx(2.0)


def test_terminal_metric_snapshot_refresh_runs_once_per_physics_step() -> None:
    calls = []
    env = SimpleNamespace(_evaluation_metric_snapshot_pending=True)
    env._refresh_evaluation_metric_snapshots = lambda: calls.append("refresh")
    env._capture_evaluation_scenario_summary = lambda: {
        "functional_success": True
    }

    PlatoonEnv._refresh_evaluation_metric_snapshots_once(env)
    PlatoonEnv._refresh_evaluation_metric_snapshots_once(env)

    assert calls == ["refresh"]
    assert env._evaluation_metric_snapshot_pending is False
    assert env._evaluation_scenario_summary == {"functional_success": True}


def test_precomputed_trajectory_controls_are_mapped_once_and_executed_unchanged() -> None:
    class StatefulEnv:
        def __init__(self):
            self.mapping_calls = {agent_id: 0 for agent_id in ("agent0", "agent1", "agent2")}
            self.executed = None

        def trajectory_to_control(self, agent_id, _trajectory):
            self.mapping_calls[agent_id] += 1
            return np.asarray(
                [self.mapping_calls[agent_id], -self.mapping_calls[agent_id]],
                dtype=np.float32,
            )

        def low_level_step(self, controls, *, control_mode):
            self.executed = {key: value.copy() for key, value in controls.items()}
            assert control_mode == "trajectory"
            return {}, {}, {}, {}, {}

    env = StatefulEnv()
    trajectories = {
        agent_id: np.zeros((8, 3), dtype=np.float32)
        for agent_id in ("agent0", "agent1", "agent2")
    }
    controls = _map_trajectory_controls(
        env,
        trajectories,
        model_id="stage1_a",
        scenario=("S5_hard_brake_lead", "R1_entry_straight"),
        seed=31,
        step_index=4,
    )
    _step_precomputed_trajectory_controls(env, trajectories, controls)

    assert env.mapping_calls == {"agent0": 1, "agent1": 1, "agent2": 1}
    for agent_id in controls:
        assert np.array_equal(env.executed[agent_id], controls[agent_id])
        assert np.array_equal(
            env._pending_step_trajectories[agent_id], trajectories[agent_id]
        )


def test_optimizer_timing_accumulates_selected_and_safe_stop_attempts(
    monkeypatch,
) -> None:
    clock = iter((1.0, 1.1, 2.0, 2.25))
    monkeypatch.setattr(evaluator_module.time, "perf_counter", lambda: next(clock))
    elapsed_ms = [0.0]

    assert _timed_optimizer_call(elapsed_ms, lambda: "selected") == "selected"
    assert _timed_optimizer_call(elapsed_ms, lambda: "safe_stop") == "safe_stop"

    assert elapsed_ms[0] == pytest.approx(350.0)


def test_execution_mask_failure_is_recorded_as_episode_rejection() -> None:
    class RejectingOptimizer:
        def execution_mode_valid_mask(self, *args):
            raise TrajectoryOptimizationError(
                "calibrated execution contract rejected STOP"
            )

    values = SimpleNamespace(
        coarse_trajectories=torch.zeros((3, 10, 8, 3)).numpy(),
        ego_state=torch.zeros((3, 8)).numpy(),
        mode_valid_mask=torch.ones((3, 10), dtype=torch.bool).numpy(),
    )
    raw = _empty_metrics()

    result = _execution_mask_or_record_rejection(
        values,
        optimizer=RejectingOptimizer(),
        raw=raw,
        model_id="stage1_a",
        scenario=("S6_background_merge_in", "R6_mainline_merge_approach"),
        seed=31,
        step_index=92,
    )

    assert result is None
    assert raw["execution_rejections"] == [
        {
            "model": "stage1_a",
            "scenario": "S6_background_merge_in",
            "route": "R6_mainline_merge_approach",
            "seed": 31,
            "step": 92,
            "stage": "execution_mode_mask",
            "reason": "calibrated execution contract rejected STOP",
        }
    ]


@pytest.mark.parametrize(
    "stage", ("active_commitment_compatibility", "proposal_action_match")
)
def test_rule_mismatch_safe_stop_marks_episode_rejected(stage: str) -> None:
    raw = _empty_metrics()
    rejected = _record_safe_stop_rejection(
        raw,
        {
            "scenario": "S9_narrow_channel_negotiation",
            "route": "R8_narrow_channel",
            "seed": 31,
            "step": 12,
            "stage": stage,
        },
    )

    assert rejected is True
    assert raw["execution_rejections"][0]["stage"] == stage


@pytest.mark.parametrize(
    ("run_mode", "all_gates_passed", "expected"),
    (
        ("diagnostic", True, False),
        ("diagnostic", False, False),
        ("formal", False, False),
        ("formal", True, True),
    ),
)
def test_formal_conclusion_eligibility_requires_mode_and_gates(
    run_mode: str, all_gates_passed: bool, expected: bool
) -> None:
    assert (
        _formal_conclusions_eligible(run_mode, all_gates_passed) is expected
    )


def test_deterministic_inference_requires_process_hash_seed(monkeypatch) -> None:
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    with pytest.raises(ModelEvaluationError, match="PYTHONHASHSEED=0"):
        _configure_deterministic_inference(torch.device("cpu"))
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    _configure_deterministic_inference(torch.device("cpu"))
    assert torch.are_deterministic_algorithms_enabled()


def test_initial_state_signature_is_joint_first_and_strict() -> None:
    env = SimpleNamespace(
        agents={
            f"agent{role}": SimpleNamespace(
                position=(float(role), -float(role)),
                heading_theta=0.1 * role,
                speed_km_h=20.0 + role,
            )
            for role in range(3)
        }
    )
    signature = _initial_state_signature(env)
    assert signature.shape == (3, 4)
    assert signature[2].tolist() == pytest.approx([2.0, -2.0, 0.2, 22.0])
    del env.agents["agent1"]
    with pytest.raises(ModelEvaluationError, match="agent1"):
        _initial_state_signature(env)


def test_initial_scene_hash_includes_background_but_not_unstable_name() -> None:
    def build(background_name: str, background_speed: float):
        agents = {
            f"agent{role}": SimpleNamespace(
                name=f"agent{role}",
                position=(float(role), 0.0),
                heading_theta=0.0,
                speed_km_h=20.0,
                lane=None,
            )
            for role in range(3)
        }
        background = SimpleNamespace(
            name=background_name,
            position=(10.0, 2.0),
            heading_theta=0.1,
            speed_km_h=background_speed,
            lane=None,
        )
        engine = SimpleNamespace(
            traffic_manager=SimpleNamespace(_traffic_vehicles=[background]),
            get_policy=lambda _: SimpleNamespace(),
        )
        return SimpleNamespace(agents=agents, engine=engine)

    baseline = _initial_scene_sha256(build("uuid-one", 18.0))
    assert baseline == _initial_scene_sha256(build("uuid-two", 18.0))
    assert baseline != _initial_scene_sha256(build("uuid-two", 19.0))


def test_behavior_hash_excludes_timing_but_not_policy_metrics() -> None:
    model_order = ["stage1_a", "grpo_open", "grpo_exec"]
    report = {
        "model_order": model_order,
        "models": {
            name: {"joint_safety": {"collision_rate": 0.0}, "timing": {"p95": 1.0}}
            for name in model_order
        },
    }
    baseline = _behavior_sha256(report)
    report["models"]["stage1_a"]["timing"]["p95"] = 99.0
    assert _behavior_sha256(report) == baseline
    report["models"]["stage1_a"]["overall"] = {
        "timing": {"planning_tick_ms_p95_ms_mean": 100.0}
    }
    nested_baseline = _behavior_sha256(report)
    report["models"]["stage1_a"]["overall"]["timing"][
        "planning_tick_ms_p95_ms_mean"
    ] = 300.0
    assert _behavior_sha256(report) == nested_baseline
    report["models"]["stage1_a"]["artifacts"] = {"root": "/tmp/repeat_2"}
    assert _behavior_sha256(report) == nested_baseline
    report["models"]["stage1_a"]["joint_safety"]["collision_rate"] = 1.0
    assert _behavior_sha256(report) != baseline


def test_four_model_comparison_rejects_mixed_reward_hashes() -> None:
    common = ("stage2_joint_reward_v2", "a" * 64, "b" * 64)
    assert _validate_common_reward_binding(
        {"grpo_open": common, "grpo_exec": common}
    ) == {
        "reward_contract_version": "stage2_joint_reward_v2",
        "reward_contract_sha256": "a" * 64,
        "reward_config_sha256": "b" * 64,
    }
    with pytest.raises(ModelEvaluationError, match="mix reward contract/config"):
        _validate_common_reward_binding(
            {
                "grpo_open": common,
                "grpo_exec": ("stage2_joint_reward_v2", "a" * 64, "c" * 64),
            }
        )


def test_grpo_evaluation_flags_are_strict_in_smoke_and_formal_modes() -> None:
    smoke = {
        "run_mode": "smoke",
        "diagnostic_only": True,
        "eligible_for_formal_training": False,
        "calibration_required": False,
    }
    formal = {
        **smoke,
        "run_mode": "formal",
        "diagnostic_only": False,
        "eligible_for_formal_training": True,
    }
    _validate_grpo_evaluation_eligibility(
        smoke, formal=False, model_id="grpo_open"
    )
    _validate_grpo_evaluation_eligibility(
        formal, formal=True, model_id="grpo_open"
    )

    for field, bad_value in (
        ("run_mode", "smoke"),
        ("diagnostic_only", True),
        ("eligible_for_formal_training", False),
        ("calibration_required", True),
    ):
        invalid = {**formal, field: bad_value}
        with pytest.raises(ModelEvaluationError, match=field):
            _validate_grpo_evaluation_eligibility(
                invalid, formal=True, model_id="grpo_open"
            )


def test_grpo_open_evaluator_requires_exact_tau_d_application_contract() -> None:
    payload = {
        "reward_application_contract": GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
        "reward_application_contract_sha256": (
            GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256
        ),
        "reward_input_domain": "tau_d",
        "candidate_selection_domain": "tau_d",
        "execution_input_domain": "tau_cmd",
        "best_checkpoint_metric": "validation/raw_proxy_reward_mean",
        "tracking_expansion_enabled": False,
        "calibration_required": False,
    }
    _validate_grpo_application_contract(payload, model_id="grpo_open")
    payload["reward_input_domain"] = "tau_cmd"
    with pytest.raises(ModelEvaluationError, match="metadata mismatch"):
        _validate_grpo_application_contract(payload, model_id="grpo_open")
    payload["reward_input_domain"] = "tau_d"
    payload["reward_application_contract_sha256"] = "0" * 64
    with pytest.raises(ModelEvaluationError, match="SHA mismatch"):
        _validate_grpo_application_contract(payload, model_id="grpo_open")


def _episode_metric_row(
    *,
    model: str,
    gap_m: float,
    collision: bool,
    progress_m: float,
    scenario: str = "S9",
    route: str = "R8",
    seed: int = 31,
) -> dict:
    comfort_fields = {
        "longitudinal_acceleration_abs_mean_mps2": 0.5,
        "longitudinal_acceleration_abs_p95_mps2": 0.7,
        "longitudinal_acceleration_abs_max_mps2": 0.8,
        "lateral_acceleration_abs_mean_mps2": 0.1,
        "lateral_acceleration_abs_p95_mps2": 0.2,
        "lateral_acceleration_abs_max_mps2": 0.3,
        "longitudinal_jerk_abs_mean_mps3": 0.2,
        "longitudinal_jerk_abs_p95_mps3": 0.3,
        "longitudinal_jerk_abs_max_mps3": 0.4,
        "yaw_rate_abs_mean_rad_s": 0.1,
        "yaw_rate_abs_p95_rad_s": 0.2,
        "yaw_rate_abs_max_rad_s": 0.3,
        "yaw_acceleration_abs_mean_rad_s2": 0.1,
        "yaw_acceleration_abs_p95_rad_s2": 0.2,
        "yaw_acceleration_abs_max_rad_s2": 0.3,
        "steering_command_slew_abs_mean_per_s": 0.05,
        "steering_command_slew_abs_p95_per_s": 0.1,
        "steering_command_slew_abs_max_per_s": 0.2,
        "throttle_command_slew_abs_mean_per_s": 0.05,
        "throttle_command_slew_abs_p95_per_s": 0.1,
        "throttle_command_slew_abs_max_per_s": 0.2,
    }
    timing = {
        component: {
            "p50_ms": 10.0,
            "p95_ms": 20.0,
            "p99_ms": 22.0,
            "max_ms": 25.0,
            **(
                {"over_200_ms_fraction": 0.0}
                if component == "planning_tick_ms"
                else {}
            ),
        }
        for component in (
            "bev_build_ms",
            "model_inference_ms",
            "control_mapping_ms",
            "planning_tick_ms",
            "trajectory_optimizer_ms",
            "rule_maker_ms",
        )
    }
    return {
        "model": model,
        "scenario": scenario,
        "route": route,
        "seed": seed,
        "dt_s": 0.1,
        "steps": 100,
        "safety": {
            "collision": collision,
            "out_of_road": False,
            "per_agent": {
                agent_id: {"collision": collision, "out_of_road": False}
                for agent_id in ("agent0", "agent1", "agent2")
            },
            "background_gap_violation": gap_m < 5.0,
            "background_gap_exposure_fraction": max(5.0 - gap_m, 0.0),
            "background_gap_deficit_integral_m_s": max(5.0 - gap_m, 0.0),
            "minimum_background_gap_m": gap_m,
            "platoon_gap_violation": False,
            "platoon_gap_exposure_fraction": 0.0,
            "platoon_gap_deficit_integral_m_s": 0.0,
            "minimum_platoon_gap_m": 8.0,
            "minimum_ttc_s": None,
            "ttc_below_1_5_s_exposure_fraction": 0.0,
            "max_drac_mps2": 0.0,
            "execution_rejected": False,
            "execution_rejection_tick_fraction": 0.0,
        },
        "efficiency": {
            "scenario_realized": True,
            "functional_success_final": False,
            "functional_success_ever": False,
            "functional_success_stable": False,
            "first_stable_success_time_s": None,
            "per_agent": {
                agent_id: {
                    "route_progress_m": progress_m,
                    "route_progress_fraction": progress_m / 100.0,
                }
                for agent_id in ("agent0", "agent1", "agent2")
            },
            "team_mean_route_progress_m": progress_m,
            "team_min_route_progress_m": progress_m,
            "team_mean_route_progress_fraction": progress_m / 100.0,
            "team_min_route_progress_fraction": progress_m / 100.0,
            "completion_time_s": None,
            "episode_duration_s": 10.0,
        },
        "cooperation": {
            "locked_spacing_error_mean_m": 1.0,
            "locked_spacing_error_p95_m": 1.5,
            "locked_spacing_error_max_m": 2.0,
            "locked_speed_spread_mean_mps": 0.2,
            "unlock_count": 0,
            "unlocked_duration_s": 0.0,
            "relock_recovery_success": None,
            "recovery_censored": None,
            "relock_recovery_time_s": None,
        },
        "comfort": {
            "per_agent": {
                agent_id: dict(comfort_fields)
                for agent_id in ("agent0", "agent1", "agent2")
            }
        },
        "timing": timing,
    }


def _repeat_report(
    *,
    gap_m: float,
    collision: bool = False,
    model_order: tuple[str, ...] = ("stage1_a", "grpo_open", "grpo_exec"),
) -> dict:
    models = {}
    for index, name in enumerate(model_order):
        roles = {
            agent_id: {
                "collision_rate": float(collision),
                "out_of_road_rate": 0.0,
                "progress_mean_m": 10.0 + index,
                "speed_mean_km_h": 20.0,
                "minimum_gap_m": gap_m,
                "mode_distribution": {"0": 90, "9": 10},
                "stop_rate": 0.1,
                "acceleration_abs_mean_mps2": 0.5,
                "jerk_abs_mean": 0.2,
                "yaw_rate_abs_mean_rad_s": 0.1,
                "steering_change_abs_mean": 0.05,
            }
            for agent_id in ("agent0", "agent1", "agent2")
        }
        models[name] = {
            "episode_metrics": [
                _episode_metric_row(
                    model=name,
                    gap_m=gap_m,
                    collision=collision,
                    progress_m=10.0 + index,
                )
            ],
            "roles": roles,
            "joint_safety": {
                "collision_rate": float(collision),
                "out_of_road_rate": 0.0,
                "gap_5m_violation_rate": 0.1 * float(gap_m < 5.0),
                "gap_7m_violation_rate": 0.0,
            },
            "formation": {
                "mean_error_m": 1.0,
                "p95_error_m": 1.5,
                "maximum_spread_m": 2.0,
                "recovery_time_mean_s": 0.5,
            },
            "efficiency": {
                "completion_rate": 0.0,
                "joint_reward_mean": 1.0 + 0.2 * index,
            },
            "execution": {
                "rejection_count": 0,
                "rejection_rate": 0.0,
                "rejections": [],
                "mean_modes_removed_per_joint_state": 0.01,
            },
            "trajectory_optimization": {
                "intervention_ade_mean_m": 0.1,
                "intervention_fde_mean_m": 0.2,
                "retained_raw_fraction_mean": 0.9,
            },
            "episode_outcomes": [
                {
                    "scenario": "S9",
                    "route": "R8",
                    "seed": 31,
                    "collision": collision,
                    "out_of_road": False,
                    "completed": False,
                    "execution_rejected": False,
                    "gap_5m_violation": gap_m < 5.0,
                    "gap_7m_violation": False,
                    "minimum_background_gap_m": gap_m,
                    "minimum_platoon_gap_m": 8.0,
                }
            ],
            "deterministic_probe": {
                "input_and_noise_sha256": "a" * 64,
                "output_sha256": "b" * 64,
                "identical_replay": True,
            },
            "timing": {"planning_tick_ms": {"p95_ms": 100.0}},
        }
    comparisons = (
        (
            ComparisonSpec("open_vs_stage1", "stage1_a", "grpo_open"),
            ComparisonSpec("exec_vs_open", "grpo_open", "grpo_exec"),
        )
        if len(model_order) > 1
        else ()
    )
    return {
        "format": "bev_model_evaluation_v3",
        "run_mode": "formal",
        "all_gates_passed": True,
        "initial_state_sha256": "c" * 64,
        "initial_scene_sha256": "d" * 64,
        "model_order": list(model_order),
        "evaluation_protocol": {},
        "metric_definitions": {},
        "grpo_reward_binding": None,
        "models": models,
        "comparisons": compare_models(models, comparisons),
    }


def test_single_model_and_explicit_pairwise_comparisons() -> None:
    single = _repeat_report(gap_m=6.0, model_order=("stage1_a",))
    assert single["comparisons"] == {}
    result = _repeat_report(gap_m=6.0)
    open_progress = result["comparisons"]["open_vs_stage1"]["metrics"][
        "efficiency.team_mean_route_progress_m_mean"
    ]
    assert open_progress["candidate_minus_baseline"] == pytest.approx(1.0)
    assert open_progress["n_pairs"] == 1
    assert open_progress["ci95"] == pytest.approx([1.0, 1.0])
    assert "conclusion" not in open_progress


def test_model_aggregates_use_scenario_macro_but_keep_rate_ci_estimand() -> None:
    rows = [
        _episode_metric_row(
            model="stage1_a",
            gap_m=6.0,
            collision=False,
            progress_m=10.0,
            scenario=scenario,
            route=route,
            seed=seed,
        )
        for scenario, route, seed in (
            ("S5", "R1", 31),
            ("S5", "R1", 47),
            ("S9", "R8", 31),
            ("S9", "R8", 47),
        )
    ]
    rows[0]["efficiency"]["completion_time_s"] = 1.0
    rows[1]["efficiency"]["completion_time_s"] = 3.0
    rows[2]["efficiency"]["completion_time_s"] = 100.0
    rows[3]["efficiency"]["completion_time_s"] = None
    rows[0]["cooperation"]["relock_recovery_success"] = True
    rows[0]["cooperation"]["recovery_censored"] = False
    rows[1]["cooperation"]["relock_recovery_success"] = False
    rows[1]["cooperation"]["recovery_censored"] = True
    rows[2]["cooperation"]["relock_recovery_success"] = True
    rows[2]["cooperation"]["recovery_censored"] = False

    by_scenario, overall = _build_model_aggregates(rows)

    assert by_scenario["S5"]["efficiency"]["completion_time_s_mean"] == 2.0
    assert by_scenario["S9"]["efficiency"]["completion_time_s_mean"] == 100.0
    assert overall["efficiency"]["completion_time_s_mean"] == pytest.approx(51.0)
    recovery = overall["cooperation"]
    assert recovery["relock_recovery_success_sample_count"] == 3
    assert recovery["relock_recovery_success_rate"] == pytest.approx(2.0 / 3.0)
    assert recovery["relock_recovery_success_rate_ci95"] == pytest.approx(
        _exact_binomial_ci(2, 3)
    )
    assert overall["aggregation"] == "equal_weight_scenario_macro"

    with pytest.raises(ModelEvaluationError, match="equal episode counts"):
        _build_model_aggregates(rows[:-1])


def test_comparison_is_strictly_paired_and_scenario_macro_bootstrapped() -> None:
    baseline_rows = [
        _episode_metric_row(
            model="stage1_a",
            gap_m=6.0,
            collision=False,
            progress_m=10.0,
            scenario="S5",
            route="R1",
            seed=31,
        ),
        _episode_metric_row(
            model="stage1_a",
            gap_m=6.0,
            collision=False,
            progress_m=20.0,
            scenario="S9",
            route="R8",
            seed=31,
        ),
    ]
    candidate_rows = [
        _episode_metric_row(
            model="grpo_open",
            gap_m=6.0,
            collision=False,
            progress_m=11.0,
            scenario="S5",
            route="R1",
            seed=31,
        ),
        _episode_metric_row(
            model="grpo_open",
            gap_m=6.0,
            collision=False,
            progress_m=23.0,
            scenario="S9",
            route="R8",
            seed=31,
        ),
    ]
    spec = (ComparisonSpec("open_vs_stage1", "stage1_a", "grpo_open"),)
    compared = compare_models(
        {
            "stage1_a": {"episode_metrics": baseline_rows},
            "grpo_open": {"episode_metrics": candidate_rows},
        },
        spec,
    )
    metric = compared["open_vs_stage1"]["metrics"][
        "efficiency.team_mean_route_progress_m_mean"
    ]
    assert metric["candidate_minus_baseline"] == pytest.approx(2.0)
    assert metric["n_pairs"] == 2
    assert metric["ci95"] == pytest.approx([2.0, 2.0])

    with pytest.raises(ModelEvaluationError, match="episode sets do not match"):
        compare_models(
            {
                "stage1_a": {"episode_metrics": baseline_rows},
                "grpo_open": {"episode_metrics": candidate_rows[:1]},
            },
            spec,
        )
    with pytest.raises(ModelEvaluationError, match="duplicate comparison episode"):
        compare_models(
            {
                "stage1_a": {"episode_metrics": baseline_rows},
                "grpo_open": {
                    "episode_metrics": [candidate_rows[0], candidate_rows[0]]
                },
            },
            spec,
        )


def test_tolerance_repeat_gate_accepts_near_threshold_physics_drift() -> None:
    result = compare_repeated_reports(
        [_repeat_report(gap_m=5.0086), _repeat_report(gap_m=4.9834)]
    )
    assert result["initial_state_exact"] is True
    assert result["deterministic_model_replay"] is True
    assert result["critical_discrete_outcomes_exact"] is True
    assert result["continuous_metrics_within_tolerance"] is True
    assert result["statistical_conclusions_stable"] is True
    assert result["threshold_boundary_disagreements"]
    assert result["tolerance_gate_passed"] is True


def test_tolerance_repeat_gate_rejects_collision_or_large_metric_change() -> None:
    collision_result = compare_repeated_reports(
        [_repeat_report(gap_m=6.0), _repeat_report(gap_m=6.0, collision=True)]
    )
    assert collision_result["critical_discrete_outcomes_exact"] is False
    assert collision_result["tolerance_gate_passed"] is False

    changed = _repeat_report(gap_m=6.0)
    changed["models"]["stage1_a"]["episode_metrics"][0]["comfort"][
        "per_agent"
    ]["agent0"]["longitudinal_acceleration_abs_mean_mps2"] = 1.5
    metric_result = compare_repeated_reports(
        [_repeat_report(gap_m=6.0), changed],
        ReproducibilityToleranceConfig(distance_m=0.1),
    )
    assert metric_result["continuous_metrics_within_tolerance"] is False
    assert metric_result["tolerance_gate_passed"] is False


def test_repeat_gate_covers_v3_functional_recovery_and_ignores_timing_drift() -> None:
    functional = _repeat_report(gap_m=6.0)
    functional["models"]["stage1_a"]["episode_metrics"][0]["efficiency"][
        "functional_success_final"
    ] = True
    functional_result = compare_repeated_reports(
        [_repeat_report(gap_m=6.0), functional]
    )
    assert functional_result["critical_discrete_outcomes_exact"] is False

    recovery = _repeat_report(gap_m=6.0)
    recovery["models"]["stage1_a"]["episode_metrics"][0]["cooperation"][
        "relock_recovery_success"
    ] = True
    recovery["models"]["stage1_a"]["episode_metrics"][0]["cooperation"][
        "recovery_censored"
    ] = False
    recovery_result = compare_repeated_reports(
        [_repeat_report(gap_m=6.0), recovery]
    )
    assert recovery_result["critical_discrete_outcomes_exact"] is False

    timing = _repeat_report(gap_m=6.0)
    for model in timing["models"].values():
        model["episode_metrics"][0]["timing"]["planning_tick_ms"][
            "p95_ms"
        ] = 150.0
    timing["comparisons"] = compare_models(
        timing["models"],
        (
            ComparisonSpec("open_vs_stage1", "stage1_a", "grpo_open"),
            ComparisonSpec("exec_vs_open", "grpo_open", "grpo_exec"),
        ),
    )
    timing_result = compare_repeated_reports(
        [_repeat_report(gap_m=6.0), timing]
    )
    assert timing_result["tolerance_gate_passed"] is True


@pytest.mark.parametrize(
    ("repeat_gates", "tolerance_gate", "expected"),
    (
        ((True, True), True, True),
        ((True, False), True, False),
        ((True, True), False, False),
    ),
)
def test_repeated_wrapper_combines_hard_and_tolerance_gates_without_expanding_n(
    monkeypatch,
    tmp_path,
    repeat_gates: tuple[bool, bool],
    tolerance_gate: bool,
    expected: bool,
) -> None:
    reports = []
    for passed in repeat_gates:
        report = copy.deepcopy(_repeat_report(gap_m=6.0))
        report["all_gates_passed"] = passed
        reports.append(report)
    pending = iter(reports)
    monkeypatch.setattr(
        evaluator_module,
        "evaluate_models",
        lambda *_args, **_kwargs: next(pending),
    )
    monkeypatch.setattr(
        evaluator_module,
        "compare_repeated_reports",
        lambda *_args, **_kwargs: {
            "tolerance_gate_passed": tolerance_gate
        },
    )

    combined = evaluate_models_repeated(
        tmp_path / "manifest.json",
        tmp_path / "report.json",
        ModelEvaluationConfig(run_mode="formal", device="cpu"),
        repeats=2,
    )

    assert combined["all_gates_passed"] is expected
    assert combined["eligible_for_formal_conclusions"] is expected
    assert combined["evaluation_status"] == (
        "completed" if expected else "completed_with_gate_failure"
    )
    assert combined["models"] == reports[0]["models"]
    assert combined["comparisons"] == reports[0]["comparisons"]
    assert combined["comparisons"]["open_vs_stage1"]["metrics"][
        "safety.collision_rate"
    ]["n_pairs"] == 1
