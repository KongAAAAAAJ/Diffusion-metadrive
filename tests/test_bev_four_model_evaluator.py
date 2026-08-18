from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from evaluation.bev_four_model_evaluator import (
    DIAGNOSTIC_EVAL_SCENARIOS,
    ModelEvaluationConfig,
    ModelEvaluationError,
    ReproducibilityToleranceConfig,
    _configure_deterministic_inference,
    _behavior_sha256,
    _empty_metrics,
    _execution_mask_or_record_rejection,
    _initial_state_signature,
    _initial_scene_sha256,
    _validate_common_reward_binding,
    _validate_grpo_evaluation_eligibility,
    _summarize,
    compare_models,
    compare_repeated_reports,
)
from evaluation.bev_model_manifest import ComparisonSpec
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
    assert result["execution"]["rejection_rate"] == pytest.approx(0.5)
    assert result["timing"]["model_inference_ms"]["p95_ms"] == pytest.approx(19.5)


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
    report["models"]["stage1_a"]["artifacts"] = {"root": "/tmp/repeat_2"}
    assert _behavior_sha256(report) == baseline
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
        "calibration_gate_bypassed": False,
        "calibration_report_passed": True,
        "calibration_blockers": [],
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
        ("calibration_gate_bypassed", True),
        ("calibration_report_passed", False),
        ("calibration_blockers", ["tracking"]),
    ):
        invalid = {**formal, field: bad_value}
        with pytest.raises(ModelEvaluationError, match=field):
            _validate_grpo_evaluation_eligibility(
                invalid, formal=True, model_id="grpo_open"
            )


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
        "initial_state_sha256": "c" * 64,
        "initial_scene_sha256": "d" * 64,
        "model_order": list(model_order),
        "models": models,
        "comparisons": compare_models(models, comparisons),
    }


def test_single_model_and_explicit_pairwise_comparisons() -> None:
    single = _repeat_report(gap_m=6.0, model_order=("stage1_a",))
    assert single["comparisons"] == {}
    result = _repeat_report(gap_m=6.0)
    open_reward = result["comparisons"]["open_vs_stage1"]["metrics"][
        "efficiency.joint_reward_mean"
    ]
    assert open_reward["candidate_minus_baseline"] == pytest.approx(0.2)
    assert open_reward["conclusion"] == "better"


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
    changed["models"]["stage1_a"]["formation"]["mean_error_m"] = 1.5
    metric_result = compare_repeated_reports(
        [_repeat_report(gap_m=6.0), changed],
        ReproducibilityToleranceConfig(distance_m=0.1),
    )
    assert metric_result["continuous_metrics_within_tolerance"] is False
    assert metric_result["tolerance_gate_passed"] is False
