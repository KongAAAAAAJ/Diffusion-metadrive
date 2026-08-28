from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pytest

from evaluation.bev_comparison_charts import (
    CHART_NAMES,
    COMFORT_SPECS,
    ComparisonChartError,
    _directional_color_delta,
    _load_and_validate,
    generate_comparison_charts,
)


AGENTS = ("agent0", "agent1", "agent2")
COMPONENTS = (
    "bev_build_ms",
    "model_inference_ms",
    "rule_maker_ms",
    "trajectory_optimizer_ms",
    "control_mapping_ms",
    "planning_tick_ms",
)


def _timing(scale: float) -> dict:
    values = {}
    for index, component in enumerate(COMPONENTS, start=1):
        p50 = scale * index
        values[component] = {
            "p50_ms": p50,
            "p95_ms": p50 * 1.4,
            "p99_ms": p50 * 1.6,
            "max_ms": p50 * 1.8,
        }
    values["planning_tick_ms"]["over_200_ms_fraction"] = 0.25 if scale > 20.0 else 0.0
    return values


def _comfort(value: float) -> dict:
    return {
        "longitudinal_acceleration_abs_mean_mps2": value * 0.7,
        "longitudinal_acceleration_abs_p95_mps2": value,
        "longitudinal_acceleration_abs_max_mps2": value * 1.2,
        "lateral_acceleration_abs_mean_mps2": value * 0.5,
        "lateral_acceleration_abs_p95_mps2": value * 0.8,
        "lateral_acceleration_abs_max_mps2": value,
        "longitudinal_jerk_abs_mean_mps3": value * 0.6,
        "longitudinal_jerk_abs_p95_mps3": value * 1.1,
        "longitudinal_jerk_abs_max_mps3": value * 1.4,
        "yaw_rate_abs_mean_rad_s": value * 0.02,
        "yaw_rate_abs_p95_rad_s": value * 0.04,
        "yaw_rate_abs_max_rad_s": value * 0.06,
        "yaw_acceleration_abs_mean_rad_s2": value * 0.03,
        "yaw_acceleration_abs_p95_rad_s2": value * 0.05,
        "yaw_acceleration_abs_max_rad_s2": value * 0.08,
        "steering_command_slew_abs_mean_per_s": value * 0.01,
        "steering_command_slew_abs_p95_per_s": value * 0.02,
        "steering_command_slew_abs_max_per_s": value * 0.03,
        "throttle_command_slew_abs_mean_per_s": value * 0.015,
        "throttle_command_slew_abs_p95_per_s": value * 0.025,
        "throttle_command_slew_abs_max_per_s": value * 0.035,
    }


def _episode(
    model: str,
    scenario: str,
    route: str,
    seed: int,
    model_bias: float,
    *,
    has_ttc: bool,
) -> dict:
    event = bool(seed == 47 and model == "stage1_a")
    success = not event
    return {
        "model": model,
        "scenario": scenario,
        "route": route,
        "seed": seed,
        "dt_s": 0.1,
        "steps": 800,
        "safety": {
            "collision": event,
            "out_of_road": False,
            "per_agent": {
                agent: {"collision": event and agent == "agent2", "out_of_road": False}
                for agent in AGENTS
            },
            "background_gap_violation": event,
            "background_gap_exposure_fraction": 0.08 if event else 0.0,
            "background_gap_deficit_integral_m_s": 0.5 if event else 0.0,
            "minimum_background_gap_m": 4.0 if event else 7.0 + model_bias,
            "platoon_gap_violation": event,
            "platoon_gap_exposure_fraction": 0.05 if event else 0.0,
            "platoon_gap_deficit_integral_m_s": 0.4 if event else 0.0,
            "minimum_platoon_gap_m": 6.0 if event else 8.0 + model_bias,
            "minimum_ttc_s": 1.2 + model_bias if has_ttc else None,
            "ttc_below_1_5_s_exposure_fraction": 0.03 if has_ttc else 0.0,
            "max_drac_mps2": 2.5 - model_bias if has_ttc else None,
            "execution_rejected": event,
            "execution_rejection_tick_fraction": 0.02 if event else 0.0,
        },
        "efficiency": {
            "scenario_realized": True,
            "functional_success_final": success,
            "functional_success_ever": success,
            "functional_success_stable": success,
            "first_stable_success_time_s": 55.0 - model_bias if success else None,
            "per_agent": {
                agent: {
                    "route_progress_m": 100.0 + model_bias - index,
                    "route_progress_fraction": 0.8 + 0.01 * model_bias - 0.01 * index,
                }
                for index, agent in enumerate(AGENTS)
            },
            "team_mean_route_progress_m": 99.0 + model_bias,
            "team_min_route_progress_m": 98.0 + model_bias,
            "team_mean_route_progress_fraction": 0.79 + 0.01 * model_bias,
            "team_min_route_progress_fraction": 0.78 + 0.01 * model_bias,
            "completion_time_s": 60.0 - model_bias if success else None,
            "episode_duration_s": 80.0 + model_bias,
        },
        "cooperation": {
            "locked_spacing_error_mean_m": 1.0 - 0.05 * model_bias,
            "locked_spacing_error_p95_m": 1.5 - 0.05 * model_bias,
            "locked_spacing_error_max_m": 2.0 - 0.05 * model_bias,
            "locked_speed_spread_mean_mps": 0.8 - 0.03 * model_bias,
            "unlock_count": 1,
            "unlocked_duration_s": 5.0 - 0.1 * model_bias,
            "relock_recovery_success": True,
            "recovery_censored": False,
            "relock_recovery_time_s": 4.0 - 0.1 * model_bias,
        },
        "comfort": {
            "per_agent": {
                agent: _comfort(1.0 + 0.1 * index - 0.02 * model_bias)
                for index, agent in enumerate(AGENTS)
            }
        },
        "timing": _timing(20.0 + model_bias),
    }


def _mean(rows: list[dict], *path: str) -> float | None:
    values = []
    for row in rows:
        value = row
        for key in path:
            value = value[key]
        if value is not None:
            values.append(float(value))
    return sum(values) / len(values) if values else None


def _aggregate(rows: list[dict]) -> dict:
    episodes = len(rows)
    safety = {}
    for stem in (
        "collision",
        "out_of_road",
        "background_gap_violation",
        "platoon_gap_violation",
        "execution_rejected",
    ):
        count = sum(bool(row["safety"][stem]) for row in rows)
        rate = count / episodes
        aggregate_stem = (
            "execution_rejection_episode" if stem == "execution_rejected" else stem
        )
        safety[f"{aggregate_stem}_count"] = count
        safety[f"{aggregate_stem}_sample_count"] = episodes
        safety[f"{aggregate_stem}_rate"] = rate
        safety[f"{aggregate_stem}_rate_ci95"] = [
            max(0.0, rate - 0.2),
            min(1.0, rate + 0.2),
        ]
    for stem in (
        "background_gap_exposure_fraction",
        "background_gap_deficit_integral_m_s",
        "minimum_background_gap_m",
        "platoon_gap_exposure_fraction",
        "platoon_gap_deficit_integral_m_s",
        "minimum_platoon_gap_m",
        "minimum_ttc_s",
        "ttc_below_1_5_s_exposure_fraction",
        "max_drac_mps2",
        "execution_rejection_tick_fraction",
    ):
        safety[f"{stem}_mean"] = _mean(rows, "safety", stem)

    efficiency = {}
    for stem in (
        "scenario_realized",
        "functional_success_final",
        "functional_success_ever",
        "functional_success_stable",
    ):
        count = sum(bool(row["efficiency"][stem]) for row in rows)
        rate = count / episodes
        efficiency[f"{stem}_count"] = count
        efficiency[f"{stem}_rate"] = rate
        efficiency[f"{stem}_rate_ci95"] = [max(0.0, rate - 0.2), min(1.0, rate + 0.2)]
    for stem in (
        "first_stable_success_time_s",
        "team_mean_route_progress_m",
        "team_min_route_progress_m",
        "team_mean_route_progress_fraction",
        "team_min_route_progress_fraction",
        "completion_time_s",
        "episode_duration_s",
    ):
        efficiency[f"{stem}_mean"] = _mean(rows, "efficiency", stem)

    cooperation = {
        "locked_spacing_error_mean_m": _mean(rows, "cooperation", "locked_spacing_error_mean_m"),
        "locked_spacing_error_p95_m_mean": _mean(rows, "cooperation", "locked_spacing_error_p95_m"),
        "locked_spacing_error_max_m_mean": _mean(rows, "cooperation", "locked_spacing_error_max_m"),
        "locked_speed_spread_mean_mps": _mean(rows, "cooperation", "locked_speed_spread_mean_mps"),
        "unlock_count_mean": _mean(rows, "cooperation", "unlock_count"),
        "unlocked_duration_s_mean": _mean(rows, "cooperation", "unlocked_duration_s"),
        "relock_recovery_applicable_count": episodes,
        "relock_recovery_success_count": episodes,
        "relock_recovery_success_rate": 1.0,
        "relock_recovery_success_rate_ci95": [0.8, 1.0],
        "recovery_censored_count": 0,
        "relock_recovery_time_s_mean": _mean(rows, "cooperation", "relock_recovery_time_s"),
    }
    comfort = {
        "per_agent": {
            agent: {
                key: sum(row["comfort"]["per_agent"][agent][key] for row in rows) / episodes
                for key in rows[0]["comfort"]["per_agent"][agent]
            }
            for agent in AGENTS
        }
    }
    timing = {
        f"{component}_{stat}_mean": _mean(rows, "timing", component, stat)
        for component in COMPONENTS
        for stat in ("p50_ms", "p95_ms", "p99_ms", "max_ms")
    }
    timing["planning_tick_ms_over_200_ms_fraction_mean"] = _mean(
        rows, "timing", "planning_tick_ms", "over_200_ms_fraction"
    )
    return {
        "episodes": episodes,
        "safety": safety,
        "efficiency": efficiency,
        "cooperation": cooperation,
        "comfort": comfort,
        "timing": timing,
    }


def _model(model_id: str, bias: float, *, ttc: bool) -> dict:
    rows = [
        _episode(model_id, scenario, route, seed, bias, has_ttc=ttc)
        for scenario, route in (("S5_hard_brake_lead", "R1"), ("S9_narrow_channel_negotiation", "R8"))
        for seed in (31, 47)
    ]
    by_scenario = {
        scenario: _aggregate([row for row in rows if row["scenario"] == scenario])
        for scenario in ("S5_hard_brake_lead", "S9_narrow_channel_negotiation")
    }
    timing = _timing(20.0 + bias)
    return {
        "episode_metrics": rows,
        "by_scenario": by_scenario,
        "overall": _aggregate(rows),
        "timing": timing,
        "gate_results": {
            "planning_tick_p95_ms": {
                "threshold_ms": 200.0,
                "observed_ms": timing["planning_tick_ms"]["p95_ms"],
                "passed": timing["planning_tick_ms"]["p95_ms"] <= 200.0,
            }
        },
    }


def _comparison(baseline: str, candidate: str, n_pairs: int, *, ttc: bool) -> dict:
    metrics = {
        "safety.collision_rate": (0.25, 0.0, "episode rate", "lower"),
        "efficiency.functional_success_stable_rate": (0.75, 1.0, "episode rate", "higher"),
        "efficiency.completion_time_s_mean": (60.0, 59.0, "s", "lower"),
        "efficiency.episode_duration_s_mean": (80.0, 81.0, "s", "descriptive"),
        "cooperation.locked_spacing_error_p95_m_mean": (1.5, 1.4, "m", "lower"),
        "comfort.per_agent.agent0.longitudinal_jerk_abs_p95_mps3": (1.1, 1.0, "m/s³", "lower"),
        "timing.planning_tick_ms_p95_ms_mean": (168.0, 176.0, "ms", "lower"),
    }
    output = {}
    for name, (baseline_value, candidate_value, unit, direction) in metrics.items():
        delta = candidate_value - baseline_value
        output[name] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "candidate_minus_baseline": delta,
            "unit": unit,
            "direction": direction,
            "n_pairs": n_pairs,
            "ci95": [delta - 0.05, delta + 0.05],
        }
    output["safety.minimum_ttc_s_mean"] = {
        "baseline": 1.2 if ttc else None,
        "candidate": 1.4 if ttc else None,
        "candidate_minus_baseline": 0.2 if ttc else None,
        "unit": "s",
        "direction": "higher",
        "n_pairs": n_pairs if ttc else 0,
        "ci95": [0.1, 0.3] if ttc else None,
    }
    return {"baseline": baseline, "candidate": candidate, "metrics": output}


def _report(model_ids: tuple[str, ...]) -> dict:
    models = {
        model_id: _model(model_id, float(index), ttc=model_id != "stage1_a")
        for index, model_id in enumerate(model_ids)
    }
    comparisons = {
        f"{candidate}_vs_{baseline}": _comparison(
            baseline, candidate, n_pairs=4, ttc=baseline != "stage1_a"
        )
        for baseline, candidate in zip(model_ids, model_ids[1:])
    }
    first_metrics = next(iter(comparisons.values()))["metrics"]
    metric_definitions = {
        name: {
            "unit": metric["unit"],
            "direction": metric["direction"],
            "statistical_unit": "episode",
        }
        for name, metric in first_metrics.items()
    }
    return {
        "format": "bev_model_evaluation_v3",
        "evaluation_status": "completed",
        "all_gates_passed": True,
        "evaluation_protocol": {
            "scenarios": [["S5_hard_brake_lead", "R1"], ["S9_narrow_channel_negotiation", "R8"]],
            "seeds": [31, 47],
            "max_steps": 800,
            "dt_s": 0.1,
        },
        "metric_definitions": metric_definitions,
        "model_order": list(model_ids),
        "models": models,
        "comparisons": comparisons,
    }


def _write_report(path: Path, report: dict) -> Path:
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def _set_overview_cooperation_na(report: dict) -> None:
    model = report["models"]["stage1_a"]
    for episode in model["episode_metrics"]:
        episode["cooperation"]["locked_spacing_error_p95_m"] = None
    model["overall"]["cooperation"]["locked_spacing_error_p95_m_mean"] = None
    for aggregate in model["by_scenario"].values():
        aggregate["cooperation"]["locked_spacing_error_p95_m_mean"] = None
    comparison = next(iter(report["comparisons"].values()))
    metric = comparison["metrics"][
        "cooperation.locked_spacing_error_p95_m_mean"
    ]
    metric.update(
        {
            "baseline": None,
            "candidate": None,
            "candidate_minus_baseline": None,
            "n_pairs": 0,
            "ci95": None,
        }
    )


def _set_null_and_zero_comfort(report: dict) -> None:
    null_field = "longitudinal_jerk_abs_p95_mps3"
    zero_field = "yaw_acceleration_abs_p95_rad_s2"
    for model_id, field, value in (
        ("stage1_a", null_field, None),
        ("grpo_open", zero_field, 0.0),
    ):
        model = report["models"][model_id]
        for episode in model["episode_metrics"]:
            for agent in AGENTS:
                episode["comfort"]["per_agent"][agent][field] = value
        for aggregate in (model["overall"], *model["by_scenario"].values()):
            for agent in AGENTS:
                aggregate["comfort"]["per_agent"][agent][field] = value
    comparison = next(iter(report["comparisons"].values()))
    metric = comparison["metrics"][
        "comfort.per_agent.agent0.longitudinal_jerk_abs_p95_mps3"
    ]
    metric.update(
        {
            "baseline": None,
            "candidate": None,
            "candidate_minus_baseline": None,
            "n_pairs": 0,
            "ci95": None,
        }
    )


def _repeat_report(*, later_gate_passed: bool, tolerance_gate_passed: bool) -> dict:
    first = _report(("stage1_a", "grpo_open"))
    second = copy.deepcopy(first)
    if not later_gate_passed:
        gate = second["models"]["grpo_open"]["gate_results"][
            "planning_tick_p95_ms"
        ]
        gate.update({"observed_ms": 210.0, "passed": False})
        second["evaluation_status"] = "completed_with_gate_failure"
        second["all_gates_passed"] = False
    combined_gate = later_gate_passed and tolerance_gate_passed
    return {
        "format": "bev_model_fixed_process_repeat_v3",
        "evaluation_status": (
            "completed" if combined_gate else "completed_with_gate_failure"
        ),
        "all_gates_passed": combined_gate,
        "repeat_count": 2,
        "model_order": copy.deepcopy(first["model_order"]),
        "evaluation_protocol": copy.deepcopy(first["evaluation_protocol"]),
        "metric_definitions": copy.deepcopy(first["metric_definitions"]),
        "models": copy.deepcopy(first["models"]),
        "comparisons": copy.deepcopy(first["comparisons"]),
        "fixed_process_reproducibility": {
            "tolerance_comparison": {
                "tolerance_gate_passed": tolerance_gate_passed,
            }
        },
        "repeat_reports": [first, second],
    }


@pytest.mark.parametrize(
    "model_ids",
    [
        ("stage1_a", "grpo_open"),
        ("stage1_a", "grpo_open", "grpo_exec"),
    ],
)
def test_generate_complete_chart_bundle_for_two_and_three_models(
    tmp_path: Path, model_ids: tuple[str, ...]
) -> None:
    report = _report(model_ids)
    if len(model_ids) == 2:
        _set_overview_cooperation_na(report)
        _set_null_and_zero_comfort(report)
    report_path = _write_report(tmp_path / "report.json", report)
    output_root = tmp_path / "output"

    index = generate_comparison_charts(report_path, output_root)

    assert [record["id"] for record in index["charts"]] == list(CHART_NAMES)
    for record in index["charts"]:
        for kind in ("png", "pdf"):
            path = Path(record[kind])
            assert path.is_file()
            assert path.stat().st_size > 100
    for record in index["tables"]:
        path = Path(record["path"])
        assert path.is_file()
        assert path.stat().st_size > 20
    stored_index = json.loads((output_root / "chart_index.json").read_text(encoding="utf-8"))
    assert stored_index["model_order"] == list(model_ids)
    assert stored_index["evaluation_status"] == "completed"
    assert stored_index["all_gates_passed"] is True
    assert stored_index["repeat_tolerance_gate_passed"] is None
    assert "minimum_ttc_s:stage1_a:N/A" in stored_index["na_observations"]
    if len(model_ids) == 2:
        assert (
            "overview:locked_spacing_error_p95_m:stage1_a:N/A"
            in stored_index["na_observations"]
        )
        assert (
            "overview:longitudinal_jerk_p95:stage1_a:N/A"
            in stored_index["na_observations"]
        )

    with (output_root / "tables" / "episode_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        episode_rows = list(csv.DictReader(stream))
    assert [row["model"] for row in episode_rows] == [
        model_id for model_id in model_ids for _ in range(4)
    ]
    assert episode_rows[0]["scenario"] == "S5_hard_brake_lead"
    assert episode_rows[0]["safety.minimum_ttc_s"] == "N/A"
    if len(model_ids) == 2:
        assert (
            episode_rows[0][
                "comfort.per_agent.agent0.longitudinal_jerk_abs_p95_mps3"
            ]
            == "N/A"
        )
        first_grpo = next(row for row in episode_rows if row["model"] == "grpo_open")
        assert float(
            first_grpo[
                "comfort.per_agent.agent0.yaw_acceleration_abs_p95_rad_s2"
            ]
        ) == pytest.approx(0.0)
    assert float(episode_rows[-1]["efficiency.team_mean_route_progress_m"]) == pytest.approx(
        99.0 + len(model_ids) - 1
    )

    with (output_root / "tables" / "scenario_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        scenario_rows = list(csv.DictReader(stream))
    assert [(row["model"], row["scenario"]) for row in scenario_rows[:2]] == [
        (model_ids[0], "S5_hard_brake_lead"),
        (model_ids[0], "S9_narrow_channel_negotiation"),
    ]
    if len(model_ids) == 2:
        assert (
            scenario_rows[0][
                "comfort.per_agent.agent0.longitudinal_jerk_abs_p95_mps3"
            ]
            == "N/A"
        )

    with (output_root / "tables" / "comparison_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        comparison_rows = list(csv.DictReader(stream))
    collision = next(
        row for row in comparison_rows if row["metric"] == "safety.collision_rate"
    )
    assert float(collision["baseline"]) == pytest.approx(0.25)
    assert float(collision["candidate_minus_baseline"]) == pytest.approx(-0.25)
    assert collision["n_pairs"] == "4"
    duration = next(
        row
        for row in comparison_rows
        if row["metric"] == "efficiency.episode_duration_s_mean"
    )
    completion = next(
        row
        for row in comparison_rows
        if row["metric"] == "efficiency.completion_time_s_mean"
    )
    assert duration["direction"] == "descriptive"
    assert float(duration["candidate_minus_baseline"]) == pytest.approx(1.0)
    assert completion["direction"] == "lower"
    if len(model_ids) == 2:
        comfort = next(
            row
            for row in comparison_rows
            if row["metric"]
            == "comfort.per_agent.agent0.longitudinal_jerk_abs_p95_mps3"
        )
        assert comfort["baseline"] == "N/A"
        assert comfort["candidate_minus_baseline"] == "N/A"


def test_repeat_preserves_combined_gate_when_a_later_repeat_fails(
    tmp_path: Path,
) -> None:
    repeat = _repeat_report(later_gate_passed=False, tolerance_gate_passed=True)
    path = _write_report(tmp_path / "repeat.json", repeat)

    normalized = _load_and_validate(path)
    index = generate_comparison_charts(path, tmp_path / "output")

    assert normalized["format"] == "bev_model_fixed_process_repeat_v3"
    assert normalized["repeat_count"] == 2
    assert len(normalized["models"]["stage1_a"]["episode_metrics"]) == 4
    assert normalized["evaluation_status"] == "completed_with_gate_failure"
    assert normalized["all_gates_passed"] is False
    assert index["evaluation_status"] == "completed_with_gate_failure"
    assert index["all_gates_passed"] is False
    assert index["repeat_tolerance_gate_passed"] is True


def test_repeat_combined_gate_includes_tolerance_gate(tmp_path: Path) -> None:
    repeat = _repeat_report(later_gate_passed=True, tolerance_gate_passed=False)
    path = _write_report(tmp_path / "tolerance_failure.json", repeat)

    normalized = _load_and_validate(path)

    assert all(item["all_gates_passed"] for item in normalized["repeat_reports"])
    assert normalized["all_gates_passed"] is False
    assert normalized["fixed_process_reproducibility"]["tolerance_comparison"][
        "tolerance_gate_passed"
    ] is False

    invalid = copy.deepcopy(repeat)
    invalid.update({"evaluation_status": "completed", "all_gates_passed": True})
    invalid_path = _write_report(tmp_path / "invalid_combined_gate.json", invalid)
    with pytest.raises(ComparisonChartError, match="all repeat gates AND"):
        _load_and_validate(invalid_path)


def test_steering_command_slew_uses_per_second_unit() -> None:
    steering = next(spec for spec in COMFORT_SPECS if spec.key == "steering_slew_mean")
    assert steering.unit == "command/s"


def test_descriptive_direction_has_neutral_heatmap_color() -> None:
    assert _directional_color_delta(2.0, "descriptive") == 0.0
    assert _directional_color_delta(2.0, "higher") == 2.0
    assert _directional_color_delta(2.0, "lower") == -2.0


def test_null_comfort_is_na_while_measured_zero_remains_numeric(
    tmp_path: Path,
) -> None:
    report = _report(("stage1_a", "grpo_open"))
    _set_null_and_zero_comfort(report)
    path = _write_report(tmp_path / "null_comfort.json", report)

    validated = _load_and_validate(path)

    stage1_comfort = validated["models"]["stage1_a"]["episode_metrics"][0][
        "comfort"
    ]["per_agent"]["agent0"]
    grpo_comfort = validated["models"]["grpo_open"]["episode_metrics"][0][
        "comfort"
    ]["per_agent"]["agent0"]
    assert stage1_comfort["longitudinal_jerk_abs_p95_mps3"] is None
    assert grpo_comfort["yaw_acceleration_abs_p95_rad_s2"] == 0.0


def test_validator_accepts_authoritative_evaluator_aggregates(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from evaluation.bev_four_model_evaluator import _build_model_aggregates

    report = _report(("stage1_a", "grpo_open"))
    for model in report["models"].values():
        by_scenario, overall = _build_model_aggregates(model["episode_metrics"])
        model["by_scenario"] = by_scenario
        model["overall"] = overall
    path = _write_report(tmp_path / "authoritative.json", report)

    validated = _load_and_validate(path)

    assert validated["models"]["stage1_a"]["overall"]["aggregation"] == (
        "equal_weight_scenario_macro"
    )


def test_validator_accepts_not_applicable_formation_recovery(tmp_path: Path) -> None:
    report = _report(("stage1_a", "grpo_open"))
    episode = report["models"]["stage1_a"]["episode_metrics"][0]
    episode["cooperation"].update(
        {
            "unlock_count": 0,
            "relock_recovery_success": None,
            "recovery_censored": None,
            "relock_recovery_time_s": None,
        }
    )
    path = _write_report(tmp_path / "no_unlock.json", report)

    validated = _load_and_validate(path)

    cooperation = validated["models"]["stage1_a"]["episode_metrics"][0][
        "cooperation"
    ]
    assert cooperation["relock_recovery_success"] is None
    assert cooperation["recovery_censored"] is None


def test_missing_required_episode_metric_fails_before_writing(tmp_path: Path) -> None:
    report = _report(("stage1_a", "grpo_open"))
    del report["models"]["stage1_a"]["episode_metrics"][0]["safety"]["minimum_ttc_s"]
    path = _write_report(tmp_path / "invalid.json", report)

    with pytest.raises(ComparisonChartError, match="minimum_ttc_s"):
        generate_comparison_charts(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_non_v3_report_is_rejected(tmp_path: Path) -> None:
    report = _report(("stage1_a", "grpo_open"))
    report["format"] = "bev_model_evaluation_v2"
    path = _write_report(tmp_path / "v2.json", report)

    with pytest.raises(ComparisonChartError, match="report.format"):
        _load_and_validate(path)
