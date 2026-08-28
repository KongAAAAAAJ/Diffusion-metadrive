"""Generate publication-ready Stage1/GRPO comparison charts from a v3 report."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from evaluation.evaluation_helper import (
    _PUBLICATION_AGENT_COLORS,
    _PUBLICATION_RC,
    _style_publication_axis,
)


SINGLE_REPORT_FORMAT = "bev_model_evaluation_v3"
REPEAT_REPORT_FORMAT = "bev_model_fixed_process_repeat_v3"
CHART_INDEX_FORMAT = "bev_comparison_chart_index_v1"
METRIC_DIRECTIONS = ("higher", "lower", "descriptive")
AGENT_IDS = ("agent0", "agent1", "agent2")
TIMING_COMPONENTS = (
    "bev_build_ms",
    "model_inference_ms",
    "rule_maker_ms",
    "trajectory_optimizer_ms",
    "control_mapping_ms",
    "planning_tick_ms",
)
TIMING_LABELS = {
    "bev_build_ms": "BEV build",
    "model_inference_ms": "Model",
    "rule_maker_ms": "RuleMaker",
    "trajectory_optimizer_ms": "Optimizer",
    "control_mapping_ms": "Control",
    "planning_tick_ms": "Full tick",
}
COMFORT_REQUIRED_FIELDS = (
    "longitudinal_acceleration_abs_mean_mps2",
    "longitudinal_acceleration_abs_p95_mps2",
    "longitudinal_acceleration_abs_max_mps2",
    "lateral_acceleration_abs_mean_mps2",
    "lateral_acceleration_abs_p95_mps2",
    "lateral_acceleration_abs_max_mps2",
    "longitudinal_jerk_abs_mean_mps3",
    "longitudinal_jerk_abs_p95_mps3",
    "longitudinal_jerk_abs_max_mps3",
    "yaw_rate_abs_mean_rad_s",
    "yaw_rate_abs_p95_rad_s",
    "yaw_rate_abs_max_rad_s",
    "yaw_acceleration_abs_mean_rad_s2",
    "yaw_acceleration_abs_p95_rad_s2",
    "yaw_acceleration_abs_max_rad_s2",
    "steering_command_slew_abs_mean_per_s",
    "steering_command_slew_abs_p95_per_s",
    "steering_command_slew_abs_max_per_s",
    "throttle_command_slew_abs_mean_per_s",
    "throttle_command_slew_abs_p95_per_s",
    "throttle_command_slew_abs_max_per_s",
)
CHART_NAMES = (
    "comparison_overview",
    "safety_rates",
    "risk_ecdf",
    "efficiency_paired",
    "cooperation_paired",
    "comfort_paired",
    "latency_breakdown",
    "scenario_delta_heatmap",
    "paired_delta_forest",
)
CORE_FOREST_METRICS = (
    "safety.collision_rate",
    "safety.out_of_road_rate",
    "safety.background_gap_violation_rate",
    "safety.platoon_gap_violation_rate",
    "safety.minimum_ttc_s_mean",
    "safety.ttc_below_1_5_s_exposure_fraction_mean",
    "safety.max_drac_mps2_mean",
    "safety.execution_rejection_episode_rate",
    "safety.execution_rejection_tick_fraction_mean",
    "efficiency.scenario_realized_rate",
    "efficiency.functional_success_stable_rate",
    "efficiency.team_mean_route_progress_fraction_mean",
    "efficiency.completion_time_s_mean",
    "efficiency.episode_duration_s_mean",
    "cooperation.locked_spacing_error_p95_m_mean",
    "cooperation.locked_speed_spread_mean_mps",
    "cooperation.unlocked_duration_s_mean",
    "cooperation.relock_recovery_success_rate",
    *(
        f"comfort.per_agent.{agent_id}.{field}"
        for agent_id in AGENT_IDS
        for field in (
            "longitudinal_acceleration_abs_p95_mps2",
            "lateral_acceleration_abs_p95_mps2",
            "longitudinal_jerk_abs_p95_mps3",
            "yaw_rate_abs_p95_rad_s",
            "yaw_acceleration_abs_p95_rad_s2",
        )
    ),
    "timing.planning_tick_ms_p50_ms_mean",
    "timing.planning_tick_ms_p95_ms_mean",
    "timing.planning_tick_ms_p99_ms_mean",
    "timing.planning_tick_ms_max_ms_mean",
    "timing.planning_tick_ms_over_200_ms_fraction_mean",
)


class ComparisonChartError(ValueError):
    """Raised when a persisted evaluation report violates the v3 contract."""


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    unit: str
    direction: str
    episode_path: tuple[str, ...]
    aggregate_path: tuple[str, ...]
    nullable: bool = False


# All report-specific paths live here. Plot code never embeds schema paths.
EPISODE_FIELD_PATHS: dict[str, tuple[str, ...]] = {
    "collision": ("safety", "collision"),
    "out_of_road": ("safety", "out_of_road"),
    "background_gap_violation": ("safety", "background_gap_violation"),
    "background_gap_exposure_fraction": (
        "safety",
        "background_gap_exposure_fraction",
    ),
    "background_gap_deficit_integral_m_s": (
        "safety",
        "background_gap_deficit_integral_m_s",
    ),
    "minimum_background_gap_m": ("safety", "minimum_background_gap_m"),
    "platoon_gap_violation": ("safety", "platoon_gap_violation"),
    "platoon_gap_exposure_fraction": (
        "safety",
        "platoon_gap_exposure_fraction",
    ),
    "platoon_gap_deficit_integral_m_s": (
        "safety",
        "platoon_gap_deficit_integral_m_s",
    ),
    "minimum_platoon_gap_m": ("safety", "minimum_platoon_gap_m"),
    "minimum_ttc_s": ("safety", "minimum_ttc_s"),
    "ttc_exposure_fraction": (
        "safety",
        "ttc_below_1_5_s_exposure_fraction",
    ),
    "max_drac_mps2": ("safety", "max_drac_mps2"),
    "execution_rejected": ("safety", "execution_rejected"),
    "execution_rejection_tick_fraction": (
        "safety",
        "execution_rejection_tick_fraction",
    ),
    "scenario_realized": ("efficiency", "scenario_realized"),
    "functional_success_final": ("efficiency", "functional_success_final"),
    "functional_success_ever": ("efficiency", "functional_success_ever"),
    "functional_success_stable": ("efficiency", "functional_success_stable"),
    "first_stable_success_time_s": (
        "efficiency",
        "first_stable_success_time_s",
    ),
    "team_mean_route_progress_m": (
        "efficiency",
        "team_mean_route_progress_m",
    ),
    "team_min_route_progress_m": ("efficiency", "team_min_route_progress_m"),
    "team_mean_route_progress_fraction": (
        "efficiency",
        "team_mean_route_progress_fraction",
    ),
    "team_min_route_progress_fraction": (
        "efficiency",
        "team_min_route_progress_fraction",
    ),
    "completion_time_s": ("efficiency", "completion_time_s"),
    "episode_duration_s": ("efficiency", "episode_duration_s"),
    "locked_spacing_error_mean_m": (
        "cooperation",
        "locked_spacing_error_mean_m",
    ),
    "locked_spacing_error_p95_m": (
        "cooperation",
        "locked_spacing_error_p95_m",
    ),
    "locked_spacing_error_max_m": (
        "cooperation",
        "locked_spacing_error_max_m",
    ),
    "locked_speed_spread_mean_mps": (
        "cooperation",
        "locked_speed_spread_mean_mps",
    ),
    "unlock_count": ("cooperation", "unlock_count"),
    "unlocked_duration_s": ("cooperation", "unlocked_duration_s"),
    "relock_recovery_success": ("cooperation", "relock_recovery_success"),
    "recovery_censored": ("cooperation", "recovery_censored"),
    "relock_recovery_time_s": ("cooperation", "relock_recovery_time_s"),
    "longitudinal_acceleration_p95": (
        "comfort",
        "per_agent",
        "*",
        "longitudinal_acceleration_abs_p95_mps2",
    ),
    "lateral_acceleration_p95": (
        "comfort",
        "per_agent",
        "*",
        "lateral_acceleration_abs_p95_mps2",
    ),
    "longitudinal_jerk_p95": (
        "comfort",
        "per_agent",
        "*",
        "longitudinal_jerk_abs_p95_mps3",
    ),
    "yaw_rate_p95": (
        "comfort",
        "per_agent",
        "*",
        "yaw_rate_abs_p95_rad_s",
    ),
    "yaw_acceleration_p95": (
        "comfort",
        "per_agent",
        "*",
        "yaw_acceleration_abs_p95_rad_s2",
    ),
    "steering_slew_mean": (
        "comfort",
        "per_agent",
        "*",
        "steering_command_slew_abs_mean_per_s",
    ),
}

AGGREGATE_FIELD_PATHS: dict[str, tuple[str, ...]] = {
    "collision_rate": ("safety", "collision_rate"),
    "out_of_road_rate": ("safety", "out_of_road_rate"),
    "background_gap_violation_rate": (
        "safety",
        "background_gap_violation_rate",
    ),
    "minimum_background_gap_m": (
        "safety",
        "minimum_background_gap_m_mean",
    ),
    "platoon_gap_violation_rate": ("safety", "platoon_gap_violation_rate"),
    "minimum_platoon_gap_m": ("safety", "minimum_platoon_gap_m_mean"),
    "minimum_ttc_s": ("safety", "minimum_ttc_s_mean"),
    "ttc_exposure_fraction": (
        "safety",
        "ttc_below_1_5_s_exposure_fraction_mean",
    ),
    "max_drac_mps2": ("safety", "max_drac_mps2_mean"),
    "execution_rejected_rate": ("safety", "execution_rejection_episode_rate"),
    "functional_success_stable_rate": (
        "efficiency",
        "functional_success_stable_rate",
    ),
    "team_mean_route_progress_m": (
        "efficiency",
        "team_mean_route_progress_m_mean",
    ),
    "team_min_route_progress_fraction": (
        "efficiency",
        "team_min_route_progress_fraction_mean",
    ),
    "completion_time_s": ("efficiency", "completion_time_s_mean"),
    "episode_duration_s": ("efficiency", "episode_duration_s_mean"),
    "locked_spacing_error_p95_m": (
        "cooperation",
        "locked_spacing_error_p95_m_mean",
    ),
    "locked_speed_spread_mean_mps": (
        "cooperation",
        "locked_speed_spread_mean_mps",
    ),
    "unlocked_duration_s": ("cooperation", "unlocked_duration_s_mean"),
    "relock_recovery_time_s": (
        "cooperation",
        "relock_recovery_time_s_mean",
    ),
    "longitudinal_jerk_p95": (
        "comfort",
        "per_agent",
        "*",
        "longitudinal_jerk_abs_p95_mps3",
    ),
}


SAFETY_SPECS = (
    MetricSpec(
        "collision_rate",
        "Collision",
        "episode rate",
        "lower",
        EPISODE_FIELD_PATHS["collision"],
        AGGREGATE_FIELD_PATHS["collision_rate"],
    ),
    MetricSpec(
        "out_of_road_rate",
        "Out of road",
        "episode rate",
        "lower",
        EPISODE_FIELD_PATHS["out_of_road"],
        AGGREGATE_FIELD_PATHS["out_of_road_rate"],
    ),
    MetricSpec(
        "background_gap_violation_rate",
        "Background gap < 5 m",
        "episode rate",
        "lower",
        EPISODE_FIELD_PATHS["background_gap_violation"],
        AGGREGATE_FIELD_PATHS["background_gap_violation_rate"],
    ),
    MetricSpec(
        "platoon_gap_violation_rate",
        "Platoon gap < 7 m",
        "episode rate",
        "lower",
        EPISODE_FIELD_PATHS["platoon_gap_violation"],
        AGGREGATE_FIELD_PATHS["platoon_gap_violation_rate"],
    ),
    MetricSpec(
        "execution_rejected_rate",
        "Execution rejected",
        "episode rate",
        "lower",
        EPISODE_FIELD_PATHS["execution_rejected"],
        AGGREGATE_FIELD_PATHS["execution_rejected_rate"],
    ),
)

RISK_SPECS = (
    MetricSpec(
        "minimum_background_gap_m",
        "Minimum background gap",
        "m",
        "higher",
        EPISODE_FIELD_PATHS["minimum_background_gap_m"],
        AGGREGATE_FIELD_PATHS["minimum_background_gap_m"],
        nullable=True,
    ),
    MetricSpec(
        "minimum_platoon_gap_m",
        "Minimum platoon gap",
        "m",
        "higher",
        EPISODE_FIELD_PATHS["minimum_platoon_gap_m"],
        AGGREGATE_FIELD_PATHS["minimum_platoon_gap_m"],
        nullable=True,
    ),
    MetricSpec(
        "minimum_ttc_s",
        "Minimum closing TTC",
        "s",
        "higher",
        EPISODE_FIELD_PATHS["minimum_ttc_s"],
        AGGREGATE_FIELD_PATHS["minimum_ttc_s"],
        nullable=True,
    ),
    MetricSpec(
        "max_drac_mps2",
        "Maximum DRAC",
        "m/s²",
        "lower",
        EPISODE_FIELD_PATHS["max_drac_mps2"],
        AGGREGATE_FIELD_PATHS["max_drac_mps2"],
        nullable=True,
    ),
)

EFFICIENCY_SPECS = (
    MetricSpec(
        "functional_success_stable",
        "Stable functional success",
        "0/1",
        "higher",
        EPISODE_FIELD_PATHS["functional_success_stable"],
        AGGREGATE_FIELD_PATHS["functional_success_stable_rate"],
    ),
    MetricSpec(
        "team_mean_route_progress_m",
        "Team mean route progress",
        "m",
        "higher",
        EPISODE_FIELD_PATHS["team_mean_route_progress_m"],
        AGGREGATE_FIELD_PATHS["team_mean_route_progress_m"],
    ),
    MetricSpec(
        "team_min_route_progress_fraction",
        "Team minimum route progress",
        "fraction",
        "higher",
        EPISODE_FIELD_PATHS["team_min_route_progress_fraction"],
        AGGREGATE_FIELD_PATHS["team_min_route_progress_fraction"],
    ),
    MetricSpec(
        "completion_time_s",
        "Completion time (successful episodes)",
        "s",
        "lower",
        EPISODE_FIELD_PATHS["completion_time_s"],
        AGGREGATE_FIELD_PATHS["completion_time_s"],
        nullable=True,
    ),
)

COOPERATION_SPECS = (
    MetricSpec(
        "locked_spacing_error_p95_m",
        "Locked spacing error P95",
        "m",
        "lower",
        EPISODE_FIELD_PATHS["locked_spacing_error_p95_m"],
        AGGREGATE_FIELD_PATHS["locked_spacing_error_p95_m"],
        nullable=True,
    ),
    MetricSpec(
        "locked_speed_spread_mean_mps",
        "Locked speed spread",
        "m/s",
        "lower",
        EPISODE_FIELD_PATHS["locked_speed_spread_mean_mps"],
        AGGREGATE_FIELD_PATHS["locked_speed_spread_mean_mps"],
        nullable=True,
    ),
    MetricSpec(
        "unlocked_duration_s",
        "Unlocked duration",
        "s",
        "lower",
        EPISODE_FIELD_PATHS["unlocked_duration_s"],
        AGGREGATE_FIELD_PATHS["unlocked_duration_s"],
    ),
    MetricSpec(
        "relock_recovery_time_s",
        "Relock recovery time",
        "s",
        "lower",
        EPISODE_FIELD_PATHS["relock_recovery_time_s"],
        AGGREGATE_FIELD_PATHS["relock_recovery_time_s"],
        nullable=True,
    ),
)

COMFORT_SPECS = (
    MetricSpec(
        "longitudinal_acceleration_p95",
        "Longitudinal acceleration P95",
        "m/s²",
        "lower",
        EPISODE_FIELD_PATHS["longitudinal_acceleration_p95"],
        ("comfort", "per_agent", "*", "longitudinal_acceleration_abs_p95_mps2"),
        nullable=True,
    ),
    MetricSpec(
        "lateral_acceleration_p95",
        "Lateral acceleration P95",
        "m/s²",
        "lower",
        EPISODE_FIELD_PATHS["lateral_acceleration_p95"],
        ("comfort", "per_agent", "*", "lateral_acceleration_abs_p95_mps2"),
        nullable=True,
    ),
    MetricSpec(
        "longitudinal_jerk_p95",
        "Longitudinal jerk P95",
        "m/s³",
        "lower",
        EPISODE_FIELD_PATHS["longitudinal_jerk_p95"],
        AGGREGATE_FIELD_PATHS["longitudinal_jerk_p95"],
        nullable=True,
    ),
    MetricSpec(
        "yaw_rate_p95",
        "Yaw rate P95",
        "rad/s",
        "lower",
        EPISODE_FIELD_PATHS["yaw_rate_p95"],
        ("comfort", "per_agent", "*", "yaw_rate_abs_p95_rad_s"),
        nullable=True,
    ),
    MetricSpec(
        "yaw_acceleration_p95",
        "Yaw acceleration P95",
        "rad/s²",
        "lower",
        EPISODE_FIELD_PATHS["yaw_acceleration_p95"],
        ("comfort", "per_agent", "*", "yaw_acceleration_abs_p95_rad_s2"),
        nullable=True,
    ),
    MetricSpec(
        "steering_slew_mean",
        "Steering command slew",
        "command/s",
        "lower",
        EPISODE_FIELD_PATHS["steering_slew_mean"],
        ("comfort", "per_agent", "*", "steering_command_slew_abs_mean_per_s"),
        nullable=True,
    ),
)

TIMING_SPEC = MetricSpec(
    "planning_tick_p95_ms",
    "Planning tick P95",
    "ms",
    "lower",
    ("timing", "planning_tick_ms", "p95_ms"),
    ("timing", "planning_tick_ms_p95_ms_mean"),
)


def _mapping(value: object, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComparisonChartError(f"{context} must be an object")
    return value


def _finite_number(value: object, context: str, *, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparisonChartError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ComparisonChartError(f"{context} must be a finite number")
    return result


def _path_value(
    root: Mapping[str, Any],
    path: Sequence[str],
    context: str,
    *,
    nullable: bool,
) -> float | None:
    current: object = root
    for index, key in enumerate(path):
        if key == "*":
            values = _mapping(current, f"{context}.{'.'.join(path[:index])}")
            if set(values) != set(AGENT_IDS):
                raise ComparisonChartError(
                    f"{context}.{'.'.join(path[:index])} must contain {AGENT_IDS}"
                )
            resolved = [
                _path_value(
                    _mapping(values[agent], f"{context}.{agent}"),
                    path[index + 1 :],
                    f"{context}.{agent}",
                    nullable=True,
                )
                for agent in AGENT_IDS
            ]
            numeric = [value for value in resolved if value is not None]
            if not numeric:
                if nullable:
                    return None
                raise ComparisonChartError(
                    f"{context}.{'.'.join(path)} has no numeric agent values"
                )
            return float(np.mean(numeric))
        mapping = _mapping(current, f"{context}.{'.'.join(path[:index])}")
        if key not in mapping:
            raise ComparisonChartError(
                f"{context} is missing required field {'.'.join(path)}"
            )
        current = mapping[key]
    if isinstance(current, bool):
        return float(current)
    return _finite_number(current, f"{context}.{'.'.join(path)}", nullable=nullable)


def _require_bool(
    root: Mapping[str, Any], path: Sequence[str], context: str, *, nullable: bool = False
) -> None:
    current: object = root
    for index, key in enumerate(path):
        mapping = _mapping(current, f"{context}.{'.'.join(path[:index])}")
        if key not in mapping:
            raise ComparisonChartError(
                f"{context} is missing required field {'.'.join(path)}"
            )
        current = mapping[key]
    if current is None and nullable:
        return
    if not isinstance(current, bool):
        raise ComparisonChartError(f"{context}.{'.'.join(path)} must be boolean")


def _validate_per_agent(row: Mapping[str, Any], category: str, context: str) -> None:
    category_value = _mapping(row.get(category), f"{context}.{category}")
    agents = _mapping(category_value.get("per_agent"), f"{context}.{category}.per_agent")
    if set(agents) != set(AGENT_IDS):
        raise ComparisonChartError(
            f"{context}.{category}.per_agent must contain {AGENT_IDS}"
        )
    for agent_id in AGENT_IDS:
        _mapping(agents[agent_id], f"{context}.{category}.per_agent.{agent_id}")


def _validate_episode(row: object, model_id: str, index: int) -> tuple[str, str, int]:
    context = f"models.{model_id}.episode_metrics[{index}]"
    value = _mapping(row, context)
    if value.get("model") != model_id:
        raise ComparisonChartError(f"{context}.model must equal {model_id}")
    scenario = value.get("scenario")
    route = value.get("route")
    seed = value.get("seed")
    if not isinstance(scenario, str) or not scenario:
        raise ComparisonChartError(f"{context}.scenario must be a non-empty string")
    if not isinstance(route, str) or not route:
        raise ComparisonChartError(f"{context}.route must be a non-empty string")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ComparisonChartError(f"{context}.seed must be an integer")
    dt_s = _finite_number(value.get("dt_s"), f"{context}.dt_s")
    if dt_s is None or dt_s <= 0.0:
        raise ComparisonChartError(f"{context}.dt_s must be positive")
    steps = value.get("steps")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ComparisonChartError(f"{context}.steps must be a positive integer")

    for category in ("safety", "efficiency", "cooperation", "comfort", "timing"):
        _mapping(value.get(category), f"{context}.{category}")
    for category in ("safety", "efficiency", "comfort"):
        _validate_per_agent(value, category, context)

    for name in (
        "collision",
        "out_of_road",
        "background_gap_violation",
        "platoon_gap_violation",
        "execution_rejected",
        "scenario_realized",
        "functional_success_final",
        "functional_success_ever",
        "functional_success_stable",
    ):
        _require_bool(value, EPISODE_FIELD_PATHS[name], context)
    for name in ("relock_recovery_success", "recovery_censored"):
        _require_bool(value, EPISODE_FIELD_PATHS[name], context, nullable=True)

    nullable_fields = {
        "minimum_background_gap_m",
        "minimum_platoon_gap_m",
        "minimum_ttc_s",
        "max_drac_mps2",
        "first_stable_success_time_s",
        "completion_time_s",
        "locked_spacing_error_mean_m",
        "locked_spacing_error_p95_m",
        "locked_spacing_error_max_m",
        "locked_speed_spread_mean_mps",
        "relock_recovery_time_s",
    }
    scalar_fields = (
        "background_gap_exposure_fraction",
        "background_gap_deficit_integral_m_s",
        "minimum_background_gap_m",
        "platoon_gap_exposure_fraction",
        "platoon_gap_deficit_integral_m_s",
        "minimum_platoon_gap_m",
        "minimum_ttc_s",
        "ttc_exposure_fraction",
        "max_drac_mps2",
        "execution_rejection_tick_fraction",
        "first_stable_success_time_s",
        "team_mean_route_progress_m",
        "team_min_route_progress_m",
        "team_mean_route_progress_fraction",
        "team_min_route_progress_fraction",
        "completion_time_s",
        "episode_duration_s",
        "locked_spacing_error_mean_m",
        "locked_spacing_error_p95_m",
        "locked_spacing_error_max_m",
        "locked_speed_spread_mean_mps",
        "unlock_count",
        "unlocked_duration_s",
        "relock_recovery_time_s",
    )
    for name in scalar_fields:
        _path_value(
            value,
            EPISODE_FIELD_PATHS[name],
            context,
            nullable=name in nullable_fields,
        )
    for spec in COMFORT_SPECS:
        _path_value(value, spec.episode_path, context, nullable=True)

    safety_agents = _mapping(
        _mapping(value["safety"], f"{context}.safety")["per_agent"],
        f"{context}.safety.per_agent",
    )
    efficiency_agents = _mapping(
        _mapping(value["efficiency"], f"{context}.efficiency")["per_agent"],
        f"{context}.efficiency.per_agent",
    )
    comfort_agents = _mapping(
        _mapping(value["comfort"], f"{context}.comfort")["per_agent"],
        f"{context}.comfort.per_agent",
    )
    for agent_id in AGENT_IDS:
        for key in ("collision", "out_of_road"):
            _require_bool(
                _mapping(safety_agents[agent_id], f"{context}.safety.{agent_id}"),
                (key,),
                f"{context}.safety.per_agent.{agent_id}",
            )
        for key in ("route_progress_m", "route_progress_fraction"):
            _path_value(
                _mapping(
                    efficiency_agents[agent_id],
                    f"{context}.efficiency.per_agent.{agent_id}",
                ),
                (key,),
                f"{context}.efficiency.per_agent.{agent_id}",
                nullable=False,
            )
        for key in COMFORT_REQUIRED_FIELDS:
            _path_value(
                _mapping(
                    comfort_agents[agent_id],
                    f"{context}.comfort.per_agent.{agent_id}",
                ),
                (key,),
                f"{context}.comfort.per_agent.{agent_id}",
                nullable=True,
            )

    timing = _mapping(value["timing"], f"{context}.timing")
    for component in TIMING_COMPONENTS:
        summary = _mapping(timing.get(component), f"{context}.timing.{component}")
        for stat in ("p50_ms", "p95_ms", "p99_ms", "max_ms"):
            _finite_number(summary.get(stat), f"{context}.timing.{component}.{stat}")
        if component == "planning_tick_ms":
            fraction = _finite_number(
                summary.get("over_200_ms_fraction"),
                f"{context}.timing.{component}.over_200_ms_fraction",
            )
            if fraction is None or not 0.0 <= fraction <= 1.0:
                raise ComparisonChartError(
                    f"{context}.timing.{component}.over_200_ms_fraction must be in [0,1]"
                )
    return scenario, route, int(seed)


def _validate_aggregate(value: object, context: str) -> None:
    aggregate = _mapping(value, context)
    episodes = aggregate.get("episodes")
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ComparisonChartError(f"{context}.episodes must be a positive integer")
    for category in ("safety", "efficiency", "cooperation", "comfort", "timing"):
        _mapping(aggregate.get(category), f"{context}.{category}")
    for spec in (
        *SAFETY_SPECS,
        *RISK_SPECS,
        *EFFICIENCY_SPECS,
        *COOPERATION_SPECS,
        *COMFORT_SPECS,
        TIMING_SPEC,
    ):
        _path_value(aggregate, spec.aggregate_path, context, nullable=spec.nullable)
    for spec in SAFETY_SPECS:
        safety = _mapping(aggregate["safety"], f"{context}.safety")
        stem = spec.aggregate_path[-1].removesuffix("_rate")
        count = safety.get(f"{stem}_count")
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= episodes:
            raise ComparisonChartError(f"{context}.safety.{stem}_count is invalid")
        ci = safety.get(f"{stem}_rate_ci95")
        if not isinstance(ci, list) or len(ci) != 2:
            raise ComparisonChartError(f"{context}.safety.{stem}_rate_ci95 must have two values")
        low = _finite_number(ci[0], f"{context}.safety.{stem}_rate_ci95[0]")
        high = _finite_number(ci[1], f"{context}.safety.{stem}_rate_ci95[1]")
        if low is None or high is None or not 0.0 <= low <= high <= 1.0:
            raise ComparisonChartError(f"{context}.safety.{stem}_rate_ci95 is invalid")
    comfort_agents = _mapping(
        _mapping(aggregate["comfort"], f"{context}.comfort").get("per_agent"),
        f"{context}.comfort.per_agent",
    )
    if set(comfort_agents) != set(AGENT_IDS):
        raise ComparisonChartError(
            f"{context}.comfort.per_agent must contain {AGENT_IDS}"
        )
    for agent_id in AGENT_IDS:
        agent = _mapping(
            comfort_agents[agent_id], f"{context}.comfort.per_agent.{agent_id}"
        )
        for field in COMFORT_REQUIRED_FIELDS:
            _finite_number(
                agent.get(field),
                f"{context}.comfort.per_agent.{agent_id}.{field}",
                nullable=True,
            )


def _validate_timing_and_gate(model: Mapping[str, Any], context: str) -> None:
    timing = _mapping(model.get("timing"), f"{context}.timing")
    for component in TIMING_COMPONENTS:
        summary = _mapping(timing.get(component), f"{context}.timing.{component}")
        for stat in ("p50_ms", "p95_ms", "p99_ms", "max_ms"):
            _finite_number(summary.get(stat), f"{context}.timing.{component}.{stat}")
        if component == "planning_tick_ms":
            fraction = _finite_number(
                summary.get("over_200_ms_fraction"),
                f"{context}.timing.{component}.over_200_ms_fraction",
            )
            if fraction is None or not 0.0 <= fraction <= 1.0:
                raise ComparisonChartError(
                    f"{context}.timing.{component}.over_200_ms_fraction must be in [0,1]"
                )
    gates = _mapping(model.get("gate_results"), f"{context}.gate_results")
    gate = _mapping(
        gates.get("planning_tick_p95_ms"),
        f"{context}.gate_results.planning_tick_p95_ms",
    )
    _finite_number(
        gate.get("threshold_ms"),
        f"{context}.gate_results.planning_tick_p95_ms.threshold_ms",
    )
    _finite_number(
        gate.get("observed_ms"),
        f"{context}.gate_results.planning_tick_p95_ms.observed_ms",
    )
    if not isinstance(gate.get("passed"), bool):
        raise ComparisonChartError(
            f"{context}.gate_results.planning_tick_p95_ms.passed must be boolean"
        )


def _validate_comparisons(
    comparisons: object,
    model_ids: Sequence[str],
    metric_definitions: Mapping[str, Any],
) -> Mapping[str, Any]:
    values = _mapping(comparisons, "comparisons")
    if not values:
        raise ComparisonChartError("comparisons must contain at least one pair")
    for comparison_id, raw in values.items():
        if not isinstance(comparison_id, str) or not comparison_id:
            raise ComparisonChartError("comparison ids must be non-empty strings")
        comparison = _mapping(raw, f"comparisons.{comparison_id}")
        baseline = comparison.get("baseline")
        candidate = comparison.get("candidate")
        if baseline not in model_ids or candidate not in model_ids or baseline == candidate:
            raise ComparisonChartError(f"comparisons.{comparison_id} has invalid model ids")
        metrics = _mapping(comparison.get("metrics"), f"comparisons.{comparison_id}.metrics")
        if set(metrics) != set(metric_definitions):
            raise ComparisonChartError(
                f"comparisons.{comparison_id}.metrics must exactly match metric_definitions"
            )
        for metric_name, raw_metric in metrics.items():
            context = f"comparisons.{comparison_id}.metrics.{metric_name}"
            if not isinstance(metric_name, str) or "." not in metric_name:
                raise ComparisonChartError(f"{context} must use a dotted metric path")
            metric = _mapping(raw_metric, context)
            unit = metric.get("unit")
            direction = metric.get("direction")
            n_pairs = metric.get("n_pairs")
            if not isinstance(unit, str) or not unit:
                raise ComparisonChartError(f"{context}.unit must be a non-empty string")
            if direction not in METRIC_DIRECTIONS:
                raise ComparisonChartError(
                    f"{context}.direction must be one of {METRIC_DIRECTIONS}"
                )
            definition = _mapping(
                metric_definitions[metric_name], f"metric_definitions.{metric_name}"
            )
            if definition.get("unit") != unit or definition.get("direction") != direction:
                raise ComparisonChartError(
                    f"{context} unit/direction differs from metric_definitions"
                )
            if isinstance(n_pairs, bool) or not isinstance(n_pairs, int) or n_pairs < 0:
                raise ComparisonChartError(f"{context}.n_pairs must be non-negative")
            baseline_value = _finite_number(
                metric.get("baseline"), f"{context}.baseline", nullable=True
            )
            candidate_value = _finite_number(
                metric.get("candidate"), f"{context}.candidate", nullable=True
            )
            delta = _finite_number(
                metric.get("candidate_minus_baseline"),
                f"{context}.candidate_minus_baseline",
                nullable=True,
            )
            ci = metric.get("ci95")
            if ci is None and n_pairs == 0:
                low = high = None
            elif isinstance(ci, list) and len(ci) == 2:
                low = _finite_number(ci[0], f"{context}.ci95[0]", nullable=True)
                high = _finite_number(ci[1], f"{context}.ci95[1]", nullable=True)
            else:
                raise ComparisonChartError(
                    f"{context}.ci95 must have two values, or be null when n_pairs is zero"
                )
            if n_pairs > 0 and (
                baseline_value is None
                or candidate_value is None
                or delta is None
                or low is None
                or high is None
            ):
                raise ComparisonChartError(f"{context} has data missing for n_pairs > 0")
            if (low is None) != (high is None) or (
                low is not None and high is not None and low > high
            ):
                raise ComparisonChartError(f"{context}.ci95 is invalid")
    return values


def _validate_single_report(report: Mapping[str, Any], context: str = "report") -> None:
    if report.get("format") != SINGLE_REPORT_FORMAT:
        raise ComparisonChartError(f"{context}.format must be {SINGLE_REPORT_FORMAT}")
    if report.get("evaluation_status") not in (
        "completed",
        "completed_with_gate_failure",
    ):
        raise ComparisonChartError(f"{context}.evaluation_status is invalid")
    if not isinstance(report.get("all_gates_passed"), bool):
        raise ComparisonChartError(f"{context}.all_gates_passed must be boolean")
    _mapping(report.get("evaluation_protocol"), f"{context}.evaluation_protocol")
    metric_definitions = _mapping(
        report.get("metric_definitions"), f"{context}.metric_definitions"
    )
    if not metric_definitions:
        raise ComparisonChartError(f"{context}.metric_definitions must not be empty")
    for metric_name, raw_definition in metric_definitions.items():
        if not isinstance(metric_name, str) or "." not in metric_name:
            raise ComparisonChartError(
                f"{context}.metric_definitions keys must be dotted metric paths"
            )
        definition = _mapping(
            raw_definition, f"{context}.metric_definitions.{metric_name}"
        )
        if (
            not isinstance(definition.get("unit"), str)
            or not definition["unit"]
            or definition.get("direction") not in METRIC_DIRECTIONS
            or definition.get("statistical_unit") != "episode"
        ):
            raise ComparisonChartError(
                f"{context}.metric_definitions.{metric_name} is invalid"
            )
    model_order = report.get("model_order")
    models = _mapping(report.get("models"), f"{context}.models")
    if (
        not isinstance(model_order, list)
        or len(model_order) < 2
        or any(not isinstance(item, str) or not item for item in model_order)
        or len(set(model_order)) != len(model_order)
        or set(model_order) != set(models)
    ):
        raise ComparisonChartError(f"{context}.model_order must exactly match models")

    reference_keys: set[tuple[str, str, int]] | None = None
    for model_id in model_order:
        model = _mapping(models[model_id], f"{context}.models.{model_id}")
        rows = model.get("episode_metrics")
        if not isinstance(rows, list) or not rows:
            raise ComparisonChartError(
                f"{context}.models.{model_id}.episode_metrics must be a non-empty list"
            )
        episode_keys = {
            _validate_episode(row, model_id, index) for index, row in enumerate(rows)
        }
        if len(episode_keys) != len(rows):
            raise ComparisonChartError(
                f"{context}.models.{model_id}.episode_metrics contains duplicate episodes"
            )
        if reference_keys is None:
            reference_keys = episode_keys
        elif episode_keys != reference_keys:
            raise ComparisonChartError(
                f"{context}.models.{model_id}.episode_metrics is not strictly paired"
            )
        overall = model.get("overall")
        _validate_aggregate(overall, f"{context}.models.{model_id}.overall")
        if overall["episodes"] != len(rows):
            raise ComparisonChartError(
                f"{context}.models.{model_id}.overall.episodes does not match episode_metrics"
            )
        by_scenario = _mapping(
            model.get("by_scenario"), f"{context}.models.{model_id}.by_scenario"
        )
        scenario_ids = {key[0] for key in episode_keys}
        if set(by_scenario) != scenario_ids:
            raise ComparisonChartError(
                f"{context}.models.{model_id}.by_scenario does not match episode scenarios"
            )
        for scenario, aggregate in by_scenario.items():
            _validate_aggregate(
                aggregate, f"{context}.models.{model_id}.by_scenario.{scenario}"
            )
            expected_count = sum(key[0] == scenario for key in episode_keys)
            if aggregate["episodes"] != expected_count:
                raise ComparisonChartError(
                    f"{context}.models.{model_id}.by_scenario.{scenario}.episodes is inconsistent"
                )
        _validate_timing_and_gate(model, f"{context}.models.{model_id}")
    _validate_comparisons(report.get("comparisons"), model_order, metric_definitions)
    observed_all_gates = all(
        bool(models[model_id]["gate_results"]["planning_tick_p95_ms"]["passed"])
        for model_id in model_order
    )
    if report["all_gates_passed"] != observed_all_gates or (
        report["evaluation_status"] == "completed"
    ) != observed_all_gates:
        raise ComparisonChartError(
            f"{context} evaluation status disagrees with model gate_results"
        )


def _load_and_validate(report_path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ComparisonChartError(f"report is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise ComparisonChartError(f"cannot read report {report_path}: {exc}") from exc
    report = dict(_mapping(raw, "report"))
    report_format = report.get("format")
    if report_format == SINGLE_REPORT_FORMAT:
        _validate_single_report(report)
        return report
    if report_format != REPEAT_REPORT_FORMAT:
        raise ComparisonChartError(
            f"report.format must be {SINGLE_REPORT_FORMAT} or {REPEAT_REPORT_FORMAT}"
        )
    repeats = report.get("repeat_reports")
    repeat_count = report.get("repeat_count")
    if (
        not isinstance(repeats, list)
        or len(repeats) < 2
        or isinstance(repeat_count, bool)
        or not isinstance(repeat_count, int)
        or repeat_count != len(repeats)
    ):
        raise ComparisonChartError("repeat report has an invalid repeat_count/repeat_reports")
    for index, repeat in enumerate(repeats, start=1):
        _validate_single_report(
            _mapping(repeat, f"repeat_reports[{index - 1}]"),
            f"repeat_reports[{index - 1}]",
        )
    first = _mapping(repeats[0], "repeat_reports[0]")
    for key in (
        "evaluation_status",
        "all_gates_passed",
        "model_order",
        "evaluation_protocol",
        "metric_definitions",
        "models",
        "comparisons",
        "fixed_process_reproducibility",
    ):
        if key not in report:
            raise ComparisonChartError(f"repeat report is missing top-level {key}")
    if report["evaluation_status"] not in (
        "completed",
        "completed_with_gate_failure",
    ) or not isinstance(report["all_gates_passed"], bool):
        raise ComparisonChartError("repeat report evaluation status is invalid")
    for index, repeat in enumerate(repeats[1:], start=2):
        if repeat["model_order"] != first["model_order"]:
            raise ComparisonChartError(
                f"repeat {index} model_order differs from repeat 1"
            )
        if (
            repeat["evaluation_protocol"] != first["evaluation_protocol"]
            or repeat["metric_definitions"] != first["metric_definitions"]
        ):
            raise ComparisonChartError(
                f"repeat {index} protocol/metric definitions differ from repeat 1"
            )
    if (
        report["model_order"] != first["model_order"]
        or report["evaluation_protocol"] != first["evaluation_protocol"]
        or report["metric_definitions"] != first["metric_definitions"]
    ):
        raise ComparisonChartError(
            "repeat report top-level protocol differs from repeat 1"
        )
    if report["models"] != first["models"] or report["comparisons"] != first["comparisons"]:
        raise ComparisonChartError(
            "repeat report top-level models/comparisons must equal repeat 1"
        )
    reproducibility = _mapping(
        report["fixed_process_reproducibility"],
        "fixed_process_reproducibility",
    )
    tolerance = _mapping(
        reproducibility.get("tolerance_comparison"),
        "fixed_process_reproducibility.tolerance_comparison",
    )
    tolerance_gate_passed = tolerance.get("tolerance_gate_passed")
    if not isinstance(tolerance_gate_passed, bool):
        raise ComparisonChartError(
            "fixed_process_reproducibility.tolerance_comparison."
            "tolerance_gate_passed must be boolean"
        )
    expected_combined_gate = tolerance_gate_passed and all(
        bool(repeat["all_gates_passed"]) for repeat in repeats
    )
    if report["all_gates_passed"] != expected_combined_gate or (
        report["evaluation_status"] == "completed"
    ) != expected_combined_gate:
        raise ComparisonChartError(
            "repeat report combined status must equal all repeat gates AND "
            "the tolerance gate"
        )
    return report


def _model_color(model_order: Sequence[str], model_id: str) -> str:
    return _PUBLICATION_AGENT_COLORS[model_order.index(model_id) % len(_PUBLICATION_AGENT_COLORS)]


def _episode_index(model: Mapping[str, Any]) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    return {
        (str(row["scenario"]), str(row["route"]), int(row["seed"])): row
        for row in model["episode_metrics"]
    }


def _episode_metric(row: Mapping[str, Any], spec: MetricSpec) -> float | None:
    return _path_value(row, spec.episode_path, "episode", nullable=spec.nullable)


def _aggregate_metric(aggregate: Mapping[str, Any], spec: MetricSpec) -> float | None:
    return _path_value(aggregate, spec.aggregate_path, "aggregate", nullable=spec.nullable)


def _flatten(prefix: str, value: object, output: dict[str, object]) -> None:
    if isinstance(value, Mapping):
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            _flatten(child, value[key], output)
    elif isinstance(value, list):
        output[prefix] = json.dumps(value, separators=(",", ":"))
    else:
        output[prefix] = value


def _csv_value(value: object) -> object:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _write_csv(path: Path, rows: list[dict[str, object]], leading: Sequence[str]) -> None:
    columns = list(leading)
    columns.extend(
        sorted({key for row in rows for key in row if key not in set(leading)})
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in columns})


def _write_tables(report: Mapping[str, Any], table_root: Path) -> list[dict[str, str]]:
    model_order = report["model_order"]
    models = report["models"]
    episode_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    for model_id in model_order:
        model = models[model_id]
        for raw in sorted(
            model["episode_metrics"],
            key=lambda row: (str(row["scenario"]), str(row["route"]), int(row["seed"])),
        ):
            flat: dict[str, object] = {}
            _flatten("", raw, flat)
            episode_rows.append(flat)
        for scenario in sorted(model["by_scenario"]):
            flat = {"model": model_id, "scenario": scenario}
            _flatten("", model["by_scenario"][scenario], flat)
            scenario_rows.append(flat)

    comparison_rows: list[dict[str, object]] = []
    for comparison_id in sorted(report["comparisons"]):
        comparison = report["comparisons"][comparison_id]
        for metric_name in sorted(comparison["metrics"]):
            metric = comparison["metrics"][metric_name]
            ci = metric["ci95"] or (None, None)
            comparison_rows.append(
                {
                    "comparison": comparison_id,
                    "baseline_model": comparison["baseline"],
                    "candidate_model": comparison["candidate"],
                    "metric": metric_name,
                    "baseline": metric["baseline"],
                    "candidate": metric["candidate"],
                    "candidate_minus_baseline": metric["candidate_minus_baseline"],
                    "unit": metric["unit"],
                    "direction": metric["direction"],
                    "n_pairs": metric["n_pairs"],
                    "ci95_low": ci[0],
                    "ci95_high": ci[1],
                }
            )

    paths = (
        ("episode_metrics", table_root / "episode_metrics.csv", episode_rows, ("model", "scenario", "route", "seed")),
        ("scenario_metrics", table_root / "scenario_metrics.csv", scenario_rows, ("model", "scenario")),
        (
            "comparison_metrics",
            table_root / "comparison_metrics.csv",
            comparison_rows,
            ("comparison", "baseline_model", "candidate_model", "metric"),
        ),
    )
    records = []
    for table_id, path, rows, leading in paths:
        _write_csv(path, rows, leading)
        records.append({"id": table_id, "path": str(path)})
    return records


def _save_figure(fig: plt.Figure, chart_root: Path, name: str) -> dict[str, str]:
    png = chart_root / f"{name}.png"
    pdf = chart_root / f"{name}.pdf"
    chart_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(pdf, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return {"id": name, "png": str(png), "pdf": str(pdf)}


def _plot_overview(
    report: Mapping[str, Any],
) -> tuple[plt.Figure, list[str]]:
    model_order = report["model_order"]
    models = report["models"]
    overview_specs = (
        ("Safety", SAFETY_SPECS[0]),
        ("Efficiency", EFFICIENCY_SPECS[0]),
        ("Cooperation", COOPERATION_SPECS[0]),
        ("Comfort", COMFORT_SPECS[2]),
    )
    fig, axes = plt.subplots(1, 5, figsize=(13.0, 2.8), squeeze=False)
    na_notes: list[str] = []
    for ax, (dimension, spec) in zip(axes[0, :4], overview_specs):
        values = [_aggregate_metric(models[name]["overall"], spec) for name in model_order]
        ax.bar(
            np.arange(len(model_order)),
            [np.nan if value is None else value for value in values],
            color=[_model_color(model_order, name) for name in model_order],
            width=0.7,
        )
        for model_index, (model_id, value) in enumerate(zip(model_order, values)):
            if value is not None:
                continue
            ax.text(
                model_index,
                0.04,
                "N/A",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=7,
                color="#333333",
            )
            na_notes.append(f"overview:{spec.key}:{model_id}:N/A")
        ax.set_title(f"{dimension}\n{spec.label}")
        ax.set_ylabel(spec.unit)
        ax.set_xticks(np.arange(len(model_order)), model_order, rotation=25, ha="right")
        _style_publication_axis(ax)
    latency_ax = axes[0, 4]
    timing_values = [float(models[name]["timing"]["planning_tick_ms"]["p95_ms"]) for name in model_order]
    latency_ax.bar(
        np.arange(len(model_order)),
        timing_values,
        color=[_model_color(model_order, name) for name in model_order],
        width=0.7,
    )
    latency_ax.axhline(200.0, color="#D55E00", linestyle="--", linewidth=1.0, label="200 ms gate")
    latency_ax.set_title("Real-time\nPlanning tick P95")
    latency_ax.set_ylabel("ms")
    latency_ax.set_xticks(np.arange(len(model_order)), model_order, rotation=25, ha="right")
    _style_publication_axis(latency_ax)
    latency_ax.legend(frameon=False, loc="best")
    fig.suptitle("Stage1 / GRPO five-dimension comparison (raw metrics)", fontsize=10)
    fig.tight_layout()
    return fig, na_notes


def _rate_ci(aggregate: Mapping[str, Any], spec: MetricSpec) -> tuple[float, float]:
    stem = spec.aggregate_path[-1].removesuffix("_rate")
    ci = aggregate["safety"][f"{stem}_rate_ci95"]
    return float(ci[0]), float(ci[1])


def _plot_safety_rates(report: Mapping[str, Any]) -> plt.Figure:
    model_order = report["model_order"]
    models = report["models"]
    x = np.arange(len(SAFETY_SPECS), dtype=np.float64)
    width = 0.82 / len(model_order)
    fig, ax = plt.subplots(figsize=(8.5, 3.5))
    for model_index, model_id in enumerate(model_order):
        aggregate = models[model_id]["overall"]
        values = np.asarray([_aggregate_metric(aggregate, spec) for spec in SAFETY_SPECS], dtype=np.float64)
        intervals = [_rate_ci(aggregate, spec) for spec in SAFETY_SPECS]
        lower = values - np.asarray([item[0] for item in intervals])
        upper = np.asarray([item[1] for item in intervals]) - values
        positions = x - 0.41 + width / 2.0 + model_index * width
        ax.bar(
            positions,
            values,
            width,
            color=_model_color(model_order, model_id),
            label=model_id,
            yerr=np.vstack((lower, upper)),
            capsize=2,
            error_kw={"linewidth": 0.7},
        )
    ax.set_xticks(x, [spec.label for spec in SAFETY_SPECS], rotation=18, ha="right")
    ax.set_ylabel("Episode rate with 95% CI")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Safety event rates")
    _style_publication_axis(ax)
    ax.legend(frameon=False, ncol=min(3, len(model_order)))
    fig.tight_layout()
    return fig


def _plot_risk_ecdf(report: Mapping[str, Any]) -> tuple[plt.Figure, list[str]]:
    model_order = report["model_order"]
    models = report["models"]
    thresholds = {
        "minimum_background_gap_m": 5.0,
        "minimum_platoon_gap_m": 7.0,
        "minimum_ttc_s": 1.5,
    }
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8), squeeze=False)
    na_notes: list[str] = []
    for ax, spec in zip(axes.flat, RISK_SPECS):
        plotted = False
        missing_models = []
        for model_id in model_order:
            values = [
                _episode_metric(row, spec)
                for row in models[model_id]["episode_metrics"]
            ]
            numeric = np.asarray([value for value in values if value is not None], dtype=np.float64)
            if numeric.size == 0:
                missing_models.append(model_id)
                na_notes.append(f"{spec.key}:{model_id}:N/A")
                continue
            numeric.sort()
            probability = np.arange(1, numeric.size + 1, dtype=np.float64) / numeric.size
            ax.step(
                numeric,
                probability,
                where="post",
                color=_model_color(model_order, model_id),
                label=f"{model_id} (n={numeric.size})",
            )
            plotted = True
        if spec.key in thresholds:
            ax.axvline(
                thresholds[spec.key],
                color="#777777",
                linestyle="--",
                linewidth=0.9,
                label=f"{thresholds[spec.key]:g} {spec.unit}",
            )
        if not plotted:
            message = (
                "N/A — no closing TTC observations"
                if spec.key == "minimum_ttc_s"
                else "N/A — no valid observations"
            )
            ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center")
        elif missing_models:
            ax.text(
                0.02,
                0.04,
                "N/A: " + ", ".join(missing_models),
                transform=ax.transAxes,
                fontsize=7,
                va="bottom",
            )
        ax.set_title(spec.label)
        ax.set_xlabel(spec.unit)
        ax.set_ylabel("ECDF")
        ax.set_ylim(0.0, 1.02)
        _style_publication_axis(ax, grid_axis="both")
        if plotted:
            ax.legend(frameon=False, fontsize=6)
    fig.suptitle("Episode-level safety-risk distributions", fontsize=10)
    fig.tight_layout()
    return fig, na_notes


def _comparison_pairs(report: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    return [
        (comparison_id, value["baseline"], value["candidate"])
        for comparison_id, value in sorted(report["comparisons"].items())
    ]


def _plot_paired_metrics(
    report: Mapping[str, Any],
    specs: Sequence[MetricSpec],
    *,
    title: str,
    columns: int = 2,
) -> tuple[plt.Figure, list[str]]:
    model_order = report["model_order"]
    models = report["models"]
    pairs = _comparison_pairs(report)
    rows = math.ceil(len(specs) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.1 * columns, 2.8 * rows), squeeze=False)
    na_notes: list[str] = []
    for ax, spec in zip(axes.flat, specs):
        tick_positions: list[float] = []
        tick_labels: list[str] = []
        plotted = False
        for pair_index, (comparison_id, baseline_id, candidate_id) in enumerate(pairs):
            baseline = _episode_index(models[baseline_id])
            candidate = _episode_index(models[candidate_id])
            left = float(pair_index * 3)
            right = left + 1.0
            tick_positions.extend((left, right))
            tick_labels.extend((f"{comparison_id}\n{baseline_id}", f"{comparison_id}\n{candidate_id}"))
            pair_values = []
            for key in sorted(baseline):
                baseline_value = _episode_metric(baseline[key], spec)
                candidate_value = _episode_metric(candidate[key], spec)
                if baseline_value is None or candidate_value is None:
                    continue
                pair_values.append((baseline_value, candidate_value))
            if not pair_values:
                na_notes.append(f"{spec.key}:{comparison_id}:N/A")
                continue
            plotted = True
            for baseline_value, candidate_value in pair_values:
                ax.plot((left, right), (baseline_value, candidate_value), color="#999999", alpha=0.35, linewidth=0.7)
            ax.scatter(
                np.full(len(pair_values), left),
                [item[0] for item in pair_values],
                color=_model_color(model_order, baseline_id),
                s=13,
                zorder=3,
            )
            ax.scatter(
                np.full(len(pair_values), right),
                [item[1] for item in pair_values],
                color=_model_color(model_order, candidate_id),
                s=13,
                zorder=3,
            )
        if not plotted:
            ax.text(0.5, 0.5, "N/A — no paired observations", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(spec.label)
        ax.set_ylabel(spec.unit)
        ax.set_xticks(tick_positions, tick_labels, rotation=20, ha="right")
        _style_publication_axis(ax, grid_axis="both")
    for ax in axes.flat[len(specs) :]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig, na_notes


def _plot_latency(report: Mapping[str, Any]) -> plt.Figure:
    model_order = report["model_order"]
    models = report["models"]
    fig, axes = plt.subplots(
        len(model_order),
        1,
        figsize=(8.4, max(3.0, 2.45 * len(model_order))),
        squeeze=False,
    )
    x = np.arange(len(TIMING_COMPONENTS), dtype=np.float64)
    width = 0.36
    for ax, model_id in zip(axes[:, 0], model_order):
        timing = models[model_id]["timing"]
        p50 = [float(timing[name]["p50_ms"]) for name in TIMING_COMPONENTS]
        p95 = [float(timing[name]["p95_ms"]) for name in TIMING_COMPONENTS]
        ax.bar(x - width / 2.0, p50, width, color="#56B4E9", label="P50")
        ax.bar(x + width / 2.0, p95, width, color="#D55E00", label="P95")
        ax.axhline(200.0, color="#333333", linestyle="--", linewidth=0.9, label="200 ms gate")
        planning = timing["planning_tick_ms"]
        ax.set_title(
            f"{model_id}: full tick P99={planning['p99_ms']:.1f} ms, "
            f"max={planning['max_ms']:.1f} ms, "
            f">200 ms={100.0 * planning['over_200_ms_fraction']:.1f}%"
        )
        ax.set_ylabel("Latency (ms)")
        ax.set_xticks(x, [TIMING_LABELS[name] for name in TIMING_COMPONENTS], rotation=18, ha="right")
        _style_publication_axis(ax)
        ax.legend(frameon=False, ncol=3)
    fig.suptitle("Planning latency breakdown", fontsize=10)
    fig.tight_layout()
    return fig


def _scenario_order(report: Mapping[str, Any]) -> list[str]:
    first_model = report["models"][report["model_order"][0]]
    return sorted(first_model["by_scenario"])


def _directional_color_delta(delta: float, direction: str) -> float:
    """Map raw delta to improvement color without judging descriptive metrics."""

    if direction == "higher":
        return float(delta)
    if direction == "lower":
        return -float(delta)
    if direction == "descriptive":
        return 0.0
    raise ComparisonChartError(f"unknown metric direction: {direction}")


def _plot_scenario_heatmap(report: Mapping[str, Any]) -> plt.Figure:
    models = report["models"]
    scenarios = _scenario_order(report)
    pairs = _comparison_pairs(report)
    specs = (
        SAFETY_SPECS[0],
        EFFICIENCY_SPECS[0],
        COOPERATION_SPECS[0],
        COMFORT_SPECS[2],
        TIMING_SPEC,
    )
    fig, axes = plt.subplots(
        len(pairs),
        1,
        figsize=(max(7.0, 1.2 * len(scenarios)), 2.2 + 2.0 * len(pairs)),
        squeeze=False,
    )
    images = []
    for ax, (comparison_id, baseline_id, candidate_id) in zip(axes[:, 0], pairs):
        raw = np.full((len(specs), len(scenarios)), np.nan, dtype=np.float64)
        oriented = np.full_like(raw, np.nan)
        for row_index, spec in enumerate(specs):
            for column_index, scenario in enumerate(scenarios):
                baseline = _aggregate_metric(models[baseline_id]["by_scenario"][scenario], spec)
                candidate = _aggregate_metric(models[candidate_id]["by_scenario"][scenario], spec)
                if baseline is None or candidate is None:
                    continue
                delta = candidate - baseline
                raw[row_index, column_index] = delta
                oriented[row_index, column_index] = _directional_color_delta(
                    delta, spec.direction
                )
        normalized = np.full_like(oriented, np.nan)
        for row_index in range(len(specs)):
            finite = np.abs(oriented[row_index, np.isfinite(oriented[row_index])])
            scale = float(np.max(finite)) if finite.size else 0.0
            if scale > 0.0:
                normalized[row_index] = oriented[row_index] / scale
            else:
                normalized[row_index, np.isfinite(oriented[row_index])] = 0.0
        image = ax.imshow(normalized, cmap="PuOr", vmin=-1.0, vmax=1.0, aspect="auto")
        images.append(image)
        ax.set_title(f"{comparison_id}: {candidate_id} − {baseline_id}")
        ax.set_xticks(np.arange(len(scenarios)), scenarios, rotation=25, ha="right")
        ax.set_yticks(np.arange(len(specs)), [spec.label for spec in specs])
        for row_index, spec in enumerate(specs):
            for column_index in range(len(scenarios)):
                value = raw[row_index, column_index]
                annotation = "N/A" if not np.isfinite(value) else f"{value:+.3g}\n{spec.unit}"
                color_value = normalized[row_index, column_index]
                text_color = (
                    "white"
                    if np.isfinite(color_value) and abs(float(color_value)) >= 0.55
                    else "black"
                )
                ax.text(
                    column_index,
                    row_index,
                    annotation,
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=text_color,
                )
        ax.tick_params(length=0)
    if images:
        fig.colorbar(
            images[0],
            ax=list(axes[:, 0]),
            fraction=0.025,
            pad=0.03,
            label="Direction-adjusted row-scaled delta",
        )
    fig.suptitle("Scenario-level raw deltas (annotations); color is row-scaled", fontsize=10)
    return fig


def _human_metric_name(path: str) -> str:
    return path.replace(".", " / ").replace("_", " ")


def _plot_forest(report: Mapping[str, Any]) -> tuple[plt.Figure, list[str]]:
    model_order = report["model_order"]
    grouped: dict[str, list[tuple[str, str, Mapping[str, Any]]]] = defaultdict(list)
    na_notes: list[str] = []
    for comparison_id, comparison in sorted(report["comparisons"].items()):
        for metric_name, metric in sorted(comparison["metrics"].items()):
            if metric_name not in CORE_FOREST_METRICS:
                continue
            grouped[str(metric["unit"])].append((comparison_id, metric_name, metric))
    units = sorted(grouped)
    figure_height = max(3.0, sum(max(1.6, 0.34 * len(grouped[unit])) for unit in units))
    fig, axes = plt.subplots(len(units), 1, figsize=(9.5, figure_height), squeeze=False)
    for ax, unit in zip(axes[:, 0], units):
        rows = grouped[unit]
        positions = np.arange(len(rows), dtype=np.float64)
        valid_values: list[float] = []
        for position, (comparison_id, metric_name, metric) in zip(positions, rows):
            delta = metric["candidate_minus_baseline"]
            ci = metric["ci95"]
            low, high = (None, None) if ci is None else ci
            candidate_id = report["comparisons"][comparison_id]["candidate"]
            if delta is None or low is None or high is None:
                ax.text(0.0, position, "N/A", ha="center", va="center", fontsize=7)
                na_notes.append(f"{comparison_id}:{metric_name}:N/A")
                continue
            delta_value = float(delta)
            low_value = float(low)
            high_value = float(high)
            valid_values.extend((delta_value, low_value, high_value))
            color = _model_color(model_order, candidate_id)
            ax.hlines(position, low_value, high_value, color=color, linewidth=1.2)
            ax.plot(delta_value, position, marker="o", color=color, markersize=4)
        labels = [f"{comparison_id}: {_human_metric_name(metric_name)}" for comparison_id, metric_name, _ in rows]
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.axvline(0.0, color="#555555", linestyle="--", linewidth=0.8)
        if not valid_values:
            ax.set_xlim(-1.0, 1.0)
        ax.set_xlabel(f"Candidate − baseline ({unit})")
        _style_publication_axis(ax, grid_axis="x")
    candidate_order = list(
        dict.fromkeys(
            comparison["candidate"]
            for _, comparison in sorted(report["comparisons"].items())
        )
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=_model_color(model_order, name),
            label=name,
        )
        for name in candidate_order
    ]
    fig.legend(handles=handles, loc="upper center", ncol=min(4, len(handles)), frameon=False)
    fig.suptitle("Paired episode delta estimates with 95% confidence intervals", fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    return fig, na_notes


def generate_comparison_charts(report_path: Path, output_root: Path) -> dict[str, Any]:
    """Validate one v3 report and materialize its chart/table bundle."""

    report_path = Path(report_path)
    output_root = Path(output_root)
    report = _load_and_validate(report_path)
    chart_root = output_root / "charts"
    table_root = output_root / "tables"

    table_records = _write_tables(report, table_root)
    chart_records: list[dict[str, str]] = []
    na_notes: list[str] = []
    with plt.rc_context(_PUBLICATION_RC):
        figure, notes = _plot_overview(report)
        chart_records.append(
            _save_figure(figure, chart_root, "comparison_overview")
        )
        na_notes.extend(notes)
        chart_records.append(_save_figure(_plot_safety_rates(report), chart_root, "safety_rates"))
        figure, notes = _plot_risk_ecdf(report)
        chart_records.append(_save_figure(figure, chart_root, "risk_ecdf"))
        na_notes.extend(notes)
        figure, notes = _plot_paired_metrics(
            report, EFFICIENCY_SPECS, title="Paired episode efficiency metrics"
        )
        chart_records.append(_save_figure(figure, chart_root, "efficiency_paired"))
        na_notes.extend(notes)
        figure, notes = _plot_paired_metrics(
            report, COOPERATION_SPECS, title="Paired episode cooperation metrics"
        )
        chart_records.append(_save_figure(figure, chart_root, "cooperation_paired"))
        na_notes.extend(notes)
        figure, notes = _plot_paired_metrics(
            report,
            COMFORT_SPECS,
            title="Paired episode comfort and command-slew diagnostics",
            columns=3,
        )
        chart_records.append(_save_figure(figure, chart_root, "comfort_paired"))
        na_notes.extend(notes)
        chart_records.append(_save_figure(_plot_latency(report), chart_root, "latency_breakdown"))
        chart_records.append(
            _save_figure(_plot_scenario_heatmap(report), chart_root, "scenario_delta_heatmap")
        )
        figure, notes = _plot_forest(report)
        chart_records.append(_save_figure(figure, chart_root, "paired_delta_forest"))
        na_notes.extend(notes)

    index = {
        "format": CHART_INDEX_FORMAT,
        "source_report": str(report_path),
        "source_report_format": report["format"],
        "evaluation_status": report["evaluation_status"],
        "all_gates_passed": report["all_gates_passed"],
        "repeat_tolerance_gate_passed": (
            report["fixed_process_reproducibility"]["tolerance_comparison"][
                "tolerance_gate_passed"
            ]
            if report["format"] == REPEAT_REPORT_FORMAT
            else None
        ),
        "model_order": list(report["model_order"]),
        "charts": chart_records,
        "tables": table_records,
        "na_observations": sorted(set(na_notes)),
    }
    index_path = output_root / "chart_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        generate_comparison_charts(arguments.report, arguments.output_root)
    except ComparisonChartError as exc:
        parser.exit(2, f"bev comparison chart error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
