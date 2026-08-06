from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.control_tracking_benchmark import (
    ControlBenchmarkError,
    evaluate_gap_feedback_contract,
    run_control_tracking_benchmark,
    standard_control_cases,
    summarize_control_result,
)


def _trace(
    *, lateral=0.0, heading=0.0, saturation=0.0, speed=4.0, gap_error=0.0
):
    count = 8
    return {
        "reference_world": np.zeros((count, 3)).tolist(),
        "actual_world": np.zeros((count, 3)).tolist(),
        "longitudinal_errors_m": [0.1] * count,
        "lateral_errors_m": [lateral] * count,
        "heading_errors_rad": [heading] * count,
        "position_error_speed_increment_mps": [0.0] * count,
        "actual_acceleration_mps2": [0.0] * count,
        "desired_acceleration_mps2": [0.0] * count,
        "acceleration_bias_mps2": [0.0] * count,
        "controller_speed_error_mps": [0.0] * count,
        "formation_control_increment": [0.0] * count,
        "formation_gap_error_m": [gap_error] * count,
        "control_saturated": [False] * count,
        "lateral_heading_contaminated": [False] * count,
        "actual_speed_mps": [speed] * count,
        "steering": [0.0] * count,
        "throttle": [0.0] * count,
        "reference_feedforward_speed_mps": [speed] * count,
        "reference_feedforward_acceleration_mps2": [0.0] * count,
        "maximum_continuous_saturation_s": saturation,
    }


class _Reward:
    def __init__(self, unsafe=False):
        self.rewards = np.asarray([0.0], dtype=np.float32)
        self.unsafe = np.asarray([unsafe], dtype=bool)


def _result(case, *, lateral=0.0, heading=0.0, unsafe=False):
    stop = case.category == "stop"
    speed = 0.0 if stop else max(case.initial_speed_mps, 0.1)
    return SimpleNamespace(
        reward=_Reward(unsafe),
        initial_speed_mps=np.full((1, 3), case.initial_speed_mps),
        tracking_traces=(
            tuple(
                _trace(lateral=lateral, heading=heading, speed=speed)
                for _ in range(3)
            ),
        ),
        failure_reasons=((),),
    )


def test_standard_cases_are_unique_and_hard_valid() -> None:
    cases = standard_control_cases()
    assert len(cases) == 9
    assert len({case.name for case in cases}) == len(cases)
    assert {case.category for case in cases} == {
        "longitudinal",
        "stop",
        "lateral",
        "curve",
        "merge",
    }
    assert all(case.trajectory.shape == (8, 3) for case in cases)
    assert all(case.trajectory.dtype == np.float32 for case in cases)


def test_summary_separates_lateral_and_heading_thresholds() -> None:
    case = standard_control_cases()[0]
    passed = summarize_control_result(case, _result(case))
    assert passed["passed"] is True
    failed = summarize_control_result(
        case,
        _result(case, lateral=0.51, heading=0.11),
    )
    assert failed["passed"] is False
    assert "lateral_p95" in failed["blockers"]
    assert "heading_p95" in failed["blockers"]


def test_summary_uses_speed_for_independent_and_gap_for_locked() -> None:
    case = standard_control_cases()[0]
    independent = summarize_control_result(
        case,
        _result(case),
        control_mode="independent",
    )
    assert independent["signed_speed_error_p95_mps"] == pytest.approx(0.0)
    assert independent["passed"] is True

    result = _result(case)
    for trace in result.tracking_traces[0][1:]:
        trace["formation_gap_error_m"] = [2.0] * 8
    locked = summarize_control_result(case, result, control_mode="locked")
    assert locked["passed"] is False
    assert "desired_center_gap_error_p95" in locked["blockers"]


def test_gap_feedback_contract_has_bounded_direction() -> None:
    report = evaluate_gap_feedback_contract()
    assert report["passed"] is True
    values = {
        row["name"]: row for row in report["cases"]
    }
    assert values["independent"]["applied_gap_feedback_mps2"] == 0.0
    assert values["locked_nominal"]["applied_gap_feedback_mps2"] == 0.0
    assert values["follower_too_close"]["applied_gap_feedback_mps2"] == -1.0
    assert values["follower_too_far"]["applied_gap_feedback_mps2"] == 1.0


class _Evaluator:
    def _make_env(self, spec):
        class _Env:
            def __init__(self):
                self.agents = {
                    f"agent{role}": SimpleNamespace(
                        position=np.asarray([0.0, -15.74 * role]),
                        heading_theta=0.0,
                    )
                    for role in range(3)
                }

            def close(self):
                return None

        return _Env()

    def evaluate(self, spec, prefix, trajectories):
        del spec, prefix
        case = next(
            item
            for item in standard_control_cases()
            if np.array_equal(item.trajectory, trajectories[0, 0])
        )
        return _result(case)


def test_benchmark_writes_json_and_numeric_npz(tmp_path) -> None:
    report = run_control_tracking_benchmark(
        tmp_path,
        case_names=("constant_4", "lane_change_left"),
        evaluator=_Evaluator(),
        control_modes=("locked",),
    )
    assert report["passed"] is True
    assert (tmp_path / "control_benchmark_summary.json").is_file()
    for name in ("constant_4", "lane_change_left"):
        with np.load(tmp_path / f"locked_{name}.npz", allow_pickle=False) as data:
            assert data["trajectory"].shape == (8, 3)
            assert data["actual_world"].shape == (3, 8, 3)
            assert data["steering"].shape == (3, 8)
            assert data["actual_acceleration_mps2"].shape == (3, 8)


def test_unknown_case_is_rejected(tmp_path) -> None:
    with pytest.raises(ControlBenchmarkError, match="unknown control cases"):
        run_control_tracking_benchmark(
            tmp_path,
            case_names=("missing",),
            evaluator=_Evaluator(),
            control_modes=("locked",),
        )


def test_invalid_control_mode_is_rejected(tmp_path) -> None:
    with pytest.raises(ControlBenchmarkError, match="control_modes"):
        run_control_tracking_benchmark(
            tmp_path,
            case_names=("constant_4",),
            evaluator=_Evaluator(),
            control_modes=("unknown",),
        )
