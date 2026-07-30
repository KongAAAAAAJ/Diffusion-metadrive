from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.longitudinal_tracking_diagnostics import (
    LongitudinalDiagnosticError,
    audit_longitudinal_trajectory,
    build_longitudinal_tracking_report,
    run_longitudinal_tracking_benchmark,
    summarize_trajectory_audits,
)


def _constant(speed_mps: float) -> np.ndarray:
    trajectory = np.zeros((8, 3), dtype=np.float32)
    trajectory[:, 0] = (
        np.arange(1, 9, dtype=np.float32) * 0.5 * speed_mps
    )
    return trajectory


def _constant_acceleration(
    current_speed_mps: float, acceleration_mps2: float
) -> np.ndarray:
    speed = float(current_speed_mps)
    distance = 0.0
    trajectory = np.zeros((8, 3), dtype=np.float32)
    for index in range(8):
        next_speed = max(0.0, speed + acceleration_mps2 * 0.5)
        distance += 0.5 * (speed + next_speed) * 0.5
        trajectory[index, 0] = distance
        speed = next_speed
    return trajectory


def test_constant_acceleration_braking_and_stop_are_valid() -> None:
    constant = audit_longitudinal_trajectory(_constant(5.0), 5.0)
    accelerate = audit_longitudinal_trajectory(
        _constant_acceleration(5.0, 2.0), 5.0
    )
    stop = audit_longitudinal_trajectory(
        _constant_acceleration(4.0, -4.0), 4.0
    )
    assert constant.valid and accelerate.valid and stop.valid
    np.testing.assert_allclose(constant.sample_times_s, np.arange(1, 9) * 0.5)
    # The contract treats segment speed as the interval-average speed.  The
    # first acceleration therefore compares 5.5 m/s with the instantaneous
    # 5.0 m/s at t=0; subsequent interval-average speeds differ by 1 m/s.
    assert accelerate.acceleration_mps2[0] == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(
        accelerate.acceleration_mps2[1:], 2.0, atol=1e-5
    )
    assert stop.speed_mps[-1] == pytest.approx(0.0)
    assert np.all(np.diff(stop.cumulative_distance_m) >= 0.0)


def test_backward_and_acceleration_spike_have_explicit_reasons() -> None:
    trajectory = _constant(5.0)
    trajectory[2, 0] = trajectory[1, 0] - 1.0
    trajectory[3, 0] = 20.0
    audit = audit_longitudinal_trajectory(trajectory, 5.0)
    assert not audit.valid
    assert "non_forward_motion" in audit.violations
    assert "acceleration_above_max" in audit.violations
    assert "outside_reachable_distance" in audit.violations


def test_first_waypoint_at_time_zero_is_identified() -> None:
    trajectory = _constant(5.0)
    trajectory[1:] = trajectory[:-1]
    trajectory[0] = 0.0
    audit = audit_longitudinal_trajectory(trajectory, 5.0)
    assert "first_waypoint_at_time_zero" in audit.violations
    assert "acceleration_below_min" in audit.violations


def test_heading_alignment_and_lateral_dynamics_are_checked() -> None:
    trajectory = _constant(5.0)
    trajectory[:, 2] = np.asarray([0.0, 1.4] * 4, dtype=np.float32)
    audit = audit_longitudinal_trajectory(trajectory, 5.0)
    assert "heading_alignment" in audit.violations
    assert "yaw_rate_limit" in audit.violations
    assert "lateral_acceleration_limit" in audit.violations


@pytest.mark.parametrize(
    ("trajectory", "speed"),
    (
        (np.zeros((7, 3), dtype=np.float32), 0.0),
        (np.zeros((8, 3), dtype=np.int64), 0.0),
        (np.full((8, 3), np.nan, dtype=np.float32), 0.0),
        (np.zeros((8, 3), dtype=np.float32), -1.0),
    ),
)
def test_invalid_audit_inputs_are_rejected(
    trajectory: np.ndarray, speed: float
) -> None:
    with pytest.raises(LongitudinalDiagnosticError):
        audit_longitudinal_trajectory(trajectory, speed)


def _trace(
    *,
    errors: tuple[float, ...] = (0.1, 0.2),
    contaminated: tuple[bool, ...] = (False, False),
    saturation_s: float = 0.5,
    terminal_speed: float = 2.0,
) -> dict[str, object]:
    return {
        "longitudinal_errors_m": list(errors),
        "lateral_heading_contaminated": list(contaminated),
        "position_error_speed_increment_mps": [0.2] * len(errors),
        "actual_acceleration_mps2": [0.1] * len(errors),
        "formation_control_increment": [0.0] * len(errors),
        "control_saturated": [False] * len(errors),
        "actual_speed_mps": [terminal_speed] * len(errors),
        "maximum_continuous_saturation_s": saturation_s,
    }


def _branch_result(
    trace: dict[str, object],
    *,
    speed_mps: float = 2.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        initial_speed_mps=np.full((1, 3), speed_mps, dtype=np.float32),
        tracking_traces=((trace, trace.copy(), trace.copy()),),
    )


def test_tracking_report_excludes_lateral_heading_contamination() -> None:
    trace = _trace(errors=(0.1, 9.0), contaminated=(False, True))
    candidates = np.stack([np.stack([_constant(2.0)] * 3)])
    report = build_longitudinal_tracking_report(
        _branch_result(trace), candidates
    )
    assert report.clean_sample_count == 3
    assert report.contaminated_sample_count == 3
    assert report.longitudinal_error_p95_m == pytest.approx(0.1)
    assert (
        report.control_decomposition["leader"][
            "clean_longitudinal_error_p95_m"
        ]
        == pytest.approx(0.1)
    )
    assert report.passed


def test_tracking_report_detects_stop_and_saturation_blockers() -> None:
    trace = _trace(saturation_s=1.1, terminal_speed=0.5)
    candidates = np.zeros((1, 3, 8, 3), dtype=np.float32)
    report = build_longitudinal_tracking_report(
        _branch_result(trace, speed_mps=0.0), candidates
    )
    assert not report.passed
    assert "continuous_control_saturation" in report.blockers
    assert "stop_terminal_speed" in report.blockers


def test_public_benchmark_uses_supplied_evaluator_without_mutation() -> None:
    candidates = np.stack([np.stack([_constant(2.0)] * 3)])
    before = candidates.copy()
    result = _branch_result(_trace())
    evaluator = SimpleNamespace(
        evaluate=lambda spec, prefix, trajectories: result
    )
    report = run_longitudinal_tracking_benchmark(
        object(), (), candidates, evaluator=evaluator
    )
    assert report.passed
    np.testing.assert_array_equal(candidates, before)


def test_audit_summary_preserves_source_role_mode_breakdown() -> None:
    valid = audit_longitudinal_trajectory(_constant(2.0), 2.0)
    invalid_trajectory = _constant(2.0)
    invalid_trajectory[1, 0] = -1.0
    invalid = audit_longitudinal_trajectory(invalid_trajectory, 2.0)
    summary = summarize_trajectory_audits(
        (
            {
                "source": "expert",
                "scenario": "S5",
                "role": 0,
                "mode": 0,
                "audit": valid,
            },
            {
                "source": "diffusion_A",
                "scenario": "S5",
                "role": 1,
                "mode": 3,
                "audit": invalid,
            },
        )
    )
    assert summary["total"] == 2
    assert summary["valid"] == 1
    assert summary["violation_counts"]["non_forward_motion"] == 1
    assert summary["breakdown"]["expert/S5/0/0"]["valid"] == 1
    assert summary["source_breakdown"]["expert"]["valid"] == 1
    assert (
        summary["source_breakdown"]["diffusion_A"]["violation_counts"][
            "non_forward_motion"
        ]
        == 1
    )
