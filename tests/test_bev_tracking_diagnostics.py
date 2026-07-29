from __future__ import annotations

import numpy as np
import pytest

from evaluation.tracking_diagnostics import (
    TrackingDiagnosticError,
    aggregate_tracking_rows,
    classify_false_safe,
    derive_tracking_envelope,
)


def test_tracking_envelope_uses_p99_padding_and_caps() -> None:
    envelope = derive_tracking_envelope(
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 0.4]),
        np.asarray([0.0, 0.05]),
    )
    assert envelope.longitudinal_margin_m == pytest.approx(1.19)
    assert envelope.lateral_margin_m == pytest.approx(0.596)
    assert envelope.heading_margin_rad == pytest.approx(0.0695)
    assert envelope.within_controller_limits is True

    capped = derive_tracking_envelope(
        np.asarray([2.0]), np.asarray([1.2]), np.asarray([0.2])
    )
    assert capped.longitudinal_margin_m == 1.5
    assert capped.lateral_margin_m == 1.0
    assert capped.heading_margin_rad == 0.15
    assert capped.within_controller_limits is False


def test_tracking_report_is_grouped_by_scenario_and_role() -> None:
    rows = [
        {
            "scenario": "S7",
            "role": role,
            "longitudinal_errors_m": [0.1, 0.2],
            "lateral_errors_m": [0.05, -0.1],
            "heading_errors_rad": [0.01, -0.02],
        }
        for role in range(3)
    ]
    report = aggregate_tracking_rows(rows)
    assert len(report["by_scenario_role"]) == 3
    assert report["overall"]["lateral_m"]["p95"] <= 0.1
    with pytest.raises(TrackingDiagnosticError):
        aggregate_tracking_rows([])


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"replay_position_error_m": 0.02}, "replay_or_state_mismatch"),
        (
            {"failure_reasons": ("simulator:agent_removed_without_failure_flag",)},
            "termination_semantics",
        ),
        ({"maximum_lateral_error_m": 0.6}, "controller_tracking"),
        ({"simulator_out_of_drivable": True}, "road_footprint"),
        (
            {
                "simulator_collision": True,
                "minimum_background_gap_m": -0.1,
            },
            "background_prediction",
        ),
        (
            {
                "simulator_collision": True,
                "minimum_platoon_gap_m": -0.1,
            },
            "platoon_interaction",
        ),
    ],
)
def test_false_safe_attribution(kwargs, expected: str) -> None:
    values = {
        "simulator_collision": False,
        "simulator_out_of_drivable": False,
        "failure_reasons": (),
        "replay_position_error_m": 0.0,
        "replay_heading_error_rad": 0.0,
        "maximum_lateral_error_m": 0.0,
        "maximum_heading_error_rad": 0.0,
        "minimum_platoon_gap_m": 10.0,
        "minimum_background_gap_m": 10.0,
    }
    values.update(kwargs)
    assert classify_false_safe(**values) == expected
