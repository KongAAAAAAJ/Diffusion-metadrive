from __future__ import annotations

import numpy as np
import pytest

from evaluation.calibrate_longitudinal_brake import (
    BrakeCalibrationError,
    fit_brake_scale,
)


def test_brake_scale_fit_recovers_affine_mapping() -> None:
    command = np.repeat(
        np.asarray([-0.1, -0.2, -0.4, -0.6, -0.8, -1.0]), 5
    )
    acceleration = 9.5 * command - 0.02
    result = fit_brake_scale(command, acceleration)
    assert result.sample_count == command.size
    assert result.slope_mps2_per_brake == pytest.approx(9.5)
    assert result.intercept_mps2 == pytest.approx(-0.02)
    assert result.correlation == pytest.approx(1.0)
    assert result.recommended_brake_acceleration_scale_mps2 == pytest.approx(9.5)


def test_brake_scale_fit_rejects_insufficient_commands() -> None:
    command = np.full((30,), -0.2)
    acceleration = np.full((30,), -2.0)
    with pytest.raises(BrakeCalibrationError, match="four negative commands"):
        fit_brake_scale(command, acceleration)
