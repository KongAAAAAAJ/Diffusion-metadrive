from __future__ import annotations

import numpy as np
import pytest

from evaluation.calibrate_longitudinal_actuator import (
    ActuatorCalibrationError,
    calibrate_actuator_from_directory,
    fit_actuator_mapping,
)


def test_fit_recovers_delay_slope_and_negative_zero_bias() -> None:
    throttle = np.random.default_rng(17).uniform(-1.0, 1.0, size=(12, 40))
    throttle[:, ::10] = 0.0
    acceleration = np.zeros_like(throttle)
    acceleration[:, 2:] = 1.5 * throttle[:, :-2] - 0.2
    result, _, _ = fit_actuator_mapping(throttle, acceleration, maximum_lag_steps=4)
    assert result.lag_steps == 2
    assert result.slope_mps2_per_throttle == pytest.approx(1.5)
    assert result.intercept_mps2 == pytest.approx(-0.2)
    assert result.recommended_acceleration_bias_mps2 < 0.0


def test_fit_rejects_missing_near_zero_samples() -> None:
    throttle = np.ones((3, 20), dtype=np.float64)
    with pytest.raises(ActuatorCalibrationError):
        fit_actuator_mapping(throttle, throttle)


def test_directory_calibration_writes_json_and_png(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    throttle = np.tile(np.linspace(-1.0, 1.0, 40), (12, 1))
    acceleration = 1.4 * throttle - 0.1
    np.savez_compressed(
        source / "independent_case.npz",
        throttle=throttle,
        actual_acceleration_mps2=acceleration,
    )
    report = calibrate_actuator_from_directory(
        source, output, skip_steps=0, maximum_lag_steps=0
    )
    assert report["fit"]["intercept_mps2"] == pytest.approx(-0.1)
    assert (output / "actuator_calibration.json").is_file()
    assert (output / "throttle_acceleration.png").is_file()
