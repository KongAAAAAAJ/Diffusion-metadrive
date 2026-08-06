"""Fit and visualize the longitudinal throttle-to-acceleration mapping."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


class ActuatorCalibrationError(RuntimeError):
    """Raised when recorded controller traces cannot support calibration."""


@dataclass(frozen=True)
class ActuatorCalibrationResult:
    sample_count: int
    lag_steps: int
    slope_mps2_per_throttle: float
    intercept_mps2: float
    correlation: float
    zero_throttle_sample_count: int
    zero_throttle_acceleration_median_mps2: float
    zero_throttle_acceleration_p10_mps2: float
    zero_throttle_acceleration_p90_mps2: float
    recommended_acceleration_bias_mps2: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "sample_count": self.sample_count,
            "lag_steps": self.lag_steps,
            "slope_mps2_per_throttle": self.slope_mps2_per_throttle,
            "intercept_mps2": self.intercept_mps2,
            "correlation": self.correlation,
            "zero_throttle_sample_count": self.zero_throttle_sample_count,
            "zero_throttle_acceleration_median_mps2": (
                self.zero_throttle_acceleration_median_mps2
            ),
            "zero_throttle_acceleration_p10_mps2": (
                self.zero_throttle_acceleration_p10_mps2
            ),
            "zero_throttle_acceleration_p90_mps2": (
                self.zero_throttle_acceleration_p90_mps2
            ),
            "recommended_acceleration_bias_mps2": (
                self.recommended_acceleration_bias_mps2
            ),
        }


def _load_traces(
    input_root: Path,
    *,
    skip_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    paths = sorted(input_root.glob("independent_*.npz"))
    if not paths:
        raise ActuatorCalibrationError("no independent_*.npz traces found")
    throttle_rows: list[np.ndarray] = []
    acceleration_rows: list[np.ndarray] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as trace:
            if "throttle" not in trace or "actual_acceleration_mps2" not in trace:
                raise ActuatorCalibrationError(f"missing actuator fields in {path}")
            throttle = np.asarray(trace["throttle"], dtype=np.float64)
            acceleration = np.asarray(
                trace["actual_acceleration_mps2"], dtype=np.float64
            )
        if throttle.shape != acceleration.shape or throttle.ndim != 2:
            raise ActuatorCalibrationError(
                f"throttle/acceleration must share [role,time] shape in {path}"
            )
        if skip_steps >= throttle.shape[1]:
            raise ActuatorCalibrationError("skip_steps removes the entire trace")
        throttle_rows.append(throttle[:, skip_steps:])
        acceleration_rows.append(acceleration[:, skip_steps:])
    return np.concatenate(throttle_rows, axis=0), np.concatenate(
        acceleration_rows, axis=0
    )


def fit_actuator_mapping(
    throttle: np.ndarray,
    acceleration: np.ndarray,
    *,
    maximum_lag_steps: int = 5,
    zero_throttle_tolerance: float = 0.05,
    maximum_abs_acceleration_mps2: float = 8.0,
) -> tuple[ActuatorCalibrationResult, np.ndarray, np.ndarray]:
    u = np.asarray(throttle, dtype=np.float64)
    a = np.asarray(acceleration, dtype=np.float64)
    if u.shape != a.shape or u.ndim != 2 or u.size == 0:
        raise ActuatorCalibrationError("calibration arrays must share non-empty [N,T]")
    if maximum_lag_steps < 0 or maximum_lag_steps >= u.shape[1]:
        raise ActuatorCalibrationError("maximum_lag_steps is outside the trace")

    best: tuple[float, int, float, float, np.ndarray, np.ndarray] | None = None
    for lag in range(maximum_lag_steps + 1):
        delayed_u = u[:, : u.shape[1] - lag] if lag else u
        delayed_a = a[:, lag:] if lag else a
        flat_u = delayed_u.reshape(-1)
        flat_a = delayed_a.reshape(-1)
        valid = (
            np.isfinite(flat_u)
            & np.isfinite(flat_a)
            & (np.abs(flat_a) <= float(maximum_abs_acceleration_mps2))
        )
        flat_u = flat_u[valid]
        flat_a = flat_a[valid]
        if flat_u.size < 20 or np.std(flat_u) <= 1.0e-8:
            continue
        slope, intercept = np.linalg.lstsq(
            np.column_stack((flat_u, np.ones_like(flat_u))),
            flat_a,
            rcond=None,
        )[0]
        correlation = float(np.corrcoef(flat_u, flat_a)[0, 1])
        score = abs(correlation)
        if best is None or score > best[0]:
            best = (
                score,
                lag,
                float(slope),
                float(intercept),
                flat_u,
                flat_a,
            )
    if best is None:
        raise ActuatorCalibrationError("insufficient finite variation for mapping fit")
    _, lag, slope, intercept, fit_u, fit_a = best
    zero = fit_a[np.abs(fit_u) <= float(zero_throttle_tolerance)]
    if zero.size < 20:
        raise ActuatorCalibrationError(
            "at least 20 near-zero throttle samples are required"
        )
    p10, median, p90 = np.percentile(zero, (10, 50, 90))
    result = ActuatorCalibrationResult(
        sample_count=int(fit_u.size),
        lag_steps=int(lag),
        slope_mps2_per_throttle=slope,
        intercept_mps2=intercept,
        correlation=float(np.corrcoef(fit_u, fit_a)[0, 1]),
        zero_throttle_sample_count=int(zero.size),
        zero_throttle_acceleration_median_mps2=float(median),
        zero_throttle_acceleration_p10_mps2=float(p10),
        zero_throttle_acceleration_p90_mps2=float(p90),
        recommended_acceleration_bias_mps2=float(median),
    )
    return result, fit_u, fit_a


def calibrate_actuator_from_directory(
    input_root: Path | str,
    output_root: Path | str,
    *,
    skip_steps: int = 10,
    maximum_lag_steps: int = 5,
) -> dict[str, object]:
    source = Path(input_root)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    throttle, acceleration = _load_traces(source, skip_steps=int(skip_steps))
    result, fit_u, fit_a = fit_actuator_mapping(
        throttle,
        acceleration,
        maximum_lag_steps=int(maximum_lag_steps),
    )
    report: dict[str, object] = {
        "format": "bev_longitudinal_actuator_calibration_v1",
        "input_root": str(source.resolve()),
        "skip_steps": int(skip_steps),
        "maximum_lag_steps": int(maximum_lag_steps),
        "fit": result.as_dict(),
        "warning": (
            "observational fit; dedicated fixed-command step responses are still "
            "required before changing acceleration scale"
        ),
    }
    (output / "actuator_calibration.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(fit_u, fit_a, s=10, alpha=0.25, label="recorded samples")
    x = np.linspace(-1.0, 1.0, 200)
    y = result.slope_mps2_per_throttle * x + result.intercept_mps2
    axis.plot(x, y, color="tab:red", linewidth=2.0, label="least-squares fit")
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("throttle")
    axis.set_ylabel("actual acceleration (m/s²)")
    axis.set_title("Independent control: throttle-to-acceleration")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output / "throttle_acceleration.png", dpi=160)
    plt.close(figure)
    return report


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--skip-steps", type=int, default=10)
    parser.add_argument("--maximum-lag-steps", type=int, default=5)
    args = parser.parse_args(argv)
    report = calibrate_actuator_from_directory(
        args.input_root,
        args.output_root,
        skip_steps=args.skip_steps,
        maximum_lag_steps=args.maximum_lag_steps,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "ActuatorCalibrationError",
    "ActuatorCalibrationResult",
    "calibrate_actuator_from_directory",
    "fit_actuator_mapping",
]
