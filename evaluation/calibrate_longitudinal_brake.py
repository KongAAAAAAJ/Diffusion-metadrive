"""Calibrate the negative-throttle brake scale from physical step responses."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from evaluation.control_tracking_benchmark import (
    AGENT_IDS,
    DEFAULT_ROUTE,
    DEFAULT_SCENARIO,
    _ControlBenchmarkBranchEvaluator,
    _IndependentControlEnv,
)
from evaluation.joint_simulator_branch import JointEpisodeSpec
from expert_dataset.collect_joint_bev import simulator_decision_dt_s
from models.controller.longitudinal_reference import signed_longitudinal_speed_mps


class BrakeCalibrationError(RuntimeError):
    """Raised when fixed-command braking data cannot support calibration."""


@dataclass(frozen=True)
class BrakeScaleFit:
    sample_count: int
    slope_mps2_per_brake: float
    intercept_mps2: float
    correlation: float
    rmse_mps2: float
    through_origin_scale_mps2_per_brake: float

    @property
    def recommended_brake_acceleration_scale_mps2(self) -> float:
        # The controller already has an independently calibrated zero-throttle
        # bias, so its one-parameter brake inverse must use the through-origin
        # gain rather than the slope of a separate affine fit.
        return float(self.through_origin_scale_mps2_per_brake)

    def as_dict(self) -> dict[str, int | float]:
        return {
            "sample_count": self.sample_count,
            "slope_mps2_per_brake": self.slope_mps2_per_brake,
            "intercept_mps2": self.intercept_mps2,
            "correlation": self.correlation,
            "rmse_mps2": self.rmse_mps2,
            "through_origin_scale_mps2_per_brake": (
                self.through_origin_scale_mps2_per_brake
            ),
            "recommended_brake_acceleration_scale_mps2": (
                self.recommended_brake_acceleration_scale_mps2
            ),
        }


def fit_brake_scale(
    throttle: np.ndarray,
    acceleration_mps2: np.ndarray,
) -> BrakeScaleFit:
    """Fit ``a_real = scale * throttle + intercept`` on braking samples."""

    command = np.asarray(throttle, dtype=np.float64).reshape(-1)
    acceleration = np.asarray(acceleration_mps2, dtype=np.float64).reshape(-1)
    valid = (
        np.isfinite(command)
        & np.isfinite(acceleration)
        & (command < -1.0e-6)
        & (command >= -1.0)
        & (acceleration < 0.0)
    )
    command = command[valid]
    acceleration = acceleration[valid]
    if command.size < 20 or np.unique(np.round(command, 6)).size < 4:
        raise BrakeCalibrationError(
            "at least 20 samples across four negative commands are required"
        )
    slope, intercept = np.linalg.lstsq(
        np.column_stack((command, np.ones_like(command))),
        acceleration,
        rcond=None,
    )[0]
    prediction = slope * command + intercept
    correlation = float(np.corrcoef(command, acceleration)[0, 1])
    rmse = float(np.sqrt(np.mean((prediction - acceleration) ** 2)))
    through_origin_scale = float(
        np.dot(command, acceleration) / np.dot(command, command)
    )
    if not math.isfinite(slope) or slope <= 0.0 or correlation < 0.90:
        raise BrakeCalibrationError("brake response is not sufficiently linear")
    return BrakeScaleFit(
        sample_count=int(command.size),
        slope_mps2_per_brake=float(slope),
        intercept_mps2=float(intercept),
        correlation=correlation,
        rmse_mps2=rmse,
        through_origin_scale_mps2_per_brake=through_origin_scale,
    )


def _all_done(value: object) -> bool:
    if isinstance(value, Mapping):
        return bool(value.get("__all__", False))
    return bool(value)


def _natural_warmup(env: object, target_speed_mps: float) -> dict[str, object]:
    dt_s = float(simulator_decision_dt_s(env))
    stable_required = max(1, int(math.ceil(1.0 / dt_s)))
    stable = 0
    speed_history: list[list[float]] = []
    throttle_history: list[list[float]] = []
    for _ in range(max(1, int(math.ceil(20.0 / dt_s)))):
        speeds = [
            float(signed_longitudinal_speed_mps(env.agents[agent_id]))
            for agent_id in AGENT_IDS
        ]
        actions: dict[str, np.ndarray] = {}
        throttles = []
        for agent_id, speed in zip(AGENT_IDS, speeds):
            command = float(
                np.clip(0.05 + 0.22 * (target_speed_mps - speed), -0.25, 0.65)
            )
            actions[agent_id] = np.asarray([0.0, command], dtype=np.float32)
            throttles.append(command)
        speed_history.append(speeds)
        throttle_history.append(throttles)
        _, _, terminated, truncated, _ = env.step(actions)
        if _all_done(terminated) or _all_done(truncated):
            raise BrakeCalibrationError("warmup terminated before target speed")
        next_speeds = [
            float(signed_longitudinal_speed_mps(env.agents[agent_id]))
            for agent_id in AGENT_IDS
        ]
        stable = (
            stable + 1
            if max(abs(target_speed_mps - value) for value in next_speeds) <= 0.15
            else 0
        )
        if stable >= stable_required:
            return {
                "duration_s": len(speed_history) * dt_s,
                "speed_history_mps": speed_history,
                "throttle_history": throttle_history,
                "terminal_speed_mps": next_speeds,
            }
    raise BrakeCalibrationError("warmup did not stabilize within 20 seconds")


def run_brake_calibration(
    output_root: Path | str,
    *,
    commands: Sequence[float] = (-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.8, -1.0),
    initial_speed_mps: float = 8.0,
    pulse_duration_s: float = 0.4,
    seed: int = 17,
) -> dict[str, object]:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    command_values = tuple(float(value) for value in commands)
    if (
        len(command_values) < 4
        or any(not math.isfinite(value) or not -1.0 <= value < 0.0 for value in command_values)
        or initial_speed_mps <= 0.0
        or pulse_duration_s <= 0.0
    ):
        raise BrakeCalibrationError("invalid brake calibration configuration")

    evaluator = _ControlBenchmarkBranchEvaluator(env_factory=_IndependentControlEnv)
    records: list[dict[str, float | int]] = []
    warmups: list[dict[str, object]] = []
    for trial, command in enumerate(command_values):
        spec = JointEpisodeSpec(
            DEFAULT_SCENARIO,
            DEFAULT_ROUTE,
            int(seed),
            np.zeros((3, 3), dtype=np.float64),
            env_config={"initial_speed_km_h": 0.0},
        )
        env = evaluator._make_env(spec)
        try:
            warmups.append(_natural_warmup(env, float(initial_speed_mps)))
            dt_s = float(simulator_decision_dt_s(env))
            pulse_steps = max(1, int(round(float(pulse_duration_s) / dt_s)))
            for pulse_step in range(pulse_steps):
                before = [
                    float(signed_longitudinal_speed_mps(env.agents[agent_id]))
                    for agent_id in AGENT_IDS
                ]
                action = {
                    agent_id: np.asarray([0.0, command], dtype=np.float32)
                    for agent_id in AGENT_IDS
                }
                _, _, terminated, truncated, _ = env.step(action)
                if _all_done(terminated) or _all_done(truncated):
                    raise BrakeCalibrationError("brake pulse terminated the episode")
                after = [
                    float(signed_longitudinal_speed_mps(env.agents[agent_id]))
                    for agent_id in AGENT_IDS
                ]
                for role in range(3):
                    records.append(
                        {
                            "trial": trial,
                            "role": role,
                            "pulse_step": pulse_step,
                            "command": command,
                            "speed_before_mps": before[role],
                            "speed_after_mps": after[role],
                            "actual_acceleration_mps2": (
                                after[role] - before[role]
                            )
                            / dt_s,
                        }
                    )
        finally:
            env.close()

    command_array = np.asarray([row["command"] for row in records], dtype=np.float64)
    acceleration_array = np.asarray(
        [row["actual_acceleration_mps2"] for row in records], dtype=np.float64
    )
    fit = fit_brake_scale(command_array, acceleration_array)
    per_command = []
    for command in command_values:
        selected = acceleration_array[np.isclose(command_array, command)]
        per_command.append(
            {
                "command": command,
                "sample_count": int(selected.size),
                "acceleration_mean_mps2": float(np.mean(selected)),
                "acceleration_p50_mps2": float(np.median(selected)),
                "acceleration_min_mps2": float(np.min(selected)),
                "acceleration_max_mps2": float(np.max(selected)),
            }
        )
    report: dict[str, object] = {
        "format": "bev_longitudinal_brake_calibration_v1",
        "scenario": DEFAULT_SCENARIO,
        "route": DEFAULT_ROUTE,
        "seed": int(seed),
        "initial_speed_mps": float(initial_speed_mps),
        "pulse_duration_s": float(pulse_duration_s),
        "commands": list(command_values),
        "fit": fit.as_dict(),
        "per_command": per_command,
        "warmup_duration_s": [float(row["duration_s"]) for row in warmups],
    }
    (output / "brake_calibration.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        output / "brake_step_responses.npz",
        command=command_array,
        acceleration_mps2=acceleration_array,
        speed_before_mps=np.asarray(
            [row["speed_before_mps"] for row in records], dtype=np.float64
        ),
        speed_after_mps=np.asarray(
            [row["speed_after_mps"] for row in records], dtype=np.float64
        ),
        role=np.asarray([row["role"] for row in records], dtype=np.int64),
        pulse_step=np.asarray(
            [row["pulse_step"] for row in records], dtype=np.int64
        ),
    )

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(command_array, acceleration_array, alpha=0.5, s=22)
    x = np.linspace(min(command_values), 0.0, 200)
    axis.plot(
        x,
        fit.slope_mps2_per_brake * x + fit.intercept_mps2,
        color="tab:red",
        label="affine brake fit",
    )
    axis.set_xlabel("negative throttle / brake command")
    axis.set_ylabel("actual acceleration (m/s²)")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output / "brake_scale_fit.png", dpi=160)
    plt.close(figure)
    return report


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--initial-speed-mps", type=float, default=8.0)
    parser.add_argument("--pulse-duration-s", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--commands", nargs="+", type=float, default=None)
    args = parser.parse_args(argv)
    report = run_brake_calibration(
        args.output_root,
        commands=(
            tuple(args.commands)
            if args.commands is not None
            else (-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.8, -1.0)
        ),
        initial_speed_mps=args.initial_speed_mps,
        pulse_duration_s=args.pulse_duration_s,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "BrakeCalibrationError",
    "BrakeScaleFit",
    "fit_brake_scale",
    "run_brake_calibration",
]
