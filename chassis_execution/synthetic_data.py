"""Deterministic virtual chassis data for diagnostic-only CF pipeline bring-up."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .contracts import (
    CHASSIS_STATE_FIELDS,
    CONTROLLER_CONTEXT_FIELDS,
    CONTROL_FIELDS,
    INITIAL_STATE_FIELDS,
    VEHICLE_CONDITION_FIELDS,
    ChassisExecutionTarget,
)
from .dataset import ChassisExecutionDatasetMetadata, canonical_sha256
from .storage import (
    ChassisExecutionSampleIdentity,
    ChassisExecutionStorageProvenance,
    verify_chassis_execution_dataset,
    write_chassis_execution_dataset,
)


MANEUVERS = (
    "constant_speed",
    "accelerate",
    "brake",
    "stop",
    "lane_change_left",
    "lane_change_right",
    "brake_and_steer",
    "formation_gap_recovery",
)


@dataclass(frozen=True)
class SyntheticChassisConfig:
    sample_count: int = 2048
    group_size: int = 4
    seed: int = 17
    horizon_s: float = 4.0
    command_dt_s: float = 0.5
    execution_dt_s: float = 0.1
    longitudinal_time_constant_s: float = 0.55
    heading_time_constant_s: float = 0.35
    roll_time_constant_s: float = 0.45
    split_salt: str = "chassis-synthetic-bootstrap-v1"

    def __post_init__(self) -> None:
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.sample_count <= 0 or self.sample_count % self.group_size != 0:
            raise ValueError("sample_count must be positive and divisible by group_size")
        if (self.horizon_s, self.command_dt_s, self.execution_dt_s) != (4.0, 0.5, 0.1):
            raise ValueError("synthetic timing must match the frozen 4.0/0.5/0.1 contract")
        if min(
            self.longitudinal_time_constant_s,
            self.heading_time_constant_s,
            self.roll_time_constant_s,
        ) <= 0.0:
            raise ValueError("synthetic response time constants must be positive")

    def sha256(self) -> str:
        return canonical_sha256(asdict(self))


def _label_sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _smoothstep5(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return 10.0 * clipped**3 - 15.0 * clipped**4 + 6.0 * clipped**5


def _wrap(value: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(value), np.cos(value))


def _command_for_role(
    maneuver: str,
    *,
    role: int,
    initial_speed: float,
    variant_scale: float,
) -> np.ndarray:
    times = np.arange(1, 9, dtype=np.float64) * 0.5
    acceleration = 0.0
    if maneuver == "accelerate":
        acceleration = 1.4 * variant_scale
    elif maneuver == "brake":
        acceleration = -2.5 * variant_scale
    elif maneuver == "stop":
        acceleration = -max(2.0, initial_speed / 2.5)
    elif maneuver == "brake_and_steer":
        acceleration = -2.0 * variant_scale
    elif maneuver == "formation_gap_recovery":
        acceleration = (0.3 + 0.35 * role) * variant_scale
    if maneuver == "stop":
        stop_time = initial_speed / max(-acceleration, 1e-6)
        moving_time = np.minimum(times, stop_time)
        x = initial_speed * moving_time + 0.5 * acceleration * moving_time**2
    else:
        x = initial_speed * times + 0.5 * acceleration * times**2
        x = np.maximum.accumulate(np.maximum(x, 0.0))
    lateral_target = 0.0
    if maneuver == "lane_change_left":
        lateral_target = 3.5
    elif maneuver == "lane_change_right":
        lateral_target = -3.5
    elif maneuver == "brake_and_steer":
        lateral_target = 3.2 if role != 2 else -3.2
    y = lateral_target * _smoothstep5(times / 3.2)
    x_with_origin = np.concatenate(([0.0], x))
    y_with_origin = np.concatenate(([0.0], y))
    heading = np.arctan2(np.diff(y_with_origin), np.maximum(np.diff(x_with_origin), 1e-6))
    if maneuver == "stop":
        last_moving = 0.0
        for index in range(heading.size):
            if x_with_origin[index + 1] - x_with_origin[index] > 1e-5:
                last_moving = float(heading[index])
            else:
                heading[index] = last_moving
    return np.stack((x, y, heading), axis=-1).astype(np.float32)


def _simulate_execution(
    command: np.ndarray,
    initial_speed: float,
    condition: np.ndarray,
    config: SyntheticChassisConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = config.execution_dt_s
    times = np.arange(1, 41, dtype=np.float64) * dt
    command_times = np.arange(1, 9, dtype=np.float64) * 0.5
    reference = np.empty((40, 3), dtype=np.float64)
    reference[:, 0] = np.interp(times, np.r_[0.0, command_times], np.r_[0.0, command[:, 0]])
    reference[:, 1] = np.interp(times, np.r_[0.0, command_times], np.r_[0.0, command[:, 1]])
    heading_unwrapped = np.unwrap(np.r_[0.0, command[:, 2]])
    reference[:, 2] = np.interp(times, np.r_[0.0, command_times], heading_unwrapped)
    reference_speed = np.diff(np.r_[0.0, reference[:, 0]]) / dt
    mass_scale = float(condition[0] / 12_000.0)
    friction = float(min(condition[10], condition[11]))
    drive_delay = float(condition[12])
    brake_delay = float(condition[13])
    x = y = heading = roll = 0.0
    speed = float(initial_speed)
    previous_roll = 0.0
    executed = np.zeros((40, 3), dtype=np.float32)
    chassis = np.zeros((40, len(CHASSIS_STATE_FIELDS)), dtype=np.float32)
    control = np.zeros((40, len(CONTROL_FIELDS)), dtype=np.float32)
    for step in range(40):
        speed_error_now = reference_speed[step] - speed
        delay = drive_delay if speed_error_now >= 0.0 else brake_delay
        delayed_step = max(0, step - int(round(delay / dt)))
        speed_reference = max(0.0, float(reference_speed[delayed_step]))
        acceleration_command = np.clip(
            (speed_reference - speed) / (config.longitudinal_time_constant_s * mass_scale),
            -6.0 * friction,
            3.0 * friction,
        )
        speed = max(0.0, speed + float(acceleration_command) * dt)
        heading_reference = float(reference[delayed_step, 2])
        yaw_rate = np.clip(
            float(_wrap(heading_reference - heading)) / config.heading_time_constant_s,
            -0.8 * friction,
            0.8 * friction,
        )
        heading = float(_wrap(heading + yaw_rate * dt))
        x += speed * np.cos(heading) * dt
        y += speed * np.sin(heading) * dt
        lateral_acceleration = speed * yaw_rate
        track = max(float(0.5 * (condition[4] + condition[5])), 1e-3)
        roll_target = 0.18 * lateral_acceleration * float(condition[2]) / (9.81 * track)
        roll += (roll_target - roll) * dt / config.roll_time_constant_s
        roll_rate = (roll - previous_roll) / dt
        previous_roll = roll
        rollover = np.clip(
            0.30 * lateral_acceleration * float(condition[2]) / (9.81 * (track / 2.0)),
            -1.0,
            1.0,
        )
        lateral_speed = 0.08 * (float(reference[step, 1]) - y)
        road_wheel_angle = np.arctan(float(condition[3]) * yaw_rate / max(speed, 0.5))
        throttle = np.clip(acceleration_command / 3.0, 0.0, 1.0)
        brake = np.clip(-acceleration_command / 6.0, 0.0, 1.0)
        executed[step] = (x, y, heading)
        chassis[step] = (
            speed,
            lateral_speed,
            acceleration_command,
            lateral_acceleration,
            yaw_rate,
            roll,
            roll_rate,
            rollover,
        )
        control[step] = (road_wheel_angle, throttle, brake)
    return executed, chassis, control


def generate_synthetic_chassis_batch(
    config: SyntheticChassisConfig,
) -> tuple[
    ChassisExecutionTarget,
    list[ChassisExecutionSampleIdentity],
    ChassisExecutionDatasetMetadata,
    ChassisExecutionStorageProvenance,
]:
    """Generate deterministic, non-TruckSim diagnostic data matching CF-1."""

    rng = np.random.default_rng(config.seed)
    count = config.sample_count
    tau_cmd = np.zeros((count, 3, 8, 3), dtype=np.float32)
    initial = np.zeros((count, 3, len(INITIAL_STATE_FIELDS)), dtype=np.float32)
    conditions = np.zeros((count, 3, len(VEHICLE_CONDITION_FIELDS)), dtype=np.float32)
    context = np.zeros((count, 3, len(CONTROLLER_CONTEXT_FIELDS)), dtype=np.float32)
    modes = np.zeros((count, 3), dtype=np.int64)
    roles = np.broadcast_to(np.arange(3, dtype=np.int64), (count, 3)).copy()
    executed = np.zeros((count, 3, 40, 3), dtype=np.float32)
    chassis = np.zeros((count, 3, 40, len(CHASSIS_STATE_FIELDS)), dtype=np.float32)
    controls = np.zeros((count, 3, 40, len(CONTROL_FIELDS)), dtype=np.float32)
    identities: list[ChassisExecutionSampleIdentity] = []
    for sample in range(count):
        group = sample // config.group_size
        variant = sample % config.group_size
        maneuver = MANEUVERS[sample % len(MANEUVERS)]
        variant_scale = 0.85 + 0.10 * variant
        for role in range(3):
            initial_speed = float(rng.uniform(5.0, 14.0) + 0.25 * (2 - role))
            condition = np.asarray(
                [
                    rng.uniform(9_000.0, 16_000.0),
                    rng.uniform(0.0, 6_000.0),
                    rng.uniform(1.0, 2.1),
                    rng.uniform(4.8, 7.0),
                    rng.uniform(1.8, 2.6),
                    rng.uniform(1.8, 2.6),
                    rng.uniform(50_000.0, 140_000.0),
                    rng.uniform(50_000.0, 140_000.0),
                    rng.uniform(180_000.0, 700_000.0),
                    rng.uniform(15_000.0, 80_000.0),
                    rng.uniform(0.45, 1.0),
                    rng.uniform(0.45, 1.0),
                    rng.uniform(0.05, 0.30),
                    rng.uniform(0.05, 0.35),
                ],
                dtype=np.float32,
            )
            command = _command_for_role(
                maneuver,
                role=role,
                initial_speed=initial_speed,
                variant_scale=variant_scale,
            )
            tau_cmd[sample, role] = command
            initial[sample, role, 0] = initial_speed
            initial[sample, role, 9] = initial_speed / 0.5
            conditions[sample, role] = condition
            context[sample, role, 5] = 14.0 + 2.0 * role
            context[sample, role, 6] = 15.0
            context[sample, role, 7] = 0.25 * (role - 1)
            result = _simulate_execution(command, initial_speed, condition, config)
            executed[sample, role], chassis[sample, role], controls[sample, role] = result
        if maneuver == "formation_gap_recovery":
            modes[sample] = 1
        identities.append(
            ChassisExecutionSampleIdentity(
                run_id=f"synthetic_run_{sample:06d}",
                run_group_id=f"synthetic_group_{group:06d}",
                maneuver=maneuver,
            )
        )
    target = ChassisExecutionTarget(
        tau_cmd=torch.from_numpy(tau_cmd),
        initial_state=torch.from_numpy(initial),
        vehicle_condition=torch.from_numpy(conditions),
        controller_context=torch.from_numpy(context),
        controller_mode=torch.from_numpy(modes),
        agent_role=torch.from_numpy(roles),
        executed_trajectory=torch.from_numpy(executed),
        chassis_state=torch.from_numpy(chassis),
        applied_control=torch.from_numpy(controls),
        state_valid_mask=torch.ones((count, 3, 40), dtype=torch.bool),
    )
    ranges = {
        "total_mass_kg": (8_000.0, 18_000.0),
        "payload_mass_kg": (0.0, 8_000.0),
        "cg_height_m": (0.8, 2.5),
        "wheelbase_m": (4.0, 8.0),
        "front_track_m": (1.5, 3.0),
        "rear_track_m": (1.5, 3.0),
        "front_cornering_stiffness_n_per_rad": (20_000.0, 200_000.0),
        "rear_cornering_stiffness_n_per_rad": (20_000.0, 200_000.0),
        "roll_stiffness_nm_per_rad": (100_000.0, 1_000_000.0),
        "roll_damping_nms_per_rad": (10_000.0, 100_000.0),
        "tire_friction_coefficient": (0.2, 1.2),
        "road_friction_coefficient": (0.2, 1.2),
        "drive_actuator_delay_s": (0.001, 1.0),
        "brake_actuator_delay_s": (0.001, 1.0),
    }
    metadata = ChassisExecutionDatasetMetadata(
        format="chassis_execution_dataset_v1",
        schema_version=1,
        controller_contract_sha256=_label_sha256("synthetic_virtual_controller_v1"),
        trajectory_optimizer_sha256=_label_sha256("synthetic_virtual_tau_cmd_generator_v1"),
        trucksim_project_sha256=_label_sha256("not_trucksim_synthetic_virtual_dynamics_v1"),
        signal_mapping_sha256=_label_sha256("synthetic_virtual_direct_si_mapping_v1"),
        split_salt=config.split_salt,
        command_dt_s=0.5,
        execution_dt_s=0.1,
        horizon_s=4.0,
        coordinate_frame="per_role_current_ego_local",
        units="SI",
        vehicle_parameter_ranges=ranges,
    )
    provenance = ChassisExecutionStorageProvenance(
        data_origin="synthetic_virtual",
        diagnostic_only=True,
        eligible_for_formal_training=False,
        cf2_real_windows_smoke_passed=False,
        diagnostic_override_id="cf2_real_smoke_diagnostic_override_v1",
        generator_config_sha256=config.sha256(),
    )
    return target, identities, metadata, provenance


def generate_synthetic_dataset(root: Path | str, config: SyntheticChassisConfig) -> dict[str, object]:
    target, identities, metadata, provenance = generate_synthetic_chassis_batch(config)
    write_chassis_execution_dataset(
        root,
        target=target,
        identities=identities,
        metadata=metadata,
        provenance=provenance,
    )
    return verify_chassis_execution_dataset(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    config = SyntheticChassisConfig(sample_count=args.samples, seed=args.seed)
    report = generate_synthetic_dataset(args.output, config)
    report = {**report, "synthetic_generator_config": asdict(config)}
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
