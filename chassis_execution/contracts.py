"""Strict machine interfaces for the frozen ``tau_cmd -> tau_a`` boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


NUM_ROLES = 3
NUM_GROUPS = 4
COMMAND_STEPS = 8
EXECUTION_STEPS = 40
TRAJECTORY_DIM = 3
ENSEMBLE_SIZE = 3

COMMAND_TIMES_S = tuple(0.5 * index for index in range(1, COMMAND_STEPS + 1))
EXECUTION_TIMES_S = tuple(0.1 * index for index in range(1, EXECUTION_STEPS + 1))

INITIAL_STATE_FIELDS = (
    "longitudinal_speed_mps",
    "lateral_speed_mps",
    "longitudinal_acceleration_mps2",
    "lateral_acceleration_mps2",
    "yaw_rate_rad_s",
    "roll_rad",
    "roll_rate_rad_s",
    "road_wheel_angle_rad",
    "signed_longitudinal_control",
    "mean_wheel_speed_rad_s",
)

VEHICLE_CONDITION_FIELDS = (
    "total_mass_kg",
    "payload_mass_kg",
    "cg_height_m",
    "wheelbase_m",
    "front_track_m",
    "rear_track_m",
    "front_cornering_stiffness_n_per_rad",
    "rear_cornering_stiffness_n_per_rad",
    "roll_stiffness_nm_per_rad",
    "roll_damping_nms_per_rad",
    "tire_friction_coefficient",
    "road_friction_coefficient",
    "drive_actuator_delay_s",
    "brake_actuator_delay_s",
)

CONTROLLER_CONTEXT_FIELDS = (
    "speed_integral_error",
    "previous_lateral_error_rad",
    "lateral_integral_error_rad_s",
    "previous_heading_error_rad",
    "longitudinal_arc_error_m",
    "actual_predecessor_gap_m",
    "desired_predecessor_gap_m",
    "predecessor_relative_speed_mps",
)

CHASSIS_STATE_FIELDS = (
    "longitudinal_speed_mps",
    "lateral_speed_mps",
    "longitudinal_acceleration_mps2",
    "lateral_acceleration_mps2",
    "yaw_rate_rad_s",
    "roll_rad",
    "roll_rate_rad_s",
    "rollover_index",
)

CONTROL_FIELDS = (
    "road_wheel_angle_rad",
    "throttle_normalized",
    "brake_normalized",
)


class ChassisExecutionContractError(ValueError):
    """Raised when a chassis-execution tensor violates the frozen contract."""


def _require_tensor(
    value: object,
    *,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int | None, ...],
    finite: bool = True,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise ChassisExecutionContractError(f"{name} must be a torch.Tensor")
    if value.dtype != dtype:
        raise ChassisExecutionContractError(
            f"{name} must have dtype {dtype}, got {value.dtype}"
        )
    if value.ndim != len(shape) or any(
        expected is not None and int(actual) != expected
        for actual, expected in zip(value.shape, shape)
    ):
        raise ChassisExecutionContractError(
            f"{name} shape {tuple(value.shape)} does not match {shape}"
        )
    if finite and value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ChassisExecutionContractError(f"{name} must contain only finite values")
    return value


def _validate_roles(agent_role: Tensor, batch_size: int) -> None:
    _require_tensor(
        agent_role,
        name="agent_role",
        dtype=torch.int64,
        shape=(batch_size, NUM_ROLES),
        finite=False,
    )
    expected = torch.arange(NUM_ROLES, dtype=torch.int64, device=agent_role.device)
    if not bool(torch.equal(agent_role, expected.unsqueeze(0).expand(batch_size, -1))):
        raise ChassisExecutionContractError(
            "agent_role must be [0,1,2] for every joint sample"
        )


@dataclass(frozen=True)
class ChassisExecutionCommand:
    """A batch of optimized commands evaluated by the frozen surrogate.

    Context tensors do not contain a group dimension because every GRPO group
    starts from the same physical online state. The surrogate expands that
    context over the four commanded candidates internally.
    """

    tau_cmd: Tensor
    initial_state: Tensor
    vehicle_condition: Tensor
    controller_context: Tensor
    controller_mode: Tensor
    agent_role: Tensor
    source: str = "tau_cmd"

    def __post_init__(self) -> None:
        command = _require_tensor(
            self.tau_cmd,
            name="tau_cmd",
            dtype=torch.float32,
            shape=(None, NUM_GROUPS, NUM_ROLES, COMMAND_STEPS, TRAJECTORY_DIM),
        )
        batch_size = int(command.shape[0])
        _require_tensor(
            self.initial_state,
            name="initial_state",
            dtype=torch.float32,
            shape=(batch_size, NUM_ROLES, len(INITIAL_STATE_FIELDS)),
        )
        conditions = _require_tensor(
            self.vehicle_condition,
            name="vehicle_condition",
            dtype=torch.float32,
            shape=(batch_size, NUM_ROLES, len(VEHICLE_CONDITION_FIELDS)),
        )
        _require_tensor(
            self.controller_context,
            name="controller_context",
            dtype=torch.float32,
            shape=(batch_size, NUM_ROLES, len(CONTROLLER_CONTEXT_FIELDS)),
        )
        modes = _require_tensor(
            self.controller_mode,
            name="controller_mode",
            dtype=torch.int64,
            shape=(batch_size, NUM_ROLES),
            finite=False,
        )
        _validate_roles(self.agent_role, batch_size)
        if str(self.source) != "tau_cmd":
            raise ChassisExecutionContractError(
                "surrogate input source must be tau_cmd, never raw tau_d"
            )
        if not bool(((modes == 0) | (modes == 1)).all()):
            raise ChassisExecutionContractError(
                "controller_mode must use 0=independent or 1=formation_locked"
            )
        positive_indices = (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
        if not bool((conditions[..., positive_indices] > 0.0).all()):
            raise ChassisExecutionContractError(
                "physical vehicle conditions and friction values must be positive"
            )
        if not bool((conditions[..., 1] >= 0.0).all()):
            raise ChassisExecutionContractError("payload_mass_kg must be non-negative")
        if not bool((conditions[..., 12:] >= 0.0).all()):
            raise ChassisExecutionContractError("actuator delays must be non-negative")


@dataclass(frozen=True)
class ChassisExecutionTarget:
    """One fixed-window TruckSim training batch without a GRPO group axis."""

    tau_cmd: Tensor
    initial_state: Tensor
    vehicle_condition: Tensor
    controller_context: Tensor
    controller_mode: Tensor
    agent_role: Tensor
    executed_trajectory: Tensor
    chassis_state: Tensor
    applied_control: Tensor
    state_valid_mask: Tensor

    def __post_init__(self) -> None:
        command = _require_tensor(
            self.tau_cmd,
            name="target.tau_cmd",
            dtype=torch.float32,
            shape=(None, NUM_ROLES, COMMAND_STEPS, TRAJECTORY_DIM),
        )
        count = int(command.shape[0])
        _require_tensor(
            self.initial_state,
            name="target.initial_state",
            dtype=torch.float32,
            shape=(count, NUM_ROLES, len(INITIAL_STATE_FIELDS)),
        )
        _require_tensor(
            self.vehicle_condition,
            name="target.vehicle_condition",
            dtype=torch.float32,
            shape=(count, NUM_ROLES, len(VEHICLE_CONDITION_FIELDS)),
        )
        _require_tensor(
            self.controller_context,
            name="target.controller_context",
            dtype=torch.float32,
            shape=(count, NUM_ROLES, len(CONTROLLER_CONTEXT_FIELDS)),
        )
        _require_tensor(
            self.controller_mode,
            name="target.controller_mode",
            dtype=torch.int64,
            shape=(count, NUM_ROLES),
            finite=False,
        )
        _validate_roles(self.agent_role, count)
        _require_tensor(
            self.executed_trajectory,
            name="target.executed_trajectory",
            dtype=torch.float32,
            shape=(count, NUM_ROLES, EXECUTION_STEPS, TRAJECTORY_DIM),
        )
        _require_tensor(
            self.chassis_state,
            name="target.chassis_state",
            dtype=torch.float32,
            shape=(count, NUM_ROLES, EXECUTION_STEPS, len(CHASSIS_STATE_FIELDS)),
        )
        _require_tensor(
            self.applied_control,
            name="target.applied_control",
            dtype=torch.float32,
            shape=(count, NUM_ROLES, EXECUTION_STEPS, len(CONTROL_FIELDS)),
        )
        mask = _require_tensor(
            self.state_valid_mask,
            name="target.state_valid_mask",
            dtype=torch.bool,
            shape=(count, NUM_ROLES, EXECUTION_STEPS),
            finite=False,
        )
        if not bool(mask.all()):
            raise ChassisExecutionContractError(
                "CF-1 fixed-window targets require all 4 seconds to be valid"
            )


@dataclass(frozen=True)
class ChassisExecutionMemberPrediction:
    """Raw probabilistic outputs from exactly three independently trained members."""

    trajectory_mean: Tensor
    trajectory_log_variance: Tensor
    chassis_mean: Tensor
    chassis_log_variance: Tensor
    control_mean: Tensor
    control_log_variance: Tensor

    def __post_init__(self) -> None:
        trajectory = _require_tensor(
            self.trajectory_mean,
            name="trajectory_mean",
            dtype=torch.float32,
            shape=(
                ENSEMBLE_SIZE,
                None,
                NUM_GROUPS,
                NUM_ROLES,
                EXECUTION_STEPS,
                TRAJECTORY_DIM,
            ),
        )
        common = tuple(trajectory.shape[:5])
        _require_tensor(
            self.trajectory_log_variance,
            name="trajectory_log_variance",
            dtype=torch.float32,
            shape=common + (TRAJECTORY_DIM,),
        )
        _require_tensor(
            self.chassis_mean,
            name="chassis_mean",
            dtype=torch.float32,
            shape=common + (len(CHASSIS_STATE_FIELDS),),
        )
        _require_tensor(
            self.chassis_log_variance,
            name="chassis_log_variance",
            dtype=torch.float32,
            shape=common + (len(CHASSIS_STATE_FIELDS),),
        )
        _require_tensor(
            self.control_mean,
            name="control_mean",
            dtype=torch.float32,
            shape=common + (len(CONTROL_FIELDS),),
        )
        _require_tensor(
            self.control_log_variance,
            name="control_log_variance",
            dtype=torch.float32,
            shape=common + (len(CONTROL_FIELDS),),
        )


@dataclass(frozen=True)
class ChassisExecutionPrediction:
    executed_trajectory_mean: Tensor
    chassis_state_mean: Tensor
    control_mean: Tensor
    trajectory_aleatoric_variance: Tensor
    trajectory_epistemic_variance: Tensor
    trajectory_total_variance: Tensor
    chassis_aleatoric_variance: Tensor
    chassis_epistemic_variance: Tensor
    chassis_total_variance: Tensor
    control_aleatoric_variance: Tensor
    control_epistemic_variance: Tensor
    control_total_variance: Tensor

    def __post_init__(self) -> None:
        trajectory_mean = _require_tensor(
            self.executed_trajectory_mean,
            name="executed_trajectory_mean",
            dtype=torch.float32,
            shape=(None, NUM_GROUPS, NUM_ROLES, EXECUTION_STEPS, TRAJECTORY_DIM),
        )
        batch_size = int(trajectory_mean.shape[0])
        trajectory_shape = (
            batch_size,
            NUM_GROUPS,
            NUM_ROLES,
            EXECUTION_STEPS,
            TRAJECTORY_DIM,
        )
        chassis_shape = (
            batch_size,
            NUM_GROUPS,
            NUM_ROLES,
            EXECUTION_STEPS,
            len(CHASSIS_STATE_FIELDS),
        )
        control_shape = (
            batch_size,
            NUM_GROUPS,
            NUM_ROLES,
            EXECUTION_STEPS,
            len(CONTROL_FIELDS),
        )
        groups = (
            ("chassis_state_mean", chassis_shape, False),
            ("control_mean", control_shape, False),
            ("trajectory_aleatoric_variance", trajectory_shape, True),
            ("trajectory_epistemic_variance", trajectory_shape, True),
            ("trajectory_total_variance", trajectory_shape, True),
            ("chassis_aleatoric_variance", chassis_shape, True),
            ("chassis_epistemic_variance", chassis_shape, True),
            ("chassis_total_variance", chassis_shape, True),
            ("control_aleatoric_variance", control_shape, True),
            ("control_epistemic_variance", control_shape, True),
            ("control_total_variance", control_shape, True),
        )
        for name, shape, is_variance in groups:
            value = _require_tensor(
                getattr(self, name), name=name, dtype=torch.float32, shape=shape
            )
            if is_variance and not bool((value >= 0.0).all()):
                raise ChassisExecutionContractError(f"{name} must be non-negative")
        for prefix in ("trajectory", "chassis", "control"):
            aleatoric = getattr(self, f"{prefix}_aleatoric_variance")
            epistemic = getattr(self, f"{prefix}_epistemic_variance")
            total = getattr(self, f"{prefix}_total_variance")
            if not bool(torch.allclose(total, aleatoric + epistemic)):
                raise ChassisExecutionContractError(
                    f"{prefix} total variance must equal aleatoric + epistemic"
                )


def _aggregate(mean: Tensor, log_variance: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    aggregate_mean = mean.mean(dim=0)
    aleatoric = torch.exp(log_variance).mean(dim=0)
    epistemic = mean.var(dim=0, unbiased=False)
    return aggregate_mean, aleatoric, epistemic, aleatoric + epistemic


def aggregate_ensemble_predictions(
    members: ChassisExecutionMemberPrediction,
) -> ChassisExecutionPrediction:
    """Apply the law of total variance to the three member predictions."""

    trajectory = _aggregate(
        members.trajectory_mean, members.trajectory_log_variance
    )
    chassis = _aggregate(members.chassis_mean, members.chassis_log_variance)
    control = _aggregate(members.control_mean, members.control_log_variance)
    return ChassisExecutionPrediction(
        executed_trajectory_mean=trajectory[0],
        chassis_state_mean=chassis[0],
        control_mean=control[0],
        trajectory_aleatoric_variance=trajectory[1],
        trajectory_epistemic_variance=trajectory[2],
        trajectory_total_variance=trajectory[3],
        chassis_aleatoric_variance=chassis[1],
        chassis_epistemic_variance=chassis[2],
        chassis_total_variance=chassis[3],
        control_aleatoric_variance=control[1],
        control_epistemic_variance=control[2],
        control_total_variance=control[3],
    )


@runtime_checkable
class ChassisExecutionSurrogate(Protocol):
    """Interface implemented by the frozen three-member ensemble in CF-4."""

    def predict(self, command: ChassisExecutionCommand) -> ChassisExecutionPrediction:
        ...
