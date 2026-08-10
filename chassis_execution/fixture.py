"""Small deterministic fixtures for contract and downstream adapter tests."""

from __future__ import annotations

import torch

from .contracts import (
    CHASSIS_STATE_FIELDS,
    CONTROLLER_CONTEXT_FIELDS,
    CONTROL_FIELDS,
    ENSEMBLE_SIZE,
    EXECUTION_STEPS,
    INITIAL_STATE_FIELDS,
    NUM_GROUPS,
    NUM_ROLES,
    TRAJECTORY_DIM,
    VEHICLE_CONDITION_FIELDS,
    ChassisExecutionCommand,
    ChassisExecutionMemberPrediction,
    ChassisExecutionTarget,
)


def synthetic_command(batch_size: int = 2) -> ChassisExecutionCommand:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    times = torch.arange(1, 9, dtype=torch.float32) * 0.5
    tau_cmd = torch.zeros((batch_size, NUM_GROUPS, NUM_ROLES, 8, 3), dtype=torch.float32)
    tau_cmd[..., 0] = times
    initial = torch.zeros((batch_size, NUM_ROLES, len(INITIAL_STATE_FIELDS)), dtype=torch.float32)
    initial[..., 0] = 8.0
    conditions = torch.ones(
        (batch_size, NUM_ROLES, len(VEHICLE_CONDITION_FIELDS)), dtype=torch.float32
    )
    conditions[..., 0] = 12_000.0
    conditions[..., 1] = 2_000.0
    conditions[..., 2] = 1.4
    conditions[..., 3] = 5.7
    conditions[..., 4:6] = 2.1
    conditions[..., 6:8] = 80_000.0
    conditions[..., 8] = 300_000.0
    conditions[..., 9] = 30_000.0
    conditions[..., 10:12] = 0.8
    conditions[..., 12:] = 0.2
    context = torch.zeros(
        (batch_size, NUM_ROLES, len(CONTROLLER_CONTEXT_FIELDS)), dtype=torch.float32
    )
    context[..., 5] = 10.0
    context[..., 6] = 10.0
    controller_mode = torch.zeros((batch_size, NUM_ROLES), dtype=torch.int64)
    roles = torch.arange(NUM_ROLES, dtype=torch.int64).unsqueeze(0).expand(batch_size, -1).clone()
    return ChassisExecutionCommand(
        tau_cmd=tau_cmd,
        initial_state=initial,
        vehicle_condition=conditions,
        controller_context=context,
        controller_mode=controller_mode,
        agent_role=roles,
    )


def synthetic_member_predictions(
    command: ChassisExecutionCommand,
) -> ChassisExecutionMemberPrediction:
    batch_size = int(command.tau_cmd.shape[0])
    prefix = (ENSEMBLE_SIZE, batch_size, NUM_GROUPS, NUM_ROLES, EXECUTION_STEPS)
    member_offsets = torch.tensor([-0.05, 0.0, 0.05], dtype=torch.float32).reshape(
        ENSEMBLE_SIZE, 1, 1, 1, 1, 1
    )
    trajectory = torch.zeros(prefix + (TRAJECTORY_DIM,), dtype=torch.float32)
    trajectory[..., 0] = (
        torch.arange(1, EXECUTION_STEPS + 1, dtype=torch.float32) * 0.1
    )
    trajectory = trajectory + member_offsets
    chassis = torch.zeros(prefix + (len(CHASSIS_STATE_FIELDS),), dtype=torch.float32)
    chassis[..., 0] = 8.0
    chassis = chassis + member_offsets
    control = torch.zeros(prefix + (len(CONTROL_FIELDS),), dtype=torch.float32)
    control = control + member_offsets
    return ChassisExecutionMemberPrediction(
        trajectory_mean=trajectory,
        trajectory_log_variance=torch.full_like(trajectory, -4.0),
        chassis_mean=chassis,
        chassis_log_variance=torch.full_like(chassis, -3.0),
        control_mean=control,
        control_log_variance=torch.full_like(control, -2.0),
    )


def synthetic_target(sample_count: int = 6) -> ChassisExecutionTarget:
    """Return a complete 4-second target batch for dataset/verifier tests."""

    command = synthetic_command(batch_size=sample_count)
    execution_times = (
        torch.arange(1, EXECUTION_STEPS + 1, dtype=torch.float32) * 0.1
    )
    executed = torch.zeros(
        (sample_count, NUM_ROLES, EXECUTION_STEPS, TRAJECTORY_DIM),
        dtype=torch.float32,
    )
    executed[..., 0] = 8.0 * execution_times
    chassis = torch.zeros(
        (sample_count, NUM_ROLES, EXECUTION_STEPS, len(CHASSIS_STATE_FIELDS)),
        dtype=torch.float32,
    )
    chassis[..., 0] = 8.0
    control = torch.zeros(
        (sample_count, NUM_ROLES, EXECUTION_STEPS, len(CONTROL_FIELDS)),
        dtype=torch.float32,
    )
    return ChassisExecutionTarget(
        tau_cmd=command.tau_cmd[:, 0],
        initial_state=command.initial_state,
        vehicle_condition=command.vehicle_condition,
        controller_context=command.controller_context,
        controller_mode=command.controller_mode,
        agent_role=command.agent_role,
        executed_trajectory=executed,
        chassis_state=chassis,
        applied_control=control,
        state_valid_mask=torch.ones(
            (sample_count, NUM_ROLES, EXECUTION_STEPS), dtype=torch.bool
        ),
    )
