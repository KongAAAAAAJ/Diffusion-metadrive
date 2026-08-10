from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from chassis_execution import (
    ChassisExecutionCommand,
    ChassisExecutionContractError,
    ChassisExecutionDatasetError,
    ChassisExecutionDatasetMetadata,
    ChassisExecutionPrediction,
    aggregate_ensemble_predictions,
    assign_run_group_split,
    canonical_sha256,
    validate_atomic_run_group_splits,
)
from chassis_execution.fixture import synthetic_command, synthetic_member_predictions
from chassis_execution.fixture import synthetic_target


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "schemas" / "chassis_execution_surrogate_v1.json"


def _metadata() -> ChassisExecutionDatasetMetadata:
    return ChassisExecutionDatasetMetadata(
        format="chassis_execution_dataset_v1",
        schema_version=1,
        controller_contract_sha256="1" * 64,
        trajectory_optimizer_sha256="2" * 64,
        trucksim_project_sha256="3" * 64,
        signal_mapping_sha256="4" * 64,
        split_salt="fixture-v1",
        command_dt_s=0.5,
        execution_dt_s=0.1,
        horizon_s=4.0,
        coordinate_frame="per_role_current_ego_local",
        units="SI",
        vehicle_parameter_ranges={"total_mass_kg": (8_000.0, 18_000.0)},
    )


def test_protocol_json_is_frozen_and_canonical() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert payload["policy_boundary"]["tau_cmd"]["surrogate_input"] is True
    assert payload["policy_boundary"]["tau_a"]["shape"] == ["B", 4, 3, 40, 3]
    assert payload["uncertainty"]["ensemble_size"] == 3
    assert payload["online_grpo"]["metadrive_candidate_branches"] == 0
    assert payload["online_grpo"]["metadrive_steps_after_selection"] == 1
    assert canonical_sha256(payload) == canonical_sha256(
        json.loads(PROTOCOL.read_text(encoding="utf-8"))
    )


def test_valid_command_and_three_member_aggregation() -> None:
    command = synthetic_command()
    members = synthetic_member_predictions(command)
    result = aggregate_ensemble_predictions(members)
    assert result.executed_trajectory_mean.shape == (2, 4, 3, 40, 3)
    assert result.chassis_state_mean.shape == (2, 4, 3, 40, 8)
    assert torch.all(result.trajectory_total_variance >= 0.0)
    assert torch.allclose(
        result.trajectory_total_variance,
        result.trajectory_aleatoric_variance
        + result.trajectory_epistemic_variance,
    )


def test_complete_training_target_fixture() -> None:
    target = synthetic_target()
    assert target.tau_cmd.shape == (6, 3, 8, 3)
    assert target.executed_trajectory.shape == (6, 3, 40, 3)
    assert target.chassis_state.shape == (6, 3, 40, 8)
    assert target.applied_control.shape == (6, 3, 40, 3)
    assert bool(target.state_valid_mask.all())


def test_ensemble_aggregation_is_member_order_invariant() -> None:
    command = synthetic_command(batch_size=1)
    members = synthetic_member_predictions(command)
    first = aggregate_ensemble_predictions(members)
    indices = torch.tensor([2, 0, 1], dtype=torch.int64)
    shuffled = type(members)(
        **{
            name: getattr(members, name).index_select(0, indices)
            for name in members.__dataclass_fields__
        }
    )
    second = aggregate_ensemble_predictions(shuffled)
    for name in first.__dataclass_fields__:
        assert torch.allclose(getattr(first, name), getattr(second, name))


@pytest.mark.parametrize("source", ["tau_d", "optimized", "raw_diffusion"])
def test_raw_policy_trajectory_cannot_enter_surrogate(source: str) -> None:
    valid = synthetic_command(batch_size=1)
    with pytest.raises(ChassisExecutionContractError, match="tau_cmd"):
        ChassisExecutionCommand(
            tau_cmd=valid.tau_cmd,
            initial_state=valid.initial_state,
            vehicle_condition=valid.vehicle_condition,
            controller_context=valid.controller_context,
            controller_mode=valid.controller_mode,
            agent_role=valid.agent_role,
            source=source,
        )


def test_wrong_command_or_output_horizon_is_rejected() -> None:
    valid = synthetic_command(batch_size=1)
    with pytest.raises(ChassisExecutionContractError, match="tau_cmd shape"):
        ChassisExecutionCommand(
            tau_cmd=valid.tau_cmd[..., :7, :],
            initial_state=valid.initial_state,
            vehicle_condition=valid.vehicle_condition,
            controller_context=valid.controller_context,
            controller_mode=valid.controller_mode,
            agent_role=valid.agent_role,
        )
    members = synthetic_member_predictions(valid)
    with pytest.raises(ChassisExecutionContractError, match="trajectory_mean shape"):
        type(members)(
            trajectory_mean=members.trajectory_mean[..., :8, :],
            trajectory_log_variance=members.trajectory_log_variance[..., :8, :],
            chassis_mean=members.chassis_mean[..., :8, :],
            chassis_log_variance=members.chassis_log_variance[..., :8, :],
            control_mean=members.control_mean[..., :8, :],
            control_log_variance=members.control_log_variance[..., :8, :],
        )


def test_wrong_ensemble_size_is_rejected() -> None:
    members = synthetic_member_predictions(synthetic_command(batch_size=1))
    with pytest.raises(ChassisExecutionContractError, match="trajectory_mean shape"):
        type(members)(
            **{
                name: getattr(members, name)[:2]
                for name in members.__dataclass_fields__
            }
        )


def test_missing_roll_signal_and_nonfinite_state_are_rejected() -> None:
    valid = synthetic_command(batch_size=1)
    with pytest.raises(ChassisExecutionContractError, match="initial_state shape"):
        ChassisExecutionCommand(
            tau_cmd=valid.tau_cmd,
            initial_state=valid.initial_state[..., :-1],
            vehicle_condition=valid.vehicle_condition,
            controller_context=valid.controller_context,
            controller_mode=valid.controller_mode,
            agent_role=valid.agent_role,
        )
    invalid = valid.initial_state.clone()
    invalid[0, 0, 5] = torch.nan
    with pytest.raises(ChassisExecutionContractError, match="finite"):
        ChassisExecutionCommand(
            tau_cmd=valid.tau_cmd,
            initial_state=invalid,
            vehicle_condition=valid.vehicle_condition,
            controller_context=valid.controller_context,
            controller_mode=valid.controller_mode,
            agent_role=valid.agent_role,
        )


def test_negative_or_inconsistent_variance_is_rejected() -> None:
    valid = aggregate_ensemble_predictions(
        synthetic_member_predictions(synthetic_command(batch_size=1))
    )
    payload = {name: getattr(valid, name) for name in valid.__dataclass_fields__}
    payload["trajectory_total_variance"] = -torch.ones_like(
        valid.trajectory_total_variance
    )
    with pytest.raises(ChassisExecutionContractError, match="non-negative"):
        ChassisExecutionPrediction(**payload)


def test_dataset_fingerprint_and_component_hashes_are_strict() -> None:
    first = _metadata()
    second = _metadata()
    assert first.fingerprint() == second.fingerprint()
    with pytest.raises(ChassisExecutionDatasetError, match="SHA256"):
        ChassisExecutionDatasetMetadata(
            **{
                **first.payload(),
                "controller_contract_sha256": "not-a-hash",
            }
        )


def test_run_groups_are_atomic_and_deterministic() -> None:
    salt = "fixture-v1"
    groups = ["trucksim-run-1", "trucksim-run-2", "trucksim-run-1"]
    rows = [
        {
            "run_group_id": group,
            "split": assign_run_group_split(group, salt=salt),
        }
        for group in groups
    ]
    validate_atomic_run_group_splits(rows, salt=salt)
    broken = copy.deepcopy(rows)
    broken[-1]["split"] = (
        "test" if broken[-1]["split"] != "test" else "train"
    )
    with pytest.raises(ChassisExecutionDatasetError):
        validate_atomic_run_group_splits(broken, salt=salt)
