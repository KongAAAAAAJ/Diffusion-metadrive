from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from chassis_execution.contracts import ChassisExecutionCommand
from chassis_execution.storage import write_chassis_execution_dataset
from chassis_execution.surrogate import (
    ChassisExecutionEnsemble,
    ChassisFeatureNormalizer,
    ChassisSurrogateConfig,
    ChassisSurrogateError,
    command_to_dense,
)
from chassis_execution.synthetic_data import (
    SyntheticChassisConfig,
    generate_synthetic_chassis_batch,
)
from chassis_execution.training import (
    ChassisTrainingConfig,
    ChassisTrainingError,
    load_surrogate_checkpoint,
    run_surrogate_training,
    save_surrogate_checkpoint,
    surrogate_loss,
)


def _batch(count: int = 32):
    return generate_synthetic_chassis_batch(SyntheticChassisConfig(sample_count=count))


def _normalizer_and_model(count: int = 32):
    target, *_ = _batch(count)
    arrays = {
        name: getattr(target, name).numpy()
        for name in (
            "tau_cmd",
            "initial_state",
            "vehicle_condition",
            "controller_context",
            "executed_trajectory",
            "chassis_state",
            "applied_control",
        )
    }
    normalizer = ChassisFeatureNormalizer.fit(arrays)
    config = ChassisSurrogateConfig(
        hidden_dim=32,
        static_dim=32,
        num_gru_layers=1,
        dropout=0.0,
    )
    return target, ChassisExecutionEnsemble(config, normalizer)


def test_command_dense_time_axis_and_heading_unwrap() -> None:
    command = torch.zeros((1, 3, 8, 3), dtype=torch.float32)
    command[..., 0] = torch.arange(1, 9, dtype=torch.float32) * 2.0
    command[..., 2] = torch.tensor(
        [3.10, -3.10, -3.05, -3.00, -2.95, -2.90, -2.85, -2.80]
    )
    dense = command_to_dense(command)
    assert dense.shape == (1, 3, 40, 3)
    torch.testing.assert_close(dense[:, :, 4::5, 0], command[..., 0])
    delta = torch.atan2(
        torch.sin(torch.diff(dense[..., 2], dim=2)),
        torch.cos(torch.diff(dense[..., 2], dim=2)),
    )
    assert float(delta.abs().max()) < 0.7


def test_config_and_normalizer_contracts_are_strict() -> None:
    with pytest.raises(ChassisSurrogateError, match="divisible"):
        ChassisSurrogateConfig(static_dim=31, transformer_heads=4)
    with pytest.raises(ChassisTrainingError, match="three distinct"):
        ChassisTrainingConfig(member_seeds=(17, 17, 31))
    target, _, _, _ = _batch()
    arrays = {"tau_cmd": target.tau_cmd.numpy()}
    with pytest.raises(ChassisSurrogateError, match="incomplete"):
        ChassisFeatureNormalizer.fit(arrays)


def test_forward_loss_backward_and_all_submodules_receive_gradients() -> None:
    target, model = _normalizer_and_model()
    batch = {name: getattr(target, name) for name in target.__dataclass_fields__}
    output = model.forward_member(0, batch)
    assert output.trajectory_mean.shape == (32, 3, 40, 3)
    assert output.chassis_mean.shape == (32, 3, 40, 8)
    assert output.control_mean.shape == (32, 3, 40, 3)
    loss, components = surrogate_loss(
        model,
        output,
        batch,
        ChassisTrainingConfig(epochs=1, patience=1, batch_size=8),
    )
    assert torch.isfinite(loss)
    assert set(components) == {
        "total",
        "trajectory_nll",
        "chassis_nll",
        "control_nll",
        "speed_consistency",
        "yaw_consistency",
    }
    loss.backward()
    for name in (
        "role_embedding.weight",
        "mode_embedding.weight",
        "static_encoder.0.weight",
        "command_encoder.0.weight",
        "temporal.weight_ih_l0",
        "output_head.3.weight",
    ):
        gradient = dict(model.members[0].named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert float(gradient.abs().sum()) > 0.0


def test_ensemble_predict_shapes_variance_identity_and_frozen_copy() -> None:
    target, model = _normalizer_and_model(count=32)
    count = 2
    command = ChassisExecutionCommand(
        tau_cmd=target.tau_cmd[:count].unsqueeze(1).expand(-1, 4, -1, -1, -1).clone(),
        initial_state=target.initial_state[:count],
        vehicle_condition=target.vehicle_condition[:count],
        controller_context=target.controller_context[:count],
        controller_mode=target.controller_mode[:count],
        agent_role=target.agent_role[:count],
    )
    model.eval()
    prediction = model.predict(command)
    assert prediction.executed_trajectory_mean.shape == (2, 4, 3, 40, 3)
    assert prediction.chassis_state_mean.shape == (2, 4, 3, 40, 8)
    assert torch.allclose(
        prediction.trajectory_total_variance,
        prediction.trajectory_aleatoric_variance
        + prediction.trajectory_epistemic_variance,
    )
    assert bool((prediction.trajectory_total_variance >= 0.0).all())
    frozen = model.frozen_copy()
    assert not any(parameter.requires_grad for parameter in frozen.parameters())
    frozen.eval()
    repeated = frozen.predict(command)
    torch.testing.assert_close(
        prediction.executed_trajectory_mean, repeated.executed_trajectory_mean
    )


def test_checkpoint_is_strict_and_diagnostic_cannot_be_loaded_silently(tmp_path: Path) -> None:
    _, model = _normalizer_and_model()
    report = {
        "data_origin": "synthetic_virtual",
        "diagnostic_only": True,
        "eligible_for_formal_training": False,
        "dataset_fingerprint": "a" * 64,
        "dataset_metadata_fingerprint": "b" * 64,
    }
    path = tmp_path / "model.pt"
    config = ChassisTrainingConfig(epochs=1, patience=1, batch_size=8)
    save_surrogate_checkpoint(
        path,
        ensemble=model,
        dataset_report=report,
        training_config=config,
        metrics={"finite": True},
        optimizer_steps=1,
    )
    with pytest.raises(ChassisTrainingError, match="diagnostic"):
        load_surrogate_checkpoint(path)
    loaded, payload = load_surrogate_checkpoint(
        path,
        allow_diagnostic=True,
        expected_dataset_fingerprint="a" * 64,
    )
    assert payload["diagnostic_only"] is True
    for first, second in zip(model.state_dict().values(), loaded.state_dict().values()):
        torch.testing.assert_close(first, second)
    with pytest.raises(ChassisTrainingError, match="fingerprint"):
        load_surrogate_checkpoint(
            path, allow_diagnostic=True, expected_dataset_fingerprint="c" * 64
        )


def test_one_epoch_training_smoke_and_formal_rejection(tmp_path: Path) -> None:
    target, identities, metadata, provenance = _batch(256)
    dataset_root = tmp_path / "dataset"
    write_chassis_execution_dataset(
        dataset_root,
        target=target,
        identities=identities,
        metadata=metadata,
        provenance=provenance,
    )
    model_config = ChassisSurrogateConfig(
        hidden_dim=32,
        static_dim=32,
        num_gru_layers=1,
        dropout=0.0,
    )
    result = run_surrogate_training(
        dataset_root,
        tmp_path / "outputs",
        model_config=model_config,
        training_config=ChassisTrainingConfig(
            epochs=1, patience=1, batch_size=64, num_workers=0
        ),
        device="cpu",
    )
    checkpoint = Path(result["best_checkpoint"])
    assert checkpoint.is_file()
    assert result["diagnostic_only"] is True
    assert result["eligible_for_formal_training"] is False
    load_surrogate_checkpoint(
        checkpoint,
        allow_diagnostic=True,
        expected_dataset_fingerprint=result["dataset_fingerprint"],
    )
    with pytest.raises(ChassisTrainingError, match="formal"):
        run_surrogate_training(
            dataset_root,
            tmp_path / "formal",
            model_config=model_config,
            training_config=ChassisTrainingConfig(
                run_mode="formal", epochs=1, patience=1, batch_size=64
            ),
            device="cpu",
        )
