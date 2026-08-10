"""CF-4 diagnostic/formal training and strict checkpoint handling."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .contracts import (
    CHASSIS_STATE_FIELDS,
    CONTROL_FIELDS,
    ENSEMBLE_SIZE,
    NUM_GROUPS,
    ChassisExecutionCommand,
)
from .storage import ChassisExecutionDataset, verify_chassis_execution_dataset
from .surrogate import (
    ChassisExecutionEnsemble,
    ChassisFeatureNormalizer,
    ChassisSurrogateConfig,
    ChassisSurrogateError,
    NormalizedMemberOutput,
)


CHECKPOINT_FORMAT = "chassis_execution_surrogate_ensemble_v1"
MEMBER_SEEDS = (17, 23, 31)


class ChassisTrainingError(RuntimeError):
    """Raised when CF-4 training or checkpoint eligibility is invalid."""


@dataclass(frozen=True)
class ChassisTrainingConfig:
    run_mode: Literal["diagnostic", "formal"] = "diagnostic"
    seed: int = 17
    member_seeds: tuple[int, int, int] = MEMBER_SEEDS
    batch_size: int = 64
    epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_grad_norm: float = 5.0
    patience: int = 15
    num_workers: int = 0
    trajectory_weight: float = 1.0
    chassis_weight: float = 1.0
    control_weight: float = 0.5
    kinematic_consistency_weight: float = 0.05

    def __post_init__(self) -> None:
        if self.run_mode not in {"diagnostic", "formal"}:
            raise ChassisTrainingError("run_mode must be diagnostic or formal")
        if len(self.member_seeds) != ENSEMBLE_SIZE or len(set(self.member_seeds)) != ENSEMBLE_SIZE:
            raise ChassisTrainingError("exactly three distinct member seeds are required")
        if min(self.batch_size, self.epochs, self.patience) <= 0 or self.num_workers < 0:
            raise ChassisTrainingError("training counts must be positive")
        if min(self.learning_rate, self.max_grad_norm) <= 0.0 or self.weight_decay < 0.0:
            raise ChassisTrainingError("optimizer values are invalid")
        if min(
            self.trajectory_weight,
            self.chassis_weight,
            self.control_weight,
            self.kinematic_consistency_weight,
        ) < 0.0:
            raise ChassisTrainingError("loss weights must be non-negative")


@dataclass(frozen=True)
class ChassisAcceptanceThresholds:
    trajectory_ade_m: float = 1.5
    trajectory_fde_m: float = 2.5
    speed_mae_mps: float = 1.0
    yaw_rate_mae_rad_s: float = 0.10
    roll_mae_rad: float = 0.03
    rollover_index_mae: float = 0.10
    throttle_mae: float = 0.15
    brake_mae: float = 0.15
    minimum_trajectory_ade_reduction: float = 0.80

    def __post_init__(self) -> None:
        if min(
            self.trajectory_ade_m,
            self.trajectory_fde_m,
            self.speed_mae_mps,
            self.yaw_rate_mae_rad_s,
            self.roll_mae_rad,
            self.rollover_index_mae,
            self.throttle_mae,
            self.brake_mae,
        ) <= 0.0:
            raise ChassisTrainingError("acceptance metric limits must be positive")
        if not 0.0 < self.minimum_trajectory_ade_reduction < 1.0:
            raise ChassisTrainingError("ADE reduction threshold must be in (0,1)")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device=device, non_blocking=True) for name, value in batch.items()}


def _wrapped_difference(first: Tensor, second: Tensor) -> Tensor:
    return torch.atan2(torch.sin(first - second), torch.cos(first - second))


def _gaussian_nll(error: Tensor, log_variance: Tensor) -> Tensor:
    return 0.5 * (torch.exp(-log_variance) * error.square() + log_variance).mean()


def surrogate_loss(
    ensemble: ChassisExecutionEnsemble,
    output: NormalizedMemberOutput,
    batch: Mapping[str, Tensor],
    config: ChassisTrainingConfig,
) -> tuple[Tensor, dict[str, Tensor]]:
    normalizer = ensemble.normalizer
    trajectory_target = normalizer.normalize("executed_trajectory", batch["executed_trajectory"])
    chassis_target = normalizer.normalize("chassis_state", batch["chassis_state"])
    control_target = normalizer.normalize("applied_control", batch["applied_control"])
    trajectory_error = output.trajectory_mean - trajectory_target
    predicted_trajectory = normalizer.denormalize("executed_trajectory", output.trajectory_mean)
    heading_error = _wrapped_difference(
        predicted_trajectory[..., 2], batch["executed_trajectory"][..., 2]
    ) / normalizer.executed_trajectory_std[2]
    trajectory_error = torch.cat(
        (trajectory_error[..., :2], heading_error.unsqueeze(-1)), dim=-1
    )
    trajectory_nll = _gaussian_nll(trajectory_error, output.trajectory_log_variance)
    chassis_nll = _gaussian_nll(
        output.chassis_mean - chassis_target, output.chassis_log_variance
    )
    control_nll = _gaussian_nll(
        output.control_mean - control_target, output.control_log_variance
    )
    predicted_chassis = normalizer.denormalize("chassis_state", output.chassis_mean)
    origin_xy = torch.zeros_like(predicted_trajectory[..., :1, :2])
    displacement = torch.diff(
        torch.cat((origin_xy, predicted_trajectory[..., :2]), dim=-2), dim=-2
    )
    derived_speed = torch.linalg.vector_norm(displacement, dim=-1) / 0.1
    speed_index = CHASSIS_STATE_FIELDS.index("longitudinal_speed_mps")
    speed_consistency = F.smooth_l1_loss(
        predicted_chassis[..., speed_index], derived_speed, beta=0.5
    )
    origin_heading = torch.zeros_like(predicted_trajectory[..., :1, 2])
    heading_sequence = torch.cat((origin_heading, predicted_trajectory[..., 2]), dim=-1)
    derived_yaw = _wrapped_difference(
        heading_sequence[..., 1:], heading_sequence[..., :-1]
    ) / 0.1
    yaw_index = CHASSIS_STATE_FIELDS.index("yaw_rate_rad_s")
    yaw_consistency = F.smooth_l1_loss(
        predicted_chassis[..., yaw_index], derived_yaw, beta=0.1
    )
    consistency = speed_consistency + yaw_consistency
    total = (
        config.trajectory_weight * trajectory_nll
        + config.chassis_weight * chassis_nll
        + config.control_weight * control_nll
        + config.kinematic_consistency_weight * consistency
    )
    components = {
        "total": total,
        "trajectory_nll": trajectory_nll,
        "chassis_nll": chassis_nll,
        "control_nll": control_nll,
        "speed_consistency": speed_consistency,
        "yaw_consistency": yaw_consistency,
    }
    return total, components


@torch.no_grad()
def evaluate_member(
    ensemble: ChassisExecutionEnsemble,
    member_index: int,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
    config: ChassisTrainingConfig,
) -> dict[str, float]:
    ensemble.eval()
    sums: dict[str, float] = {}
    count = 0
    for raw in loader:
        batch = _move_batch(raw, device)
        output = ensemble.forward_member(member_index, batch)
        _, components = surrogate_loss(ensemble, output, batch, config)
        physical = ensemble._physical_output(output)
        trajectory_error = physical.trajectory_mean - batch["executed_trajectory"]
        distance = torch.linalg.vector_norm(trajectory_error[..., :2], dim=-1)
        heading = _wrapped_difference(
            physical.trajectory_mean[..., 2], batch["executed_trajectory"][..., 2]
        ).abs()
        chassis_error = (physical.chassis_mean - batch["chassis_state"]).abs()
        control_error = (physical.control_mean - batch["applied_control"]).abs()
        metrics: dict[str, Tensor] = {
            **components,
            "trajectory_ade_m": distance.mean(),
            "trajectory_fde_m": distance[..., -1].mean(),
            "heading_mae_rad": heading.mean(),
            "speed_mae_mps": chassis_error[..., CHASSIS_STATE_FIELDS.index("longitudinal_speed_mps")].mean(),
            "yaw_rate_mae_rad_s": chassis_error[..., CHASSIS_STATE_FIELDS.index("yaw_rate_rad_s")].mean(),
            "roll_mae_rad": chassis_error[..., CHASSIS_STATE_FIELDS.index("roll_rad")].mean(),
            "rollover_index_mae": chassis_error[..., CHASSIS_STATE_FIELDS.index("rollover_index")].mean(),
            "steering_mae_rad": control_error[..., CONTROL_FIELDS.index("road_wheel_angle_rad")].mean(),
            "throttle_mae": control_error[..., CONTROL_FIELDS.index("throttle_normalized")].mean(),
            "brake_mae": control_error[..., CONTROL_FIELDS.index("brake_normalized")].mean(),
        }
        batch_count = int(batch["tau_cmd"].shape[0])
        count += batch_count
        for name, value in metrics.items():
            scalar = float(value.detach().cpu())
            if not math.isfinite(scalar):
                raise ChassisTrainingError(f"non-finite validation metric {name}")
            sums[name] = sums.get(name, 0.0) + scalar * batch_count
    if count == 0:
        raise ChassisTrainingError("validation split is empty")
    return {name: value / count for name, value in sums.items()}


@torch.no_grad()
def evaluate_ensemble(
    ensemble: ChassisExecutionEnsemble,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
) -> dict[str, float]:
    ensemble.eval()
    sums: dict[str, float] = {}
    count = 0
    for raw in loader:
        batch = _move_batch(raw, device)
        command = ChassisExecutionCommand(
            tau_cmd=batch["tau_cmd"].unsqueeze(1).expand(-1, NUM_GROUPS, -1, -1, -1),
            initial_state=batch["initial_state"],
            vehicle_condition=batch["vehicle_condition"],
            controller_context=batch["controller_context"],
            controller_mode=batch["controller_mode"],
            agent_role=batch["agent_role"],
        )
        prediction = ensemble.predict(command)
        trajectory = prediction.executed_trajectory_mean[:, 0]
        chassis = prediction.chassis_state_mean[:, 0]
        control = prediction.control_mean[:, 0]
        distance = torch.linalg.vector_norm(
            trajectory[..., :2] - batch["executed_trajectory"][..., :2], dim=-1
        )
        heading = _wrapped_difference(
            trajectory[..., 2], batch["executed_trajectory"][..., 2]
        ).abs()
        chassis_error = (chassis - batch["chassis_state"]).abs()
        control_error = (control - batch["applied_control"]).abs()
        metrics = {
            "trajectory_ade_m": distance.mean(),
            "trajectory_fde_m": distance[..., -1].mean(),
            "heading_mae_rad": heading.mean(),
            "speed_mae_mps": chassis_error[..., CHASSIS_STATE_FIELDS.index("longitudinal_speed_mps")].mean(),
            "yaw_rate_mae_rad_s": chassis_error[..., CHASSIS_STATE_FIELDS.index("yaw_rate_rad_s")].mean(),
            "roll_mae_rad": chassis_error[..., CHASSIS_STATE_FIELDS.index("roll_rad")].mean(),
            "rollover_index_mae": chassis_error[..., CHASSIS_STATE_FIELDS.index("rollover_index")].mean(),
            "steering_mae_rad": control_error[..., CONTROL_FIELDS.index("road_wheel_angle_rad")].mean(),
            "throttle_mae": control_error[..., CONTROL_FIELDS.index("throttle_normalized")].mean(),
            "brake_mae": control_error[..., CONTROL_FIELDS.index("brake_normalized")].mean(),
            "trajectory_aleatoric_variance_mean": prediction.trajectory_aleatoric_variance[:, 0].mean(),
            "trajectory_epistemic_variance_mean": prediction.trajectory_epistemic_variance[:, 0].mean(),
            "chassis_total_variance_mean": prediction.chassis_total_variance[:, 0].mean(),
        }
        batch_count = int(batch["tau_cmd"].shape[0])
        count += batch_count
        for name, value in metrics.items():
            scalar = float(value.cpu())
            if not math.isfinite(scalar):
                raise ChassisTrainingError(f"non-finite ensemble metric {name}")
            sums[name] = sums.get(name, 0.0) + scalar * batch_count
    if count == 0:
        raise ChassisTrainingError("ensemble evaluation split is empty")
    return {name: value / count for name, value in sums.items()}


def _acceptance_result(
    member_results: list[dict[str, Any]],
    test_metrics: list[dict[str, float]],
    ensemble_metrics: dict[str, float],
    thresholds: ChassisAcceptanceThresholds,
) -> dict[str, Any]:
    limits = {
        "trajectory_ade_m": thresholds.trajectory_ade_m,
        "trajectory_fde_m": thresholds.trajectory_fde_m,
        "speed_mae_mps": thresholds.speed_mae_mps,
        "yaw_rate_mae_rad_s": thresholds.yaw_rate_mae_rad_s,
        "roll_mae_rad": thresholds.roll_mae_rad,
        "rollover_index_mae": thresholds.rollover_index_mae,
        "throttle_mae": thresholds.throttle_mae,
        "brake_mae": thresholds.brake_mae,
    }
    failures: list[str] = []
    for member_index, metrics in enumerate(test_metrics):
        for name, limit in limits.items():
            if metrics[name] > limit:
                failures.append(f"member_{member_index}:{name}={metrics[name]:.6g}>{limit:.6g}")
        initial = member_results[member_index]["initial_validation"]["trajectory_ade_m"]
        best = member_results[member_index]["best_validation"]["trajectory_ade_m"]
        reduction = 1.0 - best / max(initial, 1e-12)
        member_results[member_index]["validation_trajectory_ade_reduction"] = reduction
        if reduction < thresholds.minimum_trajectory_ade_reduction:
            failures.append(
                f"member_{member_index}:trajectory_ade_reduction={reduction:.6g}"
                f"<{thresholds.minimum_trajectory_ade_reduction:.6g}"
            )
    for name, limit in limits.items():
        if ensemble_metrics[name] > limit:
            failures.append(f"ensemble:{name}={ensemble_metrics[name]:.6g}>{limit:.6g}")
    if ensemble_metrics["trajectory_aleatoric_variance_mean"] <= 0.0:
        failures.append("ensemble:aleatoric_variance_not_positive")
    if ensemble_metrics["trajectory_epistemic_variance_mean"] <= 1e-10:
        failures.append("ensemble:epistemic_variance_not_detectable")
    return {
        "passed": not failures,
        "thresholds": asdict(thresholds),
        "failures": failures,
    }


def _next_run_root(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in output_root.glob("run_*"):
        try:
            indices.append(int(path.name.split("_", 1)[1]))
        except ValueError:
            continue
    result = output_root / f"run_{max(indices, default=0) + 1}"
    result.mkdir()
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _normalizer_from_state(
    state: Mapping[str, Tensor], *, minimum_std: float
) -> ChassisFeatureNormalizer:
    statistics = {
        name: (state[f"{name}_mean"].detach().cpu(), state[f"{name}_std"].detach().cpu())
        for name in ChassisFeatureNormalizer.FEATURE_DIMS
    }
    normalizer = ChassisFeatureNormalizer(statistics, minimum_std=minimum_std)
    normalizer.load_state_dict(dict(state), strict=True)
    return normalizer


def save_surrogate_checkpoint(
    path: Path,
    *,
    ensemble: ChassisExecutionEnsemble,
    dataset_report: Mapping[str, Any],
    training_config: ChassisTrainingConfig,
    metrics: Mapping[str, Any],
    optimizer_steps: int,
) -> None:
    diagnostic = bool(dataset_report["diagnostic_only"])
    eligible = bool(dataset_report["eligible_for_formal_training"])
    if training_config.run_mode == "diagnostic" and (not diagnostic or eligible):
        raise ChassisTrainingError("diagnostic training requires diagnostic ineligible data")
    if training_config.run_mode == "formal" and (diagnostic or not eligible):
        raise ChassisTrainingError("formal training requires a formally eligible dataset")
    payload = {
        "format": CHECKPOINT_FORMAT,
        "schema_version": 1,
        "run_mode": training_config.run_mode,
        "data_origin": dataset_report["data_origin"],
        "diagnostic_only": diagnostic,
        "eligible_for_formal_training": eligible,
        "dataset_fingerprint": dataset_report["dataset_fingerprint"],
        "dataset_metadata_fingerprint": dataset_report["dataset_metadata_fingerprint"],
        "surrogate_config": asdict(ensemble.config),
        "training_config": asdict(training_config),
        "normalizer_state": {
            name: value.detach().cpu() for name, value in ensemble.normalizer.state_dict().items()
        },
        "ensemble_state": {
            name: value.detach().cpu() for name, value in ensemble.state_dict().items()
        },
        "optimizer_steps": int(optimizer_steps),
        "metrics": copy.deepcopy(dict(metrics)),
    }
    _atomic_torch_save(payload, path)


def load_surrogate_checkpoint(
    path: Path | str,
    *,
    expected_dataset_fingerprint: str | None = None,
    allow_diagnostic: bool = False,
    map_location: str | torch.device = "cpu",
) -> tuple[ChassisExecutionEnsemble, dict[str, Any]]:
    checkpoint_path = Path(path).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT or payload.get("schema_version") != 1:
        raise ChassisTrainingError("surrogate checkpoint format/schema mismatch")
    required_fields = {
        "run_mode",
        "data_origin",
        "diagnostic_only",
        "eligible_for_formal_training",
        "dataset_fingerprint",
        "dataset_metadata_fingerprint",
        "surrogate_config",
        "training_config",
        "normalizer_state",
        "ensemble_state",
        "optimizer_steps",
        "metrics",
    }
    if not required_fields.issubset(payload):
        raise ChassisTrainingError("surrogate checkpoint required field set is incomplete")
    if bool(payload.get("diagnostic_only")) and not allow_diagnostic:
        raise ChassisTrainingError("diagnostic surrogate checkpoint is not allowed")
    if bool(payload.get("diagnostic_only")) and bool(payload.get("eligible_for_formal_training")):
        raise ChassisTrainingError("checkpoint diagnostic/formal flags conflict")
    if payload.get("data_origin") == "synthetic_virtual" and (
        not bool(payload.get("diagnostic_only"))
        or bool(payload.get("eligible_for_formal_training"))
        or payload.get("run_mode") != "diagnostic"
    ):
        raise ChassisTrainingError("synthetic checkpoint provenance cannot be formal")
    if expected_dataset_fingerprint is not None and payload.get("dataset_fingerprint") != expected_dataset_fingerprint:
        raise ChassisTrainingError("surrogate checkpoint dataset fingerprint mismatch")
    try:
        config = ChassisSurrogateConfig(**payload["surrogate_config"])
        normalizer = _normalizer_from_state(
            payload["normalizer_state"], minimum_std=config.minimum_std
        )
        ensemble = ChassisExecutionEnsemble(config, normalizer)
        ensemble.load_state_dict(payload["ensemble_state"], strict=True)
    except (KeyError, TypeError, ValueError, RuntimeError, ChassisSurrogateError) as exc:
        raise ChassisTrainingError(f"invalid surrogate checkpoint state: {exc}") from exc
    return ensemble, payload


def run_surrogate_training(
    dataset_root: Path | str,
    output_root: Path | str,
    *,
    model_config: ChassisSurrogateConfig,
    training_config: ChassisTrainingConfig,
    device: str,
    acceptance_thresholds: ChassisAcceptanceThresholds = ChassisAcceptanceThresholds(),
) -> dict[str, Any]:
    dataset_report = verify_chassis_execution_dataset(dataset_root)
    if training_config.run_mode == "diagnostic":
        if dataset_report["data_origin"] != "synthetic_virtual" or not dataset_report["diagnostic_only"]:
            raise ChassisTrainingError("CF-4 diagnostic mode requires synthetic_virtual diagnostic data")
    elif dataset_report["diagnostic_only"] or not dataset_report["eligible_for_formal_training"]:
        raise ChassisTrainingError("CF-4 formal mode rejects diagnostic/ineligible data")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ChassisTrainingError("CUDA was requested but is unavailable")
    train_dataset = ChassisExecutionDataset(dataset_root, split="train")
    val_dataset = ChassisExecutionDataset(dataset_root, split="val")
    test_dataset = ChassisExecutionDataset(dataset_root, split="test")
    normalizer = ChassisFeatureNormalizer.fit(
        train_dataset.arrays, minimum_std=model_config.minimum_std
    )
    _seed_everything(training_config.seed)
    ensemble = ChassisExecutionEnsemble(model_config, normalizer).to(target_device)
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=target_device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=target_device.type == "cuda",
    )
    run_root = _next_run_root(Path(output_root).expanduser().resolve())
    checkpoint_root = run_root / "checkpoints"
    checkpoint_root.mkdir()
    (run_root / "config.json").write_text(
        json.dumps(
            {
                "model": asdict(model_config),
                "training": asdict(training_config),
                "dataset_report": dataset_report,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    member_results: list[dict[str, Any]] = []
    total_steps = 0
    for member_index, member_seed in enumerate(training_config.member_seeds):
        _seed_everything(member_seed)
        # Reinitialize this member independently after the ensemble construction.
        ensemble.members[member_index] = type(ensemble.members[member_index])(model_config).to(target_device)
        optimizer = AdamW(
            ensemble.members[member_index].parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        best_loss = math.inf
        best_state: dict[str, Tensor] | None = None
        best_metrics: dict[str, float] | None = None
        stale_epochs = 0
        generator = torch.Generator().manual_seed(member_seed)
        initial_metrics = evaluate_member(
            ensemble, member_index, val_loader, target_device, training_config
        )
        for epoch in range(1, training_config.epochs + 1):
            train_loader = DataLoader(
                train_dataset,
                batch_size=training_config.batch_size,
                shuffle=True,
                num_workers=training_config.num_workers,
                pin_memory=target_device.type == "cuda",
                generator=generator,
            )
            ensemble.train()
            for raw in train_loader:
                batch = _move_batch(raw, target_device)
                optimizer.zero_grad(set_to_none=True)
                output = ensemble.forward_member(member_index, batch)
                loss, _ = surrogate_loss(ensemble, output, batch, training_config)
                if not bool(torch.isfinite(loss)):
                    raise ChassisTrainingError("non-finite surrogate loss")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    ensemble.members[member_index].parameters(), training_config.max_grad_norm
                )
                if not bool(torch.isfinite(gradient_norm)):
                    raise ChassisTrainingError("non-finite surrogate gradient")
                optimizer.step()
                total_steps += 1
            metrics = evaluate_member(
                ensemble, member_index, val_loader, target_device, training_config
            )
            if metrics["total"] < best_loss:
                best_loss = metrics["total"]
                best_metrics = metrics
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in ensemble.members[member_index].state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= training_config.patience:
                break
        if best_state is None or best_metrics is None:
            raise ChassisTrainingError("member training did not produce a best state")
        ensemble.members[member_index].load_state_dict(best_state, strict=True)
        member_results.append(
            {
                "member_index": member_index,
                "seed": member_seed,
                "epochs_completed": epoch,
                "initial_validation": initial_metrics,
                "best_validation": best_metrics,
            }
        )
    test_metrics = [
        evaluate_member(ensemble, index, test_loader, target_device, training_config)
        for index in range(ENSEMBLE_SIZE)
    ]
    ensemble_metrics = evaluate_ensemble(ensemble, test_loader, target_device)
    acceptance = _acceptance_result(
        member_results, test_metrics, ensemble_metrics, acceptance_thresholds
    )
    summary = {
        "format": "chassis_execution_surrogate_training_result_v1",
        "run_root": str(run_root),
        "dataset_fingerprint": dataset_report["dataset_fingerprint"],
        "data_origin": dataset_report["data_origin"],
        "diagnostic_only": dataset_report["diagnostic_only"],
        "eligible_for_formal_training": dataset_report["eligible_for_formal_training"],
        "optimizer_steps": total_steps,
        "members": member_results,
        "test_metrics": test_metrics,
        "ensemble_test_metrics": ensemble_metrics,
        "acceptance": acceptance,
    }
    best_path = checkpoint_root / "best.pt"
    last_path = checkpoint_root / "last.pt"
    save_surrogate_checkpoint(
        best_path,
        ensemble=ensemble,
        dataset_report=dataset_report,
        training_config=training_config,
        metrics=summary,
        optimizer_steps=total_steps,
    )
    save_surrogate_checkpoint(
        last_path,
        ensemble=ensemble,
        dataset_report=dataset_report,
        training_config=training_config,
        metrics=summary,
        optimizer_steps=total_steps,
    )
    summary["best_checkpoint"] = str(best_path)
    summary["best_checkpoint_sha256"] = _file_sha256(best_path)
    (run_root / "metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-mode", choices=("diagnostic", "formal"), default="diagnostic")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    result = run_surrogate_training(
        args.dataset_root,
        args.output_root,
        model_config=ChassisSurrogateConfig(),
        training_config=ChassisTrainingConfig(
            run_mode=args.run_mode,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
        ),
        acceptance_thresholds=ChassisAcceptanceThresholds(),
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
