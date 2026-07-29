"""Stage 1 A training for the joint-first BEV-only diffusion planner."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from expert_dataset.joint_bev_dataset import (
    JointBEVDataset,
    JointBEVDatasetConfig,
    build_joint_bev_dataloader,
    validate_dataset_contract,
)
from models.bev_planner import (
    BEVOnlyDiffusionPlanner,
    JointStage1Loss,
    Stage1LossConfig,
    Stage1LossResult,
)


DEFAULT_CONFIG_PATH = Path("configs/train/bev_diffusion_stage1.yaml")
DEFAULT_OUTPUT_ROOT = Path(
    "/media/kong/Elements_SE/Diffusion_Data/outputs/bev_diffusion_stage1"
)
CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_FORMAT = "bev_stage1_joint_mean"
RUN_PATTERN = re.compile(r"^run_(\d+)$")


class Stage1TrainingError(RuntimeError):
    """Raised when the fixed Stage 1 A training contract is violated."""


def load_stage1_config(path: Path | str) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise Stage1TrainingError(f"invalid Stage 1 config: {config_path}") from exc
    if not isinstance(payload, Mapping):
        raise Stage1TrainingError("Stage 1 config root must be a mapping")
    config = dict(payload)
    validate_stage1_config(config)
    return config


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise Stage1TrainingError(f"config.{key} must be a mapping")
    return value


def _positive_int(value: Any, *, name: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Stage1TrainingError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise Stage1TrainingError(f"{name} must be >= {minimum}")
    return int(value)


def _positive_float(value: Any, *, name: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise Stage1TrainingError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Stage1TrainingError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result < 0.0 or (not allow_zero and result == 0.0):
        relation = "non-negative" if allow_zero else "positive"
        raise Stage1TrainingError(f"{name} must be finite and {relation}")
    return result


def validate_stage1_config(config: Mapping[str, Any]) -> None:
    experiment = _mapping(config, "experiment")
    dataset = _mapping(config, "dataset")
    training = _mapping(config, "training")
    loss = _mapping(config, "loss")
    overfit = _mapping(config, "overfit")
    if experiment.get("variant") != "A":
        raise Stage1TrainingError("Stage 1 variant must be A")
    if experiment.get("joint_update") != "joint_mean":
        raise Stage1TrainingError("Stage 1 joint_update must be joint_mean")
    if experiment.get("predecessor_condition") != "none":
        raise Stage1TrainingError("Stage 1 predecessor_condition must be none")
    dataset_root = dataset.get("root")
    if not isinstance(dataset_root, str) or not dataset_root:
        raise Stage1TrainingError("dataset.root must be a non-empty path string")
    if dataset.get("train_split") != "train" or dataset.get("val_split") != "val":
        raise Stage1TrainingError("formal Stage 1 splits must be train and val")
    for name in ("batch_size", "gradient_accumulation", "max_epochs"):
        _positive_int(training.get(name), name=f"training.{name}")
    _positive_int(
        training.get("num_workers"), name="training.num_workers", allow_zero=True
    )
    _positive_int(training.get("seed"), name="training.seed", allow_zero=True)
    _positive_int(
        training.get("gradient_diagnostics_interval"),
        name="training.gradient_diagnostics_interval",
    )
    for name in ("planner_lr", "backbone_lr", "weight_decay"):
        _positive_float(
            training.get(name),
            name=f"training.{name}",
            allow_zero=name == "weight_decay",
        )
    _positive_float(
        training.get("grad_scaler_init_scale"),
        name="training.grad_scaler_init_scale",
    )
    if not isinstance(training.get("mixed_precision"), bool):
        raise Stage1TrainingError("training.mixed_precision must be bool")
    device = training.get("device")
    if device not in ("cuda", "cpu"):
        raise Stage1TrainingError("training.device must be cuda or cpu")
    Stage1LossConfig(**dict(loss))
    _positive_int(
        overfit.get("max_optimizer_steps"),
        name="overfit.max_optimizer_steps",
    )
    for name in ("min_loss_reduction", "min_mode_accuracy"):
        value = _positive_float(
            overfit.get(name), name=f"overfit.{name}", allow_zero=True
        )
        if value > 1.0:
            raise Stage1TrainingError(f"overfit.{name} must be <= 1")
    _positive_float(overfit.get("max_gt_mode_ade_m"), name="overfit.max_gt_mode_ade_m")
    timestep = _positive_int(
        overfit.get("timestep"),
        name="overfit.timestep",
        allow_zero=True,
    )
    if timestep != 8:
        raise Stage1TrainingError("overfit.timestep must remain fixed at 8")
    _positive_int(
        overfit.get("evaluation_interval_steps"),
        name="overfit.evaluation_interval_steps",
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(configured: str) -> torch.device:
    if configured == "cuda":
        if not torch.cuda.is_available():
            raise Stage1TrainingError(
                "training.device=cuda but CUDA is unavailable; no CPU fallback is allowed"
            )
        return torch.device("cuda")
    if configured == "cpu":
        return torch.device("cpu")
    raise Stage1TrainingError("training.device must be cuda or cpu")


def create_numbered_run_dir(output_root: Path | str) -> Path:
    root = Path(output_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    existing = [
        int(match.group(1))
        for item in root.iterdir()
        if item.is_dir() and (match := RUN_PATTERN.fullmatch(item.name))
    ]
    run_dir = root / f"run_{max(existing) + 1 if existing else 1}"
    run_dir.mkdir(parents=False, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    return run_dir


def build_stage1_optimizer(
    planner: BEVOnlyDiffusionPlanner,
    *,
    planner_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> AdamW:
    backbone_parameters = list(planner.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    planner_parameters = [
        parameter
        for parameter in planner.parameters()
        if id(parameter) not in backbone_ids
    ]
    all_parameters = list(planner.parameters())
    grouped_ids = [
        id(parameter) for parameter in backbone_parameters + planner_parameters
    ]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise Stage1TrainingError("optimizer parameter groups overlap")
    if set(grouped_ids) != {id(parameter) for parameter in all_parameters}:
        raise Stage1TrainingError("optimizer parameter groups are not exhaustive")
    if not all(parameter.requires_grad for parameter in all_parameters):
        raise Stage1TrainingError("all Stage 1 planner parameters must be trainable")
    return AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": float(backbone_lr),
                "name": "backbone",
            },
            {
                "params": planner_parameters,
                "lr": float(planner_lr),
                "name": "planner",
            },
        ],
        weight_decay=float(weight_decay),
    )


def _gradient_vector(
    loss: Tensor,
    parameters: Sequence[nn.Parameter],
    *,
    retain_graph: bool,
) -> Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    parts = [
        (
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.reshape(-1)
        )
        for parameter, gradient in zip(parameters, gradients)
    ]
    return torch.cat(parts)


def role_gradient_diagnostics(
    loss_result: Stage1LossResult,
    planner: BEVOnlyDiffusionPlanner,
) -> dict[str, float]:
    """Measure each role's alignment separately for mode and trajectory heads."""

    diagnostics: dict[str, float] = {}
    role_names = ("leader", "middle", "rear")
    probes = {
        "mode_head": tuple(planner.mode_head.parameters()),
        "trajectory_head": tuple(
            planner.diffusion_decoder.trajectory_head.parameters()
        ),
    }
    eps = torch.finfo(torch.float32).eps
    for probe_name, probe_parameters in probes.items():
        vectors = [
            _gradient_vector(
                loss_result.role_total[:, role].mean(),
                probe_parameters,
                retain_graph=True,
            )
            for role in range(3)
        ]
        norms = [torch.linalg.vector_norm(vector.float()) for vector in vectors]
        prefix = f"gradient_probe/{probe_name}"
        for role_name, norm in zip(role_names, norms):
            diagnostics[f"{prefix}/{role_name}_norm"] = float(norm.detach().cpu())
        for first, second in ((0, 1), (1, 2), (0, 2)):
            denominator = (norms[first] * norms[second]).clamp_min(eps)
            cosine = (
                torch.dot(vectors[first].float(), vectors[second].float()) / denominator
            )
            diagnostics[f"{prefix}/cosine_{role_names[first]}_{role_names[second]}"] = (
                float(cosine.detach().cpu())
            )
        vector_sum = vectors[0] + vectors[1] + vectors[2]
        cancellation = torch.linalg.vector_norm(vector_sum.float()) / (
            norms[0] + norms[1] + norms[2]
        ).clamp_min(eps)
        diagnostics[f"{prefix}/cancellation_ratio"] = float(cancellation.detach().cpu())
        pair_gradient = _gradient_vector(
            loss_result.pair_joint,
            probe_parameters,
            retain_graph=True,
        )
        diagnostics[f"{prefix}/pair_norm"] = float(
            torch.linalg.vector_norm(pair_gradient.float()).detach().cpu()
        )
    if not all(math.isfinite(value) for value in diagnostics.values()):
        raise Stage1TrainingError("non-finite role-gradient diagnostic detected")
    return diagnostics


def _module_gradient_norm(module: nn.Module) -> float:
    squares = []
    for parameter in module.parameters():
        if parameter.grad is not None:
            gradient = parameter.grad.detach().float()
            if not bool(torch.isfinite(gradient).all()):
                raise Stage1TrainingError("non-finite gradient detected")
            squares.append(gradient.square().sum())
    if not squares:
        return 0.0
    return float(torch.sqrt(torch.stack(squares).sum()).cpu())


def module_gradient_norms(planner: BEVOnlyDiffusionPlanner) -> dict[str, float]:
    return {
        "gradient/backbone": _module_gradient_norm(planner.backbone),
        "gradient/bev_fusion": _module_gradient_norm(planner.bev_fusion),
        "gradient/context_encoder": _module_gradient_norm(planner.context_encoder),
        "gradient/diffusion_decoder": _module_gradient_norm(planner.diffusion_decoder),
        "gradient/trajectory_head": _module_gradient_norm(
            planner.diffusion_decoder.trajectory_head
        ),
        "gradient/mode_head": _module_gradient_norm(planner.mode_head),
    }


def move_joint_batch(
    batch: Mapping[str, Tensor], device: torch.device
) -> dict[str, Tensor]:
    return {
        name: value.to(device=device, non_blocking=device.type == "cuda")
        for name, value in batch.items()
    }


def planner_forward_from_batch(
    planner: BEVOnlyDiffusionPlanner,
    batch: Mapping[str, Tensor],
    *,
    diffusion_noise: Tensor | None = None,
    diffusion_timesteps: Tensor | None = None,
) -> dict[str, Tensor]:
    return planner(
        batch["bev"],
        batch["ego_state"],
        batch["formation_relation_state"],
        batch["relation_valid_mask"],
        batch["agent_role"],
        batch["coarse_trajectories"],
        batch["mode_valid_mask"],
        diffusion_noise=diffusion_noise,
        diffusion_timesteps=diffusion_timesteps,
    )


def loss_from_batch(
    loss_module: JointStage1Loss,
    planner_output: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
) -> Stage1LossResult:
    return loss_module(
        planner_output,
        expert_trajectory=batch["expert_trajectory"],
        gt_mode=batch["gt_mode"],
        mode_valid_mask=batch["mode_valid_mask"],
        ego_state=batch["ego_state"],
        ego_pose_global=batch["ego_pose_global"],
    )


class MetricAccumulator:
    def __init__(self) -> None:
        self.weighted: defaultdict[str, float] = defaultdict(float)
        self.samples = 0

    def update(self, metrics: Mapping[str, float], batch_size: int) -> None:
        self.samples += int(batch_size)
        for name, value in metrics.items():
            if not math.isfinite(float(value)):
                raise Stage1TrainingError(f"non-finite metric: {name}")
            self.weighted[name] += float(value) * int(batch_size)

    def compute(self) -> dict[str, float]:
        if self.samples <= 0:
            raise Stage1TrainingError("metric accumulator received no samples")
        return {name: value / self.samples for name, value in self.weighted.items()}


def deterministic_overfit_noise(
    batch: Mapping[str, Tensor],
    *,
    batch_index: int,
    seed: int,
    timestep: int = 8,
) -> tuple[Tensor, Tensor]:
    shape = (*batch["coarse_trajectories"].shape[:-1], 2)
    device = batch["coarse_trajectories"].device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + int(batch_index))
    noise = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    timesteps = torch.full(
        batch["gt_mode"].shape,
        int(timestep),
        device=device,
        dtype=torch.int64,
    )
    return noise, timesteps


def _training_amp(
    device: torch.device, enabled: bool
) -> contextlib.AbstractContextManager[Any]:
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=bool(enabled),
    )


def train_one_epoch(
    *,
    planner: BEVOnlyDiffusionPlanner,
    loss_module: JointStage1Loss,
    dataloader: Iterable[Mapping[str, Tensor]],
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    mixed_precision: bool,
    gradient_accumulation: int,
    optimizer_step: int,
    max_optimizer_steps: int | None,
    diagnostics_interval: int,
    overfit_seed: int | None,
    overfit_timestep: int = 8,
) -> tuple[dict[str, float], int, list[dict[str, float]]]:
    planner.train()
    optimizer.zero_grad(set_to_none=True)
    accumulator = MetricAccumulator()
    diagnostic_records: list[dict[str, float]] = []
    dataloader_length = len(dataloader)  # type: ignore[arg-type]
    for batch_index, cpu_batch in enumerate(dataloader):
        batch = move_joint_batch(cpu_batch, device)
        explicit_noise = explicit_timesteps = None
        if overfit_seed is not None:
            explicit_noise, explicit_timesteps = deterministic_overfit_noise(
                batch,
                batch_index=batch_index,
                seed=overfit_seed,
                timestep=overfit_timestep,
            )
        step_boundary = (
            batch_index + 1
        ) % gradient_accumulation == 0 or batch_index + 1 == dataloader_length
        next_step = optimizer_step + 1
        diagnostic_step = step_boundary and (
            next_step == 1 or next_step % diagnostics_interval == 0
        )
        with _training_amp(device, mixed_precision):
            output = planner_forward_from_batch(
                planner,
                batch,
                diffusion_noise=explicit_noise,
                diffusion_timesteps=explicit_timesteps,
            )
            loss_result = loss_from_batch(loss_module, output, batch)
        metrics = loss_result.scalar_metrics()
        accumulator.update(metrics, int(batch["bev"].shape[0]))
        if diagnostic_step:
            diagnostic_records.append(
                {
                    "optimizer_step": float(next_step),
                    **role_gradient_diagnostics(loss_result, planner),
                }
            )
        remainder = dataloader_length % gradient_accumulation
        final_window_size = remainder if remainder else gradient_accumulation
        current_window_size = (
            final_window_size
            if batch_index >= dataloader_length - final_window_size
            else gradient_accumulation
        )
        scaler.scale(loss_result.total / current_window_size).backward()
        if not step_boundary:
            continue
        scaler.unscale_(optimizer)
        gradient_metrics = module_gradient_norms(planner)
        if not any(value > 0.0 for value in gradient_metrics.values()):
            raise Stage1TrainingError("all Stage 1 gradient norms are zero")
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        diagnostic_records.append(
            {"optimizer_step": float(optimizer_step), **gradient_metrics}
        )
        if max_optimizer_steps is not None and optimizer_step >= max_optimizer_steps:
            break
    return accumulator.compute(), optimizer_step, diagnostic_records


@torch.no_grad()
def evaluate_formal(
    *,
    planner: BEVOnlyDiffusionPlanner,
    loss_module: JointStage1Loss,
    dataloader: Iterable[Mapping[str, Tensor]],
    device: torch.device,
    mixed_precision: bool,
) -> dict[str, float]:
    planner.eval()
    accumulator = MetricAccumulator()
    for cpu_batch in dataloader:
        batch = move_joint_batch(cpu_batch, device)
        with _training_amp(device, mixed_precision):
            output = planner_forward_from_batch(planner, batch)
            result = loss_from_batch(loss_module, output, batch)
        accumulator.update(result.scalar_metrics(), int(batch["bev"].shape[0]))
    return accumulator.compute()


@torch.no_grad()
def evaluate_overfit_fixed(
    *,
    planner: BEVOnlyDiffusionPlanner,
    loss_module: JointStage1Loss,
    dataloader: Iterable[Mapping[str, Tensor]],
    device: torch.device,
    mixed_precision: bool,
    seed: int,
    timestep: int,
) -> dict[str, float]:
    """Evaluate with fixed noise while restoring all BatchNorm statistics."""

    module_training_states = {
        module: bool(module.training) for module in planner.modules()
    }
    batch_norm_states = {
        module: (
            (
                module.running_mean.detach().clone()
                if module.running_mean is not None
                else None
            ),
            (
                module.running_var.detach().clone()
                if module.running_var is not None
                else None
            ),
            (
                module.num_batches_tracked.detach().clone()
                if module.num_batches_tracked is not None
                else None
            ),
        )
        for module in planner.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    }
    planner.train()
    accumulator = MetricAccumulator()
    try:
        for batch_index, cpu_batch in enumerate(dataloader):
            batch = move_joint_batch(cpu_batch, device)
            noise, timesteps = deterministic_overfit_noise(
                batch,
                batch_index=batch_index,
                seed=seed,
                timestep=timestep,
            )
            with _training_amp(device, mixed_precision):
                output = planner_forward_from_batch(
                    planner,
                    batch,
                    diffusion_noise=noise,
                    diffusion_timesteps=timesteps,
                )
                result = loss_from_batch(loss_module, output, batch)
            accumulator.update(
                result.scalar_metrics(),
                int(batch["bev"].shape[0]),
            )
    finally:
        for module, (running_mean, running_var, batches) in batch_norm_states.items():
            if running_mean is not None:
                module.running_mean.copy_(running_mean)
            if running_var is not None:
                module.running_var.copy_(running_var)
            if batches is not None:
                module.num_batches_tracked.copy_(batches)
        for module, training_state in module_training_states.items():
            module.training = training_state
    return accumulator.compute()


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def checkpoint_payload(
    *,
    planner: BEVOnlyDiffusionPlanner,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    config: Mapping[str, Any],
    dataset_fingerprint: str,
    epoch: int,
    optimizer_step: int,
    metrics: Mapping[str, float],
    diagnostic_only: bool,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "format": CHECKPOINT_FORMAT,
        "variant": "A",
        "joint_update": "joint_mean",
        "predecessor_condition": "none",
        "diagnostic_only": bool(diagnostic_only),
        "cross_split_overfit": bool(diagnostic_only),
        "eligible_for_formal_training": not bool(diagnostic_only),
        "dataset_fingerprint": str(dataset_fingerprint),
        "epoch": int(epoch),
        "optimizer_step": int(optimizer_step),
        "planner_config": _jsonable(planner.config),
        "training_config": _jsonable(dict(config)),
        "metrics": {name: float(value) for name, value in metrics.items()},
        "model_state": {
            name: value.detach().cpu() for name, value in planner.state_dict().items()
        },
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
    }


def save_stage1_checkpoint(path: Path | str, payload: Mapping[str, Any]) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(dict(payload), temporary_path)
    os.replace(temporary_path, checkpoint_path)
    return checkpoint_path


def load_stage1_checkpoint(
    path: Path | str,
    planner: BEVOnlyDiffusionPlanner,
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise Stage1TrainingError(
            f"unable to load checkpoint: {checkpoint_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise Stage1TrainingError("Stage 1 checkpoint must be a mapping")
    expected = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "format": CHECKPOINT_FORMAT,
        "variant": "A",
        "joint_update": "joint_mean",
        "predecessor_condition": "none",
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise Stage1TrainingError(f"Stage 1 checkpoint {name} mismatch")
    required_mappings = (
        "planner_config",
        "training_config",
        "metrics",
        "model_state",
        "optimizer_state",
        "scaler_state",
    )
    for name in required_mappings:
        if not isinstance(payload.get(name), Mapping):
            raise Stage1TrainingError(f"Stage 1 checkpoint {name} is invalid")
    for name in (
        "diagnostic_only",
        "cross_split_overfit",
        "eligible_for_formal_training",
    ):
        if not isinstance(payload.get(name), bool):
            raise Stage1TrainingError(f"Stage 1 checkpoint {name} is invalid")
    diagnostic_only = payload["diagnostic_only"]
    if (
        payload["cross_split_overfit"] is not diagnostic_only
        or payload["eligible_for_formal_training"] is diagnostic_only
    ):
        raise Stage1TrainingError("Stage 1 checkpoint diagnostic flags conflict")
    fingerprint = payload.get("dataset_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise Stage1TrainingError("Stage 1 checkpoint dataset fingerprint is invalid")
    try:
        int(fingerprint, 16)
    except ValueError as exc:
        raise Stage1TrainingError(
            "Stage 1 checkpoint dataset fingerprint is invalid"
        ) from exc
    for name in ("epoch", "optimizer_step"):
        value = payload.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise Stage1TrainingError(f"Stage 1 checkpoint {name} is invalid")
    state = payload.get("model_state")
    try:
        planner.load_state_dict(dict(state), strict=True)
    except RuntimeError as exc:
        raise Stage1TrainingError(
            "Stage 1 checkpoint does not strictly match the BEV planner"
        ) from exc
    return payload


def _write_json_line(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(dict(payload)), sort_keys=True) + "\n")


def _write_tensorboard(
    writer: SummaryWriter,
    metrics: Mapping[str, float],
    *,
    prefix: str,
    step: int,
) -> None:
    for name, value in metrics.items():
        writer.add_scalar(f"{prefix}/{name}", float(value), global_step=step)


def build_overfit_loader(
    dataset_root: Path | str,
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    datasets = [
        JointBEVDataset(JointBEVDatasetConfig(dataset_root, split))
        for split in ("train", "val", "test")
    ]
    combined = ConcatDataset(datasets)
    if len(combined) != 64:
        raise Stage1TrainingError(
            f"--overfit-64 requires exactly 64 joint samples, found {len(combined)}"
        )
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
    )


def _loss_config(config: Mapping[str, Any]) -> Stage1LossConfig:
    return Stage1LossConfig(**dict(_mapping(config, "loss")))


def _overfit_passed(
    initial_loss: float,
    current: Mapping[str, float],
    overfit_config: Mapping[str, Any],
) -> tuple[bool, float]:
    reduction = (initial_loss - current["loss/total"]) / max(
        abs(initial_loss), torch.finfo(torch.float32).eps
    )
    passed = (
        reduction >= float(overfit_config["min_loss_reduction"])
        and current["metric/mode_accuracy"]
        >= float(overfit_config["min_mode_accuracy"])
        and current["metric/gt_mode_ade"] <= float(overfit_config["max_gt_mode_ade_m"])
    )
    return bool(passed), float(reduction)


def run_stage1_training(
    config: Mapping[str, Any],
    *,
    output_root: Path | str,
    overfit_64: bool,
    max_optimizer_steps: int | None = None,
) -> Path:
    validate_stage1_config(config)
    training = _mapping(config, "training")
    dataset_config = _mapping(config, "dataset")
    overfit_config = _mapping(config, "overfit")
    seed = int(training["seed"])
    seed_everything(seed)
    device = resolve_device(str(training["device"]))
    mixed_precision = bool(training["mixed_precision"])
    if mixed_precision and device.type != "cuda":
        raise Stage1TrainingError("mixed precision is only supported for CUDA Stage 1")
    dataset_root = Path(str(dataset_config["root"])).expanduser()
    contract = validate_dataset_contract(dataset_root)
    dataset_fingerprint = str(contract["dataset_fingerprint"])
    run_dir = create_numbered_run_dir(output_root)
    (run_dir / "train_config.yaml").write_text(
        yaml.safe_dump(_jsonable(dict(config)), sort_keys=False),
        encoding="utf-8",
    )
    metadata = {
        "variant": "A",
        "joint_update": "joint_mean",
        "predecessor_condition": "none",
        "diagnostic_only": bool(overfit_64),
        "cross_split_overfit": bool(overfit_64),
        "eligible_for_formal_training": not bool(overfit_64),
        "dataset_fingerprint": dataset_fingerprint,
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    planner = BEVOnlyDiffusionPlanner().to(device)
    loss_module = JointStage1Loss(_loss_config(config)).to(device)
    optimizer = build_stage1_optimizer(
        planner,
        planner_lr=float(training["planner_lr"]),
        backbone_lr=float(training["backbone_lr"]),
        weight_decay=float(training["weight_decay"]),
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=mixed_precision,
        init_scale=float(training["grad_scaler_init_scale"]),
    )
    batch_size = int(training["batch_size"])
    num_workers = int(training["num_workers"])
    if overfit_64:
        train_loader = build_overfit_loader(
            dataset_root, batch_size=batch_size, num_workers=num_workers
        )
        val_loader = None
        hard_max_steps = int(overfit_config["max_optimizer_steps"])
        if max_optimizer_steps is not None:
            hard_max_steps = min(hard_max_steps, int(max_optimizer_steps))
    else:
        train_loader = build_joint_bev_dataloader(
            dataset_root,
            str(dataset_config["train_split"]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            seed=seed,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        val_loader = build_joint_bev_dataloader(
            dataset_root,
            str(dataset_config["val_split"]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            seed=seed,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        hard_max_steps = max_optimizer_steps

    metrics_path = run_dir / "metrics.jsonl"
    writer = SummaryWriter(str(run_dir / "tb"))
    optimizer_step = 0
    best_val = math.inf
    initial_overfit_loss: float | None = None
    final_metrics: dict[str, float] = {}
    passed = False
    max_epochs = int(training["max_epochs"])
    if overfit_64:
        steps_per_epoch = math.ceil(
            len(train_loader) / int(training["gradient_accumulation"])
        )
        max_epochs = math.ceil(hard_max_steps / max(steps_per_epoch, 1))
        final_metrics = evaluate_overfit_fixed(
            planner=planner,
            loss_module=loss_module,
            dataloader=train_loader,
            device=device,
            mixed_precision=mixed_precision,
            seed=seed,
            timestep=int(overfit_config["timestep"]),
        )
        initial_overfit_loss = float(final_metrics["loss/total"])
        final_metrics["metric/loss_reduction"] = 0.0
        _write_json_line(
            metrics_path,
            {
                "epoch": -1,
                "optimizer_step": 0,
                "overfit_baseline": final_metrics,
            },
        )
        _write_tensorboard(writer, final_metrics, prefix="overfit", step=0)
    try:
        for epoch in range(max_epochs):
            if not overfit_64:
                batch_sampler = getattr(train_loader, "batch_sampler", None)
                set_epoch = getattr(batch_sampler, "set_epoch", None)
                if callable(set_epoch):
                    set_epoch(epoch)
            train_metrics, optimizer_step, diagnostics = train_one_epoch(
                planner=planner,
                loss_module=loss_module,
                dataloader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                mixed_precision=mixed_precision,
                gradient_accumulation=int(training["gradient_accumulation"]),
                optimizer_step=optimizer_step,
                max_optimizer_steps=hard_max_steps,
                diagnostics_interval=int(training["gradient_diagnostics_interval"]),
                overfit_seed=seed if overfit_64 else None,
                overfit_timestep=int(overfit_config["timestep"]),
            )
            payload: dict[str, Any] = {
                "epoch": epoch,
                "optimizer_step": optimizer_step,
                "train": train_metrics,
                "diagnostics": diagnostics,
            }
            _write_tensorboard(
                writer, train_metrics, prefix="train", step=optimizer_step
            )
            for record in diagnostics:
                _write_tensorboard(
                    writer,
                    {
                        name: value
                        for name, value in record.items()
                        if name != "optimizer_step"
                    },
                    prefix="diagnostics",
                    step=int(record["optimizer_step"]),
                )
            if overfit_64:
                evaluation_interval = int(overfit_config["evaluation_interval_steps"])
                should_evaluate = (
                    optimizer_step >= hard_max_steps
                    or optimizer_step % evaluation_interval == 0
                )
                if should_evaluate:
                    final_metrics = evaluate_overfit_fixed(
                        planner=planner,
                        loss_module=loss_module,
                        dataloader=train_loader,
                        device=device,
                        mixed_precision=mixed_precision,
                        seed=seed,
                        timestep=int(overfit_config["timestep"]),
                    )
                    if initial_overfit_loss is None:
                        raise Stage1TrainingError(
                            "overfit baseline was not initialized"
                        )
                    passed, reduction = _overfit_passed(
                        initial_overfit_loss,
                        final_metrics,
                        overfit_config,
                    )
                    final_metrics["metric/loss_reduction"] = reduction
                    payload["overfit"] = final_metrics
                    payload["overfit_passed"] = passed
                    _write_tensorboard(
                        writer,
                        final_metrics,
                        prefix="overfit",
                        step=optimizer_step,
                    )
            else:
                if val_loader is None:
                    raise Stage1TrainingError("formal validation loader is missing")
                val_metrics = evaluate_formal(
                    planner=planner,
                    loss_module=loss_module,
                    dataloader=val_loader,
                    device=device,
                    mixed_precision=mixed_precision,
                )
                payload["val"] = val_metrics
                _write_tensorboard(
                    writer, val_metrics, prefix="val", step=optimizer_step
                )
                final_metrics = val_metrics
                if val_metrics["loss/total"] < best_val:
                    best_val = val_metrics["loss/total"]
                    best_payload = checkpoint_payload(
                        planner=planner,
                        optimizer=optimizer,
                        scaler=scaler,
                        config=config,
                        dataset_fingerprint=dataset_fingerprint,
                        epoch=epoch,
                        optimizer_step=optimizer_step,
                        metrics=val_metrics,
                        diagnostic_only=False,
                    )
                    save_stage1_checkpoint(
                        run_dir / "checkpoints" / "best.pt", best_payload
                    )
            _write_json_line(metrics_path, payload)
            checkpoint = checkpoint_payload(
                planner=planner,
                optimizer=optimizer,
                scaler=scaler,
                config=config,
                dataset_fingerprint=dataset_fingerprint,
                epoch=epoch,
                optimizer_step=optimizer_step,
                metrics=final_metrics,
                diagnostic_only=overfit_64,
            )
            checkpoint_name = "diagnostic.pt" if overfit_64 else "last.pt"
            save_stage1_checkpoint(
                run_dir / "checkpoints" / checkpoint_name, checkpoint
            )
            print(
                f"[stage1-A] epoch={epoch} step={optimizer_step} "
                f"loss={train_metrics['loss/total']:.6f} "
                f"mode_acc={final_metrics.get('metric/mode_accuracy', train_metrics['metric/mode_accuracy']):.4f} "
                f"gt_ade={final_metrics.get('metric/gt_mode_ade', train_metrics['metric/gt_mode_ade']):.4f}",
                flush=True,
            )
            if passed:
                break
            if hard_max_steps is not None and optimizer_step >= hard_max_steps:
                break
    finally:
        writer.close()
    summary = {
        **metadata,
        "optimizer_steps": optimizer_step,
        "metrics": final_metrics,
        "overfit_passed": passed if overfit_64 else None,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if overfit_64 and not passed:
        raise Stage1TrainingError(
            "64-sample overfit gate failed: "
            f"reduction={final_metrics.get('metric/loss_reduction', float('nan')):.4f}, "
            f"mode_accuracy={final_metrics.get('metric/mode_accuracy', float('nan')):.4f}, "
            f"gt_mode_ade={final_metrics.get('metric/gt_mode_ade', float('nan')):.4f}; "
            f"report={run_dir / 'summary.json'}"
        )
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Stage 1 A joint-mean BEV diffusion planner."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--max-optimizer-steps", type=int, default=None)
    parser.add_argument("--overfit-64", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_stage1_config(args.config)
    if args.dataset_root is not None:
        config["dataset"]["root"] = str(args.dataset_root.expanduser().resolve())
    if args.device is not None:
        config["training"]["device"] = args.device
        if args.device == "cpu":
            config["training"]["mixed_precision"] = False
    validate_stage1_config(config)
    run_dir = run_stage1_training(
        config,
        output_root=args.output_root,
        overfit_64=bool(args.overfit_64),
        max_optimizer_steps=args.max_optimizer_steps,
    )
    print(f"[stage1-A] outputs={run_dir}", flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_SCHEMA_VERSION",
    "Stage1TrainingError",
    "build_overfit_loader",
    "build_stage1_optimizer",
    "checkpoint_payload",
    "create_numbered_run_dir",
    "deterministic_overfit_noise",
    "evaluate_overfit_fixed",
    "load_stage1_checkpoint",
    "load_stage1_config",
    "loss_from_batch",
    "module_gradient_norms",
    "planner_forward_from_batch",
    "resolve_device",
    "role_gradient_diagnostics",
    "run_stage1_training",
    "save_stage1_checkpoint",
    "train_one_epoch",
    "validate_stage1_config",
]
