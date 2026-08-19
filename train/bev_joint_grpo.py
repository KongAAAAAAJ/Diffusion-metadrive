"""Checkpoint and Stage 1 loading helpers for Variant-A/B joint GRPO."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from models.bev_planner import (
    BEVOnlyDiffusionPlanner,
    BEVOnlyDiffusionPlannerConfig,
    JointGRPOConfig,
    JointGRPOError,
    JointGRPOTrainerA,
    JointGRPOTrainerB,
)
from train.train_bev_diffusion_stage1 import (
    CHECKPOINT_FORMAT as STAGE1_CHECKPOINT_FORMAT,
    CHECKPOINT_SCHEMA_VERSION as STAGE1_CHECKPOINT_SCHEMA_VERSION,
    Stage1TrainingError,
    load_stage1_checkpoint,
)


GRPO_CHECKPOINT_SCHEMA_VERSION = 1
GRPO_CHECKPOINT_FORMAT = "bev_joint_grpo_a_v1"
GRPO_B_CHECKPOINT_FORMAT = "bev_joint_grpo_b_v1"


def _variant_contract(variant: str) -> tuple[str, str]:
    if variant == "A":
        return "none", GRPO_CHECKPOINT_FORMAT
    if variant == "B":
        return "predicted_detached", GRPO_B_CHECKPOINT_FORMAT
    raise JointGRPOError("GRPO variant must be A or B")


def _validate_stage1_source_metadata(
    payload: Mapping[str, Any],
    *,
    variant: str,
    allow_diagnostic_source: bool,
) -> None:
    condition, _ = _variant_contract(variant)
    if not isinstance(payload, Mapping):
        raise JointGRPOError("Stage 1 source metadata must be a mapping")
    expected = {
        "schema_version": STAGE1_CHECKPOINT_SCHEMA_VERSION,
        "format": STAGE1_CHECKPOINT_FORMAT,
        "variant": variant,
        "predecessor_condition": condition,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise JointGRPOError(f"Stage 1 source {name} mismatch")
    source_eligible = payload.get("eligible_for_formal_training")
    diagnostic = payload.get("diagnostic_only")
    if not isinstance(source_eligible, bool) or not isinstance(diagnostic, bool):
        raise JointGRPOError("Stage 1 source eligibility metadata is invalid")
    if source_eligible is diagnostic:
        raise JointGRPOError("Stage 1 source eligibility metadata conflicts")
    if not source_eligible and not bool(allow_diagnostic_source):
        raise JointGRPOError(
            "formal GRPO requires an eligible Stage 1 checkpoint; "
            "diagnostic sources require explicit opt-in"
        )
    fingerprint = payload.get("dataset_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or fingerprint != fingerprint.lower()
    ):
        raise JointGRPOError("Stage 1 source dataset fingerprint is invalid")
    try:
        int(fingerprint, 16)
    except ValueError as exc:
        raise JointGRPOError(
            "Stage 1 source dataset fingerprint is invalid"
        ) from exc


def validate_stage1_a_source_metadata(
    payload: Mapping[str, Any],
    *,
    allow_diagnostic_source: bool,
) -> None:
    """Validate the immutable Stage 1 source boundary before GRPO starts."""

    _validate_stage1_source_metadata(
        payload,
        variant="A",
        allow_diagnostic_source=allow_diagnostic_source,
    )


def validate_stage1_b_source_metadata(
    payload: Mapping[str, Any],
    *,
    allow_diagnostic_source: bool,
) -> None:
    """Validate the immutable Stage 1 B source boundary."""

    _validate_stage1_source_metadata(
        payload,
        variant="B",
        allow_diagnostic_source=allow_diagnostic_source,
    )


def sha256_checkpoint(path: Path | str) -> str:
    checkpoint = Path(path)
    digest = hashlib.sha256()
    try:
        with checkpoint.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise JointGRPOError(f"unable to hash checkpoint: {checkpoint}") from exc
    return digest.hexdigest()


def _load_stage1_for_grpo(
    path: Path | str,
    *,
    variant: str,
    device: torch.device,
    config: JointGRPOConfig | None = None,
    allow_diagnostic_source: bool = False,
) -> tuple[JointGRPOTrainerA | JointGRPOTrainerB, dict[str, Any], str]:
    condition, _ = _variant_contract(variant)
    checkpoint_path = Path(path)
    try:
        source_preview = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise JointGRPOError("unable to inspect the Stage 1 source checkpoint") from exc
    planner_config = (
        source_preview.get("planner_config", {})
        if isinstance(source_preview, Mapping)
        else {}
    )
    model_version = (
        str(planner_config.get("model_version", "v1"))
        if isinstance(planner_config, Mapping)
        else "v1"
    )
    planner = BEVOnlyDiffusionPlanner(
        BEVOnlyDiffusionPlannerConfig(
            predecessor_condition=condition,
            model_version=model_version,
        )
    )
    try:
        payload = load_stage1_checkpoint(checkpoint_path, planner)
    except Stage1TrainingError as exc:
        raise JointGRPOError(
            f"unable to load the Stage 1 {variant} source checkpoint"
        ) from exc
    _validate_stage1_source_metadata(
        payload,
        variant=variant,
        allow_diagnostic_source=allow_diagnostic_source,
    )
    planner.to(device)
    trainer = (
        JointGRPOTrainerA(planner, config)
        if variant == "A"
        else JointGRPOTrainerB(planner, config)
    )
    return trainer, payload, sha256_checkpoint(checkpoint_path)


def load_stage1_a_for_grpo(
    path: Path | str,
    *,
    device: torch.device,
    config: JointGRPOConfig | None = None,
    allow_diagnostic_source: bool = False,
) -> tuple[JointGRPOTrainerA, dict[str, Any], str]:
    trainer, payload, digest = _load_stage1_for_grpo(
        path,
        variant="A",
        device=device,
        config=config,
        allow_diagnostic_source=allow_diagnostic_source,
    )
    if not isinstance(trainer, JointGRPOTrainerA):
        raise JointGRPOError("Stage 1 A loader constructed the wrong trainer")
    return trainer, payload, digest


def load_stage1_b_for_grpo(
    path: Path | str,
    *,
    device: torch.device,
    config: JointGRPOConfig | None = None,
    allow_diagnostic_source: bool = False,
) -> tuple[JointGRPOTrainerB, dict[str, Any], str]:
    trainer, payload, digest = _load_stage1_for_grpo(
        path,
        variant="B",
        device=device,
        config=config,
        allow_diagnostic_source=allow_diagnostic_source,
    )
    if not isinstance(trainer, JointGRPOTrainerB):
        raise JointGRPOError("Stage 1 B loader constructed the wrong trainer")
    return trainer, payload, digest


def _grpo_checkpoint_payload(
    *,
    trainer: JointGRPOTrainerA | JointGRPOTrainerB,
    variant: str,
    source_stage1_sha256: str,
    source_stage1_payload: Mapping[str, Any],
    metrics: Mapping[str, float],
    diagnostic_only: bool,
    scaler_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    condition, checkpoint_format = _variant_contract(variant)
    expected_type = JointGRPOTrainerA if variant == "A" else JointGRPOTrainerB
    if not isinstance(trainer, expected_type):
        raise JointGRPOError("GRPO checkpoint trainer variant mismatch")
    if not isinstance(diagnostic_only, bool):
        raise JointGRPOError("diagnostic_only must be boolean")
    if scaler_state is not None and not isinstance(scaler_state, Mapping):
        raise JointGRPOError("scaler_state must be a mapping")
    if (
        not isinstance(source_stage1_sha256, str)
        or len(source_stage1_sha256) != 64
        or source_stage1_sha256 != source_stage1_sha256.lower()
    ):
        raise JointGRPOError("source_stage1_sha256 must be a SHA256 hex digest")
    try:
        int(source_stage1_sha256, 16)
    except ValueError as exc:
        raise JointGRPOError(
            "source_stage1_sha256 must be a SHA256 hex digest"
        ) from exc
    _validate_stage1_source_metadata(
        source_stage1_payload,
        variant=variant,
        allow_diagnostic_source=True,
    )
    if (
        not diagnostic_only
        and source_stage1_payload.get("eligible_for_formal_training") is not True
    ):
        raise JointGRPOError(
            "a diagnostic Stage 1 source cannot produce a formal GRPO checkpoint"
        )
    fingerprint = source_stage1_payload.get("dataset_fingerprint")
    checked_metrics: dict[str, float] = {}
    for name, value in metrics.items():
        if (
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise JointGRPOError("GRPO checkpoint metrics must be finite scalars")
        checked_metrics[name] = float(value)
    return {
        "schema_version": GRPO_CHECKPOINT_SCHEMA_VERSION,
        "format": checkpoint_format,
        "variant": variant,
        "predecessor_condition": condition,
        "reward_source": "external",
        "source_stage1_sha256": source_stage1_sha256,
        "source_dataset_fingerprint": fingerprint,
        "diagnostic_only": bool(diagnostic_only),
        "eligible_for_formal_training": not bool(diagnostic_only),
        "optimizer_step": int(trainer.optimizer_step),
        "grpo_config": dataclasses.asdict(trainer.config),
        "metrics": checked_metrics,
        "planner_state": {
            name: value.detach().cpu()
            for name, value in trainer.planner.state_dict().items()
        },
        "reference_state": {
            name: value.detach().cpu()
            for name, value in trainer.reference.state_dict().items()
        },
        "optimizer_state": trainer.optimizer.state_dict(),
        "scaler_state": dict(scaler_state or {}),
    }


def grpo_checkpoint_payload(
    *,
    trainer: JointGRPOTrainerA,
    source_stage1_sha256: str,
    source_stage1_payload: Mapping[str, Any],
    metrics: Mapping[str, float],
    diagnostic_only: bool,
    scaler_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _grpo_checkpoint_payload(
        trainer=trainer,
        variant="A",
        source_stage1_sha256=source_stage1_sha256,
        source_stage1_payload=source_stage1_payload,
        metrics=metrics,
        diagnostic_only=diagnostic_only,
        scaler_state=scaler_state,
    )


def grpo_b_checkpoint_payload(
    *,
    trainer: JointGRPOTrainerB,
    source_stage1_sha256: str,
    source_stage1_payload: Mapping[str, Any],
    metrics: Mapping[str, float],
    diagnostic_only: bool,
    scaler_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _grpo_checkpoint_payload(
        trainer=trainer,
        variant="B",
        source_stage1_sha256=source_stage1_sha256,
        source_stage1_payload=source_stage1_payload,
        metrics=metrics,
        diagnostic_only=diagnostic_only,
        scaler_state=scaler_state,
    )


def save_grpo_checkpoint(
    path: Path | str, payload: Mapping[str, Any]
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, checkpoint_path)
    return checkpoint_path


def _load_grpo_checkpoint(
    path: Path | str,
    trainer: JointGRPOTrainerA | JointGRPOTrainerB,
    *,
    variant: str,
    expected_source_stage1_sha256: str,
) -> dict[str, Any]:
    condition, checkpoint_format = _variant_contract(variant)
    expected_type = JointGRPOTrainerA if variant == "A" else JointGRPOTrainerB
    if not isinstance(trainer, expected_type):
        raise JointGRPOError("GRPO checkpoint trainer variant mismatch")
    if (
        not isinstance(expected_source_stage1_sha256, str)
        or len(expected_source_stage1_sha256) != 64
        or expected_source_stage1_sha256
        != expected_source_stage1_sha256.lower()
    ):
        raise JointGRPOError(
            "expected_source_stage1_sha256 must be a SHA256 hex digest"
        )
    try:
        int(expected_source_stage1_sha256, 16)
    except ValueError as exc:
        raise JointGRPOError(
            "expected_source_stage1_sha256 must be a SHA256 hex digest"
        ) from exc
    checkpoint_path = Path(path)
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise JointGRPOError(f"unable to load GRPO checkpoint: {checkpoint_path}") from exc
    if not isinstance(payload, dict):
        raise JointGRPOError("GRPO checkpoint must be a mapping")
    expected = {
        "schema_version": GRPO_CHECKPOINT_SCHEMA_VERSION,
        "format": checkpoint_format,
        "variant": variant,
        "predecessor_condition": condition,
        "reward_source": "external",
        "source_stage1_sha256": expected_source_stage1_sha256,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise JointGRPOError(f"GRPO checkpoint {name} mismatch")
    for name in ("diagnostic_only", "eligible_for_formal_training"):
        if not isinstance(payload.get(name), bool):
            raise JointGRPOError(f"GRPO checkpoint {name} is invalid")
    if payload["eligible_for_formal_training"] is payload["diagnostic_only"]:
        raise JointGRPOError("GRPO checkpoint eligibility flags conflict")
    fingerprint = payload.get("source_dataset_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or fingerprint != fingerprint.lower()
    ):
        raise JointGRPOError(
            "GRPO checkpoint source_dataset_fingerprint is invalid"
        )
    try:
        int(fingerprint, 16)
    except ValueError as exc:
        raise JointGRPOError(
            "GRPO checkpoint source_dataset_fingerprint is invalid"
        ) from exc
    step = payload.get("optimizer_step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise JointGRPOError("GRPO checkpoint optimizer_step is invalid")
    if payload.get("grpo_config") != dataclasses.asdict(trainer.config):
        raise JointGRPOError("GRPO checkpoint config mismatch")
    for name in (
        "metrics",
        "planner_state",
        "reference_state",
        "optimizer_state",
        "scaler_state",
    ):
        if not isinstance(payload.get(name), Mapping):
            raise JointGRPOError(f"GRPO checkpoint {name} is invalid")
    metrics = payload["metrics"]
    if any(
        not isinstance(name, str)
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for name, value in metrics.items()
    ):
        raise JointGRPOError("GRPO checkpoint metrics must be finite scalars")
    try:
        trainer.planner.load_state_dict(dict(payload["planner_state"]), strict=True)
        trainer.reference.load_state_dict(
            dict(payload["reference_state"]), strict=True
        )
        trainer.optimizer.load_state_dict(dict(payload["optimizer_state"]))
    except (RuntimeError, ValueError, KeyError) as exc:
        raise JointGRPOError(
            "GRPO checkpoint does not strictly match the trainer"
        ) from exc
    trainer.reference.requires_grad_(False)
    trainer.reference.eval()
    trainer.planner.eval()
    trainer.optimizer_step = int(step)
    return payload


def load_grpo_checkpoint(
    path: Path | str,
    trainer: JointGRPOTrainerA,
    *,
    expected_source_stage1_sha256: str,
) -> dict[str, Any]:
    return _load_grpo_checkpoint(
        path,
        trainer,
        variant="A",
        expected_source_stage1_sha256=expected_source_stage1_sha256,
    )


def load_grpo_b_checkpoint(
    path: Path | str,
    trainer: JointGRPOTrainerB,
    *,
    expected_source_stage1_sha256: str,
) -> dict[str, Any]:
    return _load_grpo_checkpoint(
        path,
        trainer,
        variant="B",
        expected_source_stage1_sha256=expected_source_stage1_sha256,
    )


__all__ = [
    "GRPO_B_CHECKPOINT_FORMAT",
    "GRPO_CHECKPOINT_FORMAT",
    "GRPO_CHECKPOINT_SCHEMA_VERSION",
    "grpo_b_checkpoint_payload",
    "grpo_checkpoint_payload",
    "load_grpo_b_checkpoint",
    "load_grpo_checkpoint",
    "load_stage1_a_for_grpo",
    "load_stage1_b_for_grpo",
    "save_grpo_checkpoint",
    "sha256_checkpoint",
    "validate_stage1_a_source_metadata",
    "validate_stage1_b_source_metadata",
]
