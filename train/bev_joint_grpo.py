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
    JointGRPOConfig,
    JointGRPOError,
    JointGRPOTrainerA,
    JointGRPOTrainerB,
    joint_grpo_optimizer_contract,
)
from train.train_bev_diffusion_stage1 import (
    CHECKPOINT_FORMAT as STAGE1_CHECKPOINT_FORMAT,
    CHECKPOINT_SCHEMA_VERSION as STAGE1_CHECKPOINT_SCHEMA_VERSION,
    Stage1TrainingError,
    load_stage1_checkpoint,
    stage1_planner_config_from_mapping,
)


GRPO_CHECKPOINT_SCHEMA_VERSION = 3
GRPO_CHECKPOINT_FORMAT = "bev_joint_grpo_a_v3"
GRPO_B_CHECKPOINT_FORMAT = "bev_joint_grpo_b_v3"
GRPO_REWARD_SOURCE = "external_per_vehicle_same_mode_counterfactual"
PREVIOUS_GRPO_CHECKPOINT_SCHEMA_VERSION = 2
PREVIOUS_GRPO_CHECKPOINT_FORMAT = "bev_joint_grpo_a_v2"
PREVIOUS_GRPO_B_CHECKPOINT_FORMAT = "bev_joint_grpo_b_v2"
PREVIOUS_GRPO_REWARD_SOURCE = "external_with_explicit_pretrain_baseline"
LEGACY_GRPO_CHECKPOINT_SCHEMA_VERSION = 1
LEGACY_GRPO_CHECKPOINT_FORMAT = "bev_joint_grpo_a_v1"
LEGACY_GRPO_REWARD_SOURCE = "external"


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
    if not isinstance(source_preview, Mapping):
        raise JointGRPOError("Stage 1 source checkpoint must be a mapping")
    planner_config = source_preview.get("planner_config")
    if (
        source_preview.get("model_version") != "v2"
        or not isinstance(planner_config, Mapping)
        or planner_config.get("model_version") != "v2"
    ):
        raise JointGRPOError(
            "GRPO requires an explicit v2 Stage 1 source checkpoint"
        )
    try:
        checkpoint_config = stage1_planner_config_from_mapping(planner_config)
    except Stage1TrainingError as exc:
        raise JointGRPOError("Stage 1 source planner_config is invalid") from exc
    if checkpoint_config.predecessor_condition != condition:
        raise JointGRPOError("Stage 1 source predecessor condition mismatch")
    planner = BEVOnlyDiffusionPlanner(checkpoint_config)
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
        "reward_source": GRPO_REWARD_SOURCE,
        "optimizer_contract_version": joint_grpo_optimizer_contract()[
            "version"
        ],
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


def _load_grpo_payload(path: Path | str) -> Mapping[str, Any]:
    checkpoint_path = Path(path)
    try:
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise JointGRPOError(
            f"unable to load GRPO checkpoint: {checkpoint_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise JointGRPOError("GRPO checkpoint must be a mapping")
    return payload


def _grpo_config_from_payload(payload: Mapping[str, Any]) -> JointGRPOConfig:
    raw_config = payload.get("grpo_config")
    if not isinstance(raw_config, Mapping):
        raise JointGRPOError("GRPO checkpoint config must be a mapping")

    default_values = dataclasses.asdict(JointGRPOConfig())
    if set(raw_config) != set(default_values) or any(
        type(raw_config[name]) is not type(default_values[name])
        for name in default_values
    ):
        raise JointGRPOError("GRPO checkpoint config is invalid")
    try:
        config = JointGRPOConfig(**dict(raw_config))
    except (TypeError, ValueError, JointGRPOError) as exc:
        raise JointGRPOError("GRPO checkpoint config is invalid") from exc
    if dict(raw_config) != dataclasses.asdict(config):
        raise JointGRPOError("GRPO checkpoint config is invalid")
    return config


def load_grpo_config_from_checkpoint(
    path: Path | str,
) -> JointGRPOConfig:
    """Rebuild a current config for strict training/checkpoint resume."""

    payload = _load_grpo_payload(path)
    if payload.get("schema_version") != GRPO_CHECKPOINT_SCHEMA_VERSION:
        raise JointGRPOError("GRPO checkpoint schema_version mismatch")
    if payload.get("format") not in {
        GRPO_CHECKPOINT_FORMAT,
        GRPO_B_CHECKPOINT_FORMAT,
    }:
        raise JointGRPOError("GRPO checkpoint format mismatch")
    return _grpo_config_from_payload(payload)


def _validate_grpo_a_evaluation_identity(payload: Mapping[str, Any]) -> str:
    current_identity = {
        "schema_version": GRPO_CHECKPOINT_SCHEMA_VERSION,
        "format": GRPO_CHECKPOINT_FORMAT,
        "variant": "A",
        "predecessor_condition": "none",
        "reward_source": GRPO_REWARD_SOURCE,
        "optimizer_contract_version": joint_grpo_optimizer_contract()["version"],
    }
    legacy_identity = {
        "schema_version": LEGACY_GRPO_CHECKPOINT_SCHEMA_VERSION,
        "format": LEGACY_GRPO_CHECKPOINT_FORMAT,
        "variant": "A",
        "predecessor_condition": "none",
        "reward_source": LEGACY_GRPO_REWARD_SOURCE,
    }
    if all(payload.get(name) == value for name, value in current_identity.items()):
        return "current"
    if (
        payload.get("schema_version") == PREVIOUS_GRPO_CHECKPOINT_SCHEMA_VERSION
        and payload.get("format") == PREVIOUS_GRPO_CHECKPOINT_FORMAT
        and payload.get("variant") == "A"
        and payload.get("predecessor_condition") == "none"
        and payload.get("reward_source") == PREVIOUS_GRPO_REWARD_SOURCE
        and payload.get("optimizer_contract_version")
        in {
            "stage2_joint_grpo_optimizer_v3",
            "stage2_joint_grpo_optimizer_v4",
        }
    ):
        return "previous"
    if all(payload.get(name) == value for name, value in legacy_identity.items()):
        return "legacy"
    raise JointGRPOError("GRPO evaluation checkpoint identity mismatch")


def _historical_grpo_config_for_evaluation(
    payload: Mapping[str, Any],
) -> JointGRPOConfig:
    """Translate only the sampling size needed to inspect historical planners."""

    raw_config = payload.get("grpo_config")
    if not isinstance(raw_config, Mapping):
        raise JointGRPOError("GRPO checkpoint config must be a mapping")
    values = dataclasses.asdict(JointGRPOConfig())
    for name, default in tuple(values.items()):
        if name in raw_config:
            value = raw_config[name]
            if type(value) is not type(default):
                raise JointGRPOError("historical GRPO checkpoint config is invalid")
            values[name] = value
    if "trajectories_per_mode" not in raw_config:
        old_size = raw_config.get("group_size")
        if isinstance(old_size, bool) or not isinstance(old_size, int) or old_size < 2:
            raise JointGRPOError("historical GRPO checkpoint config is invalid")
        values["trajectories_per_mode"] = old_size
    try:
        return JointGRPOConfig(**values)
    except (TypeError, ValueError, JointGRPOError) as exc:
        raise JointGRPOError("historical GRPO checkpoint config is invalid") from exc


def load_grpo_a_config_for_evaluation(
    path: Path | str,
) -> JointGRPOConfig:
    """Read an exact current or historical Variant-A config for evaluation."""

    payload = _load_grpo_payload(path)
    generation = _validate_grpo_a_evaluation_identity(payload)
    if generation == "current":
        return _grpo_config_from_payload(payload)
    return _historical_grpo_config_for_evaluation(payload)


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
        "reward_source": GRPO_REWARD_SOURCE,
        "optimizer_contract_version": joint_grpo_optimizer_contract()[
            "version"
        ],
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


def load_grpo_a_checkpoint_for_evaluation(
    path: Path | str,
    trainer: JointGRPOTrainerA,
    *,
    expected_source_stage1_sha256: str,
) -> dict[str, Any]:
    """Load only Variant-A planner weights for current or historical evaluation."""

    if not isinstance(trainer, JointGRPOTrainerA):
        raise JointGRPOError("GRPO evaluation checkpoint trainer variant mismatch")
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

    payload = _load_grpo_payload(path)
    generation = _validate_grpo_a_evaluation_identity(payload)
    if payload.get("source_stage1_sha256") != expected_source_stage1_sha256:
        raise JointGRPOError("GRPO evaluation checkpoint source_stage1_sha256 mismatch")
    for name in ("diagnostic_only", "eligible_for_formal_training"):
        if not isinstance(payload.get(name), bool):
            raise JointGRPOError(f"GRPO evaluation checkpoint {name} is invalid")
    if payload["eligible_for_formal_training"] is payload["diagnostic_only"]:
        raise JointGRPOError("GRPO evaluation checkpoint eligibility flags conflict")
    fingerprint = payload.get("source_dataset_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or fingerprint != fingerprint.lower()
    ):
        raise JointGRPOError(
            "GRPO evaluation checkpoint source_dataset_fingerprint is invalid"
        )
    try:
        int(fingerprint, 16)
    except ValueError as exc:
        raise JointGRPOError(
            "GRPO evaluation checkpoint source_dataset_fingerprint is invalid"
        ) from exc
    if not isinstance(payload.get("planner_state"), Mapping):
        raise JointGRPOError("GRPO evaluation checkpoint planner_state is invalid")
    if generation == "current":
        step = payload.get("optimizer_step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise JointGRPOError("GRPO evaluation checkpoint optimizer_step is invalid")
        if payload.get("grpo_config") != dataclasses.asdict(trainer.config):
            raise JointGRPOError("GRPO evaluation checkpoint config mismatch")
        for name in ("metrics", "reference_state", "optimizer_state", "scaler_state"):
            if not isinstance(payload.get(name), Mapping):
                raise JointGRPOError(f"GRPO evaluation checkpoint {name} is invalid")
        metrics = payload["metrics"]
        if any(
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for name, value in metrics.items()
        ):
            raise JointGRPOError(
                "GRPO evaluation checkpoint metrics must be finite scalars"
            )
    try:
        trainer.planner.load_state_dict(
            dict(payload["planner_state"]), strict=True
        )
    except (RuntimeError, ValueError, KeyError) as exc:
        raise JointGRPOError(
            "GRPO evaluation checkpoint does not strictly match the planner"
        ) from exc
    trainer.planner.eval()
    return dict(payload)


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
    "LEGACY_GRPO_CHECKPOINT_FORMAT",
    "LEGACY_GRPO_CHECKPOINT_SCHEMA_VERSION",
    "PREVIOUS_GRPO_B_CHECKPOINT_FORMAT",
    "PREVIOUS_GRPO_CHECKPOINT_FORMAT",
    "PREVIOUS_GRPO_CHECKPOINT_SCHEMA_VERSION",
    "grpo_b_checkpoint_payload",
    "grpo_checkpoint_payload",
    "load_grpo_a_checkpoint_for_evaluation",
    "load_grpo_a_config_for_evaluation",
    "load_grpo_b_checkpoint",
    "load_grpo_checkpoint",
    "load_grpo_config_from_checkpoint",
    "load_stage1_a_for_grpo",
    "load_stage1_b_for_grpo",
    "save_grpo_checkpoint",
    "sha256_checkpoint",
    "validate_stage1_a_source_metadata",
    "validate_stage1_b_source_metadata",
]
