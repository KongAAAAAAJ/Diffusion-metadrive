"""Strict shared manifest for single- and multi-model BEV evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


MANIFEST_FORMAT = "bev_model_evaluation_manifest_v2"
MODEL_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ModelManifestError(ValueError):
    """Raised when an evaluation manifest violates the v2 contract."""


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    kind: str
    variant: str
    checkpoint: Path
    checkpoint_sha256: str
    reward_domain: str | None = None
    source_checkpoint: Path | None = None
    source_checkpoint_sha256: str | None = None


@dataclass(frozen=True)
class ComparisonSpec:
    comparison_id: str
    baseline: str
    candidate: str


@dataclass(frozen=True)
class ModelEvaluationManifest:
    path: Path
    models: tuple[ModelSpec, ...]
    comparisons: tuple[ComparisonSpec, ...]

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(model.model_id for model in self.models)

    def model_map(self) -> dict[str, ModelSpec]:
        return {model.model_id: model for model in self.models}


def file_sha256(path: Path) -> str:
    try:
        stream = path.open("rb")
    except OSError as exc:
        raise ModelManifestError(f"unable to read artifact: {path}") from exc
    digest = hashlib.sha256()
    with stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_fields(
    value: Mapping[str, object], expected: set[str], *, label: str
) -> None:
    if set(value) != expected:
        raise ModelManifestError(
            f"{label} fields must be exactly {sorted(expected)}"
        )


def _identifier(value: object, *, label: str) -> str:
    if not isinstance(value, str) or MODEL_ID_PATTERN.fullmatch(value) is None:
        raise ModelManifestError(f"{label} is not a valid identifier")
    return value


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ModelManifestError(f"{label} must be a lowercase SHA256 digest")
    return value


def _artifact(
    value: Mapping[str, object], path_field: str, hash_field: str, *, label: str
) -> tuple[Path, str]:
    raw_path = value.get(path_field)
    if not isinstance(raw_path, str) or not raw_path:
        raise ModelManifestError(f"{label}.{path_field} must be a non-empty path")
    path = Path(raw_path).expanduser().resolve()
    digest = _digest(value.get(hash_field), label=f"{label}.{hash_field}")
    if file_sha256(path) != digest:
        raise ModelManifestError(f"{label}.{path_field} SHA256 mismatch")
    return path, digest


def _model_spec(value: object, *, index: int) -> ModelSpec:
    if not isinstance(value, Mapping):
        raise ModelManifestError(f"models[{index}] must be an object")
    label = f"models[{index}]"
    kind = value.get("kind")
    if kind not in {"stage1", "grpo"}:
        raise ModelManifestError(f"{label}.kind must be stage1 or grpo")
    common_fields = {"id", "kind", "variant", "checkpoint", "checkpoint_sha256"}
    if kind == "stage1":
        _require_fields(value, common_fields, label=label)
    else:
        _require_fields(
            value,
            common_fields
            | {
                "reward_domain",
                "source_checkpoint",
                "source_checkpoint_sha256",
            },
            label=label,
        )
    model_id = _identifier(value.get("id"), label=f"{label}.id")
    variant = value.get("variant")
    if variant not in {"A", "B"}:
        raise ModelManifestError(f"{label}.variant must be A or B")
    checkpoint, checkpoint_digest = _artifact(
        value, "checkpoint", "checkpoint_sha256", label=label
    )
    if kind == "stage1":
        return ModelSpec(
            model_id=model_id,
            kind=kind,
            variant=str(variant),
            checkpoint=checkpoint,
            checkpoint_sha256=checkpoint_digest,
        )
    reward_domain = value.get("reward_domain")
    if reward_domain not in {"tau_cmd", "tau_a"}:
        raise ModelManifestError(
            f"{label}.reward_domain must be tau_cmd or tau_a"
        )
    source_checkpoint, source_digest = _artifact(
        value,
        "source_checkpoint",
        "source_checkpoint_sha256",
        label=label,
    )
    return ModelSpec(
        model_id=model_id,
        kind=kind,
        variant=str(variant),
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_digest,
        reward_domain=str(reward_domain),
        source_checkpoint=source_checkpoint,
        source_checkpoint_sha256=source_digest,
    )


def _comparison_spec(value: object, *, index: int) -> ComparisonSpec:
    if not isinstance(value, Mapping):
        raise ModelManifestError(f"comparisons[{index}] must be an object")
    label = f"comparisons[{index}]"
    _require_fields(value, {"id", "baseline", "candidate"}, label=label)
    return ComparisonSpec(
        comparison_id=_identifier(value.get("id"), label=f"{label}.id"),
        baseline=_identifier(value.get("baseline"), label=f"{label}.baseline"),
        candidate=_identifier(value.get("candidate"), label=f"{label}.candidate"),
    )


def load_model_manifest(path: Path | str) -> ModelEvaluationManifest:
    manifest_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelManifestError(f"invalid model manifest: {manifest_path}") from exc
    if not isinstance(payload, Mapping):
        raise ModelManifestError("model manifest root must be an object")
    _require_fields(payload, {"format", "models", "comparisons"}, label="manifest")
    if payload.get("format") != MANIFEST_FORMAT:
        raise ModelManifestError(f"model manifest format must be {MANIFEST_FORMAT}")
    raw_models = payload.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ModelManifestError("manifest.models must be a non-empty list")
    models = tuple(
        _model_spec(value, index=index) for index, value in enumerate(raw_models)
    )
    model_ids = [model.model_id for model in models]
    if len(set(model_ids)) != len(model_ids):
        raise ModelManifestError("manifest model ids must be unique")
    raw_comparisons = payload.get("comparisons")
    if not isinstance(raw_comparisons, list):
        raise ModelManifestError("manifest.comparisons must be a list")
    comparisons = tuple(
        _comparison_spec(value, index=index)
        for index, value in enumerate(raw_comparisons)
    )
    comparison_ids = [comparison.comparison_id for comparison in comparisons]
    if len(set(comparison_ids)) != len(comparison_ids):
        raise ModelManifestError("manifest comparison ids must be unique")
    known_models = set(model_ids)
    for comparison in comparisons:
        if comparison.baseline not in known_models:
            raise ModelManifestError(
                f"comparison {comparison.comparison_id} baseline is unknown"
            )
        if comparison.candidate not in known_models:
            raise ModelManifestError(
                f"comparison {comparison.comparison_id} candidate is unknown"
            )
        if comparison.baseline == comparison.candidate:
            raise ModelManifestError(
                f"comparison {comparison.comparison_id} cannot compare a model to itself"
            )
    return ModelEvaluationManifest(
        path=manifest_path,
        models=models,
        comparisons=comparisons,
    )


__all__ = [
    "ComparisonSpec",
    "MANIFEST_FORMAT",
    "ModelEvaluationManifest",
    "ModelManifestError",
    "ModelSpec",
    "file_sha256",
    "load_model_manifest",
]
