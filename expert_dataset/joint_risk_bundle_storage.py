"""Atomic transaction index for the joint-BEV/RiskEntry dataset bundle."""

from __future__ import annotations

import fcntl
import json
import os
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from expert_dataset.joint_risk_bundle_contract import (
    BUNDLE_FORMAT,
    BUNDLE_SCHEMA_VERSION,
    bundle_protocol_sha256,
)


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMPONENT_STATUSES = frozenset({"committed", "rejected"})


class JointRiskBundleStorageError(RuntimeError):
    """Raised when the cross-store transaction index is invalid."""


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    _atomic_write(path, encoded)


@dataclass(frozen=True)
class BundleEpisodeAttempt:
    episode_index: int
    split: str
    scenario_id: str
    local_route: str
    spawn_seed: int

    def __post_init__(self) -> None:
        if isinstance(self.episode_index, bool) or int(self.episode_index) < 0:
            raise JointRiskBundleStorageError("episode_index must be non-negative")
        if self.split not in {"train", "val", "test"}:
            raise JointRiskBundleStorageError("invalid bundle episode split")
        if not self.scenario_id or not self.local_route:
            raise JointRiskBundleStorageError("scenario_id/local_route must be non-empty")
        if isinstance(self.spawn_seed, bool) or int(self.spawn_seed) < 0:
            raise JointRiskBundleStorageError("spawn_seed must be non-negative")


@dataclass(frozen=True)
class BundleEpisodeResult:
    episode_index: int
    split: str
    scenario_id: str
    local_route: str
    spawn_seed: int
    base_status: str
    base_rejection_reason: str | None
    sidecar_status: str
    sidecar_rejection_reason: str | None
    raw_steps: int
    base_samples: int
    outcome: str

    def __post_init__(self) -> None:
        BundleEpisodeAttempt(
            self.episode_index,
            self.split,
            self.scenario_id,
            self.local_route,
            self.spawn_seed,
        )
        if self.base_status not in COMPONENT_STATUSES:
            raise JointRiskBundleStorageError("invalid base_status")
        if self.sidecar_status not in COMPONENT_STATUSES:
            raise JointRiskBundleStorageError("invalid sidecar_status")
        if self.base_status == "committed" and self.base_rejection_reason is not None:
            raise JointRiskBundleStorageError("committed base cannot have rejection reason")
        if self.base_status == "rejected" and not self.base_rejection_reason:
            raise JointRiskBundleStorageError("rejected base requires a reason")
        if self.sidecar_status == "committed" and self.sidecar_rejection_reason is not None:
            raise JointRiskBundleStorageError("committed sidecar cannot have rejection reason")
        if self.sidecar_status == "rejected" and not self.sidecar_rejection_reason:
            raise JointRiskBundleStorageError("rejected sidecar requires a reason")
        if self.base_status == "committed" and self.sidecar_status != "committed":
            raise JointRiskBundleStorageError("base-only bundle episodes are forbidden")
        if int(self.raw_steps) < 0 or int(self.base_samples) < 0:
            raise JointRiskBundleStorageError("raw_steps/base_samples must be non-negative")
        if self.base_status == "rejected" and int(self.base_samples) != 0:
            raise JointRiskBundleStorageError("rejected base must report zero stored samples")
        if not str(self.outcome):
            raise JointRiskBundleStorageError("outcome must be non-empty")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "BundleEpisodeResult":
        expected = set(cls.__dataclass_fields__)
        if set(payload) != expected:
            raise JointRiskBundleStorageError("bundle result fields do not match contract")
        return cls(**dict(payload))  # type: ignore[arg-type]


class JointRiskBundleIndex:
    """One durable result row per attempted episode, plus a crash intent record."""

    MANIFEST_FILE = "dataset_bundle_manifest.json"
    INDEX_FILE = "bundle_episode_index.jsonl"
    PENDING_FILE = ".bundle_episode_pending.json"
    LOCK_FILE = ".bundle_writer.lock"

    def __init__(
        self,
        bundle_root: Path | str,
        *,
        base_directory: str,
        sidecar_directory: str,
        base_dataset_fingerprint: str,
        sidecar_dataset_fingerprint: str,
        scenario_contract_sha256: str,
        split_seed: int,
        resume: bool,
        planner_version: str = "v1",
    ) -> None:
        self.bundle_root = Path(bundle_root).expanduser()
        self.bundle_root.mkdir(parents=True, exist_ok=True)
        if base_directory == sidecar_directory:
            raise JointRiskBundleStorageError("base and sidecar directories must differ")
        for value, name in (
            (base_dataset_fingerprint, "base_dataset_fingerprint"),
            (sidecar_dataset_fingerprint, "sidecar_dataset_fingerprint"),
            (scenario_contract_sha256, "scenario_contract_sha256"),
        ):
            if SHA256_PATTERN.fullmatch(str(value)) is None:
                raise JointRiskBundleStorageError(f"{name} must be a SHA256 digest")
        self.manifest = {
            "format": BUNDLE_FORMAT,
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "protocol_sha256": bundle_protocol_sha256(
                planner_version=planner_version
            ),
            "base_directory": str(base_directory),
            "sidecar_directory": str(sidecar_directory),
            "base_dataset_fingerprint": str(base_dataset_fingerprint),
            "sidecar_dataset_fingerprint": str(sidecar_dataset_fingerprint),
            "scenario_contract_sha256": str(scenario_contract_sha256),
            "decision_dt_s": 0.1,
            "split_seed": int(split_seed),
        }
        self.manifest_path = self.bundle_root / self.MANIFEST_FILE
        self.index_path = self.bundle_root / self.INDEX_FILE
        self.pending_path = self.bundle_root / self.PENDING_FILE
        self._lock_stream = (self.bundle_root / self.LOCK_FILE).open("a+b")
        try:
            fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_stream.close()
            raise JointRiskBundleStorageError("another bundle writer is active") from exc
        try:
            if resume:
                self._load_manifest()
            else:
                if self.manifest_path.exists() or self.index_path.exists() or self.pending_path.exists():
                    raise JointRiskBundleStorageError(
                        "bundle transaction files exist and resume is disabled"
                    )
                _atomic_write_json(self.manifest_path, self.manifest)
                _atomic_write(self.index_path, b"")
            self.rows = self._load_rows()
            pending = self.pending_attempt
            if pending is not None and pending.episode_index < self.next_episode_index:
                row = self.rows[pending.episode_index]
                if (
                    row.episode_index,
                    row.split,
                    row.scenario_id,
                    row.local_route,
                    row.spawn_seed,
                ) != (
                    pending.episode_index,
                    pending.split,
                    pending.scenario_id,
                    pending.local_route,
                    pending.spawn_seed,
                ):
                    raise JointRiskBundleStorageError(
                        "completed row conflicts with pending bundle attempt"
                    )
                self.pending_path.unlink()
                _fsync_directory(self.bundle_root)
        except Exception:
            self.close()
            raise

    def __enter__(self) -> "JointRiskBundleIndex":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()

    def close(self) -> None:
        stream = getattr(self, "_lock_stream", None)
        if stream is None or stream.closed:
            return
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()

    def _load_manifest(self) -> None:
        try:
            observed = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise JointRiskBundleStorageError("resume requires a valid bundle manifest") from exc
        if observed != self.manifest:
            raise JointRiskBundleStorageError("bundle resume manifest mismatch")

    def _load_rows(self) -> list[BundleEpisodeResult]:
        if not self.index_path.is_file():
            raise JointRiskBundleStorageError("bundle episode index is missing")
        rows: list[BundleEpisodeResult] = []
        try:
            lines = self.index_path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                payload = json.loads(line)
                if not isinstance(payload, Mapping):
                    raise JointRiskBundleStorageError("bundle row must be an object")
                rows.append(BundleEpisodeResult.from_mapping(payload))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise JointRiskBundleStorageError("invalid bundle episode index") from exc
        if [row.episode_index for row in rows] != list(range(len(rows))):
            raise JointRiskBundleStorageError(
                "bundle episode index must be contiguous and ordered"
            )
        return rows

    @property
    def next_episode_index(self) -> int:
        return len(self.rows)

    @property
    def pending_attempt(self) -> BundleEpisodeAttempt | None:
        if not self.pending_path.exists():
            return None
        try:
            payload = json.loads(self.pending_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise JointRiskBundleStorageError("invalid pending bundle attempt") from exc
        if not isinstance(payload, Mapping) or set(payload) != set(BundleEpisodeAttempt.__dataclass_fields__):
            raise JointRiskBundleStorageError("pending bundle attempt fields mismatch")
        return BundleEpisodeAttempt(**dict(payload))  # type: ignore[arg-type]

    def begin_attempt(self, attempt: BundleEpisodeAttempt) -> None:
        if self.pending_attempt is not None:
            raise JointRiskBundleStorageError("a bundle attempt is already pending")
        if attempt.episode_index != self.next_episode_index:
            raise JointRiskBundleStorageError(
                f"expected bundle episode {self.next_episode_index}, got {attempt.episode_index}"
            )
        _atomic_write_json(self.pending_path, asdict(attempt))

    def finalize(self, result: BundleEpisodeResult) -> None:
        pending = self.pending_attempt
        if pending is None:
            raise JointRiskBundleStorageError("no bundle attempt is pending")
        identity = (
            result.episode_index,
            result.split,
            result.scenario_id,
            result.local_route,
            result.spawn_seed,
        )
        if identity != (
            pending.episode_index,
            pending.split,
            pending.scenario_id,
            pending.local_route,
            pending.spawn_seed,
        ):
            raise JointRiskBundleStorageError("bundle result identity differs from pending attempt")
        rows = [*self.rows, result]
        encoded = b"".join(
            (
                json.dumps(asdict(row), sort_keys=True, ensure_ascii=False) + "\n"
            ).encode("utf-8")
            for row in rows
        )
        _atomic_write(self.index_path, encoded)
        self.pending_path.unlink()
        _fsync_directory(self.bundle_root)
        self.rows = rows


__all__ = [
    "BundleEpisodeAttempt",
    "BundleEpisodeResult",
    "JointRiskBundleIndex",
    "JointRiskBundleStorageError",
]
