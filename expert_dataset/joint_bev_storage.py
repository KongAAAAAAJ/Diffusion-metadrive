"""Atomic episode storage for the joint-first BEV dataset.

Each completed episode is committed to exactly one split as a directory of
``.npy`` arrays.  Semantic BEV tensors use the frozen lossless bit-packed
representation while all logical tensors remain memory-mappable.  The
episode-directory rename is the durable commit marker.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from expert_dataset.collect_joint_bev import (
    JOINT_SAMPLE_DTYPES,
    JOINT_SAMPLE_SHAPES,
    V2_JOINT_SAMPLE_DTYPES,
    V2_JOINT_SAMPLE_SHAPES,
    JointBEVSample,
    JointBEVSampleV2,
)
from expert_dataset.semantic_bev_codec import (
    PACKED_BEV_SHAPE,
    SemanticBEVCodecError,
    pack_semantic_bev,
    packed_bev_contract,
)


STORAGE_SCHEMA_VERSION = 2
STORAGE_FORMAT = "joint-first-packed-semantic-bev-npy-episodes"
V2_STORAGE_SCHEMA_VERSION = 3
V2_STORAGE_FORMAT = "joint-first-packed-semantic-bev-rule-conditioned-npy-episodes"
SPLIT_NAMES = ("train", "val", "test")
EPISODE_PATTERN = re.compile(r"^episode_(\d{8})$")
PACKED_BEV_FIELD = "bev_packed"
STORED_FIELD_SHAPES = {
    PACKED_BEV_FIELD: (JOINT_SAMPLE_SHAPES["bev"][0], *PACKED_BEV_SHAPE),
    **{
        name: shape
        for name, shape in JOINT_SAMPLE_SHAPES.items()
        if name != "bev"
    },
}
STORED_FIELD_DTYPES = {
    PACKED_BEV_FIELD: np.dtype(np.uint8),
    **{
        name: dtype
        for name, dtype in JOINT_SAMPLE_DTYPES.items()
        if name != "bev"
    },
}
V2_STORED_FIELD_SHAPES = {
    PACKED_BEV_FIELD: (V2_JOINT_SAMPLE_SHAPES["bev"][0], *PACKED_BEV_SHAPE),
    **{
        name: shape
        for name, shape in V2_JOINT_SAMPLE_SHAPES.items()
        if name != "bev"
    },
}
V2_STORED_FIELD_DTYPES = {
    PACKED_BEV_FIELD: np.dtype(np.uint8),
    **{
        name: dtype
        for name, dtype in V2_JOINT_SAMPLE_DTYPES.items()
        if name != "bev"
    },
}


@dataclass(frozen=True)
class JointSampleStorageContract:
    planner_version: str
    schema_version: int
    storage_format: str
    sample_shapes: Mapping[str, tuple[int, ...]]
    sample_dtypes: Mapping[str, np.dtype]
    stored_shapes: Mapping[str, tuple[int, ...]]
    stored_dtypes: Mapping[str, np.dtype]


JOINT_SAMPLE_STORAGE_CONTRACTS = {
    "v1": JointSampleStorageContract(
        planner_version="v1",
        schema_version=STORAGE_SCHEMA_VERSION,
        storage_format=STORAGE_FORMAT,
        sample_shapes=JOINT_SAMPLE_SHAPES,
        sample_dtypes=JOINT_SAMPLE_DTYPES,
        stored_shapes=STORED_FIELD_SHAPES,
        stored_dtypes=STORED_FIELD_DTYPES,
    ),
    "v2": JointSampleStorageContract(
        planner_version="v2",
        schema_version=V2_STORAGE_SCHEMA_VERSION,
        storage_format=V2_STORAGE_FORMAT,
        sample_shapes=V2_JOINT_SAMPLE_SHAPES,
        sample_dtypes=V2_JOINT_SAMPLE_DTYPES,
        stored_shapes=V2_STORED_FIELD_SHAPES,
        stored_dtypes=V2_STORED_FIELD_DTYPES,
    ),
}


def joint_sample_storage_contract(planner_version: str) -> JointSampleStorageContract:
    try:
        return JOINT_SAMPLE_STORAGE_CONTRACTS[str(planner_version)]
    except KeyError as exc:
        raise JointStorageError("planner_version must be v1 or v2") from exc


class JointStorageError(RuntimeError):
    """Raised when storage state is incompatible, corrupt, or unsafe to resume."""


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint_payload(payload: object) -> str:
    """Return a stable SHA256 fingerprint for immutable dataset settings."""

    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise JointStorageError("metadata floating-point values must be finite")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _jsonable(value.item())
    raise JointStorageError(f"metadata value is not JSON serializable: {type(value).__name__}")


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JointStorageError(f"invalid JSON state: {path}") from exc
    if not isinstance(payload, dict):
        raise JointStorageError(f"JSON state must be an object: {path}")
    return payload


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    try:
        with temporary.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        # A temporary file is not a commit marker and is ignored by resume.
        raise


@dataclass(frozen=True)
class EpisodeSplitConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 0

    def __post_init__(self) -> None:
        ratios = np.asarray(
            [self.train_ratio, self.val_ratio, self.test_ratio], dtype=np.float64
        )
        if not np.isfinite(ratios).all() or np.any(ratios < 0.0):
            raise JointStorageError("split ratios must be finite and non-negative")
        total = float(ratios.sum())
        if total <= 0.0:
            raise JointStorageError("at least one split ratio must be positive")
        ratios /= total
        object.__setattr__(self, "train_ratio", float(ratios[0]))
        object.__setattr__(self, "val_ratio", float(ratios[1]))
        object.__setattr__(self, "test_ratio", float(ratios[2]))
        if (
            isinstance(self.seed, (bool, np.bool_))
            or not isinstance(self.seed, (int, np.integer))
        ):
            raise JointStorageError("split seed must be an integer")
        object.__setattr__(self, "seed", int(self.seed))

    def as_dict(self) -> dict[str, object]:
        return {
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "seed": self.seed,
        }


class EpisodeSplitAssigner:
    """Deterministically assign a whole episode from its global index."""

    def __init__(self, config: EpisodeSplitConfig) -> None:
        self.config = config
        self._train_boundary = config.train_ratio
        self._val_boundary = config.train_ratio + config.val_ratio

    def split_for_episode(self, episode_index: int) -> str:
        if (
            isinstance(episode_index, (bool, np.bool_))
            or not isinstance(episode_index, (int, np.integer))
            or int(episode_index) < 0
        ):
            raise JointStorageError("episode_index must be a non-negative integer")
        return self._split_for_token(str(int(episode_index)))

    def split_for_key(self, group_key: str) -> str:
        """Assign all episodes in one immutable group to the same split."""

        if not isinstance(group_key, str) or not group_key.strip():
            raise JointStorageError("split group key must be a non-empty string")
        return self._split_for_token(group_key)

    def _split_for_token(self, token: str) -> str:
        digest = hashlib.sha256(
            f"{self.config.seed}:{token}".encode("utf-8")
        ).digest()
        unit = int.from_bytes(digest[:8], "big") / float(1 << 64)
        if unit < self._train_boundary:
            return "train"
        if unit < self._val_boundary:
            return "val"
        return "test"


def _sample_contract(
    contract: JointSampleStorageContract,
) -> dict[str, dict[str, object]]:
    return {
        name: {
            "shape": list(contract.sample_shapes[name]),
            "dtype": str(contract.sample_dtypes[name]),
        }
        for name in contract.sample_shapes
    }


def _physical_storage_contract(
    contract: JointSampleStorageContract,
) -> dict[str, dict[str, object]]:
    return {
        name: {
            "shape": list(contract.stored_shapes[name]),
            "dtype": str(contract.stored_dtypes[name]),
        }
        for name in contract.stored_shapes
    }


@dataclass(frozen=True)
class StoredEpisode:
    episode_index: int
    split: str
    directory: str
    joint_samples: int
    attributes: Mapping[str, object]

    def manifest_entry(self) -> dict[str, object]:
        return {
            "episode_index": self.episode_index,
            "directory": self.directory,
            "joint_samples": self.joint_samples,
            "attributes": dict(self.attributes),
        }


class JointEpisodeWriter:
    """One split-local writer; episode directories never cross split roots."""

    def __init__(
        self, split_root: Path, split: str, *, planner_version: str = "v1"
    ) -> None:
        if split not in SPLIT_NAMES:
            raise JointStorageError(f"unknown split: {split}")
        self.split_root = split_root
        self.split = split
        self.contract = joint_sample_storage_contract(planner_version)
        self.episodes_root = split_root / "episodes"
        self.manifest_path = split_root / "manifest.json"
        self.episodes_root.mkdir(parents=True, exist_ok=True)
        self.episodes: dict[int, StoredEpisode] = {}
        self.ignored_temporary_directories: tuple[str, ...] = ()

    @staticmethod
    def _episode_name(episode_index: int) -> str:
        return f"episode_{episode_index:08d}"

    def _validate_episode_directory(self, path: Path) -> StoredEpisode:
        match = EPISODE_PATTERN.fullmatch(path.name)
        if match is None:
            raise JointStorageError(f"unexpected episode directory: {path}")
        episode_index = int(match.group(1))
        metadata_path = path / "episode.json"
        if not metadata_path.is_file():
            raise JointStorageError(f"committed episode has no metadata: {path}")
        metadata = _read_json(metadata_path)
        if metadata.get("complete") is not True:
            raise JointStorageError(f"committed episode is not marked complete: {path}")
        if int(metadata.get("schema_version", -1)) != self.contract.schema_version:
            raise JointStorageError(f"episode schema version mismatch: {path}")
        if metadata.get("format") != self.contract.storage_format:
            raise JointStorageError(f"episode storage format mismatch: {path}")
        if int(metadata.get("episode_index", -1)) != episode_index:
            raise JointStorageError(f"episode directory/index mismatch: {path}")
        if metadata.get("split") != self.split:
            raise JointStorageError(f"episode split metadata mismatch: {path}")
        joint_samples = int(metadata.get("joint_samples", -1))
        if joint_samples <= 0:
            raise JointStorageError(f"episode must contain at least one joint sample: {path}")
        attributes = metadata.get("attributes", {})
        if not isinstance(attributes, dict):
            raise JointStorageError(f"episode attributes must be an object: {path}")

        expected_names = {"episode.json"} | {
            f"{field_name}.npy" for field_name in self.contract.stored_shapes
        }
        actual_names = {item.name for item in path.iterdir()}
        if actual_names != expected_names:
            raise JointStorageError(
                f"episode file set mismatch at {path}: "
                f"missing={sorted(expected_names - actual_names)}, "
                f"unexpected={sorted(actual_names - expected_names)}"
            )

        for field_name, sample_shape in self.contract.stored_shapes.items():
            array_path = path / f"{field_name}.npy"
            try:
                array = np.load(array_path, mmap_mode="r", allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise JointStorageError(f"unable to mmap episode array: {array_path}") from exc
            expected_shape = (joint_samples, *sample_shape)
            if array.shape != expected_shape:
                raise JointStorageError(
                    f"stored {field_name} shape mismatch at {path}: "
                    f"expected {expected_shape}, got {array.shape}"
                )
            if array.dtype != self.contract.stored_dtypes[field_name]:
                raise JointStorageError(
                    f"stored {field_name} dtype mismatch at {path}: "
                    f"expected {self.contract.stored_dtypes[field_name]}, got {array.dtype}"
                )

        return StoredEpisode(
            episode_index=episode_index,
            split=self.split,
            directory=path.name,
            joint_samples=joint_samples,
            attributes=attributes,
        )

    def scan(self) -> dict[int, StoredEpisode]:
        allowed_split_entries = {"episodes", "manifest.json"}
        unexpected_split_entries = []
        for item in self.split_root.iterdir():
            if item.name in allowed_split_entries:
                continue
            if item.name.startswith(".manifest.json.tmp-"):
                continue
            unexpected_split_entries.append(item.name)
        if unexpected_split_entries:
            raise JointStorageError(
                f"unexpected entries in {self.split} split root: "
                f"{sorted(unexpected_split_entries)}"
            )
        if self.manifest_path.exists():
            # Invalid JSON is a corruption signal. A valid but stale manifest is
            # rebuilt from atomic episode-directory commit markers below.
            _read_json(self.manifest_path)
        episodes: dict[int, StoredEpisode] = {}
        temporary = []
        for path in sorted(self.episodes_root.iterdir()):
            if path.is_dir() and path.name.startswith(".episode_") and ".tmp-" in path.name:
                temporary.append(path.name)
                continue
            if not path.is_dir():
                raise JointStorageError(f"unexpected file in episode root: {path}")
            episode = self._validate_episode_directory(path)
            if episode.episode_index in episodes:
                raise JointStorageError(f"duplicate global episode index: {episode.episode_index}")
            episodes[episode.episode_index] = episode
        self.episodes = episodes
        self.ignored_temporary_directories = tuple(temporary)
        self.write_manifest()
        return dict(episodes)

    def write_manifest(self) -> None:
        entries = [
            self.episodes[index].manifest_entry() for index in sorted(self.episodes)
        ]
        _atomic_write_json(
            self.manifest_path,
            {
                "schema_version": self.contract.schema_version,
                "format": self.contract.storage_format,
                "split": self.split,
                "episode_count": len(entries),
                "joint_samples": sum(int(item["joint_samples"]) for item in entries),
                "episodes": entries,
            },
        )

    def commit(
        self,
        episode_index: int,
        samples: Sequence[JointBEVSample | JointBEVSampleV2],
        attributes: Mapping[str, object] | None = None,
    ) -> StoredEpisode:
        if not samples:
            raise JointStorageError("cannot commit an empty episode")
        expected_type = (
            JointBEVSampleV2
            if self.contract.planner_version == "v2"
            else JointBEVSample
        )
        if any(type(sample) is not expected_type for sample in samples):
            raise JointStorageError(
                f"{self.contract.planner_version} storage received the wrong sample type"
            )
        if episode_index in self.episodes:
            raise JointStorageError(f"episode {episode_index} already exists")
        final_path = self.episodes_root / self._episode_name(episode_index)
        if final_path.exists():
            raise JointStorageError(f"episode path already exists: {final_path}")
        temporary = self.episodes_root / (
            f".{self._episode_name(episode_index)}.tmp-{uuid.uuid4().hex}"
        )
        temporary.mkdir()

        packed_bev = np.empty(
            (len(samples), *self.contract.stored_shapes[PACKED_BEV_FIELD]),
            dtype=np.uint8,
        )
        try:
            for sample_index, sample in enumerate(samples):
                packed_bev[sample_index] = pack_semantic_bev(sample.bev)
        except SemanticBEVCodecError as exc:
            raise JointStorageError(
                "episode BEV violates the packed semantic storage contract"
            ) from exc

        stacked: dict[str, np.ndarray] = {PACKED_BEV_FIELD: packed_bev}
        for field_name in self.contract.sample_shapes:
            if field_name == "bev":
                continue
            values = [sample.as_dict()[field_name] for sample in samples]
            array = np.ascontiguousarray(np.stack(values, axis=0))
            expected_shape = (len(samples), *self.contract.sample_shapes[field_name])
            if (
                array.shape != expected_shape
                or array.dtype != self.contract.sample_dtypes[field_name]
            ):
                raise JointStorageError(
                    f"episode field {field_name} violates the frozen storage contract"
                )
            stacked[field_name] = array

        for field_name, array in stacked.items():
            output_path = temporary / f"{field_name}.npy"
            with output_path.open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())

        metadata = {
            "schema_version": self.contract.schema_version,
            "format": self.contract.storage_format,
            "complete": True,
            "episode_index": int(episode_index),
            "split": self.split,
            "joint_samples": len(samples),
            "attributes": _jsonable(dict(attributes or {})),
        }
        _atomic_write_json(temporary / "episode.json", metadata)
        _fsync_directory(temporary)
        os.replace(temporary, final_path)
        _fsync_directory(self.episodes_root)

        episode = self._validate_episode_directory(final_path)
        self.episodes[episode_index] = episode
        self.write_manifest()
        return episode


class JointBEVDatasetStore:
    """Own three split writers and reconcile durable state for safe resume."""

    CONTRACT_FILE = "dataset_contract.json"
    STATE_FILE = "collection_state.json"
    LOCK_FILE = ".writer.lock"

    def __init__(
        self,
        dataset_root: Path | str,
        *,
        split_config: EpisodeSplitConfig,
        dataset_fingerprint: str,
        resume: bool,
        planner_version: str = "v1",
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser()
        self.split_config = split_config
        self.contract = joint_sample_storage_contract(planner_version)
        self.assigner = EpisodeSplitAssigner(split_config)
        self.dataset_fingerprint = str(dataset_fingerprint)
        if not re.fullmatch(r"[0-9a-f]{64}", self.dataset_fingerprint):
            raise JointStorageError("dataset_fingerprint must be a lowercase SHA256 hex digest")

        existed = self.dataset_root.exists()
        existing_entries = set(self.dataset_root.iterdir()) if existed else set()
        if existing_entries and not resume:
            raise JointStorageError(
                f"dataset root is not empty and resume is disabled: {self.dataset_root}"
            )
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self._lock_stream = (self.dataset_root / self.LOCK_FILE).open("a+b")
        try:
            fcntl.flock(self._lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_stream.close()
            raise JointStorageError(
                f"another writer is active for dataset root: {self.dataset_root}"
            ) from exc

        self.contract_path = self.dataset_root / self.CONTRACT_FILE
        self.state_path = self.dataset_root / self.STATE_FILE
        self.writers = {
            split: JointEpisodeWriter(
                self.dataset_root / split,
                split,
                planner_version=self.contract.planner_version,
            )
            for split in SPLIT_NAMES
        }
        self.state: dict[str, object] = {}
        try:
            if resume:
                self._load_and_validate_contract()
            else:
                self._initialize_contract()
            self._reconcile_state(resume=resume)
        except Exception:
            self.close()
            raise

    def __enter__(self) -> "JointBEVDatasetStore":
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

    def _expected_contract(self) -> dict[str, object]:
        payload = {
            "schema_version": self.contract.schema_version,
            "format": self.contract.storage_format,
            "joint_first": True,
            "sample_contract": _sample_contract(self.contract),
            "physical_storage_contract": _physical_storage_contract(self.contract),
            "packed_semantic_bev": packed_bev_contract(),
            "split_assignment": self.split_config.as_dict(),
            "dataset_fingerprint": self.dataset_fingerprint,
        }
        if self.contract.planner_version == "v2":
            payload["planner_version"] = "v2"
        return payload

    def _initialize_contract(self) -> None:
        allowed = {self.LOCK_FILE, *SPLIT_NAMES}
        unexpected = {path.name for path in self.dataset_root.iterdir()} - allowed
        if unexpected:
            raise JointStorageError(
                f"new dataset root contains unexpected entries: {sorted(unexpected)}"
            )
        _atomic_write_json(self.contract_path, self._expected_contract())

    def _load_and_validate_contract(self) -> None:
        if not self.contract_path.is_file():
            raise JointStorageError("resume requires dataset_contract.json")
        contract = _read_json(self.contract_path)
        if contract != self._expected_contract():
            raise JointStorageError(
                "resume contract mismatch: schema, split settings, or dataset fingerprint changed"
            )
        allowed = {
            self.LOCK_FILE,
            self.CONTRACT_FILE,
            self.STATE_FILE,
            *SPLIT_NAMES,
        }
        unexpected = set()
        for path in self.dataset_root.iterdir():
            if path.name in allowed:
                continue
            if path.name.startswith(
                (f".{self.CONTRACT_FILE}.tmp-", f".{self.STATE_FILE}.tmp-")
            ):
                continue
            unexpected.add(path.name)
        if unexpected:
            raise JointStorageError(
                f"dataset root contains unexpected entries: {sorted(unexpected)}"
            )

    @staticmethod
    def _validated_non_negative(state: Mapping[str, object], key: str, default: int) -> int:
        try:
            value = int(state.get(key, default))
        except (TypeError, ValueError) as exc:
            raise JointStorageError(f"collection state {key} must be an integer") from exc
        if value < 0:
            raise JointStorageError(f"collection state {key} must be non-negative")
        return value

    def _reconcile_state(self, *, resume: bool) -> None:
        stored_by_id: dict[int, StoredEpisode] = {}
        for split, writer in self.writers.items():
            for episode_index, episode in writer.scan().items():
                if episode_index in stored_by_id:
                    raise JointStorageError(
                        f"episode {episode_index} exists in more than one split"
                    )
                expected_split = self.assigner.split_for_episode(episode_index)
                if split != expected_split:
                    raise JointStorageError(
                        f"episode {episode_index} is stored in {split}, expected {expected_split}"
                    )
                stored_by_id[episode_index] = episode

        raw_state: dict[str, object] = {}
        if resume and self.state_path.exists():
            raw_state = _read_json(self.state_path)
            if int(raw_state.get("schema_version", -1)) != self.contract.schema_version:
                raise JointStorageError("collection state schema version mismatch")

        disk_next = max(stored_by_id, default=-1) + 1
        state_next = self._validated_non_negative(raw_state, "next_episode_index", 0)
        state_attempted = self._validated_non_negative(raw_state, "attempted_episodes", 0)
        next_episode = max(disk_next, state_next, state_attempted)
        stored_count = len(stored_by_id)
        total_samples = sum(item.joint_samples for item in stored_by_id.values())
        rejected_count = max(
            self._validated_non_negative(raw_state, "rejected_episodes", 0),
            next_episode - stored_count,
        )
        reasons = raw_state.get("rejection_reasons", {})
        if not isinstance(reasons, dict):
            raise JointStorageError("collection state rejection_reasons must be an object")
        normalized_reasons = {}
        for name, value in reasons.items():
            count = int(value)
            if count < 0:
                raise JointStorageError("rejection reason counts must be non-negative")
            normalized_reasons[str(name)] = count

        self.state = {
            "schema_version": self.contract.schema_version,
            "next_episode_index": next_episode,
            "attempted_episodes": next_episode,
            "stored_episodes": stored_count,
            "rejected_episodes": rejected_count,
            "total_joint_samples": total_samples,
            "rejection_reasons": normalized_reasons,
        }
        self._write_state()

    def _write_state(self) -> None:
        _atomic_write_json(self.state_path, self.state)

    @property
    def next_episode_index(self) -> int:
        return int(self.state["next_episode_index"])

    @property
    def total_joint_samples(self) -> int:
        return int(self.state["total_joint_samples"])

    def commit_episode(
        self,
        episode_index: int,
        samples: Sequence[JointBEVSample | JointBEVSampleV2],
        attributes: Mapping[str, object] | None = None,
    ) -> StoredEpisode:
        self.assigner.split_for_episode(episode_index)
        if int(episode_index) != self.next_episode_index:
            raise JointStorageError(
                f"expected episode {self.next_episode_index}, got {episode_index}"
            )
        split = self.assigner.split_for_episode(episode_index)
        episode = self.writers[split].commit(episode_index, samples, attributes)
        self.state["next_episode_index"] = episode_index + 1
        self.state["attempted_episodes"] = episode_index + 1
        self.state["stored_episodes"] = int(self.state["stored_episodes"]) + 1
        self.state["total_joint_samples"] = (
            int(self.state["total_joint_samples"]) + episode.joint_samples
        )
        self._write_state()
        return episode

    def record_rejected_episode(self, episode_index: int, reason: str) -> None:
        self.assigner.split_for_episode(episode_index)
        if int(episode_index) != self.next_episode_index:
            raise JointStorageError(
                f"expected episode {self.next_episode_index}, got {episode_index}"
            )
        reason = str(reason).strip()
        if not reason:
            raise JointStorageError("rejection reason must not be empty")
        reasons = Counter(self.state.get("rejection_reasons", {}))
        reasons[reason] += 1
        self.state["rejection_reasons"] = dict(sorted(reasons.items()))
        self.state["next_episode_index"] = episode_index + 1
        self.state["attempted_episodes"] = episode_index + 1
        self.state["rejected_episodes"] = int(self.state["rejected_episodes"]) + 1
        self._write_state()

    def summary(self) -> dict[str, object]:
        split_summary = {
            split: {
                "episodes": len(writer.episodes),
                "joint_samples": sum(
                    episode.joint_samples for episode in writer.episodes.values()
                ),
                "ignored_temporary_directories": list(
                    writer.ignored_temporary_directories
                ),
            }
            for split, writer in self.writers.items()
        }
        return {**self.state, "splits": split_summary}


__all__ = [
    "EpisodeSplitAssigner",
    "EpisodeSplitConfig",
    "JointBEVDatasetStore",
    "JointEpisodeWriter",
    "JointSampleStorageContract",
    "JointStorageError",
    "PACKED_BEV_FIELD",
    "SPLIT_NAMES",
    "STORAGE_FORMAT",
    "STORAGE_SCHEMA_VERSION",
    "STORED_FIELD_DTYPES",
    "STORED_FIELD_SHAPES",
    "V2_STORAGE_FORMAT",
    "V2_STORAGE_SCHEMA_VERSION",
    "V2_STORED_FIELD_DTYPES",
    "V2_STORED_FIELD_SHAPES",
    "joint_sample_storage_contract",
    "StoredEpisode",
    "fingerprint_payload",
]
