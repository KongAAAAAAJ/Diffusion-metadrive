"""PyTorch dataset for the packed joint-first semantic BEV episode store."""

from __future__ import annotations

import json
import os
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from expert_dataset.collect_joint_bev import JOINT_SAMPLE_DTYPES, JOINT_SAMPLE_SHAPES
from expert_dataset.joint_bev_storage import (
    EPISODE_PATTERN,
    PACKED_BEV_FIELD,
    SPLIT_NAMES,
    STORAGE_FORMAT,
    STORAGE_SCHEMA_VERSION,
    STORED_FIELD_DTYPES,
    STORED_FIELD_SHAPES,
)
from expert_dataset.semantic_bev_codec import (
    packed_bev_contract,
    unpack_semantic_bev,
)


class JointBEVDatasetError(RuntimeError):
    """Raised when a packed joint-first dataset is invalid or incompatible."""


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JointBEVDatasetError(f"invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise JointBEVDatasetError(f"JSON root must be an object: {path}")
    return payload


def _logical_contract() -> dict[str, dict[str, object]]:
    return {
        name: {
            "shape": list(JOINT_SAMPLE_SHAPES[name]),
            "dtype": str(JOINT_SAMPLE_DTYPES[name]),
        }
        for name in JOINT_SAMPLE_SHAPES
    }


def _physical_contract() -> dict[str, dict[str, object]]:
    return {
        name: {
            "shape": list(STORED_FIELD_SHAPES[name]),
            "dtype": str(STORED_FIELD_DTYPES[name]),
        }
        for name in STORED_FIELD_SHAPES
    }


def validate_dataset_contract(dataset_root: Path | str) -> dict[str, object]:
    """Validate and return the immutable schema-v2 dataset contract."""

    root = Path(dataset_root).expanduser()
    contract = _read_json(root / "dataset_contract.json")
    expected_keys = {
        "schema_version",
        "format",
        "joint_first",
        "sample_contract",
        "physical_storage_contract",
        "packed_semantic_bev",
        "split_assignment",
        "dataset_fingerprint",
    }
    if set(contract) != expected_keys:
        raise JointBEVDatasetError(
            "dataset contract fields do not match the packed joint-first schema"
        )
    if int(contract.get("schema_version", -1)) != STORAGE_SCHEMA_VERSION:
        raise JointBEVDatasetError(
            f"dataset schema must be version {STORAGE_SCHEMA_VERSION}"
        )
    if contract.get("format") != STORAGE_FORMAT:
        raise JointBEVDatasetError("dataset storage format mismatch")
    if contract.get("joint_first") is not True:
        raise JointBEVDatasetError("dataset must declare joint_first=true")
    if contract.get("sample_contract") != _logical_contract():
        raise JointBEVDatasetError("logical sample contract mismatch")
    if contract.get("physical_storage_contract") != _physical_contract():
        raise JointBEVDatasetError("physical storage contract mismatch")
    if contract.get("packed_semantic_bev") != packed_bev_contract():
        raise JointBEVDatasetError("packed semantic BEV contract mismatch")
    split_assignment = contract.get("split_assignment")
    if (
        not isinstance(split_assignment, Mapping)
        or set(split_assignment)
        != {"train_ratio", "val_ratio", "test_ratio", "seed"}
    ):
        raise JointBEVDatasetError(
            "split_assignment must match the frozen train/val/test fields"
        )
    fingerprint = contract.get("dataset_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or fingerprint != fingerprint.lower()
    ):
        raise JointBEVDatasetError("dataset_fingerprint must be a SHA256 digest")
    try:
        int(fingerprint, 16)
    except ValueError as exc:
        raise JointBEVDatasetError(
            "dataset_fingerprint must be a SHA256 digest"
        ) from exc
    return contract


@dataclass(frozen=True)
class JointEpisodeRecord:
    episode_index: int
    directory: str
    joint_samples: int
    attributes: Mapping[str, object]
    start_index: int
    stop_index: int


@dataclass(frozen=True)
class JointBEVDatasetConfig:
    dataset_root: Path | str
    split: str
    mmap_cache_episodes: int = 2

    def __post_init__(self) -> None:
        root = Path(self.dataset_root).expanduser()
        if self.split not in SPLIT_NAMES:
            raise JointBEVDatasetError(
                f"split must be one of {SPLIT_NAMES}, got {self.split!r}"
            )
        if (
            isinstance(self.mmap_cache_episodes, bool)
            or not isinstance(self.mmap_cache_episodes, Integral)
            or int(self.mmap_cache_episodes) <= 0
        ):
            raise JointBEVDatasetError("mmap_cache_episodes must be a positive integer")
        object.__setattr__(self, "dataset_root", root)
        object.__setattr__(
            self, "mmap_cache_episodes", int(self.mmap_cache_episodes)
        )


class JointBEVDataset(Dataset):
    """Flat sample view over split-local atomic episode directories."""

    def __init__(self, config: JointBEVDatasetConfig) -> None:
        self.config = config
        self.dataset_root = Path(config.dataset_root)
        self.contract = validate_dataset_contract(self.dataset_root)
        self.split_root = self.dataset_root / config.split
        manifest = _read_json(self.split_root / "manifest.json")
        if int(manifest.get("schema_version", -1)) != STORAGE_SCHEMA_VERSION:
            raise JointBEVDatasetError("manifest schema version mismatch")
        if manifest.get("format") != STORAGE_FORMAT:
            raise JointBEVDatasetError("manifest storage format mismatch")
        if manifest.get("split") != config.split:
            raise JointBEVDatasetError("manifest split mismatch")
        entries = manifest.get("episodes")
        if not isinstance(entries, list):
            raise JointBEVDatasetError("manifest episodes must be a list")
        if int(manifest.get("episode_count", -1)) != len(entries):
            raise JointBEVDatasetError("manifest episode_count mismatch")

        records = []
        cursor = 0
        previous_index = -1
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise JointBEVDatasetError("manifest episode entry must be an object")
            episode_index = int(entry.get("episode_index", -1))
            directory = str(entry.get("directory", ""))
            joint_samples = int(entry.get("joint_samples", -1))
            attributes = entry.get("attributes", {})
            if episode_index <= previous_index:
                raise JointBEVDatasetError(
                    "manifest episodes must use strictly increasing global indices"
                )
            if (
                EPISODE_PATTERN.fullmatch(directory) is None
                or directory != f"episode_{episode_index:08d}"
            ):
                raise JointBEVDatasetError("manifest episode directory/index mismatch")
            if joint_samples <= 0:
                raise JointBEVDatasetError("manifest episode must contain samples")
            if not isinstance(attributes, Mapping):
                raise JointBEVDatasetError("manifest episode attributes must be an object")
            records.append(
                JointEpisodeRecord(
                    episode_index=episode_index,
                    directory=directory,
                    joint_samples=joint_samples,
                    attributes=dict(attributes),
                    start_index=cursor,
                    stop_index=cursor + joint_samples,
                )
            )
            cursor += joint_samples
            previous_index = episode_index
        if int(manifest.get("joint_samples", -1)) != cursor:
            raise JointBEVDatasetError("manifest joint_samples mismatch")

        self.records = tuple(records)
        self._stop_indices = tuple(record.stop_index for record in self.records)
        self._length = cursor
        self._episode_cache: OrderedDict[int, dict[str, np.memmap]] = OrderedDict()
        self._cache_pid: int | None = None

    def __len__(self) -> int:
        return self._length

    @staticmethod
    def _close_arrays(arrays: Mapping[str, np.memmap]) -> None:
        for array in arrays.values():
            mmap_object = getattr(array, "_mmap", None)
            if mmap_object is not None:
                mmap_object.close()

    def close(self) -> None:
        while self._episode_cache:
            _, arrays = self._episode_cache.popitem(last=False)
            self._close_arrays(arrays)
        self._cache_pid = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self) -> dict[str, object]:
        state = dict(self.__dict__)
        state["_episode_cache"] = OrderedDict()
        state["_cache_pid"] = None
        return state

    def _ensure_worker_cache(self) -> None:
        process_id = os.getpid()
        if self._cache_pid == process_id:
            return
        self.close()
        self._cache_pid = process_id

    def _load_episode(self, record_index: int) -> dict[str, np.memmap]:
        self._ensure_worker_cache()
        cached = self._episode_cache.pop(record_index, None)
        if cached is not None:
            self._episode_cache[record_index] = cached
            return cached

        record = self.records[record_index]
        episode_root = self.split_root / "episodes" / record.directory
        metadata = _read_json(episode_root / "episode.json")
        if (
            metadata.get("complete") is not True
            or int(metadata.get("schema_version", -1)) != STORAGE_SCHEMA_VERSION
            or metadata.get("format") != STORAGE_FORMAT
            or int(metadata.get("episode_index", -1)) != record.episode_index
            or metadata.get("split") != self.config.split
            or int(metadata.get("joint_samples", -1)) != record.joint_samples
        ):
            raise JointBEVDatasetError(
                f"episode metadata mismatch: {episode_root}"
            )
        expected_names = {"episode.json"} | {
            f"{name}.npy" for name in STORED_FIELD_SHAPES
        }
        try:
            actual_names = {path.name for path in episode_root.iterdir()}
        except OSError as exc:
            raise JointBEVDatasetError(
                f"unable to inspect episode: {episode_root}"
            ) from exc
        if actual_names != expected_names:
            raise JointBEVDatasetError(f"episode file set mismatch: {episode_root}")

        arrays: dict[str, np.memmap] = {}
        try:
            for field_name, sample_shape in STORED_FIELD_SHAPES.items():
                array = np.load(
                    episode_root / f"{field_name}.npy",
                    mmap_mode="r",
                    allow_pickle=False,
                )
                expected_shape = (record.joint_samples, *sample_shape)
                if not isinstance(array, np.memmap):
                    raise JointBEVDatasetError(
                        f"episode field is not memory-mapped: {field_name}"
                    )
                if (
                    array.shape != expected_shape
                    or array.dtype != STORED_FIELD_DTYPES[field_name]
                ):
                    raise JointBEVDatasetError(
                        f"episode field contract mismatch: {field_name}"
                    )
                arrays[field_name] = array
        except (OSError, ValueError) as exc:
            self._close_arrays(arrays)
            raise JointBEVDatasetError(
                f"unable to mmap episode: {episode_root}"
            ) from exc
        except Exception:
            self._close_arrays(arrays)
            raise

        self._episode_cache[record_index] = arrays
        while len(self._episode_cache) > self.config.mmap_cache_episodes:
            _, evicted = self._episode_cache.popitem(last=False)
            self._close_arrays(evicted)
        return arrays

    def _locate(self, index: int) -> tuple[int, int]:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("dataset index must be an integer")
        normalized = int(index)
        if normalized < 0:
            normalized += self._length
        if normalized < 0 or normalized >= self._length:
            raise IndexError(index)
        record_index = bisect_right(self._stop_indices, normalized)
        record = self.records[record_index]
        return record_index, normalized - record.start_index

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record_index, sample_index = self._locate(index)
        arrays = self._load_episode(record_index)
        output: dict[str, torch.Tensor] = {}
        for field_name in JOINT_SAMPLE_SHAPES:
            if field_name == "bev":
                value = unpack_semantic_bev(
                    np.asarray(arrays[PACKED_BEV_FIELD][sample_index])
                )
            else:
                value = np.array(arrays[field_name][sample_index], copy=True)
            output[field_name] = torch.from_numpy(value)
        return output


class EpisodeChunkBatchSampler(Sampler[list[int]]):
    """Shuffle episode-local chunks while keeping mmap access bounded."""

    def __init__(
        self,
        dataset: JointBEVDataset,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int = 0,
        drop_last: bool = False,
        chunk_size: int = 32,
    ) -> None:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, Integral)
            or int(batch_size) <= 0
        ):
            raise ValueError("batch_size must be positive")
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, Integral)
            or int(chunk_size) < int(batch_size)
        ):
            raise ValueError("chunk_size must be at least batch_size")
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError("seed must be an integer")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.chunk_size = int(chunk_size)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, Integral):
            raise ValueError("epoch must be an integer")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        chunks: list[np.ndarray] = []
        for record in self.dataset.records:
            indices = np.arange(
                record.start_index, record.stop_index, dtype=np.int64
            )
            if self.shuffle:
                rng.shuffle(indices)
            chunks.extend(
                indices[start : start + self.chunk_size]
                for start in range(0, len(indices), self.chunk_size)
            )
        if self.shuffle:
            rng.shuffle(chunks)

        pending: list[int] = []
        for chunk in chunks:
            pending.extend(int(value) for value in chunk)
            while len(pending) >= self.batch_size:
                yield pending[: self.batch_size]
                del pending[: self.batch_size]
        if pending and not self.drop_last:
            yield pending


def build_joint_bev_dataloader(
    dataset_root: Path | str,
    split: str,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    chunk_size: int = 32,
    mmap_cache_episodes: int = 2,
) -> DataLoader:
    """Build the locality-aware DataLoader used by later BEV training stages."""

    if (
        isinstance(num_workers, bool)
        or not isinstance(num_workers, Integral)
        or int(num_workers) < 0
    ):
        raise ValueError("num_workers must be a non-negative integer")
    dataset = JointBEVDataset(
        JointBEVDatasetConfig(
            dataset_root=dataset_root,
            split=split,
            mmap_cache_episodes=mmap_cache_episodes,
        )
    )
    batch_sampler = EpisodeChunkBatchSampler(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
        chunk_size=chunk_size,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=False,
    )


__all__ = [
    "EpisodeChunkBatchSampler",
    "JointBEVDataset",
    "JointBEVDatasetConfig",
    "JointBEVDatasetError",
    "JointEpisodeRecord",
    "build_joint_bev_dataloader",
    "validate_dataset_contract",
]
