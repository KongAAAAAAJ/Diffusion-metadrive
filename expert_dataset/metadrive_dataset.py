from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_DATASET_ROOT = Path("/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metadrive_ppo")


@dataclass(frozen=True)
class MetaDriveDatasetConfig:
    dataset_root: Path = DEFAULT_DATASET_ROOT
    split: str = "all"
    condition_key: str = "condition_feature_259"
    bev_key: str = "bev_raster"
    trajectory_key: str = "trajectory"
    include_metadata: bool = False
    max_samples: Optional[int] = None
    shard_paths: Optional[Tuple[Path, ...]] = None


class MetaDriveShardDataset(Dataset):
    """Read shard-compressed MetaDrive samples and emit features/targets for the lightweight planner."""

    def __init__(self, config: MetaDriveDatasetConfig):
        self.config = config
        self.dataset_root = Path(config.dataset_root)
        self.shard_paths = self._resolve_shards(config)
        if not self.shard_paths:
            raise FileNotFoundError(f"No shard files found under {self.dataset_root}")

        self._index: List[Tuple[int, int]] = []
        self._shard_lengths: List[int] = []
        self._cached_shard_idx: Optional[int] = None
        self._cached_shard: Optional[Dict[str, np.ndarray]] = None

        self._build_index()

    def _resolve_shards(self, config: MetaDriveDatasetConfig) -> List[Path]:
        if config.shard_paths is not None:
            return sorted(Path(path) for path in config.shard_paths)

        shard_dir = self.dataset_root / "shards"
        if not shard_dir.is_dir():
            raise FileNotFoundError(f"Shard directory does not exist: {shard_dir}")

        shard_paths = sorted(shard_dir.glob("*.npz"))
        if config.split == "all":
            return shard_paths

        split_file = self.dataset_root / "splits" / f"{config.split}.txt"
        if not split_file.is_file():
            raise FileNotFoundError(f"Split file does not exist: {split_file}")
        shard_names = {line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()}
        return [path for path in shard_paths if path.name in shard_names]

    def _build_index(self) -> None:
        total_samples = 0
        for shard_idx, shard_path in enumerate(self.shard_paths):
            with np.load(shard_path, allow_pickle=False) as shard:
                shard_len = int(shard[self.config.trajectory_key].shape[0])
            self._shard_lengths.append(shard_len)
            for sample_idx in range(shard_len):
                self._index.append((shard_idx, sample_idx))
                total_samples += 1
                if self.config.max_samples is not None and total_samples >= self.config.max_samples:
                    return

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        shard_idx, sample_idx = self._index[idx]
        shard = self._load_shard(shard_idx)

        condition_feature = torch.from_numpy(shard[self.config.condition_key][sample_idx]).float()
        bev_raster = torch.from_numpy(shard[self.config.bev_key][sample_idx])
        trajectory = torch.from_numpy(shard[self.config.trajectory_key][sample_idx]).float()

        features = {
            "condition_feature": condition_feature,
            "bev_raster": bev_raster,
        }
        targets = {
            "trajectory": trajectory,
        }

        if not self.config.include_metadata:
            return features, targets

        metadata = {}
        for key in shard.keys():
            if key in {self.config.condition_key, self.config.bev_key, self.config.trajectory_key}:
                continue
            value = shard[key][sample_idx]
            if np.isscalar(value) or getattr(value, "shape", ()) == ():
                metadata[key] = value.item() if hasattr(value, "item") else value
            else:
                metadata[key] = value
        metadata["shard_path"] = str(self.shard_paths[shard_idx])
        metadata["sample_index"] = sample_idx
        return features, targets, metadata

    def _load_shard(self, shard_idx: int) -> Dict[str, np.ndarray]:
        if self._cached_shard_idx == shard_idx and self._cached_shard is not None:
            return self._cached_shard

        shard_path = self.shard_paths[shard_idx]
        with np.load(shard_path, allow_pickle=False) as raw_shard:
            shard = {key: raw_shard[key] for key in raw_shard.files}
        self._cached_shard_idx = shard_idx
        self._cached_shard = shard
        return shard


def build_dataset(
    dataset_root: Path | str = DEFAULT_DATASET_ROOT,
    split: str = "all",
    condition_key: str = "condition_feature_259",
    include_metadata: bool = False,
    max_samples: Optional[int] = None,
) -> MetaDriveShardDataset:
    return MetaDriveShardDataset(
        MetaDriveDatasetConfig(
            dataset_root=Path(dataset_root),
            split=split,
            condition_key=condition_key,
            include_metadata=include_metadata,
            max_samples=max_samples,
        )
    )


def split_shards(
    dataset_root: Path | str = DEFAULT_DATASET_ROOT,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 0,
) -> Dict[str, List[str]]:
    dataset_root = Path(dataset_root)
    shard_dir = dataset_root / "shards"
    split_dir = dataset_root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = sorted(shard_dir.glob("*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found under {shard_dir}")

    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios < 0.0):
        raise ValueError(f"Split ratios must be non-negative, got {ratios.tolist()}")
    if float(ratios.sum()) <= 0.0:
        raise ValueError("At least one split ratio must be greater than zero")
    ratios = ratios / ratios.sum()

    rng = np.random.RandomState(seed)
    indices = np.arange(len(shard_paths))
    rng.shuffle(indices)

    total_shards = len(indices)
    train_count = int(np.floor(total_shards * ratios[0]))
    val_count = int(np.floor(total_shards * ratios[1]))
    test_count = total_shards - train_count - val_count

    # Keep the training split non-empty when shards are available.
    if total_shards > 0 and train_count == 0:
        train_count = 1
        if val_count > test_count and val_count > 0:
            val_count -= 1
        elif test_count > 0:
            test_count -= 1

    train_slice = indices[:train_count]
    val_slice = indices[train_count:train_count + val_count]
    test_slice = indices[train_count + val_count:train_count + val_count + test_count]

    train_names = [shard_paths[idx].name for idx in train_slice]
    val_names = [shard_paths[idx].name for idx in val_slice]
    test_names = [shard_paths[idx].name for idx in test_slice]

    (split_dir / "train.txt").write_text("\n".join(train_names) + "\n", encoding="utf-8")
    (split_dir / "val.txt").write_text("\n".join(val_names) + "\n", encoding="utf-8")
    (split_dir / "test.txt").write_text("\n".join(test_names) + "\n", encoding="utf-8")

    return {"train": train_names, "val": val_names, "test": test_names}