from enum import IntEnum
import math
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import zipfile

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from metadrive.policy.diffusion_policy.transfuser_config import TransfuserConfig

PROCESSED_DIR_FIELDS = (
    "camera_feature",
    "lidar_feature",
    "status_feature",
    "ego_state",
    "target_point",
    "trajectory",
    "agent_states",
    "agent_labels",
    "bev_semantic_map",
)

# ego_state layout (19D): vehicle_state (9D) + navigation_info (10D)
# [0] lateral_to_left, [1] lateral_to_right, [2] heading_diff, [3] speed,
# [4] steering, [5] last_steering, [6] last_throttle, [7] yaw_rate,
# [8] lateral_lane_offset, [9-13] navi_checkpoint_1, [14-18] navi_checkpoint_2


# 枚举类
class BoundingBox2DIndex(IntEnum):
    _X = 0
    _Y = 1
    _HEADING = 2
    _LENGTH = 3
    _WIDTH = 4

    @classmethod
    def size(cls) -> int:
        return 5

    @classmethod
    @property
    def X(cls):
        return cls._X

    @classmethod
    @property
    def Y(cls):
        return cls._Y

    @classmethod
    @property
    def HEADING(cls):
        return cls._HEADING

    @classmethod
    @property
    def LENGTH(cls):
        return cls._LENGTH

    @classmethod
    @property
    def WIDTH(cls):
        return cls._WIDTH

    @classmethod
    @property
    def POINT(cls):
        return slice(cls._X, cls._Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls):
        return slice(cls._X, cls._HEADING + 1)


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)


def _to_hwc_image(image: np.ndarray) -> np.ndarray:
    image = _to_numpy(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {tuple(image.shape)}")
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(
            "Expected raw camera image in HWC layout with channel-last shape (H, W, C), "
            f"got shape {tuple(image.shape)}. If you are using an older raw dataset saved in CHW layout, "
            "convert it to HWC first."
        )
    return image


def stitch_three_cameras(
    left_camera: np.ndarray,
    front_camera: np.ndarray,
    right_camera: np.ndarray,
    config: TransfuserConfig,
) -> torch.Tensor:
    left = _to_hwc_image(left_camera)
    front = _to_hwc_image(front_camera)
    right = _to_hwc_image(right_camera)
    stitched = np.concatenate([left, front, right], axis=1)
    resized = cv2.resize(stitched, (config.camera_width, config.camera_height), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float()
    if resized.dtype == np.uint8:
        tensor = tensor / 255.0
    return tensor


def build_status_feature(ego_state: np.ndarray, config: TransfuserConfig) -> torch.Tensor:
    ego_state = _to_numpy(ego_state).astype(np.float32, copy=False)
    status = np.zeros(config.status_feature_dim, dtype=np.float32)
    take = min(config.status_feature_dim, ego_state.shape[0])
    status[:take] = ego_state[:take]
    return torch.from_numpy(status)


def _world_pose_to_local_xy(current_pose: np.ndarray, future_pose: np.ndarray) -> np.ndarray:
    current_pose = _to_numpy(current_pose).astype(np.float32, copy=False)
    future_pose = _to_numpy(future_pose).astype(np.float32, copy=False)
    dx = float(future_pose[0] - current_pose[0])
    dy = float(future_pose[1] - current_pose[1])
    heading = float(current_pose[2])
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    return np.asarray(
        [
            cos_h * dx + sin_h * dy,
            -sin_h * dx + cos_h * dy,
        ],
        dtype=np.float32,
    )


def _vehicle_pose_to_array(vehicle) -> np.ndarray:
    return np.asarray(
        [float(vehicle.position[0]), float(vehicle.position[1]), float(vehicle.heading_theta)],
        dtype=np.float32,
    )


def _zero_target_point() -> torch.Tensor:
    return torch.zeros((2,), dtype=torch.float32)


def _target_progress_distance(config: TransfuserConfig, speed_mps: float) -> float:
    horizon_s = 4.0
    return float(max(speed_mps * horizon_s, config.target_point_min_forward_distance_m))


def _interpolate_local_target(local_points_xy: np.ndarray, target_distance: float) -> np.ndarray:
    local_points_xy = np.asarray(local_points_xy, dtype=np.float32)
    if local_points_xy.size == 0:
        return np.zeros((2,), dtype=np.float32)
    if local_points_xy.shape[0] == 1:
        return local_points_xy[0].astype(np.float32, copy=False)

    segment_lengths = np.linalg.norm(np.diff(local_points_xy, axis=0), axis=1)
    cumulative = np.concatenate(
        [np.zeros((1,), dtype=np.float32), np.cumsum(segment_lengths, dtype=np.float32)],
        axis=0,
    )
    if target_distance <= 0.0:
        return local_points_xy[0].astype(np.float32, copy=False)
    if target_distance >= float(cumulative[-1]):
        return local_points_xy[-1].astype(np.float32, copy=False)

    segment_idx = int(np.searchsorted(cumulative, target_distance, side="right") - 1)
    segment_idx = max(0, min(segment_idx, local_points_xy.shape[0] - 2))
    segment_start = float(cumulative[segment_idx])
    segment_length = float(segment_lengths[segment_idx])
    if segment_length <= 1e-6:
        return local_points_xy[segment_idx + 1].astype(np.float32, copy=False)
    ratio = float((target_distance - segment_start) / segment_length)
    start = local_points_xy[segment_idx]
    end = local_points_xy[segment_idx + 1]
    return (start + ratio * (end - start)).astype(np.float32, copy=False)


def compute_target_point(vehicle, config: TransfuserConfig) -> torch.Tensor:
    if vehicle is None:
        return _zero_target_point()
    navigation = getattr(vehicle, "navigation", None)
    current_ref_lanes = getattr(navigation, "current_ref_lanes", None) if navigation is not None else None
    if not current_ref_lanes:
        return _zero_target_point()
    current_lane = current_ref_lanes[0]
    if current_lane is None:
        return _zero_target_point()

    s_ego, _ = current_lane.local_coordinates(vehicle.position)
    speed_mps = float(getattr(vehicle, "speed_km_h", 0.0)) / 3.6
    delta_s = _target_progress_distance(config, speed_mps)
    s_target = min(float(s_ego) + delta_s, float(current_lane.length))
    target_world = np.asarray(current_lane.position(s_target, 0.0), dtype=np.float32)
    current_pose = _vehicle_pose_to_array(vehicle)
    return torch.from_numpy(_world_pose_to_local_xy(current_pose, np.asarray([target_world[0], target_world[1], 0.0], dtype=np.float32)))


def compute_target_point_from_sample(sample: Dict[str, np.ndarray], config: TransfuserConfig) -> torch.Tensor:
    if "reference_pose_world" not in sample or "future_reference_pose_world" not in sample:
        return _zero_target_point()

    current_pose = _to_numpy(sample["reference_pose_world"]).astype(np.float32, copy=False)
    future_reference = _to_numpy(sample["future_reference_pose_world"]).astype(np.float32, copy=False)
    if future_reference.ndim != 2 or future_reference.shape[0] == 0:
        return _zero_target_point()

    current_lane_index = int(_to_numpy(sample.get("reference_lane_index", np.asarray(-1, dtype=np.int16))).reshape(-1)[0])
    future_lane_index = sample.get("future_reference_lane_index")
    if future_lane_index is not None:
        future_lane_index = _to_numpy(future_lane_index).astype(np.int64, copy=False).reshape(-1)
        same_lane_mask = future_lane_index == current_lane_index
        if same_lane_mask.any():
            future_reference = future_reference[same_lane_mask]
        else:
            future_reference = future_reference[:1]

    local_points_xy = np.stack(
        [_world_pose_to_local_xy(current_pose, pose) for pose in future_reference],
        axis=0,
    ).astype(np.float32, copy=False)
    local_points_xy = np.concatenate(
        [np.zeros((1, 2), dtype=np.float32), local_points_xy],
        axis=0,
    )
    speed_mps = float(_to_numpy(sample.get("ego_speed_km_h", np.asarray(0.0, dtype=np.float32))).reshape(-1)[0]) / 3.6
    delta_s = _target_progress_distance(config, speed_mps)
    target_point = _interpolate_local_target(local_points_xy, delta_s)
    return torch.from_numpy(target_point)


# !这里与 TransFuser 原论文中的 BEV 处理方式不同!
def lidar_to_histogram(
    lidar: np.ndarray,
    config: TransfuserConfig,
) -> torch.Tensor:
    lidar = _to_numpy(lidar).astype(np.float32, copy=False)
    if lidar.ndim != 1:
        raise ValueError(f"Expected 1D lidar vector, got shape {tuple(lidar.shape)}")

    distances = np.clip(lidar, 0.0, 1.0) * config.lidar_max_distance
    num_lasers = max(len(distances), 1)
    angles = np.linspace(0.0, 2.0 * np.pi, num_lasers, endpoint=False)

    x = distances * np.cos(angles)
    y = -distances * np.sin(angles)

    valid = (
        (config.lidar_min_x <= x) & (x <= config.lidar_max_x) &
        (config.lidar_min_y <= y) & (y <= config.lidar_max_y)
    )
    x = x[valid]
    y = y[valid]

    xbins = np.linspace(config.lidar_min_x, config.lidar_max_x, config.lidar_resolution_width + 1)
    ybins = np.linspace(config.lidar_min_y, config.lidar_max_y, config.lidar_resolution_height + 1)
    hist = np.histogramdd(np.stack([x, y], axis=1), bins=(xbins, ybins))[0]
    hist = np.clip(hist, 0, config.hist_max_per_pixel) / config.hist_max_per_pixel
    hist = hist.astype(np.float32)[None, ...]
    return torch.from_numpy(hist)


def bev_raster_to_target(bev_raster: np.ndarray, config: TransfuserConfig) -> torch.Tensor:
    bev_raster = _to_numpy(bev_raster)
    if bev_raster.ndim != 3:
        raise ValueError(f"Expected BEV raster as CHW, got shape {tuple(bev_raster.shape)}")

    road = bev_raster[0] > 0
    ego_history = bev_raster[1] > 0 if bev_raster.shape[0] > 1 else np.zeros_like(road)
    traffic = bev_raster[2:].max(axis=0) > 0 if bev_raster.shape[0] > 2 else np.zeros_like(road)

    semantic = np.zeros(road.shape, dtype=np.int64)
    semantic[road] = 1
    semantic[traffic] = 2
    semantic[ego_history] = 3

    return semantic_map_to_target(semantic, config)


def semantic_map_to_target(bev_semantic_map: np.ndarray, config: TransfuserConfig) -> torch.Tensor:
    semantic = _to_numpy(bev_semantic_map).astype(np.int64, copy=False)
    if semantic.ndim != 2:
        raise ValueError(f"Expected BEV semantic map as HW, got shape {tuple(semantic.shape)}")

    target_h, target_w = config.bev_semantic_frame
    if semantic.shape != (target_h, target_w):
        semantic = cv2.resize(
            semantic.astype(np.uint8),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int64)

    return torch.from_numpy(semantic)


def normalize_agent_targets(
    agent_states: np.ndarray,
    agent_labels: np.ndarray,
    config: TransfuserConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    agent_states = _to_numpy(agent_states).astype(np.float32, copy=False)
    agent_labels = _to_numpy(agent_labels).astype(bool, copy=False)

    if agent_states.shape != (config.num_bounding_boxes, BoundingBox2DIndex.size()):
        padded = np.zeros((config.num_bounding_boxes, BoundingBox2DIndex.size()), dtype=np.float32)
        valid = min(config.num_bounding_boxes, agent_states.shape[0])
        padded[:valid] = agent_states[:valid]
        agent_states = padded

    if agent_labels.shape != (config.num_bounding_boxes,):
        padded_labels = np.zeros((config.num_bounding_boxes,), dtype=bool)
        valid = min(config.num_bounding_boxes, agent_labels.shape[0])
        padded_labels[:valid] = agent_labels[:valid]
        agent_labels = padded_labels

    return torch.from_numpy(agent_states), torch.from_numpy(agent_labels)  # ?是否按照距离 ego 最近排序?


def sample_to_features_targets(sample: Dict[str, np.ndarray], config: TransfuserConfig):
    if "camera_feature" in sample and "lidar_feature" in sample and "status_feature" in sample:
        return processed_sample_to_features_targets(sample)
    features = {
        "camera_feature": stitch_three_cameras(
            sample["left_camera"], sample["front_camera"], sample["right_camera"], config
        ),
        "lidar_feature": lidar_to_histogram(sample["lidar"], config),
        "status_feature": build_status_feature(sample["ego_state"], config),
        "ego_state": torch.from_numpy(_to_numpy(sample["ego_state"]).astype(np.float32, copy=False)),
        "target_point": compute_target_point_from_sample(sample, config),
    }
    agent_states, agent_labels = normalize_agent_targets(
        sample["agent_states"], sample["agent_labels"], config
    )
    targets = {
        "trajectory": torch.from_numpy(_to_numpy(sample["trajectory"]).astype(np.float32, copy=False)),
        "agent_states": agent_states,
        "agent_labels": agent_labels,
        "bev_semantic_map": semantic_map_to_target(sample["bev_semantic_map"], config)
        if "bev_semantic_map" in sample
        else bev_raster_to_target(sample["bev_raster"], config),
    }
    return features, targets


def processed_sample_to_features_targets(sample: Dict[str, np.ndarray]):
    features = {
        "camera_feature": torch.from_numpy(_to_numpy(sample["camera_feature"]).astype(np.float32, copy=True)),
        "lidar_feature": torch.from_numpy(_to_numpy(sample["lidar_feature"]).astype(np.float32, copy=True)),
        "status_feature": torch.from_numpy(_to_numpy(sample["status_feature"]).astype(np.float32, copy=True)),
        "ego_state": torch.from_numpy(_to_numpy(sample["ego_state"]).astype(np.float32, copy=True)),
        "target_point": torch.from_numpy(_to_numpy(sample["target_point"]).astype(np.float32, copy=True))
        if "target_point" in sample
        else _zero_target_point(),
    }
    targets = {
        "trajectory": torch.from_numpy(_to_numpy(sample["trajectory"]).astype(np.float32, copy=True)),
        "agent_states": torch.from_numpy(_to_numpy(sample["agent_states"]).astype(np.float32, copy=True)),
        "agent_labels": torch.from_numpy(_to_numpy(sample["agent_labels"]).astype(bool, copy=True)),
        "bev_semantic_map": torch.from_numpy(_to_numpy(sample["bev_semantic_map"]).astype(np.int64, copy=True)),
    }
    return features, targets


def observation_to_features(
    observation: Dict[str, np.ndarray],
    config: TransfuserConfig,
    vehicle=None,
) -> Dict[str, torch.Tensor]:
    return {
        "camera_feature": stitch_three_cameras(
            observation["rgb_left"], observation["rgb_front"], observation["rgb_right"], config
        ),
        "lidar_feature": lidar_to_histogram(observation["lidar"], config),
        "status_feature": build_status_feature(observation["ego_state"], config),
        "ego_state": torch.from_numpy(_to_numpy(observation["ego_state"]).astype(np.float32, copy=False)),
        "target_point": compute_target_point(vehicle, config) if vehicle is not None else _zero_target_point(),
    }


class MetaDriveTransfuserDataset(Dataset):
    """Dataset adapter from MetaDrive npz shards to TransFuser feature/target dicts."""

    def __init__(
        self,
        dataset_root: Union[str, Path],
        config: TransfuserConfig,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.config = config
        self.split = split
        self.max_samples = max_samples
        self._cache_all_shards = config.cache_shards_in_memory
        self.shard_paths = self._resolve_shards()
        self._index = []
        self._cached_shard_idx = None
        self._cached_shard = None
        self._all_shards = {}
        self._build_index()
        if self._cache_all_shards:
            self._preload_shards()

    def _resolve_shards(self):
        shard_dir = self.dataset_root / "shards"
        if not shard_dir.exists():
            raise FileNotFoundError(
                f"Dataset shard directory does not exist: {shard_dir}. "
                f"Expected dataset_root to point to a dataset folder containing 'shards/'."
            )
        shard_paths = self._list_shard_entries(shard_dir)
        if self.split == "all":
            return shard_paths
        split_path = self.dataset_root / "splits" / f"{self.split}.txt"
        if not split_path.exists():
            return shard_paths
        names = {line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()}
        names = {Path(name).stem if name.endswith(".npz") else name for name in names}
        resolved = [path for path in shard_paths if path.stem in names or path.name in names]
        if not resolved:
            raise FileNotFoundError(
                f"No shards found for split '{self.split}' under {self.dataset_root}. "
                f"Checked split file: {split_path}"
            )
        return resolved

    def _list_shard_entries(self, shard_dir: Path):
        entries = {}
        for path in sorted(shard_dir.iterdir()):
            if path.is_dir() and path.name.startswith("shard_"):
                entries[path.name] = path
        for path in sorted(shard_dir.glob("*.npz")):
            entries.setdefault(path.stem, path)
        return [entries[key] for key in sorted(entries)]

    def _is_processed_dir_shard(self, shard_path: Path) -> bool:
        return shard_path.is_dir()

    def _load_processed_dir_meta(self, shard_path: Path) -> Dict:
        meta_path = shard_path / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Processed shard metadata not found: {meta_path}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def _build_index(self):
        total = 0
        for shard_idx, shard_path in enumerate(self.shard_paths):
            try:
                if self._is_processed_dir_shard(shard_path):
                    trajectory = np.load(shard_path / "trajectory.npy", mmap_mode="r", allow_pickle=False)
                    length = int(trajectory.shape[0])
                else:
                    with np.load(shard_path, allow_pickle=False) as shard:
                        length = int(shard["trajectory"].shape[0])
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to build dataset index from shard '{shard_path}'. "
                    "Run the dataset integrity verifier before training."
                ) from exc
            for sample_idx in range(length):
                self._index.append((shard_idx, sample_idx))
                total += 1
                if self.max_samples is not None and total >= self.max_samples:
                    return

    def __len__(self):
        return len(self._index)

    def get_sample_metadata(self, idx: int) -> Dict[str, Union[int, str]]:
        shard_idx, sample_idx = self._index[idx]
        shard_path = self.shard_paths[shard_idx]
        metadata = {
            "sample_index": int(idx),
            "shard_index": int(shard_idx),
            "shard_name": shard_path.name,
            "shard_stem": shard_path.stem,
            "local_index": int(sample_idx),
        }
        shard = self._load_shard(shard_idx)
        if "trajectory_mode" in shard:
            value = np.asarray(shard["trajectory_mode"][sample_idx]).reshape(-1)
            if value.size > 0:
                metadata["trajectory_mode"] = int(value[0])
        return metadata

    # 用于按索引获取一个样本的特征和目标
    def __getitem__(self, idx):
        shard_idx, sample_idx = self._index[idx]
        shard = self._load_shard(shard_idx)
        sample = {key: shard[key][sample_idx] for key in shard.keys()}
        return sample_to_features_targets(sample, self.config)

    def _load_shard(self, shard_idx: int) -> Dict[str, np.ndarray]:
        if self._cache_all_shards:
            return self._all_shards[shard_idx]
        if self._cached_shard_idx == shard_idx and self._cached_shard is not None:
            return self._cached_shard
        shard_path = self.shard_paths[shard_idx]
        if self._is_processed_dir_shard(shard_path):
            shard = self._load_processed_dir_shard(shard_path, mmap_mode="r")
            self._cached_shard_idx = shard_idx
            self._cached_shard = shard
            return shard
        try:
            with np.load(shard_path, allow_pickle=False) as raw:
                shard = {}
                for key in raw.files:
                    try:
                        shard[key] = raw[key]
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to read key '{key}' from shard '{shard_path}'. "
                            "Run the dataset integrity verifier before training."
                        ) from exc
        except RuntimeError:
            raise
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            raise RuntimeError(
                f"Failed to load shard '{shard_path}'. Run the dataset integrity verifier before training."
            ) from exc
        self._cached_shard_idx = shard_idx
        self._cached_shard = shard
        return shard

    def _preload_shards(self) -> None:
        for shard_idx in range(len(self.shard_paths)):
            shard_path = self.shard_paths[shard_idx]
            if self._is_processed_dir_shard(shard_path):
                self._all_shards[shard_idx] = self._load_processed_dir_shard(shard_path, mmap_mode=None)
                continue
            try:
                with np.load(shard_path, allow_pickle=False) as raw:
                    preloaded = {}
                    for key in raw.files:
                        try:
                            preloaded[key] = raw[key]
                        except Exception as exc:
                            raise RuntimeError(
                                f"Failed to preload key '{key}' from shard '{shard_path}'. "
                                "Run the dataset integrity verifier before training."
                            ) from exc
                    self._all_shards[shard_idx] = preloaded
            except RuntimeError:
                raise
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                raise RuntimeError(
                    f"Failed to preload shard '{shard_path}'. Run the dataset integrity verifier before training."
                ) from exc

    def _load_processed_dir_shard(self, shard_path: Path, mmap_mode: Optional[str]) -> Dict[str, np.ndarray]:
        try:
            meta = self._load_processed_dir_meta(shard_path)
            fields = tuple(meta.get("fields", {}).keys()) if isinstance(meta.get("fields"), dict) else PROCESSED_DIR_FIELDS
            shard = {}
            for key in fields:
                field_path = shard_path / f"{key}.npy"
                if not field_path.exists():
                    raise FileNotFoundError(f"Processed shard field not found: {field_path}")
                array = np.load(field_path, mmap_mode=mmap_mode, allow_pickle=False)
                shard[key] = np.asarray(array) if mmap_mode is None else array
            return shard
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load processed directory shard '{shard_path}'. "
                "Run the dataset integrity verifier before training."
            ) from exc
