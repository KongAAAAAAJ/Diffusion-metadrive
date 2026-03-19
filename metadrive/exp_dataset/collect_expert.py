"""collect_expert.py

Collect expert demonstration data from DatasetCollectEnv.

Strategy: a single environment instance is created once with the default
hybrid map (hybrid_map_sequence="SSXCOCSS", num_scenarios=1).  The map
is held fixed across all episodes; only the background traffic density is
re-sampled in [traffic_density_min, traffic_density_max] before each reset,
producing diverse traffic flow on the same road layout.

Each training sample contains:
  ego_state        – ego vehicle state vector              float32
  others_state     – surrounding vehicle state vector      float32
  lidar            – lidar observation vector              float32
  bev_raster       – BEV multi-channel image CHW           uint8   (C,H,W)
  left_camera      – left RGB camera image HWC             uint8   (H,W,3)
  front_camera     – front RGB camera image HWC            uint8   (H,W,3)
  right_camera     – right RGB camera image HWC            uint8   (H,W,3)
  trajectory       – 8 future waypoints in ego-local frame float32 (8,3)
  ego_pose_world   – current [x, y, heading_theta]         float32 (3,)
  action           – last applied [steer, accel]           float32 (2,)
  traffic_density  – episode traffic density scalar        float32 ()

Usage:
    python -m metadrive.dataset.collect_expert \\
        --output-root /data/datasets \\
        --dataset-name metadrive_ppo_hybrid \\
        --expert-type ppo \\
        --target-samples 50000
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, List

import numpy as np

from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.envs.diffusion_envs.base_multi_env import DatasetCollectEnv
from metadrive.examples.ppo_expert import expert as ppo_expert
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.policy.diffusion_policy.transfuser_features import BoundingBox2DIndex
from metadrive.utils import Config
from metadrive.exp_dataset.metadrive_dataset import split_shards





# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExpertCollectorConfig:
    # Output
    output_root: Path = Path("/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets")
    dataset_name: str = "metadrive_ppo"
    expert_type: str = "ppo"
    target_samples: int = 100
    start_seed: int = 0
    samples_per_shard: int = 2048

    # Episode
    max_episode_steps: int = 1000  # 每个episode的最大步数

    # Trajectory supervision
    horizon_steps: int = 100       # minimum future context required per sample
    target_stride_steps: int = 5  # step interval between consecutive waypoints
    sample_stride_steps: int = 2  # sliding-window stride inside one episode
    trajectory_num_poses: int = 8 # number of future waypoints per sample
    num_bounding_boxes: int = 16
    lidar_min_x: float = -32.0  # lidar的感知范围（相对于ego车坐标系）
    lidar_max_x: float = 32.0
    lidar_min_y: float = -32.0
    lidar_max_y: float = 32.0

    # Traffic (map is fixed; only density varies)
    traffic_density_min: float = 0.06
    traffic_density_max: float = 0.08

    # Dataset split
    train_split_ratio: float = 0.8
    val_split_ratio: float = 0.1
    test_split_ratio: float = 0.1
    split_seed: int = 0


# ---------------------------------------------------------------------------
# Traffic sampling
# ---------------------------------------------------------------------------

def sample_traffic_density(rng: np.random.RandomState, config: ExpertCollectorConfig) -> float:
    return float(rng.uniform(config.traffic_density_min, config.traffic_density_max))


def _to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Config):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, type):
        return value.__name__
    if callable(value):
        return getattr(value, "__name__", str(value))
    return str(value)


# ---------------------------------------------------------------------------
# Per-frame helpers
# ---------------------------------------------------------------------------

def _to_numpy_array(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)

def pose_to_array(vehicle) -> np.ndarray:
    return np.asarray(
        [vehicle.position[0], vehicle.position[1], vehicle.heading_theta],
        dtype=np.float32,
    )


def world_future_to_local(current_pose: np.ndarray, future_pose: np.ndarray) -> np.ndarray:
    """Transform a future world pose into the current ego-local coordinate frame."""
    dx = future_pose[0] - current_pose[0]
    dy = future_pose[1] - current_pose[1]
    heading = current_pose[2]
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    return np.asarray(
        [
            cos_h * dx + sin_h * dy,
            -sin_h * dx + cos_h * dy,
            future_pose[2] - heading,
        ],
        dtype=np.float32,
    )


def extract_last_action(vehicle) -> np.ndarray:
    return np.asarray(
        getattr(vehicle, "last_current_action", [None, [0.0, 0.0]])[1],
        dtype=np.float32,
    )


def _to_chw_uint8(image_obs: np.ndarray) -> np.ndarray:
    image_obs = _to_numpy_array(image_obs)
    return np.clip(np.moveaxis(image_obs, -1, 0) * 255.0, 0.0, 255.0).astype(np.uint8)


def _to_hwc_uint8(image_obs: np.ndarray) -> np.ndarray:
    image_obs = _to_numpy_array(image_obs)
    return np.clip(image_obs * 255.0, 0.0, 255.0).astype(np.uint8)


def _wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _world_to_ego_xy(vehicle, obj_position: np.ndarray) -> np.ndarray:
    dx = obj_position[0] - vehicle.position[0]
    dy = obj_position[1] - vehicle.position[1]
    heading = vehicle.heading_theta
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    return np.asarray(
        [cos_h * dx + sin_h * dy, -sin_h * dx + cos_h * dy],
        dtype=np.float32,
    )


def extract_agent_targets(vehicle, config: ExpertCollectorConfig) -> tuple[np.ndarray, np.ndarray]:
    # 捕获在lidar范围内的所有车
    candidates = []
    for obj in vehicle.engine.get_objects(lambda o: isinstance(o, BaseVehicle)).values():
        if obj is vehicle:
            continue
        local_xy = _world_to_ego_xy(vehicle, obj.position)
        x, y = float(local_xy[0]), float(local_xy[1])
        if not (config.lidar_min_x <= x <= config.lidar_max_x and config.lidar_min_y <= y <= config.lidar_max_y):  
            continue
        heading = _wrap_to_pi(float(obj.heading_theta - vehicle.heading_theta))
        candidates.append(np.asarray([x, y, heading, float(obj.LENGTH), float(obj.WIDTH)], dtype=np.float32))

    # 从 candidates 中选出的、距离 ego 车最近的前 num_bounding_boxes 个目标，并构建 agent_states 和 agent_labels
    agent_states = np.zeros((config.num_bounding_boxes, BoundingBox2DIndex.size()), dtype=np.float32)
    agent_labels = np.zeros((config.num_bounding_boxes,), dtype=bool)
    if candidates:
        candidates = np.stack(candidates, axis=0)
        distances = np.linalg.norm(candidates[:, BoundingBox2DIndex.POINT], axis=-1)
        order = np.argsort(distances)[:config.num_bounding_boxes]
        selected = candidates[order]
        agent_states[: len(selected)] = selected
        agent_labels[: len(selected)] = True
    return agent_states, agent_labels


# ?用于? 将 BEV 多通道图像转为单通道的语义分割标签图
# 返回一个二维整型数组，每个像素的值表示其语义类别（0=背景，1=道路，2=交通，3=自车轨迹）。
def build_bev_semantic_map(bev_raster: np.ndarray) -> np.ndarray:
    road = bev_raster[0] > 0
    ego_history = bev_raster[1] > 0 if bev_raster.shape[0] > 1 else np.zeros_like(road)
    traffic = bev_raster[2:].max(axis=0) > 0 if bev_raster.shape[0] > 2 else np.zeros_like(road)

    semantic = np.zeros(road.shape, dtype=np.int64)
    semantic[road] = 1
    semantic[traffic] = 2
    semantic[ego_history] = 3
    return semantic


def build_frame(
    vehicle,
    collector_config: ExpertCollectorConfig,
    ego_state_obs: np.ndarray,
    others_state_obs: np.ndarray,
    lidar_obs: np.ndarray,
    topdown_obs: np.ndarray,
    left_camera_obs: np.ndarray,
    front_camera_obs: np.ndarray,
    right_camera_obs: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Pack one timestep observation into numpy arrays.

    Args:
        vehicle:    MetaDrive vehicle instance (for pose and last action).
        ego_state_obs: ego state vector.
        others_state_obs: surrounding vehicle state vector.
        lidar_obs: pure lidar vector.
        topdown_obs: BEV image (H, W, C) float32 in [0, 1].
        left_camera_obs/front_camera_obs/right_camera_obs: RGB images (H, W, C) float32 in [0, 1].
    """
    ego_state = _to_numpy_array(ego_state_obs).astype(np.float32, copy=False)
    other_states = _to_numpy_array(others_state_obs).astype(np.float32, copy=False)
    lidar = _to_numpy_array(lidar_obs).astype(np.float32, copy=False)
    state_275 = np.concatenate([ego_state, other_states, lidar], axis=0).astype(np.float32)
    bev_raster = _to_chw_uint8(topdown_obs)
    agent_states, agent_labels = extract_agent_targets(vehicle, collector_config)
    return {
        "ego_state": ego_state,
        "other_states": other_states,
        "lidar": lidar,
        "state_275": state_275,
        "bev_raster": bev_raster,
        "bev_semantic_map": build_bev_semantic_map(bev_raster),
        "agent_states": agent_states,
        "agent_labels": agent_labels,
        "left_camera": _to_hwc_uint8(left_camera_obs),
        "front_camera": _to_hwc_uint8(front_camera_obs),
        "right_camera": _to_hwc_uint8(right_camera_obs),
        "ego_pose_world": pose_to_array(vehicle),
        "action": extract_last_action(vehicle),
    }


# ---------------------------------------------------------------------------
# Sample construction (sliding window over one episode)
# ---------------------------------------------------------------------------

def build_episode_samples(
    frames: List[Dict[str, np.ndarray]],
    config: ExpertCollectorConfig,
    traffic_density: float,
) -> List[Dict[str, np.ndarray]]:
    """Slide a window over episode frames to build labelled training samples."""
    frame_count = len(frames)
    if frame_count <= config.horizon_steps:
        return []

    future_offsets = tuple(
        config.target_stride_steps * (i + 1) for i in range(config.trajectory_num_poses)
    )
    last_offset = future_offsets[-1]
    td_arr = np.asarray(traffic_density, dtype=np.float32)

    samples: List[Dict[str, np.ndarray]] = []
    for start_idx in range(0, frame_count - config.horizon_steps, config.sample_stride_steps):
        if start_idx + last_offset >= frame_count:
            continue

        current_frame = frames[start_idx]
        current_pose = current_frame["ego_pose_world"]
        trajectory = np.stack(
            [
                world_future_to_local(current_pose, frames[start_idx + offset]["ego_pose_world"])
                for offset in future_offsets
            ],
            axis=0,
        )  # (trajectory_num_poses, 3)

        samples.append(
            {
                "ego_state": current_frame["ego_state"],
                "other_states": current_frame["other_states"],
                "lidar": current_frame["lidar"],
                "bev_raster": current_frame["bev_raster"],
                "bev_semantic_map": current_frame["bev_semantic_map"],
                "agent_states": current_frame["agent_states"],
                "agent_labels": current_frame["agent_labels"],
                "left_camera": current_frame["left_camera"],
                "front_camera": current_frame["front_camera"],
                "right_camera": current_frame["right_camera"],
                "state_275": current_frame["state_275"],
                "trajectory": trajectory,
                "ego_pose_world": current_frame["ego_pose_world"],
                "action": current_frame["action"],
                "traffic_density": td_arr,
            }
        )

    return samples


def format_eta(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    return f"{minutes:02d}m{secs:02d}s"


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------

class ShardWriter:
    """Accumulates samples in memory and flushes to .npz shards once full."""

    def __init__(self, output_dir: Path, samples_per_shard: int):
        self.output_dir = output_dir
        self.samples_per_shard = samples_per_shard
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.buffers: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.buffer_size = 0
        self.shard_index = 0

    def add_samples(self, samples: List[Dict[str, np.ndarray]]) -> int:
        if not samples:
            return 0
        for sample in samples:
            for key, value in sample.items():
                self.buffers[key].append(value)
        self.buffer_size += len(samples)

        written = 0
        while self.buffer_size >= self.samples_per_shard:
            self._flush_one(self.samples_per_shard)
            written += self.samples_per_shard
        return written

    def close(self) -> int:
        if self.buffer_size == 0:
            return 0
        remaining = self.buffer_size
        self._flush_one(remaining)
        return remaining

    def _flush_one(self, count: int) -> None:
        stacked = {
            key: np.stack(values[:count], axis=0)
            for key, values in self.buffers.items()
        }
        for values in self.buffers.values():
            del values[:count]
        self.buffer_size -= count
        shard_path = self.output_dir / f"shard_{self.shard_index:06d}.npz"
        np.savez_compressed(shard_path, **stacked)
        self.shard_index += 1


# ---------------------------------------------------------------------------
# Episode rollout
# ---------------------------------------------------------------------------

def rollout_episode(
    env: DatasetCollectEnv,
    config: ExpertCollectorConfig,
) -> List[Dict[str, np.ndarray]]:
    """Drive one episode with the selected expert; return raw frame list."""
    obs_dict, _ = env.reset()
    agent_id = list(obs_dict.keys())[0]  # single-agent env → always one key

    frames: List[Dict[str, np.ndarray]] = []
    done = False
    step = 0
    idm_policy = None

    while not done and step <= config.max_episode_steps:
        vehicle = env.agents.get(agent_id)
        if vehicle is None:
            break  # agent was removed (crash / out-of-road)

        obs = obs_dict[agent_id]
        frames.append(
            build_frame(
                vehicle,
                config,
                obs["ego_state"],
                obs["others_state"],
                obs["lidar"],
                obs["topdown"],
                obs["rgb_left"],
                obs["rgb_front"],
                obs["rgb_right"],
            )
        )

        if config.expert_type == "ppo":
            action = ppo_expert(vehicle, deterministic=True)
        elif config.expert_type == "idm":
            if idm_policy is None:
                idm_policy = IDMPolicy(vehicle, random_seed=config.start_seed)
            action = idm_policy.act()
        else:
            raise ValueError(f"Unsupported expert_type: {config.expert_type}")
        obs_dict, _, terminated, truncated, _ = env.step({agent_id: action})

        # Episode ends when all agents are done or the controlled agent is gone
        done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
        if agent_id not in obs_dict:
            done = True
        step += 1

    return frames


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(
    config: ExpertCollectorConfig,
    dataset_root: Path,
    report_dir: Path,
    total_samples: int,
    total_episodes: int,
    split_summary: Dict,
) -> None:
    manifest = {
        "dataset_name": config.dataset_name,
        "target_samples": config.target_samples,
        "collected_samples": total_samples,
        "episodes": total_episodes,
        "output_root": str(dataset_root),
        "expert_type": config.expert_type,
        "map": "hybrid_fixed (SSXCOCSS)",
        "traffic_density_range": [config.traffic_density_min, config.traffic_density_max],
        "splits": {name: len(shards) for name, shards in split_summary.items()},
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in asdict(config).items()
        },
    }
    (report_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def run_collection(config: ExpertCollectorConfig) -> None:
    dataset_root = config.output_root / config.dataset_name
    shard_dir = dataset_root / "shards"
    report_dir = dataset_root / "reports"
    shard_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(config.start_seed)
    writer = ShardWriter(shard_dir, config.samples_per_shard)

    # Create a single env with the default hybrid map (num_scenarios=1 ensures
    # the same map is used on every reset; only traffic_density is updated).
    env = DatasetCollectEnv({"use_render": False, "num_scenarios": 1, "image_on_cuda": True})

    # 保存环境参数配置
    config_path = dataset_root / "env_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(env.config), f, indent=2, ensure_ascii=False)

    total_samples = 0
    total_episodes = 0
    total_env_steps = 0
    wall_start = time.perf_counter()

    try:
        while total_samples < config.target_samples:
            # Sample a new traffic density and inject it before reset so the
            # traffic manager uses the updated value while the map stays fixed.
            traffic_density = sample_traffic_density(rng, config)
            env.config["traffic_density"] = traffic_density

            frames = rollout_episode(env, config)
            samples = build_episode_samples(frames, config, traffic_density)
            writer.add_samples(samples)
            total_samples += len(samples)
            total_episodes += 1
            total_env_steps += len(frames)

            elapsed = max(time.perf_counter() - wall_start, 1e-6)
            step_per_sec = total_env_steps / elapsed
            sample_per_sec = total_samples / elapsed
            remaining_samples = max(config.target_samples - total_samples, 0)
            eta_seconds = remaining_samples / sample_per_sec if sample_per_sec > 0 else float("inf")

            print(
                f"[ep={total_episodes}] total_samples={total_samples}/{config.target_samples} "
                f"density={traffic_density:.3f} frames={len(frames)} ep_samples={len(samples)} "
                f"step/s={step_per_sec:.2f} sample/s={sample_per_sec:.2f} ETA={format_eta(eta_seconds)}"
            )
    finally:
        env.close()

    writer.close()

    split_summary = split_shards(
        dataset_root=dataset_root,
        train_ratio=config.train_split_ratio,
        val_ratio=config.val_split_ratio,
        test_ratio=config.test_split_ratio,
        seed=config.split_seed,
    )
    write_manifest(config, dataset_root, report_dir, total_samples, total_episodes, split_summary)

    print("Splits: " + ", ".join(f"{k}={len(v)} shards" for k, v in split_summary.items()))
    print(f"Done. samples={total_samples}  output={dataset_root}")

    # TODO: 添加提取anchors过程


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> ExpertCollectorConfig:
    config = ExpertCollectorConfig()
    parser = argparse.ArgumentParser(
        description="Collect fixed-map expert trajectories from DatasetCollectEnv."
    )
    for f in fields(ExpertCollectorConfig):
        default = getattr(config, f.name)
        opts: dict = {"default": default, "dest": f.name}
        if isinstance(default, bool):
            opts["action"] = "store_true"
        elif isinstance(default, Path):
            opts["type"] = Path
        else:
            opts["type"] = type(default)
        if f.name == "expert_type":
            opts["choices"] = ("ppo", "idm")
        parser.add_argument(f"--{f.name.replace('_', '-')}", **opts)
    parser.parse_args(namespace=config)
    return config


if __name__ == "__main__":
    run_collection(parse_args())
