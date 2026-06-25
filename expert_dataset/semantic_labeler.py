"""轨迹语义标注器
根据专家轨迹数据的运动学和几何特征，为每条轨迹分配一个语义标签。
判别以轨迹自身特征（纵向进度比、横向位移、航向变化）为主，辅以前车距离和道路拓扑信息。
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Mapping

import numpy as np

# NOTE: TrajectoryMode 不再被标注逻辑直接依赖，仅在 label_dataset 中
# 保留 trajectory_mode 字段用于统计/诊断。


# ---------------------------------------------------------------------------
# 7类驾驶行为模式
#   纵向: CRUISE(巡航) / FOLLOW(跟车) / DECELERATE(减速)
#   横向: LANE_CHANGE_LEFT / LANE_CHANGE_RIGHT
#   拓扑: TURN_LEFT / TURN_RIGHT
# ---------------------------------------------------------------------------
class BehaviorMode(IntEnum):
    CRUISE = 0
    FOLLOW = 1
    DECELERATE = 2
    LANE_CHANGE_LEFT = 3
    LANE_CHANGE_RIGHT = 4
    TURN_LEFT = 5
    TURN_RIGHT = 6


# 向后兼容别名，供旧代码过渡使用
KEEP_LANE = BehaviorMode.CRUISE
FOLLOW_BRAKE = BehaviorMode.FOLLOW
YIELD = BehaviorMode.DECELERATE


# 语义标注所需字段
REQUIRED_LABEL_FIELDS = (
    "trajectory",
    "trajectory_mode",
    "ego_speed_km_h",
    "front_object_distance",
    "front_object_speed_km_h",
    "lane_index",
    "reference_lane_index",
    "reference_longitudinal",
    "reference_lateral",
    "lane_width",
    "current_ref_lane_count",
    "next_ref_lane_count",
    "reference_pose_world",
)

# ---------------------------------------------------------------------------
# 运动学阈值（集中定义，便于调参）
# ---------------------------------------------------------------------------
PROGRESS_RATIO_CRUISE = 0.7       # 纵向进度比 > 此值 → 巡航
PROGRESS_RATIO_DECEL = 0.3        # 纵向进度比 < 此值 → 减速
FRONT_DISTANCE_CLOSE = 15.0       # 前车距离(m) < 此值视为"有前车约束"
DECEL_SPEED_THRESH = 5.0          # km/h，低于此速度+低进度 → 减速
DECEL_PROGRESS_THRESH = 2.0       # m，极低进度绝对值
HEADING_CHANGE_TURN = 0.5         # rad，航向变化 > 此值可能为转弯
LATERAL_LANE_RATIO = 0.55         # 横向位移 > lane_width * 此值 → 变道
HORIZON_SEC = 4.0                 # 轨迹时间跨度(s)，8个点×0.5s


def _wrap_to_pi(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _as_scalar(sample: Mapping[str, object], key: str, default: float = 0.0) -> float:
    value = sample.get(key, default)
    array = np.asarray(value)
    if array.size == 0:
        return float(default)
    return float(array.reshape(-1)[0])


def _trajectory(sample: Mapping[str, object]) -> np.ndarray:
    trajectory = np.asarray(sample["trajectory"], dtype=np.float32)
    if trajectory.shape != (8, 3):
        raise ValueError(f"Expected trajectory shape (8, 3), got {trajectory.shape}")
    return trajectory


def _lane_change_direction(end_lateral: float, reference_lateral: float) -> BehaviorMode:
    """根据横向位移判断变道方向，reference_lateral 作为辅助。"""
    signal = end_lateral if abs(end_lateral) > 0.5 else reference_lateral
    return BehaviorMode.LANE_CHANGE_LEFT if signal >= 0.0 else BehaviorMode.LANE_CHANGE_RIGHT


def _compute_progress_ratio(end_x: float, ego_speed_km_h: float) -> float:
    """纵向进度比 = 实际纵向位移 / 匀速预期位移。"""
    expected = max((ego_speed_km_h / 3.6) * HORIZON_SEC, 1.0)
    return end_x / expected


def label_sample(sample: Mapping[str, np.ndarray | float | int]) -> BehaviorMode:
    """根据轨迹运动学和几何特征进行语义标注。

    判别优先级:
      1. 转弯  — 航向变化大 + 道路拓扑变化
      2. 变道  — 横向位移大（几何驱动）
      3. 减速  — 纵向进度比极低 或 极低速接近停车
      4. 跟车  — 前车距离近 + 纵向进度受约束
      5. 巡航  — 兜底
    """
    trajectory = _trajectory(sample)
    end_x = float(trajectory[-1, 0])
    end_y = float(trajectory[-1, 1])
    delta_heading = _wrap_to_pi(float(trajectory[-1, 2]) - float(trajectory[0, 2]))
    lane_width = max(_as_scalar(sample, "lane_width", 4.0), 1.0)
    reference_lateral = _as_scalar(sample, "reference_lateral", 0.0)
    front_distance = _as_scalar(sample, "front_object_distance", -1.0)
    ego_speed_km_h = _as_scalar(sample, "ego_speed_km_h", 0.0)
    current_ref_lane_count = int(round(_as_scalar(sample, "current_ref_lane_count", 1.0)))
    next_ref_lane_count = int(round(_as_scalar(sample, "next_ref_lane_count", -1.0)))

    max_abs_lateral = float(np.max(np.abs(trajectory[:, 1])))
    progress_ratio = _compute_progress_ratio(end_x, ego_speed_km_h)
    has_front_vehicle = 0.0 < front_distance < FRONT_DISTANCE_CLOSE

    # --- 1. 转弯：航向变化大 + 道路拓扑变化（优先级最高） ---
    topology_changed = (next_ref_lane_count == -1
                        or next_ref_lane_count != current_ref_lane_count)
    if abs(delta_heading) > HEADING_CHANGE_TURN and topology_changed:
        return BehaviorMode.TURN_LEFT if delta_heading < 0 else BehaviorMode.TURN_RIGHT

    # --- 2. 变道：横向位移大 + 航向变化小（几何驱动） ---
    lateral_is_lane_change = max_abs_lateral > lane_width * LATERAL_LANE_RATIO
    if lateral_is_lane_change and abs(delta_heading) <= HEADING_CHANGE_TURN:
        return _lane_change_direction(end_y, reference_lateral)

    # --- 3. 减速：纵向进度比极低 或 极低速接近停车 ---
    if progress_ratio < PROGRESS_RATIO_DECEL:
        return BehaviorMode.DECELERATE
    if ego_speed_km_h < DECEL_SPEED_THRESH and end_x < DECEL_PROGRESS_THRESH:
        return BehaviorMode.DECELERATE

    # --- 4. 跟车：前车距离近 + 纵向进度受约束 ---
    if has_front_vehicle and progress_ratio < PROGRESS_RATIO_CRUISE:
        return BehaviorMode.FOLLOW

    # --- 5. 巡航 ---
    return BehaviorMode.CRUISE


def _load_required_arrays(shard_path: Path) -> dict[str, np.ndarray]:
    with np.load(shard_path, allow_pickle=False) as shard:
        missing = [key for key in REQUIRED_LABEL_FIELDS if key not in shard]
        if missing:
            raise RuntimeError(
                f"Shard {shard_path.name} is missing required semantic-anchor fields: {', '.join(sorted(missing))}"
            )
        arrays = {key: np.asarray(shard[key]) for key in REQUIRED_LABEL_FIELDS}
    return arrays


def label_dataset(shard_paths: list[Path]) -> dict[str, np.ndarray]:
    labels = []
    sample_indices = []
    source_indices = []
    source_shards = []
    trajectory_modes = []
    global_offset = 0

    for shard_path in shard_paths:
        arrays = _load_required_arrays(shard_path)
        sample_count = int(arrays["trajectory"].shape[0])
        for local_idx in range(sample_count):
            sample = {key: value[local_idx] for key, value in arrays.items()}
            labels.append(int(label_sample(sample)))
            sample_indices.append(global_offset + local_idx)
            source_indices.append(local_idx)
            source_shards.append(shard_path.name)
            trajectory_modes.append(int(np.asarray(sample["trajectory_mode"]).reshape(-1)[0]))
        global_offset += sample_count

    return {
        "labels": np.asarray(labels, dtype=np.int16),
        "sample_index": np.asarray(sample_indices, dtype=np.int64),
        "source_sample_index": np.asarray(source_indices, dtype=np.int64),
        "source_shard": np.asarray(source_shards, dtype="U128"),
        "trajectory_mode": np.asarray(trajectory_modes, dtype=np.int16),
    }
