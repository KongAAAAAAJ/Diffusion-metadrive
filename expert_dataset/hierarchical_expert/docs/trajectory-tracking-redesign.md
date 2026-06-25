# Trajectory Tracking Redesign — Pure Pursuit Controller

> Date: 2026-03-26
> Status: Approved
> Scope: `expert_dataset/hierarchical_expert/`

## 1. Problem

`HierarchicalExpertIDMPolicy.act()` 将 `LaneChangeManager` 返回的 `steering_target`
（可能是 PointLane 轨迹或普通 Lane）统一传入 `ExpertIDMPolicy.steering_control()`。
该控制器为车道跟随设计（preview-based dual-PID），参数针对小横向偏差场景调优。
换道轨迹是大曲率曲线，导致：

1. **横向 PID 积分累积** — 换道过程横向偏差持续存在，积分项膨胀引起超调和振荡
2. **Preview 距离不匹配** — 轨迹长度有限（3~6s × speed），预瞄点可能超出轨迹末端
3. **双 PID 耦合** — heading PID 和 lateral PID 在大偏差时互相干扰

## 2. Solution: Pure Pursuit for Trajectory Tracking

在 `HierarchicalExpertIDMPolicy` 中新增 `trajectory_steering_control()` 方法，
换道期间（EXECUTING / ABORTING）使用 Pure Pursuit 算法，IDLE 时仍复用父类
`steering_control()` 跟踪车道中心线。

### 2.1 Why Pure Pursuit

| 特性 | 双PID (现有) | Pure Pursuit (新) |
|------|-------------|------------------|
| 参数数量 | 6 (两组 Kp/Ki/Kd) + preview | 1 (lookahead distance) |
| 曲率适应 | 需要手动补偿项 | 几何天然适应 |
| 振荡风险 | 高 (积分项 + 双回路耦合) | 低 (无积分, 纯几何) |
| 大偏差表现 | 差 (线性 PID 失效) | 好 (几何关系始终成立) |
| 实现复杂度 | 中 | 低 |

### 2.2 Pure Pursuit Algorithm

```
输入: ego_position, ego_heading, ego_speed, trajectory (PointLane)
输出: steering angle ∈ [-MAX_STEERING, MAX_STEERING]

1. 计算自适应前视距离:
   L_d = clip(k_speed * ego_speed, L_min, L_max)

2. 在轨迹上找前视点:
   current_s, lateral_err = trajectory.local_coordinates(ego_position)
   lookahead_s = clip(current_s + L_d, 0, trajectory.length)
   target_point = trajectory.position(lookahead_s, 0)

3. 变换到车辆坐标系:
   dx = target_x - ego_x
   dy = target_y - ego_y
   # 车辆坐标系: x=前, y=左
   local_x =  dx * cos(ego_heading) + dy * sin(ego_heading)
   local_y = -dx * sin(ego_heading) + dy * cos(ego_heading)

4. 计算曲率和转角:
   # Pure Pursuit 核心公式
   curvature = 2 * local_y / (L_d^2)
   steering = atan(curvature * wheelbase)

5. 限幅:
   steering = clip(steering, -MAX_STEERING, MAX_STEERING)
```

### 2.3 Parameters

```python
# Pure Pursuit 参数 (trajectory_tracker.py)
PP_LOOKAHEAD_SPEED_GAIN = 0.6    # [s]  前视距离 = gain * speed
PP_MIN_LOOKAHEAD = 3.0           # [m]  最小前视距离
PP_MAX_LOOKAHEAD = 15.0          # [m]  最大前视距离
PP_WHEELBASE = 2.8               # [m]  轴距
MAX_STEERING = 0.3149            # [rad] 最大转向角 (与父类一致)
```

**参数来源**:
- `PP_LOOKAHEAD_SPEED_GAIN = 0.6`: 典型范围 0.4~1.0，0.6 在 30~60 km/h 下表现稳定
- `PP_MIN_LOOKAHEAD = 3.0`: 与父类 `MIN_PREVIEW_DISTANCE=3.2` 近似
- `PP_MAX_LOOKAHEAD = 15.0`: 大于父类 `MAX_PREVIEW_DISTANCE=9.5`，因为换道轨迹更长

### 2.4 Trajectory Completion Detection

Pure Pursuit 需要额外处理轨迹接近末端的情况:
```
if current_s >= trajectory.length - completion_threshold:
    标记轨迹跟踪完成
```
`completion_threshold = 2.0m`（配合 `LaneChangeManager` 中 `lateral_error < 0.5` 的判断）

## 3. Implementation Plan

### 3.1 New File: `trajectory_tracker.py`

```python
"""Pure Pursuit trajectory tracker for lane-change maneuvers."""

from __future__ import annotations
import math
import numpy as np


class PurePursuitTracker:
    """Computes steering angle via Pure Pursuit on a PointLane trajectory."""

    PP_LOOKAHEAD_SPEED_GAIN: float = 0.6
    PP_MIN_LOOKAHEAD: float = 3.0
    PP_MAX_LOOKAHEAD: float = 15.0
    PP_WHEELBASE: float = 2.8
    MAX_STEERING: float = 0.3149183697964353

    def compute_steering(
        self,
        ego_position: np.ndarray,
        ego_heading: float,
        ego_speed: float,
        trajectory,  # PointLane
    ) -> float:
        """Return steering angle in [-MAX_STEERING, MAX_STEERING].

        Args:
            ego_position: (x, y) world coordinates.
            ego_heading: heading in radians.
            ego_speed: longitudinal speed in m/s.
            trajectory: PointLane object with .local_coordinates() and .position().

        Returns:
            Steering angle (radians). Positive = turn left.
        """
        lookahead = float(np.clip(
            self.PP_LOOKAHEAD_SPEED_GAIN * ego_speed,
            self.PP_MIN_LOOKAHEAD,
            self.PP_MAX_LOOKAHEAD,
        ))

        current_s, _ = trajectory.local_coordinates(ego_position)
        current_s = float(current_s)
        traj_length = float(trajectory.length)
        lookahead_s = min(current_s + lookahead, traj_length)
        target_point = trajectory.position(lookahead_s, 0)

        dx = float(target_point[0]) - float(ego_position[0])
        dy = float(target_point[1]) - float(ego_position[1])

        cos_h = math.cos(ego_heading)
        sin_h = math.sin(ego_heading)
        local_y = -dx * sin_h + dy * cos_h

        ld_sq = dx * dx + dy * dy
        if ld_sq < 1e-6:
            return 0.0

        curvature = 2.0 * local_y / ld_sq
        steering = math.atan(curvature * self.PP_WHEELBASE)
        return float(np.clip(steering, -self.MAX_STEERING, self.MAX_STEERING))

    def is_trajectory_ended(
        self,
        ego_position: np.ndarray,
        trajectory,
        threshold: float = 2.0,
    ) -> bool:
        """Check if ego has reached the end of the trajectory."""
        current_s, _ = trajectory.local_coordinates(ego_position)
        return float(current_s) >= float(trajectory.length) - threshold
```

### 3.2 Modify: `hierarchical_policy.py`

**Changes**:

1. Import `PurePursuitTracker`
2. 在 `__init__` 中创建 tracker 实例
3. 在 `act()` 中根据 maneuver state 选择控制器

```python
# --- hierarchical_policy.py 修改后的 act() ---

from expert_dataset.hierarchical_expert.trajectory_tracker import PurePursuitTracker

class HierarchicalExpertIDMPolicy(ExpertIDMPolicy):
    def __init__(self, control_object, random_seed=0, style_profile=None):
        super().__init__(control_object, random_seed)
        # ... (existing style / safety / planner / manager init) ...
        self.trajectory_tracker = PurePursuitTracker()

    def act(self, *args, **kwargs):
        success = self.move_to_next_road()
        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)
        current_lanes = self.control_object.navigation.current_ref_lanes
        next_lanes = self.control_object.navigation.next_ref_lanes
        try:
            if success:
                steering_target, acc_front_obj, acc_front_dist = self.manager.update(
                    ego=self.control_object,
                    all_objects=all_objects,
                    routing_target_lane=self.routing_target_lane,
                    current_lanes=current_lanes,
                    next_lanes=next_lanes,
                )
            else:
                steering_target = self.routing_target_lane
                acc_front_obj = None
                acc_front_dist = 5
        except Exception:
            steering_target = self.routing_target_lane
            acc_front_obj = None
            acc_front_dist = 5

        # ---- KEY CHANGE: dispatch steering by maneuver state ----
        if self.manager.state in (ManeuverState.EXECUTING, ManeuverState.ABORTING):
            steering = self.trajectory_tracker.compute_steering(
                ego_position=self.control_object.position,
                ego_heading=self.control_object.heading_theta,
                ego_speed=self.control_object.speed,
                trajectory=steering_target,
            )
        else:
            steering = self.steering_control(steering_target)

        acc = self.acceleration(acc_front_obj, acc_front_dist)
        action = [steering, acc]
        self.action_info["action"] = action
        self.action_info["maneuver_state"] = self.manager.state.name
        self.action_info["style_aggression"] = self.style.aggression
        return action
```

### 3.3 New Test: `tests/test_trajectory_tracker.py`

测试用例:

| # | 用例 | 输入 | 验收标准 |
|---|------|------|----------|
| 1 | 直线轨迹跟踪 | 直线 PointLane, ego 在中心线上 | steering ≈ 0 (|steering| < 0.01) |
| 2 | 直线偏移修正 | 直线 PointLane, ego 偏移 1m | steering 朝修正方向, 收敛后 < 0.05 |
| 3 | 典型换道轨迹 | 五次多项式 PointLane, 3.7m 横向 | 全程 |steering| < MAX_STEERING, 无振荡 |
| 4 | 低速场景 | speed=2 m/s | steering 有效, 不产生 NaN/除零 |
| 5 | 高速场景 | speed=30 m/s | steering 平滑, 不超限 |
| 6 | 轨迹末端 | ego 接近轨迹终点 | `is_trajectory_ended()` 返回 True |
| 7 | 退出轨迹 (abort) | abort PointLane 回原车道 | 跟踪平滑, 无超调 |

**振荡检测方法** (关键验收标准):
```python
def has_oscillation(steering_history, threshold=0.05, max_sign_changes=4):
    """Check for sign changes in steering delta exceeding threshold."""
    deltas = np.diff(steering_history)
    significant = deltas[np.abs(deltas) > threshold]
    sign_changes = np.sum(np.diff(np.sign(significant)) != 0)
    return sign_changes > max_sign_changes
```

## 4. Acceptance Criteria

### Phase A: `trajectory_tracker.py` (New File)

- [ ] A.1 `PurePursuitTracker` class with `compute_steering()` and `is_trajectory_ended()`
- [ ] A.2 Lookahead distance is speed-adaptive: `L_d = clip(k * speed, L_min, L_max)`
- [ ] A.3 All inputs from `PointLane` interface only (`.local_coordinates()`, `.position()`, `.length`)
- [ ] A.4 Output clipped to `[-MAX_STEERING, MAX_STEERING]`
- [ ] A.5 No division-by-zero for zero speed or coincident points
- [ ] A.6 No external dependencies beyond numpy and math

### Phase B: `hierarchical_policy.py` (Modify)

- [ ] B.1 Import and instantiate `PurePursuitTracker` in `__init__`
- [ ] B.2 `act()` dispatches to `trajectory_tracker.compute_steering()` when state is EXECUTING or ABORTING
- [ ] B.3 `act()` dispatches to `self.steering_control()` (parent class) when state is IDLE
- [ ] B.4 `self.heading_pid` and `self.lateral_pid` are reset when transitioning from EXECUTING/ABORTING back to IDLE (prevent stale integral term)
- [ ] B.5 Existing `reset()` method also resets tracker state if any

### Phase C: `tests/test_trajectory_tracker.py` (New File)

- [ ] C.1 Test cases 1~7 from section 3.3 all pass
- [ ] C.2 振荡检测: 换道轨迹跟踪过程中 steering 符号变化 ≤ 4 次 (threshold=0.05)
- [ ] C.3 收敛性: 直线偏移修正后 lateral_error < 0.3m within 50 steps (dt=0.05)
- [ ] C.4 全部测试可通过 `pytest tests/test_trajectory_tracker.py -v` 独立运行

### Phase D: Integration Verification

- [ ] D.1 `HierarchicalExpertIDMPolicy` 在换道过程中 steering 输出平滑, 无高频振荡
- [ ] D.2 换道轨迹的横向偏差 (相对 PointLane) 峰值 < 1.0m
- [ ] D.3 换道完成后回到 IDLE, 车道跟随恢复正常 (PID 无残留积分)
- [ ] D.4 Abort 轨迹跟踪平滑回原车道

## 5. Execution Order

```
Phase A → Phase C → Phase B → Phase D
```

- A 和 C 可以并行编写 (C 依赖 A 的接口但可以 mock)
- B 依赖 A 完成
- D 依赖 B 完成

## 6. Notes

- **不修改** `ExpertIDMPolicy.steering_control()` — 保持向后兼容
- **不修改** `LaneChangeManager` — 它只负责返回 steering_target，不关心跟踪方式
- **不修改** `QuinticLaneChangePlanner` — 轨迹生成与跟踪解耦
- PID reset (B.4) 是关键细节: 换道结束回到车道跟随时, 如果 PID 积分项残留,
  会导致短暂的转向偏差。在状态回到 IDLE 时调用 `self.heading_pid.reset()`
  和 `self.lateral_pid.reset()`
