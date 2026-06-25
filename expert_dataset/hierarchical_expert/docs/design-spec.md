# HierarchicalExpertIDMPolicy — Design Specification

> Date: 2026-03-26
> Status: Approved
> Location: `expert_dataset/hierarchical_expert/`

## 1. Overview

将 `ExpertIDMPolicy` 从"隐式车道切换"升级为"显式分层决策 + 轨迹生成"架构。
换道不再通过切换 `target_lane` 后让 PID 直接跟踪目标车道中心线，而是经过
**战略决策 → 安全评估 → 轨迹生成 → 轨迹跟踪** 四层流水线。

## 2. Architecture

```
HierarchicalExpertIDMPolicy.act()
│
├── LaneChangeManager.update()           ← 战略层
│   ├── _compute_route_urgency()         ← 路由紧迫度
│   ├── _compute_lane_change_utility()   ← 效用函数 (U_MOBIL + U_route + U_safety + U_style)
│   ├── SafetyAssessor.is_gap_acceptable()   ← 安全门控
│   ├── QuinticLaneChangePlanner.plan()      ← 轨迹生成
│   └── SafetyAssessor.monitor_ongoing()     ← 执行中监测
│
├── steering = steering_control(active_trajectory or target_lane)   ← 跟踪层
└── acc = acceleration(front_obj, dist)                             ← IDM 纵向
```

### 2.1 Maneuver State Machine

```
IDLE ──(utility > threshold & gap ok)──► EXECUTING
  ▲                                         │
  │              ◄──(complete)──────────────┘
  │              ◄──(abort: unsafe)─── ABORTING
  │                                         │
  └──────────(abort complete)───────────────┘
```

States:
- **IDLE**: 车道跟随，每步评估换道效用
- **EXECUTING**: 跟踪五次多项式生成的 PointLane 轨迹
- **ABORTING**: 安全监测触发中止，跟踪回退轨迹到原车道

## 3. Component Design

### 3.1 LaneChangeManager

**文件**: `lane_change_manager.py`

**职责**: 维护状态机，协调决策/安全/轨迹三个组件。

**关键数据结构**:
```python
class ManeuverState(IntEnum):
    IDLE = 0
    EXECUTING = 1
    ABORTING = 2

@dataclass
class ManeuverCommand:
    direction: int           # -1=左, +1=右
    target_lane: object
    source_lane: object
    urgency: float           # 0~1
    is_mandatory: bool
```

**核心方法**:
- `update(ego, perception, routing_target_lane, current_lanes, next_lanes)`
  → 返回 `(steering_target_lane_or_trajectory, acc_front_obj, acc_front_dist)`
- IDLE 时: 对左右各算效用，取最大值，超过阈值则进入 EXECUTING
- EXECUTING 时: 用 `monitor_ongoing()` 检查安全，不安全则 ABORTING
- ABORTING 时: 跟踪回退轨迹，完成后回 IDLE

**效用函数**:
```
U = w_mobil * U_MOBIL + w_route * U_route + w_safety * U_safety + w_style * U_style
```
- U_MOBIL: `(a_target - a_current) - politeness * Δa_follower`
- U_route: `urgency * route_urgency_weight` (urgency 基于距路段终点距离)
- U_safety: `gap_acceptance_score()` 归一化到 [-1, 1]
- U_style: `left_bias * direction` (正=偏好该方向)

**决策规则**:
- 强制换道: `U_safety > safety_threshold_mandatory` 即执行
- 主动换道: `U > utility_threshold` 且 `U_safety > 0`
- 冷却期: 换道完成后至少 `cooldown_steps` 才能再次换道

### 3.2 SafetyAssessor

**文件**: `safety_assessor.py`

**职责**: 提供换道前安全门控 + 执行中持续监测。

**GapInfo 数据结构**:
```python
@dataclass
class GapInfo:
    front_distance: float
    front_speed: float
    rear_distance: float
    rear_speed: float
    front_ttc: float
    rear_ttc: float
```

**核心方法**:

1. `is_gap_acceptable(direction, ego, perception) -> bool`
   - 最小静态间距: front ≥ min_front_gap, rear ≥ min_rear_gap
   - TTC 检查: front_ttc ≥ min_front_ttc, rear_ttc ≥ min_rear_ttc
   - RSS 安全距离: `d_rss = v_rear * t_react + v_rear²/(2*a_min) - v_ego²/(2*a_max)`

2. `monitor_ongoing(ego, perception, active_command) -> bool`
   - 使用初始阈值的 60% (更宽松，避免过度中止)
   - 硬限制: front/rear < 3.0m 时必须中止

3. `gap_acceptance_score(direction, ego, perception) -> float`
   - 归一化安全分数供效用函数使用

**TTC 计算**:
```
前车TTC = gap / max(v_ego - v_front, ε)
后车TTC = gap / max(v_rear - v_ego, ε)
无接近趋势时返回 inf
```

### 3.3 QuinticLaneChangePlanner

**文件**: `trajectory_planner.py`

**职责**: 用五次多项式在 Frenet 坐标系下生成动力学可行的换道轨迹。

**五次多项式**:
```
d(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
```
边界条件:
- t=0: d=d0(当前横向偏移), d'=d0_dot(横向速度), d''=0
- t=T: d=df(目标车道中心偏移), d'=0, d''=0

**换道时间确定**:
```
T = (|lateral_distance| / desired_lateral_speed) * speed_factor * urgency_factor
speed_factor = clip(ego_speed / 20, 0.8, 1.5)
urgency_factor = 1.0 - 0.3 * urgency
T ∈ [min_duration, max_duration]  (默认 3~6s)
```

**Frenet → Cartesian**: 在时间域采样 (s,d)，其中 s(t) = s0 + v*t，
用 `source_lane.position(s, d)` 转为 Cartesian 坐标，构建 `PointLane`。

**曲率验证**:
```
κ = |d''| / (1 + d'²)^(3/2)
κ_max = tan(δ_max) / wheelbase * 0.8
```
超过限制则增大 T 重试（最多 3 次）。

**Abort 轨迹**: 同样用五次多项式，从当前位置回到原车道中心。

### 3.4 DrivingStyleProfile

**文件**: `driving_style.py`

**设计原则**: 简化——用 `aggression` 元参数驱动所有子参数插值。

```python
@dataclass
class DrivingStyleProfile:
    # 元参数
    aggression: float = 0.5          # 0=保守, 1=激进

    # IDM 纵向 (由 aggression 插值填充)
    desired_speed_ratio: float = 1.0
    time_headway: float = 1.5        # [s]
    max_accel: float = 1.5           # [m/s²]
    comfortable_decel: float = 2.0   # [m/s²]
    min_jam_distance: float = 2.0    # [m]
    velocity_exponent: float = 4.0

    # 换道决策
    politeness: float = 0.2
    lane_change_threshold: float = 0.2

    # 安全 (由 aggression 插值)
    min_front_ttc: float = 3.0       # [s]
    min_rear_ttc: float = 3.5        # [s]
    reaction_time: float = 0.5       # [s]

    # 轨迹
    desired_lateral_speed: float = 1.0  # [m/s]
```

**采样器**: `StyleSampler.sample()` — aggression 从 Beta(2,5) 采样（偏保守），
其余参数在 conservative/aggressive 两端间线性插值 + ±10% 高斯噪声。

**预设**: conservative / normal / aggressive 三个模板。

## 4. Integration with Existing Code

### 4.1 HierarchicalExpertIDMPolicy

**文件**: `hierarchical_policy.py`

继承 `ExpertIDMPolicy`，重写 `act()`:

```python
class HierarchicalExpertIDMPolicy(ExpertIDMPolicy):
    def __init__(self, control_object, random_seed=0, style_profile=None):
        super().__init__(control_object, random_seed)
        self.style = style_profile or DrivingStyleProfile()
        # 用风格参数覆盖 IDM 类属性
        self._apply_style_to_idm()
        # 初始化子组件
        self.safety = SafetyAssessor(self.style)
        self.planner = QuinticLaneChangePlanner(self.style)
        self.manager = LaneChangeManager(self.safety, self.planner, self.style)

    def act(self, *args, **kwargs):
        success = self.move_to_next_road()
        all_objects = self.control_object.lidar.get_surrounding_objects(self.control_object)
        current_lanes = self.control_object.navigation.current_ref_lanes
        next_lanes = self.control_object.navigation.next_ref_lanes

        # 分层决策
        steering_target, acc_front_obj, acc_front_dist = self.manager.update(
            ego=self.control_object,
            all_objects=all_objects,
            routing_target_lane=self.routing_target_lane,
            current_lanes=current_lanes,
            next_lanes=next_lanes,
        )

        steering = self.steering_control(steering_target)
        acc = self.acceleration(acc_front_obj, acc_front_dist)
        action = [steering, acc]
        self.action_info["action"] = action
        self.action_info["maneuver_state"] = self.manager.state.name
        return action
```

### 4.2 collect_expert.py 集成

在 policy 创建时注入 `StyleSampler`:
```python
sampler = StyleSampler(seed=episode_seed)
style = sampler.sample()
policy = HierarchicalExpertIDMPolicy(vehicle, random_seed, style_profile=style)
```

### 4.3 不修改的文件

- `ExpertIDMPolicy` 本身不改动（保持向后兼容）
- `IDMPolicy` 不改动
- `PIDController` 不改动
- `trajectory_correction.py` 不改动

## 5. File Structure

```
expert_dataset/hierarchical_expert/
├── __init__.py
├── hierarchical_policy.py       # HierarchicalExpertIDMPolicy
├── lane_change_manager.py       # LaneChangeManager + ManeuverState
├── safety_assessor.py           # SafetyAssessor + GapInfo
├── trajectory_planner.py        # QuinticLaneChangePlanner
├── driving_style.py             # DrivingStyleProfile + StyleSampler + PRESETS
├── tests/
│   ├── __init__.py
│   ├── test_quintic_polynomial.py
│   ├── test_safety_assessor.py
│   ├── test_lane_change_manager.py
│   ├── test_driving_style.py
│   └── test_integration.py
└── docs/
    ├── design-spec.md           # 本文档
    └── implementation-plan.md   # 实施计划
```
