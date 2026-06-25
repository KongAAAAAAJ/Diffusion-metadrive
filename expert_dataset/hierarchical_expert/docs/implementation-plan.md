# HierarchicalExpertIDMPolicy — Implementation Plan

> Date: 2026-03-26
> Spec: [design-spec.md](design-spec.md)
> Target directory: `expert_dataset/hierarchical_expert/`
> 交付方式: 按 Phase 顺序执行，每个 Phase 完成后需通过对应验收指标

---

## 关键上下文（执行前必读）

### 代码库位置
- 项目根目录: `/home/kong/diffusion_codes/Diffusion-meta/new_edit_expert/Diffusion-metadrive/`
- 新模块目录: `expert_dataset/hierarchical_expert/`
- 设计规格: `expert_dataset/hierarchical_expert/docs/design-spec.md`

### 必须阅读的现有文件（按依赖顺序）
1. `metadrive/component/vehicle/PID_controller.py` — PIDController 接口
2. `metadrive/policy/base_policy.py` — BasePolicy 基类
3. `metadrive/policy/idm_policy.py` — IDMPolicy + FrontBackObjects（纵向控制、换道检测、周围车辆感知）
4. `expert_dataset/expert_idm_policy.py` — ExpertIDMPolicy（preview-PID 横向控制）
5. `metadrive/component/lane/point_lane.py` — PointLane 接口（position, local_coordinates, heading_theta_at, length, width）
6. `expert_dataset/trajectory_correction.py` — TrajectoryMode 枚举（参考换道模式分类）
7. `expert_dataset/collect_expert.py` — 数据采集流程（了解 policy 如何被创建和使用）

### 约束
- **不修改**任何现有文件（ExpertIDMPolicy、IDMPolicy、PIDController 等保持不变）
- 所有新代码写在 `expert_dataset/hierarchical_expert/` 下
- Python 3.9+ 兼容，仅依赖 numpy（测试可用 pytest）
- 继承 `ExpertIDMPolicy`，复用其 `steering_control()` 和 `acceleration()` 方法
- `act()` 返回值格式不变: `[steering, acceleration]`

---

## Phase 1: DrivingStyleProfile — 风格参数与采样

### 目标
创建 `driving_style.py`，实现风格参数容器和采样器。

### 文件: `expert_dataset/hierarchical_expert/driving_style.py`

### 具体任务

1. 创建 `DrivingStyleProfile` dataclass:
   ```python
   @dataclass
   class DrivingStyleProfile:
       aggression: float = 0.5          # 元参数, 0=保守 1=激进
       # IDM 纵向
       desired_speed_ratio: float = 1.0 # 相对限速倍率
       time_headway: float = 1.5        # [s]
       max_accel: float = 1.5           # [m/s²]
       comfortable_decel: float = 2.0   # [m/s²]
       min_jam_distance: float = 2.0    # [m]
       velocity_exponent: float = 4.0
       # 换道决策
       politeness: float = 0.2          # MOBIL 礼让因子
       lane_change_threshold: float = 0.2
       # 安全
       min_front_ttc: float = 3.0       # [s]
       min_rear_ttc: float = 3.5        # [s]
       reaction_time: float = 0.5       # [s]
       # 轨迹
       desired_lateral_speed: float = 1.0  # [m/s]
   ```

2. 创建 `STYLE_PRESETS` 字典，包含 `"conservative"`, `"normal"`, `"aggressive"` 三个预设:
   - conservative: aggression=0.2, time_headway=2.0, max_accel=1.0, comfortable_decel=1.5, min_jam_distance=4.0, velocity_exponent=6.0, politeness=0.5, lane_change_threshold=0.5, min_front_ttc=4.0, min_rear_ttc=5.0, reaction_time=0.7, desired_lateral_speed=0.7
   - normal: 使用默认值
   - aggressive: aggression=0.85, time_headway=0.8, max_accel=2.5, comfortable_decel=3.0, min_jam_distance=1.0, velocity_exponent=2.0, politeness=0.0, lane_change_threshold=0.05, min_front_ttc=2.0, min_rear_ttc=2.5, reaction_time=0.3, desired_lateral_speed=1.5

3. 创建 `StyleSampler` 类:
   - `__init__(self, seed: int = 0)` — 初始化 `np.random.RandomState`
   - `sample(self) -> DrivingStyleProfile` — aggression 从 Beta(2,5) 采样，其余参数在 conservative/aggressive 之间按 aggression 线性插值，叠加 ±10% 高斯噪声 `N(1.0, 0.1)`

4. 创建 `__init__.py`，导出 `DrivingStyleProfile`, `STYLE_PRESETS`, `StyleSampler`

### 测试文件: `expert_dataset/hierarchical_expert/tests/test_driving_style.py`

### 验收指标
- [ ] `DrivingStyleProfile()` 默认构造成功，所有字段类型为 float/int
- [ ] `STYLE_PRESETS` 包含 3 个 key，每个 value 是 `DrivingStyleProfile` 实例
- [ ] `StyleSampler(seed=42).sample()` 返回 `DrivingStyleProfile`，aggression ∈ [0,1]
- [ ] 相同 seed 两次 `sample()` 返回不同结果（内部状态递进），但相同 seed 新建的 sampler 首次结果相同（可复现）
- [ ] 采样 1000 次，aggression 均值 < 0.4（Beta(2,5) 偏保守）
- [ ] 所有参数 > 0（噪声不会导致负值，需 clip）

---

## Phase 2: SafetyAssessor — 安全评估

### 目标
创建 `safety_assessor.py`，实现 gap acceptance + TTC + RSS 安全评估。

### 文件: `expert_dataset/hierarchical_expert/safety_assessor.py`

### 具体任务

1. 创建 `GapInfo` dataclass:
   ```python
   @dataclass
   class GapInfo:
       front_distance: float   # 目标车道前车距离 [m]
       front_speed: float      # 目标车道前车速度 [m/s]
       rear_distance: float    # 目标车道后车距离 [m]
       rear_speed: float       # 目标车道后车速度 [m/s]
       front_ttc: float        # 与前车 TTC [s]
       rear_ttc: float         # 后车与自车 TTC [s]
   ```

2. 创建 `SafetyAssessor` 类:

   **`__init__(self, style: DrivingStyleProfile)`**

   **`compute_gap_info(self, direction, ego_speed, perception: FrontBackObjects) -> GapInfo`**
   - direction=-1 时取 left_front/left_back; direction=+1 时取 right_front/right_back
   - 提取距离和速度（无车时 distance=999.0, speed=0.0）
   - 计算 TTC:
     - `front_ttc = front_distance / max(ego_speed_m_s - front_speed_m_s, 0.01)` 若 ego 更快; 否则 inf
     - `rear_ttc = rear_distance / max(rear_speed_m_s - ego_speed_m_s, 0.01)` 若后车更快; 否则 inf
   - 注意: FrontBackObjects 中的速度属性是 `speed_km_h`，需转换为 m/s（除以 3.6）
   - 无邻车道时返回全 inf/999 的 GapInfo

   **`_rss_safe_distance(self, v_rear, v_front, reaction_time, a_max_brake, a_min_brake) -> float`**
   - `d = v_rear * reaction_time + v_rear² / (2*a_min_brake) - v_front² / (2*a_max_brake)`
   - 返回 `max(d, 2.0)`
   - 所有速度参数单位 m/s

   **`is_gap_acceptable(self, direction, ego_speed, perception) -> bool`**
   - 调用 `compute_gap_info()`
   - 检查: front_distance ≥ style.min_front_gap (默认用 8m 折算 — 注意 style 里没有 min_front_gap/min_rear_gap 字段，需要从 style 推导: `min_front_gap = min_front_ttc * 2.0`, `min_rear_gap = min_rear_ttc * 2.5` 或写为固定公式)
   - **简化**: 直接用 TTC 和 RSS 做判断，不再单独存 min_front_gap/min_rear_gap:
     - `front_ttc ≥ style.min_front_ttc`
     - `rear_ttc ≥ style.min_rear_ttc`
     - `rear_distance ≥ _rss_safe_distance(rear_speed, ego_speed, style.reaction_time, style.comfortable_decel, style.comfortable_decel * 0.75)`
   - 任一不满足返回 False

   **`gap_acceptance_score(self, direction, ego_speed, perception) -> float`**
   - 归一化到 [-1, 1]
   - `score = min(front_ttc / style.min_front_ttc, rear_ttc / style.min_rear_ttc, rss_ratio) - 1.0`
   - `rss_ratio = rear_distance / rss_safe_distance`
   - clip 到 [-1, 1]

   **`monitor_ongoing(self, direction, ego_speed, perception) -> bool`**
   - 同 `is_gap_acceptable` 但阈值打 0.6 折:
     - `front_ttc ≥ style.min_front_ttc * 0.6`
     - `rear_ttc ≥ style.min_rear_ttc * 0.6`
   - 硬限制: front_distance < 3.0 或 rear_distance < 3.0 → False

### 测试文件: `expert_dataset/hierarchical_expert/tests/test_safety_assessor.py`

### 验收指标
- [ ] `_rss_safe_distance(20, 15, 0.5, 2.0, 1.5)` 返回值 > 0 且数学上正确（手算验证: 20*0.5 + 20²/(2*1.5) - 15²/(2*2.0) = 10 + 133.3 - 56.25 = 87.08）
- [ ] TTC 计算: ego=30m/s, front=25m/s, gap=50m → ttc=10s; ego=20m/s, front=25m/s → ttc=inf
- [ ] `is_gap_acceptable` 在 front_ttc=1.0s（低于阈值 3.0s）时返回 False
- [ ] `is_gap_acceptable` 在 front_ttc=5.0s, rear_ttc=6.0s, rss 满足时返回 True
- [ ] `monitor_ongoing` 阈值为 is_gap_acceptable 的 60%: front_ttc=2.0s 时 monitor 返回 True（2.0 > 3.0*0.6=1.8）但 is_gap_acceptable 返回 False（2.0 < 3.0）
- [ ] `gap_acceptance_score` 返回值 ∈ [-1, 1]
- [ ] 不依赖 MetaDrive 运行时 — 测试中 mock FrontBackObjects 即可

---

## Phase 3: QuinticLaneChangePlanner — 轨迹生成

### 目标
创建 `trajectory_planner.py`，实现五次多项式换道轨迹生成。

### 文件: `expert_dataset/hierarchical_expert/trajectory_planner.py`

### 具体任务

1. 创建 `LaneChangeTrajectoryConfig` dataclass:
   ```python
   @dataclass
   class LaneChangeTrajectoryConfig:
       min_duration: float = 3.0       # [s]
       max_duration: float = 6.0       # [s]
       num_sample_points: int = 60
       max_curvature_factor: float = 0.8
   ```

2. 创建 `QuinticLaneChangePlanner` 类:

   **`__init__(self, style: DrivingStyleProfile, config: LaneChangeTrajectoryConfig = None)`**

   **`_solve_quintic_coefficients(self, d0, d0_dot, d0_ddot, df, df_dot, df_ddot, T) -> np.ndarray`**
   - 返回 shape (6,) 的系数数组 [a0..a5]
   - a0 = d0, a1 = d0_dot, a2 = d0_ddot / 2
   - 求解 3×3 线性方程组得 a3, a4, a5（见 design-spec）

   **`_evaluate_quintic(self, coeffs, t) -> tuple[float, float, float]`**
   - 返回 (d, d_dot, d_ddot) 在时刻 t 的值
   - d = Σ coeffs[k] * t^k
   - d_dot = Σ k * coeffs[k] * t^(k-1)
   - d_ddot = Σ k*(k-1) * coeffs[k] * t^(k-2)

   **`_compute_duration(self, ego_speed, lateral_distance, urgency) -> float`**
   - `base_T = |lateral_distance| / max(style.desired_lateral_speed, 0.5)`
   - `speed_factor = clip(ego_speed / 20.0, 0.8, 1.5)`
   - `urgency_factor = 1.0 - 0.3 * urgency`
   - `T = clip(base_T * speed_factor * urgency_factor, min_duration, max_duration)`

   **`_validate_curvature(self, coeffs, T, ego_speed, wheel_base=2.8) -> bool`**
   - 在 T 上均匀采样 30 个点
   - κ = |d_ddot| / (1 + d_dot²)^1.5
   - κ_max = tan(0.3149) / wheel_base * max_curvature_factor
   - 所有点 κ < κ_max 返回 True

   **`plan(self, ego_position, ego_heading, ego_speed, source_lane, target_lane, direction, urgency=0.0) -> Optional[PointLane]`**
   - 计算 ego 在 source_lane 上的 (longitudinal, lateral)
   - 计算目标横向偏移: `df = target_lane 中心相对于 source_lane 的横向距离`
     - 方法: `_, df = source_lane.local_coordinates(target_lane.position(target_long, 0))` 其中 target_long 为对应纵向位置
   - 估算当前横向速度: `d0_dot = ego_speed * sin(ego_heading - lane_heading)`
   - 调用 `_compute_duration()` 得 T
   - 调用 `_solve_quintic_coefficients(d0=lateral, d0_dot, 0, df, 0, 0, T)`
   - 调用 `_validate_curvature()`; 不通过则 T *= 1.3 重试，最多 3 次
   - 在时间域 [0, T] 上采样 num_sample_points 个点:
     - `s(t) = longitudinal + ego_speed * t` (纵向匀速假设)
     - `d(t)` 从五次多项式
     - `(x, y) = source_lane.position(clip(s, 0, length), d)`
   - 用采样点构建 `PointLane(center_line_points=points_array, width=source_lane.width)`
   - 返回 PointLane; 3 次重试仍不通过返回 None

   **`plan_abort(self, ego_position, ego_heading, ego_speed, source_lane) -> PointLane`**
   - 计算当前 (longitudinal, lateral) on source_lane
   - d0_dot = ego_speed * sin(heading_diff)
   - df = 0.0 (回到 source_lane 中心)
   - T = clip(|lateral| / max(desired_lateral_speed * 0.8, 0.3), 2.0, 5.0)
   - 同上生成 PointLane

### 测试文件: `expert_dataset/hierarchical_expert/tests/test_quintic_polynomial.py`

### 验收指标
- [ ] `_solve_quintic_coefficients(0, 0, 0, 3.7, 0, 0, 4.0)` 返回 shape (6,) 数组
- [ ] 用返回系数 evaluate 在 t=0 得 d≈0, t=T 得 d≈3.7, d_dot≈0, d_ddot≈0（误差 < 1e-6）
- [ ] 边界条件精度: |d(0)-d0| < 1e-10, |d(T)-df| < 1e-6, |d'(T)| < 1e-6, |d''(T)| < 1e-6
- [ ] `_validate_curvature` 对合理参数（lane_width=3.7, T=4s, speed=8m/s）返回 True
- [ ] `_validate_curvature` 对极端参数（lane_width=3.7, T=0.5s, speed=30m/s）返回 False
- [ ] `plan()` 对 mock 的 StraightLane 场景返回非 None 的 PointLane
- [ ] 返回的 PointLane 起点接近 ego 当前位置（误差 < 1m）
- [ ] 返回的 PointLane 终点横向偏移接近目标车道中心（误差 < 0.5m）
- [ ] `plan_abort()` 返回的 PointLane 终点横向偏移接近 0（回到原车道中心）
- [ ] 纯数学测试，不依赖 MetaDrive 运行时（测试中用 StraightLane mock 或直接构造 StraightLane）

---

## Phase 4: LaneChangeManager — 状态机与决策

### 目标
创建 `lane_change_manager.py`，实现状态机 + 效用决策 + 组件协调。

### 文件: `expert_dataset/hierarchical_expert/lane_change_manager.py`

### 具体任务

1. 创建 `ManeuverState` 枚举:
   ```python
   class ManeuverState(IntEnum):
       IDLE = 0
       EXECUTING = 1
       ABORTING = 2
   ```

2. 创建 `ManeuverCommand` dataclass:
   ```python
   @dataclass
   class ManeuverCommand:
       direction: int        # -1=左, +1=右
       target_lane: object
       source_lane: object
       urgency: float
       is_mandatory: bool
   ```

3. 创建 `LaneChangeManager`:

   **`__init__(self, safety: SafetyAssessor, planner: QuinticLaneChangePlanner, style: DrivingStyleProfile)`**
   - `self.state = ManeuverState.IDLE`
   - `self.active_trajectory = None`  # 当前换道 PointLane
   - `self.active_command = None`
   - `self.source_lane = None`
   - `self.cooldown_timer = 0`

   **`update(self, ego, all_objects, routing_target_lane, current_lanes, next_lanes)`**
   - 返回 `(steering_target, acc_front_obj, acc_front_dist)`
   - steering_target: PointLane（换道中）或 lane 对象（车道跟随）

   **IDLE 状态处理**:
   - `cooldown_timer > 0` 时递减并跳过决策
   - 调用 `_compute_route_urgency(routing_target_lane, current_lanes, next_lanes, ego)`
   - 对可行方向（左/右）调用 `_compute_lane_change_utility()`
   - 取最大效用，判断是否超过阈值
   - 超过则: 调用 `planner.plan()` 生成轨迹
     - 成功: state → EXECUTING, 记录 active_trajectory/command/source_lane
     - 失败: 保持 IDLE
   - 未超过: 正常车道跟随

   **EXECUTING 状态处理**:
   - 调用 `safety.monitor_ongoing()` 检查安全
   - 不安全: 调用 `planner.plan_abort()`, state → ABORTING
   - 安全: 检查是否完成（ego 已到达目标车道中心附近，lateral_error < 0.5m）
     - 完成: state → IDLE, cooldown_timer = style.lane_change_cooldown（默认 30）, 更新 routing_target_lane
     - 未完成: 返回 active_trajectory 作为 steering_target

   **ABORTING 状态处理**:
   - 跟踪 abort 轨迹
   - 完成（lateral_error < 0.5m on source_lane）: state → IDLE, cooldown_timer = style.lane_change_cooldown
   - 未完成: 继续跟踪

   **`_compute_route_urgency(self, routing_target_lane, current_lanes, next_lanes, ego) -> tuple[float, bool, int]`**
   - 返回 (urgency, is_mandatory, direction)
   - 逻辑参考 `IDMPolicy.lane_change_policy()` 的 lane_num_diff 处理:
     - `lane_num_diff = len(current_lanes) - len(next_lanes)` if next_lanes else 0
     - 如果 lane_num_diff > 0 且当前车道不在合法范围:
       - 计算合法 index 范围
       - `remaining = lane.length - ego_longitudinal`
       - `urgency = clip(1.0 - remaining / 200.0, 0.0, 1.0)`
       - direction: 当前 idx > 合法最大 → -1(左); 否则 +1(右)
       - is_mandatory = True
     - 否则: (0.0, False, 0)

   **`_compute_lane_change_utility(self, direction, ego, perception, urgency, is_mandatory) -> float`**
   - perception: `FrontBackObjects`
   - 读取当前车道/目标车道的前车速度，计算 MOBIL 加速度收益:
     - `a_current` = 基于当前车道前车的 IDM 加速度
     - `a_target` = 基于目标车道前车的 IDM 加速度
     - `u_mobil = a_target - a_current`（简化版，不计算后车影响）
   - `u_route = urgency * style.route_urgency_weight`（默认 weight=2.0）
   - `u_safety = safety.gap_acceptance_score(direction, ego_speed, perception)`
   - `u_style = 0.0`（简化，保留接口）
   - `U = 0.3 * u_mobil + 0.4 * u_route + 0.2 * u_safety + 0.1 * u_style`
   - 强制换道 (is_mandatory): 只要 `safety.is_gap_acceptable()` 返回 True 就执行（忽略效用阈值）

### 测试文件: `expert_dataset/hierarchical_expert/tests/test_lane_change_manager.py`

### 验收指标
- [ ] 初始状态为 IDLE
- [ ] IDLE → EXECUTING: 当效用 > threshold 且 gap acceptable 且 planner 返回非 None
- [ ] EXECUTING → IDLE: 当 ego 到达目标车道中心 (lateral_error < 0.5m)
- [ ] EXECUTING → ABORTING: 当 monitor_ongoing 返回 False
- [ ] ABORTING → IDLE: 当回到原车道中心
- [ ] cooldown: 刚完成换道后 cooldown_timer > 0 期间不触发新换道
- [ ] 强制换道: is_mandatory=True 时跳过效用阈值检查
- [ ] 无邻车道时不触发换道（FrontBackObjects.left_lane_exist()/right_lane_exist() 返回 False）
- [ ] `_compute_route_urgency` 对 lane_num_diff=0 返回 urgency=0
- [ ] 测试用 mock 对象，不依赖 MetaDrive 运行时

---

## Phase 5: HierarchicalExpertIDMPolicy — 主策略集成

### 目标
创建 `hierarchical_policy.py`，将四个组件集成为完整策略。

### 文件: `expert_dataset/hierarchical_expert/hierarchical_policy.py`

### 具体任务

1. 创建 `HierarchicalExpertIDMPolicy(ExpertIDMPolicy)`:

   **`__init__(self, control_object, random_seed=0, style_profile=None)`**
   - `super().__init__(control_object, random_seed)`
   - `self.style = style_profile or DrivingStyleProfile()`
   - 用风格参数覆盖 IDM 属性:
     ```python
     self.DISTANCE_WANTED = self.style.min_jam_distance
     self.TIME_WANTED = self.style.time_headway
     self.DELTA = self.style.velocity_exponent
     self.ACC_FACTOR = self.style.max_accel
     self.DEACC_FACTOR = -self.style.comfortable_decel
     self.target_speed = self.NORMAL_SPEED * self.style.desired_speed_ratio
     ```
   - 初始化子组件:
     ```python
     self.safety = SafetyAssessor(self.style)
     self.planner = QuinticLaneChangePlanner(self.style)
     self.manager = LaneChangeManager(self.safety, self.planner, self.style)
     ```

   **`act(self, *args, **kwargs)`**
   - 调用 `self.move_to_next_road()` 维护 routing_target_lane
   - 获取 all_objects: `self.control_object.lidar.get_surrounding_objects(self.control_object)`
   - 获取 current_lanes/next_lanes from navigation
   - 用 try/except 包裹核心逻辑（与 IDMPolicy.act 保持一致的容错）:
     ```python
     try:
         steering_target, acc_front_obj, acc_front_dist = self.manager.update(
             ego=self.control_object,
             all_objects=all_objects,
             routing_target_lane=self.routing_target_lane,
             current_lanes=current_lanes,
             next_lanes=next_lanes,
         )
     except Exception:
         # fallback: 与原 IDMPolicy 相同的降级行为
         acc_front_obj = None
         acc_front_dist = 5
         steering_target = self.routing_target_lane
     ```
   - `steering = self.steering_control(steering_target)`
   - `acc = self.acceleration(acc_front_obj, acc_front_dist)`
   - 写入 action_info:
     ```python
     self.action_info["action"] = [steering, acc]
     self.action_info["maneuver_state"] = self.manager.state.name
     self.action_info["style_aggression"] = self.style.aggression
     ```
   - 返回 `[steering, acc]`

   **`reset(self)`**
   - `super().reset()`
   - `self.manager.state = ManeuverState.IDLE`
   - `self.manager.active_trajectory = None`
   - `self.manager.active_command = None`
   - `self.manager.cooldown_timer = 0`

2. 更新 `__init__.py` 导出所有公开类

### 验收指标
- [ ] `HierarchicalExpertIDMPolicy` 是 `ExpertIDMPolicy` 的子类
- [ ] `act()` 返回 `[float, float]` 格式，steering ∈ [-1, 1], acc 为有限浮点数
- [ ] `action_info` 包含 "maneuver_state" 和 "style_aggression" 键
- [ ] `reset()` 将 manager 状态恢复到 IDLE
- [ ] 当 manager 处于 IDLE 时行为等价于父类 ExpertIDMPolicy（纵向 IDM + 横向 PID 跟踪车道中心线）
- [ ] 用不同 DrivingStyleProfile 构造的实例，IDM 参数确实不同（如 DISTANCE_WANTED、TIME_WANTED）

---

## Phase 6: 集成测试 — 独立仿真验证

### 目标
复用 `expert_idm_policy.py` 中的 MockVehicle + _simulate_step 模式，在不启动 MetaDrive 的情况下验证完整换道流程。

### 文件: `expert_dataset/hierarchical_expert/tests/test_integration.py`

### 测试场景

**场景 A: 双车道直道换道**
- 构建两条平行 StraightLane（间距 3.7m）
- ego 在右车道，前方 30m 放置慢车（mock 对象）
- 期望: manager 从 IDLE → EXECUTING → IDLE，ego 最终在左车道

**场景 B: 换道中止**
- 同场景 A，但在换道执行到一半时在目标车道后方注入一辆快车
- 期望: manager 从 EXECUTING → ABORTING → IDLE，ego 回到原车道

**场景 C: 强制换道（车道数减少）**
- 三车道 → 两车道，ego 在即将消失的车道上
- 期望: 强制触发换道，urgency > 0

**场景 D: 风格差异**
- 同一场景分别用 conservative 和 aggressive 风格
- 期望: aggressive 换道时间更短（T 更小）

### 验收指标
- [ ] 场景 A: 换道完成后 ego 横向偏移接近目标车道中心（误差 < 0.8m）
- [ ] 场景 A: 换道过程中轨迹连续（相邻点间距 < 2m，无跳变）
- [ ] 场景 B: 最终 ego 回到原车道（横向偏移 < 0.8m）
- [ ] 场景 C: manager.active_command.is_mandatory == True
- [ ] 场景 D: aggressive 的换道步数 < conservative 的换道步数
- [ ] 所有场景: steering 输出无 NaN/Inf
- [ ] 所有场景: 不依赖 MetaDrive engine（纯 numpy + mock）

---

## Phase 依赖关系

```
Phase 1 (DrivingStyle) ──┐
                         ├──► Phase 4 (LaneChangeManager)
Phase 2 (Safety)    ─────┤                │
                         │                ▼
Phase 3 (Planner)   ─────┘    Phase 5 (HierarchicalPolicy)
                                          │
                                          ▼
                              Phase 6 (Integration Tests)
```

**可并行**: Phase 1、2、3 之间无硬依赖（Phase 2/3 依赖 Phase 1 的 DrivingStyleProfile 类型定义，但可以先用默认值 mock）

**必须串行**: Phase 4 依赖 1+2+3; Phase 5 依赖 4; Phase 6 依赖 5

---

## 通用编码规范

1. 所有文件顶部: `from __future__ import annotations`
2. 类型注解: 使用 `from typing import Optional, Tuple` 等
3. 数值计算: 仅依赖 numpy，不引入 scipy
4. 所有速度内部统一使用 m/s，与外部接口交互时注意 km/h → m/s 转换（÷3.6）
5. 所有角度内部统一使用 rad
6. 浮点除法避免除零: `max(denominator, 1e-6)` 或 `not_zero()` 工具函数
7. dataclass 使用 `@dataclass` 装饰器，不需要 `frozen=True`（需要运行时修改）
8. 每个文件末尾无空行或仅一个空行
9. import 顺序: stdlib → third-party (numpy) → local
