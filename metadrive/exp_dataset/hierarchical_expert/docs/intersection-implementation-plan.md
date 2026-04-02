# Intersection Conflict Speed Regulator — Implementation Plan

> Date: 2026-03-27
> Design input: [intersection.md](intersection.md)
> Target directory: `metadrive/exp_dataset/hierarchical_expert/`
> 目标: 为 `HierarchicalExpertIDMPolicy` 增加交叉口冲突感知的纵向减速让行规则，不替换现有 IDM，不改换道模块接口

---

## 关键上下文（执行前必读）

### 当前接线位置
- `run_expert.py --expert-type idm` 当前实际使用 `HierarchicalExpertIDMPolicy`
- `HierarchicalExpertIDMPolicy.act()` 当前纵向控制链路是：
  - `LaneChangeManager.update(...)`
  - `steering` 计算
  - `acc = self.acceleration(...)`
  - 返回 `[steering, acc]`
- 当前代码库中没有交叉口专用纵向冲突调节器

### 本次固定约束
- **不替换** IDM，只在 IDM 输出后做保守覆盖
- **不改** `LaneChangeManager`、`QuinticLaneChangePlanner`、`PurePursuitTracker` 的公开接口
- **不做** 路权推理、抢行加速、复杂地图 block 类型检测
- **只做** v1 的“减速让行”规则
- 背景车来源直接使用当前 policy 已有的 `all_objects = lidar.get_surrounding_objects(...)`

### 必须阅读的文件
1. `metadrive/exp_dataset/hierarchical_expert/docs/intersection.md`
2. `metadrive/exp_dataset/hierarchical_expert/hierarchical_policy.py`
3. `metadrive/exp_dataset/expert_idm_policy.py`
4. `metadrive/exp_dataset/hierarchical_expert/tests/test_hierarchical_policy.py`
5. `metadrive/exp_dataset/hierarchical_expert/tests/test_integration.py`

---

## 实现总览

### 新增模块
创建 `metadrive/exp_dataset/hierarchical_expert/intersection_regulator.py`，提供：

```python
class IntersectionConflictRegulator:
    def adjust_acceleration(self, ego, all_objects, reference_target, idm_acc, maneuver_state=None) -> float:
        ...
```

### 职责边界
- `IntersectionConflictRegulator` 只负责：
  - 候选背景车筛选
  - ego / 背景车短时预测
  - 冲突识别
  - 所需减速度计算
  - 对 IDM 输出做保守覆盖
- `HierarchicalExpertIDMPolicy` 只负责：
  - 在 `act()` 中实例化并调用 regulator
  - 记录最小诊断信息到 `action_info`

### v1 固定决策
- ego 预测路径来源：
  - `EXECUTING / ABORTING`: 沿 `manager.active_trajectory`
  - 其他状态: 沿当前 lane-follow 目标（`steering_target` 或 `routing_target_lane`）
- 背景车预测模型：常速度直线
- 预测参数：
  - horizon: `3.0s`
  - dt: `0.2s`
  - candidate radius: `40m`
  - conflict distance: `4.5m`
  - safe time gap: `1.5s`
  - max braking: `-4.5 m/s^2`
  - accel rate limit: `1.0 m/s^2` per step
- 融合规则：
  - `a_final = min(a_idm, a_conf_limited)`
- v1 仅允许减速，不允许通过 regulator 主动加速抢行

---

## Phase 1: 规则类与单测

### 目标
实现一个可独立测试的交叉口纵向冲突调节器。

### 文件
- 新建: `metadrive/exp_dataset/hierarchical_expert/intersection_regulator.py`
- 新建测试: `metadrive/exp_dataset/hierarchical_expert/tests/test_intersection_regulator.py`

### 需要实现的接口

#### 1. `IntersectionConflictRegulator`

建议包含以下内部 helper：

- `_filter_candidate_vehicles(self, ego, all_objects) -> list`
- `_predict_ego_path(self, ego, reference_target, maneuver_state) -> np.ndarray`
- `_predict_object_path(self, obj) -> np.ndarray`
- `_find_most_dangerous_conflict(self, ego_path, candidates) -> dict | None`
- `_compute_required_acceleration(self, s_ego, v_ego, s_obj, v_obj) -> float`
- `_limit_accel_change(self, accel_cmd) -> float`

#### 2. 激活与筛选规则
- 不单独做 intersection map-type 检测
- regulator 每步可调用，但只保留距离 ego 小于 `40m` 的动态背景车
- 只处理：
  - 有 `position`
  - 有 `speed` 或 `speed_km_h`
  - 不是 ego 自身
  - 速度大于很小阈值的背景车

#### 3. ego 未来轨迹预测
- 如果 `manejver_state in {EXECUTING, ABORTING}` 且 `reference_target` 是 `PointLane`/轨迹对象：
  - 沿该轨迹按 ego 当前速度做时间采样
- 否则：
  - 沿当前 lane-follow 目标采样
- 输出固定为 shape `(T, 2)` 的未来位置序列

#### 4. 背景车未来轨迹预测
- 使用常速度直线模型：
  - `p(t) = p0 + v * t`
- 若对象缺少 heading 向量，则用当前速度方向或默认前向单位向量兜底

#### 5. 冲突判定
- 对 ego 和每辆候选背景车的未来轨迹按相同时间步比较
- 以“最小距离 + 首次低于冲突阈值的时刻”定义冲突
- 取最危险对象的规则固定为：
  - 优先更小 `time_gap`
  - 若 `time_gap` 接近，则取更小 `min_distance`

#### 6. 所需减速度
- 用文档中的固定公式：

```python
t_obj = s_obj / max(v_obj, eps)
t_target = t_obj + safe_time_gap
a_conf = 2 * (s_ego - v_ego * t_target) / (t_target ** 2)
```

- 然后：
  - `a_conf = min(a_conf, idm_acc)`
  - clip 到 `[max_braking, +inf)`

#### 7. 已承诺通过规则
- 若 ego 的首个冲突时刻 `< 0.8s` 且 `ego.speed > 3.0 m/s`
  - 不执行温和让行
  - 仅当预测 `min_distance < 2.5m` 时才触发 emergency brake
  - 否则保持 `a_idm`

### 单测验收
- [ ] 无候选车时返回 `idm_acc`
- [ ] 候选车存在但未来轨迹不冲突时返回 `idm_acc`
- [ ] 单一横向来车冲突时输出更保守负加速度
- [ ] 多辆冲突车时选择最严格 `a_conf`
- [ ] 输出始终有限且被限幅
- [ ] 已承诺通过场景不会因为轻度冲突停在路口中央

---

## Phase 2: Policy 集成

### 目标
将 regulator 接入 `HierarchicalExpertIDMPolicy` 的纵向控制链路。

### 文件
- 修改: `metadrive/exp_dataset/hierarchical_expert/hierarchical_policy.py`
- 修改: `metadrive/exp_dataset/hierarchical_expert/__init__.py`
- 修改测试: `metadrive/exp_dataset/hierarchical_expert/tests/test_hierarchical_policy.py`

### 具体任务
1. 在 `hierarchical_policy.py` 中导入 `IntersectionConflictRegulator`
2. 在 `__init__` 中新增：
   - `self.intersection_regulator = IntersectionConflictRegulator()`
3. 在 `act()` 中：
   - 保持现有 steering 路径不变
   - 保持 lane-change 期间 `LANE_CHANGE_SPEED_FACTOR` 逻辑不变
   - 在现有 `acc = self.acceleration(...)` 后调用：

```python
acc = self.intersection_regulator.adjust_acceleration(
    ego=self.control_object,
    all_objects=all_objects,
    reference_target=steering_target,
    idm_acc=acc,
    maneuver_state=self.manager.state,
)
```

4. `action_info` 增加最小诊断字段：
   - `intersection_conflict_active`
   - `intersection_conflict_count`
   - `intersection_conflict_acc`
5. 在 `__init__.py` 中导出 `IntersectionConflictRegulator`

### Policy 测试验收
- [ ] `act()` 返回格式仍为 `[steering, acceleration]`
- [ ] regulator 在 IDM 加速度之后调用
- [ ] 最终 `acc <= a_idm`
- [ ] lane-change speed factor 与 regulator 组合后行为可预测
- [ ] steering 路径不受影响

---

## Phase 3: 闭环集成测试

### 目标
在不启动完整 MetaDrive 环境的前提下，用 synthetic crossing 场景验证 ego 会让行。

### 文件
- 修改: `metadrive/exp_dataset/hierarchical_expert/tests/test_integration.py`

### 具体任务
新增 3 个场景：

1. `ego` 直行、横向背景车穿越
- 期望：ego 在冲突窗口前减速让行

2. 横向背景车存在但未来不与 ego 轨迹相交
- 期望：ego 维持原有通过行为，不被无故限速

3. 冲突解除后恢复
- 期望：ego 的纵向控制平滑恢复，不出现连续大幅正负切换

### 集成验收
- [ ] synthetic crossing 场景中 ego 明确比 baseline 更保守
- [ ] 不出现高频纵向振荡
- [ ] 现有 lane-change / abort / mandatory change 测试保持通过

---

## Phase 4: 最终验收与运行口径

### 代码层验收命令

```bash
python -m pytest metadrive/exp_dataset/hierarchical_expert/tests/test_intersection_regulator.py -q
python -m pytest metadrive/exp_dataset/hierarchical_expert/tests/test_hierarchical_policy.py -q
python -m pytest metadrive/exp_dataset/hierarchical_expert/tests/test_integration.py -q
python -m pytest metadrive/exp_dataset/hierarchical_expert/tests -q
```

### 真实环境诊断命令

```bash
python -m metadrive.exp_dataset.run_expert --expert-type idm --episodes 10 --render 0 --print-episode-summary 1 --print-run-summary 1
```

### 通过标准

#### 代码层（必须）
- [ ] 新 regulator 单测全绿
- [ ] 现有 `hierarchical_expert` 全量测试不回归
- [ ] synthetic intersection 场景里 ego 会让行
- [ ] 非交叉口场景下纵向行为与现有 IDM 路径一致

#### 真实环境（Phase B / 本地验收）
- [ ] 交叉口附近可观测到 regulator 干预
- [ ] 非交叉口 route completion 不回归
- [ ] 交叉口冲突事件相对 baseline 减少

> 注：若当前机器的 `run_expert.py` offscreen 环境仍不稳定，则真实环境统计不作为代码层通过前置条件，但需要在本地可运行环境中补做。

---

## 默认实现细节（执行时不要再二次决策）

- `reference_target` 若是轨迹对象，优先按轨迹采样 ego 未来路径
- `reference_target` 若是 lane，对 lane 的 `position(s, 0)` 做时间采样
- 若背景车没有 lane 信息，不阻塞规则运行，直接使用直线常速度预测
- 若 `t_target` 极小或公式分母接近 0，直接返回 `idm_acc`
- regulator 不修改 `target_speed`，只输出最终加速度
- 所有新逻辑放在 `hierarchical_expert/` 目录内，不回改旧版 `expert_idm_policy.py`

---

## 建议提交顺序

1. 先提交 `test_intersection_regulator.py` 红灯
2. 再提交 `intersection_regulator.py` 最小实现
3. 再提交 `hierarchical_policy.py` 集成
4. 最后提交 integration / diagnostics 补强

