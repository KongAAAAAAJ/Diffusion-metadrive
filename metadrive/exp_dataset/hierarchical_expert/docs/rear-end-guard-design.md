# Rear-End Guard Regulator — Design

> Date: 2026-03-27
> Scope: `HierarchicalExpertIDMPolicy`
> Goal: 在不修改横向控制与换道接口的前提下，优先降低 `rear_end` 追尾碰撞，并尽量提升 episode 通过距离与 `arrive_dest`

---

## 1. 背景与目标

当前 `HierarchicalExpertIDMPolicy` 已经接入：

- `LaneChangeManager`
- `PurePursuitTracker`
- `IntersectionConflictRegulator`

在真实 `run_expert.py --expert-type idm` 运行中，交叉口碰撞已经压低，但主要失败模式仍集中在：

- `rear_end`
- `out_of_road`

本轮只处理第一优先级问题：

- 明显降低 `rear_end`
- 同时优先提升 `arrive_dest`
- 不把优化做成“更早停住、通过距离更短”

---

## 2. 固定约束

- 不修改 `LaneChangeManager`、`QuinticLaneChangePlanner`、`PurePursuitTracker` 的公开接口
- 不改换道状态机
- 不改 `run_expert.py` 的专家类型接线方式
- 第一轮只增强纵向防追尾，不同时处理 `out_of_road`
- 新逻辑必须是对现有 IDM 输出的保守覆盖，而不是替换 IDM

---

## 3. 方案概览

新增一个独立规则类：

```python
class RearEndGuardRegulator:
    def adjust_acceleration(self, ego, front_obj, front_dist, idm_acc) -> float:
        ...

    def reset(self) -> None:
        ...
```

其职责仅限于：

- 利用 ego / 前车相对状态估计追尾风险
- 在追尾风险升高时，对 `a_idm` 施加更保守的减速度上限
- 只允许“更保守”，不允许额外加速

纵向控制链路调整为：

```text
IDM acceleration
  -> RearEndGuardRegulator
  -> IntersectionConflictRegulator
  -> final acceleration
```

这样可以先处理同车道前向碰撞风险，再处理交叉口横向冲突风险。

---

## 4. 核心规则

### 4.1 输入

- `gap = front_dist`
- `v_ego`
- `v_front`
- `closing_speed = max(v_ego - v_front, 0.0)`
- `idm_acc`

`v_front` 的获取方式在本轮固定为：

- 若 `front_obj` 有 `speed` 属性，使用 `float(front_obj.speed)`
- 否则若有 `speed_km_h`，使用 `float(front_obj.speed_km_h) / 3.6`
- 否则若有 `velocity_km_h`，使用 `np.linalg.norm(front_obj.velocity_km_h) / 3.6`
- 若 `front_obj is None` 或以上都不存在，则视为“无可用前车速度信息”，直接返回 `idm_acc`

实现上不新增 env/helper 依赖，也不修改旧版 `expert_idm_policy.py`。

### 4.2 风险量

使用两个量同时约束：

1. 动态安全距离

```text
safe_gap = BASE_MIN_GAP + HEADWAY_GUARD * v_ego
```

2. TTC

```text
ttc = gap / max(closing_speed, eps)
```

若 `closing_speed <= 0`，则视为无主动追尾风险。

### 4.3 决策分层

#### 无风险

- 条件：`gap >= safe_gap` 且 `ttc >= TTC_SOFT`
- 输出：`a_guard = idm_acc`

#### 软保护

- 条件：`gap < safe_gap` 或 `ttc < TTC_SOFT`
- 输出：`a_guard = min(idm_acc, SOFT_BRAKE)`

#### 硬保护

- 条件：`ttc < TTC_HARD` 或 `gap` 显著低于动态安全距离
- 输出：`a_guard = min(idm_acc, HARD_BRAKE)`

最后统一：

- `a_guard = max(a_guard, MAX_BRAKE)` 仅用于限制新增 guard 刹车不要超过本轮允许的最强 guard 制动
- 若 `idm_acc < MAX_BRAKE`，保留 `idm_acc`，不得把更强的原始制动放松回 `MAX_BRAKE`
- 经过 `ACC_RATE_LIMIT` 限制纵向变化率，避免纵向抖动

### 4.4 状态与 reset

`RearEndGuardRegulator` 是轻量 stateful 组件，仅保存上一帧输出加速度用于变化率限制：

- 内部状态：`self._last_accel`
- `reset()` 行为：将 `self._last_accel = None`
- `HierarchicalExpertIDMPolicy.reset()` 中必须调用 `self.rear_end_guard.reset()`

这样可以避免 episode 之间泄露上一轮的纵向限幅历史。

---

## 5. 第一版默认参数

第一版先固定为保守但不过分僵硬的一组：

- `BASE_MIN_GAP = 6.0`
- `HEADWAY_GUARD = 1.6`
- `TTC_SOFT = 3.0`
- `TTC_HARD = 1.6`
- `SOFT_BRAKE = -1.8`
- `HARD_BRAKE = -4.0`
- `MAX_BRAKE = -4.5`
- `ACC_RATE_LIMIT = 1.0`

这组参数的意图是：

- 在 closing speed 明显时更早介入
- 优先阻断“IDM 还在温和反应，但已经来不及”的追尾场景
- 不通过极端刹车直接把 episode 推向 `out_of_road`

---

## 6. 与现有模块的关系

### 6.1 `HierarchicalExpertIDMPolicy`

新增模块与测试文件路径在本轮固定为：

- 新增实现文件：
  - `metadrive/exp_dataset/hierarchical_expert/rear_end_guard.py`
- 修改接线文件：
  - `metadrive/exp_dataset/hierarchical_expert/hierarchical_policy.py`
  - `metadrive/exp_dataset/hierarchical_expert/__init__.py`
- 新增/修改测试文件：
  - `metadrive/exp_dataset/hierarchical_expert/tests/test_rear_end_guard.py`
  - `metadrive/exp_dataset/hierarchical_expert/tests/test_hierarchical_policy.py`
  - `metadrive/exp_dataset/hierarchical_expert/tests/test_integration.py`

在 `act()` 中：

1. 保持现有 steering 路径不变
2. 保持 lane-change 期间 `LANE_CHANGE_SPEED_FACTOR` 不变
3. 先计算现有 `acc = self.acceleration(...)`
4. 再调用 `RearEndGuardRegulator.adjust_acceleration(...)`
5. 再调用 `IntersectionConflictRegulator.adjust_acceleration(...)`

### 6.2 `IntersectionConflictRegulator`

不重写、不删除。

保留它作为纵向链路中的第二层保守覆盖，以避免交叉口碰撞回归。

---

## 7. 诊断输出

为了让真实 `run_expert.py` 更容易定位收益来源，`action_info` 至少新增：

- `rear_end_guard_active`
- `rear_end_guard_acc`
- `rear_end_guard_ttc`
- `rear_end_guard_gap`

如 `front_obj is None`，则：

- `rear_end_guard_active = False`
- `rear_end_guard_ttc = None`
- `rear_end_guard_gap = front_dist`

这组诊断字段只用于运行统计和调试，不参与决策闭环。

---

## 8. 验收标准

### 8.1 代码层

- `python -m pytest metadrive/exp_dataset/hierarchical_expert/tests/test_rear_end_guard.py -q` 通过
- `python -m pytest metadrive/exp_dataset/hierarchical_expert/tests/test_hierarchical_policy.py -q` 通过
- `python -m pytest metadrive/exp_dataset/hierarchical_expert/tests/test_integration.py -q` 通过
- regulator 单测全绿
- `HierarchicalExpertIDMPolicy` 集成测试不回归
- 现有 `intersection` 相关测试继续通过
- steering 行为与换道接口保持不变

### 8.2 真实环境

使用真实 `meta_drive` 环境运行：

```bash
HOME=/tmp XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp \
/home/kong/anaconda3/envs/meta_drive/bin/python -m metadrive.exp_dataset.run_expert \
  --expert-type idm \
  --episodes 10 \
  --render 0 \
  --print-obs-summary 0 \
  --print-idm-debug 0 \
  --print-episode-summary 1 \
  --print-run-summary 1
```

通过标准：

- 固定基线定义为本轮设计时的最近一次 10-episode 统计：
  - `collision_rear_end = 4`
  - `arrive_dest = 1`
  - `out_of_road = 5`
  - `collision_intersection = 0`
- 本轮通过标准锁定为：
  - `collision_rear_end <= 2`
  - `arrive_dest >= 2`
  - `collision_intersection = 0`
  - `out_of_road <= 6`
- `collision_intersection` 保持为 `0`

---

## 9. 非目标

本轮明确不做：

- 换道策略重构
- 横向控制器重写
- `out_of_road` 修复
- 复杂路权推理
- 抢行策略

这些留到下一轮。
