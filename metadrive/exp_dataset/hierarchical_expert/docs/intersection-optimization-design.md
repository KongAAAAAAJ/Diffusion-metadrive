# Intersection Conflict Regulator Optimization — Design

> Date: 2026-03-27
> Scope: `HierarchicalExpertIDMPolicy`
> Goal: 在保持现有横向控制、换道接口和 rear-end guard 不变的前提下，显著降低真实 `meta_drive` 运行中的 `intersection` 碰撞，并尽量不牺牲 `arrive_dest`

---

## 1. 背景

在修正 `run_expert.py` 的 episode stats 采集后，真实 `meta_drive` 10-episode 结果显示当前主要问题已从 `rear_end` 转移为交叉口碰撞：

- `collision_intersection = 5`
- `collision_lane_change = 1`
- `collision_rear_end = 0`
- `arrive_dest = 1`
- `out_of_road = 3`
- `lane_changes = 37`

这说明当前系统已经会主动换道、也不再以追尾为主失败，但在交叉口仍然存在明显的误放行问题。

---

## 2. 固定约束

- 不修改 `LaneChangeManager`、`QuinticLaneChangePlanner`、`PurePursuitTracker` 的公开接口
- 不改变 `run_expert.py --expert-type idm` 的专家接线路径
- 保留现有 `RearEndGuardRegulator`
- 优化范围只落在 `IntersectionConflictRegulator` 及其 policy 诊断接线
- 只允许交叉口规则比上游纵向控制更保守，不允许主动加速抢行

---

## 3. 总体方案

继续保留现有 `IntersectionConflictRegulator` 类边界，不新增新的调度层。优化集中在其内部规则：

1. 候选冲突车更早进入评估
2. 冲突排序从“最早冲突”改成“最危险优先”
3. 让行决策从单一连续约束升级为三段式规则机
4. 已承诺通过逻辑收紧，减少误放行
5. 诊断输出增强，便于真实运行时定位

纵向链路保持不变：

```text
IDM acceleration
  -> RearEndGuardRegulator
  -> IntersectionConflictRegulator
  -> final acceleration
```

---

## 4. 规则改造

### 4.1 候选车筛选更早

当前交叉口冲突问题的核心不是“根本没检测到车”，而是“检测和让行都偏晚”。因此候选筛选改为更早进入评估：

- 放宽 `candidate_radius`
- 适当拉长预测时域 `horizon`
- 只要背景车未来轨迹会进入 ego 前方交叉冲突区，就进入候选，而不是等它已经接近 ego

这一步不引入地图 block 级别识别，仍然使用现有上帝视角背景车与短时轨迹预测。

### 4.2 冲突排序更稳

当前排序偏向“最早发生冲突”，容易选中“时间更早但危险度没那么高”的对象，导致真正危险对象被低估。

优化后每个候选车都计算：

- `min_distance`
- `first_conflict_time`
- `time_gap`

排序规则改为：

1. 先看是否进入硬危险区
2. 再按更小 `min_distance`
3. 若接近，再按更小 `time_gap`

目标是优先处理“更危险”的对象，而不是单纯处理“更早碰上”的对象。

### 4.3 三段式让行规则

将当前偏连续的约束输出改成更可解释的 3 段模式：

- `PASS`
  - 冲突窗口足够大
  - 保持上游纵向控制

- `YIELD`
  - 存在一般冲突风险
  - 直接给出明显减速，不再只是轻微压制 IDM

- `EMERGENCY_BRAKE`
  - `min_distance` 很小或 `time_gap` 极短
  - 强制给出更强制动上限

这个模式本质上仍然输出加速度上限，但规则语义更清晰，更适合真实运行时调试。

### 4.4 已承诺通过规则收紧

仍然保留“已进入路口中央时不要犹豫停车”的原则，但要比当前更保守：

- 只有 ego 已明显接近通过完成，才允许 hold course
- 一般的“接近冲突区但还没完全承诺通过”不再轻易放行

这样能减少当前真实运行中出现的交叉口误放行碰撞。

---

## 5. 诊断输出

在现有基础上增强 `action_info`：

- `intersection_conflict_active`
- `intersection_conflict_count`
- `intersection_conflict_acc`
- `intersection_conflict_mode`
- `intersection_min_distance`
- `intersection_time_gap`

这些字段只用于统计与调试，不改变控制接口。

---

## 6. 验收目标

真实 `meta_drive` 10-episode 的当前权威基线是：

- `collision_intersection = 5`
- `collision_lane_change = 1`
- `collision_rear_end = 0`
- `arrive_dest = 1`
- `out_of_road = 3`
- `lane_changes = 37`

本轮目标：

- `collision_intersection <= 2`
- `collision_rear_end = 0` 或至少不回升
- `arrive_dest >= 1`，最好提升
- `lane_changes >= 20`，避免被过度保守压成几乎不换道

---

## 7. 非目标

本轮不做：

- `out_of_road` 专项修复
- 换道 planner / tracker 重写
- 路权图或复杂地图 block 识别
- 主动抢行策略
- `run_expert.py` 专用控制补丁

这些问题留到后续轮次。

