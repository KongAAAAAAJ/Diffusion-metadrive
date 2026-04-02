# 单车语义行为 Anchor 提取 —— 可执行实施计划

> **前置依赖**：Phase 6B.2（RL 训练稳定性修复）已合入
> **目标**：替换全局 K-Means anchor 为语义分桶 + 桶内聚类 + medoid 的三阶段方案
> **总 anchor 数**：≤ 10 个（每类 1～2 子类型）
> **路网说明**：PlatoonEnv 采用 `hybrid_map_sequence="SSXCOCSS"`，包含直道(S)、交叉口(X)、弯道(C)、环岛(O)

---

## 0. 数据字段审计与补采集

### 0.1 当前 shard 已保存的字段

每个 shard `.npz` 中每条样本已包含：

| 字段 | 形状 | 说明 |
|------|------|------|
| `trajectory` | (8, 3) | ego-local 未来轨迹 (x, y, heading)，float32 |
| `trajectory_raw` | (8, 3) | 修正前原始轨迹（如 save_raw_trajectory=True） |
| `trajectory_mode` | int8 | 已有6类: KEEP_LANE(0) FOLLOW(1) LC_LEFT(2) LC_RIGHT(3) OVERTAKE(4) MERGE(5) |
| `ego_pose_world` | (3,) | 世界坐标 [x, y, heading_theta] |
| `action` | (2,) | [steer, accel] |
| `ego_state` | 向量 | ego 状态 |
| `other_states` | 向量 | 周围车辆状态 |
| `agent_states` | (16, 5) | 附近车辆 bounding box [x, y, heading, length, width] |
| `agent_labels` | (16,) | 哪些 slot 有效 |
| `traffic_density` | scalar | 交通密度 |

### 0.2 `build_frame()` 中已计算但未保存的字段

以下字段在 `collect_expert.py` 的 `build_frame()` 函数中已经计算，但在 `build_episode_samples()` 构建最终样本时未写入 shard：

| 字段 | 类型 | 用途（7类标签判定） |
|------|------|---------------------|
| `ego_speed_km_h` | float32 | FOLLOW_BRAKE vs KEEP_LANE 区分 |
| `front_object_distance` | float32 | 前车约束检测 |
| `front_object_speed_km_h` | float32 | 前车速度 |
| `lane_index` | int16 | 当前车道编号 |
| `reference_lane_index` | int16 | 参考车道编号 |
| `reference_longitudinal` | float32 | 沿车道纵向坐标 |
| `reference_lateral` | float32 | 横向偏移（相对车道中心） |
| `lane_width` | float32 | 车道宽度（换道阈值） |
| `current_ref_lane_count` | int16 | 当前参考车道数 |
| `next_ref_lane_count` | int16 | 下一路段参考车道数（拓扑变化检测） |
| `reference_pose_world` | (3,) float32 | 参考车道世界坐标（TURN 检测的 heading 差） |

### 0.3 需要执行的修改

**文件**：`metadrive/exp_dataset/collect_expert.py`

**修改位置**：`build_episode_samples()` 函数中构建 samples dict 的部分（约 L500-528）

**修改内容**：在 `samples.append({...})` 的字典中追加以下字段：

```python
# --- 新增：语义标签判定所需字段 ---
"ego_speed_km_h": current_frame["ego_speed_km_h"],
"front_object_distance": current_frame["front_object_distance"],
"front_object_speed_km_h": current_frame["front_object_speed_km_h"],
"lane_index": current_frame["lane_index"],
"reference_lane_index": current_frame["reference_lane_index"],
"reference_longitudinal": current_frame["reference_longitudinal"],
"reference_lateral": current_frame["reference_lateral"],
"lane_width": current_frame["lane_width"],
"current_ref_lane_count": current_frame["current_ref_lane_count"],
"next_ref_lane_count": current_frame["next_ref_lane_count"],
"reference_pose_world": current_frame["reference_pose_world"],
```

**注意**：`build_frame()` 已经计算了这些字段并放入 frame dict，所以只需透传到 sample dict 即可。无需修改 `build_frame()` 本身。

修改后重新运行数据采集：
```bash
python -m metadrive.exp_dataset.collect_expert \
    --expert-type idm \
    --target-samples 50000
```

---

## 1. 目标与总体原则

### 1.1 目标

从单车 expert 数据集中提取一组可解释、可组合、可复用的轨迹 anchors，作为后续编队任务中单个 agent 的行为基元。

这些 anchors 不是直接描述编队行为，而是描述：
- 单车在不同语义行为下的典型轨迹原型
- 后续在编队控制/强化学习中，可由多个单车 anchor 组合成联合行为

### 1.2 核心原则

anchor 提取不再采用"对全量单车轨迹直接做全局 K-Means 聚类"的方式，而采用以下三阶段流程：

1. 先按单车语义行为分桶（规则判定，7 类）
2. 再在每个行为桶内按轨迹形态细分（K-Medoids 聚类，每类 1～2 个子类型）
3. 最后用真实 medoid 轨迹（而不是均值中心）作为 anchor

### 1.3 为什么这样做

- 全局几何聚类容易把不同驾驶意图混到一起
- 单车 anchor 最终要用于编队中的组合，必须保证语义清晰
- 强化学习更需要"稳定的行为基元"，而不是"几何均值轨迹"

---

## 2. Anchor 提取的两层结构

### 第一层：主语义行为类别（7 类）

| 编号 | 类别 | 含义 |
|------|------|------|
| 0 | KEEP_LANE | 正常巡航，无显著减速、无换道、无转向 |
| 1 | FOLLOW_BRAKE | 受前车约束的跟驰/减速 |
| 2 | YIELD | 因横向冲突（交叉口/merge/环岛）减速让行 |
| 3 | LANE_CHANGE_LEFT | 向左相邻车道换道（非路口左转） |
| 4 | LANE_CHANGE_RIGHT | 向右相邻车道换道 |
| 5 | TURN_LEFT | 通过交叉口/连接段完成左转拓扑切换 |
| 6 | TURN_RIGHT | 通过交叉口/连接段完成右转拓扑切换 |

### 第二层：桶内子类型（聚类产生，每类 1～2 个）

第二层不做手工定义，由桶内 K-Medoids 聚类自动产生。每类取 1～2 个 medoid。

**第一版 anchor 数量分配**（总计 9 个）：

| 类别 | 子类型数 | 预期子类型语义 |
|------|----------|----------------|
| KEEP_LANE | 2 | 匀速巡航 / 加速巡航 |
| FOLLOW_BRAKE | 2 | 轻度跟驰 / 强减速 |
| YIELD | 1 | 减速让行 |
| LANE_CHANGE_LEFT | 1 | 标准左换道 |
| LANE_CHANGE_RIGHT | 1 | 标准右换道 |
| TURN_LEFT | 1 | 左转弯 |
| TURN_RIGHT | 1 | 右转弯 |

> 若某类样本数 < 20，则该类只取 1 个 medoid。若某类样本数 = 0，则不产生该类 anchor，总数相应减少。

---

## 3. 已有 TrajectoryMode 与目标 7 类的关系

当前 `trajectory_correction.py` 中已有 6 类 `TrajectoryMode`：

| 已有类 | 值 | 目标映射 |
|--------|---|----------|
| KEEP_LANE | 0 | → KEEP_LANE |
| FOLLOW | 1 | → FOLLOW_BRAKE |
| LANE_CHANGE_LEFT | 2 | → LANE_CHANGE_LEFT |
| LANE_CHANGE_RIGHT | 3 | → LANE_CHANGE_RIGHT |
| OVERTAKE | 4 | → LANE_CHANGE_LEFT 或 LANE_CHANGE_RIGHT（根据横向位移方向） |
| MERGE | 5 | → YIELD（路段缩并场景） |

**新增需要的类**（当前判定逻辑不覆盖）：
- **YIELD**：需要检测交叉口/环岛区域内的减速让行（区别于 FOLLOW 的前车约束）
- **TURN_LEFT / TURN_RIGHT**：需要检测 heading 大幅变化 + 路网拓扑切换

---

## 4. 7 类语义标签的判定规则

### 4.1 判定优先级（从高到低）

1. TURN_LEFT / TURN_RIGHT
2. LANE_CHANGE_LEFT / LANE_CHANGE_RIGHT
3. YIELD
4. FOLLOW_BRAKE
5. KEEP_LANE

### 4.2 判定逻辑（使用已保存 + 补采集字段）

#### TURN_LEFT / TURN_RIGHT

**依据字段**：`trajectory`（heading 序列）、`reference_pose_world`、`ego_pose_world`、`next_ref_lane_count`

**判定条件**（满足全部）：
1. 轨迹末端 heading 变化 `|Δheading| > 0.5 rad`（≈ 28.6°）
2. `next_ref_lane_count` 与 `current_ref_lane_count` 不同（路段拓扑发生切换），或 `next_ref_lane_count == -1`（无下一段信息，通常表示交叉口内部）
3. 不满足换道判定（末端横向位移 < lane_width）

**方向**：`Δheading < -0.5` → TURN_LEFT，`Δheading > 0.5` → TURN_RIGHT

#### LANE_CHANGE_LEFT / LANE_CHANGE_RIGHT

**依据字段**：`lane_index`、`reference_lane_index`、`trajectory`（横向位移）、`lane_width`

**判定条件**（满足任一）：
- 条件 A：采集时已有 `trajectory_mode` ∈ {LC_LEFT(2), LC_RIGHT(3), OVERTAKE(4)} → 直接映射
- 条件 B：末端 `|reference_lateral|` > `lane_width × 0.55` 且 heading 变化 `|Δheading| ≤ 0.5 rad`

**方向**：OVERTAKE 根据末端横向位移符号决定 LEFT/RIGHT。

#### YIELD

**依据字段**：`trajectory_mode`、`next_ref_lane_count`、`current_ref_lane_count`、`front_object_distance`、`ego_speed_km_h`

**判定条件**（满足全部）：
1. 已有 `trajectory_mode == MERGE(5)`，**或**：
2. `next_ref_lane_count < current_ref_lane_count`（路段缩并），**或**：
3. 轨迹纵向位移很短（`trajectory[-1, 0] < 5.0m`）**且** `ego_speed_km_h < 10` **且** `front_object_distance` 为 -1（无前车）或 > 15m
4. 不满足 TURN / LC 判定

> 条件 3 的含义：低速、短位移、无前车约束 → 不是跟驰，更可能是让行或等待。

#### FOLLOW_BRAKE

**依据字段**：`trajectory_mode`、`front_object_distance`、`ego_speed_km_h`

**判定条件**（满足全部）：
1. 已有 `trajectory_mode == FOLLOW(1)`，**或**：
2. `front_object_distance > 0` 且 `front_object_distance < 15.0m` 且轨迹纵向位移 < 正常巡航的 50%
3. 不满足 TURN / LC / YIELD 判定

#### KEEP_LANE

以上均不满足 → 归为 KEEP_LANE。

---

## 5. 桶内聚类特征

完成第一层语义分桶后，在每个桶内做 K-Medoids 聚类。每类使用独立特征空间（不要求所有类共用同一特征）。

### 5.1 KEEP_LANE

**子类型数**：2

**聚类特征**：
- 终点纵向位移 `trajectory[-1, 0]`
- 平均横向偏移 `mean(trajectory[:, 1])`
- 最大横向偏移绝对值 `max(|trajectory[:, 1]|)`

### 5.2 FOLLOW_BRAKE

**子类型数**：2

**聚类特征**：
- 终点纵向位移 `trajectory[-1, 0]`
- 纵向位移梯度方差（反映减速强度）`var(diff(trajectory[:, 0]))`
- `front_object_distance`

### 5.3 YIELD

**子类型数**：1

**聚类特征**：（样本量可能不足，仅取 1 个 medoid）
- 终点纵向位移
- 最大 heading 变化

### 5.4 LANE_CHANGE_LEFT / RIGHT

**子类型数**：各 1

**聚类特征**：
- 末端横向位移 `trajectory[-1, 1]`
- 最大横向位移绝对值
- 终点纵向位移

### 5.5 TURN_LEFT / RIGHT

**子类型数**：各 1

**聚类特征**：
- heading 变化总量 `trajectory[-1, 2] - trajectory[0, 2]`
- 平均曲率（由相邻点 heading 差估计）
- 终点纵向位移

---

## 6. 聚类方式与 medoid 选择

### 6.1 聚类方式

- 规则分桶 + 桶内 K-Medoids（**不是 K-Means**）
- 不做全局聚类
- 桶内样本先做特征标准化（z-score），再跑 K-Medoids

### 6.2 为什么用 K-Medoids 而不是 K-Means

K-Means 的聚类中心是均值点，可能不对应任何真实轨迹。K-Medoids 的中心天然就是数据集中的真实样本，直接作为 anchor 即可，无需再"找最近样本"这一步。

### 6.3 空桶处理

若某类样本数 = 0：跳过，不产生该类 anchor。
若某类样本数 < 子类型数：子类型数退化为 min(样本数, 目标子类型数)。

---

## 7. Anchor 输出格式

### 7.1 兼容格式

为兼容现有 `abstract_anchors.py` 的输出格式，仍输出一个 numpy 数组：
- `anchors.npy`：shape `(N, 8, 2)` 或 `(N, 8, 3)`，N ≤ 10，与现有 anchor 形状一致

### 7.2 元信息文件

同时输出 `anchors_meta.json`：

```json
[
  {
    "anchor_id": 0,
    "behavior_mode": "KEEP_LANE",
    "subtype_id": 0,
    "source_sample_index": 12345,
    "source_shard": "shard_000003.npz",
    "cluster_size": 8234,
    "cluster_inertia": 0.23
  },
  ...
]
```

### 7.3 统计报告

输出 `anchors_stats.json`：

```json
{
  "total_samples": 50000,
  "label_distribution": {
    "KEEP_LANE": 28000,
    "FOLLOW_BRAKE": 12000,
    "YIELD": 1500,
    "LANE_CHANGE_LEFT": 800,
    "LANE_CHANGE_RIGHT": 600,
    "TURN_LEFT": 3500,
    "TURN_RIGHT": 3600
  },
  "anchor_count": 9,
  "anchors_per_class": {
    "KEEP_LANE": 2,
    "FOLLOW_BRAKE": 2,
    "YIELD": 1,
    "LANE_CHANGE_LEFT": 1,
    "LANE_CHANGE_RIGHT": 1,
    "TURN_LEFT": 1,
    "TURN_RIGHT": 1
  }
}
```

---

## 8. 可视化要求

### 8.1 Anchor 轨迹 BEV 图

类似现有 `metadrive_anchors.png`，但按语义类别着色，图例标注行为类型：

```
[KEEP_LANE-0] 蓝色    [KEEP_LANE-1] 浅蓝
[FOLLOW_BRAKE-0] 橙色  [FOLLOW_BRAKE-1] 红色
[YIELD-0] 紫色
[LC_LEFT-0] 绿色       [LC_RIGHT-0] 青色
[TURN_LEFT-0] 棕色     [TURN_RIGHT-0] 粉色
```

### 8.2 语义分布饼图

展示 7 类在数据集中的样本占比。

### 8.3 桶内散点图

对每个非空桶，绘制 2D 散点图（用前两个聚类特征作为 x/y 轴），标注 medoid 位置。

---

## 9. 质量检查要求

### 9.1 语义纯度检查

- 检查每个 anchor 所在簇的样本，原始 `trajectory_mode` 标签分布
- LANE_CHANGE_LEFT anchor 的簇内不应混入大量 TURN_LEFT
- FOLLOW_BRAKE anchor 的簇内不应混入大量 YIELD
- 纯度要求：主语义标签占比 ≥ 80%

### 9.2 几何可行性检查

- KEEP_LANE anchor：末端横向位移 < 2.0m
- LANE_CHANGE anchor：末端横向位移在 [lane_width × 0.4, lane_width × 1.5] 范围内
- TURN anchor：heading 变化 > 0.5 rad
- 所有 anchor 的轨迹点间距应单调递增（无倒退）

### 9.3 Medoid 真实性检查

每个 anchor 必须是数据集中真实存在的样本轨迹，不是插值或均值。通过 `source_sample_index` + `source_shard` 可回溯验证。

---

## 10. 代码修改清单

### Task 0：补采集字段（修改 collect_expert.py）

**文件**：`metadrive/exp_dataset/collect_expert.py`
**修改**：在 `build_episode_samples()` 的 `samples.append({...})` 中追加 11 个字段（见第 0.3 节）
**验收**：重新采集后，读取任一 shard，确认新字段存在且类型正确

### Task 1：新建语义标签判定模块

**新建文件**：`metadrive/exp_dataset/semantic_labeler.py`

功能：
- `class BehaviorMode(IntEnum)` 定义 7 类枚举
- `def label_sample(sample: dict) -> BehaviorMode` 按第 4 节规则判定
- `def label_dataset(shard_paths: List[Path]) -> np.ndarray` 批量打标签

### Task 2：替换 abstract_anchors.py 的聚类逻辑

**修改文件**：`metadrive/exp_dataset/abstract_anchors.py`

替换 `generate_plan_anchors()` 的流程为：
1. 加载全量轨迹 + 新字段
2. 调用 `label_dataset()` 获取每条样本的语义标签
3. 按语义标签分桶
4. 桶内提取特征 → K-Medoids 聚类 → 取 medoid
5. 汇总所有 medoid 为最终 anchors
6. 输出 `anchors.npy` + `anchors_meta.json` + `anchors_stats.json`

保留 `plot_plan_anchors()` 接口但更新着色逻辑。

### Task 3：可视化与统计

**修改文件**：`metadrive/exp_dataset/abstract_anchors.py`

- 语义着色的 BEV 轨迹图
- 分布饼图
- 桶内散点图 + medoid 标注

### Task 4：验收测试

**新建文件**：`tests/acceptance/test_semantic_anchors.py`

测试项：
1. `test_label_distribution`：统计 7 类分布，每类占比 > 0%（假设数据充分）或者至少 KEEP_LANE + FOLLOW_BRAKE 合计 > 50%
2. `test_anchor_count`：总 anchor 数 ≤ 10
3. `test_medoid_is_real_sample`：每个 anchor 的 `source_sample_index` 对应的真实轨迹与 anchor 轨迹完全一致
4. `test_anchor_shape_compatible`：输出 `anchors.npy` 的 shape 为 `(N, 8, 2)` 或 `(N, 8, 3)`，与模型输入兼容
5. `test_semantic_purity`：每个 anchor 簇的主标签占比 ≥ 80%
6. `test_geometric_sanity`：KEEP_LANE anchor 末端横向位移 < 2.0m

---

## 11. 实施顺序

```
Task 0: 补采集字段 → 重新跑数据采集
  ↓
Task 1: 新建 semantic_labeler.py → 跑一次统计分布
  ↓
Task 2: 替换 abstract_anchors.py 聚类逻辑
  ↓
Task 3: 可视化与统计
  ↓
Task 4: 验收测试
```

Task 0 需要重新采集数据（耗时较长），在等待期间可以先写 Task 1 的代码骨架。

---

## 总体要求

1. **先分桶，再聚类**，不允许直接对全量轨迹全局 K-Means
2. 主语义行为必须按 **7 类**定义
3. 每类使用**独立特征空间**，不要求所有类共用同一特征
4. 最终 anchor 必须是 **medoid 真实轨迹**，不是均值中心
5. 最终输出必须保留 anchor 的**元标签**
6. 提供**可视化与统计接口**，用于人工验收
7. 总 anchor 数 **≤ 10**，每类 1～2 个子类型
8. 输出格式须与现有 `anchors.npy` 的 shape 兼容，确保下游模型无需修改
