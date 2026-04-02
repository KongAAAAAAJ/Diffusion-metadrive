# Phase 6：实验评估与消融

> **前置依赖**：Phase 5（训练好的编队 checkpoint）
> **后续 Phase**：Phase 7（论文材料需要评估结果）

## 目标
形成论文主结果与消融结果。

---

## 全局约定（本 Phase 所需）

```bash
cd /home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive
export PYTHONPATH="$PWD:$PYTHONPATH"
```

### 关键 import
```python
from envs.platoon_env import PlatoonEnv
from models.platoon.platoon_diffusion_planner import PlatoonDiffusionPlanner
from evaluation.platoon_metrics import PlatoonMetrics
from scenarios.hazard_scenarios import get_hazard_scenario_configs
```

### 基线方法（需实现或集成）
| 基线 | 实现路径 |
|------|---------|
| 从零训练多车 RL (MAPPO) | 复用 MultiAgentMetaDrive + EPyMARL/MARLlib |
| 分层编队（PPO高层 + IDM低层）| PPO 集中式决策 + IDMPolicy |
| 无单车预训练 | 随机初始化 PlatoonDiffusionPlanner + GRPO |
| 无关系建模 | PlatoonDiffusionPlanner 但 relation_encoder 输出置零 |

### 评价指标
**通用**：success_rate, collision_rate, drivable_area_violation, avg_speed, comfort(jerk/yaw_rate), TTC
**编队**：formation_error, recovery_time, min_inter_vehicle_gap, 编队通过率
**泛化**：未见道路/工况性能, 不同 traffic density 鲁棒性
**工程**：单步推理时延, 显存占用

### 消融清单（来自 AGENTS.md 第十节）
1. 去掉单车预训练
2. 去掉编队关系建模
3. 去掉图像模态
4. 去掉 LiDAR 模态
5. 去掉编队相对关系
6. 去掉危险场景 curriculum
7. 仅 IL vs IL+RL
8. 不同编队规模
9. 不同 diffusion step / 候选轨迹数
10. 不同奖励权重组合

---

## 子任务与验收指标

### ☐ 6.1 创建主表实验脚本
- [ ] 待完成
- **交付物**：`scripts/run_main_benchmarks.py`、`configs/eval/main.yaml`、`tests/acceptance/test_phase6_task1.py`
- **验收指标**（6 项）：
  1. 接受 `--config` 和 `--output` 参数
  2. 覆盖 ≥ 3 种方法（ours、from-scratch-rl、hierarchical）
  3. 覆盖 ≥ 3 种场景（normal、hazard、mixed）
  4. 每组合 ≥ 3 seed
  5. CSV/JSON 含：方法名、场景名、seed、success_rate、collision_rate、formation_error、recovery_time、min_gap、推理时延
  6. 行数 = 方法数 × 场景数 × seed 数
- **验收命令**：
  ```bash
  python scripts/run_main_benchmarks.py --config configs/eval/main.yaml --output outputs/main_results/
  pytest tests/acceptance/test_phase6_task1.py -v
  ```

### ☐ 6.2 创建消融实验脚本
- [ ] 待完成
- **交付物**：`scripts/run_ablations.py`、`configs/eval/ablations.yaml`、`tests/acceptance/test_phase6_task2.py`
- **验收指标**（4 项）：
  1. 覆盖 ≥ 5 项消融
  2. 每项 ≥ 3 seed
  3. 输出格式与 6.1 一致
  4. 自动计算与完整模型的性能差
- **验收命令**：`pytest tests/acceptance/test_phase6_task2.py -v`

### ☐ 6.3 创建结果汇总工具
- [ ] 待完成
- **交付物**：`tools/aggregate_results.py`、`tests/acceptance/test_phase6_task3.py`
- **验收指标**（5 项）：
  1. 读取 6.1 和 6.2 输出
  2. 生成 LaTeX 主表（mean ± std）
  3. 生成消融表
  4. 自动标注最优值（bold）
  5. 输出到 `paper/tables/`
- **验收命令**：`pytest tests/acceptance/test_phase6_task3.py -v`

### ☐ 6.4 失败案例分析
- [ ] 待完成
- **交付物**：`tools/failure_analysis.py`、可视化输出
- **验收指标**（3 项）：
  1. 自动筛选 collision/failure episode
  2. 每种失败类型 ≥ 3 可视化 case
  3. 输出分类统计（碰撞/超时/出道路/队形崩溃各占比）

---

## 跨 Phase 数据流

```
Phase 5 checkpoints/platoon_rl/ ──→ 加载模型权重进行评估
Phase 1 PlatoonEnv + hazard_scenarios ──→ 评估环境
Phase 6 outputs/ ──→ Phase 7 论文表格和图表
```
