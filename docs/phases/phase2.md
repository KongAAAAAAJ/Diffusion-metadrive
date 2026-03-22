# Phase 2：单车专家数据采集

> **前置依赖**：Phase 0
> **后续 Phase**：Phase 3（训练需要数据集）、Phase 4（间接，通过 Phase 3 checkpoint）
> **可并行**：Phase 1、Phase 3（但 Phase 3 训练需要本 Phase 数据）

## 目标
得到覆盖关键驾驶技能的单车 IL 数据集。

---

## 全局约定（本 Phase 所需）

```bash
cd /home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive
export PYTHONPATH="$PWD:$PYTHONPATH"
export DATA_DIR="${DATA_DIR:-/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets}"
```

### 核心现有文件（不需重建）
- `metadrive/exp_dataset/collect_expert.py` — 主采集入口
- `metadrive/exp_dataset/trajectory_correction.py` — 轨迹修正
- `scripts/run_dataset_collect.sh` — 采集驱动脚本

### 数据字段规范
以 `docs/io_spec.md` 为准。采集数据中必须包含：`camera`（或 `rgb`）、`lidar`、`ego_state`、`trajectory`。

---

## 子任务与验收指标

### ☐ 2.1 确认现有采集脚本可运行
- [ ] 待完成
- **交付物**：`tests/acceptance/test_phase2_task1.py`、运行日志
- **验收指标**（4 项）：
  1. `bash scripts/run_dataset_collect.sh` 无 ImportError/ModuleNotFoundError
  2. 产生 ≥ 1 个 shard 文件（.npz 或 .pkl），大小 > 0
  3. 采集 50 样本耗时 < 10 分钟
  4. 数据含必要字段：`camera`/`rgb`、`lidar`、`ego_state`、`trajectory`
- **验收命令**：
  ```bash
  python -m metadrive.exp_dataset.collect_expert \
      --target-samples 50 --output-root /tmp/phase2_test --dataset-name test_run \
      --expert-type idm --trajectory-correction-enabled 0
  pytest tests/acceptance/test_phase2_task1.py -v
  ```

### ☐ 2.2 创建数据统计脚本
- [ ] 待完成
- **交付物**：`tools/check_dataset_stats.py`、`tests/acceptance/test_phase2_task2.py`
- **验收指标**（4 项）：
  1. 接受 `--dataset-root` 参数
  2. 输出含 `total_samples`(int)、`scenario_coverage`(dict)、`trajectory_stats`(dict, x/y/heading 的 min/max/mean/std)
  3. 空目录不抛异常，输出 `total_samples=0`
  4. 对有效数据集，`total_samples` 与实际样本数一致
- **验收命令**：
  ```bash
  python tools/check_dataset_stats.py --dataset-root /tmp/phase2_test/test_run
  pytest tests/acceptance/test_phase2_task2.py -v
  ```

### ☐ 2.3 扩充采集场景覆盖
- [ ] 待完成
- **交付物**：更新 `scripts/run_dataset_collect.sh`、`tests/acceptance/test_phase2_task3.py`
- **验收指标**（4 项）：
  1. 覆盖 ≥ 4 种道路结构（直道、弯道、交叉口、环岛中至少 4 种）
  2. ≥ 2 种 traffic density（如 0.02 和 0.08）
  3. 总 seed 数 ≥ 10
  4. 每种配置有注释说明道路结构
- **验收命令**：`pytest tests/acceptance/test_phase2_task3.py -v`

### ☐ 2.4 数据集完整性验收
- [ ] 待完成
- **交付物**：完整数据集 + 统计报告、`tests/acceptance/test_phase2_task4.py`
- **验收指标**（4 项）：
  1. 总样本 ≥ 500
  2. 技能覆盖：直行 ≥ 100、转向 ≥ 50、换道 ≥ 30、避障 ≥ 20
  3. 轨迹范围合理：x ∈ [-5, 100]、y ∈ [-30, 30]、heading ∈ [-π, π]
  4. 无 NaN/Inf
- **验收命令**：
  ```bash
  python tools/check_dataset_stats.py --dataset-root $DATA_DIR --check-integrity
  pytest tests/acceptance/test_phase2_task4.py -v
  ```

---

## 跨 Phase 数据流

```
Phase 2 数据集 ($DATA_DIR) ──→ Phase 3 train_transfuser.py --dataset-root $DATA_DIR
  数据由 collect_expert.py 产出，train_transfuser.py 的 DataLoader 直接消费，字段一致。
```
