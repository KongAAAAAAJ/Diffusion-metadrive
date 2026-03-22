# Phase 3：单车扩散规划模型训练

> **前置依赖**：Phase 0、Phase 2（需要数据集）
> **后续 Phase**：Phase 4（需要 `checkpoints/single_vehicle/best.ckpt`）
> **可并行**：Phase 1

## 目标
形成可闭环运行的单车基础规划器，作为编队阶段模型初始化权重。

---

## 全局约定（本 Phase 所需）

```bash
cd /home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive
export PYTHONPATH="$PWD:$PYTHONPATH"
export DATA_DIR="${DATA_DIR:-/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets}"
```

### 关键 import
```python
from metadrive.policy.diffusion_policy.transfuser_model_v2 import V2TransfuserModel
from metadrive.policy.diffusion_policy.transfuser_config import TransfuserConfig
from metadrive.policy.diffusion_policy.transfuser_features import (
    stitch_three_cameras,   # → [3, 256, 1024]
    lidar_to_histogram,     # → [1, 256, 256]
    build_status_feature,   # → [8]
)
```

### 模型架构概要
- 主干：`TransfuserBackbone`（ResNet34 + Transformer 多尺度 BEV 融合）
- 扩散轨迹头：`ConditionalUnet1D`
- 调度器：`DDIMScheduler`
- 输入：camera `[3,256,1024]` + lidar `[1,256,256]` + status `[8]`
- 输出：trajectory `[ego_fut_mode, 8, 3]`（x, y, heading × 8 步）

### 硬件约束
- RTX 4080 16GB
- 单车训练 batch=16 ≈ 8GB

### 默认路径
| 用途 | 路径 |
|------|------|
| 数据集 | `$DATA_DIR` |
| 输出 checkpoint | `checkpoints/single_vehicle/best.ckpt` |
| plan anchor | `metadrive/exp_dataset/metadrive_anchors.npy` |

---

## 子任务与验收指标

### ☐ 3.1 轨迹归一化配置化
- [ ] 待完成
- **交付物**：修改 `transfuser_config.py` + `transfuser_model_v2.py`、`tests/acceptance/test_phase3_task1.py`
- **背景**：当前归一化常量硬编码（x: [-1.2, 55.7], y: [-20, 26], heading: [-2, 1.9]），需提取到 config。
- **验收指标**（5 项）：
  1. `TransfuserConfig` 新增 6 属性：`traj_norm_x_min/max, traj_norm_y_min/max, traj_norm_heading_min/max`
  2. 默认值与原硬编码一致
  3. `transfuser_model_v2.py` 中 grep 不到 `55.7` 或 `-1.2` 字面量
  4. 修改 config 归一化参数后模型输出轨迹范围变化
  5. 现有测试 `pytest metadrive/tests/test_policy/ -v` 全通过
- **验收命令**：
  ```bash
  pytest tests/acceptance/test_phase3_task1.py -v
  pytest metadrive/tests/test_policy/ -v --timeout=120
  ```

### ☐ 3.2 训练可运行性验证
- [ ] 待完成
- **交付物**：训练日志、`tests/acceptance/test_phase3_task2.py`
- **验收指标**（5 项）：
  1. 训练 ≥ 10 step 不崩溃
  2. 前 10 step loss 为有限值
  3. 10 step 后 loss < 初始 loss
  4. 显存 < 15GB
  5. checkpoint 可保存和加载
- **验收命令**：
  ```bash
  python -m metadrive.policy.diffusion_policy.train_transfuser \
      --model-size small --dataset-root $DATA_DIR \
      --plan-anchor-path metadrive/exp_dataset/metadrive_anchors.npy \
      --max-steps 10 2>&1 | tee logs/phase3_task2.log
  pytest tests/acceptance/test_phase3_task2.py -v
  ```

### ☐ 3.3 闭环测试与 checkpoint 验收
- [ ] 待完成
- **交付物**：`checkpoints/single_vehicle/best.ckpt`、日志、`tests/acceptance/test_phase3_task3.py`
- **验收指标**（5 项）：
  1. `checkpoints/single_vehicle/best.ckpt` 存在，> 1MB
  2. 闭环 10 episodes：`success_rate > 0.5`
  3. 闭环 10 episodes：`collision_rate < 0.3`
  4. 单步推理 < 30ms（RTX 4080）
  5. checkpoint 含 model state_dict + config，可被 `V2TransfuserModel` 加载
- **验收命令**：
  ```bash
  python -m metadrive.policy.diffusion_policy.test_transfuser_policy \
      --checkpoint checkpoints/single_vehicle/best.ckpt --episodes 10 --render 0
  pytest tests/acceptance/test_phase3_task3.py -v
  ```

---

## 跨 Phase 数据流

```
Phase 2 $DATA_DIR ──→ train_transfuser.py DataLoader（直接消费，字段一致）

Phase 3 checkpoints/single_vehicle/best.ckpt ──→ Phase 4 migrate_single_to_platoon()
  checkpoint 内容：model state_dict（含 _status_encoding.weight [tf_d_model, 8]）+ TransfuserConfig
```
