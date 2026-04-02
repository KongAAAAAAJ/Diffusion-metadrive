# Phase 4：编队端到端合作扩散规划器

> **前置依赖**：Phase 1（编队环境，含 1.6/1.7）、Phase 3（单车 checkpoint）
> **后续 Phase**：Phase 5（RL 微调需要 planner + 迁移后的权重）

## 目标
将单车 planner 扩展为编队 planner（完全共享方案），加载单车预训练权重并验证前向传播。

---

## 全局约定（本 Phase 所需）

```bash
cd /home/kong/diffusion_codes/Diffusion-meta/Diffusion-metadrive
export PYTHONPATH="$PWD:$PYTHONPATH"
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
from models.platoon.relation_encoder import RelationEncoder
from models.platoon.platoon_diffusion_planner import PlatoonDiffusionPlanner
from models.platoon.weight_migration import migrate_single_to_platoon
from envs.platoon_env import PlatoonEnv
```

### 设计决策
- **不做编队 IL**：没有编队专家数据，直接从单车预训练权重初始化后进入 Phase 5 RL
- **参数共享**：3 车共用 1 个 `V2TransfuserModel` 实例，每车独立 forward
- **关系编码注入**：12 维 `formation_relation_state` → `RelationEncoder(MLP)` → 12 维 → cat with 8 维 `status` → 20 维 → `_status_encoding(20, tf_d_model)`
- **权重迁移**：`_status_encoding.weight` 前 8 列复用单车权重，后 12 列 Kaiming init

### formation_relation_state（冻结，同 Phase 1）
12 维 = 2 邻居 × 6（Δx, Δy, Δheading, Δv, Δx_target, Δy_target）。

### 硬件约束
- 3 车推理 ≈ 4.5GB，需 < 6GB
- 3 车 forward < 50ms（RTX 4080）

---

## 核心接口签名

```python
# models/platoon/relation_encoder.py
class RelationEncoder(nn.Module):
    """MLP: 12 → 64 (ReLU) → 12 (LayerNorm)
    输出 cat 到 status_feature [8]，共 [20] 维进入 _status_encoding"""
    def __init__(self, input_dim=12, hidden_dim=64, output_dim=12): ...
    def forward(self, x: Tensor) -> Tensor: ...  # [B, 12] → [B, 12]

# models/platoon/platoon_diffusion_planner.py
class PlatoonDiffusionPlanner(nn.Module):
    def __init__(self, config: TransfuserConfig, num_vehicles: int = 3):
        self.model = V2TransfuserModel(config)  # 单实例共享
        self.relation_encoder = RelationEncoder(12, 64, 12)

    def forward(self, batch: dict[str, dict]) -> dict[str, Tensor]:
        """batch[agent_id] = {
            "camera": Tensor [3, 256, 1024],
            "lidar": Tensor [1, 256, 256],
            "status": Tensor [8],
            "formation_relation_state": Tensor [12],
        }
        return {agent_id: trajectory [ego_fut_mode, 8, 3]}"""

# models/platoon/weight_migration.py
def migrate_single_to_platoon(ckpt_path: str, model: PlatoonDiffusionPlanner) -> PlatoonDiffusionPlanner:
    """加载单车权重，_status_encoding 前 8 维复用，后 12 维 Kaiming init"""
```

---

## 子任务与验收指标

### ☐ 4.0 升级 PlatoonEnv 观测为多模态格式（Phase 1→4 桥接）
- [ ] 待完成
- **背景**：Phase 1 使用 `LidarStateObservation`（flat ndarray），但 planner 需要分离的 camera/lidar/status/relation 张量。
- **交付物**：更新 `envs/platoon_env.py`、`tests/acceptance/test_phase4_task0.py`
- **设计**：新增 `observation_mode` config（`"lidar_state"` 默认 | `"multimodal"`），提供 `obs_to_tensor()` 工具函数。
- **验收指标**（7 项）：
  1. `PlatoonEnv({"observation_mode": "lidar_state"})` 行为不变（回归）
  2. `PlatoonEnv({"observation_mode": "multimodal"}).reset()` obs 含 camera/lidar/status/formation_relation_state
  3. `camera.shape == (3, 256, 1024)`
  4. `lidar.shape == (1, 256, 256)`
  5. `status.shape == (8,)`
  6. `formation_relation_state.shape == (12,)`
  7. `obs_to_tensor()` 输出 `torch.float32`，可在 GPU
- **验收命令**：`pytest tests/acceptance/test_phase4_task0.py -v`（7 PASSED）

### ☐ 4.1 创建 RelationEncoder
- [ ] 待完成
- **交付物**：`models/platoon/relation_encoder.py`、`models/platoon/__init__.py`、`models/__init__.py`、`tests/acceptance/test_phase4_task1.py`
- **验收指标**（7 项）：
  1. 继承 `nn.Module`
  2. 默认 `RelationEncoder(12, 64, 12)`
  3. `torch.randn(1,12)` → shape `(1,12)`
  4. `torch.randn(8,12)` → shape `(8,12)`
  5. 参数量 < 10000
  6. 含 LayerNorm
  7. 3 组随机输入无 NaN
- **验收命令**：`pytest tests/acceptance/test_phase4_task1.py -v`（7 PASSED）

### ☐ 4.2 创建 PlatoonDiffusionPlanner
- [ ] 待完成
- **交付物**：`models/platoon/platoon_diffusion_planner.py`、`tests/acceptance/test_phase4_task2.py`
- **验收指标**（7 项）：
  1. 继承 `nn.Module`
  2. 接受 `config: TransfuserConfig, num_vehicles: int`
  3. 含 `self.model`(V2TransfuserModel) 和 `self.relation_encoder`(RelationEncoder)
  4. `forward(batch)` 接受 dict[str, dict]，返回 dict[str, Tensor]
  5. 3 车 dummy forward 输出每车含 `[8, 3]`
  6. 只有 1 个 V2TransfuserModel 实例
  7. 3 车 forward < 100ms
- **验收命令**：`pytest tests/acceptance/test_phase4_task2.py -v`（7 PASSED）

### ☐ 4.3 创建 weight_migration
- [ ] 待完成
- **交付物**：`models/platoon/weight_migration.py`、`tests/acceptance/test_phase4_task3.py`
- **验收指标**（6 项）：
  1. `migrate_single_to_platoon()` 函数存在
  2. 非 `_status_encoding` 层权重与原 ckpt 一致（`torch.allclose`, atol=1e-6）
  3. `_status_encoding.weight` 前 8 列与单车一致
  4. 后 12 列非全零（Kaiming init）
  5. `relation_encoder` 已初始化（非全零）
  6. 迁移后可正常 forward
- **验收命令**：`pytest tests/acceptance/test_phase4_task3.py -v`（6 PASSED）

### ☐ 4.4 Phase 4 集成验收
- [ ] 待完成
- **交付物**：`tests/acceptance/test_phase4_task4.py`
- **验收指标**（5 项）：
  1. 完整流程：加载 ckpt → 迁移 → 3 车 forward → shape 正确
  2. 总时延 < 50ms（forward only）
  3. GPU 峰值 < 6GB
  4. 轨迹值合理（x ∈ [-10,100]、y ∈ [-30,30]，无 NaN/Inf）
  5. 相同输入重复 3 次输出一致
- **验收命令**：`pytest tests/acceptance/test_phase4_task4.py -v`（5 PASSED）

---

## 跨 Phase 数据流

```
Phase 1 PlatoonEnv(multimodal) obs ──→ Phase 4 PlatoonDiffusionPlanner.forward(batch)
  obs_to_tensor() 将 numpy obs 转为 GPU tensor

Phase 3 best.ckpt ──→ Phase 4 migrate_single_to_platoon()
  _status_encoding: 前8列复用，后12列 Kaiming init

Phase 4 PlatoonDiffusionPlanner (迁移后) ──→ Phase 5 MultiAgentGRPOTrainer(model=planner)
Phase 4 DiffusionRLScheduler ──→ Phase 5 sample_with_log_prob + KL 计算
```
