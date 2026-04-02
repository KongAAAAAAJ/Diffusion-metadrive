# 5v2.7 训练入口升级 + 配置

> **依赖**：5v2.1-6 全部完成
> **改动文件**：`train/train_platoon_rl.py`, `configs/train/platoon_grpo_v2.yaml`（新建）

## 背景

将所有升级集成到训练入口，支持渐进式训练配置。

## 新增训练模式

当前 `train_platoon_rl.py` 支持 `--mode toy-single` 和 `--mode platoon`。新增：

| 模式 | 说明 | 启用功能 |
|------|------|----------|
| `toy-single` | 单车 toy 验证 | ref-reg + intra-anchor |
| `platoon` | 3 车半闭环（现有） | ref-reg + intra-anchor |
| `platoon-closedloop` | 3 车闭环 + 联合组 | 全部升级 |

## 具体代码改动

### 改动 1：新增 `platoon_grpo_v2.yaml`

**文件**：`configs/train/platoon_grpo_v2.yaml`

```yaml
# === 基础训练参数 ===
group_size: 2
num_agents: 3
lr: 0.00005
kl_threshold: 5.0
max_grad_norm: 5.0
total_steps: 500
checkpoint_interval: 50

# === Diffusion 参数 ===
ddim_steps: 4
ddim_eta: 0.02
advantage_discount_gamma: 0.8

# === 参考策略正则（替代 IL loss） ===
beta_reg_max: 1.0
beta_reg_min: 0.1
beta_reg_warmup_frac: 0.3
beta_reg_decay_frac: 0.4

# === 分层 Advantage ===
lambda_local: 0.7
lambda_team: 0.3
joint_top_k: 2
num_joint_groups: 8
use_closedloop: true

# === 局部 Reward 配置 ===
reward_config:
  delta_s_max: 10.0
  d_norm: 10.0
  d_safe: 8.0
  w_progress: 1.0
  w_formation: 0.5
  w_safety: 0.3
  w_collision: 10.0
  w_road: 5.0
  w_comfort: 0.1

  # Team reward 权重
  w_team_formation: 0.5
  w_team_safety: 1.0
  w_team_efficiency: 0.3
  w_team_collision: 10.0

# === 环境配置 ===
traffic_density: 0.04
horizon: 100
max_env_steps_per_rollout: 20

# === 冻结策略 ===
freeze_backbone: true
freeze_tf_decoder: false
freeze_trajectory_head: false
```

### 改动 2：`build_runtime()` 支持新模式和配置

**文件**：`train/train_platoon_rl.py`
**位置**：`build_runtime()` 函数（line 233）

主要改动点：

```python
def build_runtime(mode, config_path, ...):
    # ... (加载配置同前)

    # 新增：根据 freeze 配置冻结参数
    if config.get("freeze_backbone", True):
        for name, param in model.named_parameters():
            if "_backbone" in name or "_bev_" in name or "_keyval_" in name:
                param.requires_grad_(False)

    # 新增：ToyEnv 也需要 get_state/set_state
    if mode.startswith("toy"):
        # ToyEnv 已有 stub 实现（5v2.3 中添加）
        pass

    # 新增：platoon-closedloop 模式
    if mode == "platoon-closedloop":
        config.setdefault("use_closedloop", True)

    # 优化器只包含可训练参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # ... 传给 trainer
```

### 改动 3：参数冻结控制

**文件**：`train/train_platoon_rl.py`

```python
def _apply_freeze_config(model, config: dict):
    """根据配置冻结指定模块的参数。"""
    freeze_backbone = config.get("freeze_backbone", True)
    freeze_tf_decoder = config.get("freeze_tf_decoder", False)
    freeze_trajectory_head = config.get("freeze_trajectory_head", False)

    for name, param in model.named_parameters():
        should_freeze = False
        if freeze_backbone and any(k in name for k in ["_backbone", "_bev_upscale", "_bev_downscale", "_keyval_embedding"]):
            should_freeze = True
        if freeze_tf_decoder and "_tf_decoder" in name:
            should_freeze = True
        if freeze_trajectory_head and "_trajectory_head" in name:
            should_freeze = True

        # 永远不冻结：relation_encoder, _status_encoding（编队新增部分）
        if "relation_encoder" in name or "_status_encoding" in name:
            should_freeze = False

        param.requires_grad_(not should_freeze)

    # 打印可训练参数统计
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[freeze] Total params: {total:,}, Trainable: {trainable:,} ({100*trainable/total:.1f}%)")
```

### 改动 4：TensorBoard 日志扩展

**文件**：`train/train_platoon_rl.py`
**位置**：`_log_metrics()` 函数（line 162）

新增日志字段：
```python
def _log_metrics(writer, metrics: dict, step: int):
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            writer.add_scalar(f"train/{key}", value, step)
    # 新增字段会自动被记录：
    # ref_reg_loss, beta_reg, team_reward_mean, lambda_local, lambda_team
```

### 改动 5：`parse_args()` 支持新模式

```python
def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="toy-single",
                        choices=["toy-single", "platoon", "platoon-closedloop"])
    # ... 其他参数同前
```

## 渐进式训练脚本

```bash
# 阶段 1：单车 toy 验证（ref-reg + intra-anchor）
python train/train_platoon_rl.py \
  --mode toy-single --steps 100 --render 0

# 阶段 2：3 车半闭环（ref-reg + intra-anchor，无联合组）
python train/train_platoon_rl.py \
  --mode platoon \
  --config configs/train/platoon_grpo_v2.yaml \
  --steps 500 --render 0

# 阶段 3：3 车闭环（全部升级）
python train/train_platoon_rl.py \
  --mode platoon-closedloop \
  --config configs/train/platoon_grpo_v2.yaml \
  --steps 500 --render 0
```

## 验收标准

**交付物**：修改后的 `train/train_platoon_rl.py`、`configs/train/platoon_grpo_v2.yaml`、`tests/acceptance/test_phase5v2_task7.py`

**验收指标**（8 项）：
1. `platoon_grpo_v2.yaml` 存在且包含所有新增字段
2. `--mode toy-single --steps 5` 可运行（使用 ref-reg 替代 IL loss）
3. **【修复要求】`--mode platoon --steps 5` 可运行**：当前 `test_phase5v2_task7.py` 缺少对 `platoon` 模式实际训练运行的测试（现有 `test_build_runtime_supports_closedloop_mode_with_monkeypatched_runtime` 只测试 `platoon-closedloop` 且 `run_training=False`）。需在 `test_phase5v2_task7.py` 中新增如下测试：
   ```python
   def test_build_runtime_runs_platoon_mode(monkeypatch, tmp_path):
       import torch
       monkeypatch.setattr(entry, 'PlatoonEnv', lambda cfg: entry.ToyEnv(num_agents=int(cfg.get('num_agents', 3)), mode='platoon'))
       monkeypatch.setattr(entry, 'PlatoonDiffusionPlanner', lambda *a, **kw: entry.ToyPlanner(num_agents=3))
       monkeypatch.setattr(entry, 'migrate_single_to_platoon', lambda ckpt, model: model)
       monkeypatch.setattr(entry, 'build_transfuser_config', lambda *a, **kw: {'dummy': True})

       summary = entry.build_runtime(
           mode='platoon',
           config_path='configs/train/platoon_grpo_v2.yaml',
           steps=2,
           render=False,
           checkpoint_dir=str(tmp_path / 'ckpt'),
           log_dir=str(tmp_path / 'logs'),
           ckpt_path='dummy.ckpt',
           run_training=True,
       )
       assert 'loss' in summary
       assert torch.isfinite(torch.tensor(float(summary['loss'][0])))
       assert 'ref_reg_loss' in summary
   ```
4. `--mode platoon-closedloop --steps 5` 可运行（需 MetaDrive 环境）
5. `_apply_freeze_config()` 正确冻结 backbone（frozen 参数 requires_grad=False）
6. relation_encoder 和 _status_encoding 始终可训练
7. TensorBoard 日志包含 ref_reg_loss 和 beta_reg 字段
8. **【修复要求】5 步 < 5 分钟（toy-single）**：当前 `test_full_train_loop_toy`（位于 `test_phase5v2_integration.py`）无显式计时断言，超时不会主动 FAIL。需在该测试内部包裹 `time.time()`，添加如下断言：
   ```python
   import time
   t0 = time.time()
   summary = entry.build_runtime(mode="toy-single", ..., steps=5, run_training=True)
   elapsed = time.time() - t0
   assert elapsed < 300, f"toy-single 5 steps took {elapsed:.1f}s, exceeds 5-minute limit"
   ```
   注意：`platoon-closedloop` 的 10 分钟限制因 MetaDrive 初始化耗时较长，暂不要求添加，避免 CI 不稳定。

**验收命令**：
```bash
python train/train_platoon_rl.py --mode toy-single --steps 5 --render 0
pytest tests/acceptance/test_phase5v2_task7.py -v
```
