# Phase 6 Hotfix: 修正 iterations 计算 + planner CUDA 推理

## 背景

run_4 训练暴露两个问题：
1. `--steps 20000` 意图跑 20k env_steps，实际跑了 40,500 步但只 10 iterations——iterations 计算公式低估了每轮实际采样量
2. 每 iteration 采样耗时 ~1300s（占 99.5%），瓶颈是 4 个 worker 在 CPU 上跑 diffusion planner 推理

---

## Task 1: 修正 `estimate_env_steps_per_iteration` 与 `resolve_max_iterations`

### 问题根因

`train/train_selector.py` 中 `estimate_env_steps_per_iteration()` 的公式是：

```python
rollout_fragment_length * num_rollout_workers * num_envs_per_worker
# = 500 * 4 * 1 = 2000
```

但 RLlib PPO 每 iteration 实际采集的 env_steps 由 `train_batch_size` 决定（当 `count_steps_by="env_steps"` 时），不是 fragment×workers。RLlib 会连续收集 fragments 直到总量 ≥ `train_batch_size`。实际每 iteration 采了 ~4050 env_steps ≈ `train_batch_size=4000`。

所以 `resolve_max_iterations(20000 / 2000) = 10`，但实际 10 iterations 产生了 40,500 步——是目标的 2 倍。

### 修改文件

**`train/train_selector.py`**

#### 修改 1: `estimate_env_steps_per_iteration()`

将公式改为取 `train_batch_size` 和 `fragment_length * workers * envs_per_worker` 二者中的较大值：

```python
def estimate_env_steps_per_iteration(cfg: dict[str, Any]) -> int:
    rollout_fragment_length = max(1, int(cfg.get("rollout_fragment_length", 200)))
    num_rollout_workers = max(1, int(cfg.get("num_rollout_workers", 2)))
    num_envs_per_worker = max(1, int(cfg.get("num_envs_per_worker", 1)))
    fragment_total = rollout_fragment_length * num_rollout_workers * num_envs_per_worker
    train_batch_size = int(cfg.get("train_batch_size", 4000))
    return max(fragment_total, train_batch_size)
```

逻辑：RLlib PPO 每 iteration 至少采集 `train_batch_size` 个 env_steps（`count_steps_by=env_steps` 时）。当 fragment_total 已 ≥ train_batch_size 时以 fragment_total 为准；否则以 train_batch_size 为准。

#### 修改 2: 同步更新 `resolve_max_iterations()` 的 docstring/注释

在函数体上方加一行注释说明计算逻辑，使意图明确：

```python
def resolve_max_iterations(cfg: dict[str, Any]) -> int:
    """Return max training iterations, derived from total_env_steps / steps_per_iter."""
    total_env_steps = cfg.get("total_env_steps")
    if total_env_steps is None:
        return max(1, int(cfg.get("max_iterations", 500)))
    return max(1, math.ceil(int(total_env_steps) / estimate_env_steps_per_iteration(cfg)))
```

（函数体逻辑不变，只确保 `estimate_env_steps_per_iteration` 修复后计算正确。）

### 验收标准

1. **公式验证**：给定 `train_batch_size=4000, rollout_fragment_length=500, num_rollout_workers=4, num_envs_per_worker=1`：
   - `estimate_env_steps_per_iteration()` 返回 `4000`（不再是 2000）
   - `resolve_max_iterations(total_env_steps=20000)` 返回 `5`（不再是 10）

2. **smoke config 验证**：给定 `platoon_mappo_smoke.yaml` 中 `train_batch_size=64, rollout_fragment_length=16, num_rollout_workers=1, num_envs_per_worker=1`：
   - `estimate_env_steps_per_iteration()` 返回 `64`（fragment_total=16 < train_batch_size=64）

3. **fragment 主导场景**：给定 `train_batch_size=1000, rollout_fragment_length=500, num_rollout_workers=4`：
   - `estimate_env_steps_per_iteration()` 返回 `2000`（fragment_total=2000 > train_batch_size=1000）

4. **单元测试**：在 `tests/acceptance/test_phase6_iterations_fix.py` 中新增测试：
   ```python
   def test_estimate_env_steps_uses_train_batch_size():
       cfg = {"train_batch_size": 4000, "rollout_fragment_length": 500,
              "num_rollout_workers": 4, "num_envs_per_worker": 1}
       assert estimate_env_steps_per_iteration(cfg) == 4000

   def test_estimate_env_steps_fragment_dominant():
       cfg = {"train_batch_size": 1000, "rollout_fragment_length": 500,
              "num_rollout_workers": 4, "num_envs_per_worker": 1}
       assert estimate_env_steps_per_iteration(cfg) == 2000

   def test_resolve_max_iterations_with_fixed_estimate():
       cfg = {"total_env_steps": 20000, "train_batch_size": 4000,
              "rollout_fragment_length": 500, "num_rollout_workers": 4,
              "num_envs_per_worker": 1}
       assert resolve_max_iterations(cfg) == 5
   ```

---

## Task 2: planner 推理从 CPU 切换到 CUDA

### 问题根因

当前 `planner_device: cpu`，每个 env.step() 中的 `forward_selector()` 调用需要 ~1.3s（ResNet backbone + 8×DDIM 去噪），导致每 iteration 的 sample 阶段耗时 ~1300s。GPU 推理预期提速 5-10 倍。

### 需要修改的文件（共 4 处）

#### 修改 1: `envs/selector_platoon_env.py` — `_to_torch_batch` 加 device 参数

当前 `_to_torch_batch()` 把 numpy 数组转为 torch tensor 但不指定 device，当 planner 在 CUDA 上时输入仍在 CPU，导致 `_build_model_inputs()` 中 `self.relation_encoder(relation_feature)` 报 device mismatch 错误。

修改模块级函数 `_to_torch_batch`，增加可选 `device` 参数：

```python
def _to_torch_batch(value: Any, device: Optional[torch.device] = None) -> Any:
    if torch is None:
        return value
    if isinstance(value, Mapping):
        return {key: _to_torch_batch(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_torch_batch(item, device) for item in value)
    if isinstance(value, np.ndarray):
        t = torch.as_tensor(value)
        return t.to(device) if device is not None else t
    if isinstance(value, (int, float, np.integer, np.floating, bool, np.bool_)):
        t = torch.as_tensor(value)
        return t.to(device) if device is not None else t
    return value
```

#### 修改 2: `envs/selector_platoon_env.py` — `__init__` 保存 planner device

在 `SelectorPlatoonEnv.__init__` 中，generator 创建完成后，探测并保存其 device：

```python
# 在 self.generator 赋值之后
self._planner_device = None
if torch is not None and hasattr(self.generator, "parameters"):
    try:
        self._planner_device = next(self.generator.parameters()).device
    except StopIteration:
        pass
```

#### 修改 3: `envs/selector_platoon_env.py` — `_generate_candidate_payload` 传 device

在调用 `_to_torch_batch` 时传入 device：

```python
# 在 _generate_candidate_payload 中 forward_selector 分支
batch = {agent_id: _to_torch_batch(dict(agent_obs), self._planner_device)}
```

#### 修改 4: 配置文件修改

**`configs/train/selector.yaml`**：
```yaml
planner_device: cuda    # 原为 cpu
num_gpus: 0.5           # driver 进程预留 0.5 GPU，剩余给 worker 共享
```

**`configs/train/platoon_mappo_smoke.yaml`**：
```yaml
planner_device: cuda    # 原为 cpu
num_gpus: 0             # smoke 测试保留 0 GPU 以兼容无 GPU 环境
```

注意：smoke yaml 中 `planner_device` 改为 `cuda`，但 smoke 测试需要在 CI/无 GPU 环境也能跑。因此需要额外处理——见修改 5。

#### 修改 5: `envs/selector_platoon_env.py` — `_build_default_candidate_generator` 中 CUDA fallback

当配置指定 `cuda` 但机器没有 GPU 时，自动回退到 CPU：

```python
def _build_default_candidate_generator(config: Mapping[str, Any]):
    ...
    planner_device = str(config.get("planner_device", "cpu"))
    # CUDA fallback: 如果请求 CUDA 但不可用，回退到 CPU
    if planner_device.startswith("cuda") and (torch is None or not torch.cuda.is_available()):
        planner_device = "cpu"
    ...
    planner = planner.to(torch.device(planner_device))
    return planner
```

### RLlib GPU 资源说明

`num_gpus: 0.5` 告诉 RLlib driver 进程预留 0.5 个 GPU。rollout workers 中的 frozen planner 通过 `torch.device("cuda")` 直接使用 GPU，不受 `num_gpus` 分配控制（因为 planner 在 env 中，不在 policy 中）。

单卡 RTX 4080 (16GB) 下：
- 4 个 worker 各加载一份 frozen planner（small model ~150MB），总 VRAM ~600MB
- 加上 driver 侧的 selector policy（很小），总 VRAM < 2GB
- 足够安全

如果 VRAM 不足，可以减少 worker 数，但 4 个 worker 共享小模型不会有问题。

### 验收标准

1. **CUDA 推理正确性**：
   - `planner_device: cuda` 配置下，`SelectorPlatoonEnv` 能成功完成 `reset()` + `step()` 无 device mismatch 错误
   - planner 参数确实在 CUDA 上：`next(env.generator.parameters()).device.type == "cuda"`
   - 输出轨迹 shape 不变：`trajectory_candidates.shape == (K, 8, 3)`

2. **CPU fallback 正确性**：
   - `planner_device: cuda` + 无 GPU 环境下，planner 自动 fallback 到 CPU，不报错

3. **推理速度提升**：
   - 单次 `forward_selector()` 调用在 GPU 上耗时 < 200ms（CPU 上约 1300ms）
   - 可通过简单计时验证，不需要精确 benchmark

4. **smoke 测试通过**：
   - `pytest tests/acceptance/test_phase6_mappo_selector.py -v` 全部通过
   - `pytest tests/acceptance/test_selector_platoon_env.py -v` 全部通过

5. **单元测试**：在 `tests/acceptance/test_phase6_iterations_fix.py` 中增加：
   ```python
   def test_to_torch_batch_respects_device():
       """_to_torch_batch with device=cpu should produce CPU tensors."""
       import torch
       batch = {"x": np.zeros((2, 3))}
       result = _to_torch_batch(batch, device=torch.device("cpu"))
       assert result["x"].device.type == "cpu"
   ```

6. **配置验证**：
   - `selector.yaml` 中 `planner_device: cuda`
   - `platoon_mappo_smoke.yaml` 中 `planner_device: cuda`

---

## 文件修改清单

| 文件 | 修改内容 |
|------|----------|
| `train/train_selector.py` | `estimate_env_steps_per_iteration()` 公式修正 |
| `envs/selector_platoon_env.py` | `_to_torch_batch()` 加 device 参数；`__init__` 保存 planner device；`_generate_candidate_payload` 传 device；`_build_default_candidate_generator` 加 CUDA fallback |
| `configs/train/selector.yaml` | `planner_device: cuda`, `num_gpus: 0.5` |
| `configs/train/platoon_mappo_smoke.yaml` | `planner_device: cuda` |
| `tests/acceptance/test_phase6_iterations_fix.py` | 新建：iterations 计算 + device 传递的单元测试 |

## 不修改的文件

| 文件 | 原因 |
|------|------|
| `models/platoon/platoon_diffusion_planner.py` | `_forward_model` 已正确处理 CUDA device fork_rng，不需要改 |
| `train/train_selector.py` | 主训练入口，需同步引用新路径 |
| `train/selector_callbacks.py` | 不涉及 |

---

## 执行顺序

```
Task 1 (iterations fix)  →  Task 2 (CUDA planner)  →  运行测试验证
```

Task 1 和 Task 2 互相独立，可以并行修改，但建议先完成 Task 1 再跑完整训练验证。
