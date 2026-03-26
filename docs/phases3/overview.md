# Phase 6A：闭环执行器性能优化

> **前置依赖**：Phase 5v2 全部完成
> **目标**：将闭环联合组评估从 ~54s/step 降至 ~3s/step（去掉多余 reset）→ ~0.5s/step（多进程并行）
> **根因**：`ClosedLoopExecutor._restore_state()` 每次调用 `env.reset()` 重建整个场景（~5.5s），8 个 joint group 导致 9 次 reset = ~50s

## 问题诊断（Profile 数据）

```
[profile][summary] step_mean=54.129s
  collect_mean=0.139s  (0.3%)
  joint_mean=53.656s   (99.1%)  ← 瓶颈
  update_mean=0.258s   (0.5%)
  env_mean=0.074s      (0.1%)
```

`joint_mean=53.656s` 中：
- 9 次 `_restore_state()` → 9 × `env.reset()` ≈ 49.5s（场景重建）
- 64 次 `env.step()` ≈ 3.2s（实际有效计算）
- **97% 时间浪费在无意义的场景重建上**

## 附加正确性问题

当前 `_restore_state()` 的 `env.reset()` 将背景交通流重置到场景起始位置，而 `set_state()` 只恢复编队车到路中段位置。这导致：
- 编队车在路中段行驶，背景车却在路口初始位置
- 碰撞/安全间距评估基于错误的交通流状态

## 子任务列表

| 编号 | 任务 | 改动文件 | 依赖 | 文档 |
|------|------|----------|------|------|
| 6A.1 | 扩展 get_state/set_state 覆盖背景交通流 | `envs/platoon_env.py` | 无 | [task1_traffic_state.md](task1_traffic_state.md) |
| 6A.2 | 去除 _restore_state 中的 reset + 正确性验证 | `train/closedloop_executor.py`, 测试 | 6A.1 | [task2_remove_reset.md](task2_remove_reset.md) |
| 6A.3 | 多进程并行闭环执行（可选） | `train/closedloop_executor.py`, `train/train_platoon_rl.py` | 6A.2 | [task3_parallel_exec.md](task3_parallel_exec.md) |

## 实施顺序

```
6A.1 扩展 state → 6A.2 去 reset + 验证 → 6A.3 并行（可选）
```

## 预期性能目标

| 阶段 | 预计耗时 | 加速比 |
|------|---------|--------|
| 现状 | ~54s/step | 1× |
| 6A.1 + 6A.2 完成后 | ~3s/step | 18× |
| 6A.3 完成后 | ~0.5s/step | 108× |

## 硬件环境

- CPU: 24 核
- RAM: 78GB（可用 ~40GB）
- GPU: RTX 4080 16GB
- MetaDrive env 实例内存占用: ~500MB-1GB
