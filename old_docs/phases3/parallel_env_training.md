# Phase 6B：多环境并行化训练（远期规划）

> **状态**：⏳ 未执行（待 Phase 6A 闭环执行器优化完成 + 单环境训练方案评估通过后启动）
> **前置条件**：Phase 6A 完成（闭环执行已优化），单 env 闭环训练可正常收敛
> **目标**：在不改变 MA-GRPO 算法逻辑的前提下，通过多进程并行化 CPU 密集的环境交互层，提升训练吞吐量
> **与 Phase 6A 的关系**：Phase 6A 解决闭环执行器的性能瓶颈（54s→3s→0.5s），本文档关注更大范围的多 env 并行训练架构改造（Phase B/C）

---

## 1. 当前瓶颈分析

单 env 串行训练流程（`train/ma_grpo_trainer.py` + `train/train_platoon_rl.py`）：

```
每个 training step:
  1. GPU: diffusion DDIM sample        — 已 batch，较快
  2. CPU: evaluate_trajectory_group()   — M anchors × N agents，串行，慢 ★
  3. CPU: ClosedLoopExecutor            — K joint groups，串行，慢 ★
  4. GPU: replay + ref-reg + update     — 已 batch，较快
```

- **GPU 侧**（扩散采样 + 梯度更新）已做 batch 化，不是瓶颈
- **CPU 侧** 环境评估占总时间 60%+，是主要瓶颈：
  - `collect_group_samples()`：对每个 agent 的 M 条候选轨迹逐一调用 `evaluate_trajectory_group()`
  - `ClosedLoopExecutor.execute_joint_groups()`：对 K 个联合组逐一执行闭环模拟

## 2. MetaDrive 原生并行架构参考

MetaDrive 示例（`metadrive/examples/train_generalization_experiment.py`）使用 RLlib：

```python
tune.run("PPO", config={
    "env": MetaDriveEnv,
    "num_workers": 5,
    "train_batch_size": 20000,
    "rollout_fragment_length": 200,
})
```

RLlib 在每个 worker 进程中创建独立 env 实例，各 worker 并行 rollout，汇聚后统一更新。

## 3. 不直接套用 RLlib 的原因

| 差异点 | RLlib PPO | MA-GRPO |
|--------|-----------|---------|
| 动作空间 | 单步 action | 8 步轨迹（diffusion sample） |
| 采样方式 | `env.step()` 逐步 | diffusion DDIM 一次性生成 M 条候选 |
| 奖励计算 | env 返回 scalar | surrogate reward + team reward |
| advantage | GAE | intra-anchor 归一化 + 分层 (local+team) |
| 梯度更新 | PPO clip | GRPO replay + ref-reg loss |

直接用 RLlib 需重写 Policy 类、Trainer 类、自定义 rollout 逻辑，工作量大且丧失当前代码可读性。

**核心思路**：保留自研 MA-GRPO 训练逻辑不动，仅在 CPU 密集的环境交互层引入多进程并行。

---

## 4. 三层渐进式并行化方案

### Phase A：多进程轨迹评估（改动最小，收益最大）

**目标**：将 `evaluate_trajectory_group()` 的串行调用改为 `multiprocessing.Pool` 并行。

**改动文件**：`train/ma_grpo_trainer.py`

**改动范围**：仅 `collect_group_samples()` 方法

**设计**：

```python
# ma_grpo_trainer.py
from multiprocessing import Pool

def _eval_worker(args):
    """在子进程中评估单条轨迹组（每个 worker 独立创建 env）"""
    env_cfg, agent_id, trajectories, reward_config = args
    env = PlatoonEnv(env_cfg)
    result = env.evaluate_trajectory_group(agent_id, trajectories)
    env.close()
    return agent_id, result

class MultiAgentGRPOTrainer:
    def collect_group_samples_parallel(self, group_size, obs, num_workers=4):
        # 1. GPU batch sample（不变）
        all_trajectories = self._sample_trajectories(obs, group_size)

        # 2. 构建评估任务列表
        eval_tasks = []
        for agent_id in self.agent_ids:
            for anchor_idx in range(group_size):
                eval_tasks.append((
                    self.env_config, agent_id,
                    all_trajectories[agent_id][anchor_idx],
                    self.config["reward_config"]
                ))

        # 3. 并行评估
        with Pool(num_workers) as pool:
            results = pool.map(_eval_worker, eval_tasks)

        # 4. 重组结果（不变）
        return self._organize_rollouts(results)
```

**预估加速**：N agents × M anchors = 3×4 = 12 次评估 → 4 workers → ~3× 加速

**注意事项**：
- `evaluate_trajectory_group()` 是无状态的 surrogate 评估，无 env 状态依赖，天然适合多进程
- 每个 worker 独立创建 env 实例，无需共享内存
- GPU tensor 需在主进程 `.cpu().numpy()` 后再传入 worker

### Phase B：多进程闭环执行

**目标**：将 `ClosedLoopExecutor.execute_joint_groups()` 的串行循环改为多进程。

**改动文件**：`train/closedloop_executor.py`、`train/train_platoon_rl.py`

**设计**：

```python
# closedloop_executor.py
class ParallelClosedLoopExecutor:
    def __init__(self, env_config, reward_config, num_workers=4):
        self.env_config = env_config
        self.reward_config = reward_config
        self.num_workers = num_workers

    def execute_joint_groups_parallel(self, joint_groups, base_state):
        """每个 joint group 在独立 env 实例中并行执行"""
        tasks = [
            (self.env_config, base_state, group, self.reward_config)
            for group in joint_groups
        ]
        with Pool(self.num_workers) as pool:
            results = pool.map(_execute_single_group, tasks)
        return results

def _execute_single_group(args):
    env_config, base_state, joint_actions, reward_config = args
    env = PlatoonEnv(env_config)
    env.set_state(base_state)
    # 逐步执行 joint trajectory，收集 step_infos
    step_infos = []
    for t in range(trajectory_length):
        actions_t = {aid: joint_actions[aid][t] for aid in joint_actions}
        obs, reward, done, info = env.step(actions_t)
        step_infos.append(info)
    env.close()
    return {"step_infos": step_infos, ...}
```

**预估加速**：K 个 joint groups（默认 8）→ 4 workers → ~2× 加速

**注意事项**：
- env state 需要可序列化（`pickle`），当前 `get_state()` 返回的数据结构需确认可 pickle
- 每个 worker 需要能独立 `set_state()` + 执行 + 关闭，无资源泄漏

### Phase C：多 env 交错训练（可选）

**目标**：多个 env 实例交替提供 obs，增大每个 training step 的有效 batch。

**改动文件**：`train/train_platoon_rl.py`、`train/ma_grpo_trainer.py`

**设计**：

```python
# train_platoon_rl.py
class ParallelTrainingLoop:
    def __init__(self, num_envs=4, env_config=None):
        self.envs = [PlatoonEnv(env_config) for _ in range(num_envs)]

    def train(self, trainer, steps):
        obs_list = [env.reset() for env in self.envs]
        for step in range(steps):
            # 从多个 env 收集 rollout
            all_rollouts = []
            for env, obs in zip(self.envs, obs_list):
                trainer.env = env
                rollouts = trainer.collect_group_samples(obs=obs)
                all_rollouts.append(rollouts)

            # 合并 rollout 做一次大 batch 更新
            merged = merge_rollouts(all_rollouts)
            trainer.update(merged)

            # 各 env 独立推进
            for i, env in enumerate(self.envs):
                obs_list[i] = env.step(best_action[i])
```

**预估加速**：等效 batch 扩大 num_envs 倍，梯度估计更稳定

**注意事项**：
- 改动较大，需修改训练主循环和 trainer 的 update 逻辑
- 多 env 的 rollout 合并需要对齐 agent_id 和 anchor 维度
- 可与 Phase A/B 的多进程评估叠加

---

## 5. 实施路线图

| 阶段 | 内容 | 改动文件 | 改动量 | 加速预估 | 风险 |
|------|------|---------|--------|---------|------|
| **Phase A** | 多进程轨迹评估 | `ma_grpo_trainer.py` | ~100 行 | 2-3× | 低 |
| **Phase B** | 多进程闭环执行 | `closedloop_executor.py`, `train_platoon_rl.py` | ~80 行 | 1.5-2× | 中 |
| **Phase C** | 多 env 交错训练 | `train_platoon_rl.py`, `ma_grpo_trainer.py` | ~200 行 | 2× | 高 |

**推荐路径**：Phase A → 验证加速效果 → Phase B → 验证 → 评估是否需要 Phase C

Phase A 改动局部、风险低、收益最大（评估占总时间 60%+），应优先实施。

## 6. 验收标准

### Phase A 验收

- [ ] `collect_group_samples()` 支持 `num_workers` 参数
- [ ] `num_workers=1` 时行为与原串行版本一致（结果可复现）
- [ ] `num_workers=4` 时端到端训练 5 步无报错
- [ ] 单步耗时对比：并行版 < 串行版 × 0.5（至少 2× 加速）
- [ ] 现有测试全部通过（`pytest tests/acceptance/test_phase5v2_*.py`）

### Phase B 验收

- [ ] `ClosedLoopExecutor` 支持并行执行模式
- [ ] env state 可正确序列化/反序列化
- [ ] 联合组并行执行结果与串行一致
- [ ] 叠加 Phase A 后端到端训练无报错

### Phase C 验收

- [ ] 多 env 训练循环可正常运行
- [ ] rollout 合并后梯度更新无数值异常
- [ ] 训练收敛曲线与单 env 可比

## 7. 配置参数

建议在 `configs/train/platoon_grpo_v2.yaml` 中新增：

```yaml
# 并行化配置
parallel:
  eval_workers: 4        # Phase A: 轨迹评估 worker 数
  closedloop_workers: 4  # Phase B: 闭环执行 worker 数
  num_envs: 1            # Phase C: 并行 env 数（1 = 不启用）
```
