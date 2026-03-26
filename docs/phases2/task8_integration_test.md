# 5v2.8 集成冒烟测试

> **依赖**：5v2.1-7 全部完成
> **改动文件**：`tests/acceptance/test_phase5v2_integration.py`（新建）

## 背景

验证 Phase 5v2 所有升级模块**端到端连通**，包括 ref-reg loss、intra-anchor advantage、闭环执行、联合组、分层 advantage。

## 测试设计

### 测试 1：ref-reg loss 端到端

```python
def test_ref_reg_replaces_il():
    """ref-reg loss 替代 IL loss，且反向传播正常。"""
    # setup: ToyPlanner + ToyEnv + MultiAgentGRPOTrainer
    # 1. collect_group_samples()
    # 2. update() 调用 compute_ref_reg_loss()（非 compute_il_loss）
    # 3. 验证 metrics 包含 ref_reg_loss，不包含 il_loss
    # 4. 验证 ref_reg_loss > 0 且 < 100
    # 5. 验证 beta_reg 在 [beta_reg_min, beta_reg_max] 范围内
```

### 测试 2：intra-anchor advantage 正确性

```python
def test_intra_anchor_advantage():
    """每个 anchor 独立归一化，不跨 anchor 比较。"""
    # 1. collect_group_samples() → reward_per_anchor [G, M]
    # 2. compute_advantages() → advantages [G, M, step_num]
    # 3. 对 anchor k1 和 k2，分别验证：
    #    advantages[:, k1, :] 的归一化独立于 advantages[:, k2, :]
    # 4. crash 轨迹的 advantage = -1.0
```

### 测试 3：env state round-trip（ToyEnv）

```python
def test_env_state_roundtrip_toy():
    """ToyEnv 的 get_state/set_state 不报错。"""
    # 1. env.reset()
    # 2. state = env.get_state()
    # 3. env.step(action)
    # 4. env.set_state(state)
    # 5. 不抛异常
```

### 测试 4：闭环执行器基本功能

```python
def test_closedloop_executor():
    """闭环执行器在 ToyEnv 上可运行。"""
    # 1. 创建 ClosedLoopExecutor(env, reward_config)
    # 2. joint_actions = {agent_id: random [8, 3]}
    # 3. result = executor.execute_joint_trajectory(joint_actions)
    # 4. 验证 result 包含 step_infos, crash_flags
    # 5. 验证 env 状态恢复
```

### 测试 5：联合组构建

```python
def test_joint_group_building():
    """联合组正确构建。"""
    # 1. per_agent_candidates = {agent_id: [(0,0), (1,1)]}
    # 2. groups = build_joint_groups(candidates, num_groups=4)
    # 3. 验证每个 group 包含所有 agent_id
    # 4. 验证 group 数量 ≤ 4
```

### 测试 6：team reward 计算

```python
def test_team_reward():
    """team reward 对碰撞场景返回低分。"""
    # 1. 构造 per_agent_step_infos（无碰撞）
    # 2. r_safe = compute_team_reward(infos, config)
    # 3. 构造 per_agent_step_infos（有碰撞）
    # 4. r_crash = compute_team_reward(infos, config)
    # 5. 验证 r_safe > r_crash
```

### 测试 7：分层 advantage 合并

```python
def test_layered_advantage():
    """分层 advantage = λ_local × local + λ_team × team。"""
    # 1. 构造 local_advantages [G, M, step_num]
    # 2. 构造 team_advantages [G, M, step_num]（大部分为 0）
    # 3. combined = compute_combined_advantages(local, team)
    # 4. 验证 combined 的非零位置 = λ_local * local + λ_team * team
    # 5. 验证零位置 = λ_local * local（因为 team=0）
```

### 测试 8：完整训练循环（toy 模式）

```python
def test_full_train_loop_toy():
    """5 步 toy 训练不报错，metrics 完整，且在 5 分钟内完成。"""
    # 1. build_runtime(mode="toy-single", steps=5)
    # 2. 运行训练，记录耗时
    # 3. 验证 metrics 包含 rl_loss, ref_reg_loss, beta_reg, mean_reward, kl
    # 4. 所有 metrics 为有限值
    # 5. 断言 elapsed < 300（5 分钟限制）
```

**【修复要求】为测试 8 添加计时断言**：当前实现无时间上限检查，超时不会主动 FAIL。需修改 `test_full_train_loop_toy` 如下：

```python
def test_full_train_loop_toy(tmp_path):
    import time
    t0 = time.time()
    summary = entry.build_runtime(
        mode="toy-single",
        config_path="configs/train/platoon_grpo_v2.yaml",
        steps=5,
        render=False,
        checkpoint_dir=str(tmp_path / "ckpt"),
        log_dir=str(tmp_path / "logs"),
        ckpt_path=entry.DEFAULT_SINGLE_CKPT,
        run_training=True,
    )
    elapsed = time.time() - t0
    _assert_numeric_metrics_are_finite(summary, ["loss", "rl_loss", "ref_reg_loss", "beta_reg", "mean_reward", "kl"])
    assert elapsed < 300, f"toy-single 5 steps took {elapsed:.1f}s, exceeds 5-minute limit"
```

### 测试 9：完整训练循环（platoon 模式，含联合组）

```python
def test_full_train_loop_platoon_closedloop():
    """3 步 platoon-closedloop 训练不报错。"""
    # 需要 MetaDrive 环境
    # 1. build_runtime(mode="platoon-closedloop", steps=3)
    # 2. 运行训练
    # 3. 验证 metrics 包含 team_reward_mean
    # 4. 所有 metrics 为有限值
    # 可标记为 @pytest.mark.skipif(not METADRIVE_AVAILABLE)
```

### 测试 10：配置文件完整性

```python
def test_config_v2_completeness():
    """platoon_grpo_v2.yaml 包含所有必需字段。"""
    # 加载 yaml
    # 验证包含：beta_reg_max, beta_reg_min, lambda_local, lambda_team,
    #           joint_top_k, num_joint_groups, use_closedloop,
    #           w_team_formation, w_team_safety, w_team_efficiency
```

## 验收标准

**交付物**：`tests/acceptance/test_phase5v2_integration.py`

**验收指标**：上述 10 个测试全部 PASSED

**验收命令**：
```bash
pytest tests/acceptance/test_phase5v2_integration.py -v
```

## 注意事项

1. 测试 9（platoon-closedloop）依赖 MetaDrive 环境，在无 MetaDrive 的机器上应 skip
2. 测试应使用 ToyEnv/ToyPlanner 尽可能覆盖，减少对真实环境的依赖
3. 所有测试应在 60 秒内完成
4. 测试不应修改任何文件或全局状态
