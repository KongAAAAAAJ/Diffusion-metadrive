# 5v2.5 联合组构建与 Team Reward

> **依赖**：5v2.4（闭环执行器）
> **改动文件**：`train/joint_group.py`（新建）, `evaluation/reward_terms.py`（扩展）

## 背景

单车 intra-anchor advantage 只解决"一辆车在一个意图下如何选最优轨迹"。编队合作还需要"多辆车如何选择意图组合"——即联合组（joint group）。

## 设计概要

```
输入：
  per_agent_rollouts = {
    agent_id: {
      trajectory_all: [G, M, 8, 3],
      reward_per_anchor: [G, M],
    }
  }

流程：
  1. 每辆车选 top-K 候选（按 local reward 排序）
  2. 从 N 辆车的 top-K 中随机组合 M_joint 个联合组
  3. 闭环执行每个联合组
  4. 计算 team reward

输出：
  joint_results: list of {
    members: {agent_id: (g_idx, k_idx)},   # 每辆车选了哪条轨迹
    team_reward: float,
    per_agent_info: {agent_id: step_infos},
  }
```

## 具体代码

### 新建 `train/joint_group.py`

```python
"""联合组构建：从多车候选轨迹中组合联合执行方案。"""

from __future__ import annotations

import itertools
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor


def select_top_k_candidates(
    reward_per_anchor: Tensor,
    crash_per_anchor: Tensor,
    top_k: int = 2,
) -> List[Tuple[int, int]]:
    """从 [G, M] reward 中选出 top-K 候选（g_idx, k_idx）。

    优先选非碰撞的，再按 reward 排序。

    Parameters
    ----------
    reward_per_anchor : Tensor [G, M]
    crash_per_anchor : Tensor [G, M] bool
    top_k : int

    Returns
    -------
    candidates : list of (g_idx, k_idx)，长度为 min(top_k, G*M)
    """
    G, M = reward_per_anchor.shape
    # 碰撞轨迹 reward 设为 -inf
    masked_reward = reward_per_anchor.clone()
    masked_reward[crash_per_anchor] = float('-inf')

    # 展平后取 top-K
    flat = masked_reward.view(-1)
    k = min(top_k, flat.numel())
    _, indices = torch.topk(flat, k)

    candidates = []
    for idx in indices:
        g = int(idx) // M
        k_anchor = int(idx) % M
        candidates.append((g, k_anchor))

    return candidates


def build_joint_groups(
    per_agent_candidates: Dict[str, List[Tuple[int, int]]],
    num_groups: int = 8,
    seed: int | None = None,
) -> List[Dict[str, Tuple[int, int]]]:
    """从每辆车的 top-K 候选中随机组合联合组。

    Parameters
    ----------
    per_agent_candidates : dict[agent_id -> list of (g_idx, k_idx)]
    num_groups : int
        生成的联合组数量。
    seed : int | None
        随机种子（可选）。

    Returns
    -------
    groups : list of dict[agent_id -> (g_idx, k_idx)]
        每个 dict 描述一个联合组中每辆车选了哪条轨迹。
    """
    agent_ids = sorted(per_agent_candidates.keys())
    candidate_lists = [per_agent_candidates[aid] for aid in agent_ids]

    # 笛卡尔积的总数
    total_combinations = 1
    for cl in candidate_lists:
        total_combinations *= len(cl)

    if total_combinations <= num_groups:
        # 组合数少于要求，穷举
        combos = list(itertools.product(*candidate_lists))
    else:
        # 随机采样
        rng = random.Random(seed)
        combos = set()
        max_attempts = num_groups * 10
        for _ in range(max_attempts):
            combo = tuple(rng.choice(cl) for cl in candidate_lists)
            combos.add(combo)
            if len(combos) >= num_groups:
                break
        combos = list(combos)

    groups = []
    for combo in combos[:num_groups]:
        group = {agent_ids[i]: combo[i] for i in range(len(agent_ids))}
        groups.append(group)

    return groups


def extract_joint_trajectories(
    joint_group: Dict[str, Tuple[int, int]],
    rollouts: dict,
) -> Dict[str, np.ndarray]:
    """从 rollouts 中提取联合组对应的轨迹。

    Parameters
    ----------
    joint_group : dict[agent_id -> (g_idx, k_idx)]
    rollouts : 包含每辆车 trajectory_all 的 dict

    Returns
    -------
    trajectories : dict[agent_id -> np.ndarray [8, 3]]
    """
    trajectories = {}
    for agent_id, (g_idx, k_idx) in joint_group.items():
        traj = rollouts[agent_id]["trajectory_all"][g_idx, k_idx]  # [8, 3]
        trajectories[agent_id] = traj.detach().cpu().numpy()
    return trajectories
```

### 扩展 `evaluation/reward_terms.py`：新增 team reward

**文件**：`evaluation/reward_terms.py`
**位置**：在 `compute_trajectory_reward()` 之后新增

```python
def compute_team_reward(
    per_agent_step_infos: Dict[str, List[dict]],
    config: dict,
) -> float:
    """计算联合组的 team reward。

    包含：
    1. 编队整体 formation reward（所有车的平均编队误差）
    2. 编队安全 team-safety reward（任意两车碰撞惩罚）
    3. 编队效率 efficiency reward（编队整体前进距离）

    Parameters
    ----------
    per_agent_step_infos : dict[agent_id -> list[dict]]
        每辆车每步的 info，来自闭环执行。
    config : dict
        权重配置。

    Returns
    -------
    team_reward : float
    """
    config = config or {}
    w_formation = _as_float(config, "w_team_formation", 0.5)
    w_safety = _as_float(config, "w_team_safety", 1.0)
    w_efficiency = _as_float(config, "w_team_efficiency", 0.3)
    collision_penalty = _as_float(config, "w_team_collision", 10.0)

    agent_ids = list(per_agent_step_infos.keys())
    num_steps = max(len(infos) for infos in per_agent_step_infos.values()) if agent_ids else 0

    if num_steps == 0:
        return 0.0

    # 1. formation: 所有车所有步的平均编队误差
    total_formation_error = 0.0
    count = 0
    for agent_id, step_infos in per_agent_step_infos.items():
        for info in step_infos:
            total_formation_error += abs(_as_float(info, "formation_error", 0.0))
            count += 1
    avg_formation_error = total_formation_error / max(count, 1)
    d_norm = max(_as_float(config, "d_norm", 10.0), 1e-6)
    r_formation = -(avg_formation_error / d_norm)

    # 2. team-safety: 任意车碰撞
    any_crash = False
    for step_infos in per_agent_step_infos.values():
        for info in step_infos:
            if _as_bool(info, "crash", False):
                any_crash = True
                break
    r_safety = -collision_penalty if any_crash else 0.0

    # 3. efficiency: 编队整体前进距离
    total_progress = 0.0
    for step_infos in per_agent_step_infos.values():
        for info in step_infos:
            total_progress += max(_as_float(info, "progress", 0.0), 0.0)
    avg_progress = total_progress / max(len(agent_ids), 1)
    delta_s_max = max(_as_float(config, "delta_s_max", 5.0), 1e-6)
    r_efficiency = avg_progress / delta_s_max

    team_reward = (
        w_formation * r_formation
        + w_safety * r_safety
        + w_efficiency * r_efficiency
    )
    return float(team_reward)
```

## 验收标准

**交付物**：`train/joint_group.py`、修改后的 `evaluation/reward_terms.py`、`tests/acceptance/test_phase5v2_task5.py`

**验收指标**（9 项）：
1. `select_top_k_candidates()` 返回 list of (g, k) tuples，长度 = top_k
2. 返回的候选按 reward 降序排列
3. 碰撞轨迹不会出现在 top-K 中（除非全部碰撞）
4. **【新增】全部碰撞时的 fallback**：当 `crash_per_anchor` 全为 True 时，`select_top_k_candidates()` 仍需返回长度为 top_k 的候选列表（而非空列表），从所有碰撞轨迹中按 reward 取 top-k。测试方法：构造 `crash_per_anchor = torch.ones(G, M, dtype=torch.bool)`，调用后断言 `len(result) == top_k`
5. `build_joint_groups()` 返回 list of dict，长度 ≤ num_groups
6. 每个联合组包含所有 agent_id
7. `extract_joint_trajectories()` 返回每辆车 [8, 3] 的 np.ndarray
8. `compute_team_reward()` 返回 float，碰撞时 reward < 0
9. 无碰撞时 team_reward > 碰撞时的 team_reward

**验收命令**：
```bash
pytest tests/acceptance/test_phase5v2_task5.py -v
```

## 注意事项

1. **组合数控制**：3 车 × top-2 = 2³=8 个组合，刚好 M_joint=8，可以穷举。如果车数增多（如 5 车），2⁵=32 > 8，需要随机采样
2. **top-K 的 K 值**：建议 K=2。K=1 退化为确定性选择（无多样性），K=3 组合数 3³=27 太大
3. **team reward 与 local reward 的量纲要匹配**：team reward 和 local reward 在 advantage 合并时需要数值可比，建议两者都做归一化后再加权
4. **闭环执行顺序**：M 个联合组的执行是串行的（每组 8 步），总计 M×8 次 env.step()
