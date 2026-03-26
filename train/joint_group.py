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
    """根据奖励和碰撞信息选择每个智能体的top-k候选轨迹索引"""
    masked_reward = reward_per_anchor.clone()
    masked_reward[crash_per_anchor] = float('-inf')
    if bool(torch.isneginf(masked_reward).all().item()):
        masked_reward = reward_per_anchor.clone()

    flat = masked_reward.view(-1)
    k = min(int(top_k), flat.numel())
    _, indices = torch.topk(flat, k)

    num_anchors = reward_per_anchor.shape[1]
    candidates: List[Tuple[int, int]] = []
    for flat_idx in indices.tolist():
        candidates.append((int(flat_idx) // num_anchors, int(flat_idx) % num_anchors))
    return candidates


def build_joint_groups(
    per_agent_candidates: Dict[str, List[Tuple[int, int]]],
    num_groups: int = 8,
    seed: int | None = None,
) -> List[Dict[str, Tuple[int, int]]]:
    agent_ids = sorted(per_agent_candidates.keys())
    candidate_lists = [per_agent_candidates[agent_id] for agent_id in agent_ids]

    total_combinations = 1
    for candidates in candidate_lists:
        total_combinations *= max(len(candidates), 1)

    if total_combinations <= num_groups:
        combos = list(itertools.product(*candidate_lists))
    else:
        rng = random.Random(seed)
        combos = []
        seen = set()
        max_attempts = max(num_groups * 10, 10)
        for _ in range(max_attempts):
            combo = tuple(rng.choice(candidates) for candidates in candidate_lists)
            if combo in seen:
                continue
            seen.add(combo)
            combos.append(combo)
            if len(combos) >= num_groups:
                break

    groups: List[Dict[str, Tuple[int, int]]] = []
    for combo in combos[:num_groups]:
        groups.append({agent_ids[idx]: combo[idx] for idx in range(len(agent_ids))})
    return groups


def extract_joint_trajectories(
    joint_group: Dict[str, Tuple[int, int]],
    rollouts: dict,
) -> Dict[str, np.ndarray]:
    trajectories: Dict[str, np.ndarray] = {}
    for agent_id, (g_idx, k_idx) in joint_group.items():
        traj = rollouts[agent_id]["trajectory_all"][g_idx, k_idx]
        trajectories[agent_id] = traj.detach().cpu().numpy()
    return trajectories
