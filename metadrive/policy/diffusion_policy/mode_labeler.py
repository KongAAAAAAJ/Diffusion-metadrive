from __future__ import annotations

import numpy as np

from metadrive.policy.diffusion_policy.mode_definitions import MODE_SLOTS, ModeSlot


def label_mode_from_expert_decision(
    lateral_decision: int,
    gt_trajectory: np.ndarray,
    coarse_trajectories: np.ndarray,
    mode_valid_mask: np.ndarray,
    mode_slots: tuple[ModeSlot, ...] | None = None,
) -> int:
    """Map GT trajectory to a dynamic mode slot using expert lateral decision."""
    slots = MODE_SLOTS if mode_slots is None else tuple(mode_slots)
    gt = np.asarray(gt_trajectory, dtype=np.float32)          # (T, 2)
    coarse = np.asarray(coarse_trajectories, dtype=np.float32) # (N, T, 2)
    mask = np.asarray(mode_valid_mask, dtype=bool)             # (N,)

    if lateral_decision == 0:
        candidate_slots = [
            slot.index for slot in slots
            if slot.semantic_group == "KEEP_LANE"
        ]
    elif lateral_decision < 0:
        candidate_slots = [
            slot.index for slot in slots
            if slot.lateral_direction == "left"
        ]
    else:
        candidate_slots = [
            slot.index for slot in slots
            if slot.lateral_direction == "right"
        ]

    # --- Stage 2: speed profile by L2 distance within the lateral group ---
    best_slot, best_dist = -1, float("inf")
    for slot_idx in candidate_slots:
        if slot_idx >= coarse.shape[0] or slot_idx >= mask.shape[0] or not mask[slot_idx]:
            continue
        dist = float(np.linalg.norm(gt[None] - coarse[slot_idx], axis=-1).mean())
        if dist < best_dist:
            best_dist, best_slot = dist, slot_idx

    if best_slot >= 0:
        return best_slot

    # Fallback 1: KEEP_LANE group (always partially valid)
    for slot in slots:
        if slot.semantic_group == "KEEP_LANE" and slot.index < mask.shape[0] and mask[slot.index]:
            return slot.index

    # Fallback 2: any valid slot
    valid = np.nonzero(mask)[0]
    return int(valid[0]) if valid.size > 0 else 0


def label_hierarchical_mode(
    gt_trajectory: np.ndarray,
    coarse_trajectories: np.ndarray,
    mode_valid_mask: np.ndarray,
) -> int:
    """Map GT trajectory to the closest valid mode slot.

    Args:
        gt_trajectory: (T, 2) GT trajectory in ego-local frame.
        coarse_trajectories: (num_modes, T, 2) per-slot coarse trajectories.
        mode_valid_mask: (num_modes,) bool, True = slot is valid.

    Returns:
        Index of the valid slot whose coarse trajectory is closest to gt_trajectory
        by mean per-step L2 distance.  Falls back to slot 0 if no valid slot has a
        non-zero coarse trajectory.
    """
    gt = np.asarray(gt_trajectory, dtype=np.float32)          # (T, 2)
    coarse = np.asarray(coarse_trajectories, dtype=np.float32) # (N, T, 2)
    mask = np.asarray(mode_valid_mask, dtype=bool)             # (N,)

    dists = np.linalg.norm(gt[None] - coarse, axis=-1).mean(axis=-1)  # (N,)
    dists[~mask] = np.inf

    best = int(np.argmin(dists))
    if not np.isfinite(dists[best]):
        return 0
    return best
