from __future__ import annotations

import numpy as np

from metadrive.policy.diffusion_policy.mode_definitions import NUM_MODE_SLOTS


def label_mode_from_expert_decision(
    lateral_decision: int,
    gt_trajectory: np.ndarray,
    coarse_trajectories: np.ndarray,
    mode_valid_mask: np.ndarray,
) -> int:
    """Map GT trajectory to a MODE_SLOTS index using a two-stage strategy.

    Stage 1 — lateral direction from expert IDM decision (immune to curve bias):
        lateral_decision == 0  → candidate slots {0, 1, 2}  (KEEP_LANE_*)
        lateral_decision <  0  → candidate slots {3, 4, 5}  (LANE_CHANGE_LEFT_*)
        lateral_decision >  0  → candidate slots {6, 7, 8}  (LANE_CHANGE_RIGHT_*)

    Stage 2 — speed profile by L2 distance within the candidate group:
        Among the valid slots in the lateral group, pick the one whose
        coarse_trajectory is closest to gt_trajectory (mean per-step L2).
        This avoids the cross-group mislabeling that occurred with pure L2
        matching in curved road sections.

    Slot layout (must match MODE_SLOTS in mode_definitions.py):
        0: KEEP_LANE_HIGH      3: LANE_CHANGE_LEFT_HIGH    6: LANE_CHANGE_RIGHT_HIGH
        1: KEEP_LANE_MEDIUM    4: LANE_CHANGE_LEFT_MEDIUM  7: LANE_CHANGE_RIGHT_MEDIUM
        2: KEEP_LANE_LOW       5: LANE_CHANGE_LEFT_LOW     8: LANE_CHANGE_RIGHT_LOW
        9: EMERGENCY_STOP

    Args:
        lateral_decision: -1 = CHANGE_LEFT, 0 = KEEP, +1 = CHANGE_RIGHT.
        gt_trajectory: (T, 2) GT trajectory in ego-local frame.
        coarse_trajectories: (NUM_MODE_SLOTS, T, 2) per-slot coarse trajectories.
        mode_valid_mask: (NUM_MODE_SLOTS,) bool validity mask from ModeTrajectoryGenerator.

    Returns:
        Index of the best matching valid mode slot.
    """
    gt = np.asarray(gt_trajectory, dtype=np.float32)          # (T, 2)
    coarse = np.asarray(coarse_trajectories, dtype=np.float32) # (N, T, 2)
    mask = np.asarray(mode_valid_mask, dtype=bool)             # (N,)

    # --- Stage 1: lateral group from expert decision ---
    if lateral_decision == 0:
        candidate_slots = [0, 1, 2]
    elif lateral_decision < 0:
        candidate_slots = [3, 4, 5]
    else:
        candidate_slots = [6, 7, 8]

    # --- Stage 2: speed profile by L2 distance within the lateral group ---
    best_slot, best_dist = -1, float("inf")
    for slot in candidate_slots:
        if not mask[slot]:
            continue
        dist = float(np.linalg.norm(gt[None] - coarse[slot], axis=-1).mean())
        if dist < best_dist:
            best_dist, best_slot = dist, slot

    if best_slot >= 0:
        return best_slot

    # Fallback 1: KEEP_LANE group (always partially valid)
    for slot in [0, 1, 2]:
        if mask[slot]:
            return slot

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
        coarse_trajectories: (NUM_MODE_SLOTS, T, 2) per-slot coarse trajectories.
        mode_valid_mask: (NUM_MODE_SLOTS,) bool, True = slot is valid.

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
