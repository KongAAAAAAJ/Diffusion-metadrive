from __future__ import annotations

import numpy as np
import pytest

from models.diffusion.mode_definitions import NUM_MODE_SLOTS
from models.diffusion.mode_labeler import label_hierarchical_mode


def _zeros() -> np.ndarray:
    return np.zeros((NUM_MODE_SLOTS, 8, 2), dtype=np.float32)


def _all_valid() -> np.ndarray:
    return np.ones((NUM_MODE_SLOTS,), dtype=bool)


def test_straight_gt_selects_keep_lane_slot():
    """A forward-going GT trajectory should match one of the KEEP_LANE slots (0-2)."""
    coarse = _zeros()
    # Slot 0: straight ahead, high speed (x increases fast)
    coarse[0] = np.stack([np.linspace(0.5, 52.0, 8), np.zeros(8)], axis=1).astype(np.float32)
    # Slot 1: straight ahead, medium speed
    coarse[1] = np.stack([np.linspace(0.5, 32.0, 8), np.zeros(8)], axis=1).astype(np.float32)

    gt = np.stack([np.linspace(0.5, 50.0, 8), np.zeros(8)], axis=1).astype(np.float32)

    label = label_hierarchical_mode(gt, coarse, _all_valid())
    assert label in {0, 1}, f"Expected KEEP_LANE slot, got {label}"


def test_left_gt_selects_left_lane_change_slot():
    """A GT trajectory with strong leftward offset should match slot 3/4/5."""
    coarse = _zeros()
    # Slot 3: lane-change-left high speed
    coarse[3] = np.stack([np.linspace(0.5, 52.0, 8), np.linspace(0.0, 4.0, 8)], axis=1).astype(np.float32)
    # Slot 0: keep lane
    coarse[0] = np.stack([np.linspace(0.5, 52.0, 8), np.zeros(8)], axis=1).astype(np.float32)

    gt = np.stack([np.linspace(0.5, 50.0, 8), np.linspace(0.0, 4.2, 8)], axis=1).astype(np.float32)

    label = label_hierarchical_mode(gt, coarse, _all_valid())
    assert label in {3, 4, 5}, f"Expected left LC slot, got {label}"


def test_right_gt_selects_right_lane_change_slot():
    """A GT trajectory with strong rightward offset should match slot 6/7/8."""
    coarse = _zeros()
    coarse[6] = np.stack([np.linspace(0.5, 52.0, 8), np.linspace(0.0, -4.0, 8)], axis=1).astype(np.float32)
    coarse[0] = np.stack([np.linspace(0.5, 52.0, 8), np.zeros(8)], axis=1).astype(np.float32)

    gt = np.stack([np.linspace(0.5, 50.0, 8), np.linspace(0.0, -4.2, 8)], axis=1).astype(np.float32)

    label = label_hierarchical_mode(gt, coarse, _all_valid())
    assert label in {6, 7, 8}, f"Expected right LC slot, got {label}"


def test_invalid_slots_are_not_selected():
    """Invalid slots must never be chosen even if they are closest."""
    coarse = _zeros()
    # Slot 3 is the exact match but is invalid
    coarse[3] = np.stack([np.linspace(0.5, 50.0, 8), np.linspace(0.0, 4.0, 8)], axis=1).astype(np.float32)
    # Slot 0 is valid but less close
    coarse[0] = np.stack([np.linspace(0.5, 52.0, 8), np.zeros(8)], axis=1).astype(np.float32)

    gt = np.stack([np.linspace(0.5, 50.0, 8), np.linspace(0.0, 4.0, 8)], axis=1).astype(np.float32)

    mask = _all_valid()
    mask[3] = False
    mask[4] = False
    mask[5] = False

    label = label_hierarchical_mode(gt, coarse, mask)
    assert label not in {3, 4, 5}, f"Selected invalid slot {label}"


def test_fallback_to_slot_0_when_all_valid_coarse_are_zero():
    """If all valid coarse trajectories are zero, fall back to slot 0."""
    coarse = _zeros()  # all zero
    mask = np.zeros((NUM_MODE_SLOTS,), dtype=bool)
    mask[0] = True

    gt = np.stack([np.linspace(1.0, 8.0, 8), np.zeros(8)], axis=1).astype(np.float32)

    # All valid slots have zero coarse — distances will all be the same
    label = label_hierarchical_mode(gt, coarse, mask)
    assert label == 0


def test_emergency_stop_selected_when_only_valid():
    """Slot 9 (EMERGENCY_STOP) should be selectable if it is the only valid slot."""
    coarse = _zeros()
    coarse[9] = np.stack([np.linspace(0.0, 0.5, 8), np.zeros(8)], axis=1).astype(np.float32)

    gt = np.stack([np.linspace(0.0, 0.4, 8), np.zeros(8)], axis=1).astype(np.float32)

    mask = np.zeros((NUM_MODE_SLOTS,), dtype=bool)
    mask[9] = True

    label = label_hierarchical_mode(gt, coarse, mask)
    assert label == 9
