from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np

from models.diffusion.mode_context import ModeContext
from models.diffusion.mode_definitions import BehaviorType, MODE_SLOTS, ModeSlot


@dataclass
class ModeFeasibilityResult:
    geometric_valid_mask: np.ndarray
    traffic_valid_mask: np.ndarray

    @property
    def valid_mask(self) -> np.ndarray:
        return np.logical_and(self.geometric_valid_mask, self.traffic_valid_mask)


class GeometricFeasibilityChecker:
    def evaluate(self, ctx: ModeContext, mode_slots: Sequence[ModeSlot] | None = None) -> np.ndarray:
        slots = MODE_SLOTS if mode_slots is None else tuple(mode_slots)
        mask = np.ones((len(slots),), dtype=bool)
        for slot in slots:
            if slot.behavior_type != BehaviorType.LANE_CHANGE:
                continue
            if slot.lateral_direction == "left" and not (ctx.has_left_adjacent or ctx.has_left_branch):
                mask[slot.index] = False
            if slot.lateral_direction == "right" and not (ctx.has_right_adjacent or ctx.has_right_branch):
                mask[slot.index] = False
        return mask


class TrafficFeasibilityChecker:
    def __init__(
        self,
        lane_change_min_gap_m: float = 12.0,
        keep_lane_medium_front_window_m: float = 45.0,
    ) -> None:
        self.lane_change_min_gap_m = float(lane_change_min_gap_m)
        self.keep_lane_medium_front_window_m = float(keep_lane_medium_front_window_m)

    def evaluate(
        self,
        ctx: ModeContext,
        geometric_mask: np.ndarray | None = None,
        mode_slots: Sequence[ModeSlot] | None = None,
    ) -> np.ndarray:
        slots = MODE_SLOTS if mode_slots is None else tuple(mode_slots)
        mask = np.ones((len(slots),), dtype=bool)
        if geometric_mask is not None:
            mask = np.logical_and(mask, geometric_mask)

        front_distance = float(ctx.front_object_distance)
        has_front_vehicle = front_distance >= 0.0 and front_distance <= self.keep_lane_medium_front_window_m
        if not has_front_vehicle:
            for slot in slots:
                if slot.semantic_group == "KEEP" and 0.0 < slot.level_fraction < 1.0:
                    mask[slot.index] = False

        left_gap = float(ctx.left_lane_gap)
        right_gap = float(ctx.right_lane_gap)
        for slot in slots:
            if slot.behavior_type != BehaviorType.LANE_CHANGE:
                continue
            if slot.lateral_direction == "left" and left_gap >= 0.0 and left_gap < self.lane_change_min_gap_m:
                mask[slot.index] = False
            if slot.lateral_direction == "right" and right_gap >= 0.0 and right_gap < self.lane_change_min_gap_m:
                mask[slot.index] = False
        return mask


def build_mode_valid_mask(
    ctx: ModeContext,
    geometric_checker: GeometricFeasibilityChecker | None = None,
    traffic_checker: TrafficFeasibilityChecker | None = None,
    mode_slots: Sequence[ModeSlot] | None = None,
) -> ModeFeasibilityResult:
    slots = MODE_SLOTS if mode_slots is None else tuple(mode_slots)
    geometric_checker = geometric_checker or GeometricFeasibilityChecker()
    geometric_valid_mask = geometric_checker.evaluate(ctx, slots)
    return ModeFeasibilityResult(
        geometric_valid_mask=geometric_valid_mask,
        traffic_valid_mask=np.ones(len(slots), dtype=bool),
    )


def valid_mode_names(mask: np.ndarray, mode_slots: Sequence[ModeSlot] | None = None) -> Iterable[str]:
    slots = MODE_SLOTS if mode_slots is None else tuple(mode_slots)
    for slot, is_valid in zip(slots, mask):
        if bool(is_valid):
            yield slot.name


def mask_to_dict(mask: np.ndarray, mode_slots: Sequence[ModeSlot] | None = None) -> Dict[str, bool]:
    slots = MODE_SLOTS if mode_slots is None else tuple(mode_slots)
    return {slot.name: bool(mask[slot.index]) for slot in slots}
