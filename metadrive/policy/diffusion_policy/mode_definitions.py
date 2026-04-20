from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple


class BehaviorType(IntEnum):
    KEEP_LANE = 0
    LANE_CHANGE = 1


class SpeedProfile(IntEnum):
    HIGH = 0
    MEDIUM = 1
    LOW = 2

@dataclass(frozen=True)
class ModeSlot:
    index: int
    name: str
    behavior_type: BehaviorType
    speed_profile: SpeedProfile
    lateral_direction: Optional[str] = None


MODE_SLOTS: Tuple[ModeSlot, ...] = (
    ModeSlot(0, "KEEP_LANE_HIGH", BehaviorType.KEEP_LANE, SpeedProfile.HIGH),
    ModeSlot(1, "KEEP_LANE_MEDIUM", BehaviorType.KEEP_LANE, SpeedProfile.MEDIUM),
    ModeSlot(2, "KEEP_LANE_LOW", BehaviorType.KEEP_LANE, SpeedProfile.LOW),
    ModeSlot(3, "LANE_CHANGE_LEFT_HIGH", BehaviorType.LANE_CHANGE, SpeedProfile.HIGH, lateral_direction="left"),
    ModeSlot(4, "LANE_CHANGE_LEFT_MEDIUM", BehaviorType.LANE_CHANGE, SpeedProfile.MEDIUM, lateral_direction="left"),
    ModeSlot(5, "LANE_CHANGE_LEFT_LOW", BehaviorType.LANE_CHANGE, SpeedProfile.LOW, lateral_direction="left"),
    ModeSlot(6, "LANE_CHANGE_RIGHT_HIGH", BehaviorType.LANE_CHANGE, SpeedProfile.HIGH, lateral_direction="right"),
    ModeSlot(7, "LANE_CHANGE_RIGHT_MEDIUM", BehaviorType.LANE_CHANGE, SpeedProfile.MEDIUM, lateral_direction="right"),
    ModeSlot(8, "LANE_CHANGE_RIGHT_LOW", BehaviorType.LANE_CHANGE, SpeedProfile.LOW, lateral_direction="right"),
    ModeSlot(9, "EMERGENCY_STOP", BehaviorType.KEEP_LANE, SpeedProfile.LOW),
)

NUM_MODE_SLOTS = len(MODE_SLOTS)


def get_mode_slot(index: int) -> ModeSlot:
    if index < 0 or index >= NUM_MODE_SLOTS:
        raise IndexError(f"Invalid mode slot index: {index}")
    return MODE_SLOTS[index]
