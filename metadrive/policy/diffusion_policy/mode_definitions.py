from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Sequence, Tuple


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
    level_index: int = 0
    level_count: int = 1
    semantic_group: str = ""

    @property
    def level_fraction(self) -> float:
        """Normalized group level: 1.0 is the fastest/highest slot, 0.0 the slowest."""
        if self.level_count <= 1:
            return 1.0
        return 1.0 - float(self.level_index) / float(self.level_count - 1)


DEFAULT_KEEP_LANE_COUNT = 3
DEFAULT_LANE_CHANGE_LEFT_COUNT = 3
DEFAULT_LANE_CHANGE_RIGHT_COUNT = 3
DEFAULT_EMERGENCY_STOP_COUNT = 1


def _profile_for_level(level_index: int, level_count: int) -> SpeedProfile:
    if level_count <= 1:
        return SpeedProfile.MEDIUM
    fraction = 1.0 - float(level_index) / float(level_count - 1)
    if fraction >= 2.0 / 3.0:
        return SpeedProfile.HIGH
    if fraction >= 1.0 / 3.0:
        return SpeedProfile.MEDIUM
    return SpeedProfile.LOW


def _slot_name(group: str, level_index: int, level_count: int) -> str:
    if group == "EMERGENCY_STOP":
        return "EMERGENCY_STOP" if level_count == 1 else f"EMERGENCY_STOP_LEVEL_{level_index}"
    if level_count == 3:
        return f"{group}_{('HIGH', 'MEDIUM', 'LOW')[level_index]}"
    return f"{group}_LEVEL_{level_index}"


def _append_level_slots(
    slots: list[ModeSlot],
    *,
    group: str,
    count: int,
    behavior_type: BehaviorType,
    lateral_direction: Optional[str] = None,
) -> None:
    count = max(1, int(count))
    for level_index in range(count):
        slots.append(
            ModeSlot(
                index=len(slots),
                name=_slot_name(group, level_index, count),
                behavior_type=behavior_type,
                speed_profile=_profile_for_level(level_index, count),
                lateral_direction=lateral_direction,
                level_index=level_index,
                level_count=count,
                semantic_group=group,
            )
        )


def build_mode_slots(
    keep_lane_count: int = DEFAULT_KEEP_LANE_COUNT,
    lane_change_left_count: int = DEFAULT_LANE_CHANGE_LEFT_COUNT,
    lane_change_right_count: int = DEFAULT_LANE_CHANGE_RIGHT_COUNT,
    emergency_stop_count: int = DEFAULT_EMERGENCY_STOP_COUNT,
) -> Tuple[ModeSlot, ...]:
    """Build semantic mode slots with configurable density per group.

    Default 3/3/3/1 preserves the historical 10-mode names and indices.
    Increasing a group count keeps the semantic range fixed and samples more
    levels inside the same group.
    """
    slots: list[ModeSlot] = []
    _append_level_slots(slots, group="KEEP_LANE", count=keep_lane_count, behavior_type=BehaviorType.KEEP_LANE)
    _append_level_slots(
        slots,
        group="LANE_CHANGE_LEFT",
        count=lane_change_left_count,
        behavior_type=BehaviorType.LANE_CHANGE,
        lateral_direction="left",
    )
    _append_level_slots(
        slots,
        group="LANE_CHANGE_RIGHT",
        count=lane_change_right_count,
        behavior_type=BehaviorType.LANE_CHANGE,
        lateral_direction="right",
    )
    _append_level_slots(
        slots,
        group="EMERGENCY_STOP",
        count=emergency_stop_count,
        behavior_type=BehaviorType.KEEP_LANE,
    )
    return tuple(slots)


def mode_slot_count(
    keep_lane_count: int = DEFAULT_KEEP_LANE_COUNT,
    lane_change_left_count: int = DEFAULT_LANE_CHANGE_LEFT_COUNT,
    lane_change_right_count: int = DEFAULT_LANE_CHANGE_RIGHT_COUNT,
    emergency_stop_count: int = DEFAULT_EMERGENCY_STOP_COUNT,
) -> int:
    return (
        max(1, int(keep_lane_count))
        + max(1, int(lane_change_left_count))
        + max(1, int(lane_change_right_count))
        + max(1, int(emergency_stop_count))
    )


MODE_SLOTS: Tuple[ModeSlot, ...] = build_mode_slots()

NUM_MODE_SLOTS = len(MODE_SLOTS)


def get_mode_slot(index: int, mode_slots: Sequence[ModeSlot] | None = None) -> ModeSlot:
    slots = MODE_SLOTS if mode_slots is None else tuple(mode_slots)
    if index < 0 or index >= len(slots):
        raise IndexError(f"Invalid mode slot index: {index}")
    return slots[index]
