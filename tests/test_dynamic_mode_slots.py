import numpy as np

from metadrive.policy.diffusion_policy.mode_context import ModeContext
from metadrive.policy.diffusion_policy.mode_definitions import build_mode_slots
from metadrive.policy.diffusion_policy.mode_trajectory_generator import ModeTrajectoryGenerator
from metadrive.policy.diffusion_policy.transfuser_config import TransfuserConfig


def _straight_polyline(offset_y: float = 0.0) -> np.ndarray:
    xs = np.linspace(0.0, 80.0, 41, dtype=np.float32)
    ys = np.full_like(xs, float(offset_y))
    return np.stack([xs, ys], axis=1)


def _mode_context() -> ModeContext:
    return ModeContext(
        ego_speed_mps=8.0,
        ego_heading=0.0,
        current_lane_polyline=_straight_polyline(0.0),
        current_lane_width=3.5,
        left_lane_polyline=_straight_polyline(3.5),
        right_lane_polyline=_straight_polyline(-3.5),
        left_branch_polyline=None,
        right_branch_polyline=None,
        front_object_distance=-1.0,
        front_object_speed_mps=-1.0,
        left_lane_gap=-1.0,
        right_lane_gap=-1.0,
        current_ref_lane_count=1,
        next_ref_lane_count=1,
        has_left_adjacent=True,
        has_right_adjacent=True,
        has_left_branch=False,
        has_right_branch=False,
    )


def test_default_mode_slots_use_compact_group_names() -> None:
    slots = build_mode_slots()
    assert [slot.name for slot in slots] == [
        "KEEP_HIGH",
        "KEEP_MEDIUM",
        "KEEP_LOW",
        "LEFT_LC_HIGH",
        "LEFT_LC_MEDIUM",
        "LEFT_LC_LOW",
        "RIGHT_LC_HIGH",
        "RIGHT_LC_MEDIUM",
        "RIGHT_LC_LOW",
        "STOP",
    ]
    assert [slot.index for slot in slots] == list(range(10))
    assert [slot.semantic_group for slot in slots] == [
        "KEEP",
        "KEEP",
        "KEEP",
        "LEFT_LC",
        "LEFT_LC",
        "LEFT_LC",
        "RIGHT_LC",
        "RIGHT_LC",
        "RIGHT_LC",
        "STOP",
    ]


def test_mode_slots_can_expand_lane_change_density() -> None:
    slots = build_mode_slots(
        keep_lane_count=3,
        lane_change_left_count=6,
        lane_change_right_count=6,
        emergency_stop_count=1,
    )

    assert len(slots) == 16
    assert [slot.name for slot in slots[:3]] == [
        "KEEP_HIGH",
        "KEEP_MEDIUM",
        "KEEP_LOW",
    ]
    assert [slot.name for slot in slots[3:9]] == [f"LEFT_LC_LEVEL_{i}" for i in range(6)]
    assert [slot.name for slot in slots[9:15]] == [f"RIGHT_LC_LEVEL_{i}" for i in range(6)]
    assert slots[-1].name == "STOP"


def test_transfuser_config_aligns_ego_fut_mode_with_mode_counts() -> None:
    config = TransfuserConfig(
        mode_keep_lane_count=3,
        mode_lane_change_left_count=6,
        mode_lane_change_right_count=6,
        mode_emergency_stop_count=1,
    )
    assert config.ego_fut_mode == 16


def test_mode_trajectory_generator_uses_configured_slot_count() -> None:
    generator = ModeTrajectoryGenerator(
        keep_lane_level_count=3,
        lane_change_left_level_count=6,
        lane_change_right_level_count=6,
        emergency_stop_level_count=1,
    )
    output = generator.generate(_mode_context())

    assert output.coarse_trajectories.shape == (16, 8, 2)
    assert output.mode_valid_mask.shape == (16,)
    assert output.mode_valid_mask[-1]
