from __future__ import annotations

import numpy as np
import pytest

from envs.observations.semantic_bev import BEVChannel, SemanticBEVRasterizer
from models.bev_planner.mode_contract import (
    MODE_NAMES,
    NUM_MODES,
    TRAJECTORY_SHAPE,
    HardModeMaskResult,
    ModeContractError,
    ModeIndex,
    ModeTopology,
    RuleAction,
    build_hard_mode_valid_mask,
    label_gt_mode,
)


EGO_SPEED_MPS = 8.0


def _trajectory_from_xy(xy: np.ndarray) -> np.ndarray:
    points = np.asarray(xy, dtype=np.float32)
    segments = np.diff(np.concatenate([np.zeros((1, 2), dtype=np.float32), points], axis=0), axis=0)
    heading = np.arctan2(segments[:, 1], segments[:, 0]).astype(np.float32)
    return np.concatenate([points, heading[:, None]], axis=1)


def _straight(speed_mps: float, lateral_m: float = 0.0) -> np.ndarray:
    times = np.arange(1, 9, dtype=np.float32) * 0.5
    xy = np.stack(
        [speed_mps * times, np.full_like(times, float(lateral_m))], axis=1
    )
    return _trajectory_from_xy(xy)


def _lane_change(speed_mps: float, direction: float) -> np.ndarray:
    progress = np.arange(1, 9, dtype=np.float32) / 8.0
    blend = 10.0 * progress**3 - 15.0 * progress**4 + 6.0 * progress**5
    xy = np.stack([speed_mps * progress * 4.0, direction * 3.5 * blend], axis=1)
    return _trajectory_from_xy(xy)


def _stop() -> np.ndarray:
    segment_speeds = np.asarray([8.0, 6.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    x = np.cumsum(segment_speeds * 0.5)
    return _trajectory_from_xy(np.stack([x, np.zeros_like(x)], axis=1))


def _coarse() -> np.ndarray:
    trajectories = np.zeros(TRAJECTORY_SHAPE, dtype=np.float32)
    for index, speed in zip((0, 1, 2), (10.0, 8.0, 6.0)):
        trajectories[index] = _straight(speed)
    for index, speed in zip((3, 4, 5), (10.0, 8.0, 6.0)):
        trajectories[index] = _lane_change(speed, 1.0)
    for index, speed in zip((6, 7, 8), (10.0, 8.0, 6.0)):
        trajectories[index] = _lane_change(speed, -1.0)
    trajectories[ModeIndex.STOP] = _stop()
    return trajectories


def _road_bev(half_width_m: float = 5.0) -> np.ndarray:
    bev = np.zeros((8, 256, 256), dtype=np.uint8)
    rasterizer = SemanticBEVRasterizer()
    left_col = rasterizer.ego_to_pixel(np.asarray([[0.0, half_width_m]], dtype=np.float32))[0, 0]
    right_col = rasterizer.ego_to_pixel(np.asarray([[0.0, -half_width_m]], dtype=np.float32))[0, 0]
    col_min = max(0, int(np.floor(min(left_col, right_col))))
    col_max = min(255, int(np.ceil(max(left_col, right_col))))
    bev[BEVChannel.DRIVABLE, :, col_min : col_max + 1] = 255
    return bev


def _all_reachable() -> ModeTopology:
    return ModeTopology(left_reachable=True, right_reachable=True)


def _mask(bev: np.ndarray | None = None, coarse: np.ndarray | None = None) -> HardModeMaskResult:
    return build_hard_mode_valid_mask(
        _road_bev() if bev is None else bev,
        _coarse() if coarse is None else coarse,
        EGO_SPEED_MPS,
        _all_reachable(),
    )


def test_fixed_mode_contract() -> None:
    assert NUM_MODES == 10
    assert TRAJECTORY_SHAPE == (10, 8, 3)
    assert MODE_NAMES == (
        "KEEP_HIGH",
        "KEEP_MEDIUM",
        "KEEP_LOW",
        "LEFT_HIGH",
        "LEFT_MEDIUM",
        "LEFT_LOW",
        "RIGHT_HIGH",
        "RIGHT_MEDIUM",
        "RIGHT_LOW",
        "STOP",
    )
    result = _mask()
    for component in (
        result.topology_mask,
        result.road_mask,
        result.kinematic_mask,
        result.valid_mask,
    ):
        assert component.shape == (10,)
        assert component.dtype == np.bool_
        assert not component.flags.writeable


def test_topology_disables_unreachable_lateral_group() -> None:
    result = build_hard_mode_valid_mask(
        _road_bev(),
        _coarse(),
        EGO_SPEED_MPS,
        ModeTopology(left_reachable=False, right_reachable=True),
    )
    assert not result.valid_mask[3:6].any()
    assert result.valid_mask[6:9].all()
    assert result.valid_mask[0:3].all()


def test_road_mask_checks_swept_vehicle_footprint() -> None:
    coarse = _coarse()
    coarse[ModeIndex.KEEP_HIGH] = _straight(8.0, lateral_m=8.0)
    result = _mask(coarse=coarse)
    assert not result.road_mask[ModeIndex.KEEP_HIGH]
    assert result.road_mask[ModeIndex.KEEP_MEDIUM]
    assert not result.valid_mask[ModeIndex.KEEP_HIGH]


def test_kinematic_mask_rejects_speed_acceleration_and_sharp_turn() -> None:
    coarse = _coarse()
    coarse[ModeIndex.KEEP_HIGH] = _straight(30.0)

    segment_speeds = np.asarray([8.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0, 14.0], dtype=np.float32)
    accel_x = np.cumsum(segment_speeds * 0.5)
    coarse[ModeIndex.KEEP_MEDIUM] = _trajectory_from_xy(
        np.stack([accel_x, np.zeros_like(accel_x)], axis=1)
    )

    sharp_x = np.arange(1, 9, dtype=np.float32) * 3.0
    sharp_y = np.asarray([0.0, 3.0, -3.0, 3.0, -3.0, 3.0, -3.0, 3.0], dtype=np.float32)
    coarse[ModeIndex.KEEP_LOW] = _trajectory_from_xy(np.stack([sharp_x, sharp_y], axis=1))
    result = _mask(bev=np.full((8, 256, 256), 255, dtype=np.uint8), coarse=coarse)
    assert not result.kinematic_mask[ModeIndex.KEEP_HIGH]
    assert not result.kinematic_mask[ModeIndex.KEEP_MEDIUM]
    assert not result.kinematic_mask[ModeIndex.KEEP_LOW]


def test_stop_is_always_valid_when_all_road_checks_fail() -> None:
    result = _mask(bev=np.zeros((8, 256, 256), dtype=np.uint8))
    assert not result.road_mask.any()
    assert result.valid_mask[ModeIndex.STOP]
    assert result.valid_mask.sum() == 1


def test_non_drivable_channels_do_not_change_hard_mask() -> None:
    base = _road_bev()
    changed = base.copy()
    rng = np.random.default_rng(7)
    changed[1:] = rng.integers(0, 256, size=changed[1:].shape, dtype=np.uint8)
    before = _mask(bev=base)
    after = _mask(bev=changed)
    np.testing.assert_array_equal(before.valid_mask, after.valid_mask)
    np.testing.assert_array_equal(before.road_mask, after.road_mask)


@pytest.mark.parametrize(
    ("action", "expected_mode"),
    (
        (RuleAction.LEFT, ModeIndex.LEFT_MEDIUM),
        (RuleAction.KEEP, ModeIndex.KEEP_MEDIUM),
        (RuleAction.RIGHT, ModeIndex.RIGHT_MEDIUM),
    ),
)
def test_gt_label_stays_in_rule_group_and_is_hard_valid(action, expected_mode) -> None:
    coarse = _coarse()
    valid_mask = _mask(coarse=coarse).valid_mask.copy()
    selected = label_gt_mode(action, coarse[expected_mode], coarse, valid_mask)
    assert selected == expected_mode
    assert valid_mask[selected]


def test_keep_and_stop_compete_for_emergency_braking_label() -> None:
    coarse = _coarse()
    valid_mask = _mask(coarse=coarse).valid_mask.copy()
    selected = label_gt_mode(RuleAction.KEEP, coarse[ModeIndex.STOP], coarse, valid_mask)
    assert selected == ModeIndex.STOP
    assert valid_mask[selected]


def test_invalid_rule_group_raises_without_relabeling() -> None:
    coarse = _coarse()
    valid_mask = np.zeros((NUM_MODES,), dtype=bool)
    valid_mask[ModeIndex.STOP] = True
    with pytest.raises(ModeContractError, match="joint sample must be discarded"):
        label_gt_mode(RuleAction.LEFT, coarse[ModeIndex.LEFT_MEDIUM], coarse, valid_mask)


def test_non_integer_rule_action_is_not_silently_coerced() -> None:
    coarse = _coarse()
    valid_mask = np.ones((NUM_MODES,), dtype=bool)
    with pytest.raises(ModeContractError, match="one of -1, 0, 1"):
        label_gt_mode(0.5, coarse[ModeIndex.KEEP_MEDIUM], coarse, valid_mask)


def test_invalid_shapes_dtypes_and_non_finite_values_raise() -> None:
    with pytest.raises(ModeContractError, match="bev must have dtype uint8"):
        build_hard_mode_valid_mask(
            _road_bev().astype(np.float32), _coarse(), EGO_SPEED_MPS, _all_reachable()
        )
    with pytest.raises(ModeContractError, match="coarse_trajectories must have shape"):
        build_hard_mode_valid_mask(
            _road_bev(), np.zeros((10, 8, 2), dtype=np.float32), EGO_SPEED_MPS, _all_reachable()
        )
    coarse = _coarse()
    coarse[0, 0, 0] = np.nan
    with pytest.raises(ModeContractError, match="non-finite"):
        build_hard_mode_valid_mask(_road_bev(), coarse, EGO_SPEED_MPS, _all_reachable())
