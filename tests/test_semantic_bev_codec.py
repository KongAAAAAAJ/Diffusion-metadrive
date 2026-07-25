from __future__ import annotations

import numpy as np
import pytest

from envs.observations.semantic_bev import BEVChannel
from expert_dataset.semantic_bev_codec import (
    BEV_COMPRESSION_RATIO,
    LOGICAL_BEV_SHAPE,
    PACKED_BEV_SHAPE,
    SemanticBEVCodecError,
    pack_semantic_bev,
    packed_bev_contract,
    unpack_semantic_bev,
)


def _legal_bev(shape: tuple[int, ...] = (3, *LOGICAL_BEV_SHAPE)) -> np.ndarray:
    rng = np.random.default_rng(7)
    bev = np.zeros(shape, dtype=np.uint8)
    for channel in (
        BEVChannel.DRIVABLE,
        BEVChannel.NAVIGATION_ROUTE,
        BEVChannel.PLATOON_VEHICLES,
        BEVChannel.BACKGROUND_T0,
        BEVChannel.BACKGROUND_T_MINUS_0_5,
        BEVChannel.BACKGROUND_T_MINUS_1_0,
    ):
        bev[..., int(channel), :, :] = rng.choice(
            np.asarray((0, 255), dtype=np.uint8),
            size=shape[:-3] + shape[-2:],
        )
    bev[..., int(BEVChannel.LANE_GEOMETRY), :, :] = rng.choice(
        np.asarray((0, 128, 255), dtype=np.uint8),
        size=shape[:-3] + shape[-2:],
    )
    bev[..., int(BEVChannel.EGO_HISTORY), :, :] = rng.choice(
        np.asarray((0, 85, 170, 255), dtype=np.uint8),
        size=shape[:-3] + shape[-2:],
    )
    return bev


def test_codec_round_trip_supports_joint_and_episode_leading_dimensions() -> None:
    bev = _legal_bev((2, 3, *LOGICAL_BEV_SHAPE))
    packed = pack_semantic_bev(bev)
    restored = unpack_semantic_bev(packed)

    assert packed.shape == (2, 3, *PACKED_BEV_SHAPE)
    assert packed.dtype == np.uint8
    assert packed.flags.c_contiguous
    assert restored.shape == bev.shape
    assert restored.dtype == np.uint8
    assert restored.flags.c_contiguous
    np.testing.assert_array_equal(restored, bev)
    assert bev.nbytes / packed.nbytes == BEV_COMPRESSION_RATIO == 6.4


def test_little_endian_bit_order_is_frozen() -> None:
    bev = np.zeros(LOGICAL_BEV_SHAPE, dtype=np.uint8)
    bev[int(BEVChannel.DRIVABLE), 0, 0] = 255
    bev[int(BEVChannel.DRIVABLE), 0, 7] = 255
    bev[int(BEVChannel.LANE_GEOMETRY), 1, 0] = 128
    bev[int(BEVChannel.LANE_GEOMETRY), 1, 1] = 255

    packed = pack_semantic_bev(bev)

    assert int(packed[0, 0, 0]) == 0b10000001
    assert int(packed[1, 1, 0]) == 0b00000001
    assert int(packed[2, 1, 0]) == 0b00000010
    np.testing.assert_array_equal(unpack_semantic_bev(packed), bev)


@pytest.mark.parametrize(
    "mutator,match",
    (
        (lambda value: value.astype(np.float32), "dtype uint8"),
        (lambda value: value[..., :7, :, :], "end with shape"),
        (
            lambda value: np.pad(
                value,
                ((0, 0), (0, 0), (0, 0), (0, 1)),
                mode="constant",
            ),
            "end with shape",
        ),
    ),
)
def test_pack_rejects_wrong_dtype_or_shape(mutator, match: str) -> None:
    with pytest.raises(SemanticBEVCodecError, match=match):
        pack_semantic_bev(mutator(_legal_bev()))


@pytest.mark.parametrize(
    ("channel", "value", "match"),
    (
        (BEVChannel.DRIVABLE, 1, "drivable contains invalid"),
        (BEVChannel.LANE_GEOMETRY, 85, "lane_geometry contains invalid"),
        (BEVChannel.EGO_HISTORY, 128, "ego_history contains invalid"),
        (BEVChannel.BACKGROUND_T0, 128, "background_t0 contains invalid"),
    ),
)
def test_pack_rejects_illegal_semantic_values(
    channel: BEVChannel, value: int, match: str
) -> None:
    bev = _legal_bev()
    bev[0, int(channel), 0, 0] = np.uint8(value)
    with pytest.raises(SemanticBEVCodecError, match=match):
        pack_semantic_bev(bev)


def test_unpack_rejects_shape_dtype_and_reserved_lane_code() -> None:
    packed = np.zeros((3, *PACKED_BEV_SHAPE), dtype=np.uint8)
    with pytest.raises(SemanticBEVCodecError, match="dtype uint8"):
        unpack_semantic_bev(packed.astype(np.int16))
    with pytest.raises(SemanticBEVCodecError, match="end with shape"):
        unpack_semantic_bev(packed[..., :-1])

    packed[0, 1, 0, 0] = np.uint8(1)
    packed[0, 2, 0, 0] = np.uint8(1)
    with pytest.raises(SemanticBEVCodecError, match="reserved code 3"):
        unpack_semantic_bev(packed)


def test_contract_records_all_physical_encoding_details() -> None:
    contract = packed_bev_contract()
    assert contract["logical_shape"] == [8, 256, 256]
    assert contract["packed_shape"] == [10, 256, 32]
    assert contract["bitorder"] == "little"
    assert len(contract["plane_names"]) == 10
    assert contract["compression_ratio"] == 6.4
