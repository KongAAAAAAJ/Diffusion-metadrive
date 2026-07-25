"""Lossless bit packing for the frozen eight-channel semantic BEV contract."""

from __future__ import annotations

from typing import Final

import numpy as np

from envs.observations.semantic_bev import BEVChannel, BEV_CHANNEL_NAMES


LOGICAL_BEV_SHAPE: Final = (8, 256, 256)
PACKED_BEV_SHAPE: Final = (10, 256, 32)
PACKED_BITORDER: Final = "little"
PACKED_PLANE_NAMES: Final = (
    "drivable",
    "lane_geometry_bit0",
    "lane_geometry_bit1",
    "navigation_route",
    "ego_history_bit0",
    "ego_history_bit1",
    "platoon_vehicles",
    "background_t0",
    "background_t_minus_0_5",
    "background_t_minus_1_0",
)
BINARY_CHANNELS: Final = (
    BEVChannel.DRIVABLE,
    BEVChannel.NAVIGATION_ROUTE,
    BEVChannel.PLATOON_VEHICLES,
    BEVChannel.BACKGROUND_T0,
    BEVChannel.BACKGROUND_T_MINUS_0_5,
    BEVChannel.BACKGROUND_T_MINUS_1_0,
)
BINARY_PLANE_BY_CHANNEL: Final = {
    BEVChannel.DRIVABLE: 0,
    BEVChannel.NAVIGATION_ROUTE: 3,
    BEVChannel.PLATOON_VEHICLES: 6,
    BEVChannel.BACKGROUND_T0: 7,
    BEVChannel.BACKGROUND_T_MINUS_0_5: 8,
    BEVChannel.BACKGROUND_T_MINUS_1_0: 9,
}
LANE_VALUES: Final = np.asarray((0, 128, 255), dtype=np.uint8)
EGO_HISTORY_VALUES: Final = np.asarray((0, 85, 170, 255), dtype=np.uint8)
RAW_BITS_PER_PIXEL: Final = 8 * len(BEVChannel)
PACKED_BITS_PER_PIXEL: Final = len(PACKED_PLANE_NAMES)
BEV_COMPRESSION_RATIO: Final = RAW_BITS_PER_PIXEL / PACKED_BITS_PER_PIXEL


class SemanticBEVCodecError(ValueError):
    """Raised when a BEV tensor violates the frozen semantic encoding."""


def _validate_tail(array: np.ndarray, tail: tuple[int, ...], name: str) -> None:
    if array.dtype != np.uint8:
        raise SemanticBEVCodecError(f"{name} must have dtype uint8, got {array.dtype}")
    if array.ndim < len(tail) or tuple(array.shape[-len(tail) :]) != tail:
        raise SemanticBEVCodecError(
            f"{name} must end with shape {tail}, got {array.shape}"
        )


def _require_values(
    channel: np.ndarray,
    allowed: np.ndarray,
    channel_name: str,
) -> None:
    if not np.isin(channel, allowed).all():
        observed = np.unique(channel)
        invalid = observed[~np.isin(observed, allowed)]
        raise SemanticBEVCodecError(
            f"{channel_name} contains invalid values {invalid.tolist()}; "
            f"allowed={allowed.tolist()}"
        )


def _pack_plane(bits: np.ndarray) -> np.ndarray:
    return np.packbits(bits, axis=-1, bitorder=PACKED_BITORDER)


def pack_semantic_bev(bev: np.ndarray) -> np.ndarray:
    """Pack ``[...,8,256,256] uint8`` BEV tensors into ten bit-planes."""

    array = np.asarray(bev)
    _validate_tail(array, LOGICAL_BEV_SHAPE, "bev")
    for channel in BINARY_CHANNELS:
        _require_values(
            array[..., int(channel), :, :],
            np.asarray((0, 255), dtype=np.uint8),
            BEV_CHANNEL_NAMES[int(channel)],
        )
    lane = array[..., int(BEVChannel.LANE_GEOMETRY), :, :]
    history = array[..., int(BEVChannel.EGO_HISTORY), :, :]
    _require_values(lane, LANE_VALUES, "lane_geometry")
    _require_values(history, EGO_HISTORY_VALUES, "ego_history")

    leading = array.shape[: -len(LOGICAL_BEV_SHAPE)]
    packed = np.empty((*leading, *PACKED_BEV_SHAPE), dtype=np.uint8)
    for channel, plane in BINARY_PLANE_BY_CHANNEL.items():
        packed[..., plane, :, :] = _pack_plane(
            array[..., int(channel), :, :] == np.uint8(255)
        )

    lane_code = np.zeros(lane.shape, dtype=np.uint8)
    lane_code[lane == np.uint8(128)] = np.uint8(1)
    lane_code[lane == np.uint8(255)] = np.uint8(2)
    packed[..., 1, :, :] = _pack_plane(lane_code & np.uint8(1))
    packed[..., 2, :, :] = _pack_plane((lane_code >> np.uint8(1)) & np.uint8(1))

    history_code = history // np.uint8(85)
    packed[..., 4, :, :] = _pack_plane(history_code & np.uint8(1))
    packed[..., 5, :, :] = _pack_plane(
        (history_code >> np.uint8(1)) & np.uint8(1)
    )
    return np.ascontiguousarray(packed)


def unpack_semantic_bev(packed: np.ndarray) -> np.ndarray:
    """Restore packed semantic BEV tensors to the logical uint8 contract."""

    array = np.asarray(packed)
    _validate_tail(array, PACKED_BEV_SHAPE, "packed_bev")
    bits = np.unpackbits(
        array,
        axis=-1,
        count=LOGICAL_BEV_SHAPE[-1],
        bitorder=PACKED_BITORDER,
    )
    leading = array.shape[: -len(PACKED_BEV_SHAPE)]
    bev = np.zeros((*leading, *LOGICAL_BEV_SHAPE), dtype=np.uint8)
    for channel, plane in BINARY_PLANE_BY_CHANNEL.items():
        bev[..., int(channel), :, :] = bits[..., plane, :, :] * np.uint8(255)

    lane_code = bits[..., 1, :, :] | (bits[..., 2, :, :] << np.uint8(1))
    if np.any(lane_code == np.uint8(3)):
        raise SemanticBEVCodecError("packed lane_geometry contains reserved code 3")
    lane_values = np.asarray((0, 128, 255), dtype=np.uint8)
    bev[..., int(BEVChannel.LANE_GEOMETRY), :, :] = lane_values[lane_code]

    history_code = bits[..., 4, :, :] | (bits[..., 5, :, :] << np.uint8(1))
    history_values = np.asarray((0, 85, 170, 255), dtype=np.uint8)
    bev[..., int(BEVChannel.EGO_HISTORY), :, :] = history_values[history_code]
    return np.ascontiguousarray(bev)


def packed_bev_contract() -> dict[str, object]:
    """Return the JSON-serializable physical BEV storage contract."""

    return {
        "logical_shape": list(LOGICAL_BEV_SHAPE),
        "packed_shape": list(PACKED_BEV_SHAPE),
        "dtype": "uint8",
        "bitorder": PACKED_BITORDER,
        "plane_names": list(PACKED_PLANE_NAMES),
        "binary_values": [0, 255],
        "lane_geometry_mapping": {"0": 0, "128": 1, "255": 2},
        "ego_history_mapping": {"0": 0, "85": 1, "170": 2, "255": 3},
        "raw_bits_per_pixel": RAW_BITS_PER_PIXEL,
        "packed_bits_per_pixel": PACKED_BITS_PER_PIXEL,
        "compression_ratio": BEV_COMPRESSION_RATIO,
    }


__all__ = [
    "BEV_COMPRESSION_RATIO",
    "BINARY_CHANNELS",
    "LOGICAL_BEV_SHAPE",
    "PACKED_BEV_SHAPE",
    "PACKED_BITORDER",
    "PACKED_PLANE_NAMES",
    "SemanticBEVCodecError",
    "pack_semantic_bev",
    "packed_bev_contract",
    "unpack_semantic_bev",
]
