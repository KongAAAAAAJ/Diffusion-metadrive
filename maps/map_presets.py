"""Hybrid map block configurations and route preset definitions.

These constants define the project's map geometry. They are the canonical
source; base_multi_env.py re-declares identical values for backward
compatibility with MetaDrive's existing import chain.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Route preset block sequences
# ---------------------------------------------------------------------------

DEFAULT_MAIN_ROUTE_BLOCK_IDS: Tuple[str, ...] = (
    "s0",
    "c0",
    "c1",
    "g0",
    "s_main0",
    "x0",
    "s_main1",
    "c2",
    "g1",
    "c3",
    "merge0",
    "s_main2",
    "split0",
    "c4",
)

DEFAULT_RAMP_MERGE_ROUTE_BLOCK_IDS: Tuple[str, ...] = (
    "s0",
    "c0",
    "c1",
    "g0",
    "s_ramp0",
    "c0_ramp0",
    "s_ramp1",
    "c1_ramp0",
    "h_ramp0",
    "g1",
    "c3",
    "merge0",
    "s_main2",
    "split0",
    "c4",
)

DEFAULT_ROUTE_PRESET: str = "ramp_merge"

ROUTE_PRESET_BLOCK_IDS: Dict[str, Tuple[str, ...]] = {
    "mainline": DEFAULT_MAIN_ROUTE_BLOCK_IDS,
    "ramp_merge": DEFAULT_RAMP_MERGE_ROUTE_BLOCK_IDS,
}

# ---------------------------------------------------------------------------
# Full hybrid map block config (18 blocks)
# ---------------------------------------------------------------------------

DEFAULT_HYBRID_MAP_CONFIG: List[Dict] = [
    {"block_id": "s0",      "id": "S", "parent_block_id": "root",      "parent_socket_index": 0, "length": 600.0},
    {"block_id": "c0",      "id": "C", "parent_block_id": "s0",        "parent_socket_index": 0, "length": 200.0, "radius": 80.0,  "angle": 150.0, "dir": 1},
    {"block_id": "c1",      "id": "C", "parent_block_id": "c0",        "parent_socket_index": 0, "length": 200.0, "radius": 80.0,  "angle": 180.0, "dir": 0},
    {"block_id": "g0",      "id": "G", "parent_block_id": "c1",        "parent_socket_index": 0, "length": 200.0, "extension_length": 80.0},
    {"block_id": "s_main0", "id": "S", "parent_block_id": "g0",        "parent_socket_index": 0, "length": 400.0},
    {"block_id": "x0",      "id": "X", "parent_block_id": "s_main0",   "parent_socket_index": 0, "radius": 120.0,  "change_lane_num": 0, "decrease_increase": 0},
    {"block_id": "s_main1", "id": "S", "parent_block_id": "x0",        "parent_socket_index": 0, "length": 300.0},
    {"block_id": "c2",      "id": "C", "parent_block_id": "s_main1",   "parent_socket_index": 0, "length": 200.0, "radius": 110.0,  "angle": 90.0,  "dir": 1},
    {"block_id": "g1",      "id": "g", "parent_block_id": "c2",        "parent_socket_index": 0, "length": 200.0, "extension_length": 160.0},
    {"block_id": "c3",      "id": "C", "parent_block_id": "g1",        "parent_socket_index": 0, "length": 200.0, "radius": 70.0,  "angle": 30.0,  "dir": 1},
    {"block_id": "merge0",  "id": "y", "parent_block_id": "c3",        "parent_socket_index": 0, "length": 200.0, "lane_num": 2},
    {"block_id": "s_main2", "id": "S", "parent_block_id": "merge0",    "parent_socket_index": 0, "length": 40.0},
    {"block_id": "split0",  "id": "Y", "parent_block_id": "s_main2",   "parent_socket_index": 0, "length": 400.0, "lane_num": 2},
    {"block_id": "c4",      "id": "C", "parent_block_id": "split0",    "parent_socket_index": 0, "length": 300.0, "radius": 90.0,  "angle": 60.0,  "dir": 1},
    {"block_id": "s_ramp0",   "id": "s", "parent_block_id": "g0",        "parent_socket_index": 1, "length": 120.0},
    {"block_id": "c0_ramp0",  "id": "c", "parent_block_id": "s_ramp0",   "parent_socket_index": 0, "length": 100.0, "radius": 100.0,  "angle": 50.0,  "dir": 1},
    {"block_id": "s_ramp1",   "id": "s", "parent_block_id": "c0_ramp0",  "parent_socket_index": 0, "length": 120.0},
    {"block_id": "c1_ramp0",  "id": "c", "parent_block_id": "s_ramp1",   "parent_socket_index": 0, "length": 100.0, "radius": 160.0,  "angle": 130.0, "dir": 1},
    {
        "block_id": "h_ramp0",
        "id": "H",
        "parent_block_id": "c1_ramp0",
        "parent_socket_index": 0,
        "secondary_parent_block_id": "g1",
        "secondary_parent_socket_index": 1,
    },
]
