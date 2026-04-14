"""Local route segment definitions for per-episode data collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class RouteDefinition:
    name: str
    blocks: Tuple[str, ...]
    family: str
    required_preset: str
    description: str


ROUTE_DEFINITIONS: Tuple[RouteDefinition, ...] = (
    RouteDefinition("R1_entry_straight", ("s0",), "straight", "mainline", "入口直道段"),
    RouteDefinition("R2_entry_curve", ("c0", "c1"), "curve", "mainline", "入口弯道段"),
    RouteDefinition("R3_mainline_straight", ("s_main0",), "straight", "mainline", "主线前段直道"),
    RouteDefinition("R3_post_transition_straight", ("s_main1",), "straight", "mainline", "主线后段直道"),
    RouteDefinition("R4_mainline_transition", ("x0",), "transition", "mainline", "主线右偏过渡段"),
    RouteDefinition("R5_ramp_curve", ("s_ramp0", "c0_ramp0", "s_ramp1", "c1_ramp0"), "curve", "ramp_merge", "匝道弯道段"),
    RouteDefinition("R6_mainline_merge_approach", ("c2", "g1", "c3"), "merge", "mainline", "主线并入干扰观察段"),
    RouteDefinition("R6_exit_to_ramp", ("g0", "s_ramp0", "c0_ramp0"), "exit", "ramp_merge", "主线驶离/汇出段"),
    RouteDefinition("R7_merge_core", ("h_ramp0", "g1", "c3"), "merge", "ramp_merge", "匝道汇入主线核心段"),
    RouteDefinition("R8_narrow_channel", ("c3", "merge0", "s_main2", "split0"), "constrained", "mainline", "合流-分流受限通道段"),
    RouteDefinition("R9_post_split_curve", ("c4",), "curve", "mainline", "分流后曲线段"),
)

ROUTE_BY_NAME: Dict[str, RouteDefinition] = {route.name: route for route in ROUTE_DEFINITIONS}
DEFAULT_LOCAL_ROUTE_WEIGHTS: Dict[str, float] = {route.name: 1.0 for route in ROUTE_DEFINITIONS}


def get_route_definition(route_name: str) -> RouteDefinition:
    return ROUTE_BY_NAME[route_name]


def get_route_blocks(route_name: str) -> Tuple[str, ...]:
    return get_route_definition(route_name).blocks


def get_required_preset(route_name: str) -> str:
    return get_route_definition(route_name).required_preset
