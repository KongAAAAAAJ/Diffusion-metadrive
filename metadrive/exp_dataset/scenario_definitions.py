"""Scenario definitions for Phase 2 route-conditioned data collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from metadrive.exp_dataset.route_definitions import get_required_preset


@dataclass(frozen=True)
class TriggerSpec:
    block_id: str
    longitudinal_min: float
    longitudinal_max: float


@dataclass(frozen=True)
class RecipeSpec:
    operation: str
    params: Dict[str, object]


@dataclass(frozen=True)
class ScenarioDefinition:
    code: str
    scenario_id: str
    allowed_local_routes: Tuple[str, ...]
    trigger_by_local_route: Dict[str, TriggerSpec]
    traffic_recipes: Tuple[RecipeSpec, ...]
    ego_spawn_lane_preference: str | None
    ego_spawn_lane_probabilities: Dict[str, float] | None
    expert_recipe: str
    description: str

    @property
    def allowed_route_presets(self) -> Tuple[str, ...]:
        return tuple(sorted({get_required_preset(route_name) for route_name in self.allowed_local_routes}))

    def get_trigger_spec(self, local_route: str) -> TriggerSpec:
        return self.trigger_by_local_route[local_route]


SCENARIO_DEFINITIONS: Tuple[ScenarioDefinition, ...] = (
    ScenarioDefinition(
        code="S1",
        scenario_id="S1_free_cruise_straight",
        allowed_local_routes=("R1_entry_straight", "R3_mainline_straight", "R3_post_transition_straight"),
        trigger_by_local_route={
            "R1_entry_straight": TriggerSpec("s0", 60.0, 220.0),
            "R3_mainline_straight": TriggerSpec("s_main0", 25.0, 170.0),
            "R3_post_transition_straight": TriggerSpec("s_main1", 20.0, 120.0),
        },
        traffic_recipes=tuple(),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        expert_recipe="平稳巡航",
        description="直道自由巡航",
    ),
    ScenarioDefinition(
        code="S2",
        scenario_id="S2_free_cruise_curve",
        allowed_local_routes=("R2_entry_curve", "R5_ramp_curve", "R9_post_split_curve"),
        trigger_by_local_route={
            "R2_entry_curve": TriggerSpec("c0", 10.0, 120.0),
            "R5_ramp_curve": TriggerSpec("c0_ramp0", 5.0, 85.0),
            "R9_post_split_curve": TriggerSpec("c4", 10.0, 120.0),
        },
        traffic_recipes=tuple(),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        expert_recipe="稍保守速度",
        description="弯道自由巡航",
    ),
    ScenarioDefinition(
        code="S3",
        scenario_id="S3_straight_following",
        allowed_local_routes=("R1_entry_straight", "R3_mainline_straight", "R3_post_transition_straight"),
        trigger_by_local_route={
            "R1_entry_straight": TriggerSpec("s0", 50.0, 210.0),
            "R3_mainline_straight": TriggerSpec("s_main0", 20.0, 150.0),
            "R3_post_transition_straight": TriggerSpec("s_main1", 15.0, 110.0),
        },
        traffic_recipes=(
            RecipeSpec("ensure_lead_vehicle", {"distance_m": 20.0, "target_speed_kmh": 24.0}),
        ),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        expert_recipe="偏纵向跟驰",
        description="直道跟驰",
    ),
    ScenarioDefinition(
        code="S4",
        scenario_id="S4_curve_following",
        allowed_local_routes=("R2_entry_curve", "R5_ramp_curve"),
        trigger_by_local_route={
            "R2_entry_curve": TriggerSpec("c0", 10.0, 110.0),
            "R5_ramp_curve": TriggerSpec("c0_ramp0", 5.0, 80.0),
        },
        traffic_recipes=(
            RecipeSpec("ensure_lead_vehicle", {"distance_m": 16.0, "target_speed_kmh": 20.0}),
        ),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        expert_recipe="偏保守跟驰",
        description="弯道跟驰",
    ),
    ScenarioDefinition(
        code="S5",
        scenario_id="S5_hard_brake_lead",  
        allowed_local_routes=("R1_entry_straight", "R3_mainline_straight", "R3_post_transition_straight"),
        trigger_by_local_route={
            "R1_entry_straight": TriggerSpec("s0", 80.0, 220.0),
            "R3_mainline_straight": TriggerSpec("s_main0", 60.0, 160.0),
            "R3_post_transition_straight": TriggerSpec("s_main1", 60.0, 115.0),
        },
        traffic_recipes=(
            RecipeSpec(
                "hard_brake_lead",
                {
                    "lead_distance_m": 10.0,
                    "lead_target_speed_kmh": 21.0,
                    "front_distance_min_m": 10.0,
                    "front_distance_max_m": 24.0,
                    "brake_target_speed_kmh": 1.0,
                    "brake_duration_steps": 50,
                },
            ),
        ),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        expert_recipe="提高安全时距",
        description="前车急减速",
    ),
    ScenarioDefinition(
        code="S6",
        scenario_id="S6_background_merge_in",  # !ego车只是固定出生在靠近匝道的车道，后面行驶时是可以自由换道的
        allowed_local_routes=("R6_mainline_merge_approach",),
        trigger_by_local_route={
            "R6_mainline_merge_approach": TriggerSpec("c2", 10.0, 95.0),
        },
        traffic_recipes=(
            RecipeSpec(
                "inject_background_vehicle",
                {
                    "reference_kind": "block_socket_road",
                    "block_id": "g1",
                    "socket_index": 1,
                    "spawn_longitude": 12.0,
                    "target_speed_kmh": 24.0,
                },
            ),
        ),
        ego_spawn_lane_preference="rightmost",
        ego_spawn_lane_probabilities=None,
        expert_recipe="保守让行",
        description="背景车并入 ego 所在主线",
    ),
    ScenarioDefinition(
        code="S7",
        scenario_id="S7_ego_merge_from_ramp",
        allowed_local_routes=("R7_merge_core",),
        trigger_by_local_route={
            "R7_merge_core": TriggerSpec("h_ramp0", 5.0, 60.0),
        },
        traffic_recipes=(
            RecipeSpec(
                "inject_background_vehicle",
                {
                    "reference_kind": "block_route_road",
                    "block_id": "g1",
                    "spawn_longitude": 18.0,
                    "target_speed_kmh": 25.0,
                },
            ),
        ),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        expert_recipe="汇入博弈",
        description="ego 从匝道汇入主线",
    ),
    ScenarioDefinition(
        code="S8",
        scenario_id="S8_ego_exit_to_ramp",
        allowed_local_routes=("R6_exit_to_ramp",),
        trigger_by_local_route={
            "R6_exit_to_ramp": TriggerSpec("g0", 5.0, 45.0),
        },
        traffic_recipes=(
            RecipeSpec(
                "inject_background_vehicle",
                {
                    "reference_kind": "block_socket_road",
                    "block_id": "g0",
                    "socket_index": 1,
                    "spawn_longitude": 30.0,
                    "target_speed_kmh": 18.0,
                },
            ),
        ),
        ego_spawn_lane_preference="rightmost",
        ego_spawn_lane_probabilities=None,
        expert_recipe="提前换道驶离",
        description="ego 从主线驶出",
    ),
    ScenarioDefinition(
        code="S9", 
        scenario_id="S9_narrow_channel_negotiation",
        allowed_local_routes=("R8_narrow_channel",),
        trigger_by_local_route={
            "R8_narrow_channel": TriggerSpec("merge0", 10.0, 110.0),
        },
        traffic_recipes=(
            RecipeSpec(
                "inject_background_vehicle",
                {
                    "reference_kind": "block_route_road",
                    "block_id": "merge0",
                    "spawn_longitude": 12.0,
                    "target_speed_kmh": 14.0,
                },
            ),
            RecipeSpec(
                "inject_background_vehicle",
                {
                    "reference_kind": "block_route_road",
                    "block_id": "split0",
                    "spawn_longitude": 18.0,
                    "target_speed_kmh": 16.0,
                },
            ),
        ),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities={
            "rightmost": 0.4,
            "middle": 0.4,
            "leftmost": 0.2,
        },
        expert_recipe="保守通过",
        description="合流-分流窄通道博弈",
    ),
)


SCENARIO_BY_ID: Dict[str, ScenarioDefinition] = {
    scenario.scenario_id: scenario for scenario in SCENARIO_DEFINITIONS
}
DEFAULT_SCENARIO_WEIGHTS: Dict[str, float] = {
    scenario.scenario_id: 1.0 for scenario in SCENARIO_DEFINITIONS
}
SCENARIO_EXPERT_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "S1_free_cruise_straight": {},
    "S2_free_cruise_curve": {
        "normal_speed_kmh": 25.0,
    },
    "S3_straight_following": {
        "time_wanted": 1.2,
        "enable_lane_change": False,
    },
    "S4_curve_following": {
        "normal_speed_kmh": 22.0,
        "time_wanted": 1.4,
        "enable_lane_change": False,
    },
    "S5_hard_brake_lead": {
        "time_wanted": 1.5,
        "distance_wanted": 12.0,
    },
    "S6_background_merge_in": {
        "time_wanted": 1.3,
        "enable_lane_change": True,
    },
    "S7_ego_merge_from_ramp": {
        "enable_lane_change": True,
        "lane_change_freq": 20,
        "safe_lane_change_distance": 12.0,
    },
    "S8_ego_exit_to_ramp": {
        "enable_lane_change": True,
        "lane_change_freq": 25,
        "safe_lane_change_distance": 8.0,
        "ignore_continuous_line_check": True,
    },
    "S9_narrow_channel_negotiation": {
        "normal_speed_kmh": 20.0,
        "time_wanted": 1.5,
        "enable_lane_change": True,
    },
}
SCENARIO_TO_ROUTES: Dict[str, Tuple[str, ...]] = {
    scenario.scenario_id: scenario.allowed_local_routes for scenario in SCENARIO_DEFINITIONS
}
ROUTE_TO_SCENARIOS: Dict[str, Tuple[str, ...]] = {}
for scenario in SCENARIO_DEFINITIONS:
    for route_name in scenario.allowed_local_routes:
        ROUTE_TO_SCENARIOS.setdefault(route_name, tuple())
        ROUTE_TO_SCENARIOS[route_name] = ROUTE_TO_SCENARIOS[route_name] + (scenario.scenario_id,)


def get_scenario_definition(scenario_id: str) -> ScenarioDefinition:
    return SCENARIO_BY_ID[scenario_id]
