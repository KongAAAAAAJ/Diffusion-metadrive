"""Scenario definitions for route-conditioned data collection and training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from routes.route_definitions import get_required_preset
from scenarios.S6_inject_vehicles import build_s6_traffic_recipes
from scenarios.S7_inject_vehicles import build_s7_traffic_recipes


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
    # 若 True，数据采集时将 episode 帧裁剪到换道事件附近（换道前 window_before 帧 + 换道后 window_after 帧）
    trim_to_lane_change: bool = False
    trim_window_before: int = 60   # frames before lane-index change
    trim_window_after: int = 60    # frames after lane-index change completes
    # 若设置，覆盖全局采样的 traffic_density（用于换道场景等需要稀疏交通的场合）
    override_traffic_density: float | None = None
    # Scenario-specific MetaDrive env config overrides.
    env_overrides: Dict[str, object] | None = None
    # Ego platoon fixed-route spawn controls.
    # ego_spawn_longitude_m is agent0's longitudinal coordinate from the selected road start.
    ego_spawn_reference_block_id: str | None = None
    ego_spawn_reference_kind: str | None = None
    ego_spawn_internal_road_index: int | None = None
    ego_spawn_longitude_m: float | Tuple[float, float] | None = None
    ego_spawn_lane_id: int | None = None
    # Backward-compatible alternative: distance from selected road end.
    ego_spawn_distance_to_route_end_m: float | None = None
    ego_initial_speed_km_h: float | Tuple[float, float] | None = None

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
            # "R1_entry_straight": TriggerSpec("s0", 60.0, 220.0),
            "R3_mainline_straight": TriggerSpec("s_main0", 25.0, 170.0),
            # "R3_post_transition_straight": TriggerSpec("s_main1", 20.0, 120.0),
        },
        traffic_recipes=tuple(),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        ego_initial_speed_km_h=30.0,
        expert_recipe="平稳巡航",
        description="直道自由巡航",
    ),
    ScenarioDefinition(
        code="S2",
        scenario_id="S2_free_cruise_curve",
        allowed_local_routes=("R2_entry_curve",),
        trigger_by_local_route={
            "R2_entry_curve": TriggerSpec("c0", 10.0, 120.0),
            "R5_ramp_curve": TriggerSpec("c0_ramp0", 5.0, 85.0),
            "R9_post_split_curve": TriggerSpec("c4", 10.0, 120.0),
        },
        traffic_recipes=tuple(),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        ego_initial_speed_km_h=25.0,
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
        ego_initial_speed_km_h=26.0,
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
        ego_initial_speed_km_h=22.0,
        expert_recipe="偏保守跟驰",
        description="弯道跟驰",
    ),
    ScenarioDefinition(
        code="S5",
        scenario_id="S5_hard_brake_lead",
        allowed_local_routes=("R1_entry_straight",),
        trigger_by_local_route={
            "R1_entry_straight": TriggerSpec("s0", 50.0, 220.0),
        },
        traffic_recipes=(
            RecipeSpec(
                "hard_brake_lead",
                {
                    "trigger_after_s": 3.0,
                    "lead_bumper_gap_range_m": (9.0, 13.0),
                    "lead_target_speed_range_kmh": (22.0, 26.0),
                    "brake_target_speed_kmh": 1.0,
                    "brake_target_speed_range_kmh": (0.5, 2.0),
                    "brake_deceleration_range_mps2": (5.0, 7.0),
                },
            ),
            RecipeSpec(
                "inject_adjacent_lane_vehicles",
                {
                    "trigger_on_start": True,
                    "clearance_scope": "same_lane",
                    "vehicles": (
                        {
                            "name": "left_side",
                            "lane_side": "left",
                            "spawn_longitude_offset_range_m": (-15.0, -13.0),
                            "target_speed_kmh": 24.0,
                        },
                        {
                            "name": "right_side",
                            "lane_side": "right",
                            "spawn_longitude_offset_range_m": (13.0, 15.0),
                            "target_speed_kmh": 24.0,
                        },
                    ),
                },
            ),
        ),
        ego_spawn_lane_preference="middle",
        ego_spawn_lane_probabilities=None,
        ego_initial_speed_km_h=24.0,
        override_traffic_density=0.0,
        expert_recipe="提高安全时距",
        description="前车急减速",
    ),
    ScenarioDefinition(
        code="S6",
        scenario_id="S6_background_merge_in",
        allowed_local_routes=("R6_mainline_merge_approach",),
        trigger_by_local_route={
            "R6_mainline_merge_approach": TriggerSpec("g1", 10.0, 95.0),
        },
        traffic_recipes=build_s6_traffic_recipes(RecipeSpec),
        ego_spawn_lane_preference="rightmost",
        ego_spawn_lane_probabilities=None,
        ego_spawn_reference_block_id="g1",
        ego_spawn_reference_kind="block_internal_road",
        ego_spawn_internal_road_index=1,
        ego_spawn_longitude_m=(72.0, 78.0),
        ego_initial_speed_km_h=(23.0, 25.0),
        override_traffic_density=0.0,
        env_overrides={
            "traffic_spawn_exclusion_ahead_m": 100.0,
            "traffic_spawn_exclusion_behind_m": 100.0,
        },
        expert_recipe="保守让行",
        description="背景车并入 ego 所在主线",
    ),
    ScenarioDefinition(
        code="S7", 
        scenario_id="S7_ego_merge_from_ramp",
        allowed_local_routes=("R7_merge_core",),
        trigger_by_local_route={
            "R7_merge_core": TriggerSpec("h_ramp0", 5.0, 60.0),  # h_ramp0
        },
        traffic_recipes=build_s7_traffic_recipes(RecipeSpec),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        ego_spawn_longitude_m=(5.0, 6.0),  # *
        ego_spawn_reference_block_id="h_ramp0",
        ego_initial_speed_km_h=(22, 23),
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
                    "reference_kind": "block_internal_road",
                    "block_id": "g0",
                    "internal_road_index": 0,
                    "lane_index": 2,
                    "spawn_longitude": 20.0,
                    "target_speed_kmh": 18.0,
                },
            ),
            RecipeSpec(
                "inject_background_vehicle",
                {
                    "reference_kind": "block_internal_road",
                    "block_id": "g0",
                    "internal_road_index": 0,
                    "lane_index": 2,
                    "spawn_longitude": 50.0,
                    "target_speed_kmh": 18.0,
                },
            ),
        ),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        ego_spawn_reference_block_id="g0",
        ego_spawn_longitude_m=30,
        ego_spawn_lane_id=1,
        ego_initial_speed_km_h=24.0,
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
        ego_spawn_reference_block_id="c3",
        ego_spawn_reference_kind="block_internal_road",
        ego_spawn_internal_road_index=1,
        ego_spawn_longitude_m=10.0,
        ego_initial_speed_km_h=18.0,
        expert_recipe="保守通过",
        description="合流-分流窄通道博弈",
    ),
    ScenarioDefinition(
        code="S10",
        scenario_id="S10_straight_lane_change",
        allowed_local_routes=("R1_entry_straight", "R3_mainline_straight", "R3_post_transition_straight"),
        trigger_by_local_route={
            "R1_entry_straight":             TriggerSpec("s0",      80.0, 180.0),
            "R3_mainline_straight":          TriggerSpec("s_main0", 30.0, 130.0),
            "R3_post_transition_straight":   TriggerSpec("s_main1", 20.0,  90.0),
        },
        traffic_recipes=(
            RecipeSpec(
                "ensure_lead_vehicle",
                {
                    "distance_m": 12.0,
                    "target_speed_kmh": 8.0,
                },
            ),
        ),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        ego_initial_speed_km_h=24.0,
        expert_recipe="激进换道",
        description="直道前方低速车插入，ego 主动换道绕行",
        trim_to_lane_change=True,
        trim_window_before=60,
        trim_window_after=60,
        override_traffic_density=0.03,
    ),
    ScenarioDefinition(
        code="S11",
        scenario_id="S11_curve_lane_change",
        allowed_local_routes=("R2_entry_curve",),
        trigger_by_local_route={
            "R2_entry_curve": TriggerSpec("c0", 15.0, 100.0),
        },
        traffic_recipes=(
            RecipeSpec(
                "ensure_lead_vehicle",
                {
                    "distance_m": 12.0,
                    "target_speed_kmh": 8.0,
                },
            ),
        ),
        ego_spawn_lane_preference=None,
        ego_spawn_lane_probabilities=None,
        ego_initial_speed_km_h=20.0,
        expert_recipe="弯道换道",
        description="弯道前方低速车插入，ego 主动换道绕行",
        trim_to_lane_change=True,
        trim_window_before=60,
        trim_window_after=60,
        override_traffic_density=0.03,
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
    "S10_straight_lane_change": {
        "enable_lane_change": True,
        "lane_change_freq": 15,
        "safe_lane_change_distance": 10.0,
        "time_wanted": 1.0,
    },
    "S11_curve_lane_change": {
        "enable_lane_change": True,
        "lane_change_freq": 15,
        "safe_lane_change_distance": 10.0,
        "normal_speed_kmh": 22.0,
        "time_wanted": 1.0,
    },
}
SCENARIO_TO_ROUTES: Dict[str, Tuple[str, ...]] = {
    scenario.scenario_id: scenario.allowed_local_routes for scenario in SCENARIO_DEFINITIONS
}
ROUTE_TO_SCENARIOS: Dict[str, Tuple[str, ...]] = {}
for _scenario in SCENARIO_DEFINITIONS:
    for _route_name in _scenario.allowed_local_routes:
        ROUTE_TO_SCENARIOS.setdefault(_route_name, tuple())
        ROUTE_TO_SCENARIOS[_route_name] = ROUTE_TO_SCENARIOS[_route_name] + (_scenario.scenario_id,)


def get_scenario_definition(scenario_id: str) -> ScenarioDefinition:
    return SCENARIO_BY_ID[scenario_id]
