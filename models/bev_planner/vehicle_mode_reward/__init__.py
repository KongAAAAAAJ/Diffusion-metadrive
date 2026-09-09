"""Vehicle-mode counterfactual reward package.

This package contains only the active GRPO reward implementation.
The legacy three-vehicle joint reward implementation has been removed.
"""

from models.platoon_planner.collision_geometry import shared_corridor_gap_series

from .config import (
    JointRewardError,
    VehicleModeRewardConfig,
    vehicle_mode_reward_config_sha256,
)
from .contracts import (
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT,
    GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256,
    VEHICLE_MODE_REWARD_CONTRACT,
    VEHICLE_MODE_REWARD_CONTRACT_SHA256,
)
from .counterfactual import VehicleModeCounterfactualReward
from .geometry import (
    drivable_signed_distance_m,
    footprint_outside_drivable_series,
    footprint_road_margin_series,
    tracking_aware_half_extents,
)
from .results import (
    RewardGeometryContext,
    VehicleModePretrainRewardResult,
    VehicleModeRewardResult,
)
from .risk import (
    aggregate_temporal_risk,
    closing_ttc_from_gap_series,
    soft_threshold_risk,
    ttc_risk_from_gap_series,
)

__all__ = [
    "GRPO_OPEN_REWARD_APPLICATION_CONTRACT",
    "GRPO_OPEN_REWARD_APPLICATION_CONTRACT_SHA256",
    "VEHICLE_MODE_REWARD_CONTRACT",
    "VEHICLE_MODE_REWARD_CONTRACT_SHA256",
    "JointRewardError",
    "RewardGeometryContext",
    "VehicleModeCounterfactualReward",
    "VehicleModePretrainRewardResult",
    "VehicleModeRewardConfig",
    "VehicleModeRewardResult",
    "aggregate_temporal_risk",
    "closing_ttc_from_gap_series",
    "drivable_signed_distance_m",
    "footprint_outside_drivable_series",
    "footprint_road_margin_series",
    "shared_corridor_gap_series",
    "soft_threshold_risk",
    "tracking_aware_half_extents",
    "ttc_risk_from_gap_series",
    "vehicle_mode_reward_config_sha256",
]
