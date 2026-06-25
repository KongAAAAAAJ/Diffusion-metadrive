"""Project-level platoon evaluation utilities and entrypoints."""

from evaluation.platoon_metrics import PlatoonMetrics
from evaluation.platoon_performance import (
    build_pdms_params,
    build_platoon_metric_params,
    compute_pairwise_formation_reward,
    compute_pdms_reward_batch,
)

__all__ = [
    "PlatoonMetrics",
    "build_pdms_params",
    "build_platoon_metric_params",
    "compute_pairwise_formation_reward",
    "compute_pdms_reward_batch",
]
