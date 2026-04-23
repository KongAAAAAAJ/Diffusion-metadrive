from __future__ import annotations

import numpy as np

from models.co_preference.geometry import point_to_polyline_distance


def compute_preference_reward(
    *,
    target_point: np.ndarray,
    selected_polyline: np.ndarray,
    previous_target_point: np.ndarray | None = None,
    reachable_distance_m: float = 30.0,
    corridor_half_width_m: float = 1.75,
    max_target_jump_m: float = 8.0,
    w_corridor: float = 1.0,
    w_reachable: float = 0.5,
    w_jump: float = 0.2,
) -> tuple[float, dict[str, float]]:
    target = np.asarray(target_point, dtype=np.float32)[:2]
    distance, s = point_to_polyline_distance(target, selected_polyline)
    corridor_violation = max(0.0, float(distance) - float(corridor_half_width_m))
    longitudinal_distance = float(np.linalg.norm(target))
    reachable_violation = max(0.0, longitudinal_distance - float(reachable_distance_m))
    jump_violation = 0.0
    if previous_target_point is not None:
        previous = np.asarray(previous_target_point, dtype=np.float32)[:2]
        jump_violation = max(0.0, float(np.linalg.norm(target - previous)) - float(max_target_jump_m))

    reward = -(
        float(w_corridor) * corridor_violation
        + float(w_reachable) * reachable_violation
        + float(w_jump) * jump_violation
    )
    terms = {
        "corridor_violation_m": float(corridor_violation),
        "reachable_violation_m": float(reachable_violation),
        "jump_violation_m": float(jump_violation),
        "target_polyline_s": float(s),
        "preference_reward": float(reward),
    }
    return float(reward), terms
