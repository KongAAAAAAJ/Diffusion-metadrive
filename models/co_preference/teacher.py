from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from .geometry import project_point_to_best_polyline


def make_teacher_label(
    teacher_target_point: np.ndarray,
    polylines: Mapping[str, Optional[np.ndarray]],
) -> dict[str, object]:
    choice, s, projected, distance = project_point_to_best_polyline(teacher_target_point, polylines)
    return {
        "teacher_topology_choice": int(choice),
        "teacher_s": float(s),
        "teacher_target_point": projected.astype(np.float32),
        "teacher_projection_distance_m": float(distance),
    }
