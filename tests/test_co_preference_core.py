import numpy as np
import torch

from models.co_preference.geometry import (
    TOPOLOGY_BRANCH,
    TOPOLOGY_CURRENT,
    TOPOLOGY_LEFT,
    build_topology_mask,
    map_preference_to_target_point,
)
from models.co_preference.model import CoPreferenceModel
from models.co_preference.teacher import make_teacher_label
from evaluation.co_preference_reward import compute_preference_reward
from models.platoon.platoon_diffusion_planner import PlatoonDiffusionPlanner
from metadrive.policy.diffusion_policy.transfuser_config import build_transfuser_config


def test_co_preference_model_outputs_masked_topology_and_unit_s() -> None:
    model = CoPreferenceModel(status_dim=8, relation_dim=12, hidden=(16,))
    status = torch.zeros((3, 8), dtype=torch.float32)
    relation = torch.ones((3, 12), dtype=torch.float32)
    mask = torch.tensor(
        [
            [True, False, True, False],
            [True, True, False, False],
            [True, False, False, True],
        ],
        dtype=torch.bool,
    )

    out = model(status_feature=status, formation_relation_state=relation, topology_mask=mask)

    assert out["topology_logits"].shape == (3, 4)
    assert out["s"].shape == (3,)
    assert torch.all((out["s"] >= 0.0) & (out["s"] <= 1.0))
    assert torch.all(out["topology_logits"][~mask] < -1e8)


def test_geometry_maps_choice_and_s_to_polyline_target_point() -> None:
    polylines = {
        "current": np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32),
        "left": np.asarray([[0.0, 3.5], [20.0, 3.5]], dtype=np.float32),
        "right": None,
        "branch": np.asarray([[0.0, 0.0], [0.0, 10.0]], dtype=np.float32),
    }

    mask = build_topology_mask(polylines)
    point, debug = map_preference_to_target_point(TOPOLOGY_LEFT, 0.25, polylines)

    np.testing.assert_array_equal(mask, np.asarray([True, True, False, True]))
    np.testing.assert_allclose(point, np.asarray([5.0, 3.5], dtype=np.float32), atol=1e-5)
    assert debug["co_preference_choice"] == TOPOLOGY_LEFT
    assert debug["co_preference_s"] == 0.25


def test_teacher_label_projects_target_to_nearest_legal_polyline() -> None:
    polylines = {
        "current": np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32),
        "left": np.asarray([[0.0, 3.5], [10.0, 3.5]], dtype=np.float32),
        "right": None,
        "branch": None,
    }

    label = make_teacher_label(np.asarray([7.5, 3.2], dtype=np.float32), polylines)

    assert label["teacher_topology_choice"] == TOPOLOGY_LEFT
    assert abs(label["teacher_s"] - 0.75) < 1e-5
    np.testing.assert_allclose(label["teacher_target_point"], np.asarray([7.5, 3.5], dtype=np.float32), atol=1e-5)


def test_preference_reward_penalizes_invalid_corridor_reachability_and_jump() -> None:
    reward, terms = compute_preference_reward(
        target_point=np.asarray([12.0, 4.5], dtype=np.float32),
        selected_polyline=np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32),
        previous_target_point=np.asarray([0.0, 0.0], dtype=np.float32),
        reachable_distance_m=8.0,
        corridor_half_width_m=1.75,
        max_target_jump_m=5.0,
    )

    assert reward < 0.0
    assert terms["corridor_violation_m"] > 0.0
    assert terms["reachable_violation_m"] > 0.0
    assert terms["jump_violation_m"] > 0.0


def test_platoon_planner_exposes_preference_path_not_selector_path() -> None:
    planner = PlatoonDiffusionPlanner(build_transfuser_config("small"), num_vehicles=2)

    assert hasattr(planner, "forward_with_preference")
    assert not hasattr(planner, "forward_selector")
    assert not hasattr(planner, "freeze_for_selector")
