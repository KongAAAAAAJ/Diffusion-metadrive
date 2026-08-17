from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.validate_bev_stage1_ab import (
    OPEN_LOOP_COMPARISON_POLICY,
    _global_agent_poses,
    compare_open_loop_models,
    parse_args,
)


def _model(*, loss: float, accuracy: float, ade: float) -> dict:
    return {
        "open_loop": {
            "metrics": {
                "loss/total": loss,
                "metric/mode_accuracy": accuracy,
                "metric/gt_mode_ade": ade,
                "metric/gt_mode_fde": ade * 2.0,
                "metric/selected_ade": ade + 0.1,
                "metric/selected_fde": ade * 2.0 + 0.1,
            }
        }
    }


def test_single_model_has_no_comparisons() -> None:
    assert compare_open_loop_models({"only": _model(loss=1.0, accuracy=0.8, ade=0.5)}, ()) == {}


def test_multiple_models_use_frozen_direction_and_tolerance() -> None:
    models = {
        "baseline": _model(loss=1.0, accuracy=0.8, ade=0.5),
        "candidate": _model(loss=0.9, accuracy=0.802, ade=0.495),
    }
    comparisons = (
        SimpleNamespace(
            comparison_id="candidate_vs_baseline",
            baseline="baseline",
            candidate="candidate",
        ),
    )
    result = compare_open_loop_models(models, comparisons)
    metrics = result["candidate_vs_baseline"]["metrics"]
    assert metrics["loss/total"]["conclusion"] == "better"
    assert metrics["metric/mode_accuracy"]["conclusion"] == "better"
    assert metrics["metric/gt_mode_ade"]["conclusion"] == "equivalent"
    assert OPEN_LOOP_COMPARISON_POLICY["metric/gt_mode_ade"] == ("lower", 0.01)


def test_legacy_checkpoint_arguments_are_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "validate_bev_stage1_ab.py",
            "--checkpoint-a",
            "a.pt",
            "--checkpoint-b",
            "b.pt",
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_s1_global_poses_are_read_from_environment_in_joint_role_order() -> None:
    env = SimpleNamespace(
        agents={
            "agent2": SimpleNamespace(position=(30.0, -2.0), heading_theta=0.3),
            "agent0": SimpleNamespace(position=(10.0, 0.0), heading_theta=0.1),
            "agent1": SimpleNamespace(position=(20.0, -1.0), heading_theta=0.2),
        }
    )

    poses = _global_agent_poses(env)

    np.testing.assert_allclose(
        poses,
        np.asarray(
            [[10.0, 0.0, 0.1], [20.0, -1.0, 0.2], [30.0, -2.0, 0.3]]
        ),
    )
