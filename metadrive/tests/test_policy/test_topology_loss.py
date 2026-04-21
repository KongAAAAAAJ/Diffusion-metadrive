from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

import metadrive.policy.diffusion_policy.transfuser_loss as transfuser_loss_module
from metadrive.policy.diffusion_policy.preprocess_transfuser_dataset import PROCESSED_FIELDS
from metadrive.policy.diffusion_policy.transfuser_config import build_transfuser_config
from metadrive.policy.diffusion_policy.transfuser_features import (
    PROCESSED_DIR_FIELDS,
    processed_sample_to_features_targets,
    sample_to_features_targets,
)
from metadrive.policy.diffusion_policy.transfuser_loss import (
    _topology_consistency_loss,
    transfuser_loss,
)
from metadrive.policy.diffusion_policy.transfuser_model_v2 import TrajectoryHead


def _raw_sample() -> dict:
    return {
        "left_camera": np.zeros((10, 16, 3), dtype=np.uint8),
        "front_camera": np.zeros((10, 16, 3), dtype=np.uint8),
        "right_camera": np.zeros((10, 16, 3), dtype=np.uint8),
        "lidar": np.zeros((32,), dtype=np.float32),
        "ego_state": np.zeros((19,), dtype=np.float32),
        "trajectory": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [7.0, 0.0, 0.0],
                [8.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "agent_states": np.zeros((16, 5), dtype=np.float32),
        "agent_labels": np.zeros((16,), dtype=bool),
        "bev_raster": np.zeros((3, 16, 16), dtype=np.uint8),
        "reference_pose_world": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        "future_reference_pose_world": np.asarray(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [6.0, 0.0, 0.0],
                [7.0, 0.0, 0.0],
                [8.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "reference_lane_index": np.asarray(0, dtype=np.int16),
        "future_reference_lane_index": np.asarray([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int16),
        "ego_speed_km_h": np.asarray(10.0, dtype=np.float32),
        "current_lane_polyline": np.asarray(
            [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0]],
            dtype=np.float32,
        ),
        "left_lane_polyline": np.asarray(
            [[0.0, 3.5], [2.0, 3.5], [4.0, 3.5], [6.0, 3.5], [8.0, 3.5]],
            dtype=np.float32,
        ),
        "right_lane_polyline": np.asarray(
            [[0.0, -3.5], [2.0, -3.5], [4.0, -3.5], [6.0, -3.5], [8.0, -3.5]],
            dtype=np.float32,
        ),
    }


def _processed_sample() -> dict:
    return {
        "camera_feature": np.zeros((3, 4, 4), dtype=np.float32),
        "lidar_feature": np.zeros((1, 4, 4), dtype=np.float32),
        "status_feature": np.zeros((19,), dtype=np.float32),
        "ego_state": np.zeros((19,), dtype=np.float32),
        "target_point": np.zeros((2,), dtype=np.float32),
        "trajectory": np.zeros((8, 3), dtype=np.float32),
        "agent_states": np.zeros((16, 5), dtype=np.float32),
        "agent_labels": np.zeros((16,), dtype=bool),
        "bev_semantic_map": np.zeros((4, 4), dtype=np.uint8),
        "topology_polyline": np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
    }


def test_sample_to_features_targets_adds_topology_polyline_for_keep():
    config = build_transfuser_config("small")
    sample = _raw_sample()

    _, targets = sample_to_features_targets(sample, config, prev_reference_lane_index=0)

    assert "topology_polyline" in targets
    assert tuple(targets["topology_polyline"].shape) == (5, 2)
    assert targets["topology_polyline"].dtype == torch.float32
    np.testing.assert_allclose(
        targets["topology_polyline"].numpy(),
        sample["current_lane_polyline"],
    )


def test_sample_to_features_targets_prefers_adjacent_topology_polyline_with_safe_fallback():
    config = build_transfuser_config("small")
    sample = _raw_sample()

    _, left_targets = sample_to_features_targets(sample, config, prev_reference_lane_index=1)
    np.testing.assert_allclose(left_targets["topology_polyline"].numpy(), sample["left_lane_polyline"])

    sample_without_right = _raw_sample()
    sample_without_right.pop("right_lane_polyline")
    sample_without_right["reference_lane_index"] = np.asarray(1, dtype=np.int16)
    _, fallback_targets = sample_to_features_targets(sample_without_right, config, prev_reference_lane_index=0)
    np.testing.assert_allclose(
        fallback_targets["topology_polyline"].numpy(),
        sample_without_right["current_lane_polyline"],
    )


def test_processed_sample_to_features_targets_reads_topology_polyline():
    features, targets = processed_sample_to_features_targets(_processed_sample())

    assert "topology_polyline" in targets
    assert "target_point" in features
    np.testing.assert_allclose(
        targets["topology_polyline"].numpy(),
        np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32),
    )


def test_topology_processed_field_lists_include_topology_polyline():
    assert "topology_polyline" in PROCESSED_DIR_FIELDS
    assert "topology_polyline" in PROCESSED_FIELDS


def test_topology_consistency_loss_is_near_zero_for_aligned_straight_trajectory():
    config = build_transfuser_config("small", topology_weight=1.0, lane_direction_weight=1.0, corridor_weight=1.0)
    predictions = {
        "trajectory": torch.tensor(
            [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
    }
    targets = {
        "topology_polyline": torch.tensor(
            [[[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0]]],
            dtype=torch.float32,
        )
    }

    topology_loss, lane_direction_loss, corridor_loss = _topology_consistency_loss(predictions, targets, config)

    assert topology_loss.item() == pytest.approx(0.0, abs=1e-5)
    assert lane_direction_loss.item() == pytest.approx(0.0, abs=1e-5)
    assert corridor_loss.item() == pytest.approx(0.0, abs=1e-5)


def test_topology_consistency_loss_penalizes_corridor_violation():
    config = build_transfuser_config(
        "small",
        topology_weight=1.0,
        lane_direction_weight=1.0,
        corridor_weight=1.0,
        corridor_half_width_m=0.5,
    )
    predictions = {
        "trajectory": torch.tensor(
            [[[0.0, 2.0, 0.0], [2.0, 2.0, 0.0], [4.0, 2.0, 0.0], [6.0, 2.0, 0.0]]],
            dtype=torch.float32,
        )
    }
    targets = {
        "topology_polyline": torch.tensor(
            [[[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0]]],
            dtype=torch.float32,
        )
    }

    topology_loss, lane_direction_loss, corridor_loss = _topology_consistency_loss(predictions, targets, config)

    assert topology_loss.item() > 0.0
    assert corridor_loss.item() > 0.0
    assert lane_direction_loss.item() == pytest.approx(0.0, abs=1e-5)


def test_topology_consistency_loss_prefers_hierarchical_mode_candidate():
    config = build_transfuser_config(
        "small",
        topology_weight=1.0,
        lane_direction_weight=1.0,
        corridor_weight=1.0,
        corridor_half_width_m=0.5,
    )
    predictions = {
        "trajectory": torch.tensor(
            [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        "trajectory_candidates_train": torch.tensor(
            [[
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
                [[0.0, 3.0, 0.0], [2.0, 3.0, 0.0], [4.0, 3.0, 0.0], [6.0, 3.0, 0.0]],
            ]],
            dtype=torch.float32,
        ),
        "trajectory_mode_logits_train": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
    }
    targets = {
        "hierarchical_mode_label": torch.tensor([1], dtype=torch.int64),
        "topology_polyline": torch.tensor(
            [[[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0]]],
            dtype=torch.float32,
        ),
    }

    topology_loss, _, corridor_loss = _topology_consistency_loss(predictions, targets, config)

    assert topology_loss.item() > 0.0
    assert corridor_loss.item() > 0.0


def test_topology_consistency_loss_returns_zero_without_topology_polyline():
    config = build_transfuser_config("small")
    predictions = {
        "trajectory": torch.tensor(
            [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
    }
    targets = {}

    topology_loss, lane_direction_loss, corridor_loss = _topology_consistency_loss(predictions, targets, config)

    assert topology_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert lane_direction_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert corridor_loss.item() == pytest.approx(0.0, abs=1e-6)


def test_transfuser_loss_reports_topology_terms(monkeypatch):
    config = build_transfuser_config(
        "small",
        topology_weight=1.0,
        lane_direction_weight=1.0,
        corridor_weight=1.0,
        corridor_half_width_m=0.5,
    )
    monkeypatch.setattr(transfuser_loss_module, "_agent_loss", lambda targets, predictions, config: (torch.tensor(0.0), torch.tensor(0.0)))
    targets = {
        "trajectory": torch.tensor(
            [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0], [6.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        "topology_polyline": torch.tensor(
            [[[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0]]],
            dtype=torch.float32,
        ),
        "bev_semantic_map": torch.zeros((1, 2, 2), dtype=torch.long),
        "agent_states": torch.zeros((1, 1, 5), dtype=torch.float32),
        "agent_labels": torch.zeros((1, 1), dtype=torch.bool),
    }
    predictions = {
        "trajectory": torch.tensor(
            [[[0.0, 1.0, 0.0], [2.0, 1.0, 0.0], [4.0, 1.0, 0.0], [6.0, 1.0, 0.0]]],
            dtype=torch.float32,
        ),
        "trajectory_candidates_train": torch.tensor(
            [[[
                [0.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [4.0, 1.0, 0.0],
                [6.0, 1.0, 0.0],
            ]]],
            dtype=torch.float32,
        ),
        "trajectory_mode_logits_train": torch.tensor([[1.0]], dtype=torch.float32),
        "bev_semantic_map": torch.zeros((1, config.num_bev_classes, 2, 2), dtype=torch.float32),
        "agent_states": torch.zeros((1, 1, 5), dtype=torch.float32),
        "agent_labels": torch.zeros((1, 1), dtype=torch.float32),
        "trajectory_loss": torch.tensor(0.0),
    }

    loss_dict = transfuser_loss(targets, predictions, config)

    assert "topology_loss" in loss_dict
    assert "lane_direction_loss" in loss_dict
    assert "corridor_loss" in loss_dict
    assert loss_dict["loss"].item() >= loss_dict["topology_loss"].item()


def test_trajectory_head_forward_train_exposes_train_candidates(monkeypatch, tmp_path):
    class _ZeroLossComputer(nn.Module):
        def forward(self, *args, **kwargs):
            return torch.tensor(0.0)

    config = build_transfuser_config(
        "small",
        use_dynamic_anchors=True,
        plan_anchor_path=str(tmp_path / "missing_anchors.npy"),
    )
    head = TrajectoryHead(
        num_poses=config.trajectory_sampling.num_poses,
        d_ffn=config.tf_d_ffn,
        d_model=config.tf_d_model,
        plan_anchor_path=config.plan_anchor_path,
        config=config,
    )
    monkeypatch.setattr(head, "loss_computer", _ZeroLossComputer())
    bs = 2
    ego_query = torch.zeros((bs, 1, config.tf_d_model), dtype=torch.float32)
    agents_query = torch.zeros((bs, config.num_bounding_boxes, config.tf_d_model), dtype=torch.float32)
    bev_feature = torch.zeros((bs, config.tf_d_model, 8, 8), dtype=torch.float32)
    bev_spatial_shape = (8, 8)
    status_encoding = torch.zeros((bs, 1, config.tf_d_model), dtype=torch.float32)
    target_point_embed = torch.zeros((bs, config.target_point_dim), dtype=torch.float32)
    coarse_trajectories = torch.zeros(
        (bs, config.ego_fut_mode, config.trajectory_sampling.num_poses, 2),
        dtype=torch.float32,
    )
    mode_valid_mask = torch.ones((bs, config.ego_fut_mode), dtype=torch.bool)
    targets = {
        "trajectory": torch.zeros((bs, config.trajectory_sampling.num_poses, 3), dtype=torch.float32),
    }

    outputs = head.forward_train(
        ego_query,
        agents_query,
        bev_feature,
        bev_spatial_shape,
        status_encoding,
        targets=targets,
        target_point_embed=target_point_embed,
        coarse_trajectories=coarse_trajectories,
        mode_valid_mask=mode_valid_mask,
    )

    assert "trajectory_candidates_train" in outputs
    assert "trajectory_mode_logits_train" in outputs
    assert tuple(outputs["trajectory_candidates_train"].shape) == (
        bs,
        config.ego_fut_mode,
        config.trajectory_sampling.num_poses,
        3,
    )
    assert tuple(outputs["trajectory_mode_logits_train"].shape) == (bs, config.ego_fut_mode)
