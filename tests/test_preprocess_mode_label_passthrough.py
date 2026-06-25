import numpy as np
import pytest
import torch

from models.diffusion import preprocess_transfuser_dataset as preprocess
from models.diffusion import transfuser_features as features_mod
from models.diffusion.transfuser_config import TransfuserConfig


def test_preprocess_keeps_recomputed_gt_mode_label_and_preserves_expert_decision(
    tmp_path,
    monkeypatch,
) -> None:
    shard_path = tmp_path / "shard_000000.npz"
    np.savez(
        shard_path,
        trajectory=np.zeros((2, 8, 3), dtype=np.float32),
        gt_mode_label=np.asarray([2, 3], dtype=np.int8),
        expert_lateral_decision=np.asarray([-1, 1], dtype=np.int8),
        reference_lane_index=np.asarray([1, 1], dtype=np.int16),
        future_reference_lane_index=np.asarray([[1, 1, 0, 0], [1, 1, 2, 2]], dtype=np.int16),
        scenario_id=np.asarray(["S1", "S1"]),
    )

    recomputed_labels = [12, 15]

    def fake_sample_to_features_targets(sample, config):
        sample_idx = int(sample["expert_lateral_decision"] > 0)
        features = {
            "camera_feature": torch.zeros((3, 4, 4), dtype=torch.float32),
            "lidar_feature": torch.zeros((2, 4, 4), dtype=torch.float32),
            "status_feature": torch.zeros((8,), dtype=torch.float32),
            "ego_state": torch.zeros((8,), dtype=torch.float32),
            "target_point": torch.zeros((2,), dtype=torch.float32),
            "target_line": torch.zeros((8, 2), dtype=torch.float32),
            "coarse_trajectories": torch.zeros((16, 8, 2), dtype=torch.float32),
            "mode_valid_mask": torch.ones((16,), dtype=torch.bool),
        }
        targets = {
            "trajectory": torch.zeros((8, 3), dtype=torch.float32),
            "topology_polyline": torch.zeros((32, 2), dtype=torch.float32),
            "agent_states": torch.zeros((32, 8), dtype=torch.float32),
            "agent_labels": torch.zeros((32,), dtype=torch.bool),
            "bev_semantic_map": torch.zeros((64, 64), dtype=torch.int64),
            "gt_mode_label": torch.tensor(recomputed_labels[sample_idx], dtype=torch.int8),
        }
        return features, targets

    monkeypatch.setattr(preprocess, "sample_to_features_targets", fake_sample_to_features_targets)

    payload = preprocess.build_processed_payload(shard_path, config=None)

    assert payload["gt_mode_label"].tolist() == recomputed_labels
    assert payload["expert_lateral_decision"].tolist() == [-1, 1]
    assert payload["reference_lane_index"].tolist() == [1, 1]
    assert payload["future_reference_lane_index"].tolist() == [[1, 1, 0, 0], [1, 1, 2, 2]]
    assert payload["scenario_id"].tolist() == ["S1", "S1"]


def test_sample_to_features_targets_requires_expert_lateral_decision(monkeypatch) -> None:
    config = TransfuserConfig()
    sample = {
        "left_camera": np.zeros((4, 4, 3), dtype=np.uint8),
        "front_camera": np.zeros((4, 4, 3), dtype=np.uint8),
        "right_camera": np.zeros((4, 4, 3), dtype=np.uint8),
        "lidar": np.zeros((1, 4), dtype=np.float32),
        "ego_state": np.zeros((8,), dtype=np.float32),
        "agent_states": np.zeros((32, 8), dtype=np.float32),
        "agent_labels": np.zeros((32,), dtype=bool),
        "bev_semantic_map": np.zeros((64, 64), dtype=np.int64),
        "trajectory": np.zeros((8, 3), dtype=np.float32),
        "reference_lane_index": np.asarray(0, dtype=np.int16),
    }

    monkeypatch.setattr(features_mod, "stitch_three_cameras", lambda *args, **kwargs: np.zeros((3, 4, 4), dtype=np.float32))
    monkeypatch.setattr(features_mod, "lidar_to_histogram", lambda *args, **kwargs: torch.zeros((2, 4, 4), dtype=torch.float32))
    monkeypatch.setattr(features_mod, "build_status_feature", lambda *args, **kwargs: torch.zeros((8,), dtype=torch.float32))
    monkeypatch.setattr(features_mod, "compute_target_point_from_sample", lambda *args, **kwargs: torch.zeros((2,), dtype=torch.float32))
    monkeypatch.setattr(features_mod, "_build_topology_polyline_from_sample", lambda *args, **kwargs: torch.zeros((8, 2), dtype=torch.float32))
    monkeypatch.setattr(features_mod, "_build_target_line", lambda *args, **kwargs: torch.zeros((8, 2), dtype=torch.float32))
    monkeypatch.setattr(
        features_mod,
        "_build_mode_features",
        lambda *args, **kwargs: {
            "coarse_trajectories": torch.zeros((10, 8, 2), dtype=torch.float32),
            "mode_valid_mask": torch.ones((10,), dtype=torch.bool),
        },
    )
    monkeypatch.setattr(
        features_mod,
        "normalize_agent_targets",
        lambda *args, **kwargs: (torch.zeros((32, 8), dtype=torch.float32), torch.zeros((32,), dtype=torch.bool)),
    )
    monkeypatch.setattr(features_mod, "semantic_map_to_target", lambda *args, **kwargs: torch.zeros((64, 64), dtype=torch.int64))

    with pytest.raises(ValueError, match="expert_lateral_decision"):
        features_mod.sample_to_features_targets(sample, config)


def test_processed_sample_recomputes_gt_mode_label_from_expert_decision() -> None:
    config = TransfuserConfig(
        mode_keep_lane_count=5,
        mode_lane_change_left_count=5,
        mode_lane_change_right_count=5,
        mode_emergency_stop_count=1,
    )
    sample = {
        "camera_feature": np.zeros((3, 4, 4), dtype=np.float32),
        "lidar_feature": np.zeros((2, 4, 4), dtype=np.float32),
        "status_feature": np.zeros((8,), dtype=np.float32),
        "ego_state": np.zeros((8,), dtype=np.float32),
        "target_point": np.zeros((2,), dtype=np.float32),
        "target_line": np.zeros((8, 2), dtype=np.float32),
        "topology_polyline": np.zeros((8, 2), dtype=np.float32),
        "coarse_trajectories": np.zeros((16, 8, 2), dtype=np.float32),
        "mode_valid_mask": np.ones((16,), dtype=bool),
        "trajectory": np.zeros((8, 3), dtype=np.float32),
        "agent_states": np.zeros((32, 8), dtype=np.float32),
        "agent_labels": np.zeros((32,), dtype=bool),
        "bev_semantic_map": np.zeros((64, 64), dtype=np.int64),
        "expert_lateral_decision": np.asarray(0, dtype=np.int8),
        "gt_mode_label": np.asarray(4, dtype=np.int8),
    }

    _, targets = features_mod.processed_sample_to_features_targets(sample, config)

    assert int(targets["gt_mode_label"]) == 0


def test_sample_to_features_targets_prefers_future_lane_window_over_instant_expert_decision(monkeypatch) -> None:
    config = TransfuserConfig(
        mode_keep_lane_count=5,
        mode_lane_change_left_count=5,
        mode_lane_change_right_count=5,
        mode_emergency_stop_count=1,
    )
    sample = {
        "left_camera": np.zeros((4, 4, 3), dtype=np.uint8),
        "front_camera": np.zeros((4, 4, 3), dtype=np.uint8),
        "right_camera": np.zeros((4, 4, 3), dtype=np.uint8),
        "lidar": np.zeros((1, 4), dtype=np.float32),
        "ego_state": np.zeros((8,), dtype=np.float32),
        "agent_states": np.zeros((32, 8), dtype=np.float32),
        "agent_labels": np.zeros((32,), dtype=bool),
        "bev_semantic_map": np.zeros((64, 64), dtype=np.int64),
        "trajectory": np.zeros((8, 3), dtype=np.float32),
        "reference_lane_index": np.asarray(1, dtype=np.int16),
        "future_reference_lane_index": np.asarray([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int16),
        "expert_lateral_decision": np.asarray(-1, dtype=np.int8),
    }

    monkeypatch.setattr(features_mod, "stitch_three_cameras", lambda *args, **kwargs: np.zeros((3, 4, 4), dtype=np.float32))
    monkeypatch.setattr(features_mod, "lidar_to_histogram", lambda *args, **kwargs: torch.zeros((2, 4, 4), dtype=torch.float32))
    monkeypatch.setattr(features_mod, "build_status_feature", lambda *args, **kwargs: torch.zeros((8,), dtype=torch.float32))
    monkeypatch.setattr(features_mod, "compute_target_point_from_sample", lambda *args, **kwargs: torch.zeros((2,), dtype=torch.float32))
    monkeypatch.setattr(features_mod, "_build_topology_polyline_from_sample", lambda *args, **kwargs: torch.zeros((8, 2), dtype=torch.float32))
    monkeypatch.setattr(features_mod, "_build_target_line", lambda *args, **kwargs: torch.zeros((8, 2), dtype=torch.float32))
    monkeypatch.setattr(
        features_mod,
        "_build_mode_features",
        lambda *args, **kwargs: {
            "coarse_trajectories": torch.zeros((16, 8, 2), dtype=torch.float32),
            "mode_valid_mask": torch.ones((16,), dtype=torch.bool),
        },
    )
    monkeypatch.setattr(
        features_mod,
        "normalize_agent_targets",
        lambda *args, **kwargs: (torch.zeros((32, 8), dtype=torch.float32), torch.zeros((32,), dtype=torch.bool)),
    )
    monkeypatch.setattr(features_mod, "semantic_map_to_target", lambda *args, **kwargs: torch.zeros((64, 64), dtype=torch.int64))

    _, targets = features_mod.sample_to_features_targets(sample, config)

    assert int(targets["gt_mode_label"]) == 0
