import numpy as np

from metadrive.policy.diffusion_policy.transfuser_config import TransfuserConfig
from metadrive.policy.diffusion_policy.transfuser_features import (
    _build_target_line_from_polyline,
    processed_sample_to_features_targets,
)


def test_target_line_samples_to_projected_target_point_on_polyline():
    config = TransfuserConfig(target_line_num_points=5)
    polyline = np.asarray([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    target_point = np.asarray([6.0, 2.0], dtype=np.float32)

    target_line = _build_target_line_from_polyline(polyline, target_point, config)

    assert target_line.shape == (5, 2)
    assert target_line.dtype == np.float32
    np.testing.assert_allclose(target_line[:, 0], np.linspace(0.0, 6.0, 5), atol=1e-5)
    np.testing.assert_allclose(target_line[:, 1], 0.0, atol=1e-5)


def test_processed_sample_rebuilds_missing_target_line_from_topology_polyline():
    config = TransfuserConfig()
    sample = {
        "camera_feature": np.zeros((3, 8, 8), dtype=np.float32),
        "lidar_feature": np.zeros((1, 8, 8), dtype=np.float32),
        "status_feature": np.zeros((config.status_feature_dim,), dtype=np.float32),
        "ego_state": np.zeros((config.status_feature_dim,), dtype=np.float32),
        "target_point": np.asarray([4.0, 1.0], dtype=np.float32),
        "topology_polyline": np.asarray([[0.0, 0.0], [8.0, 0.0]], dtype=np.float32),
        "trajectory": np.zeros((config.trajectory_sampling.num_poses, 3), dtype=np.float32),
        "agent_states": np.zeros((config.num_bounding_boxes, 5), dtype=np.float32),
        "agent_labels": np.zeros((config.num_bounding_boxes,), dtype=bool),
        "bev_semantic_map": np.zeros(config.bev_semantic_frame, dtype=np.int64),
    }

    features, _ = processed_sample_to_features_targets(sample)

    target_line = features["target_line"].numpy()
    assert target_line.shape == (config.target_line_num_points, 2)
    np.testing.assert_allclose(target_line[-1], np.asarray([4.0, 0.0], dtype=np.float32), atol=1e-5)
