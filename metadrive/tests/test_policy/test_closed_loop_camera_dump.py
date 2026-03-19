from __future__ import annotations

from pathlib import Path

import numpy as np

from metadrive.policy.diffusion_policy.test_transfuser_policy import save_triplet_cameras


def test_save_triplet_cameras_writes_three_views(tmp_path: Path):
    observation = {
        "rgb_left": np.zeros((8, 12, 3), dtype=np.uint8),
        "rgb_front": np.ones((8, 12, 3), dtype=np.uint8) * 127,
        "rgb_right": np.ones((8, 12, 3), dtype=np.uint8) * 255,
    }

    save_triplet_cameras(observation, tmp_path, episode_idx=1, step_idx=10)

    assert (tmp_path / "ep001_step00010_left.png").exists()
    assert (tmp_path / "ep001_step00010_front.png").exists()
    assert (tmp_path / "ep001_step00010_right.png").exists()
