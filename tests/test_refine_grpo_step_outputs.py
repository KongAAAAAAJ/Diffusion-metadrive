from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

import models.diffusion.test_refine_grpo as module


class _Writer:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(np.asarray(frame).copy())


def test_save_step_outputs_locked_uses_plain_2d_frame_and_writes_once(tmp_path, monkeypatch):
    plain_rgb = np.full((4, 5, 3), 17, dtype=np.uint8)
    capture_calls = []

    def capture_plain(base_env):
        capture_calls.append(base_env)
        return plain_rgb

    monkeypatch.setattr(module, "_capture_plain_2d_frame", capture_plain)
    monkeypatch.setattr(
        module,
        "_get_vehicle_pose",
        lambda env, aid: np.asarray([11.0, 2.0, 0.5], dtype=np.float32),
    )

    writer_2d = _Writer()
    records = []
    records_path = tmp_path / "records.jsonl"
    env = SimpleNamespace(base_env=object())

    writer_2d, writer_3d = module._save_step_outputs(
        env=env,
        output_dir=tmp_path,
        records_path=records_path,
        all_records=records,
        test_config={"save_trajectory_data": True, "save_2d_video": True, "save_3d_video": False},
        video_fps=10,
        vid_2d=writer_2d,
        vid_3d=None,
        episode_idx=0,
        step_idx=1,
        agent_ids=["agent0"],
        control_backend="locked_lqr_follow",
        formation_locked=True,
        env_reward=1.25,
        pdms_reward=0.0,
        pdms_by_agent={},
        selected_modes=[],
        gt_modes=[],
        mode_valid_mask=[[True]],
        all_rewards=[],
        executed_trajs={"agent0": np.zeros((2, 3), dtype=np.float32)},
        diffusion_2d_frame=None,
    )

    assert capture_calls == [env.base_env]
    assert len(writer_2d.frames) == 1
    assert writer_3d is None
    assert len(records) == 1
    assert records[0]["control_backend"] == "locked_lqr_follow"
    assert records[0]["step"] == 1
    assert records_path.read_text(encoding="utf-8").count("\n") == 1

    trajectory_path = tmp_path / "trajectory_data" / "episode_000" / "step_00001.json"
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    assert trajectory["agents"]["agent0"]["pose"] == [11.0, 2.0, 0.5]
    assert trajectory["agents"]["agent0"]["selected_mode"] is None


def test_save_step_outputs_diffusion_reuses_overlay_frame_without_plain_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(
        module,
        "_capture_plain_2d_frame",
        lambda base_env: (_ for _ in ()).throw(AssertionError("plain capture must not run")),
    )
    overlay_rgb = np.full((4, 5, 3), 23, dtype=np.uint8)
    writer_2d = _Writer()

    writer_2d, _ = module._save_step_outputs(
        env=SimpleNamespace(base_env=object()),
        output_dir=tmp_path,
        records_path=tmp_path / "records.jsonl",
        all_records=[],
        test_config={"save_trajectory_data": False, "save_2d_video": True, "save_3d_video": False},
        video_fps=10,
        vid_2d=writer_2d,
        vid_3d=None,
        episode_idx=0,
        step_idx=1,
        agent_ids=["agent0"],
        control_backend="diffusion_planner",
        formation_locked=False,
        env_reward=0.5,
        pdms_reward=0.4,
        pdms_by_agent={"agent0": 0.4},
        selected_modes=[2],
        gt_modes=[1],
        mode_valid_mask=[[True, True, True]],
        all_rewards=[[0.1, 0.2, 0.4]],
        executed_trajs={"agent0": np.ones((2, 3), dtype=np.float32)},
        diffusion_2d_frame=overlay_rgb,
    )

    assert len(writer_2d.frames) == 1
