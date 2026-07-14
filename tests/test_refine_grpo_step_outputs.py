from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import models.diffusion.test_refine_grpo as module


class _Writer:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(np.asarray(frame).copy())


def test_format_episode_end_reason_reports_terminated_agents_without_inferring_cause():
    message = module._format_episode_end_reason(
        episode_idx=1,
        step_idx=52,
        info={
            "termination_flags": {
                "agent0": True,
                "agent1": False,
                "agent2": False,
                "__all__": True,
            },
            "truncation_flags": {
                "agent0": False,
                "agent1": False,
                "agent2": False,
                "__all__": False,
            },
        },
        fallback_reason="terminated",
    )

    assert message == (
        "[test-refine-grpo] episode=1 ended: step=52 reason=terminated "
        "terminated_agents=agent0"
    )


def test_format_episode_end_reason_reports_terminated_and_truncated_agents():
    message = module._format_episode_end_reason(
        episode_idx=0,
        step_idx=300,
        info={
            "termination_flags": {"agent0": False, "agent1": True, "__all__": True},
            "truncation_flags": {"agent0": False, "agent1": True, "__all__": True},
        },
        fallback_reason="terminated",
    )

    assert message == (
        "[test-refine-grpo] episode=0 ended: step=300 reason=terminated,truncated "
        "terminated_agents=agent1 truncated_agents=agent1"
    )


def test_format_episode_end_reason_uses_non_vehicle_fallback_reason():
    for reason in (
        "max_steps",
        "planner_data_missing",
        "env_step_error",
        "terminated",
        "truncated",
        "unknown",
    ):
        message = module._format_episode_end_reason(
            episode_idx=0,
            step_idx=7,
            info={},
            fallback_reason=reason,
        )
        assert message == f"[test-refine-grpo] episode=0 ended: step=7 reason={reason}"


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
    env = SimpleNamespace(base_env=SimpleNamespace(agents={}))

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
        info={},
        pdms={},
        planning_debug={},
        execution_debug={"actions": {}, "control_debug": {}},
        grpo_debug={"enabled": False},
        previous_speed_mps={},
        dt=0.1,
        executed_trajs={"agent0": np.zeros((2, 3), dtype=np.float32)},
        diffusion_2d_frame=None,
    )

    assert capture_calls == [env.base_env]
    assert len(writer_2d.frames) == 1
    assert writer_3d is None
    assert len(records) == 1
    assert records[0]["control_backend"] == "locked_lqr_follow"
    assert records[0]["step_idx"] == 1
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
        env=SimpleNamespace(base_env=SimpleNamespace(agents={})),
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
        info={},
        pdms={"agent0": {"reward": 0.4}},
        planning_debug={},
        execution_debug={"actions": {}, "control_debug": {}},
        grpo_debug={
            "enabled": True,
            "selected_mode": {"agent0": 2},
            "best_mode": {"agent0": 1},
        },
        previous_speed_mps={},
        dt=0.1,
        executed_trajs={"agent0": np.ones((2, 3), dtype=np.float32)},
        diffusion_2d_frame=overlay_rgb,
    )

    assert len(writer_2d.frames) == 1


def test_save_step_outputs_uses_common_evaluation_record(tmp_path, monkeypatch):
    calls = []

    def collect(**kwargs):
        calls.append(kwargs)
        return {
            "step_idx": kwargs["step_idx"],
            "actions": {"agent0": {"steer": 0.2, "throttle": 0.4}},
            "info": kwargs["info"],
            "pdms": kwargs["pdms"],
            "planning": kwargs["planning_debug"],
            "control_debug": kwargs["control_debug"],
            "vehicles": {"agent0": {"x": 1.0, "y": 2.0}},
        }

    monkeypatch.setattr(module, "_collect_episode_step_record", collect, raising=False)
    records = []
    records_path = tmp_path / "records.jsonl"
    base_env = SimpleNamespace(agents={})
    grpo = {
        "enabled": True,
        "selected_mode": {"agent0": 1},
        "best_group": {"agent0": 0},
        "best_mode": {"agent0": 2},
        "best_reward": {"agent0": 0.8},
        "valid_mask": {"agent0": [True, True, True]},
        "rewards": {"agent0": [[0.1, 0.4, 0.8]]},
        "pdms_debug": {"agent0": {"quality": 0.4}},
    }
    pdms = {"agent0": {"reward": 0.4, "progress": 0.5}}
    planning = {
        "planning_policy": "diffusion_refine_grpo",
        "coordinate_frame": "world",
        "trajectories_by_agent": {"agent0": [[1.0, 2.0, 0.0], [2.0, 2.0, 0.0]]},
        "candidates_by_agent": {"agent0": [{"mode": 1, "selected": True}]},
    }
    execution_debug = {
        "actions": {"agent0": np.asarray([0.2, 0.4], dtype=np.float32)},
        "control_debug": {"agent0": {"lookahead": 2}},
    }

    module._save_step_outputs(
        env=SimpleNamespace(base_env=base_env),
        output_dir=tmp_path,
        records_path=records_path,
        all_records=records,
        test_config={"save_trajectory_data": False, "save_2d_video": False, "save_3d_video": False},
        video_fps=10,
        vid_2d=None,
        vid_3d=None,
        episode_idx=3,
        step_idx=0,
        agent_ids=["agent0"],
        control_backend="diffusion_planner",
        formation_locked=False,
        env_reward=1.25,
        pdms_reward=0.4,
        info={"termination_flags": {"agent0": False, "__all__": False}},
        pdms=pdms,
        planning_debug=planning,
        execution_debug=execution_debug,
        grpo_debug=grpo,
        previous_speed_mps={},
        dt=0.1,
        executed_trajs={"agent0": np.zeros((2, 3), dtype=np.float32)},
        diffusion_2d_frame=None,
    )

    assert len(calls) == 1
    assert calls[0]["env"] is base_env
    assert calls[0]["actions"] is execution_debug["actions"]
    assert calls[0]["pdms"] is pdms
    assert calls[0]["planning_debug"] is planning
    assert calls[0]["control_debug"] is execution_debug["control_debug"]
    assert records[0]["episode_idx"] == 3
    assert records[0]["algorithm"] == "refine_grpo"
    assert records[0]["grpo"] == grpo
    assert "selected_modes" not in records[0]
    assert json.loads(records_path.read_text(encoding="utf-8"))["pdms"]["agent0"]["reward"] == 0.4


def test_local_trajectory_to_world_rotates_xy_and_heading():
    local = np.asarray([[1.0, 0.0, 0.0], [2.0, 1.0, 0.25]], dtype=np.float32)

    world = module._local_trajectory_to_world(
        local,
        np.asarray([10.0, 20.0, np.pi / 2], dtype=np.float32),
    )

    np.testing.assert_allclose(world[:, :2], [[10.0, 21.0], [9.0, 22.0]], atol=1e-5)
    np.testing.assert_allclose(world[:, 2], [np.pi / 2, np.pi / 2 + 0.25], atol=1e-5)


def test_extract_pdms_components_uses_executed_candidate_index():
    reward_debug = {
        key: np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
        for key in (
            "reward",
            "progress",
            "formation_lon",
            "formation_lat",
            "speed",
            "comfort",
            "consistency",
            "gate",
        )
    }

    components = module._extract_pdms_components(reward_debug, 1)

    assert set(components) == {
        "reward",
        "progress",
        "formation_lon",
        "formation_lat",
        "speed",
        "comfort",
        "consistency",
        "gate",
    }
    assert components["reward"] == pytest.approx(0.2)


def test_score_planned_trajectories_pdms_returns_common_components(monkeypatch):
    def fake_pdms(trajectory, *args, **kwargs):
        del args, kwargs
        values = np.asarray([float(trajectory[0, -1, 0])], dtype=np.float32)
        return values, {key: values for key in module._PDMS_COMPONENT_KEYS}

    monkeypatch.setattr(module, "_compute_pdms_reward_batch", fake_pdms)
    monkeypatch.setattr(
        module,
        "compute_pairwise_formation_reward",
        lambda *args, **kwargs: (np.ones(1, dtype=np.float32), np.ones(1, dtype=np.float32)),
    )
    trajectories = {
        "agent0": np.asarray([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32),
        "agent1": np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32),
    }
    poses = {
        "agent0": np.asarray([10.0, 0.0, 0.0], dtype=np.float32),
        "agent1": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    }

    result = module._score_planned_trajectories_pdms(
        ["agent0", "agent1"],
        trajectories,
        poses,
        {
            "desired_gap_m": 10.0,
            "lon_decay_m": 5.0,
            "lat_decay_m": 0.5,
            "waypoint_decay_gamma": 0.9,
        },
    )

    assert set(result) == {"agent0", "agent1"}
    assert set(result["agent0"]) == set(module._PDMS_COMPONENT_KEYS)
    assert result["agent0"]["reward"] == pytest.approx(4.0)
    assert result["agent1"]["reward"] == pytest.approx(3.0)


def test_summarize_results_accepts_new_and_legacy_records():
    records = [
        {
            "env_reward": 1.0,
            "pdms_reward": 0.3,
            "selected_modes": [1],
            "pdms_by_agent": {"agent0": 0.3},
        },
        {
            "env_reward": 2.0,
            "pdms_reward": 0.4,
            "pdms": {"agent0": {"reward": 0.4}},
            "grpo": {"enabled": True, "selected_mode": {"agent0": 2}},
        },
    ]

    summary = module._summarize_results(
        records,
        episodes=1,
        success=0,
        crash=0,
        out_of_road=0,
        metadata={},
    )

    assert summary["per_mode"]["1"]["pdms_reward_mean"] == pytest.approx(0.3)
    assert summary["per_mode"]["2"]["pdms_reward_mean"] == pytest.approx(0.4)


def test_compact_planner_debug_replaces_candidate_trajectory_with_endpoint():
    compact = module._compact_planner_debug(
        {
            "agent0": {
                "candidate_count": 1,
                "candidates": [
                    {
                        "score": 0.7,
                        "selected": True,
                        "trajectory_world": [[0.0, 0.0, 0.0], [5.0, 1.0, 0.2]],
                    }
                ],
            }
        }
    )

    candidate = compact["agent0"]["candidates"][0]
    assert "trajectory_world" not in candidate
    assert candidate["trajectory_world_points"] == 2
    assert candidate["endpoint_world"] == [5.0, 1.0, 0.2]
    assert candidate["score"] == pytest.approx(0.7)
