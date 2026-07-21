from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

from expert_dataset import collect_multi_experts as multi


def test_multi_defaults_are_three_agents_and_s5_s6_only() -> None:
    config = multi.MultiExpertCollectorConfig()

    assert config.num_agents == 3
    assert config.expert_type == "rule_planner"
    assert config.target_samples == 40000
    assert config.scenario_weights == {}
    assert config.dataset_config_path == Path("configs/dataset/data_collect.yaml")


def test_env_config_loads_dataset_yaml_and_applies_collection_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "data_collect.yaml"
    config_path.write_text(
        "env_config:\n"
        "  horizon: 77\n"
        "  image_on_cuda: true\n"
        "  num_agents: 3\n"
        "  scenario_ids:\n"
        "    - S5_hard_brake_lead\n"
        "    - S6_background_merge_in\n",
        encoding="utf-8",
    )
    config = multi.MultiExpertCollectorConfig(dataset_config_path=config_path)

    env_config = multi.load_collection_env_config(config)

    assert env_config["horizon"] == 77
    assert env_config["num_agents"] == 3
    assert env_config["image_on_cuda"] is False
    assert env_config["use_render"] is False
    assert env_config["scenario_ids"] == list(multi.DEFAULT_SCENARIOS)
    assert config.scenario_weights == multi.DEFAULT_SCENARIOS


def test_parse_args_uses_collection_yaml_as_defaults_and_allows_cli_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "data_collect.yaml"
    config_path.write_text(
        "env_config:\n"
        "  num_agents: 3\n"
        "  scenario_ids: [S5_hard_brake_lead, S6_background_merge_in]\n"
        "collection:\n"
        "  target_samples: 1234\n"
        "  start_seed: 21\n"
        "  resume: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_multi_experts.py",
            "--dataset-config-path",
            str(config_path),
            "--target-samples",
            "99",
        ],
    )

    config = multi.parse_args()

    assert config.dataset_config_path == config_path
    assert config.target_samples == 99
    assert config.start_seed == 21
    assert config.resume is True


def test_info_failure_reason_detects_any_agent_failure() -> None:
    agent_ids = ["agent0", "agent1", "agent2"]

    assert multi._info_failure_reason({"agent1": {"crash_vehicle": True}}, agent_ids) == "crash:agent1"
    assert multi._info_failure_reason({"agent2": {"out_of_road": True}}, agent_ids) == "out_of_road:agent2"
    assert multi._info_failure_reason({}, agent_ids) is None


def test_flattened_samples_are_time_major_and_keep_joint_metadata(monkeypatch) -> None:
    config = multi.MultiExpertCollectorConfig(
        horizon_steps=1,
        target_stride_steps=1,
        trajectory_num_poses=1,
    )
    frames_by_agent = {}
    for agent_index in range(3):
        agent_id = f"agent{agent_index}"
        frames_by_agent[agent_id] = [
            {
                "agent_id": np.asarray(agent_id),
                "agent_index": np.asarray(agent_index, dtype=np.int8),
                "agent_role": np.asarray("leader" if agent_index == 0 else "follower"),
                "formation_relation_state": np.full((12,), agent_index, dtype=np.float32),
                "joint_step_index": np.asarray(step, dtype=np.int32),
                "num_agents": np.asarray(3, dtype=np.int8),
            }
            for step in range(2)
        ]

    def fake_build_episode_samples(frames, *args, **kwargs):
        return [
            {
                "_sample_index": np.asarray(step, dtype=np.int32),
                "trajectory": np.zeros((8, 3), dtype=np.float32),
            }
            for step in range(2)
        ]

    monkeypatch.setattr(multi, "build_episode_samples", fake_build_episode_samples)
    rollout = multi.PlatoonEpisodeRollout(frames_by_agent, [], False, None, {})
    spec = multi.EpisodeSpec("S5_hard_brake_lead", "mainline", "route", 0.0, 1)

    samples = multi.build_flattened_episode_samples(rollout, config, spec, episode_index=4)

    order = [
        (int(sample["joint_step_index"]), int(sample["agent_index"]))
        for sample in samples
    ]
    assert order == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert all(sample["formation_relation_state"].shape == (12,) for sample in samples)
