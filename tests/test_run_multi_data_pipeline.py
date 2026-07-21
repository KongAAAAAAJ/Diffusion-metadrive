from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from expert_dataset import run_multi_data_pipeline as pipeline


def _write_config(tmp_path: Path, anchor_method: str = "k_means", extra: str = "") -> Path:
    model_path = tmp_path / "model.yaml"
    model_path.write_text(
        "model:\n  anchor_method: dynamic\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "data_collect.yaml"
    config_path.write_text(
        f"""
pipeline:
  model_config_path: {model_path}
dataset:
  name: platoon_test
  output_root: {tmp_path / 'datasets'}
env_config:
  num_agents: 3
collection:
  target_samples: 10
anchors:
  method: {anchor_method}
  output_path: expert_dataset/test_platoon_anchors.npy
  figure_path: expert_dataset/test_platoon_anchors.png
  trajectory_key: trajectory
  num_anchors: 12
  seed: 7
  split: train
  max_trajectories: 321
preprocess:
  output_suffix: _processed
  output_format: dir
  skip_existing: true
{extra}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _command(config: pipeline.MultiDataPipelineConfig, stage: str) -> list[str]:
    commands = pipeline.build_stage_commands(config, stage, python_executable="python-test")
    return next(item.argv for item in commands if item.stage == stage)


def test_load_config_resolves_paths_and_derives_outputs(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    config = pipeline.load_pipeline_config(config_path)

    assert config.dataset_name == "platoon_test"
    assert config.collect_output == tmp_path / "datasets/platoon_test"
    assert config.preprocess_output == tmp_path / "datasets/platoon_test_processed"
    assert config.anchors_output == pipeline.REPO_ROOT / "expert_dataset/test_platoon_anchors.npy"
    assert config.anchor_method == "k_means"


def test_all_stage_builds_three_commands_with_yaml_parameters(tmp_path: Path) -> None:
    config = pipeline.load_pipeline_config(_write_config(tmp_path))

    commands = pipeline.build_stage_commands(config, "all", python_executable="python-test")

    assert [item.stage for item in commands] == ["collect", "anchors", "preprocess"]
    collect = commands[0].argv
    assert collect[:3] == ["python-test", "-m", "expert_dataset.collect_multi_experts"]
    assert collect[collect.index("--dataset-name") + 1] == "platoon_test"
    assert "--target-samples" not in collect

    anchors = commands[1].argv
    assert anchors[anchors.index("--num-anchors") + 1] == "12"
    assert anchors[anchors.index("--max-trajectories") + 1] == "321"
    assert anchors[anchors.index("--split") + 1] == "train"

    preprocess = commands[2].argv
    assert preprocess[preprocess.index("--output-format") + 1] == "dir"
    assert preprocess[preprocess.index("--skip-existing") + 1] == "1"


def test_dynamic_anchor_method_skips_static_command(tmp_path: Path) -> None:
    config = pipeline.load_pipeline_config(_write_config(tmp_path, anchor_method="dynamic"))

    assert pipeline.build_stage_commands(config, "anchors") == []
    assert [item.stage for item in pipeline.build_stage_commands(config, "all")] == [
        "collect",
        "preprocess",
    ]


@pytest.mark.parametrize("stage", ["collect", "anchors", "preprocess"])
def test_individual_stage_builds_only_requested_command(tmp_path: Path, stage: str) -> None:
    config = pipeline.load_pipeline_config(_write_config(tmp_path))

    commands = pipeline.build_stage_commands(config, stage)

    assert [item.stage for item in commands] == [stage]


def test_invalid_config_and_stage_are_rejected(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, extra="unknown_section: {}")
    with pytest.raises(ValueError, match="Unknown dataset config sections"):
        pipeline.load_pipeline_config(config_path)

    valid_config = pipeline.load_pipeline_config(_write_config(tmp_path))
    with pytest.raises(ValueError, match="Unknown stage"):
        pipeline.build_stage_commands(valid_config, "invalid")

    with pytest.raises(SystemExit) as exc_info:
        pipeline.parse_args(["--stage", "invalid"])
    assert exc_info.value.code == 2


def test_pipeline_stops_after_first_failed_stage(tmp_path: Path, monkeypatch) -> None:
    config = pipeline.load_pipeline_config(_write_config(tmp_path))
    visited: list[str] = []

    def fail_first(command: pipeline.PipelineCommand) -> None:
        visited.append(command.stage)
        raise subprocess.CalledProcessError(2, command.argv)

    monkeypatch.setattr(pipeline, "_run_command", fail_first)

    with pytest.raises(subprocess.CalledProcessError):
        pipeline.run_pipeline(config, "all")
    assert visited == ["collect"]
