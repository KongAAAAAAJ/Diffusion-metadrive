"""Run multi-agent collection, anchor preparation, and preprocessing from one YAML file."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPO_ROOT = Path(__file__).resolve().parents[1]
VALID_STAGES = ("all", "collect", "anchors", "preprocess")
VALID_ANCHOR_METHODS = ("dynamic", "k_means")
VALID_OUTPUT_FORMATS = ("dir", "npz-legacy")

SECTION_KEYS = {
    "pipeline": {"model_config_path"},
    "dataset": {"name", "output_root"},
    "anchors": {
        "method",
        "output_path",
        "figure_path",
        "trajectory_key",
        "num_anchors",
        "seed",
        "split",
        "max_trajectories",
    },
    "preprocess": {"output_suffix", "output_format", "skip_existing"},
    "collection": {
        "target_samples",
        "start_seed",
        "max_episodes",
        "samples_per_shard",
        "resume",
        "trajectory_visualization_enabled",
        "save_videos",
    },
}
TOP_LEVEL_KEYS = set(SECTION_KEYS) | {"env_config"}


@dataclass(frozen=True)
class PipelineCommand:
    stage: str
    argv: List[str]
    log_path: Optional[Path] = None


@dataclass(frozen=True)
class MultiDataPipelineConfig:
    config_path: Path
    model_config_path: Path
    dataset_name: str
    output_root: Path
    collect_output: Path
    preprocess_output: Path
    anchors_output: Path
    anchors_figure: Path
    trajectory_key: str
    num_anchors: int
    anchor_seed: int
    anchor_split: str
    anchor_max_trajectories: int
    output_format: str
    skip_existing: bool
    anchor_method: str

    @property
    def collection_log(self) -> Path:
        return self.output_root / f"collect_multi_command_{self.dataset_name}.log"


def _mapping(payload: Mapping[str, object], name: str) -> Dict[str, object]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"'{name}' must be a mapping in the dataset config")
    result = dict(value)
    allowed = SECTION_KEYS.get(name)
    if allowed is not None:
        unknown = sorted(set(result) - allowed)
        if unknown:
            raise ValueError(f"Unknown '{name}' config fields: {unknown}")
    return result


def _required(section: Mapping[str, object], section_name: str, key: str) -> object:
    if key not in section:
        raise ValueError(f"Missing required config field: {section_name}.{key}")
    return section[key]


def _repo_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_pipeline_config(config_path: Path) -> MultiDataPipelineConfig:
    config_path = config_path.expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("Dataset config root must be a mapping")
    unknown_sections = sorted(set(payload) - TOP_LEVEL_KEYS)
    if unknown_sections:
        raise ValueError(f"Unknown dataset config sections: {unknown_sections}")

    # Validate sections consumed by the collector even though this orchestrator
    # deliberately leaves their individual keys to collect_multi_experts.
    _mapping(payload, "env_config")
    _mapping(payload, "collection")
    pipeline = _mapping(payload, "pipeline")
    dataset = _mapping(payload, "dataset")
    anchors = _mapping(payload, "anchors")
    preprocess = _mapping(payload, "preprocess")

    model_config_path = _repo_path(
        _required(pipeline, "pipeline", "model_config_path")
    )
    dataset_name = str(_required(dataset, "dataset", "name")).strip()
    if not dataset_name or Path(dataset_name).name != dataset_name:
        raise ValueError("dataset.name must be a non-empty directory name")
    output_root = _repo_path(_required(dataset, "dataset", "output_root"))
    collect_output = output_root / dataset_name

    output_suffix = str(_required(preprocess, "preprocess", "output_suffix"))
    if not output_suffix:
        raise ValueError("preprocess.output_suffix must not be empty")
    preprocess_output = Path(f"{collect_output}{output_suffix}")
    output_format = str(_required(preprocess, "preprocess", "output_format"))
    if output_format not in VALID_OUTPUT_FORMATS:
        raise ValueError(
            f"preprocess.output_format must be one of {VALID_OUTPUT_FORMATS}, got {output_format!r}"
        )
    skip_existing = _required(preprocess, "preprocess", "skip_existing")
    if not isinstance(skip_existing, bool):
        raise ValueError("preprocess.skip_existing must be a boolean")

    num_anchors = int(_required(anchors, "anchors", "num_anchors"))
    max_trajectories = int(
        _required(anchors, "anchors", "max_trajectories")
    )
    if num_anchors <= 0 or max_trajectories <= 0:
        raise ValueError("anchors.num_anchors and max_trajectories must be positive")

    anchor_method = str(_required(anchors, "anchors", "method"))
    if anchor_method not in VALID_ANCHOR_METHODS:
        raise ValueError(
            f"anchors.method must be one of {VALID_ANCHOR_METHODS}, got {anchor_method!r}"
        )

    return MultiDataPipelineConfig(
        config_path=config_path,
        model_config_path=model_config_path,
        dataset_name=dataset_name,
        output_root=output_root,
        collect_output=collect_output,
        preprocess_output=preprocess_output,
        anchors_output=_repo_path(_required(anchors, "anchors", "output_path")),
        anchors_figure=_repo_path(_required(anchors, "anchors", "figure_path")),
        trajectory_key=str(_required(anchors, "anchors", "trajectory_key")),
        num_anchors=num_anchors,
        anchor_seed=int(_required(anchors, "anchors", "seed")),
        anchor_split=str(_required(anchors, "anchors", "split")),
        anchor_max_trajectories=max_trajectories,
        output_format=output_format,
        skip_existing=skip_existing,
        anchor_method=anchor_method,
    )


def build_stage_commands(
    config: MultiDataPipelineConfig,
    stage: str,
    python_executable: str = sys.executable,
) -> List[PipelineCommand]:
    if stage not in VALID_STAGES:
        raise ValueError(f"Unknown stage {stage!r}; use one of {VALID_STAGES}")

    commands: List[PipelineCommand] = []
    if stage in ("all", "collect"):
        commands.append(
            PipelineCommand(
                stage="collect",
                argv=[
                    python_executable,
                    "-m",
                    "expert_dataset.collect_multi_experts",
                    "--output-root",
                    str(config.output_root),
                    "--dataset-name",
                    config.dataset_name,
                    "--model-config-path",
                    str(config.model_config_path),
                    "--dataset-config-path",
                    str(config.config_path),
                ],
                log_path=config.collection_log,
            )
        )
    if stage in ("all", "anchors") and config.anchor_method == "k_means":
        commands.append(
            PipelineCommand(
                stage="anchors",
                argv=[
                    python_executable,
                    "-m",
                    "expert_dataset.abstract_anchors_default",
                    "--dataset-root",
                    str(config.collect_output),
                    "--output-path",
                    str(config.anchors_output),
                    "--split",
                    config.anchor_split,
                    "--trajectory-key",
                    config.trajectory_key,
                    "--num-anchors",
                    str(config.num_anchors),
                    "--max-trajectories",
                    str(config.anchor_max_trajectories),
                    "--seed",
                    str(config.anchor_seed),
                    "--figure-path",
                    str(config.anchors_figure),
                    "--no-show",
                ],
            )
        )
    if stage in ("all", "preprocess"):
        commands.append(
            PipelineCommand(
                stage="preprocess",
                argv=[
                    python_executable,
                    "-m",
                    "models.diffusion.preprocess_transfuser_dataset",
                    "--input-root",
                    str(config.collect_output),
                    "--output-root",
                    str(config.preprocess_output),
                    "--output-format",
                    config.output_format,
                    "--skip-existing",
                    str(int(config.skip_existing)),
                    "--model-config-path",
                    str(config.model_config_path),
                ],
            )
        )
    return commands


def _run_command(command: PipelineCommand) -> None:
    if command.log_path is None:
        subprocess.run(command.argv, cwd=REPO_ROOT, check=True)
        return

    command.log_path.parent.mkdir(parents=True, exist_ok=True)
    with command.log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command.argv,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command.argv)


def run_pipeline(config: MultiDataPipelineConfig, stage: str) -> None:
    commands = build_stage_commands(config, stage)
    print("Multi-agent pipeline config:")
    print(f"  stage={stage} anchor_method={config.anchor_method}")
    print(f"  dataset_config={config.config_path}")
    print(f"  model_config={config.model_config_path}")
    print(f"  collect_output={config.collect_output}")
    print(f"  anchors_output={config.anchors_output}")
    print(f"  preprocess_output={config.preprocess_output}")

    if stage in ("all", "anchors") and config.anchor_method == "dynamic":
        print("Dynamic anchors enabled. Static K-Means extraction will be skipped.")

    labels = {
        "collect": "Multi-agent Expert Collection",
        "anchors": "Anchor Preparation",
        "preprocess": "Diffusion Preprocess",
    }
    for command in commands:
        print(f"=== {labels[command.stage]} ===", flush=True)
        _run_command(command)
        print(f"=== {command.stage} done ===", flush=True)
    print("=== Multi-agent pipeline complete ===")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/dataset/data_collect.yaml",
    )
    parser.add_argument("--stage", choices=VALID_STAGES, default="all")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run_pipeline(load_pipeline_config(args.config), args.stage)


if __name__ == "__main__":
    main()
