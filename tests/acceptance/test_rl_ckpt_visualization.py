from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

SCRIPT_PATH = Path("scripts/test_platoon_rl_ckpt.py")
SPEC = importlib.util.spec_from_file_location("test_platoon_rl_ckpt", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
rl_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rl_eval)


def _write_yaml(tmp_path: Path, **overrides) -> Path:
    data = yaml.safe_load(Path("configs/train/selector.yaml").read_text(encoding="utf-8"))
    data.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _make_rllib_ckpt(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / ".is_checkpoint").write_text("", encoding="utf-8")
    (path / "rllib_checkpoint.json").write_text(json.dumps({"type": "Algorithm"}), encoding="utf-8")
    return path


def test_resolve_latest_checkpoint_selects_latest_run_and_checkpoint_dir(tmp_path):
    root = tmp_path / "diffusion_rl"
    _make_rllib_ckpt(root / "run_1" / "checkpoints" / "checkpoint_000001")
    latest = _make_rllib_ckpt(root / "run_2" / "checkpoints" / "checkpoint_000005")
    resolved = rl_eval.resolve_rl_checkpoint("latest", root)
    assert resolved == latest


def test_build_output_dir_defaults_to_run_and_checkpoint_name(tmp_path):
    ckpt = _make_rllib_ckpt(tmp_path / "run_9" / "checkpoints" / "checkpoint_000030")
    out = rl_eval.resolve_output_dir(None, ckpt)
    assert out == Path("outputs/rl_eval") / "run_9_checkpoint_000030"


def test_load_eval_config_prefers_summary_env_config_and_cli_overrides(tmp_path):
    config_path = _write_yaml(tmp_path, num_agents=3, traffic_density=0.08, pretrained_ckpt="/tmp/original.ckpt")
    args = rl_eval.parse_args(
        [
            "--checkpoint",
            str(tmp_path / "dummy"),
            "--config",
            str(config_path),
            "--num-agents",
            "5",
            "--hazard-scenario",
            "dynamic_cut_in",
            "--single-ckpt",
            "/tmp/override.ckpt",
            "--device",
            "cpu",
        ]
    )
    merged = rl_eval.load_eval_config(
        args,
        {
            "env_config": {
                "traffic_density": 0.12,
                "pretrained_ckpt": "/tmp/from_summary.ckpt",
                "num_agents": 4,
            }
        },
    )
    assert merged["num_agents"] == 5
    assert merged["hazard_scenario"] == "dynamic_cut_in"
    assert float(merged["traffic_density"]) == 0.12
    assert merged["pretrained_ckpt"] == "/tmp/override.ckpt"
    assert merged["planner_device"] == "cpu"


def test_extract_action_value_supports_tuple_and_array():
    assert rl_eval.extract_action_value(3) == 3
    assert rl_eval.extract_action_value(np.array([4], dtype=np.int64)) == 4
    assert rl_eval.extract_action_value((5, {"state": 1}, {"extra": 2})) == 5


def test_write_summary_json(tmp_path):
    summary_path = tmp_path / "summary.json"
    payload = {"success_rate": 1.0, "videos": ["episode_000.mp4"]}
    rl_eval.write_summary(summary_path, payload)
    assert json.loads(summary_path.read_text(encoding="utf-8")) == payload
