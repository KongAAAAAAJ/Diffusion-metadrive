from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml

from metadrive.policy.diffusion_policy.transfuser_config import build_transfuser_config
from models.platoon.platoon_diffusion_planner import PlatoonDiffusionPlanner

SCRIPT_PATH = Path("scripts/test_platoon_rl_ckpt.py")
SPEC = importlib.util.spec_from_file_location("test_platoon_rl_ckpt", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
rl_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rl_eval)


def _write_yaml(tmp_path: Path, **overrides) -> Path:
    data = yaml.safe_load(Path("configs/train/platoon_grpo_v2.yaml").read_text(encoding="utf-8"))
    data.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _make_rl_ckpt(path: Path, num_agents: int = 3) -> Path:
    config_tf = build_transfuser_config("small", plan_anchor_path="metadrive/exp_dataset/anchors.npy")
    model = PlatoonDiffusionPlanner(config_tf, num_vehicles=num_agents)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": 7, "model_state_dict": model.state_dict(), "metrics": {"loss": 1.0}}, path)
    return path


def test_resolve_latest_checkpoint_selects_latest_run_and_step(tmp_path):
    root = tmp_path / "diffusion_rl"
    _make_rl_ckpt(root / "run_1" / "checkpoints" / "step_10.ckpt")
    latest = _make_rl_ckpt(root / "run_2" / "checkpoints" / "step_20.ckpt")
    resolved = rl_eval.resolve_rl_checkpoint("latest", root)
    assert resolved == latest


def test_build_output_dir_defaults_to_run_and_step_name(tmp_path):
    ckpt = _make_rl_ckpt(tmp_path / "run_9" / "checkpoints" / "step_30.ckpt")
    out = rl_eval.resolve_output_dir(None, ckpt)
    assert out == Path("outputs/rl_eval") / "run_9_step_30"


def test_load_rl_checkpoint_into_planner(tmp_path):
    ckpt = _make_rl_ckpt(tmp_path / "step_12.ckpt")
    config_tf = build_transfuser_config("small", plan_anchor_path="metadrive/exp_dataset/anchors.npy")
    model = PlatoonDiffusionPlanner(config_tf, num_vehicles=3)
    payload = rl_eval.load_rl_checkpoint(ckpt, model)
    assert int(payload["step"]) == 7


def test_cli_overrides_config_values(tmp_path):
    config_path = _write_yaml(tmp_path, num_agents=3, traffic_density=0.08)
    args = rl_eval.parse_args([
        "--checkpoint", str(tmp_path / "dummy.ckpt"),
        "--config", str(config_path),
        "--num-agents", "5",
        "--hazard-scenario", "dynamic_cut_in",
    ])
    merged = rl_eval.load_eval_config(args)
    assert merged["num_agents"] == 5
    assert merged["hazard_scenario"] == "dynamic_cut_in"
    assert float(merged["traffic_density"]) == 0.08


def test_write_summary_json(tmp_path):
    summary_path = tmp_path / "summary.json"
    payload = {"success_rate": 1.0, "videos": ["episode_000.mp4"]}
    rl_eval.write_summary(summary_path, payload)
    assert json.loads(summary_path.read_text(encoding="utf-8")) == payload
