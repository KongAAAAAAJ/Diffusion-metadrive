from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "train" / "train_platoon_rl.py"
CONFIG = ROOT / "configs" / "train" / "platoon_grpo_v2.yaml"
PYTHON_BIN = "/home/kong/anaconda3/envs/meta_drive/bin/python"


def _run(mode: str, tmp_path: Path):
    log_dir = tmp_path / f"logs_{mode}"
    cmd = [
        PYTHON_BIN,
        str(SCRIPT),
        "--config",
        str(CONFIG),
        "--mode",
        mode,
        "--steps",
        "5",
        "--render",
        "0",
        "--log-dir",
        str(log_dir),
    ]
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    return proc, elapsed, log_dir


def test_train_platoon_rl_entrypoint(tmp_path: Path):
    assert SCRIPT.exists(), SCRIPT
    assert CONFIG.exists(), CONFIG

    with CONFIG.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)
    for key in [
        "group_size",
        "lr",
        "kl_threshold",
        "reward_config",
        "total_steps",
        "checkpoint_interval",
        "ddim_steps",
        "ddim_eta",
        "advantage_discount_gamma",
        "beta_reg_max",
        "beta_reg_min",
        "lambda_local",
        "lambda_team",
        "use_closedloop",
    ]:
        assert key in config, key

    toy_proc, toy_elapsed, toy_log_dir = _run("toy-single", tmp_path)
    assert toy_proc.returncode == 0, toy_proc.stderr
    assert toy_elapsed < 300.0
    assert toy_log_dir.exists()
    assert any(path.name.startswith("events.out.tfevents") for path in toy_log_dir.rglob("*"))

    platoon_proc, platoon_elapsed, platoon_log_dir = _run("platoon", tmp_path)
    assert platoon_proc.returncode == 0, platoon_proc.stderr
    assert platoon_elapsed < 300.0
    assert platoon_log_dir.exists()
    assert any(path.name.startswith("events.out.tfevents") for path in platoon_log_dir.rglob("*"))
