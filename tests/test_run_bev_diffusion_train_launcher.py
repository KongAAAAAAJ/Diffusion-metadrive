from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_bev_diffusion_train.sh"


def _fake_python(path: Path) -> Path:
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" > "$CALL_LOG"
""",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _environment(tmp_path: Path) -> dict[str, str]:
    environment = {
        **os.environ,
        "PYTHON_BIN": str(_fake_python(tmp_path / "fake-python")),
        "OUTPUT_ROOT": str(tmp_path / "outputs"),
        "CALL_LOG": str(tmp_path / "call.log"),
    }
    for name in (
        "CONFIG",
        "DATASET_ROOT",
        "DEVICE",
        "MAX_OPTIMIZER_STEPS",
        "RUN_MODE",
        "VARIANT",
    ):
        environment.pop(name, None)
    return environment


def _run_launcher(environment: dict[str, str]) -> str:
    subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(environment["CALL_LOG"]).read_text(encoding="utf-8").strip()


def test_launcher_defaults_to_rule_conditioned_v2_config(tmp_path: Path) -> None:
    call = _run_launcher(_environment(tmp_path))

    assert "-m train.train_bev_diffusion_stage1" in call
    assert f"--config {ROOT / 'configs/train/bev_diffusion_stage1_v2.yaml'}" in call


def test_launcher_preserves_explicit_config_override(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    custom_config = tmp_path / "custom.yaml"
    environment["CONFIG"] = str(custom_config)

    call = _run_launcher(environment)

    assert f"--config {custom_config}" in call
