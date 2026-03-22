from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_dataset_collect.sh"


def test_script_declares_both_collection_modes_and_structure_comments():
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "COLLECTION_MODE" in text
    assert "fixed_hybrid" in text
    assert "random_road" in text

    # 每种配置都要有道路结构注释说明。
    assert "straight road structure" in text
    assert "curve road structure" in text
    assert "intersection road structure" in text
    assert "roundabout road structure" in text


def test_script_declares_two_densities_and_at_least_ten_seeds():
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "0.02" in text
    assert "0.08" in text

    match = re.search(r'SEED_LIST="\$\{SEED_LIST:-([^}]*)\}"', text)
    assert match is not None, text
    seeds = [item.strip() for item in match.group(1).split(",") if item.strip()]
    assert len(seeds) >= 10, seeds


def test_phase2_plan_dry_run_prints_fixed_hybrid_and_random_road_jobs():
    env = dict(os.environ)
    env["PYTHON_BIN"] = "/home/kong/anaconda3/envs/meta_drive/bin/python"
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT_PATH),
        ],
        cwd=str(REPO_ROOT),
        env={
            **env,
            "DRY_RUN": "1",
            "COLLECTION_MODE": "phase2_plan",
            "TARGET_SAMPLES": "50",
            "OUTPUT_ROOT": "/tmp/phase2_task3",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    output = result.stdout
    assert "mode=fixed_hybrid" in output, output
    assert "mode=random_road" in output, output
    assert output.count("seed=") >= 10, output

