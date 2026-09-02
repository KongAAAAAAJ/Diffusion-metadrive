from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_checkout_has_no_partial_distribution_metadata() -> None:
    broken_local_distributions = [
        str(getattr(distribution, "_path", ""))
        for distribution in importlib.metadata.distributions(path=[str(REPO_ROOT)])
        if distribution.metadata is None
    ]

    assert broken_local_distributions == []


def test_sensorless_collection_import_does_not_load_training_frameworks() -> None:
    code = """
import json
import sys
import envs.observations.semantic_bev
import models.diffusion.transfuser_config
print(json.dumps({
    'pytorch_lightning': 'pytorch_lightning' in sys.modules,
    'torchmetrics': 'torchmetrics' in sys.modules,
    'transfuser_agent': 'models.diffusion.transfuser_agent' in sys.modules,
}))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPO_ROOT), environment.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {
        "pytorch_lightning": False,
        "torchmetrics": False,
        "transfuser_agent": False,
    }


def test_lazy_public_imports_remain_available() -> None:
    from envs import PlatoonEnv, PlatoonEnvConfig
    from models.diffusion import TransfuserConfig

    assert PlatoonEnv is not None
    assert PlatoonEnvConfig is not None
    assert TransfuserConfig is not None
