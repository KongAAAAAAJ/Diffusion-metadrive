"""Shared fixtures for acceptance tests."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Ensure project root is on sys.path so that envs/, models/, etc. are importable.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def data_dir() -> Path:
    """Return the dataset root directory from env var or default."""
    return Path(
        os.environ.get(
            "DATA_DIR",
            "/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets",
        )
    )


@pytest.fixture(scope="session")
def single_vehicle_ckpt(repo_root: Path) -> Path:
    return repo_root / "checkpoints" / "single_vehicle" / "best.ckpt"
