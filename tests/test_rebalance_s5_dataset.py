from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools.rebalance_s5_dataset import (
    S5RebalanceError,
    _payload_sha256,
    _validate_swap_paths,
)


def test_curation_fingerprint_is_canonical_and_sensitive():
    left = {"source": {"fingerprint": "a" * 64}, "episodes": [3, 7, 9]}
    reordered = {"episodes": [3, 7, 9], "source": {"fingerprint": "a" * 64}}
    changed = {"source": {"fingerprint": "b" * 64}, "episodes": [3, 7, 9]}

    assert _payload_sha256(left) == _payload_sha256(reordered)
    assert _payload_sha256(left) != _payload_sha256(changed)
    assert len(_payload_sha256(left)) == hashlib.sha256().digest_size * 2


def test_swap_paths_require_exact_sibling_names(tmp_path: Path):
    formal = tmp_path / "dataset"
    staging = tmp_path / "dataset.s5_rebalance_staging"
    backup = tmp_path / "dataset.pre_s5_rebalance_4060c643"
    _validate_swap_paths(formal, staging, backup)

    with pytest.raises(S5RebalanceError, match="staging"):
        _validate_swap_paths(formal, tmp_path / "staging", backup)
    with pytest.raises(S5RebalanceError, match="backup"):
        _validate_swap_paths(formal, staging, tmp_path / "backup")
    with pytest.raises(S5RebalanceError, match="share one parent"):
        _validate_swap_paths(formal, staging, tmp_path / "nested" / backup.name)


def test_swap_paths_reject_symbolic_links(tmp_path: Path):
    formal = tmp_path / "dataset"
    formal.mkdir()
    staging = tmp_path / "dataset.s5_rebalance_staging"
    staging.symlink_to(formal, target_is_directory=True)
    backup = tmp_path / "dataset.pre_s5_rebalance_4060c643"

    with pytest.raises(S5RebalanceError, match="symbolic"):
        _validate_swap_paths(formal, staging, backup)
