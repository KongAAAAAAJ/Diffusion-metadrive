from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.bev_model_manifest import (
    MANIFEST_FORMAT,
    ModelManifestError,
    file_sha256,
    load_model_manifest,
)


def _stage1(path: Path, model_id: str = "stage1_a") -> dict[str, str]:
    return {
        "id": model_id,
        "kind": "stage1",
        "variant": "A",
        "checkpoint": str(path),
        "checkpoint_sha256": file_sha256(path),
    }


def _grpo(path: Path, source: Path, model_id: str) -> dict[str, str]:
    return {
        "id": model_id,
        "kind": "grpo",
        "variant": "A",
        "reward_domain": "tau_cmd" if model_id == "grpo_open" else "tau_a",
        "checkpoint": str(path),
        "checkpoint_sha256": file_sha256(path),
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": file_sha256(source),
    }


def _write(path: Path, models: list[dict], comparisons: list[dict]) -> Path:
    path.write_text(
        json.dumps(
            {
                "format": MANIFEST_FORMAT,
                "models": models,
                "comparisons": comparisons,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_single_and_three_model_manifests(tmp_path: Path) -> None:
    stage1 = tmp_path / "stage1.pt"
    grpo_open = tmp_path / "open.pt"
    grpo_exec = tmp_path / "exec.pt"
    stage1.write_bytes(b"stage1")
    grpo_open.write_bytes(b"open")
    grpo_exec.write_bytes(b"exec")
    single = load_model_manifest(
        _write(tmp_path / "single.json", [_stage1(stage1)], [])
    )
    assert single.model_ids == ("stage1_a",)
    assert single.comparisons == ()

    three = load_model_manifest(
        _write(
            tmp_path / "three.json",
            [
                _stage1(stage1),
                _grpo(grpo_open, stage1, "grpo_open"),
                _grpo(grpo_exec, stage1, "grpo_exec"),
            ],
            [
                {
                    "id": "open_vs_stage1",
                    "baseline": "stage1_a",
                    "candidate": "grpo_open",
                },
                {
                    "id": "exec_vs_open",
                    "baseline": "grpo_open",
                    "candidate": "grpo_exec",
                },
            ],
        )
    )
    assert three.model_ids == ("stage1_a", "grpo_open", "grpo_exec")
    assert [value.comparison_id for value in three.comparisons] == [
        "open_vs_stage1",
        "exec_vs_open",
    ]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda payload: payload.update(models=[]), "non-empty"),
        (
            lambda payload: payload["models"].append(dict(payload["models"][0])),
            "unique",
        ),
        (
            lambda payload: payload.update(
                comparisons=[
                    {"id": "bad", "baseline": "stage1_a", "candidate": "missing"}
                ]
            ),
            "candidate is unknown",
        ),
        (
            lambda payload: payload.update(
                comparisons=[
                    {
                        "id": "self",
                        "baseline": "stage1_a",
                        "candidate": "stage1_a",
                    }
                ]
            ),
            "itself",
        ),
    ],
)
def test_manifest_rejects_invalid_model_sets(
    tmp_path: Path, mutate, match: str
) -> None:
    checkpoint = tmp_path / "stage1.pt"
    checkpoint.write_bytes(b"stage1")
    payload = {
        "format": MANIFEST_FORMAT,
        "models": [_stage1(checkpoint)],
        "comparisons": [],
    }
    mutate(payload)
    manifest = tmp_path / "invalid.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ModelManifestError, match=match):
        load_model_manifest(manifest)


def test_manifest_rejects_hash_mismatch_and_incomplete_grpo(tmp_path: Path) -> None:
    checkpoint = tmp_path / "stage1.pt"
    checkpoint.write_bytes(b"stage1")
    bad_hash = _stage1(checkpoint)
    bad_hash["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ModelManifestError, match="SHA256 mismatch"):
        load_model_manifest(_write(tmp_path / "bad_hash.json", [bad_hash], []))

    incomplete = _stage1(checkpoint, "grpo_open")
    incomplete["kind"] = "grpo"
    with pytest.raises(ModelManifestError, match="fields must be exactly"):
        load_model_manifest(
            _write(tmp_path / "incomplete_grpo.json", [incomplete], [])
        )


def test_v1_manifest_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "v1.json"
    path.write_text(
        json.dumps({"format": "bev_four_model_manifest_v1", "models": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ModelManifestError, match="fields must be exactly|format"):
        load_model_manifest(path)
