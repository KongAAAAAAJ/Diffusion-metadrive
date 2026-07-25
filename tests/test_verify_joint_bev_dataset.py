from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from expert_dataset.collect_joint_bev import (
    AgentRole,
    JOINT_SAMPLE_DTYPES,
    JOINT_SAMPLE_SHAPES,
    JointBEVSample,
)
from expert_dataset.joint_bev_storage import (
    EpisodeSplitConfig,
    JointBEVDatasetStore,
    PACKED_BEV_FIELD,
    fingerprint_payload,
)
from expert_dataset.verify_joint_bev_dataset import (
    JointBEVVerificationError,
    main,
    verify_joint_bev_dataset,
)
from models.bev_planner.mode_contract import ModeIndex


def _sample(marker: int) -> JointBEVSample:
    values = {
        name: np.zeros(shape, dtype=JOINT_SAMPLE_DTYPES[name])
        for name, shape in JOINT_SAMPLE_SHAPES.items()
    }
    values["bev"][:, 0, marker % 256, marker % 256] = 255
    values["bev"][:, 1, 0, 0] = np.asarray((0, 128, 255), dtype=np.uint8)[
        marker % 3
    ]
    values["bev"][:, 3, 0, 1] = np.asarray(
        (0, 85, 170, 255), dtype=np.uint8
    )[marker % 4]
    values["ego_state"][:] = np.float32(marker)
    values["relation_valid_mask"][:] = True
    values["agent_role"] = np.asarray(list(AgentRole), dtype=np.int64)
    values["mode_valid_mask"][:, ModeIndex.STOP] = True
    values["gt_mode"][:] = int(ModeIndex.STOP)
    return JointBEVSample(**values)


def _create(root: Path, episode_count: int = 2, samples_per_episode: int = 3) -> None:
    with JointBEVDatasetStore(
        root,
        split_config=EpisodeSplitConfig(1.0, 0.0, 0.0, seed=3),
        dataset_fingerprint=fingerprint_payload({"test": "verifier"}),
        resume=False,
    ) as store:
        marker = 0
        for episode_index in range(episode_count):
            samples = []
            for _ in range(samples_per_episode):
                samples.append(_sample(marker))
                marker += 1
            store.commit_episode(
                episode_index,
                samples,
                {
                    "scenario_id": (
                        "S5_hard_brake_lead"
                        if episode_index % 2 == 0
                        else "S6_background_merge_in"
                    ),
                    "local_route": "R3_mainline_straight",
                },
            )


def _first_episode(root: Path) -> Path:
    return next((root / "train" / "episodes").iterdir())


def test_full_verifier_reports_schema_statistics_and_throughput(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    _create(root)
    report = verify_joint_bev_dataset(
        root,
        min_decode_samples_per_s=50.0,
    )
    assert report["schema_version"] == 2
    assert report["complete_scan"] is True
    assert report["episodes"] == 2
    assert report["joint_samples"] == 6
    assert report["scanned_joint_samples"] == 6
    assert report["bev_compression_ratio"] == 6.4
    assert report["decode_joint_samples_per_s"] >= 50.0
    train = report["splits"]["train"]
    assert train["gt_mode_counts"]["STOP"] == 18
    assert train["mode_valid_rate"]["STOP"] == 1.0
    assert set(train["scenario_joint_samples"]) == {
        "S5_hard_brake_lead",
        "S6_background_merge_in",
    }


def test_verifier_cli_prints_json_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "dataset"
    _create(root, episode_count=1, samples_per_episode=2)
    assert (
        main(
            [
                "--dataset-root",
                str(root),
                "--max-samples-per-split",
                "1",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["scanned_joint_samples"] == 1
    assert report["complete_scan"] is False


def test_verifier_rejects_reserved_lane_code(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _create(root, episode_count=1, samples_per_episode=1)
    path = _first_episode(root) / f"{PACKED_BEV_FIELD}.npy"
    packed = np.load(path, mmap_mode="r+", allow_pickle=False)
    packed[0, 0, 1, 0, 0] = np.uint8(1)
    packed[0, 0, 2, 0, 0] = np.uint8(1)
    packed.flush()
    with pytest.raises(JointBEVVerificationError, match="reserved code 3"):
        verify_joint_bev_dataset(root)


@pytest.mark.parametrize(
    ("field_name", "mutate", "match"),
    (
        (
            "gt_mode",
            lambda array: array.__setitem__((0, 0), 0),
            r"mode_valid_mask\[gt_mode\]",
        ),
        (
            "agent_role",
            lambda array: array.__setitem__((0, 0), 2),
            "agent role order",
        ),
        (
            "ego_state",
            lambda array: array.__setitem__((0, 0, 0), np.nan),
            "non-finite",
        ),
    ),
)
def test_verifier_rejects_cross_field_contract_conflicts(
    tmp_path: Path, field_name: str, mutate, match: str
) -> None:
    root = tmp_path / "dataset"
    _create(root, episode_count=1, samples_per_episode=1)
    array = np.load(
        _first_episode(root) / f"{field_name}.npy",
        mmap_mode="r+",
        allow_pickle=False,
    )
    mutate(array)
    array.flush()
    with pytest.raises(JointBEVVerificationError, match=match):
        verify_joint_bev_dataset(root)


def test_verifier_rejects_episode_present_in_two_splits(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    _create(root, episode_count=1, samples_per_episode=1)
    source = _first_episode(root)
    destination = root / "val" / "episodes" / source.name
    shutil.copytree(source, destination)
    manifest_path = root / "val" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_manifest = json.loads(
        (root / "train" / "manifest.json").read_text(encoding="utf-8")
    )
    manifest["episodes"] = [train_manifest["episodes"][0]]
    manifest["episode_count"] = 1
    manifest["joint_samples"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(JointBEVVerificationError, match="appears in both"):
        verify_joint_bev_dataset(root)
