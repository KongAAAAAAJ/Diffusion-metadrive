from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from metadrive.policy.diffusion_policy.eval_transfuser_open_loop import (
    build_eval_dataloader,
    compute_ade_fde,
    evaluate_open_loop,
    summarize_open_loop_records,
)
from metadrive.policy.diffusion_policy.preprocess_transfuser_dataset import (
    OUTPUT_FORMAT_DIR,
    output_shard_path,
    preprocess_shard,
)
from metadrive.policy.diffusion_policy.transfuser_callback import render_open_loop_prediction
from metadrive.policy.diffusion_policy.transfuser_config import build_transfuser_config


def _write_split(dataset_root: Path, split: str, shard_names) -> None:
    split_path = dataset_root / "splits" / f"{split}.txt"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text("\n".join(shard_names) + "\n", encoding="utf-8")


def _raw_payload(num_samples: int = 2):
    return {
        "left_camera": np.zeros((num_samples, 10, 16, 3), dtype=np.uint8),
        "front_camera": np.zeros((num_samples, 10, 16, 3), dtype=np.uint8),
        "right_camera": np.zeros((num_samples, 10, 16, 3), dtype=np.uint8),
        "lidar": np.zeros((num_samples, 32), dtype=np.float32),
        "ego_state": np.zeros((num_samples, 19), dtype=np.float32),
        "trajectory": np.zeros((num_samples, 8, 3), dtype=np.float32),
        "agent_states": np.zeros((num_samples, 16, 5), dtype=np.float32),
        "agent_labels": np.zeros((num_samples, 16), dtype=bool),
        "bev_raster": np.zeros((num_samples, 3, 16, 16), dtype=np.uint8),
    }


def _write_npz(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


class _DummyOpenLoopModel:
    def eval(self):
        return self

    def to(self, _device):
        return self

    def __call__(self, features, targets):
        batch_size = features["camera_feature"].shape[0]
        pred_trajectory = targets["trajectory"].clone()
        pred_trajectory[:, -1, 1] += 0.2
        pred_bev = torch.nn.functional.one_hot(targets["bev_semantic_map"].long(), num_classes=4)
        pred_bev = pred_bev.permute(0, 3, 1, 2).float()
        return {
            "trajectory": pred_trajectory,
            "trajectory_mode_idx": torch.full((batch_size,), 7, dtype=torch.long, device=pred_trajectory.device),
            "trajectory_mode_logits": torch.nn.functional.one_hot(
                torch.full((batch_size,), 7, dtype=torch.long, device=pred_trajectory.device),
                num_classes=8,
            ).float(),
            "bev_semantic_map": pred_bev,
            "agent_states": targets["agent_states"].clone(),
            "agent_labels": torch.full_like(targets["agent_labels"].float(), -8.0),
        }


def test_render_open_loop_prediction_supports_missing_mode_idx():
    config = build_transfuser_config("small")
    features = {
        "camera_feature": torch.zeros((1, 3, 256, 768), dtype=torch.float32),
        "lidar_feature": torch.zeros((1, 1, 256, 256), dtype=torch.float32),
        "status_feature": torch.zeros((1, 8), dtype=torch.float32),
        "ego_state": torch.zeros((1, 8), dtype=torch.float32),
    }
    targets = {
        "trajectory": torch.zeros((1, 8, 3), dtype=torch.float32),
        "agent_states": torch.zeros((1, 16, 5), dtype=torch.float32),
        "agent_labels": torch.zeros((1, 16), dtype=torch.bool),
        "bev_semantic_map": torch.zeros((1, 128, 256), dtype=torch.int64),
    }
    predictions = {
        "trajectory": torch.zeros((1, 8, 3), dtype=torch.float32),
        "agent_states": torch.zeros((1, 16, 5), dtype=torch.float32),
        "agent_labels": torch.full((1, 16), -8.0, dtype=torch.float32),
        "bev_semantic_map": torch.zeros((1, 4, 128, 256), dtype=torch.float32),
    }

    image = render_open_loop_prediction(
        features=features,
        targets=targets,
        predictions=predictions,
        config=config,
        anchors=np.zeros((8, 8, 2), dtype=np.float32),
    )

    assert image.shape == (512, 1024, 3)


def test_summarize_open_loop_records_tracks_mode_hist():
    summary = summarize_open_loop_records([
        {
            "trajectory_l1": 0.1,
            "trajectory_final_l2": 0.2,
            "ade": 0.1,
            "fde": 0.2,
            "signed_final_y_error": 0.3,
            "pred_final_xy": [1.0, 0.5],
            "gt_final_xy": [1.0, 0.2],
            "mode_idx": 7,
            "trajectory_mode": 2,
        },
        {
            "trajectory_l1": 0.2,
            "trajectory_final_l2": 0.4,
            "ade": 0.2,
            "fde": 0.4,
            "signed_final_y_error": -0.1,
            "pred_final_xy": [1.0, -0.2],
            "gt_final_xy": [1.0, -0.1],
            "mode_idx": 3,
            "trajectory_mode": 2,
        },
    ])

    assert summary["mode_hist"] == {"3": 1, "7": 1}
    assert summary["trajectory_mode_hist"] == {"2": 2}
    assert "7" in summary["mode_final_y_mean"]
    assert summary["ade_mean"] == pytest.approx(0.15)
    assert summary["fde_mean"] == pytest.approx(0.3)


def test_compute_ade_fde_for_simple_offset():
    gt = np.zeros((4, 3), dtype=np.float32)
    pred = np.zeros((4, 3), dtype=np.float32)
    pred[:, 1] = 0.5

    ade, fde = compute_ade_fde(pred, gt)

    assert ade == pytest.approx(0.5)
    assert fde == pytest.approx(0.5)


def test_evaluate_open_loop_writes_summary_and_images(tmp_path: Path):
    dataset_root = tmp_path / "processed_dataset"
    raw_shard = tmp_path / "source" / "shards" / "shard_000000.npz"
    _write_npz(raw_shard, _raw_payload())
    processed_path = output_shard_path(dataset_root, raw_shard.name, OUTPUT_FORMAT_DIR)
    preprocess_shard(raw_shard, processed_path, build_transfuser_config("small"), output_format=OUTPUT_FORMAT_DIR)
    _write_split(dataset_root, "val", [raw_shard.stem])

    anchor_path = tmp_path / "anchors.npy"
    np.save(anchor_path, np.zeros((8, 8, 2), dtype=np.float32))
    config = build_transfuser_config(
        "small",
        dataset_root=str(dataset_root),
        plan_anchor_path=str(anchor_path),
        batch_size=1,
    )
    dataset, dataloader = build_eval_dataloader(config, split="val", num_samples=1, num_workers=0)

    summary = evaluate_open_loop(
        model=_DummyOpenLoopModel(),
        dataset=dataset,
        dataloader=dataloader,
        device=torch.device("cpu"),
        config=config,
        output_dir=tmp_path / "eval_output",
        save_images=True,
        save_json=True,
        save_trajectory_plots=True,
        save_csv=True,
        overlay_all_anchors=True,
        mode_focus=7,
    )

    assert summary["metrics"]["mode_hist"]["7"] == 1
    assert summary["metrics"]["ade_mean"] == pytest.approx(0.025)
    assert summary["metrics"]["fde_mean"] == pytest.approx(0.2)
    assert (tmp_path / "eval_output" / "open_loop_summary.json").exists()
    assert (tmp_path / "eval_output" / "open_loop_samples.json").exists()
    assert (tmp_path / "eval_output" / "open_loop_samples.csv").exists()
    assert list((tmp_path / "eval_output" / "images").glob("*.png"))
    assert list((tmp_path / "eval_output" / "trajectory_plots").glob("*.png"))
    assert list((tmp_path / "eval_output" / "mode_7").glob("*.png"))

    samples = json.loads((tmp_path / "eval_output" / "open_loop_samples.json").read_text(encoding="utf-8"))
    assert "pred_trajectory_xy" in samples[0]
    assert "gt_trajectory_xy" in samples[0]
    assert "ade" in samples[0]
    assert "fde" in samples[0]
    assert "trajectory_mode" in samples[0]
