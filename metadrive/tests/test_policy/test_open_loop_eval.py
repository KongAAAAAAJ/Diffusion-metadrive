from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from metadrive.policy.diffusion_policy.eval_transfuser_open_loop import (
    _build_even_scenario_subset_indices,
    _predict_open_loop,
    build_eval_dataloader,
    compute_ade_fde,
    evaluate_open_loop,
    save_trajectory_comparison_plot,
    summarize_open_loop_records,
)
from metadrive.policy.diffusion_policy.preprocess_transfuser_dataset import (
    OUTPUT_FORMAT_DIR,
    _write_output_splits,
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
        "scenario_id": np.asarray(["S1_free_cruise_straight"] * num_samples),
        "local_route": np.asarray(["R1_entry_straight"] * num_samples),
        "trajectory_mode": np.zeros((num_samples,), dtype=np.int8),
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
            "trajectory_candidates": torch.stack(
                [
                    pred_trajectory,
                    pred_trajectory + torch.tensor([[[0.0, 0.2, 0.0]]], dtype=pred_trajectory.dtype),
                    pred_trajectory + torch.tensor([[[0.0, -0.2, 0.0]]], dtype=pred_trajectory.dtype),
                ],
                dim=1,
            ),
            "trajectory_mode_idx": torch.full((batch_size,), 7, dtype=torch.long, device=pred_trajectory.device),
            "trajectory_mode_logits": torch.nn.functional.one_hot(
                torch.full((batch_size,), 7, dtype=torch.long, device=pred_trajectory.device),
                num_classes=8,
            ).float(),
            "bev_semantic_map": pred_bev,
            "agent_states": targets["agent_states"].clone(),
            "agent_labels": torch.full_like(targets["agent_labels"].float(), -8.0),
        }


class _DummyAgentWithInferMultimodal:
    def __init__(self):
        self._transfuser_model = self

    def eval(self):
        return self

    def to(self, _device):
        return self

    def infer_multimodal(self, features):
        batch_size = features["camera_feature"].shape[0]
        trajectory = torch.zeros((batch_size, 8, 3), dtype=torch.float32)
        return {
            "trajectory": trajectory,
            "trajectory_candidates": torch.zeros((batch_size, 3, 8, 3), dtype=torch.float32),
            "trajectory_mode_idx": torch.zeros((batch_size,), dtype=torch.long),
            "trajectory_mode_logits": torch.zeros((batch_size, 3), dtype=torch.float32),
            "bev_semantic_map": torch.zeros((batch_size, 4, 128, 256), dtype=torch.float32),
            "agent_states": torch.zeros((batch_size, 16, 5), dtype=torch.float32),
            "agent_labels": torch.zeros((batch_size, 16), dtype=torch.float32),
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


def test_predict_open_loop_prefers_infer_multimodal():
    model = _DummyAgentWithInferMultimodal()
    features = {"camera_feature": torch.zeros((1, 3, 256, 768), dtype=torch.float32)}
    targets = {"trajectory": torch.zeros((1, 8, 3), dtype=torch.float32)}

    predictions = _predict_open_loop(model, features, targets)

    assert "trajectory_candidates" in predictions


def test_save_trajectory_comparison_plot_draws_other_modes(tmp_path: Path):
    output_path = tmp_path / "plot.png"
    pred = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float32)
    gt = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.2, 0.0], [4.0, 0.2, 0.0]], dtype=np.float32)
    candidates = np.asarray(
        [
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[0.0, 0.1, 0.0], [2.0, 0.3, 0.0], [4.0, 0.3, 0.0]],
            [[0.0, -0.1, 0.0], [2.0, -0.3, 0.0], [4.0, -0.3, 0.0]],
        ],
        dtype=np.float32,
    )

    save_trajectory_comparison_plot(
        pred_traj=pred,
        gt_traj=gt,
        output_path=output_path,
        sample_index=0,
        pred_mode_idx=0,
        gt_mode_idx=1,
        pred_mode_name="KEEP_LANE_HIGH",
        gt_mode_name="KEEP_LANE_MEDIUM",
        ade=0.1,
        fde=0.2,
        target_point=np.asarray([4.0, 0.0], dtype=np.float32),
        trajectory_candidates=candidates,
    )

    assert output_path.exists()


def test_evaluate_open_loop_writes_summary_and_images(tmp_path: Path):
    dataset_root = tmp_path / "processed_dataset"
    raw_shard = tmp_path / "source" / "shards" / "shard_000000.npz"
    payload = _raw_payload()
    payload["trajectory_mode"][:] = np.asarray([7, 7], dtype=np.int8)
    payload["trajectory"][0, :, 0] = np.linspace(1.0, 8.0, 8, dtype=np.float32)
    payload["trajectory"][0, :, 1] = np.linspace(0.2, 1.6, 8, dtype=np.float32)
    _write_npz(raw_shard, payload)
    processed_path = output_shard_path(dataset_root, raw_shard.name, OUTPUT_FORMAT_DIR)
    preprocess_shard(raw_shard, processed_path, build_transfuser_config("small"), output_format=OUTPUT_FORMAT_DIR)
    _write_split(dataset_root, "val", [raw_shard.stem])

    processed_target_point = np.load(processed_path / "target_point.npy")
    np.testing.assert_allclose(processed_target_point[0], np.asarray([8.0, 1.6], dtype=np.float32))

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
    )

    assert summary["metrics"]["mode_hist"]["7"] == 1
    assert summary["metrics"]["mode_accuracy"] == pytest.approx(0.0)
    assert summary["metrics"]["gt_mode_hist"]["0"] == 1
    assert summary["metrics"]["ade_mean"] == pytest.approx(0.025)
    assert summary["metrics"]["fde_mean"] == pytest.approx(0.2)
    assert (tmp_path / "eval_output" / "open_loop_summary.json").exists()
    assert (tmp_path / "eval_output" / "open_loop_samples.json").exists()
    assert (tmp_path / "eval_output" / "open_loop_samples.csv").exists()
    assert list((tmp_path / "eval_output" / "images").glob("*.png"))
    assert list((tmp_path / "eval_output" / "trajectory_plots").glob("*.png"))
    assert not (tmp_path / "eval_output" / "mode_7").exists()

    samples = json.loads((tmp_path / "eval_output" / "open_loop_samples.json").read_text(encoding="utf-8"))
    assert "pred_trajectory_xy" in samples[0]
    assert "gt_trajectory_xy" in samples[0]
    assert "ade" in samples[0]
    assert "fde" in samples[0]
    assert "trajectory_mode" in samples[0]
    assert samples[0]["gt_mode_idx"] == 0
    assert samples[0]["pred_mode_idx"] == 7
    assert "gt_mode_name" in samples[0]
    assert "pred_mode_name" in samples[0]


def test_write_output_splits_normalizes_zero_padded_shard_names(tmp_path: Path):
    input_root = tmp_path / "raw"
    output_root = tmp_path / "processed"
    raw_shard = input_root / "shards" / "shard_000003.npz"
    _write_npz(raw_shard, _raw_payload(num_samples=1))
    _write_split(input_root, "val", ["shard_00003"])

    _write_output_splits(input_root, output_root, OUTPUT_FORMAT_DIR)

    val_lines = (output_root / "splits" / "val.txt").read_text(encoding="utf-8").splitlines()
    assert val_lines == ["shard_000003"]


def test_build_even_scenario_subset_indices_balances_across_scenarios():
    metadata = [
        {"sample_index": 0, "scenario_id": "S1"},
        {"sample_index": 1, "scenario_id": "S1"},
        {"sample_index": 2, "scenario_id": "S1"},
        {"sample_index": 3, "scenario_id": "S2"},
        {"sample_index": 4, "scenario_id": "S2"},
        {"sample_index": 5, "scenario_id": "S3"},
    ]

    indices = _build_even_scenario_subset_indices(metadata, num_samples=5)

    assert indices == [0, 3, 5, 1, 4]


def test_build_eval_dataloader_uses_even_scenario_sampling(tmp_path: Path):
    dataset_root = tmp_path / "processed_dataset"
    raw_shard = tmp_path / "source" / "shards" / "shard_000000.npz"
    payload = _raw_payload(num_samples=6)
    payload["scenario_id"] = np.asarray(["S1", "S1", "S1", "S2", "S2", "S3"])
    payload["local_route"] = np.asarray(["R1", "R1", "R1", "R2", "R2", "R3"])
    _write_npz(raw_shard, payload)
    processed_path = output_shard_path(dataset_root, raw_shard.name, OUTPUT_FORMAT_DIR)
    preprocess_shard(raw_shard, processed_path, build_transfuser_config("small"), output_format=OUTPUT_FORMAT_DIR)
    _write_split(dataset_root, "val", [raw_shard.stem])

    config = build_transfuser_config("small", dataset_root=str(dataset_root), batch_size=1)
    dataset, dataloader = build_eval_dataloader(config, split="val", num_samples=5, num_workers=0)

    assert len(dataset) == 5
    scenario_ids = [dataset.get_sample_metadata(i)["scenario_id"] for i in range(len(dataset))]
    assert scenario_ids == ["S1", "S2", "S3", "S1", "S2"]
    assert len(dataloader.dataset) == 5


def test_save_trajectory_comparison_plot_marks_target_point_and_modes(tmp_path: Path):
    from metadrive.policy.diffusion_policy.eval_transfuser_open_loop import save_trajectory_comparison_plot

    output_path = tmp_path / "traj.png"
    save_trajectory_comparison_plot(
        pred_traj=np.zeros((8, 3), dtype=np.float32),
        gt_traj=np.zeros((8, 3), dtype=np.float32),
        output_path=output_path,
        sample_index=3,
        pred_mode_idx=7,
        gt_mode_idx=4,
        pred_mode_name="LANE_CHANGE_RIGHT_MEDIUM",
        gt_mode_name="LANE_CHANGE_LEFT_MEDIUM",
        ade=0.1,
        fde=0.2,
        target_point=np.asarray([1.0, 2.0], dtype=np.float32),
    )

    assert output_path.exists()
