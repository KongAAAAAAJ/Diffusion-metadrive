from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

from metadrive.policy.diffusion_policy.transfuser_agent import TransfuserAgent
from metadrive.policy.diffusion_policy.transfuser_callback import render_open_loop_prediction
from metadrive.policy.diffusion_policy.transfuser_config import TransfuserConfig, build_transfuser_config
from metadrive.policy.diffusion_policy.transfuser_features import MetaDriveTransfuserDataset
from metadrive.policy.diffusion_policy.run_dir_utils import create_numbered_run_dir
from metadrive.policy.diffusion_policy.verify_transfuser_dataset import AUTO_DATASET_FORMAT, verify_dataset


DEFAULT_DATASET_ROOT = "/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metadrive_ppo_preprocessed_small_dir"
DEFAULT_PLAN_ANCHOR_PATH = "metadrive/exp_dataset/metadrive_anchors.npy"
DEFAULT_OUTPUT_DIR = "/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion/open_loop_eval"

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Open-loop evaluation for MetaDrive TransFuser.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--model-size", type=str, default="auto")
    parser.add_argument("--dataset-format", type=str, default=AUTO_DATASET_FORMAT)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-images", type=int, choices=(0, 1), default=1)
    parser.add_argument("--save-json", type=int, choices=(0, 1), default=1)
    parser.add_argument("--save-trajectory-plots", type=int, choices=(0, 1), default=1)
    parser.add_argument("--save-csv", type=int, choices=(0, 1), default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--plan-anchor-path", type=str, default=DEFAULT_PLAN_ANCHOR_PATH)
    parser.add_argument("--overlay-all-anchors", type=int, choices=(0, 1), default=1)
    parser.add_argument("--mode-focus", type=int, default=7)
    parser.add_argument("--verify-dataset-before-eval", type=int, choices=(0, 1), default=1)
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _extract_state_dict(checkpoint_obj):
    return checkpoint_obj.get("state_dict", checkpoint_obj)


def infer_model_size_from_checkpoint(checkpoint_path: Path) -> str:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    for key in (
        "_transfuser_model._query_embedding.weight",
        "agent._transfuser_model._query_embedding.weight",
        "_query_embedding.weight",
    ):
        if key in state_dict:
            query_embedding = state_dict[key]
            if tuple(query_embedding.shape) == (31, 256):
                return "base"
            if tuple(query_embedding.shape) == (17, 128):
                return "small"
    raise ValueError(f"Unable to infer model size from checkpoint: {checkpoint_path}")


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def build_eval_dataloader(config: TransfuserConfig, split: str, num_samples: int, num_workers: int) -> tuple[MetaDriveTransfuserDataset, DataLoader]:
    dataset = MetaDriveTransfuserDataset(
        config.dataset_root,
        config,
        split=split,
        max_samples=(num_samples if num_samples > 0 else None),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=max(1, config.batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, dataloader


def compute_ade_fde(pred_traj: np.ndarray, gt_traj: np.ndarray) -> tuple[float, float]:
    pred_xy = np.asarray(pred_traj[:, :2], dtype=np.float64)
    gt_xy = np.asarray(gt_traj[:, :2], dtype=np.float64)
    l2 = np.linalg.norm(pred_xy - gt_xy, axis=-1)
    return float(l2.mean()), float(l2[-1])


def save_trajectory_comparison_plot(
    pred_traj: np.ndarray,
    gt_traj: np.ndarray,
    output_path: Path,
    sample_index: int,
    mode_idx: Optional[int],
    ade: float,
    fde: float,
) -> None:
    pred_xy = np.asarray(pred_traj[:, :2], dtype=np.float64)
    gt_xy = np.asarray(gt_traj[:, :2], dtype=np.float64)
    plt.figure(figsize=(5, 2))
    plt.plot(gt_xy[:, 0], gt_xy[:, 1], marker="o", linewidth=2, label="GT")
    plt.plot(pred_xy[:, 0], pred_xy[:, 1], marker="o", linewidth=2, label="Pred")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"sample={sample_index} mode={mode_idx} ADE={ade:.3f} FDE={fde:.3f}")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_open_loop_csv(records: List[Dict], output_path: Path) -> None:
    fieldnames = [
        "sample_index",
        "shard_name",
        "local_index",
        "mode_idx",
        "trajectory_mode",
        "pred_final_x",
        "pred_final_y",
        "gt_final_x",
        "gt_final_y",
        "signed_final_y_error",
        "trajectory_l1",
        "trajectory_final_l2",
        "ade",
        "fde",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sample_index": record["sample_index"],
                    "shard_name": record["shard_name"],
                    "local_index": record["local_index"],
                    "mode_idx": record["mode_idx"],
                    "trajectory_mode": record.get("trajectory_mode"),
                    "pred_final_x": record["pred_final_xy"][0],
                    "pred_final_y": record["pred_final_xy"][1],
                    "gt_final_x": record["gt_final_xy"][0],
                    "gt_final_y": record["gt_final_xy"][1],
                    "signed_final_y_error": record["signed_final_y_error"],
                    "trajectory_l1": record["trajectory_l1"],
                    "trajectory_final_l2": record["trajectory_final_l2"],
                    "ade": record["ade"],
                    "fde": record["fde"],
                }
            )


def summarize_open_loop_records(records: List[Dict]) -> Dict[str, object]:
    if not records:
        return {
            "num_samples": 0,
            "trajectory_l1": 0.0,
            "trajectory_final_l2": 0.0,
            "ade_mean": 0.0,
            "fde_mean": 0.0,
            "signed_final_y_error": 0.0,
            "mean_pred_final_y": 0.0,
            "mean_gt_final_y": 0.0,
            "rightward_bias_fraction": 0.0,
            "mode_hist": {},
            "mode_final_y_mean": {},
            "trajectory_mode_hist": {},
        }
    traj_l1 = np.asarray([record["trajectory_l1"] for record in records], dtype=np.float64)
    final_l2 = np.asarray([record["trajectory_final_l2"] for record in records], dtype=np.float64)
    ade = np.asarray([record["ade"] for record in records], dtype=np.float64)
    fde = np.asarray([record["fde"] for record in records], dtype=np.float64)
    signed_y = np.asarray([record["signed_final_y_error"] for record in records], dtype=np.float64)
    pred_y = np.asarray([record["pred_final_xy"][1] for record in records], dtype=np.float64)
    gt_y = np.asarray([record["gt_final_xy"][1] for record in records], dtype=np.float64)
    mode_hist = Counter(int(record["mode_idx"]) for record in records if record["mode_idx"] is not None)
    trajectory_mode_hist = Counter(int(record["trajectory_mode"]) for record in records if record.get("trajectory_mode") is not None)
    mode_y = defaultdict(list)
    for record in records:
        if record["mode_idx"] is not None:
            mode_y[int(record["mode_idx"])].append(float(record["pred_final_xy"][1]))
    return {
        "num_samples": len(records),
        "trajectory_l1": float(traj_l1.mean()),
        "trajectory_final_l2": float(final_l2.mean()),
        "ade_mean": float(ade.mean()),
        "fde_mean": float(fde.mean()),
        "signed_final_y_error": float(signed_y.mean()),
        "mean_pred_final_y": float(pred_y.mean()),
        "mean_gt_final_y": float(gt_y.mean()),
        "rightward_bias_fraction": float((pred_y > 0).mean()),
        "mode_hist": {str(key): int(value) for key, value in sorted(mode_hist.items())},
        "mode_final_y_mean": {str(key): float(np.mean(value)) for key, value in sorted(mode_y.items())},
        "trajectory_mode_hist": {str(key): int(value) for key, value in sorted(trajectory_mode_hist.items())},
    }


def evaluate_open_loop(
    model,
    dataset: MetaDriveTransfuserDataset,
    dataloader: DataLoader,
    device: torch.device,
    config: TransfuserConfig,
    output_dir: Path,
    save_images: bool,
    save_json: bool,
    save_trajectory_plots: bool,
    save_csv: bool,
    overlay_all_anchors: bool,
    mode_focus: int,
) -> Dict[str, object]:
    image_dir = output_dir / "images"
    mode_dir = output_dir / f"mode_{mode_focus}"
    trajectory_plot_dir = output_dir / "trajectory_plots"
    if save_images:
        image_dir.mkdir(parents=True, exist_ok=True)
        mode_dir.mkdir(parents=True, exist_ok=True)
    if save_trajectory_plots:
        trajectory_plot_dir.mkdir(parents=True, exist_ok=True)

    anchors = None
    anchor_path = Path(config.plan_anchor_path)
    if anchor_path.exists():
        anchors = np.load(anchor_path)

    records: List[Dict] = []
    global_index = 0
    model.eval()
    for batch_idx, batch in enumerate(dataloader):
        features, targets = batch
        batch_size = features["camera_feature"].shape[0]
        features_device = _to_device(features, device)
        targets_device = _to_device(targets, device)
        with torch.no_grad():
            predictions = model(features_device, targets_device)
        predictions_cpu = {key: value.detach().cpu() for key, value in predictions.items()}
        features_cpu = {key: value.detach().cpu() for key, value in features.items()}
        targets_cpu = {key: value.detach().cpu() for key, value in targets.items()}

        for in_batch_idx in range(batch_size):
            metadata = dataset.get_sample_metadata(global_index)
            pred_traj = predictions_cpu["trajectory"][in_batch_idx].numpy()
            gt_traj = targets_cpu["trajectory"][in_batch_idx].numpy()
            mode_idx = None
            topk_mode_logits = []
            if "trajectory_mode_idx" in predictions_cpu:
                mode_idx = int(predictions_cpu["trajectory_mode_idx"][in_batch_idx].item())
            if "trajectory_mode_logits" in predictions_cpu:
                logits = predictions_cpu["trajectory_mode_logits"][in_batch_idx].numpy()
                topk_indices = np.argsort(logits)[::-1][: min(3, logits.shape[0])]
                topk_mode_logits = [
                    {"mode_idx": int(idx), "logit": float(logits[idx])}
                    for idx in topk_indices
                ]
            ade, fde = compute_ade_fde(pred_traj, gt_traj)

            record = {
                **metadata,
                "mode_idx": mode_idx,
                "trajectory_mode": metadata.get("trajectory_mode"),
                "pred_final_xy": [float(pred_traj[-1, 0]), float(pred_traj[-1, 1])],
                "gt_final_xy": [float(gt_traj[-1, 0]), float(gt_traj[-1, 1])],
                "pred_trajectory_xy": np.asarray(pred_traj[:, :2], dtype=np.float64).tolist(),
                "gt_trajectory_xy": np.asarray(gt_traj[:, :2], dtype=np.float64).tolist(),
                "ade": ade,
                "fde": fde,
                "signed_final_y_error": float(pred_traj[-1, 1] - gt_traj[-1, 1]),
                "trajectory_l1": float(np.abs(pred_traj - gt_traj).mean()),
                "trajectory_final_l2": float(np.linalg.norm(pred_traj[-1, :2] - gt_traj[-1, :2])),
                "topk_mode_logits": topk_mode_logits,
            }
            records.append(record)

            if save_trajectory_plots:
                plot_name = f"{metadata['sample_index']:05d}_mode{mode_idx if mode_idx is not None else 'na'}_traj.png"
                save_trajectory_comparison_plot(
                    pred_traj=pred_traj,
                    gt_traj=gt_traj,
                    output_path=trajectory_plot_dir / plot_name,
                    sample_index=metadata["sample_index"],
                    mode_idx=mode_idx,
                    ade=ade,
                    fde=fde,
                )

            if save_images:
                single_features = {key: value[in_batch_idx:in_batch_idx + 1] for key, value in features_cpu.items()}
                single_targets = {key: value[in_batch_idx:in_batch_idx + 1] for key, value in targets_cpu.items()}
                single_predictions = {key: value[in_batch_idx:in_batch_idx + 1] for key, value in predictions_cpu.items()}
                metadata_text = [
                    f"sample={metadata['sample_index']} shard={metadata['shard_name']} local={metadata['local_index']}",
                    f"y_err={record['signed_final_y_error']:+.3f} l1={record['trajectory_l1']:.3f} final_l2={record['trajectory_final_l2']:.3f}",
                ]
                image = render_open_loop_prediction(
                    features=single_features,
                    targets=single_targets,
                    predictions=single_predictions,
                    config=config,
                    sample_idx=0,
                    anchors=anchors,
                    overlay_all_anchors=overlay_all_anchors,
                    metadata_text=metadata_text,
                )
                image_name = f"{metadata['sample_index']:05d}_mode{mode_idx if mode_idx is not None else 'na'}.png"
                cv2.imwrite(str(image_dir / image_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                if mode_idx == mode_focus:
                    cv2.imwrite(str(mode_dir / image_name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            global_index += 1
        print(f"[open_loop] processed_batches={batch_idx + 1} processed_samples={global_index}")

    metrics = summarize_open_loop_records(records)
    summary = {
        "checkpoint": getattr(model, "_checkpoint_path", None),
        "dataset_root": str(dataset.dataset_root),
        "split": dataset.split,
        "num_samples": len(records),
        "metrics": metrics,
        "mode_hist": metrics["mode_hist"],
        "output_dir": str(output_dir),
    }
    if save_json:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "open_loop_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (output_dir / "open_loop_samples.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    if save_csv:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_open_loop_csv(records, output_dir / "open_loop_samples.csv")
    return summary


def main():
    args = parse_args()
    output_root = Path(args.output_dir)
    output_dir = create_numbered_run_dir(output_root)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    resolved_model_size = infer_model_size_from_checkpoint(checkpoint_path) if args.model_size == "auto" else args.model_size
    config = build_transfuser_config(
        resolved_model_size,
        dataset_root=args.dataset_root or TransfuserConfig().dataset_root,
        plan_anchor_path=args.plan_anchor_path or TransfuserConfig().plan_anchor_path,
        batch_size=args.batch_size,
        cache_shards_in_memory=False,
    )
    device = resolve_device(args.device)
    if bool(args.verify_dataset_before_eval):
        verify_dataset(
            dataset_root=Path(config.dataset_root),
            split=args.split,
            model_size=resolved_model_size,
            dataset_format=args.dataset_format,
            repair_corrupt_shards=False,
        )
    dataset, dataloader = build_eval_dataloader(
        config=config,
        split=args.split,
        num_samples=args.num_samples,
        num_workers=args.num_workers,
    )
    model = TransfuserAgent(config=config, checkpoint_path=str(checkpoint_path))
    model._checkpoint_path = str(checkpoint_path)
    model.eval()
    model.to(device)
    summary = evaluate_open_loop(
        model=model,
        dataset=dataset,
        dataloader=dataloader,
        device=device,
        config=config,
        output_dir=output_dir,
        save_images=bool(args.save_images),
        save_json=bool(args.save_json),
        save_trajectory_plots=bool(args.save_trajectory_plots),
        save_csv=bool(args.save_csv),
        overlay_all_anchors=bool(args.overlay_all_anchors),
        mode_focus=args.mode_focus,
    )
    print(f"[open_loop] summary={summary['metrics']}")
    print(f"[open_loop] output_dir={summary['output_dir']}")


if __name__ == "__main__":
    main()
