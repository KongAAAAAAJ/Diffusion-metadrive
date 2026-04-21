import argparse
from pathlib import Path
from typing import Optional
import re

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader

from metadrive.policy.diffusion_policy.transfuser_agent import TransfuserAgent
from metadrive.policy.diffusion_policy.transfuser_config import (
    TransfuserConfig,
    build_transfuser_config,
    diffusion_model_config_to_overrides,
    load_diffusion_model_config,
    resolve_model_config_value,
)
from metadrive.policy.diffusion_policy.transfuser_features import MetaDriveTransfuserDataset
from metadrive.policy.diffusion_policy.verify_transfuser_dataset import (
    AUTO_DATASET_FORMAT,
    PROCESSED_DIR_DATASET_FORMAT,
    PROCESSED_NPZ_LEGACY_DATASET_FORMAT,
    resolve_source_dataset_root,
    resolve_split_shards,
    verify_dataset,
)


DEFAULT_DATASET_ROOT = "/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/metadrive_ppo"
DEFAULT_PLAN_ANCHOR_PATH = "metadrive/exp_dataset/metadrive_anchors.npy"
DEFAULT_MODEL_CONFIG_PATH = "configs/diffusion/model.yaml"
RUN_DIR_PATTERN = re.compile(r"^run_(\d+)$")


def build_dataloader(config: TransfuserConfig, split: str, shuffle: bool) -> DataLoader:
    dataset = MetaDriveTransfuserDataset(config.dataset_root, config, split=split)
    pin_memory = torch.cuda.is_available() if config.pin_memory is None else config.pin_memory
    dataloader_kwargs = dict(
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    if config.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = config.persistent_workers
        dataloader_kwargs["prefetch_factor"] = config.prefetch_factor
    return DataLoader(
        dataset,
        **dataloader_kwargs,
    )


def validate_runtime_paths(config: TransfuserConfig) -> None:
    dataset_root = Path(config.dataset_root)
    shard_dir = dataset_root / "shards"
    split_dir = dataset_root / "splits"
    plan_anchor_path = Path(config.plan_anchor_path)

    if not shard_dir.exists():
        raise FileNotFoundError(
            f"Dataset root is invalid: {dataset_root}. Missing shard directory: {shard_dir}"
        )
    if not any(shard_dir.iterdir()):
        raise FileNotFoundError(f"No shard files found under: {shard_dir}")
    if not config.use_dynamic_anchors and not plan_anchor_path.exists():
        raise FileNotFoundError(f"Plan anchor file not found: {plan_anchor_path}")

    train_split = split_dir / f"{config.train_split}.txt"
    val_split = split_dir / f"{config.val_split}.txt"
    train_count = len(train_split.read_text(encoding="utf-8").splitlines()) if train_split.exists() else 0
    val_count = len(val_split.read_text(encoding="utf-8").splitlines()) if val_split.exists() else 0
    if train_count <= 0:
        raise FileNotFoundError(f"Train split is empty or missing: {train_split}")
    if val_count <= 0:
        raise FileNotFoundError(f"Validation split is empty or missing: {val_split}")

    print(f"[train] dataset_root={dataset_root}")
    print(f"[train] train_shards={train_count} val_shards={val_count}")
    print(f"[train] anchor_method={'dynamic' if config.use_dynamic_anchors else 'k_means'}")
    if config.use_dynamic_anchors:
        print(f"[train] plan_anchor_path={plan_anchor_path} (optional in dynamic mode)")
    else:
        print(f"[train] plan_anchor_path={plan_anchor_path}")


def verify_runtime_dataset(
    config: TransfuserConfig,
    model_size: str,
    dataset_format: str = AUTO_DATASET_FORMAT,
    source_dataset_root: Optional[str] = None,
    repair_corrupt_shards: bool = True,
    fail_on_repair_error: bool = True,
) -> dict:
    dataset_root = Path(config.dataset_root)
    train_shards = resolve_split_shards(dataset_root, config.train_split)
    val_shards = resolve_split_shards(dataset_root, config.val_split)
    selected_shards = train_shards + [path for path in val_shards if path not in train_shards]
    resolved_source_root = resolve_source_dataset_root(dataset_root, source_dataset_root)
    processed_formats = (PROCESSED_DIR_DATASET_FORMAT, PROCESSED_NPZ_LEGACY_DATASET_FORMAT)
    effective_repair_request = bool(repair_corrupt_shards and (dataset_format == AUTO_DATASET_FORMAT or dataset_format in processed_formats))
    if dataset_format not in (AUTO_DATASET_FORMAT,) + processed_formats and repair_corrupt_shards:
        print(
            "[train] repair_corrupt_shards requested for non-processed dataset; "
            "ignoring repair and running verification only"
        )
    report = verify_dataset(
        dataset_root=dataset_root,
        split="all",
        model_size=model_size,
        dataset_format=dataset_format,
        repair_corrupt_shards=effective_repair_request,
        source_dataset_root=resolved_source_root,
        fail_on_repair_error=fail_on_repair_error,
        shard_paths=selected_shards,
        split_label=f"{config.train_split}+{config.val_split}",
    )
    print(f"[train] dataset_format={report['dataset_format']}")
    print(f"[train] integrity_summary={report['summary']}")
    print(f"[train] integrity_report={report['report_path']}")
    return report


def resolve_precision(precision: str) -> str:
    if precision == "auto":
        return "16-mixed" if torch.cuda.is_available() else "32-true"
    return precision


def create_next_run_dir(output_root: Path) -> Path:
    existing_nums = []
    if output_root.exists():
        for path in output_root.iterdir():
            m = RUN_DIR_PATTERN.match(path.name)
            if m and path.is_dir():
                existing_nums.append(int(m.group(1)))
    next_num = max(existing_nums) + 1 if existing_nums else 1
    run_dir = output_root / f"run_{next_num}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Train MetaDrive Diffusion Policy.")
    parser.add_argument("--model-config-path", type=str, default=DEFAULT_MODEL_CONFIG_PATH)
    parser.add_argument("--model-size", type=str, default=None)  # “small” or “base”，决定模型规模
    parser.add_argument("--dataset-root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--plan-anchor-path", type=str, default=None)
    parser.add_argument("--anchor-method", type=str, choices=("k_means", "dynamic"), default=None)
    parser.add_argument("--trajectory-reg-decoder-type", type=str, choices=("mlp", "gru"), default=None)
    parser.add_argument("--target-guidance-type", type=str, choices=("point", "line"), default=None)
    parser.add_argument("--target-line-num-points", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--persistent-workers", type=int, default=1)  # worker 进程复用，会多占内存
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--cache-shards-in-memory", action="store_true")  # 把 .npz shard 预先整块读进内存，后续训练直接从内存取
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--precision", type=str, default="auto")    # 混合精度训练，更快但可能不稳定
    parser.add_argument("--check-val-every-n-epoch", type=int, default=1)
    parser.add_argument("--val-visualization-interval", type=int, default=5)
    parser.add_argument("--enable-val-visualization", action="store_true")
    parser.add_argument("--dataset-format", type=str, default=AUTO_DATASET_FORMAT)
    parser.add_argument("--verify-dataset-before-train", type=int, default=1)
    parser.add_argument("--repair-corrupt-shards", type=int, default=1)
    parser.add_argument("--source-dataset-root", type=str, default=None)
    parser.add_argument("--fail-on-repair-error", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="/media/kong/Elements_SE/Diffusion_Data/outputs/diffusion")
    parser.add_argument("--max-steps", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    model_config = load_diffusion_model_config(args.model_config_path)
    model_size = resolve_model_config_value(args.model_size, model_config, "model_size", "small")
    config_overrides = diffusion_model_config_to_overrides(model_config)
    if args.plan_anchor_path is not None:
        config_overrides["plan_anchor_path"] = args.plan_anchor_path
    if args.anchor_method is not None:
        config_overrides["use_dynamic_anchors"] = args.anchor_method == "dynamic"
    if args.trajectory_reg_decoder_type is not None:
        config_overrides["trajectory_reg_decoder_type"] = args.trajectory_reg_decoder_type
    if args.target_guidance_type is not None:
        config_overrides["target_guidance_type"] = args.target_guidance_type
    if args.target_line_num_points is not None:
        config_overrides["target_line_num_points"] = args.target_line_num_points
    config_overrides.update(
        dataset_root=args.dataset_root or TransfuserConfig().dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=bool(args.persistent_workers),
        prefetch_factor=args.prefetch_factor,
        cache_shards_in_memory=args.cache_shards_in_memory,
        max_epochs=args.max_epochs,
        precision=args.precision,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        val_visualization_interval=args.val_visualization_interval,
        enable_val_visualization=args.enable_val_visualization,
    )
    config = build_transfuser_config(model_size, **config_overrides)
    validate_runtime_paths(config)
    if bool(args.verify_dataset_before_train):
        verify_runtime_dataset(
            config=config,
            model_size=model_size,
            dataset_format=args.dataset_format,
            source_dataset_root=args.source_dataset_root,
            repair_corrupt_shards=bool(args.repair_corrupt_shards),
            fail_on_repair_error=bool(args.fail_on_repair_error),
        )
    resolved_precision = resolve_precision(config.precision)
    print(f"[train] model_config_path={args.model_config_path}")
    print(f"[train] model_size={config.model_size}")
    print(
        "[train] dataloader "
        f"batch_size={config.batch_size} num_workers={config.num_workers} "
        f"persistent_workers={config.persistent_workers and config.num_workers > 0} "
        f"prefetch_factor={config.prefetch_factor if config.num_workers > 0 else 'n/a'} "
        f"cache_shards_in_memory={config.cache_shards_in_memory}"
    )
    print(
        "[train] runtime "
        f"precision={resolved_precision} camera=({config.camera_height}, {config.camera_width}) "
        f"trajectory_reg_decoder_type={config.trajectory_reg_decoder_type} "
        f"target_guidance_type={config.target_guidance_type} "
        f"target_line_num_points={config.target_line_num_points} "
        f"check_val_every_n_epoch={config.check_val_every_n_epoch} "
        f"val_visualization_interval={config.val_visualization_interval} "
        f"enable_val_visualization={config.enable_val_visualization}"
    )

    model = TransfuserAgent(config=config, lr=args.lr, checkpoint_path=args.checkpoint)
    train_loader = build_dataloader(config, config.train_split, shuffle=True)
    val_loader = build_dataloader(config, config.val_split, shuffle=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = create_next_run_dir(output_dir)
    print(f"[train] run_dir={run_dir}")
    logger = TensorBoardLogger(save_dir=str(run_dir), name="tb")
    checkpoint = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        save_top_k=3,
        monitor="val/loss",
        mode="min",
        filename="diffusion-{epoch:02d}",
    )

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        max_steps=args.max_steps,
        logger=logger,
        callbacks=[checkpoint] + model.get_training_callbacks(),
        accelerator="auto",  # 优先选择GPU，如果没有则使用CPU
        devices=1,
        precision=resolved_precision,
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
