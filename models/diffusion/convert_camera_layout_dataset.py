from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


CAMERA_KEYS = ("left_camera", "front_camera", "right_camera")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert raw MetaDrive camera shards from CHW layout to HWC layout.")
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--max-shards", type=int, default=0)
    return parser.parse_args()


def chw_to_hwc(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(f"Expected 3D camera image, got shape {tuple(image.shape)}")
    if image.shape[0] not in (1, 3, 4):
        raise ValueError(f"Expected CHW image with channel-first shape, got {tuple(image.shape)}")
    return np.moveaxis(image, 0, -1)


def preprocess_shard(shard_path: Path, output_path: Path) -> int:
    with np.load(shard_path, allow_pickle=False) as raw:
        shard = {key: raw[key] for key in raw.files}

    converted = {}
    sample_count = None
    for key, value in shard.items():
        if key in CAMERA_KEYS:
            converted[key] = np.stack([chw_to_hwc(sample) for sample in value], axis=0)
        else:
            converted[key] = value
        if sample_count is None:
            sample_count = int(converted[key].shape[0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **converted)
    return sample_count or 0


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    shard_paths = sorted((input_root / "shards").glob("*.npz"))
    if args.max_shards > 0:
        shard_paths = shard_paths[:args.max_shards]
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found under {input_root / 'shards'}")

    total_samples = 0
    for shard_idx, shard_path in enumerate(shard_paths, start=1):
        output_path = output_root / "shards" / shard_path.name
        num_samples = preprocess_shard(shard_path, output_path)
        total_samples += num_samples
        print(f"[convert_camera_layout] {shard_idx}/{len(shard_paths)} {shard_path.name} samples={num_samples}")

    split_dir = input_root / "splits"
    if split_dir.exists():
        shutil.copytree(split_dir, output_root / "splits", dirs_exist_ok=True)

    reports_dir = input_root / "reports"
    if reports_dir.exists():
        shutil.copytree(reports_dir, output_root / "reports", dirs_exist_ok=True)

    manifest = {
        "source_dataset": str(input_root),
        "num_shards": len(shard_paths),
        "num_samples": total_samples,
        "camera_layout": "HWC",
        "converted_keys": list(CAMERA_KEYS),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "camera_layout_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[convert_camera_layout] output_root={output_root} total_samples={total_samples}")


if __name__ == "__main__":
    main()
