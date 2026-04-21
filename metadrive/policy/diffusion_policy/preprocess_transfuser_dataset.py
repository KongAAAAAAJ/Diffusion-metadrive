from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict
import uuid

import numpy as np

from metadrive.policy.diffusion_policy.transfuser_config import (
    build_transfuser_config,
    diffusion_model_config_to_overrides,
    load_diffusion_model_config,
    resolve_model_config_value,
)
from metadrive.policy.diffusion_policy.transfuser_features import sample_to_features_targets


OUTPUT_FORMAT_DIR = "dir"
OUTPUT_FORMAT_NPZ_LEGACY = "npz-legacy"
SUPPORTED_OUTPUT_FORMATS = (OUTPUT_FORMAT_DIR, OUTPUT_FORMAT_NPZ_LEGACY)
PROCESSED_DIRSET_FORMAT = "processed_dir"
PROCESSED_FIELDS = (
    "camera_feature",
    "lidar_feature",
    "status_feature",
    "ego_state",
    "target_point",
    "target_line",
    "topology_polyline",
    "trajectory",
    "agent_states",
    "agent_labels",
    "bev_semantic_map",
)
DEFAULT_MODEL_CONFIG_PATH = "configs/diffusion/model.yaml"
OPTIONAL_PASSTHROUGH_FIELDS = (
    "scenario_id",
    "local_route",
    "hierarchical_mode_label",
    "trajectory_mode",
    "trajectory_raw",
    "trajectory_correction_strength",
    "trajectory_mean_abs_lateral_before",
    "trajectory_mean_abs_lateral_after",
    "trajectory_final_abs_lateral_before",
    "trajectory_final_abs_lateral_after",
)


def _resolve_split_shard_name(raw_name: str, available_shards: Dict[str, Path]) -> str:
    normalized = raw_name.strip()
    if not normalized:
        raise ValueError("Split shard name must not be empty.")

    direct_keys = [normalized, Path(normalized).stem]
    for key in direct_keys:
        if key in available_shards:
            return available_shards[key].name

    stem = Path(normalized).stem
    if stem.startswith("shard_"):
        suffix = stem[len("shard_"):]
        if suffix.isdigit():
            numeric_suffix = int(suffix)
            for key, path in available_shards.items():
                key_stem = Path(key).stem
                if key_stem.startswith("shard_") and key_stem[len("shard_"):].isdigit():
                    if int(key_stem[len("shard_"):]) == numeric_suffix:
                        return path.name

    return normalized


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess MetaDrive dataset into TransFuser-ready tensors.")
    parser.add_argument("--input-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--model-config-path", type=str, default=DEFAULT_MODEL_CONFIG_PATH)
    parser.add_argument("--model-size", type=str, default=None)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--output-format", type=str, default=OUTPUT_FORMAT_DIR, choices=SUPPORTED_OUTPUT_FORMATS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", type=int, default=1)
    parser.add_argument(
        "--only-shards",
        nargs="+",
        default=None,
        help="Only preprocess the listed shard filenames, e.g. shard_000002.npz shard_000008.npz",
    )
    return parser.parse_args()


def tensor_to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _write_json_atomic(path: Path, payload: Dict) -> None:
    temp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temp_path.replace(path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    temp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp.npy"
    try:
        np.save(temp_path, array)
        temp_path.replace(path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def save_npz_atomic(output_path: Path, payload) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.parent / f".{output_path.stem}.{uuid.uuid4().hex}.tmp.npz"
    try:
        np.savez(temp_path, **payload)
        temp_path.replace(output_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def save_processed_dir_atomic(output_dir: Path, payload: Dict[str, np.ndarray], model_size: str) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir.parent / f".{output_dir.name}.{uuid.uuid4().hex}.tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=False)

    try:
        fields_meta = {}
        num_samples = None
        for field_name, value in payload.items():
            array = np.asarray(value)
            _save_npy_atomic(temp_dir / f"{field_name}.npy", array)
            fields_meta[field_name] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
            if num_samples is None:
                num_samples = int(array.shape[0])

        meta = {
            "format_version": 1,
            "dataset_format": PROCESSED_DIRSET_FORMAT,
            "storage_format": OUTPUT_FORMAT_DIR,
            "num_samples": num_samples or 0,
            "model_size": model_size,
            "fields": fields_meta,
        }
        _write_json_atomic(temp_dir / "meta.json", meta)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temp_dir.replace(output_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


def build_processed_payload(shard_path: Path, config) -> Dict[str, np.ndarray]:
    with np.load(shard_path, allow_pickle=False) as raw:
        shard = {key: raw[key] for key in raw.files}

    num_samples = int(shard["trajectory"].shape[0])
    processed = {field: [] for field in PROCESSED_FIELDS}
    hierarchical_mode_labels = []

    for sample_idx in range(num_samples):
        sample = {key: shard[key][sample_idx] for key in shard.keys()}
        features, targets = sample_to_features_targets(sample, config)
        processed["camera_feature"].append(tensor_to_numpy(features["camera_feature"]).astype(np.float16))
        processed["lidar_feature"].append(tensor_to_numpy(features["lidar_feature"]).astype(np.float16))
        processed["status_feature"].append(tensor_to_numpy(features["status_feature"]).astype(np.float32))
        processed["ego_state"].append(tensor_to_numpy(features["ego_state"]).astype(np.float32))
        processed["target_point"].append(tensor_to_numpy(features["target_point"]).astype(np.float32))
        processed["target_line"].append(tensor_to_numpy(features["target_line"]).astype(np.float32))
        processed["topology_polyline"].append(tensor_to_numpy(targets["topology_polyline"]).astype(np.float32))
        processed["trajectory"].append(tensor_to_numpy(targets["trajectory"]).astype(np.float32))
        processed["agent_states"].append(tensor_to_numpy(targets["agent_states"]).astype(np.float32))
        processed["agent_labels"].append(tensor_to_numpy(targets["agent_labels"]).astype(bool))
        processed["bev_semantic_map"].append(tensor_to_numpy(targets["bev_semantic_map"]).astype(np.uint8))
        if "hierarchical_mode_label" in targets:
            hierarchical_mode_labels.append(int(tensor_to_numpy(targets["hierarchical_mode_label"]).reshape(-1)[0]))

    payload = {key: np.stack(values, axis=0) for key, values in processed.items()}
    if hierarchical_mode_labels:
        payload["hierarchical_mode_label"] = np.asarray(hierarchical_mode_labels, dtype=np.int8)
    for field in OPTIONAL_PASSTHROUGH_FIELDS:
        if field in shard:
            payload[field] = np.asarray(shard[field])
    return payload


def output_shard_path(output_root: Path, shard_name: str, output_format: str) -> Path:
    if output_format == OUTPUT_FORMAT_DIR:
        return output_root / "shards" / Path(shard_name).stem
    if output_format == OUTPUT_FORMAT_NPZ_LEGACY:
        return output_root / "shards" / shard_name
    raise ValueError(f"Unsupported output format: {output_format}")


def preprocess_shard(
    shard_path: Path,
    output_path: Path,
    config,
    output_format: str = OUTPUT_FORMAT_DIR,
    overwrite: bool = False,
) -> int:
    if output_path.exists():
        if overwrite:
            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                output_path.unlink()
        else:
            raise FileExistsError(f"Output shard already exists: {output_path}")

    payload = build_processed_payload(shard_path, config)
    if output_format == OUTPUT_FORMAT_DIR:
        save_processed_dir_atomic(output_path, payload, config.model_size)
    elif output_format == OUTPUT_FORMAT_NPZ_LEGACY:
        save_npz_atomic(output_path, payload)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    return int(payload["trajectory"].shape[0])


def _write_output_splits(input_root: Path, output_root: Path, output_format: str) -> None:
    split_dir = input_root / "splits"
    if not split_dir.exists():
        return
    output_split_dir = output_root / "splits"
    output_split_dir.mkdir(parents=True, exist_ok=True)
    available_input_shards = {}
    for shard_path in sorted((input_root / "shards").glob("*.npz")):
        available_input_shards[shard_path.name] = shard_path
        available_input_shards[shard_path.stem] = shard_path
    for split_path in sorted(split_dir.glob("*.txt")):
        shard_names = []
        for raw_name in split_path.read_text(encoding="utf-8").splitlines():
            if not raw_name.strip():
                continue
            resolved_input_name = _resolve_split_shard_name(raw_name, available_input_shards)
            normalized_output_name = output_shard_path(output_root, resolved_input_name, output_format).name
            shard_names.append(normalized_output_name)
        (output_split_dir / split_path.name).write_text("\n".join(shard_names) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    model_config = load_diffusion_model_config(args.model_config_path)
    model_size = resolve_model_config_value(args.model_size, model_config, "model_size", "small")
    config = build_transfuser_config(
        model_size,
        **diffusion_model_config_to_overrides(model_config),
    )

    shard_paths = sorted((input_root / "shards").glob("*.npz"))
    if args.only_shards:
        requested = {name.strip() for name in args.only_shards if name.strip()}
        shard_paths = [path for path in shard_paths if path.name in requested]
        missing = sorted(requested - {path.name for path in shard_paths})
        if missing:
            raise FileNotFoundError(f"Requested shard(s) not found under {input_root / 'shards'}: {missing}")
    if args.max_shards > 0:
        shard_paths = shard_paths[:args.max_shards]

    if not shard_paths:
        raise FileNotFoundError(f"No shard files found under {input_root / 'shards'}")

    skip_existing = bool(args.skip_existing)
    total_samples = 0
    processed_shards = 0
    for shard_idx, shard_path in enumerate(shard_paths, start=1):
        target_path = output_shard_path(output_root, shard_path.name, args.output_format)
        if skip_existing and target_path.exists() and not args.overwrite:
            print(f"[preprocess] {shard_idx}/{len(shard_paths)} {shard_path.name} skipped existing")
            continue
        num_samples = preprocess_shard(
            shard_path=shard_path,
            output_path=target_path,
            config=config,
            output_format=args.output_format,
            overwrite=args.overwrite,
        )
        total_samples += num_samples
        processed_shards += 1
        print(f"[preprocess] {shard_idx}/{len(shard_paths)} {shard_path.name} samples={num_samples}")

    _write_output_splits(input_root, output_root, args.output_format)

    reports_dir = input_root / "reports"
    if reports_dir.exists():
        shutil.copytree(reports_dir, output_root / "reports", dirs_exist_ok=True)

    manifest = {
        "source_dataset": str(input_root),
        "model_size": model_size,
        "model_config_path": args.model_config_path,
        "num_shards": processed_shards,
        "num_samples": total_samples,
        "format": "transfuser_preprocessed",
        "format_version": 1,
        "storage_format": args.output_format,
        "recommended_for_training": args.output_format == OUTPUT_FORMAT_DIR,
        "fields": list(PROCESSED_FIELDS),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "preprocess_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[preprocess] output_root={output_root} total_samples={total_samples} "
        f"processed_shards={processed_shards} storage_format={args.output_format}"
    )


if __name__ == "__main__":
    main()
