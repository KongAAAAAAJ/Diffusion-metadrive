from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
import zipfile

import numpy as np

from metadrive.policy.diffusion_policy.preprocess_transfuser_dataset import (
    OUTPUT_FORMAT_DIR,
    OUTPUT_FORMAT_NPZ_LEGACY,
    PROCESSED_DIRSET_FORMAT,
    PROCESSED_FIELDS,
    output_shard_path,
    preprocess_shard,
)
from metadrive.policy.diffusion_policy.transfuser_config import build_transfuser_config


RAW_DATASET_FORMAT = "raw"
PROCESSED_DIR_DATASET_FORMAT = "processed_dir"
PROCESSED_NPZ_LEGACY_DATASET_FORMAT = "processed_npz_legacy"
AUTO_DATASET_FORMAT = "auto"
SUPPORTED_DATASET_FORMATS = (
    AUTO_DATASET_FORMAT,
    RAW_DATASET_FORMAT,
    PROCESSED_DIR_DATASET_FORMAT,
    PROCESSED_NPZ_LEGACY_DATASET_FORMAT,
)
REQUIRED_RAW_KEYS: Tuple[str, ...] = (
    "left_camera",
    "front_camera",
    "right_camera",
    "lidar",
    "ego_state",
    "trajectory",
    "agent_states",
    "agent_labels",
    "bev_raster",
)
REQUIRED_PREPROCESSED_KEYS: Tuple[str, ...] = PROCESSED_FIELDS
FINAL_BAD_STATUSES = {"corrupt", "repair_failed", "missing_source"}


@dataclass
class ShardVerificationResult:
    name: str
    path: str
    schema: str
    status: str
    storage_format: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    repaired_from: Optional[str] = None
    validated_keys: Optional[List[str]] = None
    num_samples: Optional[int] = None


def parse_args():
    parser = argparse.ArgumentParser(description="Verify and optionally repair TransFuser dataset shards.")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--split", type=str, default="all", choices=("train", "val", "test", "all"))
    parser.add_argument("--dataset-format", type=str, default=AUTO_DATASET_FORMAT, choices=SUPPORTED_DATASET_FORMATS)
    parser.add_argument("--source-dataset-root", type=str, default=None)
    parser.add_argument("--model-size", type=str, default="small")
    parser.add_argument("--repair-corrupt-shards", action="store_true")
    parser.add_argument("--fail-on-repair-error", type=int, default=1)
    parser.add_argument("--report-path", type=str, default=None)
    return parser.parse_args()


def _storage_format_for_path(shard_path: Path) -> str:
    if shard_path.is_dir():
        return OUTPUT_FORMAT_DIR
    return "npz"


def _list_shard_entries(shard_dir: Path) -> List[Path]:
    entries = {}
    for path in sorted(shard_dir.iterdir()):
        if path.is_dir() and path.name.startswith("shard_"):
            entries[path.name] = path
    for path in sorted(shard_dir.glob("*.npz")):
        entries.setdefault(path.stem, path)
    return [entries[key] for key in sorted(entries)]


def resolve_split_shards(dataset_root: Path, split: str) -> List[Path]:
    shard_dir = dataset_root / "shards"
    if not shard_dir.exists():
        raise FileNotFoundError(f"Dataset shard directory does not exist: {shard_dir}")
    shard_paths = _list_shard_entries(shard_dir)
    if split == "all":
        return shard_paths
    split_path = dataset_root / "splits" / f"{split}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file does not exist: {split_path}")
    names = {line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    canonical_names = {Path(name).stem if name.endswith(".npz") else name for name in names}
    resolved = [path for path in shard_paths if path.name in names or path.stem in canonical_names]
    if not resolved:
        raise FileNotFoundError(f"No shard files resolved for split '{split}' under {dataset_root}")
    return resolved


def resolve_source_dataset_root(dataset_root: Path, explicit_source_root: Optional[str] = None) -> Optional[Path]:
    if explicit_source_root:
        return Path(explicit_source_root)

    manifest_path = dataset_root / "preprocess_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_dataset = manifest.get("source_dataset")
        if source_dataset:
            return Path(source_dataset)
    return None


def _validate_zip_structure(shard_path: Path) -> None:
    with zipfile.ZipFile(shard_path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise zipfile.BadZipFile(f"Bad CRC-32 for file '{bad_member}'")


def _validate_processed_npz_legacy_shard(shard_path: Path) -> Tuple[List[str], int]:
    validated_keys: List[str] = []
    expected_keys = set(REQUIRED_PREPROCESSED_KEYS)

    with np.load(shard_path, allow_pickle=False) as raw:
        actual_keys = set(raw.files)
        missing_keys = sorted(expected_keys - actual_keys)
        unexpected_keys = sorted(actual_keys - expected_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                f"Unexpected shard schema for {shard_path.name}. "
                f"missing_keys={missing_keys} unexpected_keys={unexpected_keys}"
            )

        sample_count: Optional[int] = None
        for key in REQUIRED_PREPROCESSED_KEYS:
            value = np.asarray(raw[key])
            validated_keys.append(key)
            if value.ndim == 0:
                raise ValueError(f"Key '{key}' in {shard_path.name} is scalar; expected batched array")
            key_samples = int(value.shape[0])
            if sample_count is None:
                sample_count = key_samples
            elif key_samples != sample_count:
                raise ValueError(
                    f"Inconsistent sample count in {shard_path.name}: "
                    f"key '{key}' has {key_samples}, expected {sample_count}"
                )

        if sample_count is None or sample_count <= 0:
            raise ValueError(f"Shard {shard_path.name} has no samples")

    return validated_keys, sample_count


def _validate_raw_shard(shard_path: Path) -> Tuple[List[str], int]:
    validated_keys: List[str] = []
    expected_keys = set(REQUIRED_RAW_KEYS)

    with np.load(shard_path, allow_pickle=False) as raw:
        actual_keys = set(raw.files)
        missing_keys = sorted(expected_keys - actual_keys)
        if missing_keys:
            raise ValueError(f"Unexpected raw shard schema for {shard_path.name}. missing_keys={missing_keys}")

        sample_count: Optional[int] = None
        for key in REQUIRED_RAW_KEYS:
            value = np.asarray(raw[key])
            validated_keys.append(key)
            if value.ndim == 0:
                raise ValueError(f"Key '{key}' in {shard_path.name} is scalar; expected batched array")
            key_samples = int(value.shape[0])
            if sample_count is None:
                sample_count = key_samples
            elif key_samples != sample_count:
                raise ValueError(
                    f"Inconsistent sample count in {shard_path.name}: "
                    f"key '{key}' has {key_samples}, expected {sample_count}"
                )

        if sample_count is None or sample_count <= 0:
            raise ValueError(f"Shard {shard_path.name} has no samples")

        for camera_key in ("left_camera", "front_camera", "right_camera"):
            camera = np.asarray(raw[camera_key])
            if camera.ndim != 4 or camera.shape[-1] not in (1, 3, 4):
                raise ValueError(
                    f"Key '{camera_key}' in {shard_path.name} must be batched HWC image data, got shape {camera.shape}"
                )

        lidar = np.asarray(raw["lidar"])
        if lidar.ndim != 2:
            raise ValueError(f"Key 'lidar' in {shard_path.name} must be a 2D batched array, got shape {lidar.shape}")

        bev_raster = np.asarray(raw["bev_raster"])
        if bev_raster.ndim != 4:
            raise ValueError(
                f"Key 'bev_raster' in {shard_path.name} must be a 4D batched array, got shape {bev_raster.shape}"
            )

        if "bev_semantic_map" in raw.files:
            bev_semantic_map = np.asarray(raw["bev_semantic_map"])
            validated_keys.append("bev_semantic_map")
            if bev_semantic_map.ndim != 3:
                raise ValueError(
                    f"Key 'bev_semantic_map' in {shard_path.name} must be a 3D batched array, got shape {bev_semantic_map.shape}"
                )
            if int(bev_semantic_map.shape[0]) != sample_count:
                raise ValueError(
                    f"Inconsistent sample count in {shard_path.name}: "
                    f"key 'bev_semantic_map' has {bev_semantic_map.shape[0]}, expected {sample_count}"
                )

    return validated_keys, sample_count


def _validate_processed_dir_shard(shard_path: Path) -> Tuple[List[str], int]:
    meta_path = shard_path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Processed shard metadata not found: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("dataset_format") != PROCESSED_DIRSET_FORMAT:
        raise ValueError(f"Unexpected dataset_format in {meta_path}: {meta.get('dataset_format')}")

    fields_meta = meta.get("fields")
    if not isinstance(fields_meta, dict):
        raise ValueError(f"Invalid fields metadata in {meta_path}")

    validated_keys: List[str] = []
    sample_count: Optional[int] = None
    for key in REQUIRED_PREPROCESSED_KEYS:
        field_path = shard_path / f"{key}.npy"
        if not field_path.exists():
            raise FileNotFoundError(f"Processed shard field not found: {field_path}")
        array = np.load(field_path, mmap_mode="r", allow_pickle=False)
        validated_keys.append(key)
        if array.ndim == 0:
            raise ValueError(f"Key '{key}' in {shard_path.name} is scalar; expected batched array")
        key_samples = int(array.shape[0])
        if sample_count is None:
            sample_count = key_samples
        elif key_samples != sample_count:
            raise ValueError(
                f"Inconsistent sample count in {shard_path.name}: "
                f"key '{key}' has {key_samples}, expected {sample_count}"
            )

        expected = fields_meta.get(key)
        if not isinstance(expected, dict):
            raise ValueError(f"Missing metadata for key '{key}' in {meta_path}")
        if list(array.shape) != expected.get("shape"):
            raise ValueError(
                f"Shape mismatch for key '{key}' in {shard_path.name}: "
                f"metadata={expected.get('shape')} actual={list(array.shape)}"
            )
        if str(array.dtype) != expected.get("dtype"):
            raise ValueError(
                f"Dtype mismatch for key '{key}' in {shard_path.name}: "
                f"metadata={expected.get('dtype')} actual={array.dtype}"
            )

    if sample_count is None or sample_count <= 0:
        raise ValueError(f"Shard {shard_path.name} has no samples")
    if int(meta.get("num_samples", sample_count)) != sample_count:
        raise ValueError(
            f"Metadata num_samples mismatch in {shard_path.name}: "
            f"metadata={meta.get('num_samples')} actual={sample_count}"
        )

    return validated_keys, sample_count


def detect_dataset_format_from_keys(keys: Set[str]) -> str:
    if set(REQUIRED_PREPROCESSED_KEYS).issubset(keys):
        return PROCESSED_NPZ_LEGACY_DATASET_FORMAT
    if set(REQUIRED_RAW_KEYS).issubset(keys):
        return RAW_DATASET_FORMAT
    raise ValueError(
        "Unable to infer dataset format from shard keys. "
        f"available_keys={sorted(keys)}"
    )


def infer_dataset_format(shard_path: Path) -> str:
    if shard_path.is_dir():
        _validate_processed_dir_shard(shard_path)
        return PROCESSED_DIR_DATASET_FORMAT
    _validate_zip_structure(shard_path)
    with np.load(shard_path, allow_pickle=False) as raw:
        return detect_dataset_format_from_keys(set(raw.files))


def verify_shard(shard_path: Path, dataset_format: str) -> ShardVerificationResult:
    storage_format = _storage_format_for_path(shard_path)
    try:
        if dataset_format == RAW_DATASET_FORMAT:
            _validate_zip_structure(shard_path)
            validated_keys, sample_count = _validate_raw_shard(shard_path)
        elif dataset_format == PROCESSED_NPZ_LEGACY_DATASET_FORMAT:
            _validate_zip_structure(shard_path)
            validated_keys, sample_count = _validate_processed_npz_legacy_shard(shard_path)
        elif dataset_format == PROCESSED_DIR_DATASET_FORMAT:
            validated_keys, sample_count = _validate_processed_dir_shard(shard_path)
        else:
            raise ValueError(f"Unsupported dataset_format: {dataset_format}")
        return ShardVerificationResult(
            name=shard_path.name,
            path=str(shard_path),
            schema=dataset_format,
            status="ok",
            storage_format=storage_format,
            validated_keys=validated_keys,
            num_samples=sample_count,
        )
    except Exception as exc:
        return ShardVerificationResult(
            name=shard_path.name,
            path=str(shard_path),
            schema=dataset_format,
            status="corrupt",
            storage_format=storage_format,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


def verify_preprocessed_shard(shard_path: Path) -> ShardVerificationResult:
    dataset_format = PROCESSED_DIR_DATASET_FORMAT if shard_path.is_dir() else PROCESSED_NPZ_LEGACY_DATASET_FORMAT
    return verify_shard(shard_path, dataset_format)


def repair_preprocessed_shard(
    shard_path: Path,
    dataset_root: Path,
    source_dataset_root: Optional[Path],
    model_size: str,
) -> ShardVerificationResult:
    shard_name = f"{shard_path.stem}.npz"
    if source_dataset_root is None:
        return ShardVerificationResult(
            name=shard_path.name,
            path=str(shard_path),
            schema=PROCESSED_DIR_DATASET_FORMAT,
            status="missing_source",
            storage_format=_storage_format_for_path(shard_path),
            error_type="FileNotFoundError",
            error_message="No source dataset root configured for shard repair",
        )

    source_shard = source_dataset_root / "shards" / shard_name
    if not source_shard.exists():
        return ShardVerificationResult(
            name=shard_path.name,
            path=str(shard_path),
            schema=PROCESSED_DIR_DATASET_FORMAT,
            status="missing_source",
            storage_format=_storage_format_for_path(shard_path),
            error_type="FileNotFoundError",
            error_message=f"Source shard not found: {source_shard}",
        )

    repaired_output = output_shard_path(dataset_root, shard_name, OUTPUT_FORMAT_DIR)
    try:
        preprocess_shard(
            shard_path=source_shard,
            output_path=repaired_output,
            config=build_transfuser_config(model_size),
            output_format=OUTPUT_FORMAT_DIR,
            overwrite=True,
        )
        verified = verify_shard(repaired_output, PROCESSED_DIR_DATASET_FORMAT)
        if verified.status != "ok":
            return ShardVerificationResult(
                name=repaired_output.name,
                path=str(repaired_output),
                schema=PROCESSED_DIR_DATASET_FORMAT,
                status="repair_failed",
                storage_format=OUTPUT_FORMAT_DIR,
                error_type=verified.error_type,
                error_message=verified.error_message,
                repaired_from=str(source_shard),
            )
        verified.status = "repaired"
        verified.repaired_from = str(source_shard)
        return verified
    except Exception as exc:
        return ShardVerificationResult(
            name=repaired_output.name,
            path=str(repaired_output),
            schema=PROCESSED_DIR_DATASET_FORMAT,
            status="repair_failed",
            storage_format=OUTPUT_FORMAT_DIR,
            error_type=type(exc).__name__,
            error_message=str(exc),
            repaired_from=str(source_shard),
        )


def _build_summary(results: Sequence[ShardVerificationResult]) -> Dict[str, int]:
    summary: Dict[str, int] = {
        "total": len(results),
        "ok": 0,
        "repaired": 0,
        "corrupt": 0,
        "repair_failed": 0,
        "missing_source": 0,
    }
    for result in results:
        summary[result.status] = summary.get(result.status, 0) + 1
    return summary


def write_integrity_report(
    dataset_root: Path,
    split_label: str,
    dataset_format: str,
    results: Sequence[ShardVerificationResult],
    source_dataset_root: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> Path:
    if report_path is None:
        report_path = dataset_root / "reports" / "integrity_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_root": str(dataset_root),
        "dataset_format": dataset_format,
        "storage_format": (
            OUTPUT_FORMAT_DIR
            if dataset_format == PROCESSED_DIR_DATASET_FORMAT
            else "npz-legacy" if dataset_format == PROCESSED_NPZ_LEGACY_DATASET_FORMAT else "npz"
        ),
        "legacy_format_detected": any(result.schema == PROCESSED_NPZ_LEGACY_DATASET_FORMAT for result in results),
        "source_dataset_root": str(source_dataset_root) if source_dataset_root is not None else None,
        "split": split_label,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "summary": _build_summary(results),
        "shards": [asdict(result) for result in results],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def verify_dataset(
    dataset_root: Path,
    split: str,
    model_size: str,
    dataset_format: str = AUTO_DATASET_FORMAT,
    repair_corrupt_shards: bool = False,
    source_dataset_root: Optional[Path] = None,
    fail_on_repair_error: bool = True,
    report_path: Optional[Path] = None,
    shard_paths: Optional[Sequence[Path]] = None,
    split_label: Optional[str] = None,
) -> Dict:
    selected_shards = list(shard_paths) if shard_paths is not None else resolve_split_shards(dataset_root, split)
    if not selected_shards:
        raise FileNotFoundError(f"No shard files resolved for split '{split}' under {dataset_root}")
    inferred_format = None
    if dataset_format == AUTO_DATASET_FORMAT:
        inferred_format = infer_dataset_format(selected_shards[0])
        resolved_dataset_format = inferred_format
    else:
        resolved_dataset_format = dataset_format
        try:
            inferred_format = infer_dataset_format(selected_shards[0])
        except Exception:
            inferred_format = None
        if inferred_format is not None and resolved_dataset_format != inferred_format:
            raise ValueError(
                f"Requested dataset_format='{dataset_format}' does not match inferred format "
                f"'{inferred_format}' from shard '{selected_shards[0].name}'"
            )
    resolved_source_root = resolve_source_dataset_root(dataset_root, str(source_dataset_root) if source_dataset_root else None)
    effective_repair = bool(
        repair_corrupt_shards and resolved_dataset_format in (PROCESSED_DIR_DATASET_FORMAT, PROCESSED_NPZ_LEGACY_DATASET_FORMAT)
    )
    results: List[ShardVerificationResult] = []

    for shard_path in selected_shards:
        result = verify_shard(shard_path, resolved_dataset_format)
        if result.status == "corrupt" and effective_repair:
            result = repair_preprocessed_shard(shard_path, dataset_root, resolved_source_root, model_size)
        results.append(result)

    report_file = write_integrity_report(
        dataset_root=dataset_root,
        split_label=split_label or split,
        dataset_format=resolved_dataset_format,
        results=results,
        source_dataset_root=(
            resolved_source_root
            if resolved_dataset_format in (PROCESSED_DIR_DATASET_FORMAT, PROCESSED_NPZ_LEGACY_DATASET_FORMAT)
            else None
        ),
        report_path=report_path,
    )
    summary = _build_summary(results)
    final_bad = [result for result in results if result.status in FINAL_BAD_STATUSES]
    if fail_on_repair_error and final_bad:
        names = ", ".join(f"{result.name}({result.status})" for result in final_bad)
        raise RuntimeError(
            f"Dataset integrity verification failed for {names}. See report: {report_file}"
        )
    return {
        "dataset_root": str(dataset_root),
        "dataset_format": resolved_dataset_format,
        "storage_format": (
            OUTPUT_FORMAT_DIR
            if resolved_dataset_format == PROCESSED_DIR_DATASET_FORMAT
            else "npz-legacy" if resolved_dataset_format == PROCESSED_NPZ_LEGACY_DATASET_FORMAT else "npz"
        ),
        "legacy_format_detected": any(result.schema == PROCESSED_NPZ_LEGACY_DATASET_FORMAT for result in results),
        "source_dataset_root": (
            str(resolved_source_root)
            if resolved_dataset_format in (PROCESSED_DIR_DATASET_FORMAT, PROCESSED_NPZ_LEGACY_DATASET_FORMAT)
            and resolved_source_root is not None
            else None
        ),
        "split": split_label or split,
        "summary": summary,
        "shards": [asdict(result) for result in results],
        "report_path": str(report_file),
    }


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    source_dataset_root = Path(args.source_dataset_root) if args.source_dataset_root else None
    report_path = Path(args.report_path) if args.report_path else None
    report = verify_dataset(
        dataset_root=dataset_root,
        split=args.split,
        model_size=args.model_size,
        dataset_format=args.dataset_format,
        repair_corrupt_shards=args.repair_corrupt_shards,
        source_dataset_root=source_dataset_root,
        fail_on_repair_error=bool(args.fail_on_repair_error),
        report_path=report_path,
    )
    print(f"[verify] dataset_root={report['dataset_root']}")
    print(f"[verify] dataset_format={report['dataset_format']}")
    print(f"[verify] storage_format={report['storage_format']}")
    print(f"[verify] split={report['split']} summary={report['summary']}")
    print(f"[verify] report_path={report['report_path']}")


if __name__ == "__main__":
    main()
