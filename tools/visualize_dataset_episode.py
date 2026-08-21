"""Render one committed joint-BEV dataset episode as a world-space MP4."""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import mediapy
import numpy as np

from expert_dataset.joint_bev_dataset import validate_dataset_contract
from expert_dataset.joint_bev_storage import SPLIT_NAMES
from expert_dataset.riskentry_sidecar_storage import (
    ACTOR_STATE_CHANNELS,
    SIDECAR_ARRAY_DTYPES,
    SIDECAR_FORMAT,
    SIDECAR_SCHEMA_VERSION,
    sidecar_dataset_contract,
)


DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 10.0
DEFAULT_VIEW_WIDTH_M = 100.0
DEFAULT_VIEW_HEIGHT_M = 56.25
DEFAULT_HISTORY_SECONDS = 5.0
HUD_HEIGHT = 72
PLATOON_COLORS_BGR = {
    "P0": (235, 99, 37),
    "P1": (105, 150, 5),
    "P2": (38, 38, 220),
}
EXTERNAL_COLOR_BGR = (145, 145, 145)
SCENARIO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class DatasetEpisodeVisualizationError(RuntimeError):
    """Raised when selection, stored data, or video output is invalid."""


@dataclass(frozen=True)
class DatasetRoots:
    bundle_root: Path
    base_root: Path
    sidecar_root: Path
    dataset_fingerprint: str
    base_format: str
    base_schema_version: int


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    split: str
    scenario_id: str
    spawn_seed: int
    joint_samples: int
    base_directory: Path
    sidecar_directory: Path
    raw_steps: int
    duration_s: float


@dataclass(frozen=True)
class RenderConfig:
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: float = DEFAULT_FPS
    camera_mode: str = "follow"
    view_width_m: float = DEFAULT_VIEW_WIDTH_M
    view_height_m: float = DEFAULT_VIEW_HEIGHT_M
    history_seconds: float = DEFAULT_HISTORY_SECONDS

    def __post_init__(self) -> None:
        if self.width < 320 or self.height < 240:
            raise DatasetEpisodeVisualizationError("video dimensions must be at least 320x240")
        if not math.isfinite(self.fps) or self.fps <= 0.0:
            raise DatasetEpisodeVisualizationError("fps must be finite and positive")
        if self.camera_mode not in {"follow", "episode"}:
            raise DatasetEpisodeVisualizationError("camera_mode must be follow or episode")
        for name, value in (
            ("view_width_m", self.view_width_m),
            ("view_height_m", self.view_height_m),
            ("history_seconds", self.history_seconds),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise DatasetEpisodeVisualizationError(f"{name} must be finite and positive")


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetEpisodeVisualizationError(f"invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise DatasetEpisodeVisualizationError(f"JSON root must be an object: {path}")
    return payload


def resolve_dataset_roots(dataset_root: Path | str) -> DatasetRoots:
    requested = Path(dataset_root).expanduser().resolve()
    if (requested / "platoon_joint_bev").is_dir():
        bundle_root = requested
        base_root = requested / "platoon_joint_bev"
    elif requested.name == "platoon_joint_bev" and requested.is_dir():
        bundle_root = requested.parent
        base_root = requested
    else:
        raise DatasetEpisodeVisualizationError(
            "dataset root must be a bundle root containing platoon_joint_bev "
            "or the platoon_joint_bev directory itself"
        )
    sidecar_root = bundle_root / "riskentry_actor_sidecar"
    if not sidecar_root.is_dir():
        raise DatasetEpisodeVisualizationError(
            f"matching riskentry_actor_sidecar is missing: {sidecar_root}"
        )
    try:
        base_contract = validate_dataset_contract(base_root)
    except Exception as exc:
        raise DatasetEpisodeVisualizationError(
            f"invalid joint-BEV dataset contract: {base_root}"
        ) from exc
    fingerprint = str(base_contract["dataset_fingerprint"])
    observed_sidecar_contract = _read_json(sidecar_root / "dataset_contract.json")
    expected_sidecar_contract = sidecar_dataset_contract(
        fingerprint,
        base_format=str(base_contract["format"]),
        base_schema_version=int(base_contract["schema_version"]),
    )
    if observed_sidecar_contract != expected_sidecar_contract:
        raise DatasetEpisodeVisualizationError(
            "sidecar contract or base dataset fingerprint does not match"
        )
    return DatasetRoots(
        bundle_root,
        base_root,
        sidecar_root,
        fingerprint,
        str(base_contract["format"]),
        int(base_contract["schema_version"]),
    )


def _validate_sidecar_episode(
    path: Path,
    *,
    episode_index: int,
    split: str,
    scenario_id: str,
    spawn_seed: int,
    dataset_fingerprint: str,
) -> tuple[int, float]:
    metadata = _read_json(path / "episode.json")
    expected = {
        "complete": True,
        "format": SIDECAR_FORMAT,
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "episode_index": episode_index,
        "split": split,
        "scenario_id": scenario_id,
        "spawn_seed": spawn_seed,
        "base_dataset_fingerprint": dataset_fingerprint,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise DatasetEpisodeVisualizationError(
                f"sidecar episode {episode_index} has invalid {key}: {metadata.get(key)!r}"
            )
    timestamp_path = path / "timestamp_s.npy"
    if not timestamp_path.is_file():
        raise DatasetEpisodeVisualizationError(
            f"sidecar episode {episode_index} is missing timestamp_s.npy"
        )
    timestamps = np.load(timestamp_path, mmap_mode="r", allow_pickle=False)
    if timestamps.dtype != SIDECAR_ARRAY_DTYPES["timestamp_s"] or timestamps.ndim != 1:
        raise DatasetEpisodeVisualizationError(
            f"sidecar episode {episode_index} has invalid timestamp_s array"
        )
    if timestamps.size == 0 or not np.isfinite(timestamps).all():
        raise DatasetEpisodeVisualizationError(
            f"sidecar episode {episode_index} has empty or non-finite timestamps"
        )
    if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0.0):
        raise DatasetEpisodeVisualizationError(
            f"sidecar episode {episode_index} timestamps must be strictly increasing"
        )
    return int(timestamps.size), float(timestamps[-1] - timestamps[0])


def _validate_base_episode(
    path: Path,
    *,
    episode_index: int,
    split: str,
    joint_samples: int,
    attributes: Mapping[str, object],
    base_format: str,
    base_schema_version: int,
) -> None:
    metadata = _read_json(path / "episode.json")
    expected = {
        "complete": True,
        "format": base_format,
        "schema_version": base_schema_version,
        "episode_index": episode_index,
        "split": split,
        "joint_samples": joint_samples,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise DatasetEpisodeVisualizationError(
                f"base episode {episode_index} has invalid {key}: {metadata.get(key)!r}"
            )
    if metadata.get("attributes") != dict(attributes):
        raise DatasetEpisodeVisualizationError(
            f"base episode {episode_index} attributes do not match its manifest"
        )


def discover_episodes(roots: DatasetRoots) -> tuple[EpisodeRecord, ...]:
    records: list[EpisodeRecord] = []
    seen_indices: set[int] = set()
    for split in SPLIT_NAMES:
        manifest_path = roots.base_root / split / "manifest.json"
        manifest = _read_json(manifest_path)
        if (
            manifest.get("schema_version") != roots.base_schema_version
            or manifest.get("format") != roots.base_format
            or manifest.get("split") != split
        ):
            raise DatasetEpisodeVisualizationError(f"invalid base manifest: {manifest_path}")
        entries = manifest.get("episodes")
        if not isinstance(entries, list) or manifest.get("episode_count") != len(entries):
            raise DatasetEpisodeVisualizationError(f"invalid episode list: {manifest_path}")
        manifest_joint_samples = 0
        previous_split_index = -1
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise DatasetEpisodeVisualizationError(f"invalid episode entry: {manifest_path}")
            episode_index = int(entry.get("episode_index", -1))
            directory = str(entry.get("directory", ""))
            joint_samples = int(entry.get("joint_samples", -1))
            attributes = entry.get("attributes")
            if (
                episode_index < 0
                or episode_index in seen_indices
                or directory != f"episode_{episode_index:08d}"
                or joint_samples <= 0
                or not isinstance(attributes, Mapping)
            ):
                raise DatasetEpisodeVisualizationError(
                    f"invalid or duplicate base episode entry: {manifest_path}"
                )
            if episode_index <= previous_split_index:
                raise DatasetEpisodeVisualizationError(
                    f"base manifest indices are not strictly increasing: {manifest_path}"
                )
            scenario_id = str(attributes.get("scenario_id", ""))
            spawn_seed = attributes.get("spawn_seed")
            if not scenario_id or isinstance(spawn_seed, bool) or not isinstance(spawn_seed, int):
                raise DatasetEpisodeVisualizationError(
                    f"episode {episode_index} lacks scenario_id or integer spawn_seed"
                )
            base_directory = roots.base_root / split / "episodes" / directory
            sidecar_directory = roots.sidecar_root / split / "episodes" / directory
            if not base_directory.is_dir() or not sidecar_directory.is_dir():
                raise DatasetEpisodeVisualizationError(
                    f"episode {episode_index} lacks matching base or sidecar directory"
                )
            _validate_base_episode(
                base_directory,
                episode_index=episode_index,
                split=split,
                joint_samples=joint_samples,
                attributes=attributes,
                base_format=roots.base_format,
                base_schema_version=roots.base_schema_version,
            )
            raw_steps, duration_s = _validate_sidecar_episode(
                sidecar_directory,
                episode_index=episode_index,
                split=split,
                scenario_id=scenario_id,
                spawn_seed=int(spawn_seed),
                dataset_fingerprint=roots.dataset_fingerprint,
            )
            records.append(
                EpisodeRecord(
                    episode_index=episode_index,
                    split=split,
                    scenario_id=scenario_id,
                    spawn_seed=int(spawn_seed),
                    joint_samples=joint_samples,
                    base_directory=base_directory,
                    sidecar_directory=sidecar_directory,
                    raw_steps=raw_steps,
                    duration_s=duration_s,
                )
            )
            seen_indices.add(episode_index)
            previous_split_index = episode_index
            manifest_joint_samples += joint_samples
        if manifest.get("joint_samples") != manifest_joint_samples:
            raise DatasetEpisodeVisualizationError(
                f"base manifest joint_samples mismatch: {manifest_path}"
            )
    return tuple(sorted(records, key=lambda item: item.episode_index))


def episodes_for_scenario(
    episodes: Sequence[EpisodeRecord], scenario_id: str
) -> tuple[EpisodeRecord, ...]:
    matches = tuple(item for item in episodes if item.scenario_id == scenario_id)
    if not matches:
        available = sorted({item.scenario_id for item in episodes})
        raise DatasetEpisodeVisualizationError(
            f"unknown scenario_id {scenario_id!r}; available scenarios: {available}"
        )
    return matches


def select_episode(
    episodes: Sequence[EpisodeRecord], scenario_id: str, episode_number: int
) -> tuple[EpisodeRecord, int]:
    if isinstance(episode_number, bool) or episode_number <= 0:
        raise DatasetEpisodeVisualizationError("episode-number is 1-based and must be positive")
    matches = episodes_for_scenario(episodes, scenario_id)
    if episode_number > len(matches):
        raise DatasetEpisodeVisualizationError(
            f"episode-number {episode_number} is out of range for {scenario_id}; "
            f"valid range is 1..{len(matches)}"
        )
    return matches[episode_number - 1], len(matches)


def vehicle_polygon_world(
    center_xy: Sequence[float], heading_rad: float, length_m: float, width_m: float
) -> np.ndarray:
    center = np.asarray(center_xy, dtype=np.float64)
    forward = np.asarray((math.cos(heading_rad), math.sin(heading_rad)))
    left = np.asarray((-forward[1], forward[0]))
    half_length = 0.5 * float(length_m)
    half_width = 0.5 * float(width_m)
    return np.stack(
        (
            center + forward * half_length + left * half_width,
            center + forward * half_length - left * half_width,
            center - forward * half_length - left * half_width,
            center - forward * half_length + left * half_width,
        )
    )


def world_to_pixels(
    points_xy: np.ndarray,
    *,
    center_xy: Sequence[float],
    view_width_m: float,
    view_height_m: float,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    center = np.asarray(center_xy, dtype=np.float64)
    plot_height = image_height - HUD_HEIGHT
    scale = min(image_width / view_width_m, plot_height / view_height_m)
    pixel = np.empty_like(points)
    pixel[..., 0] = image_width * 0.5 + (points[..., 0] - center[0]) * scale
    pixel[..., 1] = HUD_HEIGHT + plot_height * 0.5 - (points[..., 1] - center[1]) * scale
    return np.rint(pixel).astype(np.int32)


def _load_sidecar_arrays(record: EpisodeRecord) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    metadata = _read_json(record.sidecar_directory / "episode.json")
    names = ("step_index", "timestamp_s", "actor_state", "actor_state_valid_mask", "actor_valid_mask")
    try:
        arrays = {
            name: np.load(
                record.sidecar_directory / f"{name}.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            for name in names
        }
    except (OSError, ValueError) as exc:
        raise DatasetEpisodeVisualizationError(
            f"episode {record.episode_index} has a missing or unreadable sidecar array"
        ) from exc
    timeline = record.raw_steps
    actor_state = arrays["actor_state"]
    actor_count = len(metadata.get("actors", [])) if isinstance(metadata.get("actors"), list) else -1
    expected_shapes = {
        "step_index": (timeline,),
        "timestamp_s": (timeline,),
        "actor_state": (timeline, actor_count, len(ACTOR_STATE_CHANNELS)),
        "actor_state_valid_mask": (timeline, actor_count, len(ACTOR_STATE_CHANNELS)),
        "actor_valid_mask": (timeline, actor_count),
    }
    for name, expected_shape in expected_shapes.items():
        if arrays[name].dtype != SIDECAR_ARRAY_DTYPES[name] or arrays[name].shape != expected_shape:
            raise DatasetEpisodeVisualizationError(
                f"episode {record.episode_index} has invalid {name}: "
                f"shape={arrays[name].shape}, dtype={arrays[name].dtype}"
            )
    if actor_count < 3 or actor_state.shape[1] != actor_count:
        raise DatasetEpisodeVisualizationError("sidecar must register at least P0/P1/P2")
    actors = metadata["actors"]
    for index, actor in enumerate(actors):
        if not isinstance(actor, Mapping) or actor.get("actor_index") != index:
            raise DatasetEpisodeVisualizationError("sidecar actor table is not contiguous")
        if not all(
            math.isfinite(float(actor.get(name, float("nan")))) and float(actor[name]) > 0.0
            for name in ("length_m", "width_m")
        ):
            raise DatasetEpisodeVisualizationError("sidecar actor dimensions must be positive")
    if [actor.get("actor_id") for actor in actors[:3]] != ["P0", "P1", "P2"]:
        raise DatasetEpisodeVisualizationError("sidecar actor rows 0..2 must be P0/P1/P2")
    pose_valid = arrays["actor_valid_mask"] & arrays["actor_state_valid_mask"][..., :3].all(axis=-1)
    if not np.isfinite(actor_state[..., :3][pose_valid]).all():
        raise DatasetEpisodeVisualizationError("valid actor pose channels contain non-finite values")
    return metadata, arrays


def _episode_camera(
    actor_state: np.ndarray, pose_valid: np.ndarray, config: RenderConfig
) -> tuple[np.ndarray, float, float]:
    points = actor_state[..., :2][pose_valid]
    if points.size == 0:
        raise DatasetEpisodeVisualizationError("episode contains no valid actor positions")
    lower = points.min(axis=0) - 5.0
    upper = points.max(axis=0) + 5.0
    center = 0.5 * (lower + upper)
    width = max(config.view_width_m, float(upper[0] - lower[0]))
    height = max(config.view_height_m, float(upper[1] - lower[1]))
    plot_aspect = config.width / (config.height - HUD_HEIGHT)
    if width / height < plot_aspect:
        width = height * plot_aspect
    else:
        height = width / plot_aspect
    return center, width, height


def _follow_camera_centers(actor_state: np.ndarray, pose_valid: np.ndarray) -> np.ndarray:
    platoon_valid = pose_valid[:, :3]
    has_platoon = platoon_valid.any(axis=1)
    if not has_platoon.any():
        raise DatasetEpisodeVisualizationError("episode contains no valid platoon positions")
    centers = np.zeros((actor_state.shape[0], 2), dtype=np.float64)
    for frame_index in np.flatnonzero(has_platoon):
        centers[frame_index] = actor_state[frame_index, :3, :2][
            platoon_valid[frame_index]
        ].mean(axis=0)
    first_valid = int(np.flatnonzero(has_platoon)[0])
    centers[:first_valid] = centers[first_valid]
    last_center = centers[first_valid]
    for frame_index in range(first_valid, actor_state.shape[0]):
        if has_platoon[frame_index]:
            last_center = centers[frame_index]
        else:
            centers[frame_index] = last_center
    return centers


def _draw_grid(
    image: np.ndarray, center: np.ndarray, view_width: float, view_height: float
) -> None:
    left, right = center[0] - view_width / 2.0, center[0] + view_width / 2.0
    bottom, top = center[1] - view_height / 2.0, center[1] + view_height / 2.0
    for x in np.arange(math.floor(left / 10.0) * 10.0, right + 10.0, 10.0):
        points = world_to_pixels(
            np.asarray(((x, bottom), (x, top))), center_xy=center,
            view_width_m=view_width, view_height_m=view_height,
            image_width=image.shape[1], image_height=image.shape[0]
        )
        cv2.line(image, tuple(points[0]), tuple(points[1]), (225, 225, 225), 1, cv2.LINE_AA)
    for y in np.arange(math.floor(bottom / 10.0) * 10.0, top + 10.0, 10.0):
        points = world_to_pixels(
            np.asarray(((left, y), (right, y))), center_xy=center,
            view_width_m=view_width, view_height_m=view_height,
            image_width=image.shape[1], image_height=image.shape[0]
        )
        cv2.line(image, tuple(points[0]), tuple(points[1]), (225, 225, 225), 1, cv2.LINE_AA)


def render_frame(
    metadata: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    record: EpisodeRecord,
    episode_number: int,
    frame_index: int,
    config: RenderConfig,
    camera: tuple[np.ndarray, float, float],
) -> np.ndarray:
    actor_state = arrays["actor_state"]
    actor_valid = arrays["actor_valid_mask"]
    state_valid = arrays["actor_state_valid_mask"]
    pose_valid = actor_valid & state_valid[..., :3].all(axis=-1)
    center, view_width, view_height = camera

    image = np.full((config.height, config.width, 3), 248, dtype=np.uint8)
    image[:HUD_HEIGHT] = (32, 37, 43)
    _draw_grid(image, np.asarray(center), view_width, view_height)
    actors = metadata["actors"]
    timestamps = arrays["timestamp_s"]
    cutoff = float(timestamps[frame_index]) - config.history_seconds
    history_start = int(np.searchsorted(timestamps, cutoff, side="left"))
    for actor_index, actor in enumerate(actors):
        color = PLATOON_COLORS_BGR.get(str(actor["actor_id"]), EXTERNAL_COLOR_BGR)
        valid_history = pose_valid[history_start : frame_index + 1, actor_index]
        history = actor_state[history_start : frame_index + 1, actor_index, :2][valid_history]
        if history.shape[0] >= 2:
            pixels = world_to_pixels(
                history, center_xy=center, view_width_m=view_width,
                view_height_m=view_height, image_width=config.width,
                image_height=config.height
            )
            cv2.polylines(image, [pixels], False, color, 2, cv2.LINE_AA)
        if not pose_valid[frame_index, actor_index]:
            continue
        state = actor_state[frame_index, actor_index]
        polygon = vehicle_polygon_world(
            state[:2], float(state[2]), float(actor["length_m"]), float(actor["width_m"])
        )
        pixels = world_to_pixels(
            polygon, center_xy=center, view_width_m=view_width,
            view_height_m=view_height, image_width=config.width,
            image_height=config.height
        )
        cv2.fillConvexPoly(image, pixels, color, cv2.LINE_AA)
        cv2.polylines(image, [pixels], True, (25, 25, 25), 1, cv2.LINE_AA)
        label_position = world_to_pixels(
            np.asarray([state[:2]]), center_xy=center, view_width_m=view_width,
            view_height_m=view_height, image_width=config.width,
            image_height=config.height
        )[0]
        cv2.putText(
            image, str(actor["actor_id"]), (int(label_position[0]) + 5, int(label_position[1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA
        )

    title = (
        f"{record.scenario_id}  episode #{episode_number}  "
        f"global={record.episode_index}  split={record.split}  seed={record.spawn_seed}"
    )
    timing = (
        f"t={float(timestamps[frame_index]):.1f}s  step={int(arrays['step_index'][frame_index])}  "
        f"actors={int(pose_valid[frame_index].sum())}/{len(actors)}  camera={config.camera_mode}"
    )
    cv2.putText(image, title, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(image, timing, (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (205, 215, 225), 1, cv2.LINE_AA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def render_episode_video(
    record: EpisodeRecord,
    episode_number: int,
    output_path: Path | str,
    *,
    config: RenderConfig = RenderConfig(),
    overwrite: bool = False,
) -> Path:
    output = Path(output_path).expanduser().resolve()
    if output.suffix.lower() != ".mp4":
        raise DatasetEpisodeVisualizationError("output path must end with .mp4")
    if output.exists() and not overwrite:
        raise DatasetEpisodeVisualizationError(
            f"output already exists; pass --overwrite to replace it: {output}"
        )
    metadata, arrays = _load_sidecar_arrays(record)
    pose_valid = arrays["actor_valid_mask"] & arrays["actor_state_valid_mask"][..., :3].all(axis=-1)
    episode_camera = (
        _episode_camera(arrays["actor_state"], pose_valid, config)
        if config.camera_mode == "episode"
        else None
    )
    follow_centers = (
        _follow_camera_centers(arrays["actor_state"], pose_valid)
        if config.camera_mode == "follow"
        else None
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial-{os.getpid()}.mp4")
    if temporary.exists():
        temporary.unlink()
    try:
        with mediapy.VideoWriter(
            temporary,
            (config.height, config.width),
            codec="h264",
            fps=config.fps,
        ) as writer:
            for frame_index in range(record.raw_steps):
                camera = episode_camera
                if camera is None:
                    camera = (
                        follow_centers[frame_index],
                        config.view_width_m,
                        config.view_height_m,
                    )
                writer.add_image(
                    render_frame(
                        metadata, arrays, record, episode_number, frame_index,
                        config, camera
                    )
                )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def default_output_path(
    roots: DatasetRoots, record: EpisodeRecord, episode_number: int
) -> Path:
    return (
        Path("outputs")
        / "dataset_episode_visualizations"
        / roots.bundle_root.name
        / record.scenario_id
        / f"episode_{episode_number:03d}_global_{record.episode_index:08d}.mp4"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scenario-id")
    parser.add_argument("--episode-number", type=int)
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument("--list-episodes", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--camera-mode", choices=("follow", "episode"), default="follow")
    parser.add_argument("--view-width-m", type=float, default=DEFAULT_VIEW_WIDTH_M)
    parser.add_argument("--view-height-m", type=float, default=DEFAULT_VIEW_HEIGHT_M)
    parser.add_argument("--history-seconds", type=float, default=DEFAULT_HISTORY_SECONDS)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    roots = resolve_dataset_roots(args.dataset_root)
    episodes = discover_episodes(roots)
    if args.list_scenarios:
        counts = {scenario: sum(item.scenario_id == scenario for item in episodes) for scenario in sorted({item.scenario_id for item in episodes})}
        for scenario, count in counts.items():
            print(f"{scenario}\t{count}")
        return 0
    if not args.scenario_id:
        raise DatasetEpisodeVisualizationError("--scenario-id is required unless --list-scenarios is used")
    if not SCENARIO_PATTERN.fullmatch(args.scenario_id):
        raise DatasetEpisodeVisualizationError("scenario-id contains unsupported characters")
    matches = episodes_for_scenario(episodes, args.scenario_id)
    if args.list_episodes:
        for number, record in enumerate(matches, start=1):
            print(
                f"{number}\tglobal={record.episode_index}\tsplit={record.split}\t"
                f"seed={record.spawn_seed}\tduration_s={record.duration_s:.1f}"
            )
        return 0
    if args.episode_number is None:
        raise DatasetEpisodeVisualizationError("--episode-number is required for rendering")
    record, total = select_episode(episodes, args.scenario_id, args.episode_number)
    output = Path(args.output) if args.output else default_output_path(roots, record, args.episode_number)
    config = RenderConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        camera_mode=args.camera_mode,
        view_width_m=args.view_width_m,
        view_height_m=args.view_height_m,
        history_seconds=args.history_seconds,
    )
    result = render_episode_video(
        record, args.episode_number, output, config=config, overwrite=args.overwrite
    )
    print(
        f"rendered scenario={record.scenario_id} episode={args.episode_number}/{total} "
        f"global={record.episode_index} split={record.split} frames={record.raw_steps} output={result}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DatasetEpisodeVisualizationError as exc:
        raise SystemExit(f"error: {exc}") from exc
