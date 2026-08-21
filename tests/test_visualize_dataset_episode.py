from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from expert_dataset.joint_bev_storage import (
    EpisodeSplitConfig,
    JointBEVDatasetStore,
    SPLIT_NAMES,
    STORAGE_FORMAT,
    STORAGE_SCHEMA_VERSION,
)
from expert_dataset.riskentry_sidecar_storage import (
    ACTOR_STATE_CHANNELS,
    SIDECAR_ARRAY_DTYPES,
    SIDECAR_FORMAT,
    SIDECAR_SCHEMA_VERSION,
    sidecar_dataset_contract,
)
from tools.visualize_dataset_episode import (
    DatasetEpisodeVisualizationError,
    RenderConfig,
    default_output_path,
    discover_episodes,
    episodes_for_scenario,
    main,
    render_episode_video,
    resolve_dataset_roots,
    select_episode,
    vehicle_polygon_world,
    world_to_pixels,
)


FINGERPRINT = "a" * 64


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _actor_table() -> list[dict[str, object]]:
    return [
        {
            "actor_id": actor_id,
            "actor_index": index,
            "actor_type": "platoon" if index < 3 else "background",
            "platoon_role": role,
            "source_object_id": f"source-{actor_id}",
            "first_seen_step": 0,
            "length_m": 5.0,
            "width_m": 2.0,
        }
        for index, (actor_id, role) in enumerate(
            (("P0", "leader"), ("P1", "middle"), ("P2", "tail"), ("V000", None))
        )
    ]


def _add_episode(
    bundle: Path,
    manifests: dict[str, list[dict[str, object]]],
    *,
    episode_index: int,
    split: str,
    scenario_id: str,
    spawn_seed: int,
    add_base: bool = True,
) -> None:
    directory = f"episode_{episode_index:08d}"
    attributes = {"scenario_id": scenario_id, "spawn_seed": spawn_seed}
    if add_base:
        base_episode = bundle / "platoon_joint_bev" / split / "episodes" / directory
        base_episode.mkdir(parents=True)
        _write_json(
            base_episode / "episode.json",
            {
                "complete": True,
                "format": STORAGE_FORMAT,
                "schema_version": STORAGE_SCHEMA_VERSION,
                "episode_index": episode_index,
                "split": split,
                "joint_samples": 3,
                "attributes": attributes,
            },
        )
        manifests[split].append(
            {
                "episode_index": episode_index,
                "directory": directory,
                "joint_samples": 3,
                "attributes": attributes,
            }
        )

    sidecar_episode = bundle / "riskentry_actor_sidecar" / split / "episodes" / directory
    sidecar_episode.mkdir(parents=True)
    _write_json(
        sidecar_episode / "episode.json",
        {
            "complete": True,
            "format": SIDECAR_FORMAT,
            "schema_version": SIDECAR_SCHEMA_VERSION,
            "episode_index": episode_index,
            "split": split,
            "scenario_id": scenario_id,
            "spawn_seed": spawn_seed,
            "base_dataset_fingerprint": FINGERPRINT,
            "actors": _actor_table(),
        },
    )
    timeline, actor_count = 4, 4
    actor_state = np.zeros((timeline, actor_count, len(ACTOR_STATE_CHANNELS)), dtype=np.float32)
    for step in range(timeline):
        for actor in range(actor_count):
            actor_state[step, actor, 0] = float(step * 2 + actor * 7)
            actor_state[step, actor, 1] = float((actor - 1) * 3)
            actor_state[step, actor, 2] = np.float32(0.05 * actor)
    actor_valid = np.ones((timeline, actor_count), dtype=np.bool_)
    actor_valid[1, 3] = False
    actor_valid[3, :3] = False
    state_valid = np.repeat(actor_valid[..., None], len(ACTOR_STATE_CHANNELS), axis=2)
    arrays = {
        "step_index": np.arange(timeline, dtype=np.int64),
        "timestamp_s": np.arange(timeline, dtype=np.float64) * 0.1,
        "actor_state": actor_state,
        "actor_state_valid_mask": state_valid,
        "actor_valid_mask": actor_valid,
    }
    for name, array in arrays.items():
        assert array.dtype == SIDECAR_ARRAY_DTYPES[name]
        np.save(sidecar_episode / f"{name}.npy", array, allow_pickle=False)


@pytest.fixture()
def mini_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "mini_bundle"
    base = bundle / "platoon_joint_bev"
    with JointBEVDatasetStore(
        base,
        split_config=EpisodeSplitConfig(),
        dataset_fingerprint=FINGERPRINT,
        resume=False,
    ):
        pass
    sidecar = bundle / "riskentry_actor_sidecar"
    _write_json(sidecar / "dataset_contract.json", sidecar_dataset_contract(FINGERPRINT))
    for split in SPLIT_NAMES:
        (sidecar / split / "episodes").mkdir(parents=True)

    manifests: dict[str, list[dict[str, object]]] = {split: [] for split in SPLIT_NAMES}
    _add_episode(
        bundle,
        manifests,
        episode_index=10,
        split="train",
        scenario_id="S5_hard_brake_lead",
        spawn_seed=110,
    )
    _add_episode(
        bundle,
        manifests,
        episode_index=2,
        split="val",
        scenario_id="S5_hard_brake_lead",
        spawn_seed=102,
    )
    _add_episode(
        bundle,
        manifests,
        episode_index=7,
        split="test",
        scenario_id="S6_stopped_vehicle",
        spawn_seed=107,
    )
    _add_episode(
        bundle,
        manifests,
        episode_index=1,
        split="train",
        scenario_id="S5_hard_brake_lead",
        spawn_seed=101,
        add_base=False,
    )
    for split, entries in manifests.items():
        entries.sort(key=lambda item: int(item["episode_index"]))
        _write_json(
            base / split / "manifest.json",
            {
                "schema_version": STORAGE_SCHEMA_VERSION,
                "format": STORAGE_FORMAT,
                "split": split,
                "episode_count": len(entries),
                "joint_samples": sum(int(item["joint_samples"]) for item in entries),
                "episodes": entries,
            },
        )
    return bundle


def test_discovery_filters_sidecar_only_and_sorts_globally(mini_bundle: Path) -> None:
    roots = resolve_dataset_roots(mini_bundle)
    assert resolve_dataset_roots(mini_bundle / "platoon_joint_bev") == roots
    records = discover_episodes(roots)
    assert [record.episode_index for record in records] == [2, 7, 10]
    s5 = episodes_for_scenario(records, "S5_hard_brake_lead")
    assert [(record.episode_index, record.split) for record in s5] == [(2, "val"), (10, "train")]
    selected, total = select_episode(records, "S5_hard_brake_lead", 2)
    assert selected.episode_index == 10
    assert total == 2
    assert default_output_path(roots, selected, 2).as_posix().endswith(
        "mini_bundle/S5_hard_brake_lead/episode_002_global_00000010.mp4"
    )


def test_selection_errors_include_available_range(mini_bundle: Path) -> None:
    records = discover_episodes(resolve_dataset_roots(mini_bundle))
    with pytest.raises(DatasetEpisodeVisualizationError, match="available scenarios"):
        episodes_for_scenario(records, "missing")
    with pytest.raises(DatasetEpisodeVisualizationError, match=r"valid range is 1\.\.2"):
        select_episode(records, "S5_hard_brake_lead", 3)
    with pytest.raises(DatasetEpisodeVisualizationError, match="1-based"):
        select_episode(records, "S5_hard_brake_lead", 0)


def test_contract_and_episode_corruption_fail_fast(mini_bundle: Path) -> None:
    sidecar_contract = mini_bundle / "riskentry_actor_sidecar" / "dataset_contract.json"
    payload = json.loads(sidecar_contract.read_text(encoding="utf-8"))
    payload["base_dataset_fingerprint"] = "b" * 64
    _write_json(sidecar_contract, payload)
    with pytest.raises(DatasetEpisodeVisualizationError, match="fingerprint"):
        resolve_dataset_roots(mini_bundle)


def test_resolve_dataset_roots_accepts_v2_sidecar_contract(tmp_path: Path) -> None:
    bundle = tmp_path / "v2_bundle"
    base = bundle / "platoon_joint_bev"
    with JointBEVDatasetStore(
        base,
        split_config=EpisodeSplitConfig(),
        dataset_fingerprint=FINGERPRINT,
        resume=False,
        planner_version="v2",
    ):
        pass
    base_contract = json.loads(
        (base / "dataset_contract.json").read_text(encoding="utf-8")
    )
    sidecar = bundle / "riskentry_actor_sidecar"
    _write_json(
        sidecar / "dataset_contract.json",
        sidecar_dataset_contract(
            FINGERPRINT,
            base_format=str(base_contract["format"]),
            base_schema_version=int(base_contract["schema_version"]),
        ),
    )

    roots = resolve_dataset_roots(bundle)

    assert roots.dataset_fingerprint == FINGERPRINT
    assert roots.sidecar_root == sidecar
    assert roots.base_schema_version == int(base_contract["schema_version"])
    assert roots.base_format == str(base_contract["format"])
    assert discover_episodes(roots) == ()


def test_vehicle_geometry_and_world_pixel_conversion() -> None:
    polygon = vehicle_polygon_world((10.0, 5.0), 0.0, 4.0, 2.0)
    assert np.allclose(
        polygon,
        np.asarray(((12.0, 6.0), (12.0, 4.0), (8.0, 4.0), (8.0, 6.0))),
    )
    center_pixel = world_to_pixels(
        np.asarray(((10.0, 5.0),)),
        center_xy=(10.0, 5.0),
        view_width_m=40.0,
        view_height_m=20.0,
        image_width=400,
        image_height=272,
    )
    assert center_pixel.tolist() == [[200, 172]]


def test_cli_lists_scenarios_and_episodes(mini_bundle: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--dataset-root", str(mini_bundle), "--list-scenarios"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "S5_hard_brake_lead\t2",
        "S6_stopped_vehicle\t1",
    ]
    assert main(
        [
            "--dataset-root",
            str(mini_bundle),
            "--scenario-id",
            "S5_hard_brake_lead",
            "--list-episodes",
        ]
    ) == 0
    listed = capsys.readouterr().out.splitlines()
    assert "1\tglobal=2\tsplit=val\tseed=102\tduration_s=0.3" in listed
    assert "2\tglobal=10\tsplit=train\tseed=110\tduration_s=0.3" in listed


def test_streaming_mp4_and_overwrite_protection(mini_bundle: Path, tmp_path: Path) -> None:
    records = discover_episodes(resolve_dataset_roots(mini_bundle))
    record, _ = select_episode(records, "S5_hard_brake_lead", 1)
    output = tmp_path / "render.mp4"
    config = RenderConfig(
        width=320,
        height=240,
        fps=10.0,
        camera_mode="follow",
        view_width_m=30.0,
        view_height_m=18.0,
        history_seconds=0.2,
    )
    assert render_episode_video(record, 1, output, config=config) == output.resolve()
    assert output.stat().st_size > 0

    capture = cv2.VideoCapture(str(output))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == record.raw_steps
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == config.width
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == config.height
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(config.fps, abs=0.1)
    ok, frame = capture.read()
    capture.release()
    assert ok and frame.shape == (config.height, config.width, 3)

    with pytest.raises(DatasetEpisodeVisualizationError, match="already exists"):
        render_episode_video(record, 1, output, config=config)
    assert render_episode_video(record, 1, output, config=config, overwrite=True) == output.resolve()


def test_invalid_sidecar_shape_is_rejected(mini_bundle: Path, tmp_path: Path) -> None:
    records = discover_episodes(resolve_dataset_roots(mini_bundle))
    record, _ = select_episode(records, "S5_hard_brake_lead", 1)
    actor_state_path = record.sidecar_directory / "actor_state.npy"
    actor_state = np.load(actor_state_path, allow_pickle=False)
    np.save(actor_state_path, actor_state[:, :3], allow_pickle=False)
    with pytest.raises(DatasetEpisodeVisualizationError, match="invalid actor_state"):
        render_episode_video(record, 1, tmp_path / "invalid-shape.mp4")


def test_non_finite_valid_pose_is_rejected(mini_bundle: Path, tmp_path: Path) -> None:
    records = discover_episodes(resolve_dataset_roots(mini_bundle))
    record, _ = select_episode(records, "S5_hard_brake_lead", 1)
    actor_state_path = record.sidecar_directory / "actor_state.npy"
    actor_state = np.load(actor_state_path, allow_pickle=False)
    actor_state[0, 0, 0] = np.nan
    np.save(actor_state_path, actor_state, allow_pickle=False)
    with pytest.raises(DatasetEpisodeVisualizationError, match="non-finite"):
        render_episode_video(record, 1, tmp_path / "non-finite.mp4")
