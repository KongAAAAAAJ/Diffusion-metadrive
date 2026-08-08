import json
from pathlib import Path

import numpy as np

from evaluation.round13_97g_dataset_quality import audit_bundle_quality


def _write_episode(root: Path, scenario: str, modes: np.ndarray, speed: float) -> None:
    split = "train"
    episode_id = f"episode_{len(list((root / split / 'episodes').glob('*'))):08d}"
    episode = root / split / "episodes" / episode_id
    episode.mkdir(parents=True, exist_ok=True)
    count = modes.shape[0]
    np.save(episode / "gt_mode.npy", modes.astype(np.int64))
    np.save(episode / "mode_valid_mask.npy", np.ones((count, 3, 10), dtype=bool))
    state = np.zeros((count, 3, 8), dtype=np.float32); state[..., 0] = speed
    np.save(episode / "ego_state.npy", state)
    expert = np.zeros((count, 3, 8, 3), dtype=np.float32); expert[..., -1, 0] = speed * 4
    np.save(episode / "expert_trajectory.npy", expert)
    manifest = root / split / "manifest.json"
    payload = json.loads(manifest.read_text()) if manifest.exists() else {"episodes": []}
    payload["episodes"].append({"episode_id": episode_id, "attributes": {"scenario_id": scenario}})
    manifest.write_text(json.dumps(payload))


def test_s7_stop_collapse_blocks_formal_collection(tmp_path: Path):
    base = tmp_path / "platoon_joint_bev"
    for split in ("train", "val", "test"):
        (base / split).mkdir(parents=True)
        (base / split / "manifest.json").write_text('{"episodes": []}')
    for scenario in (
        "S5_hard_brake_lead", "S6_background_merge_in", "S7_ego_merge_from_ramp",
        "S8_ego_exit_to_ramp", "S9_narrow_channel_negotiation",
    ):
        mode = 9 if scenario.startswith("S7") else 1
        _write_episode(base, scenario, np.full((2, 3), mode), 0.0 if mode == 9 else 4.0)
    # Ensure val/test are non-empty without duplicating episode files.
    for split in ("val", "test"):
        src = json.loads((base / "train" / "manifest.json").read_text())["episodes"][0]
        eid = f"episode_{split}"
        dst = base / split / "episodes" / eid; dst.mkdir(parents=True)
        for filename in ("gt_mode.npy", "mode_valid_mask.npy", "ego_state.npy", "expert_trajectory.npy"):
            data = np.load(base / "train" / "episodes" / src["episode_id"] / filename)
            np.save(dst / filename, data)
        (base / split / "manifest.json").write_text(json.dumps({"episodes": [{"episode_id": eid, "attributes": src["attributes"]}]}))
    (tmp_path / "riskentry_actor_sidecar" / "train" / "episodes").mkdir(parents=True)
    report = audit_bundle_quality(tmp_path)
    assert report["status"] == "statistical_quality_blocked"
    assert not report["gates"]["s7_stop_fraction"]
    assert not report["eligible_for_formal_50k_collection"]


def test_s7_focused_corrective_gate_accepts_lateral_motion(tmp_path: Path):
    base = tmp_path / "platoon_joint_bev"
    for split in ("train", "val", "test"):
        (base / split).mkdir(parents=True)
        (base / split / "manifest.json").write_text('{"episodes": []}')
    modes = np.asarray([[3, 3, 3], [1, 1, 1], [1, 1, 1]], dtype=np.int64)
    _write_episode(base, "S7_ego_merge_from_ramp", modes, 6.0)
    (tmp_path / "riskentry_actor_sidecar" / "train" / "episodes").mkdir(parents=True)

    report = audit_bundle_quality(
        tmp_path,
        required_scenarios=("S7_ego_merge_from_ramp",),
        require_all_splits=False,
    )

    assert report["status"] == "statistical_quality_accepted"
    assert report["gates"]["s7_lateral_mode_fraction"]
