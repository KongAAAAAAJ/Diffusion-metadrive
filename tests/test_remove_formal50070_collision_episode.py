from __future__ import annotations

import pytest

from tools.remove_formal50070_collision_episode import (
    Formal50070EpisodeRemovalError,
    REMOVAL_REASON,
    _curated_base_manifest,
    _curated_row,
    _curated_sidecar_manifest,
)


def _source_row() -> dict[str, object]:
    return {
        "episode_index": 194,
        "split": "train",
        "scenario_id": "S6_background_merge_in",
        "local_route": "R6_mainline_merge_approach",
        "spawn_seed": 1751228647,
        "base_status": "committed",
        "base_rejection_reason": None,
        "sidecar_status": "committed",
        "sidecar_rejection_reason": None,
        "raw_steps": 261,
        "base_samples": 200,
        "outcome": "collision",
    }


def test_curation_rejects_only_base_and_retains_sidecar_evidence() -> None:
    curated = _curated_row(_source_row())

    assert curated["base_status"] == "rejected"
    assert curated["base_rejection_reason"] == REMOVAL_REASON
    assert curated["base_samples"] == 0
    assert curated["sidecar_status"] == "committed"
    assert curated["raw_steps"] == 261
    assert curated["outcome"] == "collision"


def test_curation_updates_base_and_sidecar_manifests() -> None:
    base = {
        "episode_count": 2,
        "joint_samples": 390,
        "episodes": [
            {"episode_index": 193, "joint_samples": 190},
            {"episode_index": 194, "joint_samples": 200},
        ],
    }
    sidecar = {
        "episode_count": 2,
        "base_samples": 390,
        "episodes": [
            {"episode_index": 193, "base_samples": 190, "outcome": "success"},
            {"episode_index": 194, "base_samples": 200, "outcome": "collision"},
        ],
    }

    curated_base = _curated_base_manifest(base)
    curated_sidecar = _curated_sidecar_manifest(sidecar)

    assert curated_base["episode_count"] == 1
    assert curated_base["joint_samples"] == 190
    assert [row["episode_index"] for row in curated_base["episodes"]] == [193]
    assert curated_sidecar["episode_count"] == 2
    assert curated_sidecar["base_samples"] == 190
    assert curated_sidecar["episodes"][1]["base_samples"] == 0


def test_curation_stops_if_the_frozen_target_row_drifted() -> None:
    source = _source_row()
    source["spawn_seed"] = 1

    with pytest.raises(Formal50070EpisodeRemovalError, match="drifted"):
        _curated_row(source)
