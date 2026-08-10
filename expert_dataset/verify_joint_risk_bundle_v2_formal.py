"""Verify the frozen cross-partition RiskEntry bundle-v2 formal gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from expert_dataset.joint_bev_storage import EpisodeSplitAssigner, EpisodeSplitConfig


PARTITIONS = ("id", "compositional_ood", "topology_ood")
FUTURE_OFFSETS = (10, 30, 50)
PLATOON_IDS = ("P0", "P1", "P2")
MAX_EXTERNAL_CANDIDATES = 12


class FormalBundleVerificationError(RuntimeError):
    """Raised when a formal bundle violates a frozen cross-project gate."""


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FormalBundleVerificationError(f"JSON object required: {path}")
    return payload


def _rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FormalBundleVerificationError(f"missing bundle index: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not isinstance(row, dict) for row in rows):
        raise FormalBundleVerificationError(f"invalid bundle index row: {path}")
    return rows


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _static_lane_graph(metadata: Mapping[str, object]) -> dict[str, object]:
    actors = metadata.get("lanes")
    topology = metadata.get("topology")
    if not isinstance(actors, list) or not isinstance(topology, Mapping):
        raise FormalBundleVerificationError("episode lacks lanes/topology metadata")
    lane_sources = {
        str(row["lane_id"]): str(row["source_lane_id"])
        for row in actors
        if isinstance(row, Mapping)
    }
    relations = []
    for row in topology.get("lane_relations", []):
        if not isinstance(row, Mapping):
            raise FormalBundleVerificationError("invalid topology relation")
        source = lane_sources.get(str(row.get("source_lane_id")))
        target = lane_sources.get(str(row.get("target_lane_id")))
        if source is None or target is None:
            raise FormalBundleVerificationError("topology relation does not resolve")
        relations.append((source, str(row.get("relation")), target))
    return {
        "physical_route": topology.get("physical_route"),
        "source_lane_ids": sorted(lane_sources.values()),
        "relations": sorted(relations),
    }


def _episode_dir(partition_root: Path, row: Mapping[str, object]) -> Path:
    episode_index = int(row["episode_index"])
    split = str(row["split"])
    return (
        partition_root
        / "riskentry_actor_sidecar"
        / split
        / "episodes"
        / f"episode_{episode_index:08d}"
    )


def _actor_index(metadata: Mapping[str, object], actor_id: str) -> int:
    actors = metadata.get("actors")
    if not isinstance(actors, list):
        raise FormalBundleVerificationError("episode actors must be an array")
    matches = [
        int(row["actor_index"])
        for row in actors
        if isinstance(row, Mapping) and row.get("actor_id") == actor_id
    ]
    if len(matches) != 1:
        raise FormalBundleVerificationError(f"actor {actor_id} is not unique")
    return matches[0]


def _validate_anchor(
    metadata: Mapping[str, object], episode_dir: Path, anchor: int
) -> None:
    actor_valid = np.load(
        episode_dir / "actor_valid_mask.npy", mmap_mode="r", allow_pickle=False
    )
    state_valid = np.load(
        episode_dir / "actor_state_valid_mask.npy", mmap_mode="r", allow_pickle=False
    )
    timeline = int(actor_valid.shape[0])
    p_indices = [_actor_index(metadata, actor_id) for actor_id in PLATOON_IDS]
    for offset in FUTURE_OFFSETS:
        future = int(anchor) + offset
        if future >= timeline:
            raise FormalBundleVerificationError(
                f"anchor {anchor} lacks the {offset / 10:g}s future"
            )
        if not bool(np.asarray(actor_valid[future, p_indices]).all()):
            raise FormalBundleVerificationError(
                f"platoon truth invalid at anchor={anchor}, future={future}"
            )
        if not bool(np.asarray(state_valid[future, p_indices, :]).all()):
            raise FormalBundleVerificationError(
                f"platoon state invalid at anchor={anchor}, future={future}"
            )


def _validate_near_source(
    metadata: Mapping[str, object], episode_dir: Path, anchors: Sequence[int]
) -> dict[str, object]:
    key_actor_ids = metadata.get("key_actor_ids")
    entry = metadata.get("entry_event")
    if not isinstance(key_actor_ids, Mapping) or not key_actor_ids:
        raise FormalBundleVerificationError("near-critical key_actor_ids is empty")
    if not isinstance(entry, Mapping):
        raise FormalBundleVerificationError("near-critical entry_event is missing")
    source_id = str(entry.get("source_actor_id"))
    target_id = str(entry.get("target_actor_id"))
    if source_id not in set(map(str, key_actor_ids.values())):
        raise FormalBundleVerificationError("entry source is not a key actor")
    source_index = _actor_index(metadata, source_id)
    target_index = _actor_index(metadata, target_id)
    p_indices = [_actor_index(metadata, actor_id) for actor_id in PLATOON_IDS]
    onset = entry.get("onset_step")
    if isinstance(onset, bool) or not isinstance(onset, int) or not 60 <= onset <= 120:
        raise FormalBundleVerificationError(
            f"near-critical onset_step is outside [60,120]: {onset}"
        )

    actor_valid = np.load(
        episode_dir / "actor_valid_mask.npy", mmap_mode="r", allow_pickle=False
    )
    state_valid = np.load(
        episode_dir / "actor_state_valid_mask.npy", mmap_mode="r", allow_pickle=False
    )
    observed = np.load(
        episode_dir / "actor_observation_mask.npy", mmap_mode="r", allow_pickle=False
    )
    state = np.load(episode_dir / "actor_state.npy", mmap_mode="r", allow_pickle=False)
    if onset >= actor_valid.shape[0]:
        raise FormalBundleVerificationError("entry onset is outside the timeline")
    if not bool(np.asarray(actor_valid[onset, p_indices]).all()):
        raise FormalBundleVerificationError("platoon truth is invalid at entry onset")
    if not bool(np.asarray(state_valid[onset, p_indices, :]).all()):
        raise FormalBundleVerificationError("platoon state is invalid at entry onset")
    if not bool(actor_valid[onset, source_index]) or not bool(
        actor_valid[onset, target_index]
    ):
        raise FormalBundleVerificationError("entry edge is invalid at onset")
    if not bool(observed[onset, source_index]):
        raise FormalBundleVerificationError("entry source is not observed at onset")

    actors = metadata["actors"]
    external_indices = [
        int(row["actor_index"])
        for row in actors
        if isinstance(row, Mapping) and row.get("actor_type") == "external"
    ]
    for anchor in anchors:
        if not bool(observed[anchor, source_index]):
            raise FormalBundleVerificationError(
                f"entry source is not observed at anchor {anchor}"
            )
        center = np.asarray(state[anchor, p_indices, :2], dtype=np.float64).mean(axis=0)
        candidates = [
            index
            for index in external_indices
            if bool(observed[anchor, index])
            and bool(actor_valid[anchor, index])
            and bool(np.asarray(state_valid[anchor, index, :2]).all())
        ]
        ranked = sorted(
            candidates,
            key=lambda index: (
                float(np.linalg.norm(state[anchor, index, :2] - center)), index
            ),
        )[:MAX_EXTERNAL_CANDIDATES]
        if source_index not in ranked:
            raise FormalBundleVerificationError(
                f"entry source is outside top-{MAX_EXTERNAL_CANDIDATES} at anchor {anchor}"
            )
    return {"source_actor_id": source_id, "target_actor_id": target_id, "onset": onset}


def verify_formal_bundle(root: Path | str) -> dict[str, object]:
    root = Path(root).expanduser().resolve()
    contract = _json(root / "formal_v2_run_contract.json")
    prefix = str(contract["dataset_instance_prefix"])
    split_seed = int(contract["split_seed"])
    target_windows = {str(k): int(v) for k, v in contract["target_windows"].items()}
    assigner = EpisodeSplitAssigner(EpisodeSplitConfig(seed=split_seed))

    topology_ids: dict[str, set[str]] = defaultdict(set)
    static_graphs: dict[str, set[str]] = defaultdict(set)
    report_partitions: dict[str, object] = {}
    all_pairs: dict[str, list[tuple[str, str, int]]] = defaultdict(list)

    for partition in PARTITIONS:
        partition_root = root / f"{prefix}_{partition}_v2"
        rows = _rows(partition_root / "bundle_episode_index.jsonl")
        windows = 0
        families: Counter[str] = Counter()
        splits: Counter[str] = Counter()
        near_events: list[dict[str, object]] = []
        near_candidate_windows = 0
        for row in rows:
            episode_dir = _episode_dir(partition_root, row)
            metadata = _json(episode_dir / "episode.json")
            if int(metadata["episode_index"]) != int(row["episode_index"]):
                raise FormalBundleVerificationError("episode index metadata mismatch")
            if str(metadata["split"]) != str(row["split"]):
                raise FormalBundleVerificationError("episode split metadata mismatch")
            pair_id = str(metadata["matched_pair_id"])
            severity = str(metadata["severity"])
            spawn_seed = int(metadata["spawn_seed"])
            all_pairs[pair_id].append((str(row["split"]), severity, spawn_seed))
            if partition == "id":
                expected_split = assigner.split_for_key(pair_id)
                if str(row["split"]) != expected_split:
                    raise FormalBundleVerificationError(
                        f"matched pair {pair_id} has incorrect ID split"
                    )
            elif str(row["split"]) != "test":
                raise FormalBundleVerificationError(f"{partition} must use test split")

            anchors = np.load(
                episode_dir / "base_sample_step_index.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            anchor_values = tuple(int(value) for value in np.asarray(anchors))
            if anchor_values != tuple(int(value) for value in metadata["eligible_anchor_steps"]):
                raise FormalBundleVerificationError("eligible anchor metadata mismatch")
            for anchor in anchor_values:
                _validate_anchor(metadata, episode_dir, anchor)
            if severity == "near_critical":
                near_events.append(_validate_near_source(metadata, episode_dir, anchor_values))
                near_candidate_windows += len(anchor_values)

            topology = metadata.get("topology")
            if not isinstance(topology, Mapping):
                raise FormalBundleVerificationError("episode topology is missing")
            topology_id = str(metadata["topology_id"])
            if topology_id != str(topology.get("topology_id")):
                raise FormalBundleVerificationError("topology_id metadata mismatch")
            if topology_id.startswith(("id_", "topology_ood_", "compositional_ood_")):
                raise FormalBundleVerificationError("topology_id contains a partition prefix")
            topology_ids[partition].add(topology_id)
            static_graphs[partition].add(_canonical_hash(_static_lane_graph(metadata)))
            windows += len(anchor_values)
            families[str(metadata["scenario_family"])] += len(anchor_values)
            splits[str(row["split"])] += 1

        if windows != target_windows[partition]:
            raise FormalBundleVerificationError(
                f"{partition} eligible windows={windows}, expected={target_windows[partition]}"
            )
        report_partitions[partition] = {
            "episodes": len(rows),
            "eligible_windows": windows,
            "future_truth_checkpoints": windows * len(FUTURE_OFFSETS),
            "near_source_candidate_windows": near_candidate_windows,
            "windows_by_family": dict(sorted(families.items())),
            "episodes_by_split": dict(sorted(splits.items())),
            "near_events": near_events,
            "topology_ids": sorted(topology_ids[partition]),
            "static_graph_signatures": sorted(static_graphs[partition]),
        }

    for pair_id, members in all_pairs.items():
        if len(members) != 2:
            raise FormalBundleVerificationError(
                f"matched pair {pair_id} has {len(members)} members"
            )
        if {severity for _, severity, _ in members} != {"control", "near_critical"}:
            raise FormalBundleVerificationError(f"matched pair {pair_id} is incomplete")
        if len({split for split, _, _ in members}) != 1:
            raise FormalBundleVerificationError(f"matched pair {pair_id} crosses splits")
        if len({seed for _, _, seed in members}) != 1:
            raise FormalBundleVerificationError(f"matched pair {pair_id} changes seed")

    topology_overlap = topology_ids["id"] & topology_ids["topology_ood"]
    graph_overlap = static_graphs["id"] & static_graphs["topology_ood"]
    if topology_overlap:
        raise FormalBundleVerificationError(
            f"ID/topology-OOD topology IDs overlap: {sorted(topology_overlap)}"
        )
    if graph_overlap:
        raise FormalBundleVerificationError(
            f"ID/topology-OOD static lane graphs overlap: {sorted(graph_overlap)}"
        )

    return {
        "format": "metadrive-joint-risk-bundle-v2-formal-verification",
        "passed": True,
        "root": str(root),
        "eligible_windows_total": sum(target_windows.values()),
        "matched_pairs": len(all_pairs),
        "partitions": report_partitions,
        "cross_partition": {
            "topology_id_overlap": [],
            "static_lane_graph_overlap": [],
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = verify_formal_bundle(args.root)
    encoded = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FormalBundleVerificationError", "verify_formal_bundle"]
