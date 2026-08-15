"""Statistical quality gate for the shared BEV/RiskEntry pilot bundle."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from expert_dataset.run_joint_bev_collection import load_run_config


MODE_NAMES = (
    "KEEP_HIGH", "KEEP_MEDIUM", "KEEP_LOW",
    "LEFT_HIGH", "LEFT_MEDIUM", "LEFT_LOW",
    "RIGHT_HIGH", "RIGHT_MEDIUM", "RIGHT_LOW", "STOP",
)
ROLE_NAMES = ("leader", "middle", "rear")
PRIMARY_SCENARIOS = (
    "S5_hard_brake_lead", "S6_background_merge_in",
    "S7_ego_merge_from_ramp", "S8_ego_exit_to_ramp",
    "S9_narrow_channel_negotiation",
)


class DatasetQualityError(RuntimeError):
    """Raised when the bundle cannot be audited without guessing."""


def _episodes(base_root: Path) -> Iterable[tuple[str, Path, Mapping[str, object]]]:
    for split in ("train", "val", "test"):
        manifest_path = base_root / split / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in payload["episodes"]:
            directory = row.get("directory", row.get("episode_id"))
            if not isinstance(directory, str):
                raise DatasetQualityError(f"manifest episode has no directory in {manifest_path}")
            episode_dir = base_root / split / "episodes" / directory
            yield split, episode_dir, row


def _quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "p05": float(np.quantile(array, 0.05)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
    }


def audit_bundle_quality(
    bundle_root: Path,
    *,
    infrastructure_report: Path | None = None,
    s7_max_stop_fraction: float = 0.40,
    s7_max_low_speed_fraction: float = 0.40,
    s7_min_lateral_mode_fraction: float = 0.10,
    required_scenarios: tuple[str, ...] = PRIMARY_SCENARIOS,
    require_all_splits: bool = True,
    formal_collection_config: Path | None = None,
) -> dict[str, object]:
    bundle_root = Path(bundle_root)
    base_root = bundle_root / "platoon_joint_bev"
    sidecar_root = bundle_root / "riskentry_actor_sidecar"
    if not base_root.is_dir() or not sidecar_root.is_dir():
        raise DatasetQualityError("bundle must contain both base and sidecar roots")

    scenario_modes: dict[str, Counter[str]] = defaultdict(Counter)
    role_modes: dict[str, Counter[str]] = defaultdict(Counter)
    scenario_speed: dict[str, list[float]] = defaultdict(list)
    scenario_displacement: dict[str, list[float]] = defaultdict(list)
    scenario_samples: Counter[str] = Counter()
    episode_counts: Counter[str] = Counter()
    scenario_splits: dict[str, set[str]] = defaultdict(set)
    scenario_seeds: dict[str, list[int]] = defaultdict(list)
    scenario_background_counts: dict[str, set[int]] = defaultdict(set)
    scenario_behavior_categories: dict[str, set[str]] = defaultdict(set)
    max_episode_samples: Counter[str] = Counter()
    formal_metadata_valid = True
    base_episode_keys: set[tuple[str, str]] = set()
    split_samples: Counter[str] = Counter()
    global_modes: Counter[str] = Counter()
    valid_counts = np.zeros(len(MODE_NAMES), dtype=np.int64)
    valid_total = 0

    for split, episode_dir, row in _episodes(base_root):
        attrs = row["attributes"]
        base_episode_keys.add((split, episode_dir.name))
        scenario = str(attrs["scenario_id"])
        gt_mode = np.load(episode_dir / "gt_mode.npy", mmap_mode="r")
        mask = np.load(episode_dir / "mode_valid_mask.npy", mmap_mode="r")
        state = np.load(episode_dir / "ego_state.npy", mmap_mode="r")
        expert = np.load(episode_dir / "expert_trajectory.npy", mmap_mode="r")
        if gt_mode.ndim != 2 or gt_mode.shape[1] != 3:
            raise DatasetQualityError(f"bad gt_mode shape in {episode_dir}: {gt_mode.shape}")
        sample_count = int(gt_mode.shape[0])
        scenario_samples[scenario] += sample_count
        split_samples[split] += sample_count
        episode_counts[scenario] += 1
        scenario_splits[scenario].add(split)
        max_episode_samples[scenario] = max(
            max_episode_samples[scenario], sample_count
        )
        spawn_seed = attrs.get("spawn_seed")
        coverage = attrs.get("formal_coverage")
        if isinstance(spawn_seed, bool) or not isinstance(spawn_seed, int):
            formal_metadata_valid = False
        else:
            scenario_seeds[scenario].append(int(spawn_seed))
        if not isinstance(coverage, Mapping):
            formal_metadata_valid = False
        else:
            background_count = coverage.get(
                "incidental_background_actor_count"
            )
            behavior_category = coverage.get("behavior_category")
            if isinstance(background_count, bool) or not isinstance(
                background_count, int
            ):
                formal_metadata_valid = False
            else:
                scenario_background_counts[scenario].add(
                    int(background_count)
                )
            if behavior_category in (None, ""):
                formal_metadata_valid = False
            else:
                scenario_behavior_categories[scenario].add(
                    str(behavior_category)
                )
        for role in range(3):
            for mode in np.asarray(gt_mode[:, role], dtype=np.int64):
                name = MODE_NAMES[int(mode)]
                global_modes[name] += 1
                scenario_modes[scenario][name] += 1
                role_modes[ROLE_NAMES[role]][name] += 1
        valid_counts += np.asarray(mask, dtype=bool).sum(axis=(0, 1))
        valid_total += int(mask.shape[0] * mask.shape[1])
        scenario_speed[scenario].extend(np.asarray(state[..., 0]).reshape(-1).tolist())
        displacement = np.linalg.norm(np.asarray(expert[..., -1, :2]), axis=-1)
        scenario_displacement[scenario].extend(displacement.reshape(-1).tolist())

    event_actor_type: dict[str, Counter[str]] = defaultdict(Counter)
    committed_event_actor_type: dict[str, Counter[str]] = defaultdict(Counter)
    sidecar_only_event_actor_type: dict[str, Counter[str]] = defaultdict(Counter)
    for episode_path in sidecar_root.glob("*/episodes/*/episode.json"):
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        split = str(episode.get("split", episode_path.parents[2].name))
        scoped_counts = (
            committed_event_actor_type
            if (split, episode_path.parent.name) in base_episode_keys
            else sidecar_only_event_actor_type
        )
        actor_type = {
            str(actor["actor_id"]): str(actor["actor_type"])
            for actor in episode["actors"]
        }
        for event in episode["events"]:
            event_type = str(event["event_type"])
            for actor_id in event.get("actor_ids", []):
                resolved_type = actor_type.get(str(actor_id), "unknown")
                event_actor_type[event_type][resolved_type] += 1
                scoped_counts[event_type][resolved_type] += 1

    scenario_stats: dict[str, object] = {}
    audited_scenarios = tuple(
        dict.fromkeys(tuple(required_scenarios) + tuple(sorted(scenario_samples)))
    )
    for scenario in audited_scenarios:
        speeds = scenario_speed[scenario]
        displacements = scenario_displacement[scenario]
        mode_total = sum(scenario_modes[scenario].values())
        if not speeds or not displacements or mode_total <= 0:
            continue
        lateral_total = sum(
            scenario_modes[scenario][MODE_NAMES[index]]
            for index in (3, 4, 5, 6, 7, 8)
        )
        scenario_stats[scenario] = {
            "episodes": int(episode_counts[scenario]),
            "joint_samples": int(scenario_samples[scenario]),
            "role_labels": int(mode_total),
            "gt_mode_counts": dict(scenario_modes[scenario]),
            "stop_fraction": float(scenario_modes[scenario]["STOP"] / mode_total),
            "lateral_mode_fraction": float(lateral_total / mode_total),
            "speed_mps": _quantiles(speeds),
            "speed_below_0_3_fraction": float(np.mean(np.asarray(speeds) < 0.3)),
            "expert_terminal_displacement_m": _quantiles(displacements),
        }

    s7 = scenario_stats.get("S7_ego_merge_from_ramp")
    gates = {
        "required_scenarios_present": all(
            scenario_samples[name] > 0 for name in required_scenarios
        ),
        "required_splits_non_empty": (
            all(split_samples[name] > 0 for name in ("train", "val", "test"))
            if require_all_splits
            else any(split_samples[name] > 0 for name in ("train", "val", "test"))
        ),
        "sidecar_no_platoon_collision_events": all(
            committed_event_actor_type[name]["platoon"] == 0
            for name in (
                "collision_vehicle",
                "collision_object",
                "collision_sidewalk",
            )
        ),
        "sidecar_no_platoon_out_of_road_events": (
            committed_event_actor_type["out_of_road"]["platoon"] == 0
        ),
    }
    if s7 is not None:
        gates.update(
            s7_stop_fraction=bool(
                s7["stop_fraction"] <= s7_max_stop_fraction
            ),
            s7_low_speed_fraction=bool(
                s7["speed_below_0_3_fraction"] <= s7_max_low_speed_fraction
            ),
            s7_lateral_mode_fraction=bool(
                s7["lateral_mode_fraction"] >= s7_min_lateral_mode_fraction
            ),
        )
    formal_contract = None
    if formal_collection_config is not None:
        formal_run = load_run_config(formal_collection_config)
        requirements = formal_run.formal_diversity
        if formal_run.formal_scenario_quotas is None or requirements is None:
            raise DatasetQualityError(
                "formal collection config has no diversity contract"
            )
        quotas = dict(formal_run.formal_scenario_quotas)
        required_splits = set(("train", "val", "test"))
        required_seeds = set(requirements.required_spawn_seeds)
        required_background = set(
            requirements.required_incidental_background_actor_counts
        )
        gates.update(
            formal_scenario_joint_sample_quotas=(
                dict(scenario_samples) == quotas
            ),
            formal_minimum_independent_episodes=all(
                episode_counts[name] >= requirements.min_episodes_per_scenario
                and len(set(scenario_seeds[name])) == episode_counts[name]
                for name in quotas
            ),
            formal_episode_sample_cap=all(
                max_episode_samples[name] <= requirements.max_samples_per_episode
                for name in quotas
            ),
            formal_all_scenarios_in_each_split=all(
                required_splits <= scenario_splits[name]
                for name in quotas
            ),
            formal_required_seed_coverage=all(
                required_seeds <= set(scenario_seeds[name])
                for name in quotas
            ),
            formal_background_count_coverage=all(
                required_background <= scenario_background_counts[name]
                for name in quotas
            ),
            formal_background_counts_in_contract=all(
                scenario_background_counts[name] <= required_background
                for name in quotas
            ),
            formal_behavior_category_coverage=all(
                set(requirements.required_behavior_categories[name])
                <= scenario_behavior_categories[name]
                for name in quotas
            ),
            formal_coverage_metadata_valid=formal_metadata_valid,
        )
        formal_contract = {
            "config": str(Path(formal_collection_config).resolve()),
            "scenario_quotas": quotas,
            "diversity_requirements": requirements.as_dict(),
            "observed": {
                name: {
                    "joint_samples": int(scenario_samples[name]),
                    "episodes": int(episode_counts[name]),
                    "max_samples_per_episode": int(max_episode_samples[name]),
                    "splits": sorted(scenario_splits[name]),
                    "unique_spawn_seeds": len(set(scenario_seeds[name])),
                    "required_spawn_seeds_observed": sorted(
                        required_seeds & set(scenario_seeds[name])
                    ),
                    "incidental_background_actor_counts": sorted(
                        scenario_background_counts[name]
                    ),
                    "behavior_categories": sorted(
                        scenario_behavior_categories[name]
                    ),
                }
                for name in quotas
            },
        }
    infrastructure = None
    if infrastructure_report is not None:
        infrastructure = json.loads(Path(infrastructure_report).read_text(encoding="utf-8"))
        gates["infrastructure_accepted"] = infrastructure.get("status") == "formal_15000_pilot_accepted"
    accepted = all(gates.values())
    return {
        "format": "bev_shared_bundle_statistical_quality_v1",
        "round": "13.97g",
        "bundle_root": str(bundle_root),
        "status": "statistical_quality_accepted" if accepted else "statistical_quality_blocked",
        "eligible_for_formal_50k_collection": bool(accepted),
        "gates": gates,
        "thresholds": {
            "s7_max_stop_fraction": float(s7_max_stop_fraction),
            "s7_max_low_speed_fraction": float(s7_max_low_speed_fraction),
            "s7_min_lateral_mode_fraction": float(
                s7_min_lateral_mode_fraction
            ),
            "required_scenarios": list(required_scenarios),
            "require_all_splits": bool(require_all_splits),
        },
        "split_joint_samples": dict(split_samples),
        "scenario_statistics": scenario_stats,
        "global_gt_mode_counts": dict(global_modes),
        "role_gt_mode_counts": {name: dict(value) for name, value in role_modes.items()},
        "mode_valid_rate": {
            MODE_NAMES[index]: float(valid_counts[index] / valid_total)
            for index in range(len(MODE_NAMES))
        },
        "sidecar_event_actor_types": {
            name: dict(value) for name, value in event_actor_type.items()
        },
        "sidecar_safety_audit": {
            "base_committed_platoon_collision_events": int(
                sum(
                    committed_event_actor_type[name]["platoon"]
                    for name in (
                        "collision_vehicle",
                        "collision_object",
                        "collision_sidewalk",
                    )
                )
            ),
            "base_committed_platoon_out_of_road_events": int(
                committed_event_actor_type["out_of_road"]["platoon"]
            ),
            "sidecar_only_platoon_collision_events_retained": int(
                sum(
                    sidecar_only_event_actor_type[name]["platoon"]
                    for name in (
                        "collision_vehicle",
                        "collision_object",
                        "collision_sidewalk",
                    )
                )
            ),
            "sidecar_only_platoon_out_of_road_events_retained": int(
                sidecar_only_event_actor_type["out_of_road"]["platoon"]
            ),
            "external_collision_events_retained_as_context": int(
                sum(
                    event_actor_type[name]["external"]
                    for name in (
                        "collision_vehicle",
                        "collision_object",
                        "collision_sidewalk",
                    )
                )
            ),
            "external_out_of_road_events_retained_as_context": int(
                event_actor_type["out_of_road"]["external"]
            ),
            "interpretation": (
                "Raw external-actor events remain losslessly retained for RiskEntry; "
                "formal BEV safety eligibility is gated on platoon actors from "
                "base-committed episodes only. Sidecar-only failures remain evidence."
            ),
        },
        "infrastructure_report_status": None if infrastructure is None else infrastructure.get("status"),
        "formal_collection_contract": formal_contract,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_root", type=Path)
    parser.add_argument("--infrastructure-report", type=Path)
    parser.add_argument("--formal-collection-config", type=Path)
    parser.add_argument(
        "--required-scenarios",
        nargs="+",
        default=list(PRIMARY_SCENARIOS),
    )
    parser.add_argument("--allow-empty-splits", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_bundle_quality(
        args.bundle_root,
        infrastructure_report=args.infrastructure_report,
        required_scenarios=tuple(args.required_scenarios),
        require_all_splits=not args.allow_empty_splits,
        formal_collection_config=args.formal_collection_config,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": report["gates"]}, indent=2))


if __name__ == "__main__":
    main()
