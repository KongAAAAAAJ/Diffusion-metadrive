"""Audit the S5 targeted supplement and build its logical composition manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Mapping

import numpy as np

from expert_dataset.run_joint_bev_collection import (
    FORMAL_SPLITS,
    TARGETED_LATERAL_MODES,
    JointCollectionRunConfig,
    _count_true_runs,
    _lateral_run_directions,
    load_run_config,
)
from expert_dataset.verify_joint_risk_bundle import verify_joint_risk_bundle
from scenarios.bev_round13_contract import (
    CANDIDATE_V3_CONTRACT_ID,
    CANDIDATE_V4_CONTRACT_ID,
)


DEFAULT_ORIGINAL_BUNDLE = Path(
    "/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/"
    "bev_joint_risk_candidate_v3_formal50k_v1"
)
DEFAULT_COMPOSITION_PATH = Path(
    "/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/"
    "bev_joint_risk_candidate_v3_formal50k_plus_s5_release20_v1/"
    "dataset_composition.json"
)
DEFAULT_CANDIDATE_V4_COMPOSITION_PATH = Path(
    "/media/kong/Elements_SE/Diffusion_Data/metadrive_datasets/"
    "bev_joint_risk_candidate_v3_formal50k_plus_candidate_v4_s5_release20_v1/"
    "dataset_composition.json"
)


class TargetedSupplementAuditError(RuntimeError):
    """Raised when the supplement is incomplete or violates its contract."""


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TargetedSupplementAuditError(f"JSON root must be an object: {path}")
    return payload


def _read_rows(bundle_root: Path) -> list[dict[str, object]]:
    rows = []
    path = bundle_root / "bundle_episode_index.jsonl"
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TargetedSupplementAuditError("bundle index row must be an object")
        rows.append(row)
    if [int(row["episode_index"]) for row in rows] != list(range(len(rows))):
        raise TargetedSupplementAuditError("bundle episode indexes are not contiguous")
    return rows


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _episode_path(
    bundle_root: Path, component: str, split: str, episode_index: int
) -> Path:
    return (
        bundle_root
        / component
        / split
        / "episodes"
        / f"episode_{episode_index:08d}"
    )


def _physical_episode_audit(
    episode_path: Path,
    *,
    minimum_lateral_range_m: float,
    maximum_return_error_m: float,
) -> dict[str, object]:
    gt_mode = np.load(episode_path / "gt_mode.npy", mmap_mode="r", allow_pickle=False)
    pose = np.load(
        episode_path / "ego_pose_global.npy", mmap_mode="r", allow_pickle=False
    )
    if gt_mode.ndim != 2 or gt_mode.shape[1] != 3 or pose.shape != (
        gt_mode.shape[0],
        3,
        3,
    ):
        raise TargetedSupplementAuditError(
            f"malformed targeted arrays: {episode_path}"
        )
    if not np.isfinite(pose).all():
        raise TargetedSupplementAuditError(
            f"non-finite targeted pose evidence: {episode_path}"
        )
    runs = []
    directions = []
    ranges = []
    returns = []
    for role_index in range(3):
        runs.append(
            _count_true_runs(
                np.isin(gt_mode[:, role_index], tuple(TARGETED_LATERAL_MODES))
            )
        )
        directions.append(_lateral_run_directions(gt_mode[:, role_index]))
        y = np.asarray(pose[:, role_index, 1], dtype=np.float64)
        ranges.append(float(np.ptp(y)))
        returns.append(float(abs(y[-1] - y[0])))
    passed = (
        runs == [2, 2, 2]
        and all(
            len(values) == 2
            and values[0] != values[1]
            and "mixed" not in values
            for values in directions
        )
        and len({values[0] for values in directions}) >= 2
        and all(value >= minimum_lateral_range_m for value in ranges)
        and all(value <= maximum_return_error_m for value in returns)
    )
    return {
        "passed": passed,
        "lateral_mode_runs_by_role": runs,
        "lateral_run_directions_by_role": directions,
        "lateral_range_m_by_role": ranges,
        "return_error_m_by_role": returns,
    }


def audit_targeted_supplement(
    config: JointCollectionRunConfig,
    *,
    stop_reason: str | None = None,
    initial_gate_failure_waived: bool = False,
) -> dict[str, object]:
    requirements = config.targeted_supplement
    if requirements is None:
        raise TargetedSupplementAuditError("config is not targeted_supplement mode")
    rows = _read_rows(config.bundle_root)
    rejection_reasons: Counter[str] = Counter()
    behavior_categories: Counter[str] = Counter()
    safety_event_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    background_counts: Counter[int] = Counter()
    seeds: list[int] = []
    attempt_seeds: list[int] = []
    attempt_audit_rows: list[dict[str, object]] = []
    initial_gate_rows: list[dict[str, object]] = []
    physical_rows = []
    accepted_contract_rows = []
    simulator_attempts = 0
    for row in rows:
        if row.get("outcome") == "scheduler_skip":
            continue
        episode_safety_events: set[str] = set()
        simulator_attempts += 1
        attempt_seed = int(row["spawn_seed"])
        attempt_seeds.append(attempt_seed)
        if row.get("base_status") != "committed":
            rejection_reasons[str(row.get("base_rejection_reason"))] += 1
        sidecar_metadata = None
        if row.get("sidecar_status") == "committed":
            sidecar_metadata = _read_json(
                _episode_path(
                    config.bundle_root,
                    config.sidecar_root.name,
                    str(row["split"]),
                    int(row["episode_index"]),
                )
                / "episode.json"
            )
            parameters = sidecar_metadata.get("scenario_parameters", {})
            if isinstance(parameters, Mapping):
                observed = parameters.get("observed_behavior_class")
                behavior_categories[str(observed or "unknown")] += 1
            events = sidecar_metadata.get("events", [])
            if isinstance(events, list):
                for event in events:
                    if isinstance(event, Mapping):
                        actor_ids = event.get("actor_ids", [])
                        if isinstance(actor_ids, list) and any(
                            actor_id in {"P0", "P1", "P2"}
                            for actor_id in actor_ids
                        ):
                            episode_safety_events.add(
                                str(event.get("event_type"))
                            )
            safety_event_counts.update(episode_safety_events)
        parameters = (
            sidecar_metadata.get("scenario_parameters", {})
            if isinstance(sidecar_metadata, Mapping)
            else {}
        )
        if not isinstance(parameters, Mapping):
            parameters = {}
        attempt_audit_row = {
            "attempt": simulator_attempts,
            "episode_index": int(row["episode_index"]),
            "spawn_seed": attempt_seed,
            "base_committed": row.get("base_status") == "committed",
            "sidecar_committed": row.get("sidecar_status") == "committed",
            "target_background_condition_sampled": parameters.get(
                "target_background_condition_sampled"
            ),
            "scenario_contract_id": parameters.get("scenario_contract_id"),
            "sampling_policy_id": parameters.get("sampling_policy_id"),
            "target_background_condition_realized": parameters.get(
                "target_background_condition_realized"
            ),
            "left_relation": parameters.get("left_relation"),
            "right_relation": parameters.get("right_relation"),
            "left_offset_m": parameters.get("left_offset_m"),
            "right_offset_m": parameters.get("right_offset_m"),
            "actual_background_evidence": parameters.get(
                "s5_adjacent_background_realization"
            ),
            "behavior_category": parameters.get("observed_behavior_class"),
            "mixed_direction_lane_change_completed": parameters.get(
                "mixed_direction_lane_change_completed"
            ),
            "reassembly_completed": parameters.get("reassembly_completed"),
            "formation_recovered_after_hazard": parameters.get(
                "formation_recovered_after_hazard"
            ),
            "rejection_reason": row.get("base_rejection_reason"),
        }
        attempt_audit_rows.append(attempt_audit_row)
        if simulator_attempts <= requirements.initial_feasibility_attempts:
            initial_gate_rows.append(attempt_audit_row)
        if row.get("base_status") != "committed":
            continue
        split = str(row["split"])
        episode_index = int(row["episode_index"])
        metadata = _read_json(
            _episode_path(
                config.bundle_root,
                config.dataset_root.name,
                split,
                episode_index,
            )
            / "episode.json"
        )
        attributes = metadata.get("attributes", {})
        evidence = (
            attributes.get("targeted_supplement_evidence", {})
            if isinstance(attributes, Mapping)
            else {}
        )
        if not isinstance(evidence, Mapping):
            evidence = {}
        split_counts[split] += 1
        background_counts[int(evidence.get("incidental_background_actor_count", -1))] += 1
        seeds.append(int(row["spawn_seed"]))
        physical = _physical_episode_audit(
            _episode_path(
                config.bundle_root,
                config.dataset_root.name,
                split,
                episode_index,
            ),
            minimum_lateral_range_m=requirements.minimum_lateral_range_m,
            maximum_return_error_m=requirements.maximum_return_error_m,
        )
        sidecar_events = (
            sidecar_metadata.get("events", [])
            if isinstance(sidecar_metadata, Mapping)
            else []
        )
        unsafe_sidecar_events = []
        if isinstance(sidecar_events, list):
            for event in sidecar_events:
                if not isinstance(event, Mapping):
                    continue
                actor_ids = event.get("actor_ids", [])
                if (
                    event.get("event_type")
                    in {
                        "collision_vehicle",
                        "collision_object",
                        "collision_sidewalk",
                        "out_of_road",
                    }
                    and isinstance(actor_ids, list)
                    and any(actor in {"P0", "P1", "P2"} for actor in actor_ids)
                ):
                    unsafe_sidecar_events.append(str(event.get("event_type")))
        contract_passed = all(
            (
                int(metadata.get("joint_samples", -1))
                == requirements.required_samples_per_episode,
                attributes.get("scenario_id") == requirements.scenario_id,
                attributes.get("rule_maker_profile_id")
                == requirements.rule_maker_profile_id,
                evidence.get("behavior_category")
                == requirements.behavior_category,
                evidence.get("platoon_safety_events") == [],
                not unsafe_sidecar_events,
                config.scenario_contract_id != CANDIDATE_V4_CONTRACT_ID
                or attributes.get("scenario_contract_id")
                == CANDIDATE_V4_CONTRACT_ID,
            )
        )
        accepted_contract_rows.append(contract_passed)
        physical_rows.append(
            {
                "episode_index": episode_index,
                "split": split,
                "acceptance_contract_passed": contract_passed,
                "unsafe_sidecar_events": sorted(set(unsafe_sidecar_events)),
                **physical,
            }
        )
    accepted = len(physical_rows)
    pending_transaction = any(
        (
            (config.bundle_root / ".bundle_episode_pending.json").exists(),
            (config.bundle_root / ".targeted_batch_pending.json").exists(),
        )
    )
    complete = all(
        (
            accepted == requirements.target_episodes,
            dict(split_counts) == dict(requirements.accepted_episode_quotas),
            dict(background_counts)
            == dict(requirements.accepted_background_count_quotas),
            len(seeds) == len(set(seeds)),
            all(accepted_contract_rows),
            all(bool(row["passed"]) for row in physical_rows),
            not pending_transaction,
        )
    )
    gate_attempts_complete = (
        len(initial_gate_rows) == requirements.initial_feasibility_attempts
    )
    initial_attempt_seeds = attempt_seeds[
        : requirements.initial_feasibility_attempts
    ]
    gate_seed_unique = len(initial_attempt_seeds) == len(
        set(initial_attempt_seeds)
    )
    gate_accepted = sum(
        bool(row["base_committed"]) for row in initial_gate_rows
    )
    initial_gate_passed = bool(
        gate_attempts_complete
        and gate_seed_unique
        and gate_accepted
        >= requirements.initial_feasibility_min_accepted_episodes
        and not pending_transaction
    )
    if config.scenario_contract_id == CANDIDATE_V4_CONTRACT_ID:
        sampled_target_rows = [
            row
            for row in initial_gate_rows
            if row["target_background_condition_sampled"] is True
        ]
        realized_target_rows = [
            row
            for row in initial_gate_rows
            if row["target_background_condition_realized"] is True
        ]
        expected_seeds = list(
            requirements.bootstrap_spawn_seeds[
                : requirements.initial_feasibility_attempts
            ]
        )
        evidence_complete = all(
            isinstance(row["target_background_condition_sampled"], bool)
            and isinstance(row["target_background_condition_realized"], bool)
            and row["target_background_condition_sampled"]
            == row["target_background_condition_realized"]
            and row["scenario_contract_id"] == CANDIDATE_V4_CONTRACT_ID
            and row["sampling_policy_id"] == "s5_release_enriched_80_v1"
            for row in initial_gate_rows
        )
        all_target_rows_accepted = all(
            bool(row["base_committed"]) for row in sampled_target_rows
        )
        initial_gate_passed = bool(
            initial_gate_passed
            and len(sampled_target_rows) == 8
            and len(realized_target_rows) == 8
            and all_target_rows_accepted
            and evidence_complete
            and initial_attempt_seeds == expected_seeds
        )
    else:
        sampled_target_rows = []
        realized_target_rows = []
    initial_gate_waiver_applied = bool(
        initial_gate_failure_waived
        and gate_attempts_complete
        and not initial_gate_passed
    )
    complete = bool(
        complete and (initial_gate_passed or initial_gate_waiver_applied)
    )

    failure_categories = {
        "scenario_not_realized": int(
            rejection_reasons["targeted_scenario_not_realized"]
        ),
        "sustained_emergency_braking": int(
            rejection_reasons["targeted_sustained_emergency_braking"]
        ),
        "release_without_recovery": int(
            sum(
                1
                for row in attempt_audit_rows
                if row["mixed_direction_lane_change_completed"] is True
                and row["reassembly_completed"] is not True
            )
        ),
        "formation_not_stable": sum(
            1
            for row in attempt_audit_rows
            if row["behavior_category"]
            == "temporary_formation_release_and_recovery"
            and row["reassembly_completed"] is True
            and row["formation_recovered_after_hazard"] is not True
        ),
        "collision": sum(
            int(safety_event_counts[name])
            for name in (
                "collision_vehicle",
                "collision_object",
                "collision_sidewalk",
            )
        ),
        "out_of_road": int(safety_event_counts["out_of_road"]),
        "safety_rejected": int(rejection_reasons["targeted_safety_rejected"]),
        "trajectory_infeasible": int(
            rejection_reasons["targeted_trajectory_infeasible"]
        ),
        "transaction_failure": sum(
            1
            for row in attempt_audit_rows
            if not bool(row["sidecar_committed"])
        ),
    }
    return {
        "format": "s5-targeted-supplement-audit-v2",
        "bundle_root": str(config.bundle_root),
        "dataset_fingerprint": config.immutable_fingerprint(),
        "stop_reason": stop_reason,
        "complete": complete,
        "accepted_episodes": accepted,
        "simulator_attempts": simulator_attempts,
        "scheduler_skips": len(rows) - simulator_attempts,
        "split_counts": {name: int(split_counts[name]) for name in FORMAL_SPLITS},
        "background_count_counts": {
            str(key): int(background_counts[key])
            for key in requirements.accepted_background_count_quotas
        },
        "duplicate_spawn_seeds": sorted(
            seed for seed, count in Counter(seeds).items() if count > 1
        ),
        "behavior_category_counts": dict(sorted(behavior_categories.items())),
        "rejection_reason_counts": dict(sorted(rejection_reasons.items())),
        "failure_category_counts": failure_categories,
        "initial_gate": {
            "passed": initial_gate_passed,
            "failure_waiver_applied": initial_gate_waiver_applied,
            "effective_passed": bool(
                initial_gate_passed or initial_gate_waiver_applied
            ),
            "attempts": len(initial_gate_rows),
            "required_attempts": requirements.initial_feasibility_attempts,
            "accepted_episodes": gate_accepted,
            "minimum_accepted_episodes": (
                requirements.initial_feasibility_min_accepted_episodes
            ),
            "unique_spawn_seeds": gate_seed_unique,
            "target_background_episodes_sampled": len(sampled_target_rows),
            "target_background_episodes_realized": len(realized_target_rows),
            "rows": initial_gate_rows,
        },
        "physical_episode_audit": physical_rows,
        "pending_transaction": pending_transaction,
        "requirements": requirements.as_dict(),
    }


def write_targeted_supplement_report(
    config: JointCollectionRunConfig,
    *,
    stop_reason: str | None = None,
    initial_gate_failure_waived: bool = False,
) -> dict[str, object]:
    report = audit_targeted_supplement(
        config,
        stop_reason=stop_reason,
        initial_gate_failure_waived=initial_gate_failure_waived,
    )
    _atomic_write_json(
        config.bundle_root / "targeted_supplement_report.json", report
    )
    return report


def _source_statistics(bundle_root: Path) -> dict[str, object]:
    manifest = _read_json(bundle_root / "dataset_bundle_manifest.json")
    rows = _read_rows(bundle_root)
    split_samples: Counter[str] = Counter()
    split_episodes: Counter[str] = Counter()
    scenarios: Counter[str] = Counter()
    behaviors: Counter[str] = Counter()
    for row in rows:
        if row.get("base_status") != "committed":
            continue
        split = str(row["split"])
        metadata = _read_json(
            _episode_path(
                bundle_root,
                str(manifest["base_directory"]),
                split,
                int(row["episode_index"]),
            )
            / "episode.json"
        )
        samples = int(metadata["joint_samples"])
        attributes = metadata.get("attributes", {})
        if not isinstance(attributes, Mapping):
            raise TargetedSupplementAuditError("episode attributes are missing")
        scenario = str(attributes.get("scenario_id"))
        coverage = attributes.get("formal_coverage")
        targeted = attributes.get("targeted_supplement_evidence")
        evidence = coverage if isinstance(coverage, Mapping) else targeted
        behavior = (
            str(evidence.get("behavior_category"))
            if isinstance(evidence, Mapping)
            else "unknown"
        )
        split_samples[split] += samples
        split_episodes[split] += 1
        scenarios[scenario] += 1
        behaviors[f"{scenario}/{behavior}"] += 1
    index_path = bundle_root / "bundle_episode_index.jsonl"
    return {
        "path": str(bundle_root.resolve()),
        "dataset_fingerprint": str(manifest["base_dataset_fingerprint"]),
        "bundle_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "joint_samples": sum(split_samples.values()),
        "episodes": sum(split_episodes.values()),
        "split_joint_samples": {
            name: int(split_samples[name]) for name in FORMAL_SPLITS
        },
        "split_episode_counts": {
            name: int(split_episodes[name]) for name in FORMAL_SPLITS
        },
        "scenario_episode_counts": dict(sorted(scenarios.items())),
        "behavior_episode_counts": dict(sorted(behaviors.items())),
    }


def build_composition_manifest(
    config: JointCollectionRunConfig,
    *,
    original_bundle: Path = DEFAULT_ORIGINAL_BUNDLE,
    output_path: Path | None = None,
    initial_gate_failure_waived: bool = False,
) -> dict[str, object]:
    report = write_targeted_supplement_report(
        config,
        initial_gate_failure_waived=initial_gate_failure_waived,
    )
    if not report["complete"]:
        raise TargetedSupplementAuditError(
            "targeted supplement is incomplete; composition was not generated"
        )
    if output_path is None:
        output_path = (
            DEFAULT_CANDIDATE_V4_COMPOSITION_PATH
            if config.scenario_contract_id == CANDIDATE_V4_CONTRACT_ID
            else DEFAULT_COMPOSITION_PATH
        )
    original_verification = verify_joint_risk_bundle(
        original_bundle, scenario_contract_id=CANDIDATE_V3_CONTRACT_ID
    )
    supplement_verification = verify_joint_risk_bundle(
        config.bundle_root, scenario_contract_id=config.scenario_contract_id
    )
    sources = [
        _source_statistics(original_bundle),
        _source_statistics(config.bundle_root),
    ]
    combined_samples = sum(int(source["joint_samples"]) for source in sources)
    combined_episodes = sum(int(source["episodes"]) for source in sources)
    scenario_counts: Counter[str] = Counter()
    behavior_counts: Counter[str] = Counter()
    combined_split_samples: Counter[str] = Counter()
    combined_split_episodes: Counter[str] = Counter()
    for source in sources:
        scenario_counts.update(source["scenario_episode_counts"])
        behavior_counts.update(source["behavior_episode_counts"])
        combined_split_samples.update(source["split_joint_samples"])
        combined_split_episodes.update(source["split_episode_counts"])
    if (
        combined_samples != 53_800
        or combined_episodes != 282
        or scenario_counts["S5_hard_brake_lead"] != 73
        or behavior_counts[
            "S5_hard_brake_lead/keep_emergency_braking"
        ]
        != 52
        or behavior_counts[
            "S5_hard_brake_lead/temporary_formation_release_and_recovery"
        ]
        != 21
    ):
        raise TargetedSupplementAuditError(
            "source statistics do not match the frozen 50k + S5-release20 contract"
        )
    fingerprint_payload = {
        "format": "logical-joint-risk-bundle-composition-v1",
        "sources": sources,
        "joint_samples": combined_samples,
        "episodes": combined_episodes,
        "split_joint_samples": {
            name: int(combined_split_samples[name]) for name in FORMAL_SPLITS
        },
        "split_episode_counts": {
            name: int(combined_split_episodes[name]) for name in FORMAL_SPLITS
        },
        "scenario_episode_counts": dict(sorted(scenario_counts.items())),
        "behavior_episode_counts": dict(sorted(behavior_counts.items())),
    }
    composite_fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        **fingerprint_payload,
        "composite_fingerprint": composite_fingerprint,
        "source_verification": [original_verification, supplement_verification],
        "targeted_supplement_audit": report,
        "training_eligibility": "not_automatically_granted",
    }
    _atomic_write_json(output_path, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument("--original-bundle", type=Path, default=DEFAULT_ORIGINAL_BUNDLE)
    parser.add_argument("--composition-output", type=Path)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--stop-reason")
    parser.add_argument(
        "--allow-failed-initial-gate-continue",
        action="store_true",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_run_config(args.config)
    if args.report_only:
        report = write_targeted_supplement_report(
            config,
            stop_reason=args.stop_reason,
            initial_gate_failure_waived=(
                args.allow_failed_initial_gate_continue
            ),
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["complete"] else 2
    manifest = build_composition_manifest(
        config,
        original_bundle=args.original_bundle,
        output_path=args.composition_output,
        initial_gate_failure_waived=args.allow_failed_initial_gate_continue,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
