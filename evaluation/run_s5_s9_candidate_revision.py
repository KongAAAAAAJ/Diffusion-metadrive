"""Sequential five-seed S5--S9 candidate-v3 expert/video evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2

from evaluation.preview_and_evaluation import run_scenario
from scenarios.bev_round13_contract import candidate_scenario_contract_v2


SEEDS = (17, 23, 31, 47, 59)
S5_PROFILES = ("brake_first", "balanced", "evasive")
SCENARIOS = (
    ("S6_background_merge_in", "R6_mainline_merge_approach"),
    ("S7_ego_merge_from_ramp", "R7_merge_core"),
    ("S8_ego_exit_to_ramp", "R6_exit_to_ramp"),
    ("S9_narrow_channel_negotiation", "R8_narrow_channel"),
)
SCENARIO_HORIZONS = {
    "S6_background_merge_in": 260,
    "S9_narrow_channel_negotiation": 800,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _video_decodable(path: Path) -> bool:
    capture = cv2.VideoCapture(str(path))
    try:
        ok, frame = capture.read()
        return bool(ok and frame is not None and frame.size > 0)
    finally:
        capture.release()


def _artifact_row(path: Path) -> dict[str, object]:
    exists = path.is_file()
    size = path.stat().st_size if exists else 0
    decodable = _video_decodable(path) if exists and path.suffix == ".mp4" else exists
    return {
        "absolute_path": str(path.resolve()),
        "exists": exists,
        "size_bytes": size,
        "decodable": decodable,
        "sha256": _sha256(path) if exists and size > 0 else None,
    }


def _s5_physical_behavior_gate(rows: list[dict[str, object]]) -> dict[str, object]:
    """Validate the latest S5 five-seed memory from physical evidence."""

    behavior_classes = {
        str((row.get("conflict_evidence") or {}).get("observed_behavior_class"))
        for row in rows
        if (row.get("conflict_evidence") or {}).get("observed_behavior_class")
    }
    lane_change_rows = [
        row
        for row in rows
        if bool((row.get("conflict_evidence") or {}).get("real_lane_change_completed"))
    ]
    asynchronous_rows = []
    synchronous_same_direction_rows = []
    for row in lane_change_rows:
        evidence = row.get("conflict_evidence") or {}
        directions = tuple(
            str(value)
            for value in (evidence.get("lane_change_direction_by_agent") or {}).values()
            if str(value) in {"left", "right"}
        )
        steps = tuple(
            int(value)
            for value in (evidence.get("lane_change_completion_steps") or {}).values()
        )
        if len(steps) >= 2 and len(set(steps)) >= 2:
            asynchronous_rows.append(row)
        if len(directions) == 3 and len(set(directions)) == 1:
            synchronous_same_direction_rows.append(row)

    all_keep = bool(rows) and all(
        (row.get("conflict_evidence") or {}).get("observed_behavior_class")
        == "keep_emergency_braking"
        for row in rows
    )
    all_synchronous_same_direction = bool(rows) and len(
        synchronous_same_direction_rows
    ) == len(rows)
    passed = bool(
        len(rows) == len(SEEDS)
        and len(behavior_classes) >= 2
        and lane_change_rows
        and asynchronous_rows
        and not all_keep
        and not all_synchronous_same_direction
    )
    return {
        "passed": passed,
        "episode_count": len(rows),
        "behavior_classes": sorted(behavior_classes),
        "physical_lane_change_seeds": sorted(
            int(row["seed"]) for row in lane_change_rows
        ),
        "asynchronous_response_seeds": sorted(
            int(row["seed"]) for row in asynchronous_rows
        ),
        "all_keep_lane": all_keep,
        "all_synchronous_same_direction_lane_change": (
            all_synchronous_same_direction
        ),
    }


def _s8_physical_behavior_gate(rows: list[dict[str, object]]) -> dict[str, object]:
    """Require five-seed asynchronous front/behind crossing diversity."""

    behavior_classes = {
        str((row.get("conflict_evidence") or {}).get("observed_lane_change_behavior_class"))
        for row in rows
        if (row.get("conflict_evidence") or {}).get(
            "observed_lane_change_behavior_class"
        )
    }
    straddled_rows = [
        row
        for row in rows
        if bool(
            (row.get("conflict_evidence") or {}).get(
                "constraint_straddled_by_lane_changes"
            )
        )
    ]
    asynchronous_rows = [
        row
        for row in rows
        if bool(
            (row.get("conflict_evidence") or {}).get(
                "non_simultaneous_right_lane_changes"
            )
        )
    ]
    relations = {
        str(value)
        for row in rows
        for value in (
            (row.get("conflict_evidence") or {}).get(
                "constraint_relation_at_lane_change_by_agent"
            )
            or {}
        ).values()
    }
    passed = bool(
        len(rows) == len(SEEDS)
        and len(straddled_rows) == len(SEEDS)
        and len(asynchronous_rows) == len(SEEDS)
        and relations == {"ahead", "behind"}
        and len(behavior_classes) >= 2
    )
    return {
        "passed": passed,
        "episode_count": len(rows),
        "behavior_classes": sorted(behavior_classes),
        "constraint_straddled_seeds": sorted(
            int(row["seed"]) for row in straddled_rows
        ),
        "asynchronous_right_lane_change_seeds": sorted(
            int(row["seed"]) for row in asynchronous_rows
        ),
        "realized_constraint_relations": sorted(relations),
    }


def _s7_parallel_constraint_gate(rows: list[dict[str, object]]) -> dict[str, object]:
    failures = []
    counts = []
    for row in rows:
        evidence = row.get("conflict_evidence") or {}
        declared = int(evidence.get("parallel_constraint_declared_count", 0))
        realized = int(evidence.get("parallel_constraint_realized_count", 0))
        counts.append(realized)
        passed = bool(
            1 <= declared == realized <= 3
            and evidence.get("parallel_constraint_initial_region_valid", False)
            and evidence.get("parallel_constraint_roles_present", False)
            and evidence.get("physical_split_observed", False)
            and evidence.get("non_simultaneous_mainline_entries", False)
            and evidence.get("formation_recovered_after_merge", False)
        )
        if not passed:
            failures.append(
                {
                    "seed": row.get("seed"),
                    "declared": declared,
                    "realized": realized,
                    "initial_region_valid": evidence.get(
                        "parallel_constraint_initial_region_valid", False
                    ),
                    "physical_split_observed": evidence.get(
                        "physical_split_observed", False
                    ),
                    "non_simultaneous_mainline_entries": evidence.get(
                        "non_simultaneous_mainline_entries", False
                    ),
                    "formation_recovered_after_merge": evidence.get(
                        "formation_recovered_after_merge", False
                    ),
                }
            )
    return {
        "passed": bool(len(rows) == len(SEEDS) and not failures),
        "realized_count_range": [min(counts), max(counts)] if counts else None,
        "failures": failures,
    }


def _s9_physical_behavior_gate(rows: list[dict[str, object]]) -> dict[str, object]:
    """Require the designated split, narrow traversal, return and recovery."""

    failures = []
    for row in rows:
        evidence = row.get("conflict_evidence") or {}
        route = row.get("route_completion") or {}
        relations = set(
            (evidence.get("split_actor_relation_at_left_completion_by_agent") or {}).values()
        )
        passed = bool(
            evidence.get("left_bypass_trigger_actor_declared_count") == 1
            and evidence.get("left_bypass_trigger_actor_realized_count") == 1
            and evidence.get("designated_split_actor_initial_region_valid", False)
            and evidence.get("s9_actor_roles_present", False)
            and evidence.get("non_simultaneous_left_lane_changes", False)
            and evidence.get("split_actor_straddled_by_left_completions", False)
            and relations == {"ahead", "behind"}
            and evidence.get("causal_split_actor_interaction_observed", False)
            and evidence.get("left_completion_clearance_satisfied", False)
            and route.get("all_agents_passed_blocker", False)
            and route.get("all_agents_traversed_narrow_section", False)
            and route.get("all_agents_returned_to_original_lane", False)
            and not evidence.get(
                "return_before_narrow_section_clear_observed", False
            )
            and evidence.get("formation_recovered_after_return", False)
        )
        if not passed:
            failures.append(
                {
                    "seed": row.get("seed"),
                    "declared_split_actors": evidence.get(
                        "left_bypass_trigger_actor_declared_count"
                    ),
                    "realized_split_actors": evidence.get(
                        "left_bypass_trigger_actor_realized_count"
                    ),
                    "relations": sorted(str(value) for value in relations),
                    "traversed_narrow_section": route.get(
                        "all_agents_traversed_narrow_section", False
                    ),
                    "returned_to_original_lane": route.get(
                        "all_agents_returned_to_original_lane", False
                    ),
                    "formation_recovered_after_return": evidence.get(
                        "formation_recovered_after_return", False
                    ),
                }
            )
    return {
        "passed": bool(len(rows) == len(SEEDS) and not failures),
        "failures": failures,
    }


def _incidental_background_gate(rows: list[dict[str, object]]) -> dict[str, object]:
    failures = []
    counts = []
    for row in rows:
        evidence = row.get("conflict_evidence") or {}
        declared = int(evidence.get("incidental_background_declared_count", 0))
        realized = int(evidence.get("incidental_background_realized_count", 0))
        counts.append(realized)
        if not 3 <= declared == realized <= 6:
            failures.append(
                {
                    "scenario_id": row.get("scenario_id"),
                    "profile_id": row.get("profile_id"),
                    "seed": row.get("seed"),
                    "declared": declared,
                    "realized": realized,
                }
            )
    return {
        "passed": bool(rows and not failures),
        "realized_count_range": [min(counts), max(counts)] if counts else None,
        "failures": failures,
    }


def _run_batch(root: Path, scenario_id: str, route: str, profile: str | None) -> Path:
    batch_root = root if profile is None else root / "S5_profiles" / profile
    run_scenario(
        scenario_id=scenario_id,
        local_route=route,
        num_agents=3,
        num_episodes=5,
        output_root=batch_root,
        heading_up=False,
        traffic_density=0.0,
        start_seed=SEEDS[0],
        fps=10,
        horizon=SCENARIO_HORIZONS.get(scenario_id, 200),
        seeds=SEEDS,
        evaluate=True,
        save_topdown_video=True,
        save_semantic_bev_video=True,
        save_trajectory_data=True,
        rule_maker_profile=profile,
    )
    return batch_root / scenario_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/s5_s9_candidate_revision_eval_20260811"),
    )
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    batches: list[tuple[str, str | None, Path]] = []
    for profile in S5_PROFILES:
        batches.append((
            "S5_hard_brake_lead",
            profile,
            _run_batch(root, "S5_hard_brake_lead", "R1_entry_straight", profile),
        ))
    for scenario_id, route in SCENARIOS:
        batches.append((scenario_id, None, _run_batch(root, scenario_id, route, None)))

    episodes = []
    artifacts = []
    for scenario_id, profile, batch in batches:
        for episode, seed in enumerate(SEEDS):
            expert_path = batch / "metrices" / f"episode_{episode:04d}" / "expert_episode.json"
            payload = json.loads(expert_path.read_text(encoding="utf-8"))
            summary = payload.get("scenario_summary", {})
            episode_row = {
                "scenario_id": scenario_id,
                "profile_id": profile,
                "episode": episode,
                "seed": seed,
                "failure_reason": payload.get("failure_reason"),
                "collision": bool(payload.get("collision")),
                "out_of_road": bool(payload.get("out_of_road")),
                "planning_success_rate": payload.get("native_joint_planning_success_rate"),
                "functional_success": bool(summary.get("functional_success")),
                "actor_manifest": summary.get("actor_manifest"),
                "conflict_evidence": summary.get("conflict_evidence"),
                "route_completion": summary.get("route_completion"),
                "rule_action_counts": payload.get("rule_action_counts"),
                "expert_episode_path": str(expert_path.resolve()),
            }
            episodes.append(episode_row)
            for path in (
                batch / "video" / f"episode_{episode:04d}.mp4",
                batch / "semantic_bev_video" / f"episode_{episode:04d}.mp4",
                batch / "trajectories" / f"episode_{episode:04d}.npz",
                expert_path,
            ):
                artifacts.append(_artifact_row(path))

    base_ok = all(
        not row["collision"]
        and not row["out_of_road"]
        and row["failure_reason"] is None
        and row["planning_success_rate"] == 1.0
        and row["functional_success"]
        for row in episodes
    )
    files_ok = all(
        row["exists"] and row["size_bytes"] > 0 and row["decodable"]
        for row in artifacts
    )
    s6_gaps = {
        row["conflict_evidence"].get("target_gap_id")
        for row in episodes if row["scenario_id"] == "S6_background_merge_in"
    }
    s7_behaviors = {
        row["conflict_evidence"].get("expected_behavior")
        for row in episodes if row["scenario_id"] == "S7_ego_merge_from_ramp"
    }
    s7_rows = [
        row for row in episodes if row["scenario_id"] == "S7_ego_merge_from_ramp"
    ]
    s7_parallel_constraint_gate = _s7_parallel_constraint_gate(s7_rows)
    s5_categories = {}
    s5_physical_behavior_gates = {}
    for profile in S5_PROFILES:
        actions = set()
        profile_rows = [
            row for row in episodes if row["profile_id"] == profile
        ]
        for row in profile_rows:
            actions.update(
                key.rsplit(":", 1)[-1]
                for key, count in (row["rule_action_counts"] or {}).items()
                if count
            )
        s5_categories[profile] = sorted(actions)
        s5_physical_behavior_gates[profile] = _s5_physical_behavior_gate(
            profile_rows
        )
    profile_diversity = len({tuple(value) for value in s5_categories.values()}) >= 2
    s5_memory_gate = any(
        bool(value["passed"])
        for value in s5_physical_behavior_gates.values()
    )
    s8_rows = [
        row for row in episodes if row["scenario_id"] == "S8_ego_exit_to_ramp"
    ]
    s8_physical_behavior_gate = _s8_physical_behavior_gate(s8_rows)
    s9_rows = [
        row
        for row in episodes
        if row["scenario_id"] == "S9_narrow_channel_negotiation"
    ]
    s9_physical_behavior_gate = _s9_physical_behavior_gate(s9_rows)
    incidental_background_gate = _incidental_background_gate(episodes)
    hard_gates_passed = bool(
        base_ok and files_ok
        and s6_gaps == {"agent0-agent1", "agent1-agent2"}
        and s7_behaviors == {"pass_first", "yield_then_merge"}
        and s7_parallel_constraint_gate["passed"]
        and profile_diversity
        and s5_memory_gate
        and s8_physical_behavior_gate["passed"]
        and s9_physical_behavior_gate["passed"]
        and incidental_background_gate["passed"]
    )
    contract = candidate_scenario_contract_v2(frozen=hard_gates_passed)
    manifest = {
        "format": "s5_s9_candidate_revision_artifact_manifest_v1",
        "hard_gates_passed": hard_gates_passed,
        "contract": contract,
        "fixed_seeds": list(SEEDS),
        "episode_count": len(episodes),
        "topdown_video_count": sum(a["absolute_path"].endswith(".mp4") and "/video/" in a["absolute_path"] for a in artifacts),
        "semantic_bev_video_count": sum(a["absolute_path"].endswith(".mp4") and "/semantic_bev_video/" in a["absolute_path"] for a in artifacts),
        "trajectory_npz_count": sum(a["absolute_path"].endswith(".npz") for a in artifacts),
        "s5_behavior_categories": s5_categories,
        "s5_physical_behavior_gates": s5_physical_behavior_gates,
        "s5_latest_memory_gate_passed": s5_memory_gate,
        "s7_parallel_constraint_gate": s7_parallel_constraint_gate,
        "s8_physical_behavior_gate": s8_physical_behavior_gate,
        "s9_physical_behavior_gate": s9_physical_behavior_gate,
        "incidental_background_gate": incidental_background_gate,
        "s6_target_gap_coverage": sorted(value for value in s6_gaps if value),
        "s7_behavior_coverage": sorted(
            value for value in s7_behaviors if value
        ),
        "episodes": episodes,
        "artifacts": artifacts,
    }
    (root / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not hard_gates_passed:
        raise SystemExit("candidate evaluation gates failed; v2 remains unfrozen")


if __name__ == "__main__":
    main()
