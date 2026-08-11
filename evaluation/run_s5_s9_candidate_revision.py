"""Sequential five-seed S5--S9 candidate-v2 expert/video evaluation."""

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
        horizon=200,
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
    s5_categories = {}
    for profile in S5_PROFILES:
        actions = set()
        for row in episodes:
            if row["profile_id"] == profile:
                actions.update(
                    key.rsplit(":", 1)[-1]
                    for key, count in (row["rule_action_counts"] or {}).items()
                    if count
                )
        s5_categories[profile] = sorted(actions)
    profile_diversity = len({tuple(value) for value in s5_categories.values()}) >= 2
    hard_gates_passed = bool(
        base_ok and files_ok
        and s6_gaps == {"agent0-agent1", "agent1-agent2"}
        and s7_behaviors == {"pass_first", "yield_then_merge"}
        and profile_diversity
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
