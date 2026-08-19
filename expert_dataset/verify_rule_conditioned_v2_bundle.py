"""Verify exact alignment between rule-conditioned v2 samples and sidecar events."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from expert_dataset.collect_joint_bev import SCENARIO_CODE_BY_ID
from expert_dataset.joint_bev_storage import (
    V2_STORAGE_FORMAT,
    V2_STORAGE_SCHEMA_VERSION,
)
from expert_dataset.verify_joint_bev_dataset import (
    JointBEVVerificationError,
    verify_joint_bev_dataset,
)
from expert_dataset.verify_joint_risk_bundle import (
    JointRiskBundleVerificationError,
    verify_joint_risk_bundle,
)
from expert_dataset.verify_riskentry_sidecar import (
    RiskEntrySidecarVerificationError,
    verify_riskentry_sidecar_dataset,
)
from models.bev_planner.mode_contract import mode_index_to_rule_action
from scenarios.bev_round13_contract import (
    CANDIDATE_V3_CONTRACT_ID,
    CANDIDATE_V4_CONTRACT_ID,
    FORMAL_V1_CONTRACT_ID,
    scenario_contract_for_id,
)


AGENT_IDS = ("agent0", "agent1", "agent2")
RULE_ACTIONS = frozenset((-1, 0, 1))
REQUIRED_CONDITION_DETAILS = frozenset(
    {
        "trajectory_source",
        "diffusion_input_actions",
        "normal_planner_accepted_actions",
        "normal_planner_accepted_proposal_rank",
        "rule_formation_state",
        "proposal_batch_id",
        "execution_id",
        "background_actor_state",
        "background_actor_valid_mask",
        "physical_mode_action",
        "rule_feedback_action",
        "s7_left_to_keep_exception",
    }
)


class RuleConditionedV2VerificationError(RuntimeError):
    """Raised when v2 sample conditions and RuleMaker audit events diverge."""


def _read_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuleConditionedV2VerificationError(f"invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuleConditionedV2VerificationError(
            f"JSON root must be an object: {path}"
        )
    return payload


def _strict_int(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuleConditionedV2VerificationError(
            f"{name} must be an integer >= {minimum}"
        )
    return int(value)


def _actions(value: object, *, name: str) -> tuple[int, int, int]:
    if not isinstance(value, Mapping) or tuple(value) != AGENT_IDS:
        raise RuleConditionedV2VerificationError(
            f"{name} must contain ordered agent0--agent2"
        )
    result = tuple(
        _strict_int(value[agent_id], name=f"{name}.{agent_id}", minimum=-1)
        for agent_id in AGENT_IDS
    )
    if any(action not in RULE_ACTIONS for action in result):
        raise RuleConditionedV2VerificationError(
            f"{name} contains an invalid RuleMaker action"
        )
    return result  # type: ignore[return-value]


def _exception_mask(value: object) -> tuple[bool, bool, bool]:
    if not isinstance(value, Mapping) or tuple(value) != AGENT_IDS:
        raise RuleConditionedV2VerificationError(
            "s7_left_to_keep_exception must contain ordered agent0--agent2"
        )
    if any(not isinstance(value[agent_id], bool) for agent_id in AGENT_IDS):
        raise RuleConditionedV2VerificationError(
            "s7_left_to_keep_exception values must be booleans"
        )
    return tuple(bool(value[agent_id]) for agent_id in AGENT_IDS)  # type: ignore[return-value]


def _episode_path(root: Path, split: str, episode_index: int) -> Path:
    return root / split / "episodes" / f"episode_{episode_index:08d}"


def _load_array(path: Path) -> np.ndarray:
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise RuleConditionedV2VerificationError(
            f"unable to mmap condition array: {path}"
        ) from exc


def _require_equal(
    observed: np.ndarray,
    expected: np.ndarray,
    *,
    name: str,
    context: str,
) -> None:
    if observed.shape != expected.shape or not np.array_equal(observed, expected):
        raise RuleConditionedV2VerificationError(
            f"{name} mismatch at {context}"
        )


def _condition_events(
    metadata: Mapping[str, object],
    *,
    episode_index: int,
) -> dict[int, Mapping[str, object]]:
    rows = metadata.get("events")
    if not isinstance(rows, list):
        raise RuleConditionedV2VerificationError(
            f"episode {episode_index} sidecar events are missing"
        )
    by_step: dict[int, Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or row.get("event_type") != "rule_maker_condition":
            continue
        step = _strict_int(
            row.get("step_index"),
            name=f"episode {episode_index} condition step",
            minimum=0,
        )
        if step in by_step:
            raise RuleConditionedV2VerificationError(
                f"duplicate rule_maker_condition event at episode {episode_index} "
                f"raw step {step}"
            )
        if row.get("actor_ids") != [] or row.get("terminal") is not False:
            raise RuleConditionedV2VerificationError(
                f"condition event identity is invalid at episode {episode_index} "
                f"raw step {step}"
            )
        by_step[step] = row
    return by_step


def _verify_event(
    event: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    sample_index: int,
    *,
    episode_index: int,
    raw_step: int,
    scenario_id: str,
    local_route: str,
) -> dict[str, object]:
    context = (
        f"episode={episode_index} sample={sample_index} raw_step={raw_step}"
    )
    details = event.get("details")
    if not isinstance(details, Mapping) or not REQUIRED_CONDITION_DETAILS.issubset(
        details
    ):
        missing = (
            sorted(REQUIRED_CONDITION_DETAILS - set(details))
            if isinstance(details, Mapping)
            else sorted(REQUIRED_CONDITION_DETAILS)
        )
        raise RuleConditionedV2VerificationError(
            f"condition event details are incomplete at {context}: missing={missing}"
        )

    expected_scenario_code = SCENARIO_CODE_BY_ID.get(scenario_id)
    observed_scenario_code = int(arrays["scenario_code"][sample_index])
    if expected_scenario_code is None or observed_scenario_code != expected_scenario_code:
        raise RuleConditionedV2VerificationError(
            f"scenario_code mismatch at {context}"
        )
    if "scenario_id" in details and details["scenario_id"] != scenario_id:
        raise RuleConditionedV2VerificationError(
            f"event scenario_id mismatch at {context}"
        )
    if "local_route" in details and details["local_route"] != local_route:
        raise RuleConditionedV2VerificationError(
            f"event local_route mismatch at {context}"
        )
    if "scenario_code" in details and _strict_int(
        details["scenario_code"], name="event scenario_code", minimum=1
    ) != expected_scenario_code:
        raise RuleConditionedV2VerificationError(
            f"event scenario_code mismatch at {context}"
        )

    input_actions = _actions(
        details["diffusion_input_actions"], name="diffusion_input_actions"
    )
    accepted_actions = _actions(
        details["normal_planner_accepted_actions"],
        name="normal_planner_accepted_actions",
    )
    physical_actions = _actions(
        details["physical_mode_action"], name="physical_mode_action"
    )
    feedback_actions = _actions(
        details["rule_feedback_action"], name="rule_feedback_action"
    )
    exceptions = _exception_mask(details["s7_left_to_keep_exception"])

    sample_actions = tuple(
        int(value) for value in arrays["rule_action_condition"][sample_index]
    )
    if sample_actions != input_actions:
        raise RuleConditionedV2VerificationError(
            f"rule_action_condition mismatch at {context}"
        )
    expected_physical = tuple(
        int(mode_index_to_rule_action(int(mode)))
        for mode in arrays["gt_mode"][sample_index]
    )
    if physical_actions != expected_physical:
        raise RuleConditionedV2VerificationError(
            f"physical_mode_action mismatch at {context}"
        )

    formation_name = details["rule_formation_state"]
    if formation_name not in {"LOCKED", "UNLOCKED"}:
        raise RuleConditionedV2VerificationError(
            f"rule_formation_state is invalid at {context}"
        )
    formation_value = 1 if formation_name == "LOCKED" else 0
    if int(arrays["rule_formation_state"][sample_index]) != formation_value:
        raise RuleConditionedV2VerificationError(
            f"rule_formation_state mismatch at {context}"
        )

    actor_state_raw = np.asarray(details["background_actor_state"])
    if not np.issubdtype(actor_state_raw.dtype, np.number):
        raise RuleConditionedV2VerificationError(
            f"background_actor_state is not numeric at {context}"
        )
    actor_state = actor_state_raw.astype(np.float32, copy=False)
    if not np.isfinite(actor_state).all():
        raise RuleConditionedV2VerificationError(
            f"background_actor_state is non-finite at {context}"
        )
    _require_equal(
        actor_state,
        np.asarray(arrays["background_actor_state"][sample_index]),
        name="background_actor_state",
        context=context,
    )
    actor_valid = np.asarray(details["background_actor_valid_mask"])
    if actor_valid.dtype != np.dtype(np.bool_):
        raise RuleConditionedV2VerificationError(
            f"background_actor_valid_mask is not boolean at {context}"
        )
    _require_equal(
        actor_valid,
        np.asarray(arrays["background_actor_valid_mask"][sample_index]),
        name="background_actor_valid_mask",
        context=context,
    )

    trajectory_source = details["trajectory_source"]
    if trajectory_source not in {"new_native_plan", "committed_roll"}:
        raise RuleConditionedV2VerificationError(
            f"trajectory_source is invalid at {context}"
        )
    proposal_batch_id = _strict_int(
        details["proposal_batch_id"], name="proposal_batch_id", minimum=0
    )
    accepted_rank = _strict_int(
        details["normal_planner_accepted_proposal_rank"],
        name="normal_planner_accepted_proposal_rank",
        minimum=0,
    )
    execution_id = _strict_int(
        details["execution_id"], name="execution_id", minimum=-1
    )
    if trajectory_source == "committed_roll" and execution_id < 0:
        raise RuleConditionedV2VerificationError(
            f"committed_roll lacks execution_id at {context}"
        )
    if trajectory_source == "new_native_plan":
        if feedback_actions != accepted_actions:
            raise RuleConditionedV2VerificationError(
                f"new plan feedback/accepted actions mismatch at {context}"
            )
        if accepted_rank == 0 and accepted_actions != input_actions:
            raise RuleConditionedV2VerificationError(
                f"rank-0 accepted/input actions mismatch at {context}"
            )
    elif feedback_actions != input_actions:
        raise RuleConditionedV2VerificationError(
            f"committed input/feedback actions mismatch at {context}"
        )

    for role, (physical, feedback, exception) in enumerate(
        zip(physical_actions, feedback_actions, exceptions)
    ):
        valid_exception = bool(
            scenario_id == "S7_ego_merge_from_ramp"
            and local_route == "R7_merge_core"
            and physical == -1
            and feedback == 0
        )
        if exception != valid_exception:
            raise RuleConditionedV2VerificationError(
                f"S7 exception flag mismatch for {AGENT_IDS[role]} at {context}"
            )
        if not exception and physical != feedback:
            raise RuleConditionedV2VerificationError(
                f"physical/feedback actions mismatch at {context}"
            )

    return {
        "scenario_id": scenario_id,
        "formation": formation_name,
        "input_actions": ",".join(str(value) for value in input_actions),
        "trajectory_source": trajectory_source,
        "accepted_rank": accepted_rank,
        "proposal_batch_id": proposal_batch_id,
        "execution_id": execution_id,
        "s7_exceptions": sum(exceptions),
    }


def verify_rule_conditioned_v2_bundle(
    bundle_root: Path | str,
    *,
    scenario_contract_id: str = CANDIDATE_V4_CONTRACT_ID,
) -> dict[str, object]:
    """Run component verification and exact base/sidecar condition alignment."""

    root = Path(bundle_root).expanduser()
    manifest = _read_object(root / "dataset_bundle_manifest.json")
    base_root = root / str(manifest.get("base_directory", ""))
    sidecar_root = root / str(manifest.get("sidecar_directory", ""))
    base_contract = _read_object(base_root / "dataset_contract.json")
    sidecar_contract = _read_object(sidecar_root / "dataset_contract.json")
    if (
        base_contract.get("planner_version") != "v2"
        or base_contract.get("schema_version") != V2_STORAGE_SCHEMA_VERSION
        or base_contract.get("format") != V2_STORAGE_FORMAT
    ):
        raise RuleConditionedV2VerificationError(
            "rule-conditioned verification requires planner_version v2/schema3"
        )
    if (
        sidecar_contract.get("base_schema_version") != V2_STORAGE_SCHEMA_VERSION
        or sidecar_contract.get("base_format") != V2_STORAGE_FORMAT
    ):
        raise RuleConditionedV2VerificationError(
            "sidecar is not bound to the v2/schema3 base contract"
        )

    expected_scenario_sha256 = str(
        scenario_contract_for_id(scenario_contract_id)["sha256"]
    )
    try:
        base_report = verify_joint_bev_dataset(
            base_root,
            expected_scenario_contract_sha256=expected_scenario_sha256,
        )
        sidecar_report = verify_riskentry_sidecar_dataset(sidecar_root)
        bundle_report = verify_joint_risk_bundle(
            root, scenario_contract_id=scenario_contract_id
        )
    except (
        JointBEVVerificationError,
        RiskEntrySidecarVerificationError,
        JointRiskBundleVerificationError,
    ) as exc:
        raise RuleConditionedV2VerificationError(str(exc)) from exc

    index_path = root / "bundle_episode_index.jsonl"
    try:
        rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuleConditionedV2VerificationError(
            "unable to read bundle episode index"
        ) from exc

    aligned_episodes = 0
    aligned_samples = 0
    formation_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    rank_counts: Counter[int] = Counter()
    scenario_counts: Counter[str] = Counter()
    s7_exceptions = 0
    proposal_batches: set[tuple[int, int]] = set()
    executions: set[tuple[int, int]] = set()

    for row in rows:
        if not isinstance(row, Mapping) or row.get("base_status") != "committed":
            continue
        episode_index = _strict_int(
            row.get("episode_index"), name="bundle episode_index", minimum=0
        )
        split = str(row.get("split", ""))
        base_path = _episode_path(base_root, split, episode_index)
        sidecar_path = _episode_path(sidecar_root, split, episode_index)
        base_metadata = _read_object(base_path / "episode.json")
        sidecar_metadata = _read_object(sidecar_path / "episode.json")
        attributes = base_metadata.get("attributes")
        if not isinstance(attributes, Mapping):
            raise RuleConditionedV2VerificationError(
                f"base episode {episode_index} attributes are missing"
            )
        scenario_id = str(attributes.get("scenario_id", ""))
        local_route = str(attributes.get("local_route", ""))
        if (
            sidecar_metadata.get("scenario_id") != scenario_id
            or sidecar_metadata.get("local_route") != local_route
        ):
            raise RuleConditionedV2VerificationError(
                f"episode {episode_index} scenario context mismatch"
            )

        arrays = {
            name: _load_array(base_path / f"{name}.npy")
            for name in (
                "background_actor_state",
                "background_actor_valid_mask",
                "scenario_code",
                "rule_formation_state",
                "rule_action_condition",
                "gt_mode",
            )
        }
        mapping = _load_array(sidecar_path / "base_sample_step_index.npy")
        events = _condition_events(sidecar_metadata, episode_index=episode_index)
        expected_steps = tuple(int(value) for value in mapping)
        if set(events) != set(expected_steps) or len(events) != len(expected_steps):
            missing = sorted(set(expected_steps) - set(events))
            unexpected = sorted(set(events) - set(expected_steps))
            raise RuleConditionedV2VerificationError(
                f"condition/base step mapping mismatch for episode {episode_index}: "
                f"missing={missing}, unexpected={unexpected}"
            )
        if any(len(array) != len(mapping) for array in arrays.values()):
            raise RuleConditionedV2VerificationError(
                f"condition array length mismatch for episode {episode_index}"
            )

        for sample_index, raw_step in enumerate(expected_steps):
            summary = _verify_event(
                events[raw_step],
                arrays,
                sample_index,
                episode_index=episode_index,
                raw_step=raw_step,
                scenario_id=scenario_id,
                local_route=local_route,
            )
            formation_counts[str(summary["formation"])] += 1
            action_counts[str(summary["input_actions"])] += 1
            source_counts[str(summary["trajectory_source"])] += 1
            rank_counts[int(summary["accepted_rank"])] += 1
            scenario_counts[str(summary["scenario_id"])] += 1
            s7_exceptions += int(summary["s7_exceptions"])
            proposal_batches.add(
                (episode_index, int(summary["proposal_batch_id"]))
            )
            execution_id = int(summary["execution_id"])
            if execution_id >= 0:
                executions.add((episode_index, execution_id))
            aligned_samples += 1
        aligned_episodes += 1

    if aligned_samples != int(base_report["joint_samples"]):
        raise RuleConditionedV2VerificationError(
            "aligned sample count differs from verified base sample count"
        )
    if aligned_episodes != int(bundle_report["committed_base_episodes"]):
        raise RuleConditionedV2VerificationError(
            "aligned episode count differs from verified bundle episode count"
        )
    if aligned_samples != int(sidecar_report["base_samples"]):
        raise RuleConditionedV2VerificationError(
            "aligned sample count differs from verified sidecar base mapping"
        )

    return {
        "status": "pass",
        "bundle_root": str(root.resolve()),
        "planner_version": "v2",
        "base_schema_version": V2_STORAGE_SCHEMA_VERSION,
        "scenario_contract_id": scenario_contract_id,
        "scenario_contract_sha256": expected_scenario_sha256,
        "aligned_episodes": aligned_episodes,
        "aligned_samples": aligned_samples,
        "proposal_batches": len(proposal_batches),
        "executions": len(executions),
        "s7_left_to_keep_exceptions": s7_exceptions,
        "formation_state_counts": dict(sorted(formation_counts.items())),
        "input_action_counts": dict(sorted(action_counts.items())),
        "trajectory_source_counts": dict(sorted(source_counts.items())),
        "accepted_proposal_rank_counts": {
            str(key): value for key, value in sorted(rank_counts.items())
        },
        "scenario_sample_counts": dict(sorted(scenario_counts.items())),
        "bundle_sha256": bundle_report["bundle_sha256"],
        "base_dataset_fingerprint": bundle_report["base_dataset_fingerprint"],
        "sidecar_dataset_fingerprint": bundle_report[
            "sidecar_dataset_fingerprint"
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument(
        "--scenario-contract",
        choices=(
            FORMAL_V1_CONTRACT_ID,
            CANDIDATE_V3_CONTRACT_ID,
            CANDIDATE_V4_CONTRACT_ID,
        ),
        default=CANDIDATE_V4_CONTRACT_ID,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = verify_rule_conditioned_v2_bundle(
        args.bundle_root, scenario_contract_id=args.scenario_contract
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RuleConditionedV2VerificationError",
    "main",
    "verify_rule_conditioned_v2_bundle",
]
