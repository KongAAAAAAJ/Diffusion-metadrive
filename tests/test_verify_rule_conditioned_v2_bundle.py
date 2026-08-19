from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from expert_dataset.collect_joint_bev import (
    AgentRole,
    JOINT_SAMPLE_DTYPES,
    JOINT_SAMPLE_SHAPES,
    V2_JOINT_SAMPLE_SHAPES,
    JointBEVSample,
    JointBEVSampleV2,
)
from expert_dataset.joint_bev_storage import (
    EpisodeSplitConfig,
    JointBEVDatasetStore,
    fingerprint_payload,
    joint_sample_storage_contract,
)
from expert_dataset.joint_risk_bundle_storage import (
    BundleEpisodeAttempt,
    BundleEpisodeResult,
    JointRiskBundleIndex,
)
from expert_dataset.riskentry_sidecar_adapter import (
    SidecarActorRecord,
    SidecarActorSnapshot,
    SidecarLaneRecord,
    SidecarRawEvent,
)
from expert_dataset.riskentry_sidecar_storage import (
    RiskEntrySidecarDatasetStore,
    SidecarEpisodeStart,
)
from expert_dataset.verify_rule_conditioned_v2_bundle import (
    RuleConditionedV2VerificationError,
    main,
    verify_rule_conditioned_v2_bundle,
)
from models.bev_planner.mode_contract import ModeIndex
from scenarios.bev_round13_contract import candidate_scenario_contract_v4


AGENT_IDS = ("agent0", "agent1", "agent2")
SCENARIO_ID = "S6_background_merge_in"
LOCAL_ROUTE = "R6_mainline_merge_approach"


def _sample() -> JointBEVSampleV2:
    values = {
        name: np.zeros(shape, dtype=JOINT_SAMPLE_DTYPES[name])
        for name, shape in JOINT_SAMPLE_SHAPES.items()
    }
    values["agent_role"] = np.asarray(list(AgentRole), dtype=np.int64)
    values["mode_valid_mask"][:, ModeIndex.STOP] = True
    values["gt_mode"][:] = int(ModeIndex.STOP)
    base = JointBEVSample(**values)
    return JointBEVSampleV2(
        base=base,
        background_actor_state=np.zeros(
            V2_JOINT_SAMPLE_SHAPES["background_actor_state"], dtype=np.float32
        ),
        background_actor_valid_mask=np.zeros(
            V2_JOINT_SAMPLE_SHAPES["background_actor_valid_mask"], dtype=np.bool_
        ),
        scenario_code=np.asarray(2, dtype=np.int64),
        rule_formation_state=np.asarray(1, dtype=np.int64),
        rule_action_condition=np.asarray((-1, 0, 1), dtype=np.int64),
    )


ACTORS = (
    SidecarActorRecord(0, "P0", "agent0", "platoon", "leader", 0, 5.74, 2.3),
    SidecarActorRecord(1, "P1", "agent1", "platoon", "middle", 0, 5.74, 2.3),
    SidecarActorRecord(2, "P2", "agent2", "platoon", "rear", 0, 5.74, 2.3),
)
LANES = (SidecarLaneRecord(0, "L000", "fixture-lane", "mainline"),)


def _snapshot(record: SidecarActorRecord, step: int) -> SidecarActorSnapshot:
    x = 20.0 - 8.0 * record.actor_index + 0.4 * step
    return SidecarActorSnapshot(
        actor_id=record.actor_id,
        source_object_id=record.source_object_id,
        actor_type=record.actor_type,
        world_x_m=x,
        world_y_m=0.0,
        heading_rad=0.0,
        velocity_x_mps=4.0,
        velocity_y_mps=0.0,
        length_m=record.length_m,
        width_m=record.width_m,
        acceleration_valid=step > 0,
        yaw_rate_valid=step > 0,
        lane_id="L000",
        lane_s_m=x,
        lane_lateral_m=0.0,
        lane_heading_error_rad=0.0,
        lane_width_m=3.5,
        lane_valid=True,
    )


def _details(sample: JointBEVSampleV2) -> dict[str, object]:
    zeros = {agent_id: 0 for agent_id in AGENT_IDS}
    return {
        "trajectory_source": "new_native_plan",
        "diffusion_input_actions": {
            agent_id: action
            for agent_id, action in zip(AGENT_IDS, (-1, 0, 1))
        },
        "normal_planner_accepted_actions": dict(zeros),
        "normal_planner_accepted_proposal_rank": 1,
        "rule_formation_state": "LOCKED",
        "proposal_batch_id": 4,
        "execution_id": -1,
        "background_actor_state": sample.background_actor_state.tolist(),
        "background_actor_valid_mask": (
            sample.background_actor_valid_mask.tolist()
        ),
        "physical_mode_action": dict(zeros),
        "rule_feedback_action": dict(zeros),
        "s7_left_to_keep_exception": {
            agent_id: False for agent_id in AGENT_IDS
        },
    }


def _build_bundle(root: Path) -> Path:
    bundle_root = root / "bundle"
    base_root = bundle_root / "platoon_joint_bev"
    sidecar_root = bundle_root / "riskentry_actor_sidecar"
    fingerprint = fingerprint_payload({"test": "rule-conditioned-v2-verifier"})
    scenario_hash = str(candidate_scenario_contract_v4()["sha256"])
    split_config = EpisodeSplitConfig(1.0, 0.0, 0.0, seed=17)
    storage_contract = joint_sample_storage_contract("v2")
    sample = _sample()

    with JointBEVDatasetStore(
        base_root,
        split_config=split_config,
        dataset_fingerprint=fingerprint,
        resume=False,
        planner_version="v2",
    ) as base_store, RiskEntrySidecarDatasetStore(
        sidecar_root,
        base_dataset_fingerprint=fingerprint,
        resume=False,
        base_format=storage_contract.storage_format,
        base_schema_version=storage_contract.schema_version,
    ) as sidecar_store, JointRiskBundleIndex(
        bundle_root,
        base_directory=base_root.name,
        sidecar_directory=sidecar_root.name,
        base_dataset_fingerprint=fingerprint,
        sidecar_dataset_fingerprint=sidecar_store.dataset_fingerprint,
        scenario_contract_sha256=scenario_hash,
        split_seed=17,
        resume=False,
        planner_version="v2",
    ) as bundle:
        bundle.begin_attempt(
            BundleEpisodeAttempt(0, "train", SCENARIO_ID, LOCAL_ROUTE, 17)
        )
        base_store.commit_episode(
            0,
            [sample],
            {
                "scenario_id": SCENARIO_ID,
                "local_route": LOCAL_ROUTE,
                "spawn_seed": 17,
                "selected_sample_steps": [1],
                "decision_dt_s": 0.1,
                "raw_timeline_length": 3,
                "sidecar_dataset_fingerprint": sidecar_store.dataset_fingerprint,
                "scenario_contract_sha256": scenario_hash,
            },
        )
        sidecar_store.begin_episode(
            SidecarEpisodeStart(
                episode_index=0,
                split="train",
                scenario_id=SCENARIO_ID,
                local_route=LOCAL_ROUTE,
                spawn_seed=17,
                decision_dt_s=0.1,
                base_dataset_fingerprint=fingerprint,
                scenario_parameters={
                    "scenario_contract_sha256": scenario_hash,
                },
            )
        )
        sidecar_store.update_registry(
            actor_records=ACTORS,
            lane_records=LANES,
            key_actor_ids={},
        )
        for step in range(3):
            sidecar_store.append_frame(
                step_index=step,
                timestamp_s=step * 0.1,
                actors=tuple(_snapshot(record, step) for record in ACTORS),
            )
            if step == 1:
                sidecar_store.append_event(
                    SidecarRawEvent(
                        "rule_maker_condition",
                        1,
                        0.1,
                        details=_details(sample),
                    )
                )
        sidecar_store.commit_episode(base_sample_step_indices=(1,))
        bundle.finalize(
            BundleEpisodeResult(
                episode_index=0,
                split="train",
                scenario_id=SCENARIO_ID,
                local_route=LOCAL_ROUTE,
                spawn_seed=17,
                base_status="committed",
                base_rejection_reason=None,
                sidecar_status="committed",
                sidecar_rejection_reason=None,
                raw_steps=3,
                base_samples=1,
                outcome="success",
            )
        )
    return bundle_root


def _sidecar_metadata(bundle_root: Path) -> Path:
    return (
        bundle_root
        / "riskentry_actor_sidecar/train/episodes/episode_00000000/episode.json"
    )


def _mutate_event(bundle_root: Path, mutation) -> None:
    path = _sidecar_metadata(bundle_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload["events"])
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_verifier_aligns_schema3_conditions_and_cli_prints_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle_root = _build_bundle(tmp_path)
    report = verify_rule_conditioned_v2_bundle(bundle_root)
    assert report["status"] == "pass"
    assert report["planner_version"] == "v2"
    assert report["base_schema_version"] == 3
    assert report["aligned_episodes"] == 1
    assert report["aligned_samples"] == 1
    assert report["input_action_counts"] == {"-1,0,1": 1}
    assert report["accepted_proposal_rank_counts"] == {"1": 1}

    assert main(["--bundle-root", str(bundle_root)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["aligned_samples"] == 1
    assert printed["status"] == "pass"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda events: events.clear(), "condition/base step mapping mismatch"),
        (lambda events: events.append(dict(events[0])), "duplicate rule_maker_condition"),
        (
            lambda events: events[0].update(
                {"step_index": 2, "timestamp_s": 0.2}
            ),
            "condition/base step mapping mismatch",
        ),
        (
            lambda events: events[0]["details"]["background_actor_state"][0][0].__setitem__(
                0, 1.0
            ),
            "background_actor_state mismatch",
        ),
        (
            lambda events: events[0]["details"]["diffusion_input_actions"].__setitem__(
                "agent0", 0
            ),
            "rule_action_condition mismatch",
        ),
        (
            lambda events: events[0]["details"].__setitem__(
                "normal_planner_accepted_proposal_rank", -1
            ),
            "normal_planner_accepted_proposal_rank must be an integer",
        ),
        (
            lambda events: events[0]["details"]["s7_left_to_keep_exception"].__setitem__(
                "agent0", True
            ),
            "S7 exception flag mismatch",
        ),
    ],
)
def test_verifier_rejects_missing_duplicate_misaligned_or_inconsistent_events(
    tmp_path: Path, mutation, match: str
) -> None:
    bundle_root = _build_bundle(tmp_path)
    _mutate_event(bundle_root, mutation)
    with pytest.raises(RuleConditionedV2VerificationError, match=match):
        verify_rule_conditioned_v2_bundle(bundle_root)


def test_verifier_rejects_non_v2_contract_before_component_scan(
    tmp_path: Path,
) -> None:
    bundle_root = _build_bundle(tmp_path)
    path = bundle_root / "platoon_joint_bev/dataset_contract.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("planner_version")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with pytest.raises(
        RuleConditionedV2VerificationError,
        match="requires planner_version v2/schema3",
    ):
        verify_rule_conditioned_v2_bundle(bundle_root)
