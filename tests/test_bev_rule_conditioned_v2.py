from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from expert_dataset.collect_joint_bev import (
    AgentRole,
    JOINT_SAMPLE_DTYPES,
    JOINT_SAMPLE_SHAPES,
    MAX_BACKGROUND_ACTORS,
    V2_JOINT_SAMPLE_DTYPES,
    V2_JOINT_SAMPLE_SHAPES,
    JointBEVSample,
    JointBEVSampleV2,
)
from expert_dataset.joint_bev_dataset import (
    JointBEVDataset,
    JointBEVDatasetConfig,
)
from expert_dataset.joint_bev_storage import (
    EpisodeSplitConfig,
    JointBEVDatasetStore,
    fingerprint_payload,
)
from expert_dataset.joint_risk_bundle_contract import (
    bundle_protocol_sha256,
    load_bundle_protocol,
)
from expert_dataset.riskentry_sidecar_storage import RiskEntrySidecarDatasetStore
from expert_dataset.verify_joint_bev_dataset import verify_joint_bev_dataset
from expert_dataset.verify_riskentry_sidecar import (
    validate_sidecar_dataset_contract,
)
from models.bev_planner.bev_only_diffusion_planner import (
    BEVOnlyDiffusionPlanner,
    BEVOnlyDiffusionPlannerConfig,
    BEVPlannerError,
    JointStateRelationEncoder,
)
from models.bev_planner.mode_contract import ModeIndex, mode_index_to_rule_action
from models.decisioner.rule_decisioner import (
    JointActionProposal,
    RuleMakerProposalBatch,
    diffusion_mode_feedback_actions,
    joint_proposal_actions,
    match_joint_action_proposal,
)


AGENT_IDS = ("agent0", "agent1", "agent2")


def _proposal(
    proposal_id: int, rank: int, actions: tuple[int, int, int]
) -> JointActionProposal:
    return JointActionProposal(
        proposal_id=proposal_id,
        rank=rank,
        rule_score=float(-rank),
        decisions={
            agent_id: {
                "action": int(actions[index]),
                "target_point": np.zeros((2,), dtype=np.float32),
            }
            for index, agent_id in enumerate(AGENT_IDS)
        },
    )


def _v2_sample() -> JointBEVSampleV2:
    values = {
        name: np.zeros(shape, dtype=JOINT_SAMPLE_DTYPES[name])
        for name, shape in JOINT_SAMPLE_SHAPES.items()
    }
    values["agent_role"] = np.asarray(list(AgentRole), dtype=np.int64)
    values["mode_valid_mask"][:, ModeIndex.STOP] = True
    values["gt_mode"][:] = int(ModeIndex.STOP)
    base = JointBEVSample(**values)
    actor_state = np.zeros(
        V2_JOINT_SAMPLE_SHAPES["background_actor_state"], dtype=np.float32
    )
    actor_state[:, 0, 0] = np.asarray([10.0, 20.0, 30.0], dtype=np.float32)
    actor_mask = np.zeros(
        V2_JOINT_SAMPLE_SHAPES["background_actor_valid_mask"], dtype=np.bool_
    )
    actor_mask[:, 0] = True
    return JointBEVSampleV2(
        base=base,
        background_actor_state=actor_state,
        background_actor_valid_mask=actor_mask,
        scenario_code=np.asarray(3, dtype=np.int64),
        rule_formation_state=np.asarray(1, dtype=np.int64),
        rule_action_condition=np.asarray((-1, 0, 1), dtype=np.int64),
    )


def test_mode_feedback_and_s7_exception_are_explicit() -> None:
    assert [int(mode_index_to_rule_action(index)) for index in range(10)] == [
        0,
        0,
        0,
        -1,
        -1,
        -1,
        1,
        1,
        1,
        0,
    ]
    physical, feedback, exceptions = diffusion_mode_feedback_actions(
        (3, 2, 9),
        AGENT_IDS,
        scenario_id="S7_ego_merge_from_ramp",
        local_route="R7_merge_core",
    )
    assert physical == {"agent0": -1, "agent1": 0, "agent2": 0}
    assert feedback == {"agent0": 0, "agent1": 0, "agent2": 0}
    assert exceptions == {"agent0": True, "agent1": False, "agent2": False}

    _, strict_feedback, strict_exceptions = diffusion_mode_feedback_actions(
        (3, 2, 9),
        AGENT_IDS,
        scenario_id="S6_background_merge_in",
        local_route="R3_mainline_straight",
    )
    assert strict_feedback["agent0"] == -1
    assert not any(strict_exceptions.values())


def test_exact_proposal_matching_supports_rank_zero_and_lower_rank() -> None:
    rank0 = _proposal(0, 0, (0, 0, 0))
    rank1 = _proposal(1, 1, (-1, 0, 1))
    batch = RuleMakerProposalBatch(batch_id=7, proposals=(rank0, rank1))

    assert match_joint_action_proposal(
        batch, joint_proposal_actions(rank0, AGENT_IDS), AGENT_IDS
    ) is rank0
    assert match_joint_action_proposal(
        batch, {"agent0": -1, "agent1": 0, "agent2": 1}, AGENT_IDS
    ) is rank1
    assert (
        match_joint_action_proposal(
            batch, {"agent0": 1, "agent1": 1, "agent2": 1}, AGENT_IDS
        )
        is None
    )


def test_v2_actor_and_rule_conditions_change_context_without_invalid_actor_nans() -> None:
    encoder = JointStateRelationEncoder(
        32, 4, 64, 0.0, model_version="v2"
    ).eval()
    ego = torch.zeros((1, 3, 8), dtype=torch.float32)
    relation = torch.zeros((1, 3, 12), dtype=torch.float32)
    relation_mask = torch.ones((1, 3, 2), dtype=torch.bool)
    roles = torch.arange(3, dtype=torch.int64).unsqueeze(0)
    bev_feature = torch.randn(
        (1, 3, 32, 8, 8), generator=torch.Generator().manual_seed(17)
    )
    actor_state = torch.zeros(
        (1, 3, MAX_BACKGROUND_ACTORS, 8), dtype=torch.float32
    )
    actor_mask = torch.zeros(
        (1, 3, MAX_BACKGROUND_ACTORS), dtype=torch.bool
    )
    scenario = torch.tensor([1], dtype=torch.int64)
    formation = torch.tensor([0], dtype=torch.int64)
    actions = torch.zeros((1, 3), dtype=torch.int64)

    with torch.no_grad():
        empty = encoder(
            ego,
            relation,
            relation_mask,
            roles,
            bev_feature,
            actor_state,
            actor_mask,
            scenario,
            formation,
            actions,
        )
        actor_state[:, :, 0, :2] = torch.tensor([12.0, -3.0])
        actor_mask[:, :, 0] = True
        actors_present = encoder(
            ego,
            relation,
            relation_mask,
            roles,
            bev_feature,
            actor_state,
            actor_mask,
            scenario,
            formation,
            actions,
        )
        changed_rule = encoder(
            ego,
            relation,
            relation_mask,
            roles,
            bev_feature,
            actor_state,
            actor_mask,
            torch.tensor([5], dtype=torch.int64),
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([[-1, 0, 1]], dtype=torch.int64),
        )

    assert torch.isfinite(empty).all()
    assert not torch.allclose(empty, actors_present)
    assert not torch.allclose(actors_present, changed_rule)


def test_v2_planner_requires_conditions_and_rule_action_does_not_override_hard_mask() -> None:
    planner = BEVOnlyDiffusionPlanner(
        BEVOnlyDiffusionPlannerConfig(model_version="v2")
    ).eval()
    bev = torch.zeros((1, 3, 8, 256, 256), dtype=torch.uint8)
    ego = torch.zeros((1, 3, 8), dtype=torch.float32)
    relation = torch.zeros((1, 3, 12), dtype=torch.float32)
    relation_mask = torch.ones((1, 3, 2), dtype=torch.bool)
    roles = torch.arange(3, dtype=torch.int64).unsqueeze(0)
    anchors = torch.zeros((1, 3, 10, 8, 3), dtype=torch.float32)
    hard_mask = torch.zeros((1, 3, 10), dtype=torch.bool)
    hard_mask[..., ModeIndex.STOP] = True
    conditions = {
        "background_actor_state": torch.zeros(
            (1, 3, MAX_BACKGROUND_ACTORS, 8), dtype=torch.float32
        ),
        "background_actor_valid_mask": torch.zeros(
            (1, 3, MAX_BACKGROUND_ACTORS), dtype=torch.bool
        ),
        "scenario_code": torch.tensor([3], dtype=torch.int64),
        "rule_formation_state": torch.tensor([1], dtype=torch.int64),
        "rule_action_condition": torch.full((1, 3), -1, dtype=torch.int64),
    }

    with pytest.raises(BEVPlannerError, match="requires all explicit condition"):
        planner(bev, ego, relation, relation_mask, roles, anchors, hard_mask)
    with torch.no_grad():
        output = planner(
            bev,
            ego,
            relation,
            relation_mask,
            roles,
            anchors,
            hard_mask,
            **conditions,
            diffusion_noise=torch.zeros((1, 3, 10, 8, 2)),
        )

    assert torch.equal(
        output["selected_mode"], torch.full((1, 3), int(ModeIndex.STOP))
    )
    assert torch.isneginf(output["mode_logits"][..., :9]).all()


def test_v2_storage_round_trip_preserves_v1_and_new_fields(tmp_path: Path) -> None:
    sample = _v2_sample()
    root = tmp_path / "dataset"
    with JointBEVDatasetStore(
        root,
        split_config=EpisodeSplitConfig(1.0, 0.0, 0.0, seed=17),
        dataset_fingerprint=fingerprint_payload({"test": "bev-v2"}),
        resume=False,
        planner_version="v2",
    ) as store:
        episode = store.commit_episode(
            0,
            [sample],
            {
                "scenario_id": "S7_ego_merge_from_ramp",
                "local_route": "R7_merge_core",
            },
        )

    dataset = JointBEVDataset(
        JointBEVDatasetConfig(root, episode.split, mmap_cache_episodes=1)
    )
    loaded = dataset[0]
    assert set(loaded) == set(V2_JOINT_SAMPLE_SHAPES)
    for name, expected in sample.as_dict().items():
        assert loaded[name].numpy().dtype == V2_JOINT_SAMPLE_DTYPES[name]
        np.testing.assert_array_equal(loaded[name].numpy(), expected)
    report = verify_joint_bev_dataset(root, splits=("train",))
    assert report["schema_version"] == 3
    assert report["planner_version"] == "v2"


def test_v2_bundle_protocol_and_sidecar_base_binding_are_validated(
    tmp_path: Path,
) -> None:
    protocol = load_bundle_protocol(planner_version="v2")
    assert protocol["components"]["base"]["schema_version"] == 3
    assert bundle_protocol_sha256(planner_version="v2") != bundle_protocol_sha256()
    base_contract = protocol["components"]["base"]
    sidecar_root = tmp_path / "sidecar"
    with RiskEntrySidecarDatasetStore(
        sidecar_root,
        base_dataset_fingerprint=fingerprint_payload({"test": "v2-sidecar"}),
        resume=False,
        base_format=str(base_contract["format"]),
        base_schema_version=int(base_contract["schema_version"]),
    ):
        pass
    validated = validate_sidecar_dataset_contract(sidecar_root)
    assert validated["base_schema_version"] == 3


def test_online_path_directly_reuses_rule_maker_without_decision_wrapper() -> None:
    root = Path(__file__).resolve().parents[1]
    evaluator_source = (
        root / "evaluation" / "bev_four_model_evaluator.py"
    ).read_text(encoding="utf-8")
    collector_source = (
        root / "expert_dataset" / "collect_joint_bev.py"
    ).read_text(encoding="utf-8")
    rule_source = (
        root / "models" / "decisioner" / "rule_decisioner.py"
    ).read_text(encoding="utf-8")

    assert "PlatoonNormalPlanner" not in evaluator_source
    assert "make_rule_maker" in evaluator_source
    assert ".accept_joint_action(" in evaluator_source
    assert ".accept_joint_action(" in collector_source
    class_names = {
        node.name
        for node in ast.walk(ast.parse(rule_source))
        if isinstance(node, ast.ClassDef)
    }
    assert "RuleMakerHintProvider" not in class_names
