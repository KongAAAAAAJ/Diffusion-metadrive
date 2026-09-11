import pytest
import torch

from models.bev_planner import JointGRPOError, RiskPACTRolloutContext


def test_risk_pact_rollout_context_accepts_expected_actor_contract():
    state = torch.zeros((2, 3, 16, 8), dtype=torch.float32)
    valid = torch.ones((2, 3, 16), dtype=torch.bool)
    context = RiskPACTRolloutContext(
        background_actor_state=state,
        background_actor_valid_mask=valid,
    )
    assert context.batch_size == 2
    assert context.background_actor_state.requires_grad is False
    assert context.background_actor_valid_mask.requires_grad is False


def test_risk_pact_rollout_context_rejects_wrong_feature_width():
    state = torch.zeros((1, 3, 16, 7), dtype=torch.float32)
    valid = torch.ones((1, 3, 16), dtype=torch.bool)
    with pytest.raises(JointGRPOError, match=r"\[B,3,A,8\]"):
        RiskPACTRolloutContext(
            background_actor_state=state,
            background_actor_valid_mask=valid,
        )


def test_risk_pact_rollout_context_rejects_partial_or_wrong_mask_shape():
    state = torch.zeros((1, 3, 16, 8), dtype=torch.float32)
    valid = torch.ones((1, 3, 15), dtype=torch.bool)
    with pytest.raises(JointGRPOError, match=r"\[B,3,A\]"):
        RiskPACTRolloutContext(
            background_actor_state=state,
            background_actor_valid_mask=valid,
        )
