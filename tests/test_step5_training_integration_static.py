"""Dependency-light Step-5 contract checks.

These checks do not import the full BEV planner stack (timm/diffusers are not
required in the artifact build environment).  Full rollout smoke testing is
intentionally deferred until the overlay is applied in the project env.
"""
from pathlib import Path
import ast
import importlib.util
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
JOINT = ROOT / "models" / "bev_planner" / "joint_grpo.py"
FEA = ROOT / "models" / "bev_planner" / "time_varying_feasibility.py"
RUNNER = ROOT / "train" / "bev_joint_grpo_online" / "runner.py"


def _load_feasibility():
    spec = importlib.util.spec_from_file_location("_step5_fea", FEA)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_step5_sources_parse():
    ast.parse(JOINT.read_text())
    ast.parse(RUNNER.read_text())


def test_post_guard_skips_autograd_teacher():
    text = JOINT.read_text()
    assert "include_risk_pact_teacher=False" in text
    assert "rollout.require_risk_pact_context()" in text
    assert "risk_pact_active_count" in text
    assert "auxiliary_signal" in text


def test_runner_uses_unified_safety_switch():
    text = RUNNER.read_text()
    assert "safety = training_config.safety_post_training" in text
    assert "safety_post_training_mode=safety.mode" in text
    assert "risk_pact_distill_weight=risk_pact.distill_weight" in text


def test_feasibility_loss_has_gradient():
    m = _load_feasibility()
    zigzag = torch.tensor(
        [[0.,0.],[1.,0.],[2.,.9],[3.,-.9],[4.,.9],[5.,-.9],[6.,.9],[7.,0.]],
        requires_grad=True,
    )
    out = m.steering_feasibility_loss(
        zigzag,
        steering_limit_deg=24.13,
        steering_rate_limit_deg_s=60.0,
        wheelbase_m=5.6,
        trajectory_dt_s=0.5,
        min_segment_length_m=0.2,
    )
    loss = out.steering_loss + out.steering_rate_loss
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert zigzag.grad is not None
    assert bool(torch.isfinite(zigzag.grad).all())
    assert float(torch.linalg.vector_norm(zigzag.grad)) > 0.0


def test_mode_group_flatten_identity():
    # [B,R,M,N,H,D] -> [B,R,M*N,H,D] must preserve M-major/N-minor identity.
    B,R,M,N,H,D = 1,3,10,4,8,3
    x = torch.arange(B*R*M*N*H*D).reshape(B,R,M,N,H,D)
    flat = x.reshape(B,R,M*N,H,D)
    for mode in range(M):
        for group in range(N):
            assert torch.equal(flat[:, :, mode*N + group], x[:, :, mode, group])
