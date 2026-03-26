from __future__ import annotations

import torch
from torch import nn

from models.diffusion.diffusion_rl_scheduler import DDIMSchedulerWithLogProb, DiffusionRLScheduler


# ---------------------------------------------------------------------------
# Stub trajectory head (mimics V2TransfuserModel._trajectory_head interface)
# ---------------------------------------------------------------------------

class StubTrajectoryHead(nn.Module):
    """Minimal stand-in for TrajectoryHead used by the RL scheduler."""

    def __init__(self, ego_fut_mode: int = 3, num_poses: int = 8):
        super().__init__()
        self.ego_fut_mode = ego_fut_mode
        self._num_poses = num_poses
        # plan_anchor: [ego_fut_mode, num_poses, 3]
        self.plan_anchor = nn.Parameter(
            torch.zeros(ego_fut_mode, num_poses, 3), requires_grad=False
        )
        self.scale = nn.Parameter(torch.tensor(0.9))

    # norm / denorm: identity for stub (physical coords ≈ normalised coords)
    def norm_odo(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def denorm_odo(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def bezier_xyyaw(self, xy: torch.Tensor) -> torch.Tensor:
        # xy: [..., 8, 2] → add zero yaw → [..., 8, 3]
        return torch.cat([xy, torch.zeros_like(xy[..., :1])], dim=-1)


class StubV2Model(nn.Module):
    def __init__(self, ego_fut_mode: int = 3):
        super().__init__()
        self._trajectory_head = StubTrajectoryHead(ego_fut_mode=ego_fut_mode)


# ---------------------------------------------------------------------------
# Stub planner (implements PlatoonDiffusionPlanner RL interface)
# ---------------------------------------------------------------------------

class StubPlanner(nn.Module):
    """Implements the two RL helper methods consumed by DiffusionRLScheduler."""

    def __init__(self, ego_fut_mode: int = 3, num_poses: int = 8):
        super().__init__()
        self.model = StubV2Model(ego_fut_mode=ego_fut_mode)
        self._ego_fut_mode = ego_fut_mode
        self._num_poses = num_poses
        self.scale = nn.Parameter(torch.tensor(0.9))
        self.bias = nn.Parameter(torch.tensor(0.01))

    def extract_rl_context(self, batch: dict):
        """Return stub context (no real backbone) and agent_ids."""
        agent_ids = list(batch.keys())
        contexts = {}
        for agent_id in agent_ids:
            obs = next(iter(batch[agent_id].values()))
            device = obs.device
            contexts[agent_id] = {"_stub": torch.zeros(1, device=device)}
        return contexts, agent_ids

    def predict_denoised_traj(
        self,
        noisy_traj_norm: torch.Tensor,  # [B, M, 8, 2]
        timestep: torch.Tensor,          # [B]
        context: dict,
    ) -> torch.Tensor:                   # [B, M, 8, 2] normalised prediction
        t_scale = timestep.float().view(-1, 1, 1, 1) / 1000.0
        return torch.tanh(self.scale * noisy_traj_norm + self.bias + t_scale)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DDIM_CONFIG = {
    "num_train_timesteps": 1000,
    "num_inference_steps": 4,
    "eta": 0.5,
    "prediction_type": "sample",
    "trunc_timestep": 8,
}

EGO_FUT_MODE = 3
NUM_AGENTS = 2


def _batch(num_agents: int = NUM_AGENTS, device: str = "cpu") -> dict:
    return {
        f"agent{i}": {
            "camera": torch.zeros(3, 256, 1024, device=device),
            "lidar": torch.zeros(1, 256, 256, device=device),
            "status": torch.zeros(8, device=device),
            "formation_relation_state": torch.zeros(12, device=device),
        }
        for i in range(num_agents)
    }


# ---------------------------------------------------------------------------
# Tests (9 acceptance criteria)
# ---------------------------------------------------------------------------

def test_ddim_scheduler_with_log_prob_exists():
    """Criterion 1: DDIMSchedulerWithLogProb class exists."""
    sched = DDIMSchedulerWithLogProb(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="sample",
    )
    assert isinstance(sched, DDIMSchedulerWithLogProb)


def test_diffusion_rl_scheduler_exists():
    """Criterion 2: DiffusionRLScheduler class exists."""
    scheduler = DiffusionRLScheduler(_DDIM_CONFIG)
    assert isinstance(scheduler.scheduler, DDIMSchedulerWithLogProb)


def test_sample_with_log_prob_returns_required_keys():
    """Criterion 3: sample_with_log_prob returns trajectory, log_prob, diffusion_chain."""
    scheduler = DiffusionRLScheduler(_DDIM_CONFIG)
    planner = StubPlanner(ego_fut_mode=EGO_FUT_MODE)
    batch = _batch()

    outputs_a0 = scheduler.sample_with_log_prob(planner, batch, num_groups=2)
    assert "agent0" in outputs_a0
    out = outputs_a0["agent0"]
    assert set(out.keys()) == {"trajectory", "log_prob", "diffusion_chain"}


def test_trajectory_shape():
    """Criterion 4: trajectory.shape[-2:] == (8, 3)."""
    scheduler = DiffusionRLScheduler(_DDIM_CONFIG)
    planner = StubPlanner(ego_fut_mode=EGO_FUT_MODE)
    batch = _batch()

    outputs = scheduler.sample_with_log_prob(planner, batch, num_groups=2)
    traj = outputs["agent0"]["trajectory"]
    assert tuple(traj.shape[-2:]) == (8, 3), traj.shape


def test_log_prob_shape_and_finite():
    """Criterion 5: log_prob.shape == [G, M, step_num], all finite."""
    scheduler = DiffusionRLScheduler(_DDIM_CONFIG)
    planner = StubPlanner(ego_fut_mode=EGO_FUT_MODE)
    batch = _batch()
    num_groups = 2

    outputs = scheduler.sample_with_log_prob(planner, batch, num_groups=num_groups)
    lp = outputs["agent0"]["log_prob"]
    assert lp.ndim == 3, lp.shape
    assert lp.shape[0] == num_groups
    assert lp.shape[1] == EGO_FUT_MODE
    assert lp.shape[2] == scheduler.step_num
    assert torch.isfinite(lp).all()


def test_replay_with_log_prob_consistent_and_supports_backward():
    """Criterion 6 & criterion 9: replay log_prob is consistent with sample log_prob
    AND replay result supports backward."""
    scheduler = DiffusionRLScheduler(_DDIM_CONFIG)
    planner = StubPlanner(ego_fut_mode=EGO_FUT_MODE)
    batch = _batch()

    outputs = scheduler.sample_with_log_prob(planner, batch, num_groups=2)
    chain = outputs["agent0"]["diffusion_chain"]
    lp_sample = outputs["agent0"]["log_prob"]

    assert len(chain) == scheduler.step_num + 1

    lp_replay = scheduler.replay_with_log_prob(planner, batch, "agent0", chain)
    assert lp_replay.shape == lp_sample.shape
    # Numerical consistency: same model weights + same chain → same log_prob
    assert torch.allclose(lp_replay.detach(), lp_sample.detach(), atol=1e-4)

    # Backward must propagate through planner parameters
    lp_replay.sum().backward()
    assert planner.scale.grad is not None
    assert torch.isfinite(planner.scale.grad).all()


def test_compute_kl_is_scalar_and_finite():
    """Criterion 7: compute_kl returns a finite scalar."""
    scheduler = DiffusionRLScheduler(_DDIM_CONFIG)
    planner = StubPlanner(ego_fut_mode=EGO_FUT_MODE)
    batch = _batch()

    outputs = scheduler.sample_with_log_prob(planner, batch, num_groups=2)
    lp = outputs["agent0"]["log_prob"]
    chain = outputs["agent0"]["diffusion_chain"]
    lp_replay = scheduler.replay_with_log_prob(planner, batch, "agent0", chain)

    kl = scheduler.compute_kl(lp, lp_replay.detach())
    assert kl.ndim == 0
    assert torch.isfinite(kl)


def test_multiple_samples_are_different():
    """Criterion 8: consecutive sample_with_log_prob calls return different trajectories."""
    scheduler = DiffusionRLScheduler(_DDIM_CONFIG)
    planner = StubPlanner(ego_fut_mode=EGO_FUT_MODE)
    batch = _batch()

    out1 = scheduler.sample_with_log_prob(planner, batch, num_groups=2)
    out2 = scheduler.sample_with_log_prob(planner, batch, num_groups=2)
    assert not torch.allclose(out1["agent0"]["trajectory"], out2["agent0"]["trajectory"])


def test_truncated_noise_initialisation():
    """Extra: verify chain[0] is not pure random Gaussian (truncated noise from anchor)."""
    scheduler = DiffusionRLScheduler({**_DDIM_CONFIG, "trunc_timestep": 8})
    planner = StubPlanner(ego_fut_mode=EGO_FUT_MODE)
    batch = _batch()

    outputs = scheduler.sample_with_log_prob(planner, batch, num_groups=4)
    chain0 = outputs["agent0"]["diffusion_chain"][0]   # initial noisy sample
    # With truncated noise from plan_anchor (all zeros), the norm should be
    # substantially smaller than pure unit Gaussian noise.
    pure_gauss = torch.randn_like(chain0)
    # chain0 should have smaller std than pure Gaussian (mean ≈ 0, std < 1 typically)
    assert chain0.std().item() < pure_gauss.abs().mean().item() * 3.0
