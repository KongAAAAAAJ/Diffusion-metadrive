from __future__ import annotations

from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from expert_dataset.collect_joint_bev import (
    AgentRole,
    JOINT_SAMPLE_DTYPES,
    JOINT_SAMPLE_SHAPES,
    JointBEVSample,
    JointBEVSampleV2,
    MAX_BACKGROUND_ACTORS,
)
from expert_dataset.joint_bev_dataset import JointBEVDataset, JointBEVDatasetConfig
from expert_dataset.joint_bev_storage import (
    EpisodeSplitConfig,
    JointBEVDatasetStore,
    fingerprint_payload,
)
from models.bev_planner import (
    BEVOnlyDiffusionPlanner,
    BEVOnlyDiffusionPlannerConfig,
    BEVPlannerContext,
    BEVPlannerError,
    JointStage1Loss,
)
from models.bev_planner.mode_contract import ModeIndex
from train.train_bev_diffusion_stage1 import loss_from_batch, planner_forward_from_batch


def _planner_config(condition: str) -> BEVOnlyDiffusionPlannerConfig:
    return BEVOnlyDiffusionPlannerConfig(
        d_model=32,
        num_heads=4,
        ffn_dim=64,
        decoder_layers=1,
        predecessor_condition=condition,
    )


def _planner_batch(batch_size: int = 1) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(91)
    time = torch.arange(1, 9, dtype=torch.float32) * 0.5
    anchors = torch.zeros((batch_size, 3, 10, 8, 3), dtype=torch.float32)
    for role in range(3):
        for mode in range(10):
            anchors[:, role, mode, :, 0] = time * (6.0 + 0.2 * role) + 0.1 * mode
            anchors[:, role, mode, :, 1] = 0.05 * (mode - 4)
            anchors[:, role, mode, :, 2] = 0.01 * (mode - 4)
    return {
        "bev": torch.randint(
            0,
            256,
            (batch_size, 3, 8, 256, 256),
            dtype=torch.uint8,
            generator=generator,
        ),
        "ego_state": torch.zeros((batch_size, 3, 8), dtype=torch.float32),
        "formation_relation_state": torch.zeros(
            (batch_size, 3, 12), dtype=torch.float32
        ),
        "relation_valid_mask": torch.ones((batch_size, 3, 2), dtype=torch.bool),
        "agent_role": torch.arange(3, dtype=torch.int64)
        .unsqueeze(0)
        .repeat(batch_size, 1),
        "coarse_trajectories": anchors,
        "mode_valid_mask": torch.ones((batch_size, 3, 10), dtype=torch.bool),
        "background_actor_state": torch.zeros(
            (batch_size, 3, MAX_BACKGROUND_ACTORS, 8), dtype=torch.float32
        ),
        "background_actor_valid_mask": torch.zeros(
            (batch_size, 3, MAX_BACKGROUND_ACTORS), dtype=torch.bool
        ),
        "scenario_code": torch.ones((batch_size,), dtype=torch.int64),
        "rule_formation_state": torch.zeros((batch_size,), dtype=torch.int64),
        "rule_action_condition": torch.zeros(
            (batch_size, 3), dtype=torch.int64
        ),
    }


def _call(
    planner: BEVOnlyDiffusionPlanner,
    batch: dict[str, torch.Tensor],
    *,
    noise: torch.Tensor,
    timesteps: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    return planner(
        batch["bev"],
        batch["ego_state"],
        batch["formation_relation_state"],
        batch["relation_valid_mask"],
        batch["agent_role"],
        batch["coarse_trajectories"],
        batch["mode_valid_mask"],
        background_actor_state=batch["background_actor_state"],
        background_actor_valid_mask=batch["background_actor_valid_mask"],
        scenario_code=batch["scenario_code"],
        rule_formation_state=batch["rule_formation_state"],
        rule_action_condition=batch["rule_action_condition"],
        diffusion_noise=noise,
        diffusion_timesteps=timesteps,
    )


def _copy_shared_state(
    source: BEVOnlyDiffusionPlanner,
    target: BEVOnlyDiffusionPlanner,
) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    for name, value in source_state.items():
        if name in target_state:
            target_state[name] = value.detach().clone()
    target.load_state_dict(target_state, strict=True)


@pytest.fixture(scope="module")
def planners() -> tuple[BEVOnlyDiffusionPlanner, BEVOnlyDiffusionPlanner]:
    torch.manual_seed(7)
    planner_a = BEVOnlyDiffusionPlanner(_planner_config("none"))
    planner_b = BEVOnlyDiffusionPlanner(_planner_config("predicted_detached"))
    _copy_shared_state(planner_a, planner_b)
    return planner_a, planner_b


def _context(planner: BEVOnlyDiffusionPlanner) -> BEVPlannerContext:
    generator = torch.Generator().manual_seed(23)
    return BEVPlannerContext(
        bev_feature=torch.randn(
            (1, 3, planner.config.d_model, 64, 64),
            generator=generator,
        ),
        role_tokens=torch.randn(
            (1, 3, planner.config.d_model),
            generator=generator,
        ),
    )


def _direct_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(31)
    noisy = torch.randn(
        (1, 3, 10, 8, 2),
        dtype=torch.float32,
        generator=generator,
    ).clamp(-1.0, 1.0)
    coarse = torch.randn(
        (1, 3, 10, 8, 3),
        dtype=torch.float32,
        generator=generator,
    )
    timesteps = torch.full((1, 3), 8, dtype=torch.int64)
    return noisy, coarse, timesteps


def test_config_and_conditional_module_contract() -> None:
    with pytest.raises(BEVPlannerError, match="predecessor_condition"):
        _planner_config("teacher_forcing")
    planner_a = BEVOnlyDiffusionPlanner(_planner_config("none"))
    planner_b = BEVOnlyDiffusionPlanner(_planner_config("predicted_detached"))
    assert planner_a.diffusion_decoder.predecessor_action_encoder is None
    assert planner_a.diffusion_decoder.predecessor_residual_gate is None
    assert not any("predecessor_" in name for name, _ in planner_a.named_parameters())
    assert planner_b.diffusion_decoder.predecessor_action_encoder is not None
    torch.testing.assert_close(
        planner_b.diffusion_decoder.predecessor_residual_gate,
        torch.zeros(()),
    )


def test_zero_gate_matches_parallel_a_with_shared_weights(
    planners: tuple[BEVOnlyDiffusionPlanner, BEVOnlyDiffusionPlanner],
) -> None:
    planner_a, planner_b = planners
    batch = _planner_batch()
    noise = torch.randn(
        (1, 3, 10, 8, 2),
        generator=torch.Generator().manual_seed(41),
    )
    timesteps = torch.full((1, 3), 8, dtype=torch.int64)
    planner_a.train()
    planner_b.train()
    output_a = _call(planner_a, batch, noise=noise, timesteps=timesteps)
    output_b = _call(planner_b, batch, noise=noise, timesteps=timesteps)
    for name in output_a:
        torch.testing.assert_close(
            output_b[name],
            output_a[name],
            rtol=1e-4,
            atol=1e-5,
        )


def test_inference_decodes_every_step_in_role_order(
    planners: tuple[BEVOnlyDiffusionPlanner, BEVOnlyDiffusionPlanner],
) -> None:
    _, planner = planners
    batch = _planner_batch()
    noise = torch.zeros((1, 3, 10, 8, 2), dtype=torch.float32)
    order: list[int] = []
    original = planner.diffusion_decoder.forward_role

    def traced(*args, **kwargs):
        order.append(int(kwargs["role_index"]))
        return original(*args, **kwargs)

    planner.eval()
    with mock.patch.object(
        planner.diffusion_decoder,
        "forward_role",
        side_effect=traced,
    ):
        with torch.no_grad():
            _call(planner, batch, noise=noise)
    assert order == [0, 1, 2, 0, 1, 2]


def test_nonzero_gate_propagates_predictions_only_downstream() -> None:
    torch.manual_seed(47)
    planner = BEVOnlyDiffusionPlanner(
        _planner_config("predicted_detached")
    )
    batch = _planner_batch()
    noise = torch.randn(
        (1, 3, 10, 8, 2),
        generator=torch.Generator().manual_seed(51),
    )
    timesteps = torch.full((1, 3), 8, dtype=torch.int64)
    planner.train()
    with torch.no_grad():
        planner.diffusion_decoder.predecessor_residual_gate.fill_(0.7)
        torch.nn.init.normal_(
            planner.diffusion_decoder.trajectory_head[-1].weight,
            std=0.01,
        )
        baseline = _call(planner, batch, noise=noise, timesteps=timesteps)
        leader_changed_noise = noise.clone()
        leader_changed_noise[:, 0] += 0.5
        leader_changed = _call(
            planner,
            batch,
            noise=leader_changed_noise,
            timesteps=timesteps,
        )
        downstream_changed_noise = noise.clone()
        downstream_changed_noise[:, 1:] -= 0.5
        downstream_changed = _call(
            planner,
            batch,
            noise=downstream_changed_noise,
            timesteps=timesteps,
        )
    assert not torch.allclose(
        leader_changed["mode_logits"][:, 1:],
        baseline["mode_logits"][:, 1:],
    )
    assert not torch.allclose(
        leader_changed["trajectory_candidates"][:, 2],
        baseline["trajectory_candidates"][:, 2],
    )
    torch.testing.assert_close(
        downstream_changed["mode_logits"][:, 0],
        baseline["mode_logits"][:, 0],
    )
    torch.testing.assert_close(
        downstream_changed["trajectory_candidates"][:, 0],
        baseline["trajectory_candidates"][:, 0],
    )


def test_predecessor_normalization_is_bounded_wrapped_and_detached(
    planners: tuple[BEVOnlyDiffusionPlanner, BEVOnlyDiffusionPlanner],
) -> None:
    _, planner = planners
    trajectory = torch.zeros((1, 8, 3), dtype=torch.float32)
    trajectory[..., 0] = torch.linspace(-100.0, 100.0, 8)
    trajectory[..., 1] = torch.linspace(100.0, -100.0, 8)
    trajectory[..., 2] = 1.5 * torch.pi
    trajectory.requires_grad_(True)
    normalized = planner._normalize_predecessor_action(trajectory)
    assert normalized.shape == (1, 8, 3)
    assert normalized.requires_grad is False
    assert bool((normalized[..., :2].abs() <= 1.0).all())
    torch.testing.assert_close(
        normalized[..., 2],
        torch.full((1, 8), -0.5),
        atol=1e-6,
        rtol=0.0,
    )


def test_hard_mask_selects_only_valid_predecessor_and_stop_only_runs(
    planners: tuple[BEVOnlyDiffusionPlanner, BEVOnlyDiffusionPlanner],
) -> None:
    _, planner = planners
    candidates = torch.zeros((1, 10, 8, 3), dtype=torch.float32)
    candidates[:, :, :, 0] = torch.arange(10).reshape(1, 10, 1)
    logits = torch.arange(10, dtype=torch.float32).reshape(1, 10)
    mask = torch.zeros((1, 10), dtype=torch.bool)
    mask[:, int(ModeIndex.KEEP_LOW)] = True
    selected = planner._select_predecessor_trajectory(candidates, logits, mask)
    torch.testing.assert_close(
        selected[..., 0],
        torch.full((1, 8), float(ModeIndex.KEEP_LOW)),
    )

    batch = _planner_batch()
    batch["mode_valid_mask"][:] = False
    batch["mode_valid_mask"][..., int(ModeIndex.STOP)] = True
    noise = torch.zeros((1, 3, 10, 8, 2), dtype=torch.float32)
    planner.eval()
    with torch.no_grad():
        output = _call(planner, batch, noise=noise)
    assert torch.equal(
        output["selected_mode"],
        torch.full((1, 3), int(ModeIndex.STOP), dtype=torch.int64),
    )


def test_downstream_losses_do_not_backpropagate_to_predecessor_actions(
    planners: tuple[BEVOnlyDiffusionPlanner, BEVOnlyDiffusionPlanner],
) -> None:
    _, planner = planners
    planner.train()
    with torch.no_grad():
        planner.diffusion_decoder.predecessor_residual_gate.fill_(0.5)
        torch.nn.init.normal_(
            planner.diffusion_decoder.trajectory_head[-1].weight,
            std=0.01,
        )
    noisy, coarse_value, timesteps = _direct_inputs()
    coarse = coarse_value.requires_grad_(True)
    mask = torch.ones((1, 3, 10), dtype=torch.bool)
    candidates, logits = planner.predict_denoised_candidates(
        noisy,
        timesteps,
        _context(planner),
        coarse,
        mask,
    )
    middle_loss = candidates[:, 1].sum() + logits[:, 1].sum()
    middle_gradient = torch.autograd.grad(
        middle_loss,
        coarse,
        retain_graph=True,
    )[0]
    assert torch.count_nonzero(middle_gradient[:, 0]) == 0
    assert torch.count_nonzero(middle_gradient[:, 1]) > 0

    rear_loss = candidates[:, 2].sum() + logits[:, 2].sum()
    rear_gradient = torch.autograd.grad(rear_loss, coarse)[0]
    assert torch.count_nonzero(rear_gradient[:, :2]) == 0
    assert torch.count_nonzero(rear_gradient[:, 2]) > 0


def test_gate_and_action_encoder_gradient_phases(
    planners: tuple[BEVOnlyDiffusionPlanner, BEVOnlyDiffusionPlanner],
) -> None:
    _, planner = planners
    planner.train()
    noisy, coarse, timesteps = _direct_inputs()
    mask = torch.ones((1, 3, 10), dtype=torch.bool)
    gate = planner.diffusion_decoder.predecessor_residual_gate
    encoder = planner.diffusion_decoder.predecessor_action_encoder
    assert gate is not None and encoder is not None

    with torch.no_grad():
        gate.zero_()
    planner.zero_grad(set_to_none=True)
    _, logits = planner.predict_denoised_candidates(
        noisy,
        timesteps,
        _context(planner),
        coarse,
        mask,
    )
    logits[:, 1, 0].sum().backward()
    assert gate.grad is not None
    assert torch.isfinite(gate.grad)
    assert gate.grad.abs() > 0
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in encoder.parameters()
    )

    with torch.no_grad():
        gate.fill_(0.5)
    planner.zero_grad(set_to_none=True)
    _, logits = planner.predict_denoised_candidates(
        noisy,
        timesteps,
        _context(planner),
        coarse,
        mask,
    )
    logits[:, 1, 0].sum().backward()
    assert any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and torch.count_nonzero(parameter.grad) > 0
        for parameter in encoder.parameters()
    )


def _packed_sample() -> JointBEVSampleV2:
    values = {
        name: np.zeros(shape, dtype=JOINT_SAMPLE_DTYPES[name])
        for name, shape in JOINT_SAMPLE_SHAPES.items()
    }
    values["bev"][:, 0] = 255
    values["ego_state"][:, 0] = 6.0
    values["ego_pose_global"][:, 0] = np.asarray((20.0, 10.0, 0.0))
    values["relation_valid_mask"][:] = True
    values["agent_role"] = np.asarray(list(AgentRole), dtype=np.int64)
    times = np.arange(1, 9, dtype=np.float32) * np.float32(0.5)
    for mode in range(10):
        values["coarse_trajectories"][:, mode, :, 0] = times * np.float32(
            6.0
        ) + np.float32(mode * 0.05)
    values["mode_valid_mask"][:] = True
    values["gt_mode"][:] = int(ModeIndex.KEEP_MEDIUM)
    values["expert_trajectory"][:] = values["coarse_trajectories"][
        :, int(ModeIndex.KEEP_MEDIUM)
    ]
    return JointBEVSampleV2(
        base=JointBEVSample(**values),
        background_actor_state=np.zeros(
            (3, MAX_BACKGROUND_ACTORS, 8), dtype=np.float32
        ),
        background_actor_valid_mask=np.zeros(
            (3, MAX_BACKGROUND_ACTORS), dtype=np.bool_
        ),
        scenario_code=np.asarray(1, dtype=np.int64),
        rule_formation_state=np.asarray(0, dtype=np.int64),
        rule_action_condition=np.zeros((3,), dtype=np.int64),
    )


def _packed_batch(tmp_path: Path) -> dict[str, torch.Tensor]:
    root = tmp_path / "dataset"
    with JointBEVDatasetStore(
        root,
        split_config=EpisodeSplitConfig(1.0, 0.0, 0.0, seed=9),
        dataset_fingerprint=fingerprint_payload({"round": 9}),
        resume=False,
        planner_version="v2",
    ) as store:
        store.commit_episode(
            0,
            [_packed_sample()],
            {"scenario_id": "round9_test", "local_route": "straight"},
        )
    dataset = JointBEVDataset(JointBEVDatasetConfig(root, "train"))
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    dataset.close()
    return batch


def test_real_packed_batch_b_forward_loss_and_backward(tmp_path: Path) -> None:
    planner = BEVOnlyDiffusionPlanner(_planner_config("predicted_detached"))
    planner.train()
    with torch.no_grad():
        planner.diffusion_decoder.predecessor_residual_gate.fill_(0.25)
    batch = _packed_batch(tmp_path)
    noise = torch.randn(
        (1, 3, 10, 8, 2),
        generator=torch.Generator().manual_seed(71),
    )
    timesteps = torch.full((1, 3), 8, dtype=torch.int64)
    output = planner_forward_from_batch(
        planner,
        batch,
        diffusion_noise=noise,
        diffusion_timesteps=timesteps,
    )
    result = loss_from_batch(JointStage1Loss(), output, batch)
    result.total.backward()

    modules = {
        "backbone": planner.backbone,
        "fusion": planner.bev_fusion,
        "context": planner.context_encoder,
        "decoder": planner.diffusion_decoder,
        "trajectory_head": planner.diffusion_decoder.trajectory_head,
        "mode_head": planner.mode_head,
        "action_encoder": planner.diffusion_decoder.predecessor_action_encoder,
    }
    assert torch.isfinite(result.total)
    for name, module in modules.items():
        assert module is not None, name
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.grad is not None
        ]
        assert gradients, name
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)
    gate = planner.diffusion_decoder.predecessor_residual_gate
    assert gate is not None and gate.grad is not None
    assert torch.isfinite(gate.grad) and gate.grad.abs() > 0
