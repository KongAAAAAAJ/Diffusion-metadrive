from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import pytest
import torch

from expert_dataset.collect_joint_bev import MAX_BACKGROUND_ACTORS
from models.bev_planner import (
    BEVOnlyDiffusionPlanner,
    BEVOnlyDiffusionPlannerConfig,
    BEVPlannerError,
    DDIMNoiseBundle,
    DEFAULT_DDIM_PATH,
    MetricTrajectoryBEVSampler,
)


def _joint_batch(batch_size: int = 1) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1234)
    bev = torch.randint(
        0,
        256,
        (batch_size, 3, 8, 256, 256),
        dtype=torch.uint8,
        generator=generator,
    )
    ego_state = torch.tensor(
        [
            [8.0, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0],
            [7.8, 0.1, 0.0, 0.1, 0.0, 0.1, 0.01, 1.2],
            [7.6, 0.0, 0.0, 0.1, 0.0, -0.1, -0.01, 1.4],
        ],
        dtype=torch.float32,
    ).unsqueeze(0).repeat(batch_size, 1, 1)
    relation = torch.tensor(
        [
            [-16.0, 0.0, 0.0, -1.0, -16.0, 0.0, -32.0, 0.0, 0.0, -2.0, -32.0, 0.0],
            [16.0, 0.0, 0.0, 1.0, 16.0, 0.0, -16.0, 0.0, 0.0, -1.0, -16.0, 0.0],
            [32.0, 0.0, 0.0, 2.0, 32.0, 0.0, 16.0, 0.0, 0.0, 1.0, 16.0, 0.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(0).repeat(batch_size, 1, 1)
    relation_valid = torch.ones(
        (batch_size, 3, 2),
        dtype=torch.bool,
    )
    roles = torch.arange(3, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)

    time = torch.arange(1, 9, dtype=torch.float32) * 0.5
    anchors = torch.zeros((batch_size, 3, 10, 8, 3), dtype=torch.float32)
    for role in range(3):
        for mode in range(10):
            anchors[:, role, mode, :, 0] = 7.0 * time + 0.1 * mode
            if 3 <= mode <= 5:
                anchors[:, role, mode, :, 1] = torch.linspace(0.0, 3.5, 8)
                anchors[:, role, mode, :, 2] = 0.08
            elif 6 <= mode <= 8:
                anchors[:, role, mode, :, 1] = torch.linspace(0.0, -3.5, 8)
                anchors[:, role, mode, :, 2] = -0.08
            elif mode == 9:
                anchors[:, role, mode, :, 0] = torch.linspace(2.0, 6.0, 8)
    mask = torch.ones((batch_size, 3, 10), dtype=torch.bool)
    return {
        "bev": bev,
        "ego_state": ego_state,
        "formation_relation_state": relation,
        "relation_valid_mask": relation_valid,
        "agent_role": roles,
        "coarse_trajectories": anchors,
        "mode_valid_mask": mask,
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


@pytest.fixture(scope="module")
def planner() -> BEVOnlyDiffusionPlanner:
    result = BEVOnlyDiffusionPlanner()
    result.eval()
    return result


def _call(
    planner: BEVOnlyDiffusionPlanner,
    batch: dict[str, torch.Tensor],
    **kwargs,
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
        **kwargs,
    )


def _zero_noise_bundle(shape: tuple[int, ...]) -> DDIMNoiseBundle:
    return DDIMNoiseBundle(
        initial_noise=torch.zeros(shape, dtype=torch.float32),
        transition_noises=tuple(
            torch.zeros(shape, dtype=torch.float32)
            for _ in range(DEFAULT_DDIM_PATH.stochastic_transition_count)
        ),
    )


def test_public_forward_shapes_mask_and_fixed_noise_determinism(
    planner: BEVOnlyDiffusionPlanner,
) -> None:
    batch = _joint_batch()
    batch["mode_valid_mask"][:, :, :9] = False
    noise = _zero_noise_bundle((1, 3, 10, 8, 2))
    with torch.no_grad():
        first = _call(planner, batch, ddim_noise_bundle=noise)
        second = _call(planner, batch, ddim_noise_bundle=noise)

    assert set(first) == {
        "trajectory_candidates",
        "mode_logits",
        "selected_mode",
        "selected_trajectory",
    }
    assert first["trajectory_candidates"].shape == (1, 3, 10, 8, 3)
    assert first["trajectory_candidates"].dtype == torch.float32
    assert torch.isfinite(first["trajectory_candidates"]).all()
    assert first["mode_logits"].shape == (1, 3, 10)
    assert torch.isneginf(first["mode_logits"][..., :9]).all()
    assert torch.isfinite(first["mode_logits"][..., 9]).all()
    assert torch.equal(first["selected_mode"], torch.full((1, 3), 9))
    assert first["selected_mode"].dtype == torch.int64
    assert first["selected_trajectory"].shape == (1, 3, 8, 3)
    for key in first:
        assert torch.equal(first[key], second[key])


def test_default_inference_seed_does_not_change_global_rng(
    planner: BEVOnlyDiffusionPlanner,
) -> None:
    batch = _joint_batch()
    rng_before = torch.random.get_rng_state().clone()
    with torch.no_grad():
        first = _call(planner, batch)
    rng_after = torch.random.get_rng_state().clone()
    with torch.no_grad():
        second = _call(planner, batch)

    assert torch.equal(rng_before, rng_after)
    for key in first:
        assert torch.equal(first[key], second[key])


def test_context_shapes_and_dynamic_anchor_dependency(
    planner: BEVOnlyDiffusionPlanner,
) -> None:
    batch = _joint_batch()
    with torch.no_grad():
        context = planner.encode_context(
            batch["bev"],
            batch["ego_state"],
            batch["formation_relation_state"],
            batch["relation_valid_mask"],
            batch["agent_role"],
            background_actor_state=batch["background_actor_state"],
            background_actor_valid_mask=batch["background_actor_valid_mask"],
            scenario_code=batch["scenario_code"],
            rule_formation_state=batch["rule_formation_state"],
            rule_action_condition=batch["rule_action_condition"],
        )
        normalized = planner._normalize_xy(batch["coarse_trajectories"][..., :2])
        timesteps = torch.zeros((1, 3), dtype=torch.int64)
        candidates, logits = planner.predict_denoised_candidates(
            normalized,
            timesteps,
            context,
            batch["coarse_trajectories"],
            batch["mode_valid_mask"],
        )
        changed_anchors = batch["coarse_trajectories"].clone()
        changed_anchors[..., 2] = 0.4
        changed_normalized = normalized.clone()
        changed_normalized[..., 0] += 0.05
        changed_candidates, _ = planner.predict_denoised_candidates(
            changed_normalized,
            timesteps,
            context,
            changed_anchors,
            batch["mode_valid_mask"],
        )

    assert context.bev_feature.shape == (1, 3, 128, 64, 64)
    assert context.role_tokens.shape == (1, 3, 128)
    assert candidates.shape == (1, 3, 10, 8, 3)
    assert logits.shape == (1, 3, 10)
    assert not torch.allclose(candidates, changed_candidates)


def test_relation_mask_and_joint_role_information(
    planner: BEVOnlyDiffusionPlanner,
) -> None:
    encoder = planner.context_encoder
    encoder.eval()
    batch = _joint_batch()
    batch["relation_valid_mask"][:, 0, 0] = False
    bev_feature = torch.randn(
        (1, 3, planner.config.d_model, 64, 64),
        generator=torch.Generator().manual_seed(9),
    )
    relation_changed = batch["formation_relation_state"].clone()
    relation_changed[:, 0, :6] = torch.tensor(
        [999.0, -999.0, 2.0, 500.0, 600.0, -700.0]
    )
    with torch.no_grad():
        base = encoder(
            batch["ego_state"],
            batch["formation_relation_state"],
            batch["relation_valid_mask"],
            batch["agent_role"],
            bev_feature,
            batch["background_actor_state"],
            batch["background_actor_valid_mask"],
            batch["scenario_code"],
            batch["rule_formation_state"],
            batch["rule_action_condition"],
        )
        masked_changed = encoder(
            batch["ego_state"],
            relation_changed,
            batch["relation_valid_mask"],
            batch["agent_role"],
            bev_feature,
            batch["background_actor_state"],
            batch["background_actor_valid_mask"],
            batch["scenario_code"],
            batch["rule_formation_state"],
            batch["rule_action_condition"],
        )
        leader_changed_state = batch["ego_state"].clone()
        leader_changed_state[:, 0, 0] += 12.0
        leader_changed = encoder(
            leader_changed_state,
            batch["formation_relation_state"],
            batch["relation_valid_mask"],
            batch["agent_role"],
            bev_feature,
            batch["background_actor_state"],
            batch["background_actor_valid_mask"],
            batch["scenario_code"],
            batch["rule_formation_state"],
            batch["rule_action_condition"],
        )

    torch.testing.assert_close(base, masked_changed)
    assert not torch.allclose(base[:, 2], leader_changed[:, 2])


def test_metric_sampler_matches_semantic_bev_axes() -> None:
    sampler = MetricTrajectoryBEVSampler()
    height = width = 65
    row = torch.arange(height, dtype=torch.float32).view(1, 1, height, 1)
    col = torch.arange(width, dtype=torch.float32).view(1, 1, 1, width)
    feature = row * 100.0 + col
    points = torch.zeros((1, 10, 8, 2), dtype=torch.float32)
    points[:, :, 0] = torch.tensor([52.0, 32.0])
    points[:, :, 1] = torch.tensor([-12.0, -32.0])
    points[:, :, 2] = torch.tensor([20.0, 0.0])
    sampled = sampler(feature, points)

    torch.testing.assert_close(sampled[:, :, 0, 0], torch.zeros((1, 10)))
    torch.testing.assert_close(
        sampled[:, :, 1, 0],
        torch.full((1, 10), (height - 1) * 100.0 + width - 1),
    )
    expected_center = ((52.0 - 20.0) / 64.0 * 64.0) * 100.0 + 32.0
    torch.testing.assert_close(
        sampled[:, :, 2, 0],
        torch.full((1, 10), expected_center),
        atol=1e-4,
        rtol=0.0,
    )


def test_diffusion_timestep_contract_is_preserved(
    planner: BEVOnlyDiffusionPlanner,
) -> None:
    assert planner.diffusion_scheduler.config.num_train_timesteps == 1000
    assert planner.diffusion_scheduler.config.beta_schedule == "scaled_linear"
    assert planner.diffusion_scheduler.config.prediction_type == "sample"
    assert planner.config.train_timestep_upper == 50
    assert DEFAULT_DDIM_PATH.initial_timestep == 8
    assert planner.inference_roll_timesteps() == (8, 5, 3, 0)
    assert DEFAULT_DDIM_PATH.previous_timesteps == (5, 3, 0, -1)
    assert DEFAULT_DDIM_PATH.eta == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field", "mutator", "match"),
    [
        (
            "bev",
            lambda value: value.float(),
            "bev must use dtype",
        ),
        (
            "ego_state",
            lambda value: value[:, :, :7],
            "ego_state must have shape",
        ),
        (
            "formation_relation_state",
            lambda value: value.clone().fill_(float("nan")),
            "formation_relation_state contains non-finite",
        ),
        (
            "agent_role",
            lambda value: value.flip(-1),
            "agent_role must be",
        ),
        (
            "coarse_trajectories",
            lambda value: value.double(),
            "coarse_trajectories must use dtype",
        ),
        (
            "mode_valid_mask",
            lambda value: value.float(),
            "mode_valid_mask must use dtype",
        ),
    ],
)
def test_strict_input_contract_rejects_invalid_values(
    planner: BEVOnlyDiffusionPlanner,
    field: str,
    mutator,
    match: str,
) -> None:
    batch = _joint_batch()
    batch[field] = mutator(batch[field])
    with pytest.raises(BEVPlannerError, match=match):
        _call(planner, batch)


def test_stop_must_always_be_valid(planner: BEVOnlyDiffusionPlanner) -> None:
    batch = _joint_batch()
    batch["mode_valid_mask"][0, 1, 9] = False
    with pytest.raises(BEVPlannerError, match="STOP must be valid"):
        _call(planner, batch)


def test_eval_rejects_training_timestep_override(
    planner: BEVOnlyDiffusionPlanner,
) -> None:
    batch = _joint_batch()
    with pytest.raises(BEVPlannerError, match="only valid during training"):
        _call(
            planner,
            batch,
            diffusion_timesteps=torch.zeros((1, 3), dtype=torch.int64),
        )


def test_training_rejects_timestep_outside_current_range(
    planner: BEVOnlyDiffusionPlanner,
) -> None:
    batch = _joint_batch()
    planner.train()
    try:
        with pytest.raises(BEVPlannerError, match=r"must be in \[0,50\)"):
            _call(
                planner,
                batch,
                diffusion_noise=torch.zeros((1, 3, 10, 8, 2)),
                diffusion_timesteps=torch.full((1, 3), 50, dtype=torch.int64),
            )
    finally:
        planner.eval()


def _has_finite_nonzero_gradient(parameter: torch.nn.Parameter) -> bool:
    return (
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and bool((parameter.grad != 0).any())
    )


def test_backward_reaches_all_stage_one_components(
    planner: BEVOnlyDiffusionPlanner,
) -> None:
    batch = _joint_batch()
    planner.train()
    planner.zero_grad(set_to_none=True)
    try:
        output = _call(
            planner,
            batch,
            diffusion_noise=torch.zeros((1, 3, 10, 8, 2)),
            diffusion_timesteps=torch.zeros((1, 3), dtype=torch.int64),
        )
        loss = (
            output["trajectory_candidates"].square().mean()
            + output["mode_logits"].mean()
        )
        loss.backward()
        parameters = {
            "backbone": planner.backbone.encoder.conv1.weight,
            "fusion": planner.bev_fusion.lateral[0].weight,
            "state": planner.context_encoder.ego_encoder[0].weight,
            "relation": planner.context_encoder.relation_encoder[0].weight,
            "decoder": planner.diffusion_decoder.blocks[
                0
            ].bev_attention.in_proj_weight,
            "trajectory_head": planner.diffusion_decoder.trajectory_head[-1].weight,
            "mode_head": planner.mode_head.weight,
        }
        missing = [
            name
            for name, parameter in parameters.items()
            if not _has_finite_nonzero_gradient(parameter)
        ]
        assert not missing, f"missing finite nonzero gradients: {missing}"
    finally:
        planner.zero_grad(set_to_none=True)
        planner.eval()


def test_heading_is_wrapped_to_pi(planner: BEVOnlyDiffusionPlanner) -> None:
    batch = _joint_batch()
    batch["coarse_trajectories"][..., 2] = math.pi
    noise = _zero_noise_bundle((1, 3, 10, 8, 2))
    with torch.no_grad():
        output = _call(planner, batch, ddim_noise_bundle=noise)
    heading = output["trajectory_candidates"][..., 2]
    assert bool((heading >= -math.pi).all())
    assert bool((heading <= math.pi).all())


@pytest.mark.parametrize(
    "missing",
    [
        "background_actor_state",
        "background_actor_valid_mask",
        "scenario_code",
        "rule_formation_state",
        "rule_action_condition",
    ],
)
def test_each_v2_condition_is_required(
    planner: BEVOnlyDiffusionPlanner, missing: str
) -> None:
    batch = _joint_batch()
    conditions = {
        name: batch[name]
        for name in (
            "background_actor_state",
            "background_actor_valid_mask",
            "scenario_code",
            "rule_formation_state",
            "rule_action_condition",
        )
    }
    conditions.pop(missing)
    with pytest.raises(BEVPlannerError, match="requires all explicit condition"):
        planner(
            batch["bev"],
            batch["ego_state"],
            batch["formation_relation_state"],
            batch["relation_valid_mask"],
            batch["agent_role"],
            batch["coarse_trajectories"],
            batch["mode_valid_mask"],
            **conditions,
        )


def test_planner_source_has_no_legacy_model_imports() -> None:
    source_path = Path(inspect.getfile(BEVOnlyDiffusionPlanner))
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    forbidden = {
        "models.diffusion.transfuser_model_v2",
        "models.diffusion.transfuser_backbone",
        "models.platoon_planner.platoon_diffusion_planner",
        "models.selector.intent_selector",
        "models.occupancy",
    }
    assert imported_modules.isdisjoint(forbidden)
    source_lower = source_path.read_text(encoding="utf-8").lower()
    assert "v2transfusermodel" not in source_lower
    assert "agenthead" not in source_lower


def test_config_rejects_inconsistent_architecture() -> None:
    with pytest.raises(BEVPlannerError, match="divisible"):
        BEVOnlyDiffusionPlannerConfig(d_model=130, num_heads=4)
    with pytest.raises(BEVPlannerError, match="train_timestep_upper"):
        BEVOnlyDiffusionPlannerConfig(train_timestep_upper=1001)
    with pytest.raises(BEVPlannerError, match="must be v2"):
        BEVOnlyDiffusionPlannerConfig(model_version="v1")  # type: ignore[arg-type]
