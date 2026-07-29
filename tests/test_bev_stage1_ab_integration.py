from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from models.bev_planner import (
    BEVOnlyDiffusionPlanner,
    BEVOnlyDiffusionPlannerConfig,
)
from train.train_bev_diffusion_stage1 import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_SCHEMA_VERSION,
    Stage1TrainingError,
    build_stage1_optimizer,
    checkpoint_payload,
    configure_stage1_variant,
    load_stage1_checkpoint,
    save_stage1_checkpoint,
)


def _planner(variant: str) -> BEVOnlyDiffusionPlanner:
    condition = "none" if variant == "A" else "predicted_detached"
    return BEVOnlyDiffusionPlanner(
        BEVOnlyDiffusionPlannerConfig(
            d_model=32,
            num_heads=4,
            ffn_dim=64,
            decoder_layers=1,
            predecessor_condition=condition,
        )
    )


def _config(variant: str) -> dict[str, object]:
    base: dict[str, object] = {
        "experiment": {
            "variant": "A",
            "joint_update": "joint_mean",
            "predecessor_condition": "none",
        },
        "dataset": {
            "root": "/tmp/data",
            "train_split": "train",
            "val_split": "val",
        },
        "training": {
            "device": "cpu",
            "mixed_precision": False,
            "grad_scaler_init_scale": 1024.0,
            "seed": 17,
            "batch_size": 8,
            "gradient_accumulation": 2,
            "max_epochs": 2,
            "num_workers": 0,
            "planner_lr": 1e-4,
            "backbone_lr": 1e-5,
            "weight_decay": 1e-4,
            "gradient_diagnostics_interval": 100,
        },
        "loss": {
            "xy_weight": 1.0,
            "heading_weight": 0.2,
            "mode_weight": 1.0,
            "motion_weight": 0.05,
            "pair_weight": 0.25,
            "xy_beta_m": 1.0,
            "heading_beta_rad": 0.1,
            "motion_beta": 1.0,
            "pair_beta_m": 1.0,
            "dt_s": 0.5,
        },
        "overfit": {
            "max_optimizer_steps": 500,
            "evaluation_interval_steps": 20,
            "timestep": 8,
            "min_loss_reduction": 0.70,
            "min_mode_accuracy": 0.95,
            "max_gt_mode_ade_m": 1.0,
        },
    }
    return configure_stage1_variant(base, variant)  # type: ignore[arg-type]


def _checkpoint(
    tmp_path: Path,
    planner: BEVOnlyDiffusionPlanner,
    variant: str,
) -> Path:
    optimizer = build_stage1_optimizer(
        planner,
        planner_lr=1e-4,
        backbone_lr=1e-5,
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    payload = checkpoint_payload(
        planner=planner,
        optimizer=optimizer,
        scaler=scaler,
        config=_config(variant),
        dataset_fingerprint="b" * 64,
        epoch=0,
        optimizer_step=4,
        metrics={"loss/total": 1.0},
        run_mode="smoke",
    )
    return save_stage1_checkpoint(tmp_path / f"{variant}.pt", payload)


def test_same_seed_keeps_all_shared_a_b_parameters_bitwise_equal() -> None:
    torch.manual_seed(17)
    planner_a = _planner("A")
    state_a = planner_a.state_dict()
    torch.manual_seed(17)
    planner_b = _planner("B")
    state_b = planner_b.state_dict()

    common = sorted(set(state_a).intersection(state_b))
    assert common
    assert all(torch.equal(state_a[name], state_b[name]) for name in common)
    assert set(state_b).difference(state_a) == {
        "diffusion_decoder.predecessor_residual_gate",
        "diffusion_decoder.predecessor_action_encoder.network.0.weight",
        "diffusion_decoder.predecessor_action_encoder.network.0.bias",
        "diffusion_decoder.predecessor_action_encoder.network.1.weight",
        "diffusion_decoder.predecessor_action_encoder.network.1.bias",
        "diffusion_decoder.predecessor_action_encoder.network.3.weight",
        "diffusion_decoder.predecessor_action_encoder.network.3.bias",
    }


@pytest.mark.parametrize("variant", ["A", "B"])
def test_schema_v2_smoke_checkpoint_round_trip(
    tmp_path: Path, variant: str
) -> None:
    planner = _planner(variant)
    path = _checkpoint(tmp_path, planner, variant)
    payload = load_stage1_checkpoint(path, planner)
    assert payload["schema_version"] == CHECKPOINT_SCHEMA_VERSION == 2
    assert payload["format"] == CHECKPOINT_FORMAT
    assert payload["variant"] == variant
    assert payload["run_mode"] == "smoke"
    assert payload["diagnostic_only"] is True
    assert payload["cross_split_overfit"] is False
    assert payload["eligible_for_formal_training"] is False


def test_a_b_checkpoint_cross_load_is_rejected(tmp_path: Path) -> None:
    planner_a = _planner("A")
    path_a = _checkpoint(tmp_path, planner_a, "A")
    with pytest.raises(Stage1TrainingError, match="variant mismatch"):
        load_stage1_checkpoint(path_a, _planner("B"))

    planner_b = _planner("B")
    path_b = _checkpoint(tmp_path, planner_b, "B")
    with pytest.raises(Stage1TrainingError, match="variant mismatch"):
        load_stage1_checkpoint(path_b, _planner("A"))


def test_schema_v1_is_explicitly_rejected(tmp_path: Path) -> None:
    planner = _planner("A")
    path = _checkpoint(tmp_path, planner, "A")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    legacy = copy.deepcopy(payload)
    legacy["schema_version"] = 1
    legacy_path = tmp_path / "schema_v1.pt"
    torch.save(legacy, legacy_path)
    with pytest.raises(Stage1TrainingError, match="schema_version mismatch"):
        load_stage1_checkpoint(legacy_path, planner)
