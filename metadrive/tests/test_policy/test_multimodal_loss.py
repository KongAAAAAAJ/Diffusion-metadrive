from __future__ import annotations

import pytest
import torch

from metadrive.policy.diffusion_policy.modules.multimodal_loss import LossComputer
from metadrive.policy.diffusion_policy.transfuser_config import build_transfuser_config


def _make_loss_computer() -> LossComputer:
    config = build_transfuser_config(
        "small",
        trajectory_cls_weight=1.0,
        trajectory_reg_weight=1.0,
    )
    return LossComputer(config)


def test_loss_computer_prefers_hierarchical_mode_label_over_nearest_anchor():
    loss_computer = _make_loss_computer()

    poses_reg = torch.tensor(
        [[
            [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]],
        dtype=torch.float32,
    )
    poses_cls = torch.tensor([[-10.0, 10.0]], dtype=torch.float32)
    targets = {
        "trajectory": torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float32),
        "hierarchical_mode_label": torch.tensor([1], dtype=torch.int64),
    }
    plan_anchor = torch.tensor(
        [[
            [[0.0, 0.0], [0.0, 0.0]],
            [[10.0, 0.0], [10.0, 0.0]],
        ]],
        dtype=torch.float32,
    )
    mode_valid_mask = torch.tensor([[True, True]])

    loss = loss_computer(poses_reg, poses_cls, targets, plan_anchor, mode_valid_mask=mode_valid_mask)

    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_loss_computer_falls_back_when_hierarchical_mode_label_is_invalid():
    loss_computer = _make_loss_computer()

    poses_reg = torch.tensor(
        [[
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        ]],
        dtype=torch.float32,
    )
    poses_cls = torch.tensor([[10.0, -10.0]], dtype=torch.float32)
    targets = {
        "trajectory": torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float32),
        "hierarchical_mode_label": torch.tensor([1], dtype=torch.int64),
    }
    plan_anchor = torch.tensor(
        [[
            [[0.0, 0.0], [0.0, 0.0]],
            [[10.0, 0.0], [10.0, 0.0]],
        ]],
        dtype=torch.float32,
    )
    mode_valid_mask = torch.tensor([[True, False]])

    loss = loss_computer(poses_reg, poses_cls, targets, plan_anchor, mode_valid_mask=mode_valid_mask)

    assert loss.item() == pytest.approx(0.0, abs=1e-3)
