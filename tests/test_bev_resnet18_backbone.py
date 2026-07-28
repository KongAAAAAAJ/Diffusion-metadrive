from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

import models.bev_planner.bev_resnet18_backbone as backbone_module
from models.bev_planner.bev_resnet18_backbone import (
    BEVBackboneError,
    BEVResNet18Backbone,
    BEVResNet18Config,
    DEFAULT_RESNET18_A1_WEIGHTS,
    RESNET18_A1_SHA256,
    sha256_file,
)


@pytest.fixture(scope="module")
def backbone() -> BEVResNet18Backbone:
    return BEVResNet18Backbone()


def test_real_safetensors_hash_and_strict_config_are_frozen() -> None:
    config = BEVResNet18Config()

    assert config.weights_path == DEFAULT_RESNET18_A1_WEIGHTS
    assert config.expected_sha256 == RESNET18_A1_SHA256
    assert sha256_file(config.weights_path) == RESNET18_A1_SHA256


def test_bn_statistics_reset_but_affine_weights_remain_pretrained(
    backbone: BEVResNet18Backbone,
) -> None:
    pretrained = load_file(str(DEFAULT_RESNET18_A1_WEIGHTS), device="cpu")
    batch_norms = [
        module
        for module in backbone.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]

    assert batch_norms
    for module in batch_norms:
        torch.testing.assert_close(
            module.running_mean,
            torch.zeros_like(module.running_mean),
        )
        torch.testing.assert_close(
            module.running_var,
            torch.ones_like(module.running_var),
        )
        assert int(module.num_batches_tracked.item()) == 0
    torch.testing.assert_close(backbone.encoder.bn1.weight, pretrained["bn1.weight"])
    torch.testing.assert_close(backbone.encoder.bn1.bias, pretrained["bn1.bias"])


def test_conv1_uses_mean_repeat_and_three_eighths_scaling(
    backbone: BEVResNet18Backbone,
) -> None:
    rgb_weight = load_file(
        str(DEFAULT_RESNET18_A1_WEIGHTS),
        device="cpu",
    )["conv1.weight"]
    expected = rgb_weight.mean(dim=1, keepdim=True).repeat(1, 8, 1, 1)
    expected.mul_(3.0 / 8.0)

    assert backbone.encoder.conv1.weight.shape == (64, 8, 7, 7)
    torch.testing.assert_close(backbone.encoder.conv1.weight, expected)


def test_four_scale_output_shapes_values_and_determinism(
    backbone: BEVResNet18Backbone,
) -> None:
    generator = torch.Generator().manual_seed(7)
    bev = torch.randint(
        0,
        256,
        (2, 8, 256, 256),
        dtype=torch.uint8,
        generator=generator,
    )
    backbone.eval()
    with torch.no_grad():
        first = backbone(bev)
        second = backbone(bev)

    assert backbone.feature_channels == (64, 128, 256, 512)
    assert backbone.feature_strides == (4, 8, 16, 32)
    assert tuple(tensor.shape for tensor in first) == (
        (2, 64, 64, 64),
        (2, 128, 32, 32),
        (2, 256, 16, 16),
        (2, 512, 8, 8),
    )
    for first_tensor, second_tensor in zip(first, second):
        assert first_tensor.dtype == torch.float32
        assert torch.isfinite(first_tensor).all()
        torch.testing.assert_close(first_tensor, second_tensor, rtol=0.0, atol=0.0)


def test_uint8_input_is_scaled_only_by_255(
    backbone: BEVResNet18Backbone,
) -> None:
    captured = []

    def capture_input(module, inputs):
        captured.append(inputs[0].detach().clone())

    handle = backbone.encoder.conv1.register_forward_pre_hook(capture_input)
    backbone.eval()
    try:
        with torch.no_grad():
            backbone(torch.full((1, 8, 256, 256), 255, dtype=torch.uint8))
    finally:
        handle.remove()

    assert len(captured) == 1
    torch.testing.assert_close(captured[0], torch.ones_like(captured[0]))


def test_all_backbone_stages_receive_gradients(
    backbone: BEVResNet18Backbone,
) -> None:
    generator = torch.Generator().manual_seed(11)
    bev = torch.randint(
        0,
        256,
        (2, 8, 256, 256),
        dtype=torch.uint8,
        generator=generator,
    )
    backbone.train()
    backbone.zero_grad(set_to_none=True)
    features = backbone(bev)
    loss = sum(feature.square().mean() for feature in features)
    loss.backward()

    assert all(parameter.requires_grad for parameter in backbone.parameters())
    for parameter in (
        backbone.encoder.conv1.weight,
        backbone.encoder.layer4[1].conv2.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_joint_first_batch_can_flatten_and_restore_role_axis(
    backbone: BEVResNet18Backbone,
) -> None:
    batch_size = 1
    joint_bev = torch.zeros(
        (batch_size, 3, 8, 256, 256),
        dtype=torch.uint8,
    )
    backbone.eval()
    with torch.no_grad():
        flat_features = backbone(
            joint_bev.reshape(batch_size * 3, 8, 256, 256)
        )
    restored = tuple(
        feature.reshape(batch_size, 3, *feature.shape[1:])
        for feature in flat_features
    )

    assert tuple(tensor.shape for tensor in restored) == (
        (1, 3, 64, 64, 64),
        (1, 3, 128, 32, 32),
        (1, 3, 256, 16, 16),
        (1, 3, 512, 8, 8),
    )


@pytest.mark.parametrize(
    "bev,match",
    [
        (torch.zeros((1, 8, 256, 256)), "torch.uint8"),
        (torch.zeros((8, 256, 256), dtype=torch.uint8), "shape"),
        (torch.zeros((1, 7, 256, 256), dtype=torch.uint8), "shape"),
        (torch.zeros((1, 8, 128, 256), dtype=torch.uint8), "shape"),
        (torch.zeros((0, 8, 256, 256), dtype=torch.uint8), "positive"),
    ],
)
def test_forward_rejects_invalid_input(
    backbone: BEVResNet18Backbone,
    bev: torch.Tensor,
    match: str,
) -> None:
    with pytest.raises(BEVBackboneError, match=match):
        backbone(bev)


def test_missing_file_and_wrong_digest_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(BEVBackboneError, match="does not exist"):
        BEVResNet18Backbone(
            BEVResNet18Config(
                weights_path=tmp_path / "missing.safetensors",
                expected_sha256="0" * 64,
            )
        )

    monkeypatch.setattr(
        backbone_module,
        "sha256_file",
        lambda path: RESNET18_A1_SHA256,
    )
    with pytest.raises(BEVBackboneError, match="SHA256 mismatch"):
        BEVResNet18Backbone(
            BEVResNet18Config(
                weights_path=DEFAULT_RESNET18_A1_WEIGHTS,
                expected_sha256="0" * 64,
            )
        )


def test_wrong_pretrained_architecture_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weights_path = tmp_path / "model.safetensors"
    weights_path.symlink_to(DEFAULT_RESNET18_A1_WEIGHTS)
    source_config = json.loads(
        DEFAULT_RESNET18_A1_WEIGHTS.with_name("config.json").read_text(
            encoding="utf-8"
        )
    )
    source_config["architecture"] = "resnet34"
    (tmp_path / "config.json").write_text(
        json.dumps(source_config),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        backbone_module,
        "sha256_file",
        lambda path: RESNET18_A1_SHA256,
    )

    with pytest.raises(BEVBackboneError, match="architecture"):
        BEVResNet18Backbone(
            BEVResNet18Config(
                weights_path=weights_path,
                expected_sha256=RESNET18_A1_SHA256,
            )
        )
