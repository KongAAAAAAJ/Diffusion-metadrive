"""Strict ResNet18.a1 backbone for eight-channel semantic BEV inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import timm
import torch
from safetensors.torch import load_file
from torch import Tensor, nn


DEFAULT_RESNET18_A1_WEIGHTS: Final[Path] = Path(
    "/media/kong/Elements_SE/Diffusion_Data/ckpts/"
    "resnet18.a1_in1k/model.safetensors"
)
RESNET18_A1_SHA256: Final[str] = (
    "80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612"
)


class BEVBackboneError(RuntimeError):
    """Raised when the frozen BEV backbone contract cannot be satisfied."""


@dataclass(frozen=True)
class BEVResNet18Config:
    """Immutable local-weight contract for the BEV ResNet18 backbone."""

    weights_path: Path = DEFAULT_RESNET18_A1_WEIGHTS
    expected_sha256: str = RESNET18_A1_SHA256

    def __post_init__(self) -> None:
        weights_path = Path(self.weights_path).expanduser()
        digest = str(self.expected_sha256)
        if len(digest) != 64 or digest != digest.lower():
            raise BEVBackboneError("expected_sha256 must be a lowercase SHA256 digest")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise BEVBackboneError(
                "expected_sha256 must be a lowercase SHA256 digest"
            ) from exc
        if weights_path.suffix != ".safetensors":
            raise BEVBackboneError("weights_path must reference a .safetensors file")
        object.__setattr__(self, "weights_path", weights_path)
        object.__setattr__(self, "expected_sha256", digest)


def sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Return the SHA256 digest of one local file."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_bytes)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise BEVBackboneError(f"unable to read weights file: {path}") from exc
    return digest.hexdigest()


def _validate_pretrained_config(weights_path: Path) -> None:
    config_path = weights_path.with_name("config.json")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BEVBackboneError(
            f"invalid ResNet18 pretrained config: {config_path}"
        ) from exc
    pretrained_cfg = payload.get("pretrained_cfg")
    if payload.get("architecture") != "resnet18":
        raise BEVBackboneError("pretrained architecture must be resnet18")
    if not isinstance(pretrained_cfg, dict):
        raise BEVBackboneError("pretrained_cfg must be an object")
    if pretrained_cfg.get("tag") != "a1_in1k":
        raise BEVBackboneError("pretrained tag must be a1_in1k")
    if pretrained_cfg.get("first_conv") != "conv1":
        raise BEVBackboneError("pretrained first_conv must be conv1")


def _adapt_first_conv_to_bev(rgb_weight: Tensor) -> Tensor:
    if tuple(rgb_weight.shape[1:]) != (3, 7, 7):
        raise BEVBackboneError(
            "pretrained conv1.weight must have shape [out_channels,3,7,7]"
        )
    mean_kernel = rgb_weight.mean(dim=1, keepdim=True)
    return mean_kernel.repeat(1, 8, 1, 1).mul(3.0 / 8.0)


class BEVResNet18Backbone(nn.Module):
    """Trainable four-scale ResNet18 encoder for uint8 semantic BEV."""

    feature_channels: Final[tuple[int, int, int, int]] = (64, 128, 256, 512)
    feature_strides: Final[tuple[int, int, int, int]] = (4, 8, 16, 32)
    input_shape: Final[tuple[int, int, int]] = (8, 256, 256)

    def __init__(self, config: BEVResNet18Config | None = None) -> None:
        super().__init__()
        self.config = config or BEVResNet18Config()
        weights_path = self.config.weights_path
        if not weights_path.is_file():
            raise BEVBackboneError(f"weights file does not exist: {weights_path}")
        actual_sha256 = sha256_file(weights_path)
        if actual_sha256 != self.config.expected_sha256:
            raise BEVBackboneError(
                "ResNet18 weights SHA256 mismatch: "
                f"expected {self.config.expected_sha256}, got {actual_sha256}"
            )
        _validate_pretrained_config(weights_path)

        encoder = timm.create_model(
            "resnet18",
            pretrained=False,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )
        try:
            pretrained_state = load_file(str(weights_path), device="cpu")
        except Exception as exc:
            raise BEVBackboneError(
                f"unable to load safetensors weights: {weights_path}"
            ) from exc
        feature_state = {
            name: value
            for name, value in pretrained_state.items()
            if not name.startswith("fc.")
        }
        try:
            encoder.load_state_dict(feature_state, strict=True)
        except RuntimeError as exc:
            raise BEVBackboneError(
                "ResNet18 feature weights do not strictly match the architecture"
            ) from exc

        rgb_conv = encoder.conv1
        bev_conv = nn.Conv2d(
            in_channels=8,
            out_channels=rgb_conv.out_channels,
            kernel_size=rgb_conv.kernel_size,
            stride=rgb_conv.stride,
            padding=rgb_conv.padding,
            dilation=rgb_conv.dilation,
            groups=rgb_conv.groups,
            bias=False,
            padding_mode=rgb_conv.padding_mode,
        )
        with torch.no_grad():
            bev_conv.weight.copy_(_adapt_first_conv_to_bev(rgb_conv.weight))
        encoder.conv1 = bev_conv

        for module in encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.reset_running_stats()

        if tuple(encoder.feature_info.channels()) != self.feature_channels:
            raise BEVBackboneError("unexpected ResNet18 feature channel contract")
        if tuple(encoder.feature_info.reduction()) != self.feature_strides:
            raise BEVBackboneError("unexpected ResNet18 feature stride contract")
        self.encoder = encoder

    def forward(self, bev: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if not isinstance(bev, Tensor):
            raise BEVBackboneError("bev must be a torch.Tensor")
        if bev.dtype is not torch.uint8:
            raise BEVBackboneError("bev must use torch.uint8")
        if bev.ndim != 4 or tuple(bev.shape[1:]) != self.input_shape:
            raise BEVBackboneError("bev must have shape [N,8,256,256]")
        if int(bev.shape[0]) <= 0:
            raise BEVBackboneError("bev batch dimension must be positive")
        normalized = bev.to(dtype=torch.float32).mul_(1.0 / 255.0)
        features = tuple(self.encoder(normalized))
        if len(features) != 4:
            raise BEVBackboneError("ResNet18 must return four feature scales")
        return features  # type: ignore[return-value]


__all__ = [
    "BEVBackboneError",
    "BEVResNet18Backbone",
    "BEVResNet18Config",
    "DEFAULT_RESNET18_A1_WEIGHTS",
    "RESNET18_A1_SHA256",
    "sha256_file",
]
