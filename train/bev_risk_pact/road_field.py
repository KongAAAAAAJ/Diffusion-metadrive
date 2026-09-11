from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .config import RiskPACTConfig


# Frozen SemanticBEV metric contract from envs/observations/semantic_bev.py.
BEV_HEIGHT = 256
BEV_WIDTH = 256
BEV_X_MIN_M = -12.0
BEV_X_MAX_M = 52.0
BEV_Y_MIN_M = -32.0
BEV_Y_MAX_M = 32.0
DRIVABLE_CHANNEL = 0


@dataclass(frozen=True)
class RoadRiskResult:
    risk: Tensor
    signed_distance_m: Tensor


def _drivable_binary_from_bev(bev: Tensor) -> np.ndarray:
    """Return a CPU bool drivable mask with shape [B,R,H,W]."""

    if not isinstance(bev, Tensor):
        raise ValueError("bev must be a torch.Tensor")
    if bev.ndim != 5 or int(bev.shape[1]) != 3 or int(bev.shape[2]) <= DRIVABLE_CHANNEL:
        raise ValueError("bev must have shape [B,3,C,H,W]")
    if tuple(bev.shape[-2:]) != (BEV_HEIGHT, BEV_WIDTH):
        raise ValueError(
            f"Risk-PACT road field expects frozen BEV size {(BEV_HEIGHT, BEV_WIDTH)}, "
            f"got {tuple(bev.shape[-2:])}"
        )

    layer = bev[:, :, DRIVABLE_CHANNEL].detach().cpu()
    if layer.is_floating_point():
        # The online path normally carries uint8 [0,255].  Accept normalized
        # [0,1] tensors as well so the contract is robust to future loaders.
        threshold = 0.5 if float(layer.max()) <= 1.5 else 127.5
    else:
        threshold = 127.5
    return (layer.numpy() > threshold)


def build_drivable_signed_distance(bev: Tensor) -> Tensor:
    """Build a metric signed-distance field from the frozen DRIVABLE channel.

    Positive values are inside the drivable region, negative values are outside.
    The field itself is detached scene geometry; gradients later flow only
    through differentiable trajectory-coordinate sampling.
    """

    mask = _drivable_binary_from_bev(bev)
    b, r, h, w = mask.shape
    x_resolution = (BEV_X_MAX_M - BEV_X_MIN_M) / float(h - 1)
    y_resolution = (BEV_Y_MAX_M - BEV_Y_MIN_M) / float(w - 1)
    if abs(x_resolution - y_resolution) > 1.0e-6:
        raise ValueError("current Risk-PACT EDT assumes equal BEV x/y pixel resolution")
    resolution_m = float(x_resolution)

    sdf = np.empty((b, r, h, w), dtype=np.float32)
    for batch_index in range(b):
        for role_index in range(r):
            binary = np.ascontiguousarray(mask[batch_index, role_index].astype(np.uint8))
            inside_px = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            outside_px = cv2.distanceTransform(1 - binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            sdf[batch_index, role_index] = (inside_px - outside_px) * resolution_m

    return torch.as_tensor(sdf, dtype=torch.float32, device=bev.device).detach()


class RoadBoundaryRiskField:
    """Differentiable road-boundary risk queried from a detached BEV SDF."""

    def __init__(self, config: RiskPACTConfig | None = None) -> None:
        self.config = config or RiskPACTConfig()

    @staticmethod
    def _metric_to_grid(trajectory_xy: Tensor) -> Tensor:
        x = trajectory_xy[..., 0]
        y = trajectory_xy[..., 1]
        # SemanticBEV maps ego-left to smaller image columns and ego-forward to
        # smaller image rows.  align_corners=True reproduces that exact mapping.
        col_norm = 2.0 * (BEV_Y_MAX_M - y) / (BEV_Y_MAX_M - BEV_Y_MIN_M) - 1.0
        row_norm = 2.0 * (BEV_X_MAX_M - x) / (BEV_X_MAX_M - BEV_X_MIN_M) - 1.0
        return torch.stack((col_norm, row_norm), dim=-1)

    def query(self, trajectory_xy: Tensor, road_sdf: Tensor) -> RoadRiskResult:
        squeeze_mode = False
        if trajectory_xy.ndim == 4:
            trajectory_xy = trajectory_xy.unsqueeze(2)
            squeeze_mode = True
        if trajectory_xy.ndim != 5 or trajectory_xy.shape[-1] != 2:
            raise ValueError("trajectory_xy must have shape [B,R,M,H,2] or [B,R,H,2]")
        if road_sdf.ndim != 4 or tuple(road_sdf.shape[:2]) != tuple(trajectory_xy.shape[:2]):
            raise ValueError("road_sdf must have shape [B,R,H_bev,W_bev] matching trajectory B/R")
        if tuple(road_sdf.shape[-2:]) != (BEV_HEIGHT, BEV_WIDTH):
            raise ValueError("road_sdf has an unexpected BEV resolution")

        b, r, m, horizon, _ = trajectory_xy.shape
        grid = self._metric_to_grid(trajectory_xy)
        # Clamp to the raster boundary for a stable boundary sample.  Metric
        # out-of-bounds distance below then makes points outside progressively
        # less safe and provides an inward gradient.
        clamped_grid = grid.clamp(-1.0, 1.0)
        flat_sdf = road_sdf.to(dtype=trajectory_xy.dtype, device=trajectory_xy.device).reshape(
            b * r, 1, BEV_HEIGHT, BEV_WIDTH
        )
        flat_grid = clamped_grid.reshape(b * r, m * horizon, 1, 2)
        sampled = F.grid_sample(
            flat_sdf,
            flat_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).reshape(b, r, m, horizon)

        x = trajectory_xy[..., 0]
        y = trajectory_xy[..., 1]
        outside_x = torch.relu(BEV_X_MIN_M - x) + torch.relu(x - BEV_X_MAX_M)
        outside_y = torch.relu(BEV_Y_MIN_M - y) + torch.relu(y - BEV_Y_MAX_M)
        outside_distance = outside_x + outside_y
        effective_signed_distance = sampled - outside_distance

        required_clearance = (
            0.5 * float(self.config.ego_width_m)
            + float(self.config.road_safety_margin_m)
        )
        temperature = float(self.config.road_temperature_m)
        risk = torch.sigmoid((required_clearance - effective_signed_distance) / temperature)

        if squeeze_mode:
            risk = risk.squeeze(2)
            effective_signed_distance = effective_signed_distance.squeeze(2)
        return RoadRiskResult(risk=risk, signed_distance_m=effective_signed_distance)
