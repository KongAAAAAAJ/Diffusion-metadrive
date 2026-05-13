"""Lightweight ego-centric feasible occupancy predictor.

The predictor estimates a future feasible driving region for the current
vehicle in ego-local coordinates.  It starts from rule-based feasibility masks
(route corridor, reachability, dynamic actors, formation prior) and predicts a
small residual heatmap correction.  The output is used only for trajectory
classification/re-ranking; it does not generate or modify planner candidates.

All coordinates are ego-local (x=forward, y=left).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .heatmap_utils import rasterize_to_heatmap


class OccupancyPredictor(nn.Module):
    """Predict a residual correction to an ego-local feasible occupancy map.

    Parameters
    ----------
    traj_points : int
        Number of waypoints in target trajectory (default 8).
    status_dim : int
        Dimension of the ego status feature vector.
    hidden_dim : int
        Hidden dimension of the MLP.
    grid_h, grid_w : int
        Output heatmap grid resolution.
    lon_range, lat_range : tuple[float, float]
        Ego-local spatial range in metres (longitudinal / lateral).
    sigma_m : float
        Gaussian blob sigma in metres.
    residual_scale : float
        Maximum auxiliary target-line residual, kept for backwards-compatible
        diagnostics and optional supervision.
    """

    def __init__(
        self,
        traj_points: int = 8,
        status_dim: int = 19,
        hidden_dim: int = 128,
        grid_h: int = 64,
        grid_w: int = 64,
        lon_range: tuple[float, float] = (-2.0, 30.0),
        lat_range: tuple[float, float] = (-10.0, 10.0),
        sigma_m: float = 1.0,
        residual_scale: float = 2.0,
    ) -> None:
        super().__init__()
        self.traj_points = int(traj_points)
        self.status_dim = int(status_dim)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.lon_range = tuple(lon_range)
        self.lat_range = tuple(lat_range)
        self.sigma_m = float(sigma_m)
        self.residual_scale = float(residual_scale)

        # Auxiliary target-line residual.  This is no longer the main output,
        # but retaining it keeps older training/debug code compatible.
        in_dim = traj_points * 2 + self.status_dim + 1

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, traj_points * 2),
        )

        # Initialise last layer near-zero so residual starts small
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # Feasible occupancy residual. Channels:
        #   base, target-line prior, route corridor, reachable, dynamic unsafe,
        #   formation prior.
        self.residual_cnn = nn.Sequential(
            nn.Conv2d(6, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.residual_cnn[-2].weight)
        nn.init.zeros_(self.residual_cnn[-2].bias)

        # Exact zero-init makes the first forward pass an identity transform:
        # feasible_occupancy_heatmap == rule-based base mask.
        self.residual_gate = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _coerce_mask(mask: torch.Tensor | None, like: torch.Tensor, fill: float) -> torch.Tensor:
        if mask is None:
            return torch.full_like(like, float(fill))
        if mask.ndim == 4 and mask.shape[1] == 1:
            mask = mask[:, 0]
        return mask.to(device=like.device, dtype=like.dtype).clamp(0.0, 1.0)

    def forward(
        self,
        target_traj: torch.Tensor,    # [N, T, 2]  ego-local (x, y)
        status: torch.Tensor,          # [N, status_dim]
        lane_decision: torch.Tensor,   # [N, 1] or [N]
        base_feasible_mask: torch.Tensor | None = None,      # [N, H, W]
        route_corridor_mask: torch.Tensor | None = None,     # [N, H, W]
        reachable_mask: torch.Tensor | None = None,          # [N, H, W]
        dynamic_occupancy_mask: torch.Tensor | None = None,  # [N, H, W], 1=unsafe/occupied
        formation_prior_mask: torch.Tensor | None = None,    # [N, H, W]
    ) -> dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with:
          feasible_occupancy_heatmap : [N, H, W] final feasible region
          base_feasible_mask         : [N, H, W] rule-based prior
          residual_heatmap           : [N, H, W] learned residual in [-1, 1]
          corrected_traj             : [N, T, 2] auxiliary target_traj + residual
          delta_traj                 : [N, T, 2] auxiliary predicted residual
          heatmap                    : alias of feasible_occupancy_heatmap
        """
        N, T, _ = target_traj.shape
        device = target_traj.device

        ld = lane_decision.view(N, 1).float().to(device)
        status = status.float().to(device)
        if status.shape[-1] < self.status_dim:
            pad = torch.zeros(N, self.status_dim - status.shape[-1], device=device, dtype=status.dtype)
            status = torch.cat([status, pad], dim=-1)
        elif status.shape[-1] > self.status_dim:
            status = status[:, : self.status_dim]
        flat_traj = target_traj.reshape(N, T * 2)
        x = torch.cat([flat_traj, status, ld], dim=-1)

        delta_flat = self.mlp(x)                           # [N, T*2]
        delta_traj = delta_flat.view(N, T, 2)
        delta_traj = torch.tanh(delta_traj) * self.residual_scale

        corrected_traj = target_traj + delta_traj          # [N, T, 2]
        target_prior = rasterize_to_heatmap(
            corrected_traj,
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            lon_range=self.lon_range,
            lat_range=self.lat_range,
            sigma_m=self.sigma_m,
        )                                                   # [N, H, W]

        if base_feasible_mask is None:
            # Backwards-compatible fallback: use the target-line prior when no
            # richer rule-based feasible mask is available.
            base = target_prior
        else:
            base = base_feasible_mask.to(device=device, dtype=target_traj.dtype).clamp(0.0, 1.0)

        route = self._coerce_mask(route_corridor_mask, base, fill=1.0)
        reachable = self._coerce_mask(reachable_mask, base, fill=1.0)
        dynamic_unsafe = self._coerce_mask(dynamic_occupancy_mask, base, fill=0.0)
        formation = self._coerce_mask(formation_prior_mask, base, fill=0.0)

        # Rule feasibility: stay in route/reachable space, avoid dynamic actors.
        base = (base * route * reachable * (1.0 - dynamic_unsafe)).clamp(0.0, 1.0)
        # Formation prior is an additional desirable feasible region.  It should
        # not erase route/reachable constraints, so gate it through them too.
        formation_feasible = (formation * route * reachable * (1.0 - dynamic_unsafe)).clamp(0.0, 1.0)
        base = torch.maximum(base, formation_feasible)

        residual_in = torch.stack([base, target_prior, route, reachable, dynamic_unsafe, formation], dim=1)
        residual = self.residual_cnn(residual_in).squeeze(1)
        feasible_heatmap = (base + self.residual_gate * residual).clamp(0.0, 1.0)

        return {
            "corrected_traj": corrected_traj,
            "delta_traj": delta_traj,
            "base_feasible_mask": base,
            "target_prior_heatmap": target_prior,
            "residual_heatmap": residual,
            "feasible_occupancy_heatmap": feasible_heatmap,
            "heatmap": feasible_heatmap,
        }
