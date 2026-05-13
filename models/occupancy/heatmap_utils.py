"""Gaussian heatmap rasterization and candidate trajectory matching utilities.

All coordinates are in ego-local frame:
  x = forward (longitudinal), y = left (lateral).
Grid conventions:
  row 0 = lon_max (far ahead), row H-1 = lon_min (behind ego)
  col 0 = lat_max (far left),  col W-1 = lat_min (far right)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def world_to_ego_local_xy(points_world: torch.Tensor, ego_pose: torch.Tensor) -> torch.Tensor:
    """Convert world xy points to ego-local xy.

    Parameters
    ----------
    points_world : Tensor [..., 2]
    ego_pose     : Tensor [..., 3] or [3] containing x, y, heading.
    """
    while ego_pose.ndim < points_world.ndim:
        ego_pose = ego_pose.unsqueeze(-2)
    dx = points_world[..., 0] - ego_pose[..., 0]
    dy = points_world[..., 1] - ego_pose[..., 1]
    cos_h = torch.cos(ego_pose[..., 2])
    sin_h = torch.sin(ego_pose[..., 2])
    local_x = cos_h * dx + sin_h * dy
    local_y = -sin_h * dx + cos_h * dy
    return torch.stack([local_x, local_y], dim=-1)


def ego_local_to_world_xy(points_local: torch.Tensor, ego_pose: torch.Tensor) -> torch.Tensor:
    """Convert ego-local xy points to world xy."""
    while ego_pose.ndim < points_local.ndim:
        ego_pose = ego_pose.unsqueeze(-2)
    cos_h = torch.cos(ego_pose[..., 2])
    sin_h = torch.sin(ego_pose[..., 2])
    world_x = ego_pose[..., 0] + cos_h * points_local[..., 0] - sin_h * points_local[..., 1]
    world_y = ego_pose[..., 1] + sin_h * points_local[..., 0] + cos_h * points_local[..., 1]
    return torch.stack([world_x, world_y], dim=-1)


def rasterize_to_heatmap(
    traj_local: torch.Tensor,
    grid_h: int = 64,
    grid_w: int = 64,
    lon_range: tuple[float, float] = (-2.0, 30.0),
    lat_range: tuple[float, float] = (-10.0, 10.0),
    sigma_m: float = 1.0,
) -> torch.Tensor:
    """Rasterize an ego-local trajectory to a Gaussian occupancy heatmap.

    Parameters
    ----------
    traj_local : Tensor [..., T, 2]  (x=forward, y=left)
    Returns
    -------
    heatmap : Tensor [..., H, W]  values in [0, 1]
    """
    *batch, T, _ = traj_local.shape
    device = traj_local.device
    dtype = traj_local.dtype

    lon_min, lon_max = float(lon_range[0]), float(lon_range[1])
    lat_min, lat_max = float(lat_range[0]), float(lat_range[1])

    # pixel size in metres
    px_lon = (lon_max - lon_min) / grid_h   # metres per row
    px_lat = (lat_max - lat_min) / grid_w   # metres per col

    # normalised sigma in pixel units
    sigma_row = sigma_m / px_lon
    sigma_col = sigma_m / px_lat

    # grid coordinates [H, W]
    rows = torch.linspace(lon_max - 0.5 * px_lon, lon_min + 0.5 * px_lon, grid_h, device=device, dtype=dtype)
    cols = torch.linspace(lat_max - 0.5 * px_lat, lat_min + 0.5 * px_lat, grid_w, device=device, dtype=dtype)
    # rows shape [H, 1], cols shape [1, W]
    rows = rows.view(grid_h, 1)
    cols = cols.view(1, grid_w)

    # traj_local [..., T, 2] → x_t [..., T], y_t [..., T]
    x_t = traj_local[..., 0]   # longitudinal
    y_t = traj_local[..., 1]   # lateral

    # Broadcast: [..., T, H, W]
    x_t = x_t.unsqueeze(-1).unsqueeze(-1)   # [..., T, 1, 1]
    y_t = y_t.unsqueeze(-1).unsqueeze(-1)

    dist_lon = (rows - x_t) ** 2 / (2.0 * sigma_row ** 2)
    dist_lat = (cols - y_t) ** 2 / (2.0 * sigma_col ** 2)
    gauss = torch.exp(-(dist_lon + dist_lat))   # [..., T, H, W]

    heatmap = gauss.max(dim=-3).values          # [..., H, W]  max-pool over T
    return heatmap.clamp(0.0, 1.0)


def rasterize_polyline_mask(
    polyline_local: torch.Tensor,
    grid_h: int = 64,
    grid_w: int = 64,
    lon_range: tuple[float, float] = (-2.0, 30.0),
    lat_range: tuple[float, float] = (-10.0, 10.0),
    sigma_m: float = 1.5,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Rasterize an ego-local polyline/corridor to a soft binary mask."""
    heatmap = rasterize_to_heatmap(
        polyline_local,
        grid_h=grid_h,
        grid_w=grid_w,
        lon_range=lon_range,
        lat_range=lat_range,
        sigma_m=sigma_m,
    )
    return (heatmap / max(float(threshold), 1e-6)).clamp(0.0, 1.0)


def build_reachable_mask(
    batch_size: int,
    speed_mps: torch.Tensor,
    grid_h: int = 64,
    grid_w: int = 64,
    lon_range: tuple[float, float] = (-2.0, 30.0),
    lat_range: tuple[float, float] = (-10.0, 10.0),
    horizon_s: float = 4.0,
    max_accel_mps2: float = 1.5,
    max_lateral_m: float = 6.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a simple ego-local reachability mask from speed and horizon."""
    device = device or speed_mps.device
    lon_min, lon_max = float(lon_range[0]), float(lon_range[1])
    lat_min, lat_max = float(lat_range[0]), float(lat_range[1])
    px_lon = (lon_max - lon_min) / grid_h
    px_lat = (lat_max - lat_min) / grid_w
    rows = torch.linspace(lon_max - 0.5 * px_lon, lon_min + 0.5 * px_lon, grid_h, device=device, dtype=dtype)
    cols = torch.linspace(lat_max - 0.5 * px_lat, lat_min + 0.5 * px_lat, grid_w, device=device, dtype=dtype)
    lon = rows.view(1, grid_h, 1)
    lat = cols.view(1, 1, grid_w)
    speed = speed_mps.to(device=device, dtype=dtype).view(batch_size, 1, 1).clamp_min(0.0)
    max_forward = speed * float(horizon_s) + 0.5 * float(max_accel_mps2) * float(horizon_s) ** 2
    forward_ok = (lon >= 0.0) & (lon <= max_forward)
    lateral_ok = lat.abs() <= float(max_lateral_m)
    return (forward_ok & lateral_ok).to(dtype)


def heatmap_matching(
    candidates: torch.Tensor,
    heatmap: torch.Tensor,
    corrected_traj: torch.Tensor | None = None,
    grid_h: int = 64,
    grid_w: int = 64,
    lon_range: tuple[float, float] = (-2.0, 30.0),
    lat_range: tuple[float, float] = (-10.0, 10.0),
    route_corridor_mask: torch.Tensor | None = None,
    dynamic_occupancy_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute per-candidate matching features against a feasible occupancy map.

    Parameters
    ----------
    candidates    : Tensor [N, M, T, 3]  ego-local (x, y, heading)
    heatmap       : Tensor [N, H, W]
    corrected_traj: Tensor [N, T, 2] or None
        Optional auxiliary target trajectory. When supplied, endpoint errors are
        reported for backwards compatibility. When omitted, the endpoint errors
        are replaced by occupancy/corridor-derived penalties.

    Returns
    -------
    features : Tensor [N, M, 6]
        dim 0: matching_score   (mean heatmap value along candidate) ∈ [0,1]
        dim 1: min_score        minimum heatmap value along candidate ∈ [0,1]
        dim 2: endpoint_score   heatmap value at candidate endpoint ∈ [0,1]
        dim 3: offroad_penalty  1 - route corridor score, or 1 - matching_score
        dim 4: dynamic_overlap  mean dynamic occupancy along candidate ∈ [0,1]
        dim 5: endpoint_err     optional target endpoint L2 error, else 1-endpoint_score
    """
    N, M, T, _ = candidates.shape
    device = candidates.device
    dtype = candidates.dtype

    lon_min, lon_max = float(lon_range[0]), float(lon_range[1])
    lat_min, lat_max = float(lat_range[0]), float(lat_range[1])

    cand_xy = candidates[..., :2]   # [N, M, T, 2]

    # ── matching score via bilinear sampling ───────────────────────────────
    # normalise to [-1, 1] for grid_sample (y_grid=lon, x_grid=lat)
    lon_norm = 1.0 - 2.0 * (cand_xy[..., 0] - lon_min) / (lon_max - lon_min)   # [N, M, T]
    lat_norm = 1.0 - 2.0 * (cand_xy[..., 1] - lat_min) / (lat_max - lat_min)   # [N, M, T]
    lon_norm = lon_norm.clamp(-1.0, 1.0)
    lat_norm = lat_norm.clamp(-1.0, 1.0)

    # grid_sample expects [N, C, H_out, W_out] input and [N, H_out, W_out, 2] grid
    # We sample M*T points per batch item
    grid = torch.stack([lat_norm, lon_norm], dim=-1)   # [N, M, T, 2]
    grid_flat = grid.view(N, M * T, 1, 2)              # [N, M*T, 1, 2]

    heatmap_4d = heatmap.unsqueeze(1)                  # [N, 1, H, W]
    sampled = F.grid_sample(
        heatmap_4d.float(), grid_flat.float(),
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )   # [N, 1, M*T, 1]
    sampled = sampled.squeeze(1).squeeze(-1)            # [N, M*T]
    sampled = sampled.view(N, M, T).to(dtype)
    matching_score = sampled.mean(dim=-1)               # [N, M]
    min_score = sampled.min(dim=-1).values
    endpoint_score = sampled[:, :, -1]

    def _sample_optional_mask(mask: torch.Tensor | None, default: float) -> torch.Tensor:
        if mask is None:
            return torch.full((N, M, T), float(default), device=device, dtype=dtype)
        if mask.ndim == 4 and mask.shape[1] == 1:
            mask_4d = mask.to(device=device, dtype=dtype)
        else:
            mask_4d = mask.to(device=device, dtype=dtype).unsqueeze(1)
        values = F.grid_sample(
            mask_4d.float(), grid_flat.float(),
            mode="bilinear", padding_mode="zeros", align_corners=True,
        )
        return values.squeeze(1).squeeze(-1).view(N, M, T).to(dtype)

    route_values = _sample_optional_mask(route_corridor_mask, default=1.0)
    dyn_values = _sample_optional_mask(dynamic_occupancy_mask, default=0.0)
    if route_corridor_mask is None:
        offroad_penalty = 1.0 - matching_score
    else:
        offroad_penalty = 1.0 - route_values.mean(dim=-1)
    dynamic_overlap = dyn_values.mean(dim=-1)

    if corrected_traj is not None:
        cand_end = cand_xy[:, :, -1, :]                 # [N, M, 2]
        target_end = corrected_traj[:, -1, :]           # [N, 2]
        endpoint_err = torch.norm(cand_end - target_end.unsqueeze(1), dim=-1)
    else:
        endpoint_err = 1.0 - endpoint_score

    return torch.stack(
        [matching_score, min_score, endpoint_score, offroad_penalty, dynamic_overlap, endpoint_err],
        dim=-1,
    )
