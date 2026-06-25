from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


MODE_NAME_MAP = {
    0: "straight",
    1: "curve_left",
    2: "curve_right",
    3: "lane_change_left",
    4: "lane_change_right",
    -1: "unknown",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 local-route verification plots.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-traj-plots", type=int, default=500)
    parser.add_argument("--heatmap-bins", type=int, default=200)
    return parser.parse_args()


def decode_string_array(values: np.ndarray) -> np.ndarray:
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        elif isinstance(value, np.bytes_):
            decoded.append(bytes(value).decode("utf-8"))
        else:
            decoded.append(str(value))
    return np.asarray(decoded, dtype=object)


def load_manifest(dataset_root: Path) -> Dict | None:
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        manifest_path = dataset_root / "reports" / "manifest.json"
    if not manifest_path.is_file():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def iter_shard_paths(dataset_root: Path) -> List[Path]:
    shard_dir = dataset_root / "shards"
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"Shard directory not found: {shard_dir}")
    shard_paths = sorted(shard_dir.glob("*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found under {shard_dir}")
    return shard_paths


def load_dataset(dataset_root: Path) -> Dict[str, np.ndarray]:
    ego_pose_chunks: List[np.ndarray] = []
    future_pose_chunks: List[np.ndarray] = []
    route_id_chunks: List[np.ndarray] = []
    local_route_chunks: List[np.ndarray] = []
    scenario_id_chunks: List[np.ndarray] = []
    mode_chunks: List[np.ndarray] = []
    episode_id_chunks: List[np.ndarray] = []

    for shard_path in iter_shard_paths(dataset_root):
        with np.load(shard_path, allow_pickle=True) as shard:
            if "ego_pose_world" in shard:
                ego_pose_chunks.append(np.asarray(shard["ego_pose_world"], dtype=np.float32))
            if "future_ego_pose_world" in shard:
                future_pose_chunks.append(np.asarray(shard["future_ego_pose_world"], dtype=np.float32))
            if "route_id" in shard:
                route_id_chunks.append(decode_string_array(np.asarray(shard["route_id"], dtype=object)))
            if "local_route" in shard:
                local_route_chunks.append(decode_string_array(np.asarray(shard["local_route"], dtype=object)))
            if "scenario_id" in shard:
                scenario_id_chunks.append(decode_string_array(np.asarray(shard["scenario_id"], dtype=object)))
            if "trajectory_mode" in shard:
                mode_chunks.append(np.asarray(shard["trajectory_mode"], dtype=np.int16))
            if "episode_id" in shard:
                episode_id_chunks.append(np.asarray(shard["episode_id"], dtype=np.int32))

    if not ego_pose_chunks or not future_pose_chunks:
        raise RuntimeError("Required keys ego_pose_world/future_ego_pose_world are missing from dataset shards.")

    total_samples = int(sum(chunk.shape[0] for chunk in ego_pose_chunks))
    route_id = (
        np.concatenate(route_id_chunks, axis=0)
        if route_id_chunks
        else np.asarray(["unknown"] * total_samples, dtype=object)
    )
    local_route = (
        np.concatenate(local_route_chunks, axis=0)
        if local_route_chunks
        else route_id.copy()
    )
    scenario_id = (
        np.concatenate(scenario_id_chunks, axis=0)
        if scenario_id_chunks
        else np.asarray(["unknown"] * total_samples, dtype=object)
    )
    trajectory_mode = (
        np.concatenate(mode_chunks, axis=0)
        if mode_chunks
        else np.full((total_samples,), -1, dtype=np.int16)
    )
    episode_id = np.concatenate(episode_id_chunks, axis=0) if episode_id_chunks else None

    data = {
        "ego_pose_world": np.concatenate(ego_pose_chunks, axis=0),
        "future_ego_pose_world": np.concatenate(future_pose_chunks, axis=0),
        "route_id": route_id,
        "local_route": local_route,
        "scenario_id": scenario_id,
        "trajectory_mode": trajectory_mode,
    }
    if episode_id is not None:
        data["episode_id"] = episode_id
    return data


def route_sort_key(route: str) -> tuple[int, int, str]:
    route = str(route)
    if route.startswith("R"):
        prefix, _, remainder = route.partition("_")
        numeric = prefix[1:]
        if numeric.isdigit():
            return (0, int(numeric), route)
    if route == "unknown":
        return (2, 10**6, route)
    return (1, 10**6, route)


def ordered_routes(route_ids: Iterable[str]) -> List[str]:
    return sorted({str(route) for route in route_ids}, key=route_sort_key)


def scenario_sort_key(scenario_id: str) -> tuple[int, int, str]:
    scenario_id = str(scenario_id)
    if scenario_id.startswith("S"):
        prefix = scenario_id.split("_", 1)[0]
        numeric = prefix[1:]
        if numeric.isdigit():
            return (0, int(numeric), scenario_id)
    if scenario_id == "unknown":
        return (2, 10**6, scenario_id)
    return (1, 10**6, scenario_id)


def ordered_scenarios(scenario_ids: Iterable[str]) -> List[str]:
    return sorted({str(scenario_id) for scenario_id in scenario_ids}, key=scenario_sort_key)


def has_episode_ids(data: Dict[str, np.ndarray]) -> bool:
    episode_ids = data.get("episode_id")
    return episode_ids is not None and len(np.asarray(episode_ids)) == len(np.asarray(data["local_route"]))


def count_routes(data: Dict[str, np.ndarray]) -> tuple[Dict[str, int], str]:
    route_ids = np.asarray(data["local_route"], dtype=object)
    routes = ordered_routes(route_ids)
    if has_episode_ids(data):
        episode_ids = np.asarray(data["episode_id"], dtype=np.int64)
        counts = {
            route: int(np.unique(episode_ids[route_ids == route]).size)
            for route in routes
        }
        return counts, "episode"
    counts = {route: int(np.sum(route_ids == route)) for route in routes}
    return counts, "sample"


def count_scenarios(data: Dict[str, np.ndarray]) -> tuple[Dict[str, int], str]:
    scenario_ids = np.asarray(data["scenario_id"], dtype=object)
    scenarios = ordered_scenarios(scenario_ids)
    if has_episode_ids(data):
        episode_ids = np.asarray(data["episode_id"], dtype=np.int64)
        counts = {
            scenario_id: int(np.unique(episode_ids[scenario_ids == scenario_id]).size)
            for scenario_id in scenarios
        }
        return counts, "episode"
    counts = {scenario_id: int(np.sum(scenario_ids == scenario_id)) for scenario_id in scenarios}
    return counts, "sample"


def save_heatmap(data: Dict[str, np.ndarray], output_dir: Path, bins: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ego_xy = np.asarray(data["ego_pose_world"][:, :2], dtype=np.float32)
    future_xy = np.asarray(data["future_ego_pose_world"][:, :, :2], dtype=np.float32)
    all_points = np.concatenate([ego_xy[:, None, :], future_xy], axis=1).reshape(-1, 2)
    hist, x_edges, y_edges = np.histogram2d(all_points[:, 0], all_points[:, 1], bins=bins)
    hist = np.asarray(hist, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#f6f4ef")
    ax.set_facecolor("#fbf8f1")
    positive_hist = hist[hist > 0.0]
    norm = None
    if positive_hist.size > 0:
        vmax = float(np.max(positive_hist))
        if vmax > 1.0:
            norm = colors.LogNorm(vmin=1.0, vmax=vmax)
    image = ax.imshow(
        hist.T,
        origin="lower",
        cmap="YlOrRd",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        aspect="equal",
        interpolation="bilinear",
        norm=norm,
    )
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("visit density", rotation=90)
    cbar.outline.set_linewidth(0.6)
    ax.set_title(
        f"Trajectory Density Heatmap (N={data['ego_pose_world'].shape[0]} samples)",
        fontsize=13,
        pad=12,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(color="#ffffff", linewidth=0.6, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_color("#d8d0c2")
        spine.set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_route_counts(data: Dict[str, np.ndarray], manifest: Dict | None, output_dir: Path) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    route_ids = np.asarray(data["local_route"], dtype=object)
    routes = ordered_routes(route_ids)
    counts, count_basis = count_routes(data)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(routes))
    bars = ax.bar(x, [counts[route] for route in routes], color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels(routes, rotation=20, ha="right")
    ax.set_ylabel(f"{count_basis.title()} count")
    ax.set_title(f"{count_basis.title()} Count by Local Route")

    for bar, route in zip(bars, routes):
        height = float(bar.get_height())
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, str(counts[route]), ha="center", va="bottom")

    if manifest is not None:
        manifest_counts = manifest.get("local_route_distribution") or manifest.get("route_distribution") or {}
        if manifest_counts:
            if count_basis == "sample":
                match = True
                for route in routes:
                    if int(manifest_counts.get(route, 0)) != counts[route]:
                        match = False
                        break
                status = "[PASS]" if match else "[MISMATCH]"
            else:
                status = "[SKIPPED: sample-based manifest vs episode-based plot]"
            fig.text(0.5, 0.01, f"Manifest cross-check: {status}", ha="center", fontsize=9)

    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    fig.savefig(output_dir / "fig2_route_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return counts


def save_scenario_counts(data: Dict[str, np.ndarray], manifest: Dict | None, output_dir: Path) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_ids = np.asarray(data["scenario_id"], dtype=object)
    scenarios = ordered_scenarios(scenario_ids)
    counts, count_basis = count_scenarios(data)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(scenarios))
    bars = ax.bar(x, [counts[scenario_id] for scenario_id in scenarios], color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=25, ha="right")
    ax.set_ylabel(f"{count_basis.title()} count")
    ax.set_title(f"{count_basis.title()} Count by Scenario")
    for bar, scenario_id in zip(bars, scenarios):
        height = float(bar.get_height())
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, str(counts[scenario_id]), ha="center", va="bottom")

    if manifest is not None:
        manifest_counts = manifest.get("scenario_distribution") or {}
        if manifest_counts:
            status = "[SKIPPED]" if count_basis == "episode" else (
                "[PASS]" if all(int(manifest_counts.get(scenario_id, 0)) == counts[scenario_id] for scenario_id in scenarios) else "[MISMATCH]"
            )
            fig.text(0.5, 0.01, f"Manifest cross-check: {status}", ha="center", fontsize=9)

    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    fig.savefig(output_dir / "fig3_scenario_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return counts


def save_scenario_route_matrix(data: Dict[str, np.ndarray], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_ids = np.asarray(data["scenario_id"], dtype=object)
    route_ids = np.asarray(data["local_route"], dtype=object)
    scenarios = ordered_scenarios(scenario_ids)
    routes = ordered_routes(route_ids)
    matrix = np.zeros((len(scenarios), len(routes)), dtype=np.int32)

    if has_episode_ids(data):
        episode_ids = np.asarray(data["episode_id"], dtype=np.int64)
        for scenario_idx, scenario_id in enumerate(scenarios):
            for route_idx, route_id in enumerate(routes):
                mask = (scenario_ids == scenario_id) & (route_ids == route_id)
                matrix[scenario_idx, route_idx] = int(np.unique(episode_ids[mask]).size)
    else:
        for scenario_idx, scenario_id in enumerate(scenarios):
            for route_idx, route_id in enumerate(routes):
                matrix[scenario_idx, route_idx] = int(np.sum((scenario_ids == scenario_id) & (route_ids == route_id)))

    fig_width = max(8.0, 1.2 * len(routes))
    fig_height = max(5.0, 0.6 * len(scenarios))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(routes)))
    ax.set_xticklabels(routes, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(scenarios)))
    ax.set_yticklabels(scenarios)
    ax.set_xlabel("Local route")
    ax.set_ylabel("Scenario")
    ax.set_title("Scenario x Local Route Count Matrix")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(col_idx, row_idx, str(int(matrix[row_idx, col_idx])), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(output_dir / "fig4_scenario_route_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_ego_marker(ax, start_xy: np.ndarray, label: str = "ego"):
    x = float(start_xy[0])
    y = float(start_xy[1])
    outer = ax.scatter([x], [y], s=120, c="#fff7ed", edgecolors="#c2410c", linewidths=1.4, zorder=5)
    inner = ax.scatter([x], [y], s=34, c="#ea580c", edgecolors="white", linewidths=0.7, zorder=6)
    text = ax.text(
        x,
        y + 2.2,
        label,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#9a3412",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff7ed", edgecolor="#fdba74", linewidth=0.8, alpha=0.95),
        zorder=7,
    )
    return outer, inner, text


def save_per_route_trajectories(data: Dict[str, np.ndarray], output_dir: Path, max_traj_plots: int) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    route_ids = np.asarray(data["local_route"], dtype=object)
    ego_pose_world = np.asarray(data["ego_pose_world"], dtype=np.float32)
    future_ego_pose_world = np.asarray(data["future_ego_pose_world"], dtype=np.float32)
    trajectory_mode = np.asarray(data["trajectory_mode"], dtype=np.int16)
    routes = ordered_routes(route_ids)
    rng = np.random.RandomState(0)
    per_route_root = output_dir / "per_route"
    per_route_root.mkdir(parents=True, exist_ok=True)

    saved_counts: Dict[str, int] = {}
    for route in routes:
        route_dir = per_route_root / route
        route_dir.mkdir(parents=True, exist_ok=True)
        indices = np.flatnonzero(route_ids == route)
        if indices.size > max_traj_plots:
            indices = np.sort(rng.choice(indices, size=max_traj_plots, replace=False))

        total = int(indices.size)
        for done, sample_index in enumerate(indices, start=1):
            fig, ax = plt.subplots(figsize=(6, 6))
            future_xy = future_ego_pose_world[sample_index, :, :2]
            start_xy = ego_pose_world[sample_index, :2]
            all_xy = np.concatenate([start_xy[None, :], future_xy], axis=0)
            xy_min = np.min(all_xy, axis=0)
            xy_max = np.max(all_xy, axis=0)
            xy_span = np.maximum(xy_max - xy_min, 1e-3)
            margin = np.maximum(0.15 * xy_span, 5.0)
            min_span = np.asarray([30.0, 30.0], dtype=np.float32)
            final_span = np.maximum(xy_span + 2.0 * margin, min_span)
            center = 0.5 * (xy_min + xy_max)
            ax.plot(future_xy[:, 0], future_xy[:, 1], "o-", markersize=3, color="#1f77b4")
            draw_ego_marker(ax, start_xy)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title(f"Route: {route} | traj {int(sample_index)}")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(center[0] - final_span[0] / 2.0, center[0] + final_span[0] / 2.0)
            ax.set_ylim(center[1] - final_span[1] / 2.0, center[1] + final_span[1] / 2.0)
            ax.grid(True, alpha=0.25, linewidth=0.5)
            ax.text(0.02, 0.97, "scenario: [Phase 2]", transform=ax.transAxes, va="top", fontsize=8, color="gray")
            mode_name = MODE_NAME_MAP.get(int(trajectory_mode[sample_index]), "unknown")
            ax.text(0.02, 0.91, f"mode: {mode_name}", transform=ax.transAxes, va="top", fontsize=8)
            fig.tight_layout()
            fig.savefig(route_dir / f"traj_{int(sample_index):06d}.png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            if done % 50 == 0:
                print(f"[{route}] {done}/{total} saved...")

        saved_counts[route] = total
    return saved_counts


def save_per_scenario_trajectories(data: Dict[str, np.ndarray], output_dir: Path, max_traj_plots: int) -> Dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_ids = np.asarray(data["scenario_id"], dtype=object)
    local_route_ids = np.asarray(data["local_route"], dtype=object)
    ego_pose_world = np.asarray(data["ego_pose_world"], dtype=np.float32)
    future_ego_pose_world = np.asarray(data["future_ego_pose_world"], dtype=np.float32)
    trajectory_mode = np.asarray(data["trajectory_mode"], dtype=np.int16)
    scenarios = ordered_scenarios(scenario_ids)
    rng = np.random.RandomState(0)
    per_scenario_root = output_dir / "per_scenario"
    per_scenario_root.mkdir(parents=True, exist_ok=True)

    saved_counts: Dict[str, int] = {}
    for scenario_id in scenarios:
        scenario_dir = per_scenario_root / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        indices = np.flatnonzero(scenario_ids == scenario_id)
        if indices.size > max_traj_plots:
            indices = np.sort(rng.choice(indices, size=max_traj_plots, replace=False))

        total = int(indices.size)
        for done, sample_index in enumerate(indices, start=1):
            fig, ax = plt.subplots(figsize=(6, 6))
            future_xy = future_ego_pose_world[sample_index, :, :2]
            start_xy = ego_pose_world[sample_index, :2]
            all_xy = np.concatenate([start_xy[None, :], future_xy], axis=0)
            xy_min = np.min(all_xy, axis=0)
            xy_max = np.max(all_xy, axis=0)
            xy_span = np.maximum(xy_max - xy_min, 1e-3)
            margin = np.maximum(0.15 * xy_span, 5.0)
            min_span = np.asarray([30.0, 30.0], dtype=np.float32)
            final_span = np.maximum(xy_span + 2.0 * margin, min_span)
            center = 0.5 * (xy_min + xy_max)
            ax.plot(future_xy[:, 0], future_xy[:, 1], "o-", markersize=3, color="#ff7f0e")
            draw_ego_marker(ax, start_xy)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title(f"Scenario: {scenario_id} | traj {int(sample_index)}")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(center[0] - final_span[0] / 2.0, center[0] + final_span[0] / 2.0)
            ax.set_ylim(center[1] - final_span[1] / 2.0, center[1] + final_span[1] / 2.0)
            ax.grid(True, alpha=0.25, linewidth=0.5)
            ax.text(0.02, 0.97, f"local_route: {local_route_ids[sample_index]}", transform=ax.transAxes, va="top", fontsize=8)
            mode_name = MODE_NAME_MAP.get(int(trajectory_mode[sample_index]), "unknown")
            ax.text(0.02, 0.91, f"mode: {mode_name}", transform=ax.transAxes, va="top", fontsize=8)
            fig.tight_layout()
            fig.savefig(scenario_dir / f"traj_{int(sample_index):06d}.png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            if done % 50 == 0:
                print(f"[{scenario_id}] {done}/{total} saved...")

        saved_counts[scenario_id] = total
    return saved_counts


def print_summary(
    data: Dict[str, np.ndarray],
    route_counts: Dict[str, int],
    scenario_counts: Dict[str, int],
    saved_counts: Dict[str, int],
    scenario_saved_counts: Dict[str, int],
    manifest: Dict | None,
    output_dir: Path,
) -> None:
    total_samples = int(data["ego_pose_world"].shape[0])
    routes = ordered_routes(data["local_route"])
    route_basis = "sample"
    route_counts, route_basis = count_routes(data)
    total_route_count = int(sum(route_counts.values()))
    manifest_counts = {}
    manifest_status = "[MISSING]"
    if manifest is not None:
        manifest_counts = manifest.get("local_route_distribution") or manifest.get("route_distribution") or {}
        if manifest_counts:
            if route_basis == "sample":
                manifest_status = "[PASS]" if all(int(manifest_counts.get(route, 0)) == route_counts[route] for route in routes) else "[MISMATCH]"
            else:
                manifest_status = "[SKIPPED]"

    print("=== Phase 1 Verification Summary ===")
    print(f"Total samples: {total_samples}")
    if route_basis == "episode":
        total_episodes = int(np.unique(np.asarray(data["episode_id"], dtype=np.int64)).size)
        print(f"Total episodes: {total_episodes}")
    print("Local routes:")
    for route in routes:
        count = route_counts[route]
        print(f"  {route:24s} {count:6d} ({count / max(total_route_count, 1):5.1%})")
    print("Scenarios:")
    total_scenario_count = max(int(sum(scenario_counts.values())), 1)
    for scenario_id in ordered_scenarios(data["scenario_id"]):
        count = scenario_counts[scenario_id]
        print(f"  {scenario_id:24s} {count:6d} ({count / total_scenario_count:5.1%})")
    print(f"Count basis: {route_basis}")
    print(f"Manifest cross-check: {manifest_status}")
    print("Per-route trajectory plots:")
    for route in routes:
        print(f"  {route}: {saved_counts.get(route, 0)} plots -> {output_dir / 'per_route' / route}")
    print("Per-scenario trajectory plots:")
    for scenario_id in ordered_scenarios(data["scenario_id"]):
        print(f"  {scenario_id}: {scenario_saved_counts.get(scenario_id, 0)} plots -> {output_dir / 'per_scenario' / scenario_id}")
    print(f"Output: {output_dir}/")


def main() -> None:
    args = parse_args()
    data = load_dataset(args.dataset_root)
    manifest = load_manifest(args.dataset_root)
    route_counts = save_route_counts(data, manifest=manifest, output_dir=args.output_dir)
    scenario_counts = save_scenario_counts(data, manifest=manifest, output_dir=args.output_dir)
    save_heatmap(data, args.output_dir, bins=args.heatmap_bins)
    save_scenario_route_matrix(data, output_dir=args.output_dir)
    saved_counts = save_per_route_trajectories(data, args.output_dir, max_traj_plots=args.max_traj_plots)
    scenario_saved_counts = save_per_scenario_trajectories(data, args.output_dir, max_traj_plots=args.max_traj_plots)
    print_summary(data, route_counts, scenario_counts, saved_counts, scenario_saved_counts, manifest, args.output_dir)


if __name__ == "__main__":
    main()
