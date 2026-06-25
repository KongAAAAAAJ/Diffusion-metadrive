from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required for saving BaseMultiEnv map images.") from exc

from envs.diffusion_envs.base_multi_env import BaseMultiEnv
from metadrive.utils.draw_top_down_map import draw_top_down_map


def _iter_block_roads(block) -> Iterable[object]:
    seen = set()
    for road in getattr(block, "get_respawn_roads", lambda: [])() or []:
        key = (getattr(road, "start_node", None), getattr(road, "end_node", None))
        if None not in key and key not in seen:
            seen.add(key)
            yield road
    for socket in getattr(block, "get_socket_list", lambda: [])() or []:
        road = getattr(socket, "positive_road", None)
        key = (getattr(road, "start_node", None), getattr(road, "end_node", None))
        if road is not None and None not in key and key not in seen:
            seen.add(key)
            yield road


def _lane_midpoint(current_map, road) -> tuple[float, float] | None:
    try:
        lanes = current_map.road_network.graph[road.start_node][road.end_node]
    except Exception:
        return None
    if not lanes:
        return None
    lane = lanes[len(lanes) // 2]
    length = float(getattr(lane, "length", 0.0))
    if length <= 0.0 or not hasattr(lane, "position"):
        return None
    try:
        point = lane.position(length * 0.5, 0.0)
        return float(point[0]), float(point[1])
    except Exception:
        return None


def _collect_block_label_positions(current_map) -> list[tuple[str, float, float]]:
    labels: list[tuple[str, float, float]] = []
    for block in getattr(current_map, "blocks", []) or []:
        block_id = getattr(block, "graph_block_id", None)
        if block_id is None:
            continue
        points = [
            point for road in _iter_block_roads(block)
            if (point := _lane_midpoint(current_map, road)) is not None
        ]
        if not points:
            continue
        x = sum(point[0] for point in points) / len(points)
        y = sum(point[1] for point in points) / len(points)
        labels.append((str(block_id), float(x), float(y)))
    return labels


def _world_to_image_pixel(current_map, x: float, y: float, width: int, height: int) -> tuple[float, float] | None:
    try:
        min_x, max_x, min_y, max_y = current_map.road_network.get_bounding_box()
    except Exception:
        return None
    x_len = float(max_x) - float(min_x)
    y_len = float(max_y) - float(min_y)
    max_len = max(x_len, y_len)
    if max_len <= 0.0:
        return None
    film_width = film_height = 2000.0
    scaling = film_height / max_len - 0.1
    center_x = (float(min_x) + float(max_x)) / 2.0
    center_y = (float(min_y) + float(max_y)) / 2.0
    origin_x = center_x - 0.5 * film_width / scaling
    origin_y = center_y - 0.5 * film_height / scaling
    film_px = math.ceil((float(x) - origin_x) * scaling)
    film_py = film_height - math.ceil((float(y) - origin_y) * scaling)
    return film_px * float(width) / film_width, film_py * float(height) / film_height


def _draw_block_labels(current_map, surface) -> None:
    try:
        height, width = int(len(surface)), int(len(surface[0]))
    except Exception:
        shape = getattr(surface, "shape", None)
        if shape is None or len(shape) < 2:
            return
        height, width = int(shape[0]), int(shape[1])
    if width <= 0 or height <= 0:
        return
    for block_id, x, y in _collect_block_label_positions(current_map):
        pixel = _world_to_image_pixel(current_map, x, y, width, height)
        if pixel is None:
            continue
        plt.text(
            pixel[0],
            pixel[1],
            block_id,
            color="black",
            fontsize=5,
            # fontweight="bold",
            ha="center",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round,pad=0.18", "alpha": 0.82},
        )


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_blocks_config(value: str | None):
    if value is None:
        return None

    candidate = Path(value)
    payload = candidate.read_text(encoding="utf-8") if candidate.exists() else value
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            "Invalid hybrid map blocks config. Provide a JSON array or a path to a JSON file."
        ) from exc

    if parsed is None:
        return None
    if not isinstance(parsed, list):
        raise argparse.ArgumentTypeError("hybrid_map_blocks_config must decode to a JSON array.")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save one BaseMultiEnv map as a top-down PNG image.")
    parser.add_argument("--output", type=Path, default=Path("outputs/base_multi_env_map.png"))
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--num-scenarios", type=int, default=1)
    parser.add_argument("--route-preset", type=str, default=None, help="BaseMultiEnv route preset override.")
    parser.add_argument("--use-hybrid-map", type=_parse_bool, default=None)
    parser.add_argument(
        "--hybrid-map-blocks-config",
        type=_parse_blocks_config,
        default=None,
        help="JSON array or path to JSON file describing block configs.",
    )
    parser.add_argument("--map", dest="map_config", default=None, help="Optional map override.")
    parser.add_argument("--render", type=_parse_bool, default=False)
    return parser


def build_env_config(args: argparse.Namespace) -> dict[str, object]:
    default_env_config = BaseMultiEnv.default_config()
    config: dict[str, object] = {
        "use_render": bool(args.render),
        "start_seed": int(args.start_seed),
        "num_scenarios": int(args.num_scenarios),
        "traffic_density": 0.0,
        "random_traffic": False,
        "route_preset": str(
            default_env_config.get("route_preset", "mainline") if args.route_preset is None else args.route_preset
        ),
        "use_hybrid_map": bool(
            default_env_config["use_hybrid_map"] if args.use_hybrid_map is None else args.use_hybrid_map
        ),
        "hybrid_map_blocks_config": (
            default_env_config.get("hybrid_map_blocks_config")
            if args.hybrid_map_blocks_config is None
            else args.hybrid_map_blocks_config
        ),
    }
    if args.map_config is not None:
        config["map"] = args.map_config
    return config


def save_map_image(args: argparse.Namespace) -> Path:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = BaseMultiEnv(build_env_config(args))
    try:
        env.reset()
        current_map = getattr(env, "current_map", None)
        if current_map is None and hasattr(env, "engine"):
            current_map = getattr(getattr(env.engine, "map_manager", None), "current_map", None)
        if current_map is None:
            raise RuntimeError("BaseMultiEnv did not produce a current_map after reset().")

        surface = draw_top_down_map(current_map, resolution=(int(args.resolution), int(args.resolution)))
        plt.imshow(surface, cmap="Greys")
        _draw_block_labels(current_map, surface)
        plt.axis("off")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
    finally:
        env.close()
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    env_config = build_env_config(args)
    saved_path = save_map_image(args)
    print(f"saved_map={saved_path}")
    print(f"seed={int(args.start_seed)}")
    print(f"blocks_config={'set' if env_config['hybrid_map_blocks_config'] else 'unset'}")
    print(f"route_preset={env_config['route_preset']}")
    print(f"resolution={int(args.resolution)}")
    print(f"use_hybrid_map={env_config['use_hybrid_map']}")
    print(f"map={args.map_config if args.map_config is not None else 'default'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
