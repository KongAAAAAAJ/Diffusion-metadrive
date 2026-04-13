from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit("matplotlib is required for saving BaseMultiEnv map images.") from exc

from metadrive.envs.diffusion_envs.base_multi_env import BaseMultiEnv
from metadrive.utils.draw_top_down_map import draw_top_down_map


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
