from __future__ import annotations

import argparse
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save one BaseMultiEnv map as a top-down PNG image.")
    parser.add_argument("--output", type=Path, default=Path("outputs/base_multi_env_map.png"))
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--num-scenarios", type=int, default=1)
    parser.add_argument("--use-hybrid-map", type=_parse_bool, default=True)
    parser.add_argument("--hybrid-map-sequence", type=str, default="SSXCOCSS")
    parser.add_argument("--map", dest="map_config", default=None, help="Optional map override.")
    parser.add_argument("--render", type=_parse_bool, default=False)
    return parser


def build_env_config(args: argparse.Namespace) -> dict[str, object]:
    config: dict[str, object] = {
        "use_render": bool(args.render),
        "start_seed": int(args.start_seed),
        "num_scenarios": int(args.num_scenarios),
        "use_hybrid_map": bool(args.use_hybrid_map),
        "hybrid_map_sequence": str(args.hybrid_map_sequence),
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
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
    finally:
        env.close()
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    saved_path = save_map_image(args)
    print(f"saved_map={saved_path}")
    print(f"seed={int(args.start_seed)}")
    print(f"sequence={str(args.hybrid_map_sequence)}")
    print(f"resolution={int(args.resolution)}")
    print(f"use_hybrid_map={bool(args.use_hybrid_map)}")
    print(f"map={args.map_config if args.map_config is not None else 'default'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
