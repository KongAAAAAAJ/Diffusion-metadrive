"""preview_platoon.py

Run PlatoonEnv with ExpertIDMPolicy for all agents and save top-down videos.
Used by scripts/preview_scenario.sh --platoon to manually verify scenario designs.

Follows the same rendering strategy as collect_expert.py:
  - use_render=False (no 3D Panda3D window)
  - top-down frames captured via env.render(mode="top_down", window=False, ...)
  - per-vehicle labels (ego1, ego2, ...) overlaid using the same style as the
    single-vehicle "EGO" label in TopDownRenderer._draw_tracked_agent_highlight
  - videos written with mediapy

Usage:
    python -m metadrive.exp_dataset.preview_platoon \
        --scenario-id S5_hard_brake_lead \
        --local-route R3_mainline_straight \
        --num-agents 3 \
        --num-episodes 3 \
        --output-root /tmp/platoon_preview
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_TOPDOWN_SCREEN_SIZE = 800
_TOPDOWN_FILM_SIZE = 3000

# Label style mirrors TopDownRenderer.EGO_LABEL_*
_LABEL_BG = (255, 247, 237)
_LABEL_FG = (154, 52, 18)
_LABEL_BORDER = (251, 146, 60)


def _build_render_kwargs(
    camera_position: tuple[float, float] | None = None,
    heading_up: bool = False,
) -> dict:
    kwargs: dict = {
        "mode": "top_down",
        "window": False,
        "screen_size": (_TOPDOWN_SCREEN_SIZE, _TOPDOWN_SCREEN_SIZE),
        "film_size": (_TOPDOWN_FILM_SIZE, _TOPDOWN_FILM_SIZE),
        "target_agent_heading_up": heading_up,
    }
    if camera_position is not None:
        kwargs["camera_position"] = camera_position
    return kwargs


def _overlay_platoon_labels(
    frame: np.ndarray,
    env,
    agent_ids: list[str],
) -> np.ndarray:
    """Draw ego1/ego2/... labels at each platoon vehicle's screen position.

    Mirrors the style of TopDownRenderer._draw_tracked_agent_highlight.
    The coordinate transform matches _world_to_screen_position exactly:
        screen_pos = frame_canvas.pos2pix(world_pos) - off
    where off is reconstructed from renderer.position (camera center).
    """
    renderer = getattr(env, "top_down_renderer", None)
    if renderer is None:
        return frame

    agents = getattr(env, "agents", {}) or {}
    screen_w, screen_h = frame.shape[1], frame.shape[0]

    # Reconstruct off: the frame-canvas pixel that maps to screen (0, 0)
    cam_pos = getattr(renderer, "position", None)
    if cam_pos is None:
        tracked = getattr(renderer, "current_track_agent", None)
        if tracked is None:
            return frame
        cam_pos = tracked.position

    try:
        frame_px = renderer._frame_canvas.pos2pix(float(cam_pos[0]), float(cam_pos[1]))
        off = np.asarray([frame_px[0] - screen_w / 2, frame_px[1] - screen_h / 2], dtype=np.float32)
    except Exception:
        return frame

    import pygame
    if not pygame.font.get_init():
        pygame.font.init()
    if not pygame.get_init():
        pygame.init()
    font = pygame.font.Font(None, 16)

    surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

    for i, agent_id in enumerate(agent_ids):
        vehicle = agents.get(agent_id)
        if vehicle is None:
            continue
        try:
            screen_pos = renderer._world_to_screen_position(vehicle.position, off)
        except Exception:
            continue
        if screen_pos is None:
            continue

        sx, sy = int(screen_pos[0]), int(screen_pos[1])
        # Skip if clearly off-screen
        if sx < -40 or sx > screen_w + 40 or sy < -40 or sy > screen_h + 40:
            continue

        label = f"ego{i + 1}"
        text_surf = font.render(label, True, _LABEL_FG, _LABEL_BG)
        text_rect = text_surf.get_rect()
        # Offset same as _draw_tracked_agent_highlight: +10, -12 from vehicle center
        text_rect.center = (sx + 10, sy - 12)
        padded_rect = text_rect.inflate(8, 4)
        pygame.draw.rect(surf, _LABEL_BG, padded_rect, border_radius=6)
        pygame.draw.rect(surf, _LABEL_BORDER, padded_rect, width=1, border_radius=6)
        surf.blit(text_surf, text_rect)

    return pygame.surfarray.array3d(surf).swapaxes(0, 1)


def _capture_frame(
    env,
    lead_id: str,
    heading_up: bool,
    agent_ids: list[str],
) -> Optional[np.ndarray]:
    """Capture a top-down frame and overlay platoon vehicle labels."""
    agents = getattr(env, "agents", {}) or {}
    vehicle = agents.get(lead_id)
    if vehicle is None:
        return None
    cam_pos = (float(vehicle.position[0]), float(vehicle.position[1]))
    renderer = getattr(env, "top_down_renderer", None)
    if renderer is not None:
        renderer.position = cam_pos

    try:
        if renderer is None:
            frame = env.render(**_build_render_kwargs(cam_pos, heading_up))
            # sync again after first render initialises the renderer
            renderer2 = getattr(env, "top_down_renderer", None)
            if renderer2 is not None:
                renderer2.position = cam_pos
        else:
            frame = env.render(**_build_render_kwargs(heading_up=heading_up))
        if frame is None:
            return None
        frame = np.asarray(frame)
    except Exception:
        return None

    return _overlay_platoon_labels(frame, env, agent_ids)


def _write_video(path: Path, frames: list, fps: int) -> None:
    if not frames:
        return
    import mediapy
    path.parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(str(path), frames, fps=int(fps))


def _pick_local_route(scenario_id: str) -> str:
    from scenarios.definitions import SCENARIO_BY_ID
    defn = SCENARIO_BY_ID.get(scenario_id)
    if defn is None:
        raise ValueError(f"Unknown scenario: {scenario_id}")
    routes = defn.allowed_local_routes
    if not routes:
        raise ValueError(f"Scenario {scenario_id} has no allowed_local_routes")
    return routes[0]


def _make_idm_policies(env, agent_ids: list[str], seed: int) -> dict:
    from metadrive.exp_dataset.expert_idm_policy import ExpertIDMPolicy
    policies = {}
    for i, agent_id in enumerate(agent_ids):
        vehicle = env.agents.get(agent_id)
        if vehicle is not None:
            policies[agent_id] = ExpertIDMPolicy(
                control_object=vehicle, random_seed=seed + i
            )
    return policies


def run_preview(
    scenario_id: str,
    local_route: Optional[str],
    num_agents: int,
    num_episodes: int,
    output_root: Path,
    heading_up: bool,
    traffic_density: float,
    start_seed: int,
    fps: int,
) -> Path:
    from envs.platoon_env import PlatoonEnv

    if local_route is None:
        local_route = _pick_local_route(scenario_id)

    video_dir = output_root / scenario_id / "reports" / "videos" / scenario_id
    video_dir.mkdir(parents=True, exist_ok=True)

    # use_render=False: matches collect_expert — top-down frames via env.render(mode="top_down")
    env = PlatoonEnv({
        "num_agents": num_agents,
        "use_render": False,
        "use_hybrid_map": True,
        "num_scenarios": 1,
        "traffic_density": traffic_density,
        "scenario_id": scenario_id,
        "local_route": local_route,
        "crash_done": False,
        "out_of_road_done": False,
        "horizon": 500,
    })

    agent_ids = [f"agent{i}" for i in range(num_agents)]
    lead_id = agent_ids[0]

    try:
        for ep_idx in range(num_episodes):
            obs = env.reset()
            policies = _make_idm_policies(env, agent_ids, seed=start_seed + ep_idx)
            frames: list[np.ndarray] = []

            for step in range(600):
                # Capture current state before env.step (same order as collect_expert)
                frame = _capture_frame(env, lead_id, heading_up, agent_ids)
                if frame is not None:
                    frames.append(frame)

                # Build actions for all active agents
                actions: dict[str, np.ndarray] = {}
                for agent_id in agent_ids:
                    vehicle = env.agents.get(agent_id)
                    if vehicle is None:
                        continue
                    policy = policies.get(agent_id)
                    if policy is None:
                        actions[agent_id] = np.zeros(2, dtype=np.float32)
                        continue
                    policy.control_object = vehicle
                    try:
                        act = policy.act()
                        actions[agent_id] = np.asarray(act, dtype=np.float32)
                    except Exception:
                        actions[agent_id] = np.zeros(2, dtype=np.float32)

                if not actions:
                    break

                obs, reward, terminated, truncated, info = env.low_level_step(actions)

                if terminated.get("__all__", False) or truncated.get("__all__", False):
                    break

            video_path = video_dir / f"episode_{ep_idx:04d}.mp4"
            _write_video(video_path, frames, fps)
            print(f"  episode {ep_idx + 1}/{num_episodes} -> {video_path} ({len(frames)} frames)")

    finally:
        try:
            env.close()
        except Exception:
            pass

    return video_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Platoon scenario top-down video preview")
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--local-route", default=None)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--heading-up", default="false")
    parser.add_argument("--traffic-density", type=float, default=0.10)
    parser.add_argument("--start-seed", type=int, default=59)
    parser.add_argument("--video-fps", type=int, default=10)
    args = parser.parse_args()

    video_dir = run_preview(
        scenario_id=args.scenario_id,
        local_route=args.local_route,
        num_agents=args.num_agents,
        num_episodes=args.num_episodes,
        output_root=Path(args.output_root),
        heading_up=args.heading_up.lower() in ("true", "1", "yes"),
        traffic_density=args.traffic_density,
        start_seed=args.start_seed,
        fps=args.video_fps,
    )
    print(f"\n=== Platoon videos saved to: {video_dir} ===")


if __name__ == "__main__":
    main()
