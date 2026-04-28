"""preview_platoon.py

Run PlatoonEnv with a selectable expert policy and save top-down videos.
Used by scripts/preview_scenario.sh --platoon for manual scenario validation.

Supported experts:
  --expert idm  (default) ExpertIDMPolicy per vehicle
  --expert lqr            PlatoonLQRExpert (LQR longitudinal control, steer=0)

Rendering strategy (matches collect_expert.py):
  - use_render=False — no 3D Panda3D window
  - env.render(mode="top_down", window=False, ...) for frame capture
  - per-vehicle labels (ego1, ego2, ...) overlaid on each frame
  - videos written via mediapy

Usage:
    # single scenario
    python -m metadrive.exp_dataset.preview_platoon \\
        --scenario-id S5_hard_brake_lead --local-route R3_mainline_straight \\
        --num-agents 3 --num-episodes 2 --expert lqr \\
        --output-root /tmp/platoon_preview

    # all 11 scenarios
    python -m metadrive.exp_dataset.preview_platoon \\
        --all-scenarios --num-agents 3 --num-episodes 2 --expert lqr \\
        --output-root /tmp/platoon_preview
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional

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


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

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
    """Draw ego1/ego2/... labels at each platoon vehicle's screen position."""
    renderer = getattr(env, "top_down_renderer", None)
    if renderer is None:
        return frame

    agents = getattr(env, "agents", {}) or {}
    screen_w, screen_h = frame.shape[1], frame.shape[0]

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
        if sx < -40 or sx > screen_w + 40 or sy < -40 or sy > screen_h + 40:
            continue

        label = f"ego{i + 1}"
        text_surf = font.render(label, True, _LABEL_FG, _LABEL_BG)
        text_rect = text_surf.get_rect()
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
    """Capture top-down frame and overlay per-vehicle labels."""
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


# ---------------------------------------------------------------------------
# Expert factories
# ---------------------------------------------------------------------------

def _make_idm_policies(env, agent_ids: list[str], seed: int) -> dict[str, object]:
    """Create per-vehicle IDM policies for one episode."""
    from metadrive.exp_dataset.expert_idm_policy import ExpertIDMPolicy
    policies = {}
    for i, agent_id in enumerate(agent_ids):
        vehicle = env.agents.get(agent_id)
        if vehicle is not None:
            policies[agent_id] = ExpertIDMPolicy(
                control_object=vehicle, random_seed=seed + i
            )
    return policies


def _make_idm_action_fn(env, agent_ids: list[str], seed: int) -> Callable:
    """IDM expert: creates per-vehicle policies, returns per-step action function."""
    policies = _make_idm_policies(env, agent_ids, seed)

    def action_fn(env) -> dict[str, np.ndarray]:
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
        return actions

    return action_fn


def _make_lqr_action_fn(lqr_config: dict | None = None) -> Callable:
    """LQR expert: stateless, returns a fixed per-step action function."""
    from experts.platoon_lqr_expert import PlatoonLQRConfig, PlatoonLQRExpert
    cfg = PlatoonLQRConfig(**(lqr_config or {}))
    expert = PlatoonLQRExpert(cfg)

    def action_fn(env) -> dict[str, np.ndarray]:
        return expert.compute_actions(env)

    return action_fn


def _build_action_fn_factory(
    expert_type: str,
    lqr_config: dict | None = None,
) -> Callable:
    """
    Return a factory: factory(env, agent_ids, seed) -> action_fn(env) -> actions.

    IDM: reinitialised each episode (binds to specific vehicle objects).
    LQR: stateless; same action_fn reused across all episodes.
    """
    if expert_type == "lqr":
        action_fn = _make_lqr_action_fn(lqr_config)
        # LQR is stateless — same fn for every episode / seed
        return lambda env, agent_ids, seed: action_fn
    else:
        return _make_idm_action_fn


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _episode_failed_immediately(frames, terminated, truncated, info) -> bool:
    if len(frames) > 1:
        return False
    if terminated is None and truncated is None:
        return False
    if truncated is not None and bool(truncated.get("__all__", False)):
        return False
    if terminated is not None and bool(terminated.get("__all__", False)):
        for agent_info in (info or {}).values():
            if bool(agent_info.get("crash", False) or agent_info.get("crash_vehicle", False)):
                return True
    return False


def _run_single_episode(env, agent_ids, lead_id, heading_up, seed, action_fn_factory):
    spawn_manager = getattr(getattr(env, "engine", None), "spawn_manager", None)
    if spawn_manager is not None and hasattr(spawn_manager, "set_episode_spawn_seed"):
        spawn_manager.set_episode_spawn_seed(int(seed))
    obs = env.reset()
    action_fn = action_fn_factory(env, agent_ids, seed)
    frames: list[np.ndarray] = []
    terminated = truncated = info = None

    for _step in range(600):
        frame = _capture_frame(env, lead_id, heading_up, agent_ids)
        if frame is not None:
            frames.append(frame)

        actions = action_fn(env)
        if not actions:
            break

        obs, reward, terminated, truncated, info = env.low_level_step(actions)
        if terminated.get("__all__", False) or truncated.get("__all__", False):
            break

    return frames, terminated, truncated, info


# ---------------------------------------------------------------------------
# Route / scenario helpers
# ---------------------------------------------------------------------------

def _pick_local_route(scenario_id: str) -> str:
    from scenarios.definitions import SCENARIO_BY_ID
    defn = SCENARIO_BY_ID.get(scenario_id)
    if defn is None:
        raise ValueError(f"Unknown scenario: {scenario_id}")
    routes = defn.allowed_local_routes
    if not routes:
        raise ValueError(f"Scenario {scenario_id} has no allowed_local_routes")
    return routes[0]


def _all_scenario_route_pairs() -> list[tuple[str, str]]:
    """Return [(scenario_id, first_route), ...] for all defined scenarios, sorted."""
    from scenarios.definitions import SCENARIO_BY_ID
    pairs = []
    for sid, defn in sorted(SCENARIO_BY_ID.items()):
        routes = defn.allowed_local_routes
        if routes:
            pairs.append((sid, routes[0]))
    return pairs


# ---------------------------------------------------------------------------
# Main preview runner
# ---------------------------------------------------------------------------

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
    expert_type: str = "idm",
    lqr_config: Optional[dict] = None,
    max_episode_retries: int = 3,
    env_factory: Optional[Callable] = None,
) -> Path:
    from envs.platoon_env import PlatoonEnv

    if local_route is None:
        local_route = _pick_local_route(scenario_id)

    video_dir = output_root / scenario_id / "reports" / "videos" / scenario_id
    video_dir.mkdir(parents=True, exist_ok=True)

    env_config = {
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
    }
    env = env_factory(env_config) if env_factory is not None else PlatoonEnv(env_config)
    agent_ids = [f"agent{i}" for i in range(num_agents)]
    lead_id = agent_ids[0]
    action_fn_factory = _build_action_fn_factory(expert_type, lqr_config)

    try:
        for ep_idx in range(num_episodes):
            frames: list[np.ndarray] = []
            used_seed = start_seed + ep_idx
            for retry in range(max(1, int(max_episode_retries))):
                episode_seed = used_seed + retry
                frames, terminated, truncated, info = _run_single_episode(
                    env, agent_ids, lead_id, heading_up, episode_seed, action_fn_factory
                )
                if not _episode_failed_immediately(frames, terminated, truncated, info):
                    used_seed = episode_seed
                    break
                print(f"  retry ep {ep_idx + 1} seed={episode_seed + 1} (frames={len(frames)})")

            video_path = video_dir / f"episode_{ep_idx:04d}.mp4"
            _write_video(video_path, frames, fps)
            print(
                f"  [{scenario_id}] ep {ep_idx + 1}/{num_episodes} -> {video_path.name} "
                f"({len(frames)} frames, seed={used_seed})"
            )
    finally:
        try:
            env.close()
        except Exception:
            pass

    return video_dir


def run_all_scenarios(
    num_agents: int,
    num_episodes: int,
    output_root: Path,
    heading_up: bool,
    traffic_density: float,
    start_seed: int,
    fps: int,
    expert_type: str = "idm",
    lqr_config: Optional[dict] = None,
) -> None:
    """Evaluate all defined scenarios (one env per scenario to avoid map conflicts)."""
    pairs = _all_scenario_route_pairs()
    print(f"\n=== Evaluating {len(pairs)} scenarios with expert={expert_type} ===\n")

    results: list[tuple[str, str, str]] = []  # (scenario_id, route, status)
    for scenario_id, route in pairs:
        print(f"--- {scenario_id} / {route} ---")
        try:
            video_dir = run_preview(
                scenario_id=scenario_id,
                local_route=route,
                num_agents=num_agents,
                num_episodes=num_episodes,
                output_root=output_root,
                heading_up=heading_up,
                traffic_density=traffic_density,
                start_seed=start_seed,
                fps=fps,
                expert_type=expert_type,
                lqr_config=lqr_config,
            )
            results.append((scenario_id, route, f"OK  -> {video_dir}"))
        except Exception as exc:
            results.append((scenario_id, route, f"FAIL: {exc}"))
            print(f"  ERROR: {exc}")

    print("\n" + "=" * 70)
    print(f"{'SCENARIO':<40} {'ROUTE':<30} STATUS")
    print("=" * 70)
    for sid, route, status in results:
        print(f"{sid:<40} {route:<30} {status}")
    print("=" * 70)
    ok = sum(1 for _, _, s in results if s.startswith("OK"))
    print(f"\n{ok}/{len(results)} scenarios completed successfully.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Platoon scenario top-down video preview")
    parser.add_argument("--scenario-id", default=None,
                        help="Scenario ID (ignored when --all-scenarios is set)")
    parser.add_argument("--local-route", default=None)
    parser.add_argument("--all-scenarios", action="store_true",
                        help="Run all defined scenarios sequentially")
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--heading-up", default="false")
    parser.add_argument("--traffic-density", type=float, default=0.10)
    parser.add_argument("--start-seed", type=int, default=59)
    parser.add_argument("--video-fps", type=int, default=10)
    # Expert selection
    parser.add_argument("--expert", default="idm", choices=["idm", "lqr"],
                        help="Policy used to drive platoon vehicles (default: idm)")
    # LQR tuning (only used when --expert lqr)
    parser.add_argument("--lqr-desired-speed-kmh", type=float, default=30.0)
    parser.add_argument("--lqr-headway-time", type=float, default=0.5)
    parser.add_argument("--lqr-standstill-gap", type=float, default=4.0)
    parser.add_argument("--lqr-q-gap", type=float, default=1.0)
    parser.add_argument("--lqr-q-rel-vel", type=float, default=2.0)
    parser.add_argument("--lqr-q-speed", type=float, default=0.5)
    parser.add_argument("--lqr-r-accel", type=float, default=0.5)
    args = parser.parse_args()

    heading_up = args.heading_up.lower() in ("true", "1", "yes")
    output_root = Path(args.output_root)

    lqr_config = None
    if args.expert == "lqr":
        lqr_config = {
            "desired_speed_km_h": args.lqr_desired_speed_kmh,
            "headway_time_s": args.lqr_headway_time,
            "standstill_gap_m": args.lqr_standstill_gap,
            "q_gap": args.lqr_q_gap,
            "q_rel_vel": args.lqr_q_rel_vel,
            "q_speed": args.lqr_q_speed,
            "r_accel": args.lqr_r_accel,
        }

    if args.all_scenarios:
        run_all_scenarios(
            num_agents=args.num_agents,
            num_episodes=args.num_episodes,
            output_root=output_root,
            heading_up=heading_up,
            traffic_density=args.traffic_density,
            start_seed=args.start_seed,
            fps=args.video_fps,
            expert_type=args.expert,
            lqr_config=lqr_config,
        )
    else:
        if not args.scenario_id:
            parser.error("--scenario-id is required when --all-scenarios is not set")
        video_dir = run_preview(
            scenario_id=args.scenario_id,
            local_route=args.local_route,
            num_agents=args.num_agents,
            num_episodes=args.num_episodes,
            output_root=output_root,
            heading_up=heading_up,
            traffic_density=args.traffic_density,
            start_seed=args.start_seed,
            fps=args.video_fps,
            expert_type=args.expert,
            lqr_config=lqr_config,
        )
        print(f"\n=== Videos saved to: {video_dir} ===")


if __name__ == "__main__":
    main()
