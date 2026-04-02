from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metadrive.exp_dataset.expert_idm_policy import ExpertIDMConfig, ExpertIDMPolicy

DEFAULT_OUTPUT_DIR = "/media/kong/Elements_SE/Diffusion_Data/outputs/expert"
DEFAULT_VIDEO_FPS = 10
DEFAULT_TOPDOWN_CAMERA_HEIGHT = 180.0
DEFAULT_TOPDOWN_SCREEN_SIZE = 800
DEFAULT_TOPDOWN_FILM_SIZE = 3000
DEFAULT_TEXT_CORNER = "top_left"


@dataclass
class EpisodeMetrics:
    route_completion: float
    arrive_dest: bool
    crash: bool
    out_of_road: bool
    lane_change_count: float
    min_ttc: float
    avg_speed_km_h: float
    avg_abs_jerk: float
    max_abs_jerk: float
    episode_length: int
    episode_reward: float


@dataclass
class EvalSummary:
    n_episodes: int
    success_rate: float
    collision_rate: float
    out_of_road_rate: float
    mean_lane_change_count: float
    mean_route_completion: float
    mean_min_ttc: float
    mean_avg_speed_km_h: float
    mean_avg_abs_jerk: float
    mean_max_abs_jerk: float
    per_episode: list[EpisodeMetrics]


def build_run_output_dir(output_root: str, expert_name: str, seed: int, timestamp: str | None = None) -> str:
    resolved_timestamp = timestamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_root, f"{expert_name}_seed{seed}_{resolved_timestamp}")


def build_episode_video_path(run_dir: str, episode_index: int) -> str:
    return os.path.join(run_dir, f"episode_{episode_index:03d}.mp4")


def build_summary_json_path(run_dir: str) -> str:
    return os.path.join(run_dir, "summary.json")


def build_base_env_config(
    seed: int,
    n_episodes: int,
    topdown_camera_height: float = DEFAULT_TOPDOWN_CAMERA_HEIGHT,
    env_config: dict | None = None,
) -> dict[str, object]:
    merged_config: dict[str, object] = {
        "use_render": False,
        "start_seed": int(seed),
        "num_scenarios": max(int(n_episodes), 1),
        "prefer_track_agent": "agent0",
        "top_down_camera_initial_z": float(topdown_camera_height),
    }
    if env_config:
        merged_config.update(env_config)
    return merged_config


def build_topdown_render_kwargs(
    expert_name: str,
    episode_index: int,
    step_count: int,
    screen_size: int = DEFAULT_TOPDOWN_SCREEN_SIZE,
    film_size: int = DEFAULT_TOPDOWN_FILM_SIZE,
    camera_position: tuple[float, float] | None = None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "mode": "top_down",
        "window": False,
        "screen_size": (int(screen_size), int(screen_size)),
        "film_size": (int(film_size), int(film_size)),
        "target_agent_heading_up": False,
    }
    if camera_position is not None:
        kwargs["camera_position"] = camera_position
    return kwargs


def compute_heading_up_rotation_deg(heading_rad: float) -> float:
    return float(-np.rad2deg(float(heading_rad)) + 90.0)


def build_overlay_lines(expert_name: str, episode_index: int, step_count: int) -> list[str]:
    return [
        f"expert: {expert_name}",
        f"episode: {episode_index}",
        f"step: {step_count}",
    ]


def compute_ttc(ego_speed_ms: float, front_speed_ms: float, gap: float) -> float:
    closing = float(ego_speed_ms) - float(front_speed_ms)
    if closing <= 0.0 or gap <= 0.0:
        return float("inf")
    return float(gap) / closing


def compute_jerk_stats(speed_ms_list: Iterable[float], dt: float) -> tuple[float, float]:
    speeds = np.asarray(list(speed_ms_list), dtype=np.float32)
    if speeds.size < 3 or dt <= 0.0:
        return 0.0, 0.0
    accel = np.diff(speeds) / dt
    jerk = np.diff(accel) / dt
    if jerk.size == 0:
        return 0.0, 0.0
    jerk_abs = np.abs(jerk)
    return float(np.mean(jerk_abs)), float(np.max(jerk_abs))


def aggregate_episode_metrics(agent_metrics: list[EpisodeMetrics]) -> EpisodeMetrics:
    if not agent_metrics:
        return EpisodeMetrics(
            route_completion=0.0,
            arrive_dest=False,
            crash=False,
            out_of_road=False,
            lane_change_count=0.0,
            min_ttc=float("inf"),
            avg_speed_km_h=0.0,
            avg_abs_jerk=0.0,
            max_abs_jerk=0.0,
            episode_length=0,
            episode_reward=0.0,
        )

    return EpisodeMetrics(
        route_completion=float(np.mean([m.route_completion for m in agent_metrics])),
        arrive_dest=all(m.arrive_dest for m in agent_metrics),
        crash=any(m.crash for m in agent_metrics),
        out_of_road=any(m.out_of_road for m in agent_metrics),
        lane_change_count=float(np.mean([m.lane_change_count for m in agent_metrics])),
        min_ttc=float(np.mean([m.min_ttc for m in agent_metrics])),
        avg_speed_km_h=float(np.mean([m.avg_speed_km_h for m in agent_metrics])),
        avg_abs_jerk=float(np.mean([m.avg_abs_jerk for m in agent_metrics])),
        max_abs_jerk=float(np.mean([m.max_abs_jerk for m in agent_metrics])),
        episode_length=int(round(float(np.mean([m.episode_length for m in agent_metrics])))),
        episode_reward=float(np.mean([m.episode_reward for m in agent_metrics])),
    )


def summarize_metrics(per_episode: list[EpisodeMetrics]) -> EvalSummary:
    if not per_episode:
        return EvalSummary(
            n_episodes=0,
            success_rate=0.0,
            collision_rate=0.0,
            out_of_road_rate=0.0,
            mean_lane_change_count=0.0,
            mean_route_completion=0.0,
            mean_min_ttc=float("inf"),
            mean_avg_speed_km_h=0.0,
            mean_avg_abs_jerk=0.0,
            mean_max_abs_jerk=0.0,
            per_episode=[],
        )

    return EvalSummary(
        n_episodes=len(per_episode),
        success_rate=float(np.mean([m.arrive_dest for m in per_episode])),
        collision_rate=float(np.mean([m.crash for m in per_episode])),
        out_of_road_rate=float(np.mean([m.out_of_road for m in per_episode])),
        mean_lane_change_count=float(np.mean([m.lane_change_count for m in per_episode])),
        mean_route_completion=float(np.mean([m.route_completion for m in per_episode])),
        mean_min_ttc=float(np.mean([m.min_ttc for m in per_episode])),
        mean_avg_speed_km_h=float(np.mean([m.avg_speed_km_h for m in per_episode])),
        mean_avg_abs_jerk=float(np.mean([m.avg_abs_jerk for m in per_episode])),
        mean_max_abs_jerk=float(np.mean([m.max_abs_jerk for m in per_episode])),
        per_episode=per_episode,
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _safe_lane_ordinal(lane_index) -> int | None:
    if isinstance(lane_index, (tuple, list)) and lane_index:
        candidate = lane_index[-1]
    else:
        candidate = lane_index
    if isinstance(candidate, (int, np.integer)):
        return int(candidate)
    return None


def get_lane_ordinal(vehicle) -> int | None:
    return _safe_lane_ordinal(getattr(vehicle, "lane_index", None))


def _select_reference_lane(vehicle):
    navigation = getattr(vehicle, "navigation", None)
    current_ref_lanes = getattr(navigation, "current_ref_lanes", None)
    if current_ref_lanes:
        lane_idx = _safe_lane_ordinal(getattr(vehicle, "lane_index", None))
        if lane_idx is not None and 0 <= lane_idx < len(current_ref_lanes):
            return current_ref_lanes[lane_idx]
        lane = getattr(vehicle, "lane", None)
        if lane in current_ref_lanes:
            return lane
        return current_ref_lanes[0]
    return getattr(vehicle, "lane", None)


def get_front_vehicle_state(vehicle) -> tuple[float, float]:
    try:
        from metadrive.policy.idm_policy import FrontBackObjects, IDMPolicy
    except Exception:
        return float("inf"), 0.0

    if not hasattr(vehicle, "lidar"):
        return float("inf"), 0.0
    ref_lane = _select_reference_lane(vehicle)
    if ref_lane is None:
        return float("inf"), 0.0

    try:
        current_ref_lanes = getattr(vehicle.navigation, "current_ref_lanes", None)
        surrounding_objects = FrontBackObjects.get_find_front_back_objs(
            vehicle.lidar.get_surrounding_objects(vehicle),
            ref_lane,
            vehicle.position,
            max_distance=IDMPolicy.MAX_LONG_DIST,
            ref_lanes=current_ref_lanes if current_ref_lanes and ref_lane in current_ref_lanes else None,
        )
        front_object = surrounding_objects.front_object()
        if front_object is None:
            return float("inf"), 0.0
        front_distance = float(surrounding_objects.front_min_distance())
        front_speed = getattr(front_object, "speed", None)
        if front_speed is None:
            speed_km_h = float(getattr(front_object, "speed_km_h", 0.0))
            front_speed = speed_km_h / 3.6
        return front_distance, max(float(front_speed), 0.0)
    except Exception:
        return float("inf"), 0.0


def _get_env_dt(env) -> float:
    config = getattr(env, "config", {}) or {}
    physics_dt = float(config.get("physics_world_step_size", 0.02))
    decision_repeat = int(config.get("decision_repeat", 5))
    return physics_dt * decision_repeat


def _build_agent_metric(
    step_count: int,
    episode_reward: float,
    lane_change_count: int,
    agent_info: dict,
    speed_ms: list[float],
    speed_km_h: list[float],
    ttc_list: list[float],
) -> EpisodeMetrics:
    avg_abs_jerk, max_abs_jerk = compute_jerk_stats(speed_ms, dt=max(agent_info.get("_dt", 0.1), 1e-6))
    return EpisodeMetrics(
        route_completion=float(agent_info.get("route_completion", 0.0)),
        arrive_dest=bool(agent_info.get("arrive_dest", False)),
        crash=bool(agent_info.get("crash", False)),
        out_of_road=bool(agent_info.get("out_of_road", False)),
        lane_change_count=float(lane_change_count),
        min_ttc=float(min(ttc_list)) if ttc_list else float("inf"),
        avg_speed_km_h=float(np.mean(speed_km_h)) if speed_km_h else 0.0,
        avg_abs_jerk=avg_abs_jerk,
        max_abs_jerk=max_abs_jerk,
        episode_length=int(step_count),
        episode_reward=float(episode_reward),
    )


def get_primary_agent_id(env) -> str | None:
    agents = getattr(env, "agents", {}) or {}
    if "agent0" in agents:
        return "agent0"
    return next(iter(agents.keys()), None)


def sync_topdown_camera_with_agent(env, agent_id: str | None) -> tuple[float, float] | None:
    if agent_id is None:
        return None
    agents = getattr(env, "agents", {}) or {}
    vehicle = agents.get(agent_id)
    if vehicle is None:
        return None
    camera_position = (float(vehicle.position[0]), float(vehicle.position[1]))
    renderer = getattr(env, "top_down_renderer", None)
    if renderer is not None:
        renderer.position = camera_position
    return camera_position


def rotate_frame_heading_up(frame_array: np.ndarray, heading_rad: float) -> np.ndarray:
    import pygame

    height, width = frame_array.shape[:2]
    rotation_deg = compute_heading_up_rotation_deg(heading_rad)
    source = pygame.surfarray.make_surface(frame_array.swapaxes(0, 1))
    canvas_size = max(width, height) * 2
    canvas = pygame.Surface((canvas_size, canvas_size))
    canvas.fill((255, 255, 255))
    canvas.blit(source, ((canvas_size - width) // 2, (canvas_size - height) // 2))
    rotated = pygame.transform.rotozoom(canvas, rotation_deg, 1.0)
    crop_x = max(rotated.get_width() // 2 - width // 2, 0)
    crop_y = max(rotated.get_height() // 2 - height // 2, 0)
    cropped = pygame.Surface((width, height))
    cropped.fill((255, 255, 255))
    cropped.blit(rotated, (0, 0), (crop_x, crop_y, width, height))
    return pygame.surfarray.array3d(cropped).swapaxes(0, 1)


def overlay_text_on_frame(frame_array: np.ndarray, lines: list[str], corner: str = DEFAULT_TEXT_CORNER) -> np.ndarray:
    import pygame

    if not lines:
        return frame_array
    height, width = frame_array.shape[:2]
    surface = pygame.surfarray.make_surface(frame_array.swapaxes(0, 1))
    if not pygame.font.get_init():
        pygame.font.init()
    font = pygame.font.SysFont("Arial", 24)
    rendered = [font.render(line, True, (0, 0, 0)) for line in lines]
    max_width = max(text.get_width() for text in rendered)
    line_height = max(text.get_height() for text in rendered)
    padding = 14
    total_height = line_height * len(rendered)
    if corner == "top_right":
        x = max(width - max_width - padding, padding)
    else:
        x = padding
    y = padding
    for idx, text_surface in enumerate(rendered):
        surface.blit(text_surface, (x, y + idx * line_height))
    return pygame.surfarray.array3d(surface).swapaxes(0, 1)


def _capture_topdown_frame(env, expert_name: str, episode_index: int, step_count: int) -> np.ndarray:
    primary_agent_id = get_primary_agent_id(env)
    camera_position = sync_topdown_camera_with_agent(env, primary_agent_id)
    if getattr(env, "top_down_renderer", None) is None:
        frame = env.render(
            **build_topdown_render_kwargs(
                expert_name,
                episode_index,
                step_count,
                camera_position=camera_position,
            )
        )
        sync_topdown_camera_with_agent(env, primary_agent_id)
    else:
        frame = env.render(**build_topdown_render_kwargs(expert_name, episode_index, step_count))
    frame_array = np.asarray(frame)
    if frame_array.ndim >= 2:
        frame_array = frame_array.swapaxes(0, 1)
    agents = getattr(env, "agents", {}) or {}
    primary_agent = agents.get(primary_agent_id) if primary_agent_id is not None else None
    if primary_agent is not None:
        frame_array = rotate_frame_heading_up(frame_array, float(primary_agent.heading_theta))
    frame_array = overlay_text_on_frame(frame_array, build_overlay_lines(expert_name, episode_index, step_count))
    return frame_array


def _write_video(video_path: str, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    import mediapy

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    mediapy.write_video(video_path, frames, fps=fps)


def build_expert_policy(vehicle, random_seed: int, idm_config: ExpertIDMConfig | None = None):
    return ExpertIDMPolicy(
        control_object=vehicle,
        random_seed=random_seed,
        idm_config=idm_config,
    )


def run_evaluation(
    n_episodes: int = 10,
    env_config: dict | None = None,
    render: bool = False,
    seed: int = 0,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    topdown_camera_height: float = DEFAULT_TOPDOWN_CAMERA_HEIGHT,
    expert_idm_config: ExpertIDMConfig | None = None,
) -> EvalSummary:
    from metadrive.envs.diffusion_envs.base_multi_env import BaseMultiEnv

    expert_name = "idm"

    run_dir = build_run_output_dir(output_dir, expert_name=expert_name, seed=seed)
    os.makedirs(run_dir, exist_ok=True)

    merged_config = build_base_env_config(
        seed=seed,
        n_episodes=n_episodes,
        topdown_camera_height=topdown_camera_height,
        env_config=env_config,
    )
    env = BaseMultiEnv(config=merged_config)

    def get_action(agent_id, vehicle, idm_cache):
        if agent_id not in idm_cache:
            idm_cache[agent_id] = build_expert_policy(
                vehicle,
                random_seed=seed,
                idm_config=expert_idm_config,
            )
        return idm_cache[agent_id].act()

    per_episode: list[EpisodeMetrics] = []
    idm_cache: dict[str, object] = {}

    try:
        env.reset()
        dt = _get_env_dt(env)
        episode_count = 0
        step_count = 0
        episode_rewards: dict[str, float] = {}
        speed_ms: dict[str, list[float]] = {}
        speed_km_h: dict[str, list[float]] = {}
        ttc_history: dict[str, list[float]] = {}
        lane_change_counts: dict[str, int] = {}
        previous_lane_ordinals: dict[str, int | None] = {}
        episode_video_frames: list[np.ndarray] = []

        def reset_buffers():
            episode_rewards.clear()
            speed_ms.clear()
            speed_km_h.clear()
            ttc_history.clear()
            lane_change_counts.clear()
            previous_lane_ordinals.clear()
            episode_video_frames.clear()
            for agent_id in env.agents.keys():
                episode_rewards[agent_id] = 0.0
                speed_ms[agent_id] = []
                speed_km_h[agent_id] = []
                ttc_history[agent_id] = []
                lane_change_counts[agent_id] = 0
                previous_lane_ordinals[agent_id] = get_lane_ordinal(env.agents[agent_id])

        reset_buffers()

        while episode_count < n_episodes:
            current_vehicles = {agent_id: env.agents[agent_id] for agent_id in list(env.agents.keys())}
            actions = {
                agent_id: get_action(agent_id, vehicle, idm_cache)
                for agent_id, vehicle in current_vehicles.items()
            }
            _, reward, terminated, truncated, info = env.step(actions)
            step_count += 1

            for agent_id, vehicle in current_vehicles.items():
                speed_ms.setdefault(agent_id, []).append(float(getattr(vehicle, "speed", 0.0)))
                speed_km_h.setdefault(agent_id, []).append(float(getattr(vehicle, "speed_km_h", 0.0)))
                front_dist, front_speed = get_front_vehicle_state(vehicle)
                ttc_history.setdefault(agent_id, []).append(compute_ttc(vehicle.speed, front_speed, front_dist))
                episode_rewards[agent_id] = episode_rewards.get(agent_id, 0.0) + float(reward.get(agent_id, 0.0))
                current_lane_ordinal = get_lane_ordinal(vehicle)
                previous_lane_ordinal = previous_lane_ordinals.get(agent_id)
                if previous_lane_ordinal is not None and current_lane_ordinal is not None and current_lane_ordinal != previous_lane_ordinal:
                    lane_change_counts[agent_id] = lane_change_counts.get(agent_id, 0) + 1
                previous_lane_ordinals[agent_id] = current_lane_ordinal

            if render:
                episode_video_frames.append(
                    _capture_topdown_frame(
                        env,
                        expert_name=expert_name,
                        episode_index=episode_count + 1,
                        step_count=step_count,
                    )
                )

            done = bool(terminated.get("__all__", False) or truncated.get("__all__", False))
            if not done:
                continue

            agent_metrics: list[EpisodeMetrics] = []
            agent_ids = list(episode_rewards.keys())
            for agent_id in agent_ids:
                agent_info = dict(info.get(agent_id, {}))
                agent_info["_dt"] = dt
                agent_metrics.append(
                    _build_agent_metric(
                        step_count=step_count,
                        episode_reward=episode_rewards.get(agent_id, 0.0),
                        lane_change_count=lane_change_counts.get(agent_id, 0),
                        agent_info=agent_info,
                        speed_ms=speed_ms.get(agent_id, []),
                        speed_km_h=speed_km_h.get(agent_id, []),
                        ttc_list=ttc_history.get(agent_id, []),
                    )
                )

            aggregated = aggregate_episode_metrics(agent_metrics)
            per_episode.append(aggregated)
            video_path = build_episode_video_path(run_dir, episode_count + 1)
            if render:
                _write_video(video_path, episode_video_frames, fps=DEFAULT_VIDEO_FPS)
            print(
                f"[{expert_name}] ep={episode_count + 1:2d} "
                f"steps={aggregated.episode_length:4d} "
                f"reward={aggregated.episode_reward:8.2f} "
                f"route_completion={aggregated.route_completion:6.3f} "
                f"arrive={aggregated.arrive_dest} "
                f"crash={aggregated.crash} "
                f"oor={aggregated.out_of_road} "
                f"lane_changes={aggregated.lane_change_count:5.1f} "
                f"min_ttc={aggregated.min_ttc:6.2f} "
                f"video={video_path}"
            )

            episode_count += 1
            step_count = 0
            idm_cache.clear()
            if episode_count >= n_episodes:
                break
            env.reset()
            dt = _get_env_dt(env)
            reset_buffers()
    finally:
        env.close()

    summary = summarize_metrics(per_episode)
    save_json(summary, build_summary_json_path(run_dir))
    return summary


def print_summary(summary: EvalSummary) -> None:
    print("\nEvaluation Summary")
    print("=" * 72)
    print(
        f"Episodes={summary.n_episodes}  "
        f"Success={summary.success_rate:.3f}  "
        f"Collision={summary.collision_rate:.3f}  "
        f"OutOfRoad={summary.out_of_road_rate:.3f}"
    )
    print(
        f"LaneChanges={summary.mean_lane_change_count:.3f}  "
        f"RouteCompletion={summary.mean_route_completion:.3f}  "
        f"MeanMinTTC={summary.mean_min_ttc:.3f}  "
        f"AvgSpeed(km/h)={summary.mean_avg_speed_km_h:.3f}"
    )
    print(
        f"AvgAbsJerk={summary.mean_avg_abs_jerk:.3f}  "
        f"MaxAbsJerk={summary.mean_max_abs_jerk:.3f}"
    )
    if not summary.per_episode:
        return

    print("-" * 72)
    print("ep  route   arrive crash oor lanechg  min_ttc  avg_speed  avg_jerk  max_jerk")
    for idx, episode in enumerate(summary.per_episode, start=1):
        print(
            f"{idx:>2d}  "
            f"{episode.route_completion:>5.3f}   "
            f"{str(episode.arrive_dest):>6s} "
            f"{str(episode.crash):>5s} "
            f"{str(episode.out_of_road):>3s} "
            f"{episode.lane_change_count:>7.1f}   "
            f"{episode.min_ttc:>7.3f}   "
            f"{episode.avg_speed_km_h:>8.3f}  "
            f"{episode.avg_abs_jerk:>8.3f}  "
            f"{episode.max_abs_jerk:>8.3f}"
        )


def save_json(summary: EvalSummary, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(asdict(summary)), f, ensure_ascii=False, indent=2)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate single-expert driving performance in MetaDrive.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--num-agents", type=int, default=1)
    parser.add_argument("--traffic-density", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=88)
    parser.add_argument("--topdown-camera-height", type=float, default=DEFAULT_TOPDOWN_CAMERA_HEIGHT)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory for per-episode videos and summary JSON.")
    parser.add_argument("--output", type=str, default=None, help="Optional extra JSON output path.")
    parser.add_argument("--expert-idm-distance-wanted", type=float, default=ExpertIDMConfig.distance_wanted)
    parser.add_argument("--expert-idm-time-wanted", type=float, default=ExpertIDMConfig.time_wanted)
    parser.add_argument("--expert-idm-delta", type=float, default=ExpertIDMConfig.delta)
    parser.add_argument("--expert-idm-acc-factor", type=float, default=ExpertIDMConfig.acc_factor)
    parser.add_argument("--expert-idm-deacc-factor", type=float, default=ExpertIDMConfig.deacc_factor)
    parser.add_argument("--expert-idm-normal-speed-kmh", type=float, default=ExpertIDMConfig.normal_speed_kmh)
    parser.add_argument("--expert-idm-max-speed-kmh", type=float, default=ExpertIDMConfig.max_speed_kmh)
    parser.add_argument("--expert-idm-enable-lane-change", type=int, choices=(0, 1), default=int(ExpertIDMConfig.enable_lane_change))
    parser.add_argument("--expert-idm-lane-change-freq", type=int, default=ExpertIDMConfig.lane_change_freq)
    parser.add_argument("--expert-idm-lane-change-speed-increase", type=float, default=ExpertIDMConfig.lane_change_speed_increase)
    parser.add_argument("--expert-idm-safe-lane-change-distance", type=float, default=ExpertIDMConfig.safe_lane_change_distance)
    parser.add_argument("--expert-idm-max-long-dist", type=float, default=ExpertIDMConfig.max_long_dist)
    parser.add_argument("--expert-idm-heading-pid-kp", type=float, default=ExpertIDMConfig.heading_pid_kp)
    parser.add_argument("--expert-idm-heading-pid-ki", type=float, default=ExpertIDMConfig.heading_pid_ki)
    parser.add_argument("--expert-idm-heading-pid-kd", type=float, default=ExpertIDMConfig.heading_pid_kd)
    parser.add_argument("--expert-idm-lateral-pid-kp", type=float, default=ExpertIDMConfig.lateral_pid_kp)
    parser.add_argument("--expert-idm-lateral-pid-ki", type=float, default=ExpertIDMConfig.lateral_pid_ki)
    parser.add_argument("--expert-idm-lateral-pid-kd", type=float, default=ExpertIDMConfig.lateral_pid_kd)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    expert_idm_config = ExpertIDMConfig(
        distance_wanted=float(args.expert_idm_distance_wanted),
        time_wanted=float(args.expert_idm_time_wanted),
        delta=float(args.expert_idm_delta),
        acc_factor=float(args.expert_idm_acc_factor),
        deacc_factor=float(args.expert_idm_deacc_factor),
        normal_speed_kmh=float(args.expert_idm_normal_speed_kmh),
        max_speed_kmh=float(args.expert_idm_max_speed_kmh),
        enable_lane_change=bool(args.expert_idm_enable_lane_change),
        lane_change_freq=int(args.expert_idm_lane_change_freq),
        lane_change_speed_increase=float(args.expert_idm_lane_change_speed_increase),
        safe_lane_change_distance=float(args.expert_idm_safe_lane_change_distance),
        max_long_dist=float(args.expert_idm_max_long_dist),
        heading_pid_kp=float(args.expert_idm_heading_pid_kp),
        heading_pid_ki=float(args.expert_idm_heading_pid_ki),
        heading_pid_kd=float(args.expert_idm_heading_pid_kd),
        lateral_pid_kp=float(args.expert_idm_lateral_pid_kp),
        lateral_pid_ki=float(args.expert_idm_lateral_pid_ki),
        lateral_pid_kd=float(args.expert_idm_lateral_pid_kd),
    )
    summary = run_evaluation(
        n_episodes=args.episodes,
        env_config={
            "num_agents": args.num_agents,
            "traffic_density": args.traffic_density,
        },
        render=True,  # render打开以保存视频，评估时默认开启
        seed=args.seed,
        output_dir=args.output_dir,
        topdown_camera_height=args.topdown_camera_height,
        expert_idm_config=expert_idm_config,
    )
    print_summary(summary)
    if args.output:
        save_json(summary, args.output)


if __name__ == "__main__":
    main()
