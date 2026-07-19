"""Collect three-vehicle rule-planner expert demonstrations.

The collector intentionally reuses the single-vehicle dataset schema and
helpers from :mod:`expert_dataset.collect_expert`.  A synchronized platoon
timestep is flattened in time-major order (agent0, agent1, agent2), while
``episode_id`` and ``joint_step_index`` retain the joint grouping.

Example:
    python -m expert_dataset.collect_multi_experts \
        --target-samples 40000 \
        --dataset-name platoon_rule_expert
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np
import yaml

from envs.platoon_env import PlatoonEnv
from evaluation.preview_and_evaluation import build_rule_planner_pipeline_factory
from expert_dataset.collect_expert import (
    EpisodeSpec,
    EpisodeSpecSampler,
    ExpertCollectorConfig,
    ShardWriter,
    _coerce_bool,
    _filter_samples,
    _prepare_samples_for_storage,
    _save_sample_visualizations,
    _to_jsonable,
    build_episode_rngs,
    build_episode_samples,
    build_frame,
    build_map_visualization_geometry,
    build_trajectory_filter_pipeline,
    build_trajectory_mode_summary,
    detect_existing_state,
    fast_forward_episode_rngs,
    format_eta,
    trajectory_mode_name,
    write_episode_video,
)
from expert_dataset.metadrive_dataset import split_shards


DEFAULT_SCENARIOS = {
    "S5_hard_brake_lead": 1.0,
    "S6_background_merge_in": 1.0,
}
PLATOON_WRAPPER_ONLY_ENV_KEYS = {
    "planner_device",
    "lookahead_index",
    "controller_type",
    "scenario_ids",
}
FAILURE_CRASH_KEYS = (
    "crash",
    "crash_vehicle",
    "crash_object",
    "crash_building",
    "crash_human",
)
FAILURE_ROAD_KEYS = ("out_of_road", "out_of_route")


@dataclass
class MultiExpertCollectorConfig(ExpertCollectorConfig):
    """Single-collector-compatible configuration with platoon additions."""

    dataset_name: str = "platoon_rule_expert"
    expert_type: str = "rule_planner"
    target_samples: int = 40000
    max_episode_steps: int = 0  # 0 means use env_config.horizon
    dataset_config_path: Path = Path("configs/dataset/data_collect.yaml")
    num_agents: int = 3
    decision_policy: str = "rule_maker"
    planning_policy: str = "lattice"
    control_policy: str = "adaptive"
    scenario_weights: Dict[str, float] = field(default_factory=dict)
    traffic_density_min: float = 0.0
    traffic_density_max: float = 0.0
    trajectory_correction_enabled: bool = False


@dataclass
class PlatoonEpisodeRollout:
    frames_by_agent: Dict[str, List[Dict[str, np.ndarray]]]
    video_frames: List[np.ndarray]
    failed: bool
    failure_reason: Optional[str]
    scenario_summary: Dict[str, object]


class RawObservationPlatoonEnv(PlatoonEnv):
    """PlatoonEnv variant exposing DatasetCollectObservation before encoding."""

    def _augment_observations(self, obs: Mapping[str, object]) -> dict:
        return {str(agent_id): dict(agent_obs) for agent_id, agent_obs in obs.items()}


def load_collection_env_config(config: MultiExpertCollectorConfig) -> dict:
    payload = dict(yaml.safe_load(config.dataset_config_path.read_text(encoding="utf-8")) or {})
    env_config = dict(payload.get("env_config") or {})
    if config.scenario_weights:
        env_config["scenario_ids"] = list(config.scenario_weights)
    else:
        scenario_ids = [str(value) for value in (env_config.get("scenario_ids") or [])]
        if not scenario_ids:
            raise ValueError("dataset config env_config.scenario_ids must not be empty")
        config.scenario_weights = {scenario_id: 1.0 for scenario_id in scenario_ids}
    env_config.update(
        {
            "num_agents": int(config.num_agents),
            "observation_mode": "multimodal",
            "use_render": False,
            # DatasetCollectEnv explicitly supports CUDA image collection only
            # for num_agents == 1.  Three-agent raw collection must stay on CPU.
            "image_on_cuda": False,
        }
    )
    return env_config


def _set_runtime_density(env: PlatoonEnv, density: float) -> None:
    env.config["traffic_density"] = float(density)
    env.platoon_config.traffic_density = float(density)
    global_config = getattr(getattr(env, "engine", None), "global_config", None)
    if global_config is not None:
        global_config["traffic_density"] = float(density)


def _scenario_summary(env: PlatoonEnv) -> Dict[str, object]:
    orchestrator = getattr(env, "_scenario_orchestrator", None)
    if orchestrator is None:
        return {
            "scenario_triggered": False,
            "scenario_realized": False,
            "scenario_notes": ["missing_scenario_orchestrator"],
        }
    return dict(orchestrator.get_episode_summary())


def _info_failure_reason(info: Mapping[str, object], agent_ids: List[str]) -> Optional[str]:
    for agent_id in agent_ids:
        raw_agent_info = info.get(agent_id, {}) if isinstance(info, Mapping) else {}
        agent_info = raw_agent_info if isinstance(raw_agent_info, Mapping) else {}
        if any(bool(agent_info.get(key, False)) for key in FAILURE_CRASH_KEYS):
            return f"crash:{agent_id}"
        if any(bool(agent_info.get(key, False)) for key in FAILURE_ROAD_KEYS):
            return f"out_of_road:{agent_id}"
    return None


def _raw_observation_parts(agent_obs: Mapping[str, object]) -> tuple[np.ndarray, ...]:
    required = (
        "ego_state",
        "others_state",
        "lidar",
        "topdown",
        "rgb_left",
        "rgb_front",
        "rgb_right",
    )
    missing = [key for key in required if key not in agent_obs]
    if missing:
        raise KeyError(f"Raw platoon observation missing keys: {missing}")
    return tuple(np.asarray(agent_obs[key]) for key in required)


def rollout_platoon_episode(
    env: RawObservationPlatoonEnv,
    config: MultiExpertCollectorConfig,
    spec: EpisodeSpec,
    episode_index: int,
) -> PlatoonEpisodeRollout:
    agent_ids = [f"agent{i}" for i in range(int(config.num_agents))]
    spawn_manager = getattr(getattr(env, "engine", None), "spawn_manager", None)
    if spawn_manager is not None and hasattr(spawn_manager, "set_episode_spawn_seed"):
        spawn_manager.set_episode_spawn_seed(int(spec.spawn_seed))

    env.set_runtime_scenario_route(spec.scenario_id, spec.local_route)
    _set_runtime_density(env, spec.traffic_density)
    # spawn_seed belongs to RouteSpawnManager.  MetaDrive's reset(seed=...)
    # selects a map and only accepts values inside its small scenario pool.
    obs = env.reset()
    action_fn = build_rule_planner_pipeline_factory(
        config.decision_policy,
        config.planning_policy,
        config.control_policy,
    )(env, agent_ids, int(spec.spawn_seed))

    frames_by_agent: Dict[str, List[Dict[str, np.ndarray]]] = {
        agent_id: [] for agent_id in agent_ids
    }
    video_frames: List[np.ndarray] = []
    failed = False
    failure_reason: Optional[str] = None
    env_horizon = int(getattr(env, "config", {}).get("horizon", 100))
    max_steps = int(config.max_episode_steps) if int(config.max_episode_steps) > 0 else env_horizon

    for joint_step in range(max_steps):
        missing = [agent_id for agent_id in agent_ids if agent_id not in env.agents or agent_id not in obs]
        if missing:
            failed, failure_reason = True, f"missing_agent:{','.join(missing)}"
            break

        actions = action_fn(env)
        missing_actions = [agent_id for agent_id in agent_ids if agent_id not in actions]
        if missing_actions:
            failed, failure_reason = True, f"missing_action:{','.join(missing_actions)}"
            break

        for agent_index, agent_id in enumerate(agent_ids):
            vehicle = env.agents[agent_id]
            frame = build_frame(vehicle, config, *_raw_observation_parts(obs[agent_id]))
            frame["agent_id"] = np.asarray(agent_id)
            frame["agent_index"] = np.asarray(agent_index, dtype=np.int8)
            frame["agent_role"] = np.asarray(env.get_agent_role(agent_id))
            frame["formation_relation_state"] = np.asarray(
                env.get_formation_relation_state(agent_id), dtype=np.float32
            ).reshape(12)
            frame["joint_step_index"] = np.asarray(joint_step, dtype=np.int32)
            frame["num_agents"] = np.asarray(config.num_agents, dtype=np.int8)
            frames_by_agent[agent_id].append(frame)

        if bool(config.save_videos):
            from tools.topdown_view import capture_topdown_frame

            frame = capture_topdown_frame(env, agent_ids[0], config.topdown_heading_up, agent_ids)
            if frame is not None:
                video_frames.append(frame)

        obs, _, terminated, truncated, info = env.low_level_step(actions)
        failure_reason = _info_failure_reason(info or {}, agent_ids)
        if failure_reason is not None:
            failed = True
            break
        if bool((terminated or {}).get("__all__", False) or (truncated or {}).get("__all__", False)):
            break

    summary = _scenario_summary(env)
    if not failed and not bool(summary.get("scenario_triggered", False)):
        failed, failure_reason = True, "scenario_not_triggered"
    if not failed and not bool(summary.get("scenario_realized", False)):
        failed, failure_reason = True, "scenario_not_realized"

    return PlatoonEpisodeRollout(
        frames_by_agent=frames_by_agent,
        video_frames=video_frames,
        failed=failed,
        failure_reason=failure_reason,
        scenario_summary=summary,
    )


def build_flattened_episode_samples(
    rollout: PlatoonEpisodeRollout,
    config: MultiExpertCollectorConfig,
    spec: EpisodeSpec,
    episode_index: int,
) -> List[Dict[str, np.ndarray]]:
    """Build per-agent samples, then merge by (joint step, agent index)."""

    flattened: List[Dict[str, np.ndarray]] = []
    for agent_index in range(int(config.num_agents)):
        agent_id = f"agent{agent_index}"
        frames = rollout.frames_by_agent[agent_id]
        samples = build_episode_samples(
            frames,
            config,
            spec.traffic_density,
            route_id=spec.route_preset,
            local_route=spec.local_route,
            scenario_id=spec.scenario_id,
            idm_variant="rule_planner",
            episode_index=episode_index,
        )
        for sample in samples:
            frame_index = int(np.asarray(sample["_sample_index"]).item())
            source_frame = frames[frame_index]
            for key in (
                "agent_id",
                "agent_index",
                "agent_role",
                "formation_relation_state",
                "joint_step_index",
                "num_agents",
            ):
                sample[key] = np.asarray(source_frame[key]).copy()
            flattened.append(sample)

    flattened.sort(
        key=lambda sample: (
            int(np.asarray(sample["joint_step_index"]).item()),
            int(np.asarray(sample["agent_index"]).item()),
        )
    )
    return flattened


def _save_agent_visualizations(
    samples: List[Dict[str, np.ndarray]],
    root: Optional[Path],
    config: MultiExpertCollectorConfig,
    episode_index: int,
    map_geometry,
) -> None:
    if root is None:
        return
    by_agent: Dict[str, List[Dict[str, np.ndarray]]] = defaultdict(list)
    for sample in samples:
        by_agent[str(np.asarray(sample["agent_id"]).item())].append(sample)
    for agent_id, agent_samples in by_agent.items():
        _save_sample_visualizations(
            agent_samples,
            root / agent_id,
            config,
            episode_index,
            map_geometry,
        )


def _multi_manifest(
    *,
    config: MultiExpertCollectorConfig,
    dataset_root: Path,
    effective_env_config: dict,
    total_samples: int,
    total_episodes: int,
    successful_episodes: int,
    discarded_episodes: int,
    failure_counts: Counter,
    split_summary: dict,
    mode_counts: Mapping[str, int],
    route_counts: Mapping[str, int],
    local_route_counts: Mapping[str, int],
    scenario_counts: Mapping[str, int],
    agent_counts: Mapping[str, int],
    role_counts: Mapping[str, int],
    filter_stats: Mapping[str, object],
    wall_time_sec: float,
) -> dict:
    return {
        "dataset_name": config.dataset_name,
        "collector_type": "multi_rule_planner",
        "expert_type": config.expert_type,
        "target_samples": int(config.target_samples),
        "target_samples_semantics": "flattened_per_agent_rows",
        "collected_samples": int(total_samples),
        "episodes": int(total_episodes),
        "successful_episodes": int(successful_episodes),
        "discarded_episodes": int(discarded_episodes),
        "discard_reasons": dict(failure_counts),
        "output_root": str(dataset_root),
        "num_agents": int(config.num_agents),
        "decision_policy": config.decision_policy,
        "planning_policy": config.planning_policy,
        "control_policy": config.control_policy,
        "environment_config_source": str(config.dataset_config_path),
        "environment_config_overrides": {
            "num_agents": int(config.num_agents),
            "observation_mode": "multimodal",
            "use_render": False,
            "image_on_cuda": False,
            "image_on_cuda_reason": "raw multi-agent image collection is CPU-only",
        },
        "effective_env_config": _to_jsonable(effective_env_config),
        "route_distribution": dict(route_counts),
        "local_route_distribution": dict(local_route_counts),
        "scenario_distribution": dict(scenario_counts),
        "agent_distribution": dict(agent_counts),
        "role_distribution": dict(role_counts),
        "trajectory_modes": build_trajectory_mode_summary(dict(mode_counts), total_samples),
        "trajectory_filter": {
            **dict(filter_stats),
            "rejection_reasons": dict(filter_stats.get("rejection_reasons", {})),
        },
        "splits": {name: len(shards) for name, shards in split_summary.items()},
        "collection_wall_time_sec": float(wall_time_sec),
        "config": _to_jsonable(asdict(config)),
    }


def run_collection(config: MultiExpertCollectorConfig) -> None:
    if config.expert_type != "rule_planner":
        raise ValueError("Multi-agent collection only supports expert_type='rule_planner'.")
    if int(config.num_agents) != 3:
        raise ValueError("This expert demonstration collector currently requires num_agents=3.")
    effective_env_config = load_collection_env_config(config)
    if set(config.scenario_weights) - set(DEFAULT_SCENARIOS):
        raise ValueError("Only S5_hard_brake_lead and S6_background_merge_in are enabled.")

    dataset_root = config.output_root / config.dataset_name
    shard_dir = dataset_root / "shards"
    report_dir = dataset_root / "reports"
    visualization_dir = report_dir / "trajectory_visualizations"
    qual_dir = visualization_dir / "qual"
    unqual_dir = visualization_dir / "unqual"
    video_dir = report_dir / "videos"
    shard_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    existing = detect_existing_state(shard_dir, report_dir) if config.resume else {
        "shard_count": 0,
        "total_samples": 0,
        "total_episodes": 0,
        "mode_counts": {},
        "route_counts": {},
        "local_route_counts": {},
        "scenario_counts": {},
    }
    total_samples = int(existing["total_samples"])
    total_episodes = int(existing["total_episodes"])
    mode_counts = Counter(existing.get("mode_counts", {}))
    route_counts = Counter(existing.get("route_counts", {}))
    local_route_counts = Counter(existing.get("local_route_counts", {}))
    scenario_counts = Counter(existing.get("scenario_counts", {}))

    existing_manifest_path = report_dir / "manifest.json"
    previous_manifest = (
        json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if config.resume and existing_manifest_path.exists()
        else {}
    )
    if previous_manifest.get("trajectory_modes"):
        mode_counts.update(
            {
                str(name): int(stats.get("count", 0))
                for name, stats in dict(previous_manifest["trajectory_modes"]).items()
            }
        )
    successful_episodes = int(previous_manifest.get("successful_episodes", 0))
    discarded_episodes = int(previous_manifest.get("discarded_episodes", 0))
    failure_counts = Counter(previous_manifest.get("discard_reasons", {}))
    agent_counts = Counter(previous_manifest.get("agent_distribution", {}))
    role_counts = Counter(previous_manifest.get("role_distribution", {}))

    rng = build_episode_rngs(config)
    fast_forward_episode_rngs(rng, config, total_episodes)
    sampler = EpisodeSpecSampler(config, rng)
    writer = ShardWriter(shard_dir, config.samples_per_shard, int(existing["shard_count"]))
    trajectory_filter = build_trajectory_filter_pipeline(config)
    filter_stats = {
        "enabled": bool(trajectory_filter is not None),
        "total_checked": 0,
        "accepted": 0,
        "rejected": 0,
        "reject_rate": 0.0,
        "rejection_reasons": {},
        "missing_reference_lane_points": 0,
    }

    runtime_env_config = {
        key: value
        for key, value in effective_env_config.items()
        if key not in PLATOON_WRAPPER_ONLY_ENV_KEYS
    }
    env = RawObservationPlatoonEnv(runtime_env_config)
    (dataset_root / "env_config.json").write_text(
        json.dumps(_to_jsonable(dict(env.config)), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    map_geometry = None
    wall_start = time.perf_counter()

    try:
        while total_samples < int(config.target_samples) and (
            int(config.max_episodes) <= 0 or total_episodes < int(config.max_episodes)
        ):
            spec = sampler.sample()
            episode_index = total_episodes + 1
            rollout = rollout_platoon_episode(env, config, spec, episode_index)
            total_episodes += 1

            if rollout.failed:
                discarded_episodes += 1
                failure_counts[rollout.failure_reason or "unknown"] += 1
                print(
                    f"[ep={total_episodes}] DISCARD scenario={spec.scenario_id} "
                    f"route={spec.local_route} reason={rollout.failure_reason}"
                )
                continue

            successful_episodes += 1
            if bool(config.save_videos):
                write_episode_video(
                    video_dir / spec.scenario_id / f"episode_{episode_index:06d}.mp4",
                    rollout.video_frames,
                    config.video_fps,
                )
            if bool(config.trajectory_visualization_enabled) and map_geometry is None:
                map_geometry = build_map_visualization_geometry(env)

            samples = build_flattened_episode_samples(rollout, config, spec, episode_index)
            accepted, rejected = _filter_samples(samples, trajectory_filter, filter_stats)
            _save_agent_visualizations(
                accepted,
                qual_dir if bool(config.trajectory_visualization_enabled) else None,
                config,
                episode_index,
                map_geometry,
            )
            _save_agent_visualizations(
                rejected,
                unqual_dir if bool(config.trajectory_visualization_enabled) else None,
                config,
                episode_index,
                map_geometry,
            )

            remaining = int(config.target_samples) - total_samples
            stored = _prepare_samples_for_storage(accepted[:remaining])
            writer.add_samples(stored)
            total_samples += len(stored)
            route_counts[spec.route_preset] += len(stored)
            local_route_counts[spec.local_route] += len(stored)
            scenario_counts[spec.scenario_id] += len(stored)
            for sample in stored:
                agent_counts[str(np.asarray(sample["agent_id"]).item())] += 1
                role_counts[str(np.asarray(sample["agent_role"]).item())] += 1
                mode_counts[trajectory_mode_name(int(sample["trajectory_mode"]))] += 1

            elapsed = max(time.perf_counter() - wall_start, 1e-6)
            rate = total_samples / elapsed
            eta = (int(config.target_samples) - total_samples) / rate if rate > 0 else float("inf")
            print(
                f"[ep={total_episodes}] samples={total_samples}/{config.target_samples} "
                f"scenario={spec.scenario_id} route={spec.local_route} "
                f"episode_rows={len(stored)} sample/s={rate:.2f} ETA={format_eta(eta)}"
            )
    finally:
        env.close()
        writer.close()

    filter_stats["reject_rate"] = float(filter_stats["rejected"]) / max(
        int(filter_stats["total_checked"]), 1
    )
    if any(shard_dir.glob("shard_*.npz")):
        split_summary = split_shards(
            dataset_root,
            config.train_split_ratio,
            config.val_split_ratio,
            config.test_split_ratio,
            config.split_seed,
        )
    else:
        split_summary = {"train": [], "val": [], "test": []}
    manifest = _multi_manifest(
        config=config,
        dataset_root=dataset_root,
        effective_env_config=effective_env_config,
        total_samples=total_samples,
        total_episodes=total_episodes,
        successful_episodes=successful_episodes,
        discarded_episodes=discarded_episodes,
        failure_counts=failure_counts,
        split_summary=split_summary,
        mode_counts=mode_counts,
        route_counts=route_counts,
        local_route_counts=local_route_counts,
        scenario_counts=scenario_counts,
        agent_counts=agent_counts,
        role_counts=role_counts,
        filter_stats=filter_stats,
        wall_time_sec=time.perf_counter() - wall_start,
    )
    (report_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Done. samples={total_samples} output={dataset_root}")


def parse_args() -> MultiExpertCollectorConfig:
    config = MultiExpertCollectorConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    for item in fields(MultiExpertCollectorConfig):
        default = getattr(config, item.name)
        opts: dict = {"default": default, "dest": item.name}
        if isinstance(default, bool):
            opts["type"] = _coerce_bool
        elif isinstance(default, Path):
            opts["type"] = Path
        elif default is None and item.name.endswith("_dir"):
            opts["type"] = Path
        elif isinstance(default, dict):
            opts["type"] = lambda value: dict(json.loads(value))
        elif isinstance(default, tuple):
            opts["type"] = lambda value: tuple(json.loads(value))
        else:
            opts["type"] = type(default)
        parser.add_argument(f"--{item.name.replace('_', '-')}", **opts)
    parser.parse_args(namespace=config)
    return config


def main() -> None:
    run_collection(parse_args())


if __name__ == "__main__":
    main()
