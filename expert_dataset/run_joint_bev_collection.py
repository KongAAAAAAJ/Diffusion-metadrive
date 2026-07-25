"""Collect and persist the three-vehicle joint-first BEV expert dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml

from expert_dataset.collect_joint_bev import (
    JointCollectionError,
    SensorlessJointBEVPlatoonEnv,
    collect_joint_episode,
)
from expert_dataset.joint_bev_storage import (
    EpisodeSplitConfig,
    JointBEVDatasetStore,
    fingerprint_payload,
)
from scenarios.definitions import SCENARIO_BY_ID, get_scenario_definition


REPO_ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL_KEYS = {"dataset", "split", "collection", "env_config"}
SECTION_KEYS = {
    "dataset": {"name", "output_root"},
    "split": {"train_ratio", "val_ratio", "test_ratio", "seed"},
    "collection": {
        "target_joint_steps",
        "start_seed",
        "max_episodes",
        "max_episode_steps",
        "resume",
        "scenario_weights",
        "traffic_density_min",
        "traffic_density_max",
    },
}


@dataclass(frozen=True)
class JointCollectionRunConfig:
    config_path: Path
    dataset_root: Path
    split_config: EpisodeSplitConfig
    target_joint_steps: int
    start_seed: int
    max_episodes: int
    max_episode_steps: int
    resume: bool
    scenario_weights: Mapping[str, float]
    traffic_density_min: float
    traffic_density_max: float
    env_config: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.target_joint_steps <= 0:
            raise ValueError("collection.target_joint_steps must be positive")
        if self.max_episodes < 0:
            raise ValueError("collection.max_episodes must be non-negative")
        if self.max_episode_steps <= 0:
            raise ValueError("collection.max_episode_steps must be positive")
        if not self.scenario_weights:
            raise ValueError("collection.scenario_weights must not be empty")
        normalized = {}
        for scenario_id, weight in self.scenario_weights.items():
            if scenario_id not in SCENARIO_BY_ID:
                raise ValueError(f"unknown scenario in scenario_weights: {scenario_id}")
            value = float(weight)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"scenario weight must be positive: {scenario_id}")
            normalized[str(scenario_id)] = value
        object.__setattr__(self, "scenario_weights", normalized)
        density_min = float(self.traffic_density_min)
        density_max = float(self.traffic_density_max)
        if (
            not np.isfinite(density_min)
            or not np.isfinite(density_max)
            or density_min < 0.0
            or density_min > density_max
        ):
            raise ValueError("traffic density range must be finite, non-negative, and ordered")
        object.__setattr__(self, "traffic_density_min", density_min)
        object.__setattr__(self, "traffic_density_max", density_max)

        env_config = dict(self.env_config)
        if int(env_config.get("num_agents", 3)) != 3:
            raise ValueError("env_config.num_agents must be exactly 3")
        if bool(env_config.get("use_render", False)):
            raise ValueError("joint BEV collection requires env_config.use_render=false")
        if bool(env_config.get("image_observation", False)):
            raise ValueError("joint BEV collection requires image_observation=false")
        if env_config.get("sensors") not in (None, {}):
            raise ValueError("joint BEV collection requires an empty sensors mapping")
        observation_mode = str(env_config.get("observation_mode", "bev_gt"))
        if observation_mode != "bev_gt":
            raise ValueError("env_config.observation_mode must be 'bev_gt'")
        env_config.update(
            {
                "num_agents": 3,
                "observation_mode": "bev_gt",
                "use_render": False,
                "image_observation": False,
                "image_on_cuda": False,
                "sensors": {},
                "ground_truth_traffic_policy": True,
            }
        )
        object.__setattr__(self, "env_config", env_config)

    def immutable_fingerprint(self) -> str:
        return fingerprint_payload(
            {
                "collector": "joint_bev_rule_planner",
                "start_seed": self.start_seed,
                "max_episode_steps": self.max_episode_steps,
                "scenario_weights": dict(self.scenario_weights),
                "traffic_density_min": self.traffic_density_min,
                "traffic_density_max": self.traffic_density_max,
                "env_config": dict(self.env_config),
            }
        )


@dataclass(frozen=True)
class JointEpisodeSpec:
    scenario_id: str
    local_route: str
    spawn_seed: int
    traffic_density: float
    initial_speed_km_h: float


def _strict_section(
    payload: Mapping[str, object],
    name: str,
    *,
    required: bool = True,
) -> dict[str, object]:
    value = payload.get(name)
    if value is None and not required:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"'{name}' must be a mapping")
    result = dict(value)
    allowed = SECTION_KEYS.get(name)
    if allowed is not None:
        unknown = sorted(set(result) - allowed)
        if unknown:
            raise ValueError(f"unknown '{name}' fields: {unknown}")
    return result


def _required(section: Mapping[str, object], section_name: str, key: str) -> object:
    if key not in section:
        raise ValueError(f"missing config field: {section_name}.{key}")
    return section[key]


def _resolve_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_run_config(path: Path | str) -> JointCollectionRunConfig:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("dataset config root must be a mapping")
    unknown = sorted(set(payload) - TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"unknown dataset config sections: {unknown}")

    dataset = _strict_section(payload, "dataset")
    split = _strict_section(payload, "split")
    collection = _strict_section(payload, "collection")
    env_config = _strict_section(payload, "env_config", required=False)

    dataset_name = str(_required(dataset, "dataset", "name")).strip()
    if not dataset_name or Path(dataset_name).name != dataset_name:
        raise ValueError("dataset.name must be one non-empty directory name")
    dataset_root = _resolve_path(
        _required(dataset, "dataset", "output_root")
    ) / dataset_name
    scenario_weights = _required(
        collection, "collection", "scenario_weights"
    )
    if not isinstance(scenario_weights, Mapping):
        raise ValueError("collection.scenario_weights must be a mapping")
    resume = _required(collection, "collection", "resume")
    if not isinstance(resume, bool):
        raise ValueError("collection.resume must be a boolean")

    return JointCollectionRunConfig(
        config_path=config_path,
        dataset_root=dataset_root,
        split_config=EpisodeSplitConfig(
            train_ratio=float(_required(split, "split", "train_ratio")),
            val_ratio=float(_required(split, "split", "val_ratio")),
            test_ratio=float(_required(split, "split", "test_ratio")),
            seed=int(_required(split, "split", "seed")),
        ),
        target_joint_steps=int(
            _required(collection, "collection", "target_joint_steps")
        ),
        start_seed=int(_required(collection, "collection", "start_seed")),
        max_episodes=int(_required(collection, "collection", "max_episodes")),
        max_episode_steps=int(
            _required(collection, "collection", "max_episode_steps")
        ),
        resume=resume,
        scenario_weights={
            str(name): float(weight) for name, weight in scenario_weights.items()
        },
        traffic_density_min=float(
            _required(collection, "collection", "traffic_density_min")
        ),
        traffic_density_max=float(
            _required(collection, "collection", "traffic_density_max")
        ),
        env_config=env_config,
    )


def _episode_rng(start_seed: int, episode_index: int) -> np.random.RandomState:
    digest = hashlib.sha256(f"{start_seed}:{episode_index}".encode("ascii")).digest()
    return np.random.RandomState(int.from_bytes(digest[:4], "big"))


def sample_episode_spec(
    config: JointCollectionRunConfig, episode_index: int
) -> JointEpisodeSpec:
    rng = _episode_rng(config.start_seed, episode_index)
    scenario_ids = tuple(sorted(config.scenario_weights))
    weights = np.asarray(
        [config.scenario_weights[name] for name in scenario_ids], dtype=np.float64
    )
    weights /= weights.sum()
    scenario_id = str(rng.choice(scenario_ids, p=weights))
    scenario = get_scenario_definition(scenario_id)
    local_routes = tuple(scenario.trigger_by_local_route) or tuple(
        scenario.allowed_local_routes
    )
    if not local_routes:
        raise ValueError(f"scenario {scenario_id} has no collectable local route")
    local_route = str(rng.choice(local_routes))
    density = float(
        rng.uniform(config.traffic_density_min, config.traffic_density_max)
    )
    if scenario.override_traffic_density is not None:
        density = min(density, float(scenario.override_traffic_density))
    initial_speed = scenario.ego_initial_speed_km_h
    if isinstance(initial_speed, (tuple, list)) and len(initial_speed) == 2:
        initial_speed_km_h = float(rng.uniform(float(initial_speed[0]), float(initial_speed[1])))
    elif initial_speed is None:
        initial_speed_km_h = float(config.env_config.get("initial_speed_km_h", 25.0))
    else:
        initial_speed_km_h = float(initial_speed)
    return JointEpisodeSpec(
        scenario_id=scenario_id,
        local_route=local_route,
        spawn_seed=int(rng.randint(0, 2**31 - 1)),
        traffic_density=density,
        initial_speed_km_h=initial_speed_km_h,
    )


def _configure_episode(
    env: SensorlessJointBEVPlatoonEnv,
    spec: JointEpisodeSpec,
) -> None:
    env.set_runtime_scenario_route(spec.scenario_id, spec.local_route)
    updates = {
        "traffic_density": spec.traffic_density,
        "initial_speed_km_h": spec.initial_speed_km_h,
    }
    env.config.update(updates)
    env.platoon_config.traffic_density = spec.traffic_density
    env.platoon_config.initial_speed_km_h = spec.initial_speed_km_h
    global_config = getattr(getattr(env, "engine", None), "global_config", None)
    if global_config is not None:
        global_config.update(updates)
    spawn_manager = getattr(getattr(env, "engine", None), "spawn_manager", None)
    if spawn_manager is not None and hasattr(spawn_manager, "set_episode_spawn_seed"):
        spawn_manager.set_episode_spawn_seed(spec.spawn_seed)


def run_collection(config: JointCollectionRunConfig) -> dict[str, object]:
    wall_start = time.perf_counter()
    with JointBEVDatasetStore(
        config.dataset_root,
        split_config=config.split_config,
        dataset_fingerprint=config.immutable_fingerprint(),
        resume=config.resume,
    ) as store:
        starting_joint_samples = store.total_joint_samples
        print(
            f"[INFO] dataset={config.dataset_root} "
            f"resume={config.resume} existing_joint_steps={store.total_joint_samples}",
            flush=True,
        )
        if store.total_joint_samples >= config.target_joint_steps:
            summary = store.summary()
            print("[INFO] target already satisfied; no simulator started", flush=True)
            return summary

        env = SensorlessJointBEVPlatoonEnv(dict(config.env_config))
        try:
            while store.total_joint_samples < config.target_joint_steps:
                episode_index = store.next_episode_index
                if config.max_episodes > 0 and episode_index >= config.max_episodes:
                    break
                spec = sample_episode_spec(config, episode_index)
                _configure_episode(env, spec)
                try:
                    rollout = collect_joint_episode(
                        env, max_steps=config.max_episode_steps
                    )
                except JointCollectionError as exc:
                    reason = f"collection_contract:{type(exc).__name__}"
                    store.record_rejected_episode(episode_index, reason)
                    print(
                        f"[WARNING] episode={episode_index} split="
                        f"{store.assigner.split_for_episode(episode_index)} "
                        f"scenario={spec.scenario_id} rejected={reason}: {exc}",
                        flush=True,
                    )
                    continue

                if rollout.failure_reason is not None:
                    store.record_rejected_episode(
                        episode_index, rollout.failure_reason
                    )
                    print(
                        f"[WARNING] episode={episode_index} scenario={spec.scenario_id} "
                        f"rejected={rollout.failure_reason}",
                        flush=True,
                    )
                    continue
                if not rollout.samples:
                    store.record_rejected_episode(
                        episode_index, "no_joint_samples"
                    )
                    print(
                        f"[WARNING] episode={episode_index} scenario={spec.scenario_id} "
                        "rejected=no_joint_samples",
                        flush=True,
                    )
                    continue

                stored = store.commit_episode(
                    episode_index,
                    rollout.samples,
                    {
                        "scenario_id": spec.scenario_id,
                        "local_route": spec.local_route,
                        "spawn_seed": spec.spawn_seed,
                        "traffic_density": spec.traffic_density,
                        "initial_speed_km_h": spec.initial_speed_km_h,
                        "simulator_steps": rollout.simulator_steps,
                        "rejected_joint_steps": rollout.rejected_joint_steps,
                        "terminated": rollout.terminated,
                        "truncated": rollout.truncated,
                    },
                )
                elapsed = max(time.perf_counter() - wall_start, 1e-6)
                rate = (
                    store.total_joint_samples - starting_joint_samples
                ) / elapsed
                print(
                    f"[INFO] episode={episode_index} split={stored.split} "
                    f"scenario={spec.scenario_id} route={spec.local_route} "
                    f"episode_joint_steps={stored.joint_samples} "
                    f"total={store.total_joint_samples}/{config.target_joint_steps} "
                    f"joint_steps_per_s={rate:.2f}",
                    flush=True,
                )
        finally:
            env.close()
        summary = store.summary()

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def parse_args(argv: list[str] | None = None) -> JointCollectionRunConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/dataset/data_collect.yaml",
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--target-joint-steps", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--resume", choices=("0", "1"))
    args = parser.parse_args(argv)
    config = load_run_config(args.config)
    overrides = {}
    if args.dataset_root is not None:
        overrides["dataset_root"] = args.dataset_root.expanduser().resolve()
    if args.target_joint_steps is not None:
        overrides["target_joint_steps"] = args.target_joint_steps
    if args.max_episodes is not None:
        overrides["max_episodes"] = args.max_episodes
    if args.max_episode_steps is not None:
        overrides["max_episode_steps"] = args.max_episode_steps
    if args.resume is not None:
        overrides["resume"] = args.resume == "1"
    return replace(config, **overrides)


def main(argv: list[str] | None = None) -> int:
    run_collection(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
