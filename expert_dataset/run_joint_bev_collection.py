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
    simulator_decision_dt_s,
)
from expert_dataset.joint_risk_bundle_storage import (
    BundleEpisodeAttempt,
    BundleEpisodeResult,
    JointRiskBundleIndex,
    JointRiskBundleStorageError,
)
from expert_dataset.joint_bev_storage import (
    EpisodeSplitConfig,
    JointBEVDatasetStore,
    fingerprint_payload,
)
from expert_dataset.riskentry_sidecar_storage import (
    RiskEntrySidecarDatasetStore,
    RiskEntrySidecarStorageError,
    SidecarEpisodeStart,
)
from scenarios.definitions import SCENARIO_BY_ID, get_scenario_definition
from scenarios.bev_round13_contract import (
    PRIMARY_S5_S9_SCENARIOS,
    primary_scenario_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL_KEYS = {
    "dataset",
    "split",
    "collection",
    "env_config",
    "diagnostic_64",
}
SECTION_KEYS = {
    "dataset": {"name", "sidecar_name", "output_root"},
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
    "diagnostic_64": {
        "enabled",
        "scenario_quotas",
        "sample_offsets_after_trigger_s",
        "max_attempts_per_scenario",
    },
}


@dataclass(frozen=True)
class JointCollectionRunConfig:
    config_path: Path
    bundle_root: Path
    dataset_root: Path
    sidecar_root: Path
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
    diagnostic_scenario_quotas: Mapping[str, int] | None = None
    diagnostic_sample_offsets_s: tuple[float, ...] = ()
    diagnostic_max_attempts_per_scenario: int = 0

    def __post_init__(self) -> None:
        bundle_root = Path(self.bundle_root).expanduser().resolve()
        dataset_root = Path(self.dataset_root).expanduser().resolve()
        sidecar_root = Path(self.sidecar_root).expanduser().resolve()
        if dataset_root == sidecar_root:
            raise ValueError("base and sidecar dataset roots must differ")
        if dataset_root.parent != bundle_root or sidecar_root.parent != bundle_root:
            raise ValueError("base and sidecar roots must be direct children of bundle_root")
        object.__setattr__(self, "bundle_root", bundle_root)
        object.__setattr__(self, "dataset_root", dataset_root)
        object.__setattr__(self, "sidecar_root", sidecar_root)
        if int(self.split_config.seed) != 17:
            raise ValueError("joint BEV/RiskEntry bundle split seed is frozen to 17")
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
        quotas = self.diagnostic_scenario_quotas
        if quotas is not None:
            expected = tuple(value[0] for value in PRIMARY_S5_S9_SCENARIOS)
            if tuple(quotas) != expected:
                raise ValueError(
                    "diagnostic_64 scenario_quotas must use ordered S5--S9"
                )
            normalized_quotas = {}
            for scenario_id, value in quotas.items():
                if isinstance(value, bool) or int(value) <= 0:
                    raise ValueError(
                        "diagnostic_64 scenario quotas must be positive integers"
                    )
                normalized_quotas[str(scenario_id)] = int(value)
            if sum(normalized_quotas.values()) != self.target_joint_steps:
                raise ValueError(
                    "diagnostic_64 quotas must sum to target_joint_steps"
                )
            offsets = tuple(float(value) for value in self.diagnostic_sample_offsets_s)
            if (
                not offsets
                or any(not np.isfinite(value) or value < 0.0 for value in offsets)
                or tuple(sorted(set(offsets))) != offsets
            ):
                raise ValueError(
                    "diagnostic_64 sample offsets must be unique, ordered and non-negative"
                )
            if int(self.diagnostic_max_attempts_per_scenario) <= 0:
                raise ValueError(
                    "diagnostic_64 max_attempts_per_scenario must be positive"
                )
            if self.resume:
                raise ValueError("diagnostic_64 collection does not support resume")
            object.__setattr__(
                self, "diagnostic_scenario_quotas", normalized_quotas
            )
            object.__setattr__(self, "diagnostic_sample_offsets_s", offsets)
            object.__setattr__(
                self,
                "diagnostic_max_attempts_per_scenario",
                int(self.diagnostic_max_attempts_per_scenario),
            )

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
                "diagnostic_64": (
                    None
                    if self.diagnostic_scenario_quotas is None
                    else {
                        "scenario_quotas": dict(
                            self.diagnostic_scenario_quotas
                        ),
                        "sample_offsets_after_trigger_s": list(
                            self.diagnostic_sample_offsets_s
                        ),
                        "max_attempts_per_scenario": (
                            self.diagnostic_max_attempts_per_scenario
                        ),
                        "scenario_contract": primary_scenario_contract(),
                    }
                ),
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
    diagnostic = _strict_section(
        payload, "diagnostic_64", required=False
    )

    dataset_name = str(_required(dataset, "dataset", "name")).strip()
    if not dataset_name or Path(dataset_name).name != dataset_name:
        raise ValueError("dataset.name must be one non-empty directory name")
    sidecar_name = str(_required(dataset, "dataset", "sidecar_name")).strip()
    if not sidecar_name or Path(sidecar_name).name != sidecar_name:
        raise ValueError("dataset.sidecar_name must be one non-empty directory name")
    if sidecar_name == dataset_name:
        raise ValueError("dataset.name and dataset.sidecar_name must differ")
    bundle_root = _resolve_path(_required(dataset, "dataset", "output_root"))
    dataset_root = bundle_root / dataset_name
    sidecar_root = bundle_root / sidecar_name
    scenario_weights = _required(
        collection, "collection", "scenario_weights"
    )
    if not isinstance(scenario_weights, Mapping):
        raise ValueError("collection.scenario_weights must be a mapping")
    resume = _required(collection, "collection", "resume")
    if not isinstance(resume, bool):
        raise ValueError("collection.resume must be a boolean")

    diagnostic_enabled = diagnostic.get("enabled", False)
    if not isinstance(diagnostic_enabled, bool):
        raise ValueError("diagnostic_64.enabled must be bool")
    diagnostic_quotas = None
    diagnostic_offsets: tuple[float, ...] = ()
    diagnostic_max_attempts = 0
    if diagnostic_enabled:
        raw_quotas = _required(
            diagnostic, "diagnostic_64", "scenario_quotas"
        )
        raw_offsets = _required(
            diagnostic,
            "diagnostic_64",
            "sample_offsets_after_trigger_s",
        )
        if not isinstance(raw_quotas, Mapping):
            raise ValueError("diagnostic_64.scenario_quotas must be a mapping")
        if not isinstance(raw_offsets, (list, tuple)):
            raise ValueError(
                "diagnostic_64.sample_offsets_after_trigger_s must be a list"
            )
        diagnostic_quotas = {
            str(name): int(value) for name, value in raw_quotas.items()
        }
        diagnostic_offsets = tuple(float(value) for value in raw_offsets)
        diagnostic_max_attempts = int(
            _required(
                diagnostic,
                "diagnostic_64",
                "max_attempts_per_scenario",
            )
        )

    return JointCollectionRunConfig(
        config_path=config_path,
        bundle_root=bundle_root,
        dataset_root=dataset_root,
        sidecar_root=sidecar_root,
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
        diagnostic_scenario_quotas=diagnostic_quotas,
        diagnostic_sample_offsets_s=diagnostic_offsets,
        diagnostic_max_attempts_per_scenario=diagnostic_max_attempts,
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


def sample_episode_spec_for_scenario(
    config: JointCollectionRunConfig,
    episode_index: int,
    scenario_id: str,
) -> JointEpisodeSpec:
    """Sample episode nuisance variables while freezing the diagnostic scenario."""

    if scenario_id not in config.scenario_weights:
        raise ValueError(f"scenario is not enabled by collection config: {scenario_id}")
    rng = _episode_rng(config.start_seed, episode_index)
    scenario = get_scenario_definition(scenario_id)
    route_by_scenario = dict(PRIMARY_S5_S9_SCENARIOS)
    if scenario_id not in route_by_scenario:
        raise ValueError(f"diagnostic scenario is outside S5--S9: {scenario_id}")
    local_route = route_by_scenario[scenario_id]
    density = float(
        rng.uniform(config.traffic_density_min, config.traffic_density_max)
    )
    if scenario.override_traffic_density is not None:
        density = min(density, float(scenario.override_traffic_density))
    initial_speed = scenario.ego_initial_speed_km_h
    if isinstance(initial_speed, (tuple, list)) and len(initial_speed) == 2:
        initial_speed_km_h = float(
            rng.uniform(float(initial_speed[0]), float(initial_speed[1]))
        )
    elif initial_speed is None:
        initial_speed_km_h = float(
            config.env_config.get("initial_speed_km_h", 25.0)
        )
    else:
        initial_speed_km_h = float(initial_speed)
    return JointEpisodeSpec(
        scenario_id=scenario_id,
        local_route=local_route,
        spawn_seed=int(rng.randint(0, 2**31 - 1)),
        traffic_density=density,
        initial_speed_km_h=initial_speed_km_h,
    )


def _select_diagnostic_samples(
    rollout,
    *,
    decision_dt_s: float,
    offsets_s: tuple[float, ...],
    remaining: int,
) -> tuple[tuple[object, ...], tuple[int, ...], int]:
    summary = dict(rollout.scenario_summary)
    if not bool(summary.get("scenario_triggered", False)):
        raise JointCollectionError(
            "diagnostic episode never triggered its dangerous event",
            reason_code="diagnostic_event_not_triggered",
        )
    trigger_step = summary.get("scenario_trigger_step")
    if (
        isinstance(trigger_step, bool)
        or not isinstance(trigger_step, (int, np.integer))
        or int(trigger_step) < 0
    ):
        raise JointCollectionError(
            "diagnostic episode has no valid trigger step",
            reason_code="diagnostic_event_not_triggered",
        )
    if len(rollout.samples) != len(rollout.sample_step_indices):
        raise JointCollectionError(
            "sample/step metadata are misaligned",
            reason_code="diagnostic_sample_alignment",
        )
    selected_indices: list[int] = []
    selected_steps: list[int] = []
    for offset_s in offsets_s[:remaining]:
        target_step = int(trigger_step) + int(round(offset_s / decision_dt_s))
        eligible = [
            (index, int(step))
            for index, step in enumerate(rollout.sample_step_indices)
            if int(step) >= target_step and index not in selected_indices
        ]
        if not eligible:
            raise JointCollectionError(
                f"no history-ready sample exists after trigger+{offset_s:.1f}s",
                reason_code="diagnostic_event_sample_missing",
            )
        index, actual_step = min(
            eligible, key=lambda value: (value[1] - target_step, value[1])
        )
        selected_indices.append(index)
        selected_steps.append(actual_step)
    if not selected_indices:
        raise JointCollectionError(
            "diagnostic episode selected no samples",
            reason_code="diagnostic_event_sample_missing",
        )
    return (
        tuple(rollout.samples[index] for index in selected_indices),
        tuple(selected_steps),
        int(trigger_step),
    )


def _configure_episode(
    env: SensorlessJointBEVPlatoonEnv,
    spec: JointEpisodeSpec,
) -> None:
    env.set_runtime_scenario_route(spec.scenario_id, spec.local_route)
    updates = {
        "traffic_density": spec.traffic_density,
        "initial_speed_km_h": spec.initial_speed_km_h,
        "start_seed": int(spec.spawn_seed),
    }
    env.config.update(updates)
    # MetaDrive validates reset(seed) against the construction-time scenario
    # window.  This collector deliberately reuses one num_scenarios=1 env, so
    # move that single-scenario window atomically with the episode contract.
    env.start_index = int(spec.spawn_seed)
    env.platoon_config.traffic_density = spec.traffic_density
    env.platoon_config.initial_speed_km_h = spec.initial_speed_km_h
    global_config = getattr(getattr(env, "engine", None), "global_config", None)
    if global_config is not None:
        global_config.update(updates)
    spawn_manager = getattr(getattr(env, "engine", None), "spawn_manager", None)
    if spawn_manager is not None and hasattr(spawn_manager, "set_episode_spawn_seed"):
        spawn_manager.set_episode_spawn_seed(spec.spawn_seed)


def _stored_component_episode(store, episode_index: int):
    matches = [
        writer.episodes[episode_index]
        for writer in store.writers.values()
        if episode_index in writer.episodes
    ]
    if len(matches) > 1:
        raise JointRiskBundleStorageError(
            f"episode {episode_index} exists in multiple component splits"
        )
    return matches[0] if matches else None


def _recover_pending_bundle_attempt(
    bundle: JointRiskBundleIndex,
    base_store: JointBEVDatasetStore,
    sidecar_store: RiskEntrySidecarDatasetStore,
) -> None:
    """Conservatively finish an interrupted transaction without inventing base data."""

    pending = bundle.pending_attempt
    if pending is None:
        return
    episode_index = pending.episode_index
    if episode_index != bundle.next_episode_index:
        raise JointRiskBundleStorageError("pending bundle episode is out of sequence")
    base_episode = _stored_component_episode(base_store, episode_index)
    sidecar_episode = _stored_component_episode(sidecar_store, episode_index)
    prepared = sidecar_store.recover_prepared_episode(
        episode_index, pending.split
    )
    if sidecar_episode is not None and prepared is not None:
        raise JointRiskBundleStorageError(
            "episode has both committed and prepared sidecar payloads"
        )
    if sidecar_episode is None and prepared is not None:
        if base_episode is None:
            prepared = sidecar_store.downgrade_recovered_episode_to_sidecar_only(
                prepared
            )
        sidecar_episode = sidecar_store.finalize_recovered_episode(prepared)
    if (
        base_episode is None
        and sidecar_episode is not None
        and sidecar_episode.base_samples != 0
    ):
        raise JointRiskBundleStorageError(
            "orphan committed sidecar has a non-empty base mapping"
        )
    if base_episode is not None and sidecar_episode is None:
        raise JointRiskBundleStorageError(
            "interrupted transaction contains a forbidden base-only episode"
        )
    if base_episode is None and base_store.next_episode_index == episode_index:
        base_store.record_rejected_episode(
            episode_index, "interrupted_bundle_transaction"
        )
    elif base_episode is None and base_store.next_episode_index <= episode_index:
        raise JointRiskBundleStorageError("base store is behind pending transaction")
    bundle.finalize(
        BundleEpisodeResult(
            episode_index=episode_index,
            split=pending.split,
            scenario_id=pending.scenario_id,
            local_route=pending.local_route,
            spawn_seed=pending.spawn_seed,
            base_status="committed" if base_episode is not None else "rejected",
            base_rejection_reason=(
                None if base_episode is not None else "interrupted_bundle_transaction"
            ),
            sidecar_status=(
                "committed" if sidecar_episode is not None else "rejected"
            ),
            sidecar_rejection_reason=(
                None if sidecar_episode is not None else "interrupted_bundle_transaction"
            ),
            raw_steps=(0 if sidecar_episode is None else sidecar_episode.raw_steps),
            base_samples=(0 if base_episode is None else base_episode.joint_samples),
            outcome=(
                "interrupted"
                if sidecar_episode is None
                else sidecar_episode.outcome
            ),
        )
    )


def _validate_bundle_resume_state(
    bundle: JointRiskBundleIndex,
    base_store: JointBEVDatasetStore,
    sidecar_store: RiskEntrySidecarDatasetStore,
) -> None:
    if bundle.next_episode_index != base_store.next_episode_index:
        raise JointRiskBundleStorageError(
            "bundle/base next_episode_index mismatch after recovery"
        )
    for row in bundle.rows:
        base_episode = _stored_component_episode(base_store, row.episode_index)
        sidecar_episode = _stored_component_episode(sidecar_store, row.episode_index)
        if (base_episode is not None) != (row.base_status == "committed"):
            raise JointRiskBundleStorageError("bundle/base episode status mismatch")
        if (sidecar_episode is not None) != (row.sidecar_status == "committed"):
            raise JointRiskBundleStorageError("bundle/sidecar episode status mismatch")
        if base_episode is not None and sidecar_episode is None:
            raise JointRiskBundleStorageError("base-only episode detected")


def _sidecar_start(
    *,
    episode_index: int,
    split: str,
    spec: JointEpisodeSpec,
    rollout,
    base_dataset_fingerprint: str,
    decision_dt_s: float,
) -> SidecarEpisodeStart:
    summary = dict(rollout.scenario_summary)
    return SidecarEpisodeStart(
        episode_index=episode_index,
        split=split,
        scenario_id=spec.scenario_id,
        local_route=spec.local_route,
        spawn_seed=spec.spawn_seed,
        decision_dt_s=decision_dt_s,
        base_dataset_fingerprint=base_dataset_fingerprint,
        scenario_parameters={
            "scenario_contract_sha256": primary_scenario_contract()["sha256"],
            "traffic_density": spec.traffic_density,
            "initial_speed_km_h": spec.initial_speed_km_h,
            "scenario_trigger_step": summary.get("scenario_trigger_step"),
            "scenario_realized_step": summary.get("scenario_realized_step"),
        },
    )


def _prepare_rollout_sidecar(
    store: RiskEntrySidecarDatasetStore,
    *,
    start: SidecarEpisodeStart,
    rollout,
    base_sample_step_indices: tuple[int, ...],
):
    if rollout.sidecar is None:
        raise RiskEntrySidecarStorageError("rollout has no raw sidecar timeline")
    store.begin_episode(start)
    for capture in rollout.sidecar.captures:
        store.append_capture(
            frame=capture.frame,
            events=capture.events,
            actor_records=rollout.sidecar.actor_records,
            lane_records=rollout.sidecar.lane_records,
            key_actor_ids=rollout.sidecar.key_actor_ids,
        )
    return store.prepare_episode(
        base_sample_step_indices=base_sample_step_indices
    )


def run_collection(config: JointCollectionRunConfig) -> dict[str, object]:
    wall_start = time.perf_counter()
    base_fingerprint = config.immutable_fingerprint()
    with JointBEVDatasetStore(
        config.dataset_root,
        split_config=config.split_config,
        dataset_fingerprint=base_fingerprint,
        resume=config.resume,
    ) as store, RiskEntrySidecarDatasetStore(
        config.sidecar_root,
        base_dataset_fingerprint=base_fingerprint,
        resume=config.resume,
    ) as sidecar_store, JointRiskBundleIndex(
        config.bundle_root,
        base_directory=config.dataset_root.name,
        sidecar_directory=config.sidecar_root.name,
        base_dataset_fingerprint=base_fingerprint,
        sidecar_dataset_fingerprint=sidecar_store.dataset_fingerprint,
        scenario_contract_sha256=primary_scenario_contract()["sha256"],
        split_seed=config.split_config.seed,
        resume=config.resume,
    ) as bundle:
        _recover_pending_bundle_attempt(bundle, store, sidecar_store)
        _validate_bundle_resume_state(bundle, store, sidecar_store)
        starting_joint_samples = store.total_joint_samples
        print(
            f"[INFO] base_dataset={config.dataset_root} "
            f"sidecar_dataset={config.sidecar_root} "
            f"resume={config.resume} existing_joint_steps={store.total_joint_samples}",
            flush=True,
        )
        if store.total_joint_samples >= config.target_joint_steps:
            summary = {
                "base": store.summary(),
                "sidecar": sidecar_store.summary(),
                "bundle_attempts": bundle.next_episode_index,
            }
            print("[INFO] target already satisfied; no simulator started", flush=True)
            return summary

        env: SensorlessJointBEVPlatoonEnv | None = None
        diagnostic_quotas = (
            None
            if config.diagnostic_scenario_quotas is None
            else dict(config.diagnostic_scenario_quotas)
        )
        diagnostic_counts = (
            None
            if diagnostic_quotas is None
            else {name: 0 for name in diagnostic_quotas}
        )
        diagnostic_attempts = (
            None
            if diagnostic_quotas is None
            else {name: 0 for name in diagnostic_quotas}
        )
        diagnostic_order = (
            ()
            if diagnostic_quotas is None
            else tuple(diagnostic_quotas)
        )
        try:
            while store.total_joint_samples < config.target_joint_steps:
                episode_index = store.next_episode_index
                if config.max_episodes > 0 and episode_index >= config.max_episodes:
                    break
                if diagnostic_quotas is None:
                    spec = sample_episode_spec(config, episode_index)
                else:
                    incomplete = [
                        scenario_id
                        for scenario_id in diagnostic_order
                        if diagnostic_counts[scenario_id]
                        < diagnostic_quotas[scenario_id]
                    ]
                    if not incomplete:
                        break
                    scenario_id = incomplete[
                        episode_index % len(incomplete)
                    ]
                    if (
                        diagnostic_attempts[scenario_id]
                        >= config.diagnostic_max_attempts_per_scenario
                    ):
                        raise JointCollectionError(
                            f"diagnostic quota for {scenario_id} was not met "
                            f"within {config.diagnostic_max_attempts_per_scenario} attempts",
                            reason_code="diagnostic_attempt_limit",
                        )
                    diagnostic_attempts[scenario_id] += 1
                    spec = sample_episode_spec_for_scenario(
                        config, episode_index, scenario_id
                    )
                split = store.assigner.split_for_episode(episode_index)
                bundle.begin_attempt(
                    BundleEpisodeAttempt(
                        episode_index=episode_index,
                        split=split,
                        scenario_id=spec.scenario_id,
                        local_route=spec.local_route,
                        spawn_seed=spec.spawn_seed,
                    )
                )
                # Managers such as traffic/scenario policies retain internal
                # episode state beyond BaseEnv.reset().  Recreate the
                # sensorless environment so a fixed episode spec is invariant
                # to which scenarios ran before it.
                if env is not None:
                    env.close()
                episode_env_config = dict(config.env_config)
                episode_env_config.update(
                    {"start_seed": int(spec.spawn_seed), "num_scenarios": 1}
                )
                env = SensorlessJointBEVPlatoonEnv(episode_env_config)
                _configure_episode(env, spec)
                try:
                    rollout = collect_joint_episode(
                        env,
                        max_steps=config.max_episode_steps,
                        reset_seed=spec.spawn_seed,
                    )
                except JointCollectionError as exc:
                    reason = getattr(exc, "reason_code", "collection_contract")
                    store.record_rejected_episode(episode_index, reason)
                    bundle.finalize(
                        BundleEpisodeResult(
                            episode_index=episode_index,
                            split=split,
                            scenario_id=spec.scenario_id,
                            local_route=spec.local_route,
                            spawn_seed=spec.spawn_seed,
                            base_status="rejected",
                            base_rejection_reason=reason,
                            sidecar_status="rejected",
                            sidecar_rejection_reason=reason,
                            raw_steps=0,
                            base_samples=0,
                            outcome="collection_failed_before_raw_state",
                        )
                    )
                    print(
                        f"[WARNING] episode={episode_index} split="
                        f"{store.assigner.split_for_episode(episode_index)} "
                        f"scenario={spec.scenario_id} rejected={reason}: {exc}",
                        flush=True,
                    )
                    continue

                samples_to_store = rollout.samples
                selected_steps: tuple[int, ...] = tuple(
                    int(value) for value in rollout.sample_step_indices
                )
                trigger_step = rollout.scenario_summary.get(
                    "scenario_trigger_step"
                )
                base_rejection_reason = rollout.failure_reason
                if base_rejection_reason is None and not rollout.samples:
                    base_rejection_reason = "no_joint_samples"
                if diagnostic_quotas is not None and base_rejection_reason is None:
                    remaining = (
                        diagnostic_quotas[spec.scenario_id]
                        - diagnostic_counts[spec.scenario_id]
                    )
                    try:
                        (
                            samples_to_store,
                            selected_steps,
                            trigger_step,
                        ) = _select_diagnostic_samples(
                            rollout,
                            decision_dt_s=simulator_decision_dt_s(env),
                            offsets_s=config.diagnostic_sample_offsets_s,
                            remaining=remaining,
                        )
                    except JointCollectionError as exc:
                        base_rejection_reason = exc.reason_code
                base_eligible = base_rejection_reason is None
                mapping = selected_steps if base_eligible else ()
                decision_dt_s = simulator_decision_dt_s(env)
                try:
                    sidecar_prepared = _prepare_rollout_sidecar(
                        sidecar_store,
                        start=_sidecar_start(
                            episode_index=episode_index,
                            split=split,
                            spec=spec,
                            rollout=rollout,
                            base_dataset_fingerprint=base_fingerprint,
                            decision_dt_s=decision_dt_s,
                        ),
                        rollout=rollout,
                        base_sample_step_indices=mapping,
                    )
                except RiskEntrySidecarStorageError as exc:
                    # No base commit is allowed without a valid sidecar.  If an
                    # episode directory appeared, leave the pending intent for
                    # conservative resume recovery instead of guessing whether
                    # the atomic commit completed.
                    if (
                        _stored_component_episode(sidecar_store, episode_index)
                        is not None
                        or sidecar_store.recover_prepared_episode(
                            episode_index, split
                        )
                        is not None
                    ):
                        raise
                    try:
                        sidecar_store.reject_episode(
                            reason_code="sidecar_data_integrity_invalid"
                        )
                    except RiskEntrySidecarStorageError:
                        pass
                    reason = "sidecar_data_integrity_invalid"
                    store.record_rejected_episode(episode_index, reason)
                    bundle.finalize(
                        BundleEpisodeResult(
                            episode_index=episode_index,
                            split=split,
                            scenario_id=spec.scenario_id,
                            local_route=spec.local_route,
                            spawn_seed=spec.spawn_seed,
                            base_status="rejected",
                            base_rejection_reason=reason,
                            sidecar_status="rejected",
                            sidecar_rejection_reason=reason,
                            raw_steps=0,
                            base_samples=0,
                            outcome="sidecar_data_integrity_invalid",
                        )
                    )
                    print(
                        f"[WARNING] episode={episode_index} scenario={spec.scenario_id} "
                        f"rejected={reason}: {exc}",
                        flush=True,
                    )
                    continue

                stored = None
                if base_eligible:
                    stored = store.commit_episode(
                        episode_index,
                        samples_to_store,
                        {
                            "scenario_id": spec.scenario_id,
                            "local_route": spec.local_route,
                            "spawn_seed": spec.spawn_seed,
                            "traffic_density": spec.traffic_density,
                            "initial_speed_km_h": spec.initial_speed_km_h,
                            "simulator_steps": rollout.simulator_steps,
                            "rejected_joint_steps": rollout.rejected_joint_steps,
                            "joint_step_rejection_counts": dict(
                                rollout.joint_step_rejection_counts
                            ),
                            "terminated": rollout.terminated,
                            "truncated": rollout.truncated,
                            "scenario_trigger_step": trigger_step,
                            "scenario_realized_step": (
                                rollout.scenario_summary.get(
                                    "scenario_realized_step"
                                )
                            ),
                            "selected_sample_steps": list(selected_steps),
                            "decision_dt_s": decision_dt_s,
                            "raw_timeline_length": sidecar_prepared.raw_steps,
                            "sidecar_dataset_fingerprint": (
                                sidecar_store.dataset_fingerprint
                            ),
                            "diagnostic_subsampled": (
                                diagnostic_quotas is not None
                            ),
                            "scenario_contract_sha256": (
                                primary_scenario_contract()["sha256"]
                            ),
                        },
                    )
                else:
                    store.record_rejected_episode(
                        episode_index, str(base_rejection_reason)
                    )

                sidecar_episode = sidecar_store.commit_prepared_episode()

                bundle.finalize(
                    BundleEpisodeResult(
                        episode_index=episode_index,
                        split=split,
                        scenario_id=spec.scenario_id,
                        local_route=spec.local_route,
                        spawn_seed=spec.spawn_seed,
                        base_status="committed" if stored is not None else "rejected",
                        base_rejection_reason=(
                            None if stored is not None else str(base_rejection_reason)
                        ),
                        sidecar_status="committed",
                        sidecar_rejection_reason=None,
                        raw_steps=sidecar_episode.raw_steps,
                        base_samples=0 if stored is None else stored.joint_samples,
                        outcome=sidecar_episode.outcome,
                    )
                )
                if stored is None:
                    print(
                        f"[WARNING] episode={episode_index} scenario={spec.scenario_id} "
                        f"base_rejected={base_rejection_reason} sidecar=committed",
                        flush=True,
                    )
                    continue
                if diagnostic_counts is not None:
                    diagnostic_counts[spec.scenario_id] += stored.joint_samples
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
            if env is not None:
                env.close()
        summary = {
            "base": store.summary(),
            "sidecar": sidecar_store.summary(),
            "bundle_attempts": bundle.next_episode_index,
        }
        if diagnostic_quotas is not None:
            summary["diagnostic_64"] = {
                "scenario_quotas": diagnostic_quotas,
                "scenario_counts": diagnostic_counts,
                "attempts": diagnostic_attempts,
                "scenario_contract_sha256": primary_scenario_contract()[
                    "sha256"
                ],
            }
            if diagnostic_counts != diagnostic_quotas:
                raise JointCollectionError(
                    "diagnostic collection ended before all quotas were met",
                    reason_code="diagnostic_quota_incomplete",
                )

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
    parser.add_argument("--sidecar-root", type=Path)
    parser.add_argument("--target-joint-steps", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-episode-steps", type=int)
    parser.add_argument("--resume", choices=("0", "1"))
    args = parser.parse_args(argv)
    config = load_run_config(args.config)
    overrides = {}
    if args.dataset_root is not None:
        dataset_root = args.dataset_root.expanduser().resolve()
        overrides["dataset_root"] = dataset_root
        overrides["bundle_root"] = dataset_root.parent
        if args.sidecar_root is None:
            overrides["sidecar_root"] = dataset_root.parent / "riskentry_actor_sidecar"
    if args.sidecar_root is not None:
        sidecar_root = args.sidecar_root.expanduser().resolve()
        overrides["sidecar_root"] = sidecar_root
        if args.dataset_root is None:
            overrides["bundle_root"] = sidecar_root.parent
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
