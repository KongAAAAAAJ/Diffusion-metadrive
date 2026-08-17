"""Collect and persist the three-vehicle joint-first BEV expert dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
import traceback
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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
    EpisodeSplitAssigner,
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
    FORMAL_V1_CONTRACT_ID,
    PRIMARY_S5_S9_SCENARIOS,
    primary_scenario_contract,
    scenario_contract_for_id,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL_KEYS = {
    "dataset",
    "split",
    "collection",
    "env_config",
    "diagnostic_64",
    "formal_pilot",
    "targeted_supplement",
}
SECTION_KEYS = {
    "dataset": {"name", "sidecar_name", "output_root", "scenario_contract"},
    "split": {"train_ratio", "val_ratio", "test_ratio", "seed"},
    "collection": {
        "target_joint_steps",
        "start_seed",
        "max_episodes",
        "max_episode_steps",
        "scenario_max_episode_steps",
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
    "formal_pilot": {
        "enabled",
        "scenario_quotas",
        "max_attempts_per_scenario",
        "min_episodes_per_scenario",
        "max_samples_per_episode",
        "require_all_scenarios_in_each_split",
        "required_spawn_seeds",
        "required_incidental_background_actor_counts",
        "required_behavior_categories",
        "behavior_rule_maker_profiles",
    },
    "targeted_supplement": {
        "enabled",
        "scenario_id",
        "behavior_category",
        "rule_maker_profile_id",
        "accepted_episode_quotas",
        "required_samples_per_episode",
        "max_simulator_attempts",
        "initial_feasibility_attempts",
        "minimum_lateral_range_m",
        "maximum_return_error_m",
        "accepted_background_count_quotas",
        "bootstrap_spawn_seeds",
        "parallel_workers",
    },
}

FORMAL_SPLITS = ("train", "val", "test")
FORMAL_BEHAVIOR_SOURCE = {
    "S5_hard_brake_lead": ("conflict_evidence", "observed_behavior_class"),
    "S6_background_merge_in": ("conflict_evidence", "target_gap_id"),
    "S7_ego_merge_from_ramp": ("conflict_evidence", "observed_behavior"),
    "S8_ego_exit_to_ramp": (
        "conflict_evidence",
        "exit_constraint_target_gap_id",
    ),
    "S9_narrow_channel_negotiation": (
        "conflict_evidence",
        "designated_split_actor_target_gap_id",
    ),
}


@dataclass(frozen=True)
class FormalDiversityRequirements:
    min_episodes_per_scenario: int
    max_samples_per_episode: int
    require_all_scenarios_in_each_split: bool
    required_spawn_seeds: tuple[int, ...]
    required_incidental_background_actor_counts: tuple[int, ...]
    required_behavior_categories: Mapping[str, tuple[str, ...]]
    behavior_rule_maker_profiles: Mapping[str, Mapping[str, str]]

    def as_dict(self) -> dict[str, object]:
        return {
            "min_episodes_per_scenario": self.min_episodes_per_scenario,
            "max_samples_per_episode": self.max_samples_per_episode,
            "require_all_scenarios_in_each_split": (
                self.require_all_scenarios_in_each_split
            ),
            "required_spawn_seeds": list(self.required_spawn_seeds),
            "required_incidental_background_actor_counts": list(
                self.required_incidental_background_actor_counts
            ),
            "required_behavior_categories": {
                name: list(values)
                for name, values in self.required_behavior_categories.items()
            },
            "behavior_rule_maker_profiles": {
                name: dict(values)
                for name, values in self.behavior_rule_maker_profiles.items()
            },
        }


@dataclass(frozen=True)
class TargetedSupplementRequirements:
    scenario_id: str
    behavior_category: str
    rule_maker_profile_id: str
    accepted_episode_quotas: Mapping[str, int]
    required_samples_per_episode: int
    max_simulator_attempts: int
    initial_feasibility_attempts: int
    minimum_lateral_range_m: float
    maximum_return_error_m: float
    accepted_background_count_quotas: Mapping[int, int]
    bootstrap_spawn_seeds: tuple[int, ...] = ()
    parallel_workers: int = 1

    @property
    def target_episodes(self) -> int:
        return sum(int(value) for value in self.accepted_episode_quotas.values())

    def as_dict(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "behavior_category": self.behavior_category,
            "rule_maker_profile_id": self.rule_maker_profile_id,
            "accepted_episode_quotas": dict(self.accepted_episode_quotas),
            "required_samples_per_episode": self.required_samples_per_episode,
            "max_simulator_attempts": self.max_simulator_attempts,
            "initial_feasibility_attempts": self.initial_feasibility_attempts,
            "minimum_lateral_range_m": self.minimum_lateral_range_m,
            "maximum_return_error_m": self.maximum_return_error_m,
            "accepted_background_count_quotas": {
                str(key): int(value)
                for key, value in self.accepted_background_count_quotas.items()
            },
            "bootstrap_spawn_seeds": list(self.bootstrap_spawn_seeds),
            "parallel_workers": self.parallel_workers,
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
    scenario_max_episode_steps: Mapping[str, int]
    resume: bool
    scenario_weights: Mapping[str, float]
    traffic_density_min: float
    traffic_density_max: float
    env_config: Mapping[str, object]
    diagnostic_scenario_quotas: Mapping[str, int] | None = None
    diagnostic_sample_offsets_s: tuple[float, ...] = ()
    diagnostic_max_attempts_per_scenario: int = 0
    formal_scenario_quotas: Mapping[str, int] | None = None
    formal_max_attempts_per_scenario: int = 0
    formal_diversity: FormalDiversityRequirements | None = None
    targeted_supplement: TargetedSupplementRequirements | None = None
    scenario_contract_id: str = FORMAL_V1_CONTRACT_ID

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
        scenario_steps = {}
        for scenario_id, value in self.scenario_max_episode_steps.items():
            if scenario_id not in SCENARIO_BY_ID:
                raise ValueError(
                    f"unknown scenario in scenario_max_episode_steps: {scenario_id}"
                )
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError(
                    "collection.scenario_max_episode_steps values must be positive integers"
                )
            scenario_steps[str(scenario_id)] = int(value)
        object.__setattr__(self, "scenario_max_episode_steps", scenario_steps)
        scenario_contract_for_id(self.scenario_contract_id)
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
        formal_quotas = self.formal_scenario_quotas
        enabled_modes = sum(
            value is not None
            for value in (quotas, formal_quotas, self.targeted_supplement)
        )
        if enabled_modes > 1:
            raise ValueError(
                "diagnostic_64, formal_pilot and targeted_supplement are mutually exclusive"
            )
        if formal_quotas is not None:
            expected = tuple(value[0] for value in PRIMARY_S5_S9_SCENARIOS)
            if tuple(formal_quotas) != expected:
                raise ValueError(
                    "formal_pilot scenario_quotas must use ordered S5--S9"
                )
            normalized_formal_quotas = {}
            for scenario_id, value in formal_quotas.items():
                if isinstance(value, bool) or int(value) <= 0:
                    raise ValueError(
                        "formal_pilot scenario quotas must be positive integers"
                    )
                normalized_formal_quotas[str(scenario_id)] = int(value)
            if sum(normalized_formal_quotas.values()) != self.target_joint_steps:
                raise ValueError(
                    "formal_pilot quotas must sum to target_joint_steps"
                )
            if int(self.formal_max_attempts_per_scenario) <= 0:
                raise ValueError(
                    "formal_pilot max_attempts_per_scenario must be positive"
                )
            if not self.resume:
                raise ValueError("formal_pilot collection requires resume=true")
            object.__setattr__(
                self, "formal_scenario_quotas", normalized_formal_quotas
            )
            object.__setattr__(
                self,
                "formal_max_attempts_per_scenario",
                int(self.formal_max_attempts_per_scenario),
            )
            diversity = self.formal_diversity
            if diversity is not None:
                if diversity.min_episodes_per_scenario <= 0:
                    raise ValueError(
                        "formal_pilot min_episodes_per_scenario must be positive"
                    )
                if diversity.max_samples_per_episode <= 0:
                    raise ValueError(
                        "formal_pilot max_samples_per_episode must be positive"
                    )
                if not diversity.require_all_scenarios_in_each_split:
                    raise ValueError(
                        "formal_pilot must require all S5--S9 in every split"
                    )
                if (
                    not diversity.required_spawn_seeds
                    or len(set(diversity.required_spawn_seeds))
                    != len(diversity.required_spawn_seeds)
                    or any(value < 0 for value in diversity.required_spawn_seeds)
                ):
                    raise ValueError(
                        "formal_pilot required_spawn_seeds must be unique non-negative integers"
                    )
                if (
                    tuple(sorted(set(
                        diversity.required_incidental_background_actor_counts
                    )))
                    != diversity.required_incidental_background_actor_counts
                    or not diversity.required_incidental_background_actor_counts
                    or any(
                        value < 3 or value > 6
                        for value in diversity.required_incidental_background_actor_counts
                    )
                ):
                    raise ValueError(
                        "formal_pilot background coverage must be an ordered subset of 3--6"
                    )
                if tuple(diversity.required_behavior_categories) != expected:
                    raise ValueError(
                        "formal_pilot behavior coverage must use ordered S5--S9"
                    )
                for scenario_id, values in (
                    diversity.required_behavior_categories.items()
                ):
                    if (
                        not values
                        or len(set(values)) != len(values)
                        or any(not value for value in values)
                    ):
                        raise ValueError(
                            f"formal_pilot behavior coverage is invalid for {scenario_id}"
                        )
                for scenario_id, profiles in (
                    diversity.behavior_rule_maker_profiles.items()
                ):
                    if scenario_id not in diversity.required_behavior_categories:
                        raise ValueError(
                            "formal_pilot behavior profile has an unknown scenario: "
                            f"{scenario_id}"
                        )
                    unknown = set(profiles) - set(
                        diversity.required_behavior_categories[scenario_id]
                    )
                    if unknown or any(not value for value in profiles.values()):
                        raise ValueError(
                            "formal_pilot behavior profile mapping is invalid for "
                            f"{scenario_id}: {sorted(unknown)}"
                        )
                for scenario_id, quota in normalized_formal_quotas.items():
                    if (
                        diversity.min_episodes_per_scenario
                        * diversity.max_samples_per_episode
                        < quota
                    ):
                        raise ValueError(
                            "formal_pilot episode minimum/sample cap cannot satisfy "
                            f"the quota for {scenario_id}"
                        )
                if (
                    len(diversity.required_spawn_seeds)
                    > diversity.min_episodes_per_scenario
                ):
                    raise ValueError(
                        "formal_pilot required seed count exceeds the episode minimum"
                    )
        elif self.formal_diversity is not None:
            raise ValueError(
                "formal_pilot diversity constraints require formal_pilot.enabled=true"
            )
        targeted = self.targeted_supplement
        if targeted is not None:
            if targeted.scenario_id != "S5_hard_brake_lead":
                raise ValueError("targeted_supplement supports only S5_hard_brake_lead")
            if set(self.scenario_weights) != {targeted.scenario_id}:
                raise ValueError(
                    "targeted_supplement scenario_weights must contain only its scenario"
                )
            if targeted.behavior_category != (
                "temporary_formation_release_and_recovery"
            ):
                raise ValueError(
                    "targeted_supplement behavior_category is outside the frozen contract"
                )
            if targeted.rule_maker_profile_id != "balanced":
                raise ValueError("targeted_supplement requires the balanced profile")
            if tuple(targeted.accepted_episode_quotas) != FORMAL_SPLITS:
                raise ValueError(
                    "targeted_supplement accepted_episode_quotas must use train/val/test order"
                )
            if any(
                isinstance(value, bool) or int(value) < 0
                for value in targeted.accepted_episode_quotas.values()
            ) or targeted.target_episodes <= 0:
                raise ValueError(
                    "targeted_supplement split quotas must be non-negative and non-empty"
                )
            if targeted.required_samples_per_episode <= 0:
                raise ValueError(
                    "targeted_supplement required_samples_per_episode must be positive"
                )
            if self.target_joint_steps != (
                targeted.target_episodes * targeted.required_samples_per_episode
            ):
                raise ValueError(
                    "targeted_supplement episode/sample quotas must equal target_joint_steps"
                )
            if targeted.initial_feasibility_attempts <= 0:
                raise ValueError(
                    "targeted_supplement initial_feasibility_attempts must be positive"
                )
            if (
                targeted.max_simulator_attempts
                < targeted.initial_feasibility_attempts
            ):
                raise ValueError(
                    "targeted_supplement max attempts must cover the feasibility gate"
                )
            if not self.resume:
                raise ValueError("targeted_supplement collection requires resume=true")
            if (
                not np.isfinite(targeted.minimum_lateral_range_m)
                or targeted.minimum_lateral_range_m <= 0.0
                or not np.isfinite(targeted.maximum_return_error_m)
                or targeted.maximum_return_error_m < 0.0
            ):
                raise ValueError("targeted_supplement physical thresholds are invalid")
            background_quotas = dict(targeted.accepted_background_count_quotas)
            if (
                not background_quotas
                or any(key not in (3, 4, 5, 6) for key in background_quotas)
                or any(
                    isinstance(value, bool) or int(value) <= 0
                    for value in background_quotas.values()
                )
                or sum(background_quotas.values()) != targeted.target_episodes
            ):
                raise ValueError(
                    "targeted_supplement background quotas must cover all accepted episodes"
                )
            if (
                len(set(targeted.bootstrap_spawn_seeds))
                != len(targeted.bootstrap_spawn_seeds)
                or any(value < 0 for value in targeted.bootstrap_spawn_seeds)
            ):
                raise ValueError(
                    "targeted_supplement bootstrap seeds must be unique non-negative integers"
                )
            if (
                isinstance(targeted.parallel_workers, bool)
                or not 1 <= int(targeted.parallel_workers) <= 4
            ):
                raise ValueError(
                    "targeted_supplement parallel_workers must be an integer in [1, 4]"
                )

    def immutable_fingerprint(self) -> str:
        return fingerprint_payload(
            {
                "collector": "joint_bev_rule_planner",
                "scenario_contract": self.scenario_contract(),
                "start_seed": self.start_seed,
                "max_episode_steps": self.max_episode_steps,
                "scenario_max_episode_steps": dict(self.scenario_max_episode_steps),
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
                        "scenario_contract": self.scenario_contract(),
                    }
                ),
                "formal_pilot": (
                    None
                    if self.formal_scenario_quotas is None
                    else {
                        "scenario_quotas": dict(self.formal_scenario_quotas),
                        "max_attempts_per_scenario": (
                            self.formal_max_attempts_per_scenario
                        ),
                        "scenario_contract": self.scenario_contract(),
                        **(
                            {}
                            if self.formal_diversity is None
                            else {"diversity": self.formal_diversity.as_dict()}
                        ),
                    }
                ),
                "targeted_supplement": (
                    None
                    if self.targeted_supplement is None
                    else self.targeted_supplement.as_dict()
                ),
            }
        )

    def scenario_contract(self) -> dict[str, object]:
        return scenario_contract_for_id(self.scenario_contract_id)

    def episode_step_limit(self, scenario_id: str) -> int:
        return int(
            self.scenario_max_episode_steps.get(
                str(scenario_id), self.max_episode_steps
            )
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
    formal = _strict_section(payload, "formal_pilot", required=False)
    targeted = _strict_section(
        payload, "targeted_supplement", required=False
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
    scenario_steps = collection.get("scenario_max_episode_steps", {})
    if not isinstance(scenario_steps, Mapping):
        raise ValueError("collection.scenario_max_episode_steps must be a mapping")
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

    formal_enabled = formal.get("enabled", False)
    if not isinstance(formal_enabled, bool):
        raise ValueError("formal_pilot.enabled must be bool")
    formal_quotas = None
    formal_max_attempts = 0
    formal_diversity = None
    if formal_enabled:
        raw_formal_quotas = _required(
            formal, "formal_pilot", "scenario_quotas"
        )
        if not isinstance(raw_formal_quotas, Mapping):
            raise ValueError("formal_pilot.scenario_quotas must be a mapping")
        formal_quotas = {
            str(name): int(value) for name, value in raw_formal_quotas.items()
        }
        formal_max_attempts = int(
            _required(
                formal,
                "formal_pilot",
                "max_attempts_per_scenario",
            )
        )
        diversity_fields = (
            "min_episodes_per_scenario",
            "max_samples_per_episode",
            "require_all_scenarios_in_each_split",
            "required_spawn_seeds",
            "required_incidental_background_actor_counts",
            "required_behavior_categories",
        )
        supplied_diversity_fields = [
            name for name in diversity_fields if name in formal
        ]
        if supplied_diversity_fields and len(supplied_diversity_fields) != len(
            diversity_fields
        ):
            missing = sorted(set(diversity_fields) - set(supplied_diversity_fields))
            raise ValueError(
                f"formal_pilot diversity constraints must be supplied together; missing={missing}"
            )
        if supplied_diversity_fields:
            raw_required_seeds = formal["required_spawn_seeds"]
            raw_background_counts = formal[
                "required_incidental_background_actor_counts"
            ]
            raw_behavior = formal["required_behavior_categories"]
            raw_behavior_profiles = formal.get(
                "behavior_rule_maker_profiles", {}
            )
            if not isinstance(raw_required_seeds, (list, tuple)):
                raise ValueError("formal_pilot.required_spawn_seeds must be a list")
            if not isinstance(raw_background_counts, (list, tuple)):
                raise ValueError(
                    "formal_pilot.required_incidental_background_actor_counts must be a list"
                )
            if not isinstance(raw_behavior, Mapping):
                raise ValueError(
                    "formal_pilot.required_behavior_categories must be a mapping"
                )
            if not isinstance(raw_behavior_profiles, Mapping):
                raise ValueError(
                    "formal_pilot.behavior_rule_maker_profiles must be a mapping"
                )
            behavior_categories = {}
            for scenario_id, values in raw_behavior.items():
                if not isinstance(values, (list, tuple)):
                    raise ValueError(
                        "formal_pilot behavior category values must be lists"
                    )
                behavior_categories[str(scenario_id)] = tuple(
                    str(value) for value in values
                )
            behavior_profiles = {}
            for scenario_id, values in raw_behavior_profiles.items():
                if not isinstance(values, Mapping):
                    raise ValueError(
                        "formal_pilot behavior profile values must be mappings"
                    )
                behavior_profiles[str(scenario_id)] = {
                    str(category): str(profile)
                    for category, profile in values.items()
                }
            split_coverage = formal["require_all_scenarios_in_each_split"]
            if not isinstance(split_coverage, bool):
                raise ValueError(
                    "formal_pilot.require_all_scenarios_in_each_split must be bool"
                )
            formal_diversity = FormalDiversityRequirements(
                min_episodes_per_scenario=int(
                    formal["min_episodes_per_scenario"]
                ),
                max_samples_per_episode=int(formal["max_samples_per_episode"]),
                require_all_scenarios_in_each_split=split_coverage,
                required_spawn_seeds=tuple(int(value) for value in raw_required_seeds),
                required_incidental_background_actor_counts=tuple(
                    int(value) for value in raw_background_counts
                ),
                required_behavior_categories=behavior_categories,
                behavior_rule_maker_profiles=behavior_profiles,
            )

    targeted_enabled = targeted.get("enabled", False)
    if not isinstance(targeted_enabled, bool):
        raise ValueError("targeted_supplement.enabled must be bool")
    targeted_requirements = None
    if targeted_enabled:
        raw_split_quotas = _required(
            targeted, "targeted_supplement", "accepted_episode_quotas"
        )
        raw_background_quotas = _required(
            targeted,
            "targeted_supplement",
            "accepted_background_count_quotas",
        )
        raw_bootstrap_seeds = targeted.get("bootstrap_spawn_seeds", [])
        if not isinstance(raw_split_quotas, Mapping):
            raise ValueError(
                "targeted_supplement.accepted_episode_quotas must be a mapping"
            )
        if not isinstance(raw_background_quotas, Mapping):
            raise ValueError(
                "targeted_supplement.accepted_background_count_quotas must be a mapping"
            )
        if not isinstance(raw_bootstrap_seeds, (list, tuple)):
            raise ValueError(
                "targeted_supplement.bootstrap_spawn_seeds must be a list"
            )
        targeted_requirements = TargetedSupplementRequirements(
            scenario_id=str(
                _required(targeted, "targeted_supplement", "scenario_id")
            ),
            behavior_category=str(
                _required(
                    targeted, "targeted_supplement", "behavior_category"
                )
            ),
            rule_maker_profile_id=str(
                _required(
                    targeted, "targeted_supplement", "rule_maker_profile_id"
                )
            ),
            accepted_episode_quotas={
                str(name): int(value) for name, value in raw_split_quotas.items()
            },
            required_samples_per_episode=int(
                _required(
                    targeted,
                    "targeted_supplement",
                    "required_samples_per_episode",
                )
            ),
            max_simulator_attempts=int(
                _required(
                    targeted,
                    "targeted_supplement",
                    "max_simulator_attempts",
                )
            ),
            initial_feasibility_attempts=int(
                _required(
                    targeted,
                    "targeted_supplement",
                    "initial_feasibility_attempts",
                )
            ),
            minimum_lateral_range_m=float(
                _required(
                    targeted,
                    "targeted_supplement",
                    "minimum_lateral_range_m",
                )
            ),
            maximum_return_error_m=float(
                _required(
                    targeted,
                    "targeted_supplement",
                    "maximum_return_error_m",
                )
            ),
            accepted_background_count_quotas={
                int(name): int(value)
                for name, value in raw_background_quotas.items()
            },
            bootstrap_spawn_seeds=tuple(
                int(value) for value in raw_bootstrap_seeds
            ),
            parallel_workers=int(
                targeted.get("parallel_workers", 1)
            ),
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
        scenario_max_episode_steps={
            str(name): int(value) for name, value in scenario_steps.items()
        },
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
        formal_scenario_quotas=formal_quotas,
        formal_max_attempts_per_scenario=formal_max_attempts,
        formal_diversity=formal_diversity,
        targeted_supplement=targeted_requirements,
        scenario_contract_id=str(
            dataset.get("scenario_contract", FORMAL_V1_CONTRACT_ID)
        ),
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
    *,
    spawn_seed: int | None = None,
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
        spawn_seed=(
            int(rng.randint(0, 2**31 - 1))
            if spawn_seed is None
            else int(spawn_seed)
        ),
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


def _select_formal_quota_samples(
    rollout,
    *,
    remaining: int,
    max_samples: int | None = None,
) -> tuple[tuple[object, ...], tuple[int, ...]]:
    """Keep a complete episode, clipping only its persisted sample view."""

    if remaining <= 0:
        raise JointCollectionError(
            "formal scenario quota is already complete",
            reason_code="formal_pilot_quota_complete",
        )
    sample_count = len(rollout.samples)
    if sample_count != len(rollout.sample_step_indices):
        raise JointCollectionError(
            "sample/step metadata are misaligned",
            reason_code="formal_pilot_sample_alignment",
        )
    if sample_count == 0:
        raise JointCollectionError(
            "formal episode selected no samples",
            reason_code="formal_pilot_sample_missing",
        )
    selected_count = min(
        sample_count,
        int(remaining),
        sample_count if max_samples is None else int(max_samples),
    )
    if selected_count <= 0:
        raise JointCollectionError(
            "formal episode has no diversity-safe sample capacity",
            reason_code="formal_pilot_diversity_capacity",
        )
    if selected_count == sample_count:
        indices = tuple(range(sample_count))
    else:
        indices = tuple(
            int(value)
            for value in np.linspace(
                0,
                sample_count - 1,
                num=selected_count,
                dtype=np.int64,
            )
        )
        if len(set(indices)) != selected_count:
            raise JointCollectionError(
                "formal sample clipping produced duplicate indices",
                reason_code="formal_pilot_sample_alignment",
            )
    return (
        tuple(rollout.samples[index] for index in indices),
        tuple(int(rollout.sample_step_indices[index]) for index in indices),
    )


def _stored_scenario_counts(
    store: JointBEVDatasetStore,
    quotas: Mapping[str, int],
) -> dict[str, int]:
    counts = {name: 0 for name in quotas}
    for writer in store.writers.values():
        for episode in writer.episodes.values():
            scenario_id = str(episode.attributes.get("scenario_id", ""))
            if scenario_id not in counts:
                raise JointCollectionError(
                    f"formal pilot contains unexpected scenario {scenario_id!r}",
                    reason_code="formal_pilot_resume_mismatch",
                )
            counts[scenario_id] += int(episode.joint_samples)
    for scenario_id, count in counts.items():
        if count > int(quotas[scenario_id]):
            raise JointCollectionError(
                f"formal pilot scenario {scenario_id} exceeds its quota",
                reason_code="formal_pilot_resume_mismatch",
            )
    if sum(counts.values()) != store.total_joint_samples:
        raise JointCollectionError(
            "formal pilot scenario counts do not match stored sample count",
            reason_code="formal_pilot_resume_mismatch",
        )
    return counts


def _formal_episode_coverage(
    scenario_id: str,
    scenario_summary: Mapping[str, object],
) -> dict[str, object]:
    resolved = scenario_summary.get("resolved_scenario_parameters", {})
    evidence = scenario_summary.get("conflict_evidence", {})
    if not isinstance(resolved, Mapping) or not isinstance(evidence, Mapping):
        raise JointCollectionError(
            "formal episode has malformed scenario coverage metadata",
            reason_code="formal_pilot_coverage_metadata_missing",
        )
    background_count = resolved.get("incidental_background_actor_count")
    realized_background_count = evidence.get(
        "incidental_background_realized_count"
    )
    source_section, source_key = FORMAL_BEHAVIOR_SOURCE[scenario_id]
    source = evidence if source_section == "conflict_evidence" else resolved
    behavior_category = source.get(source_key)
    if (
        isinstance(background_count, bool)
        or not isinstance(background_count, (int, np.integer))
        or isinstance(realized_background_count, bool)
        or not isinstance(realized_background_count, (int, np.integer))
        or int(realized_background_count) != int(background_count)
        or behavior_category in (None, "")
    ):
        raise JointCollectionError(
            "formal episode is missing background or behavior coverage metadata",
            reason_code="formal_pilot_coverage_metadata_missing",
        )
    return {
        "incidental_background_actor_count": int(background_count),
        "behavior_category": str(behavior_category),
    }


def _empty_formal_progress(quotas: Mapping[str, int]) -> dict[str, dict[str, object]]:
    return {
        scenario_id: {
            "joint_samples": 0,
            "episodes": 0,
            "spawn_seeds": set(),
            "splits": set(),
            "incidental_background_actor_counts": set(),
            "behavior_categories": set(),
        }
        for scenario_id in quotas
    }


def _stored_formal_progress(
    store: JointBEVDatasetStore,
    quotas: Mapping[str, int],
    requirements: FormalDiversityRequirements | None,
) -> dict[str, dict[str, object]]:
    progress = _empty_formal_progress(quotas)
    for writer in store.writers.values():
        for episode in writer.episodes.values():
            scenario_id = str(episode.attributes.get("scenario_id", ""))
            if scenario_id not in progress:
                raise JointCollectionError(
                    f"formal pilot contains unexpected scenario {scenario_id!r}",
                    reason_code="formal_pilot_resume_mismatch",
                )
            row = progress[scenario_id]
            row["joint_samples"] += int(episode.joint_samples)
            row["episodes"] += 1
            row["splits"].add(str(episode.split))
            spawn_seed = episode.attributes.get("spawn_seed")
            if isinstance(spawn_seed, bool) or not isinstance(
                spawn_seed, (int, np.integer)
            ):
                raise JointCollectionError(
                    "formal episode has no valid spawn seed",
                    reason_code="formal_pilot_resume_mismatch",
                )
            if (
                requirements is not None
                and int(spawn_seed) in row["spawn_seeds"]
            ):
                raise JointCollectionError(
                    f"formal scenario {scenario_id} reuses spawn seed {spawn_seed}",
                    reason_code="formal_pilot_duplicate_spawn_seed",
                )
            row["spawn_seeds"].add(int(spawn_seed))
            if requirements is None:
                continue
            if episode.joint_samples > requirements.max_samples_per_episode:
                raise JointCollectionError(
                    f"formal episode {episode.episode_index} exceeds the sample cap",
                    reason_code="formal_pilot_episode_sample_cap_exceeded",
                )
            coverage = episode.attributes.get("formal_coverage")
            if not isinstance(coverage, Mapping):
                raise JointCollectionError(
                    "formal episode has no persisted coverage metadata",
                    reason_code="formal_pilot_resume_mismatch",
                )
            background_count = coverage.get("incidental_background_actor_count")
            behavior_category = coverage.get("behavior_category")
            if (
                isinstance(background_count, bool)
                or not isinstance(background_count, (int, np.integer))
                or behavior_category in (None, "")
            ):
                raise JointCollectionError(
                    "formal episode has malformed persisted coverage metadata",
                    reason_code="formal_pilot_resume_mismatch",
                )
            row["incidental_background_actor_counts"].add(int(background_count))
            row["behavior_categories"].add(str(behavior_category))
    for scenario_id, row in progress.items():
        if int(row["joint_samples"]) > int(quotas[scenario_id]):
            raise JointCollectionError(
                f"formal pilot scenario {scenario_id} exceeds its quota",
                reason_code="formal_pilot_resume_mismatch",
            )
    if sum(int(row["joint_samples"]) for row in progress.values()) != (
        store.total_joint_samples
    ):
        raise JointCollectionError(
            "formal pilot scenario counts do not match stored sample count",
            reason_code="formal_pilot_resume_mismatch",
        )
    return progress


def _serializable_formal_progress(
    progress: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    result = {}
    for scenario_id, row in progress.items():
        result[scenario_id] = {
            "joint_samples": int(row["joint_samples"]),
            "episodes": int(row["episodes"]),
            "spawn_seeds": sorted(row["spawn_seeds"]),
            "splits": sorted(row["splits"]),
            "incidental_background_actor_counts": sorted(
                row["incidental_background_actor_counts"]
            ),
            "behavior_categories": sorted(row["behavior_categories"]),
        }
    return result


def _validate_formal_completion(
    progress: Mapping[str, Mapping[str, object]],
    quotas: Mapping[str, int],
    requirements: FormalDiversityRequirements | None,
) -> None:
    counts = {
        scenario_id: int(row["joint_samples"])
        for scenario_id, row in progress.items()
    }
    if counts != dict(quotas):
        raise JointCollectionError(
            "formal pilot ended before all quotas were met",
            reason_code="formal_pilot_quota_incomplete",
        )
    if requirements is None:
        return
    required_splits = set(FORMAL_SPLITS)
    required_seeds = set(requirements.required_spawn_seeds)
    required_background = set(
        requirements.required_incidental_background_actor_counts
    )
    for scenario_id, row in progress.items():
        if int(row["episodes"]) < requirements.min_episodes_per_scenario:
            raise JointCollectionError(
                f"formal scenario {scenario_id} has too few independent episodes",
                reason_code="formal_pilot_episode_diversity_incomplete",
            )
        if requirements.require_all_scenarios_in_each_split and not (
            required_splits <= set(row["splits"])
        ):
            raise JointCollectionError(
                f"formal scenario {scenario_id} is missing from a dataset split",
                reason_code="formal_pilot_split_coverage_incomplete",
            )
        if not required_seeds <= set(row["spawn_seeds"]):
            raise JointCollectionError(
                f"formal scenario {scenario_id} is missing a required seed",
                reason_code="formal_pilot_seed_coverage_incomplete",
            )
        if not required_background <= set(
            row["incidental_background_actor_counts"]
        ):
            raise JointCollectionError(
                f"formal scenario {scenario_id} lacks background-count coverage",
                reason_code="formal_pilot_background_coverage_incomplete",
            )
        required_behavior = set(
            requirements.required_behavior_categories[scenario_id]
        )
        if not required_behavior <= set(row["behavior_categories"]):
            raise JointCollectionError(
                f"formal scenario {scenario_id} lacks behavior-category coverage",
                reason_code="formal_pilot_behavior_coverage_incomplete",
            )


def _select_formal_scenario(
    order: tuple[str, ...],
    progress: Mapping[str, Mapping[str, object]],
    quotas: Mapping[str, int],
    split: str,
) -> str:
    incomplete = [
        scenario_id
        for scenario_id in order
        if int(progress[scenario_id]["joint_samples"]) < int(quotas[scenario_id])
    ]
    if not incomplete:
        raise JointCollectionError(
            "formal scenario quotas are already complete",
            reason_code="formal_pilot_quota_complete",
        )
    missing_current_split = [
        scenario_id
        for scenario_id in incomplete
        if split not in progress[scenario_id]["splits"]
    ]
    candidates = missing_current_split or incomplete
    return min(
        candidates,
        key=lambda scenario_id: (
            int(progress[scenario_id]["joint_samples"])
            / int(quotas[scenario_id]),
            order.index(scenario_id),
        ),
    )


def _formal_sample_capacity(
    *,
    scenario_id: str,
    remaining: int,
    row: Mapping[str, object],
    requirements: FormalDiversityRequirements,
    split: str,
    spawn_seed: int,
    coverage: Mapping[str, object],
) -> int:
    episodes_after = int(row["episodes"]) + 1
    missing_episode_count = max(
        requirements.min_episodes_per_scenario - episodes_after,
        0,
    )
    splits_after = set(row["splits"]) | {split}
    seeds_after = set(row["spawn_seeds"]) | {spawn_seed}
    backgrounds_after = set(row["incidental_background_actor_counts"]) | {
        int(coverage["incidental_background_actor_count"])
    }
    behaviors_after = set(row["behavior_categories"]) | {
        str(coverage["behavior_category"])
    }
    coverage_still_missing = any(
        (
            requirements.require_all_scenarios_in_each_split
            and not set(FORMAL_SPLITS) <= splits_after,
            not set(requirements.required_spawn_seeds) <= seeds_after,
            not set(requirements.required_incidental_background_actor_counts)
            <= backgrounds_after,
            not set(requirements.required_behavior_categories[
                scenario_id
            ])
            <= behaviors_after,
        )
    )
    reserved_samples = max(missing_episode_count, int(coverage_still_missing))
    return min(
        requirements.max_samples_per_episode,
        int(remaining) - reserved_samples,
    )


def _formal_rule_maker_profile(
    *,
    scenario_id: str,
    row: Mapping[str, object],
    requirements: FormalDiversityRequirements,
) -> str | None:
    """Select the configured profile for the first still-missing behavior."""

    observed = set(row["behavior_categories"])
    profiles = requirements.behavior_rule_maker_profiles.get(scenario_id, {})
    for behavior in requirements.required_behavior_categories[scenario_id]:
        if behavior not in observed and behavior in profiles:
            return str(profiles[behavior])
    return None


TARGETED_SAFETY_EVENTS = frozenset(
    {"collision_vehicle", "collision_object", "collision_sidewalk", "out_of_road"}
)
TARGETED_LATERAL_MODES = frozenset(range(3, 9))
TARGETED_LEFT_MODES = frozenset(range(3, 6))
TARGETED_RIGHT_MODES = frozenset(range(6, 9))


def _count_true_runs(mask: np.ndarray) -> int:
    values = np.asarray(mask, dtype=bool).reshape(-1)
    if values.size == 0:
        return 0
    return int(values[0]) + int(np.count_nonzero(~values[:-1] & values[1:]))


def _lateral_run_directions(modes: np.ndarray) -> list[str]:
    values = np.asarray(modes, dtype=np.int64).reshape(-1)
    lateral = np.isin(values, tuple(TARGETED_LATERAL_MODES))
    starts = np.flatnonzero(lateral & np.concatenate(([True], ~lateral[:-1])))
    stops = np.flatnonzero(lateral & np.concatenate((~lateral[1:], [True]))) + 1
    directions = []
    for start, stop in zip(starts, stops):
        run = set(int(value) for value in values[start:stop])
        if run <= TARGETED_LEFT_MODES:
            directions.append("left")
        elif run <= TARGETED_RIGHT_MODES:
            directions.append("right")
        else:
            directions.append("mixed")
    return directions


def _validate_targeted_rollout(
    rollout,
    requirements: TargetedSupplementRequirements,
) -> dict[str, object]:
    """Validate label, physical motion and safety before any base commit."""

    if rollout.failure_reason is not None:
        raise JointCollectionError(
            f"targeted rollout failed: {rollout.failure_reason}",
            reason_code="targeted_trajectory_infeasible",
        )
    if len(rollout.samples) != requirements.required_samples_per_episode:
        raise JointCollectionError(
            "targeted rollout does not contain one complete episode",
            reason_code="targeted_incomplete_joint_episode",
        )
    if len(rollout.sample_step_indices) != len(rollout.samples):
        raise JointCollectionError(
            "targeted rollout sample/step metadata are misaligned",
            reason_code="targeted_incomplete_joint_episode",
        )
    summary = dict(rollout.scenario_summary)
    evidence = summary.get("conflict_evidence", {})
    if not isinstance(evidence, Mapping):
        raise JointCollectionError(
            "targeted rollout has no conflict evidence",
            reason_code="targeted_scenario_not_realized",
        )
    observed = evidence.get("observed_behavior_class")
    if observed != requirements.behavior_category:
        if bool(evidence.get("mixed_direction_lane_change_completed", False)):
            reason = "targeted_release_without_recovery"
        elif observed == "keep_emergency_braking":
            reason = "targeted_sustained_emergency_braking"
        else:
            reason = "targeted_scenario_not_realized"
        raise JointCollectionError(
            f"targeted behavior mismatch: observed={observed!r}",
            reason_code=reason,
        )
    if rollout.sidecar is None:
        raise JointCollectionError(
            "targeted rollout has no sidecar timeline",
            reason_code="targeted_safety_evidence_missing",
        )
    unsafe_events = []
    for capture in rollout.sidecar.captures:
        for event in capture.events:
            if event.event_type in TARGETED_SAFETY_EVENTS and any(
                actor_id in {"P0", "P1", "P2"} for actor_id in event.actor_ids
            ):
                unsafe_events.append(event.event_type)
    if unsafe_events:
        raise JointCollectionError(
            f"targeted rollout has platoon safety events: {sorted(set(unsafe_events))}",
            reason_code="targeted_safety_rejected",
        )
    required_flags = (
        "mixed_direction_lane_change_completed",
        "reassembly_completed",
        "reassembly_to_initial_lane_completed",
        "hazard_cleared",
        "formation_recovered_after_hazard",
    )
    if not all(bool(evidence.get(name, False)) for name in required_flags):
        raise JointCollectionError(
            "targeted rollout released formation but did not complete recovery",
            reason_code="targeted_release_without_recovery",
        )

    gt_mode = np.stack(
        [np.asarray(sample.gt_mode, dtype=np.int64) for sample in rollout.samples]
    )
    pose = np.stack(
        [
            np.asarray(sample.ego_pose_global, dtype=np.float64)
            for sample in rollout.samples
        ]
    )
    if gt_mode.shape != (len(rollout.samples), 3) or pose.shape != (
        len(rollout.samples),
        3,
        3,
    ):
        raise JointCollectionError(
            "targeted rollout has malformed physical evidence",
            reason_code="targeted_physical_evidence_invalid",
        )
    lateral_runs = []
    lateral_directions = []
    lateral_ranges = []
    return_errors = []
    for role_index in range(3):
        lateral = np.isin(gt_mode[:, role_index], tuple(TARGETED_LATERAL_MODES))
        lateral_runs.append(_count_true_runs(lateral))
        lateral_directions.append(
            _lateral_run_directions(gt_mode[:, role_index])
        )
        y = pose[:, role_index, 1]
        lateral_ranges.append(float(np.ptp(y)))
        return_errors.append(float(abs(y[-1] - y[0])))
    if lateral_runs != [2, 2, 2]:
        raise JointCollectionError(
            f"targeted gt_mode lateral runs are {lateral_runs}, expected [2, 2, 2]",
            reason_code="targeted_lateral_mode_segments_invalid",
        )
    if (
        any(
            len(directions) != 2
            or directions[0] == directions[1]
            or "mixed" in directions
            for directions in lateral_directions
        )
        or len({directions[0] for directions in lateral_directions}) < 2
    ):
        raise JointCollectionError(
            f"targeted lateral directions are not mixed release/return: {lateral_directions}",
            reason_code="targeted_lateral_directions_invalid",
        )
    if any(
        value < requirements.minimum_lateral_range_m
        for value in lateral_ranges
    ) or any(
        value > requirements.maximum_return_error_m
        for value in return_errors
    ):
        raise JointCollectionError(
            "targeted lateral displacement/recovery threshold was not met",
            reason_code="targeted_physical_displacement_invalid",
        )
    coverage = _formal_episode_coverage(
        requirements.scenario_id, rollout.scenario_summary
    )
    return {
        "behavior_category": str(coverage["behavior_category"]),
        "incidental_background_actor_count": int(
            coverage["incidental_background_actor_count"]
        ),
        "lateral_mode_runs_by_role": lateral_runs,
        "lateral_run_directions_by_role": lateral_directions,
        "lateral_range_m_by_role": lateral_ranges,
        "return_error_m_by_role": return_errors,
        "platoon_safety_events": [],
    }


def _empty_targeted_progress(
    requirements: TargetedSupplementRequirements,
) -> dict[str, object]:
    return {
        "accepted_episodes": 0,
        "split_counts": {name: 0 for name in FORMAL_SPLITS},
        "background_count_counts": {
            int(name): 0
            for name in requirements.accepted_background_count_quotas
        },
        "spawn_seeds": set(),
    }


def _stored_targeted_progress(
    store: JointBEVDatasetStore,
    requirements: TargetedSupplementRequirements,
) -> dict[str, object]:
    progress = _empty_targeted_progress(requirements)
    for writer in store.writers.values():
        for episode in writer.episodes.values():
            attributes = episode.attributes
            evidence = attributes.get("targeted_supplement_evidence")
            if (
                attributes.get("scenario_id") != requirements.scenario_id
                or attributes.get("rule_maker_profile_id")
                != requirements.rule_maker_profile_id
                or not isinstance(evidence, Mapping)
                or evidence.get("behavior_category")
                != requirements.behavior_category
                or int(episode.joint_samples)
                != requirements.required_samples_per_episode
            ):
                raise JointCollectionError(
                    "stored targeted episode violates the frozen supplement contract",
                    reason_code="targeted_resume_mismatch",
                )
            seed = attributes.get("spawn_seed")
            if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
                raise JointCollectionError(
                    "stored targeted episode has no valid spawn seed",
                    reason_code="targeted_resume_mismatch",
                )
            if int(seed) in progress["spawn_seeds"]:
                raise JointCollectionError(
                    f"targeted supplement reuses spawn seed {seed}",
                    reason_code="targeted_duplicate_spawn_seed",
                )
            background_count = evidence.get("incidental_background_actor_count")
            if background_count not in progress["background_count_counts"]:
                raise JointCollectionError(
                    "stored targeted episode has an out-of-contract background count",
                    reason_code="targeted_resume_mismatch",
                )
            progress["accepted_episodes"] += 1
            progress["split_counts"][episode.split] += 1
            progress["background_count_counts"][int(background_count)] += 1
            progress["spawn_seeds"].add(int(seed))
    if int(progress["accepted_episodes"]) * (
        requirements.required_samples_per_episode
    ) != store.total_joint_samples:
        raise JointCollectionError(
            "targeted episode count does not match stored sample count",
            reason_code="targeted_resume_mismatch",
        )
    for split, count in progress["split_counts"].items():
        if count > int(requirements.accepted_episode_quotas[split]):
            raise JointCollectionError(
                f"targeted split {split} exceeds its quota",
                reason_code="targeted_resume_mismatch",
            )
    for background, count in progress["background_count_counts"].items():
        if count > int(requirements.accepted_background_count_quotas[background]):
            raise JointCollectionError(
                f"targeted background count {background} exceeds its quota",
                reason_code="targeted_resume_mismatch",
            )
    return progress


def _targeted_simulator_attempts(bundle: JointRiskBundleIndex) -> int:
    return sum(row.outcome != "scheduler_skip" for row in bundle.rows)


def _serializable_targeted_progress(
    progress: Mapping[str, object], simulator_attempts: int
) -> dict[str, object]:
    return {
        "accepted_episodes": int(progress["accepted_episodes"]),
        "simulator_attempts": int(simulator_attempts),
        "split_counts": dict(progress["split_counts"]),
        "background_count_counts": {
            str(key): int(value)
            for key, value in progress["background_count_counts"].items()
        },
        "spawn_seeds": sorted(progress["spawn_seeds"]),
    }


def _validate_targeted_completion(
    progress: Mapping[str, object],
    requirements: TargetedSupplementRequirements,
) -> None:
    if dict(progress["split_counts"]) != dict(
        requirements.accepted_episode_quotas
    ):
        raise JointCollectionError(
            "targeted supplement split quotas are incomplete",
            reason_code="targeted_split_quota_incomplete",
        )
    if dict(progress["background_count_counts"]) != dict(
        requirements.accepted_background_count_quotas
    ):
        raise JointCollectionError(
            "targeted supplement background quotas are incomplete",
            reason_code="targeted_background_quota_incomplete",
        )


def _targeted_hard_stop_reason(
    *,
    simulator_attempts: int,
    accepted_episodes: int,
    requirements: TargetedSupplementRequirements,
) -> str | None:
    if (
        simulator_attempts >= requirements.initial_feasibility_attempts
        and accepted_episodes == 0
    ):
        return "targeted_initial_feasibility_failed"
    if simulator_attempts >= requirements.max_simulator_attempts:
        return "targeted_max_simulator_attempts_exhausted"
    return None


def _plan_targeted_parallel_batch(
    config: JointCollectionRunConfig,
    *,
    start_episode_index: int,
    progress: Mapping[str, object],
    bundle: JointRiskBundleIndex,
    simulator_attempts: int,
) -> tuple[tuple[JointEpisodeSpec, ...], list[dict[str, object]]]:
    requirements = config.targeted_supplement
    if requirements is None:
        return (), []
    batch_limit = min(
        requirements.parallel_workers,
        requirements.max_simulator_attempts - simulator_attempts,
    )
    if config.max_episodes > 0:
        batch_limit = min(
            batch_limit, config.max_episodes - start_episode_index
        )
    if (
        int(progress["accepted_episodes"]) == 0
        and simulator_attempts < requirements.initial_feasibility_attempts
    ):
        batch_limit = min(
            batch_limit,
            requirements.initial_feasibility_attempts - simulator_attempts,
        )
    if batch_limit <= 0:
        return (), []
    used_seeds = set(progress["spawn_seeds"])
    used_seeds.update(row.spawn_seed for row in bundle.rows)
    reserved_by_split: Counter[str] = Counter()
    assigner = EpisodeSplitAssigner(config.split_config)
    specs = []
    entries = []
    for offset in range(batch_limit):
        episode_index = start_episode_index + offset
        split_name = assigner.split_for_episode(episode_index)
        remaining_split = (
            requirements.accepted_episode_quotas[split_name]
            - progress["split_counts"][split_name]
            - reserved_by_split[split_name]
        )
        if remaining_split <= 0:
            break
        forced_seed = (
            requirements.bootstrap_spawn_seeds[simulator_attempts + offset]
            if simulator_attempts + offset
            < len(requirements.bootstrap_spawn_seeds)
            else None
        )
        spec = sample_episode_spec_for_scenario(
            config,
            episode_index,
            requirements.scenario_id,
            spawn_seed=forced_seed,
        )
        unique_seed = spec.spawn_seed
        while unique_seed in used_seeds:
            unique_seed = (unique_seed + 1) % (2**31 - 1)
        if unique_seed != spec.spawn_seed:
            spec = replace(spec, spawn_seed=unique_seed)
        used_seeds.add(spec.spawn_seed)
        reserved_by_split[split_name] += 1
        specs.append(spec)
        entries.append(
            {
                "episode_index": episode_index,
                "split": split_name,
                "scenario_id": spec.scenario_id,
                "local_route": spec.local_route,
                "spawn_seed": spec.spawn_seed,
            }
        )
    return tuple(specs), entries


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


TARGETED_BATCH_PENDING_FILE = ".targeted_batch_pending.json"


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _isolated_targeted_rollout_worker(
    sender,
    config: JointCollectionRunConfig,
    spec: JointEpisodeSpec,
) -> None:
    env = None
    try:
        requirements = config.targeted_supplement
        if requirements is None:
            raise JointCollectionError(
                "isolated targeted worker received a non-targeted config",
                reason_code="targeted_worker_contract",
            )
        episode_step_limit = config.episode_step_limit(spec.scenario_id)
        episode_env_config = dict(config.env_config)
        episode_env_config.update(
            {
                "rule_maker_profile_id": requirements.rule_maker_profile_id,
                "start_seed": int(spec.spawn_seed),
                "num_scenarios": 1,
                "horizon": episode_step_limit,
            }
        )
        env = SensorlessJointBEVPlatoonEnv(episode_env_config)
        _configure_episode(env, spec)
        rollout = collect_joint_episode(
            env,
            max_steps=episode_step_limit,
            reset_seed=spec.spawn_seed,
        )
        sender.send(("ok", rollout, simulator_decision_dt_s(env)))
    except BaseException as exc:
        sender.send(
            (
                "error",
                getattr(exc, "reason_code", "collection_contract"),
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )
        )
    finally:
        if env is not None:
            env.close()
        sender.close()


def _collect_one_targeted_isolated(
    config: JointCollectionRunConfig,
    spec: JointEpisodeSpec,
) -> tuple[str, object, object]:
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_isolated_targeted_rollout_worker,
        args=(sender, config, spec),
    )
    try:
        process.start()
        sender.close()
        payload = receiver.recv()
        process.join()
        if process.exitcode != 0 and payload[0] == "ok":
            return (
                "error",
                "targeted_worker_process_failed",
                f"targeted worker exited with code {process.exitcode}",
            )
        return payload
    except (EOFError, OSError) as exc:
        return (
            "error",
            "targeted_worker_process_failed",
            f"targeted worker IPC failed: {exc}",
        )
    finally:
        receiver.close()
        if process.is_alive():
            process.terminate()
            process.join()


def _collect_targeted_batch(
    config: JointCollectionRunConfig,
    specs: tuple[JointEpisodeSpec, ...],
) -> tuple[tuple[str, object, object], ...]:
    requirements = config.targeted_supplement
    if requirements is None or not specs:
        raise JointCollectionError(
            "targeted parallel batch is empty or disabled",
            reason_code="targeted_worker_contract",
        )
    worker_count = min(requirements.parallel_workers, len(specs))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(_collect_one_targeted_isolated, config, spec)
            for spec in specs
        ]
        return tuple(future.result() for future in futures)


def _targeted_batch_path(config: JointCollectionRunConfig) -> Path:
    return config.bundle_root / TARGETED_BATCH_PENDING_FILE


def _write_targeted_batch_pending(
    config: JointCollectionRunConfig,
    entries: list[Mapping[str, object]],
) -> None:
    path = _targeted_batch_path(config)
    if entries:
        _atomic_write_json(
            path,
            {"format": "targeted-parallel-batch-v1", "entries": entries},
        )
    elif path.exists():
        path.unlink()


def _complete_targeted_batch_entry(
    config: JointCollectionRunConfig,
    entries: list[dict[str, object]],
    episode_index: int,
) -> list[dict[str, object]]:
    remaining = [
        entry
        for entry in entries
        if int(entry["episode_index"]) != int(episode_index)
    ]
    _write_targeted_batch_pending(config, remaining)
    return remaining


def _recover_targeted_batch_pending(
    config: JointCollectionRunConfig,
    bundle: JointRiskBundleIndex,
    base_store: JointBEVDatasetStore,
) -> None:
    path = _targeted_batch_path(config)
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries") if isinstance(payload, Mapping) else None
    if not isinstance(entries, list):
        raise JointRiskBundleStorageError("invalid targeted parallel batch intent")
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise JointRiskBundleStorageError("invalid targeted batch entry")
        episode_index = int(entry["episode_index"])
        if episode_index < bundle.next_episode_index:
            continue
        if episode_index != bundle.next_episode_index:
            raise JointRiskBundleStorageError(
                "targeted batch recovery is not contiguous"
            )
        split = str(entry["split"])
        reason = "interrupted_parallel_rollout"
        bundle.begin_attempt(
            BundleEpisodeAttempt(
                episode_index=episode_index,
                split=split,
                scenario_id=str(entry["scenario_id"]),
                local_route=str(entry["local_route"]),
                spawn_seed=int(entry["spawn_seed"]),
            )
        )
        base_store.record_rejected_episode(episode_index, reason)
        bundle.finalize(
            BundleEpisodeResult(
                episode_index=episode_index,
                split=split,
                scenario_id=str(entry["scenario_id"]),
                local_route=str(entry["local_route"]),
                spawn_seed=int(entry["spawn_seed"]),
                base_status="rejected",
                base_rejection_reason=reason,
                sidecar_status="rejected",
                sidecar_rejection_reason=reason,
                raw_steps=0,
                base_samples=0,
                outcome=reason,
            )
        )
    _write_targeted_batch_pending(config, [])


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


def _effective_resume_mode(config: JointCollectionRunConfig) -> bool:
    """Resolve first creation versus strict three-component resume."""

    markers = {
        "base": config.dataset_root / JointBEVDatasetStore.CONTRACT_FILE,
        "sidecar": (
            config.sidecar_root / RiskEntrySidecarDatasetStore.CONTRACT_FILE
        ),
        "bundle": config.bundle_root / JointRiskBundleIndex.MANIFEST_FILE,
    }
    present = {name: path.is_file() for name, path in markers.items()}
    if not any(present.values()):
        return False
    if not config.resume:
        return False
    if not all(present.values()):
        missing = sorted(name for name, exists in present.items() if not exists)
        raise JointRiskBundleStorageError(
            f"formal resume has a partial component initialization: missing={missing}"
        )
    return True


def _sidecar_start(
    *,
    episode_index: int,
    split: str,
    spec: JointEpisodeSpec,
    rollout,
    base_dataset_fingerprint: str,
    decision_dt_s: float,
    scenario_contract_sha256: str | None = None,
    rule_maker_profile_id: str | None = None,
) -> SidecarEpisodeStart:
    summary = dict(rollout.scenario_summary)
    conflict_evidence = summary.get("conflict_evidence", {})
    resolved = summary.get("resolved_scenario_parameters", {})
    if not isinstance(conflict_evidence, Mapping):
        conflict_evidence = {}
    if not isinstance(resolved, Mapping):
        resolved = {}
    return SidecarEpisodeStart(
        episode_index=episode_index,
        split=split,
        scenario_id=spec.scenario_id,
        local_route=spec.local_route,
        spawn_seed=spec.spawn_seed,
        decision_dt_s=decision_dt_s,
        base_dataset_fingerprint=base_dataset_fingerprint,
        scenario_parameters={
            "scenario_contract_sha256": (
                scenario_contract_for_id(FORMAL_V1_CONTRACT_ID)["sha256"]
                if scenario_contract_sha256 is None
                else str(scenario_contract_sha256)
            ),
            "traffic_density": spec.traffic_density,
            "initial_speed_km_h": spec.initial_speed_km_h,
            "scenario_trigger_step": summary.get("scenario_trigger_step"),
            "scenario_realized_step": summary.get("scenario_realized_step"),
            "rule_maker_profile_id": rule_maker_profile_id,
            "observed_behavior_class": conflict_evidence.get(
                "observed_behavior_class"
            ),
            "mixed_direction_lane_change_completed": conflict_evidence.get(
                "mixed_direction_lane_change_completed"
            ),
            "reassembly_completed": conflict_evidence.get(
                "reassembly_completed"
            ),
            "reassembly_to_initial_lane_completed": conflict_evidence.get(
                "reassembly_to_initial_lane_completed"
            ),
            "hazard_cleared": conflict_evidence.get("hazard_cleared"),
            "formation_recovered_after_hazard": conflict_evidence.get(
                "formation_recovered_after_hazard"
            ),
            "incidental_background_actor_count": resolved.get(
                "incidental_background_actor_count"
            ),
            "incidental_background_realized_count": conflict_evidence.get(
                "incidental_background_realized_count"
            ),
            "rollout_failure_reason": rollout.failure_reason,
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
    scenario_contract = config.scenario_contract()
    scenario_contract_sha256 = str(scenario_contract["sha256"])
    base_fingerprint = config.immutable_fingerprint()
    effective_resume = _effective_resume_mode(config)
    with JointBEVDatasetStore(
        config.dataset_root,
        split_config=config.split_config,
        dataset_fingerprint=base_fingerprint,
        resume=effective_resume,
    ) as store, RiskEntrySidecarDatasetStore(
        config.sidecar_root,
        base_dataset_fingerprint=base_fingerprint,
        resume=effective_resume,
    ) as sidecar_store, JointRiskBundleIndex(
        config.bundle_root,
        base_directory=config.dataset_root.name,
        sidecar_directory=config.sidecar_root.name,
        base_dataset_fingerprint=base_fingerprint,
        sidecar_dataset_fingerprint=sidecar_store.dataset_fingerprint,
        scenario_contract_sha256=scenario_contract_sha256,
        split_seed=config.split_config.seed,
        resume=effective_resume,
    ) as bundle:
        _recover_pending_bundle_attempt(bundle, store, sidecar_store)
        if config.targeted_supplement is not None:
            _recover_targeted_batch_pending(config, bundle, store)
        _validate_bundle_resume_state(bundle, store, sidecar_store)
        starting_joint_samples = store.total_joint_samples
        print(
            f"[INFO] base_dataset={config.dataset_root} "
            f"sidecar_dataset={config.sidecar_root} "
            f"resume_requested={config.resume} resume_active={effective_resume} "
            f"existing_joint_steps={store.total_joint_samples}",
            flush=True,
        )
        if store.total_joint_samples >= config.target_joint_steps:
            summary = {
                "base": store.summary(),
                "sidecar": sidecar_store.summary(),
                "bundle_attempts": bundle.next_episode_index,
            }
            if config.formal_scenario_quotas is not None:
                quotas = dict(config.formal_scenario_quotas)
                progress = _stored_formal_progress(
                    store, quotas, config.formal_diversity
                )
                _validate_formal_completion(
                    progress, quotas, config.formal_diversity
                )
                summary["formal_pilot"] = {
                    "scenario_quotas": quotas,
                    "scenario_counts": {
                        name: int(row["joint_samples"])
                        for name, row in progress.items()
                    },
                    "attempts": {
                        name: sum(row.scenario_id == name for row in bundle.rows)
                        for name in quotas
                    },
                    "scenario_contract_sha256": scenario_contract_sha256,
                    "diversity_requirements": (
                        None
                        if config.formal_diversity is None
                        else config.formal_diversity.as_dict()
                    ),
                    "diversity_observed": _serializable_formal_progress(progress),
                }
            if config.targeted_supplement is not None:
                targeted_progress = _stored_targeted_progress(
                    store, config.targeted_supplement
                )
                _validate_targeted_completion(
                    targeted_progress, config.targeted_supplement
                )
                summary["targeted_supplement"] = {
                    "requirements": config.targeted_supplement.as_dict(),
                    "observed": _serializable_targeted_progress(
                        targeted_progress,
                        _targeted_simulator_attempts(bundle),
                    ),
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
        formal_quotas = (
            None
            if config.formal_scenario_quotas is None
            else dict(config.formal_scenario_quotas)
        )
        formal_counts = (
            None
            if formal_quotas is None
            else _stored_scenario_counts(store, formal_quotas)
        )
        formal_progress = (
            None
            if formal_quotas is None
            else _stored_formal_progress(
                store, formal_quotas, config.formal_diversity
            )
        )
        formal_attempts = (
            None
            if formal_quotas is None
            else {
                name: sum(row.scenario_id == name for row in bundle.rows)
                for name in formal_quotas
            }
        )
        formal_order = () if formal_quotas is None else tuple(formal_quotas)
        targeted_requirements = config.targeted_supplement
        targeted_progress = (
            None
            if targeted_requirements is None
            else _stored_targeted_progress(store, targeted_requirements)
        )
        targeted_simulator_attempts = (
            0
            if targeted_requirements is None
            else _targeted_simulator_attempts(bundle)
        )
        targeted_precollected: dict[
            int, tuple[JointEpisodeSpec, tuple[str, object, object]]
        ] = {}
        targeted_pending_entries: list[dict[str, object]] = []
        try:
            while store.total_joint_samples < config.target_joint_steps:
                episode_index = store.next_episode_index
                if config.max_episodes > 0 and episode_index >= config.max_episodes:
                    break
                if targeted_requirements is not None:
                    hard_stop_reason = _targeted_hard_stop_reason(
                        simulator_attempts=targeted_simulator_attempts,
                        accepted_episodes=int(
                            targeted_progress["accepted_episodes"]
                        ),
                        requirements=targeted_requirements,
                    )
                    if hard_stop_reason is not None:
                        raise JointCollectionError(
                            "targeted supplement reached a production stop condition",
                            reason_code=hard_stop_reason,
                        )
                    if (
                        targeted_requirements.parallel_workers > 1
                        and not targeted_precollected
                        and targeted_progress["split_counts"][
                            store.assigner.split_for_episode(episode_index)
                        ]
                        < targeted_requirements.accepted_episode_quotas[
                            store.assigner.split_for_episode(episode_index)
                        ]
                    ):
                        batch_specs, targeted_pending_entries = (
                            _plan_targeted_parallel_batch(
                                config,
                                start_episode_index=episode_index,
                                progress=targeted_progress,
                                bundle=bundle,
                                simulator_attempts=targeted_simulator_attempts,
                            )
                        )
                        if batch_specs:
                            _write_targeted_batch_pending(
                                config, targeted_pending_entries
                            )
                            batch_results = _collect_targeted_batch(
                                config, batch_specs
                            )
                            targeted_precollected = {
                                episode_index + offset: (spec, result)
                                for offset, (spec, result) in enumerate(
                                    zip(batch_specs, batch_results)
                                )
                            }
                    scenario_id = targeted_requirements.scenario_id
                    split = store.assigner.split_for_episode(episode_index)
                    precollected = targeted_precollected.get(episode_index)
                    if precollected is not None:
                        spec = precollected[0]
                    else:
                        forced_spawn_seed = (
                            targeted_requirements.bootstrap_spawn_seeds[
                                targeted_simulator_attempts
                            ]
                            if targeted_simulator_attempts
                            < len(targeted_requirements.bootstrap_spawn_seeds)
                            else None
                        )
                        spec = sample_episode_spec_for_scenario(
                            config,
                            episode_index,
                            scenario_id,
                            spawn_seed=forced_spawn_seed,
                        )
                        used_seeds = set(targeted_progress["spawn_seeds"])
                        used_seeds.update(row.spawn_seed for row in bundle.rows)
                        unique_seed = spec.spawn_seed
                        while unique_seed in used_seeds:
                            unique_seed = (unique_seed + 1) % (2**31 - 1)
                        if unique_seed != spec.spawn_seed:
                            spec = replace(spec, spawn_seed=unique_seed)
                    if (
                        targeted_progress["split_counts"][split]
                        >= targeted_requirements.accepted_episode_quotas[split]
                    ):
                        bundle.begin_attempt(
                            BundleEpisodeAttempt(
                                episode_index=episode_index,
                                split=split,
                                scenario_id=spec.scenario_id,
                                local_route=spec.local_route,
                                spawn_seed=spec.spawn_seed,
                            )
                        )
                        reason = "targeted_split_quota_satisfied"
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
                                outcome="scheduler_skip",
                            )
                        )
                        print(
                            f"[INFO] episode={episode_index} split={split} "
                            "scheduler_skip=targeted_split_quota_satisfied",
                            flush=True,
                        )
                        continue
                elif diagnostic_quotas is None and formal_quotas is None:
                    spec = sample_episode_spec(config, episode_index)
                elif diagnostic_quotas is not None:
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
                else:
                    split = store.assigner.split_for_episode(episode_index)
                    if config.formal_diversity is None:
                        incomplete = [
                            scenario_id
                            for scenario_id in formal_order
                            if formal_counts[scenario_id]
                            < formal_quotas[scenario_id]
                        ]
                        if not incomplete:
                            break
                        scenario_id = incomplete[episode_index % len(incomplete)]
                    else:
                        scenario_id = _select_formal_scenario(
                            formal_order,
                            formal_progress,
                            formal_quotas,
                            split,
                        )
                    if (
                        formal_attempts[scenario_id]
                        >= config.formal_max_attempts_per_scenario
                    ):
                        raise JointCollectionError(
                            f"formal quota for {scenario_id} was not met within "
                            f"{config.formal_max_attempts_per_scenario} attempts",
                            reason_code="formal_pilot_attempt_limit",
                        )
                    formal_attempts[scenario_id] += 1
                    forced_spawn_seed = None
                    if config.formal_diversity is not None:
                        stored_seeds = formal_progress[scenario_id]["spawn_seeds"]
                        forced_spawn_seed = next(
                            (
                                value
                                for value in config.formal_diversity.required_spawn_seeds
                                if value not in stored_seeds
                            ),
                            None,
                        )
                    spec = sample_episode_spec_for_scenario(
                        config,
                        episode_index,
                        scenario_id,
                        spawn_seed=forced_spawn_seed,
                    )
                    if config.formal_diversity is not None:
                        stored_seeds = formal_progress[scenario_id]["spawn_seeds"]
                        unique_seed = spec.spawn_seed
                        while unique_seed in stored_seeds:
                            unique_seed = (unique_seed + 1) % (2**31 - 1)
                        if unique_seed != spec.spawn_seed:
                            spec = replace(spec, spawn_seed=unique_seed)
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
                    env = None
                episode_env_config = dict(config.env_config)
                rule_maker_profile_id = None
                if targeted_requirements is not None:
                    rule_maker_profile_id = (
                        targeted_requirements.rule_maker_profile_id
                    )
                elif config.formal_diversity is not None:
                    rule_maker_profile_id = _formal_rule_maker_profile(
                        scenario_id=spec.scenario_id,
                        row=formal_progress[spec.scenario_id],
                        requirements=config.formal_diversity,
                    )
                episode_env_config["rule_maker_profile_id"] = rule_maker_profile_id
                episode_step_limit = config.episode_step_limit(spec.scenario_id)
                episode_env_config.update(
                    {
                        "start_seed": int(spec.spawn_seed),
                        "num_scenarios": 1,
                        "horizon": episode_step_limit,
                    }
                )
                precollected_payload = (
                    None
                    if targeted_requirements is None
                    else targeted_precollected.get(episode_index)
                )
                episode_decision_dt_s = 0.1
                if precollected_payload is None:
                    env = SensorlessJointBEVPlatoonEnv(episode_env_config)
                    _configure_episode(env, spec)
                if targeted_requirements is not None:
                    targeted_simulator_attempts += 1
                try:
                    if precollected_payload is None:
                        rollout = collect_joint_episode(
                            env,
                            max_steps=episode_step_limit,
                            reset_seed=spec.spawn_seed,
                        )
                        episode_decision_dt_s = simulator_decision_dt_s(env)
                    else:
                        status, first, second = precollected_payload[1]
                        if status != "ok":
                            raise JointCollectionError(
                                str(second), reason_code=str(first)
                            )
                        rollout = first
                        episode_decision_dt_s = float(second)
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
                    if targeted_requirements is not None:
                        targeted_precollected.pop(episode_index, None)
                        targeted_pending_entries = (
                            _complete_targeted_batch_entry(
                                config,
                                targeted_pending_entries,
                                episode_index,
                            )
                        )
                        hard_stop_reason = _targeted_hard_stop_reason(
                            simulator_attempts=targeted_simulator_attempts,
                            accepted_episodes=int(
                                targeted_progress["accepted_episodes"]
                            ),
                            requirements=targeted_requirements,
                        )
                        if hard_stop_reason is not None:
                            raise JointCollectionError(
                                "targeted supplement reached a production stop condition",
                                reason_code=hard_stop_reason,
                            )
                    continue

                samples_to_store = rollout.samples
                formal_coverage = None
                selected_steps: tuple[int, ...] = tuple(
                    int(value) for value in rollout.sample_step_indices
                )
                trigger_step = rollout.scenario_summary.get(
                    "scenario_trigger_step"
                )
                base_rejection_reason = rollout.failure_reason
                if (
                    targeted_requirements is not None
                    and base_rejection_reason is not None
                ):
                    raw_failure = str(base_rejection_reason).lower()
                    base_rejection_reason = (
                        "targeted_safety_rejected"
                        if any(
                            token in raw_failure
                            for token in (
                                "collision",
                                "crash",
                                "out_of_road",
                                "out_of_route",
                            )
                        )
                        else "targeted_trajectory_infeasible"
                    )
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
                if formal_quotas is not None and base_rejection_reason is None:
                    remaining = (
                        formal_quotas[spec.scenario_id]
                        - formal_counts[spec.scenario_id]
                    )
                    try:
                        max_samples = None
                        if config.formal_diversity is not None:
                            formal_coverage = _formal_episode_coverage(
                                spec.scenario_id, rollout.scenario_summary
                            )
                            if (
                                formal_coverage[
                                    "incidental_background_actor_count"
                                ]
                                not in config.formal_diversity.required_incidental_background_actor_counts
                            ):
                                raise JointCollectionError(
                                    "formal episode background count is outside the frozen contract",
                                    reason_code="formal_pilot_background_count_invalid",
                                )
                            max_samples = _formal_sample_capacity(
                                scenario_id=spec.scenario_id,
                                remaining=remaining,
                                row=formal_progress[spec.scenario_id],
                                requirements=config.formal_diversity,
                                split=split,
                                spawn_seed=spec.spawn_seed,
                                coverage=formal_coverage,
                            )
                        samples_to_store, selected_steps = (
                            _select_formal_quota_samples(
                                rollout,
                                remaining=remaining,
                                max_samples=max_samples,
                            )
                        )
                    except JointCollectionError as exc:
                        base_rejection_reason = exc.reason_code
                targeted_evidence = None
                if (
                    targeted_requirements is not None
                    and base_rejection_reason is None
                ):
                    try:
                        targeted_evidence = _validate_targeted_rollout(
                            rollout, targeted_requirements
                        )
                        background_count = int(
                            targeted_evidence[
                                "incidental_background_actor_count"
                            ]
                        )
                        if background_count not in (
                            targeted_requirements.accepted_background_count_quotas
                        ):
                            raise JointCollectionError(
                                "targeted background count is outside the frozen quotas",
                                reason_code="targeted_background_count_invalid",
                            )
                        if (
                            targeted_progress["background_count_counts"][
                                background_count
                            ]
                            >= targeted_requirements.accepted_background_count_quotas[
                                background_count
                            ]
                        ):
                            raise JointCollectionError(
                                "targeted background-count quota is already satisfied",
                                reason_code="targeted_background_quota_satisfied",
                            )
                    except JointCollectionError as exc:
                        base_rejection_reason = exc.reason_code
                base_eligible = base_rejection_reason is None
                mapping = selected_steps if base_eligible else ()
                decision_dt_s = episode_decision_dt_s
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
                            scenario_contract_sha256=scenario_contract_sha256,
                            rule_maker_profile_id=rule_maker_profile_id,
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
                    if targeted_requirements is not None:
                        targeted_precollected.pop(episode_index, None)
                        targeted_pending_entries = (
                            _complete_targeted_batch_entry(
                                config,
                                targeted_pending_entries,
                                episode_index,
                            )
                        )
                    print(
                        f"[WARNING] episode={episode_index} scenario={spec.scenario_id} "
                        f"rejected={reason}: {exc}",
                        flush=True,
                    )
                    if targeted_requirements is not None:
                        hard_stop_reason = _targeted_hard_stop_reason(
                            simulator_attempts=targeted_simulator_attempts,
                            accepted_episodes=int(
                                targeted_progress["accepted_episodes"]
                            ),
                            requirements=targeted_requirements,
                        )
                        if hard_stop_reason is not None:
                            raise JointCollectionError(
                                "targeted supplement reached a production stop condition",
                                reason_code=hard_stop_reason,
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
                            "formal_pilot": formal_quotas is not None,
                            "formal_quota_clipped": (
                                formal_quotas is not None
                                and len(samples_to_store) < len(rollout.samples)
                            ),
                            "formal_coverage": formal_coverage,
                            "targeted_supplement": (
                                targeted_requirements is not None
                            ),
                            "targeted_supplement_evidence": targeted_evidence,
                            "rule_maker_profile_id": rule_maker_profile_id,
                            "scenario_contract_sha256": scenario_contract_sha256,
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
                if targeted_requirements is not None:
                    targeted_precollected.pop(episode_index, None)
                    targeted_pending_entries = _complete_targeted_batch_entry(
                        config,
                        targeted_pending_entries,
                        episode_index,
                    )
                if stored is None:
                    print(
                        f"[WARNING] episode={episode_index} scenario={spec.scenario_id} "
                        f"base_rejected={base_rejection_reason} sidecar=committed",
                        flush=True,
                    )
                    if targeted_requirements is not None:
                        hard_stop_reason = _targeted_hard_stop_reason(
                            simulator_attempts=targeted_simulator_attempts,
                            accepted_episodes=int(
                                targeted_progress["accepted_episodes"]
                            ),
                            requirements=targeted_requirements,
                        )
                        if hard_stop_reason is not None:
                            raise JointCollectionError(
                                "targeted supplement reached a production stop condition",
                                reason_code=hard_stop_reason,
                            )
                    continue
                if diagnostic_counts is not None:
                    diagnostic_counts[spec.scenario_id] += stored.joint_samples
                if formal_counts is not None:
                    formal_counts[spec.scenario_id] += stored.joint_samples
                    formal_progress = _stored_formal_progress(
                        store, formal_quotas, config.formal_diversity
                    )
                if targeted_requirements is not None:
                    targeted_progress = _stored_targeted_progress(
                        store, targeted_requirements
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
                "scenario_contract_sha256": scenario_contract_sha256,
            }
            if diagnostic_counts != diagnostic_quotas:
                raise JointCollectionError(
                    "diagnostic collection ended before all quotas were met",
                    reason_code="diagnostic_quota_incomplete",
                )
        if formal_quotas is not None:
            formal_progress = _stored_formal_progress(
                store, formal_quotas, config.formal_diversity
            )
            summary["formal_pilot"] = {
                "scenario_quotas": formal_quotas,
                "scenario_counts": formal_counts,
                "attempts": formal_attempts,
                "scenario_contract_sha256": scenario_contract_sha256,
                "diversity_requirements": (
                    None
                    if config.formal_diversity is None
                    else config.formal_diversity.as_dict()
                ),
                "diversity_observed": _serializable_formal_progress(
                    formal_progress
                ),
            }
            _validate_formal_completion(
                formal_progress, formal_quotas, config.formal_diversity
            )
        if targeted_requirements is not None:
            targeted_progress = _stored_targeted_progress(
                store, targeted_requirements
            )
            summary["targeted_supplement"] = {
                "requirements": targeted_requirements.as_dict(),
                "observed": _serializable_targeted_progress(
                    targeted_progress, targeted_simulator_attempts
                ),
            }
            _validate_targeted_completion(
                targeted_progress, targeted_requirements
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
    config = parse_args(argv)
    try:
        run_collection(config)
    except JointCollectionError as exc:
        if config.targeted_supplement is not None:
            from expert_dataset.finalize_s5_targeted_supplement import (
                write_targeted_supplement_report,
            )

            report = write_targeted_supplement_report(
                config, stop_reason=getattr(exc, "reason_code", None)
            )
            print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
        raise
    if config.targeted_supplement is not None:
        from expert_dataset.finalize_s5_targeted_supplement import (
            build_composition_manifest,
        )

        build_composition_manifest(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
