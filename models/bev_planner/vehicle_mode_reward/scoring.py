"""Core target-vehicle counterfactual scoring implementation."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from envs.observations.semantic_bev import BEVChannel
from models.platoon_planner.collision_geometry import (
    obb_overlap_series,
    shared_corridor_gap_series,
)

from .constants import AGENT_IDS, NUM_ROLES, TRAJECTORY_SHAPE
from .geometry import (
    _dense_local_trajectories,
    _local_to_world,
    footprint_outside_drivable_series,
    footprint_road_margin_series,
    tracking_aware_half_extents,
)
from .risk import (
    aggregate_temporal_risk,
    closing_ttc_from_gap_series,
    soft_threshold_risk,
)


class _CounterfactualScoringMixin:
    def _background_by_actor(
        self,
        env: object,
        target_role: int,
        times: np.ndarray,
    ) -> dict[str, list[tuple[str, np.ndarray, tuple[float, float]]]]:
        agents = getattr(env, "agents", {}) or {}
        background = self._prediction_planner._predicted_obstacles(
            env,
            agents[AGENT_IDS[target_role]],
            times,
            include_platoon=False,
            include_policy_branches=True,
        )
        grouped: dict[str, list[tuple[str, np.ndarray, tuple[float, float]]]] = {}
        for name, predicted, dimensions in background:
            actor = str(name).split(":policy_branch_", 1)[0]
            grouped.setdefault(actor, []).append(
                (str(name), np.asarray(predicted, dtype=np.float64), dimensions)
            )
        return grouped

    def _score_target_group(
        self,
        *,
        target_role: int,
        target_trajectories: np.ndarray,
        frozen_argmax_joint_trajectories: np.ndarray,
        poses: list[np.ndarray],
        drivable: np.ndarray,
        road_field: np.ndarray,
        background_by_actor: Mapping[
            str, list[tuple[str, np.ndarray, tuple[float, float]]]
        ],
    ) -> dict[str, object]:
        group_size = target_trajectories.shape[0]
        joint = np.broadcast_to(
            frozen_argmax_joint_trajectories,
            (group_size, NUM_ROLES, *TRAJECTORY_SHAPE),
        ).copy()
        joint[:, target_role] = target_trajectories
        dense_local, _ = _dense_local_trajectories(joint, self.config)
        dense_world = np.empty_like(dense_local)
        for role in range(NUM_ROLES):
            dense_world[:, role] = _local_to_world(dense_local[:, role], poses[role])

        progress_score = np.zeros(group_size, dtype=np.float64)
        gap_penalty = np.zeros(group_size, dtype=np.float64)
        ttc_penalty = np.zeros(group_size, dtype=np.float64)
        road_penalty = np.zeros(group_size, dtype=np.float64)
        comfort_penalty = np.zeros(group_size, dtype=np.float64)
        collision = np.zeros(group_size, dtype=np.bool_)
        out_of_drivable = np.zeros(group_size, dtype=np.bool_)
        clearance_violation = np.zeros(group_size, dtype=np.bool_)
        minimum_background_gap = np.full(
            group_size, self.config.no_risk_gap_m, dtype=np.float64
        )
        minimum_teammate_gap = np.full(
            group_size, self.config.no_risk_gap_m, dtype=np.float64
        )
        minimum_road_margin = np.full(
            group_size, self.config.no_risk_gap_m, dtype=np.float64
        )
        minimum_ttc = np.full(
            group_size, self.config.no_risk_ttc_s, dtype=np.float64
        )
        half_length, half_width = tracking_aware_half_extents(self.config)
        tracking_dimensions = (2.0 * half_length, 2.0 * half_width)
        physical_dimensions = (
            self.config.vehicle_length_m,
            self.config.vehicle_width_m,
        )

        for group in range(group_size):
            local = dense_local[group, target_role]
            world = dense_world[group, target_role]
            progress_score[group] = float(
                np.clip(
                    target_trajectories[group, -1, 0]
                    / self.config.progress_norm_m,
                    0.0,
                    1.0,
                )
            )
            out_of_drivable[group] = bool(
                np.any(
                    footprint_outside_drivable_series(
                        local[1:], drivable, self.config
                    )
                )
            )
            road_margin = footprint_road_margin_series(
                local,
                road_field,
                self.config,
                tracking_aware=True,
            )
            minimum_road_margin[group] = float(np.min(road_margin))
            road_risk = soft_threshold_risk(
                road_margin,
                warning_threshold=self.config.road_margin_warning_m,
                softness=self.config.road_margin_softness_m,
            )
            road_penalty[group] = aggregate_temporal_risk(
                road_risk,
                max_weight=self.config.temporal_max_weight,
                mean_weight=self.config.temporal_mean_weight,
            )

            motion_local = local[1:]
            speed = np.linalg.norm(
                np.diff(
                    np.concatenate(
                        (np.zeros((1, 2)), motion_local[:, :2]), axis=0
                    ),
                    axis=0,
                ),
                axis=1,
            ) / self.config.interpolation_dt_s
            acceleration = np.diff(speed, prepend=speed[0]) / (
                self.config.interpolation_dt_s
            )
            unwrapped_heading = np.unwrap(motion_local[:, 2])
            yaw_rate = np.diff(
                unwrapped_heading, prepend=unwrapped_heading[0]
            ) / self.config.interpolation_dt_s
            comfort_penalty[group] = float(
                np.clip(
                    0.5 * np.mean(np.abs(acceleration)) / 8.0
                    + 0.5 * np.mean(np.abs(yaw_rate)),
                    0.0,
                    1.0,
                )
            )

            interaction_gap_risks: list[np.ndarray] = []
            interaction_ttc_risks: list[np.ndarray] = []
            for actor, predictions in background_by_actor.items():
                branch_gap_risks = []
                branch_ttc_risks = []
                for name, predicted, other_dimensions in predictions:
                    gap_series = shared_corridor_gap_series(
                        world,
                        tracking_dimensions,
                        predicted,
                        other_dimensions,
                        no_risk_gap_m=self.config.no_risk_gap_m,
                    )
                    branch_gap_risks.append(
                        soft_threshold_risk(
                            gap_series,
                            warning_threshold=self.config.background_safe_gap_m,
                            softness=self.config.gap_softness_m,
                        )
                    )
                    ttc = closing_ttc_from_gap_series(
                        gap_series,
                        dt_s=self.config.interpolation_dt_s,
                        closing_speed_epsilon_mps=(
                            self.config.closing_speed_epsilon_mps
                        ),
                        no_risk_gap_m=self.config.no_risk_gap_m,
                        no_risk_ttc_s=self.config.no_risk_ttc_s,
                    )
                    minimum_ttc[group] = min(
                        minimum_ttc[group], float(np.min(ttc))
                    )
                    branch_ttc_risks.append(
                        soft_threshold_risk(
                            ttc,
                            warning_threshold=self.config.ttc_warning_s,
                            softness=self.config.ttc_softness_s,
                        )
                    )
                    if name == actor:
                        minimum_background_gap[group] = min(
                            minimum_background_gap[group],
                            float(np.min(gap_series)),
                        )
                        if obb_overlap_series(
                            world[1:],
                            physical_dimensions,
                            predicted[1:],
                            other_dimensions,
                            0.0,
                        ):
                            collision[group] = True
                interaction_gap_risks.append(
                    np.max(np.stack(branch_gap_risks), axis=0)
                )
                interaction_ttc_risks.append(
                    np.max(np.stack(branch_ttc_risks), axis=0)
                )

            for other_role in range(NUM_ROLES):
                if other_role == target_role:
                    continue
                teammate = dense_world[group, other_role]
                if obb_overlap_series(
                    world[1:],
                    physical_dimensions,
                    teammate[1:],
                    physical_dimensions,
                    0.0,
                ):
                    collision[group] = True
                pair_gap = shared_corridor_gap_series(
                    world,
                    tracking_dimensions,
                    teammate,
                    tracking_dimensions,
                    no_risk_gap_m=self.config.no_risk_gap_m,
                )
                minimum_teammate_gap[group] = min(
                    minimum_teammate_gap[group], float(np.min(pair_gap))
                )
                interaction_gap_risks.append(
                    soft_threshold_risk(
                        pair_gap,
                        warning_threshold=self.config.platoon_safe_gap_m,
                        softness=self.config.gap_softness_m,
                    )
                )
                pair_ttc = closing_ttc_from_gap_series(
                    pair_gap,
                    dt_s=self.config.interpolation_dt_s,
                    closing_speed_epsilon_mps=(
                        self.config.closing_speed_epsilon_mps
                    ),
                    no_risk_gap_m=self.config.no_risk_gap_m,
                    no_risk_ttc_s=self.config.no_risk_ttc_s,
                )
                minimum_ttc[group] = min(
                    minimum_ttc[group], float(np.min(pair_ttc))
                )
                interaction_ttc_risks.append(
                    soft_threshold_risk(
                        pair_ttc,
                        warning_threshold=self.config.ttc_warning_s,
                        softness=self.config.ttc_softness_s,
                    )
                )

            gap_penalty[group] = aggregate_temporal_risk(
                np.max(np.stack(interaction_gap_risks), axis=0),
                max_weight=self.config.temporal_max_weight,
                mean_weight=self.config.temporal_mean_weight,
            )
            ttc_penalty[group] = aggregate_temporal_risk(
                np.max(np.stack(interaction_ttc_risks), axis=0),
                max_weight=self.config.temporal_max_weight,
                mean_weight=self.config.temporal_mean_weight,
            )
            clearance_violation[group] = bool(
                minimum_background_gap[group] < self.config.background_safe_gap_m
                or minimum_teammate_gap[group] < self.config.platoon_safe_gap_m
            )

        rewards = (
            self.config.progress_weight * progress_score
            - self.config.gap_weight * gap_penalty
            - self.config.ttc_weight * ttc_penalty
            - self.config.road_weight * road_penalty
            - self.config.comfort_weight * comfort_penalty
            - self.config.collision_penalty * collision.astype(np.float64)
            - self.config.out_of_drivable_penalty
            * out_of_drivable.astype(np.float64)
        )
        unsafe = collision | out_of_drivable | clearance_violation
        return {
            "rewards": rewards.astype(np.float32),
            "unsafe": unsafe,
            "collision": collision,
            "out_of_drivable": out_of_drivable,
            "clearance_violation": clearance_violation,
            "components": {
                "progress_score": progress_score.astype(np.float32),
                "gap_penalty": gap_penalty.astype(np.float32),
                "ttc_penalty": ttc_penalty.astype(np.float32),
                "road_penalty": road_penalty.astype(np.float32),
                "comfort_penalty": comfort_penalty.astype(np.float32),
                "minimum_background_gap_m": minimum_background_gap.astype(
                    np.float32
                ),
                "minimum_teammate_gap_m": minimum_teammate_gap.astype(np.float32),
                "minimum_road_margin_m": minimum_road_margin.astype(np.float32),
                "minimum_ttc_s": minimum_ttc.astype(np.float32),
            },
        }
