"""Public vehicle-mode counterfactual reward scorer."""

from __future__ import annotations

import numpy as np

from envs.observations.semantic_bev import BEVChannel

from .config import JointRewardError, VehicleModeRewardConfig
from .constants import AGENT_IDS, NUM_MODES, NUM_ROLES, TRAJECTORY_SHAPE
from .geometry import (
    _agent_pose,
    _as_numpy_model_field,
    _dense_local_trajectories,
    drivable_signed_distance_m,
)
from .input_validation import _CounterfactualInputMixin
from .results import (
    RewardGeometryContext,
    VehicleModePretrainRewardResult,
    VehicleModeRewardResult,
)
from .scoring import _CounterfactualScoringMixin


class VehicleModeCounterfactualReward(
    _CounterfactualInputMixin,
    _CounterfactualScoringMixin,
):
    """Score each ``(vehicle, mode)`` against one frozen teammate context.

    Candidate and same-mode frozen rewards differ only in the target vehicle's
    raw ``tau_d``.  The two teammates always use the deterministic frozen
    Stage-1 argmax trajectories.  This scorer deliberately does not read
    formation relations and never evaluates teammate-only events.
    """

    _COMPONENT_NAMES = (
        "progress_score",
        "gap_penalty",
        "ttc_penalty",
        "road_penalty",
        "comfort_penalty",
        "minimum_background_gap_m",
        "minimum_teammate_gap_m",
        "minimum_road_margin_m",
        "minimum_ttc_s",
    )

    def __init__(self, config: VehicleModeRewardConfig | None = None) -> None:
        from models.platoon_planner.platoon_normal_planner import (
            PlatoonNormalPlanner,
        )

        self.config = config or VehicleModeRewardConfig()
        self._prediction_planner = PlatoonNormalPlanner(
            background_safe_gap_m=self.config.background_safe_gap_m,
            platoon_safe_gap_m=self.config.platoon_safe_gap_m,
        )

    def build_geometry_context(
        self,
        env: object,
        model_inputs: object,
        frozen_argmax: np.ndarray,
    ) -> RewardGeometryContext:
        """Build reusable reward geometry for one unchanged live state."""

        bev = _as_numpy_model_field(model_inputs, "bev")
        if bev.ndim == 5:
            if bev.shape[0] != 1:
                raise JointRewardError("model input BEV only supports B=1")
            bev = bev[0]
        if bev.shape != (NUM_ROLES, 8, 256, 256) or bev.dtype != np.uint8:
            raise JointRewardError("model input BEV contract mismatch")
        poses = tuple(_agent_pose(env, agent_id) for agent_id in AGENT_IDS)
        probe = frozen_argmax[None]
        _, times = _dense_local_trajectories(probe, self.config)
        backgrounds = tuple(
            self._background_by_actor(env, role, times)
            for role in range(NUM_ROLES)
        )
        road_fields = tuple(
            drivable_signed_distance_m(bev[role, int(BEVChannel.DRIVABLE)])
            for role in range(NUM_ROLES)
        )
        return RewardGeometryContext(
            bev=bev,
            poses=poses,
            backgrounds=backgrounds,
            road_fields=road_fields,
        )

    def score_pretrain(
        self,
        env: object,
        model_inputs: object,
        frozen_all_mode_trajectories: object,
        frozen_argmax_joint_trajectories: object,
        valid_mode_mask: object,
        *,
        geometry_context: RewardGeometryContext | None = None,
    ) -> VehicleModePretrainRewardResult:
        """Score all valid frozen modes once for one live-state cache."""

        frozen_all = self._frozen_all_values(frozen_all_mode_trajectories)
        frozen_argmax = self._frozen_argmax_values(
            frozen_argmax_joint_trajectories
        )
        valid = self._valid_values(valid_mode_mask)
        shape = (NUM_ROLES, NUM_MODES)
        rewards = np.zeros(shape, dtype=np.float32)
        unsafe = np.zeros(shape, dtype=np.bool_)
        collision = np.zeros(shape, dtype=np.bool_)
        out_of_drivable = np.zeros(shape, dtype=np.bool_)
        clearance = np.zeros(shape, dtype=np.bool_)
        components = {
            name: np.zeros(shape, dtype=np.float32)
            for name in self._COMPONENT_NAMES
        }
        context = (
            geometry_context
            if geometry_context is not None
            else self.build_geometry_context(env, model_inputs, frozen_argmax)
        )

        for role in range(NUM_ROLES):
            for mode in range(NUM_MODES):
                if not valid[role, mode]:
                    continue
                scored = self._score_target_group(
                    target_role=role,
                    target_trajectories=frozen_all[role, mode][None],
                    frozen_argmax_joint_trajectories=frozen_argmax,
                    poses=list(context.poses),
                    drivable=context.bev[role, int(BEVChannel.DRIVABLE)],
                    road_field=context.road_fields[role],
                    background_by_actor=context.backgrounds[role],
                )
                rewards[role, mode] = scored["rewards"][0]
                unsafe[role, mode] = scored["unsafe"][0]
                collision[role, mode] = scored["collision"][0]
                out_of_drivable[role, mode] = scored["out_of_drivable"][0]
                clearance[role, mode] = scored["clearance_violation"][0]
                scored_components = scored["components"]
                for name in self._COMPONENT_NAMES:
                    components[name][role, mode] = scored_components[name][0]

        return VehicleModePretrainRewardResult(
            rewards=rewards,
            valid_mode_mask=valid,
            unsafe=unsafe,
            collision=collision,
            out_of_drivable=out_of_drivable,
            clearance_violation=clearance,
            components=components,
        )

    def score_candidates(
        self,
        env: object,
        model_inputs: object,
        candidates: object,
        frozen_argmax_joint_trajectories: object,
        valid_mode_mask: object,
        pretrain: VehicleModePretrainRewardResult,
        *,
        geometry_context: RewardGeometryContext | None = None,
        trajectories_per_mode: int | None = None,
    ) -> VehicleModeRewardResult:
        """Score one noise resample while reusing cached pretrain rewards."""

        candidate_values = self._candidate_values(
            candidates, trajectories_per_mode=trajectories_per_mode
        )
        frozen_argmax = self._frozen_argmax_values(
            frozen_argmax_joint_trajectories
        )
        valid = self._valid_values(valid_mode_mask)
        if not isinstance(pretrain, VehicleModePretrainRewardResult):
            raise JointRewardError(
                "pretrain must be VehicleModePretrainRewardResult"
            )
        if not np.array_equal(pretrain.valid_mode_mask, valid):
            raise JointRewardError("pretrain and candidate valid_mode_mask differ")
        current_shape = (
            NUM_ROLES,
            NUM_MODES,
            candidate_values.shape[2],
        )
        rewards = np.zeros(current_shape, dtype=np.float32)
        unsafe = np.zeros(current_shape, dtype=np.bool_)
        collision = np.zeros(current_shape, dtype=np.bool_)
        out_of_drivable = np.zeros(current_shape, dtype=np.bool_)
        clearance = np.zeros(current_shape, dtype=np.bool_)
        components = {
            name: np.zeros(current_shape, dtype=np.float32)
            for name in self._COMPONENT_NAMES
        }
        context = (
            geometry_context
            if geometry_context is not None
            else self.build_geometry_context(env, model_inputs, frozen_argmax)
        )

        for role in range(NUM_ROLES):
            for mode in range(NUM_MODES):
                if not valid[role, mode]:
                    continue
                scored = self._score_target_group(
                    target_role=role,
                    target_trajectories=candidate_values[role, mode],
                    frozen_argmax_joint_trajectories=frozen_argmax,
                    poses=list(context.poses),
                    drivable=context.bev[role, int(BEVChannel.DRIVABLE)],
                    road_field=context.road_fields[role],
                    background_by_actor=context.backgrounds[role],
                )
                rewards[role, mode] = scored["rewards"]
                unsafe[role, mode] = scored["unsafe"]
                collision[role, mode] = scored["collision"]
                out_of_drivable[role, mode] = scored["out_of_drivable"]
                clearance[role, mode] = scored["clearance_violation"]
                scored_components = scored["components"]
                for name in self._COMPONENT_NAMES:
                    components[name][role, mode] = scored_components[name]

        return VehicleModeRewardResult(
            rewards=rewards,
            pretrain_rewards=pretrain.rewards,
            valid_mode_mask=valid,
            unsafe=unsafe,
            collision=collision,
            out_of_drivable=out_of_drivable,
            clearance_violation=clearance,
            pretrain_unsafe=pretrain.unsafe,
            pretrain_collision=pretrain.collision,
            pretrain_out_of_drivable=pretrain.out_of_drivable,
            pretrain_clearance_violation=pretrain.clearance_violation,
            components=components,
            pretrain_components=pretrain.components,
        )

    def score_all_mode_trajectories(
        self,
        env: object,
        model_inputs: object,
        all_mode_trajectories: object,
        frozen_argmax_joint_trajectories: object,
        valid_mode_mask: object,
        pretrain: VehicleModePretrainRewardResult,
        *,
        geometry_context: RewardGeometryContext | None = None,
    ) -> VehicleModeRewardResult:
        """Score one deterministic trajectory for every vehicle-mode pair.

        This is the exact N=1 counterpart of :meth:`score_candidates` for
        validation.  It avoids materializing N identical trajectory copies.
        """

        all_mode_values = self._frozen_all_values(all_mode_trajectories)
        return self.score_candidates(
            env,
            model_inputs,
            all_mode_values[:, :, None],
            frozen_argmax_joint_trajectories,
            valid_mode_mask,
            pretrain,
            geometry_context=geometry_context,
            trajectories_per_mode=1,
        )

    def score_counterfactuals(
        self,
        env: object,
        model_inputs: object,
        candidates: object,
        frozen_all_mode_trajectories: object,
        frozen_argmax_joint_trajectories: object,
        valid_mode_mask: object,
    ) -> VehicleModeRewardResult:
        """One-shot scoring; online retries should use the split cache API."""

        pretrain = self.score_pretrain(
            env,
            model_inputs,
            frozen_all_mode_trajectories,
            frozen_argmax_joint_trajectories,
            valid_mode_mask,
        )
        return self.score_candidates(
            env,
            model_inputs,
            candidates,
            frozen_argmax_joint_trajectories,
            valid_mode_mask,
            pretrain,
        )
