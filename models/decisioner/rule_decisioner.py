from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from models.decisioner.risk import SimpleRuleRiskDetector
from models.decisioner.risk.safety_potential import pairwise_agent_safety_score
from models.decisioner.rule_decisioner_helper import (
    save_candidate_debug_plot,
    save_lane_pair_debug_plot,
    save_s7_route_lanes_debug_plot,
    save_s8_route_lanes_debug_plot,
)

if TYPE_CHECKING:
    pass

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "decision" / "rule_maker.yaml"


def load_rule_maker_config(
    scenario_id: str | None = None,
    yaml_path: str | Path | None = None,
) -> dict:
    """Load rule-maker params from yaml, merging scenario override on top of default.

    Args:
        scenario_id: If provided, merges the matching entry under scenario_overrides.
        yaml_path: Path to the yaml file. Defaults to configs/decision/rule_maker.yaml.

    Returns:
        Flat dict of merged parameters ready to pass to MultiAgentRuleMaker(**params).
    """
    import yaml as _yaml

    path = Path(yaml_path) if yaml_path is not None else _DEFAULT_CONFIG_PATH
    raw = _yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    params = dict(raw.get("default") or {})
    if scenario_id:
        override = (raw.get("scenario_overrides") or {}).get(scenario_id)
        if override:
            params.update(override)
    return params


class RuleMaker(ABC):
    """Base class for external upper-level decision models.

    Produces a per-agent ego-local target point each step.
    Used when target_guidance_type='external_point' in model config.

    The returned target_point [x, y] (ego-local, metres) is injected into
    planner_batch before extract_rl_context(), so it flows into the model as
    features["target_point"] and conditions the trajectory decoder.
    """

    @abstractmethod
    def compute(
        self,
        env,
        agent_ids: list[str],
        planner_batch: dict[str, dict],
    ) -> dict[str, dict]:
        """Return per-agent decision in ego-local frame.

        Args:
            env: PlatoonEnv instance providing vehicle/lane state.
            agent_ids: Ordered list of active agent IDs.
            planner_batch: Current planner observation dict (read-only from caller's
                perspective; do not mutate — injection is handled externally).

        Returns:
            Mapping agent_id -> {"action": int, "target_point": np.ndarray shape [2]}.
            "action": lane change decision (-1=left, 0=keep, 1=right).
            "target_point": ego-local target [x_fwd, y_lat] in metres.
            Agents absent from the dict receive no guidance this step.
        """
        raise NotImplementedError

    def reset(self, env, agent_ids: list[str]) -> None:  # noqa: ARG002
        """Called at the start of each episode. Override to clear per-episode state."""

    @property
    def is_formation_locked(self) -> bool:
        """Return True when the platoon is in formation-locked mode."""
        return False


class MultiAgentRuleMaker(RuleMaker):
    """Simple multi-agent lane decision rule maker.

    Each controlled vehicle enumerates three coarse decisions (left lane change,
    keep lane, right lane change). The rule maker scores the joint Cartesian
    product of all active agents' coarse trajectories and returns each selected
    trajectory endpoint as the existing ego-local target_point.
    """

    ACTIONS = (-1, 0, 1)  # left, keep, right. Lane ids follow MetaDrive convention.

    def __init__(
        self,
        target_speed_km_h: float = 30.0,
        horizon_s: float = 4.0,
        *,
        num_waypoints: int = 8,
        lane_change_preference: float = 0.0,
        traffic_safety_distance_m: float = 8.0,  # min gap to traffic vehicles for any candidate to be valid
        agent_safety_distance_m: float = 7.0,  # 
        idm_time_headway_s: float = 1.2,
        idm_min_gap_m: float = 6.0,
        idm_max_accel_mps2: float = 1.8,
        idm_comfortable_brake_mps2: float = 2.5,
        idm_delta: float = 4.0,
        mobil_lane_change_threshold: float = 0.2,
        mobil_keep_bias: float = 0.15,
        mobil_rear_base_penalty: float = 3.0,
        mobil_rear_slope_penalty: float = 4.0,
        w_progress: float = 0.04,
        w_mobil: float = 1.1,
        w_keep_bias: float = 0.30,
        lc_cost: float = 0.75,
        w_traffic_clearance: float = 0.03,
        traffic_clearance_cap: float = 1.5,
        w_formation_consistent: float = 0.50,
        w_formation_inconsistent_cost: float = 0.75,
        close_pair_threshold_m: float = 12.0,
        w_close_keep: float = 1.5,
        w_close_lc_same_cost: float = 0.5,
        w_close_lc_diff_cost: float = 1.0,
        locked_on_reset: bool = True,
        risk_ttc_trigger_s: float = 3.0,
        relock_ttc_threshold_s: float = 5.0,
        ideal_following_distance_m: float = 10.0,
        relock_gap_ratio: float = 1.5,
        forced_lane_unlock_wait_steps: int = 10,
    ) -> None:
        self.target_speed_km_h = float(target_speed_km_h)
        self.horizon_s = float(horizon_s)
        self.num_waypoints = max(2, int(num_waypoints))
        self.lane_change_preference = float(lane_change_preference)
        self.traffic_safety_distance_m = float(traffic_safety_distance_m)
        self.agent_safety_distance_m = float(agent_safety_distance_m)
        self.idm_time_headway_s = float(idm_time_headway_s)
        self.idm_min_gap_m = float(idm_min_gap_m)
        self.idm_max_accel_mps2 = float(idm_max_accel_mps2)
        self.idm_comfortable_brake_mps2 = float(idm_comfortable_brake_mps2)
        self.idm_delta = float(idm_delta)
        self.mobil_lane_change_threshold = float(mobil_lane_change_threshold)
        self.mobil_keep_bias = float(mobil_keep_bias)
        self.mobil_rear_base_penalty = float(mobil_rear_base_penalty)
        self.mobil_rear_slope_penalty = float(mobil_rear_slope_penalty)
        self.w_progress = float(w_progress)
        self.w_mobil = float(w_mobil)
        self.w_keep_bias = float(w_keep_bias)
        self.lc_cost = float(lc_cost)
        self.w_traffic_clearance = float(w_traffic_clearance)
        self.traffic_clearance_cap = float(traffic_clearance_cap)
        self.w_formation_consistent = float(w_formation_consistent)
        self.w_formation_inconsistent_cost = float(w_formation_inconsistent_cost)
        self.close_pair_threshold_m = float(close_pair_threshold_m)
        self.w_close_keep = float(w_close_keep)
        self.w_close_lc_same_cost = float(w_close_lc_same_cost)
        self.w_close_lc_diff_cost = float(w_close_lc_diff_cost)
        self.locked_on_reset = bool(locked_on_reset)
        self.risk_ttc_trigger_s = float(risk_ttc_trigger_s)
        self.relock_ttc_threshold_s = float(relock_ttc_threshold_s)
        self.ideal_following_distance_m = float(ideal_following_distance_m)
        self.relock_gap_ratio = float(relock_gap_ratio)
        self.forced_lane_unlock_wait_steps = max(1, int(forced_lane_unlock_wait_steps))
        self._formation_locked = bool(self.locked_on_reset)
        self._risk_detector = SimpleRuleRiskDetector(
            ttc_trigger_s=self.risk_ttc_trigger_s,
            relock_ttc_threshold_s=self.relock_ttc_threshold_s,
            ideal_following_distance_m=self.ideal_following_distance_m,
            relock_gap_ratio=self.relock_gap_ratio,
        )
        self._last_debug: dict | None = None
        self._decision_step = 0
        self._forced_lane_first_step_by_agent: dict[str, int] = {}
        self._forced_lane_frozen_wait_steps_by_agent: dict[str, int] = {}
        self._candidate_debug_plot_counter = 0
        self._lane_pair_debug_plot_counter = 0
        self._s7_route_lanes_debug_plot_counter = 0
        self._s8_route_lanes_debug_plot_counter = 0

    def reset(self, env, agent_ids: list[str]) -> None:  # noqa: ARG002
        self._formation_locked = bool(self.locked_on_reset)
        self._last_debug = None
        self._decision_step = 0
        self._forced_lane_first_step_by_agent.clear()
        self._forced_lane_frozen_wait_steps_by_agent.clear()

    @property
    def is_formation_locked(self) -> bool:
        return self._formation_locked

    def get_last_debug(self) -> dict | None:
        if self._last_debug is None:
            return None
        return copy.deepcopy(self._last_debug)

    def compute(
        self,
        env,
        agent_ids: list[str],
        planner_batch: dict[str, dict],
    ) -> dict[str, np.ndarray]:
        agents = getattr(env, "agents", {})
        traffic_vehicles = self._traffic_vehicles(env)
        candidates_by_agent: dict[str, list[dict]] = {}
        for agent_id in agent_ids:
            vehicle = agents.get(agent_id)
            candidates = self._build_agent_candidates(env, vehicle, traffic_vehicles)
            if not candidates:
                pp = self._from_coarse_fallback(planner_batch, agent_id)
                if pp is None:
                    continue
                candidates = [
                    {
                        "action": 0,
                        "valid": True,
                        "score": 0.0,
                        "trajectory_world": None,
                        "target_point": pp,
                    }
                ]
            candidates_by_agent[agent_id] = candidates

        if not candidates_by_agent:
            return {}

        self._decision_step += 1
        ordered_agent_ids = [agent_id for agent_id in agent_ids if agent_id in candidates_by_agent]
        current_state = "LOCKED" if self._formation_locked else "UNLOCKED"
        forced_combo = self._forced_lane_combo(ordered_agent_ids, candidates_by_agent)
        forced_lane_wait_info = self._update_forced_lane_wait_info(
            ordered_agent_ids=ordered_agent_ids,
            candidates_by_agent=candidates_by_agent,
            forced_combo=forced_combo,
            freeze_wait=current_state != "LOCKED",
        )
        risk_info = self._risk_detector.detect(
            env,
            ordered_agent_ids,
            traffic_vehicles,
            current_state,
            forced_lane_wait_info=forced_lane_wait_info,
        )
        self._formation_locked = risk_info["next_state"] == "LOCKED"

        forced_lane_decision = forced_combo is not None
        if forced_lane_decision:
            best_combo = forced_combo
            best_score = 0.0
        elif self._formation_locked:
            best_combo, best_score = self._best_locked_combo(
                env=env,
                ordered_agent_ids=ordered_agent_ids,
                candidates_by_agent=candidates_by_agent,
                traffic_vehicles=traffic_vehicles,
            )
        else:
            best_combo = None
            best_score = -float("inf")
            candidate_sets = [
                self._forced_candidates_for_agent(candidates_by_agent.get(agent_id, []))
                or candidates_by_agent[agent_id]
                for agent_id in ordered_agent_ids
            ]
            for combo in product(*candidate_sets):
                score = self._score_joint_combo(
                    env=env,
                    ordered_agent_ids=ordered_agent_ids,
                    combo=combo,
                    traffic_vehicles=traffic_vehicles,
                )
                if score > best_score:
                    best_score = score
                    best_combo = combo

        result: dict[str, dict] = {}

        # !!!!!!!!!【DEBUG】
        # if best_combo[0]['action'] != 1:
        #     debug = 1


        if best_combo is None:
            return result
        best_actions: dict[str, int] = {}
        print(f"step = {self._decision_step}")
        for agent_id, selected in zip(ordered_agent_ids, best_combo):

            # !!!!![DEBUG]
            # print(
            #     f"{agent_id} action={int(selected['action'])} "
            #     f"target_lane_index={tuple(selected.get('target_lane_index', ()) or ())} "
            #     f"forced={bool(selected.get('forced_lane_change', False))} "
            #     f"locked={bool(self._formation_locked)}"
            # )

            result[agent_id] = {
                "action": int(selected["action"]),
                "target_point": np.asarray(selected["target_point"], dtype=np.float32).reshape(2),
            }
            best_actions[agent_id] = int(selected["action"])




        # !!!!!!!!!!!!!!!!!!!!!!!固定动作都为0,测试控制(删)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # for agent_id, selected in zip(ordered_agent_ids, best_combo):
        #     keep_candidate = self._candidate_for_action(candidates_by_agent.get(agent_id, []), 0)
        #     if keep_candidate is not None:
        #         selected = keep_candidate

        #     result[agent_id] = {
        #         "action": 0,
        #         "target_point": np.asarray(selected["target_point"], dtype=np.float32).reshape(2),
        #     }
        #     best_actions[agent_id] = 0
        # -----------------------------------------------------------------------------




        if self._formation_locked:
            dynamic_roles = self._locked_roles(ordered_agent_ids)
        else:
            dynamic_roles = self._resolve_dynamic_roles(
                env,
                ordered_agent_ids,
                best_actions,
                traffic_vehicles,
            )
        self._last_debug = {
            "agent_ids": list(ordered_agent_ids),
            "best_actions": best_actions,
            "best_score": float(best_score),
            "formation_locked": bool(self._formation_locked),
            "risk_triggered": bool(risk_info.get("triggered", False)),
            "state_transition": risk_info.get("transition"),
            "forced_lane_decision": bool(forced_lane_decision),
            "forced_lane_wait_info": forced_lane_wait_info,
            "risk_info": risk_info,
            "dynamic_roles": dynamic_roles,
            "candidates_by_agent": {
                agent_id: [
                    self._debug_candidate(candidate, selected=bool(int(candidate["action"]) == best_actions[agent_id]))
                    for candidate in candidates_by_agent.get(agent_id, [])
                ]
                for agent_id in ordered_agent_ids
            },
        }
        return result

    def debug_compute(
        self,
        env,
        agent_ids: list[str],
        planner_batch: dict[str, dict],  # noqa: ARG002
        manual_actions: dict[str, int],
    ) -> dict[str, np.ndarray]:
        """Return decisions for explicitly requested per-agent actions."""
        agents = getattr(env, "agents", {}) or {}
        traffic_vehicles = self._traffic_vehicles(env)
        requested_actions = {str(agent_id): int(action) for agent_id, action in dict(manual_actions).items()}

        result: dict[str, dict] = {}
        best_actions: dict[str, int] = {}
        invalid_actions: dict[str, int] = {}
        candidates_by_agent: dict[str, list[dict]] = {}

        for agent_id in agent_ids:
            vehicle = agents.get(agent_id)
            candidates = self._build_agent_candidates(env, vehicle, traffic_vehicles)
            candidates_by_agent[agent_id] = candidates
            if agent_id not in requested_actions:
                continue
            action = int(requested_actions[agent_id])
            candidate = self._candidate_for_action(candidates, action)
            if candidate is None:
                invalid_actions[agent_id] = action
                continue
            result[agent_id] = {
                "action": int(candidate["action"]),
                "target_point": np.asarray(candidate["target_point"], dtype=np.float32).reshape(2),
            }
            best_actions[agent_id] = int(candidate["action"])

        self._last_debug = {
            "manual_mode": True,
            "agent_ids": list(agent_ids),
            "requested_actions": requested_actions,
            "best_actions": best_actions,
            "invalid_actions": invalid_actions,
            "formation_locked": bool(self._formation_locked),
            "risk_triggered": False,
            "risk_info": {"triggered": False, "reasons": [], "per_agent": {}},
            "dynamic_roles": {},
            "candidates_by_agent": {
                agent_id: [
                    self._debug_candidate(
                        candidate,
                        selected=bool(agent_id in best_actions and int(candidate["action"]) == best_actions[agent_id]),
                    )
                    for candidate in candidates_by_agent.get(agent_id, [])
                ]
                for agent_id in agent_ids
            },
        }
        return result

    @staticmethod
    def _debug_candidate(candidate: dict, *, selected: bool) -> dict:
        return {
            "action": int(candidate["action"]),
            "score": float(candidate.get("score", 0.0)),
            "trajectory_world": (
                np.asarray(candidate["trajectory_world"], dtype=np.float32).tolist()
                if candidate.get("trajectory_world") is not None
                else None
            ),
            "target_point": np.asarray(candidate["target_point"], dtype=np.float32).tolist(),
            "source_lane_index": tuple(candidate.get("source_lane_index", ()) or ()),
            "target_lane_index": tuple(candidate.get("target_lane_index", ()) or ()),
            "selected": bool(selected),
            "forced_lane_change": bool(candidate.get("forced_lane_change", False)),
        }

    @staticmethod
    def _forced_lane_combo(
        ordered_agent_ids: list[str],
        candidates_by_agent: dict[str, list[dict]],
    ) -> tuple[dict, ...] | None:
        combo = []
        for agent_id in ordered_agent_ids:
            forced = [
                candidate
                for candidate in candidates_by_agent.get(agent_id, [])
                if bool(candidate.get("forced_lane_change", False))
            ]
            if not forced:
                return None
            combo.append(forced[0])
        return tuple(combo)

    @staticmethod
    def _forced_candidates_for_agent(candidates: list[dict]) -> list[dict]:
        return [
            candidate
            for candidate in candidates
            if bool(candidate.get("forced_lane_change", False))
        ]

    def _update_forced_lane_wait_info(
        self,
        *,
        ordered_agent_ids: list[str],
        candidates_by_agent: dict[str, list[dict]],
        forced_combo: tuple[dict, ...] | None,
        freeze_wait: bool = False,
    ) -> dict:
        forced_agents = [
            agent_id
            for agent_id in ordered_agent_ids
            if any(bool(candidate.get("forced_lane_change", False)) for candidate in candidates_by_agent.get(agent_id, []))
        ]
        all_forced_ready = forced_combo is not None
        current_forced = set(forced_agents)
        if not forced_agents:
            self._forced_lane_first_step_by_agent.clear()
            self._forced_lane_frozen_wait_steps_by_agent.clear()
        elif not freeze_wait:
            for agent_id in list(self._forced_lane_first_step_by_agent):
                if agent_id not in current_forced:
                    self._forced_lane_first_step_by_agent.pop(agent_id, None)
                    self._forced_lane_frozen_wait_steps_by_agent.pop(agent_id, None)
            for agent_id in forced_agents:
                self._forced_lane_first_step_by_agent.setdefault(agent_id, int(self._decision_step))
        if all_forced_ready and not freeze_wait:
            self._forced_lane_first_step_by_agent.clear()
            self._forced_lane_frozen_wait_steps_by_agent.clear()

        first_steps = {
            agent_id: int(step)
            for agent_id, step in self._forced_lane_first_step_by_agent.items()
            if agent_id in current_forced
        }
        partial_forced = bool(forced_agents) and not all_forced_ready
        wait_steps_by_agent = {
            agent_id: max(0, int(self._decision_step) - int(first_step) + 1)
            for agent_id, first_step in first_steps.items()
        }
        if freeze_wait:
            wait_steps_by_agent = {
                agent_id: int(self._forced_lane_frozen_wait_steps_by_agent.get(agent_id, 0) or 0)
                for agent_id in first_steps
            }
        else:
            self._forced_lane_frozen_wait_steps_by_agent = dict(wait_steps_by_agent)
        partial_wait_steps = max(wait_steps_by_agent.values(), default=0) if partial_forced else 0
        return {
            "decision_step": int(self._decision_step),
            "threshold_steps": int(self.forced_lane_unlock_wait_steps),
            "forced_agents": list(forced_agents),
            "all_forced_ready": bool(all_forced_ready),
            "partial_forced": bool(partial_forced),
            "partial_forced_wait_steps": int(partial_wait_steps),
            "forced_wait_active": bool(self._forced_lane_first_step_by_agent),
            "forced_wait_frozen": bool(freeze_wait),
            "forced_lane_first_step": first_steps,
            "forced_lane_wait_steps_by_agent": wait_steps_by_agent,
        }

    def _best_locked_combo(
        self,
        *,
        env,
        ordered_agent_ids: list[str],
        candidates_by_agent: dict[str, list[dict]],
        traffic_vehicles: list,
    ):
        best_combo = None
        best_score = -float("inf")
        for action in self.ACTIONS:
            combo = []
            valid = True
            for agent_id in ordered_agent_ids:
                candidate = self._candidate_for_action(candidates_by_agent.get(agent_id, []), int(action))
                if candidate is None:
                    valid = False
                    break
                combo.append(candidate)
            if not valid:
                continue
            combo_tuple = tuple(combo)
            score = self._score_joint_combo(
                env=env,
                ordered_agent_ids=ordered_agent_ids,
                combo=combo_tuple,
                traffic_vehicles=traffic_vehicles,
            )
            if score > best_score:
                best_score = score
                best_combo = combo_tuple
        return best_combo, best_score

    @staticmethod
    def _candidate_for_action(candidates: list[dict], action: int) -> dict | None:
        for candidate in candidates:
            if int(candidate.get("action", 0)) == int(action):
                return candidate
        return None

    @staticmethod
    def _locked_roles(ordered_agent_ids: list[str]) -> dict[str, str]:
        roles: dict[str, str] = {}
        for idx, agent_id in enumerate(ordered_agent_ids):
            roles[agent_id] = "leader" if idx == 0 else "follower"
        return roles

    def _resolve_dynamic_roles(
        self,
        env,
        ordered_agent_ids: list[str],
        best_actions: dict[str, int],
        traffic_vehicles: list,
    ) -> dict[str, str]:
        agents = getattr(env, "agents", {}) or {}
        roles: dict[str, str] = {}
        for idx, agent_id in enumerate(ordered_agent_ids):
            if idx == 0:
                roles[agent_id] = "leader"
                continue
            prev_agent_id = ordered_agent_ids[idx - 1]
            vehicle = agents.get(agent_id)
            prev_vehicle = agents.get(prev_agent_id)
            if (
                vehicle is None
                or prev_vehicle is None
                or int(best_actions.get(agent_id, 0)) != int(best_actions.get(prev_agent_id, 0))
                or not self._same_lane(vehicle, prev_vehicle)
                or self._has_background_vehicle_between(vehicle, prev_vehicle, traffic_vehicles)
            ):
                roles[agent_id] = "leader"
            else:
                roles[agent_id] = "follower"
        return roles

    def _build_agent_candidates(self, env, vehicle, traffic_vehicles: list) -> list[dict]:
        if vehicle is None:
            return []
        other_vehicles = self._candidate_obstacle_vehicles(env, vehicle, traffic_vehicles)
        candidates = []
        for action in self.ACTIONS:
            candidate = self._build_coarse_trajectory(env, vehicle, int(action), other_vehicles)
            if candidate is not None:
                candidates.append(candidate)

        debug = 0
        if debug:
            self._candidate_debug_plot_counter += 1
            save_candidate_debug_plot(vehicle, candidates, self._candidate_debug_plot_counter)

        keep_candidate = next((candidate for candidate in candidates if int(candidate.get("action", 0)) == 0), None)
        keep_accel = float(keep_candidate.get("terminal_accel_mps2", 0.0)) if keep_candidate is not None else 0.0
        for candidate in candidates:
            action = int(candidate.get("action", 0))
            accel = float(candidate.get("terminal_accel_mps2", 0.0))
            if action == 0:
                candidate["mobil_gain"] = float(self.mobil_keep_bias)
            else:
                rear_penalty = self._mobil_rear_safety_penalty(candidate)
                candidate["mobil_gain"] = float(accel - keep_accel - self.mobil_lane_change_threshold - rear_penalty)
        return candidates

    def _build_coarse_trajectory(self, env, vehicle, action: int, other_vehicles: list) -> dict | None:
        # Step 1: 计算当前车辆所在lane和目标lane
        source_lane = getattr(vehicle, "lane", None)
        target_lane = self._target_lane(env, vehicle, source_lane, action)

        debug = 0
        if debug == 1 and action == 1:
            self._lane_pair_debug_plot_counter += 1
            # 画当前lane和目标lane
            save_lane_pair_debug_plot(
                vehicle,
                source_lane,
                target_lane,
                action,
                self._lane_pair_debug_plot_counter,
            )

        if source_lane is None or target_lane is None:
            return None
        try:
            # Step 2: 将车辆当前位置映射到source_lane的s坐标，计算fallback_progress
            pos = np.asarray(vehicle.position[:2], dtype=np.float32)
            s_ego, _ = source_lane.local_coordinates(pos)
            fallback_progress = self._target_progress(vehicle)

            # Step 3: 将source_lane和target_lane映射到lane_chain，计算总长度，计算start_s
            source_lane_chain = self._reference_lane_chain(env, vehicle, source_lane)
            target_lane_chain = self._reference_lane_chain(env, vehicle, target_lane)
            total_reference_length = max(
                self._lane_chain_total_length(source_lane_chain),
                self._lane_chain_total_length(target_lane_chain),
            )
            lane_len = max(float(s_ego) + fallback_progress, total_reference_length)
            start_s = float(np.clip(float(s_ego), 0.0, lane_len))

            # Step 4: 根据前车和后车的距离，计算terminal_accel_mps2和terminal_speed_km_h
            target_s, _ = target_lane.local_coordinates(pos)
            front_vehicle, front_gap_m = self._front_vehicle_on_lane(
                target_lane,
                float(target_s),
                other_vehicles,
                ego_length=float(getattr(vehicle, "LENGTH", 4.5) or 4.5),
            )
            rear_vehicle, rear_gap_m = self._rear_vehicle_on_lane(
                target_lane,
                float(target_s),
                other_vehicles,
                ego_length=float(getattr(vehicle, "LENGTH", 4.5) or 4.5),
            )

            terminal_accel_mps2 = self._idm_acceleration(vehicle, front_vehicle, front_gap_m)
            terminal_speed_km_h = self._terminal_speed_km_h(vehicle, terminal_accel_mps2)
            rear_vehicle_post_accel_mps2 = self._rear_vehicle_post_accel(vehicle, rear_vehicle, rear_gap_m)

            # Step 5: 积分加速度得到车辆纵向轨迹
            longs = self._integrate_longitudinal_profile(
                start_s=start_s,
                speed_km_h=float(getattr(vehicle, "speed_km_h", 0.0) or 0.0),
                terminal_accel_mps2=terminal_accel_mps2,
                lane_len=lane_len,
            )
            end_s = float(longs[-1]) if longs.size > 0 else start_s

            # Step 6: 转换成世界坐标
            trajectory = self._build_world_trajectory_from_lane_chains(
                source_lane_chain=source_lane_chain,
                target_lane_chain=target_lane_chain,
                longitudinals=longs,
                action=action,
            )

            # Step 7: 计算目标点在车辆坐标系下的坐标
            target_point = self._world_to_ego_local(vehicle, trajectory[-1])

            candidate = {
                "action": int(action),
                "valid": True,
                "score": 0.0,
                "trajectory_world": trajectory,
                "target_point": target_point,
                "source_lane_index": tuple(getattr(source_lane, "index", ()) or ()),
                "target_lane_index": tuple(getattr(target_lane, "index", ()) or ()),
                "start_s": float(start_s),
                "end_s": float(end_s),
                "front_vehicle_speed_km_h": float(getattr(front_vehicle, "speed_km_h", 0.0) or 0.0) if front_vehicle is not None else None,
                "front_gap_m": None if front_gap_m is None else float(front_gap_m),
                "rear_vehicle_speed_km_h": float(getattr(rear_vehicle, "speed_km_h", 0.0) or 0.0) if rear_vehicle is not None else None,
                "rear_gap_m": None if rear_gap_m is None else float(rear_gap_m),
                "terminal_accel_mps2": float(terminal_accel_mps2),
                "terminal_speed_km_h": float(terminal_speed_km_h),
                "rear_vehicle_post_accel_mps2": None if rear_vehicle_post_accel_mps2 is None else float(rear_vehicle_post_accel_mps2),
                "mobil_gain": 0.0,
            }
            if self._is_s8_exit_route(env) and int(action) == 1:
                candidate["forced_lane_change"] = True
            if self._is_s7_forced_lane_candidate(env, candidate):
                candidate["forced_lane_change"] = True
            return candidate
        except Exception:
            return None

    def _target_progress(self, vehicle) -> float:
        speed_km_h = float(getattr(vehicle, "speed_km_h", 0.0) or 0.0)
        target_speed = max(self.target_speed_km_h, speed_km_h)
        return max(5.0, target_speed / 3.6 * self.horizon_s)

    def _integrate_longitudinal_profile(
        self,
        *,
        start_s: float,
        speed_km_h: float,
        terminal_accel_mps2: float,
        lane_len: float,
    ) -> np.ndarray:
        if self.num_waypoints <= 1:
            return np.asarray([start_s], dtype=np.float32)
        v = np.clip(float(speed_km_h) / 3.6, 0.0, float(self.target_speed_km_h) / 3.6)  # 速度上下限
        dt = float(self.horizon_s) / float(self.num_waypoints - 1)
        longs = [float(start_s)]
        s = float(start_s)
        accel = float(terminal_accel_mps2)
        for _ in range(self.num_waypoints - 1):
            s = min(float(lane_len), s + v * dt + 0.5 * accel * dt * dt)
            v = max(0.0, v + accel * dt)
            longs.append(float(s))
        return np.asarray(longs, dtype=np.float32)

    def _lane_change_lateral_offsets(
        self,
        *,
        source_lane,
        target_lane,
        action: int,
        num_points: int,
        reference_s: float,
    ) -> np.ndarray:
        if num_points <= 0:
            return np.zeros((0,), dtype=np.float32)
        if action == 0:
            return np.zeros((num_points,), dtype=np.float32)
        target_center = np.asarray(target_lane.position(float(reference_s), 0.0)[:2], dtype=np.float32)
        _, target_lateral = source_lane.local_coordinates(target_center)
        target_lateral = float(target_lateral)
        if num_points == 1:
            return np.asarray([target_lateral], dtype=np.float32)
        us = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
        # Quintic smootherstep: zero first/second derivative at both ends.
        ratios = 6.0 * us**5 - 15.0 * us**4 + 10.0 * us**3
        return (target_lateral * ratios).astype(np.float32, copy=False)

    def _build_world_trajectory_from_lane_chains(
        self,
        *,
        source_lane_chain: list,
        target_lane_chain: list,
        longitudinals: np.ndarray,
        action: int,
    ) -> np.ndarray:
        if action == 0:
            return np.asarray(
                [self._position_along_lane_chain(source_lane_chain, float(s), 0.0)[:2] for s in longitudinals],
                dtype=np.float32,
            )
        num_points = int(len(longitudinals))
        if num_points <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        us = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
        ratios = 6.0 * us**5 - 15.0 * us**4 + 10.0 * us**3
        points = []
        for s, ratio in zip(longitudinals, ratios):
            src = self._position_along_lane_chain(source_lane_chain, float(s), 0.0)[:2]
            tgt = self._position_along_lane_chain(target_lane_chain, float(s), 0.0)[:2]
            world = (1.0 - float(ratio)) * np.asarray(src, dtype=np.float32) + float(ratio) * np.asarray(tgt, dtype=np.float32)
            points.append(world.astype(np.float32, copy=False))
        return np.asarray(points, dtype=np.float32)

    def _terminal_speed_km_h(self, vehicle, terminal_accel_mps2: float) -> float:
        v0 = max(0.0, float(getattr(vehicle, "speed_km_h", 0.0) or 0.0) / 3.6)
        v1 = max(0.0, v0 + float(terminal_accel_mps2) * float(self.horizon_s))
        return float(v1 * 3.6)

    @staticmethod
    def _lane_chain_total_length(lanes: list) -> float:
        return float(sum(float(getattr(lane, "length", 0.0) or 0.0) for lane in lanes))

    def _reference_lane_chain(self, env, vehicle, source_lane) -> list:
        if self._is_s8_exit_route(env):
            s8_chain = self._S8_reference_lane_chain(env, vehicle, source_lane)
            if s8_chain is not None and len(s8_chain) > 1:
                return s8_chain
        return self._generic_reference_lane_chain(env, vehicle, source_lane)

    @staticmethod
    def _config_value(config, key: str):
        if config is None:
            return None
        if hasattr(config, "get"):
            try:
                return config.get(key)
            except Exception:
                pass
        return getattr(config, key, None)

    @classmethod
    def _is_s8_exit_route(cls, env) -> bool:
        config = getattr(env, "config", {}) or {}
        return (
            cls._config_value(config, "scenario_id") == "S8_ego_exit_to_ramp"
            and cls._config_value(config, "local_route") == "R6_exit_to_ramp"
        )

    @classmethod
    def _is_s7_merge_route(cls, env) -> bool:
        config = getattr(env, "config", {}) or {}
        return (
            cls._config_value(config, "scenario_id") == "S7_ego_merge_from_ramp"
            and cls._config_value(config, "local_route") == "R7_merge_core"
        )

    @classmethod
    def _is_s7_forced_lane_candidate(cls, env, candidate: dict) -> bool:
        if not cls._is_s7_merge_route(env):
            return False
        if int(candidate.get("action", 0)) != -1:
            return False
        target_lane_index = tuple(candidate.get("target_lane_index", ()) or ())
        source_lane_index = tuple(candidate.get("source_lane_index", ()) or ())
        if source_lane_index == ('9g0_0_', '9g1_4_', 0) and target_lane_index == ('9g0_0_', '9g0_1_', 2):
            return True
        return False

    def _S8_reference_lane_chain(self, env, vehicle, source_lane) -> list | None:
        lane_chain = [source_lane]
        lane_index = tuple(getattr(source_lane, "index", ()) or ())
        if len(lane_index) < 3:
            return None
        road_network = getattr(getattr(getattr(env, "engine", None), "current_map", None), "road_network", None)
        if road_network is None:
            return None

        navigation = getattr(vehicle, "navigation", None)
        checkpoints = list(getattr(navigation, "checkpoints", []) or [])
        if len(checkpoints) < 2:
            return None

        found = False
        next_pairs: list[tuple[str, str]] = []
        for idx in range(len(checkpoints) - 1):
            if not found:
                if checkpoints[idx] == lane_index[0] and checkpoints[idx + 1] == lane_index[1]:
                    found = True
                continue
            next_pairs.append((checkpoints[idx], checkpoints[idx + 1]))
        if not found:
            return None

        for from_node, to_node in next_pairs:
            lanes = self._graph_lanes(road_network, from_node, to_node)
            if not lanes:
                break
            if len(lanes) == 1:
                lane_chain.append(lanes[0])
                continue
            lane_chain.append(self._rightmost_lane(lanes))
        return lane_chain

    @staticmethod
    def _rightmost_lane(lanes: list):
        try:
            return max(lanes, key=lambda lane: int(tuple(getattr(lane, "index", ()) or (0, 0, 0))[2]))
        except Exception:
            return lanes[-1]

    def _generic_reference_lane_chain(self, env, vehicle, source_lane) -> list:
        lane_chain = [source_lane]
        lane_index = tuple(getattr(source_lane, "index", ()) or ())
        if len(lane_index) < 3:
            return lane_chain
        road_network = getattr(getattr(getattr(env, "engine", None), "current_map", None), "road_network", None)
        if road_network is None:
            return lane_chain

        navigation = getattr(vehicle, "navigation", None)
        checkpoints = list(getattr(navigation, "checkpoints", []) or [])
        lane_slot = int(lane_index[2])
        if len(checkpoints) >= 2:
            next_pairs: list[tuple[str, str]] = []
            found = False
            for idx in range(len(checkpoints) - 1):
                if not found:
                    if checkpoints[idx] == lane_index[0] and checkpoints[idx + 1] == lane_index[1]:
                        found = True
                    continue
                next_pairs.append((checkpoints[idx], checkpoints[idx + 1]))
            for from_node, to_node in next_pairs:
                lanes = self._graph_lanes(road_network, from_node, to_node)
                if not lanes:
                    break
                lane_chain.append(lanes[min(lane_slot, len(lanes) - 1)])
            return lane_chain

        current_to = lane_index[1]
        visited = {tuple(lane_index[:2])}
        for _ in range(4):
            next_map = getattr(road_network, "graph", {}).get(current_to, {})
            if len(next_map) != 1:
                break
            next_to = next(iter(next_map.keys()))
            if (current_to, next_to) in visited:
                break
            lanes = self._graph_lanes(road_network, current_to, next_to)
            if not lanes:
                break
            lane_chain.append(lanes[min(lane_slot, len(lanes) - 1)])
            visited.add((current_to, next_to))
            current_to = next_to
        return lane_chain

    @staticmethod
    def _graph_lanes(road_network, from_node, to_node):
        try:
            return list(road_network.graph[from_node][to_node])
        except Exception:
            return []

    @staticmethod
    def _position_along_lane_chain(lane_chain: list, longitudinal: float, lateral: float):
        if not lane_chain:
            return np.asarray([0.0, 0.0], dtype=np.float32)
        remaining = float(longitudinal)
        for lane in lane_chain:
            lane_length = float(getattr(lane, "length", 0.0) or 0.0)
            if remaining <= lane_length:
                return np.asarray(lane.position(remaining, lateral)[:2], dtype=np.float32)
            remaining -= lane_length
        last_lane = lane_chain[-1]
        last_len = float(getattr(last_lane, "length", 0.0) or 0.0)
        return np.asarray(last_lane.position(last_len, lateral)[:2], dtype=np.float32)

    def _idm_acceleration(self, vehicle, front_vehicle, front_gap_m: float | None) -> float:
        v = max(0.0, float(getattr(vehicle, "speed_km_h", 0.0) or 0.0) / 3.6)
        v_des = max(1.0, float(self.target_speed_km_h) / 3.6)
        a_max = max(0.1, self.idm_max_accel_mps2)
        b = max(0.1, self.idm_comfortable_brake_mps2)
        free_term = 1.0 - (v / v_des) ** self.idm_delta
        interaction_term = 0.0
        if front_vehicle is not None and front_gap_m is not None:
            v_front = max(0.0, float(getattr(front_vehicle, "speed_km_h", 0.0) or 0.0) / 3.6)
            delta_v = v - v_front
            s_alpha = max(0.1, float(front_gap_m))
            s_star = self.idm_min_gap_m + max(
                0.0,
                v * self.idm_time_headway_s + (v * delta_v) / (2.0 * np.sqrt(a_max * b)),
            )
            interaction_term = (s_star / s_alpha) ** 2
        accel = a_max * (free_term - interaction_term)
        return float(np.clip(accel, -b, a_max))

    def _front_vehicle_on_lane(self, lane, ego_s: float, other_vehicles: list, ego_length: float = 4.5):
        best_vehicle = None
        best_gap = None
        lane_index = tuple(getattr(lane, "index", ()) or ())
        for vehicle in other_vehicles:
            vehicle_lane = getattr(vehicle, "lane", None)
            if vehicle_lane is None or tuple(getattr(vehicle_lane, "index", ()) or ()) != lane_index:
                continue
            position = np.asarray(getattr(vehicle, "position", (0.0, 0.0))[:2], dtype=np.float32)
            veh_s, veh_t = lane.local_coordinates(position)
            lane_width = float(getattr(lane, "width", 3.5) or 3.5)
            if abs(float(veh_t)) > 0.5 * lane_width:
                continue
            gap = (
                float(veh_s)
                - float(ego_s)
                - 0.5 * float(getattr(vehicle, "LENGTH", 4.5) or 4.5)
                - 0.5 * float(ego_length)
            )
            if gap <= 0.0:
                continue
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_vehicle = vehicle
        return best_vehicle, best_gap

    def _rear_vehicle_on_lane(self, lane, ego_s: float, other_vehicles: list, ego_length: float = 4.5):
        best_vehicle = None
        best_gap = None
        lane_index = tuple(getattr(lane, "index", ()) or ())
        for vehicle in other_vehicles:
            vehicle_lane = getattr(vehicle, "lane", None)
            if vehicle_lane is None or tuple(getattr(vehicle_lane, "index", ()) or ()) != lane_index:
                continue
            position = np.asarray(getattr(vehicle, "position", (0.0, 0.0))[:2], dtype=np.float32)
            veh_s, veh_t = lane.local_coordinates(position)
            lane_width = float(getattr(lane, "width", 3.5) or 3.5)
            if abs(float(veh_t)) > 0.5 * lane_width:
                continue
            gap = (
                float(ego_s)
                - float(veh_s)
                - 0.5 * float(getattr(vehicle, "LENGTH", 4.5) or 4.5)
                - 0.5 * float(ego_length)
            )
            if gap <= 0.0:
                continue
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_vehicle = vehicle
        return best_vehicle, best_gap

    def _rear_vehicle_post_accel(self, ego_vehicle, rear_vehicle, rear_gap_m: float | None) -> float | None:
        if rear_vehicle is None or rear_gap_m is None:
            return None
        virtual_front = SimpleNamespace(speed_km_h=float(getattr(ego_vehicle, "speed_km_h", 0.0) or 0.0))
        return self._idm_acceleration(rear_vehicle, virtual_front, rear_gap_m)

    def _mobil_rear_safety_penalty(self, candidate: dict) -> float:
        rear_post_accel = candidate.get("rear_vehicle_post_accel_mps2")
        if rear_post_accel is None:
            return 0.0
        rear_brake = -float(rear_post_accel)
        threshold = self.idm_comfortable_brake_mps2
        if rear_brake < threshold:
            return 0.0
        excess_brake = max(0.0, rear_brake - threshold)
        return self.mobil_rear_base_penalty + self.mobil_rear_slope_penalty * excess_brake

    @staticmethod
    def _candidate_obstacle_vehicles(env, ego_vehicle, traffic_vehicles: list) -> list:
        others = []
        agents = getattr(env, "agents", {}) or {}
        for vehicle in list(agents.values()) + list(traffic_vehicles):
            if vehicle is None or vehicle is ego_vehicle:
                continue
            others.append(vehicle)
        return others

    def _score_joint_combo(
        self,
        env,
        ordered_agent_ids: list[str],
        combo,
        traffic_vehicles: list,
    ) -> float:
        # !!!!!!!!只保留 效率评分
        score = 0.0
        agents = getattr(env, "agents", {}) or {}

        for agent_id, candidate in zip(ordered_agent_ids, combo):
            trajectory = candidate.get("trajectory_world")
            if trajectory is None or len(trajectory) == 0:
                continue
            action = int(candidate.get("action", 0))
            progress = float(np.linalg.norm(trajectory[-1] - trajectory[0]))
            # score += self.w_progress * progress
            score += self.w_mobil * float(candidate.get("mobil_gain", 0.0))
            # if action == 0:
            #     score += self.w_keep_bias
            # else:
            #     if not (self._is_s8_exit_route(env) and s8_force_score is not None):
            #         score += self.lane_change_preference
            #     score -= self.lc_cost

        #     for traffic_vehicle in traffic_vehicles:
        #         traffic_pos = np.asarray(getattr(traffic_vehicle, "position", (0.0, 0.0))[:2], dtype=np.float32)
        #         min_dist = float(np.min(np.linalg.norm(trajectory - traffic_pos.reshape(1, 2), axis=1)))
        #         if min_dist < self.traffic_safety_distance_m:
        #             score -= 100.0 * (self.traffic_safety_distance_m - min_dist + 1.0)
        #         else:
        #             score += min(self.traffic_clearance_cap, self.w_traffic_clearance * min_dist)

        # score += self._joint_agent_safety_score(combo)

        # for idx in range(1, len(ordered_agent_ids)):
        #     prev_agent_id = ordered_agent_ids[idx - 1]
        #     agent_id = ordered_agent_ids[idx]
        #     prev_candidate = combo[idx - 1]
        #     candidate = combo[idx]
        #     prev_vehicle = agents.get(prev_agent_id)
        #     vehicle = agents.get(agent_id)
        #     if prev_vehicle is None or vehicle is None:
        #         continue
        #     prev_action = int(prev_candidate.get("action", 0))
        #     action = int(candidate.get("action", 0))
        #     if prev_action == action:
        #         score += self.w_formation_consistent
        #     else:
        #         score -= self.w_formation_inconsistent_cost

        #     same_source_lane = tuple(prev_candidate.get("source_lane_index", ())) == tuple(candidate.get("source_lane_index", ()))
        #     close_pair = self._current_pair_distance(prev_vehicle, vehicle) <= self.close_pair_threshold_m
        #     if same_source_lane and close_pair:
        #         if prev_action == 0 and action == 0:
        #             score += self.w_close_keep
        #         elif prev_action == action:
        #             score -= self.w_close_lc_same_cost
        #         else:
        #             score -= self.w_close_lc_diff_cost

        return float(score)

    def _joint_agent_safety_score(self, combo) -> float:
        score = 0.0
        for i in range(len(combo)):
            traj_i = combo[i].get("trajectory_world")
            if traj_i is None:
                continue
            for j in range(i + 1, len(combo)):
                traj_j = combo[j].get("trajectory_world")
                if traj_j is None:
                    continue
                score += pairwise_agent_safety_score(
                    traj_i,
                    traj_j,
                    safe_distance_m=self.agent_safety_distance_m,
                )
        return float(score)

    @staticmethod
    def _current_pair_distance(vehicle_a, vehicle_b) -> float:
        pos_a = np.asarray(getattr(vehicle_a, "position", (0.0, 0.0))[:2], dtype=np.float32)
        pos_b = np.asarray(getattr(vehicle_b, "position", (0.0, 0.0))[:2], dtype=np.float32)
        return float(np.linalg.norm(pos_a - pos_b))

    @staticmethod
    def _same_lane(vehicle_a, vehicle_b) -> bool:
        lane_a = getattr(getattr(vehicle_a, "lane", None), "index", None)
        lane_b = getattr(getattr(vehicle_b, "lane", None), "index", None)
        return tuple(lane_a or ()) == tuple(lane_b or ())

    @classmethod
    def _has_background_vehicle_between(cls, rear_vehicle, front_vehicle, traffic_vehicles: list) -> bool:
        rear_lane = getattr(rear_vehicle, "lane", None)
        front_lane = getattr(front_vehicle, "lane", None)
        if rear_lane is None or front_lane is None or tuple(getattr(rear_lane, "index", ()) or ()) != tuple(getattr(front_lane, "index", ()) or ()):
            return True

        rear_pos = np.asarray(getattr(rear_vehicle, "position", (0.0, 0.0))[:2], dtype=np.float32)
        front_pos = np.asarray(getattr(front_vehicle, "position", (0.0, 0.0))[:2], dtype=np.float32)
        rear_s, _ = rear_lane.local_coordinates(rear_pos)
        front_s, _ = front_lane.local_coordinates(front_pos)
        s_min = float(min(rear_s, front_s))
        s_max = float(max(rear_s, front_s))
        if s_max - s_min <= 1e-3:
            return False

        lane_width = float(getattr(rear_lane, "width", 3.5) or 3.5)
        for traffic_vehicle in traffic_vehicles:
            if traffic_vehicle is rear_vehicle or traffic_vehicle is front_vehicle:
                continue
            traffic_lane = getattr(traffic_vehicle, "lane", None)
            if traffic_lane is None:
                continue
            if tuple(getattr(traffic_lane, "index", ()) or ()) != tuple(getattr(rear_lane, "index", ()) or ()):
                continue
            traffic_pos = np.asarray(getattr(traffic_vehicle, "position", (0.0, 0.0))[:2], dtype=np.float32)
            traffic_s, traffic_t = rear_lane.local_coordinates(traffic_pos)
            if abs(float(traffic_t)) > 0.5 * lane_width:
                continue
            if s_min < float(traffic_s) < s_max:
                return True
        return False

    def _target_lane(self, env, vehicle, source_lane, action: int):
        if source_lane is None:
            return None
        if action == 0:
            return source_lane
        if self._is_s8_exit_route(env) and int(action) == 1:
            s8_target = self._S8_downstream_target_lane(env, vehicle, source_lane)
            if s8_target is not None:
                return s8_target
        if self._is_s7_merge_route(env) and int(action) == -1:
            s7_target = self._S7_downstream_target_lane(env, vehicle, source_lane)
            if s7_target is not None:
                return s7_target
        lane_index = tuple(getattr(source_lane, "index", ()) or ())
        if len(lane_index) < 3:
            return None
        target_index = tuple(list(lane_index[:2]) + [int(lane_index[2]) + int(action)])
        road_network = getattr(getattr(getattr(env, "engine", None), "current_map", None), "road_network", None)
        if road_network is None or not hasattr(road_network, "get_lane"):
            return None
        try:
            return road_network.get_lane(target_index)
        except Exception:
            return None

    def _S8_downstream_target_lane(self, env, vehicle, source_lane):
        debug = 0
        if debug:
            road_network = getattr(getattr(getattr(env, "engine", None), "current_map", None), "road_network", None)
            navigation = getattr(vehicle, "navigation", None)
            checkpoints = list(getattr(navigation, "checkpoints", []) or [])
            if road_network is not None and len(checkpoints) >= 2:
                self._s8_route_lanes_debug_plot_counter += 1
                save_s8_route_lanes_debug_plot(
                    vehicle,
                    road_network,
                    checkpoints,
                    source_lane,
                    self._s8_route_lanes_debug_plot_counter,
                )

        lane_index = tuple(getattr(source_lane, "index", ()) or ())
        if lane_index == ("3C0_1_", "4G0_0_", 2):
            road_network = getattr(getattr(getattr(env, "engine", None), "current_map", None), "road_network", None)
            if road_network is not None and hasattr(road_network, "get_lane"):
                try:
                    return road_network.get_lane(("3C0_1_", "4G1_0_", 0))
                except Exception:
                    return None
        if len(lane_index) < 3:
            return None
        s8_chain = self._S8_reference_lane_chain(env, vehicle, source_lane)
        if not s8_chain:
            return None
        source_road = tuple(lane_index[:2])
        for lane in s8_chain:
            candidate_index = tuple(getattr(lane, "index", ()) or ())
            if len(candidate_index) >= 3 and tuple(candidate_index[:2]) != source_road:
                return lane
        return None

    def _S7_downstream_target_lane(self, env, vehicle, source_lane):
        debug = 0
        if debug:
            road_network = getattr(getattr(getattr(env, "engine", None), "current_map", None), "road_network", None)
            navigation = getattr(vehicle, "navigation", None)
            checkpoints = list(getattr(navigation, "checkpoints", []) or [])
            if road_network is not None and len(checkpoints) >= 2:
                self._s7_route_lanes_debug_plot_counter += 1
                save_s7_route_lanes_debug_plot(
                    vehicle,
                    road_network,
                    checkpoints,
                    source_lane,
                    self._s7_route_lanes_debug_plot_counter,
                )

        lane_index = tuple(getattr(source_lane, "index", ()) or ())
        if lane_index == ('9g0_0_', '9g1_4_', 0):
            road_network = getattr(getattr(getattr(env, "engine", None), "current_map", None), "road_network", None)
            if road_network is not None and hasattr(road_network, "get_lane"):
                try:
                    return road_network.get_lane(('9g0_0_', '9g0_1_', 2))
                except Exception:
                    return None
        if len(lane_index) < 3:
            return None
        return None

    @classmethod
    def _nearest_background_observation(cls, vehicle, traffic_vehicles: list) -> dict:
        nearest = {"front": None, "back": None, "left": None, "right": None}
        best = {key: float("inf") for key in nearest}
        ego_pos = np.asarray(getattr(vehicle, "position", (0.0, 0.0))[:2], dtype=np.float32)
        ego_heading = float(getattr(vehicle, "heading_theta", 0.0))
        for traffic_vehicle in traffic_vehicles:
            if traffic_vehicle is vehicle:
                continue
            rel = cls._world_to_ego_local_from_pose(
                ego_pos,
                ego_heading,
                np.asarray(getattr(traffic_vehicle, "position", (0.0, 0.0))[:2], dtype=np.float32),
            )
            lon, lat = float(rel[0]), float(rel[1])
            if abs(lat) <= 2.0:
                key = "front" if lon >= 0.0 else "back"
                dist = abs(lon)
            elif lat > 0.0:
                key = "left"
                dist = float(np.linalg.norm(rel))
            else:
                key = "right"
                dist = float(np.linalg.norm(rel))
            if dist < best[key]:
                best[key] = dist
                nearest[key] = traffic_vehicle
        return nearest

    @staticmethod
    def _traffic_vehicles(env) -> list:
        traffic_manager = getattr(getattr(env, "engine", None), "traffic_manager", None)
        if traffic_manager is None:
            return []
        vehicles = getattr(traffic_manager, "traffic_vehicles", None)
        if vehicles is not None:
            try:
                return list(vehicles)
            except TypeError:
                pass
        return list(getattr(traffic_manager, "_traffic_vehicles", []) or [])

    @classmethod
    def _world_to_ego_local(cls, vehicle, target_world: np.ndarray) -> np.ndarray:
        ego_pos = np.asarray(getattr(vehicle, "position", (0.0, 0.0))[:2], dtype=np.float32)
        heading = float(getattr(vehicle, "heading_theta", 0.0))
        return cls._world_to_ego_local_from_pose(ego_pos, heading, np.asarray(target_world, dtype=np.float32))

    @staticmethod
    def _world_to_ego_local_from_pose(ego_pos: np.ndarray, heading: float, target_world: np.ndarray) -> np.ndarray:
        delta = np.asarray(target_world[:2], dtype=np.float32) - np.asarray(ego_pos[:2], dtype=np.float32)
        cos_h, sin_h = np.cos(float(heading)), np.sin(float(heading))
        return np.asarray(
            [
                cos_h * float(delta[0]) + sin_h * float(delta[1]),
                -sin_h * float(delta[0]) + cos_h * float(delta[1]),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _from_coarse_fallback(planner_batch: dict, agent_id: str) -> np.ndarray | None:
        coarse = (planner_batch.get(agent_id) or {}).get("coarse_trajectories")
        if coarse is None:
            return None
        coarse_arr = np.asarray(coarse, dtype=np.float32)
        if coarse_arr.ndim == 3 and coarse_arr.shape[0] > 0 and coarse_arr.shape[1] > 0:
            return coarse_arr[0, -1, :2].copy()
        return None

def make_rule_maker(config: dict) -> RuleMaker:
    """Factory: instantiate a RuleMaker from a config dict.

    Loads base params from configs/decision_model/rule_maker.yaml (default section),
    then applies scenario-specific overrides if 'scenario_id' is present in config.
    Any matching 'rule_maker_*' keys in config also override yaml values.

    Config keys:
        rule_maker_type (str): identifier for the implementation (default: 'multi_agent').
        scenario_id (str): if set, merges the matching scenario_overrides section from yaml.
        rule_maker_horizon_s (float): overrides yaml horizon_s.
        rule_maker_num_waypoints (int): overrides yaml num_waypoints.
        rule_maker_lane_change_preference (float): overrides yaml lane_change_preference.
        rule_maker_traffic_safety_distance_m (float): overrides yaml traffic_safety_distance_m.
        rule_maker_agent_safety_distance_m (float): overrides yaml agent_safety_distance_m.
        target_speed_km_h (float): overrides yaml target_speed_km_h.
    """
    rule_maker_type = str(config.get("rule_maker_type", "multi_agent"))
    scenario_id = config.get("scenario_id") or None
    yaml_path = config.get("rule_maker_yaml_path") or None

    yaml_params = load_rule_maker_config(scenario_id=scenario_id, yaml_path=yaml_path)

    # Apply explicit config overrides (rule_maker_* prefix or bare keys).
    _overrides = {
        "target_speed_km_h": config.get("target_speed_km_h"),
        "horizon_s": config.get("rule_maker_horizon_s"),
        "num_waypoints": config.get("rule_maker_num_waypoints"),
        "lane_change_preference": config.get("rule_maker_lane_change_preference"),
        "traffic_safety_distance_m": config.get("rule_maker_traffic_safety_distance_m"),
        "agent_safety_distance_m": config.get("rule_maker_agent_safety_distance_m"),
        "locked_on_reset": config.get("rule_maker_locked_on_reset"),
        "risk_ttc_trigger_s": config.get("rule_maker_risk_ttc_trigger_s"),
        "relock_ttc_threshold_s": config.get("rule_maker_relock_ttc_threshold_s"),
        "ideal_following_distance_m": config.get("rule_maker_ideal_following_distance_m"),
        "relock_gap_ratio": config.get("rule_maker_relock_gap_ratio"),
        "forced_lane_unlock_wait_steps": config.get("rule_maker_forced_lane_unlock_wait_steps"),
    }
    for k, v in _overrides.items():
        if v is not None:
            yaml_params[k] = v

    if rule_maker_type == "multi_agent":
        return MultiAgentRuleMaker(
            target_speed_km_h=float(yaml_params.get("target_speed_km_h", 30.0)),
            horizon_s=float(yaml_params.get("horizon_s", 4.0)),
            num_waypoints=int(yaml_params.get("num_waypoints", 8)),
            lane_change_preference=float(yaml_params.get("lane_change_preference", 0.0)),
            traffic_safety_distance_m=float(yaml_params.get("traffic_safety_distance_m", 8.0)),
            agent_safety_distance_m=float(yaml_params.get("agent_safety_distance_m", 7.0)),
            idm_time_headway_s=float(yaml_params.get("idm_time_headway_s", 1.2)),
            idm_min_gap_m=float(yaml_params.get("idm_min_gap_m", 6.0)),
            idm_max_accel_mps2=float(yaml_params.get("idm_max_accel_mps2", 1.8)),
            idm_comfortable_brake_mps2=float(yaml_params.get("idm_comfortable_brake_mps2", 2.5)),
            idm_delta=float(yaml_params.get("idm_delta", 4.0)),
            mobil_lane_change_threshold=float(yaml_params.get("mobil_lane_change_threshold", 0.2)),
            mobil_keep_bias=float(yaml_params.get("mobil_keep_bias", 0.15)),
            mobil_rear_base_penalty=float(yaml_params.get("mobil_rear_base_penalty", 3.0)),
            mobil_rear_slope_penalty=float(yaml_params.get("mobil_rear_slope_penalty", 4.0)),
            w_progress=float(yaml_params.get("w_progress", 0.04)),
            w_mobil=float(yaml_params.get("w_mobil", 1.1)),
            w_keep_bias=float(yaml_params.get("w_keep_bias", 0.30)),
            lc_cost=float(yaml_params.get("lc_cost", 0.75)),
            w_traffic_clearance=float(yaml_params.get("w_traffic_clearance", 0.03)),
            traffic_clearance_cap=float(yaml_params.get("traffic_clearance_cap", 1.5)),
            w_formation_consistent=float(yaml_params.get("w_formation_consistent", 0.50)),
            w_formation_inconsistent_cost=float(yaml_params.get("w_formation_inconsistent_cost", 0.75)),
            close_pair_threshold_m=float(yaml_params.get("close_pair_threshold_m", 12.0)),
            w_close_keep=float(yaml_params.get("w_close_keep", 1.5)),
            w_close_lc_same_cost=float(yaml_params.get("w_close_lc_same_cost", 0.5)),
            w_close_lc_diff_cost=float(yaml_params.get("w_close_lc_diff_cost", 1.0)),
            locked_on_reset=bool(yaml_params.get("locked_on_reset", True)),
            risk_ttc_trigger_s=float(yaml_params.get("risk_ttc_trigger_s", 3.0)),
            relock_ttc_threshold_s=float(yaml_params.get("relock_ttc_threshold_s", 5.0)),
            ideal_following_distance_m=float(yaml_params.get("ideal_following_distance_m", 10.0)),
            relock_gap_ratio=float(yaml_params.get("relock_gap_ratio", 1.5)),
            forced_lane_unlock_wait_steps=int(yaml_params.get("forced_lane_unlock_wait_steps", 10)),
        )
    raise ValueError(
        f"Unknown rule_maker_type: {rule_maker_type!r}. "
        "Register a new subclass of RuleMaker and add it here."
    )


def compute_target_points(
    rule_maker: RuleMaker,
    env,
    agent_ids: list[str],
    planner_batch: dict[str, dict],
) -> None:
    """Call rule_maker.compute() and inject results into planner_batch in-place.

    Must be called before planner.extract_rl_context(planner_batch) so that
    target_point flows through into model features automatically.
    """
    decisions = rule_maker.compute(env, agent_ids, planner_batch)
    for agent_id, decision in decisions.items():
        if agent_id in planner_batch:
            tp = np.asarray(decision["target_point"], dtype=np.float32).reshape(2)
            planner_batch[agent_id]["target_point"] = tp


def select_controller_by_formation(
    rule_maker: RuleMaker,
    pid_ctrl: "object",
    lqr_ctrl: "object",
) -> "object":
    """Return LQRFollowerController when formation is locked, PIDTrajectoryController otherwise.

    Args:
        rule_maker: Active RuleMaker instance (formation state is read via is_formation_locked).
        pid_ctrl: PIDTrajectoryController instance (used when formation is unlocked).
        lqr_ctrl: LQRFollowerController instance (used when formation is locked).

    Returns:
        The appropriate BaseController to call compute_actions() on this step.
    """
    return lqr_ctrl if rule_maker.is_formation_locked else pid_ctrl
