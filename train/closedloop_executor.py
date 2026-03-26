from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from typing import Dict, List, Optional

import numpy as np



def _has_active_agents(env) -> bool:
    agent_manager = getattr(env, "agent_manager", None)
    if agent_manager is None:
        return True
    active_objects = getattr(agent_manager, "_active_objects", None)
    return bool(active_objects)


def _is_all_agents_gone_assertion(exc: Exception, env) -> bool:
    if not isinstance(exc, AssertionError):
        return False
    if "Not enough objects exist!" not in str(exc):
        return False
    return not _has_active_agents(env)


def _build_terminal_fallback_info(env, agent_ids: List[str]) -> Dict[str, dict]:
    last_info = getattr(env, "_last_info", {}) or {}
    fallback = {}
    for agent_id in agent_ids:
        item = dict(last_info.get(agent_id, {}))
        item.setdefault("progress", 0.0)
        item.setdefault("formation_error", 10.0)
        item.setdefault("min_gap", 0.0)
        item.setdefault("jerk", 0.0)
        item.setdefault("delta_steering", 0.0)
        item["crash"] = True
        item["out_of_road"] = bool(item.get("out_of_road", False))
        fallback[agent_id] = item
    return fallback

def _execute_single_group_standalone(env, joint_actions: Dict[str, np.ndarray], horizon: int) -> dict:
    step_infos: Dict[str, List[dict]] = {agent_id: [] for agent_id in joint_actions}
    crash_flags = {agent_id: False for agent_id in joint_actions}
    out_of_road_flags = {agent_id: False for agent_id in joint_actions}
    terminated = False
    profile = {
        "restore_reset": 0.0,
        "restore_set_state": 0.0,
        "restore_total": 0.0,
        "step_execution": 0.0,
    }

    for step_idx in range(horizon):
        step_actions = {}
        for agent_id, trajectory in joint_actions.items():
            traj = np.asarray(trajectory, dtype=np.float32)
            if traj.ndim != 2 or traj.shape[1] != 3:
                raise ValueError(f"Expected [T, 3] trajectory for {agent_id}, got {traj.shape}")
            traj_len = traj.shape[0]
            start_idx = min(step_idx, max(traj_len - 1, 0))
            window = traj[start_idx:]
            if window.shape[0] < traj_len:
                pad = np.repeat(window[-1:, :], traj_len - window.shape[0], axis=0)
                window = np.concatenate([window, pad], axis=0)
            step_actions[agent_id] = window

        step_start = time.perf_counter()
        try:
            _, _, term, trunc, info = env.step(step_actions)
        except Exception as exc:
            profile["step_execution"] += time.perf_counter() - step_start
            if _is_all_agents_gone_assertion(exc, env):
                info = _build_terminal_fallback_info(env, list(joint_actions.keys()))
                term = {"__all__": True}
                trunc = {"__all__": False}
            else:
                raise
        else:
            profile["step_execution"] += time.perf_counter() - step_start
        for agent_id in joint_actions:
            agent_info = dict(info.get(agent_id, {}))
            step_infos[agent_id].append(agent_info)
            crash_flags[agent_id] = crash_flags[agent_id] or bool(agent_info.get("crash", False))
            out_of_road_flags[agent_id] = out_of_road_flags[agent_id] or bool(agent_info.get("out_of_road", False))

        if term.get("__all__", False) or trunc.get("__all__", False) or not _has_active_agents(env):
            terminated = True
            break

    return {
        "step_infos": step_infos,
        "crash_flags": crash_flags,
        "out_of_road_flags": out_of_road_flags,
        "terminated": terminated,
        "profile": profile,
    }


def _worker_loop(env_config: dict, task_queue, result_queue, horizon: int):
    from envs.platoon_env import PlatoonEnv

    env = PlatoonEnv(env_config)
    env.reset()
    result_queue.put(("__ready__", os.getpid(), None))
    try:
        while True:
            task = task_queue.get()
            if task is None:
                break
            batch_id, saved_state, batch_items = task
            try:
                batch_results = []
                needs_reset = False
                for task_id, joint_actions in batch_items:
                    restore_start = time.perf_counter()
                    if needs_reset:
                        env.reset()
                        needs_reset = False
                    env.set_state(saved_state)
                    restore_end = time.perf_counter()
                    result = _execute_single_group_standalone(env, joint_actions, horizon)
                    result["profile"]["restore_set_state"] = restore_end - restore_start
                    result["profile"]["restore_total"] = restore_end - restore_start
                    batch_results.append((task_id, result))
                    # If all agents terminated, we need a reset before next set_state
                    if result.get("terminated", False):
                        needs_reset = True
                result_queue.put((batch_id, batch_results, None))
            except Exception:
                result_queue.put((batch_id, None, traceback.format_exc()))
    finally:
        env.close()


class ClosedLoopExecutor:
    def __init__(self, env, reward_config: dict, horizon: int = 8):
        self.env = env
        self.reward_config = dict(reward_config or {})
        self.horizon = int(horizon)
        self.last_profile = self._empty_group_profile()

    def _stamp(self) -> float:
        return time.perf_counter()

    def _empty_restore_profile(self) -> dict[str, float]:
        return {
            "restore_reset": 0.0,
            "restore_set_state": 0.0,
            "restore_total": 0.0,
        }

    def _empty_group_profile(self) -> dict[str, float]:
        return {
            "group_count": 0.0,
            "pre_restore_reset": 0.0,
            "pre_restore_set_state": 0.0,
            "pre_restore_total": 0.0,
            "trajectory_restore_reset": 0.0,
            "trajectory_restore_set_state": 0.0,
            "trajectory_restore_total": 0.0,
            "trajectory_step_execution": 0.0,
            "final_restore_reset": 0.0,
            "final_restore_set_state": 0.0,
            "final_restore_total": 0.0,
            "all_restore_reset": 0.0,
            "all_restore_set_state": 0.0,
            "all_restore_total": 0.0,
            "all_step_execution": 0.0,
            "group_total": 0.0,
            "dispatch_total": 0.0,
            "result_wait_total": 0.0,
        }

    def _restore_state(self, saved_state: dict) -> dict[str, float]:
        profile = self._empty_restore_profile()
        restore_start = self._stamp()
        self.env.set_state(saved_state)
        restore_end = self._stamp()
        profile["restore_set_state"] = restore_end - restore_start
        profile["restore_total"] = restore_end - restore_start
        return profile

    def _accumulate_restore(self, target: dict[str, float], prefix: str, restore_profile: dict[str, float]) -> None:
        target[f"{prefix}_reset"] += float(restore_profile.get("restore_reset", 0.0))
        target[f"{prefix}_set_state"] += float(restore_profile.get("restore_set_state", 0.0))
        target[f"{prefix}_total"] += float(restore_profile.get("restore_total", 0.0))
        target["all_restore_reset"] += float(restore_profile.get("restore_reset", 0.0))
        target["all_restore_set_state"] += float(restore_profile.get("restore_set_state", 0.0))
        target["all_restore_total"] += float(restore_profile.get("restore_total", 0.0))

    def _execute_single_group(self, joint_actions: Dict[str, np.ndarray]) -> dict:
        return _execute_single_group_standalone(self.env, joint_actions, self.horizon)

    def execute_joint_trajectory(self, joint_actions: Dict[str, np.ndarray]) -> dict:
        saved_state = self.env.get_state()
        try:
            result = self._execute_single_group(joint_actions)
        finally:
            restore_profile = self._restore_state(saved_state)

        result["profile"].update(restore_profile)
        return result

    def _ensure_agents_alive(self):
        """If all agents were terminated or detached, do a lightweight reset to re-create them."""
        if not _has_active_agents(self.env):
            self.env.reset()
            return
        if hasattr(self.env, "dones") and isinstance(self.env.dones, dict):
            done_values = [bool(value) for key, value in self.env.dones.items() if key != "__all__"]
            if done_values and all(done_values):
                self.env.reset()

    def execute_joint_groups(self, joint_groups: List[Dict[str, np.ndarray]]) -> List[dict]:
        saved_state = self.env.get_state()
        results = []
        aggregate = self._empty_group_profile()
        aggregate["group_count"] = float(len(joint_groups))
        try:
            for joint_group in joint_groups:
                self._ensure_agents_alive()
                pre_restore = self._restore_state(saved_state)
                self._accumulate_restore(aggregate, "pre_restore", pre_restore)

                group_start = self._stamp()
                result = self._execute_single_group(joint_group)
                aggregate["group_total"] += self._stamp() - group_start

                result_profile = result.get("profile", {})
                self._accumulate_restore(aggregate, "trajectory_restore", result_profile)
                aggregate["trajectory_step_execution"] += float(result_profile.get("step_execution", 0.0))
                aggregate["all_step_execution"] += float(result_profile.get("step_execution", 0.0))
                results.append(result)
        finally:
            self._ensure_agents_alive()
            final_restore = self._restore_state(saved_state)
            self._accumulate_restore(aggregate, "final_restore", final_restore)
            self.last_profile = aggregate
        return results


class ParallelClosedLoopExecutor(ClosedLoopExecutor):
    def __init__(self, env, env_config: dict, reward_config: dict, num_workers: int = 4, horizon: int = 8):
        super().__init__(env=env, reward_config=reward_config, horizon=horizon)
        self.env_config = dict(env_config or {})
        self.env_config["use_render"] = False
        if str(self.env_config.get("observation_mode", "lidar_state")) == "multimodal":
            self.env_config["observation_mode"] = "lidar_state"
        self.num_workers = int(num_workers)
        self._ctx = None
        self._task_queue = None
        self._result_queue = None
        self._workers: List[mp.Process] = []
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        py_utf8 = os.environ.get("PYTHONUTF8")
        if py_utf8 not in {None, "0", "1"}:
            os.environ["PYTHONUTF8"] = "1"
        elif py_utf8 == "":
            os.environ["PYTHONUTF8"] = "1"
        self._ctx = mp.get_context("spawn")
        self._task_queue = self._ctx.Queue()
        self._result_queue = self._ctx.Queue()
        for _ in range(self.num_workers):
            worker = self._ctx.Process(
                target=_worker_loop,
                args=(dict(self.env_config), self._task_queue, self._result_queue, self.horizon),
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)
        for _ in range(self.num_workers):
            ready_tag, _, error = self._result_queue.get(timeout=60)
            if ready_tag != "__ready__" or error is not None:
                raise RuntimeError(f"Parallel worker failed during startup: {error}")
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        assert self._task_queue is not None
        for _ in self._workers:
            self._task_queue.put(None)
        for worker in self._workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
        self._workers.clear()
        if self._task_queue is not None:
            self._task_queue.close()
            self._task_queue = None
        if self._result_queue is not None:
            self._result_queue.close()
            self._result_queue = None
        self._ctx = None
        self._started = False

    def execute_joint_groups(self, joint_groups: List[Dict[str, np.ndarray]]) -> List[dict]:
        if not self._started:
            self.start()
        assert self._task_queue is not None and self._result_queue is not None

        saved_state = self.env.get_state()
        aggregate = self._empty_group_profile()
        aggregate["group_count"] = float(len(joint_groups))
        results_map: dict[int, dict] = {}
        try:
            dispatch_start = self._stamp()
            active_workers = max(1, min(len(self._workers), len(joint_groups)))
            chunks: list[list[tuple[int, Dict[str, np.ndarray]]]] = [[] for _ in range(active_workers)]
            for task_id, joint_group in enumerate(joint_groups):
                chunks[task_id % active_workers].append((task_id, joint_group))
            non_empty_chunks = [chunk for chunk in chunks if chunk]
            for batch_id, chunk in enumerate(non_empty_chunks):
                self._task_queue.put((batch_id, saved_state, chunk))
            aggregate["dispatch_total"] = self._stamp() - dispatch_start

            wait_start = self._stamp()
            for _ in range(len(non_empty_chunks)):
                batch_id, batch_results, error = self._result_queue.get(timeout=60)
                if error is not None:
                    raise RuntimeError(f"Worker error on batch {batch_id}: {error}")
                for task_id, result in batch_results:
                    results_map[int(task_id)] = result
            aggregate["result_wait_total"] = self._stamp() - wait_start

            results = [results_map[idx] for idx in range(len(joint_groups))]
            for result in results:
                result_profile = result.get("profile", {})
                self._accumulate_restore(aggregate, "trajectory_restore", result_profile)
                aggregate["trajectory_step_execution"] += float(result_profile.get("step_execution", 0.0))
                aggregate["all_step_execution"] += float(result_profile.get("step_execution", 0.0))
                aggregate["group_total"] += float(result_profile.get("restore_total", 0.0)) + float(result_profile.get("step_execution", 0.0))
            return results
        finally:
            final_restore = self._restore_state(saved_state)
            self._accumulate_restore(aggregate, "final_restore", final_restore)
            self.last_profile = aggregate

    def __del__(self):  # pragma: no cover
        try:
            self.stop()
        except Exception:
            pass
