import numpy as np

from models.diffusion.test_transfuser_policy import _compute_locked_follow_backend


class _FakeRuleMaker:
    def __init__(self, locked: bool) -> None:
        self._locked = bool(locked)
        self.calls = []

    @property
    def is_formation_locked(self) -> bool:
        return self._locked

    def compute(self, env, agent_ids, planner_batch):
        self.calls.append((env, list(agent_ids), planner_batch))
        return {
            agent_id: {
                "action": 0,
                "target_point": np.asarray([12.0, 0.0], dtype=np.float32),
            }
            for agent_id in agent_ids
        }

    def get_last_debug(self):
        return {"formation_locked": self._locked}


class _FakePlanner:
    def __init__(self) -> None:
        self.calls = []

    def plan(self, env, decisions):
        self.calls.append((env, decisions))
        return {
            agent_id: np.zeros((8, 3), dtype=np.float32)
            for agent_id in decisions
        }


class _FakeController:
    def __init__(self) -> None:
        self.calls = []

    def compute_actions(self, env, trajectories_world):
        self.calls.append((env, trajectories_world))
        return {
            agent_id: np.asarray([0.1, 0.2], dtype=np.float32)
            for agent_id in trajectories_world
        }

    def get_last_debug(self):
        return {"agent0": {"mode": "free_speed_tracking"}}


def test_locked_follow_backend_returns_low_level_actions_when_rule_maker_locked():
    rule_maker = _FakeRuleMaker(locked=True)
    planner = _FakePlanner()
    controller = _FakeController()
    planner_batch = {"agent0": {}, "agent1": {}}

    result = _compute_locked_follow_backend(
        rule_maker=rule_maker,
        normal_planner=planner,
        follow_controller=controller,
        env=object(),
        active_agent_ids=["agent0", "agent1"],
        planner_batch=planner_batch,
    )

    assert result["use_locked_follow"] is True
    assert sorted(result["low_level_actions"]) == ["agent0", "agent1"]
    assert np.allclose(result["low_level_actions"]["agent0"], [0.1, 0.2])
    assert result["control_backend"] == "locked_lqr_follow"
    assert len(planner.calls) == 1
    assert len(controller.calls) == 1


def test_locked_follow_backend_only_injects_targets_when_rule_maker_unlocked():
    rule_maker = _FakeRuleMaker(locked=False)
    planner = _FakePlanner()
    controller = _FakeController()
    planner_batch = {"agent0": {}, "agent1": {}}

    result = _compute_locked_follow_backend(
        rule_maker=rule_maker,
        normal_planner=planner,
        follow_controller=controller,
        env=object(),
        active_agent_ids=["agent0", "agent1"],
        planner_batch=planner_batch,
    )

    assert result["use_locked_follow"] is False
    assert np.allclose(planner_batch["agent0"]["target_point"], [12.0, 0.0])
    assert planner.calls == []
    assert controller.calls == []
