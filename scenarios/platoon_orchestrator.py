"""Platoon-aware scenario orchestrator.

PlatoonScenarioOrchestrator overrides the trigger and reset/step dispatch so
that the lead vehicle (agent0 by default) is always used for position-based
trigger evaluation, regardless of which agent_id is passed by the caller.

Scenario semantics (route, traffic recipes, trigger window geometry) are
inherited from the parent ScenarioOrchestrator without modification.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from scenarios.definitions import ScenarioDefinition
from scenarios.orchestrator import LeadVehicleTrigger, ScenarioOrchestrator


class PlatoonScenarioOrchestrator(ScenarioOrchestrator):
    """ScenarioOrchestrator for multi-vehicle platoon environments.

    Key differences from the single-vehicle base class:

    * Trigger: always evaluates the lead vehicle position, not the calling
      agent_id. This ensures the hazard fires at a consistent location
      regardless of which vehicle is designated as "ego" by the env.

    * Recipe execution: guarded by a platoon-split check — if any agent has
      left the episode (crashed/out-of-road) the recipe is skipped to avoid
      spawning obstacles into a partially terminated platoon.

    * Summary: extends the base summary dict with ``platoon_agent_ids`` and
      ``platoon_lead_agent_id`` for logging and diagnostics.
    """

    def __init__(
        self,
        definition: ScenarioDefinition,
        local_route: str,
        agent_ids: List[str],
        lead_agent_id: Optional[str] = None,
    ) -> None:
        effective_lead = lead_agent_id if lead_agent_id is not None else agent_ids[0]
        super().__init__(
            definition,
            local_route,
            trigger_evaluator=LeadVehicleTrigger(lead_agent_id=effective_lead),
        )
        self._agent_ids: List[str] = list(agent_ids)
        self._lead_agent_id: str = effective_lead

    # ------------------------------------------------------------------
    # Public interface overrides
    # ------------------------------------------------------------------

    def reset(self, env, agent_id: str) -> None:
        super().reset(env, self._lead_agent_id)

    def before_step(self, env, agent_id: str, step_count: int) -> None:
        super().before_step(env, self._lead_agent_id, step_count)

    def get_episode_summary(self) -> Dict[str, object]:
        summary = super().get_episode_summary()
        summary["platoon_agent_ids"] = list(self._agent_ids)
        summary["platoon_lead_agent_id"] = self._lead_agent_id
        return summary

    # ------------------------------------------------------------------
    # Recipe execution guard
    # ------------------------------------------------------------------

    def _execute_recipe(self, env, ego_vehicle, step_count: int) -> None:
        if self._platoon_is_split(env):
            self.summary.notes.append("recipe_skipped:platoon_split")
            return
        super()._execute_recipe(env, ego_vehicle, step_count)

    def _platoon_is_split(self, env) -> bool:
        """Returns True if any platoon agent is no longer active in the env."""
        agents = getattr(env, "agents", {}) or {}
        return any(aid not in agents for aid in self._agent_ids)
