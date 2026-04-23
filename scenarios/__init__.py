"""Project scenario infrastructure — definitions, orchestration, and hazard registry.

Pure-data symbols (TriggerSpec, ScenarioDefinition, SCENARIO_BY_ID, etc.) are
importable without MetaDrive installed.  Runtime classes (ScenarioOrchestrator,
PlatoonScenarioOrchestrator) require MetaDrive and are imported directly from
their sub-modules to avoid polluting this module's import-time cost.

  # Pure data — always safe:
  from scenarios import SCENARIO_BY_ID, get_scenario_definition

  # Runtime — requires MetaDrive:
  from scenarios.orchestrator import ScenarioOrchestrator
  from scenarios.platoon_orchestrator import PlatoonScenarioOrchestrator
"""

from scenarios.definitions import (  # noqa: F401
    TriggerSpec,
    RecipeSpec,
    ScenarioDefinition,
    SCENARIO_DEFINITIONS,
    SCENARIO_BY_ID,
    DEFAULT_SCENARIO_WEIGHTS,
    SCENARIO_EXPERT_OVERRIDES,
    SCENARIO_TO_ROUTES,
    ROUTE_TO_SCENARIOS,
    get_scenario_definition,
)
