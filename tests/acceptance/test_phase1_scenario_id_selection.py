from __future__ import annotations

from envs.platoon_env import PlatoonEnv
from scenarios.definitions import SCENARIO_BY_ID
import pytest


def test_platoon_env_no_longer_exposes_legacy_scenario_alias_resolver():
    legacy_resolver_name = "_resolve_" + "hazard_" + "scenario"
    assert not hasattr(PlatoonEnv, legacy_resolver_name)


def test_platoon_env_rejects_legacy_scenario_alias_key():
    legacy_key = "hazard_" + "scenario"
    with pytest.raises(ValueError, match="scenario_id"):
        PlatoonEnv({"use_render": False, legacy_key: "dynamic_cut_in"})


def test_scenario_registry_is_the_hazard_selection_source():
    assert "S6_background_merge_in" in SCENARIO_BY_ID
    assert "S9_narrow_channel_negotiation" in SCENARIO_BY_ID
