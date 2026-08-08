"""Frozen actor identity constants shared without importing dataset tensors."""

from __future__ import annotations

import re


PLATOON_AGENT_TO_ACTOR_ID = {
    "agent0": "P0",
    "agent1": "P1",
    "agent2": "P2",
}
PLATOON_ACTOR_IDS = tuple(PLATOON_AGENT_TO_ACTOR_ID.values())
EXTERNAL_ACTOR_ID_PATTERN = re.compile(r"^V[0-9]{3,}$")


__all__ = [
    "EXTERNAL_ACTOR_ID_PATTERN",
    "PLATOON_ACTOR_IDS",
    "PLATOON_AGENT_TO_ACTOR_ID",
]
