"""Process-wide compatibility shims for local training environments.

This module is auto-imported by Python when it is discoverable on ``sys.path``.
We use it to provide a small NumPy compatibility alias required by
``ray[rllib]==2.4.0`` on newer NumPy versions.
"""

from __future__ import annotations

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
if not hasattr(np, "product"):
    np.product = np.prod

for alias, target in {
    "object": object,
    "int": int,
    "float": float,
    "complex": complex,
    "str": str,
}.items():
    if not hasattr(np, alias):
        setattr(np, alias, target)

try:
    from gym.utils import seeding as gym_seeding

    _orig_generator_ctor = getattr(gym_seeding, "_generator_ctor", None)

    if callable(_orig_generator_ctor):
        def _compat_generator_ctor(bit_generator="MT19937"):  # pragma: no cover - import-time shim
            if isinstance(bit_generator, np.random.BitGenerator):
                return np.random.Generator(bit_generator)
            return _orig_generator_ctor(bit_generator)

        gym_seeding._generator_ctor = _compat_generator_ctor
except Exception:
    pass
