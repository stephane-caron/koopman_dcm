# SPDX-License-Identifier: Apache-2.0

"""Per-robot specs registry."""

from .base import RobotSpec
from .g1 import G1_SPEC

_REGISTRY: dict[str, RobotSpec] = {
    "g1": G1_SPEC,
}


def get_spec(name: str) -> RobotSpec:
    """Look up a robot spec by short name.

    Args:
        name: Short name (e.g. ``"g1"``).

    Returns:
        The corresponding ``RobotSpec``.

    Raises:
        KeyError: If ``name`` is not in the registry.
    """
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown robot '{name}'. Known: {known}")
    return _REGISTRY[name]
