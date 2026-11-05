# SPDX-License-Identifier: Apache-2.0

"""Observation schema.

The first three slices (``q_actuated``, ``com_anchor``, ``com_line``) are
the observations of the delay embedding of the DCM observable, so that
they end at ``sv.slice_of("com_line").stop``.

Sign convention: +y = left.

``com_line`` and ``zmp_line`` are normalized to [-1, 1] where +1 is the
left foot frame's lateral position and -1 is the right foot frame's.
"""

from .robots.base import RobotSpec
from .structured_vector import StructuredVector


def make_observation(spec: RobotSpec) -> StructuredVector:
    """Return a zero-filled structured observation vector for ``spec``.

    Layout, in order:

        q_actuated (nq)  Joint angles, in rad.
        com_anchor  (3)  CoM position in the anchor frame, in m.
        com_line    (1)  Normalized lateral CoM in [-1, 1].
        tau        (nq)  Joint torques, in N·m.
        zmp_anchor  (3)  ZMP position in the anchor frame, in m.
        zmp_line    (1)  Normalized lateral ZMP in [-1, 1].

    Args:
        spec: Robot spec providing ``nb_actuated``.

    Returns:
        A zero-filled ``StructuredVector`` with the layout above.
    """
    nq = spec.nb_actuated
    return (
        StructuredVector()
        .add_slice("q_actuated", nq)
        .add_slice("com_anchor", 3)
        .add_slice("com_line", 1)
        .add_slice("tau", nq)
        .add_slice("zmp_anchor", 3)
        .add_slice("zmp_line", 1)
    )
