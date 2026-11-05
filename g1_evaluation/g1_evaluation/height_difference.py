# SPDX-License-Identifier: Apache-2.0

"""Right-minus-left foot height integrator.

The DCM controller emits a scalar ``leaning_velocity``; this module
integrates it into a height difference (m), clipped to a maximum.
"""

import numpy as np


class HeightDifference:
    """Leaky integrator on (right − left) foot target height, in m."""

    value: float

    def __init__(self, max_value: float = 0.10) -> None:
        """Initialize.

        Args:
            max_value: Symmetric bound on |height_difference|, in m.
        """
        self.max_value = max_value
        self.value = 0.0

    def reset(self) -> None:
        self.value = 0.0

    def integrate(self, leaning_velocity: float, dt: float) -> None:
        """Add ``leaning_velocity * dt`` to the height difference, clipped.

        Args:
            leaning_velocity: Velocity to integrate, in m/s. Positive
                grows the height difference toward "right foot raised".
            dt: Time step, in s.
        """
        self.value = float(
            np.clip(
                self.value + leaning_velocity * dt,
                -self.max_value,
                +self.max_value,
            )
        )
