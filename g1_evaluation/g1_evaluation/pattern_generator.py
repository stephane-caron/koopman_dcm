# SPDX-License-Identifier: Apache-2.0

"""Sinusoidal ZMP and DCM references."""

import numpy as np

from .interface import Interface

# Range and default of the period of the references
MAX_PERIOD = 2.0  # s
MIN_PERIOD = 0.2  # s
NOMINAL_PERIOD = 1.2  # s


class PatternGenerator:
    r"""Sinusoidal ZMP reference and the corresponding DCM reference.

    The ZMP reference is $z(t) = \delta + A \sin(\nu t)$ with $\nu = 2 \pi / T$,
    as in Section 5 of the paper. The DCM reference is the bounded solution
    of $\dot{\xi} = \omega (\xi - z)$:

    $$
    \xi(t) = \delta + \frac{\omega A}{\sqrt{\omega^2 + \nu^2}}
    \sin(\nu t + \arctan(\nu / \omega))
    $$

    Attributes:
        amplitude: ZMP amplitude $A$, read from the interface each cycle.
        dcm: DCM reference, as a normalized line coordinate.
        offset: ZMP offset $\delta$, read from the interface each cycle.
        period: Period $T$ in seconds, read from the interface each cycle.
        zmp: ZMP reference, as a normalized line coordinate.
    """

    _interface: Interface | None
    _t: float
    amplitude: float
    dcm: float
    offset: float
    period: float
    zmp: float

    def __init__(self, initial_period: float = NOMINAL_PERIOD) -> None:
        """Initialize the generator.

        Args:
            initial_period: Initial pattern period in seconds.
        """
        self._interface = None
        self._initial_period = initial_period
        self._t = 0.0
        self.amplitude = 0.0
        self.dcm = 0.0
        self.offset = 0.0
        self.period = initial_period
        self.zmp = 0.0

    def reset(self) -> None:
        """Reset time accumulator for ZMP oscillation."""
        self._t = 0.0

    def register_interface(self, interface: Interface) -> None:
        """Register sliders with the given interface.

        Args:
            interface: Active input interface.
        """
        self._interface = interface
        interface.add_slider(
            name="amplitude",
            label="Amplitude",
            min=0.0,
            max=0.3,
            step=0.1,
            initial_value=0.0,
        )
        interface.add_slider(
            name="offset",
            label="Offset",
            min=-1.0,
            max=1.0,
            step=0.1,
            initial_value=0.0,
        )
        interface.add_slider(
            name="period",
            label="Period (s)",
            min=MIN_PERIOD,
            max=MAX_PERIOD,
            step=0.1,
            initial_value=self._initial_period,
        )

    def cycle(self, dt: float, omega: float) -> None:
        """Compute the ZMP and DCM references at the current time.

        Args:
            dt: Time step in seconds.
            omega: Natural frequency of the linear inverted pendulum model.
        """
        assert self._interface is not None, "call register_interface first"
        amplitude = self._interface.get_slider("amplitude")
        offset = self._interface.get_slider("offset")
        period = self._interface.get_slider("period")
        self.amplitude = amplitude
        self.offset = offset
        self.period = period

        # ZMP
        nu = 2.0 * np.pi / period  # rad/s
        phase = nu * self._t
        self.zmp = amplitude * np.sin(phase) + offset

        # DCM
        shift = np.arctan2(nu, omega)
        dcm_amplitude = amplitude * omega / np.sqrt(omega**2 + nu**2)
        self.dcm = dcm_amplitude * np.sin(phase + shift) + offset

        self._t += dt
