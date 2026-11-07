# SPDX-License-Identifier: Apache-2.0

"""Walking pattern generator."""

import numpy as np

from .interfaces.interface import Interface

# Our omega is between 5.0 rad/s and 5.7 rad/s, thus the nominal walking period
# for Cookie (2 pi / omega) should be between 1.1 s and 1.3 s.
MAX_PERIOD = 2.0  # s
MIN_PERIOD = 0.2  # s
NOMINAL_PERIOD = 1.2  # s


class PatternGenerator:
    r"""Modulate how task targets evolve over time.

    The DCM and ZMP targets evolve jointly as sine waves tuned to the stepping
    period $T_s$, or its equivalent natural frequency $\nu = 2 pi / T_s$.
    The ZMP oscillates according to:

    $$
    z(t) = A \sin(\nu t) + \delta
    $$

    with $A$ the amplitude and $\delta$ the offset parameters. Solving the ODE
    $\dot{\xi} = \omega (\xi - z)$ yields the corresponding DCM solution:

    $$
    \xi(t) = \frac{\omega A}{\sqrt{\omega^2 + \nu^2}} \sin(\nu t + \psi) +
    \delta
    $$

    where $\psi = \arctan(\nu / \omega)$ is the DCM-ZMP phase shift. Among the
    solutions of $\dot{\xi} = \omega (\xi - z)$ consistent with the ZMP
    profile, we thus pick the one that does not diverge (any nonzero
    coefficient on the homogeneous mode $e^{\omega t}$ would blow up over
    time).

    Feeding this DCM into the stable CoM dynamics $\dot{c} = \omega (\xi - c)$
    and keeping the bounded steady-state solution (the transient $e^{-\omega
    t}$ decays naturally) yields:

    $$
    c(t) = \frac{\omega^2 A}{\omega^2 + \nu^2} \sin(\nu t) + \delta
    $$

    The CoM lags the DCM by exactly $\psi$, which cancels the DCM-ZMP lead:
    the CoM oscillates in phase with the ZMP, with amplitude reduced by a
    factor $\cos^2 \psi = \omega^2 / (\omega^2 + \nu^2)$.

    Attributes:
        amplitude: Current ZMP oscillation amplitude, kept in sync with the
            interface slider each cycle.
        com: Normalized lateral CoM coordinate.
        dcm: Normalized lateral DCM coordinate.
        offset: Current ZMP offset, kept in sync with the interface slider
            each cycle.
        period: Current stepping period in seconds; kept in sync with the
            interface slider each cycle.
        zmp: Normalized lateral ZMP coordinate.
    """

    _interface: Interface | None
    _t: float
    amplitude: float
    com: float
    dcm: float
    offset: float
    period: float
    zmp: float

    def __init__(self, initial_period: float = NOMINAL_PERIOD) -> None:
        """Initialize the generator.

        Args:
            initial_period: Initial stepping period in seconds.
        """
        self._interface = None
        self._initial_period = initial_period
        self._t = 0.0
        self.amplitude = 0.0
        self.com = 0.0
        self.dcm = 0.0
        self.offset = 0.0
        self.period = initial_period
        self.zmp = 0.0

    @property
    def t(self) -> float:
        """Time in seconds since the last reset, in the pattern's clock.

        Read by :class:`~.dcm_mpc.DcmMpc` to evaluate the reference
        forward over its preview horizon. Exposing the clock rather than
        letting the MPC mirror it keeps the two from drifting apart: this
        balancer's ``reset`` does not rewind the pattern, so a mirrored
        clock would silently track a phase-shifted reference.
        """
        return self._t

    def reset(self) -> None:
        """Reset time accumulator for ZMP oscillation."""
        self._t = 0.0

    def register_interface(self, interface: Interface) -> None:
        """Register sliders with the given interface.

        Args:
            interface: Active input interface.
        """
        self._interface = interface
        with interface.add_folder("Pattern generator"):
            interface.add_slider(
                name="amplitude",
                label="Amplitude",
                min=0.0,
                max=1.0,
                step=0.05,
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
        """Compute desired CoM, DCM and ZMP normalized coordinates.

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

        # CoM (in phase with ZMP, reduced amplitude)
        com_amplitude = amplitude * omega**2 / (omega**2 + nu**2)
        self.com = com_amplitude * np.sin(phase) + offset

        self._t += dt

    def to_log(self) -> dict:
        """Get instantaneous values for logging.

        Returns:
            Dictionary mapping signal names to their current values.
        """
        return {
            "amplitude": self.amplitude,
            "offset": self.offset,
            "period": self.period,
        }
