# SPDX-License-Identifier: Apache-2.0

"""DCM feedback controller."""

from . import settings
from .interface import Interface


class DcmController:
    r"""Balance control by DCM feedback.

    This controller applies proportional DCM feedback at the ZMP. The resulting
    commanded ZMP is then converted to a leaning velocity.

    Attributes:
        leaning_velocity: Leaning velocity in m/s for the leg controller,
            corresponding to foot force difference control targetted at the
            commanded ZMP.
        zmp_line: Normalized line coordinate, in [-1, 1], of the commanded ZMP.
    """

    leaning_velocity: float
    zmp_line: float

    def __init__(self) -> None:
        self._interface: Interface | None = None

    def reset(self) -> None:
        pass

    def register_interface(self, interface: Interface) -> None:
        """Register sliders with the given interface.

        Args:
            interface: Active input interface.
        """
        self._interface = interface
        interface.add_slider(
            name="admittance",
            label="Admittance",
            min=0.1,
            max=5.0,
            step=0.1,
            initial_value=settings.ADMITTANCE_DEFAULT,
        )
        interface.add_slider(
            name="kp",
            label="kp",
            min=0.1,
            max=5.0,
            step=0.1,
            initial_value=1.0,
        )

    def cycle(
        self,
        dt: float,
        omega: float,
        dcm_target: float,
        dcm_observed: float,
        zmp_target: float,
        zmp_observed: float,
    ) -> None:
        """Compute desired ZMP and leaning velocity for this cycle.

        Args:
            dt: Time step in seconds.
            omega: Eigenvalue of the divergent component.
            dcm_target: Desired normalized line DCM, in [-1, 1].
            dcm_observed: Estimate of the normalized line DCM, in [-1, 1].
            zmp_target: Desired normalized line ZMP, in [-1, 1].
            zmp_observed: Estimate of the normalized line ZMP, in [-1, 1].

        Returns:
            Leaning velocity in m/s.
        """
        assert self._interface is not None, "call register_interface first"
        assert 1.0 < omega < 8.0, f"suspicious {omega=}"
        admittance = self._interface.get_slider("admittance")
        kp = self._interface.get_slider("kp")

        zmp_gain = 1 + kp / omega
        zmp_feedback = zmp_target - zmp_gain * (dcm_target - dcm_observed)
        leaning_velocity = admittance * (zmp_feedback - zmp_observed)

        self.zmp_line = zmp_feedback
        self.leaning_velocity = leaning_velocity
