# SPDX-License-Identifier: Apache-2.0

"""Base class for lateral DCM controllers.

A DCM controller consumes an observed DCM (from any of the DCM sources
selected by the ``dcm_model`` selector) and produces a commanded ZMP
``zmp_line`` together with the ``leaning_velocity`` that the leg
controller integrates into a foot height difference.

Subclasses differ in how they map the DCM to the commanded ZMP:

- :class:`~.dcm_feedback_controller.DcmFeedbackController` applies proportional
  DCM feedback, regulating ξ toward the pattern's reference.
- :class:`~.dcm_mpc.DcmMpc` solves a receding-horizon QP over the DCM
  recursion, optionally under the viability constraint ``|ξ| ≤ ξ_max``.

Their ``cycle`` signatures differ (the MPC needs the whole pattern
generator to forward-evaluate the reference over its horizon), so
``Balancer`` branches on the active controller rather than calling a
single unified method.
"""

from .interfaces.interface import Interface


class DcmController:
    """Common interface and shared plumbing of lateral DCM controllers.

    Attributes:
        leaning_velocity: Leaning velocity in m/s for the leg controller,
            corresponding to foot force difference control targetted at the
            commanded ZMP.
        zmp_line: Normalized line coordinate, in [-1, 1], of the commanded ZMP.
    """

    leaning_velocity: float
    zmp_line: float

    def __init__(self) -> None:
        """Initialize the controller with zero outputs."""
        self._interface: Interface | None = None
        self._zmp_commanded_handle = None
        self.leaning_velocity = 0.0
        self.zmp_line = 0.0

    def reset(self) -> None:
        """Reset controller state at the beginning of an episode."""
        self.leaning_velocity = 0.0
        self.zmp_line = 0.0

    def register_interface(self, interface: Interface) -> None:
        """Register sliders with the given interface.

        Subclasses are expected to call this, then add their own sliders.

        Args:
            interface: Active input interface.
        """
        self._interface = interface

    def _add_commanded_zmp_marker(self, interface: Interface) -> None:
        """Add the commanded-ZMP ball to the Viser scene, if there is one.

        Args:
            interface: Active input interface. No-op unless it carries a
                Viser server.
        """
        if not hasattr(interface, "viser_server"):
            return
        scene = interface.viser_server.scene
        self._zmp_commanded_handle = scene.add_icosphere(
            "/zmp/commanded",
            radius=0.015,
            color=(0.6, 0.0, 0.8),
            position=(0.0, 0.0, 0.0),
        )

    def update_visualizer(self, contacts: dict) -> None:
        """Update the commanded ZMP ball position in the Viser scene.

        Args:
            contacts: Contact dict from StateEstimator, keyed by side.
        """
        if self._zmp_commanded_handle is None:
            return
        alpha_left = (self.zmp_line + 1.0) / 2.0
        position = (
            alpha_left * contacts["left"].position_in_world
            + (1.0 - alpha_left) * contacts["right"].position_in_world
        )
        self._zmp_commanded_handle.position = position

    def to_log(self) -> dict:
        """Get instantaneous values for logging.

        Returns:
            Dictionary mapping signal names to their current values.
        """
        return {"leaning_velocity": self.leaning_velocity}
