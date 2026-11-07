# SPDX-License-Identifier: Apache-2.0

"""Control wheels to track ground velocity targets."""

from typing import Optional

import numpy as np
import upkie.model
from upkie.controllers.mpc_balancer import MPCBalancer

from . import settings
from .interfaces.interface import Interface


class WheelController:
    """Base class for wheel balancers.

    Attributes:
        mpc_balancer: Internal controller for sagittal balance.
    """

    mpc_balancer: MPCBalancer

    def __init__(
        self,
        model: upkie.model.Model,
        fall_pitch: float = settings.FALL_ANGLE,
        mpc_leg_length: float = settings.MPC_LEG_LENGTH,
        max_ground_accel: float = settings.MAX_GROUND_ACCEL,
        max_ground_velocity: float = settings.MAX_GROUND_VELOCITY,
        max_wheel_torque: float = settings.MAX_WHEEL_TORQUE,
    ):
        """Initialize balancer.

        Args:
            model: Robot model to read wheel radius from.
            fall_pitch: Fall pitch threshold, in radians.
            mpc_leg_length: Leg length parameter of the MPC, in meters.
            max_ground_accel: Maximum commanded ground acceleration.
            max_ground_velocity: Maximum commanded ground velocity.
            max_wheel_torque: Maximum torque applied by wheel motors, in N.m.
        """
        mpc_balancer = MPCBalancer(
            leg_length=mpc_leg_length,
            fall_pitch=fall_pitch,
            max_ground_accel=max_ground_accel,
            max_ground_velocity=max_ground_velocity,
        )
        self._interface: Optional[Interface] = None
        self._max_wheel_torque = max_wheel_torque
        self.model = model
        self.mpc_balancer = mpc_balancer

    def reset(self) -> None:
        """Reset controller state to initial values."""
        self.mpc_balancer.reset()

    def register_interface(self, interface: Interface) -> None:
        """Register sliders with the given interface.

        Args:
            interface: Active input interface.
        """
        self._interface = interface
        with interface.add_folder("Wheel controller"):
            interface.add_slider(
                name="forward_velocity",
                label="Forward velocity (m/s)",
                min=-1.0,
                max=1.0,
                step=0.05,
                initial_value=0.0,
            )
            interface.add_slider(
                name="hack_velocity",
                label="Hack velocity (m/s)",
                min=-0.70,
                max=-0.10,
                step=0.02,
                initial_value=-0.62,  # fit to PyBullet
            )

    def cycle(
        self, dt: float, observation: dict, apply_hack: bool = False
    ) -> dict:
        """Compute a new ground velocity.

        Args:
            dt: Duration in seconds until the next cycle.
            observation: Latest observation.
            apply_hack: Whether to add the velocity correction set by the
                operator (``hack_velocity`` slider) to the forward velocity.

        Returns:
            Servo action for left and right wheels.
        """
        forward_velocity = (
            self._interface.get_slider("forward_velocity")
            if self._interface is not None
            else 0.0
        )

        # Velocity correction set by the operator, to keep the robot in place
        if self._interface is not None and apply_hack:
            forward_velocity += self._interface.get_slider("hack_velocity")

        ground_velocity = self.mpc_balancer.step(
            forward_velocity, observation, dt
        )
        wheel_radius = self.model.wheel_radius
        left_wheeled = self.model.left_wheeled
        wheel_velocity = ground_velocity / wheel_radius
        left_sign: float = 1.0 if left_wheeled else -1.0
        right_sign = -left_sign
        left_wheel_velocity = left_sign * wheel_velocity
        right_wheel_velocity = right_sign * wheel_velocity
        return {
            "left_wheel": {
                "position": np.nan,
                "velocity": left_wheel_velocity,
                "maximum_torque": self._max_wheel_torque,
            },
            "right_wheel": {
                "position": np.nan,
                "velocity": right_wheel_velocity,
                "maximum_torque": self._max_wheel_torque,
            },
        }
