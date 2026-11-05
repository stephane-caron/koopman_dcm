# SPDX-License-Identifier: Apache-2.0

"""Sagittal stabilization.

This evaluation only controls the lateral (y) axis via foot height difference.
The sagittal (x) axis and yaw (z) rotation are not investigated. We keep them
from drifting via a spring-damper force along the world-x axis, and a
spring-damper torque about the world-z axis. This sagital stabilization has the
same function as the sagittal MPCBalancer in the Cookie controller.
"""

import numpy as np

from . import settings


def _quat_xyzw_to_yaw(quat_xyzw: np.ndarray) -> float:
    """Extract the yaw angle (rotation about world z) from a quaternion.

    Args:
        quat_xyzw: Quaternion in (x, y, z, w) order.

    Returns:
        Yaw angle in rad.
    """
    x, y, z, w = (float(c) for c in quat_xyzw)
    # Yaw from quaternion: atan2(2(wz + xy), 1 − 2(y² + z²)).
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


class SagittalStabilizer:
    """External force/torque PD on (x, ẋ) and (yaw, ω_z) at the base.

    Attributes:
        enabled: When False, ``compute`` returns zero force and torque.
        x0: Sagittal world-x setpoint, recorded at ``anchor``.
        yaw0: Yaw setpoint, recorded at ``anchor``.
    """

    enabled: bool
    x0: float
    yaw0: float

    def __init__(self) -> None:
        self.enabled = True
        self.x0 = 0.0
        self.yaw0 = 0.0

    def anchor(
        self,
        base_position: np.ndarray,
        base_quat_xyzw: np.ndarray,
    ) -> None:
        """Record the reset-time sagittal and yaw setpoints.

        Args:
            base_position: Base position in the world frame, in m.
            base_quat_xyzw: Base orientation as an (x, y, z, w) quaternion.
        """
        self.x0 = float(base_position[0])
        self.yaw0 = _quat_xyzw_to_yaw(base_quat_xyzw)

    def compute(
        self,
        base_position: np.ndarray,
        base_quat_xyzw: np.ndarray,
        base_linvel_world: np.ndarray,
        base_angvel_world: np.ndarray,
    ) -> tuple[float, float]:
        """Return ``(F_x_world, M_z_world)`` for one cycle.

        Args:
            base_position: World-frame base position.
            base_quat_xyzw: Base orientation as (x, y, z, w).
            base_linvel_world: World-frame linear velocity.
            base_angvel_world: World-frame angular velocity.

        Returns:
            ``(F_x, M_z)`` to apply at the base link via
            ``PyBulletBridge.apply_external_disturbance``.
        """
        if not self.enabled:
            return 0.0, 0.0
        x = float(base_position[0])
        vx = float(base_linvel_world[0])
        yaw = _quat_xyzw_to_yaw(base_quat_xyzw)
        wz = float(base_angvel_world[2])
        yaw_err = (yaw - self.yaw0 + np.pi) % (2 * np.pi) - np.pi
        Fx = -settings.SAGITTAL_KP * (x - self.x0) - settings.SAGITTAL_KD * vx
        Mz = -settings.YAW_KP * yaw_err - settings.YAW_KD * wz
        return Fx, Mz
