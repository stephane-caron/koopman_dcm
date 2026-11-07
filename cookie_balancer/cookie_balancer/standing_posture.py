# SPDX-License-Identifier: Apache-2.0

"""Default standing posture of the Cookie robot."""

import numpy as np
from pinker import custom_configuration
from pinker.kinematics import RobotWrapper


def make_standing_posture(robot: RobotWrapper) -> np.ndarray:
    """Return the default standing-posture configuration vector.

    Sets the hip and knee angles to their nominal values; other joints keep
    their neutral (zero) value.  Works with any robot wrapper regardless of
    whether it has a floating base or a fixed base.

    Args:
        robot: Pinker robot wrapper built from the Cookie URDF.

    Returns:
        Writable configuration vector with the standing-posture joint angles.
    """
    return custom_configuration(
        robot.model,
        left_hip=0.1,
        left_knee=-0.2,
        right_hip=-0.1,
        right_knee=0.2,
    )
