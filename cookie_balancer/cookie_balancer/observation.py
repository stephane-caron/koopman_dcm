# SPDX-License-Identifier: Apache-2.0

"""Observation vector of the state estimator."""

from .utils import StructuredVector


def make_observation() -> StructuredVector:
    """Return a zero-filled structured observation vector.

    This is the single source of truth for the observation schema used both
    during data collection (StateEstimator) and training (train_dcm_estimator).
    Any change to the layout must be made here only.

    Layout, in order:

        q_actuated (6)  Joint angles, in rad.
        com_anchor (3)  Center-of-mass position in the anchor frame, in m.
        com_line   (1)  Normalized line CoM coordinate in [-1, 1], where +1 is
                        the left wheel and -1 is the right wheel.
        tau        (6)  Joint torques, in N·m.
        zmp_anchor (3)  Zero-tilting moment point in the anchor frame, in m.
        zmp_line   (1)  Normalized lateral ZMP coordinate in [-1, 1], same
                        convention as com_line.

    Joint vectors are ordered left then right: [left_hip, left_knee,
    left_wheel, right_hip, right_knee, right_wheel].

    Returns:
        A zero-filled StructuredVector with the layout described above.
    """
    return (
        StructuredVector()
        .add_slice("q_actuated", 6)
        .add_slice("com_anchor", 3)
        .add_slice("com_line", 1)
        .add_slice("tau", 6)
        .add_slice("zmp_anchor", 3)
        .add_slice("zmp_line", 1)
    )
