# SPDX-License-Identifier: Apache-2.0

"""Robot specification shared by PyBullet and Pinker bridges."""

from dataclasses import dataclass, field


@dataclass
class RobotSpec:
    """Per-robot configuration.

    Attributes:
        name: Short identifier used in the registry (e.g. ``"g1"``).
        description: Name of the ``robot_descriptions`` package to load
            (e.g. ``"g1_description"``).
        actuated_joint_names: Ordered list of actuated joint names. Defines
            the convention used by both bridges for q/v slicing and for the
            ``q_actuated`` slot in the observation.
        left_foot_frame: Kinematic-model frame name for the left foot.
            Probed at startup; used as the IK frame task target.
        right_foot_frame: Kinematic-model frame name for the right foot.
        initial_base_z: Base z coordinate so the foot soles rest on z=0.
        com_height: Reference CoM height above the foot soles. Used to
            initialise omega = sqrt(g / com_height) at reset.
        half_foot_spacing: Half the lateral distance between the two foot
            frames at the neutral configuration. Used to normalise
            ``com_line``, ``zmp_line`` to [-1, 1].
        foot_half_width: Half the foot collision box width (lateral). Used
            for the support-polygon bounds in the MPC.
        foot_half_length: Half the foot collision box length (sagittal).
        standing_posture: Joint name → angle (rad) for the IK posture task
            and the initial PyBullet pose.
        kp: Joint name → ``positionGain`` (dimensionless blending factor,
            ~0.05-1.0) for PyBullet POSITION_CONTROL. NOT a torque/rad PD
            gain — PyBullet handles tracking via its constraint solver and
            this value tunes how stiffly the joint converges to the target.
        kd: Joint name → ``velocityGain`` (typically 1.0 — defaults if absent).
        max_torque: Joint name → ``force`` cap (N·m). This is the real PD
            output limit.
        transform_sole_to_frame: 3-vector offset (in foot-frame coordinates)
            from the foot frame origin to the sole centre. ``None`` if the
            foot frame is already at the sole.
    """

    name: str
    description: str
    actuated_joint_names: list[str]
    left_foot_frame: str
    right_foot_frame: str
    initial_base_z: float
    com_height: float
    half_foot_spacing: float
    foot_half_width: float
    foot_half_length: float
    standing_posture: dict[str, float]
    kp: dict[str, float]
    kd: dict[str, float]
    max_torque: dict[str, float]
    transform_sole_to_frame: tuple[float, float, float] = (0.0, 0.0, 0.0)

    posture_task_joint_names: list[str] = field(default_factory=list)
    """Subset of joints to include in the Pinker PostureTask. Defaults to all
    actuated joints if empty."""

    @property
    def nb_actuated(self) -> int:
        return len(self.actuated_joint_names)
