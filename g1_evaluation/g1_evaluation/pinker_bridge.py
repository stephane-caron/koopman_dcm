# SPDX-License-Identifier: Apache-2.0

"""Pinker kinematic model + configuration helpers.

This module is the only one in the project that loads the robot model
through ``pinker.load_robot_description``. The PyBullet bridge
loads the same URDF independently via the PyBullet loader; the two stay
in sync because we feed PyBullet's observations into Pinker every
cycle (see ``update_configuration``).

Free-flyer convention:
    q[0:3]  = base position in WORLD frame (m)
    q[3:7]  = base orientation as quaternion (x, y, z, w)
    q[7:]   = joint positions (rad)
    v[0:3]  = base linear velocity in WORLD frame, expressed in the
              LOCAL body frame at the base origin
    v[3:6]  = base angular velocity in the LOCAL body frame
    v[6:]   = joint velocities (rad/s)
"""

import numpy as np
from pinker import Configuration, load_robot_description
from pinker import kinematics as kin

from .robots.base import RobotSpec


class PinkerBridge:
    """Pinker model wrapper + PyBullet → ``pinker.Configuration`` updater.

    Attributes:
        robot: Pinker ``RobotWrapper`` (free-flyer base).
        configuration: ``pinker.Configuration`` that the rest of the project
            uses as the IK input.
        joint_qidx: For each name in ``spec.actuated_joint_names``, the
            corresponding slot in ``q[7:]``.
        joint_vidx: Same, but for ``v[6:]``.
    """

    def __init__(self, spec: RobotSpec) -> None:
        """Load the robot model and map actuated joints to q and v indices.

        Args:
            spec: Robot specification.
        """
        self.spec = spec
        self.robot = load_robot_description(
            spec.description,
            root_joint="free_flyer",
        )
        self.model = self.robot.model
        self.data = self.robot.data

        # Map actuated joint names to (idx_q, idx_v) into the full q and v
        # vectors. Joint indices include the floating base, so the offset
        # is implicit in the model.
        self.joint_qidx: list[int] = []
        self.joint_vidx: list[int] = []
        missing = []
        for name in spec.actuated_joint_names:
            if not self.model.exist_joint_name(name):
                missing.append(name)
                continue
            jid = self.model.get_joint_id(name)
            j = self.model.joints[jid]
            if j.nq != 1:
                raise ValueError(
                    f"Joint '{name}' has nq={j.nq}; only revolute joints "
                    f"with nq=1 are supported."
                )
            self.joint_qidx.append(j.idx_q)
            self.joint_vidx.append(j.idx_v)
        if missing:
            raise KeyError(f"Joints not found in kinematic model: {missing}")

        # Initialise q with neutral pose + the standing posture.
        q = kin.neutral(self.model)
        for name, angle in spec.standing_posture.items():
            j = self.model.joints[self.model.get_joint_id(name)]
            q[j.idx_q] = angle
        q[2] = spec.initial_base_z
        self.configuration = Configuration(self.model, self.data, q)
        self.q = q.copy()

        # Frame ids for the foot bodies (used by IK / state estimation).
        self.left_foot_frame_id = self.model.get_frame_id(spec.left_foot_frame)
        self.right_foot_frame_id = self.model.get_frame_id(spec.right_foot_frame)

    def update_configuration(
        self,
        base_position: np.ndarray,
        base_quat_xyzw: np.ndarray,
        q_actuated: np.ndarray,
    ) -> Configuration:
        """Update ``self.configuration`` from PyBullet's base + joint state.

        Args:
            base_position: World-frame base position (3,).
            base_quat_xyzw: Base orientation quaternion (x, y, z, w).
            q_actuated: Actuated joint positions ordered as
                ``spec.actuated_joint_names``.

        Returns:
            The updated ``pinker.Configuration`` (same instance).
        """
        q = self.q
        q[0:3] = base_position
        q[3:7] = base_quat_xyzw
        for slot, idx_q in enumerate(self.joint_qidx):
            q[idx_q] = q_actuated[slot]
        self.configuration.update(q)
        return self.configuration

    def get_q_actuated(self) -> np.ndarray:
        """Return the actuated joint positions from the current ``q``."""
        return np.array([self.q[idx_q] for idx_q in self.joint_qidx])

    def get_v_actuated_from_ik(self, ik_velocity: np.ndarray) -> np.ndarray:
        """Slice the IK velocity vector into the actuated joint velocities.

        Args:
            ik_velocity: Velocity vector from ``pinker.solve_ik`` (length nv,
                including the 6 floating-base components at the start).

        Returns:
            Actuated joint velocities ordered as ``actuated_joint_names``.
        """
        return np.array([ik_velocity[idx_v] for idx_v in self.joint_vidx])

    def compute_com(self) -> np.ndarray:
        """Return the CoM position in the world frame (3,)."""
        return kin.center_of_mass(self.model, self.data, self.q)

    def get_foot_transforms(self) -> tuple[kin.SE3, kin.SE3]:
        """Return the world-frame transforms of left and right foot frames.

        Uses Pinker's ``get_transform_frame_to_world`` which guarantees frame
        placements are up to date; ``data.oMf`` may be stale otherwise.
        """
        return (
            self.configuration.get_transform_frame_to_world(self.spec.left_foot_frame),
            self.configuration.get_transform_frame_to_world(self.spec.right_foot_frame),
        )
