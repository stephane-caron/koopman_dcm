# SPDX-License-Identifier: Apache-2.0

"""PyBullet I/O: load URDF, observe state, apply POSITION_CONTROL actions."""

from dataclasses import dataclass

import numpy as np
import pybullet
import pybullet_data
from robot_descriptions.loaders.pybullet import load_robot_description

from . import settings
from .robots.base import RobotSpec


@dataclass
class FootContact:
    """Per-foot aggregated contact state.

    Attributes:
        position: Center of pressure position in the world frame, in m.
        total_fz: Total normal force (positive upward), in N.
    """

    position: np.ndarray
    total_fz: float


@dataclass
class PyBulletObservation:
    """One cycle of PyBullet state.

    Attributes:
        base_position: World-frame base position (3,), in m.
        base_quat_xyzw: World-frame base orientation as quaternion (x,y,z,w).
        base_linvel_world: World-frame linear velocity (3,), in m/s.
        base_angvel_world: World-frame angular velocity (3,), in rad/s.
        q_actuated: Actuated joint positions, ordered per ``RobotSpec``.
        v_actuated: Actuated joint velocities.
        tau_actuated: Applied joint torques (from PyBullet).
        left_contact: Aggregated contact at the left foot.
        right_contact: Aggregated contact at the right foot.
    """

    base_position: np.ndarray
    base_quat_xyzw: np.ndarray
    base_linvel_world: np.ndarray
    base_angvel_world: np.ndarray
    q_actuated: np.ndarray
    v_actuated: np.ndarray
    tau_actuated: np.ndarray
    left_contact: FootContact
    right_contact: FootContact


class PyBulletBridge:
    """PyBullet client + URDF loader + observe/apply helpers."""

    def __init__(self, spec: RobotSpec) -> None:
        """Connect to PyBullet, load the robot URDF, build index maps.

        Args:
            spec: Robot specification.
        """
        self.spec = spec
        self.client_id = pybullet.connect(pybullet.DIRECT)
        pybullet.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.client_id
        )
        pybullet.setGravity(0, 0, -settings.GRAVITY, physicsClientId=self.client_id)
        pybullet.setTimeStep(settings.DT, physicsClientId=self.client_id)
        self.plane_id = pybullet.loadURDF("plane.urdf", physicsClientId=self.client_id)
        self.robot_id = load_robot_description(
            spec.description,
            basePosition=[0.0, 0.0, spec.initial_base_z],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            useFixedBase=False,
            physicsClientId=self.client_id,
        )

        # Build joint-name -> PyBullet joint index map.
        self.joint_name_to_index: dict[str, int] = {}
        self.link_name_to_index: dict[str, int] = {-1: -1}  # base
        for i in range(
            pybullet.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        ):
            info = pybullet.getJointInfo(
                self.robot_id, i, physicsClientId=self.client_id
            )
            jname = info[1].decode()
            lname = info[12].decode()
            self.joint_name_to_index[jname] = i
            self.link_name_to_index[lname] = i

        missing = [
            n for n in spec.actuated_joint_names if n not in self.joint_name_to_index
        ]
        if missing:
            raise KeyError(f"Actuated joints not found in PyBullet model: {missing}")

        self.actuated_indices = [
            self.joint_name_to_index[n] for n in spec.actuated_joint_names
        ]

        # Foot link indices used by getContactPoints filtering.
        if spec.left_foot_frame not in self.link_name_to_index:
            raise KeyError(
                f"Left foot frame '{spec.left_foot_frame}' not a link in "
                f"PyBullet model. Known links: "
                f"{sorted(self.link_name_to_index)}"
            )
        if spec.right_foot_frame not in self.link_name_to_index:
            raise KeyError(f"Right foot frame '{spec.right_foot_frame}' not a link.")
        self.left_foot_link = self.link_name_to_index[spec.left_foot_frame]
        self.right_foot_link = self.link_name_to_index[spec.right_foot_frame]

        # Select PyBullet's position control on actuated joints.
        for idx in self.actuated_indices:
            pybullet.setJointMotorControl2(
                self.robot_id,
                idx,
                pybullet.POSITION_CONTROL,
                targetPosition=0.0,
                force=0.0,
                physicsClientId=self.client_id,
            )

        self._last_zmp_left = np.zeros(3)
        self._last_zmp_right = np.zeros(3)

    def reset(self) -> None:
        """Reset to initial pose + standing posture, zero velocities."""
        pybullet.resetBasePositionAndOrientation(
            self.robot_id,
            [0.0, 0.0, self.spec.initial_base_z],
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client_id,
        )
        pybullet.resetBaseVelocity(
            self.robot_id,
            [0, 0, 0],
            [0, 0, 0],
            physicsClientId=self.client_id,
        )
        for name, angle in self.spec.standing_posture.items():
            pybullet.resetJointState(
                self.robot_id,
                self.joint_name_to_index[name],
                targetValue=angle,
                targetVelocity=0.0,
                physicsClientId=self.client_id,
            )
        self._last_zmp_left = np.zeros(3)
        self._last_zmp_right = np.zeros(3)

    def _foot_contact(self, link_id: int, prev_zmp: np.ndarray) -> FootContact:
        """Aggregate contact points on one foot into a single CoP + total fz.

        Args:
            link_id: PyBullet link index of the foot.
            prev_zmp: Position to fall back on if the total normal force is
                below ``settings.MIN_TOTAL_FZ``.

        Returns:
            ``FootContact(position, total_fz)``.
        """
        contacts = pybullet.getContactPoints(
            bodyA=self.robot_id,
            bodyB=self.plane_id,
            linkIndexA=link_id,
            physicsClientId=self.client_id,
        )
        total_fz = 0.0
        weighted_pos = np.zeros(3)
        for c in contacts:
            # c[6] is positionOnB (the plane); c[9] is normalForce.
            pos_b = np.array(c[6])
            fn = float(c[9])
            if fn <= 0.0:
                continue
            total_fz += fn
            weighted_pos += fn * pos_b
        if total_fz < settings.MIN_TOTAL_FZ:
            return FootContact(position=prev_zmp.copy(), total_fz=total_fz)
        return FootContact(position=weighted_pos / total_fz, total_fz=total_fz)

    def observe(self) -> PyBulletObservation:
        """Read base + joint + contact state from PyBullet."""
        base_pos, base_orn = pybullet.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.client_id
        )
        base_linvel, base_angvel = pybullet.getBaseVelocity(
            self.robot_id, physicsClientId=self.client_id
        )

        # Joint states. getJointStates is more efficient than a loop.
        states = pybullet.getJointStates(
            self.robot_id,
            self.actuated_indices,
            physicsClientId=self.client_id,
        )
        q = np.array([s[0] for s in states], dtype=np.float64)
        v = np.array([s[1] for s in states], dtype=np.float64)
        tau = np.array([s[3] for s in states], dtype=np.float64)

        left_contact = self._foot_contact(self.left_foot_link, self._last_zmp_left)
        right_contact = self._foot_contact(self.right_foot_link, self._last_zmp_right)
        # Cache for next-cycle small-force fallback.
        if left_contact.total_fz >= settings.MIN_TOTAL_FZ:
            self._last_zmp_left = left_contact.position
        if right_contact.total_fz >= settings.MIN_TOTAL_FZ:
            self._last_zmp_right = right_contact.position

        return PyBulletObservation(
            base_position=np.array(base_pos),
            base_quat_xyzw=np.array(base_orn),
            base_linvel_world=np.array(base_linvel),
            base_angvel_world=np.array(base_angvel),
            q_actuated=q,
            v_actuated=v,
            tau_actuated=tau,
            left_contact=left_contact,
            right_contact=right_contact,
        )

    def apply_action(
        self,
        q_target: np.ndarray,
        v_target: np.ndarray,
    ) -> None:
        """Send POSITION_CONTROL targets to the actuated joints.

        Args:
            q_target: Target positions, one per actuated joint (rad), ordered
                as ``spec.actuated_joint_names``.
            v_target: Target velocities, same ordering (rad/s).
        """
        kps = np.array([self.spec.kp[n] for n in self.spec.actuated_joint_names])
        kds = np.array([self.spec.kd[n] for n in self.spec.actuated_joint_names])
        forces = np.array(
            [self.spec.max_torque[n] for n in self.spec.actuated_joint_names]
        )
        pybullet.setJointMotorControlArray(
            self.robot_id,
            self.actuated_indices,
            pybullet.POSITION_CONTROL,
            targetPositions=q_target.tolist(),
            targetVelocities=v_target.tolist(),
            positionGains=kps.tolist(),
            velocityGains=kds.tolist(),
            forces=forces.tolist(),
            physicsClientId=self.client_id,
        )

    def step(self) -> None:
        """Advance the physics simulation by one ``settings.DT``."""
        pybullet.stepSimulation(physicsClientId=self.client_id)

    def apply_external_disturbance(
        self,
        base_position: np.ndarray,
        force_x: float,
        torque_z: float,
    ) -> None:
        """Queue an external force + torque on the base link for the next step.

        Both are expressed in the world frame; the force has zero y- and
        z-components and the torque has only a z-component, so a single
        call cannot contribute to the lateral (y) axis.

        Args:
            base_position: World-frame position of the base link
                (application point for the force).
            force_x: Force along world-x, in N.
            torque_z: Torque about world-z, in N·m.
        """
        if force_x != 0.0:
            pybullet.applyExternalForce(
                objectUniqueId=self.robot_id,
                linkIndex=-1,  # base link
                forceObj=[float(force_x), 0.0, 0.0],
                posObj=[float(c) for c in base_position],
                flags=pybullet.WORLD_FRAME,
                physicsClientId=self.client_id,
            )
        if torque_z != 0.0:
            pybullet.applyExternalTorque(
                objectUniqueId=self.robot_id,
                linkIndex=-1,
                torqueObj=[0.0, 0.0, float(torque_z)],
                flags=pybullet.WORLD_FRAME,
                physicsClientId=self.client_id,
            )

    def has_fallen(self, obs: PyBulletObservation) -> bool:
        """Check whether the robot has fallen.

        Args:
            obs: Simulator observation.

        Returns:
            True if the base height fell below ``FALL_HEIGHT_FRACTION``
            times the initial base height.
        """
        return (
            float(obs.base_position[2])
            < settings.FALL_HEIGHT_FRACTION * self.spec.initial_base_z
        )

    def close(self) -> None:
        pybullet.disconnect(physicsClientId=self.client_id)
