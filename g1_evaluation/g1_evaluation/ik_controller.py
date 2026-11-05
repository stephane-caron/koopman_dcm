# SPDX-License-Identifier: Apache-2.0

"""Pinker differential IK controller for humanoids."""

import numpy as np
from pinker import kinematics as kin
from pinker import solve_ik
from pinker.tasks import FrameTask, PostureTask

from .pinker_bridge import PinkerBridge
from .robots.base import RobotSpec


class IKController:
    """Differential IK controller using Pinker.

    Tasks:

    - ``root`` FrameTask: anchors the floating-base link at its reset pose.
      The target is set **once** at ``set_initial_targets()`` and never
      updated.
    - Two foot FrameTasks: their xy and rotation targets stay at the reset
      pose; only ``z`` is modulated by the balancer via ``set_foot_heights``.
    - PostureTask: regularises arm + torso joints toward the standing
      posture so the upper body doesn't flop in the unconstrained DOF.
    """

    def __init__(self, bridge: PinkerBridge, root_frame: str = "pelvis") -> None:
        """Build the IK tasks.

        Args:
            bridge: Kinematic model and configuration of the robot.
            root_frame: Floating-base frame anchored at its reset pose.
        """
        self.bridge = bridge
        self.spec: RobotSpec = bridge.spec
        if not bridge.model.exist_frame(root_frame):
            raise KeyError(f"Root frame '{root_frame}' not in kinematic model.")
        self.root_frame = root_frame

        self.root_task = FrameTask(
            root_frame,
            position_cost=[1.0, 1.0, 1.0],
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self.left_foot_task = FrameTask(
            self.spec.left_foot_frame,
            position_cost=[1.0, 1.0, 1.0],
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self.right_foot_task = FrameTask(
            self.spec.right_foot_frame,
            position_cost=[1.0, 1.0, 1.0],
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self.posture_task = PostureTask(cost=1e-2)

        self.tasks: list[object] = [
            self.root_task,
            self.left_foot_task,
            self.right_foot_task,
            self.posture_task,
        ]

        # Foot anchor poses captured at ``set_initial_targets``.
        self._anchor_left: kin.SE3 | None = None
        self._anchor_right: kin.SE3 | None = None

    def set_initial_targets(self) -> None:
        """Anchor root, foot, and posture targets to the current pose."""
        self.root_task.set_target(
            self.bridge.configuration.get_transform_frame_to_world(self.root_frame)
        )
        lf = self.bridge.configuration.get_transform_frame_to_world(
            self.spec.left_foot_frame
        )
        rf = self.bridge.configuration.get_transform_frame_to_world(
            self.spec.right_foot_frame
        )
        self._anchor_left = lf.copy()
        self._anchor_right = rf.copy()
        self.left_foot_task.set_target(lf)
        self.right_foot_task.set_target(rf)
        q = self.bridge.configuration.q.copy()
        joint_names = (
            self.spec.posture_task_joint_names or self.spec.actuated_joint_names
        )
        for name in joint_names:
            angle = self.spec.standing_posture.get(name, 0.0)
            j = self.bridge.model.joints[self.bridge.model.get_joint_id(name)]
            q[j.idx_q] = angle
        self.posture_task.set_target(q)

    def set_foot_heights(self, height_difference: float) -> None:
        """Update foot frame targets with a height-difference offset.

        Args:
            height_difference: Right-minus-left foot target height, in m.
                ``> 0`` raises the right foot and lowers the left.
        """
        assert self._anchor_left is not None
        assert self._anchor_right is not None
        half = 0.5 * height_difference
        left = self._anchor_left.copy()
        left.translation[2] -= half
        right = self._anchor_right.copy()
        right.translation[2] += half
        self.left_foot_task.set_target(left)
        self.right_foot_task.set_target(right)

    def solve(self, dt: float, solver: str = "proxqp") -> np.ndarray:
        """Compute IK velocity, integrate, and return the velocity.

        Args:
            dt: Time step (s).
            solver: QP solver passed to Pinker.

        Returns:
            The IK velocity vector of length ``nv``.
        """
        velocity = solve_ik(
            self.bridge.configuration,
            self.tasks,
            dt,
            solver=solver,
            safety_break=False,
        )
        if not np.all(np.isfinite(velocity)):
            raise RuntimeError(f"IK returned a non-finite velocity: {velocity}")
        self.bridge.configuration.integrate_inplace(velocity, dt)
        self.bridge.q[:] = self.bridge.configuration.q
        return velocity
