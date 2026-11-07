# SPDX-License-Identifier: Apache-2.0

"""Leg controller: inverse kinematics for crouching and leaning."""

import os

import numpy as np
import pinker
import upkie.model
from numpy.typing import NDArray
from pinker import solve_ik
from pinker.tasks import FrameTask, PostureTask
from upkie.logging import logger
from upkie.utils.clamp import clamp, clamp_abs
from upkie.utils.filters import abs_bounded_derivative_filter

from . import settings
from .interfaces.interface import Interface
from .standing_posture import make_standing_posture


class LegController:
    """Compute leg inverse kinematics.

    Attributes:
        max_crouch_height: Maximum distance along the vertical axis that the
            robot goes down while crouching, in meters.
        max_height_difference: Maximum height difference between the two
            contact frames, in meters.
        max_init_joint_velocity: Maximum joint velocity during the initial
            phase, in rad/s.
        max_leaning_velocity: Maximum leaning velocity, in m/s.
        tasks: Dictionary of inverse kinematics tasks.
        transform_neutral_to_world: Rest frame pose for each end effector.
        wheel_offset: Vertical wheel offset for each wheel.
    """

    max_crouch_height: float
    max_init_joint_velocity: float
    max_leaning_velocity: float
    tasks: dict
    transform_neutral_to_world: dict
    wheel_offset: dict[NDArray[float]]

    def __init__(
        self,
        model: upkie.model.Model,
        max_crouch_height: float = settings.MAX_CROUCH_HEIGHT,
        max_height_difference: float = settings.MAX_HEIGHT_DIFFERENCE,
        max_init_joint_velocity: float = settings.MAX_INIT_JOINT_VELOCITY,
        max_leaning_velocity: float = settings.MAX_LEANING_VELOCITY,
        max_leg_torque: float = settings.MAX_LEG_TORQUE,
        default_crouch_height: float = settings.DEFAULT_CROUCH_HEIGHT,
        kp_scale: float = settings.LEG_KP_SCALE,
        kd_scale: float = settings.LEG_KD_SCALE,
    ):
        """Create controller.

        Args:
            model: Upkie model providing URDF path and joint info.
            max_crouch_height: Maximum distance along the vertical axis that
                the robot goes down while crouching, in meters.
            max_height_difference: Maximum height difference between the two
                contact frames, in meters.
            max_init_joint_velocity: Maximum joint velocity during the
                initial phase, in rad/s.
            max_leaning_velocity: Maximum leaning velocity, in m/s.
            max_leg_torque: Maximum torque applied by leg actuators, in N.m.
            default_crouch_height: Initial crouch height target, in meters.
            kp_scale: Multiplier on the spine's position gain for leg servos.
            kd_scale: Multiplier on the spine's velocity gain for leg servos.
        """
        urdf_dir = os.path.dirname(model.urdf_path)
        description_dir = os.path.dirname(urdf_dir)
        robot = pinker.load_robot_urdf(
            model.urdf_path,
            root_joint=None,
            package_dirs=[urdf_dir, os.path.dirname(description_dir)],
        )
        neutral_configuration = pinker.Configuration(
            robot.model, robot.data, robot.q0
        )

        joint_names = {joint.name for joint in model.upper_leg_joints}

        wheel_names = {joint.name for joint in model.wheel_joints}
        wheel_frame_names = {}
        for joint in model.wheel_joints:
            for frame in model.kinematic_tree.frame_names:
                if frame.startswith(joint.name) and "stator" in frame:
                    wheel_frame_names[joint.name] = frame
                    break

        # Wheel frame tasks
        tasks = {
            wheel_name: FrameTask(
                wheel_frame_names[wheel_name],
                position_cost=[1.0, 1.0, 0.0],  # cost/m
                orientation_cost=0.0,  # cost/rad
                lm_damping=10.0,
            )
            for wheel_name in wheel_names
        }

        # Posture task
        tasks["posture"] = PostureTask(cost=1e-2)  # cost/rad
        tasks["posture"].set_target(make_standing_posture(robot))

        # Neutral frames, i.e. frames in the neutral configuration
        transform_neutral_to_world = {
            wheel: np.empty((4, 4)) for wheel in wheel_names
        }
        for wheel in wheel_names:
            transform_target_to_world = (
                neutral_configuration.get_transform_frame_to_world(wheel)
            )
            tasks[wheel].set_target(transform_target_to_world)
            transform_neutral_to_world[wheel] = transform_target_to_world

        self._default_crouch_height = default_crouch_height
        self._height_difference = 0.0
        self._initialized = False
        self._interface = None
        self._logged_contact = False
        self._logged_posture = False
        self._kd_scale = kd_scale
        self._kp_scale = kp_scale
        self._max_leg_torque = max_leg_torque
        self._q_init = None
        self._robot = robot
        self.ik_configuration = neutral_configuration
        self.joint_names = joint_names
        self.max_crouch_height = max_crouch_height
        self.max_height_difference = max_height_difference
        self.max_init_joint_velocity = max_init_joint_velocity
        self.max_leaning_velocity = max_leaning_velocity
        self.tasks = tasks
        self.transform_neutral_to_world = transform_neutral_to_world
        self.wheel_heights: dict[str, float] = {w: 0.0 for w in wheel_names}
        self.wheel_names = wheel_names
        self.wheel_target_heights: dict[str, float] = {
            w: 0.0 for w in wheel_names
        }

    @property
    def is_ready(self) -> bool:
        """True if and only if the initialization phase is over."""
        return self._initialized

    def reset(self) -> None:
        """Reset controller state to initial values."""
        self._height_difference = 0.0
        self._initialized = False
        self._logged_contact = False
        self._logged_posture = False
        self._q_init = None
        self.ik_configuration = pinker.Configuration(
            self._robot.model, self._robot.data, self._robot.q0
        )
        for wheel in self.wheel_names:
            self.tasks[wheel].set_target(
                self.transform_neutral_to_world[wheel]
            )
            self.wheel_heights[wheel] = 0.0
            self.wheel_target_heights[wheel] = 0.0

    def register_interface(self, interface: Interface) -> None:
        """Register sliders with the given interface.

        Args:
            interface: Active input interface.
        """
        self._interface = interface
        with interface.add_folder("Leg controller"):
            interface.add_slider(
                name="crouch_height",
                label="Crouch height (m)",
                min=0.0,
                max=self.max_crouch_height,
                step=0.01,
                initial_value=self._default_crouch_height,
            )

    def cycle(
        self,
        dt: float,
        spine_observation: dict,
        leaning_velocity: float,
    ) -> dict:
        """Compute action for a new cycle.

        Args:
            dt: Duration in seconds until next cycle.
            spine_observation: Spine observation dictionary.
            leaning_velocity: Lateral leaning velocity in m/s.

        Returns:
            Dictionary with the new action.

        Note:
            Leaning velocities are not tracked during initialization.
        """
        if not self._initialized:
            self.get_ik_action(dt, leaning_velocity=0.0)
            return self.get_init_action(spine_observation, dt)

        # Now initialized, ready to process leaning_velocity
        clamped_leaning_velocity = clamp_abs(
            leaning_velocity,
            self.max_leaning_velocity,
        )
        return self.get_ik_action(dt, clamped_leaning_velocity)

    def map_to_action(self, configuration) -> dict:
        """Map robot configuration to a servo action.

        Args:
            configuration: Robot configuration.

        Returns:
            Dictionary of position and velocity targets for each joint.
        """
        action = {}
        for joint_name in self.joint_names:
            joint_id = configuration.model.get_joint_id(joint_name)
            joint = configuration.model.joints[joint_id]
            action[joint_name] = {
                "position": configuration.q[joint.idx_q],
                "velocity": 0.0,
                "maximum_torque": self._max_leg_torque,
                "kp_scale": self._kp_scale,
                "kd_scale": self._kd_scale,
            }
        return action

    def get_init_action(self, observation: dict, dt: float) -> dict:
        """Initial phase where we return legs to the neutral configuration.

        Args:
            observation: Observation from the spine.
            dt: Duration in seconds until next cycle.

        Returns:
            Dictionary with the new action.
        """
        assert self._interface is not None, "call register_interface first"
        self._interface.set_slider(
            "crouch_height", self._default_crouch_height
        )

        if self._q_init is None:
            logger.info(
                "[%s] Moving to initial posture...",
                self.__class__.__name__,
            )
            q_init = self.ik_configuration.q.copy()  # for shape only
            for joint_name in self.joint_names:
                joint_id = self.ik_configuration.model.get_joint_id(joint_name)
                joint = self.ik_configuration.model.joints[joint_id]
                q_init[joint.idx_q] = observation["servo"][joint_name][
                    "position"
                ]
            self._q_init = q_init

        # Applying this filter is ok as it's only on joint angles
        nv = self._robot.nv
        self._q_init = abs_bounded_derivative_filter(
            prev_output=self._q_init,
            new_input=self.ik_configuration.q,
            dt=dt,
            max_derivative=np.full((nv,), self.max_init_joint_velocity),
        )

        # Difference is also OK, again only on joint angles
        q_diff = self._q_init - self.ik_configuration.q
        posture_converged = np.linalg.norm(q_diff, ord=1) < 1e-5
        floor_contact = observation["floor_contact"]["contact"]
        supervised = "ready" in self._interface._selectors
        ready = (
            self._interface.get_selector("ready") == "yes"
            if supervised
            else True
        )

        # Walk the operator through the initialization steps
        if posture_converged and not self._logged_posture:
            self._logged_posture = True
            if not floor_contact:
                logger.info(
                    "[%s] Posture initialized, waiting for floor contact...",
                    self.__class__.__name__,
                )
        if (
            posture_converged
            and floor_contact
            and supervised
            and not self._logged_contact
        ):
            self._logged_contact = True
            logger.info(
                "[%s] Posture ready and robot in contact, press the ready "
                "button to enable the leg controller",
                self.__class__.__name__,
            )

        if posture_converged and floor_contact and ready:
            if supervised:
                logger.info(
                    "[%s] Ready button pressed, let's move those legs!",
                    self.__class__.__name__,
                )
            else:
                logger.info(
                    "[%s] Posture ready and robot in contact, let's move "
                    "those legs!",
                    self.__class__.__name__,
                )
            self._initialized = True

        init_configuration = pinker.Configuration(
            self._robot.model, self._robot.data, self._q_init
        )
        return self.map_to_action(init_configuration)

    def get_ik_action(
        self,
        dt: float,
        leaning_velocity: float,
    ) -> dict:
        """Compute leg motion by differential inverse kinematics.

        Args:
            dt: Duration in seconds until next cycle.
            leaning_velocity: Lateral leaning velocity in m/s.

        Returns:
            Dictionary with the new action.
        """
        self.update_wheel_targets(dt, leaning_velocity)
        ik_velocity = solve_ik(
            self.ik_configuration,
            self.tasks.values(),
            dt,
            solver="proxqp",
            safety_break=True,
        )
        if not np.all(np.isfinite(ik_velocity)):
            raise RuntimeError(
                f"IK returned a non-finite velocity: {ik_velocity}"
            )
        self.ik_configuration.integrate_inplace(ik_velocity, dt)
        for wheel in self.wheel_names:
            self.wheel_heights[wheel] = (
                self.ik_configuration.get_transform_frame_to_world(
                    self.tasks[wheel].frame
                ).translation[2]
            )
        return self.map_to_action(self.ik_configuration)

    def update_wheel_targets(
        self,
        dt: float,
        leaning_velocity: float,
    ) -> None:
        """Update wheel contact target positions.

        Args:
            dt: Duration in seconds since the last cycle.
            leaning_velocity: Lateral leaning velocity in m/s. Positive leans
                right, negative leans left.
        """
        assert self._interface is not None, "call register_interface first"
        crouch_height = self._interface.get_slider("crouch_height")

        # Height difference is right - left wheel
        self._height_difference = clamp_abs(
            self._height_difference + leaning_velocity * dt,
            self.max_height_difference,
        )

        # Clamp targets for each wheel
        for wheel in self.wheel_names:
            sign = +1.0 if wheel == "right_wheel" else -1.0
            offset = sign * 0.5 * self._height_difference
            wheel_height = clamp(
                crouch_height + offset,
                0.0,
                self.max_crouch_height,
            )
            transform_target_to_world = self.transform_neutral_to_world[
                wheel
            ].copy()
            transform_target_to_world.translation[2] += wheel_height
            self.tasks[wheel].set_target(transform_target_to_world)
            self.wheel_target_heights[wheel] = (
                transform_target_to_world.translation[2]
            )

    def to_log(self) -> dict:
        """Get instantaneous values for logging.

        Returns:
            Dictionary mapping signal names to their current values.
        """
        return {
            "height_difference": self._height_difference,
            "is_ready": self.is_ready,
            "left_wheel_target_height": self.wheel_target_heights[
                "left_wheel"
            ],
            "right_wheel_target_height": self.wheel_target_heights[
                "right_wheel"
            ],
        }
