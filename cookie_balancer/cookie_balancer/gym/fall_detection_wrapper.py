# SPDX-License-Identifier: Apache-2.0

"""Gymnasium wrapper that terminates episodes on fall detection."""

import gymnasium as gym
import numpy as np
from upkie.logging import logger

from .. import settings


class FallDetectionWrapper(gym.Wrapper):
    """Terminate an episode when the robot falls.

    A fall is detected when the angle between the base z-axis and the world
    z-axis exceeds ``fall_angle``.
    """

    fall_angle: float

    def __init__(
        self, env: gym.Env, fall_angle: float = settings.FALL_ANGLE
    ) -> None:
        """Initialize the wrapper.

        Args:
            env: Environment to wrap.
            fall_angle: Fall detection threshold in radians. A fall is
                detected when the base z-axis tilts more than this angle away
                from the world vertical.
        """
        super().__init__(env)
        self.fall_angle = fall_angle

    def _detect_fall(self, spine_observation: dict) -> bool:
        """Detect a fall from the tilt of the base.

        Args:
            spine_observation: Spine observation dictionary.

        Returns:
            True if the base tilts beyond ``fall_angle`` from the vertical.
        """
        rotation = np.array(
            spine_observation["base_orientation"]["rotation_base_to_world"]
        ).reshape((3, 3))
        # R[:, 2] is the base z-axis in world frame; its dot product with
        # world z = [0, 0, 1] is just R[2, 2].
        tilt = np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))
        if tilt > self.fall_angle:
            logger.warning(
                "Fall detected: tilt = %.2f rad (threshold %.2f rad)",
                tilt,
                self.fall_angle,
            )
            return True
        return False

    def log(self, name: str, entry) -> None:
        """Forward log entries to the base environment.

        Args:
            name: Name of the log entry.
            entry: Value to log.
        """
        self.unwrapped.log(name, entry)

    def step(self, action):
        """Run one timestep, terminating the episode on a detected fall.

        Args:
            action: Action to apply.

        Returns:
            Observation, reward, terminated and truncated flags, and info
            dictionary, as for the wrapped environment.
        """
        observation, reward, terminated, truncated, info = self.env.step(
            action
        )
        if self._detect_fall(info["spine_observation"]):
            terminated = True
        return observation, reward, terminated, truncated, info
