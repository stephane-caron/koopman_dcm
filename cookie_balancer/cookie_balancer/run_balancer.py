# SPDX-License-Identifier: Apache-2.0

"""Run the balancer on the real robot, with the joystick interface.

The operator confirms with the square button before the leg controller
activates. See ``run_pybullet`` to run the balancer in simulation.
"""

from typing import Literal

import gymnasium as gym
import upkie.envs
import upkie.logging
from upkie.logging import logger
from upkie.utils.raspi import configure_agent_process, on_raspi

from . import settings
from .balancer import Balancer
from .gym import FallDetectionWrapper
from .pattern_generator import NOMINAL_PERIOD

# Initial stepping period of the pattern generator, in seconds
INITIAL_PERIOD: float = NOMINAL_PERIOD

# DCM controller to start with: "dcm" for proportional feedback, "mpc" for
# the DCM-state MPC with the viability constraint, or "viability-less-mpc"
# for the same MPC without it. Switchable at runtime from the interface.
CONTROLLER: str = "dcm"

# Distance the robot goes down while crouching, in meters
CROUCH_HEIGHT: float = settings.DEFAULT_CROUCH_HEIGHT


def main(
    backend: Literal["PyBullet", "Spine"],
    gui: bool,
    initial_period: float,
    interface_name: Literal["joystick", "none", "viser"],
    supervised: bool,
    controller: str,
    crouch_height: float,
) -> None:
    """Run the agent.

    Args:
        backend: Backend to create the Upkie environment with.
        gui: Whether to show the PyBullet GUI.
        initial_period: Initial stepping period in seconds for the pattern
            generator slider.
        interface_name: Name of the interface to use.
        supervised: If True, require operator confirmation before the leg
            controller activates.
        controller: DCM controller the balancer starts with. The interface
            selector can switch between the three at runtime.
        crouch_height: Distance the robot goes down while crouching, in
            meters.
    """
    upkie.envs.register()
    env_args = {
        "frequency": settings.CONTROL_FREQUENCY,
        "frequency_checks": False,
    }
    if backend == "Spine" and on_raspi():
        configure_agent_process()
    elif backend == "PyBullet":
        env_args["gui"] = gui
    with FallDetectionWrapper(
        gym.make(f"Cookie-{backend}-Servos", **env_args)
    ) as env:
        balancer = Balancer(
            env=env,
            interface_name=interface_name,
            initial_period=initial_period,
            supervised=supervised,
            controller=controller,
            crouch_height=crouch_height,
        )
        try:
            while True:
                balancer.cycle()
        except KeyboardInterrupt:
            logger.info("Terminating in response to keyboard interrupt")
        finally:
            balancer.log_perfs()


if __name__ == "__main__":
    main(
        backend="Spine",
        gui=False,
        initial_period=INITIAL_PERIOD,
        interface_name="joystick",
        supervised=True,
        controller=CONTROLLER,
        crouch_height=CROUCH_HEIGHT,
    )
