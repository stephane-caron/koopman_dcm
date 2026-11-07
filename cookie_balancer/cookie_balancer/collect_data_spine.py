# SPDX-License-Identifier: Apache-2.0

"""Real-robot data collection entry point."""

import datetime
import socket
from pathlib import Path

import gymnasium as gym
import upkie.envs
from upkie.logging import logger
from upkie.utils.raspi import configure_agent_process, on_raspi

from . import settings
from .balancer import Balancer
from .gym import FallDetectionWrapper
from .interfaces import JoystickDataCollectionInterface

upkie.envs.register()

# Duration of one collection session, in minutes. Each session is saved to
# its own file ``{hostname}-MMDD-HHMM.h5`` in the working directory.
DURATION_MIN: float = 5.0

# Duration of fixed-parameter chunks, in seconds.
CHUNK_DURATION: float = 10.0

# Aesthetic purpose :)
_DONE_ART = r"""
          \o/    \o/    \o/    \o/    \o/    \o/    \o/    \o/    \o/
           |      |      |      |      |      |      |      |      |
          / \    / \    / \    / \    / \    / \    / \    / \    / \

                                ______________
                               /    ______    \
                               |    ------    |
                               |  Collection  |
                               |   complete   |
                               [O____________O]
                                |            |
                               [O]          [O]
                                |            |
                               .'.          .'.
                               |||          |||
                               |||          |||
                               '.'          '.'

"""


def collect_data_spine(
    duration_min: float = DURATION_MIN,
    output: str | None = None,
    chunk_duration: float = CHUNK_DURATION,
) -> None:
    """Run one real-robot data collection session.

    Collects data for ``duration_min`` minutes from the spine environment,
    writes them to an HDF5 file, then exits. Run multiple times to build up
    a full dataset.

    Args:
        duration_min: Duration of the collection process in minutes.
        output: Path of the HDF5 output file. Defaults to
            ``{hostname}-MMDD-HHMM.h5``.
        chunk_duration: Duration of each fixed-parameter data collection chunk,
            in seconds.
    """
    nb_steps = int(duration_min * 60 * settings.CONTROL_FREQUENCY)
    if on_raspi():
        configure_agent_process()

    if output is None:
        hostname = socket.gethostname()
        timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
        output = f"{hostname}-{timestamp}.h5"

    output_path = Path(output)
    logger.info("Collecting %d steps from spine → %s", nb_steps, output_path)

    with JoystickDataCollectionInterface(
        output_path=output_path,
        nb_steps=nb_steps,
        chunk_duration=chunk_duration,
        frequency=settings.CONTROL_FREQUENCY,
        axis_map={
            "pad_vertical": ("hack_velocity", -1.0),
        },
        button_map={
            "square_button": "ready",
        },
    ) as interface:
        with FallDetectionWrapper(
            gym.make(
                "Cookie-Spine-Servos",
                frequency=settings.CONTROL_FREQUENCY,
                frequency_checks=False,
            )
        ) as env:
            balancer = Balancer(env=env, interface=interface, supervised=True)
            try:
                while not interface.is_complete:
                    balancer.cycle()
                interface.reset_sliders()
                logger.info(
                    "Collection complete! Set ready=no to power down the beast"
                )
                print(_DONE_ART)
                episode_ended = False
                while not episode_ended:
                    episode_ended = balancer.cycle()
            except KeyboardInterrupt:
                logger.info(
                    "Interrupted, saving %d steps collected so far",
                    interface.nb_collected,
                )

    logger.info(
        "Saved %d steps to %s (terminated: %d, truncated: %d)",
        interface.nb_collected,
        output_path,
        interface.nb_terminated,
        interface.nb_truncated,
    )


if __name__ == "__main__":
    collect_data_spine()
