# SPDX-License-Identifier: Apache-2.0

"""Run the balancer in the PyBullet GUI, with a viser interface in the browser.

Settings shared with the real robot are at the top of ``run_balancer``.
"""

from .run_balancer import CONTROLLER, CROUCH_HEIGHT, INITIAL_PERIOD, main

# Show the PyBullet GUI, in addition to the viser interface
GUI: bool = True

if __name__ == "__main__":
    main(
        backend="PyBullet",
        gui=GUI,
        initial_period=INITIAL_PERIOD,
        interface_name="viser",
        supervised=False,
        controller=CONTROLLER,
        crouch_height=CROUCH_HEIGHT,
    )
