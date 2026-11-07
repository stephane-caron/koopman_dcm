# SPDX-License-Identifier: Apache-2.0

"""Time-series plot of DCM and ZMP targets and measurements."""

import numpy as np

try:
    import viser.uplot
except ImportError:
    pass

from ..interfaces.interface import Interface


class DcmZmpPlot:
    """Time-series plot of DCM and ZMP targets and measurements."""

    def __init__(self) -> None:
        """Initialize the plot without data."""
        self._dcm_observed = None
        self._dcm_target = None
        self._elapsed = 0.0
        self._plot = None
        self._zmp_command = None
        self._zmp_observed = None
        self._zmp_target = None

    def reset(self) -> None:
        """Reset the plot, which keeps its history across episodes."""

    def register_interface(self, interface: Interface) -> None:
        """Register DCM-ZMP plot with the given interface.

        Args:
            interface: Active input interface.
        """
        if not hasattr(interface, "viser_server"):
            return

        gui = interface.viser_server.gui
        _N_PLOT = 300  # 3 s at 100 Hz
        self._plot_t = np.zeros(_N_PLOT)
        self._plot_dcm_target = np.zeros(_N_PLOT)
        self._plot_dcm_observed = np.zeros(_N_PLOT)
        self._plot_zmp_target = np.zeros(_N_PLOT)
        self._plot_zmp_observed = np.zeros(_N_PLOT)
        self._plot_zmp_command = np.zeros(_N_PLOT)
        self._elapsed = 0.0

        self._plot = gui.add_uplot(
            data=(
                self._plot_t,
                self._plot_dcm_target,
                self._plot_dcm_observed,
                self._plot_zmp_target,
                self._plot_zmp_observed,
                self._plot_zmp_command,
            ),
            series=(
                viser.uplot.Series(label="t"),
                viser.uplot.Series(label="dcm*", stroke="#9999ff", width=2),
                viser.uplot.Series(label="dcm", stroke="#0000cc", width=2),
                viser.uplot.Series(
                    label="zmp*",
                    stroke="#ff9999",
                    width=2,
                    show=True,  # task-relevant
                ),
                viser.uplot.Series(
                    label="zmp",
                    stroke="#cc0000",
                    width=2,
                    show=False,
                ),
                viser.uplot.Series(
                    label="zmpc",
                    stroke="#cc00cc",
                    width=2,
                    show=True,  # task-relevant
                ),
            ),
            title="DCM-ZMP",
            scales={
                "x": viser.uplot.Scale(time=False, auto=True),
                "y": viser.uplot.Scale(min=-1.1, max=1.1, auto=False),
            },
            legend=viser.uplot.Legend(show=True),
            aspect=2.5,
        )

    def cycle(
        self,
        dt: float,
        dcm_target: float,
        dcm_observed: float,
        zmp_target: float,
        zmp_observed: float,
        zmp_command: float,
    ) -> None:
        """Append a new sample to the DCM-ZMP plot.

        Args:
            dt: Time step in seconds.
            dcm_target: Target normalized line DCM in [-1, 1].
            dcm_observed: Estimate of the normalized line DCM in [-1, 1].
            zmp_target: Target normalized line ZMP in [-1, 1].
            zmp_observed: Estimate of the normalized line ZMP in [-1, 1].
            zmp_command: Commanded normalized line ZMP in [-1, 1].
        """
        self._elapsed += dt
        self._dcm_observed = dcm_observed
        self._dcm_target = dcm_target
        self._zmp_command = zmp_command
        self._zmp_observed = zmp_observed
        self._zmp_target = zmp_target

        if self._plot is not None:
            self.update_live_plot()

    def update_live_plot(self) -> None:
        """Append the latest values to the plotted time series."""
        self._plot_t[:-1] = self._plot_t[1:]
        self._plot_t[-1] = self._elapsed
        self._plot_dcm_target[:-1] = self._plot_dcm_target[1:]
        self._plot_dcm_target[-1] = self._dcm_target
        self._plot_dcm_observed[:-1] = self._plot_dcm_observed[1:]
        self._plot_dcm_observed[-1] = self._dcm_observed
        self._plot_zmp_target[:-1] = self._plot_zmp_target[1:]
        self._plot_zmp_target[-1] = self._zmp_target
        self._plot_zmp_observed[:-1] = self._plot_zmp_observed[1:]
        self._plot_zmp_observed[-1] = self._zmp_observed
        self._plot_zmp_command[:-1] = self._plot_zmp_command[1:]
        self._plot_zmp_command[-1] = self._zmp_command
        self._plot.data = (
            self._plot_t,
            self._plot_dcm_target,
            self._plot_dcm_observed,
            self._plot_zmp_target,
            self._plot_zmp_observed,
            self._plot_zmp_command,
        )

    def to_log(self) -> dict:
        """Get instantaneous values of the plot for logging.

        Returns:
            Dictionary mapping curve names to their current values.
        """
        return {
            k: v
            for k, v in {
                "dcm_observed": self._dcm_observed,
                "dcm_target": self._dcm_target,
                "zmp_command": self._zmp_command,
                "zmp_observed": self._zmp_observed,
                "zmp_target": self._zmp_target,
            }.items()
            if v is not None
        }
