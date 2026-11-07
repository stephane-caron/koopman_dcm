# SPDX-License-Identifier: Apache-2.0

"""Viser GUI interface and 3-D scene renderer."""

import numpy as np
import viser
import viser.uplot
from pinker import kinematics as kin
from pinker.visualizer import ViserVisualizer

from .interface import Interface


class ViserInterface(Interface):
    """Viser GUI interface and 3-D scene renderer.

    Sliders are created lazily when modules call add_slider from their
    register_interface method. The ZMP plot is appended to the GUI after
    all sliders.

    Attributes:
        viser_server: Viser server for scene rendering.
        visualizer: Visual model viewer.
    """

    viser_server: viser.ViserServer
    visualizer: ViserVisualizer

    def __init__(
        self,
        kin_model: kin.Model,
        visual_model: kin.GeometryModel,
    ) -> None:
        """Initialize Viser scene (sliders are added later via add_slider).

        Args:
            kin_model: Kinematic model of the robot.
            visual_model: Visual geometry model of the robot.
        """
        super().__init__()
        self.viser_server = viser.ViserServer()
        self.visualizer = ViserVisualizer(kin_model, visual_model)
        self.visualizer.init_viewer(viewer=self.viser_server, open=False)
        self.visualizer.load_viewer_model(rootNodeName="estimated")
        self.visualizer.display_visuals(True)
        self.visualizer.display_frames(False)
        self.viser_server.scene.set_background_image(
            np.full((1, 1, 3), 220, dtype=np.uint8)
        )
        self.viser_server.scene.add_grid(
            "/grid",
            width=2.0,
            height=2.0,
            position=(0.0, 0.0, 0.0),
            plane_opacity=0.5,
        )

        self._reset_requested = False
        _reset_button = self.viser_server.gui.add_button("Reset")

        @_reset_button.on_click
        def _(_) -> None:
            """Request a reset when the Reset button is clicked.

            Args:
                _: Click event, unused.
            """
            self._reset_requested = True

        self._slider_handles: dict[str, viser.GuiSliderHandle] = {}
        self._selector_handles: dict[str, viser.GuiDropdownHandle] = {}

    def reset(self) -> None:
        """Restore all GUI sliders and selectors to their initial values."""
        for name, default in self._slider_defaults.items():
            self._slider_handles[name].value = default
        for name, default in self._selector_defaults.items():
            self._selector_handles[name].value = default
        self._reset_requested = False

    @property
    def reset_requested(self) -> bool:
        """Whether the user clicked the Reset button."""
        return self._reset_requested

    def add_folder(self, label: str, expand_by_default: bool = True):
        """Add a GUI folder that groups subsequent sliders.

        Args:
            label: Folder label.
            expand_by_default: Whether the folder starts expanded.

        Returns:
            Viser folder handle, to be used as a context manager.
        """
        return self.viser_server.gui.add_folder(
            label, expand_by_default=expand_by_default
        )

    def add_selector(
        self,
        name: str,
        label: str,
        values: list[str],
        default: str,
    ) -> None:
        """Add a selector to the GUI as a dropdown.

        Args:
            name: Internal key used with get_selector.
            label: Display label shown in the GUI.
            values: Ordered list of selectable values.
            default: Initial value.
        """
        super().add_selector(name, label, values, default)
        self._selector_handles[name] = self.viser_server.gui.add_dropdown(
            label,
            options=values,
            initial_value=default,
        )

    def add_slider(
        self,
        name: str,
        label: str,
        min: float,
        max: float,
        step: float,
        initial_value: float,
    ) -> None:
        """Add a slider to the GUI.

        Args:
            name: Internal key used with get_slider.
            label: Display label shown in the GUI.
            min: Minimum slider value.
            max: Maximum slider value.
            step: Slider increment.
            initial_value: Initial value.
        """
        super().add_slider(name, label, min, max, step, initial_value)
        self._slider_handles[name] = self.viser_server.gui.add_slider(
            label,
            min=min,
            max=max,
            step=step,
            initial_value=initial_value,
        )

    def get_slider(self, name: str) -> float:
        """Return the current value of a GUI slider.

        Args:
            name: Internal key of the slider.

        Returns:
            Current value of the slider.
        """
        return self._slider_handles[name].value

    def set_slider(self, name: str, value: float) -> None:
        """Set the value of a GUI slider programmatically.

        Args:
            name: Internal key of the slider.
            value: New value of the slider.
        """
        self._slider_handles[name].value = value
        super().set_slider(name, value)

    def get_selector(self, name: str) -> str:
        """Return the current value of a GUI selector.

        Args:
            name: Internal key of the selector.

        Returns:
            Current value of the selector.
        """
        return self._selector_handles[name].value

    def set_selector(self, name: str, value: str) -> None:
        """Set the value of a GUI selector programmatically.

        Args:
            name: Internal key of the selector.
            value: New value, one of the selector's values.
        """
        self._selector_handles[name].value = value
        super().set_selector(name, value)
