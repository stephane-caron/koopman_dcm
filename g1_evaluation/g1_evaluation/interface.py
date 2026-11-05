# SPDX-License-Identifier: Apache-2.0

"""Base interface with no-op methods."""


class Interface:
    """Base interface; slider values are fixed at their initial values.

    Subclasses override get_slider to provide other values, e.g. the
    random ZMP references of data collection.
    """

    def __init__(self) -> None:
        self._slider_defaults: dict[str, float] = {}
        self._sliders: dict[str, float] = {}

    def reset(self) -> None:
        """Restore all sliders to their initial values."""
        for name, default in self._slider_defaults.items():
            self._sliders[name] = default

    def add_slider(
        self,
        name: str,
        label: str,
        min: float,
        max: float,
        step: float,
        initial_value: float,
    ) -> None:
        """Register a slider.

        Args:
            name: Internal key used with get_slider.
            label: Display label.
            min: Minimum slider value.
            max: Maximum slider value.
            step: Slider increment.
            initial_value: Value returned before any user interaction.
        """
        if name in self._sliders:
            raise KeyError(f"Slider '{name}' already exists")
        self._slider_defaults[name] = initial_value
        self._sliders[name] = initial_value

    def get_slider(self, name: str) -> float:
        """Return the current value of a named slider.

        Args:
            name: Slider name.

        Returns:
            Current slider value.
        """
        return self._sliders[name]

    def set_slider(self, name: str, value: float) -> None:
        """Set the value of a named slider programmatically.

        Args:
            name: Slider name.
            value: New slider value.
        """
        self._sliders[name] = value

    def cycle(self, dt: float, observation: dict) -> None:
        """Update the interface at each control cycle.

        Args:
            dt: Time step in seconds.
            observation: Observation dictionary (unused).
        """
