# SPDX-License-Identifier: Apache-2.0

"""Base interface with no-op methods."""


class _NullFolder:
    """Folder context manager that groups nothing."""

    def __enter__(self):
        """Enter the folder.

        Returns:
            This folder.
        """
        return self

    def __exit__(self, *_):
        """Exit the folder."""


class Interface:
    """Base interface; slider values are fixed at their initial values.

    Subclasses override add_slider/get_slider to route values to a GUI
    or a joystick.
    """

    def __init__(self) -> None:
        """Initialize an interface without sliders or selectors."""
        self._slider_defaults: dict[str, float] = {}
        self._sliders: dict[str, float] = {}
        self._selector_defaults: dict[str, str] = {}
        self._selector_values: dict[str, list[str]] = {}
        self._selectors: dict[str, str] = {}

    def reset(self) -> None:
        """Restore all sliders and selectors to their initial values."""
        for name, default in self._slider_defaults.items():
            self._sliders[name] = default
        for name, default in self._selector_defaults.items():
            self._selectors[name] = default

    def reset_sliders(self) -> None:
        """Restore all sliders to their initial values, leaving selectors."""
        for name, default in self._slider_defaults.items():
            self._sliders[name] = default

    @property
    def reset_requested(self) -> bool:
        """Return True if the user has requested an episode reset."""
        return False

    def add_folder(
        self, label: str, expand_by_default: bool = True
    ) -> _NullFolder:
        """Return a context manager that groups subsequent sliders.

        Args:
            label: Folder label.
            expand_by_default: Whether the folder starts expanded.

        Returns:
            Context manager, which does nothing in this base interface.
        """
        return _NullFolder()

    def add_selector(
        self,
        name: str,
        label: str,
        values: list[str],
        default: str,
    ) -> None:
        """Register a selector.

        Args:
            name: Internal key used with get_selector.
            label: Display label shown in the GUI.
            values: Ordered list of selectable values.
            default: Value returned before any user interaction.
        """
        if name in self._selectors:
            raise KeyError(f"Selector '{name}' already exists")
        self._selector_defaults[name] = default
        self._selector_values[name] = values
        self._selectors[name] = default

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
            label: Display label shown in the GUI.
            min: Minimum slider value.
            max: Maximum slider value.
            step: Slider increment.
            initial_value: Value returned before any user interaction.
        """
        if name in self._sliders:
            raise KeyError(f"Slider '{name}' already exists")
        self._slider_defaults[name] = initial_value
        self._sliders[name] = initial_value

    def has_slider(self, name: str) -> bool:
        """Return True if a slider with this name is already registered.

        Lets several modules share one slider: the first to register
        creates it, the others find it and read the same value.

        Args:
            name: Internal key of the slider.
        """
        return name in self._sliders

    def get_selector(self, name: str) -> str:
        """Return the current value of a named selector.

        Args:
            name: Internal key of the selector.

        Returns:
            Current value of the selector.
        """
        return self._selectors[name]

    def get_selector_index(self, name: str) -> int:
        """Index of the current value in the selector's value list.

        Args:
            name: Internal key of the selector.

        Returns:
            Index of the current value.
        """
        return self._selector_values[name].index(self._selectors[name])

    def set_selector(self, name: str, value: str) -> None:
        """Set the value of a named selector programmatically.

        Args:
            name: Internal key of the selector.
            value: New value, one of the selector's values.
        """
        self._selectors[name] = value

    def get_slider(self, name: str) -> float:
        """Return the current value of a named slider.

        Args:
            name: Internal key of the slider.

        Returns:
            Current value of the slider.
        """
        return self._sliders[name]

    def set_slider(self, name: str, value: float) -> None:
        """Set the value of a named slider programmatically.

        Args:
            name: Internal key of the slider.
            value: New value of the slider.
        """
        self._sliders[name] = value

    def cycle(self, dt: float, observation: dict) -> None:
        """Integrate spine observations.

        Args:
            dt: Time step in seconds.
            observation: Spine observation containing joystick state.
        """
