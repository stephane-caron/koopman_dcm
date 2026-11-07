# SPDX-License-Identifier: Apache-2.0

"""Joystick interface for the PS4 controller."""

from upkie.utils.clamp import clamp

from .interface import Interface

_AXIS_KEYS: dict[str, tuple[str, int | None]] = {
    # 2-D axes (index selects x or y component)
    "pad_horizontal": ("pad_axis", 0),
    "pad_vertical": ("pad_axis", 1),
    "left_horizontal": ("left_axis", 0),
    "left_vertical": ("left_axis", 1),
    "right_horizontal": ("right_axis", 0),
    "right_vertical": ("right_axis", 1),
    # Scalar inputs (index = None)
    "left_button": ("left_button", None),
    "right_button": ("right_button", None),
    "left_trigger": ("left_trigger", None),
    "right_trigger": ("right_trigger", None),
}

# Analog sticks use direct position-to-value mapping.
# Everything else (pad, buttons, triggers) uses the accumulator.
_DIRECT_OBS_KEYS = {"left_axis", "right_axis"}


class _Folder:
    """Folder grouping the sliders added within its context."""

    def __init__(self, interface: "JoystickInterface", label: str) -> None:
        """Initialize the folder.

        Args:
            interface: Joystick interface the sliders are added to.
            label: Folder label.
        """
        self._interface = interface
        self._label = label

    def __enter__(self) -> "_Folder":
        """Start assigning new sliders to this folder.

        Returns:
            This folder.
        """
        self._interface._folders.append((self._label, []))
        self._interface._current_folder = len(self._interface._folders) - 1
        return self

    def __exit__(self, *_) -> None:
        """Stop assigning new sliders to this folder."""
        self._interface._current_folder = None


class JoystickInterface(Interface):
    """Read slider values from a PS4 joystick.

    Attributes:
        axis_deadband: Stick value below which input is ignored.
    """

    def __init__(
        self,
        axis_map: dict[str, "str | tuple[str, float]"] = {},
        button_map: dict[str, str] = {},
        axis_deadband: float = 0.05,
    ) -> None:
        """Create joystick interface.

        Args:
            axis_map: Maps axis names (e.g. ``"right_horizontal"``) to slider
                names, or to ``(slider_name, gain)`` tuples. For non-stick
                inputs (pad, buttons, triggers) the gain magnitude sets the
                integration rate in slider steps per cycle (e.g. gain=0.1 →
                one step every ten cycles) and its sign flips direction. For
                analog stick axes the gain scales or flips the direct
                position-to-value mapping.
            button_map: Maps button observation keys (e.g.
                ``"square_button"``) to selector names. Each rising edge
                (False → True) advances the selector to the next value.
            axis_deadband: Axis magnitude below which input is ignored.
        """
        super().__init__()
        self.axis_deadband = axis_deadband
        self._axis_map: dict[str, tuple[str, float]] = {
            axis: (target, 1.0) if isinstance(target, str) else target
            for axis, target in axis_map.items()
        }
        self._button_map: dict[str, str] = dict(button_map)
        self._prev_button_states: dict[str, bool] = {
            btn: False for btn in button_map
        }
        self._current_folder: int | None = None
        self._folders: list[tuple[str, list[str]]] = []
        self._labels: dict[str, str] = {}
        self._last_snapshot: dict[str, object] = {}
        self._accumulators: dict[str, float] = {}
        self._slider_range: dict[str, tuple[float, float]] = {}
        self._slider_step: dict[str, float] = {}
        self._unfoldered: list[str] = []

    def reset(self) -> None:
        """Restore sliders and selectors to defaults, clear button states."""
        super().reset()
        for btn_key in self._prev_button_states:
            self._prev_button_states[btn_key] = False
        self._accumulators.clear()

    def add_folder(
        self, label: str, expand_by_default: bool = True
    ) -> _Folder:
        """Return a context manager that groups subsequent sliders.

        Args:
            label: Folder label, used when listing the sliders.
            expand_by_default: Unused, for compatibility with the GUI.

        Returns:
            Context manager assigning sliders added within it to the folder.
        """
        return _Folder(self, label)

    def add_selector(
        self,
        name: str,
        label: str,
        values: list[str],
        default: str,
    ) -> None:
        """Register a selector, mapped to a joystick button if any.

        Args:
            name: Internal key used with get_selector.
            label: Display label.
            values: Ordered list of selectable values.
            default: Initial value.
        """
        super().add_selector(name, label, values, default)
        self._labels[name] = label
        self._assign_to_current_folder(name)

    def add_slider(
        self,
        name: str,
        label: str,
        min: float,
        max: float,
        step: float,
        initial_value: float,
    ) -> None:
        """Register a slider, mapped to a joystick axis if any.

        Args:
            name: Internal key used with get_slider.
            label: Display label.
            min: Minimum slider value.
            max: Maximum slider value.
            step: Slider increment.
            initial_value: Initial value.
        """
        super().add_slider(name, label, min, max, step, initial_value)
        self._labels[name] = label
        self._slider_range[name] = (min, max)
        self._slider_step[name] = step
        self._assign_to_current_folder(name)

    def _assign_to_current_folder(self, name: str) -> None:
        """Assign a slider or selector to the current folder, if any.

        Args:
            name: Internal key of the slider or selector.
        """
        if self._current_folder is not None:
            self._folders[self._current_folder][1].append(name)
        else:
            self._unfoldered.append(name)

    def get_slider(self, name: str) -> float:
        """Return the current value of a named slider.

        Args:
            name: Internal key of the slider.

        Returns:
            Current value of the slider.
        """
        return self._sliders[name]

    def cycle(self, dt: float, observation: dict) -> None:
        """Integrate joystick axes into slider values.

        Args:
            dt: Time step in seconds.
            observation: Spine observation containing joystick state.
        """
        joystick = observation.get("joystick", {})
        for axis_name, (slider_name, gain) in self._axis_map.items():
            if slider_name not in self._sliders:
                continue
            obs_key, idx = _AXIS_KEYS[axis_name]
            if idx is not None:
                raw = joystick.get(obs_key, [0.0, 0.0])[idx]
            else:
                raw = max(0.0, float(joystick.get(obs_key, 0)))
            lo, hi = self._slider_range[slider_name]
            if obs_key in _DIRECT_OBS_KEYS:
                if abs(raw) < self.axis_deadband:
                    self._sliders[slider_name] = clamp(0.0, lo, hi)
                else:
                    self._sliders[slider_name] = clamp(
                        lo + (gain * raw + 1.0) / 2.0 * (hi - lo),
                        lo,
                        hi,
                    )
            else:  # pad / button / trigger: accumulate
                if abs(raw) >= self.axis_deadband:
                    step = self._slider_step[slider_name]
                    acc = clamp(
                        self._accumulators.setdefault(
                            slider_name, self._sliders[slider_name]
                        )
                        + gain * raw * step,
                        lo,
                        hi,
                    )
                    self._accumulators[slider_name] = acc
                    snapped = round((acc - lo) / step) * step + lo
                    self._sliders[slider_name] = clamp(snapped, lo, hi)
        for btn_key, selector_name in self._button_map.items():
            if selector_name not in self._selectors:
                continue
            pressed = bool(joystick.get(btn_key, False))
            if pressed and not self._prev_button_states[btn_key]:
                values = self._selector_values[selector_name]
                current = self._selectors[selector_name]
                idx = values.index(current)
                self._selectors[selector_name] = values[
                    (idx + 1) % len(values)
                ]
            self._prev_button_states[btn_key] = pressed
        snapshot = {**self._sliders, **self._selectors}
        if snapshot != self._last_snapshot:
            self._last_snapshot = snapshot
            self._print_status()

    def _format_value(self, name: str) -> str:
        """Format the value of a slider or selector for display.

        Args:
            name: Internal key of the slider or selector.

        Returns:
            Formatted value.
        """
        if name in self._selectors:
            return self._selectors[name]
        return f"{self._sliders[name]:.3f}"

    def _build_tree_rows(self) -> list[str]:
        """Build the rows of the status tree of sliders and selectors.

        Returns:
            Lines of the tree, grouped by folder.
        """
        sections: list[tuple] = [
            ("item", name) for name in self._unfoldered
        ] + [
            ("folder", label, items) for label, items in self._folders if items
        ]
        if not sections:
            return []

        entries: list[tuple[str, str, str]] = []
        for i, section in enumerate(sections):
            is_last_section = i == len(sections) - 1
            branch = "└── " if is_last_section else "├── "
            cont = "    " if is_last_section else "│   "
            if section[0] == "item":
                name = section[1]
                entries.append(
                    (
                        branch,
                        self._labels.get(name, name),
                        self._format_value(name),
                    )
                )
            else:
                _, label, item_names = section
                entries.append((branch, label, "-"))
                for j, name in enumerate(item_names):
                    is_last_item = j == len(item_names) - 1
                    entries.append(
                        (
                            cont + ("└── " if is_last_item else "├── "),
                            self._labels.get(name, name),
                            self._format_value(name),
                        )
                    )

        max_name_col = max(len(p) + len(n) for p, n, _ in entries) + 2
        rows = ["JoystickInterface"]
        for prefix, name, value in entries:
            name_part = prefix + name
            rows.append(f"{name_part:<{max_name_col}}{value}")
        return rows

    def _print_status(self) -> None:
        """Print the status tree of sliders and selectors."""
        rows = self._build_tree_rows()
        if rows:
            print("\n".join(rows))
