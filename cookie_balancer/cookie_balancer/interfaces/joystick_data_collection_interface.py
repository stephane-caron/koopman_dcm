# SPDX-License-Identifier: Apache-2.0

"""Data collection interface with optional joystick overrides."""

from pathlib import Path
from typing import Optional

from upkie.utils.clamp import clamp

from .data_collection_interface import DataCollectionInterface
from .joystick_interface import _AXIS_KEYS


class JoystickDataCollectionInterface(DataCollectionInterface):
    """Data collection interface that also reads a PS4 joystick.

    Extends DataCollectionInterface so that a subset of joystick axes can
    update sliders not managed by the curriculum (e.g. hack_velocity), and
    buttons can advance selectors (e.g. square_button → ready).  All data
    collection logic is inherited and unchanged.
    """

    def __init__(
        self,
        output_path: str | Path,
        nb_steps: int,
        chunk_duration: float,
        frequency: float,
        axis_deadband: float = 0.05,
        axis_map: dict[str, "str | tuple[str, float]"] = {},
        button_map: dict[str, str] = {},
        bar: Optional[object] = None,
    ) -> None:
        """Initialize the interface.

        Args:
            output_path: Path of the HDF5 output file. Opened on context
                manager entry, not here.
            nb_steps: Number of samples to collect (counting only steps
                where ``leg_controller.is_ready`` is True).
            chunk_duration: Duration of each fixed-parameter chunk, in
                seconds.
            frequency: Control loop frequency, in Hz.
            axis_deadband: Axis magnitude below which input is ignored.
            axis_map: Maps axis names to slider names or (slider_name, gain)
                pairs, same convention as JoystickInterface.
            button_map: Maps button observation keys to selector names; each
                rising edge advances the selector to its next value.
            bar: Optional tqdm progress bar, see DataCollectionInterface.
        """
        super().__init__(
            output_path=output_path,
            nb_steps=nb_steps,
            chunk_duration=chunk_duration,
            frequency=frequency,
            bar=bar,
        )
        self.axis_deadband = axis_deadband
        self._axis_map: dict[str, tuple[str, float]] = {
            axis: (target, 1.0) if isinstance(target, str) else target
            for axis, target in axis_map.items()
        }
        self._button_map: dict[str, str] = dict(button_map)
        self._prev_button_states: dict[str, bool] = {
            btn: False for btn in button_map
        }
        self._pad_accumulators: dict[str, float] = {}
        self._slider_range: dict[str, tuple[float, float]] = {}
        self._slider_step: dict[str, float] = {}

    def add_slider(
        self,
        name: str,
        label: str,
        min: float,
        max: float,
        step: float,
        initial_value: float,
    ) -> None:
        """Add a slider, recording its range and step for joystick input.

        Args:
            name: Slider name.
            label: Human-readable label.
            min: Minimum value.
            max: Maximum value.
            step: Value increment.
            initial_value: Initial value.
        """
        super().add_slider(name, label, min, max, step, initial_value)
        self._slider_range[name] = (min, max)
        self._slider_step[name] = step

    def reset(self) -> None:
        """Reset the collection state and the joystick button states."""
        super().reset()
        for btn_key in self._prev_button_states:
            self._prev_button_states[btn_key] = False
        self._pad_accumulators.clear()

    def cycle(self, dt: float, observation: dict) -> None:
        """Advance data collection and apply joystick inputs.

        Args:
            dt: Time step, in seconds.
            observation: Spine observation dictionary.
        """
        super().cycle(dt, observation)
        joystick = observation.get("joystick", {})
        for axis_name, (slider_name, gain) in self._axis_map.items():
            if slider_name not in self._sliders:
                continue
            obs_key, idx = _AXIS_KEYS[axis_name]
            raw = joystick.get(obs_key, [0.0, 0.0])[idx]
            lo, hi = self._slider_range[slider_name]
            if obs_key == "pad_axis":
                if abs(raw) >= self.axis_deadband:
                    step = self._slider_step[slider_name]
                    acc = clamp(
                        self._pad_accumulators.setdefault(
                            slider_name, self._sliders[slider_name]
                        )
                        + gain * raw * step,
                        lo,
                        hi,
                    )
                    self._pad_accumulators[slider_name] = acc
                    snapped = round((acc - lo) / step) * step + lo
                    self._sliders[slider_name] = clamp(snapped, lo, hi)
            else:  # analog stick: map position directly to value
                if abs(raw) < self.axis_deadband:
                    self._sliders[slider_name] = clamp(0.0, lo, hi)
                else:
                    self._sliders[slider_name] = clamp(
                        lo + (gain * raw + 1.0) / 2.0 * (hi - lo),
                        lo,
                        hi,
                    )
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
