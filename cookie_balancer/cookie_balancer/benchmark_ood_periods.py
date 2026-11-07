# SPDX-License-Identifier: Apache-2.0

"""Out-of-distribution period benchmark on the real robot.

If the DCM observable was trained on sinusoids drawn from a small grid of
periods, it might be inferring reference phase from its own action history
rather than reading robot state. Input scrambling
(``scripts/input_scrambling.py``) already shows the DCM observable barely
uses the action channel; this benchmark closes the other half by checking that
closed-loop tracking survives references the DCM observable cannot have
memorized.

Data collection drew periods uniformly from ``{0.3, 0.7, 1.0, 1.2, 1.4}`` s and
amplitudes from ``{0, 0.1, 0.2, 0.4, 0.8}``. This script rolls the learned DCM
over periods that are not in that grid. The default periods bracket the grid on
both sides: 2.0 s and 1.8 s are beyond its maximum, and 0.5 s sits between 0.3
and 0.7 s.

Alongside the markdown report, the session is recorded to an HDF5 file in
the schema of ``data/cookie.h5`` (``Y``, ``U``, ``terminated``,
``truncated``, ``amplitude``, ``offset``, ``period``), so that
``scripts/input_scrambling.py`` runs on it unchanged. ``Y`` is
captured through the same ``collect_data`` hook the training data used, so
the ``(y_k, u_k)`` pairing is identical.

Usage::

    pixi run real-benchmark-ood-periods
"""

import datetime
import math
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Sequence

import gymnasium as gym
import h5py
import numpy as np
from upkie.logging import logger
from upkie.utils.raspi import configure_agent_process, on_raspi

from . import settings
from .balancer import Balancer

# Importing this module registers the Upkie environments as a side effect,
# so this one must not call upkie.envs.register() again: a second call
# re-registers every environment and warns on each of them.
from .benchmark_dcm_spine import (
    NB_PERIODS,
    NB_SETTLING_PERIODS,
    ChunkResult,
    DcmBenchmark,
    _ci_half_width,
    resolve_dcm_model,
)
from .gym import FallDetectionWrapper
from .interfaces import JoystickInterface
from .interfaces.data_collection_interface import AMPLITUDES, PERIODS

# Periods rolled out, in seconds. None of these belong to the grid the
# training references were drawn from: 2.0 and 1.8 s sit beyond its
# maximum of 1.4 s, 0.5 s between 0.3 and 0.7 s.
OOD_PERIODS = (2.0, 1.8, 0.5)

# Normalized ZMP amplitude, also outside the training grid
OOD_AMPLITUDE = 0.3

# Cycles over the (period, model) grid: three trials per period in the paper
NB_CYCLES: int = 3

# Trained DCM observable compared to the kinematic DCM: stem of an ONNX file
# in models/
DCM_MODEL: str = "dcm-cookie"


@dataclass
class OodChunkResult(ChunkResult):
    """Chunk outcome, tagged with the period it was rolled out at.

    Attributes:
        period: Pattern period of the chunk, in seconds.
    """

    period: float = math.nan


class RecordingJoystickInterface(JoystickInterface):
    """Joystick interface that also records the DCM observable's input stream.

    The balancer calls ``collect_data`` on any interface exposing it, with
    exactly the observation and action that feed the DCM observable. Recording
    through that hook rather than sampling the balancer from outside
    guarantees the ``(y_k, u_k)`` pairing matches the training data.

    Attributes:
        recording: When False, samples are discarded. Set around chunks so
            that idle time between them does not enter the file.
    """

    recording: bool

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the interface with recording disabled."""
        super().__init__(*args, **kwargs)
        self._recording = False
        self._tag = (0, -1)  # (cycle, model index)
        self._Y: List[np.ndarray] = []
        self._U: List[float] = []
        self._terminated: List[bool] = []
        self._truncated: List[bool] = []
        self._amplitude: List[float] = []
        self._offset: List[float] = []
        self._period: List[float] = []
        self._cycle: List[int] = []
        self._model: List[int] = []

    @property
    def recording(self) -> bool:
        """Whether incoming samples are being recorded."""
        return self._recording

    @recording.setter
    def recording(self, value: bool) -> None:
        """Arm or disarm recording, closing the chunk on disarm.

        Chunks are appended to one contiguous stream, but the operator
        pause between them is not recorded. Flagging the last sample of
        each chunk as truncated keeps a delay embedding from spanning two
        chunks, which would splice unrelated states and two different DCM
        models into a single history.

        Args:
            value: True to record incoming samples.
        """
        value = bool(value)
        if self._recording and not value and self._truncated:
            self._truncated[-1] = True
        self._recording = value

    def set_tag(self, cycle: int, model_index: int) -> None:
        """Label the samples recorded from now on.

        Args:
            cycle: 1-based cycle index.
            model_index: Index of the DCM model in the benchmark list.
        """
        self._tag = (cycle, model_index)

    @property
    def nb_recorded(self) -> int:
        """Number of samples recorded so far."""
        return len(self._U)

    def collect_data(
        self,
        y_k: np.ndarray,
        u_k: float,
        terminated: bool,
        truncated: bool,
        is_ready: bool,
    ) -> None:
        """Record one transition, when recording and the legs are ready.

        Args:
            y_k: Observation vector at step k.
            u_k: ZMP line action at step k.
            terminated: Whether the episode terminated this step.
            truncated: Whether the episode was truncated this step.
            is_ready: Whether the leg controller has left its init phase.
        """
        if not self.recording or not is_ready:
            return
        cycle, model_index = self._tag
        self._Y.append(np.asarray(y_k, dtype=np.float32))
        self._U.append(float(u_k))
        self._terminated.append(bool(terminated))
        self._truncated.append(bool(truncated))
        self._amplitude.append(self.get_slider("amplitude"))
        self._offset.append(self.get_slider("offset"))
        self._period.append(self.get_slider("period"))
        self._cycle.append(cycle)
        self._model.append(model_index)

    def write_hdf5(self, path: Path, models: Sequence[str]) -> None:
        """Write the recorded stream in the training-dataset schema.

        Args:
            path: Output path of the HDF5 file.
            models: DCM model names, indexed by the recorded model column.
        """
        if not self._U:
            logger.warning("Nothing recorded, skipping %s", path)
            return
        with h5py.File(path, "w") as hf:
            hf.attrs["dt"] = 1.0 / settings.CONTROL_FREQUENCY
            hf.attrs["frequency"] = float(settings.CONTROL_FREQUENCY)
            hf.attrs["models"] = list(models)
            hf.create_dataset("Y", data=np.stack(self._Y))
            hf.create_dataset("U", data=np.asarray(self._U, dtype=np.float32))
            hf.create_dataset(
                "terminated", data=np.asarray(self._terminated, dtype=bool)
            )
            hf.create_dataset(
                "truncated", data=np.asarray(self._truncated, dtype=bool)
            )
            for name, values in (
                ("amplitude", self._amplitude),
                ("offset", self._offset),
                ("period", self._period),
            ):
                hf.create_dataset(
                    name, data=np.asarray(values, dtype=np.float32)
                )
            hf.create_dataset(
                "cycle", data=np.asarray(self._cycle, dtype=np.int32)
            )
            hf.create_dataset(
                "model", data=np.asarray(self._model, dtype=np.int32)
            )
        logger.info("Recorded %d steps to %s", len(self._U), path)


class OodPeriodBenchmark(DcmBenchmark):
    """Roll every DCM over every out-of-distribution period."""

    def __init__(
        self,
        balancer: Balancer,
        models: Sequence[str],
        amplitude: float,
        periods: Sequence[float],
        nb_cycles: int,
        dt: float,
        nb_periods: int,
    ) -> None:
        """Initialize the benchmark.

        Args:
            balancer: Balancer to drive, built in supervised mode.
            models: DCM selector values, rolled out once per period.
            amplitude: Normalized ZMP oscillation amplitude.
            periods: Pattern periods to sweep, in seconds.
            nb_cycles: Number of cycles over the whole (period, model) grid.
            dt: Control-loop time step, in seconds.
            nb_periods: Oscillation periods rolled out per chunk.
        """
        super().__init__(
            balancer=balancer,
            models=models,
            amplitude=amplitude,
            period=periods[0],
            nb_cycles=nb_cycles,
            dt=dt,
        )
        self.periods = list(periods)
        self._nb_periods = nb_periods
        self._select_period(self.periods[0])

    def _select_period(self, period: float) -> None:
        """Point the protocol at one period.

        Args:
            period: Pattern period, in seconds.
        """
        self.period = period
        self._nb_samples = round(self._nb_periods * period / self._dt)
        self._nb_settling = round(NB_SETTLING_PERIODS * period / self._dt)

    def run(self) -> List[OodChunkResult]:
        """Run the whole protocol, retrying chunks that abort."""
        interface = self.balancer.interface
        recording = isinstance(interface, RecordingJoystickInterface)
        for cycle in range(1, self.nb_cycles + 1):
            for period_index, period in enumerate(self.periods, start=1):
                self._select_period(period)
                logger.info(
                    "--- cycle %d/%d, period %.2f s (%d/%d), %s ---",
                    cycle,
                    self.nb_cycles,
                    period,
                    period_index,
                    len(self.periods),
                    "in the training grid"
                    if period in PERIODS
                    else "out of distribution",
                )
                for model_index, model in enumerate(self.models):
                    attempt = 1
                    while True:
                        if recording:
                            interface.recording = False
                        self.wait_until_ready(model)
                        if recording:
                            interface.set_tag(cycle, model_index)
                            interface.recording = True
                        result = self.run_chunk(cycle, model, attempt)
                        tagged = OodChunkResult(
                            **asdict(result), period=period
                        )
                        self.results.append(tagged)
                        if not result.aborted:
                            break
                        attempt += 1
                        logger.info(
                            "Retrying cycle %d, period %.2f s, %s",
                            cycle,
                            period,
                            model,
                        )
        if recording:
            interface.recording = False
        return self.results


def estimate_duration(
    periods: Sequence[float],
    nb_models: int,
    nb_cycles: int,
    nb_periods: int,
) -> float:
    """Total pattern time of the protocol, in seconds.

    Excludes operator time between chunks and any retries after a fall.

    Args:
        periods: Pattern periods to sweep, in seconds.
        nb_models: Number of DCM models rolled out per period.
        nb_cycles: Number of cycles over the grid.
        nb_periods: Oscillation periods per chunk.

    Returns:
        Lower bound on the wall-clock duration, in seconds.
    """
    return nb_cycles * nb_models * nb_periods * float(sum(periods))


def format_report(
    results: Sequence[OodChunkResult],
    models: Sequence[str],
    amplitude: float,
    periods: Sequence[float],
    nb_periods: int,
) -> str:
    """Render the per-chunk and per-period tables as markdown.

    Args:
        results: Chunk outcomes, in the order they were rolled out.
        models: DCM model names.
        amplitude: Normalized ZMP oscillation amplitude.
        periods: Pattern periods swept, in seconds.
        nb_periods: Oscillation periods per chunk.

    Returns:
        The report as a markdown string.
    """
    in_grid = [p for p in periods if p in PERIODS]
    lines = [
        "# Protocol",
        "",
        "Out-of-distribution periods: closed-loop tracking at references "
        "outside the grid the training data was drawn from.",
        "",
        f"- Normalized amplitude: {amplitude}"
        + (" (in the training grid!)" if amplitude in AMPLITUDES else ""),
        f"- Periods: {', '.join(f'{p:g}' for p in periods)} s",
        f"- Training grid: periods {sorted(PERIODS)} s, "
        f"amplitudes {sorted(AMPLITUDES)}",
        f"- {nb_periods} periods per chunk, "
        f"first {NB_SETTLING_PERIODS} discarded as settling",
        "- Statistics on the commanded ZMP against the reference",
        "- `gain` and `lag` from the fundamental Fourier coefficient",
    ]
    if in_grid:
        lines.append(
            f"- **Warning**: periods {in_grid} are in the training grid, "
            "so they are not out of distribution"
        )
    lines += [
        "",
        "# Experiments",
        "",
        "| Cycle | Period (s) | DCM model | RMS | Peak | Gain | Lag (rad) |",
        "|-------|-----------|-----------|------|------|------|-----------|",
    ]
    for r in results:
        if r.aborted:
            lines.append(
                f"| {r.cycle} | {r.period:g} | {r.model} | aborted after "
                f"{r.periods_done:.1f} periods | | | |"
            )
        else:
            lines.append(
                f"| {r.cycle} | {r.period:g} | {r.model} | {r.rms:.2f} | "
                f"{r.peak:.2f} | {r.gain:.2f} | {r.lag:+.2f} |"
            )

    lines += [
        "",
        "# Summary",
        "",
        "Mean over completed chunks, $\\pm$ half-width of the 95% "
        "confidence interval on the mean.",
        "",
        "| Period (s) | DCM model | n | RMS | Peak | Gain | Lag (rad) "
        "| aborted |",
        "|-----------|-----------|---|-----|------|------|-----------"
        "|---------|",
    ]
    for period in periods:
        for model in models:
            at = [
                r for r in results if r.model == model and r.period == period
            ]
            done = [r for r in at if not r.aborted]
            nb_aborted = sum(1 for r in at if r.aborted)
            if not done:
                lines.append(
                    f"| {period:g} | {model} | 0 | | | | | {nb_aborted} |"
                )
                continue
            cells = []
            for field in ("rms", "peak", "gain", "lag"):
                values = np.array(
                    [getattr(r, field) for r in done], dtype=float
                )
                cells.append(
                    f"{values.mean():.3f} ± {_ci_half_width(values):.3f}"
                )
            lines.append(
                f"| {period:g} | {model} | {len(done)} | "
                + " | ".join(cells)
                + f" | {nb_aborted} |"
            )
    return "\n".join(lines) + "\n"


def benchmark_ood_periods(
    amplitude: float,
    nb_cycles: int,
    periods: Sequence[float],
    nb_periods: int,
    dcm_model: str | None = None,
    models: Sequence[str] | None = None,
    output: str | None = None,
    record: str | None = None,
) -> None:
    """Run the out-of-distribution period protocol on the real robot.

    Args:
        amplitude: Normalized ZMP oscillation amplitude.
        nb_cycles: Number of cycles over the (period, model) grid.
        periods: Pattern periods to sweep, in seconds.
        nb_periods: Oscillation periods rolled out per chunk.
        dcm_model: Stem of the learned DCM to benchmark. Optional
            when exactly one learned DCM is available.
        models: DCM selector values to compare. Defaults to the kinematic
            baseline and the learned DCM.
        output: Path of the markdown report. Defaults to a timestamped file.
        record: Path of the HDF5 recording. Defaults to a timestamped file;
            pass an empty string to disable recording.
    """
    if on_raspi():
        configure_agent_process()
    if not 0.0 < amplitude <= 1.0:
        raise ValueError(f"amplitude must be in (0, 1], got {amplitude}")
    if nb_cycles < 1:
        raise ValueError(f"nb_cycles must be >= 1, got {nb_cycles}")
    if not periods:
        raise ValueError("at least one period is required")
    if any(p <= 0.0 for p in periods):
        raise ValueError(f"periods must be positive, got {periods}")
    if nb_periods <= NB_SETTLING_PERIODS:
        raise ValueError(
            f"nb_periods must exceed the {NB_SETTLING_PERIODS} settling "
            f"periods, got {nb_periods}"
        )

    for period in periods:
        if period in PERIODS:
            logger.warning(
                "period %.2f s is in the training grid %s, so it is not "
                "out of distribution",
                period,
                sorted(PERIODS),
            )
    if amplitude in AMPLITUDES:
        logger.warning(
            "amplitude %.2f is in the training grid %s",
            amplitude,
            sorted(AMPLITUDES),
        )

    timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
    if output is None:
        output = f"ood-periods-{timestamp}.md"
    output_path = Path(output)
    if record is None:
        record = f"{socket.gethostname()}-ood-{timestamp}.h5"
    record_path = Path(record) if record else None

    interface = RecordingJoystickInterface(
        # The d-pad drives hack_velocity, so the operator can nudge the robot
        # back into place between chunks without touching the protocol.
        axis_map={"pad_vertical": ("hack_velocity", -1.0)},
        button_map={"square_button": "ready"},
    )
    with FallDetectionWrapper(
        gym.make(
            "Cookie-Spine-Servos",
            frequency=settings.CONTROL_FREQUENCY,
            frequency_checks=False,
        )
    ) as env:
        balancer = Balancer(env=env, interface=interface, supervised=True)
        if models is None:
            models = (
                "kinematics",
                resolve_dcm_model(balancer, dcm_model),
            )
        seconds = estimate_duration(
            periods, len(models), nb_cycles, nb_periods
        )
        logger.info(
            "Benchmarking %s over periods %s s at amplitude %.2f, "
            "%d cycles: %d chunks, %.1f min of pattern time (excluding "
            "resets and retries)",
            ", ".join(models),
            ", ".join(f"{p:g}" for p in periods),
            amplitude,
            nb_cycles,
            nb_cycles * len(periods) * len(models),
            seconds / 60.0,
        )
        benchmark = OodPeriodBenchmark(
            balancer=balancer,
            models=models,
            amplitude=amplitude,
            periods=periods,
            nb_cycles=nb_cycles,
            dt=env.unwrapped.dt,
            nb_periods=nb_periods,
        )

        def emit_report() -> None:
            """Write the report and the recording to their files."""
            report = format_report(
                benchmark.results, models, amplitude, periods, nb_periods
            )
            output_path.write_text(report, encoding="utf-8")
            print()
            print(report)
            logger.info("Report written to %s", output_path)
            if record_path is not None:
                interface.write_hdf5(record_path, models)

        try:
            benchmark.run()
            emit_report()
            benchmark.stand_still()
        except KeyboardInterrupt:
            logger.info(
                "Interrupted, reporting %d chunks collected so far",
                len(benchmark.results),
            )
            emit_report()


def main() -> None:
    """Run the benchmark with the settings at the top of this module."""
    benchmark_ood_periods(
        amplitude=OOD_AMPLITUDE,
        nb_cycles=NB_CYCLES,
        periods=OOD_PERIODS,
        nb_periods=NB_PERIODS,
        dcm_model=DCM_MODEL,
    )


if __name__ == "__main__":
    main()
