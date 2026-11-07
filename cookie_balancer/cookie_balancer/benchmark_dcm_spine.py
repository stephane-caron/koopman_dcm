# SPDX-License-Identifier: Apache-2.0

"""Automated DCM benchmark on the real robot.

Rolls out a fixed number of oscillation periods under each DCM in
turn, and reports per-chunk tracking statistics. One *cycle* runs all three
DCMs back to back, in the fixed order kinematics, learned-LIP, Koopman.

Table 1 of the paper reports one session per amplitude: set ``AMPLITUDE``
below, then run::

    pixi run real-benchmark-dcm
"""

import datetime
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import gymnasium as gym
import numpy as np
import upkie.envs
from upkie.logging import logger
from upkie.utils.raspi import configure_agent_process, on_raspi

from . import settings
from .balancer import Balancer
from .gym import FallDetectionWrapper
from .interfaces import JoystickInterface
from .pattern_generator import NOMINAL_PERIOD

upkie.envs.register()

# Periods rolled out per chunk, and periods dropped at the start of each
# chunk as a settling transient.
NB_PERIODS = 10
NB_SETTLING_PERIODS = 2

# Protocol of Table 1: one session per amplitude in {0.1, ..., 0.6}, at the
# nominal period of 1.2 s, with 10 cycles over the three DCMs.
AMPLITUDE: float = 0.3  # normalized ZMP oscillation amplitude
PERIOD: float = NOMINAL_PERIOD  # seconds
NB_CYCLES: int = 10  # one cycle runs every DCM once

# Trained DCM observable compared to the kinematic and learned-LIP DCMs:
# stem of an ONNX file in models/
DCM_MODEL: str = "dcm-cookie"


@dataclass
class ChunkResult:
    """Outcome of one chunk under a single DCM.

    Attributes:
        cycle: 1-based cycle index.
        model: DCM model selector value.
        attempt: 1-based attempt index, > 1 after a retry.
        aborted: True if the chunk ended early (fall or manual reset).
        periods_done: Periods completed before an abort.
        rms: RMS of the ZMP command error over the measurement window.
        peak: Maximum absolute ZMP command error over the same window.
        gain: Command/reference amplitude ratio at the pattern frequency.
        lag: Command phase minus reference phase, in radians.
    """

    cycle: int
    model: str
    attempt: int
    aborted: bool = False
    periods_done: float = 0.0
    rms: float = math.nan
    peak: float = math.nan
    gain: float = math.nan
    lag: float = math.nan


def lock_in(signal: np.ndarray, time: np.ndarray, nu: float) -> complex:
    """Fundamental Fourier coefficient of a signal at angular frequency ``nu``.

    For ``signal = a sin(nu t + phi)`` sampled over an integer number of
    periods, the returned coefficient has magnitude ``a`` and carries the
    phase ``phi``. Broadband noise averages toward zero, which is what makes
    this robust where an RMS is not.

    Args:
        signal: Sampled signal.
        time: Sample times, in seconds.
        nu: Angular frequency of the pattern, in rad/s.

    Returns:
        Complex amplitude of the fundamental component.
    """
    return 2.0 * np.mean(signal * np.exp(-1j * nu * time))


def chunk_statistics(
    command: Sequence[float],
    reference: Sequence[float],
    dt: float,
    period: float,
) -> Tuple[float, float, float, float]:
    """Compute ``(rms, peak, gain, lag)`` over one measurement window.

    ``gain`` and ``lag`` are taken as the ratio and phase difference of the
    two signals' fundamental coefficients, so they are independent of where
    the window starts within a period.

    Args:
        command: Commanded ZMP samples.
        reference: Reference ZMP samples, same length.
        dt: Sampling period, in seconds.
        period: Pattern period, in seconds.

    Returns:
        ``(rms, peak, gain, lag)``, with ``lag`` in radians wrapped to
        ``[-pi, pi]`` and negative when the command lags the reference.
    """
    u = np.asarray(command, dtype=float)
    u_ref = np.asarray(reference, dtype=float)
    error = u - u_ref
    rms = float(np.sqrt(np.mean(error**2)))
    peak = float(np.max(np.abs(error)))

    time = np.arange(u.size) * dt
    nu = 2.0 * np.pi / period
    z_cmd = lock_in(u, time, nu)
    z_ref = lock_in(u_ref, time, nu)
    if abs(z_ref) < 1e-9:  # zero-amplitude reference: gain undefined
        return rms, peak, math.nan, math.nan
    gain = float(abs(z_cmd) / abs(z_ref))
    lag = float(np.angle(z_cmd) - np.angle(z_ref))
    lag = float((lag + np.pi) % (2.0 * np.pi) - np.pi)  # wrap to [-pi, pi]
    return rms, peak, gain, lag


class DcmBenchmark:
    """Drive the balancer through the DCM-comparison protocol."""

    def __init__(
        self,
        balancer: Balancer,
        models: Sequence[str],
        amplitude: float,
        period: float,
        nb_cycles: int,
        dt: float,
    ) -> None:
        """Initialize the benchmark.

        Args:
            balancer: Balancer to drive, built in supervised mode.
            models: DCM selector values to roll out, in order, once per cycle.
            amplitude: Normalized ZMP oscillation amplitude.
            period: Pattern period, in seconds.
            nb_cycles: Number of cycles to run.
            dt: Control-loop time step, in seconds.
        """
        self.amplitude = amplitude
        self.balancer = balancer
        self.models = list(models)
        self.nb_cycles = nb_cycles
        self.period = period
        self.results: List[ChunkResult] = []
        self._dt = dt
        self._nb_samples = round(NB_PERIODS * period / self._dt)
        self._nb_settling = round(NB_SETTLING_PERIODS * period / self._dt)

    @property
    def _is_ready(self) -> bool:
        """Whether the operator has set ``ready`` to yes."""
        return self.balancer.interface.get_selector("ready") == "yes"

    def _apply(self, model: str, amplitude: float) -> None:
        """Re-assert the settings the balancer resets on episode end.

        Args:
            model: DCM selector value.
            amplitude: Normalized ZMP oscillation amplitude.
        """
        interface = self.balancer.interface
        interface.set_selector("dcm_model", model)
        interface.set_slider("amplitude", amplitude)
        interface.set_slider("offset", 0.0)
        interface.set_slider("period", self.period)

    def wait_until_ready(self, model: str) -> None:
        """Balance in place until the operator sets ``ready`` to yes.

        The amplitude is held at zero here, so the robot does not start
        oscillating until the chunk actually begins.

        Args:
            model: DCM selector value, applied while waiting.
        """
        if not self._is_ready:
            logger.info("Waiting for ready=yes (square button)...")
        while not self._is_ready:
            self._apply(model, 0.0)
            self.balancer.cycle()

    def run_chunk(self, cycle: int, model: str, attempt: int) -> ChunkResult:
        """Roll out one chunk and return its statistics.

        Args:
            cycle: 1-based cycle index.
            model: DCM selector value.
            attempt: 1-based attempt index.

        Returns:
            The chunk result, flagged as aborted if the episode ended or the
            operator cleared ``ready`` before the chunk completed.
        """
        duration = self._nb_samples * self._dt
        logger.info(
            "cycle %d, %-14s rolling out %.0f periods of %.2f s (%.0f s)%s",
            cycle,
            model,
            duration / self.period,
            self.period,
            duration,
            "" if attempt == 1 else f" [attempt {attempt}]",
        )
        command: List[float] = []
        reference: List[float] = []
        for step in range(self._nb_samples):
            self._apply(model, self.amplitude)
            episode_ended = self.balancer.cycle()
            if episode_ended or not self._is_ready:
                periods = step * self._dt / self.period
                logger.warning(
                    "cycle %d, %s: aborted after %.1f periods",
                    cycle,
                    model,
                    periods,
                )
                return ChunkResult(
                    cycle=cycle,
                    model=model,
                    attempt=attempt,
                    aborted=True,
                    periods_done=periods,
                )
            command.append(self.balancer.dcm_controller.zmp_line)
            reference.append(self.balancer.pattern_generator.zmp)

        rms, peak, gain, lag = chunk_statistics(
            command[self._nb_settling :],
            reference[self._nb_settling :],
            self._dt,
            self.period,
        )
        logger.info(
            "cycle %d, %-14s rms=%.3f peak=%.3f gain=%.3f lag=%+.3f rad",
            cycle,
            model,
            rms,
            peak,
            gain,
            lag,
        )
        return ChunkResult(
            cycle=cycle,
            model=model,
            attempt=attempt,
            periods_done=NB_PERIODS,
            rms=rms,
            peak=peak,
            gain=gain,
            lag=lag,
        )

    def stand_still(self) -> None:
        """Keep the robot balancing in place once the protocol is over.

        Returning from the control loop would cut the spine and drop the
        robot, so instead we hold it upright on a zero-amplitude pattern
        until the operator ends the episode (cross button) or it falls.
        """
        logger.info(
            "Benchmark complete! Still balancing: press X to terminate."
        )
        episode_ended = False
        while not episode_ended:
            self._apply("kinematics", 0.0)
            episode_ended = self.balancer.cycle()

    def run(self) -> List[ChunkResult]:
        """Run the whole protocol, retrying chunks that abort."""
        for cycle in range(1, self.nb_cycles + 1):
            for model in self.models:
                attempt = 1
                while True:
                    self.wait_until_ready(model)
                    result = self.run_chunk(cycle, model, attempt)
                    self.results.append(result)
                    if not result.aborted:
                        break
                    attempt += 1
                    logger.info("Retrying cycle %d, %s", cycle, model)
        return self.results


def format_report(
    results: Sequence[ChunkResult],
    models: Sequence[str],
    amplitude: float,
    period: float,
) -> str:
    """Render the per-chunk and summary tables as markdown.

    Args:
        results: Chunk results.
        models: DCM selector values, in the order of the summary table.
        amplitude: Normalized ZMP oscillation amplitude.
        period: Pattern period, in seconds.

    Returns:
        The report as a markdown string.
    """
    lines = [
        "# Protocol",
        "",
        f"- Normalized amplitude: {amplitude}",
        f"- Period: {period} s",
        f"- {NB_PERIODS} periods per chunk, "
        f"first {NB_SETTLING_PERIODS} discarded as settling",
        "- Statistics on the commanded ZMP against the reference",
        "- `gain` and `lag` from the fundamental Fourier coefficient",
        "",
        "# Experiments",
        "",
        "| Cycle | DCM model | RMS | Peak | Gain | Lag (rad) |",
        "|-------|-----------|------|------|------|-----------|",
    ]
    for r in results:
        if r.aborted:
            lines.append(
                f"| {r.cycle} | {r.model} | aborted after "
                f"{r.periods_done:.1f} periods | | | |"
            )
        else:
            lines.append(
                f"| {r.cycle} | {r.model} | {r.rms:.2f} | {r.peak:.2f} | "
                f"{r.gain:.2f} | {r.lag:+.2f} |"
            )

    lines += [
        "",
        "# Summary",
        "",
        "Mean over completed chunks, $\\pm$ half-width of the 95% "
        "confidence interval on the mean.",
        "",
        "| DCM model | n | RMS | Peak | Gain | Lag (rad) | aborted |",
        "|-----------|---|-----|------|------|-----------|---------|",
    ]
    for model in models:
        done = [r for r in results if r.model == model and not r.aborted]
        nb_aborted = sum(1 for r in results if r.model == model and r.aborted)
        if not done:
            lines.append(f"| {model} | 0 | | | | | {nb_aborted} |")
            continue
        cells = []
        for field in ("rms", "peak", "gain", "lag"):
            values = np.array([getattr(r, field) for r in done], dtype=float)
            cells.append(f"{values.mean():.3f} ± {_ci_half_width(values):.3f}")
        lines.append(
            f"| {model} | {len(done)} | "
            + " | ".join(cells)
            + f" | {nb_aborted} |"
        )
    return "\n".join(lines) + "\n"


def _ci_half_width(values: np.ndarray, confidence: float = 0.95) -> float:
    """Half-width of the t-based confidence interval on the mean.

    Args:
        values: Samples.
        confidence: Confidence level of the interval.

    Returns:
        Half-width of the interval, NaN with fewer than two samples.
    """
    n = values.size
    if n < 2:
        return math.nan
    from scipy import stats  # local import: not needed on the robot path

    return float(
        stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, n - 1)
        * values.std(ddof=1)
        / math.sqrt(n)
    )


def resolve_dcm_model(balancer: Balancer, requested: str | None) -> str:
    """Pick the learned DCM to benchmark.

    Args:
        balancer: Balancer holding the loaded learned DCMs.
        requested: Model stem requested by the caller, or None.

    Returns:
        The selector value of the DCM observable to use.

    Raises:
        ValueError: If the request is unknown, or if it is omitted while
            several learned DCMs are available.
    """
    available = list(balancer.dcm_observables)
    if requested is not None:
        if requested not in available:
            raise ValueError(
                f"unknown model {requested!r}; available: {available}"
            )
        return requested
    if len(available) == 1:
        return available[0]
    raise ValueError(
        f"several learned DCMs available, set DCM_MODEL: {available}"
    )


def benchmark_dcm_spine(
    amplitude: float,
    nb_cycles: int,
    period: float = NOMINAL_PERIOD,
    dcm_model: str | None = None,
    output: str | None = None,
) -> None:
    """Run the DCM-comparison protocol on the real robot.

    Args:
        amplitude: Normalized ZMP oscillation amplitude.
        nb_cycles: Number of cycles; one cycle runs every DCM once.
        period: Pattern period, in seconds.
        dcm_model: Stem of the learned DCM to benchmark. Optional
            when exactly one learned DCM is available.
        output: Path of the markdown report. Defaults to a timestamped file
            in the working directory.
    """
    if on_raspi():
        configure_agent_process()
    if not 0.0 < amplitude <= 1.0:
        raise ValueError(f"amplitude must be in (0, 1], got {amplitude}")
    if nb_cycles < 1:
        raise ValueError(f"nb_cycles must be >= 1, got {nb_cycles}")
    if amplitude > 0.3:
        logger.warning(
            "amplitude %.2f exceeds the slider range used during data "
            "collection (0.3)",
            amplitude,
        )

    if output is None:
        timestamp = datetime.datetime.now().strftime("%m%d-%H%M")
        output = f"baseline-comparison-A{amplitude:g}-{timestamp}.md"
    output_path = Path(output)

    interface = JoystickInterface(
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
        models = (
            "kinematics",
            "learned-lip",
            resolve_dcm_model(balancer, dcm_model),
        )
        logger.info(
            "Benchmarking %s over %d cycles at amplitude %.2f, period %.2f s",
            ", ".join(models),
            nb_cycles,
            amplitude,
            period,
        )
        benchmark = DcmBenchmark(
            balancer=balancer,
            models=models,
            amplitude=amplitude,
            period=period,
            nb_cycles=nb_cycles,
            dt=env.unwrapped.dt,
        )

        def emit_report() -> None:
            """Write the report to its file and print it."""
            report = format_report(
                benchmark.results, models, amplitude, period
            )
            output_path.write_text(report, encoding="utf-8")
            print()
            print(report)
            logger.info("Report written to %s", output_path)

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
    benchmark_dcm_spine(
        amplitude=AMPLITUDE,
        nb_cycles=NB_CYCLES,
        period=PERIOD,
        dcm_model=DCM_MODEL,
    )


if __name__ == "__main__":
    main()
