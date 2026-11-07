# SPDX-License-Identifier: Apache-2.0

"""Reproduce the Cookie part of Table 2: proportional DCM vs. MPC.

This is the Cookie counterpart of ``g1_evaluation``'s ``evaluate_mpc``,
run in PyBullet on the robot of the hardware experiments. All controllers
consume the same learned DCM. The question is whether the viability
constraint ``|ξ| ≤ ξ_max`` prevents a fall when the reference is
unfeasible, with ZMP outside the support polygon. The third controller,
``viability-less-mpc``, is the same MPC (horizon, cost weights, input box)
with only the DCM state box dropped, isolating the contribution of the
viability constraint from the MPC's other ingredients.

Since this evaluation runs in simulation, it requires a DCM trained on
*simulation* data: the ``DCM_MODEL`` stem must start with
``dcm-pybullet``. A DCM trained on real-robot logs would measure
sim-to-real transfer instead of the controller comparison.

Per-trial metrics:

- ``did_fall``, and ``fall_time`` (NaN if no fall), in seconds since the
  start of the excited phase;
- ``rmse_zmp_last_2T``: RMS of ``zmp_observed − pattern.zmp`` over the
  last two periods, the RMSE of Table 2;
- ``rmse_zmp_cmd_last_2T``: RMS of ``zmp_commanded − pattern.zmp`` over
  the same window, the metric of the real-robot benchmarks of Table 1;
- ``dcm_constraint_rate``, ``zmp_constraint_rate`` and
  ``predicted_peak_max``: MPC constraint diagnostics (NaN for ``dcm``).
"""

import itertools
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import scipy.stats
import upkie.envs
from upkie.logging import logger

from . import settings
from .balancer import Balancer
from .dcm_mpc import DcmMpc
from .gym import FallDetectionWrapper
from .interfaces import Interface

# Sweep of Table 2 (Cookie wheeled biped): sinusoidal ZMP references
# u_ref(t) = offset + amplitude * sin(2 pi t / period). The periods are
# those of the G1 sweep scaled by the LIP timescale sqrt(h_Cookie / h_G1),
# and bracket the 1.2 s nominal period of the hardware sessions. The
# non-zero offset and larger amplitudes push the reference DCM toward the
# edge of the support polygon.
AMPLITUDES: list[float] = [0.2, 0.4, 0.6]
OFFSETS: list[float] = [0.0, 0.5]
PERIODS: list[float] = [1.0, 1.2, 1.4]  # seconds
NB_SEEDS: int = 5  # seeds per (amplitude, offset, period) cell
DURATION: float = 10.0 * max(PERIODS)  # seconds of excited phase per trial
CONTROLLERS: list[str] = ["dcm", "viability-less-mpc", "mpc"]
CROUCH_HEIGHT: float = settings.DEFAULT_CROUCH_HEIGHT  # meters

# DCM observable used by all controllers: stem of an ONNX file in models/
DCM_MODEL: str = "dcm-pybullet-cookie"

# Per-trial results
OUTPUT: Path = settings.EVALUATION_DATA_DIR / "cookie-evaluation.h5"

# Cells run in parallel processes. All trials of a cell run in the same
# process, so that 18 cells take two rounds with 9 workers.
NUM_WORKERS: int = 9

# Seconds of standing still, after the leg controller reports ready,
# before the pattern amplitude is switched on. Mirrors the warmup of
# ``DataCollectionInterface`` so trials start from a settled posture.
WARMUP_DURATION = 3.0

# Give up on a trial whose leg controller never reports ready.
MAX_INIT_DURATION = 30.0

# DCMs trained in simulation are named after their dataset, which
# ``collect_data_pybullet`` writes as ``pybullet-<MMDD>-<HHMM>.h5``.
SIM_MODEL_PREFIX = "dcm-pybullet"

# Row labels of the paper's table, in order
PAPER_LABELS: dict[str, str] = {
    "dcm": "Proportional DCM",
    "viability-less-mpc": "MPC w/o viability",
    "mpc": "MPC",
}


def make_env():
    """Create the fall-detecting PyBullet environment.

    Returns:
        Wrapped Gymnasium environment.
    """
    return FallDetectionWrapper(
        gym.make(
            "Cookie-PyBullet-Servos",
            frequency=settings.CONTROL_FREQUENCY,
            frequency_checks=False,
            gui=False,
        )
    )


def evaluate_cell(
    balancer: Balancer,
    dcm_model: str,
    amplitude: float,
    offset: float,
    period: float,
    duration: float,
) -> dict:
    """Run one trial on an already-constructed balancer.

    The balancer resets itself whenever an episode ends, and ``reset``
    restores every slider *and selector* to its default — including
    sending ``dcm_model`` back to the kinematic DCM. Both the
    pattern parameters and the DCM model are therefore re-applied after
    every cycle rather than once upfront.

    Args:
        balancer: Balancer whose controller is locked for this trial.
        dcm_model: Stem of the DCM observable the controller consumes.
        amplitude: Pattern amplitude, in normalized units.
        offset: Pattern offset, in normalized units.
        period: Stepping period, in seconds.
        duration: Duration of the excited phase, in seconds.

    Returns:
        Dict with keys ``did_fall``, ``fell_in_warmup``, ``fall_time``,
        ``rmse_zmp_last_2T`` and ``rmse_zmp_cmd_last_2T``.
    """
    dt = 1.0 / settings.CONTROL_FREQUENCY
    interface = balancer.interface

    def apply(excited: bool) -> None:
        """Apply the DCM and pattern parameters of the trial.

        Args:
            excited: If False, hold the amplitude and offset at zero.
        """
        interface.set_selector("dcm_model", dcm_model)
        interface.set_slider("amplitude", amplitude if excited else 0.0)
        interface.set_slider("offset", offset if excited else 0.0)
        interface.set_slider("period", period)

    result = dict(
        did_fall=False,
        fell_in_warmup=False,
        fall_time=float("nan"),
        rmse_zmp_last_2T=float("nan"),
        rmse_zmp_cmd_last_2T=float("nan"),
        dcm_constraint_rate=float("nan"),
        zmp_constraint_rate=float("nan"),
        predicted_peak_max=float("nan"),
    )

    # Phase 1: hold still until the leg controller has taken over, then
    # warm up for a few seconds at zero amplitude.
    balancer.reset()
    # Balancer.reset does not rewind the pattern, so rewind it here: every
    # trial then starts at phase 0, where the sinusoid is at its offset,
    # and switching the amplitude on introduces no step discontinuity.
    balancer.pattern_generator.reset()
    apply(excited=False)
    nb_warmup_steps = int(WARMUP_DURATION / dt)
    nb_ready_steps = 0
    for k in range(int(MAX_INIT_DURATION / dt)):
        if balancer.cycle():  # episode ended: the robot fell while settling
            result["did_fall"] = True
            result["fell_in_warmup"] = True
            result["fall_time"] = 0.0
            return result
        apply(excited=False)
        if balancer.leg_controller.is_ready:
            nb_ready_steps += 1
            if nb_ready_steps >= nb_warmup_steps:
                break
    else:
        logger.warning(
            "Leg controller never became ready within %.0f s; "
            "counting the trial as a warmup fall",
            MAX_INIT_DURATION,
        )
        result["did_fall"] = True
        result["fell_in_warmup"] = True
        result["fall_time"] = 0.0
        return result

    # Phase 2: switch the pattern on and measure.
    apply(excited=True)
    nb_steps = int(duration / dt)
    nb_last_2T = max(1, int(2.0 * period / dt))
    zmp_errors: list[float] = []
    zmp_cmd_errors: list[float] = []
    dcm_active: list[bool] = []
    zmp_active: list[bool] = []
    predicted_peaks: list[float] = []
    for k in range(nb_steps):
        ended = balancer.cycle()
        controller = balancer.dcm_controller
        zmp_ref = balancer.pattern_generator.zmp
        zmp_errors.append(float(balancer.state_estimator.zmp_line - zmp_ref))
        zmp_cmd_errors.append(float(controller.zmp_line - zmp_ref))
        if isinstance(controller, DcmMpc):
            dcm_active.append(controller.dcm_constraint_active)
            zmp_active.append(controller.zmp_constraint_active)
            if np.isfinite(controller.predicted_peak):
                predicted_peaks.append(controller.predicted_peak)
        if ended:
            result["did_fall"] = True
            result["fall_time"] = float(k * dt)
            break
        apply(excited=True)

    if dcm_active:
        result["dcm_constraint_rate"] = float(np.mean(dcm_active))
    if zmp_active:
        result["zmp_constraint_rate"] = float(np.mean(zmp_active))
    if predicted_peaks:
        result["predicted_peak_max"] = float(np.max(predicted_peaks))

    if len(zmp_errors) >= nb_last_2T:
        for key, errors in (
            ("rmse_zmp_last_2T", zmp_errors),
            ("rmse_zmp_cmd_last_2T", zmp_cmd_errors),
        ):
            tail = np.array(errors[-nb_last_2T:])
            result[key] = float(np.sqrt(np.mean(tail * tail)))
    return result


def run_cell(
    cell: tuple[float, float, float],
    seeds: list[int],
    duration: float,
    controllers: list[str],
    dcm_model: str,
    crouch_height: float,
    quiet: bool,
) -> list[dict]:
    """Run all trials of one cell in a fresh PyBullet instance.

    Controllers are interleaved within each seed: they run back to back on
    the same environment, so that they see the same accumulated simulator
    state and the comparison cannot be an artifact of one having run
    first. Work is therefore split by cell, never by controller.

    Args:
        cell: ``(amplitude, offset, period)`` pattern parameters.
        seeds: Random seeds, each run on every controller.
        duration: Duration of the excited phase, in seconds.
        controllers: Controllers to evaluate.
        dcm_model: Stem of the DCM observable all controllers consume.
        crouch_height: Crouch height, in meters.
        quiet: If set, only log errors, as in parallel worker processes.

    Returns:
        List of per-trial result dicts, ordered by seed, then controller.
    """
    if quiet:
        logger.setLevel(logging.ERROR)
    # Spawned workers re-import the module but never run __main__, so the
    # Gymnasium environments have to be registered here as well.
    upkie.envs.register()

    amplitude, offset, period = cell
    rows: list[dict] = []
    with make_env() as env:
        balancer = Balancer(
            env=env,
            interface=Interface(),
            controller=controllers[0],
            # Locked, so that cycle() leaves ``controller_name`` alone and
            # this loop stays in charge of which controller runs.
            controller_locked=True,
            crouch_height=crouch_height,
        )
        for seed in seeds:
            for controller in controllers:
                np.random.seed(seed)
                balancer.controller_name = controller
                result = evaluate_cell(
                    balancer=balancer,
                    dcm_model=dcm_model,
                    amplitude=amplitude,
                    offset=offset,
                    period=period,
                    duration=duration,
                )
                rows.append(
                    dict(
                        amplitude=amplitude,
                        offset=offset,
                        period=period,
                        seed=seed,
                        controller=controller,
                        **result,
                    )
                )
    return rows


def _log_progress(n_done: int, n_total: int, rows: list[dict]) -> None:
    """Log the outcome of a finished cell.

    Args:
        n_done: Number of cells finished so far, this one included.
        n_total: Total number of cells in the sweep.
        rows: Result rows returned by :func:`run_cell`.
    """
    falls = {
        controller: sum(
            1 for r in rows if r["controller"] == controller and r["did_fall"]
        )
        for controller in dict.fromkeys(r["controller"] for r in rows)
    }
    logger.info(
        "[%d/%d] amp=%.2f off=%.2f per=%.2f falls: %s",
        n_done,
        n_total,
        rows[0]["amplitude"],
        rows[0]["offset"],
        rows[0]["period"],
        ", ".join(f"{c}={n}" for c, n in falls.items()),
    )


def run_sweep(
    dcm_model: str,
    seeds: list[int],
    duration: float,
    controllers: list[str],
    cells: list[tuple[float, float, float]],
    crouch_height: float,
    output: Path,
    num_workers: int = 1,
) -> None:
    """Run the sweep and save the per-trial results to HDF5.

    Each cell runs in its own fresh environment, so results do not depend
    on the number of workers. Rows are saved in the same order whatever
    the number of workers.

    Args:
        dcm_model: Stem of the DCM observable all controllers consume.
        seeds: Random seeds, each run on every cell and controller.
        duration: Duration of the excited phase, in seconds.
        controllers: Controllers to evaluate.
        cells: ``(amplitude, offset, period)`` pattern parameters.
        crouch_height: Crouch height, in meters.
        output: Output HDF5 path.
        num_workers: Number of parallel worker processes.
    """
    n_total = len(cells)
    cell_rows: list[list[dict]] = [[] for _ in cells]
    if num_workers == 1:
        for i, cell in enumerate(cells):
            cell_rows[i] = run_cell(
                cell,
                seeds,
                duration,
                controllers,
                dcm_model,
                crouch_height,
                quiet=False,
            )
            _log_progress(i + 1, n_total, cell_rows[i])
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=num_workers, mp_context=ctx
        ) as executor:
            futures = {
                executor.submit(
                    run_cell,
                    cell,
                    seeds,
                    duration,
                    controllers,
                    dcm_model,
                    crouch_height,
                    True,
                ): i
                for i, cell in enumerate(cells)
            }
            for n_done, future in enumerate(as_completed(futures), start=1):
                i = futures[future]
                cell_rows[i] = future.result()
                _log_progress(n_done, n_total, cell_rows[i])
    rows = [row for rows in cell_rows for row in rows]

    # Record the bounds actually in force, read off a controller built
    # the same way the workers build theirs.
    reference_mpc = DcmMpc()
    save_results(
        rows,
        output,
        dcm_model,
        duration,
        crouch_height,
        num_workers=num_workers,
        xi_max=reference_mpc.xi_max,
        z_max=reference_mpc.z_max,
    )


def save_results(
    rows: list[dict],
    output: Path,
    dcm_model: str,
    duration: float,
    crouch_height: float,
    num_workers: int = 1,
    xi_max: float | None = None,
    z_max: float | None = None,
) -> None:
    """Write per-trial results to an HDF5 file.

    Args:
        rows: Per-trial result dicts.
        output: Output HDF5 path.
        dcm_model: Stem of the DCM observable used.
        duration: Duration of the excited phase, in seconds.
        crouch_height: Crouch height, in meters.
        num_workers: Number of parallel processes the sweep used.
        xi_max: Half-width of the DCM viability box, recorded as
            provenance. Omitted from the file when None.
        z_max: Half-width of the ZMP input box, recorded for the same
            reason: it dominates the fall rate on this platform. Omitted
            from the file when None.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as hf:
        hf.attrs["robot"] = "cookie"
        hf.attrs["backend"] = "PyBullet"
        hf.attrs["dcm_model"] = dcm_model
        hf.attrs["duration"] = duration
        hf.attrs["crouch_height"] = crouch_height
        hf.attrs["num_workers"] = num_workers
        if xi_max is not None:
            hf.attrs["xi_max"] = xi_max
        if z_max is not None:
            hf.attrs["z_max"] = z_max
        for key, dtype in (
            ("amplitude", np.float32),
            ("offset", np.float32),
            ("period", np.float32),
            ("seed", np.int32),
            ("did_fall", bool),
            ("fell_in_warmup", bool),
            ("fall_time", np.float32),
            ("rmse_zmp_last_2T", np.float32),
            ("rmse_zmp_cmd_last_2T", np.float32),
            ("dcm_constraint_rate", np.float32),
            ("zmp_constraint_rate", np.float32),
            ("predicted_peak_max", np.float32),
        ):
            hf.create_dataset(
                key, data=np.array([r[key] for r in rows], dtype=dtype)
            )
        hf.create_dataset(
            "controller",
            data=np.array([r["controller"] for r in rows], dtype="S20"),
        )
    logger.info("Saved %d trials to %s", len(rows), output)


def print_table(input_path: Path) -> None:
    """Print fall rates and ZMP RMSE in the format of the paper's table.

    RMSE is reported as mean ± half-width of the 95 % confidence
    interval on the mean (Student t, ``n - 1`` degrees of freedom),
    over all trials of a controller with a full measurement window,
    fallen or not.

    Args:
        input_path: HDF5 file written by :func:`save_results`.
    """
    with h5py.File(input_path, "r") as hf:
        ctrl = hf["controller"][:].astype(str)
        fell = hf["did_fall"][:]
        rmse = hf["rmse_zmp_last_2T"][:]
        fell_in_warmup = (
            hf["fell_in_warmup"][:] if "fell_in_warmup" in hf else None
        )
        robot = hf.attrs.get("robot", "")
        dcm_model = hf.attrs.get("dcm_model", "")
        crouch_height = float(hf.attrs.get("crouch_height", float("nan")))

    rows = []
    for c, label in PAPER_LABELS.items():
        mask = ctrl == c
        if not mask.any():
            continue
        n_trials = int(mask.sum())
        n_falls = int(fell[mask].sum())
        r = rmse[mask]
        r = r[~np.isnan(r)].astype(np.float64)
        if len(r) > 1:
            t_crit = scipy.stats.t.ppf(0.975, df=len(r) - 1)
            half_width = t_crit * r.std(ddof=1) / np.sqrt(len(r))
            rmse_str = f"{r.mean():.2f} ± {half_width:.2f}"
        elif len(r) == 1:
            rmse_str = f"{r[0]:.2f}"
        else:
            rmse_str = "n/a"
        fall_rate = 100.0 * n_falls / n_trials
        rows.append(
            (
                label,
                f"{fall_rate:.0f}% ({n_falls}/{n_trials})",
                rmse_str,
            )
        )

    header = ("Controller", "Fall rate", "RMSE")
    widths = [max(len(row[i]) for row in rows + [header]) for i in range(3)]
    lines = [
        f"Robot: {robot} | learned DCM: {dcm_model} | "
        f"crouch height: {crouch_height:.3f} m",
        "  ".join(h.ljust(w) for h, w in zip(header, widths)),
        "  ".join("-" * w for w in widths),
    ]
    lines += [
        "  ".join(v.ljust(w) for v, w in zip(row, widths)) for row in rows
    ]
    if fell_in_warmup is not None and fell_in_warmup.any():
        lines.append(
            f"Note: {int(fell_in_warmup.sum())}/{len(ctrl)} trials fell "
            "before the pattern was switched on; they count as falls but "
            "contribute no RMSE."
        )
    print("\n".join(lines))


def main() -> None:
    """Run the sweep, save per-trial results and print the table."""
    if not DCM_MODEL.startswith(SIM_MODEL_PREFIX):
        raise SystemExit(
            f"DCM_MODEL {DCM_MODEL!r} does not start with "
            f"'{SIM_MODEL_PREFIX}': this evaluation runs in PyBullet and "
            "calls for a DCM trained on simulation data."
        )
    model_path = settings.MODELS_DIR / f"{DCM_MODEL}.onnx"
    if not model_path.exists():
        raise SystemExit(f"DCM observable {model_path} not found")

    cells = list(itertools.product(AMPLITUDES, OFFSETS, PERIODS))
    seeds = list(range(NB_SEEDS))
    logger.info(
        "Cookie | model=%s | crouch=%.3f m | duration=%.1f s | seeds=%d | "
        "amps=%d offsets=%d periods=%d controllers=%s | %d cells, "
        "%d trials | %d workers",
        DCM_MODEL,
        CROUCH_HEIGHT,
        DURATION,
        len(seeds),
        len(AMPLITUDES),
        len(OFFSETS),
        len(PERIODS),
        ",".join(CONTROLLERS),
        len(cells),
        len(cells) * len(seeds) * len(CONTROLLERS),
        NUM_WORKERS,
    )
    run_sweep(
        dcm_model=DCM_MODEL,
        seeds=seeds,
        duration=DURATION,
        controllers=CONTROLLERS,
        cells=cells,
        crouch_height=CROUCH_HEIGHT,
        output=OUTPUT,
        num_workers=NUM_WORKERS,
    )
    print_table(OUTPUT)


if __name__ == "__main__":
    upkie.envs.register()
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted")
