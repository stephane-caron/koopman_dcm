# SPDX-License-Identifier: Apache-2.0

"""Reproduce the G1 part of Table 2: proportional DCM vs. MPC.

All controllers consume the same learned DCM. The question is whether the
viability constraint ``|ξ| ≤ ξ_max`` prevents a fall when the reference is
unfeasible, with ZMP outside the support polygon.
"""

import itertools
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from pinker.exceptions import NoSolutionFound

from . import settings
from .balancer import Balancer
from .observation import make_observation
from .onnx_dcm_observable import load_dcm_observable
from .print_table import print_table
from .robots import get_spec

logger = logging.getLogger("g1_evaluation.evaluate_mpc")

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Sweep of Table 2 (G1 humanoid): sinusoidal ZMP references
# u_ref(t) = offset + amplitude * sin(2 pi t / period). The non-zero offset
# and larger amplitudes push the reference DCM toward the edge of the
# support polygon.
ROBOT: str = "g1"  # robot name in the registry
AMPLITUDES: list[float] = [0.3, 0.6, 0.9]
OFFSETS: list[float] = [0.0, 0.5]
PERIODS: list[float] = [1.3, 1.6, 1.9]  # seconds
NB_SEEDS: int = 5  # seeds per (amplitude, offset, period) cell
DURATION: float = 10.0 * max(PERIODS)  # seconds per trial
CONTROLLERS: list[str] = ["dcm", "viability-less-mpc", "mpc"]

# DCM observable used by all controllers
MODEL_PATH: Path = settings.MODELS_DIR / "dcm-pybullet-g1.onnx"

# Per-trial results
OUTPUT: Path = settings.DATA_DIR / "g1-evaluation.h5"

NUM_WORKERS: int = 12  # trials run in parallel processes


def evaluate_cell(
    robot: str,
    amplitude: float,
    offset: float,
    period: float,
    seed: int,
    duration: float,
    controller: str,
    learned_model: Path,
) -> dict:
    """Run one trial.

    Args:
        robot: Robot name.
        amplitude, offset, period: Pattern parameters.
        seed: Numpy random seed.
        duration: Trial duration in seconds.
        controller: ``"dcm"`` (DcmController), ``"mpc"`` (DcmMpc), or
            ``"viability-less-mpc"`` (DcmMpc with the DCM state box
            removed).
        learned_model: ONNX path of the DCM observable.

    Returns:
        Dict with keys ``did_fall, fall_time, rmse_zmp_last_2T``.
    """
    np.random.seed(seed)
    spec = get_spec(robot)
    balancer = Balancer(spec, controller=controller)

    # The learned DCM overrides the LIP DCM in all controllers (see
    # Balancer.cycle, step 2).
    dcm_observable = load_dcm_observable(learned_model)

    balancer.reset()
    balancer.interface.set_slider("amplitude", amplitude)
    balancer.interface.set_slider("offset", offset)
    balancer.interface.set_slider("period", period)

    nb_last_2T = max(1, int(2.0 * period / settings.DT))
    zmp_errors: list[float] = []
    did_fall = False
    fall_time = float("nan")
    last_action = 0.0

    nb_steps = int(duration / settings.DT)
    for k in range(nb_steps):
        try:
            state, u, fell = balancer.cycle()
        except (AssertionError, NoSolutionFound) as exc:
            logger.info(
                "trial aborted at t=%.2fs: %s — counting as a fall",
                k * settings.DT,
                type(exc).__name__,
            )
            did_fall = True
            fall_time = float(k * settings.DT)
            break
        # Update the learned DCM used in the next cycle.
        q_actuated = balancer.pinker.get_q_actuated()
        com_anchor = state.com_world - state.anchor_world
        dcm_observable.cycle(
            settings.DT,
            q_actuated,
            com_anchor,
            float(state.com_line),
            last_action,
        )
        if np.isfinite(dcm_observable.dcm_line):
            balancer._dcm_y_override = float(dcm_observable.dcm_line)
        last_action = u
        zmp_errors.append(float(state.zmp_line - balancer.pattern.zmp))
        if fell:
            did_fall = True
            fall_time = float(k * settings.DT)
            break

    balancer.close()
    if len(zmp_errors) >= nb_last_2T:
        tail = np.array(zmp_errors[-nb_last_2T:])
        rmse_zmp_last_2T = float(np.sqrt(np.mean(tail * tail)))
    else:
        rmse_zmp_last_2T = float("nan")
    return dict(
        did_fall=did_fall,
        fall_time=fall_time,
        rmse_zmp_last_2T=rmse_zmp_last_2T,
    )


def _init_worker_logging() -> None:
    """Configure logging in a spawned worker like in the main process."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def _run_trial(trial: dict) -> dict:
    """Run one trial and return its result row (worker entry point).

    Args:
        trial: Keyword arguments of :func:`evaluate_cell`.

    Returns:
        Trial parameters (amplitude, offset, period, seed, controller)
        merged with the metrics returned by :func:`evaluate_cell`.
    """
    result = evaluate_cell(**trial)
    keys = ("amplitude", "offset", "period", "seed", "controller")
    return {key: trial[key] for key in keys} | result


def _log_progress(n_done: int, n_total: int, row: dict) -> None:
    """Log the outcome of a finished trial.

    Args:
        n_done: Number of trials finished so far, this one included.
        n_total: Total number of trials in the sweep.
        row: Result row returned by :func:`_run_trial`.
    """
    logger.info(
        "[%d/%d] amp=%.2f off=%.2f per=%.2f seed=%d "
        "ctrl=%s fell=%s fall_t=%.2f rmse_zmp=%.3f",
        n_done,
        n_total,
        row["amplitude"],
        row["offset"],
        row["period"],
        row["seed"],
        row["controller"],
        row["did_fall"],
        row["fall_time"],
        row["rmse_zmp_last_2T"],
    )


def run_sweep(
    robot: str,
    seeds: list[int],
    duration: float,
    learned_model: Path,
    controllers: list[str],
    output: Path,
    cells: list[tuple[float, float, float]],
    num_workers: int = 1,
) -> None:
    """Run the sweep and save to HDF5.

    Each trial builds its own simulation, controller and DCM observable, so
    trials are independent and can run in parallel. Rows are saved in
    the same order whatever the number of workers.

    Args:
        robot: Robot name in the registry.
        seeds: Random seeds, each run on every cell and controller.
        duration: Trial duration in seconds.
        learned_model: ONNX path of the DCM observable.
        controllers: Controllers to evaluate.
        output: Output HDF5 path.
        cells: ``(amplitude, offset, period)`` pattern parameters.
        num_workers: Number of parallel worker processes.
    """
    trials = [
        dict(
            robot=robot,
            amplitude=amp,
            offset=off,
            period=per,
            seed=seed,
            duration=duration,
            controller=ctrl,
            learned_model=learned_model,
        )
        for amp, off, per in cells
        for seed in seeds
        for ctrl in controllers
    ]
    n_total = len(trials)
    rows: list[dict] = [{} for _ in trials]
    if num_workers == 1:
        for i, trial in enumerate(trials):
            rows[i] = _run_trial(trial)
            _log_progress(i + 1, n_total, rows[i])
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=ctx,
            initializer=_init_worker_logging,
        ) as executor:
            futures = {
                executor.submit(_run_trial, trial): i for i, trial in enumerate(trials)
            }
            for n_done, future in enumerate(as_completed(futures), start=1):
                i = futures[future]
                rows[i] = future.result()
                _log_progress(n_done, n_total, rows[i])

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as hf:
        hf.attrs["robot"] = robot
        hf.attrs["duration"] = duration
        hf.attrs["learned_model"] = learned_model.name
        hf.attrs["num_workers"] = num_workers
        hf.create_dataset(
            "amplitude",
            data=np.array([r["amplitude"] for r in rows], dtype=np.float32),
        )
        hf.create_dataset(
            "offset",
            data=np.array([r["offset"] for r in rows], dtype=np.float32),
        )
        hf.create_dataset(
            "period",
            data=np.array([r["period"] for r in rows], dtype=np.float32),
        )
        hf.create_dataset(
            "seed", data=np.array([r["seed"] for r in rows], dtype=np.int32)
        )
        hf.create_dataset(
            "controller",
            data=np.array([r["controller"] for r in rows], dtype="S20"),
        )
        hf.create_dataset(
            "did_fall",
            data=np.array([r["did_fall"] for r in rows], dtype=bool),
        )
        hf.create_dataset(
            "fall_time",
            data=np.array([r["fall_time"] for r in rows], dtype=np.float32),
        )
        hf.create_dataset(
            "rmse_zmp_last_2T",
            data=np.array([r["rmse_zmp_last_2T"] for r in rows], dtype=np.float32),
        )
    logger.info("Saved %d trials to %s", len(rows), output)


def main() -> None:
    """Run the sweep, save per-trial results and print the table."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    # Check upfront that the DCM observable matches the robot's
    # observation, so that a mismatch fails before the sweep starts.
    spec = get_spec(ROBOT)
    expected_obs_dim = make_observation(spec).slice_of("com_line").stop
    dcm_observable = load_dcm_observable(MODEL_PATH)
    if dcm_observable.obs_dim != expected_obs_dim:
        raise SystemExit(
            f"ONNX model {MODEL_PATH} was trained with obs_dim="
            f"{dcm_observable.obs_dim}, but robot {ROBOT!r} produces obs_dim="
            f"{expected_obs_dim}."
        )

    cells = list(itertools.product(AMPLITUDES, OFFSETS, PERIODS))
    seeds = list(range(NB_SEEDS))
    logger.info(
        "Robot %s | duration=%.2fs | seeds=%d | "
        "amps=%d offsets=%d periods=%d controllers=%s | %d cells, "
        "%d trials | %d workers",
        ROBOT,
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
        robot=ROBOT,
        seeds=seeds,
        duration=DURATION,
        learned_model=MODEL_PATH,
        controllers=CONTROLLERS,
        output=OUTPUT,
        cells=cells,
        num_workers=NUM_WORKERS,
    )
    print_table(OUTPUT)


if __name__ == "__main__":
    main()
