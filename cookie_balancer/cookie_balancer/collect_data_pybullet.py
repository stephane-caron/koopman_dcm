# SPDX-License-Identifier: Apache-2.0

"""Autonomous data collection entry point."""

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import gymnasium as gym
import h5py
import numpy as np
import upkie.envs
from upkie.logging import logger

from . import settings
from .balancer import Balancer
from .gym import FallDetectionWrapper
from .interfaces import DataCollectionInterface

upkie.envs.register()

# Dataset of the Cookie column of Table 2: one hour of simulation at 100 Hz,
# in 10-second chunks of fixed pattern parameters.
NB_STEPS: int = 360_000  # minimum number of simulation steps to collect
CHUNK_DURATION: float = 10.0  # seconds
NUM_WORKERS: int = 8  # parallel PyBullet simulations
SEED: int = 0  # base random seed, worker i uses SEED + i
GUI: bool = False  # show the PyBullet GUI (single worker only)

# Output dataset
OUTPUT: Path = settings.DATA_DIR / "pybullet-new.h5"


def _flush(
    hf: h5py.File,
    datasets: List[h5py.Dataset],
    buffers: List[np.ndarray],
    nb_rows: int,
) -> None:
    """Append in-memory buffers to growable HDF5 datasets.

    Args:
        hf: Open HDF5 file (flushed to disk after the write).
        datasets: Resizable datasets to append to. The first dataset's
            current length is used as the common write offset, so all
            datasets must have the same length on entry.
        buffers: In-memory buffers aligned with ``datasets``. Each
            buffer's first ``nb_rows`` rows are written.
        nb_rows: Number of rows to append from each buffer.
    """
    base = datasets[0].shape[0]
    for ds, buf in zip(datasets, buffers):
        ds.resize(base + nb_rows, axis=0)
        ds[base : base + nb_rows] = buf[:nb_rows]
    hf.flush()


def _run_worker(args: tuple) -> Tuple[int, int, int, int]:
    """Run a single PyBullet simulation worker.

    The last entry of each worker is force-marked as truncated so that
    downstream pair construction never bridges across worker boundaries.

    Args:
        args: Tuple of (worker_id, nb_steps, output_path, chunk_duration,
            gui, seed, quiet).

            - worker_id (int): Index of this worker, used for the tqdm
              bar position.
            - nb_steps (int): Number of simulation steps to collect.
            - output_path (str): Path of this worker's HDF5 file.
            - chunk_duration (float): Duration of each fixed-parameter chunk,
              in seconds.
            - gui (bool): Whether to show the PyBullet GUI.
            - seed (int): Random seed for both NumPy and ``env.reset``.
            - quiet (bool): If True, suppress info/warning log messages.

    Returns:
        Tuple (worker_id, nb_steps_collected, nb_terminated,
        nb_truncated).
    """
    (
        worker_id,
        nb_steps,
        output_path,
        chunk_duration,
        gui,
        seed,
        quiet,
    ) = args
    if quiet:
        logger.setLevel(logging.ERROR)
    np.random.seed(seed)

    bar = None
    if quiet:
        try:
            import tqdm

            bar = tqdm.tqdm(
                total=nb_steps,
                unit="step",
                position=worker_id,
                desc=f"worker {worker_id}",
            )
        except ImportError:
            pass

    with DataCollectionInterface(
        output_path=output_path,
        nb_steps=nb_steps,
        chunk_duration=chunk_duration,
        frequency=settings.CONTROL_FREQUENCY,
        bar=bar,
    ) as interface:
        with FallDetectionWrapper(
            gym.make(
                "Cookie-PyBullet-Servos",
                frequency=settings.CONTROL_FREQUENCY,
                frequency_checks=False,
                gui=gui,
            )
        ) as env:
            balancer = Balancer(env=env, interface=interface)
            while not interface.is_complete:
                balancer.cycle()

    return (
        worker_id,
        interface.nb_collected,
        interface.nb_terminated,
        interface.nb_truncated,
    )


def _merge_files(
    part_paths: List[Path],
    output_path: Path,
    metadata: Optional[dict] = None,
) -> Tuple[int, int, int]:
    """Concatenate worker output files into a single HDF5 file.

    Datasets ``Y``, ``U``, ``amplitude``, ``offset``, ``period``,
    ``terminated`` and ``truncated`` are concatenated in the order of
    ``part_paths``. All inputs must share the same ``Y`` second-dimension
    (observation size). ``metadata`` is written as file-level attributes.

    Args:
        part_paths: Per-worker HDF5 files to merge, in concatenation
            order.
        output_path: Destination HDF5 file (overwritten).
        metadata: Scalar parameters to store as file attributes.

    Returns:
        Tuple (nb_total, nb_terminated, nb_truncated) summarizing the
        merged dataset.
    """
    with h5py.File(part_paths[0], "r") as src:
        obs_dim = src["Y"].shape[1]

    nb_terminated = 0
    nb_truncated = 0
    with h5py.File(output_path, "w") as out:
        for key, value in (metadata or {}).items():
            out.attrs[key] = value
        Y_out = out.create_dataset(
            "Y",
            shape=(0, obs_dim),
            maxshape=(None, obs_dim),
            dtype=np.float32,
        )
        U_out = out.create_dataset(
            "U", shape=(0,), maxshape=(None,), dtype=np.float32
        )
        A_out = out.create_dataset(
            "amplitude", shape=(0,), maxshape=(None,), dtype=np.float32
        )
        O_out = out.create_dataset(
            "offset", shape=(0,), maxshape=(None,), dtype=np.float32
        )
        P_out = out.create_dataset(
            "period", shape=(0,), maxshape=(None,), dtype=np.float32
        )
        T_out = out.create_dataset(
            "terminated", shape=(0,), maxshape=(None,), dtype=bool
        )
        R_out = out.create_dataset(
            "truncated", shape=(0,), maxshape=(None,), dtype=bool
        )
        for part in part_paths:
            with h5py.File(part, "r") as src:
                nb_rows = src["Y"].shape[0]
                if nb_rows == 0:
                    continue
                base = Y_out.shape[0]
                for ds, key in (
                    (Y_out, "Y"),
                    (U_out, "U"),
                    (A_out, "amplitude"),
                    (O_out, "offset"),
                    (P_out, "period"),
                    (T_out, "terminated"),
                    (R_out, "truncated"),
                ):
                    ds.resize(base + nb_rows, axis=0)
                    ds[base : base + nb_rows] = src[key][:]
                nb_terminated += int(src["terminated"][:].sum())
                nb_truncated += int(src["truncated"][:].sum())
        nb_total = Y_out.shape[0]
    return nb_total, nb_terminated, nb_truncated


def collect_data(
    nb_steps: int,
    num_workers: int,
    output: str | Path,
    chunk_duration: float,
    gui: bool,
    seed: int,
) -> None:
    """Run autonomous data collection.

    Args:
        nb_steps: Minimum total number of simulation steps to collect.
            Each worker's allotment is rounded up to a full chunk, so
            the actual total may slightly exceed this value.
        num_workers: Number of parallel PyBullet simulations.
        output: Path of the HDF5 output file.
        chunk_duration: Duration of each fixed-parameter chunk, in seconds.
        gui: Show PyBullet GUI.
        seed: Base random seed; worker i uses ``seed + i``.
    """
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if gui and num_workers > 1:
        logging.warning("PyBullet GUI won't work in subprocesses")

    steps_per_chunk = int(chunk_duration * settings.CONTROL_FREQUENCY)
    raw_per_worker = (nb_steps + num_workers - 1) // num_workers
    steps_per_worker = (
        (raw_per_worker + steps_per_chunk - 1) // steps_per_chunk
    ) * steps_per_chunk
    logger.info(
        "Collecting %d steps/worker x %d workers = %d steps total "
        "(requested >= %d, chunk-aligned to %d steps)",
        steps_per_worker,
        num_workers,
        steps_per_worker * num_workers,
        nb_steps,
        steps_per_chunk,
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if num_workers == 1:
        _, nb_collected, nb_terminated, nb_truncated = _run_worker(
            (
                0,
                steps_per_worker,
                str(output_path),
                chunk_duration,
                gui,
                seed,
                False,
            )
        )
    else:
        part_paths = [
            output_path.with_name(
                f"{output_path.stem}.part{i}{output_path.suffix}"
            )
            for i in range(num_workers)
        ]
        worker_args = [
            (
                i,
                steps_per_worker,
                str(part_paths[i]),
                chunk_duration,
                False,
                seed + i,
                True,
            )
            for i in range(num_workers)
        ]
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=num_workers, mp_context=ctx
        ) as executor:
            list(executor.map(_run_worker, worker_args))

        logger.info("Merging %d worker files into %s", num_workers, output)
        nb_collected, nb_terminated, nb_truncated = _merge_files(
            part_paths,
            output_path,
            metadata={
                "dt": 1.0 / settings.CONTROL_FREQUENCY,
                "frequency": settings.CONTROL_FREQUENCY,
                "chunk_duration": chunk_duration,
            },
        )
        for p in part_paths:
            p.unlink()

    logger.info(
        "Saved %d steps to %s (terminated: %d, truncated: %d)",
        nb_collected,
        output,
        nb_terminated,
        nb_truncated,
    )


if __name__ == "__main__":
    try:
        collect_data(
            nb_steps=NB_STEPS,
            num_workers=NUM_WORKERS,
            output=OUTPUT,
            chunk_duration=CHUNK_DURATION,
            gui=GUI,
            seed=SEED,
        )
    except KeyboardInterrupt:
        logger.info("Data collection interrupted")
