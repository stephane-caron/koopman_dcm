# SPDX-License-Identifier: Apache-2.0

"""Collect the training dataset of the DCM observable.

The robot balances in static double support under the MPC, while tracking
sinusoidal ZMP references whose amplitude, offset and period are drawn at
random from the sets in ``data_collection_interface.py`` at each chunk.
Observations follow the layout of ``observation.make_observation``.
"""

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
from pinker.exceptions import NoSolutionFound

from . import settings
from .balancer import Balancer
from .data_collection_interface import DataCollectionInterface
from .observation import make_observation
from .robots import get_spec
from .robots.base import RobotSpec

logger = logging.getLogger("g1_evaluation.collect")

ROBOT: str = "g1"  # robot name in the registry
NB_STEPS: int = 1_000_000  # control steps at 240 Hz, about 70 min
NUM_WORKERS: int = 12  # parallel PyBullet processes
CHUNK_DURATION: float = 10.0  # seconds per random ZMP reference
SEED: int = 0  # worker i uses SEED + i
OUTPUT: Path = settings.DATA_DIR / "pybullet-g1-new.h5"


def _mark_last_sample_terminated(interface: DataCollectionInterface) -> None:
    """Flip ``terminated=True`` on the most-recently recorded sample.

    Used when ``balancer.cycle()`` raises (typically Pinker's NoSolutionFound)
    so the trainer's episode-boundary logic treats the discontinuity as a
    termination. Reaches into the interface's private buffers / datasets;
    if the buffer is empty (just flushed) we patch the on-disk dataset.

    Args:
        interface: Data collection interface recording the samples.
    """
    if interface._buf_idx > 0:
        interface._T_buf[interface._buf_idx - 1] = True
        interface._nb_terminated += 1
        return
    T_ds = interface._T_ds
    if T_ds is None or T_ds.shape[0] == 0:
        return
    T_ds[-1] = True
    interface._nb_terminated += 1


def _populate_observation(spec: RobotSpec, balancer: Balancer, state) -> np.ndarray:
    """Build the observation vector for one cycle.

    Args:
        spec: Robot specification.
        balancer: Running balancer (provides bridge / kinematics access).
        state: ``State`` returned by ``balancer.cycle``.

    Returns:
        Float32 array of length ``make_observation(spec).dim``.
    """
    obs = make_observation(spec)
    obs["q_actuated"] = balancer.pinker.get_q_actuated()
    obs["com_anchor"] = state.com_world - state.anchor_world
    obs["com_line"] = state.com_line
    # tau: re-observe to snapshot appliedJointMotorTorque after the step.
    tau = balancer.pb.observe().tau_actuated
    obs["tau"] = tau
    obs["zmp_anchor"] = state.zmp_world - state.anchor_world
    obs["zmp_line"] = state.zmp_line
    return np.asarray(obs).astype(np.float32)


def _run_worker(args: tuple) -> tuple[int, int, int, int]:
    """Run a single PyBullet simulation worker (entry point for spawn).

    Args:
        args: Tuple (worker_id, nb_steps, output_path, chunk_duration,
            seed, quiet, robot_name).

    Returns:
        (worker_id, nb_collected, nb_terminated, nb_truncated).
    """
    (
        worker_id,
        nb_steps,
        output_path,
        chunk_duration,
        seed,
        quiet,
        robot_name,
    ) = args
    np.random.seed(seed)
    if quiet:
        logging.getLogger().setLevel(logging.ERROR)

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

    spec = get_spec(robot_name)
    with DataCollectionInterface(
        output_path=output_path,
        nb_steps=nb_steps,
        chunk_duration=chunk_duration,
        frequency=settings.CONTROL_FREQUENCY,
        bar=bar,
    ) as interface:
        balancer = Balancer(spec, controller="mpc")
        balancer.interface = interface
        balancer.pattern._interface = interface
        balancer.dcm_lat._interface = interface
        balancer.pattern.register_interface(interface)
        balancer.dcm_lat.register_interface(interface)

        balancer.reset()
        while not interface.is_complete:
            try:
                state, u, fell = balancer.cycle()
            except NoSolutionFound:
                _mark_last_sample_terminated(interface)
                logger.warning(
                    "[worker %d] IK NoSolutionFound — resetting episode",
                    worker_id,
                )
                balancer.reset()
                continue

            y = _populate_observation(spec, balancer, state)
            interface.collect_data(
                y_k=y,
                u_k=u,
                terminated=fell,
                truncated=False,
                is_ready=True,
            )
            interface.cycle(settings.DT, observation={})
            if fell:
                balancer.reset()
        balancer.close()

    return (
        worker_id,
        interface.nb_collected,
        interface.nb_terminated,
        interface.nb_truncated,
    )


def _merge_files(
    part_paths: list[Path],
    output_path: Path,
    metadata: dict | None = None,
) -> tuple[int, int, int]:
    """Concatenate worker HDF5 files into a single one.

    Datasets concatenated: ``Y``, ``U``, ``amplitude``, ``offset``,
    ``period``, ``terminated``, ``truncated``.

    Args:
        part_paths: HDF5 files written by the workers.
        output_path: Path of the merged HDF5 file.
        metadata: Attributes written to the merged file.

    Returns:
        Number of samples, of terminated samples and of truncated samples.
    """
    with h5py.File(part_paths[0], "r") as src:
        obs_dim = src["Y"].shape[1]

    nb_terminated = 0
    nb_truncated = 0
    with h5py.File(output_path, "w") as out:
        for key, value in (metadata or {}).items():
            out.attrs[key] = value
        Y_out = out.create_dataset(
            "Y", shape=(0, obs_dim), maxshape=(None, obs_dim), dtype=np.float32
        )
        U_out = out.create_dataset("U", shape=(0,), maxshape=(None,), dtype=np.float32)
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
    output: str,
    robot: str,
    chunk_duration: float = 10.0,
    seed: int = 0,
) -> None:
    """Collect a dataset in parallel workers and merge it into one file.

    Args:
        nb_steps: Total number of control steps, rounded up to a whole
            number of chunks per worker.
        num_workers: Number of parallel PyBullet processes.
        output: Path of the output HDF5 file.
        robot: Robot name in the registry.
        chunk_duration: Duration of each ZMP reference, in seconds.
        seed: Base random seed; worker i uses ``seed + i``.
    """
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")

    steps_per_chunk = int(chunk_duration * settings.CONTROL_FREQUENCY)
    raw_per_worker = (nb_steps + num_workers - 1) // num_workers
    steps_per_worker = (
        (raw_per_worker + steps_per_chunk - 1) // steps_per_chunk
    ) * steps_per_chunk
    logger.info(
        "Collecting %d steps/worker x %d workers = %d total (chunk %d steps)",
        steps_per_worker,
        num_workers,
        steps_per_worker * num_workers,
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
                seed,
                False,
                robot,
            )
        )
    else:
        part_paths = [
            output_path.with_name(f"{output_path.stem}.part{i}{output_path.suffix}")
            for i in range(num_workers)
        ]
        worker_args = [
            (
                i,
                steps_per_worker,
                str(part_paths[i]),
                chunk_duration,
                seed + i,
                True,
                robot,
            )
            for i in range(num_workers)
        ]
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            list(executor.map(_run_worker, worker_args))

        nb_collected, nb_terminated, nb_truncated = _merge_files(
            part_paths,
            output_path,
            metadata={
                "dt": 1.0 / settings.CONTROL_FREQUENCY,
                "frequency": settings.CONTROL_FREQUENCY,
                "chunk_duration": chunk_duration,
                "seed": seed,
                "robot": robot,
            },
        )
        # Append robot attribute to single-worker case too.
        for p in part_paths:
            p.unlink()

    # Single-worker case: open and add robot attribute.
    if num_workers == 1:
        with h5py.File(output_path, "a") as hf:
            hf.attrs.setdefault("robot", robot)
            hf.attrs.setdefault("dt", 1.0 / settings.CONTROL_FREQUENCY)

    logger.info(
        "Saved %d steps to %s (terminated: %d, truncated: %d)",
        nb_collected,
        output,
        nb_terminated,
        nb_truncated,
    )


def main() -> None:
    """Collect the training dataset of the DCM observable."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if OUTPUT.exists():
        raise SystemExit(
            f"{OUTPUT} already exists: move it away to collect a new dataset."
        )
    try:
        collect_data(
            nb_steps=NB_STEPS,
            num_workers=NUM_WORKERS,
            output=str(OUTPUT),
            robot=ROBOT,
            chunk_duration=CHUNK_DURATION,
            seed=SEED,
        )
    except KeyboardInterrupt:
        logger.info("Data collection interrupted")


if __name__ == "__main__":
    main()
