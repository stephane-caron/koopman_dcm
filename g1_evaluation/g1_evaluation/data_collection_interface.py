# SPDX-License-Identifier: Apache-2.0

"""Interface for autonomous data collection."""

import logging
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np

from .interface import Interface

logger = logging.getLogger("g1_evaluation.data_collection_interface")

# Amplitudes of normalized-ZMP oscillations
AMPLITUDES = [0.0, 0.1, 0.2, 0.4, 0.8]  # in [0, 1]

# Stationary offset of normalized-ZMP trajectories
OFFSETS = [-0.5, -0.2, 0.0, 0.2, 0.5]  # in [-1, 1]

# Periods of normalized-ZMP oscillations
PERIODS = [0.3, 0.7, 1.0, 1.2, 1.4]  # s

# Warmup is a short phase  of zero amplitude and offset after is_ready
WARMUP_DURATION = 3.0  # s

_FLUSH_INTERVAL = 10_000


class DataCollectionInterface(Interface):
    """Interface for autonomous data collection.

    The interface samples trajectory parameters from fixed sets. It splits the
    run into fixed-length time chunks. At each chunk boundary and on episode
    reset, it draws new amplitude, offset, and period values uniformly at
    random from AMPLITUDES, OFFSETS, PERIODS.

    Use as a context manager to manage the HDF5 file lifecycle::

        with DataCollectionInterface(output_path, nb_steps, ...) as iface:
            ...
    """

    def __init__(
        self,
        output_path: str | Path,
        nb_steps: int,
        chunk_duration: float,
        frequency: float,
        warmup_duration: float = WARMUP_DURATION,
        bar: Optional[object] = None,
    ) -> None:
        """Initialize the data collection interface.

        Args:
            output_path: Path of the HDF5 output file. Opened on context
                manager entry, not here.
            nb_steps: Number of samples to collect (counting only steps
                where ``leg_controller.is_ready`` is True).
            chunk_duration: Duration of each fixed-parameter chunk, in
                seconds.
            frequency: Control loop frequency, in Hz.
            warmup_duration: Seconds to hold amplitude and offset at zero
                after ``is_ready`` flips True, giving the robot time to
                stabilize before exciting it with a pattern.
            bar: Optional tqdm progress bar. If provided, the interface
                calls ``bar.update(1)`` on each collected sample and
                ``bar.close()`` on exit. If absent, progress and ``is_ready``
                transitions are logged as info messages.
        """
        super().__init__()
        self._output_path = Path(output_path)
        self._nb_steps = nb_steps
        self._steps_per_chunk = int(chunk_duration * frequency)
        self._dt = 1.0 / frequency
        self._chunk_duration = chunk_duration
        self._frequency = frequency
        self._warmup_steps = int(warmup_duration * frequency)

        self._chunk_amplitude = 0.0
        self._chunk_offset = 0.0
        self._chunk_period = 1.2
        self._chunk_step = 0
        self._warmup_remaining: int = 0
        self._sample_chunk()

        # HDF5 state
        self._hf: Optional[h5py.File] = None
        self._Y_ds: Optional[h5py.Dataset] = None
        self._U_ds: Optional[h5py.Dataset] = None
        self._A_ds: Optional[h5py.Dataset] = None
        self._O_ds: Optional[h5py.Dataset] = None
        self._P_ds: Optional[h5py.Dataset] = None
        self._T_ds: Optional[h5py.Dataset] = None
        self._R_ds: Optional[h5py.Dataset] = None

        self._Y_buf: Optional[np.ndarray] = None
        self._U_buf: Optional[np.ndarray] = None
        self._A_buf: Optional[np.ndarray] = None
        self._O_buf: Optional[np.ndarray] = None
        self._P_buf: Optional[np.ndarray] = None
        self._T_buf: Optional[np.ndarray] = None
        self._R_buf: Optional[np.ndarray] = None
        self._buf_idx: int = 0

        self._nb_collected: int = 0
        self._nb_terminated: int = 0
        self._nb_truncated: int = 0

        self._bar: Optional[object] = bar
        self._prev_is_ready: bool = True

    def __enter__(self) -> "DataCollectionInterface":
        self._hf = h5py.File(self._output_path, "w")
        self._hf.attrs["dt"] = self._dt
        self._hf.attrs["frequency"] = self._frequency
        self._hf.attrs["chunk_duration"] = self._chunk_duration

        self._Y_ds = None  # created on first collect_data call
        self._U_ds = self._hf.create_dataset(
            "U", shape=(0,), maxshape=(None,), dtype=np.float32
        )
        self._A_ds = self._hf.create_dataset(
            "amplitude", shape=(0,), maxshape=(None,), dtype=np.float32
        )
        self._O_ds = self._hf.create_dataset(
            "offset", shape=(0,), maxshape=(None,), dtype=np.float32
        )
        self._P_ds = self._hf.create_dataset(
            "period", shape=(0,), maxshape=(None,), dtype=np.float32
        )
        self._T_ds = self._hf.create_dataset(
            "terminated", shape=(0,), maxshape=(None,), dtype=bool
        )
        self._R_ds = self._hf.create_dataset(
            "truncated", shape=(0,), maxshape=(None,), dtype=bool
        )

        self._U_buf = np.empty(_FLUSH_INTERVAL, dtype=np.float32)
        self._A_buf = np.empty(_FLUSH_INTERVAL, dtype=np.float32)
        self._O_buf = np.empty(_FLUSH_INTERVAL, dtype=np.float32)
        self._P_buf = np.empty(_FLUSH_INTERVAL, dtype=np.float32)
        self._T_buf = np.empty(_FLUSH_INTERVAL, dtype=bool)
        self._R_buf = np.empty(_FLUSH_INTERVAL, dtype=bool)

        return self

    def __exit__(self, *_) -> None:
        if self._buf_idx > 0 and self._Y_ds is not None:
            self._flush_buffers()
        if self._bar is not None:
            self._bar.close()
        if self._hf is not None:
            self._hf.close()

    def _flush_buffers(self) -> None:
        """Append in-memory buffers to the HDF5 datasets."""
        nb_rows = self._buf_idx
        datasets: List[h5py.Dataset] = [
            self._Y_ds,
            self._U_ds,
            self._A_ds,
            self._O_ds,
            self._P_ds,
            self._T_ds,
            self._R_ds,
        ]
        buffers = [
            self._Y_buf,
            self._U_buf,
            self._A_buf,
            self._O_buf,
            self._P_buf,
            self._T_buf,
            self._R_buf,
        ]
        base = datasets[0].shape[0]
        for ds, buf in zip(datasets, buffers):
            ds.resize(base + nb_rows, axis=0)
            ds[base : base + nb_rows] = buf[:nb_rows]
        self._hf.flush()

    def collect_data(
        self,
        y_k: np.ndarray,
        u_k: float,
        terminated: bool,
        truncated: bool,
        is_ready: bool,
    ) -> None:
        """Record one transition sample into the HDF5 file.

        Args:
            y_k: State observation vector at step k.
            u_k: ZMP line control action at step k.
            terminated: Whether the episode terminated this step.
            truncated: Whether the episode was truncated this step.
            is_ready: Whether the leg controller has left its init phase.
        """
        if self._bar is None:  # no tqdm bar, so we log info messages
            if self._prev_is_ready and not is_ready:
                logger.info(
                    "[%s] Waiting for initialization to complete...",
                    self.__class__.__name__,
                )
            elif not self._prev_is_ready and is_ready:
                logger.info(
                    "[%s] Warming up for %.1f s before exciting pattern",
                    self.__class__.__name__,
                    self._warmup_steps / self._frequency,
                )
        if not self._prev_is_ready and is_ready:
            self._warmup_remaining = self._warmup_steps
        self._prev_is_ready = is_ready

        if not is_ready or self._nb_collected >= self._nb_steps:
            return

        truncated_step = truncated or (
            self._nb_collected == self._nb_steps - 1
        )

        if self._Y_ds is None:
            obs_dim = y_k.shape[0]
            self._Y_ds = self._hf.create_dataset(
                "Y",
                shape=(0, obs_dim),
                maxshape=(None, obs_dim),
                dtype=np.float32,
            )
            self._Y_buf = np.empty(
                (_FLUSH_INTERVAL, obs_dim), dtype=np.float32
            )

        self._Y_buf[self._buf_idx] = y_k
        self._U_buf[self._buf_idx] = u_k
        self._A_buf[self._buf_idx] = self._chunk_amplitude
        self._O_buf[self._buf_idx] = self._chunk_offset
        self._P_buf[self._buf_idx] = self._chunk_period
        self._T_buf[self._buf_idx] = bool(terminated)
        self._R_buf[self._buf_idx] = truncated_step
        self._buf_idx += 1
        self._nb_terminated += int(terminated)
        self._nb_truncated += int(truncated_step)
        self._nb_collected += 1
        if self._bar is not None:
            self._bar.update(1)
        elif self._nb_collected % _FLUSH_INTERVAL == 0:
            logger.info(
                "[%s] %d / %d steps collected",
                self.__class__.__name__,
                self._nb_collected,
                self._nb_steps,
            )

        if self._buf_idx == _FLUSH_INTERVAL:
            self._flush_buffers()
            self._buf_idx = 0

    @property
    def is_complete(self) -> bool:
        """True once the target number of samples has been collected."""
        return self._nb_collected >= self._nb_steps

    @property
    def nb_collected(self) -> int:
        """Number of samples collected so far."""
        return self._nb_collected

    @property
    def nb_terminated(self) -> int:
        """Number of collected steps that ended with termination."""
        return self._nb_terminated

    @property
    def nb_truncated(self) -> int:
        """Number of collected steps that ended with truncation."""
        return self._nb_truncated

    def _sample_chunk(self) -> None:
        self._chunk_amplitude = float(np.random.choice(AMPLITUDES))
        self._chunk_offset = float(np.random.choice(OFFSETS))
        self._chunk_period = float(np.random.choice(PERIODS))
        self._chunk_step = 0
        logger.info(
            "[%s] Chunk: amplitude=%.1f offset=%.1f period=%.1f",
            self.__class__.__name__,
            self._chunk_amplitude,
            self._chunk_offset,
            self._chunk_period,
        )

    def reset(self) -> None:
        """Reset the step counter and resample chunk parameters."""
        super().reset()
        self._prev_is_ready = True
        self._warmup_remaining = 0
        self._sample_chunk()

    def get_slider(self, name: str) -> float:
        """Return the current slider value.

        Args:
            name: Slider name.

        Returns:
            Current chunk value for pattern sliders; base value for others.
            Amplitude and offset are clamped to zero once collection is
            complete so the robot idles quietly during the cooldown phase.
        """
        if self.is_complete:
            if name == "amplitude":
                return 0.0
            if name == "offset":
                return 0.0
        in_warmup = not self._prev_is_ready or self._warmup_remaining > 0
        if name == "amplitude":
            return 0.0 if in_warmup else self._chunk_amplitude
        if name == "offset":
            return 0.0 if in_warmup else self._chunk_offset
        if name == "period":
            return self._chunk_period
        return super().get_slider(name)

    def cycle(self, dt: float, observation: dict) -> None:
        """Advance step counter and resample at chunk boundaries.

        Args:
            dt: Time step in seconds.
            observation: Observation dictionary (unused).
        """
        if self._prev_is_ready and not self.is_complete:
            if self._warmup_remaining > 0:
                self._warmup_remaining -= 1
            else:  # self._warmup_remaining == 0
                self._chunk_step += 1
                ready_to_sample = self._chunk_step % self._steps_per_chunk == 0
                almost_over = self.nb_collected > self._nb_steps - 42
                if ready_to_sample and not almost_over:
                    self._sample_chunk()
