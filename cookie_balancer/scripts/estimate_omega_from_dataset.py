# SPDX-License-Identifier: Apache-2.0

"""Fit the natural frequency of the learned-LIP ablation.

Holds the observable at the analytic form ξ = c + ċ/ω and fits ω alone,
under the same losses and hyperparameters as the learned DCM. Prints
ω̃ over five seeds; the mean goes to ``settings.LEARNED_LIP_OMEGA``.

Usage::

    pixi run python scripts/estimate_omega_from_dataset.py

Runs on CPU in about ten minutes.
"""

from typing import Tuple

import h5py
import numpy as np
import torch
from torch import nn
from upkie.logging import logger

from cookie_balancer import settings
from cookie_balancer.observation import make_observation
from cookie_balancer.train_dcm import (
    build_valid_segment_starts,
    default_nb_horizon_steps,
    lambda_from_nb_lambda_steps,
    run_epoch,
    split_indices,
)

DATASET = settings.DATA_DIR / "cookie.h5"
SEEDS = (0, 1, 2, 3, 4)

# Same values as the paper's DCM observable (H = 10), so that the two fits
# differ only in the observable.
HIST_SIZE = 10
OMEGA_INIT = 5.0
EPOCHS = 100
BATCH_SIZE = 1024
LEARNING_RATE = 3e-4
VAL_FRAC = 0.1
COM_LOSS_WEIGHT = 1.0
NB_LAMBDA_STEPS = 10


class LearnedLipObservable(nn.Module):
    """Analytic observable ξ = c + ċ/ω, with ω the only parameter.

    Drop-in replacement for ``train_dcm.DcmObservable`` in
    ``run_epoch``, consuming rows ``(c_k, ċ_k)``. Its ``log_omega`` is
    passed to ``run_epoch`` as well, which forms α = exp(−ω·δt) from
    it: ω therefore enters both the observable and the recursions, and a
    single scalar has to explain both.
    """

    def __init__(self, omega_init: float) -> None:
        """Initialize the observable.

        Args:
            omega_init: Initial natural frequency, in rad/s.
        """
        super().__init__()
        self.log_omega = nn.Parameter(torch.tensor(float(np.log(omega_init))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the DCM.

        Args:
            x: Rows ``(c_k, ċ_k)``, of shape ``(..., 2)``.

        Returns:
            DCM values, of shape ``(...)``.
        """
        return x[..., 0] + x[..., 1] / torch.exp(self.log_omega)


def load_dataset() -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, float]:
    """Load the dataset and assemble ``(c, ċ)`` rows.

    Rows are aligned exactly as ``train_dcm.build_histories``
    does, i.e. ``M = N − HIST_SIZE`` rows with row ``j`` anchored at step
    ``HIST_SIZE + j``, so the segment bookkeeping and the train/validation
    split of the neural pipeline apply unchanged.

    Returns:
        ``(samples, U_aligned, valid_starts, dt)``, with ``samples`` of
        shape ``(M, 2)`` holding ``(c, ċ)`` and ``U_aligned[j]`` the action
        driving row ``j`` to row ``j+1``.
    """
    with h5py.File(DATASET, "r") as hf:
        Y = hf["Y"][:].astype(np.float32)
        U = hf["U"][:].astype(np.float32)
        terminated = hf["terminated"][:].astype(bool)
        truncated = hf["truncated"][:].astype(bool)
        dt = float(hf.attrs["dt"])

    sv = make_observation()
    if Y.shape[1] != sv.dim:
        raise ValueError(
            f"obs_dim={Y.shape[1]} does not match the observation schema "
            f"(expected {sv.dim})"
        )
    c = Y[:, sv.slice_of("com_line").start].astype(np.float64)

    M = c.shape[0] - HIST_SIZE
    anchors = np.arange(HIST_SIZE, HIST_SIZE + M)
    samples = np.empty((M, 2), dtype=np.float32)
    samples[:, 0] = c[anchors]
    samples[:, 1] = (c[anchors] - c[anchors - 1]) / dt  # backward difference

    nb_horizon_steps = default_nb_horizon_steps(NB_LAMBDA_STEPS)
    valid_starts = build_valid_segment_starts(
        M, HIST_SIZE, nb_horizon_steps, terminated, truncated
    )
    # com_line jumps by up to 3.8 at episode resets, so a backward
    # difference straddling one would wreck the fit. Row j reaches back to
    # step HIST_SIZE + j − 1, which lies inside the flag window that
    # build_valid_segment_starts already checks; assert it rather than
    # trust the index arithmetic.
    flags = terminated | truncated
    if flags[valid_starts + HIST_SIZE - 1].any():
        raise AssertionError("backward difference crosses an episode boundary")

    logger.info(
        "Loaded %d steps from %s, %d valid segments",
        c.shape[0],
        DATASET.name,
        valid_starts.shape[0],
    )
    return (
        torch.from_numpy(samples),
        torch.from_numpy(U[anchors].astype(np.float32)),
        valid_starts,
        dt,
    )


def fit(
    samples: torch.Tensor,
    U_aligned: torch.Tensor,
    valid_starts: np.ndarray,
    dt: float,
    seed: int,
) -> float:
    """Fit ω for one seed, returning its value at the best-validation epoch.

    Args:
        samples: Rows ``(c_k, ċ_k)``, from :func:`load_dataset`.
        U_aligned: Actions aligned with the rows.
        valid_starts: Start indices of valid segments.
        dt: Time step of the dataset, in seconds.
        seed: Random seed of the split and initialization.

    Returns:
        Value of ω at the epoch with the lowest validation loss, in rad/s.
    """
    torch.manual_seed(seed)
    lambda_decay = lambda_from_nb_lambda_steps(NB_LAMBDA_STEPS)
    nb_horizon_steps = default_nb_horizon_steps(NB_LAMBDA_STEPS)
    train_idx, val_idx = split_indices(valid_starts.shape[0], VAL_FRAC, seed)
    train_starts = torch.from_numpy(valid_starts[train_idx])
    val_starts = torch.from_numpy(valid_starts[val_idx])

    observable = LearnedLipObservable(OMEGA_INIT)
    opt = torch.optim.Adam(observable.parameters(), lr=LEARNING_RATE)

    best_val, best_omega = float("inf"), OMEGA_INIT
    for epoch in range(EPOCHS):
        # Column 0 of a row is c, which run_epoch reads for the CoM loss.
        head = (observable, samples, U_aligned, 0)
        tail = (
            observable.log_omega,
            dt,
            COM_LOSS_WEIGHT,
            lambda_decay,
            nb_horizon_steps,
            BATCH_SIZE,
        )
        run_epoch(*head, train_starts, *tail, opt, shuffle=True)
        val_loss, _, _ = run_epoch(
            *head, val_starts, *tail, None, shuffle=False
        )
        omega = float(torch.exp(observable.log_omega).item())
        if val_loss < best_val:
            best_val, best_omega = val_loss, omega
        if (epoch + 1) % 25 == 0:
            logger.info(
                "  seed %d, epoch %3d/%d: val=%.3e, ω=%.4f rad/s",
                seed,
                epoch + 1,
                EPOCHS,
                val_loss,
                omega,
            )
    return best_omega


def main() -> None:
    """Fit ω̃ over all seeds and log the results."""
    samples, U_aligned, valid_starts, dt = load_dataset()
    omegas = [
        fit(samples, U_aligned, valid_starts, dt, seed) for seed in SEEDS
    ]
    for seed, omega in zip(SEEDS, omegas):
        logger.info("seed %d: ω = %.4f rad/s", seed, omega)
    logger.info(
        "ω̃ = %.4f ± %.4f rad/s over %d seeds",
        float(np.mean(omegas)),
        float(np.std(omegas)),
        len(SEEDS),
    )


if __name__ == "__main__":
    main()
