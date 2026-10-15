# SPDX-License-Identifier: Apache-2.0

"""Evaluate the DCM training pipeline on a 1-D linear inverted pendulum.

Generates synthetic data from c̈ = ω²(c − z), trains the DCM observable
ξ_θ(y, u history) jointly with ω, then plots training curves.

Usage:
    pixi run train
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lip_eval.figures import save_pyplot_figure
from lip_eval.train_dcm import train_dcm

# ---------------------------------------------------------------------------
# Training run (Appendix A)
# ---------------------------------------------------------------------------

NUM_STEPS: int = 200_000  # closed-loop simulation steps in the dataset
EPOCHS: int = 100
HIST_SIZE: int = 10  # delay-embedding horizon H
OMEGA_INIT: float = 4.0  # rad/s, initial guess for ω
COM_LOSS_WEIGHT: float = 1.0  # w_CoM
SEED: int = 0  # for both data generation and training
PLOT_NAME: str = "lip_eval_losses"

# ---------------------------------------------------------------------------
# LIP model
# ---------------------------------------------------------------------------

OMEGA_TRUE: float = 5.0  # LIP natural frequency sqrt(g/h), h = 0.4 m
DT: float = 1.0 / 200.0  # s
OBS_DIM: int = 6  # y = [cx, cy, cz, ċx, ċy, ċz]
COM_HEIGHT: float = 0.4  # m, constant CoM height


# ---------------------------------------------------------------------------
# LIP dynamics
# ---------------------------------------------------------------------------


def lip_step(
    c: float, cdot: float, z: float, omega: float, dt: float
) -> tuple[float, float]:
    """Exact ZOH integration of LIP dynamics for piecewise-constant z."""
    odt = omega * dt
    d = c - z
    c_next = np.cosh(odt) * d + np.sinh(odt) / omega * cdot + z
    cdot_next = omega * np.sinh(odt) * d + np.cosh(odt) * cdot
    return float(c_next), float(cdot_next)


def dcm_from_state(c: float, cdot: float, omega: float) -> float:
    return c + cdot / omega


def make_obs(c: float, cdot: float) -> np.ndarray:
    """6-D observation [cx, cy, cz, ċx, ċy, ċz]; cy/cz/ċy/ċz are constant."""
    return np.array([c, 0.0, COM_HEIGHT, cdot, 0.0, 0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Sinusoidal ZMP/DCM reference
# ---------------------------------------------------------------------------


def compute_targets(
    t: float, amplitude: float, offset: float, period: float, omega: float
) -> tuple[float, float]:
    """Return (zmp_ref, dcm_ref) for a sine-wave ZMP reference at time t."""
    nu = 2.0 * np.pi / period
    phase = nu * t
    zmp = amplitude * np.sin(phase) + offset
    shift = np.arctan2(nu, omega)
    dcm = amplitude * omega / np.sqrt(omega**2 + nu**2) * np.sin(phase + shift) + offset
    return zmp, dcm


# ---------------------------------------------------------------------------
# DCM controller
# ---------------------------------------------------------------------------


def compute_zmp(
    xi: float,
    xi_ref: float,
    z_ref: float,
    kp: float = 1.0,
    omega: float = OMEGA_TRUE,
) -> float:
    """z = z_ref − (1 + kp/ω)(ξ_ref − ξ), clipped to [−1, 1]."""
    zmp_gain = 1.0 + kp / omega
    return float(np.clip(z_ref - zmp_gain * (xi_ref - xi), -1.0, 1.0))


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


_PERIODS = [
    0.3,
    0.5,
    0.8,
    1.2,
    2.0,
]  # sinusoidal ZMP periods used in curriculum
_AMPLITUDES = [0.1, 0.2, 0.3, 0.4, 0.5]
_OFFSETS = [-0.2, -0.1, 0.0, 0.1, 0.2]
_CHUNK_STEPS = 2_000  # re-draw parameters every N steps


def generate_data(
    num_steps: int,
    kp: float = 1.0,
    fall_threshold: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the 1-D LIP under a varied sinusoidal curriculum.

    Parameters are re-drawn from _PERIODS / _AMPLITUDES / _OFFSETS every
    _CHUNK_STEPS steps.  Varied frequencies are required for ω to be
    identifiable: with a single frequency there are approximate local minima
    for any (learned DCM ≈ c + ċ/ω′, ω̂ = ω′).

    Returns:
        Y: Observation array of shape (num_steps, OBS_DIM).
        U: ZMP command array of shape (num_steps,).
    """
    rng = np.random.default_rng(seed)
    c, cdot, t = 0.0, 0.0, 0.0
    Y = np.empty((num_steps, OBS_DIM), dtype=np.float32)
    U = np.empty(num_steps, dtype=np.float32)
    n_resets = 0
    period = float(rng.choice(_PERIODS))
    amplitude = float(rng.choice(_AMPLITUDES))
    offset = float(rng.choice(_OFFSETS))

    for k in range(num_steps):
        if k % _CHUNK_STEPS == 0 and k > 0:
            period = float(rng.choice(_PERIODS))
            amplitude = float(rng.choice(_AMPLITUDES))
            offset = float(rng.choice(_OFFSETS))
            t = 0.0

        z_ref, xi_ref = compute_targets(t, amplitude, offset, period, OMEGA_TRUE)
        xi = dcm_from_state(c, cdot, OMEGA_TRUE)
        z = compute_zmp(xi, xi_ref, z_ref, kp)

        Y[k] = make_obs(c, cdot)
        U[k] = np.float32(z)

        c, cdot = lip_step(c, cdot, z, OMEGA_TRUE, DT)
        t += DT

        if abs(c) > fall_threshold:
            c, cdot, t = 0.0, 0.0, 0.0
            n_resets += 1

    print(f"  {num_steps:,} steps — {n_resets} resets")
    return Y, U


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------


def plot_losses(
    train_losses: list[float],
    val_losses: list[float],
    omegas: list[float],
    plot_name: str,
    val_dcm_losses: list[float] | None = None,
    val_com_losses: list[float] | None = None,
) -> None:
    """Plot per-epoch train/val losses (left) and ω (right axis).

    If ``val_dcm_losses`` and ``val_com_losses`` are provided, overlays
    the unweighted DCM and CoM validation components so the reader can
    see which term dominates and when each crosses below the other.
    """
    epochs_x = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    handles = []
    (l1,) = ax.plot(epochs_x, train_losses, lw=1.8, label="Train (total)")
    (l2,) = ax.plot(epochs_x, val_losses, lw=1.8, label="Val (total)")
    handles.extend([l1, l2])
    if val_dcm_losses is not None:
        (ldcm,) = ax.plot(
            epochs_x,
            val_dcm_losses,
            lw=1.2,
            ls="--",
            color="tab:red",
            label="Val DCM",
        )
        handles.append(ldcm)
    if val_com_losses is not None:
        (lcom,) = ax.plot(
            epochs_x,
            val_com_losses,
            lw=1.2,
            ls="--",
            color="tab:purple",
            label="Val CoM",
        )
        handles.append(lcom)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("DCM training curves on the LIP")
    ax.grid(True, which="both", alpha=0.3)
    ax2 = ax.twinx()
    (l3,) = ax2.plot(epochs_x, omegas, lw=1.5, color="tab:green", label="ω (rad/s)")
    ax2.axhline(OMEGA_TRUE, color="tab:green", lw=0.8, ls="--", alpha=0.6)
    ax2.set_ylabel("ω (rad/s)")
    handles.append(l3)
    ax.legend(handles=handles, loc="best")
    plt.tight_layout()
    save_pyplot_figure(plot_name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print("=== Data generation ===")
    Y, U = generate_data(NUM_STEPS, seed=SEED)

    print("\n=== Training ===")
    (
        _,
        omega_learned,
        train_losses,
        val_losses,
        omegas,
        val_dcm_losses,
        val_com_losses,
    ) = train_dcm(
        Y,
        U,
        dt=DT,
        hist_size=HIST_SIZE,
        omega_init=OMEGA_INIT,
        epochs=EPOCHS,
        com_loss_weight=COM_LOSS_WEIGHT,
        seed=SEED,
    )
    plot_losses(
        train_losses,
        val_losses,
        omegas,
        plot_name=PLOT_NAME,
        val_dcm_losses=val_dcm_losses,
        val_com_losses=val_com_losses,
    )

    print("\n=== Summary ===")
    print(f"  Learned ω = {omega_learned:.4f} rad/s")
    print(f"  True    ω = {OMEGA_TRUE:.4f} rad/s")
    print(
        f"  Error     = {abs(omega_learned - OMEGA_TRUE):.4f} rad/s"
        f"  ({100 * abs(omega_learned - OMEGA_TRUE) / OMEGA_TRUE:.1f}%)"
    )
    print(f"  Final val DCM loss = {val_dcm_losses[-1]:.4e}")
    print(f"  Final val CoM loss = {val_com_losses[-1]:.4e}")


if __name__ == "__main__":
    main()
