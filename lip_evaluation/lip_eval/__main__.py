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
from lip_eval.generate_data import DT, OMEGA_TRUE, generate_data
from lip_eval.train_dcm import train_dcm

NUM_STEPS: int = 200_000  # closed-loop simulation steps in the dataset
EPOCHS: int = 100
HIST_SIZE: int = 10  # delay-embedding horizon H
OMEGA_INIT: float = 4.0  # rad/s, initial guess for ω
COM_LOSS_WEIGHT: float = 1.0  # w_CoM
SEED: int = 0  # for both data generation and training
PLOT_NAME: str = "lip_eval_losses"


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

    Args:
        train_losses: Training loss at each epoch.
        val_losses: Validation loss at each epoch.
        omegas: Learned ω at each epoch, in rad/s.
        plot_name: Name of the saved figure.
        val_dcm_losses: Unweighted DCM validation loss at each epoch.
        val_com_losses: Unweighted CoM validation loss at each epoch.
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


def main() -> None:
    """Run the training of Appendix A and print a summary."""
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
