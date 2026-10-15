# SPDX-License-Identifier: Apache-2.0

"""Sweep CoM loss weights, plot validation loss and convergence of ω.

For each weight in COM_LOSS_WEIGHTS, runs NUM_SAMPLES training runs sharing
the same data seed (so the only variable is the weight), records per-epoch
validation loss and ω, and overlays mean ± individual curves on two subplots.

Usage:
    pixi run com-loss-sweep
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lip_eval.figures import save_pyplot_figure
from lip_eval.generate_data import DT, OMEGA_TRUE, generate_data
from lip_eval.train_dcm import train_dcm

COM_LOSS_WEIGHTS: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)  # w_CoM
NUM_SAMPLES: int = 3  # seeds per weight
NUM_STEPS: int = 50_000  # closed-loop simulation steps per dataset
EPOCHS: int = 50
HIST_SIZE: int = 10  # delay-embedding horizon H
OMEGA_INIT: float = 1.0  # rad/s, initial guess for ω, close to ω = 0
PLOT_NAME: str = "lip_eval_com_sweep"

W_COM: str = "$w_\\mathrm{CoM}$"  # label of the CoM loss weight in figures


def _lighten(color: tuple[float, float, float, float], alpha: float):
    """Replace the alpha channel of an RGBA color.

    Args:
        color: RGBA color.
        alpha: New alpha value.

    Returns:
        RGBA color with the new alpha value.
    """
    r, g, b, _ = color
    return (r, g, b, alpha)


def plot_sweep(
    val_curves: dict,
    omega_curves: dict,
    plot_name: str,
) -> None:
    """Two-subplot figure: validation loss (left) and ω convergence (right).

    Args:
        val_curves: Per-epoch validation losses of each run, by weight.
        omega_curves: Per-epoch learned ω of each run, by weight.
        plot_name: Name of the saved figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("tab10")

    for i, w in enumerate(COM_LOSS_WEIGHTS):
        base = cmap(i)
        light = _lighten(base, 0.3)
        val_runs = np.array(val_curves[w])  # (samples, epochs)
        omega_runs = np.array(omega_curves[w])
        epochs_x = np.arange(1, val_runs.shape[1] + 1)
        label = f"{W_COM}={w:g}"
        for run in val_runs:
            ax1.plot(epochs_x, run, color=light, lw=0.8)
        ax1.plot(epochs_x, val_runs.mean(axis=0), color=base, lw=2.5, label=label)
        for run in omega_runs:
            ax2.plot(epochs_x, run, color=light, lw=0.8)
        ax2.plot(epochs_x, omega_runs.mean(axis=0), color=base, lw=2.5, label=label)

    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation loss (MSE)")
    ax1.set_title("Validation loss vs epoch — CoM loss weight sweep")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=10, title=f"CoM loss weight {W_COM}")

    ax2.axhline(
        OMEGA_TRUE,
        color="k",
        lw=0.8,
        ls="--",
        alpha=0.6,
        label=f"ω_true = {OMEGA_TRUE}",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("ω (rad/s)")
    ax2.set_title("ω convergence — CoM loss weight sweep")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=10, title=f"CoM loss weight {W_COM}")

    plt.tight_layout()
    save_pyplot_figure(plot_name)
    plt.close(fig)


def plot_summary(
    val_curves: dict,
    omega_curves: dict,
    plot_name: str,
) -> None:
    """Plot final ω and final val loss vs CoM loss weight (mean ± std).

    Args:
        val_curves: Per-epoch validation losses of each run, by weight.
        omega_curves: Per-epoch learned ω of each run, by weight.
        plot_name: Name of the saved figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ws = np.array(COM_LOSS_WEIGHTS)
    om_finals = np.array(
        [[runs[-1] for runs in omega_curves[w]] for w in COM_LOSS_WEIGHTS]
    )
    val_finals = np.array(
        [[runs[-1] for runs in val_curves[w]] for w in COM_LOSS_WEIGHTS]
    )

    ax1.errorbar(
        ws,
        om_finals.mean(axis=1),
        yerr=om_finals.std(axis=1),
        fmt="o-",
        lw=1.8,
        capsize=4,
        color="tab:blue",
    )
    ax1.axhline(
        OMEGA_TRUE,
        color="k",
        lw=0.8,
        ls="--",
        alpha=0.6,
        label=f"ω_true = {OMEGA_TRUE}",
    )
    ax1.set_xscale("log")
    ax1.set_xlabel(f"CoM loss weight  {W_COM}")
    ax1.set_ylabel("Final ω (rad/s)")
    ax1.set_title("Final ω vs CoM weight  (mean ± std)")
    ax1.set_xticks(ws)
    ax1.set_xticklabels([f"{w:g}" for w in ws])
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.errorbar(
        ws,
        val_finals.mean(axis=1),
        yerr=val_finals.std(axis=1),
        fmt="o-",
        lw=1.8,
        capsize=4,
        color="tab:orange",
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(f"CoM loss weight  {W_COM}")
    ax2.set_ylabel("Final validation loss (MSE)")
    ax2.set_title("Final val loss vs CoM weight  (mean ± std)")
    ax2.set_xticks(ws)
    ax2.set_xticklabels([f"{w:g}" for w in ws])
    ax2.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    save_pyplot_figure(plot_name)
    plt.close(fig)


def print_summary(
    val_curves: dict,
    omega_curves: dict,
) -> None:
    """Print per-weight mean ± std of final val loss and ω.

    Args:
        val_curves: Per-epoch validation losses of each run, by weight.
        omega_curves: Per-epoch learned ω of each run, by weight.
    """
    print("\n=== CoM weight sweep summary (mean ± std over samples) ===")
    print(f"  {'w_CoM':>6s}  {'val_loss':>22s}  {'ω (rad/s)':>20s}")
    for w in COM_LOSS_WEIGHTS:
        va = np.array([r[-1] for r in val_curves[w]])
        om = np.array([r[-1] for r in omega_curves[w]])
        print(
            f"  {w:>6g}  "
            f"{va.mean():>10.3e} ± {va.std():>8.3e}  "
            f"{om.mean():>8.4f} ± {om.std():>8.4f}"
        )


def main() -> None:
    """Run the CoM loss weight sweep of Appendix A.1."""
    val_curves: dict = {w: [] for w in COM_LOSS_WEIGHTS}
    omega_curves: dict = {w: [] for w in COM_LOSS_WEIGHTS}

    # Each run s uses the same data for all weights
    for s in range(NUM_SAMPLES):
        print(f"\n=== Run {s + 1}/{NUM_SAMPLES} (seed={s}) ===")
        Y, U = generate_data(NUM_STEPS, seed=s)
        for w in COM_LOSS_WEIGHTS:
            print(f"  w_CoM={w:g}")
            _, _, _, val_l, omegas_l, _, _ = train_dcm(
                Y,
                U,
                dt=DT,
                hist_size=HIST_SIZE,
                omega_init=OMEGA_INIT,
                epochs=EPOCHS,
                com_loss_weight=w,
                seed=s,
            )
            val_curves[w].append(val_l)
            omega_curves[w].append(omegas_l)
            print(f"    final val={val_l[-1]:.3e}  ω={omegas_l[-1]:.4f} rad/s")

    plot_sweep(val_curves, omega_curves, plot_name=PLOT_NAME)
    plot_summary(val_curves, omega_curves, plot_name=f"{PLOT_NAME}_summary")
    print_summary(val_curves, omega_curves)


if __name__ == "__main__":
    main()
