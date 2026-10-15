# SPDX-License-Identifier: Apache-2.0

"""Sweep CoM loss weights, plot validation loss and convergence of ω.

For each weight in COM_LOSS_WEIGHTS, runs --samples training runs sharing
the same data seed (so the only variable is the weight), records per-epoch
validation loss and ω, and overlays mean ± individual curves on two subplots.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lip_eval.__main__ import OMEGA_TRUE, generate_data, train
from lip_eval.figures import save_pyplot_figure

COM_LOSS_WEIGHTS: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)


def _lighten(color: tuple[float, float, float, float], alpha: float):
    r, g, b, _ = color
    return (r, g, b, alpha)


def plot_sweep(
    val_curves: dict,
    omega_curves: dict,
    plot_name: str,
) -> None:
    """Two-subplot figure: validation loss (left) and ω convergence (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("tab10")

    for i, w in enumerate(COM_LOSS_WEIGHTS):
        base = cmap(i)
        light = _lighten(base, 0.3)
        val_runs = np.array(val_curves[w])  # (samples, epochs)
        omega_runs = np.array(omega_curves[w])
        epochs_x = np.arange(1, val_runs.shape[1] + 1)
        label = f"α={w:g}"
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
    ax1.legend(fontsize=10, title="CoM loss weight α")

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
    ax2.legend(fontsize=10, title="CoM loss weight α")

    plt.tight_layout()
    save_pyplot_figure(plot_name)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CoM loss weight sweep on the LIP encoder pipeline"
    )
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--hist-size", type=int, default=10)
    parser.add_argument(
        "--omega-init",
        type=float,
        default=1.0,
        help="Initial ω guess (default: 1.0 rad/s, true = 5.0 rad/s)",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--plot-name",
        type=str,
        default="lip_eval_com_sweep",
    )
    args = parser.parse_args()

    val_curves: dict = {w: [] for w in COM_LOSS_WEIGHTS}
    omega_curves: dict = {w: [] for w in COM_LOSS_WEIGHTS}

    # Each run s uses the same data for all weights
    for s in range(args.samples):
        print(f"\n=== Run {s + 1}/{args.samples} (seed={s}) ===")
        Y, U = generate_data(args.steps, seed=s)
        for w in COM_LOSS_WEIGHTS:
            print(f"  α={w:g}")
            _, _, _, val_l, omegas_l, _, _ = train(
                Y,
                U,
                hist_size=args.hist_size,
                omega_init=args.omega_init,
                com_loss_weight=w,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                val_frac=0.1,
                seed=s,
            )
            val_curves[w].append(val_l)
            omega_curves[w].append(omegas_l)
            print(f"    final val={val_l[-1]:.3e}  ω={omegas_l[-1]:.4f} rad/s")

    plot_sweep(val_curves, omega_curves, plot_name=args.plot_name)
    plot_summary(val_curves, omega_curves, plot_name=f"{args.plot_name}_summary")
    print_summary(val_curves, omega_curves)


def plot_summary(
    val_curves: dict,
    omega_curves: dict,
    plot_name: str,
) -> None:
    """Plot final ω and final val loss vs CoM loss weight (mean ± std)."""
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
    ax1.set_xlabel("CoM loss weight  $w_\\mathrm{CoM}$")
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
    ax2.set_xlabel("CoM loss weight  $w_\\mathrm{CoM}$")
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
    """Print per-weight mean ± std of final val loss and ω."""
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


if __name__ == "__main__":
    main()
