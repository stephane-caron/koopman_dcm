# SPDX-License-Identifier: Apache-2.0

"""Sweep the encoder history size and print final losses and ω.

For each history size in {1, 3, 5, 10, 20}, runs N training samples
(different seeds) and prints the mean ± std of the final train/val losses
and ω over samples.

Training delegates to ``lip_eval.__main__.train`` so the pipeline
matches the main LIP evaluation script exactly --- DCM dynamics loss +
CoM prediction loss, ω jointly learned as exp(log_ω).

Usage:
    pixi run python -m lip_eval.history_sweep \\
        --steps 50000 --epochs 50 --samples 10
"""

import argparse

import numpy as np

from lip_eval.__main__ import generate_data, train

HISTORY_SIZES: tuple[int, ...] = (1, 3, 5, 10, 20)


def print_summary(
    train_curves: dict,
    val_curves: dict,
    omega_curves: dict,
) -> None:
    """Print per-history-size mean ± std of final train/val loss and ω."""
    print("\n=== History sweep summary (mean ± std over samples) ===")
    print(f"  {'h':>3s}  {'train_loss':>22s}  {'val_loss':>22s}  {'ω (rad/s)':>20s}")
    for h in HISTORY_SIZES:
        tr = np.array([r[-1] for r in train_curves[h]])
        va = np.array([r[-1] for r in val_curves[h]])
        om = np.array([r[-1] for r in omega_curves[h]])
        print(
            f"  {h:>3d}  "
            f"{tr.mean():>10.3e} ± {tr.std():>8.3e}  "
            f"{va.mean():>10.3e} ± {va.std():>8.3e}  "
            f"{om.mean():>8.4f} ± {om.std():>8.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="History-size sweep on the LIP encoder pipeline"
    )
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--omega-init", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_curves: dict = {h: [] for h in HISTORY_SIZES}
    val_curves: dict = {h: [] for h in HISTORY_SIZES}
    omega_curves: dict = {h: [] for h in HISTORY_SIZES}

    for h in HISTORY_SIZES:
        print(f"\n=== History size {h} ===")
        for s in range(args.samples):
            seed = 1000 * h + s
            print(f"  sample {s + 1}/{args.samples}  (seed={seed})")
            Y, U = generate_data(args.steps, seed=seed)
            _, _, tr, va, om, _, _ = train(
                Y,
                U,
                hist_size=h,
                omega_init=args.omega_init,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=seed,
            )
            train_curves[h].append(tr)
            val_curves[h].append(va)
            omega_curves[h].append(om)
            print(f"    final train={tr[-1]:.3e}  val={va[-1]:.3e}  ω={om[-1]:.4f}")

    print_summary(train_curves, val_curves, omega_curves)


if __name__ == "__main__":
    main()
