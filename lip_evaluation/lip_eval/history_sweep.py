# SPDX-License-Identifier: Apache-2.0

"""Sweep the history size of the DCM observable, print final losses and ω.

For each history size in HISTORY_SIZES, runs NUM_SAMPLES training runs
(different seeds) and prints the mean ± std of the final train/val losses
and ω over samples.

Training delegates to ``lip_eval.train_dcm.train`` so the pipeline
matches the main LIP evaluation script exactly --- DCM dynamics loss +
CoM prediction loss, ω jointly learned as exp(log_ω).

Usage:
    pixi run history-sweep
"""

import numpy as np

from lip_eval.__main__ import DT, generate_data
from lip_eval.train_dcm import train

HISTORY_SIZES: tuple[int, ...] = (1, 3, 5, 10, 20)  # delay-embedding horizons H
NUM_SAMPLES: int = 10  # seeds per history size
NUM_STEPS: int = 50_000  # closed-loop simulation steps per dataset
EPOCHS: int = 50
OMEGA_INIT: float = 4.0  # rad/s, initial guess for ω
COM_LOSS_WEIGHT: float = 1.0  # w_CoM


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
    train_curves: dict = {h: [] for h in HISTORY_SIZES}
    val_curves: dict = {h: [] for h in HISTORY_SIZES}
    omega_curves: dict = {h: [] for h in HISTORY_SIZES}

    for h in HISTORY_SIZES:
        print(f"\n=== History size {h} ===")
        for s in range(NUM_SAMPLES):
            seed = 1000 * h + s
            print(f"  sample {s + 1}/{NUM_SAMPLES}  (seed={seed})")
            Y, U = generate_data(NUM_STEPS, seed=seed)
            _, _, tr, va, om, _, _ = train(
                Y,
                U,
                dt=DT,
                hist_size=h,
                omega_init=OMEGA_INIT,
                epochs=EPOCHS,
                com_loss_weight=COM_LOSS_WEIGHT,
                seed=seed,
            )
            train_curves[h].append(tr)
            val_curves[h].append(va)
            omega_curves[h].append(om)
            print(f"    final train={tr[-1]:.3e}  val={va[-1]:.3e}  ω={om[-1]:.4f}")

    print_summary(train_curves, val_curves, omega_curves)


if __name__ == "__main__":
    main()
