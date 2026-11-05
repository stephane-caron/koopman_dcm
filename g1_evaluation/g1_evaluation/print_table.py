# SPDX-License-Identifier: Apache-2.0

"""Print the G1 part of Table 2 from saved evaluation results."""

from pathlib import Path

import h5py
import numpy as np
import scipy.stats

from . import settings

# Per-trial results written by ``pixi run evaluate``
INPUT: Path = settings.DATA_DIR / "g1-evaluation.h5"

# Row labels of the paper's table, in order
PAPER_LABELS: dict[str, str] = {
    "dcm": "Proportional DCM",
    "viability-less-mpc": "MPC w/o viability",
    "mpc": "MPC",
}


def print_table(input_path: Path) -> None:
    """Print fall rates and ZMP RMSE in the format of the paper's table.

    RMSE is reported as mean ± half-width of the 95 % confidence
    interval on the mean (Student t, ``n - 1`` degrees of freedom),
    over all trials of a controller, fallen or not.

    Args:
        input_path: HDF5 file written by ``pixi run evaluate``.
    """
    with h5py.File(input_path, "r") as hf:
        ctrl = hf["controller"][:].astype(str)
        fell = hf["did_fall"][:]
        rmse = hf["rmse_zmp_last_2T"][:]
        robot = hf.attrs.get("robot", "")
        learned_model = hf.attrs.get("learned_model", "")

    rows = []
    for c, label in PAPER_LABELS.items():
        mask = ctrl == c
        if not mask.any():
            continue
        n_trials = int(mask.sum())
        n_falls = int(fell[mask].sum())
        r = rmse[mask]
        r = r[~np.isnan(r)].astype(np.float64)
        if len(r) > 1:
            t_crit = scipy.stats.t.ppf(0.975, df=len(r) - 1)
            half_width = t_crit * r.std(ddof=1) / np.sqrt(len(r))
            rmse_str = f"{r.mean():.2f} ± {half_width:.2f}"
        elif len(r) == 1:
            rmse_str = f"{r[0]:.2f}"
        else:
            rmse_str = "n/a"
        fall_rate = 100.0 * n_falls / n_trials
        rows.append(
            (
                label,
                f"{fall_rate:.0f}% ({n_falls}/{n_trials})",
                rmse_str,
            )
        )

    header = ("Controller", "Fall rate", "RMSE")
    widths = [max(len(row[i]) for row in rows + [header]) for i in range(3)]
    lines = [
        f"Robot: {robot} | learned DCM: {learned_model}",
        "  ".join(h.ljust(w) for h, w in zip(header, widths)),
        "  ".join("-" * w for w in widths),
    ]
    lines += ["  ".join(v.ljust(w) for v, w in zip(row, widths)) for row in rows]
    print("\n".join(lines))


if __name__ == "__main__":
    print_table(INPUT)
