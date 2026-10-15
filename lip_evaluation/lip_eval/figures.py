# SPDX-License-Identifier: Apache-2.0

"""Save matplotlib figures to the output directory."""

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

FIGURES_DIR: Path = Path("figures")


def save_pyplot_figure(name: str) -> Path:
    """Save the current figure as ``figures/<date>_<time>_<name>.pdf``.

    Args:
        name: Plot name.

    Returns:
        Path to the saved PDF.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    dt_str = datetime.now().astimezone().strftime("%Y-%m-%d_%H%M%S")
    pdf_path = FIGURES_DIR / f"{dt_str}_{name}.pdf"
    plt.savefig(pdf_path)
    print(f"  Plot saved → {pdf_path}")
    return pdf_path
