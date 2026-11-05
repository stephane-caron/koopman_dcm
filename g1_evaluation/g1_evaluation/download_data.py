# SPDX-License-Identifier: Apache-2.0

"""Download the training dataset of the paper from Hugging Face."""

from pathlib import Path

from . import settings
from .download import download_file

FILENAME: str = "g1/pybullet-g1.h5"

# Dataset repository, pinned to a commit so that the file cannot change
REPO_URL: str = "https://huggingface.co/datasets/stephane-caron/koopman_dcm"
REVISION: str = "041bebac1c115ce69909460aa492da944a79ac39"

SHA256: str = "422279e2fc397d478ec17eebacaa91e61fa2ed547c28ce98e2d85a0ebe0f9889"
OUTPUT: Path = settings.DATA_DIR / "pybullet-g1.h5"


if __name__ == "__main__":
    download_file(f"{REPO_URL}/resolve/{REVISION}/{FILENAME}", OUTPUT, SHA256)
