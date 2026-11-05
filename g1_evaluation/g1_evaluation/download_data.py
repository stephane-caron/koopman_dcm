# SPDX-License-Identifier: Apache-2.0

"""Download the training dataset of the paper from Hugging Face."""

from pathlib import Path

from . import settings
from .download import download_file

FILENAME: str = "g1/pybullet-g1.h5"

# Dataset repository, pinned to a commit so that the file cannot change
REPO_URL: str = "https://huggingface.co/datasets/stephane-caron/koopman_dcm"
REVISION: str = "TODO"

SHA256: str = "TODO"
OUTPUT: Path = settings.DATA_DIR / "pybullet-g1.h5"


if __name__ == "__main__":
    download_file(f"{REPO_URL}/resolve/{REVISION}/{FILENAME}", OUTPUT, SHA256)
