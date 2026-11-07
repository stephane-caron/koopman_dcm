# SPDX-License-Identifier: Apache-2.0

"""Download the datasets of the paper from Hugging Face."""

from pathlib import Path

from . import settings
from .download import download_file

# Dataset repository, pinned to a commit so that files cannot change
REPO_URL: str = "https://huggingface.co/datasets/stephane-caron/koopman_dcm"
REVISION: str = "b2af67013a0e5cbaef93f225b8d8c223857ec005"

# Files of the repository, with their SHA-256 digests
FILES: dict[str, str] = {
    # One hour of real-robot data, training data of the DCM (Section 4.1)
    "cookie/cookie.h5": (
        "95295af0f513c9286516b2bf72aa52c0abf7ca6b6ec940b1c0ab2c1a55eead96"
    ),
    # Real-robot tracking at out-of-distribution periods (Sections 4.4-4.5)
    "cookie/cookie-ood.h5": (
        "0e6cbd9bf85f4d93cb53622c29aae49d993b1baf1af17553c25c40891e586d68"
    ),
    # One hour of PyBullet data, training data of the simulation DCM
    "cookie/pybullet-cookie.h5": (
        "22ff53fdd7faa86ed7a8c8ceb1d967978290e1e8a8669bde573e009e9ec1c527"
    ),
}

# Local directory the files are saved to
OUTPUT_DIR: Path = settings.DATA_DIR


if __name__ == "__main__":
    for filename, sha256 in FILES.items():
        download_file(
            f"{REPO_URL}/resolve/{REVISION}/{filename}",
            OUTPUT_DIR / Path(filename).name,
            sha256,
        )
