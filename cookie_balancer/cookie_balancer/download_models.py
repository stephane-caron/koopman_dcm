# SPDX-License-Identifier: Apache-2.0

"""Download the DCM observables of the paper from Hugging Face."""

from pathlib import Path

from . import settings
from .download import download_file

# Model repository, pinned to a commit so that files cannot change
REPO_URL: str = "https://huggingface.co/stephane-caron/koopman_dcm"
REVISION: str = "f4275afc51a84fd75e6f66be6b94d7bc581b7213"

# Files of the repository, with their SHA-256 digests
FILES: dict[str, str] = {
    # Trained on real-robot data, Section 4 of the paper
    "dcm-cookie.onnx": (
        "3330a452b8466538d8b249ba3b856f37af7286a4b2830e331a2920b2a0fc207f"
    ),
    # Trained on PyBullet data, Table 2 of the paper
    "dcm-pybullet-cookie.onnx": (
        "3bd06a479bb51837d243fbcce484635f3503c1c5c66ea0ac210c49926af243f7"
    ),
}

# Local directory the files are saved to
OUTPUT_DIR: Path = settings.MODELS_DIR


if __name__ == "__main__":
    for filename, sha256 in FILES.items():
        download_file(
            f"{REPO_URL}/resolve/{REVISION}/{filename}",
            OUTPUT_DIR / filename,
            sha256,
        )
