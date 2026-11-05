# SPDX-License-Identifier: Apache-2.0

"""Download the DCM observable of the paper from Hugging Face."""

from pathlib import Path

from . import settings
from .download import download_file

FILENAME: str = "dcm-pybullet-g1.onnx"

# Model repository, pinned to a commit so that the file cannot change
REPO_URL: str = "https://huggingface.co/stephane-caron/koopman_dcm"
REVISION: str = "f6e1244f874125b7764833d2f64ab7ed3ad8836b"

SHA256: str = "3766f50422b1e3712d85225bdff9f7192f13cce5ae9a9a0c12a93d58556fe037"
OUTPUT: Path = settings.MODELS_DIR / FILENAME


if __name__ == "__main__":
    download_file(f"{REPO_URL}/resolve/{REVISION}/{FILENAME}", OUTPUT, SHA256)
