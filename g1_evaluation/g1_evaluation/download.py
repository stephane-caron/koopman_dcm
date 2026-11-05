# SPDX-License-Identifier: Apache-2.0

"""Download files of the paper from Hugging Face, checking their digest."""

import hashlib
import urllib.request
from pathlib import Path


def sha256sum(path: Path) -> str:
    """Compute the SHA-256 digest of a file.

    Args:
        path: File to hash.

    Returns:
        Hexadecimal digest.
    """
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def download_file(url: str, output: Path, sha256: str) -> None:
    """Download a file unless it is already present.

    An existing file at the output path is used as is, so that the download
    never overwrites a file produced locally. Only downloaded files are
    checked against the expected digest.

    Args:
        url: URL of the file.
        output: Path to save the file to.
        sha256: Expected SHA-256 digest of the downloaded file.
    """
    if output.exists():
        print(f"{output} is already present")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output.with_name(output.name + ".part")
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, tmp_path)
    if sha256sum(tmp_path) != sha256:
        tmp_path.unlink()
        raise SystemExit(f"Checksum mismatch for {url}")
    tmp_path.replace(output)
    print(f"Saved to {output}")
