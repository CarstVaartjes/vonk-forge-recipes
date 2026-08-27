"""Install the exact ARM64 NCCL runtime used by the official Mia profile."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

WHEEL_URL = (
    "https://files.pythonhosted.org/packages/d1/21/"
    "a73174c6157101bdf1ffc22b517f76ff0082613989dd9bc8f43e8034caac/"
    "nvidia_nccl_cu13-2.30.7-py3-none-manylinux_2_18_aarch64.whl"
)
WHEEL_SHA256 = "ca786ffa5a647c75d4d1f5cc72a6c4f537947e2ba8823d7c8aaf768e7a7b9f77"
WHEEL_MEMBER = "nvidia/nccl/lib/libnccl.so.2"
TARGET = Path("/opt/vonk/lib/libnccl.so.2")


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        wheel = Path(temporary) / "nccl.whl"
        digest = hashlib.sha256()
        with (
            urllib.request.urlopen(WHEEL_URL, timeout=600) as response,
            wheel.open("wb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != WHEEL_SHA256:
            raise SystemExit("NCCL wheel SHA-256 mismatch")
        with zipfile.ZipFile(wheel) as archive:
            info = archive.getinfo(WHEEL_MEMBER)
            if info.is_dir() or info.file_size < 100_000_000:
                raise SystemExit("NCCL wheel library member is invalid")
            TARGET.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, TARGET.open("wb") as output:
                shutil.copyfileobj(source, output)
    os.chmod(TARGET, 0o555)


if __name__ == "__main__":
    main()
