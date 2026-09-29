"""Install the exact ARM64 NCCL runtime used by the upstream TP4 profile."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

WHEEL_URL = (
    "https://files.pythonhosted.org/packages/65/32/"
    "ff4e28cbed87f99fed63df446ef1986e0617842258a3535eaa2ee92d6226/"
    "nvidia_nccl_cu13-2.30.4-py3-none-manylinux_2_18_aarch64.whl"
)
WHEEL_SHA256 = "e99308a3a89fba78918d50886e81072a6c8b0b4199feb02c3903e63713a6525a"
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
