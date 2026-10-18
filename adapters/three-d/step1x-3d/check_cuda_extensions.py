"""Verify compiled CUDA support on a builder without a GPU or driver."""

from __future__ import annotations

import importlib
import subprocess


def main() -> None:
    # Load libtorch before any extension (custom_rasterizer needs its symbols).
    import torch  # noqa: F401
    from pytorch3d import _C

    # This symbol is exported only under WITH_CUDA in the pinned ext.cpp.
    if not hasattr(_C, "knn_check_version"):
        raise SystemExit("pytorch3d was built without CUDA support")
    for name in ("pytorch3d._C", "custom_rasterizer_kernel", "_nvdiffrast_c"):
        extension = importlib.import_module(name)
        result = subprocess.run(
            ["cuobjdump", "--list-elf", extension.__file__],
            check=True,
            capture_output=True,
            text=True,
        )
        if "sm_121" not in result.stdout:
            raise SystemExit(f"{name} has no sm_121 CUDA kernels: {extension.__file__}")


if __name__ == "__main__":
    main()
