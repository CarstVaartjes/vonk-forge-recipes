#!/usr/bin/env python3
"""Compile FlashInfer's patched sparse_mla_sm120 module at image build, offline.

Vonk Forge packaging of the creator's scripts/build_flashinfer.sh (Reederey87 at
4ccddc8801757e3f26b162688ddd6bcdd0668bdb). The creator prebuilds this module offline because an in-engine
nvcc build raced the server for the same unified memory pool and reset every node of a 4-Spark cluster. Here
the build runs once in the image (FLASHINFER_CUDA_ARCH_LIST=12.1a, MAX_JOBS=2, no GPU needed for nvcc).

The built .so is then copied into the AOT slot (spec.aot_path) that mask_flashinfer_aot.py emptied. Workloads run
with a read-only root, and FlashInfer's JIT path (build_and_load) takes a FileLock under its writable tmp dir
before it looks for a cached .so, so a JIT-cache prebuild would fail with EROFS at runtime. FlashInfer's
try_load() loads an AOT .so directly, with no lock, ninja or writes.

Fails the image build when the AOT binary is still present (the patched .cu would be ignored), the build did not
produce the .so, it was built for another architecture, or the copy does not make spec.is_aot true.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from flashinfer.jit import env as jit_env
from flashinfer.mla._sparse_mla_sm120 import gen_sparse_mla_sm120_module

spec = gen_sparse_mla_sm120_module()
if spec.is_aot:
    raise SystemExit(
        "AOT binary is still present, so the patched .cu would be ignored. "
        "overlay/mask_flashinfer_aot.py must run before this step."
    )
spec.build()
built = Path(spec.jit_library_path)
root = Path(jit_env.FLASHINFER_CACHE_DIR)
if not built.is_file():
    raise SystemExit(f"the runtime would load {built}, which the build did not produce")
if "121a" not in built.relative_to(root).parts:
    raise SystemExit(f"module built for another architecture: {built}")
aot = Path(spec.aot_path)
aot.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(built, aot)
aot.chmod(0o755)
if not spec.is_aot or aot.stat().st_size != built.stat().st_size:
    raise SystemExit(f"the patched module is not in the AOT slot {aot}; the runtime would not load it")
print(f"built {built.relative_to(root)} ({built.stat().st_size / 1024 / 1024:.1f} MiB) -> AOT slot {aot}")
