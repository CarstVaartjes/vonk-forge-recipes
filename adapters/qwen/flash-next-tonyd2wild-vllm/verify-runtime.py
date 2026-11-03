#!/usr/bin/env python3
"""Check the image packaging at build time (ours; upstream features are not tested here)."""

from __future__ import annotations

import ast
import mmap
import struct
import sys
from pathlib import Path

PKG = Path("/usr/local/lib/python3.12/dist-packages/vllm")
NVIDIA = "models/qwen4_exp/nvidia"
# Files the upstream overlays replace: they must exist in the pinned base image.
BASE = (
    f"{NVIDIA}/ple_layer.py",
    f"{NVIDIA}/model_state.py",
    f"{NVIDIA}/mtp.py",
    f"{NVIDIA}/qsa.py",
    f"{NVIDIA}/ops/ple.py",
    f"{NVIDIA}/ops/qsa.py",
    "config/compilation.py",
    "platforms/interface.py",
    "model_executor/layers/quantization/modelopt.py",
)
# Files that must exist after the overlays are copied in, with a marker each.
OVERLAYS = {
    f"{NVIDIA}/ple_layer.py": "QWEN4EXP_PLE_STAGED",
    f"{NVIDIA}/ops/ple_mmap.py": "QWEN4EXP_PLE_MMAP_CHUNK",
    f"{NVIDIA}/model_state.py": "QWEN4EXP_PLE_STAGED",
    f"{NVIDIA}/mtp.py": "QWEN4EXP_DRAFT_VOCAB",
    "config/compilation.py": "qwen4_exp_ple_mmap_gather",
}

# CUDA fatbin container (cudafe/fatbinary): a 16-byte header, then entries of
# (header, payload). Entry header fields used here: kind u16 at 0 (1 PTX, 2 ELF
# cubin), header size u32 at 4, payload size u64 at 8, SM number u32 at 28, flags
# u64 at 40. Measured on vLLM's 8a728663 wheels: flags bit 0x100000 marks an
# arch-specific target (sm_90a, sm_120a); the CUDA 13 family target (sm_120f) is a
# plain cubin without it. An sm_120a cubin does not load on GB10 (sm_121).
FATBIN_MAGIC = b"\x50\xed\x55\xba"
ENTRY_PTX = 1
ARCH_SPECIFIC = 0x100000
FAMILY = 0x200000
GB10_SM = 121


def fatbin_entries(data, offset: int) -> list[tuple[int, int, int]]:
    """(kind, sm, flags) of each entry of the fatbin at offset; [] if it is not one."""
    if offset + 16 > len(data):
        return []
    version, header, size = struct.unpack_from("<HHQ", data, offset + 4)
    end = min(offset + header + size, len(data))
    if version != 1 or header != 16:
        return []
    entries = []
    at = offset + header
    while at + 48 <= end:
        kind, _, entry_header, payload = struct.unpack_from("<HHIQ", data, at)
        sm = struct.unpack_from("<I", data, at + 28)[0]
        flags = struct.unpack_from("<Q", data, at + 40)[0]
        if entry_header < 48:
            break
        entries.append((kind, sm, flags))
        at += entry_header + payload
    return entries


def loads_on_gb10(kind: int, sm: int, flags: int) -> bool:
    """True when the driver can run this entry on sm_121 (directly or by PTX JIT)."""
    if kind != ENTRY_PTX and sm == GB10_SM:
        return True  # sm_121 and sm_121a run on GB10 itself
    if flags & ARCH_SPECIFIC:
        return False
    if kind == ENTRY_PTX and flags & FAMILY:
        return sm in (120, GB10_SM)  # family PTX only JITs inside its own family
    return sm <= GB10_SM if kind == ENTRY_PTX else sm == 120


def stranded_fatbins(data) -> list[int]:
    """Offsets of fatbins that carry SM12x code but nothing sm_121 can load."""
    stranded = []
    at = data.find(FATBIN_MAGIC)
    while at != -1:
        entries = fatbin_entries(data, at)
        if any(sm in (120, GB10_SM) for _, sm, _ in entries) and not any(
            loads_on_gb10(*entry) for entry in entries
        ):
            stranded.append(at)
        at = data.find(FATBIN_MAGIC, at + 4)
    return stranded


def check_kernels() -> None:
    """Fail the build for a vLLM library whose SM12x kernels cannot load on GB10."""
    found = {}
    for path in sorted(PKG.rglob("*.so")):
        with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
            stranded = stranded_fatbins(data)
        if stranded:
            found[path] = stranded
    if found:
        listing = "; ".join(f"{path.relative_to(PKG)}: {len(at)}" for path, at in found.items())
        raise SystemExit(
            "SM12x kernels with no sm_121-loadable image (built for sm_120a, or a CUDA 12.x "
            f"build): {listing}"
        )


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) == 2 else ""
    if stage == "base":
        for name in BASE:
            if not (PKG / name).is_file():
                raise SystemExit(f"pinned vLLM image is missing {PKG / name}")
    elif stage == "overlays":
        for name, marker in OVERLAYS.items():
            path = PKG / name
            text = path.read_text(encoding="utf-8")
            if marker not in text:
                raise SystemExit(f"overlay is missing or wrong: {path}")
            ast.parse(text, filename=str(path))
        for name in BASE:
            ast.parse((PKG / name).read_text(encoding="utf-8"), filename=name)
        if not Path("/opt/vonk/bin/vllm").is_file():
            raise SystemExit("Controller wrapper is missing")
    elif stage == "kernels":
        check_kernels()
    else:
        raise SystemExit("usage: verify-runtime.py base|overlays|kernels")


if __name__ == "__main__":
    main()
