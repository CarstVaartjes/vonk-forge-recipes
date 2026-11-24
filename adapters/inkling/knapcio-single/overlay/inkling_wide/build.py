"""Build (or reuse) the inkling_wide CUDA extension with torch.utils.cpp_extension.

Sources: csrc/wide_ext.cu (+ csrc/bf16x_format.cuh, pico's bf16x reader, MIT). Flags: -gencode
arch=compute_121a,code=sm_121a -O3 -std=c++20 --expt-relaxed-constexpr -lineinfo. Built into <cache>/<tag>/,
<cache> = INKLING_WIDE_CACHE (default ~/.cache/inkling-wide, i.e. /root/.cache/inkling-wide in the container),
<tag> a hash of the sources, the flags and the torch / CUDA versions: a later start with the same inputs only
imports the built .so, a changed input never reuses a stale build.

    python3 -m inkling_wide.build      # build now (no GPU needed); prints the path

Needs nvcc and ninja (both in the SGLang image).
"""

import hashlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = (os.path.join(HERE, "csrc", "wide_ext.cu"),)
HEADERS = (os.path.join(HERE, "csrc", "bf16x_format.cuh"),)
NAME = "inkling_wide_ext"

# torch adds -D__CUDA_NO_{HALF,BFLOAT16}_* to every nvcc line; the kernel uses __floats2bfloat162_rn.
_UNDEF = (
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
)


def default_cache_dir():
    return os.path.join(os.path.expanduser("~"), ".cache", "inkling-wide")


def cuda_flags(arch="121a"):
    return [
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
        "-O3",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "-lineinfo",
        *_UNDEF,
    ]


def build_tag(flags, torch_version, cuda_version):
    h = hashlib.sha256()
    for path in SOURCES + HEADERS:
        with open(path, "rb") as f:
            h.update(os.path.basename(path).encode() + b"\0" + f.read() + b"\0")
    h.update("\0".join(flags).encode())
    h.update(f"\0{torch_version}\0{cuda_version}".encode())
    return h.hexdigest()[:16]


_EXT = None
INFO = {}


def load(cache_dir=None, arch="121a", verbose=False):
    """Import the extension, building it first when <cache>/<tag> holds no build of these inputs."""
    global _EXT
    if _EXT is not None:
        return _EXT
    import torch
    from torch.utils import cpp_extension

    flags = cuda_flags(arch)
    if verbose:
        flags = flags + ["-Xptxas=-v"]
    tag = build_tag(flags, torch.__version__, torch.version.cuda)
    build_dir = os.path.join(cache_dir or default_cache_dir(), tag)
    os.makedirs(build_dir, exist_ok=True)
    cached = any(f.endswith(".so") and f.startswith(NAME) for f in os.listdir(build_dir))
    t0 = time.monotonic()
    try:
        ext = cpp_extension.load(
            name=NAME,
            sources=list(SOURCES),
            extra_cuda_cflags=flags,
            extra_include_paths=[os.path.join(HERE, "csrc")],
            build_directory=build_dir,
            verbose=verbose,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"inkling_wide: building the extension in {build_dir} failed: {exc}") from exc
    for attr in ("gemm", "decode", "prepare", "bucket", "kernel_info", "prepare_certwide", "certwide_exact"):
        if not callable(getattr(ext, attr, None)):
            raise RuntimeError(f"inkling_wide: extension in {build_dir} lacks {attr}()")
    INFO.update(build_dir=build_dir, cached=cached, seconds=time.monotonic() - t0, tag=tag)
    _EXT = ext
    return ext


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache", default=os.environ.get("INKLING_WIDE_CACHE") or default_cache_dir())
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)
    ext = load(a.cache, verbose=a.verbose)
    print(
        f"inkling_wide extension {'reused' if INFO['cached'] else 'built'} in {INFO['seconds']:.1f} s: "
        f"{INFO['build_dir']} (max_rows={ext.max_rows} block_n={ext.block_n})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
