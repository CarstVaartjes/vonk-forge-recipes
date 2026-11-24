"""Build (or reuse) the inkling_attn CUDA extension with torch.utils.cpp_extension.

Sources: csrc/inkling_attn_ext.cu (binding) + csrc/attn_kernel.cuh (kernel), compiled for sm_121a into
<cache>/<tag>/, where <cache> is INKLING_ATTN_CACHE (default ~/.cache/inkling-attn, i.e. /root/.cache/inkling-attn
in the container) and <tag> hashes the sources, the flags and the torch / CUDA versions, so a changed input never
reuses a stale build. Needs nvcc and ninja (both in the SGLang image); no GPU is needed to build.

    python3 -m inkling_attn.build      # build now (once, before serving); prints the path
"""

import hashlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCES = (os.path.join(HERE, "csrc", "inkling_attn_ext.cu"),)
HEADERS = (os.path.join(HERE, "csrc", "attn_kernel.cuh"),)
NAME = "inkling_attn_ext"

_UNDEF = (
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
)


def default_cache_dir():
    return os.environ.get("INKLING_ATTN_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "inkling-attn")


def cuda_flags(arch="121a"):
    return [
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
        "-O3",
        "-std=c++20",
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
        raise RuntimeError(f"inkling_attn: building the extension in {build_dir} failed: {exc}") from exc
    if not callable(getattr(ext, "fwd", None)):
        raise RuntimeError(f"inkling_attn: extension in {build_dir} lacks fwd()")
    INFO.update(build_dir=build_dir, cached=cached, seconds=time.monotonic() - t0, tag=tag)
    _EXT = ext
    return ext


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache", default=default_cache_dir())
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)
    load(a.cache, verbose=a.verbose)
    print(f"inkling_attn extension {'reused' if INFO['cached'] else 'built'} in {INFO['seconds']:.1f} s: {INFO['build_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
