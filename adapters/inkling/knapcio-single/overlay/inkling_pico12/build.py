"""Build (or reuse) the inkling_pico12 CUDA extension with torch.utils.cpp_extension.

The sources are pico's gemv16.cu (third_party/pico, verbatim) and our wrapper csrc/pico12_ext.cu. They are
compiled with pico's own flags (-gencode arch=compute_121a,code=sm_121a -O3 -std=c++20
--expt-relaxed-constexpr) into <cache>/<tag>/, where <cache> is INKLING_PICO12_CACHE (default
~/.cache/inkling-pico12, i.e. /root/.cache/inkling-pico12 in the container) and <tag> hashes the sources, the
flags, torch's version and CUDA's version, so a later start with the same inputs only imports the built .so and
a changed input never reuses a stale build.

    python3 -m inkling_pico12.build      # build now (e.g. once in the container before serving); prints the path

Needs nvcc and ninja (both in the SGLang image). No GPU is needed to build.
"""

import hashlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PICO = os.path.join(HERE, "third_party", "pico")
SOURCES = (
    os.path.join(HERE, "csrc", "pico12_ext.cu"),
    os.path.join(PICO, "kernels", "gemv16.cu"),
)
HEADERS = (
    os.path.join(PICO, "kernels", "dsk.h"),
    os.path.join(PICO, "kernels", "common.cuh"),
    os.path.join(PICO, "common", "check.h"),
)
NAME = "inkling_pico12_ext"

# torch adds -D__CUDA_NO_{HALF,BFLOAT16}_* to every nvcc line; pico was built and checked without them.
_UNDEF = (
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
)


def default_cache_dir():
    return os.path.join(os.path.expanduser("~"), ".cache", "inkling-pico12")


def cuda_flags(std="c++20", arch="121a"):
    return [
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
        "-O3",
        f"-std={std}",
        "--expt-relaxed-constexpr",
        "-lineinfo",
        *_UNDEF,
    ]


def build_tag(flags, torch_version, cuda_version, files=None):
    h = hashlib.sha256()
    for path in files or SOURCES + HEADERS:
        with open(path, "rb") as f:
            h.update(os.path.basename(path).encode() + b"\0" + f.read() + b"\0")
    h.update("\0".join(flags).encode())
    h.update(f"\0{torch_version}\0{cuda_version}".encode())
    return h.hexdigest()[:16]


_EXT = None
INFO = {}


def load(cache_dir=None, std="c++20", arch="121a", verbose=False):
    """Import the extension, building it first when <cache>/<tag> holds no build of these inputs."""
    global _EXT
    if _EXT is not None:
        return _EXT
    import torch
    from torch.utils import cpp_extension

    flags = cuda_flags(std, arch)
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
            extra_include_paths=[PICO],
            build_directory=build_dir,
            verbose=verbose,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"inkling_pico12: building the extension in {build_dir} failed: {exc}") from exc
    for attr in ("tile", "encode", "check", "gemv", "gemv_raw"):
        if not callable(getattr(ext, attr, None)):
            raise RuntimeError(f"inkling_pico12: extension in {build_dir} lacks {attr}()")
    INFO.update(build_dir=build_dir, cached=cached, seconds=time.monotonic() - t0, tag=tag)
    _EXT = ext
    return ext


# `stack1`: a second, separate extension (built only when the lever is on, so the default build is unchanged). Its
# one source includes pico's gemv16.cu verbatim (see csrc/pico12_range.cu); gemv16.cu is therefore hashed as a header.
RANGE_SOURCES = (os.path.join(HERE, "csrc", "pico12_range.cu"),)
RANGE_NAME = "inkling_pico12_range_ext"
_RANGE = None


def load_range(cache_dir=None, std="c++20", arch="121a", verbose=False):
    """Import the stack1 range extension, building it first when <cache>/range-<tag> holds no build of its inputs."""
    global _RANGE
    if _RANGE is not None:
        return _RANGE
    import torch
    from torch.utils import cpp_extension

    flags = cuda_flags(std, arch)
    tag = build_tag(flags, torch.__version__, torch.version.cuda, files=RANGE_SOURCES + SOURCES[1:] + HEADERS)
    build_dir = os.path.join(cache_dir or default_cache_dir(), f"range-{tag}")
    os.makedirs(build_dir, exist_ok=True)
    cached = any(f.endswith(".so") and f.startswith(RANGE_NAME) for f in os.listdir(build_dir))
    t0 = time.monotonic()
    try:
        ext = cpp_extension.load(
            name=RANGE_NAME,
            sources=list(RANGE_SOURCES),
            extra_cuda_cflags=flags,
            extra_include_paths=[PICO],
            build_directory=build_dir,
            verbose=verbose,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"inkling_pico12: building the stack1 extension in {build_dir} failed: {exc}") from exc
    for attr in ("gemv_range", "gemv_stacked"):
        if not callable(getattr(ext, attr, None)):
            raise RuntimeError(f"inkling_pico12: extension in {build_dir} lacks {attr}()")
    INFO.update(range_build_dir=build_dir, range_cached=cached, range_seconds=time.monotonic() - t0)
    _RANGE = ext
    return ext


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache", default=os.environ.get("INKLING_PICO12_CACHE") or default_cache_dir())
    p.add_argument("--std", default=os.environ.get("INKLING_PICO12_STD", "c++20"))
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--stack1", action="store_true", help="also build the stack1 range extension")
    a = p.parse_args(argv)
    ext = load(a.cache, a.std, verbose=a.verbose)
    if a.stack1:
        load_range(a.cache, a.std, verbose=a.verbose)
        print(f"inkling_pico12 stack1 extension {'reused' if INFO['range_cached'] else 'built'} in "
              f"{INFO['range_seconds']:.1f} s: {INFO['range_build_dir']}")
    print(
        f"inkling_pico12 extension {'reused' if INFO['cached'] else 'built'} in {INFO['seconds']:.1f} s: "
        f"{INFO['build_dir']} (max_rows={ext.max_rows})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
