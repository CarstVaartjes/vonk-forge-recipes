"""Build (or reuse) the inkling_cert CUDA extension with torch.utils.cpp_extension.

The only source is csrc/cert_ext.cu; it #includes pico's gemv16.cu from inkling_pico12's vendored copy
(../inkling_pico12/third_party/pico, MIT, Andy Shu), so the exact pass is compiled from the very file the
`lmhead` GEMV is built from. Flags are pico's. The build directory is <cache>/<tag>, <tag> a hash of every source
and header, the flags, torch's and CUDA's versions. <cache> = INKLING_CERT_CACHE (default ~/.cache/inkling-cert).

    python3 -m inkling_cert.build      # build now (in the container, no GPU needed); prints the path
"""

import hashlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PICO = os.path.join(os.path.dirname(HERE), "inkling_pico12", "third_party", "pico")
SOURCES = (os.path.join(HERE, "csrc", "cert_ext.cu"),)
HEADERS = (
    os.path.join(HERE, "csrc", "screen6.cuh"),
    os.path.join(HERE, "csrc", "certwide_screen.cuh"),
    os.path.join(PICO, "kernels", "gemv16.cu"),
    os.path.join(PICO, "kernels", "dsk.h"),
    os.path.join(PICO, "kernels", "common.cuh"),
    os.path.join(PICO, "common", "check.h"),
)
NAME = "inkling_cert_ext"
_UNDEF = (
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
)


def default_cache_dir():
    return os.path.join(os.path.expanduser("~"), ".cache", "inkling-cert")


def cuda_flags(std="c++20", arch="121a"):
    return [f"-gencode=arch=compute_{arch},code=sm_{arch}", "-O3", f"-std={std}", "--expt-relaxed-constexpr",
            "-lineinfo", *_UNDEF]


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


def load(cache_dir=None, std="c++20", arch="121a", verbose=False):
    global _EXT
    if _EXT is not None:
        return _EXT
    import torch
    from torch.utils import cpp_extension

    missing = [p for p in SOURCES + HEADERS if not os.path.isfile(p)]
    if missing:
        raise RuntimeError(f"inkling_cert: sources missing (inkling_pico12 must sit next to inkling_cert): {missing}")
    flags = cuda_flags(std, arch)
    tag = build_tag(flags, torch.__version__, torch.version.cuda)
    build_dir = os.path.join(cache_dir or default_cache_dir(), tag)
    os.makedirs(build_dir, exist_ok=True)
    cached = any(f.endswith(".so") and f.startswith(NAME) for f in os.listdir(build_dir))
    t0 = time.monotonic()
    try:
        ext = cpp_extension.load(name=NAME, sources=list(SOURCES), extra_cuda_cflags=flags,
                                 extra_include_paths=[PICO], build_directory=build_dir, verbose=verbose)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"inkling_cert: building the extension in {build_dir} failed: {exc}") from exc
    for attr in ("screen", "exact", "prepare_certwide", "screen_wide", "screen_wide_info", "debug_select_wide"):
        if not callable(getattr(ext, attr, None)):
            raise RuntimeError(f"inkling_cert: extension in {build_dir} lacks {attr}()")
    INFO.update(build_dir=build_dir, cached=cached, seconds=time.monotonic() - t0, tag=tag)
    _EXT = ext
    return ext


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache", default=os.environ.get("INKLING_CERT_CACHE") or default_cache_dir())
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)
    load(a.cache, verbose=a.verbose)
    print(f"inkling_cert extension {'reused' if INFO['cached'] else 'built'} in {INFO['seconds']:.1f} s: "
          f"{INFO['build_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
