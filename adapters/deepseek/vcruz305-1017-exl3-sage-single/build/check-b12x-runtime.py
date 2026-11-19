import importlib.metadata as metadata
import subprocess

assert metadata.version("nvidia-cutlass-dsl") == "4.7.1"
import cutlass.base_dsl.enums  # noqa: F401  (quack-kernels, the DS4.1 L2 prefetch)
import quack.compile_utils  # noqa: F401
import b12x.attention.compressed_sparse_mla  # noqa: F401
import b12x.comm.roce  # noqa: F401
import b12x.integration.vllm.loader  # noqa: F401
import b12x.sequence.engram  # noqa: F401
from b12x.comm.roce._proxy import load
from b12x.loader._native import _build
print("RoCE proxy ABI", load().roce_abi_version())
native_path = _build()
linked = subprocess.check_output(["ldd", native_path], text=True)
# The sandbox blocks io_uring: B12X compiled against the POSIX stand-in (posix-uring), so no liburing is linked
# (the creator asserts the opposite, "liburing.so" in linked).
assert "liburing" not in linked, "B12X links a real liburing"
import ctypes
import hashlib

# torch.cuda.nccl.version() reports the NCCL torch was compiled against, so
# ask the installed library itself (NCCL_VERSION_CODE 2.30.7 = 23007).
nccl = "/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2"
version = ctypes.c_int()
assert ctypes.CDLL(nccl).ncclGetVersion(ctypes.byref(version)) == 0
assert version.value == 23007, version.value
with open(nccl, "rb") as handle:
    built = open("/opt/spark3/nccl/libnccl.so.2.sha256").read().strip()
    assert hashlib.sha256(handle.read()).hexdigest() == built, "NCCL is not the patched build"
for name in ("_C_stable_libtorch", "_moe_C_stable_libtorch"):
    with open(f"/opt/spark3/candidate/vllm/vllm/{name}.abi3.so", "rb") as handle:
        assert handle.read(4) == b"\x7fELF"
print("B12X build checks passed")
