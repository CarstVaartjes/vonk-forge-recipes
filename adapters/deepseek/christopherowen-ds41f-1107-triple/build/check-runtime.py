import ctypes
import hashlib
import importlib.metadata as metadata
import subprocess

assert metadata.version("nvidia-cutlass-dsl") == "4.7.1"
import cutlass.base_dsl.enums  # noqa: F401  (quack-kernels, the DS4.1 L2 prefetch)
import quack.compile_utils  # noqa: F401
from vllm.model_executor.model_loader.direct_io import _native as direct_native
from vllm.models.deepseek_v4_1.engram_rows import _native as engram_native

# torch.cuda.nccl.version() reports the NCCL torch was compiled against, so
# ask the installed library itself (NCCL_VERSION_CODE 2.30.7 = 23007).
nccl = "/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2"
version = ctypes.c_int()
assert ctypes.CDLL(nccl).ncclGetVersion(ctypes.byref(version)) == 0
assert version.value == 23007, version.value
with open(nccl, "rb") as handle:
    built = open("/opt/spark-ds41f/nccl/libnccl.so.2.sha256").read().strip()
    assert hashlib.sha256(handle.read()).hexdigest() == built, "NCCL is not the patched build"
for name in ("_C_stable_libtorch", "_moe_C_stable_libtorch"):
    with open(f"/opt/spark-ds41f/candidate/vllm/vllm/{name}.abi3.so", "rb") as handle:
        assert handle.read(4) == b"\x7fELF"
for label, mod in (("direct_io", direct_native), ("engram_rows", engram_native)):
    builder = getattr(mod, "prebuild", mod._build)
    native_path = builder()
    linked = subprocess.check_output(["ldd", str(native_path)], text=True)
    # The sandbox blocks io_uring: the helper compiled against the POSIX stand-in (posix-uring), so no liburing is
    # linked and no io_uring_* symbol is imported (the creator asserts the opposite, "liburing.so" in linked).
    assert "liburing" not in linked, f"{label} links a real liburing"
    undefined = subprocess.check_output(["nm", "-D", "--undefined-only", str(native_path)], text=True)
    assert "io_uring" not in undefined, f"{label} still imports io_uring symbols:\n{undefined}"
print("runtime build checks passed")
