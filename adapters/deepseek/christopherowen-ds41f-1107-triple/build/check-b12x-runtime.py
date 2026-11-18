import subprocess

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
undefined = subprocess.check_output(["nm", "-D", "--undefined-only", native_path], text=True)
assert "io_uring" not in undefined, f"B12X still imports io_uring symbols:\n{undefined}"
print("B12X build checks passed")
