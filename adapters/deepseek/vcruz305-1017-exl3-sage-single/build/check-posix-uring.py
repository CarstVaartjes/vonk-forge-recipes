"""Build the Engram row reader and B12X's loader against the POSIX liburing stand-in and prove neither
calls io_uring: no liburing in ldd, no io_uring_* symbol left undefined (the stand-in is header only)."""
import subprocess
import sys

from b12x.loader._native import _build as b12x_build
from vllm.models.deepseek_v4_1.engram_rows._native import _build as engram_build

for label, build in (("B12X loader", b12x_build), ("Engram row reader", engram_build)):
    path = str(build())
    linked = subprocess.check_output(["ldd", path], text=True)
    assert "liburing" not in linked, f"{label} links a real liburing:\n{linked}"
    undefined = subprocess.check_output(["nm", "-D", "--undefined-only", path], text=True)
    assert "io_uring" not in undefined, f"{label} still imports io_uring symbols:\n{undefined}"
    print(label, "built without io_uring:", path)
sys.exit(0)
