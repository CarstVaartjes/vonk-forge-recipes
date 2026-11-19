"""Build the Engram row reader and B12X's loader against the POSIX liburing stand-in and prove neither
calls io_uring: no liburing in ldd, no io_uring_* symbol left undefined (the stand-in is header only).

The Engram reader's builder is loaded by file path: its package imports vLLM, which is not importable without a GPU."""
import importlib.util
import subprocess

from b12x.loader._native import _build as b12x_build

ENGRAM = "/opt/spark3/candidate/vllm/vllm/models/deepseek_v4_1/engram_rows/_native.py"
spec = importlib.util.spec_from_file_location("engram_rows_native", ENGRAM)
engram = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engram)

for label, build in (("B12X loader", b12x_build), ("Engram row reader", engram._build)):
    path = str(build())
    linked = subprocess.check_output(["ldd", path], text=True)
    assert "liburing" not in linked, f"{label} links a real liburing:\n{linked}"
    undefined = subprocess.check_output(["nm", "-D", "--undefined-only", path], text=True)
    assert "io_uring" not in undefined, f"{label} still imports io_uring symbols:\n{undefined}"
    print(label, "built without io_uring:", path)
