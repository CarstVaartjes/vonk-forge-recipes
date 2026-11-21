import ctypes
import subprocess
from pathlib import Path

from cuda.pathfinder import find_nvidia_binary_utility

for tool in ("nvcc", "ptxas"):
    out = subprocess.check_output([f"/usr/local/cuda/bin/{tool}", "--version"], text=True)
    assert "release 13.5, V13.5.37" in out, out
    resolved = Path(find_nvidia_binary_utility(tool)).resolve()
    assert resolved == Path(f"/usr/local/cuda-13.5/bin/{tool}"), resolved
for lib_path in (
    "/usr/local/cuda/lib64/libcublasLt.so.13",
    "/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libcublasLt.so.13",
):
    assert Path(lib_path).resolve() == Path(
        "/usr/local/cuda-13.5/targets/sbsa-linux/lib/libcublasLt.so.13.10.0.20"
    ), Path(lib_path).resolve()
    assert ctypes.CDLL(lib_path).cublasLtGetVersion() == 131000
for name in ("libcudart.so.13", "libnvrtc.so.13", "libnvJitLink.so.13", "libcublas.so.13"):
    resolved = Path(f"/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/{name}").resolve()
    assert resolved.is_relative_to("/usr/local/cuda-13.5/targets/sbsa-linux/lib"), resolved
cudart = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so.13")
rt_ver = ctypes.c_int()
assert cudart.cudaRuntimeGetVersion(ctypes.byref(rt_ver)) == 0
assert rt_ver.value == 13050, rt_ver.value
print("CUDA 13.5 build checks passed")
