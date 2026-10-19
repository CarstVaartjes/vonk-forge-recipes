"""Honor the disabled JIT flag before importing toolkit-dependent DeepGEMM."""
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
anchor = "def _compute_enable_deep_gemm():\n"
if source.count(anchor) != 1:
    raise SystemExit("pinned SGLang DeepGEMM initialization function is missing")
source = source.replace(
    anchor,
    anchor + "    if not envs.SGLANG_ENABLE_JIT_DEEPGEMM.get():\n        return False\n",
    1,
)
compile(source, str(path), "exec")
path.write_text(source)
