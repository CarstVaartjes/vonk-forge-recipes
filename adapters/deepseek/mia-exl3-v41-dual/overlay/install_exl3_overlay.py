"""Install the EXL3 quantization overlay over the pinned vLLM (image build)."""

from pathlib import Path

import vllm

src = Path("/opt/dsv41/exl3.py")
dst = (
    Path(vllm.__file__).resolve().parent / "model_executor/layers/quantization/exl3.py"
)
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(src.read_text())
print("installed overlay exl3.py ->", dst)
