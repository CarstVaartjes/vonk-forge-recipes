#!/usr/bin/env python3
"""Fail closed if the digest-pinned Mia DeepSeek V4.1 EXL3 image is incomplete."""

from pathlib import Path

VLLM = Path("/usr/local/lib/python3.12/dist-packages/vllm")
required = (
    "/opt/dsv41/chat_template.jinja",
    "/opt/dsv41/exl3_k_map.json",
    "/opt/dsv41/engram_file_backend.py",
    "/opt/dsv41/engram_layout.py",
    "/opt/dsv41/librow_store.so",
    str(VLLM / "model_executor/layers/quantization/exl3.py"),
)
missing = [path for path in required if not Path(path).is_file()]
if missing:
    raise SystemExit(f"incomplete Mia DeepSeek V4.1 EXL3 runtime: {missing}")

try:
    # torch first: the compiled extension needs libc10/libtorch already loaded.
    import torch  # noqa: F401, I001  (load libc10/libtorch before the CUDA extension)
    import exl3_fat_moe_ext
    import exllamav3_ext
except Exception as exc:  # pragma: no cover - exercised by image build
    raise SystemExit(f"incomplete EXL3 runtime: {exc}") from exc
for name in ("exl3_moe", "exl3_fat_gemm"):
    if not hasattr(exllamav3_ext, name):
        raise SystemExit(f"incomplete EXL3 extension: {name} is absent")
if int(exl3_fat_moe_ext.exl3_fat_moe_abi()) != 2:
    raise SystemExit(
        "incomplete EXL3 extension: grouped fat-expert kernels are not ABI 2"
    )

# The container root is read-only at run time, so every overlay that rewrites an
# installed vLLM file must already be present in the image.
markers = {
    "dsv41-engram-file": "file-backed Engram tables",
    "dsv41-sm120-block64": "SM12x 64-token KV blocks",
    "dsv41-h2d-stage": "host-to-device weight staging",
}
found: set[str] = set()
for source in VLLM.rglob("*.py"):
    text = source.read_text(errors="ignore")
    found.update(marker for marker in markers if marker in text)
absent = [label for marker, label in markers.items() if marker not in found]
if absent:
    raise SystemExit(f"incomplete Mia DeepSeek V4.1 overlay: {absent}")
if "input_text" not in (VLLM / "tokenizers/deepseek_v41.py").read_text():
    raise SystemExit("incomplete Mia DeepSeek V4.1 overlay: Responses text content")

print("Mia DeepSeek V4.1 EXL3 runtime contract OK")
