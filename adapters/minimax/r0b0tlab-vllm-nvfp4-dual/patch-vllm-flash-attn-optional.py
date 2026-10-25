"""Let vllm.vllm_flash_attn import when the image was built without FA2/FA3.

The parent image builds vLLM v0.21.0 with a setup.py patch that drops the
_vllm_fa2_C and _vllm_fa3_C extensions ("skip FA2/FA3 fallback extensions").
vLLM v0.21.0's vllm/vllm_flash_attn/__init__.py:32-36 raises ImportError when
neither is present, and on a CUDA platform every model import reaches it
(models -> layers/attention/__init__.py -> mm_encoder_attention.py ->
v1/attention/backends/fa_utils.py:20), so MiniMaxM2ForCausalLM fails to be
inspected before the attention backend (FlashInfer here) is chosen.

This turns that raise into a warning. FlashAttention stays unavailable:
is_fa_version_supported() reports FA2/FA3 unsupported, so vLLM never selects
a FlashAttention backend, and the kernels raise only if something calls them.
The build fails when the v0.21.0 text is not found exactly once.
"""

from __future__ import annotations

import importlib.util
import py_compile
from pathlib import Path

OLD = """if not (FA2_AVAILABLE or FA3_AVAILABLE):
    raise ImportError(
        "vllm.vllm_flash_attn requires the CUDA flash attention extensions "
        "(_vllm_fa2_C or _vllm_fa3_C). On ROCm, use upstream flash_attn."
    )
"""
NEW = """if not (FA2_AVAILABLE or FA3_AVAILABLE):
    # Vonk Forge: this image is built without _vllm_fa2_C/_vllm_fa3_C; keep the
    # module importable so non-FlashAttention backends (FlashInfer) can run.
    from vllm.logger import init_logger as _init_logger
    from vllm.vllm_flash_attn.flash_attn_interface import (
        FA2_UNAVAILABLE_REASON,
        FA3_UNAVAILABLE_REASON,
    )

    _init_logger(__name__).warning(
        "vllm.vllm_flash_attn: CUDA flash attention extensions are missing "
        "(FA2: %s; FA3: %s); FlashAttention backends are unavailable.",
        FA2_UNAVAILABLE_REASON,
        FA3_UNAVAILABLE_REASON,
    )
"""


def main() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("vLLM is missing from the base image")
    target = (
        Path(next(iter(spec.submodule_search_locations)))
        / "vllm_flash_attn"
        / "__init__.py"
    )
    source = target.read_text(encoding="utf-8")
    if source.count(OLD) != 1:
        raise SystemExit(
            f"vllm_flash_attn import guard anchor not found exactly once in {target}"
        )
    target.write_text(source.replace(OLD, NEW, 1), encoding="utf-8")
    py_compile.compile(str(target), doraise=True)
    print(f"made the FA2/FA3 extensions optional in {target}")


if __name__ == "__main__":
    main()
