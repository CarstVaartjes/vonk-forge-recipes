from pathlib import Path


path = Path("/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py")
source = path.read_text()
anchor = "    try:\n        from flash_attn import flash_attn_varlen_func"
replacement = (
    "    from sglang.srt.utils import is_sm100_supported\n"
    "    if not is_sm100_supported():\n"
    "        from sglang.srt.layers.attention.qsa_fa_fallback import triton_varlen_attn_func\n"
    "        return triton_varlen_attn_func\n"
) + anchor
if anchor not in source:
    raise SystemExit("QSA patch anchor missing from pinned SGLang image")
if "qsa_fa_fallback" in source:
    raise SystemExit("QSA fallback already present in pinned SGLang image")
path.write_text(source.replace(anchor, replacement, 1))
