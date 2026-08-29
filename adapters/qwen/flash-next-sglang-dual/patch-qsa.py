from pathlib import Path


path = Path("/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py")
source = path.read_text()
anchor = "    try:\n        from flash_attn import flash_attn_varlen_func"
replacement = (
    "    from sglang.srt.utils import is_sm121\n"
    "\n"
    "    if is_sm121():\n"
    "        from sglang.srt.layers.attention.qsa.sm121_varlen import (\n"
    "            qsa_sm121_varlen_attention,\n"
    "        )\n"
    "\n"
    "        return qsa_sm121_varlen_attention\n"
) + anchor
if anchor not in source:
    raise SystemExit("QSA patch anchor missing from pinned SGLang image")
if "qsa.sm121_varlen" in source:
    raise SystemExit("QSA fallback already present in pinned SGLang image")
source = source.replace(anchor, replacement, 1)

marker = "dspark: SM121 must not use TRT-LLM sparse decode"
resolver = "def _resolve_trtllm_sparse_decode():"
resolver_index = source.find(resolver)
if resolver_index < 0:
    raise SystemExit("QSA TRT-LLM resolver missing from pinned SGLang image")
docstring_start = source.find('\"\"\"', resolver_index)
docstring_end = source.find('\"\"\"', docstring_start + 3)
if docstring_start < 0 or docstring_end < 0:
    raise SystemExit("QSA TRT-LLM resolver docstring is invalid")
docstring_end += 3
source = source[:docstring_end] + (
    "\n    from sglang.srt.utils import is_sm121\n"
    "\n"
    "    # dspark: SM121 must not use TRT-LLM sparse decode\n"
    "    # (sglang#36806 / #36845). That path silently emits token id 0\n"
    "    # at long context on GB10.\n"
    "    if is_sm121():\n"
    "        return None\n"
) + source[docstring_end:]
if marker not in source:
    raise SystemExit("QSA TRT-LLM SM121 guard was not applied")
path.write_text(source)
