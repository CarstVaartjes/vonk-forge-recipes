#!/usr/bin/env python3
"""DeepSeek-V4.1 sparse-MLA kernel envelope on SM12x (GB10): 64-state pages,
text-only prefill windows.

vLLM's DeepSeek-V4 indexer declares a 64-token kernel block on Hopper and 128
everywhere else, and the FlashInfer sparse-MLA DSV41 backend declares 128
only. The DeepGEMM build in this image (SM120) asserts
`block_kv == 32 or block_kv == 64` in its paged MQA logits kernel, so with
128-token blocks every compress-ratio-1 indexer layer (layer 20 and the
index-only layers) dies on the first decode:

    Assertion error (deepgemm-src/csrc/apis/attention.hpp:262): block_kv == 32 or block_kv == 64

The attention kernels gather by token slot and do not care about the page
size (the sliding-window cache already uses 32-token blocks; the native
3-Spark SGLang recipe runs the same FlashInfer SM120 kernel on 64-token
pages). This patch lets both SM12 backends accept a 64-token kernel block;
start.sh passes `--block-size ${KV_BLOCK_SIZE:-64}`.
"""
from __future__ import annotations

from pathlib import Path
import sys

MARK = "dsv41-sm120-block64"

# Helpers appended once to the end of a patched file (sentinel = def name).
ATTN_HELPER = """

# [dsv41-sm120-block64]
def _dsv41_swa_block_size() -> int:
    \"\"\"SWA cache page: 64 tokens on SM12x (FlashInfer SM120 decode dispatch), else 32.\"\"\"
    from vllm.platforms import current_platform

    return 64 if current_platform.is_device_capability_family(120) else 32


def _dsv41_cr_block_size(block_size: int, compress_ratio: int) -> int:
    \"\"\"64 states per page on SM12x: FlashInfer's SM120 dual-cache prefill and
    DeepGEMM's paged indexer both take 64-state pages only (DeepGEMM also 32),
    so a compress-ratio-r cache uses r x the token block.\"\"\"
    from vllm.platforms import current_platform

    if current_platform.is_device_capability_family(120) and int(compress_ratio) > 1:
        return int(block_size) * int(compress_ratio)
    return int(block_size)


def _dsv41_text_only_swa() -> bool:
    \"\"\"FlashInfer SM120 has no prefill kernel for the vision-widened SWA window
    (window + vision_max_n_token = 1152); on SM12x keep the 128-wide text window
    (serve with LANGUAGE_MODEL_ONLY=1).\"\"\"
    from vllm.platforms import current_platform

    return bool(current_platform.is_device_capability_family(120))
"""

SWA_META_HELPER = """

# [dsv41-sm120-block64]
def _dsv41_text_only_swa() -> bool:
    from vllm.platforms import current_platform

    return bool(current_platform.is_device_capability_family(120))
"""

JOBS = [
    (
        # Compressed-KV cache spec: block = 64 x compress_ratio tokens on SM12x.
        "models/deepseek_v4_1/attention.py",
        "        return MLAAttentionSpec(\n"
        "            block_size=vllm_config.cache_config.block_size,\n"
        "            num_kv_heads=1,\n"
        "            head_size=self.head_dim,\n"
        "            dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.kv_cache_torch_dtype,\n",
        "        return MLAAttentionSpec(\n"
        "            block_size=_dsv41_cr_block_size(  # [dsv41-sm120-block64]\n"
        "                vllm_config.cache_config.block_size, self.compress_ratio\n"
        "            ),\n"
        "            num_kv_heads=1,\n"
        "            head_size=self.head_dim,\n"
        "            dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.kv_cache_torch_dtype,\n",
        ATTN_HELPER,
    ),
    (
        # Indexer cache spec: same scaling.
        "models/deepseek_v4_1/attention.py",
        "        return MLAAttentionSpec(\n"
        "            block_size=self.cache_config.block_size,\n"
        "            num_kv_heads=1,\n"
        "            head_size=self.head_dim,\n"
        "            dtype=self.dtype,\n"
        "            tokens_per_state=self.compress_ratio,\n",
        "        return MLAAttentionSpec(\n"
        "            block_size=_dsv41_cr_block_size(  # [dsv41-sm120-block64]\n"
        "                self.cache_config.block_size, self.compress_ratio\n"
        "            ),\n"
        "            num_kv_heads=1,\n"
        "            head_size=self.head_dim,\n"
        "            dtype=self.dtype,\n"
        "            tokens_per_state=self.compress_ratio,\n",
        ATTN_HELPER,
    ),
    (
        # Attention layer: no image widening of the SWA window on SM12x.
        "models/deepseek_v4_1/attention.py",
        "        self.max_image_tokens = (\n"
        "            getattr(config, \"vision_max_n_token\", 0)\n"
        "            if getattr(config, \"vision_n_layers\", 0) > 0\n"
        "            else 0\n"
        "        )\n",
        "        self.max_image_tokens = (  # [dsv41-sm120-block64] text-only on SM12x\n"
        "            0\n"
        "            if _dsv41_text_only_swa()\n"
        "            else getattr(config, \"vision_max_n_token\", 0)\n"
        "            if getattr(config, \"vision_n_layers\", 0) > 0\n"
        "            else 0\n"
        "        )\n",
        ATTN_HELPER,
    ),
    (
        # SWA metadata builder: prefill index width = window only on SM12x.
        "v1/attention/backends/mla/sparse_swa.py",
        "        self.max_image_tokens = (\n"
        "            getattr(hf_config, \"vision_max_n_token\", 0)\n"
        "            if getattr(hf_config, \"vision_n_layers\", 0) > 0\n"
        "            else 0\n"
        "        )\n",
        "        self.max_image_tokens = (  # [dsv41-sm120-block64] text-only on SM12x\n"
        "            0\n"
        "            if _dsv41_text_only_swa()\n"
        "            else getattr(hf_config, \"vision_max_n_token\", 0)\n"
        "            if getattr(hf_config, \"vision_n_layers\", 0) > 0\n"
        "            else 0\n"
        "        )\n",
        SWA_META_HELPER,
    ),
    (
        # The sliding-window MLA cache: vLLM builds it with 32-token pages; the
        # FlashInfer SM120 DSV4 decode kernel only dispatches for
        # page_block_size == 64 ("SM120 sparse-MLA has no decode kernel for
        # this shape ... page_block_size=32"). Any multiple of 32 is fine for
        # the other backends, so use 64 on SM12x.
        "models/deepseek_v4_1/attention.py",
        "            backend_cls=self.swa_backend_cls,\n"
        "            block_size=32,\n"
        "        )\n",
        "            backend_cls=self.swa_backend_cls,\n"
        "            block_size=_dsv41_swa_block_size(),  # [dsv41-sm120-block64]\n"
        "        )\n",
        ATTN_HELPER,
    ),
    (
        # Escape hatch: DSV41_SKIP_MIXED_WARMUP=1 skips the mixed prefill+decode
        # sparse-MLA warm-up (boot 11 on 2026-09-12 hung both ranks inside it
        # with 1.5 GiB CUDA-free on the worker). The kernels then JIT on the
        # first real mixed batch instead.
        "model_executor/warmup/flashinfer_sparse_mla_warmup.py",
        '''    mixed_tokens = _clamp_warmup_tokens(_SPARSE_MLA_MIXED_WARMUP_TOKENS, max_tokens)
    if mixed_tokens <= 0:
        return

    logger.info(
        "Warming up DeepSeek V4 sparse MLA attention for mixed tokens=%s.",
        mixed_tokens,
    )
''',
        '''    mixed_tokens = _clamp_warmup_tokens(_SPARSE_MLA_MIXED_WARMUP_TOKENS, max_tokens)
    if mixed_tokens <= 0:
        return
    if __import__("os").environ.get("DSV41_SKIP_MIXED_WARMUP", "0") == "1":  # dsv41-sm120-block64
        logger.info("DSV41_SKIP_MIXED_WARMUP=1: skipping the mixed sparse MLA warm-up.")
        return

    logger.info(
        "Warming up DeepSeek V4 sparse MLA attention for mixed tokens=%s.",
        mixed_tokens,
    )
''',
        None,
    ),
    (
        # No auxiliary CUDA streams for the attention input projections /
        # indexer / compressor. Every exllamav3 trellis kernel (exl3_gemm
        # split-K tile locks, exl3_moe barriers) uses ONE lock buffer per
        # device, so two EXL3 kernels running at once on different streams
        # corrupt each other's locks and spin forever. With this all-EXL3
        # checkpoint the aux-stream GEMMs at layers >= 2 (small batches only:
        # VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD) deadlocked every boot on
        # 2026-09-12 (cuda-gdb: two exl3_gemm_kernel grids Active, thread 0
        # spinning in barrier_acquire). None = the ROCm serial path.
        "models/deepseek_v4_1/nvidia/model.py",
        '''        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]
''',
        '''        aux_stream_list = (  # dsv41-sm120-block64: EXL3 kernels share one lock buffer
            None
            if __import__("os").environ.get("DSV41_EXL3_SERIAL_STREAMS", "1") == "1"
            else [torch.cuda.Stream() for _ in range(3)]
        )
''',
        None,
    ),
    (
        # Same for the V4 MTP stack (DSpark draft layers mirror the model).
        "models/deepseek_v4/nvidia/mtp.py",
        '''        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]
''',
        '''        aux_stream_list = (  # dsv41-sm120-block64: EXL3 kernels share one lock buffer
            None
            if __import__("os").environ.get("DSV41_EXL3_SERIAL_STREAMS", "1") == "1"
            else [torch.cuda.Stream() for _ in range(3)]
        )
''',
        None,
    ),
    (
        # Debug bisect: DSV41_ENGRAM_DISABLE=1 passes the residual stream
        # through the Engram layers untouched (rows are still staged).
        "models/deepseek_v4_1/common/engram.py",
        '''        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        num_kv_tokens = hash_ids.shape[0]
''',
        '''        if __import__("os").environ.get("DSV41_ENGRAM_DISABLE", "0") == "1":  # dsv41-sm120-block64
            return hidden_states
        kv = self.wkv(self.embed(hash_ids).flatten(-2))  # dsv41
        num_kv_tokens = hash_ids.shape[0]
''',
        None,
    ),
    (
        # Right-size the sparse-indexer prefill gather workspace (the GLM
        # recipe's GLM53_INDEXER_WORKSPACE=rightsize, for this indexer): stock
        # is max_model_len * 40 entries of 132 B (0.7 GB at 128k, 2.6 GB at
        # 500k, 5.3 GB at 1M) locked for the life of the process. The chunk
        # planner only needs the summed (compressed) prefix of the prefill
        # requests it packs into one chunk, i.e. at most max_num_seqs *
        # max_model_len, so that is the default factor here
        # (DSV41_INDEXER_PREFILL_FACTOR overrides; 40 restores stock).
        "v1/attention/backends/mla/indexer.py",
        '''    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    return max_model_len * 40
''',
        '''    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    _factor = __import__("os").environ.get("DSV41_INDEXER_PREFILL_FACTOR", "")  # dsv41-sm120-block64
    if _factor.strip():
        return max_model_len * max(1, int(_factor))
    return max_model_len * max(1, min(40, int(vllm_config.scheduler_config.max_num_seqs)))
''',
        None,
    ),
    (
        # GB10: persistent_topk oversubscribes at long contexts ("persistent_topk
        # would oversubscribe and the FilteredTopK fallback requires >=128KB smem
        # per block (have 101376)", 500k boot on 2026-09-12); the GLM recipe
        # disables it on GB10 the same way. top_k_per_row_decode is the fallback.
        "model_executor/layers/sparse_attn_indexer.py",
        '''        use_persistent_topk = current_platform.is_cuda() and topk_tokens in (
            512,
            1024,
            2048,
        )
''',
        '''        use_persistent_topk = (  # dsv41-sm120-block64: GB10 smem cannot host it at long seqs
            current_platform.is_cuda()
            and topk_tokens in (512, 1024, 2048)
            and not current_platform.is_device_capability_family(120)
        )
''',
        None,
    ),
    (
        "model_executor/layers/sparse_attn_indexer_kpool.py",
        '''        if current_platform.is_cuda() and select_k in (512, 1024, 2048):
''',
        '''        if (  # dsv41-sm120-block64: no persistent_topk on GB10
            current_platform.is_cuda()
            and select_k in (512, 1024, 2048)
            and not current_platform.is_device_capability_family(120)
        ):
''',
        None,
    ),
    (
        "v1/attention/backends/mla/indexer.py",
        "        return [64 if current_platform.is_device_capability_family(90) else 128]\n",
        "        # [dsv41-sm120-block64] SM12x: the kernel block equals the manager block\n"
        "        # (64 tokens for ratio-1 caches, 128 for ratio-2), so DeepGEMM's paged\n"
        "        # MQA logits always see 64 states per page and the packed BLHNC layout\n"
        "        # never has to split a manager block.\n"
        "        if current_platform.is_device_capability_family(120):\n"
        "            return [64, 128]\n"
        "        return [64] if current_platform.is_device_capability_family(90) else [128]\n",
        None,
    ),
    (
        "models/deepseek_v4_1/nvidia/flashinfer_sparse.py",
        "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
        "        return [128]\n",
        "    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:\n"
        "        # [dsv41-sm120-block64] SM12x also runs on 64-token pages (--block-size 64)\n"
        "        from vllm.platforms import current_platform\n"
        "\n"
        "        if current_platform.is_device_capability_family(120):\n"
        "            return [64, 128]\n"
        "        return [128]\n",
        None,
    ),
]


def apply(text: str, old: str, new: str, helper: str | None = None) -> tuple[str, str]:
    # a file may carry several jobs: "skipped" only when this job's edit is present
    if new.strip() and new in text:
        return text, "skipped"
    n = text.count(old)
    if n != 1:
        return text, f"missing:{old.strip()[:60]!r} count={n}"
    out = text.replace(old, new, 1)
    if helper:
        sentinel = helper.strip().splitlines()[1] if helper.strip().startswith("#") else helper.strip().splitlines()[0]
        if sentinel not in out:
            if not out.endswith("\n"):
                out += "\n"
            out += helper
    return out, "applied"


def main() -> int:
    try:
        import vllm

        root = Path(vllm.__file__).resolve().parent
    except Exception as exc:
        print(f"WARN: vllm not importable; skip sm120 block64 patch ({exc})", file=sys.stderr)
        return 0
    rc = 0
    for rel, old, new, helper in JOBS:
        path = root / rel
        if not path.is_file():
            print(f"WARN: sm120 block64: {rel} missing", file=sys.stderr)
            rc = 1
            continue
        out, status = apply(path.read_text(), old, new, helper)
        if status == "applied":
            path.write_text(out)
            print(f"sm120 block64: applied ({rel})")
        elif status == "skipped":
            print(f"sm120 block64: already applied ({rel})")
        else:
            print(f"WARN: sm120 block64 {rel}: {status}", file=sys.stderr)
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
