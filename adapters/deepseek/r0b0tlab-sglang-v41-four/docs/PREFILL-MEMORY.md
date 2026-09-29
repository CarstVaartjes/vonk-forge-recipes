# Bounded prefill work on GB10

Active SM121 prefill is `_low_ratio_index_topk_torch`, selected because the
adapter forces `_is_sm100_or_newer` false (no tcgen05 / DeepGEMM fp8_fp4 MQA
on SM12x). Decode/verify stay on `_low_ratio_index_topk_sm90_decode` (Triton).
The dense-FP4 path (`_low_ratio_index_topk_dense`) and its 14 B×T×L coefficient
are **not** the measured byte model on this deployment.

The Torch indexer dequants visible compressed-K via
`get_low_ratio_index_k_dequant(layer, slots_j)` and scores with
`indexer.scores` → `[query_rows, visible_keys]` (heads already reduced). The
einsum intermediate is `[query_rows, index_n_heads, visible_keys]` bf16.
Upstream `_TORCH_INDEXER_SCORE_BUDGET_BYTES` only chunks **query** rows; K
dequant of the full prefix stayed unbounded. Overlay-v3 slices K, takes
per-slice topk, and merges (union of per-slice topk is exact).

The adaptive chunk hook **does fire** even when `/get_server_info` reports
`enable_dynamic_chunking: false` (that flag is the PP sizer). Scheduler line
3678 uses `self.dynamic_chunk_sizer` whenever `chunked_req` is set. Overlay-v2
set `cmax=2048` while the static first chunk was 512, so continuations **grew**
(`predict(0)==2048`; last 512k prefills logged `#new-token: 768` at processed
≈298752, which equals `BudgetChunkSizer(3e8,256,2048).predict(298752)`).
Overlay-v3 is shrink-only: `cmax` is capped at `--chunked-prefill-size`
(512 prod-c8 / 256 on production exclusive C1). Continuation chunks never exceed the
static first chunk. Token² is a scheduler bound, not a total-memory guarantee.

The candidate-copy / score budget default is restored to 1024 MiB (upstream
1 GiB). Overlay-v3 used that budget to size K slices, which yields
k_chunk=65536 at T=256 / H=32 and a `[T,H,K]` einsum of 1 GiB — 256k NIAH
NVRM'd in ~2 min. Overlay-v4 caps `k_chunk` at 2048 independently
(`DSV41_INDEXER_K_CHUNK_MAX`); einsum ~32 MiB. Token² is a scheduler bound,
not a total-memory guarantee.

References: pinned engine da64c5cbb8cf6bfd39be19da43573fdfd484c43a,
python/sglang/srt/managers/scheduler.py (`dynamic_chunk_sizer.predict`);
python/sglang/srt/layers/attention/deepseek_v4_backend.py
(`_low_ratio_index_topk_torch`). Independent discussion:
https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks/blob/79f656a65f189239cc575bf5c5d1b5cf579d4c41/docs/chunked-prefill-memory.md .
No community adapter code is copied.
