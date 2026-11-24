"""Inkling-2-Mini lossless 12-bit global-layer KV for decode / verify attention (SGLang on GB10, sm_121).

    INKLING_KV12=1                    on
    INKLING_KV12_MAX_PAGES=N          packed side slots per global layer (0 = one per page of the pool); any page
                                      can own a slot, LRU slots are reassigned to the pages being decoded
    INKLING_KV12_QCAP=N               pages packed per layer visit at most (default 8192; the rest next step)
    INKLING_KV12_SPLITS=N             split count of the attention kernel (0 = the stock heuristic, bit-identical)
    INKLING_KV12_PREFETCH=1           one-tile manual lookahead variant (also bit-identical; slower on GB10)
    INKLING_KV12_WARPS=4|8            warps of the attention kernel (default 4)
    INKLING_KV12_STATS_S=N            log coverage counters every N s (default 60, 0 = off)
    unset / empty / 0: install() does nothing and nothing is imported.

The 8 global visits (layers 5 / 11 / 17 / 23, each core pass its own cache) keep their raw BF16 KV; full committed
128-token pages are also packed into a side pool (sign + mantissa byte, 4-bit exponent code against the fixed
116..130 table, sparse escape stream), and decode / verify attention decodes packed pages in registers to the same
BF16 bits (raw tiles for the open page, unpacked or overflowed pages). Same math as stock; see README.md.
install() checks the patch targets in the installed sglang sources (without importing sglang), registers
post-import hooks and prints one line. Problems raise RuntimeError (sitecustomize then exits with code 78).
Install after inkling_vattn (this wrapper of rel_bias_partials must be the outer one).
"""

import importlib.util
import os
import sys

_SOURCE_CHECKS = (
    ("kernels/ops/attention/inkling2_attn.py", (
        "def rel_bias_partials(",
        "    return decode_partials(\n        q,\n        k_cache,\n        v_cache,\n        page_table,\n"
        "        seqused_k,\n        rel_bias,\n        softmax_scale,\n        window_size_left=window_size_left,\n    )",
    )),
    ("kernels/ops/attention/inkling2_decode_attention_sm90.py", (
        "def _decode_partials(",
        "    tl.extra.cuda.gdc_wait()\n    tl.extra.cuda.gdc_launch_dependents()\n    b = tl.program_id(0) // HEADS_K",
        "    max_splits = max(1, min(2 * sms // max(1, batch * heads_k), MAX_SPLITS))\n"
        "    block_n = 128 if qlen == 1 and span >= 128 * max_splits else 32\n"
        "    splits = min(triton.cdiv(span, block_n), max_splits)",
        "stat = (split * batch * QLEN + token) * (HEADS_K * GROUP) + qh",
        "        bias = tl.load(\n            Bias + token[:, None] * SB0 + qh[:, None] * SB1 + dist,\n"
        "            head_mask[:, None] & keep & (dist < EXTENT),\n            other=0,\n        ).to(tl.float32)",
        "        s = tl.where(keep, s + bias, float(\"-inf\"))",
        "        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)",
        "num_stages=2 if qlen == 1 else 3,",
    )),
    ("srt/model_executor/model_runner.py", (
        "capture_cuda_graphs,",
        "        capture = capture_cuda_graphs(\n            model_runner=self,",
    )),
    ("srt/model_executor/model_runner_components/cuda_graph_setup.py", (
        "def capture_cuda_graphs(\n    *,\n    model_runner: ModelRunner,",
        "    # cuda-graph capture: prefill before decode",
    )),
    ("srt/mem_cache/kv/layout/mha.py", (
        "    def load_cpu_copy(\n        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None\n    ):",
    )),
    ("srt/mem_cache/kv/alloc/band.py", (
        "    def _move_pages(self, src_pages: torch.Tensor, dst_pages: torch.Tensor) -> None:",
        "        self._move_pages(src_pages, dst_pages)",
    )),
    ("srt/mem_cache/kv/memory/snapshot.py", (
        "def restore_from_cpu(allocator, snapshot, indices, *, state_indices, request_row):",
    )),
    ("srt/mem_cache/kv/alloc/chain.py", (
        "        from sglang.srt.mem_cache.kv.memory.snapshot import restore_from_cpu",
    )),
    ("srt/layers/attention/flashattention_backend.py", (
        "    def forward_decode(",
        "from sglang.kernels.ops.attention.inkling2_attn import rel_bias_partials",
        "    def get_paged_mha_kv_cache(",
        "self.kv_layers.layer(layer.layer_id).store(",
        "                    *rel_bias_partials(\n                        q.view(-1, layer.tp_q_head_num, layer.head_dim),\n"
        "                        key_cache,\n                        value_cache,\n                        page_table,\n"
        "                        cache_seqlens,",
    )),
)

STATE = {"installed": False, "config": None}


def _log(msg):
    sys.stderr.write(f"[inkling_kv12 pid={os.getpid()}] {msg}\n")
    sys.stderr.flush()


def _on(value):
    return value.strip().lower() not in ("", "0", "off", "none", "false", "no")


def sglang_root():
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("inkling_kv12: the sglang package is not importable")
    return list(spec.submodule_search_locations)[0]


def check_sources(root=None):
    root = root or sglang_root()
    for rel, needles in _SOURCE_CHECKS:
        path = os.path.join(root, rel)
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except OSError as exc:
            raise RuntimeError(f"inkling_kv12: patch target {path} is missing") from exc
        missing = [n[:80] for n in needles if n not in text]
        if missing:
            raise RuntimeError(f"inkling_kv12: patch target {path} lacks {missing}")


def install(root=None):
    """Activate when INKLING_KV12 is on. Idempotent."""
    if STATE["installed"]:
        return STATE["config"]
    if not _on(os.environ.get("INKLING_KV12", "")):
        STATE["installed"] = True
        return None
    if _on(os.environ.get("INKLING_KV12S", "")):
        raise RuntimeError("INKLING_KV12 and INKLING_KV12S are mutually exclusive")
    config = {
        "max_pages": int(os.environ.get("INKLING_KV12_MAX_PAGES", 0)),
        "qcap": int(os.environ.get("INKLING_KV12_QCAP", 8192)),
        "splits": int(os.environ.get("INKLING_KV12_SPLITS", 0)),
        "pack_programs": int(os.environ.get("INKLING_KV12_PACK_PROGRAMS", 96)),
        "prefetch": _on(os.environ.get("INKLING_KV12_PREFETCH", "0")),
        "warps": int(os.environ.get("INKLING_KV12_WARPS", 4)),
        "stats_s": int(os.environ.get("INKLING_KV12_STATS_S", 60)),
    }
    if config["prefetch"]:
        raise RuntimeError("inkling_kv12: INKLING_KV12_PREFETCH was retired (the slot-indexed kernel has no lookahead)")
    if (config["qcap"] < 1 or config["max_pages"] < 0 or config["splits"] < 0 or config["pack_programs"] < 1
            or config["warps"] not in (4, 8)):
        raise RuntimeError(f"inkling_kv12: bad configuration {config}")
    check_sources(root)
    from . import patch

    patch.register(config)
    STATE.update(installed=True, config=config)
    _log(f"enabled max_pages={config['max_pages'] or 'all'} qcap={config['qcap']} "
         f"splits={config['splits'] or 'stock'}")
    return config
