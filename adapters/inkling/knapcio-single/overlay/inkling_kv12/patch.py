"""Patch points of inkling_kv12.

1. sglang.srt.model_executor.model_runner.capture_cuda_graphs (the name model_runner imported): before the TARGET
   runner captures anything (prefill graphs are captured first, then decode / verify), one ops.LayerPack per global
   cache layer is allocated (the 8 RadixAttention objects of the non-local Inkling2Attention modules), keyed by the
   layer's paged key-buffer data pointer, so every captured writer and reader sees the side pool.
2. FlashAttentionBackend.forward_extend and forward_decode: after the stock call (which stores K / V first), for a
   layer with a side pool, ops.clear_written sets FLAG := 0 for every page that holds a written position (prefill,
   chunked / mixed extend, target verify, decode), whichever attention kernel the call used.
3. Other raw writers: MHAView.load_cpu_copy (retraction restore, FLAG := 0 for those pages), Band._move_pages
   (unified-pool compaction copies pages: FLAG := 0 for source and destination pages), snapshot.restore_from_cpu
   (FLAG := 0 everywhere). HiCache, LMCache and PD disaggregation are refused at pre-allocation.
4. sglang.kernels.ops.attention.inkling2_attn.rel_bias_partials (compact decode / verify attention): a call for a
   layer with a side pool (BF16 paged cache, page 128, real (tokens, heads, 1024) rel bias, no window, SM12x, not
   stock's 128-key single-query decode) runs ops.pack_step + ops.kv12_partials; everything else passes through.

Every per-call action is device work on fixed grids (graph-safe); the host only reads static shapes.
"""

import inspect
import os
import sys

OPS_MODULE = "sglang.kernels.ops.attention.inkling2_attn"
BACKEND_MODULE = "sglang.srt.layers.attention.flashattention_backend"
RUNNER_MODULE = "sglang.srt.model_executor.model_runner"
MHA_MODULE = "sglang.srt.mem_cache.kv.layout.mha"
BAND_MODULE = "sglang.srt.mem_cache.kv.alloc.band"
SNAPSHOT_MODULE = "sglang.srt.mem_cache.kv.memory.snapshot"

PARTIALS_PARAMS = (
    "q", "k_cache", "v_cache", "page_table", "seqused_k", "rel_bias", "softmax_scale", "out",
    "window_size_left", "verify",
)
EXTENT = 1024

STORE = {}  # paged key-buffer data_ptr -> ops.LayerPack
STATS = {"kv12": 0, "passthrough": 0, "no_pool_global": 0, "cleared": 0, "restored": 0, "moved": 0}
CONFIG = {"max_pages": 0, "qcap": 8192, "splits": 0, "pack_programs": 96, "prefetch": False, "warps": 4,
          "stats_s": 60}
_SM12 = {}
_LOGGED = set()
_STATS_THREAD = []


def _log(msg):
    sys.stderr.write(f"[inkling_kv12 pid={os.getpid()}] {msg}\n")
    sys.stderr.flush()


def _once(key, msg):
    if key not in _LOGGED:
        _LOGGED.add(key)
        _log(msg)


def _check_params(fn, expected, what):
    params = tuple(inspect.signature(fn).parameters)
    if params != expected:
        raise RuntimeError(f"inkling_kv12: {what} signature changed: {params}")


def _is_sm12(device):
    import torch

    key = device.index
    if key not in _SM12:
        _SM12[key] = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] == 12
    return _SM12[key]


def _lookup(k_cache):
    lp = STORE.get(k_cache.data_ptr())
    if lp is None or lp.key != (k_cache.data_ptr(), tuple(k_cache.shape), tuple(k_cache.stride())):
        return None
    return lp


# ------------------------------------------------------------------------------------------------ 1. pre-allocation


def preallocate(model_runner):
    import torch
    from torch import nn

    from . import ops

    if getattr(model_runner, "is_draft_worker", False):
        return
    sa = getattr(model_runner, "server_args", None)
    for attr in ("enable_hierarchical_cache", "enable_lmcache"):
        if getattr(sa, attr, False):
            raise RuntimeError(f"inkling_kv12: not supported with {attr} (external KV load-back writes raw KV)")
    mode = getattr(sa, "disaggregation_mode", None)
    if mode not in (None, "null", "none"):
        raise RuntimeError(f"inkling_kv12: not supported with disaggregation_mode={mode!r}")
    backend = getattr(model_runner, "attn_backend", None)
    for _ in range(3):  # the Inkling hybrid (short-conv) backend wraps the FA4 one as full_attn_backend
        if backend is None or hasattr(backend, "get_paged_mha_kv_cache"):
            break
        backend = getattr(backend, "full_attn_backend", None)
    if backend is None or not hasattr(backend, "get_paged_mha_kv_cache"):
        raise RuntimeError(f"inkling_kv12: no FA attention backend with a paged MHA cache under "
                           f"{type(getattr(model_runner, 'attn_backend', None)).__name__}")
    layers = []
    for module in model_runner.model.modules():
        if type(module).__name__ == "Inkling2Attention" and not getattr(module, "is_local", True):
            atts = list(module.attn) if isinstance(module.attn, nn.ModuleList) else [module.attn]
            layers += atts
    if not layers:
        raise RuntimeError("inkling_kv12: no global Inkling2Attention layers found in the target model")
    total = 0
    for ra in layers:
        k_cache, _ = backend.get_paged_mha_kv_cache(ra)
        if k_cache.dtype != torch.bfloat16 or k_cache.shape[1] != ops.PAGE or k_cache.shape[3] != 128:
            _log(f"layer {ra.layer_id}: cache {tuple(k_cache.shape)} {k_cache.dtype} not supported, stays stock")
            continue
        if k_cache.data_ptr() in STORE:
            continue
        lp = ops.LayerPack(k_cache, CONFIG["max_pages"], CONFIG["qcap"])
        STORE[k_cache.data_ptr()] = lp
        total += lp.nbytes()
    torch.cuda.synchronize()
    ids = sorted(ra.layer_id for ra in layers)
    some = next(iter(STORE.values()), None)
    _log(f"side pools for global cache layers {ids}: {len(STORE)} x {(some.ns if some else 0)} slots for "
         f"{(some.npages if some else 0)} pages, {total / 2**30:.2f} GiB")


def _stats_loop(interval, dev):
    """Every `interval` s: coverage counters summed over the global layers (copied on a side stream, so the serving
    stream is never synchronised) and the eager hook counters."""
    import time

    import torch

    torch.cuda.set_device(dev)
    stream = torch.cuda.Stream(device=dev)
    host = torch.zeros((8,), dtype=torch.int64).pin_memory()
    # counters are read while graphs replay on the serving stream: values are approximate (mid-update), never synced
    prev = None
    while True:
        time.sleep(interval)
        try:
            tot = [0] * 8
            with torch.cuda.stream(stream):
                for lp in list(STORE.values()):
                    host.copy_(lp.stats, non_blocking=True)
                    stream.synchronize()
                    vals = host.tolist()
                    for i in range(8):
                        tot[i] = max(tot[i], vals[i]) if i == 6 else tot[i] + vals[i]
            if prev is not None and tot[3] == prev[3]:
                continue
            d = [t - (p if p is not None else 0) for t, p in zip(tot, prev or [None] * 8)]
            tiles = max(1, d[3])
            some = next(iter(STORE.values()))
            _log(f"coverage last {interval}s: tiles {d[3]} packed {100 * d[0] / tiles:.1f} % raw-unpacked "
                 f"{100 * d[1] / tiles:.1f} % raw-overflow {100 * d[2] / tiles:.1f} % | page-heads packed {d[4]} "
                 f"overflow {d[5]} | slots stolen {d[7]} | max page id {tot[6]} (pages {some.npages}, slots "
                 f"{some.ns}) | hooks {STATS}")
            prev = tot
        except Exception as exc:  # noqa: BLE001
            _log(f"stats thread stopped: {exc!r}")
            return


def make_capture(orig):
    def capture_cuda_graphs(*args, **kwargs):
        runner = kwargs.get("model_runner", args[0] if args else None)
        if runner is not None:
            preallocate(runner)
        result = orig(*args, **kwargs)
        # the counter poller starts only after the graphs are captured (no host sync may overlap a capture)
        if CONFIG["stats_s"] > 0 and STORE and not _STATS_THREAD:
            import threading

            dev = next(iter(STORE.values())).stats.device
            _STATS_THREAD.append(threading.Thread(target=_stats_loop, args=(CONFIG["stats_s"], dev), daemon=True,
                                                  name="inkling_kv12_stats"))
            _STATS_THREAD[0].start()
        return result

    capture_cuda_graphs.__wrapped__ = orig
    capture_cuda_graphs._inkling_kv12 = True
    return capture_cuda_graphs


# ------------------------------------------------------------------------------------------------ 2. / 3. invalidation


def _clear_after_store(backend, q, k, layer, forward_batch, save_kv_cache):
    from . import ops

    if not STORE or not save_kv_cache or k is None:
        return
    key_cache, _ = backend.get_paged_mha_kv_cache(layer)
    lp = _lookup(key_cache)
    if lp is None:
        return
    md = backend.forward_metadata
    seqlens = md.cache_seqlens_int32
    cu_q = getattr(md, "cu_seqlens_q", None)
    if cu_q is not None and cu_q.shape[0] == seqlens.shape[0] + 1:
        ops.clear_written(lp, md.page_table, seqlens, cu_q=cu_q)
    else:
        bs = max(1, seqlens.shape[0])
        ops.clear_written(lp, md.page_table, seqlens, qlen=max(1, q.shape[0] - bs + 1))  # bounds any real length
    STATS["cleared"] += 1


def make_forward(orig):
    def forward(self, q, k, v, layer, forward_batch, save_kv_cache=True, *args, **kwargs):
        result = orig(self, q, k, v, layer, forward_batch, save_kv_cache, *args, **kwargs)
        _clear_after_store(self, q, k, layer, forward_batch, save_kv_cache)
        return result

    forward.__wrapped__ = orig
    forward._inkling_kv12 = True
    return forward


def make_load_cpu_copy(orig):
    from . import ops

    def load_cpu_copy(self, kv_cache_cpu, indices, *args, **kwargs):
        result = orig(self, kv_cache_cpu, indices, *args, **kwargs)
        if STORE:
            import torch

            pages = torch.as_tensor(indices) // ops.PAGE
            for lp in STORE.values():
                ops.clear_pages(lp, pages)
            STATS["restored"] += 1
        return result

    load_cpu_copy.__wrapped__ = orig
    load_cpu_copy._inkling_kv12 = True
    return load_cpu_copy


def make_move_pages(orig):
    from . import ops

    def _move_pages(self, src_pages, dst_pages):
        orig(self, src_pages, dst_pages)
        if STORE:  # compaction copied raw pages: the destinations' (and sources') packed state is unknown
            for lp in STORE.values():
                ops.clear_pages(lp, dst_pages)
                ops.clear_pages(lp, src_pages)
            STATS["moved"] += 1

    _move_pages.__wrapped__ = orig
    _move_pages._inkling_kv12 = True
    return _move_pages


def make_restore(orig):
    from . import ops

    def restore_from_cpu(*args, **kwargs):
        result = orig(*args, **kwargs)
        for lp in STORE.values():  # snapshot restore scatters raw rows: forget every packed page (repacked lazily)
            ops.clear_all(lp)
        STATS["restored"] += 1
        return result

    restore_from_cpu.__wrapped__ = orig
    restore_from_cpu._inkling_kv12 = True
    return restore_from_cpu


# ------------------------------------------------------------------------------------------------ 4. attention


def make_partials(orig):
    import torch

    from . import ops

    def rel_bias_partials(q, k_cache, v_cache, page_table, seqused_k, rel_bias, softmax_scale, out, *,
                          window_size_left, verify):
        if (
            STORE
            and window_size_left is None
            and k_cache.dim() == 4
            and k_cache.dtype == torch.bfloat16
            and rel_bias.shape[-1] == EXTENT
            and seqused_k.shape[0] > 0
            and q.shape[0] % seqused_k.shape[0] == 0
            and _is_sm12(q.device)
        ):
            lp = _lookup(k_cache)
            if lp is None:
                STATS["no_pool_global"] += 1
                _once("nopool", f"global-layer call without a side pool (cache {tuple(k_cache.shape)}): stock")
            elif v_cache.stride() == k_cache.stride():
                bs = seqused_k.shape[0]
                qlen = q.shape[0] // bs
                sms = torch.cuda.get_device_properties(q.device).multi_processor_count
                splits, block_n = ops.stock_config(bs, k_cache.shape[2], qlen, page_table.shape[1] * ops.PAGE, sms)
                if block_n in (32, 64):  # stock's 128-key single-query decode passes through (bit-identity)
                    if CONFIG["splits"] > 0:
                        splits = CONFIG["splits"]
                    qmax = q.shape[0] - bs + 1  # bounds any request's real query length (ragged-safe)
                    STATS["kv12"] += 1
                    ops.pack_step(lp, k_cache, v_cache, page_table, seqused_k, qmax,
                                  pack_programs=CONFIG["pack_programs"])
                    return ops.kv12_partials(lp, q, k_cache, v_cache, page_table, seqused_k, rel_bias,
                                             softmax_scale, config=(splits, block_n), qmax=qmax,
                                             prefetch=CONFIG["prefetch"], num_warps=CONFIG["warps"])
        STATS["passthrough"] += 1
        return orig(q, k_cache, v_cache, page_table, seqused_k, rel_bias, softmax_scale, out,
                    window_size_left=window_size_left, verify=verify)

    rel_bias_partials.__wrapped__ = orig
    rel_bias_partials._inkling_kv12 = True
    return rel_bias_partials


# ------------------------------------------------------------------------------------------------ install


def patch_ops(module):
    orig = module.rel_bias_partials
    if getattr(orig, "_inkling_kv12", False):
        return
    _check_params(orig, PARTIALS_PARAMS, "rel_bias_partials")
    module.rel_bias_partials = make_partials(orig)


def patch_backend(module):
    cls = module.FlashAttentionBackend
    if not hasattr(cls, "get_paged_mha_kv_cache"):
        raise RuntimeError("inkling_kv12: FlashAttentionBackend.get_paged_mha_kv_cache is missing")
    for name in ("forward_extend", "forward_decode"):
        orig = getattr(cls, name)
        if getattr(orig, "_inkling_kv12", False):
            continue
        params = tuple(inspect.signature(orig).parameters)
        if params[:7] != ("self", "q", "k", "v", "layer", "forward_batch", "save_kv_cache"):
            raise RuntimeError(f"inkling_kv12: FlashAttentionBackend.{name} signature changed: {params}")
        setattr(cls, name, make_forward(orig))


def patch_runner(module):
    orig = getattr(module, "capture_cuda_graphs", None)
    if orig is None:
        raise RuntimeError("inkling_kv12: model_runner.capture_cuda_graphs is missing")
    if getattr(orig, "_inkling_kv12", False):
        return
    if "model_runner" not in inspect.signature(orig).parameters:
        raise RuntimeError("inkling_kv12: capture_cuda_graphs signature changed")
    module.capture_cuda_graphs = make_capture(orig)


def patch_mha(module):
    cls = module.MHAView
    orig = cls.load_cpu_copy
    if getattr(orig, "_inkling_kv12", False):
        return
    params = tuple(inspect.signature(orig).parameters)
    if params[:3] != ("self", "kv_cache_cpu", "indices"):
        raise RuntimeError(f"inkling_kv12: MHAView.load_cpu_copy signature changed: {params}")
    cls.load_cpu_copy = make_load_cpu_copy(orig)


def patch_band(module):
    cls = module.Band
    orig = cls._move_pages
    if getattr(orig, "_inkling_kv12", False):
        return
    if tuple(inspect.signature(orig).parameters) != ("self", "src_pages", "dst_pages"):
        raise RuntimeError("inkling_kv12: Band._move_pages signature changed")
    cls._move_pages = make_move_pages(orig)


def patch_snapshot(module):
    orig = module.restore_from_cpu
    if getattr(orig, "_inkling_kv12", False):
        return
    module.restore_from_cpu = make_restore(orig)


def register(config):
    from . import _hook

    CONFIG.update(config)
    _hook.when_imported(OPS_MODULE, patch_ops)
    _hook.when_imported(BACKEND_MODULE, patch_backend)
    _hook.when_imported(RUNNER_MODULE, patch_runner)
    _hook.when_imported(MHA_MODULE, patch_mha)
    _hook.when_imported(BAND_MODULE, patch_band)
    _hook.when_imported(SNAPSHOT_MODULE, patch_snapshot)
