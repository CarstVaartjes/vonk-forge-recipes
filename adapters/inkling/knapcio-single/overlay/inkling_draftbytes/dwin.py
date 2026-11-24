"""dwin: the DSpark drafter's global attention layer reads an attention sink plus the last W tokens.

Inkling-2-Mini's drafter has three layers; ``local_layer_ids = [0, 2]``, so layer 1 is a full
(global) layer. Every draft forward (one per decode step, TARGET_VERIFY mode, 8 query rows per
request) attends over the request's whole context through the drafter's own K/V in that layer:
8 KV heads x 128 x (K + V) x BF16 = 4096 B per token, 0.54 GB at 128K, 1.05 GB at 255K.

With ``dwin:W[:S]`` that layer attends over the first S tokens (the sink) plus every page that holds
one of the last W tokens of the key sequence (which includes the 8-token draft block). Both are
whole pages of the paged KV cache, so the page table stays page-granular and FA4 keeps its paged
path:

  n         = cache_seqlens (context + draft block), P = page size
  sp        = ceil(S / P)                        sink pages (S = 0: no sink)
  wp        = ceil(W / P) + 1                    window columns (the last W tokens span at most this many pages)
  start     = max(sp, max(0, n - W) // P)        the page holding token n - W
  table'    = [table[0 .. sp-1], table[start .. start+wp-1]]       (columns clamped to the table width)
  n'        = n - (start - sp) * P               (only whole skipped pages are removed)

When start == sp this is the identity (the same pages in the same order, n' = n); that holds for
every n < W + (sp + 1) * P. Above it the layer reads at most sp + wp pages instead of ceil(n / P);
the kept tail holds between W and W + P - 1 tokens (all of the last W). If the source table is
not wider than sp + wp columns, the call is left stock (it cannot read more than the window).
RoPE is applied to K when it is written (absolute positions) and to Q at its absolute position,
and the drafter's attention has no position-dependent mask (bidirectional, no window, no bias),
so attending over a subset of pages is exactly attention over that subset of keys. The new table
has a static width (sp + wp), so the few tensor ops that build it are CUDA-graph safe: they read
the backend's static metadata buffers at replay. The FA4 call also gets max_seqlen_k =
(sp + wp) * P (static), so its SplitKV planner sizes the launch for the window instead of the
whole KV pool (INKLING_DRAFT_DWIN_MAXK=0 leaves the stock planning).

Exactness: this changes only the draft proposal q. Greedy verify emits the target argmax whatever
q is; sampled verify reads the step logits the proposal was drawn from. Cost is acceptance only.

Hook: FlashAttentionBackend.forward_extend is wrapped (class level). A call whose layer carries
the ``_idb_dwin`` mark (set at draft prepare on the drafter's global RadixAttention instance only),
in TARGET_VERIFY mode with topk <= 1, swaps a shallow copy of the forward metadata with the
windowed page_table / cache_seqlens_int32 / cu_seqlens_k for the duration of the stock call and
restores it afterwards. Every other call (the target model, the drafter's local layers) goes
straight to the stock method.
"""

from __future__ import annotations

import copy
import functools
from dataclasses import dataclass
from typing import Tuple

import torch

FA_MODULE = "sglang.srt.layers.attention.flashattention_backend"
MARK = "_idb_dwin"

STATS = {"windowed_calls": 0, "passthrough_marked": 0}
_LOGGED = set()


@dataclass(frozen=True)
class DwinCfg:
    window: int  # the last `window` tokens are always kept (whole pages: up to window + page - 1)
    sink: int  # tokens kept at the start (rounded up to whole pages; 0 = none)
    maxk: bool = True  # pass the window's static max_seqlen_k to FA4 (SplitKV planning)

    def pages(self, page_size: int) -> Tuple[int, int]:
        """(sink pages, window columns)."""
        return -(-self.sink // page_size), -(-self.window // page_size) + 1


def window_page_metadata(
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_size: int,
    sink_pages: int,
    window: int,
):
    """(page_table [bs, T] page ids, cache_seqlens [bs] tokens) -> the windowed
    (page_table [bs, sp + wp] int32, cache_seqlens [bs] int32, cu_seqlens_k [bs + 1] int32), wp = ceil(W / P) + 1;
    None when T <= sp + wp (the stock table already reads no more than the window)."""
    if page_table.dim() != 2:
        raise ValueError(f"page_table must be [bs, pages], got {tuple(page_table.shape)}")
    win_pages = -(-window // page_size) + 1
    bs, width = page_table.shape
    if width <= sink_pages + win_pages:
        return None
    n = cache_seqlens[:bs].to(torch.int32)
    start = torch.clamp(torch.div(torch.clamp(n - window, min=0), page_size, rounding_mode="floor"), min=sink_pages)
    # sink column j -> page j; window column sp + i -> page start + i. Built on the device every call (no cached
    # constants: a first call inside a graph capture must not allocate or copy outside the capture stream).
    ar = torch.arange(sink_pages + win_pages, dtype=torch.int32, device=page_table.device)
    cols = torch.where(ar < sink_pages, ar, ar - sink_pages + start[:, None])
    cols = torch.clamp(cols, max=width - 1)
    table = torch.gather(page_table, 1, cols.long()).to(torch.int32)
    new_len = (n - (start - sink_pages) * page_size).to(torch.int32)
    cu = torch.zeros(bs + 1, dtype=torch.int32, device=n.device)
    cu[1:] = torch.cumsum(new_len, 0, dtype=torch.int32)
    return table, new_len, cu


def reference_keys(cache_seqlen: int, page_size: int, sink_pages: int, window: int):
    """Logical token positions the windowed layer reads, in order, from first principles (for tests): every token
    of the sink pages and of every page that holds one of the last `window` tokens, below cache_seqlen."""
    n, P = cache_seqlen, page_size
    pages = set(range(sink_pages))
    pages |= {t // P for t in range(max(0, n - window), n)}
    return [t for t in range(n) if t // P in pages]


def _log_once(key, msg, say):
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    say(msg)


def patch_fa_backend(mod, *, require, replace, say) -> None:
    require(mod, ["FlashAttentionBackend", "FlashAttentionMetadata"], FA_MODULE)
    cls = mod.FlashAttentionBackend
    require(cls, ["forward_extend"], "FlashAttentionBackend")
    orig = cls.forward_extend

    @functools.wraps(orig)
    def forward_extend(self, q, k, v, layer, forward_batch, *args, **kwargs):
        cfg = getattr(layer, MARK, None)
        if cfg is None:
            return orig(self, q, k, v, layer, forward_batch, *args, **kwargs)
        meta = self.forward_metadata
        reason = None
        if not forward_batch.forward_mode.is_target_verify():
            reason = f"mode {forward_batch.forward_mode}"
        elif getattr(self, "topk", 1) > 1:
            reason = "topk > 1"
        elif getattr(self, "use_mla", False):
            reason = "MLA"
        elif meta is None or meta.page_table is None or meta.cache_seqlens_int32 is None:
            reason = "no page table"
        elif getattr(meta, "local_attn_metadata", None) is not None:
            reason = "local attention metadata"
        elif layer.sliding_window_size is not None and layer.sliding_window_size > -1:
            reason = "sliding-window layer"
        elif getattr(layer, "window_anchor_at_start", False):
            reason = "anchored layer"
        if reason is not None:
            STATS["passthrough_marked"] += 1
            _log_once(("pass", reason), f"dwin: stock call for the marked layer ({reason})", say)
            return orig(self, q, k, v, layer, forward_batch, *args, **kwargs)
        page_size = int(self.page_size)
        sp, wp = cfg.pages(page_size)
        built = window_page_metadata(meta.page_table, meta.cache_seqlens_int32, page_size, sp, cfg.window)
        if built is None:
            STATS["passthrough_marked"] += 1
            return orig(self, q, k, v, layer, forward_batch, *args, **kwargs)
        table, lens, cu = built
        windowed = copy.copy(meta)
        windowed.page_table = table
        windowed.cache_seqlens_int32 = lens
        windowed.cu_seqlens_k = cu
        STATS["windowed_calls"] += 1
        _log_once(
            ("on", page_size, sp, wp),
            f"dwin: drafter layer {layer.layer_id} reads {sp} sink + {wp} window pages of {page_size} "
            f"(table {tuple(meta.page_table.shape)} -> {tuple(table.shape)})",
            say,
        )
        inject = cfg.maxk and "flash_attn_with_kvcache" not in self.__dict__
        if inject:
            stock_fa = self.flash_attn_with_kvcache
            max_k = int(table.shape[1]) * page_size

            def flash_attn_with_kvcache(*a, **kw):
                if kw.get("max_seqlen_k") is None and kw.get("page_table") is table:
                    kw["max_seqlen_k"] = max_k
                return stock_fa(*a, **kw)

            self.flash_attn_with_kvcache = flash_attn_with_kvcache
        self.forward_metadata = windowed
        try:
            return orig(self, q, k, v, layer, forward_batch, *args, **kwargs)
        finally:
            self.forward_metadata = meta
            if inject:
                del self.flash_attn_with_kvcache

    replace(cls, "forward_extend", forward_extend, "FlashAttentionBackend.forward_extend")


def mark_draft_layers(model, cfg: DwinCfg) -> list:
    """Mark the drafter's global attention layers; returns their layer ids."""
    marked = []
    for layer in model.layers:
        attn = layer.self_attn.attn
        swa = getattr(attn, "sliding_window_size", None)
        if swa is None or swa < 0:
            setattr(attn, MARK, cfg)
            marked.append(int(attn.layer_id))
    return marked
