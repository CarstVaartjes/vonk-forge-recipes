"""The packed global store on SGLang's static (non-unified) pools (KV12S_POOL=static).

The eight global target visits keep the v2.7.1 envelope, device writer and readers;
only their storage binding differs from the unified one (patch.py):

- sizing: HybridSWAPoolConfigurator prices a full-side target token at the packed
  envelope (8 x 3,108 B instead of 8 x 4,096 B) and takes the fixed packed bytes
  (staging, overflow and fallback arenas, encoder workspace, prefix scratch) off the
  budget first;
- storage: the target SWAKVPool's full sub-pool is a StaticPackedPool (one envelope
  buffer, one DeviceLayer per visit); the SWA sub-pool, the draft's private pools and
  the short-conv state stay stock BF16, so the stock static draft fold stays captured;
- lifetime: every free of a full page reaches PagedTokenToKVPoolAllocator._release_page_ids
  (free, free_page_ids, free_group_end); its descriptors and side slots are released
  on the forward stream, after the forward in flight and before the page's next
  writer. Static pages never move; retraction re-prefills.

Static token ids are physical rows, page-aligned (page 0 is the padding page).
"""

import logging

import torch

from . import IncompatibleImage
from .storage import DIM, PAGE, Layout, arenas_bytes, stage_slots, workspace_bytes

LOG = logging.getLogger("inkling_kv12s")

VISITS = 8
HEADS = 8
# Static sizing prices the DSpark draft's window lanes (Inkling-2-Mini's MTP: one global
# lane and two window lanes) per window token, which is what they allocate, instead of
# stock's per full token. 0 keeps stock pricing.
import os as _os

DRAFT_WINDOW_LANES = int(_os.environ.get("INKLING_KV12S_DRAFT_WINDOW_LANES", "2"))
_FORWARD_STREAM = []  # the stream serving forwards run on (pressure.outermost notes it)
# Set while the target runner warms up and captures CUDA graphs: prefill then reads the
# packed prefix with the in-kernel loader in warmup too, so capture never meets a
# kernel specialization (and its first cudaFuncSetAttribute) for the first time.
CAPTURING = []


def note_forward_stream():
    if torch.cuda.is_available() and not torch.cuda.is_current_stream_capturing():
        stream = torch.cuda.current_stream()
        if not _FORWARD_STREAM or _FORWARD_STREAM[0] != stream:
            _FORWARD_STREAM[:] = [stream]


def token_bytes(heads=HEADS, visits=VISITS):
    """Packed bytes per full-side token over all global visits."""
    return visits * (Layout(heads).size // PAGE)


def cell_bytes(heads=HEADS, visits=VISITS):
    """What the sizing charges per full-side token: the envelopes plus one byte for the
    per-page overflow-failure flags (4 B per page and visit = 0.25 B per token)."""
    return token_bytes(heads, visits) + 1


def fixed_bytes(running, verify_rows, *, overlap, heads=HEADS, visits=VISITS):
    """Packed bytes outside the per-token envelopes, charged before the token budget."""
    from . import side_count
    from .patch import _FA4_WORKSPACE, _scratch_pages
    from .storage import EVENT_WORDS, FALLBACK_PAGES

    if _FA4_WORKSPACE:
        raise IncompatibleImage("inkling_kv12s: INKLING_KV12S_FA4_WORKSPACE is not supported on the static pools")
    overflow = side_count("INKLING_KV12S_OVERFLOW_HEADS", "2048", 1)
    slots = stage_slots(running, verify_rows, overlap=overlap) + FALLBACK_PAGES
    lanes = visits * (arenas_bytes(heads, slots, overflow) + 4 * (slots + overflow + EVENT_WORDS))
    scratch = 2 * max(0, _scratch_pages()) * PAGE * heads * DIM * 2
    sink = visits * Layout(heads).size  # the padding page 0 carries an envelope too
    return lanes + workspace_bytes(heads) + scratch + sink


def view_of(pool):
    """The packed view behind a target token pool, unified or static, else None."""
    arena = getattr(pool, "_arena", None)
    view = getattr(arena, "_inkling_kv12s", None)
    if view is None:
        full = getattr(pool, "full_kv_pool", None)
        if getattr(full, "is_inkling_kv12s", False):
            view = full
    return view


def make_pool_class():
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, unwrap_write_loc

    from .device import DeviceLayer
    from .storage import EVENT_WORDS, FALLBACK_PAGES
    from .swa_layout import _NoRowView

    class StaticPackedPool(MHATokenToKVPool):
        """Full-side target sub-pool: packed page envelopes, no token-row buffers."""

        is_inkling_kv12s = True
        device_owned = True
        page_major = False

        def __init__(self, *, inkling, **kwargs):
            self._inkling = inkling
            kwargs["enable_kv_cache_copy"] = False  # no token-row moves (linear DSpark)
            super().__init__(**kwargs)

        def _create_buffers(self):
            from . import side_count
            from .attention import PrefixScratch
            from .patch import _PREFIX_SCRATCH, _scratch_pages
            from .pressure import GUARD, SideGuard

            cfg = self._inkling
            if (self.page_size != PAGE or self.store_dtype != torch.bfloat16 or self.head_dim != DIM
                    or self.v_head_dim != DIM or self.layer_num != VISITS or self.post_capture_active
                    or self.use_hnd or self.is_quantized_kv_cache):
                raise IncompatibleImage("inkling_kv12s expects eight global BF16 visits with 128-wide heads/pages")
            heads = self.head_num
            overflow = side_count("INKLING_KV12S_OVERFLOW_HEADS", "2048", 1)
            slots = stage_slots(cfg["running"], cfg["verify_rows"], overlap=cfg["overlap"])
            envelope = Layout(heads).size
            stride = envelope * self.layer_num
            self.num_pages = (self.size + self.page_size) // self.page_size
            self.page_bytes = stride
            self.raw = torch.zeros(self.num_pages * stride, dtype=torch.uint8, device=self.device)
            locks = torch.zeros((self.layer_num, slots + FALLBACK_PAGES + overflow), dtype=torch.int32,
                                device=self.device)
            events = torch.zeros((self.layer_num, EVENT_WORDS), dtype=torch.int32, device=self.device)
            self.layers = tuple(
                DeviceLayer(self.raw, heads, stride, i * envelope, slots, overflow, FALLBACK_PAGES,
                            locks=locks[i], events=events[i])
                for i in range(self.layer_num)
            )
            self.owned_layers = self.layers
            from .patch import _DFLASH_SETTLE, _lane_group

            if _DFLASH_SETTLE == "batch":  # the DFLASH settle's lane table, built here and not on the first decode step
                _lane_group(self, self.owned_layers)
            self.k_buffer = _NoRowView([None] * self.layer_num)
            self.v_buffer = _NoRowView([None] * self.layer_num)
            scratch_pages = _scratch_pages()
            if scratch_pages > 0 and not _PREFIX_SCRATCH:
                _PREFIX_SCRATCH.append(PrefixScratch(scratch_pages, heads, self.device))
            GUARD[:] = [SideGuard(self.layers, locks, events, prefill_tokens=cfg["prefill_tokens"], page=PAGE)]
            LOG.info("inkling_kv12s: static packed store, %d visits x %d pages, %d B/token, %d staging slots "
                     "per visit, prefix scratch %d pages", self.layer_num, self.num_pages,
                     stride // PAGE, slots, scratch_pages)

        def get_kv_size_bytes(self):
            return self.raw.numel(), 0

        def _refuse(self, *args, **kwargs):
            raise RuntimeError("inkling_kv12s: a raw token-row access reached the packed global store")

        get_key_buffer = get_value_buffer = get_kv_buffer = _refuse
        get_contiguous_buf_infos = get_cpu_copy = load_cpu_copy = move_kv_cache = _refuse
        get_device_pool_infos = _refuse

        def get_v_head_dim(self):
            return DIM

        def _index(self, layer, layer_id_override):
            if layer_id_override is not None:
                return layer_id_override
            return self.model_layer_ids.index(layer.layer_id)

        def set_kv_buffer(self, layer, loc_info, cache_k, cache_v, k_scale=None, v_scale=None,
                          layer_id_override=None, dcp_kv_mask=None):
            from .patch import _FORWARD

            if dcp_kv_mask is not None:
                raise RuntimeError("inkling_kv12s expects unsharded KV writes")
            loc, _, _ = unwrap_write_loc(loc_info)
            visit = _FORWARD.get()
            self._store(self._index(layer, layer_id_override), loc, cache_k, cache_v, k_scale, v_scale,
                        tentative=visit is not None and visit.tentative)

        def set_kv_buffer_prefix_valid(self, layer, loc_2d, commit_lens, cache_k, cache_v,
                                       k_scale=None, v_scale=None, layer_id_override=None):
            if loc_2d.ndim != 2 or commit_lens.shape != (loc_2d.shape[0],):
                raise ValueError("inkling_kv12s: invalid prefix commit metadata")
            rows = torch.arange(loc_2d.shape[1], device=loc_2d.device)
            masked = torch.where(rows[None, :] < commit_lens[:, None], loc_2d, 0)
            self._store(self._index(layer, layer_id_override), masked, cache_k, cache_v, k_scale, v_scale,
                        tentative=False)

        def _store(self, index, loc, cache_k, cache_v, k_scale, v_scale, *, tentative):
            if cache_k.dtype != torch.bfloat16:
                if k_scale is not None:
                    cache_k = cache_k.div(k_scale)
                if v_scale is not None:
                    cache_v = cache_v.div(v_scale)
            self.layers[index].write(loc, cache_k, cache_v, tentative=tentative)

        def release_pages(self, page_ids, *, reset_locks=False):
            """Descriptors and side slots of freed pages, on the serving forward stream:
            after the forward in flight (which may still read or write them) and before
            the pages' next writer. The ids are made on this stream first, then handed
            over, so the scheduler cannot recycle them under the release kernels."""
            ids = page_ids.reshape(-1).to(device=self.device, dtype=torch.int64)  # a new tensor here
            if not ids.numel() and not reset_locks:
                return
            if not self.raw.is_cuda:
                self._release(ids, reset_locks)
                return
            current = torch.cuda.current_stream()
            stream = _FORWARD_STREAM[0] if _FORWARD_STREAM else current
            if stream != current:
                stream.wait_stream(current)
            with torch.cuda.stream(stream):
                self._release(ids, reset_locks)
            if stream != current:
                ids.record_stream(stream)

        def _release(self, ids, reset_locks):
            for lane in self.layers:
                if ids.numel():
                    lane.release(ids)
                if reset_locks:
                    lane.clear()

        def release_all(self):
            self.release_pages(torch.arange(1, self.num_pages, device=self.device), reset_locks=True)

    return StaticPackedPool


_POOL_CLASS = []


def pool_class():
    if not _POOL_CLASS:
        _POOL_CLASS.append(make_pool_class())
    return _POOL_CLASS[0]
