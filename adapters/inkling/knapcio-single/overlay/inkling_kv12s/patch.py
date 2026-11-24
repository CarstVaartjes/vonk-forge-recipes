"""SGLang unified-pool and FA4 bindings for the image used by Inkling-2-Mini."""

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import inspect
import logging
import os

from inkling_attn._hook import when_imported

from . import IncompatibleImage

LOG = logging.getLogger(__name__)
_PLAN_VERIFY_ROWS = ContextVar("inkling_kv12s_plan_verify_rows", default=None)
_FORWARD = ContextVar("inkling_kv12s_forward", default=None)
_VERIFY_WINDOW = ContextVar("inkling_kv12s_verify_window", default=None)
PackedSpec = None
# Target sliding-window band stored page-major (see swa_layout): avoids GPU TLB misses
# of the unified arena's token-major entries in the compact verify attention.
_SWA_PAGE_MAJOR = os.environ.get("INKLING_KV12S_SWA_PAGE_MAJOR", "1").strip().lower() not in ("0", "off", "false", "no")


# Eager prefill decodes the visit's referenced pages into a bounded BF16 copy and
# runs the BF16 fused kernel (attention.PrefixScratch); 0 = packed in-kernel loader.
# 2112 pages = a 256K context plus one 8K chunk; allocated once and charged in the plan.
def _scratch_pages():
    """Prefix scratch pages: INKLING_KV12S_PREFILL_SCRATCH_PAGES, else enough for one
    request at the server's context length plus one 8K chunk (at least 2,112 = 256K + 64).
    Beyond 262,144 tokens this costs 512 KiB per page at eight KV heads (1M: 4.03 GiB)."""
    value = os.environ.get("INKLING_KV12S_PREFILL_SCRATCH_PAGES", "").strip()
    if value:
        return int(value)
    try:
        from sglang.srt.runtime_context import get_server_args

        context = int(get_server_args().context_length or 0)
    except Exception:  # noqa: BLE001 - outside a server (unit fixtures)
        context = 0
    return max(2112, -(-context // 128) + 65)
_PREFIX_SCRATCH = []
# Unit fixtures without a scratch may still use the unbounded eager gather.
_ALLOW_UNBOUNDED_GATHER = False
# Below this many new rows the per-visit decode of the whole prefix outweighs the
# packed loader's slower tiles (128K prefix: ~5 ms decode per visit).
_SCRATCH_MIN_TOKENS = int(os.environ.get("INKLING_KV12S_PREFILL_SCRATCH_MIN_TOKENS", "512"))
# Eager-only BF16 gather for FA4 calls the packed readers don't cover. The full
# extent costs ~10 GiB; without it such a call gathers into temporaries.
_FA4_WORKSPACE = os.environ.get("INKLING_KV12S_FA4_WORKSPACE", "0").strip().lower() not in ("", "0", "off", "false", "no")
# Which SGLang pools carry the packed store: "static" (non-unified SWAKVPool, static.py)
# or "unified" (the --enable-unified-memory arena, v2.7.1).
_POOL = os.environ.get("INKLING_KV12S_POOL", "static").strip().lower()


def _wrap(owner, name, params, factory):
    original = getattr(owner, name, None)
    if original is None or tuple(inspect.signature(original).parameters) != tuple(params.split()):
        raise IncompatibleImage(f"inkling_kv12s: unexpected signature for {owner.__name__}.{name}")
    setattr(owner, name, wraps(original)(factory(original)))


def _packed(arena, sub_pool):
    view = getattr(arena, "_inkling_kv12s", None)
    return view if view is not None and view.name == sub_pool else None


def _owner_view(allocator):
    return _packed(allocator.arena, allocator.roles.owner)


def _check_cache_selectors(memory, speculative_algorithm, disaggregation_mode):
    if memory.enable_flexkv:
        raise IncompatibleImage("inkling_kv12s: FlexKV requires raw global KV buffers")
    if memory.radix_cache_backend is not None:
        raise IncompatibleImage("inkling_kv12s requires the built-in unified radix cache; "
                                f"explicit backend {memory.radix_cache_backend!r} is unsupported")
    if speculative_algorithm not in (None, "DSPARK"):
        raise IncompatibleImage("inkling_kv12s supports a linear DSpark draft")
    if disaggregation_mode not in (None, "null", "none"):
        raise IncompatibleImage("inkling_kv12s: prefill/decode disaggregation and PD decode restore are unsupported")


def _check_serving_args(spec):
    """Refuse server settings whose paths the packed store does not bound."""
    from sglang.srt.runtime_context import get_server_args

    args = get_server_args()
    for name in ("tp_size", "pp_size", "dp_size", "ep_size"):
        if int(getattr(args, name, 1) or 1) != 1:
            raise IncompatibleImage(f"inkling_kv12s: {name} must be 1 (side-slot decisions are per rank)")
    if getattr(args, "enable_streaming_session", False):
        raise IncompatibleImage("inkling_kv12s: --enable-streaming-session is unsupported (packed pages are immutable)")
    if getattr(args, "enable_multimodal", False):
        raise IncompatibleImage("inkling_kv12s: --enable-multimodal routes image prefill through an unbudgeted "
                                "BF16 gather; unsupported")
    try:
        warps = int(os.environ.get("INKLING_ATTN_WARPS", "4"))
        w8_from = int(os.environ.get("INKLING_ATTN_W8_FROM", "131072"))
    except ValueError as exc:
        raise IncompatibleImage("inkling_kv12s: INKLING_ATTN_WARPS / INKLING_ATTN_W8_FROM must be integers") from exc
    if warps not in (4, 8) or not 0 <= w8_from < 2**31:
        raise IncompatibleImage("inkling_kv12s: INKLING_ATTN_WARPS must be 4 or 8 and 0 <= INKLING_ATTN_W8_FROM < 2**31")
    if spec.speculative_algorithm is not None and not _FA4_WORKSPACE:
        from sglang.srt.environ import envs

        try:  # the same accessor the model's verify dispatch reads
            compact = bool(envs.SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL.get())
        except ValueError:
            compact = False
        if not compact:
            raise IncompatibleImage("inkling_kv12s requires SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1 "
                                    "(compact verify reads the packed store; FA4 verify would gather during capture)")


def _check_verify_capture(runner):
    """Every captured verify batch must take the compact packed reader: the FA4
    fallback reads host metadata and cannot run inside graph capture."""
    if _FA4_WORKSPACE:
        return
    from sglang.srt.models.inkling_common import inkling2_attn
    from sglang.srt.runtime_context import get_server_args, get_spec

    spec = get_spec()
    if spec.speculative_algorithm is None:
        return
    from sglang.srt.runtime_context import get_exec

    args = get_server_args()
    width = int(getattr(spec, "speculative_num_draft_tokens", 0) or 0)
    try:  # the resolved capture buckets, as DSpark reads them
        bs = max(get_exec().graph.cuda_graph_config.decode.bs)
    except (AttributeError, TypeError, ValueError):
        bs = None
    cli = [getattr(args, name, None) for name in ("cuda_graph_max_bs_decode", "cuda_graph_max_bs", "max_running_requests")]
    bs = max([int(b) for b in (bs, *cli) if b])  if any((bs, *cli)) else None
    cutoff = getattr(inkling2_attn, "VERIFY_REL_BIAS_MAX_TOKENS", 32)
    if not bs or not width or int(bs) * width > cutoff:
        raise IncompatibleImage(
            f"inkling_kv12s: captured verify batches up to {bs} x {width} rows exceed the compact verify cutoff "
            f"({cutoff} rows; enable INKLING_VATTN with INKLING_VATTN_MAX_TOKENS >= that product, lower "
            "--cuda-graph-max-bs-decode, or set INKLING_KV12S_FA4_WORKSPACE=1)")


def _register_config(module):
    cls = module.KVCacheConfigurator

    def check_config(original):
        def configure(self, budget_bytes, *, cap_tokens=None):
            if self.is_draft_worker:
                return original(self, budget_bytes, cap_tokens=cap_tokens)
            from sglang.srt.model_executor.cuda_graph_config import Backend, Phase, check_cuda_graph_backend

            memory, parallel = module.get_memory(), module.get_parallel()
            if _POOL == "static":
                if memory.enable_unified_memory:
                    raise IncompatibleImage("inkling_kv12s: INKLING_KV12S_POOL=static runs without "
                                            "--enable-unified-memory (set INKLING_KV12S_POOL=unified for it)")
                if not self.is_hybrid_swa or self.use_mla_backend:
                    raise IncompatibleImage("inkling_kv12s expects the hybrid sliding-window MHA pools")
                if not (check_cuda_graph_backend(Phase.PREFILL, Backend.FULL)
                        or check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED)):
                    # Breakable prefill graphs replay attention eagerly without the fused
                    # projection marker, which would send global prefill to the FA4 gather.
                    raise IncompatibleImage("inkling_kv12s: static pools need --cuda-graph-backend-prefill=full "
                                            "(or disabled)")
            else:
                if not memory.enable_unified_memory:
                    raise IncompatibleImage("inkling_kv12s requires unified memory")
                if not check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED):
                    raise IncompatibleImage("inkling_kv12s requires --cuda-graph-backend-prefill=disabled")
            if (parallel.attn_dcp_size != 1 or parallel.enable_prefill_cp or parallel.pp_size != 1
                    or self.attn_dp_size != 1 or module.get_exec().features.enable_memory_saver):
                raise IncompatibleImage("inkling_kv12s requires unsharded, resident request ownership")
            spec = module.get_spec()
            _check_cache_selectors(memory, spec.speculative_algorithm, module.get_disagg().disaggregation_mode)
            if spec.speculative_eagle_topk not in (None, 1) or spec.speculative_dflash_draft_kv_ring:
                raise IncompatibleImage("inkling_kv12s requires a linear draft chain in unified KV storage")
            if os.environ.get("SGLANG_DEBUG_POISON_POOL", "0").strip().lower() not in ("", "0", "false"):
                raise IncompatibleImage("inkling_kv12s requires zero-initialized page envelopes")
            if any(getattr(memory, flag, False) for flag in (
                "enable_hierarchical_cache", "enable_lmcache", "enable_unified_cache_external_linker",
            )):
                raise IncompatibleImage("inkling_kv12s does not support external cache transfer")
            from sglang.srt.speculative.ragged_verify import RaggedVerifyMode, read_ragged_verify_mode

            if spec.speculative_algorithm is not None and read_ragged_verify_mode() is not RaggedVerifyMode.STATIC:
                # VATTN's SWA reader and the stock compact SWA fallback index
                # queries with one batch-wide QLEN.
                raise IncompatibleImage("inkling_kv12s requires SGLANG_RAGGED_VERIFY_MODE=static")
            _check_serving_args(spec)
            if "Inkling2LoopForConditionalGeneration" not in self.model_config.hf_config.architectures:
                raise IncompatibleImage("inkling_kv12s expects the Inkling-2-Mini loop model")
            return original(self, budget_bytes, cap_tokens=cap_tokens)
        return configure

    def plan_scope(original):
        def configure(self, budget_bytes, *, cap_tokens=None):
            rows = max(1, module.max_speculative_num_draft_tokens() or 1)
            token = _PLAN_VERIFY_ROWS.set(rows)
            try:
                return original(self, budget_bytes, cap_tokens=cap_tokens)
            finally:
                _PLAN_VERIFY_ROWS.reset(token)
        return configure

    _wrap(cls, "config_from_budget", "self budget_bytes cap_tokens", check_config)
    if _POOL != "static":
        _wrap(cls, "_unified_config_from_budget", "self budget_bytes cap_tokens", plan_scope)


def _register_plan(module, overflow_heads, draft_overflow_heads):
    import msgspec
    import torch

    from .storage import DIM, PAGE, Layout, arenas_bytes, stage_slots, workspace_bytes

    @dataclass(frozen=True, kw_only=True)
    class PackedMHASpec(module.MHASubPoolSpec):
        verify_rows: int

        def host_entry_bytes(self):
            return self.layer_num * (Layout(self.head_num).size // PAGE)

        def entry_bytes(self):
            from .storage import raw_lane_bytes

            return self.host_entry_bytes() + sum(raw_lane_bytes(h) // PAGE for h in self.raw_lane_heads())

        def lane_heads(self):
            """Packed target visits (the MTP draft lane is raw, see device.RawLane)."""
            return (self.head_num,) * self.layer_num

        def raw_lane_heads(self):
            draft = self.draft_region
            return (draft.head_num,) * draft.lane_num if draft else ()

        def lane_overflow_heads(self):
            return (overflow_heads,) * self.layer_num

        def fixed_bytes(self, running, *, overlap):
            from .storage import FALLBACK_PAGES

            slots = stage_slots(running, self.verify_rows, overlap=overlap) + FALLBACK_PAGES
            return sum(arenas_bytes(h, slots, overflow) + 4 * (slots + overflow)
                       for h, overflow in zip(self.lane_heads(), self.lane_overflow_heads()))

        def workspace_bytes(self):
            # attention.PrefixScratch: BF16 K and V of _scratch_pages() pages.
            scratch = 2 * max(0, _scratch_pages()) * PAGE * max(self.lane_heads()) * DIM * 2
            return workspace_bytes(max(self.lane_heads())) + scratch

        def gather_bytes(self, chain_bytes):
            if not _FA4_WORKSPACE:
                return 0
            pages = chain_bytes // (PAGE * self.entry_bytes())
            return pages * (4 * PAGE * DIM * max(self.lane_heads()) + 8) + 4

    global PackedSpec
    PackedSpec = PackedMHASpec

    def make_plan(original):
        def plan(decls, *, page_size, fused_draft=None, speculative_num_draft_tokens=None, derived_parts=()):
            result = original(decls, page_size=page_size, fused_draft=fused_draft,
                              speculative_num_draft_tokens=speculative_num_draft_tokens, derived_parts=derived_parts)
            verify_rows = _PLAN_VERIFY_ROWS.get()
            if verify_rows is None:
                return result
            owner = result.owner()
            spec = owner.spec
            if (page_size != PAGE or type(spec) is not module.MHASubPoolSpec or owner.parts
                    or any(part.sub_pool == owner.name for part in result.derived_parts)
                    or spec.kv_cache_dtype != torch.bfloat16 or spec.head_dim != DIM
                    or spec.v_head_dim != DIM or spec.layer_num != 8 or spec.is_mxfp8):
                raise IncompatibleImage("inkling_kv12s expects eight global BF16 visits with 128-wide heads/pages")
            draft = spec.draft_region
            if draft is not None and (
                not isinstance(draft, module.DenseDraftRegion) or draft.token_replicas != 1
                or draft.resolved_kv_dtype() != torch.bfloat16 or draft.head_dim != DIM
                or draft.resolved_v_head_dim() != DIM or draft.lane_num != 1
            ):
                raise IncompatibleImage("inkling_kv12s expects one global BF16 MTP draft lane")
            packed = PackedSpec(
                name=spec.name, layer_num=spec.layer_num, grow_direction=spec.grow_direction,
                head_num=spec.head_num, head_dim=spec.head_dim, v_head_dim=spec.v_head_dim,
                store_dtype=spec.store_dtype, verify_rows=verify_rows, draft_region=draft,
            )
            return msgspec.structs.replace(
                result, sub_pools=tuple(
                    msgspec.structs.replace(s, spec=packed) if s is owner else s for s in result.sub_pools
                ),
            )
        return plan

    _wrap(module, "plan_kv_storage", "decls page_size fused_draft speculative_num_draft_tokens derived_parts", make_plan)


def _register_sizing(module):
    def make_fixed(original):
        def fixed(plan, inp, running, strategy):
            terms = original(plan, inp, running, strategy)
            spec = plan.owner().spec
            if type(spec) is not PackedSpec:
                return terms
            return (
                *terms,
                module.FixedAllocation(
                    name="inkling_kv12s_arenas", bytes=spec.fixed_bytes(running, overlap=strategy.overlap)
                ),
                module.FixedAllocation(name="inkling_kv12s_workspace", bytes=spec.workspace_bytes()),
            )
        return fixed

    def make_derived(original):
        def derived(plan, chain_bytes):
            terms = original(plan, chain_bytes)
            spec = plan.owner().spec
            if type(spec) is not PackedSpec:
                return terms
            return (*terms, module.FixedAllocation(
                name="inkling_kv12s_fa4_workspace", bytes=spec.gather_bytes(chain_bytes)
            ))
        return derived

    _wrap(module, "_fixed_terms", "plan inp running strategy", make_fixed)
    _wrap(module, "_derived_terms", "plan chain_bytes", make_derived)


def _register_build(module):
    import torch

    from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import UnquantizedKVCacheMethod
    from sglang.srt.mem_cache.kv.memory.views import MHAKVView
    from sglang.srt.mem_cache.memory_pool import unwrap_write_loc, write_loc_is_physical

    from .attention import FA4Workspace
    from .device import DeviceLayer, RawLane
    from .storage import raw_lane_bytes
    from .storage import DIM, FALLBACK_PAGES, PAGE, Layout, stage_slots

    class PackedView(MHAKVView):
        is_inkling_kv12s = True
        device_owned = True
        dtype = store_dtype = torch.bfloat16
        requires_physical_write_loc = True
        is_mxfp8 = False
        is_quantized_kv_cache = False
        k_scale_buffer = v_scale_buffer = None
        start_layer = 0
        layer_transfer_counter = None
        # Stock routed-ID consumers (RunnerIds, KVLayerRouter.storage_name)
        # read these MHAView/DraftMHAView attributes from every routed view.
        token_replicas = 1
        host_capacity_tokens = host_capacity_bytes = None
        cpu_copy_carries_mamba = False
        layer_shard_enabled = False
        post_capture_active = False
        use_hnd = False
        kv_cache_layout = "page_major"
        enable_custom_mem_pool = False
        custom_mem_pool = None
        allocation_label = None
        mem_usage = 0.0

        def __init__(self, arena, name, layer_ids, heads, layers, *, owned_layers=None, start_layer=0,
                     page_bytes=0):
            self.name = name
            self._arena = arena
            self._sub_pool_name = name
            self.model_layer_ids = tuple(layer_ids)
            self.device = arena.device
            self.page_size = PAGE
            self.page_bytes = page_bytes
            self.head_num = heads
            self.head_dim = self.v_head_dim = DIM
            self.row_dim = self.v_row_dim = heads * DIM
            self.layer_num = len(layers)
            self.start_layer = start_layer
            self.end_layer = start_layer + self.layer_num - 1
            self.layers = layers
            self.owned_layers = layers if owned_layers is None else owned_layers
            self.num_pages = arena.extent_entries(name) // PAGE
            self.size = (self.num_pages - 1) * PAGE
            self.quant_method = UnquantizedKVCacheMethod()

        @property
        def grow_direction(self):
            return self._arena.band(self._sub_pool_name).grow_direction

        def maybe_get_custom_mem_pool(self):
            return None

        def register_layer_transfer_counter(self, counter):
            raise RuntimeError("inkling_kv12s: layer-wise host load-back is unsupported")

        def get_contiguous_buf_infos(self):
            raise RuntimeError("inkling_kv12s: packed global KV has no raw transfer buffers")

        def get_cpu_copy(self, *args, **kwargs):
            raise RuntimeError("inkling_kv12s: use the allocator snapshot for packed global KV")

        load_cpu_copy = get_cpu_copy

        def draft_view(self, arena, host_spec, layer_lanes):
            first_draft = host_spec.layer_num
            ids = sorted(layer_lanes)
            layers = tuple(self.owned_layers[first_draft + layer_lanes[i]] for i in ids)
            view = PackedView(arena, self.name, ids, host_spec.draft_region.head_num, layers,
                              start_layer=ids[0], page_bytes=self.page_bytes)
            view.layer_lanes = dict(layer_lanes)
            return view

        def get_kv_cache_quant_method(self):
            return self.quant_method

        def get_kv_size_bytes(self):
            return 0, 0  # charged in the arena and the plan's fixed terms

        def get_kv_buffer_shape(self):
            shape = torch.Size((self.num_pages * PAGE, self.head_num, DIM))
            return shape, shape

        def get_v_head_dim(self):
            return DIM

        def get_kv_buffer(self, layer_id):
            raise RuntimeError("inkling_kv12s: a raw access reached the packed global KV route")

        get_key_buffer = get_value_buffer = get_kv_buffer

        def set_kv_buffer(self, layer, loc_info, cache_k, cache_v, k_scale=None, v_scale=None,
                          layer_id_override=None, dcp_kv_mask=None):
            if not write_loc_is_physical(loc_info) or dcp_kv_mask is not None:
                raise RuntimeError("inkling_kv12s expects physical, unsharded KV writes")
            loc, _, _ = unwrap_write_loc(loc_info)
            visit = _FORWARD.get()
            self._store(layer, loc, cache_k, cache_v, k_scale, v_scale, layer_id_override,
                        tentative=visit is not None and visit.tentative)

        def set_kv_buffer_prefix_valid(self, layer, loc_2d, commit_lens, cache_k, cache_v,
                                       k_scale=None, v_scale=None, layer_id_override=None):
            if loc_2d.ndim != 2 or commit_lens.shape != (loc_2d.shape[0],):
                raise ValueError("inkling_kv12s: invalid prefix commit metadata")
            rows = torch.arange(loc_2d.shape[1], device=loc_2d.device)
            masked = torch.where(rows[None, :] < commit_lens[:, None], loc_2d, 0)
            self._store(layer, masked, cache_k, cache_v, k_scale, v_scale, layer_id_override,
                        tentative=False)

        def _store(self, layer, loc, cache_k, cache_v, k_scale, v_scale, layer_id_override, *, tentative):
            if cache_k.dtype != self.dtype:
                # Match MHAView's conversion before BF16 storage.
                if k_scale is not None:
                    cache_k = cache_k.div(k_scale)
                if v_scale is not None:
                    cache_v = cache_v.div(v_scale)
            index = layer.layer_id if layer_id_override is None else layer_id_override
            self.layers[index].write(loc, cache_k, cache_v, tentative=tentative)

    def make_view(original):
        def token_view(sub_pool, arena, plan, ctx):
            spec = sub_pool.spec
            if type(spec) is not PackedSpec:
                view = original(sub_pool, arena, plan, ctx)
                if _SWA_PAGE_MAJOR and type(plan.owner().spec) is PackedSpec:
                    from . import swa_layout

                    if swa_layout.eligible(spec):
                        view = swa_layout.convert(view, arena, spec, PAGE)
                        bands = getattr(arena, "_inkling_kv12s_rows", {})
                        bands[spec.name] = view
                        arena._inkling_kv12s_rows = bands
                        LOG.info("inkling_kv12s: window band %s page-major (%d visits, %d B/token)",
                                 spec.name, spec.layer_num, spec.entry_bytes())
                return view
            if torch.cuda.get_device_capability(ctx.device) != (12, 1):
                raise IncompatibleImage("inkling_kv12s requires GB10 (SM121)")
            heads = spec.lane_heads()
            stride = PAGE * spec.entry_bytes()
            slots = stage_slots(ctx.max_num_reqs, spec.verify_rows, overlap=not ctx.disable_overlap_schedule)
            pages = arena.extent_entries(spec.name) // PAGE
            workspace = FA4Workspace(pages, max(heads), arena.raw.device) if _FA4_WORKSPACE else None
            scratch_pages = _scratch_pages()
            if scratch_pages > 0 and not _PREFIX_SCRATCH:
                from .attention import PrefixScratch

                _PREFIX_SCRATCH.append(PrefixScratch(scratch_pages, max(heads), arena.raw.device))
                LOG.info("inkling_kv12s: prefill BF16 prefix scratch %d pages (%.2f GiB)", scratch_pages,
                         2 * scratch_pages * PAGE * max(heads) * DIM * 2 / 2**30)
            layers = []
            offset = 0
            overflows = spec.lane_overflow_heads()
            if len(set(overflows)) != 1 or len(set(heads)) != 1:
                raise IncompatibleImage("inkling_kv12s expects uniform packed target visits")
            # One lock row and one event row per visit, in shared buffers that the
            # side-slot guard reads with a single reduction per forward.
            from .pressure import GUARD, SideGuard
            from .storage import EVENT_WORDS

            locks = torch.zeros((len(heads), slots + FALLBACK_PAGES + overflows[0]), dtype=torch.int32,
                                device=arena.raw.device)
            events = torch.zeros((len(heads), EVENT_WORDS), dtype=torch.int32, device=arena.raw.device)
            for index, (h, overflow) in enumerate(zip(heads, overflows)):
                lane = DeviceLayer(arena.raw, h, stride, offset, slots, overflow, FALLBACK_PAGES,
                                   locks=locks[index], events=events[index])
                lane.gather_workspace = workspace
                layers.append(lane)
                offset += Layout(h).size
            from sglang.srt.runtime_context import get_server_args

            args = get_server_args()
            chunk = max(int(getattr(args, "max_prefill_tokens", 0) or 0),
                        int(getattr(args, "chunked_prefill_size", 0) or 0), PAGE)
            GUARD[:] = [SideGuard(layers, locks, events, prefill_tokens=chunk, page=PAGE)]
            for h in spec.raw_lane_heads():
                layers.append(RawLane(arena.raw, h, stride, offset, pages))
                offset += raw_lane_bytes(h)
            view = PackedView(arena, spec.name, sub_pool.layer_ids, spec.head_num,
                              tuple(layers[:spec.layer_num]), owned_layers=tuple(layers), page_bytes=stride)
            arena._inkling_kv12s = view
            LOG.info(
                "inkling_kv12s: target global visits %s, draft global lanes %d, %d B/token, "
                "%.1f MiB side arenas, %.1f MiB FA4 workspace",
                sub_pool.layer_ids, len(layers) - spec.layer_num, spec.entry_bytes(),
                spec.fixed_bytes(ctx.max_num_reqs, overlap=not ctx.disable_overlap_schedule) / 2**20,
                spec.gather_bytes(arena.total_bytes) / 2**20,
            )
            return view
        return token_view

    def log_capacity(original):
        def build(plan, sized, ctx):
            result = original(plan, sized, ctx)
            if type(plan.owner().spec) is PackedSpec:
                LOG.info("inkling_kv12s: FULL tokens=%d, SWA tokens=%d, running requests=%d%s",
                         sized.full_tokens, sized.window_tokens, sized.max_running_requests,
                         " (before graph capture)" if ctx.post_capture_active else "")
            return result
        return build

    _wrap(module, "_build_token_view", "sub_pool arena plan ctx", make_view)
    _wrap(module, "build_kv_storage", "plan sized ctx", log_capacity)


def _register_draft_binding(module):
    import msgspec

    original_view = module.DraftMHAView
    original_binder = module.DRAFT_BINDERS["dense"]

    def draft_view(*, arena, host_spec, layer_lanes, page_size):
        view = _packed(arena, host_spec.name)
        if view is None:
            return original_view(arena=arena, host_spec=host_spec, layer_lanes=layer_lanes, page_size=page_size)
        return view.draft_view(arena, host_spec, layer_lanes)

    def bind_dense(ctx, binding):
        # The stock owner-only shortcut creates a plain view. Keep the same
        # routed physical-ID contract for owner-only and mixed packed drafts.
        if ctx.hosts_of("dense") != (ctx.owner,) or _packed(ctx.arena, ctx.owner) is None:
            return original_binder(ctx, binding)
        ids = module.draft_kv_layer_ids(ctx.model)
        lanes = ctx.placement.lanes_for(ctx.runner, ctx.owner)
        if len(ids) != len(lanes):
            raise IncompatibleImage("inkling_kv12s: draft layers differ from the planned packed lanes")
        pool = module.routed_draft_pool(
            arena=ctx.arena, plan=ctx.plan, owner=ctx.owner, page_size=ctx.page_size,
            lanes_by_host={ctx.owner: dict(zip(ids, lanes))},
        )
        return msgspec.structs.replace(binding, token_to_kv_pool=pool)

    module.DraftMHAView = draft_view
    module.DRAFT_BINDERS["dense"] = bind_dense


def _register_lifetime(module):
    import torch

    def make_release(original):
        def release(self, pages, cause, *, host=None, readers=()):
            view = _packed(self.arena, self.sub_pool_name)
            if view is None:
                return original(self, pages, cause, host=host, readers=readers)
            if readers:
                raise RuntimeError("inkling_kv12s: external reader on a released page")
            parts = (pages,) if isinstance(pages, torch.Tensor) else tuple(pages)
            if getattr(view, "device_owned", False):
                if cause != module.Release.VACATED:
                    for ids in parts:
                        for lane in view.owned_layers:
                            lane.release(ids)
            else:
                retired = host if host is not None else [page for ids in parts for page in ids.tolist()]
                for lane in view.layers:
                    lane.release(retired)
            return original(self, parts[0] if isinstance(pages, torch.Tensor) else parts,
                            cause, host=host, readers=readers)
        return release

    def make_settle(original):
        def settle(self, through=None, prepared_holes=None):
            view = _packed(self.arena, self.sub_pool_name)
            pending = self._pending_until_retire
            if view is not None and prepared_holes is not None and pending:
                if getattr(view, "device_owned", False):
                    # Take the stock event-fenced release branch, then attach
                    # the host copy of exactly the holes it just published.
                    if through is not None and through < len(pending):
                        raise AssertionError("a partial settle with a copy")
                    result = original(self, through, None)
                    self._set_free_pages(self._free_phys_pages, host=prepared_holes)
                    return result
                chunks, _, _ = pending.oldest(len(pending) if through is None else through)
                retired = [page for ids in chunks for page in ids.tolist()]
                for lane in view.layers:
                    lane.release(retired)
            return original(self, through, prepared_holes)
        return settle

    def make_clear(original):
        def clear(self):
            result = original(self)
            view = _packed(self.arena, self.sub_pool_name)
            if view is not None:
                for lane in getattr(view, "owned_layers", view.layers):
                    lane.clear()
            return result
        return clear

    _wrap(module.Band, "release", "self pages cause host readers", make_release)
    _wrap(module.Band, "settle_pending", "self through prepared_holes", make_settle)
    _wrap(module.Band, "clear", "self", make_clear)


def _register_move(module):
    import torch

    def make_move(original):
        def move(self, src_pages, dst_pages):
            view = _packed(self.arena, self.sub_pool_name)
            if view is None or not src_pages.numel():
                return original(self, src_pages, dst_pages)
            # Stock may copy a source while an older forward still reads
            # it. Packed moves also invalidate that source, so first order
            # this move behind the latest whole-forward completion.
            event = self.usage.latest_forward
            if event is not None:
                self._wait_on_stream(event)
            elif self.forward_steps is not None and self.forward_steps.in_flight() and self.forward_stream is not None:
                self._before_wait()
                torch.cuda.current_stream(self.device).wait_stream(self.forward_stream)
            result = original(self, src_pages, dst_pages)
            # The copied descriptor owns the same slots at the destination.
            for lane in view.owned_layers:
                lane.release(src_pages, recycle=False)
            return result
        return move

    _wrap(module.Band, "_move_pages", "self src_pages dst_pages", make_move)


def _register_snapshot(module):
    import torch

    from .storage import PACK_BATCH, PAGE

    def row_bands(allocator):
        return getattr(allocator.arena, "_inkling_kv12s_rows", {})

    class WithoutPackedBand:
        def __init__(self, allocator, view):
            self.allocator = allocator
            skip = {view.name, *row_bands(allocator)}
            self.members = [band for band in allocator.members if band.sub_pool_name not in skip]

        def __getattr__(self, name):
            return getattr(self.allocator, name)

    def make_copy(original):
        def copy(allocator, indices, *, state_indices, request_row):
            view = _owner_view(allocator)
            if view is None:
                return original(allocator, indices, state_indices=state_indices, request_row=request_row)
            saved = original(WithoutPackedBand(allocator, view), indices,
                             state_indices=state_indices, request_row=request_row)
            band = allocator.full_attn_allocator
            physical = module._physical(band, indices, by_unit=False)
            positions = (physical > 0).nonzero(as_tuple=True)[0]
            locations = physical[positions].tolist()
            lanes = getattr(view, "owned_layers", view.layers)
            # Draft head geometry can differ from the target. Preserve the
            # per-lane shape and use the stock snapshot for the other bands.
            entries = [module.torch.empty(
                (len(locations), 2, lane.heads, view.head_dim), dtype=view.dtype, device="cpu"
            ) for lane in lanes]
            for dest, layer in zip(entries, lanes):
                for start in range(0, len(locations), PACK_BATCH * PAGE):
                    end = start + PACK_BATCH * PAGE
                    dest[start:end].copy_(layer.read_rows(locations[start:end]))
            saved.bands[view.name] = module.BandSnapshot(positions=positions.cpu(), entries=tuple(entries))
            # Page-major window bands: save each token's rows of every visit.
            for band in allocator.members:
                rows_view = row_bands(allocator).get(band.sub_pool_name)
                if rows_view is None:
                    continue
                ids, by_unit = module._band_ids(allocator, band, indices, state_indices, request_row)
                ids = ids.reshape(-1)
                physical = module._physical(band, ids, by_unit=by_unit)
                positions = torch.nonzero(physical > 0, as_tuple=True)[0]
                rows = physical[positions]
                values = torch.empty((rows.numel(), rows_view.layer_num, 2, rows_view.head_num, rows_view.head_dim),
                                     dtype=torch.bfloat16, device="cpu")
                from .swa_layout import SNAPSHOT_ROWS

                for start in range(0, rows.numel(), SNAPSHOT_ROWS):
                    values[start:start + SNAPSHOT_ROWS].copy_(rows_view.read_rows(rows[start:start + SNAPSHOT_ROWS]))
                saved.bands[band.sub_pool_name] = module.BandSnapshot(positions=positions.cpu(), entries=values)
            return saved
        return copy

    def make_restore(original):
        def restore(allocator, snapshot, indices, *, state_indices, request_row):
            view = _owner_view(allocator)
            if view is None:
                return original(allocator, snapshot, indices, state_indices=state_indices, request_row=request_row)
            saved = snapshot.bands[view.name]
            physical = module._physical(allocator.full_attn_allocator, indices, by_unit=False)
            locations = physical[saved.positions.to(physical.device)].tolist()
            if any(loc <= 0 for loc in locations):
                raise RuntimeError("inkling_kv12s: restore has no destination for saved global rows")
            entries = saved.entries if isinstance(saved.entries, tuple) else saved.entries.unbind(dim=1)
            lanes = getattr(view, "owned_layers", view.layers)
            if len(entries) != len(lanes):
                raise RuntimeError("inkling_kv12s: snapshot and storage lane counts differ")
            publications = [
                layer.prepare_write(locations, rows[:, 0], rows[:, 1])
                for layer, rows in zip(lanes, entries)
            ]
            original(WithoutPackedBand(allocator, view), snapshot, indices,
                     state_indices=state_indices, request_row=request_row)
            for publish in publications:
                publish()
            for band in allocator.members:
                rows_view = row_bands(allocator).get(band.sub_pool_name)
                band_saved = snapshot.bands.get(band.sub_pool_name)
                if rows_view is None or band_saved is None:
                    continue
                ids, by_unit = module._band_ids(allocator, band, indices, state_indices, request_row)
                physical = module._physical(band, ids.reshape(-1), by_unit=by_unit)
                destinations = physical[band_saved.positions.to(physical.device)]
                # A window may retain less history by restore time; restore its bound rows.
                valid = (destinations > 0).cpu()
                keep = torch.nonzero(valid, as_tuple=True)[0]
                if keep.numel():
                    from .swa_layout import SNAPSHOT_ROWS

                    # Index the host payload one tile at a time: a whole-band
                    # fancy index would duplicate it in host memory first.
                    for start in range(0, keep.numel(), SNAPSHOT_ROWS):
                        tile = keep[start:start + SNAPSHOT_ROWS]
                        rows_view.write_rows(destinations[tile.to(destinations.device)],
                                             band_saved.entries[tile])
            module.current_platform.synchronize()
        return restore

    signature = "allocator indices state_indices request_row"
    _wrap(module, "copy_to_cpu", signature, make_copy)
    _wrap(module, "restore_from_cpu", "allocator snapshot indices state_indices request_row", make_restore)


@dataclass
class Forward:
    tentative: bool
    cu_q: object
    ragged: bool
    qmax: int


def _packed_prefill_in_capture(q, layer, page_table, cu_seqlens_q, seqused_k, softmax_scale, *, r, proj, out,
                               warps=None):
    """inkling_attn.ops.packed_prefill_attention without its host-side length check
    (which reads seqused_k): geometry is validated on shapes only, so the call can be
    captured; the kernel bounds every page and length on the device."""
    import torch
    from inkling_attn import build, ops

    from .storage import validate_paged_geometry

    validate_paged_geometry(q, layer, page_table, seqused_k, cu_seqlens_q)
    if cu_seqlens_q is None or out is None:
        raise RuntimeError("inkling_kv12s: a captured packed prefill needs query offsets and an output buffer")
    t, heads, _ = q.shape
    lse = torch.empty((heads, t), dtype=torch.float32, device=q.device)
    if r.stride(-1) != 1:
        r = r.contiguous()
    projp = (torch.empty(((proj.shape[1] + 1) * ops.D_REL,), dtype=torch.bfloat16, device=q.device)
             if ops.PACKED_PROJ else None)
    planes = [*layer.planes, layer.stage, layer.overflow]
    build.load().fwd(q, layer.raw, layer.raw, page_table, cu_seqlens_q, seqused_k, r, proj, out, lse,
                     float(softmax_scale), -1, projp, ops.WARPS if warps is None else int(warps), planes,
                     ops.W8_FROM)
    return out, lse


def dispatch_packed_fa4(stock, layer, kwargs, *, warps=None):
    """Only a supported projection marker selects fused prefill; otherwise call stock FA4."""
    import torch

    from inkling_attn.ops import packed_prefill_attention
    from inkling_attn.patch import STATS, packed_rel_projection

    from .storage import PAGE

    q = kwargs["q"]
    table, seq_k = kwargs["page_table"], kwargs["cache_seqlens"]
    cu_q = kwargs.get("cu_seqlens_q")
    context = _FORWARD.get()
    if cu_q is None and context is not None:
        cu_q = context.cu_q

    bias = kwargs.get("rel_bias")
    proj = packed_rel_projection(bias) if bias is not None else None
    unsupported = ("k", "v", "qv", "q_descale", "k_descale", "v_descale", "sinks", "score_mod", "mask_mod",
                   "aux_tensors", "sfq", "sfk", "sfv", "scheduler_metadata", "attention_chunk",
                   "cache_batch_idx", "cache_leftpad", "rotary_cos", "rotary_sin", "rotary_seqlens")
    if (proj is not None and cu_q is not None and kwargs.get("out") is not None
            and q.shape[:2] == bias.shape[:2]
            and kwargs.get("causal") and kwargs.get("window_size") == (-1, -1)
            and kwargs.get("softmax_scale") is not None and kwargs.get("softcap") in (0, None)
            and kwargs.get("num_splits", 1) in (0, 1) and not kwargs.get("pack_gqa")
            and not kwargs.get("rel_bias_presheared")
            and all(kwargs.get(name) is None for name in unsupported)):
        from .static import CAPTURING

        if CAPTURING or (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
            # Prefill graph warmup and capture (static pools): the eager scratch gather reads the
            # page table on the host. A captured prefill gathers the prefix into the bounded BF16
            # scratch with device-side metadata and runs the BF16 kernel on it, or, when the batch
            # does not fit (a device flag), the in-kernel packed loader; buckets below the scratch
            # threshold use the in-kernel loader only. Both are bitwise equal to the BF16 kernel
            # on BF16 KV.
            scratch = _PREFIX_SCRATCH[0] if _PREFIX_SCRATCH else None
            if scratch is None or scratch.max_pages < 2 or q.shape[0] < _SCRATCH_MIN_TOKENS:
                # Small buckets: the eager rule (per-visit decode of the prefix outweighs).
                result = _packed_prefill_in_capture(
                    q, layer, table, cu_q, seq_k, kwargs["softmax_scale"], r=bias, proj=proj,
                    out=kwargs["out"], warps=warps,
                )
            else:
                # The prefix in the BF16 scratch when it fits (device decision), else the
                # packed loader: two launches into one output, the inactive one with no
                # query rows (a zero-row request has no work item; it writes nothing).
                from inkling_attn.ops import prefill_attention

                k, v, local, fits = scratch.gather_captured(layer, table, seq_k)
                none = torch.zeros_like(cu_q)
                _, lse_bf16 = prefill_attention(
                    q, k, v, local, torch.where(fits, cu_q, none), seq_k, bias, proj, kwargs["softmax_scale"],
                    None, out=kwargs["out"], warps=warps,
                )
                _, lse_packed = _packed_prefill_in_capture(
                    q, layer, table, torch.where(fits, none, cu_q), seq_k, kwargs["softmax_scale"], r=bias,
                    proj=proj, out=kwargs["out"], warps=warps,
                )
                result = (kwargs["out"], torch.where(fits, lse_bf16, lse_packed))
            STATS["kernel"] += 1
            return result if kwargs.get("return_softmax_lse") else result[0]
        scratch = _PREFIX_SCRATCH[0] if _PREFIX_SCRATCH else None
        copy = (scratch.gather(layer, q, table, seq_k, cu_q)
                if scratch is not None and q.shape[0] >= _SCRATCH_MIN_TOKENS else None)
        if copy is not None:
            from inkling_attn.ops import prefill_attention

            k, v, local = copy
            result = prefill_attention(
                q, k, v, local, cu_q, seq_k, bias, proj, kwargs["softmax_scale"], None,
                out=kwargs["out"], warps=warps,
            )
        else:
            result = packed_prefill_attention(
                q, layer, table, cu_q, seq_k, kwargs["softmax_scale"], r=bias, proj=proj,
                out=kwargs["out"], warps=warps,
            )
        STATS["kernel"] += 1
        return result if kwargs.get("return_softmax_lse") else result[0]

    from .attention import gather_for_fa4

    STATS["gather"] = STATS.get("gather", 0) + 1
    if STATS["gather"] in (1, 10, 1000):
        LOG.info("inkling_kv12s: packed FA4 gather fallback #%d (causal=%s, bias=%s, qmax=%s, lse=%s)",
                 STATS["gather"], kwargs.get("causal"), bias is not None, kwargs.get("max_seqlen_q"),
                 kwargs.get("return_softmax_lse"))
    workspace = getattr(layer, "gather_workspace", None)
    if workspace is None and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("inkling_kv12s: an FA4 call outside the packed readers inside CUDA graph capture "
                           f"(causal={kwargs.get('causal')}, bias={bias is not None}, qmax={kwargs.get('max_seqlen_q')})")
    if workspace is None:
        # Eager only. Use the pool-charged prefix scratch; an unbounded per-call
        # BF16 gather could exceed the memory headroom.
        scratch = _PREFIX_SCRATCH[0] if _PREFIX_SCRATCH else None
        copy = scratch.gather(layer, q, table, seq_k, cu_q) if scratch is not None else None
        if copy is None:
            if scratch is not None or not _ALLOW_UNBOUNDED_GATHER:
                raise RuntimeError("inkling_kv12s: an FA4 call outside the packed readers references more pages "
                                   "than the bounded BF16 scratch holds")
            copy = gather_for_fa4(layer, q, table, seq_k, cu_q, kwargs.get("max_seqlen_q"))
        k, v, table = copy
    else:
        k, v, table = workspace.gather(layer, table, seq_k)
    forwarded = dict(kwargs, k_cache=k, v_cache=v, page_table=table)
    if forwarded.get("max_seqlen_k") is None and forwarded.get("num_splits", 0) != 1:
        # FA4 normalizes SplitKV against the original physical cache span when
        # extend/verify omits max_seqlen_k. The gathered tensor is smaller.
        forwarded["max_seqlen_k"] = layer.pages * PAGE
    if proj is not None:
        STATS["fallback"] += 1
        forwarded["rel_bias"] = torch.einsum("thd,de->the", bias, proj)
        forwarded.pop("rel_bias_presheared", None)
    return stock(**forwarded)


def _register_backend(module):
    from .storage import PAGE, Layer

    cls = module.FlashAttentionBackend

    def get_layer(backend, layer):
        route = backend.kv_layers.layer(layer.layer_id)
        view = route.view
        if getattr(view, "is_inkling_kv12s", False):
            return view.layers[route.view_layer_id]
        return None

    def make_init(original):
        def init(self, model_runner, *args, **kwargs):
            original(self, model_runner, *args, **kwargs)
            from .static import view_of

            view = view_of(model_runner.token_to_kv_pool)
            if view is not None:
                if (self.fa_impl_ver != 4 or self.use_mla or self.topk > 1 or self.fa_skip_kv_cache
                        or self._kv_shard_pool is not None or self.page_size != PAGE):
                    raise IncompatibleImage("inkling_kv12s requires dense FA4 with linear verify")
                if not model_runner.is_draft_worker:
                    # Build the fused prefill reader outside CUDA graph capture.
                    from inkling_attn import build

                    build.load()
        return init

    def make_paged(original):
        def paged(self, layer, *, head_group_num=1):
            lane = get_layer(self, layer)
            if lane is None:
                route = self.kv_layers.layer(layer.layer_id)
                if getattr(route.view, "page_major", False):
                    if head_group_num != 1:
                        raise RuntimeError("inkling_kv12s: page-major window reads require physical KV heads")
                    return route.view.k4[route.view_layer_id], route.view.v4[route.view_layer_id]
                return original(self, layer, head_group_num=head_group_num)
            if head_group_num != 1:
                raise RuntimeError("inkling_kv12s: packed reads require physical KV heads")
            if getattr(lane, "is_raw", False):
                return lane.k, lane.v
            return lane, lane
        return paged

    def make_fused(original):
        def fused(self, layer, forward_batch):
            if get_layer(self, layer) is not None:
                return None
            return original(self, layer, forward_batch)
        return fused

    def make_forward(original):
        def forward(self, q, k, v, layer, forward_batch, *args, **kwargs):
            lane = get_layer(self, layer)
            if lane is None:
                return original(self, q, k, v, layer, forward_batch, *args, **kwargs)
            if module.is_cp_active(forward_batch):
                raise RuntimeError("inkling_kv12s requires unsharded forwards")
            meta = self.forward_metadata
            verify = forward_batch.forward_mode.is_target_verify()
            token = _FORWARD.set(Forward(
                verify or self.is_draft_runner, meta.cu_seqlens_q,
                verify and forward_batch.ragged_verify_layout is not None, meta.max_seq_len_q,
            ))
            try:
                return original(self, q, k, v, layer, forward_batch, *args, **kwargs)
            finally:
                _FORWARD.reset(token)
        return forward

    def make_fa(original):
        def flash(self, *args, ver, **kwargs):
            layer = kwargs.get("k_cache")
            if not isinstance(layer, Layer):
                return original(self, *args, ver=ver, **kwargs)
            if args or ver != 4 or kwargs.get("v_cache") is not layer:
                raise RuntimeError("inkling_kv12s: unsupported packed FA4 dispatch")
            return dispatch_packed_fa4(lambda **call: original(self, ver=ver, **call), layer, kwargs)
        return flash

    _wrap(cls, "__init__", "self model_runner skip_prefill speculative_step_id topk speculative_num_steps fa_impl_ver", make_init)
    _wrap(cls, "get_paged_mha_kv_cache", "self layer head_group_num", make_paged)
    _wrap(cls, "get_fused_kv_write_buffers", "self layer forward_batch", make_fused)
    common = ("self q k v layer forward_batch save_kv_cache q_rope k_rope sinks q_descale k_descale v_descale "
              "score_mod aux_tensors rel_bias rel_bias_event return_lse rel_bias_compact_decode rel_bias_decode_epilogue")
    _wrap(cls, "forward_extend", common + " rel_bias_presheared", make_forward)
    _wrap(cls, "forward_decode", common, make_forward)
    _wrap(cls, "flash_attn_with_kvcache", "self args ver kwargs", make_fa)


def _register_partials(module):
    from .storage import Layer

    def make_partials(original):
        def call(q, k_cache, v_cache, page_table, seqused_k, rel_bias, softmax_scale, out, *,
                 window_size_left, verify):
            if not isinstance(k_cache, Layer):
                return original(q, k_cache, v_cache, page_table, seqused_k, rel_bias, softmax_scale, out,
                                window_size_left=window_size_left, verify=verify)
            import torch
            from .attention import partials

            if window_size_left is not None:
                raise RuntimeError("inkling_kv12s: compact loader is for global visits")
            if rel_bias.shape[-1] == 128:
                from inkling_vattn.patch import _take

                proj = _take(rel_bias)
                if proj is None:
                    raise RuntimeError("inkling_kv12s: missing verify bias marker")
                rel_bias = torch.einsum("thd,de->the", rel_bias[..., :proj.shape[0]], proj)
            if rel_bias.shape[-1] != 1024:
                raise RuntimeError("inkling_kv12s: unexpected verify bias width")
            visit = _FORWARD.get()
            layout = {"cu_q": visit.cu_q, "qmax": visit.qmax} if visit is not None and visit.ragged else {}
            return partials(k_cache, q, page_table, seqused_k, rel_bias, softmax_scale,
                            device_metadata=True, **layout)
        return call

    _wrap(module, "rel_bias_partials",
          "q k_cache v_cache page_table seqused_k rel_bias softmax_scale out window_size_left verify", make_partials)


def _register_model(module):
    from inkling_vattn.patch import BYPASS_BIAS

    def keep_stock_projection(original):
        def prologue(self, *args, **kwargs):
            if self.is_local:
                return original(self, *args, **kwargs)
            # Packed partials consume the stock 1024-column BF16 projection.
            # Leave VATTN's in-kernel projection available on raw SWA layers.
            token = BYPASS_BIAS.set(True)
            try:
                return original(self, *args, **kwargs)
            finally:
                BYPASS_BIAS.reset(token)
        return prologue

    _wrap(module.Inkling2Attention, "_decode_prologue",
          "self q k v r sources positions forward_batch padded verify shear", keep_stock_projection)


def _register_dspark(module, eager=True):
    def save_window(original):
        def allocate(*args, **kwargs):
            window = original(*args, **kwargs)
            active = _VERIFY_WINDOW.get()
            if active is not None:
                active.append(window)
            return window
        return allocate

    def finish_verify(original):
        def decode(self, batch, on_publish, grammar_barrier=None, coordination_plan=None):
            from .static import view_of

            target = self.target_worker.model_runner
            view = view_of(target.token_to_kv_pool)
            if view is None:
                return original(self, batch, on_publish, grammar_barrier, coordination_plan)
            windows = []
            token = _VERIFY_WINDOW.set(windows)
            try:
                result = original(self, batch, on_publish, grammar_barrier, coordination_plan)
            finally:
                _VERIFY_WINDOW.reset(token)
            if windows:
                window, = windows
                commit = result.accept_lens
                iteration = getattr(window, "iteration_ids", None)
                if iteration is None:  # static pools: the verify ids are physical rows
                    ids = window.verify_cache_loc_2d
                else:
                    ids = iteration.write_ids(target.runner_ids).view_as(window.verify_cache_loc_2d)
                for lane in view.owned_layers:
                    lane.settle(ids, commit)
            return result
        return decode

    def eager_commit(original):
        def alloc(self, memory_pool_config=None, req_to_token_pool=None, token_to_kv_pool_allocator=None):
            # Unified FULL IDs carry a write translator, so stock's static fold
            # predicate is false, and the captured fold would pass FULL IDs only
            # (the routed draft's window layers need their own). Keep the
            # captured target verify and accept, and commit the draft eagerly
            # through Inkling2MTPVerifyExecutor.commit_hidden, which routes the
            # translated window IDs (see _register_verify).
            self._static_epilogue_needs_draft_fold = False
            return original(self, memory_pool_config, req_to_token_pool, token_to_kv_pool_allocator)
        return alloc

    _wrap(module, "alloc_verify_window",
          "batch bs device verify_num_draft_tokens block_pos_offsets model_runner seq_lens_cpu bonus anchor_ids",
          save_window)
    if eager:  # unified pools only: the static pools keep the captured draft fold
        _wrap(module.DSparkWorkerV2, "alloc_memory_pool",
              "self memory_pool_config req_to_token_pool token_to_kv_pool_allocator", eager_commit)
    _wrap(module.DSparkWorkerV2, "_forward_decode",
          "self batch on_publish grammar_barrier coordination_plan", finish_verify)


def _register_probe_dspark(module):
    from .pressure import outermost

    def probe(original):
        def forward(self, batch, on_publish=None, grammar_barrier=None, *, pp_proxy_tensors=None):
            return outermost(lambda: original(self, batch, on_publish, grammar_barrier,
                                              pp_proxy_tensors=pp_proxy_tensors), batch)
        return forward

    _wrap(module.DSparkWorkerV2, "forward_batch_generation",
          "self batch on_publish grammar_barrier pp_proxy_tensors", probe)


def _register_probe_worker(module):
    from .pressure import outermost

    def probe(original):
        def forward(self, *args, **kwargs):
            batch = args[0] if args else kwargs.get("batch")
            return outermost(lambda: original(self, *args, **kwargs), batch)
        return forward

    _wrap(module.TpModelWorker, "forward_batch_generation",
          "self batch forward_batch pp_proxy_tensors is_verify skip_attn_backend_init capture_hidden_mode "
          "pp_loras return_iteration_ids input_ids_from_graph", probe)


def _register_scheduler(module):
    from .pressure import guard

    def results(original):
        def process(self, batch, result):
            side = guard()
            if side is not None:
                side.on_result(self, batch, result)
            return original(self, batch, result)
        return process

    def next_batch(original):
        def plan(self, running_batch, last_batch):
            side = guard()
            if side is not None:
                side.before_schedule(self, running_batch, last_batch)
            return original(self, running_batch, last_batch)
        return plan

    def defer(original):
        def should_defer(self, running_batch):
            side = guard()
            return original(self, running_batch) or (side is not None and side.defer_prefill(running_batch))
        return should_defer

    _wrap(module.Scheduler, "process_batch_result", "self batch result", results)
    _wrap(module.Scheduler, "get_next_batch_to_run", "self running_batch last_batch", next_batch)
    _wrap(module.Scheduler, "_should_defer_prefill", "self running_batch", defer)


def _register_decode_capacity(module):
    from .pressure import guard

    def check(original):
        def fits(self, selected_indices=None):
            if not original(self, selected_indices):
                return False
            side = guard()
            requests = len(self.reqs) if selected_indices is None else len(selected_indices)
            # Inside the retraction loop (selected_indices given) measure again:
            # the retracted requests' slots are free now.
            return side is None or side.decode_fits(requests) or (
                selected_indices is not None and side.decode_fits(requests, measure=True))
        return fits

    _wrap(module.ScheduleBatch, "check_decode_mem", "self selected_indices", check)


def _register_verify(module, force=False):
    def no_fold(original):
        def resolve(self):
            if self.commit_ctx is None:
                return original(self)
            if force:
                return False
            target = self.commit_ctx.kv_injector
            arena = getattr(getattr(getattr(target, "model_runner", None), "token_to_kv_pool", None), "_arena", None)
            if getattr(arena, "_inkling_kv12s", None) is not None:
                return False
            return original(self)
        return resolve

    _wrap(module.DsparkVerifyEpilogue, "_resolve_draft_folds_static_commit", "self", no_fold)


def _register_sessions(module):
    def refuse_streaming(original):
        def open_session(self, recv_req):
            if getattr(recv_req, "streaming", False):
                # A streaming turn can resume inside a page that verify already
                # sealed; packed pages are immutable.
                LOG.warning("inkling_kv12s: streaming sessions are unsupported; refusing %s",
                            recv_req.session_id)
                from sglang.srt.managers.io_struct import OpenSessionReqOutput

                return OpenSessionReqOutput(session_id=recv_req.session_id, success=False)
            return original(self, recv_req)
        return open_session

    _wrap(module.SessionController, "open", "self recv_req", refuse_streaming)


def _register_runner(module):
    def verify_backend(original):
        def initialize(self):
            result = original(self)
            from torch.nn import ModuleList
            from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend

            from .static import view_of

            if _POOL == "static" and self.is_draft_worker:
                return result  # the draft keeps its private BF16 pools
            view = view_of(self.token_to_kv_pool)
            if view is None:
                raise IncompatibleImage("inkling_kv12s: no authoritative packed pool was installed")
            backend = self.attn_backend
            while hasattr(backend, "full_attn_backend"):
                backend = backend.full_attn_backend
            if not isinstance(backend, FlashAttentionBackend) or backend.fa_impl_ver != 4:
                raise IncompatibleImage("inkling_kv12s requires FA4 for the global visits")
            if self.is_draft_worker:
                routed = [
                    self.token_to_kv_pool.layer_router.layer(i)
                    for i in self.token_to_kv_pool.layer_router.layer_ids()
                ]
                draft = [route for route in routed if getattr(route.view, "is_inkling_kv12s", False)]
                if len(draft) != 1 or draft[0].view.layers[draft[0].view_layer_id] not in view.owned_layers[view.layer_num:]:
                    raise IncompatibleImage("inkling_kv12s: DSpark global layer has no packed draft lane")
                return result
            global_ids = set()
            for attention in self.model.modules():
                if type(attention).__name__ == "Inkling2Attention" and not attention.is_local:
                    groups = attention.attn if isinstance(attention.attn, ModuleList) else (attention.attn,)
                    global_ids.update(layer.layer_id for layer in groups)
            if global_ids != set(view.model_layer_ids):
                raise IncompatibleImage("inkling_kv12s: packed routes differ from the model's global visits")
            _check_verify_capture(self)
            return result
        return initialize

    _wrap(module.ModelRunner, "init_attention_backends", "self", verify_backend)


def _register_routed_cache(module):
    def make_head_dim(original):
        def value_head_dim(self):
            if getattr(self._owner_view, "is_inkling_kv12s", False):
                return self._owner_view.get_v_head_dim()
            return original(self)
        return value_head_dim

    _wrap(module.RoutedKVCache, "get_v_head_dim", "self", make_head_dim)


def _overflow_count(name, default):
    from . import side_count

    return side_count(name, default, 1)


_STATIC_SIZES = ContextVar("inkling_kv12s_static_sizes", default=None)


def _static_running():
    """The request bound the static sizing charges staging for (the server's --max-running-requests)."""
    from sglang.srt.runtime_context import get_server_args

    running = getattr(get_server_args(), "max_running_requests", None)
    if not running:
        raise IncompatibleImage("inkling_kv12s: the static pools need an explicit --max-running-requests")
    return int(running)


def _register_static_pools(module):
    """kv_cache_configurator: build the target's full sub-pool packed; tag its allocator."""
    from . import static

    cls = module.KVCacheConfigurator

    def sizes_scope(original):
        def build(self, *, sizes, is_dsa_model, is_compressed_kv_model, req_to_token_pool, state=None):
            token = _STATIC_SIZES.set(sizes)
            try:
                return original(self, sizes=sizes, is_dsa_model=is_dsa_model,
                                is_compressed_kv_model=is_compressed_kv_model,
                                req_to_token_pool=req_to_token_pool, state=state)
            finally:
                _STATIC_SIZES.reset(token)
        return build

    def packed_swa(original):
        def build(self, *, full_max_total_num_tokens, swa_max_total_num_tokens, mha_pool_class):
            if self.is_draft_worker:
                return original(self, full_max_total_num_tokens=full_max_total_num_tokens,
                                swa_max_total_num_tokens=swa_max_total_num_tokens, mha_pool_class=mha_pool_class)
            from sglang.srt.runtime_context import get_server_args

            sizes = _STATIC_SIZES.get()
            checks = {
                "mxfp8": self.kv_cache_dtype_str == "mxfp8",
                "pool_class": mha_pool_class is not module.MHATokenToKVPool,
                "no_sizes": sizes is None,
                "dtype": self.kv_cache_dtype != torch_bf16(),
                "post_capture": bool(self.post_capture_kv_active),
            }
            if any(checks.values()):
                raise IncompatibleImage(
                    "inkling_kv12s expects stock BF16 hybrid sliding-window pools: "
                    + ", ".join(name for name, bad in checks.items() if bad)
                    + f" (pool class {getattr(mha_pool_class, '__name__', mha_pool_class)}, dtype {self.kv_cache_dtype})")
            running = int(sizes.max_running_requests)
            if running > _static_running():
                raise IncompatibleImage(f"inkling_kv12s: {running} running requests exceed the "
                                        f"{_static_running()} the static sizing charged")
            args = get_server_args()
            parallel = module.get_parallel()
            head_num = self.model_config.get_num_kv_heads(parallel.attn_tp_size, parallel.attn_dcp_size)
            inkling = {
                "running": running,
                "verify_rows": max(1, module.max_speculative_num_draft_tokens() or 1),
                "overlap": not args.disable_overlap_schedule,
                "prefill_tokens": max(int(args.max_prefill_tokens or 0), int(args.chunked_prefill_size or 0)),
            }
            # Inkling's window layers declare their own head geometry ("compress"):
            # the stock builder hands it to the SWA sub-pool, as here.
            swa_kwargs = {}
            if self.is_hybrid_swa_compress:
                text = self.model_config.hf_text_config
                swa_kwargs = {
                    "swa_head_num": max(1, text.swa_num_key_value_heads // parallel.attn_tp_size),
                    "swa_head_dim": self.model_config.swa_head_dim,
                    "swa_v_head_dim": self.model_config.swa_v_head_dim,
                    "v_head_dim": self.model_config.v_head_dim,
                }
            return module.SWAKVPool(
                size=full_max_total_num_tokens,
                size_swa=swa_max_total_num_tokens,
                page_size=self.pool_page_size,
                dtype=self.kv_cache_dtype,
                head_num=head_num,
                head_dim=self.model_config.head_dim,
                swa_attention_layer_ids=self.layer_info.swa_attention_layer_ids,
                full_attention_layer_ids=self.layer_info.full_attention_layer_ids,
                device=self.device,
                enable_kv_cache_copy=(module.get_spec().speculative_algorithm is not None),
                token_to_kv_pool_class=mha_pool_class,
                full_kv_pool_class=static.pool_class(),
                full_kv_pool_kwargs={
                    "page_size": self.pool_page_size, "enable_memory_saver": False, "device": self.device,
                    "head_num": head_num, "head_dim": self.model_config.head_dim,
                    "v_head_dim": self.model_config.v_head_dim,
                    "allocation_label": "Full (packed)", "inkling": inkling,
                },
                enable_memory_saver=module.get_exec().features.enable_memory_saver,
                **swa_kwargs,
            )
        return build

    def tag_allocator(original):
        def build(self, *, sizes, token_to_kv_pool, is_compressed_kv_model, req_to_token_pool,
                  token_to_kv_pool_allocator):
            allocator = original(self, sizes=sizes, token_to_kv_pool=token_to_kv_pool,
                                 is_compressed_kv_model=is_compressed_kv_model,
                                 req_to_token_pool=req_to_token_pool,
                                 token_to_kv_pool_allocator=token_to_kv_pool_allocator)
            view = None if self.is_draft_worker else static.view_of(token_to_kv_pool)
            if view is not None:
                from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

                full = getattr(allocator, "full_attn_allocator", None)
                if (not isinstance(full, PagedTokenToKVPoolAllocator) or full.page_size != view.page_size
                        or getattr(allocator, "hands_out_virtual_ids", False)):
                    raise IncompatibleImage("inkling_kv12s expects a paged hybrid SWA allocator of physical pages")
                full._inkling_kv12s_view = view
            return allocator
        return build

    _wrap(cls, "_build_token_to_kv_pool", "self sizes is_dsa_model is_compressed_kv_model req_to_token_pool state",
          sizes_scope)
    _wrap(cls, "_build_hybrid_swa_kv_pool",
          "self full_max_total_num_tokens swa_max_total_num_tokens mha_pool_class", packed_swa)
    _wrap(cls, "_build_token_to_kv_pool_allocator",
          "self sizes token_to_kv_pool is_compressed_kv_model req_to_token_pool token_to_kv_pool_allocator",
          tag_allocator)


def torch_bf16():
    import torch

    return torch.bfloat16


def _register_static_sizing(module):
    """pool_configurator: a full-side target token costs the packed envelope; the
    fixed packed bytes come off the budget before it is divided."""
    from . import static

    cls = module.HybridSWAPoolConfigurator

    def init(original):
        def configure(self, kvc):
            original(self, kvc)
            self._inkling_kv12s = not kvc.is_draft_worker
            if self._inkling_kv12s:
                from sglang.srt.mem_cache.kv_cache_configurator import max_speculative_num_draft_tokens
                from sglang.srt.runtime_context import get_server_args

                if self._full_layers_num != static.VISITS or self._fused_full_entry is not None:
                    raise IncompatibleImage("inkling_kv12s expects eight global visits in a private full pool")
                self._inkling_fixed = static.fixed_bytes(
                    _static_running(), max(1, max_speculative_num_draft_tokens() or 1),
                    overlap=not get_server_args().disable_overlap_schedule)
                lanes = static.DRAFT_WINDOW_LANES
                lane = self._full_per_token  # one draft lane: the target's visit geometry (8 x 256 x 2 B)
                if lanes and not self._draft_follows_swa and self._draft_cell_size:
                    draft_cfg = getattr(getattr(kvc, "spec_aux_config", None), "draft_model_config", None)
                    local = layers = None
                    for hf in (getattr(draft_cfg, "hf_text_config", None), getattr(draft_cfg, "hf_config", None)):
                        mtp = getattr(hf, "mtp_config", None)
                        mtp = mtp if isinstance(mtp, dict) else {}
                        local = getattr(hf, "local_layer_ids", mtp.get("local_layer_ids"))
                        layers = getattr(hf, "num_hidden_layers", None) if "local_layer_ids" not in mtp else \
                            mtp.get("num_hidden_layers")
                        if local is not None and layers is not None:
                            break
                    if local is None or layers is None or len(set(local)) != lanes or layers != 1 + lanes:
                        raise IncompatibleImage(
                            f"inkling_kv12s: the DSpark draft has local layers {local} of {layers}, not "
                            f"{lanes} window layers plus one global (set INKLING_KV12S_DRAFT_WINDOW_LANES=0 "
                            "for stock pricing)")
                    # Stock prices the separate DSpark draft pool per FULL token
                    # (draft_num_layers lanes); its window lanes are allocated at the
                    # window pool's size (startup log: "SWA KV Cache ... #tokens: <swa
                    # tokens>"): price those per window token, as the target's.
                    if self._draft_cell_size != (1 + lanes) * lane:
                        raise IncompatibleImage(
                            f"inkling_kv12s: draft pool of {self._draft_cell_size} B/token is not one global "
                            f"lane plus {lanes} window lanes of {lane} B (set INKLING_KV12S_DRAFT_WINDOW_LANES=0 "
                            "for stock pricing)")
                    self._draft_cell_size -= lanes * lane
                    self._draft_swa_cell_size += lanes * lane
                self._recompute_cell_size()
        return configure

    def cell(original):
        def full_cell_bytes(self):
            if not getattr(self, "_inkling_kv12s", False):
                return original(self)
            return static.cell_bytes() + self._full_per_token * self._draft_full_layers_num
        return full_cell_bytes

    def budget(original):
        def calculate(self, available_bytes, page_size):
            if getattr(self, "_inkling_kv12s", False):
                available_bytes = available_bytes - self._inkling_fixed
                LOG.info("inkling_kv12s: static sizing: %.2f GiB fixed packed bytes, %d B per full token",
                         self._inkling_fixed / 2**30, self._cell_size)
            return original(self, available_bytes, page_size)
        return calculate

    _wrap(cls, "__init__", "self kvc", init)
    _wrap(cls, "_full_cell_bytes", "self", cell)
    _wrap(cls, "calculate_pool_sizes", "self available_bytes page_size", budget)
    # The request-capped subclass overrides calculate_pool_sizes without calling it.
    capped = getattr(module, "SWARequestCapPoolConfigurator", None)
    if capped is not None and capped.calculate_pool_sizes is not cls.calculate_pool_sizes:
        _wrap(capped, "calculate_pool_sizes", "self available_bytes page_size", budget)


def _register_static_release(module):
    """allocator.paged: every freed full page releases its descriptors and side slots."""
    cls = module.PagedTokenToKVPoolAllocator

    def release(original):
        def release_page_ids(self, *page_ids):
            view = getattr(self, "_inkling_kv12s_view", None)
            if view is not None and page_ids:
                import torch

                view.release_pages(torch.cat([ids.reshape(-1) for ids in page_ids]))
            return original(self, *page_ids)
        return release_page_ids

    def clear(original):
        def reset(self):
            result = original(self)
            view = getattr(self, "_inkling_kv12s_view", None)
            if view is not None:
                view.release_all()
            return result
        return reset

    _wrap(cls, "_release_page_ids", "self page_ids", release)
    _wrap(cls, "clear", "self", clear)


def _register_static_capture(module):
    """model_runner.capture_cuda_graphs: mark warmup and capture of the target runner."""
    from . import static

    original = getattr(module, "capture_cuda_graphs", None)
    if original is None or "model_runner" not in inspect.signature(original).parameters:
        raise IncompatibleImage("inkling_kv12s: unexpected model_runner.capture_cuda_graphs")
    if getattr(original, "_inkling_kv12s", False):
        return

    @wraps(original)
    def capture_cuda_graphs(*args, **kwargs):
        static.CAPTURING.append(True)
        try:
            return original(*args, **kwargs)
        finally:
            static.CAPTURING.pop()

    capture_cuda_graphs._inkling_kv12s = True
    module.capture_cuda_graphs = capture_cuda_graphs


def _register_static_swa_pool(module):
    def head_dim(original):
        def get_v_head_dim(self):
            if getattr(self.full_kv_pool, "is_inkling_kv12s", False) and self.full_layer_nums > 0:
                return self.full_kv_pool.get_v_head_dim()
            return original(self)
        return get_v_head_dim

    _wrap(module.SWAKVPool, "get_v_head_dim", "self", head_dim)


def register():
    from . import side_count

    side_count("INKLING_KV12S_FALLBACK_PAGES", "64", 0)  # refuse a bad count before any sizing
    overflow = _overflow_count("INKLING_KV12S_OVERFLOW_HEADS", "2048")
    if _POOL == "static":
        register_static()
        return
    if _POOL != "unified":
        raise IncompatibleImage(f"inkling_kv12s: INKLING_KV12S_POOL={_POOL!r} must be static or unified")
    draft_overflow = _overflow_count("INKLING_KV12S_DRAFT_OVERFLOW_HEADS", "2048")
    patches = (
        ("sglang.srt.mem_cache.kv.plan", lambda m: _register_plan(m, overflow, draft_overflow)),
        ("sglang.srt.mem_cache.kv.sizing", _register_sizing),
        ("sglang.srt.mem_cache.kv.build", _register_build),
        ("sglang.srt.mem_cache.unified_draft_pool", _register_draft_binding),
        ("sglang.srt.mem_cache.kv_cache_configurator", _register_config),
        ("sglang.srt.mem_cache.kv.alloc.band", _register_lifetime),
        ("sglang.srt.mem_cache.kv.routed_cache", _register_routed_cache),
        ("sglang.srt.model_executor.model_runner", _register_runner),
        ("sglang.srt.mem_cache.kv.alloc.band", _register_move),
        ("sglang.srt.mem_cache.kv.memory.snapshot", _register_snapshot),
        ("sglang.srt.layers.attention.flashattention_backend", _register_backend),
        ("sglang.kernels.ops.attention.inkling2_attn", _register_partials),
        ("sglang.srt.models.inkling_common.inkling2_attn", _register_model),
        ("sglang.srt.speculative.dspark_components.dspark_worker_v2", _register_dspark),
        ("sglang.srt.speculative.dspark_components.dspark_worker_v2", _register_probe_dspark),
        ("sglang.srt.managers.tp_worker", _register_probe_worker),
        ("sglang.srt.managers.scheduler", _register_scheduler),
        ("sglang.srt.managers.schedule_batch", _register_decode_capacity),
        ("sglang.srt.speculative.dspark_components.dspark_verify", _register_verify),
        ("sglang.srt.session.session_controller", _register_sessions),
    )
    for target, patch in patches:
        when_imported(target, patch)


def register_static():
    """The packed store on the static (non-unified) pools: see static.py."""
    patches = (
        ("sglang.srt.mem_cache.kv_cache_configurator", _register_config),
        ("sglang.srt.mem_cache.kv_cache_configurator", _register_static_pools),
        ("sglang.srt.model_executor.pool_configurator", _register_static_sizing),
        ("sglang.srt.mem_cache.allocator.paged", _register_static_release),
        ("sglang.srt.mem_cache.swa_memory_pool", _register_static_swa_pool),
        ("sglang.srt.model_executor.model_runner", _register_runner),
        ("sglang.srt.model_executor.model_runner", _register_static_capture),
        ("sglang.srt.layers.attention.flashattention_backend", _register_backend),
        ("sglang.kernels.ops.attention.inkling2_attn", _register_partials),
        ("sglang.srt.models.inkling_common.inkling2_attn", _register_model),
        ("sglang.srt.speculative.dspark_components.dspark_worker_v2", lambda m: _register_dspark(m, eager=False)),
        ("sglang.srt.speculative.dspark_components.dspark_worker_v2", _register_probe_dspark),
        ("sglang.srt.managers.tp_worker", _register_probe_worker),
        ("sglang.srt.managers.scheduler", _register_scheduler),
        ("sglang.srt.managers.schedule_batch", _register_decode_capacity),
        ("sglang.srt.session.session_controller", _register_sessions),
    )
    for target, patch in patches:
        when_imported(target, patch)


def register_eager_commit_only():
    """Diagnostic for stock pools (INKLING_DSPARK_EAGER_COMMIT=1 without KV12S):
    the same eager draft commit as KV12S, to separate its numerics from storage."""
    def worker(module):
        def eager_commit(original):
            def alloc(self, memory_pool_config=None, req_to_token_pool=None, token_to_kv_pool_allocator=None):
                self._static_epilogue_needs_draft_fold = False
                return original(self, memory_pool_config, req_to_token_pool, token_to_kv_pool_allocator)
            return alloc

        _wrap(module.DSparkWorkerV2, "alloc_memory_pool",
              "self memory_pool_config req_to_token_pool token_to_kv_pool_allocator", eager_commit)

    when_imported("sglang.srt.speculative.dspark_components.dspark_worker_v2", worker)
    when_imported("sglang.srt.speculative.dspark_components.dspark_verify",
                  lambda m: _register_verify(m, force=True))


def register_vattn_trace():
    """Diagnostic (INKLING_KV12S_VATTN_TRACE=N): log the compact verify attention
    arguments of every Nth call, to compare pool configurations."""
    every = max(1, int(os.environ.get("INKLING_KV12S_VATTN_TRACE", "0") or 1))
    count = [0]
    seen = set()

    def wrap(module):
        original = module.inkernel_bias_partials

        def traced(q, k_cache, v_cache, page_table, seqused_k, r, proj, softmax_scale, *, window_size_left=None,
                   config=None):
            count[0] += 1
            capturing = __import__("torch").cuda.is_current_stream_capturing()
            key = (tuple(q.shape), tuple(k_cache.stride()), tuple(page_table.shape), window_size_left, capturing)
            if key not in seen or (not capturing and count[0] % every == 0):
                seen.add(key)
                seq = "graph" if capturing else seqused_k.tolist()
                LOG.warning("vattn-trace capturing=%s q=%s k=%s kstride=%s table=%s seq=%s window=%s proj=%s",
                            capturing, tuple(q.shape), tuple(k_cache.shape), tuple(k_cache.stride()),
                            tuple(page_table.shape), seq if capturing else (min(seq), max(seq), len(seq)),
                            window_size_left, tuple(proj.shape))
            return original(q, k_cache, v_cache, page_table, seqused_k, r, proj, softmax_scale,
                            window_size_left=window_size_left, config=config)

        module.inkernel_bias_partials = traced

    when_imported("inkling_vattn.ops", wrap)
