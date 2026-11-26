"""(B) hooks: DFlashDraftKVRing.record_floors (plan + restore), DFlashWorkerV2._init_draft_kv_ring (store),
DFlashWorkerV2.forward_batch_generation (captures after the forward), DFlashWorkerV2.clear_cache_pool (flush).

A failure inside a hook never fails a request: the snapshots go inert for the process (draft-only state; the target
path is untouched either way) and the reason is logged once."""

from . import snapshot as sn

RING_MODULE = "sglang.srt.speculative.dflash_draft_ring"
WORKER_MODULE = "sglang.srt.speculative.dflash_worker_v2"
CONFIG_MODULE = "sglang.srt.mem_cache.kv_cache_configurator"
UPDATER_MODULE = "sglang.srt.managers.scheduler_components.weight_updater"
MODULES = (CONFIG_MODULE, RING_MODULE, WORKER_MODULE, UPDATER_MODULE)
MARK = "_inkling_dftail_b"
POOL_TYPES = ("MHATokenToKVPool",)  # token-major [rows, heads, dim] per layer; quantized / page-major / HND refused


def _disable(ring, why):
    store = getattr(ring, "_inkling_snap", None) if ring is not None else None
    if store is not None:
        ring._inkling_snap = None
        store.release()
    if not sn.STATE["inert"]:
        sn.STATE["inert"] = why
        sn._log(f"draft-tail snapshots inert: {why}")


def make_record_floors(orig):
    def record_floors(self, reqs, prefix_lens, req_generation):
        store = getattr(self, "_inkling_snap", None)
        if store is None:
            return orig(self, reqs, prefix_lens, req_generation)
        try:
            restores, fresh = sn.plan_floors(store, reqs, prefix_lens, req_generation)
        except Exception as exc:  # noqa: BLE001
            _disable(self, f"planning failed: {exc!r}")
            return orig(self, reqs, prefix_lens, req_generation)
        floors = [int(p) for p in prefix_lens]
        done = []
        try:
            # every copy is enqueued for the floor it will have, BEFORE that floor is published below
            for i, slot_idx, generation, slot, entry, lo, P in restores:
                store.restore(slot_idx, slot, entry, lo, P)
                floors[i] = lo
                done.append((lo, P, entry))
        except Exception as exc:  # noqa: BLE001
            # a request whose copies did not all enqueue keeps its stock floor: rows below it are never read
            floors = [int(p) for p in prefix_lens]
            done = []
            _disable(self, f"restore failed: {exc!r}")
        out = orig(self, reqs, floors, req_generation)
        st = getattr(self, "_inkling_snap", None)
        if st is not None:
            try:
                sn.record_mirror(st, fresh, floors)
                sn.plan_captures(st, reqs, prefix_lens, req_generation)
            except Exception as exc:  # noqa: BLE001
                _disable(self, f"capture planning failed: {exc!r}")
        for lo, P, entry in done:
            sn.STATE["stats"]["restores"] += 1
            sn.STATE["stats"]["restored_positions"] += P - lo
            n = sn.STATE["stats"]["restores"]
            if n == 1:
                sn._log(f"first restore: {P - lo} draft positions [{lo}, {P}) from a snapshot ending at {entry.end}")
            if n % 32 == 0:
                sn._log(f"snapshot stats: {dict(sn.STATE['stats'])}")
        return out

    record_floors.__wrapped__ = orig
    setattr(record_floors, MARK, True)
    return record_floors


def drain(store, req_generation):
    """Copy this forward's queued captures (after its appends, on the same stream)."""
    pending, store.pending = store.pending, []
    for slot_idx, generation, lo, end, digest, tail, salt, extra in pending:
        if int(req_generation[slot_idx]) != generation:
            continue
        slot = store.index.place(digest, end, lo, tail, salt, extra)
        if slot is None:
            sn.STATE["stats"]["refreshes"] += 1
            continue
        store.capture(slot_idx, slot, lo, end)
        sn.STATE["stats"]["captures"] += 1


def _capturing(store):
    try:
        return bool(store.torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001  (CPU builds)
        return False


def make_forward(orig):
    def forward_batch_generation(self, batch, *args, **kwargs):
        ring = getattr(self, "draft_kv_ring", None)
        store = getattr(ring, "_inkling_snap", None)
        if store is not None:
            store.pending = []
        out = orig(self, batch, *args, **kwargs)
        store = getattr(ring, "_inkling_snap", None)
        if store is not None and store.pending:
            if _capturing(store):
                store.pending = []  # never inside a CUDA graph capture (warm-up)
            else:
                try:
                    drain(store, self.model_runner.req_to_token_pool.req_generation)
                except Exception as exc:  # noqa: BLE001
                    _disable(ring, f"capture failed: {exc!r}")
        store = getattr(ring, "_inkling_snap", None)
        if store is not None and sn.STATE["defer"]:
            store.index.epoch += 1  # reservations pinned before this forward stop pinning
        return out

    forward_batch_generation.__wrapped__ = orig
    setattr(forward_batch_generation, MARK, True)
    return forward_batch_generation


def build_store(worker, slots):
    """The snapshot store for the worker's ring, or None (with the reason logged) when this server cannot use it."""
    ring = getattr(worker, "draft_kv_ring", None)
    if ring is None:
        _disable(None, "no DFLASH draft KV ring (SGLANG_ENABLE_DFLASH_DRAFT_KV_RING off)")
        return None
    try:
        from sglang.srt.runtime_context import get_disagg

        mode = getattr(get_disagg(), "disaggregation_mode", "null")
    except Exception:  # noqa: BLE001
        mode = "null"
    if mode not in (None, "null"):
        _disable(ring, f"PD disaggregation ({mode}) is not supported")
        return None
    if int(ring.dcp_size) != 1:
        _disable(ring, f"DCP {ring.dcp_size} is not supported")
        return None
    pool = ring.pool
    if getattr(pool, "is_quantized_kv_cache", False) or getattr(pool, "store_dtype", None) != getattr(pool, "dtype", None):
        _disable(ring, "a quantized draft KV pool is not supported")
        return None
    layout = getattr(pool, "kv_cache_layout", None)
    if type(pool).__name__ not in POOL_TYPES or getattr(pool, "use_hnd", False) or layout not in (None, "nhd"):
        _disable(ring, f"draft pool {type(pool).__name__} (layout {layout}, hnd {getattr(pool, 'use_hnd', False)}) "
                       "is not the token-major NHD pool")
        return None
    try:
        layer_ids = [int(layer.self_attn.attn.layer_id) for layer in worker.draft_model.layers]
        start = int(getattr(pool, "start_layer", 0) or 0)
        k_bufs = [pool.k_buffer[i - start] for i in layer_ids]
        v_bufs = [pool.v_buffer[i - start] for i in layer_ids]
    except Exception as exc:  # noqa: BLE001
        _disable(ring, f"draft pool layout not recognised: {exc!r}")
        return None
    charged = sn.STATE["charged"]
    page = int(ring.sink_rows) if getattr(ring, "paged", False) else 1
    per_slot = (int(ring.window_left) + 1 + sn.extra_rows(page)) * sum(
        int(b[0].numel()) * b.element_size() for b in k_bufs + v_bufs)
    if charged is None or slots * per_slot > charged:
        _disable(ring, f"the snapshot pool ({slots * per_slot} B) was not charged to the KV budget (charged {charged})")
        return None
    rows = int(ring.sink_rows) + int(ring.min_pos.shape[0]) * int(ring.rows_per_req)
    for b in k_bufs + v_bufs:
        if b.dim() != 3 or int(b.shape[0]) < rows * int(ring.dcp_size):
            _disable(ring, f"draft KV buffer shape {tuple(b.shape)} does not hold the ring's {rows} rows token-major")
            return None
    store = sn.Store(ring, k_bufs, v_bufs, slots)
    ring._inkling_snap = store
    if sn.STATE["defer"]:
        try:
            import torch.distributed as dist

            sn.STATE["solo"] = not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1
        except Exception:  # noqa: BLE001
            sn.STATE["solo"] = False
    if sn.STATE["defer"] and not sn.STATE["solo"]:
        sn._log("prefix tail does not defer to snapshots: more than one rank (each would decide on its own slots)")
    sn.STATE["stores"].add(store)
    sn._log(f"draft-tail snapshots: {slots} slots x {store.length} positions, {len(layer_ids)} draft layers, "
            f"{store.nbytes / 2**20:.0f} MiB")
    return store


def make_init_ring(orig, slots):
    def _init_draft_kv_ring(self):
        orig(self)
        try:
            build_store(self, slots)
        except Exception as exc:  # noqa: BLE001
            _disable(getattr(self, "draft_kv_ring", None), f"store allocation failed: {exc!r}")

    _init_draft_kv_ring.__wrapped__ = orig
    setattr(_init_draft_kv_ring, MARK, True)
    return _init_draft_kv_ring


def make_clear(orig):
    def clear_cache_pool(self):
        out = orig(self)
        store = getattr(getattr(self, "draft_kv_ring", None), "_inkling_snap", None)
        if store is not None:
            store.clear()
        return out

    clear_cache_pool.__wrapped__ = orig
    setattr(clear_cache_pool, MARK, True)
    return clear_cache_pool




def make_ring_bytes(orig, slots):
    """KVCacheConfigurator._dflash_draft_kv_ring_bytes: the target's KV budget is charged the ring; charge the
    snapshot pool with it (slots x window x the draft's per-token cell), whatever the static fraction is."""
    def _dflash_draft_kv_ring_bytes(self):
        ring_bytes = orig(self)
        if ring_bytes <= 0:
            return ring_bytes
        from sglang.srt.speculative.dflash_utils import get_dflash_attention_sliding_window_size

        window = int(get_dflash_attention_sliding_window_size(self.spec_aux_config.draft_model_config.hf_config)) + 1
        if sn.STATE["defer"]:  # one page per slot more: the ring's page (the stock ring-bytes expression)
            from sglang.srt.runtime_context import get_schedule

            window += sn.extra_rows(get_schedule().page_size * self.dflash_draft_loc_space_scale)
        snap = int(slots) * window * int(self.dflash_draft_cell_size_per_token)
        sn.STATE["charged"] = snap
        sn._log(f"draft-tail snapshots: {snap / 2**30:.3f} GiB charged to the KV budget ({slots} x {window} positions)")
        return ring_bytes + snap

    _dflash_draft_kv_ring_bytes.__wrapped__ = orig
    setattr(_dflash_draft_kv_ring_bytes, MARK, True)
    return _dflash_draft_kv_ring_bytes


def clear_all(why):
    for store in list(sn.STATE["stores"]):
        store.clear()


def make_select_runners(orig):
    """SchedulerWeightUpdaterManager._select_runners: every weight load (disk, tensor, distributed, IPC, plugin)
    selects its runners here; target or draft weights changing makes every snapshot stale."""
    def _select_runners(self, *args, **kwargs):
        clear_all("weight update")
        return orig(self, *args, **kwargs)

    _select_runners.__wrapped__ = orig
    setattr(_select_runners, MARK, True)
    return _select_runners


def patch_config(mod, slots, hint=""):
    cls = getattr(mod, "KVCacheConfigurator", None)
    _need(cls, ["_dflash_draft_kv_ring_bytes"], "KVCacheConfigurator", hint)
    if getattr(cls._dflash_draft_kv_ring_bytes, MARK, False):
        return False
    cls._dflash_draft_kv_ring_bytes = make_ring_bytes(cls._dflash_draft_kv_ring_bytes, slots)
    return True


def patch_updater(mod, hint=""):
    cls = getattr(mod, "SchedulerWeightUpdaterManager", None)
    _need(cls, ["_select_runners"], "SchedulerWeightUpdaterManager", hint)
    if getattr(cls._select_runners, MARK, False):
        return False
    cls._select_runners = make_select_runners(cls._select_runners)
    return True


def _need(cls, names, where, hint):
    if cls is None:
        raise RuntimeError(f"inkling_dftail: {where} is missing" + hint)
    for name in names:
        if getattr(cls, name, None) is None:
            raise RuntimeError(f"inkling_dftail: {where}.{name} is missing" + hint)


def patch_ring(mod, hint=""):
    cls = getattr(mod, "DFlashDraftKVRing", None)
    _need(cls, ["record_floors", "locs"], "DFlashDraftKVRing", hint)
    if getattr(cls.record_floors, MARK, False):
        return False
    cls.record_floors = make_record_floors(cls.record_floors)
    return True


def patch_worker(mod, slots, hint=""):
    cls = getattr(mod, "DFlashWorkerV2", None)
    _need(cls, ["_init_draft_kv_ring", "forward_batch_generation", "clear_cache_pool"], "DFlashWorkerV2", hint)
    if getattr(cls._init_draft_kv_ring, MARK, False):
        return False
    cls._init_draft_kv_ring = make_init_ring(cls._init_draft_kv_ring, slots)
    cls.forward_batch_generation = make_forward(cls.forward_batch_generation)
    cls.clear_cache_pool = make_clear(cls.clear_cache_pool)
    return True
