"""inkling_padx patches: the decode graph runner (live row count per replay, capture scope) and
Inkling2Experts._sm120_route (padding rows -> phantom expert). See the package docstring."""

import functools
import sys

from . import BIG, MARK, PHANTOM, STATE, PadxFatal


def _real_buffer(runner):
    """The runner's device int32 [1] live-row count, created at capture start (BIG: every row live)."""
    buf = getattr(runner, "_inkling_padx_real", None)
    if buf is None:
        import torch

        device = getattr(getattr(runner, "model_runner", None), "device", None)
        if device is None:
            raise PadxFatal("inkling_padx: graph runner has no model_runner.device")
        buf = torch.full((1,), BIG, dtype=torch.int32, device=device)
        runner._inkling_padx_real = buf
    return buf


def patch_runner(mod):
    cls = getattr(mod, "DecodeCudaGraphRunner", None)
    if cls is None or not hasattr(cls, "_set_dp_real_num_tokens") or not hasattr(cls, "capture"):
        raise PadxFatal("inkling_padx: DecodeCudaGraphRunner._set_dp_real_num_tokens / capture missing")
    if getattr(cls._set_dp_real_num_tokens, MARK, False):
        return False
    orig_set, orig_capture = cls._set_dp_real_num_tokens, cls.capture

    @functools.wraps(orig_set)
    def _set_dp_real_num_tokens(self, raw_num_token):
        orig_set(self, raw_num_token)
        buf = getattr(self, "_inkling_padx_real", None)
        if buf is not None:  # only runners that captured with padx
            buf.fill_(int(raw_num_token))
            STATE["fills"] += 1

    @functools.wraps(orig_capture)
    def capture(self):
        buf = _real_buffer(self)
        buf.fill_(BIG)
        prev, STATE["runner"] = STATE["runner"], self
        try:
            return orig_capture(self)
        finally:
            STATE["runner"] = prev

    setattr(_set_dp_real_num_tokens, MARK, True)
    setattr(capture, MARK, True)
    cls._set_dp_real_num_tokens = _set_dp_real_num_tokens
    cls.capture = capture
    STATE["applied"] += 1
    return True


def can_pad(runner, rows):
    """Static, decided at capture: can a replay of this graph carry padding rows? The runner pads a batch of n live
    requests to the smallest captured size >= n, so a graph of bs requests is replayed with padding only when bs - 1
    is not itself a captured size (with speculative decoding: 1..8 are all captured, then 10, 12, ...). Graphs that
    never pad keep the stock route (no extra graph nodes: the c1..c8 verify graphs are unchanged). Unknown layout
    (no capture list, rows not a multiple of the request width): mask (exact either way)."""
    sizes = getattr(runner, "capture_bs", None)
    width = int(getattr(runner, "captured_req_width", 0) or 0)
    if not sizes or width <= 0 or rows % width:
        return True
    bs = rows // width
    return bs > 1 and (bs - 1) not in set(int(b) for b in sizes)


def _noexp():
    m = sys.modules.get("inkling_noexp")
    return m if m is not None and m.STATE.get("applied") else None


def make_route(mod, inner):
    route_metadata = mod.route_metadata
    max_ids = mod.ROUTE_MAX_IDS

    def _sm120_route(self, ids, weights):
        import torch

        runner = STATE["runner"]
        real = getattr(runner, "_inkling_padx_real", None) if runner is not None else None
        if (real is None or ids.dim() != 2 or self.num_global_experts != PHANTOM or ids.numel() > max_ids
                or not ids.is_cuda or not torch.cuda.is_current_stream_capturing()
                or not can_pad(runner, ids.shape[0])):
            STATE["passthrough"] += 1
            return inner(self, ids, weights)
        m = ids.shape[0]
        live = torch.arange(m, device=ids.device, dtype=torch.int32) < real  # captured: real is read at replay
        nx = _noexp()
        base = ids
        if nx is not None:
            pending = nx.STATE.get("masked")
            if pending is not None and ids is pending[0]:  # noexp's fused remap already wrote the dead-row mask
                nx.STATE["masked"] = None
                base = pending[1]
            else:
                dead_live = nx.live_rows(m) if nx.STATE.get("applied") else None
                if dead_live is not None:
                    base = torch.where(dead_live[:, None], ids, torch.full_like(ids, PHANTOM))
        masked = torch.where(live[:, None], base.to(torch.int32), torch.full_like(base, PHANTOM, dtype=torch.int32))
        STATE["calls"] += 1
        src2dst, dst2src, gammas, _, offsets, *_ = route_metadata(masked.flatten(), weights.flatten(), 0, PHANTOM,
                                                                  PHANTOM + 1)
        return src2dst, dst2src, gammas, offsets

    _sm120_route.__wrapped__ = inner
    setattr(_sm120_route, MARK, True)
    return _sm120_route


def patch_model(mod):
    cls = getattr(mod, "Inkling2Experts", None)
    inner = getattr(cls, "_sm120_route", None) if cls is not None else None
    if inner is None or not hasattr(mod, "route_metadata") or not hasattr(mod, "ROUTE_MAX_IDS"):
        raise PadxFatal("inkling_padx: Inkling2Experts._sm120_route / route_metadata / ROUTE_MAX_IDS missing")
    if getattr(inner, MARK, False):
        return False
    cls._sm120_route = make_route(mod, inner)
    STATE["applied"] += 1
    return True
