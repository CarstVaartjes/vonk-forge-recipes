"""inkling_noexp: dead verify rows get no experts (exact for live rows).

    INKLING_NOEXP=1     on (unset / 0: nothing is imported). Needs INKLING_VERIFY_CAP with its remap (the default).

inkling_vcap verifies only rows 0..L-1 of each request; the dead rows L..7 stay in the static graph and today take
their anchor row's experts (same expert union). They still add routed rows: quantization, grouped-GEMM N tiles,
SwiGLU and combine all scale with 8 x rows. Here a dead row's 8 expert ids are replaced by the out-of-range id
E (= number of experts) and route_metadata runs over E + 1 global experts with the E real ones local: the dead
rows' routed rows sort past offsets[E] and every SM120 kernel ignores them (quantization, both grouped GEMMs and
the combine only touch rows in [offsets[0], offsets[E]); the combine gives a dead row 0).

Exact for live rows: each live routed row is quantized, multiplied and combined by the same per-row arithmetic;
only the set of other rows in the grouped GEMM tiles changes, which does not enter a row's result (GPU check:
tools/gpu_check_noexp.py compares live rows bit for bit against the anchor remap). Dead rows hold different values
than before; vcap's exactness argument never used their values (they are later than every live row of their
request, never accepted, never committed).

Patch: Inkling2Experts._sm120_route (sglang.srt.models.inkling_common.inkling2_moe). Active only while vcap marks a
capped target-verify MoE call (the same condition under which vcap remaps), only during CUDA-graph capture (eager
calls, incl. other overlays' bitwise self-checks, keep stock routing), only with 128 experts and on the
route_metadata path (<= 4096 ids); graph-safe (the live mask is read from vcap's device src_index at replay: live row r <=> src_index[r] == r).
"""

import importlib.util
import os
import sys

from . import _hook

MODEL_MODULE = "sglang.srt.models.inkling_common.inkling2_moe"
MARK = "_inkling_noexp"
_NEEDLES = (
    "def _sm120_route(self, ids, weights):",
    "src2dst, dst2src, gammas, _, offsets, *_ = route_metadata(",
    "if flat.numel() <= ROUTE_MAX_IDS:",
    "src2dst, dst2src, gammas, offsets = self._sm120_route(ids, weights)",
)
STATE = {"installed": False, "applied": False, "calls": 0, "passthrough": 0, "masked": None, "fused_mask": 0,
         "mask_ok": None}
PHANTOM = 128  # = num_global_experts of Inkling-2-Mini; _sm120_route re-checks it


class NoexpFatal(BaseException):
    """BaseException: SGLang's model registry swallows Exception during model imports."""


def _log(msg):
    print(f"[inkling_noexp] {msg}", file=sys.stderr, flush=True)


_OFF = ("", "0", "off", "false", "no", "none")


def enabled(env=None):
    env = os.environ if env is None else env
    on = env.get("INKLING_NOEXP", "").strip().lower()
    if on in _OFF:
        return False
    if on not in ("1", "on", "true", "yes"):
        raise RuntimeError(f"inkling_noexp: INKLING_NOEXP={on!r} (use 1 or 0)")
    remap = env.get("INKLING_VERIFY_CAP_REMAP", "").strip().lower()  # vcap _parse_bool, default on
    if env.get("INKLING_VERIFY_CAP", "").strip().lower() in _OFF or remap in ("0", "false", "off", "no"):
        # Without the verify cap's dead-row remap there are no dead rows to skip: stay out (stock routing).
        _log("inactive: INKLING_VERIFY_CAP or its dead-row remap is off, so there are no dead verify rows")
        return False
    return True


def check_sources(root=None):
    if root is None:
        spec = importlib.util.find_spec("sglang")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("inkling_noexp: the sglang package is not importable")
        root = list(spec.submodule_search_locations)[0]
    path = os.path.join(root, "srt/models/inkling_common/inkling2_moe.py")
    try:
        text = open(path, encoding="utf-8").read()
    except OSError as exc:
        raise RuntimeError(f"inkling_noexp: patch target {path} is missing") from exc
    missing = [n for n in _NEEDLES if n not in text]
    if missing:
        raise RuntimeError(f"inkling_noexp: patch target {path} lacks {missing}")


def live_rows(m):
    """Bool [m] device mask of live verify rows from vcap's state, or None when this call is not a capped
    target-verify MoE call (then nothing changes)."""
    vcap = sys.modules.get("inkling_vcap.state")
    st = getattr(vcap, "STATE", None) if vcap is not None else None
    if st is None or not (st.enabled and st.in_verify_moe and st.capping and st.config.remap):
        return None
    src = st.src_index
    if not st.stride or m % st.stride or m > src.numel():
        return None
    import torch

    return src[:m] == torch.arange(m, device=src.device, dtype=src.dtype)


def route_dead_free(route_metadata, ids, weights, live, experts):
    """route_metadata over E + 1 global experts with dead rows' ids set to E (non-local)."""
    import torch

    masked = torch.where(live[:, None], ids.to(torch.int32), torch.full_like(ids, experts, dtype=torch.int32))
    src2dst, dst2src, gammas, _, offsets, *_ = route_metadata(masked.flatten(), weights.flatten(), 0, experts,
                                                              experts + 1)
    return src2dst, dst2src, gammas, offsets


def _on_device(t):
    return t.is_cuda


def _route_limit():
    mod = sys.modules.get(MODEL_MODULE)
    return getattr(mod, "ROUTE_MAX_IDS", None) if mod is not None else None


def make_remap(orig):
    """vcap VcapState.remap wrapper. On the calls noexp would serve (CUDA-graph capture, vcap's fused remap in use,
    at most ROUTE_MAX_IDS ids, the mask self-check passed) one launch writes vcap's two outputs (returned unchanged:
    every gate consumer sees the v2.7.2 tensors) and the phantom-masked ids, remembered for _sm120_route, which
    routes them when it receives exactly that ids tensor. Any other case: vcap's own remap, and _sm120_route masks
    as in v2.7.2."""

    def remap(self, weights, ids):
        import torch

        STATE["masked"] = None
        m = ids.shape[0] if ids.dim() == 2 else -1
        limit = _route_limit()
        if (not STATE["applied"] or STATE["mask_ok"] is not True or not torch.cuda.is_current_stream_capturing()
                or not getattr(self, "use_fused", False) or limit is None or m <= 0 or not self.stride
                or m % self.stride or m > self.src_index.numel() or ids.numel() > limit or not _on_device(ids)
                or ids.stride(1) != 1 or weights.shape != ids.shape or weights.stride(1) != 1
                or ids.shape[1] > 64):
            return orig(self, weights, ids)
        from ._mask import remap_mask

        self.remap_recorded += 1  # as vcap's remap does on this path
        w, i, masked = remap_mask(weights, ids, self.src_index, PHANTOM)
        STATE["masked"] = (i, masked)
        STATE["fused_mask"] += 1
        return w, i

    remap.__wrapped__ = orig
    setattr(remap, MARK, True)
    return remap


def patch_vcap(mod):
    cls = getattr(mod, "VcapState", None)
    orig = getattr(cls, "remap", None) if cls is not None else None
    if orig is None:
        raise NoexpFatal("inkling_noexp: inkling_vcap.state.VcapState.remap is missing")
    if getattr(orig, MARK, False):
        return False
    cls.remap = make_remap(orig)
    return True


def patch_vcap_fused(mod):
    """Run the mask kernel's GPU self-check right after vcap's own fused-remap self-check (worker init, eager,
    before any capture); without a pass the fused mask stays off and v2.7.2's path serves."""
    orig = getattr(mod, "selfcheck", None)
    if orig is None:
        raise NoexpFatal("inkling_noexp: inkling_vcap.fused.selfcheck is missing")
    if getattr(orig, MARK, False):
        return False

    def selfcheck(device):
        err = orig(device)
        if err is None:
            try:
                from ._mask import selfcheck as mask_check

                why = mask_check(device)
            except Exception as exc:  # noqa: BLE001 - any failure keeps the v2.7.2 path
                why = repr(exc)
            STATE["mask_ok"] = why is None
            _log("fused phantom mask " + ("on (self-check passed)" if why is None else f"off ({why})"))
        return err

    selfcheck.__wrapped__ = orig
    setattr(selfcheck, MARK, True)
    mod.selfcheck = selfcheck
    return True


def make_route(mod, stock):
    route_metadata = mod.route_metadata
    max_ids = mod.ROUTE_MAX_IDS

    def _sm120_route(self, ids, weights):
        import torch

        pending = STATE["masked"]
        if (pending is not None and ids is pending[0] and self.num_global_experts == PHANTOM
                and ids.numel() <= max_ids and torch.cuda.is_current_stream_capturing()):
            # the phantom-masked twin of exactly these ids, written by the remap launch
            STATE["masked"] = None
            STATE["calls"] += 1
            src2dst, dst2src, gammas, _, offsets, *_ = route_metadata(pending[1].flatten(), weights.flatten(), 0,
                                                                      PHANTOM, PHANTOM + 1)
            return src2dst, dst2src, gammas, offsets

        # Only inside CUDA-graph capture (the served verify graphs): eager calls keep stock routing, so the eager
        # bitwise self-checks of inkling_scodes / inkling_moe2 / inkling_vquant (which compare whole output buffers,
        # including rows past offsets[E] that noexp leaves unwritten) never see a noexp call.
        live = (live_rows(ids.shape[0]) if ids.dim() == 2 and self.num_global_experts == 128
                and torch.cuda.is_current_stream_capturing() else None)
        if live is None or ids.numel() > max_ids:
            STATE["passthrough"] += 1
            return stock(self, ids, weights)
        STATE["calls"] += 1
        return route_dead_free(route_metadata, ids, weights, live, self.num_global_experts)

    _sm120_route.__wrapped__ = stock
    setattr(_sm120_route, MARK, True)
    return _sm120_route


def patch(mod):
    cls = getattr(mod, "Inkling2Experts", None)
    stock = getattr(cls, "_sm120_route", None) if cls is not None else None
    if stock is None or not hasattr(mod, "route_metadata") or not hasattr(mod, "ROUTE_MAX_IDS"):
        raise NoexpFatal("inkling_noexp: Inkling2Experts._sm120_route / route_metadata / ROUTE_MAX_IDS missing")
    if getattr(stock, MARK, False):
        return False
    cls._sm120_route = make_route(mod, stock)
    STATE["applied"] = True
    return True


def install(env=None, root=None):
    if STATE["installed"]:
        return STATE["applied"]
    STATE["installed"] = True
    if not enabled(env):
        return False
    check_sources(root)
    _hook.when_imported(MODEL_MODULE, patch)
    _hook.when_imported("inkling_vcap.state", patch_vcap)
    _hook.when_imported("inkling_vcap.fused", patch_vcap_fused)
    _log("enabled: dead verify rows get no experts (route over E + 1 experts, dead ids -> E)")
    return True
