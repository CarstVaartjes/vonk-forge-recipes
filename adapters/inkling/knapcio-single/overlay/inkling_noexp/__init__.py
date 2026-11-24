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
STATE = {"installed": False, "applied": False, "calls": 0, "passthrough": 0}


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


def make_route(mod, stock):
    route_metadata = mod.route_metadata
    max_ids = mod.ROUTE_MAX_IDS

    def _sm120_route(self, ids, weights):
        import torch

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
    _log("enabled: dead verify rows get no experts (route over E + 1 experts, dead ids -> E)")
    return True
