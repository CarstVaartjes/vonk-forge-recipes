"""Weights registered for the wide GEMM and the per-call paths (17..MAX_M rows).

A Wide entry points at the bytes the kernel streams: the layer's own BF16 weight (fmt RAW, natural [N, K]
layout, no copy) or inkling_pico12's exactly coded copy of it (fmt BF16X, pico's `bf16x` format, ~0.775 of the
bytes; read in place, never copied). Both formats give the same output bits (the decode restores the BF16
values exactly and the MMA sequence is the same).

Math: out[m, n] = sum_k x[m, k] W[n, k], BF16 operands, FP32 accumulation over K in a fixed order (per 32-value
step: two m16n8k16 MMAs; split-K slices summed in slice order), FP32 out or one rounding to BF16. Against the stock
path (cuBLAS) only the FP32 summation order differs.

Per call: torch.empty outputs (the graph pool under capture), one launch on the current stream, no host sync, no
host-device copy, a path chosen by shapes only; the split-K workspace and counters are allocated at load time.
"""

import torch

MAX_ROWS = 256  # rows the kernel takes (dsk-style buckets of 16..256)
BLOCK_N = 128  # weight rows per CTA: stacked-region bounds must be multiples of it
RAW, BF16X = 0, 2
NO_REGION = 2**31 - 1


class Settings:
    min_m = 17  # rows of a call taken (plain calls; per source for stacked q/k/v calls)
    max_m = 256
    # per weight kind: rows up to which wide beats cuBLAS on GB10 (gpu_check_wide, 2026-11-21, L2-cold): qkvr
    # 1.02x at 96 rows (stacked 1.13x at 85 per source), wo_ud 1.06x at 32 / 1.00x at 48, the head 1.18x at 256
    max_m_kind = {"qkvr": 96, "wo": 32, "head": 256}
    pdl = True
    fmt = "auto"  # auto: pico's coded copy when inkling_pico12 made one, else the raw weight; raw: always raw
    splitk = {}  # kind -> forced split-K slices (env INKLING_WIDE_SPLITK)
    sms = 48
    mt = 0  # tiles per warp above 128 rows: 0 = the kernel's default (1), 1 or 2 (env INKLING_WIDE_MT)
    mg = 2  # warp groups along M at 65..128 rows: 2 (default; wo 96 rows 89 -> 76 us, qkvr flat) or 1 (env INKLING_WIDE_MG)
    scope = "verify"  # verify: only inside target decode / target-verify forwards (env INKLING_WIDE_SCOPE); all
    pfd = 1  # L2 prefetch distance in 64-value batches (env INKLING_WIDE_PFD; 0 = off): 1 best or flat everywhere
    require_cuda = True  # CPU tests turn this off and install a stand-in EXT


SET = Settings()
EXT = None
_batch_invariant = None
_warned = set()
LOG = None
WS = {"ws": None, "cnt": None}
# True while the target model runs a decode or target-verify forward (set by the patched
# Inkling2ForConditionalGeneration.forward from forward_batch.forward_mode, a Python attribute fixed at CUDA-graph
# capture). With SET.scope == "verify" wide takes calls only then: prefill / extend chunks keep the stock path.
IN_SCOPE = [False]  # split-K workspace (FP32) and counters (int32), shared by every call


class WideFatal(BaseException):
    """A broken patch target or an impossible state: not caught by `except Exception`."""


def _warn_once(key, msg):
    if key not in _warned:
        _warned.add(key)
        if LOG is not None:
            LOG(msg)


def bucket(m):
    """m16 tiles of the kernel for m rows (the kernel's buckets)."""
    need = (m + 15) // 16
    for b in (1, 2, 3, 4, 6, 8, 12, 16):
        if b >= need:
            return b
    raise ValueError(f"inkling_wide: {m} rows > {MAX_ROWS}")


def ctas_per_sm(m):
    """The occupancy the kernel is built for at m rows (launch bounds: 2 CTAs up to 64 rows)."""
    return 2 if bucket(m) <= 4 else 1


def choose_split(n, k, m, kind=None):
    """Split-K slices for an [n, k] weight at m rows: forced per kind (INKLING_WIDE_SPLITK), else from the GB10 sweep
    (gpu_check_wide --sweep, 2026-11-21): no split once the column blocks (128 rows each) cover the SMs (qkvr: 68
    blocks, the head: 1571; every split measured slower); below that, enough slices for one CTA per SM up to 64
    rows (wo_ud, 18 blocks: 3 slices; 4 measured within 2 %) and two thirds of that above (2 slices), where the
    FP32 partials' traffic starts to cost more than idle SMs. The partials must fit in L2."""
    nb = k // 64
    if kind is not None and kind in SET.splitk:
        return max(1, min(SET.splitk[kind], nb))
    nblk = -(-n // BLOCK_N)
    if nblk >= SET.sms:
        return 1
    want = SET.sms / nblk if m <= 64 else SET.sms / nblk / 1.5
    s = max(1, min(-(-int(want * 1000) // 1000), nb // 4, 16))
    while s > 1 and s * m * n * 4 > 24 * 2**20:
        s -= 1
    return s


def shape_reason(n, k):
    if n % 8:
        return f"N={n} not a multiple of 8"
    if k % 64:
        return f"K={k} not a multiple of 64"
    if n >= 2**31 or k >= 2**31:
        return f"[{n}, {k}] too large"
    return None


class Wide:
    """One weight the wide GEMM reads: RAW (the BF16 tensor) or BF16X (pico's Pack, read through .buf)."""

    __slots__ = ("n", "k", "fmt", "weight", "pack", "wptr", "label", "kind")

    def __init__(self, weight, label, kind, pack=None):
        n, k = weight.shape
        self.n, self.k, self.label, self.kind = n, k, label, kind
        self.fmt = BF16X if pack is not None else RAW
        self.weight, self.pack = weight, pack
        self.wptr = weight.data_ptr()

    def workspace_floats(self, min_m, max_m):
        """FP32 partials the largest split call of min_m..max_m rows needs (0: never splits)."""
        need = 0
        for m in range(min_m, max_m + 1):
            s = choose_split(self.n, self.k, m, self.kind)
            if s > 1:
                need = max(need, s * m * self.n)
        return need

    @property
    def buf(self):
        return self.pack.buf if self.fmt == BF16X else self.weight

    @property
    def nbytes(self):
        return self.buf.numel() * self.buf.element_size()

    def __repr__(self):
        return f"Wide({self.label} [{self.n}x{self.k}] {'bf16x' if self.fmt == BF16X else 'raw'} {self.kind})"


def make(weight, label, kind, pico_pack=None):
    """Register [N, K] BF16 `weight`; pico_pack (inkling_pico12's Pack of the same storage) selects BF16X."""
    w = weight.detach()
    if w.dtype != torch.bfloat16 or w.dim() != 2:
        raise ValueError(f"inkling_wide: {label}: not a BF16 matrix")
    why = shape_reason(*w.shape)
    if why is not None:
        raise ValueError(f"inkling_wide: {label}: {why}")
    if not w.is_contiguous() or w.data_ptr() % 16:
        raise ValueError(f"inkling_wide: {label}: weight not contiguous or not 16-byte aligned")
    # the kernel holds a bf16x tile row's headers in 3 registers a lane: K <= 3072 (larger K reads the raw weight)
    use_pack = pico_pack is not None and SET.fmt == "auto" and w.shape[1] <= 3072
    if use_pack and ((pico_pack.n, pico_pack.k) != tuple(w.shape) or pico_pack.wptr != w.data_ptr()):
        raise WideFatal(f"inkling_wide: {label}: inkling_pico12's pack is not of this weight")
    return Wide(w, label, kind, pico_pack if use_pack else None)


def ensure_workspace(entries, min_m, max_m, device, grow=False, floats=0):
    """Allocate (load time) the split-K workspace (FP32) and counters (int32, zero) for every entry that may split
    in min_m..max_m rows. The buffers are process-wide and referenced by captured CUDA graphs: they are never
    replaced unless `grow` (offline tools); a reload that needs more raises WideFatal (restart the server).
    Calls must not run concurrently on two streams (SGLang's forward runs on one)."""
    need = max([floats] + [e.workspace_floats(min_m, max_m) for e in entries])
    nblk = max([1] + [-(-e.n // BLOCK_N) for e in entries])
    ws, cnt = WS["ws"], WS["cnt"]
    if ws is None or (grow and (ws.numel() < need or cnt.numel() < nblk)):
        WS["ws"] = torch.empty(max(need, 4), dtype=torch.float32, device=device)
        WS["cnt"] = torch.zeros(max(nblk, 64), dtype=torch.int32, device=device)
    elif ws.numel() < need or cnt.numel() < nblk:
        raise WideFatal(f"inkling_wide: split-K workspace of {ws.numel()} floats / {cnt.numel()} counters (captured in "
                        f"CUDA graphs) cannot hold {need} / {nblk} after a reload: restart the server")
    return 4 * WS["ws"].numel()


def mode_ok():
    if torch.compiler.is_compiling():
        return False
    return not (_batch_invariant is not None and _batch_invariant())


def rows_of(x):
    return x.numel() // x.shape[-1] if x.dim() >= 1 and x.shape[-1] else 0


def x_ok(x, k):
    return (
        x.dtype == torch.bfloat16
        and x.dim() >= 1
        and x.shape[-1] == k
        and (x.is_cuda or not SET.require_cuda)
        and not x.requires_grad
    )


def in_range(m, kind=None):
    if SET.scope == "verify" and not IN_SCOPE[0]:
        return False
    return SET.min_m <= m <= min(SET.max_m, SET.max_m_kind.get(kind, SET.max_m))


def weight_current(weight, entry):
    if weight.data_ptr() == entry.wptr and (entry.fmt == RAW or entry.pack.wptr == entry.wptr):
        return True
    _warn_once(("wptr", entry.label), f"{entry.label}: weight storage changed since registration; stock path from now on")
    return False


def eligible(x, entry):
    return (
        in_range(rows_of(x), entry.kind)
        and x_ok(x, entry.k)
        and (x.device == entry.weight.device or not SET.require_cuda)
        and mode_ok()
    )


def as_rows(x, k):
    """x[..., K] as [M, K] with unit column stride and 16-byte aligned rows (a copy only when x is not)."""
    x2 = x.reshape(-1, k)
    if x2.stride(-1) != 1 or (x2.shape[0] > 1 and x2.stride(0) != k):
        x2 = x2.contiguous()
    if x2.data_ptr() % 16:
        x2 = x2.clone()
    return x2


def out_ok(out, m, entry, dtype, device):
    return (
        out.dtype == dtype
        and out.device == device
        and out.dim() == 2
        and tuple(out.shape) == (m, entry.n)
        and out.stride(-1) == 1
        and (m == 1 or out.stride(0) == entry.n)
        and out.data_ptr() % (8 if dtype == torch.float32 else 4) == 0
    )


def gemm(x2, entry, out, m, x1=None, x2b=None, regions=(NO_REGION, NO_REGION, NO_REGION), roffs=(0, 0)):
    s = choose_split(entry.n, entry.k, m, entry.kind)
    EXT.gemm(
        x2, x2 if x1 is None else x1, x2 if x2b is None else x2b, m, entry.buf, entry.fmt, entry.n, entry.k,
        regions[0], regions[1], regions[2], roffs[0], roffs[1], out, s, WS["ws"], WS["cnt"], SET.pdl, SET.mt, SET.mg, SET.pfd,
    )
    return out


def linear(x, entry, out_dtype, out=None):
    """x[..., K] @ W^T in out_dtype, into `out` ([M, N], checked by the caller) when given."""
    lead = x.shape[:-1]
    x2 = as_rows(x, entry.k)
    m = x2.shape[0]
    if out is None:
        out = torch.empty((m, entry.n), dtype=out_dtype, device=x.device)
    gemm(x2, entry, out, m)
    return out.view(*lead, entry.n)


# ---- stacked qkvr ---------------------------------------------------------------------------------------


def regions_of(split_sizes):
    """Column regions (k_lo, v_lo, v_hi) of the k and v outputs of the qkvr projection (split order q, k, v, ...),
    or None if a bound is not a multiple of BLOCK_N (a CTA must serve one source)."""
    if len(split_sizes) < 3:
        return None
    a = int(split_sizes[0])
    b = a + int(split_sizes[1])
    c = b + int(split_sizes[2])
    if a % BLOCK_N or b % BLOCK_N or c % BLOCK_N or not 0 < a < b < c:
        return None
    return a, b, c


def stacked_eligible(sources, entry, regions):
    q, k, v = sources
    if regions is None or q.dim() != 2 or not (q.shape == k.shape == v.shape):
        return False
    if not (q.dtype == k.dtype == v.dtype == torch.bfloat16) or not (q.device == k.device == v.device):
        return False
    return (
        in_range(q.shape[0], entry.kind)
        and x_ok(q, entry.k)
        and (q.device == entry.weight.device or not SET.require_cuda)
        and not k.requires_grad
        and not v.requires_grad
        and mode_ok()
    )


def _rows(x, k):
    """x as [M, K] with rows of stride K (the three sources must share one row stride)."""
    if x.stride(-1) != 1 or (x.shape[0] > 1 and x.stride(0) != k) or x.data_ptr() % 16:
        x = x.contiguous()
        if x.data_ptr() % 16:
            x = x.clone()
    return x


def stacked_project(q, k, v, entry, regions):
    """BF16 [3M, N] with what qkvr(cat(q, k, v))[0] holds on every element Inkling2Attention._project_sources
    keeps: rows [0, M) on every column outside [k_lo, v_hi), rows [M, 2M) on the k columns, rows [2M, 3M) on
    the v columns, from one read of the weight. The other elements are left unwritten (never read)."""
    m = q.shape[0]
    q2, k2, v2 = _rows(q, entry.k), _rows(k, entry.k), _rows(v, entry.k)
    out = torch.empty((3 * m, entry.n), dtype=torch.bfloat16, device=q.device)
    return gemm(q2, entry, out, m, x1=k2, x2b=v2, regions=regions, roffs=(m, 2 * m))


def split_like_stock(out, split_sizes):
    """The (q, k, v, r, gate, sinks) views _project_sources returns for a stacked [3M, N] projection."""
    q_proj, k_proj, v_proj = (chunk.split(split_sizes, dim=-1)[:6] for chunk in out.chunk(3))
    q, _, _, r, gate, sinks = q_proj
    return q, k_proj[1], v_proj[2], r, gate, sinks
