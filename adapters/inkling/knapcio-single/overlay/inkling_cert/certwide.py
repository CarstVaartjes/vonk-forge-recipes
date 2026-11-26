"""Certified packed head for 17–192 activation rows. The public result is token IDs."""

import threading

import torch

from . import cert_math, quant

MIN_ROWS = 17
MAX_ROWS = 192
MAX_TILES = 2048
PARTS = 16
# Give each path 2 * C_ACC to cover the compounded rounding model.
CN = 4 * cert_math.C_ACC
MAX_WEIGHT_NORM = 2.0**28
ABS_SLOP = 2.0**-78

CERTIFIED, UNSAFE, TOO_MANY, OFF, INVALID = range(5)


def weight_radius(tables):
    """Upper bound on ||W-D|| + CN * max(||W||, ||D||), one FP32 value per vocab ID."""
    nv = tables["nv"]
    if not bool((nv <= MAX_WEIGHT_NORM).all()):
        raise ValueError("inkling_certwide: weight norms outside the supported range")
    r = tables["rt"].double().square().sum(0).sqrt() + CN * nv.double()
    return quant.round_up_f32(r * (1.0 + 2.0**-40)).contiguous()


class CertWide:
    """Caller-owned fallback logit buffer. Launches and scratch are fixed under graph capture.

    All serving replays and gate writes use one host thread and CUDA stream.
    Standalone callers must maintain the same ordering through reduction.
    """

    def __init__(self, screen_ext, wide_ext, entry, tables, capacity=MAX_TILES):
        from . import certwide_reduce

        self.screen_ext, self.wide_ext, self.entry = screen_ext, wide_ext, entry
        self.reduce = certwide_reduce.reduce
        self.q, self.sc = tables["q"], tables["sc"]
        self.radius = weight_radius(tables)
        self.V = entry.n
        if entry.k != cert_math.K or self.V <= 0 or self.V % cert_math.TILE:
            raise ValueError("inkling_certwide: invalid packed head geometry")
        self._replay_owner = None
        self._owner_lock = threading.Lock()
        self.ntiles = self.V // cert_math.TILE
        self.capacity = min(int(capacity), self.ntiles)
        if self.capacity < 1:
            raise ValueError("inkling_certwide: tile capacity must be positive")
        device = self.q.device
        self.lower = torch.empty((MAX_ROWS, self.ntiles), dtype=torch.float32, device=device)
        self.upper = torch.empty_like(self.lower)
        self.norms = torch.empty(MAX_ROWS, dtype=torch.float32, device=device)
        self.threshold = torch.empty_like(self.norms)
        self.state = torch.empty(2, dtype=torch.int32, device=device)
        self.tiles = torch.empty(self.capacity, dtype=torch.int32, device=device)
        self.groups = torch.empty_like(self.tiles)
        self.sparse = torch.empty((MAX_ROWS, self.capacity * cert_math.TILE), dtype=torch.float32, device=device)
        self.partial = torch.empty((MAX_ROWS, PARTS), dtype=torch.int64, device=device)
        self.empty = torch.empty(0, dtype=torch.float32, device=device)
        self.stats = None  # optional [MAX_ROWS + 1, 5 outcomes, UNION_BUCKETS] int64 (enable_stats)
        self.counting = True  # False while the startup / reload self-tests run
        screen_ext.prepare_certwide(device.index)
        wide_ext.prepare_certwide(device.index)

    def require_replay_stream(self):
        stream = torch.cuda.current_stream(self.q.device)
        owner = (threading.current_thread(), stream)
        with self._owner_lock:
            if self._replay_owner is None:
                self._replay_owner = owner
            elif self._replay_owner != owner:
                raise RuntimeError("inkling_certwide: verify gate and replay must use the owning thread and CUDA stream")

    def screen(self, x, on, v_real, debug=None, *, three_stages=False):
        s, lo, hi = (self.empty,) * 3 if debug is None else debug
        self.screen_ext.screen_wide(
            x, self.q, self.sc, self.radius, self.V, v_real, self.norms, self.lower, self.upper,
            self.threshold, on, self.state, self.tiles, self.groups, s, lo, hi,
            cert_math.CAP, cert_math.TWO_E, cert_math.DMAX, three_stages,
        )

    def exact(self, x, fallback, v_real, settings):
        self.wide_ext.certwide_exact(
            x, self.entry.buf, fallback, self.sparse[:x.shape[0]],
            self.state, self.tiles, self.groups, self.entry.n, v_real, settings.mt, settings.mg, settings.pfd,
        )

    def run(self, x, fallback, on, v_real, settings, *, debug=None, capped=None, three_stages=False):
        """Return [M] int64 IDs. On fallback only, also write capped, masked logits."""
        m = x.shape[0]
        if capped is not None and (capped.shape != self.sparse.shape or capped.device != self.sparse.device
                                   or capped.dtype != self.sparse.dtype or not capped.is_contiguous()):
            raise ValueError("inkling_certwide: invalid capped debug buffer")
        ids = torch.empty(m, dtype=torch.int64, device=x.device)
        self.screen(x, on, v_real, debug, three_stages=three_stages)
        self.exact(x, fallback, v_real, settings)
        self.reduce(fallback, self.sparse, self.state, self.tiles, self.groups,
                    self.partial, ids, v_real, cert_math.CAP, capped)
        if self.stats is not None and self.counting:
            self._count(m)
        return ids

    UNION_STEP = 256    # tiles per union bucket
    UNION_BUCKETS = 17  # the last one holds unions >= 16 * UNION_STEP (> the 2048-tile capacity)

    def enable_stats(self):
        """Diagnostic counters, written inside the graph after each run (three small launches, no sync)."""
        device = self.state.device
        self.stats = torch.zeros((MAX_ROWS + 1, 5, self.UNION_BUCKETS), dtype=torch.int64, device=device)
        self._one = torch.ones(1, dtype=torch.int64, device=device)

    def _count(self, m):
        reason = self.state[0:1].clamp(0, 4).long()  # CERTIFIED, UNSAFE, TOO_MANY, OFF, INVALID
        bucket = torch.div(self.state[1:2], self.UNION_STEP, rounding_mode="floor").clamp_(0, self.UNION_BUCKETS - 1)
        self.stats[m].view(-1).index_add_(0, reason * self.UNION_BUCKETS + bucket.long(), self._one)

    def describe_stats(self, host):
        names = ("cert", "unsafe", "too_many", "off", "invalid")
        parts = []
        for m in range(host.shape[0]):
            per = host[m]
            if int(per.sum()) == 0:
                continue
            counts = " ".join(f"{names[r]}={int(per[r].sum())}" for r in range(5) if int(per[r].sum()))
            union = ",".join(f"{b * self.UNION_STEP}+:{int(per[:3, b].sum())}"
                             for b in range(self.UNION_BUCKETS) if int(per[:3, b].sum()))
            parts.append(f"M={m}: {counts} (union tiles {union})")
        return "; ".join(parts) if parts else "none"

    def refresh(self, entry, tables):
        # The graph captures the table and packed-buffer addresses.
        if (entry.n != self.V or entry.k != self.entry.k
                or self.entry.buf.data_ptr() != entry.buf.data_ptr()):
            raise ValueError("inkling_certwide: reload changed a graph buffer address")
        self.radius.copy_(weight_radius(tables))
        self.entry = entry
