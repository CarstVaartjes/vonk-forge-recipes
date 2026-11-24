"""Per-call device work of the certified head (graph-safe: fixed launches, no host sync, no allocation besides
torch.empty in the graph pool) and the start-up self-test.

Cert.run(x2, out, on) launches screen (k_screen, k_rowmax, k_select) and exact (k_exact): out holds pico's FP32
logits on the candidate tiles and -inf elsewhere when the device flag `on` is 1 and the step is certifiable, and
pico's logits everywhere otherwise (the masked kernel then runs every tile: the full head, the same bits).
"""

import torch

from . import cert_math

K = cert_math.K
MAX_ROWS = 16


class CertFatal(BaseException):
    """A broken patch target or an impossible state (not caught by `except Exception`)."""


class Cert:
    def __init__(self, ext, pack, tables, wk, device):
        self.ext, self.pack, self.wk = ext, pack, int(wk)
        self.V = int(pack.n)
        if pack.k != K or self.V % 16:
            raise CertFatal(f"inkling_cert: head [{pack.n}x{pack.k}] is not [16n x {K}]")
        self.q, self.sc, self.rt, self.nv = tables["q"], tables["sc"], tables["rt"], tables["nv"]
        self.eps = float(tables["eps"])
        self.ncta = self.V // 16
        f32, i32 = torch.float32, torch.int32
        self.lt = torch.empty((MAX_ROWS, self.ncta), dtype=f32, device=device)
        self.ht = torch.empty((MAX_ROWS, self.ncta), dtype=f32, device=device)
        self.tm = torch.empty((MAX_ROWS,), dtype=f32, device=device)
        self.hm = torch.empty((MAX_ROWS,), dtype=f32, device=device)
        self.on = torch.zeros((1,), dtype=i32, device=device)    # host-written before each verify replay
        self.off = torch.zeros((1,), dtype=i32, device=device)   # eager calls: always the full head
        self.force = torch.ones((1,), dtype=i32, device=device)  # tests / self-test: certify if possible
        self.flags = torch.zeros((4,), dtype=i32, device=device)
        self.mask = torch.empty((self.ncta,), dtype=torch.uint8, device=device)
        # stats: 0 candidate tiles on certified steps, 1 certified steps, 2 eligible but not certifiable,
        # 3 (unused), 4 steps with on = 0, 5 rows checked, 6 token mismatches (check mode), 7 check steps
        self.stats = torch.zeros((8,), dtype=torch.int64, device=device)
        self.on_host = 0
        self._empty = torch.empty((0,), dtype=f32, device=device)
        self.warm = set()

    @property
    def screen_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.q, self.sc, self.rt, self.nv))

    def set_on(self, v):
        v = 1 if v else 0
        if v != self.on_host:
            self.on.fill_(v)
            self.on_host = v

    def screen(self, x2, on, v_real, cap, sdbg=None, bdbg=None):
        e = self._empty
        self.ext.screen(x2, self.q, self.sc, self.rt, self.nv, self.V, int(v_real), cert_math.CN, self.lt, self.ht,
                        self.tm, self.hm, on, self.flags, self.mask, self.stats, float(cap), cert_math.TWO_E,
                        cert_math.DMAX, e if sdbg is None else sdbg, e if bdbg is None else bdbg)

    def run(self, x2, out, on, v_real, cap):
        self.screen(x2, on, v_real, cap)
        self.ext.exact(x2, self.pack.buf, self.V, K, self.mask, out, self.wk, self.flags)
        return out

    def run_check(self, x2, out, on, v_real, cap, full_fn, softcap_fn, argmax_fn):
        """Check mode: out gets pico's full head (what is served); the certified buffer is built beside it and both
        go through the stock softcap, padding mask and argmax; mismatching rows of certified steps are counted."""
        c = torch.empty_like(out)
        self.run(x2, c, on, v_real, cap)
        full_fn(x2, out)
        a = c.clone()
        b = out.clone()
        softcap_fn(a, cap)
        softcap_fn(b, cap)
        a[:, v_real:] = float("-inf")
        b[:, v_real:] = float("-inf")
        cert = self.flags[2:3].to(torch.int64)
        self.stats[5:6].add_(cert * x2.shape[0])
        self.stats[6:7].add_((argmax_fn(a) != argmax_fn(b)).sum().view(1) * cert)
        self.stats[7:8].add_(cert)
        return out
