"""NumPy model of the certified greedy LM-head argmax (what cert_ext.cu computes), for the CPU tests and as the
statement of the argument. No torch, no sglang.

Stock pipeline (the reference the certificate must reproduce), per verify row m:
  l = pico's FP32 head GEMV of x_m (inkling_pico12 `lmhead`), f(l) = fl(fl(tanh(fl(l / c))) * c) (stock softcap,
  c = 30), y_v = f(l_v) for v < v_real and -inf for the padded ids, token = the lowest v with y_v = max y (the
  stock argmax kernel's tie rule; with finite x every l is finite, so there is no NaN).

Certified pipeline:
  screen  s_v ~ <D_v, x> (6-bit copy), B_v = sum_g Rt[g, v] ||x_g|| + CN Nv[v] ||x||, lo = s - B - slop,
          hi = s + B + slop; T = max_{v < v_real} lo_v; H = max hi.
  select  tile (16 ids) is computed iff some row has a tile member with hi >= T - delta, where delta (from T, H) makes
          every pair a < b in [T - DMAX, H] with f(a) >= f(b) satisfy b - a <= delta. Not certifiable (all tiles
          computed): the step is not eligible, a non-finite value, or the range test fails.
  exact   computed tiles hold pico's l (the same kernel body: the same bits); every other id holds -inf. Then the
          stock softcap, padding mask and argmax run unchanged.

Why the token is the same. (1) |acc - <W_v, x>| and |s - <D_v, x>| are both <= C_ACC * N_v * ||x|| (accumulation
model below; flushed subnormal inputs or products add at most K * 2^-126 * (||W_v||_1 + ||x||_1), covered by the
absolute slop while ||x|| <= 2^20, which the screen checks) and |<W_v - D_v, x>| <= sum_g ||R_vg|| ||x_g||
(Cauchy-Schwarz per group), so lo_v <= l_v <= hi_v. (2) Let w be the stock token and u the id with lo_u = T, so
T <= l_u <= hi_u <= H, and f(l_w) >= f(l_u) with |f - g| <= E, g = c tanh(. / c) increasing. If l_w >= l_u then
hi_w >= l_w >= l_u >= T. Otherwise g(l_u) - g(l_w) <= 2E. Every y in [l_u - DMAX, l_u] has |y| <= L = max(|T| + DMAX,
|H|) and g' = sech^2(y / c) >= exp(-2L / c) there; if l_u - l_w > DMAX then g(l_u) - g(l_w) >= DMAX exp(-2L / c)
>= delta exp(-2L / c) = 4E > 2E (the device checks delta <= DMAX), a contradiction; so l_u - l_w <= DMAX and then
(l_u - l_w) exp(-2L / c) <= 2E, i.e. l_u - l_w <= delta / 2. Either way hi_w >= l_w >= T - delta: w's tile is
computed. (3) Every other id holds -inf, f(-inf) = -c, and y_w >= f(l_u) >= g(T) - E >= -c + c exp(-2L / c) - E > -c
(the device checks c exp(-2L / c) > 8E), so no uncomputed id reaches the maximum. (4) Computed ids hold the stock
values, so the maximum and the set of ids attaining it among computed ids are the stock ones restricted to computed
ids, and that set contains w, the lowest stock maximiser. Hence the stock argmax returns w on the certified buffer.
E = 2^-14 covers Triton's div.full (<= 2 ulp), libdevice tanhf (<= 2 ulp) and the final multiply for every |l|
(about 2.5e-5 at |f| ~ 30; tanhf saturates exactly to +-1 for large |l|).

Accumulation model (an ASSUMPTION about the tensor cores, validated on the GPU): products of BF16 values are exact in
FP32, and each chained m16n8k16 MMA step adds at most 17 rounding errors of one FP32 ulp of the running magnitude
(Fasi, Higham, Mikaitis, Pranesh 2021); with K / 16 chained steps and the cross-warp FP32 sums this gives
C_ACC = (K / 16 * 17 + 32) * 2^-23 per unit of sum |w_k x_k| <= ||w|| ||x||. inkling_pico12's GPU check already
gates pico's head against FP64 with the tighter (K / 16 + 32) * 2^-23; tools/gpu_check_cert.py measures both the
screen and the exact pass against FP64 and against this bound.
"""

import math

import numpy as np

K = 2304
GROUP = 192
NGROUPS = K // GROUP
TILE = 16
QMAX = 31
C_ACC = (K / 16 * 17 + 32) * 2.0 ** -23
CN = 2.0 * C_ACC                  # exact path + screen
CAP = 30.0                        # final_logit_softcapping
E_SOFTCAP = 2.0 ** -14            # |stock f(l) - c tanh(l / c)| (div.full + libdevice tanhf + one multiply: ~2.5e-5)
TWO_E = 2.0 * E_SOFTCAP
DMAX = 16.0
INFL = 1.0 + 2.0 ** -16
X_ABS = 2.0 ** -58                # covers squares of tiny activations that underflow in FP32


def f32(a):
    return np.asarray(a, dtype=np.float32)


def bf16_round(a):
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) >> 16 << 16).astype(np.uint32).view(np.float32)


def round_up_f16(x):
    h = np.asarray(x, dtype=np.float64).astype(np.float16)
    bump = h.astype(np.float64) < x
    return np.where(bump, np.nextafter(h, np.float16(np.inf)), h).astype(np.float16)


def round_up_f32(x):
    f = np.asarray(x, dtype=np.float64).astype(np.float32)
    bump = f.astype(np.float64) < x
    return np.where(bump, np.nextafter(f, np.float32(np.inf)), f).astype(np.float32)


def quantize(W):
    """W [V, K] -> (q int8, sc fp16 [V, NGROUPS], D float64)."""
    V = W.shape[0]
    w = W.astype(np.float64).reshape(V, NGROUPS, GROUP)
    amax = np.abs(w).max(-1)
    sc = round_up_f16(amax / QMAX)
    s = sc.astype(np.float64)
    q = np.where(s[..., None] > 0, np.clip(np.rint(w / np.where(s > 0, s, 1)[..., None]), -QMAX, QMAX), 0)
    return q.reshape(V, K).astype(np.int8), sc, (q * s[..., None]).reshape(V, K)


def tables(W, D):
    V = W.shape[0]
    R = (W.astype(np.float64) - D).reshape(V, NGROUPS, GROUP)
    rt = round_up_f16(np.sqrt((R ** 2).sum(-1)) * (1 + 2.0 ** -40))
    nv = round_up_f32(np.maximum(np.linalg.norm(W.astype(np.float64), axis=1), np.linalg.norm(D, axis=1)) * (1 + 2.0 ** -40))
    return rt, nv


def screen(x, D, rt, nv, v_real, s_err=None):
    """-> (lo, hi) float32 [M, V], mirrors the kernel epilogue. s_err: an injected screen error (tests)."""
    M = x.shape[0]
    x64 = x.astype(np.float64)
    s = x64 @ D.T
    if s_err is not None:
        s = s + s_err
    s = f32(s)
    xg2 = f32((x64.reshape(M, NGROUPS, GROUP) ** 2).sum(-1))
    xg2 = f32(np.where(xg2 < 2.0 ** -126, 0.0, xg2))   # worst case: everything below the normal range is lost
    xg = f32(np.sqrt(xg2) * np.float32(INFL) + np.float32(X_ABS))
    xn = f32(np.sqrt(xg2.sum(-1, dtype=np.float32)) * np.float32(INFL) + np.float32(4 * X_ABS))
    b = f32(xg.astype(np.float64) @ rt.astype(np.float64).T)
    b = f32((b * np.float32(INFL) + np.float32(CN) * nv[None, :] * xn[:, None]) * np.float32(INFL))
    slop = f32((np.abs(s) + b) * np.float32(2.0 ** -20) + np.float32(1e-30))
    lo, hi = f32(s - b - slop), f32(s + b + slop)
    lo[:, v_real:] = -np.inf
    hi[:, v_real:] = -np.inf
    return lo, hi


def select(lo, hi, on=True, bad=False):
    """-> (tile mask bool [V/16], certified bool). Mirrors k_rowmax + k_select."""
    M, V = lo.shape
    T, H = lo.max(1), hi.max(1)
    ht = hi.reshape(M, V // TILE, TILE).max(-1)
    full = (not on) or bad
    thr = np.empty(M, np.float32)
    for m in range(M):
        L = max(abs(float(T[m])) + DMAX, abs(float(H[m])))
        ex = math.exp(-2.0 * L / CAP)
        delta = 2.0 * TWO_E / max(ex, 1e-30)
        ok = math.isfinite(T[m]) and math.isfinite(H[m]) and L < 1e4 and delta <= DMAX and CAP * ex > 4 * TWO_E
        full |= not ok
        thr[m] = T[m] - (delta if ok else 0.0)
    if full:
        return np.ones(V // TILE, bool), False
    return (ht >= thr[:, None]).any(0), True


def softcap(l):
    """The stock softcap in FP32 (division, tanh, multiply each rounded once)."""
    l = f32(l)
    with np.errstate(invalid="ignore", over="ignore"):
        return f32(f32(np.tanh(f32(l / np.float32(CAP)))) * np.float32(CAP))


def stock_token(l, v_real, f=softcap):
    y = f(l)
    y[..., v_real:] = -np.inf
    return y.argmax(-1)    # numpy argmax: first maximum = lowest id (no NaN here)


def certified_buffer(l, mask):
    buf = np.full_like(l, -np.inf, dtype=np.float32)
    cols = np.repeat(mask, TILE)
    buf[:, cols] = l[:, cols]
    return buf


def certified_token(l, mask, v_real, f=softcap):
    return stock_token(certified_buffer(l, mask), v_real, f)
