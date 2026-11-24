"""CPU model of the wide certificate's intervals and candidate selection."""

import numpy as np

from . import cert_math as cm
from .certwide import ABS_SLOP, CN, MAX_WEIGHT_NORM


def weight_radius(rt, nv):
    """rt is [vocab, 12] and already rounded up, as is nv [vocab]."""
    if not np.all(nv <= MAX_WEIGHT_NORM):
        raise ValueError("weight norms outside the supported range")
    radius = np.sqrt(np.square(rt.astype(np.float64)).sum(-1)) + CN * nv.astype(np.float64)
    return cm.round_up_f32(radius * (1 + 2.0**-40))


def activation_norm(x):
    x64 = x.astype(np.float64)
    norm = np.linalg.norm(x64, axis=-1).astype(np.float32)
    return norm * np.float32(cm.INFL) + np.float32(4 * cm.X_ABS)


def intervals(screen, norm, radius):
    """Reference intervals for FP32 screen logits. Absolute slop covers FTZ."""
    b = cm.round_up_f32(norm.astype(np.float64)[:, None] * radius.astype(np.float64)[None, :])
    s = screen.astype(np.float32, copy=False)
    e = (np.abs(s) + b) * np.float32(2.0**-20) + np.float32(ABS_SLOP)
    return s - b - e, s + b + e


def derivative_lower(L):
    """CPU model of the device's downward-rounded repeated squaring."""
    z = np.nextafter(2 * L.astype(np.float64) / cm.CAP, np.inf)
    base = np.nextafter(1 - np.minimum(z, 32) * 2.0**-20, -np.inf)
    for _ in range(20):
        base = np.nextafter(base * base, -np.inf)
    return np.where(z <= 32, -cm.round_up_f32(-base), np.float32(0))


def thresholds(T, H):
    """Return an inclusive cutoff and a safe flag per row."""
    with np.errstate(invalid="ignore", divide="ignore", under="ignore", over="ignore"):
        L = np.maximum(cm.round_up_f32(np.abs(T.astype(np.float64)) + cm.DMAX), np.abs(H))
        slope = derivative_lower(L)
        delta = cm.round_up_f32(2.0 * cm.TWO_E / slope.astype(np.float64))
        rounded_down = -cm.round_up_f32(delta.astype(np.float64) - T.astype(np.float64))
    ok = np.isfinite(T) & np.isfinite(H) & (delta <= cm.DMAX)
    return rounded_down, ok


def select(lo, hi, v_real, capacity, norm, on=True):
    """Return (row mask, union mask, wide activation-group mask, sparse route)."""
    m, v = lo.shape
    ht = hi.copy()
    ht[:, v_real:] = -np.inf
    ht = ht.reshape(m, v // cm.TILE, cm.TILE).max(-1)
    thr, ok = thresholds(lo[:, :v_real].max(-1), ht.max(-1))
    rows = (ht >= thr[:, None]) & (np.arange(v // cm.TILE) * cm.TILE < v_real)
    union = rows.any(0)
    groups = np.zeros(v // cm.TILE, np.uint16)
    for start in range(0, m, 16):
        groups |= rows[start:start + 16].any(0).astype(np.uint16) << (start // 16)
    count = int(union.sum())
    sparse = on and np.all(ok & (norm <= 2.0**20)) and 0 < count <= capacity and count < len(union)
    return rows, union, groups, bool(sparse)


def compact_token(logits, v_real, groups, f=cm.softcap):
    """Model the CUDA reduction, including its lowest-ID rule and NaN handling."""
    m, _ = logits.shape
    tile = np.arange(v_real) // cm.TILE
    live = (groups[tile][None, :] & (1 << (np.arange(m) // 16))[:, None]) != 0
    capped = f(logits[:, :v_real])
    return np.where(live & ~np.isnan(capped), capped, -np.inf).argmax(-1)
