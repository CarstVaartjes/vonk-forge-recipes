"""The screening copy of the LM head and its bound tables (load time; torch, on the weight's device).

Format (cert_ext.cu reads exactly this):
  q   int in [-31, 31], one FP16 scale per (row, 192-value group): D[v, k] = sc[g, v] * q[v, k] (real numbers)
  Q   uint8 [V/8 tiles][36 steps][384 B]: per (8-row tile, 64-value step), lane l = 4 * row + t (t = 0..3) holds the
      row's values 16t .. 16t + 15 as u = q + 32 (0..63): bytes [8l, 8l + 8) the low nibbles, bytes [256 + 4l, + 4)
      the high two bits. Nibble word h (h = 0, 1; bytes 8l + 4h ..): nibble slot j (bits 4j) = value 2(4h + j), slot
      j + 4 = value 2(4h + j) + 1. High word: slot p (bits 2p) = value 2p, slot p + 8 = value 2p + 1 (p = 0..7).
  sc  fp16 [12][V]  scales, rounded up so that 31 * sc >= the group's max |w|
  rt  fp16 [12][V]  ||W_vg - D_vg||_2, computed in float64, rounded up
  nv  fp32 [V]      max(||W_v||, ||D_v||), float64, rounded up
"""

import torch

K = 2304
GROUP = 192
NGROUPS = K // GROUP
STEP = 64
STEPS = K // STEP
QMAX = 31


def round_up_f32(x64):
    """float64 (>= 0 or inf) -> the smallest fp32 >= x."""
    f = x64.float()
    bump = f.double() < x64
    return torch.where(bump, (f.view(torch.int32) + 1).view(torch.float32), f)


def round_up_f16(x64):
    """float64 (>= 0) -> the smallest fp16 >= x (inf above the fp16 range)."""
    h = x64.half()
    bump = h.double() < x64
    return torch.where(bump, (h.view(torch.int16) + 1).view(torch.float16), h)


def quantize_rows(w):
    """w [r, K] (bf16 / fp32) -> (q int8 [r, K], sc fp16 [r, NGROUPS], D float64 [r, K])."""
    r = w.shape[0]
    w64 = w.double().view(r, NGROUPS, GROUP)
    amax = w64.abs().amax(-1)
    sc = round_up_f16(amax / QMAX)
    s64 = sc.double()
    safe = torch.where(s64 > 0, s64, torch.ones_like(s64))
    q = torch.round(w64 / safe[..., None]).clamp(-QMAX, QMAX)
    q = torch.where(s64[..., None] > 0, q, torch.zeros_like(q))
    d = q * s64[..., None]
    return q.view(r, K).to(torch.int8), sc, d.view(r, K)


def pack6(q):
    """q int8 [V, K] (V % 8 == 0) -> uint8 [V/8 * STEPS * 384] in the layout above."""
    V = q.shape[0]
    u = (q.to(torch.int32) + 32).view(V // 8, 8, STEPS, 4, 16)          # tile, row g, step, t, value j
    u = u.permute(0, 2, 1, 3, 4).reshape(V // 8, STEPS, 32, 16)         # tile, step, lane = 4g + t, j
    lo = u & 0xF
    hi = (u >> 4) & 0x3
    nib = torch.zeros((V // 8, STEPS, 32, 2), dtype=torch.int64, device=q.device)
    for h in range(2):
        for j in range(4):
            p = 4 * h + j
            nib[..., h] |= lo[..., 2 * p].long() << (4 * j)
            nib[..., h] |= lo[..., 2 * p + 1].long() << (16 + 4 * j)
    hw = torch.zeros((V // 8, STEPS, 32), dtype=torch.int64, device=q.device)
    for p in range(8):
        hw |= hi[..., 2 * p].long() << (2 * p)
        hw |= hi[..., 2 * p + 1].long() << (16 + 2 * p)
    nib_b = torch.stack([(nib >> (8 * b)) & 0xFF for b in range(4)], -1).view(V // 8, STEPS, 256)
    hw_b = torch.stack([(hw >> (8 * b)) & 0xFF for b in range(4)], -1).view(V // 8, STEPS, 128)
    return torch.cat([nib_b, hw_b], -1).to(torch.uint8).reshape(-1)


def unpack6(packed, V):
    """Inverse of pack6 by the kernel's own extraction rule (tests): -> q int8 [V, K]."""
    b = packed.view(V // 8, STEPS, 384).long()
    nib = b[..., :256].view(V // 8, STEPS, 32, 2, 4)
    nibw = (nib << (8 * torch.arange(4, device=b.device))).sum(-1)      # [.., 32, 2] u32 words
    hiw = (b[..., 256:].view(V // 8, STEPS, 32, 4) << (8 * torch.arange(4, device=b.device))).sum(-1)
    vals = torch.empty((V // 8, STEPS, 32, 16), dtype=torch.int64, device=b.device)
    for p in range(8):
        w = nibw[..., p // 4]
        lo = (w >> (4 * (p % 4))) & 0x000F000F
        h2 = ((hiw >> (2 * p)) & 0x00030003) << 4
        v = lo | h2
        vals[..., 2 * p] = v & 0xFFFF
        vals[..., 2 * p + 1] = v >> 16
    q = vals - 32
    return q.view(V // 8, STEPS, 8, 4, 16).permute(0, 2, 1, 3, 4).reshape(V, K).to(torch.int8)


@torch.inference_mode()
def build_tables(weight, chunk=4096):
    """weight bf16 [V, K] -> dict(q=packed uint8, sc, rt, nv, eps). Runs on weight.device, chunk rows at a time."""
    if chunk % 8:
        raise ValueError("inkling_cert: chunk must be a multiple of 8")
    return _build(weight, chunk)


def _build(weight, chunk):
    V, k = weight.shape
    if k != K or V % 16:
        raise ValueError(f"inkling_cert: head [{V}, {k}] is not [16n, {K}]")
    dev = weight.device
    packed = torch.empty((V // 8 * STEPS * 384,), dtype=torch.uint8, device=dev)
    sc = torch.empty((NGROUPS, V), dtype=torch.float16, device=dev)
    rt = torch.empty((NGROUPS, V), dtype=torch.float16, device=dev)
    nv = torch.empty((V,), dtype=torch.float32, device=dev)
    err = 0.0
    for r0 in range(0, V, chunk):
        r1 = min(V, r0 + chunk)
        w64 = weight[r0:r1].double()
        q, s, d = quantize_rows(weight[r0:r1])
        packed[r0 // 8 * STEPS * 384 : r1 // 8 * STEPS * 384] = pack6(q)
        sc[:, r0:r1] = s.t()
        res = (w64 - d).view(r1 - r0, NGROUPS, GROUP)
        rg = res.pow(2).sum(-1).sqrt() * (1.0 + 2.0 ** -40)
        rt[:, r0:r1] = round_up_f16(rg).t()
        n = torch.maximum(w64.norm(dim=1), d.norm(dim=1)) * (1.0 + 2.0 ** -40)
        nv[r0:r1] = round_up_f32(n)
        err += float((res.reshape(r1 - r0, K).norm(dim=1) / w64.norm(dim=1).clamp_min(1e-30)).sum())
    if not (torch.isfinite(rt.float()).all() and torch.isfinite(nv).all() and torch.isfinite(sc.float()).all()):
        raise ValueError("inkling_cert: non-finite bound table (head weights out of the fp16 range?)")
    return dict(q=packed, sc=sc.contiguous(), rt=rt.contiguous(), nv=nv, eps=err / V)
