"""Deterministic finite BF16 fixtures for the CPU and GPU checks."""

import torch

from .storage import DIM


def kv_values(rows, heads, device, generator, *, real_like):
    shape = (rows, heads, DIM)
    if not real_like:
        return torch.randn(shape, device=device, generator=generator, dtype=torch.bfloat16)
    # The shipped histogram reports the shared top-15 range and 2.8e-4
    # outside it. Bucket weights within that range are a synthetic profile.
    mid = torch.randint(116, 131, shape, device=device, generator=generator, dtype=torch.int32)
    peak = torch.randint(121, 129, shape, device=device, generator=generator, dtype=torch.int32)
    chance = torch.rand(shape, device=device, generator=generator)
    exponent = torch.where(chance < 0.00028, 112, torch.where(chance < 0.04, mid, peak))
    exponent.masked_fill_(chance < 0.00002, 0)
    fraction = torch.randint(0, 128, shape, device=device, generator=generator, dtype=torch.int32)
    sign = torch.randint(0, 2, shape, device=device, generator=generator, dtype=torch.int32)
    return ((sign << 15) | (exponent << 7) | fraction).to(torch.int16).view(torch.bfloat16)
