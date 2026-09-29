#!/usr/bin/env python3
"""pretune_exl3_shapes touches every distinct EXL3 shape once per row bucket (no GPU needed)."""
from __future__ import annotations

import torch
from torch import nn

from vllm.model_executor.layers.quantization.exl3 import (
    exl3_shape_key,
    iter_exl3_handles,
    pretune_exl3_shapes,
)


class FakeHandle:
    def __init__(self, in_features, out_features, K, mul1=True):
        self.in_features, self.out_features, self.K = in_features, out_features, K
        self.mcg, self.mul1 = not mul1, mul1
        self.trellis = torch.zeros(1, dtype=torch.int16)  # device = cpu
        self.calls: list[tuple] = []

    def forward(self, x, params, out_dtype=None):
        assert x.dtype == torch.float16 and x.shape[1] == self.in_features
        self.calls.append((int(x.shape[0]), out_dtype))
        return torch.zeros(x.shape[0], self.out_features)


def test_dedup_and_buckets() -> None:
    a = nn.Linear(1, 1); a._exl3_inner = FakeHandle(5120, 1280, 5)
    b = nn.Linear(1, 1); b._exl3_inner = FakeHandle(5120, 1280, 5)      # same shape -> tuned once
    c = nn.Linear(1, 1); c._exl3_inners = [FakeHandle(4096, 1024, 5), FakeHandle(4096, 1024, 5)]
    d = nn.Linear(1, 1); d._exl3_inners = [{"gate": FakeHandle(5120, 1152, 3), "up": FakeHandle(5120, 1152, 3), "down": FakeHandle(1152, 5120, 3)}]
    e = nn.Linear(1, 1); e._exl3_inner = FakeHandle(5120, 1280, 6)      # different K -> its own shape
    model = nn.Sequential(a, b, c, d, e)
    assert len(list(iter_exl3_handles(model))) == 8
    shapes, launches = pretune_exl3_shapes([model, None], sizes=(1, 2, 4, 8, 16))
    assert shapes == 5, shapes                       # (5120,1280,5) (4096,1024,5) gate/up (5120,1152,3) down (1152,5120,3) (5120,1280,6)
    assert launches == 25, launches
    assert [m for m, _ in a._exl3_inner.calls] == [1, 2, 4, 8, 16]
    assert b._exl3_inner.calls == []                 # deduped onto a
    assert all(dt == torch.float32 for _, dt in a._exl3_inner.calls)
    assert exl3_shape_key(a._exl3_inner) == (5120, 1280, 5, False, True)
    print("test_exl3_pretune: ok")


if __name__ == "__main__":
    test_dedup_and_buckets()
