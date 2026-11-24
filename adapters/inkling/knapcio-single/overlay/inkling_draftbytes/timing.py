"""GPU timing for gpu_check.py / bench_draft.py: CUDA-graph replays (no launch
overhead), with an optional L2 flush before every call whose own time is
measured separately and subtracted (DRAM-cold numbers)."""

from __future__ import annotations

import torch

_flush_buf = None


def _flush():
    global _flush_buf
    if _flush_buf is None:
        _flush_buf = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    _flush_buf.zero_()


def _graph_ms(body, iters: int, reps: int) -> float:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            body()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            body()
    graph.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end) / iters)
    del graph
    return best


def time_ms(fn, *, cold: bool = True, iters: int = 20, reps: int = 5) -> float:
    """Best-of-reps mean time of fn() in ms (fn must be graph-capturable)."""
    if not cold:
        return _graph_ms(fn, iters, reps)

    def body():
        _flush()
        fn()

    return max(_graph_ms(body, iters, reps) - _graph_ms(_flush, iters, reps), 1e-6)


def gbps(nbytes: int, ms: float) -> float:
    return nbytes / (ms * 1e-3) / 1e9
