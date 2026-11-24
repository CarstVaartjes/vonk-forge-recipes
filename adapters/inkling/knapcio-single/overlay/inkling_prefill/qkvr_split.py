"""C1 qkvr_split: column-split qkvr projection for stacked q/k/v sources at large M.

Stock (sglang/srt/models/inkling_common/inkling2_attn.py,
Inkling2Attention._project_sources): when q, k and v come from three distinct
tensors (the 4 global layers in every visit, and the 10 local core layers in loop
passes 2-3: 28 of 48 layer visits), the sources are stacked into [3M, 2304] and
multiplied by the whole qkvr weight [8600, 2304]. From the q chunk only the q, rel,
gate and sink columns are used, from the k chunk only k, from the v chunk only v:
two thirds of the GEMM is discarded (prefill 8K chunk: 10.6 ms per stacked visit,
3.8 ms for the kept columns).

This lever computes only the kept columns, each with cuBLAS (torch.mm with an
out= column view) on contiguous row slices of the same weight (views, no copy),
into ONE [M, N] buffer:
    q source x W[0:q)            -> out[:, 0:q)            q
    k source x W[q:q+k)          -> out[:, q:q+k)          k
    v source x W[q+k:q+k+v)      -> out[:, q+k:q+k+v)      v
    q source x W[q+k+v:N)        -> out[:, q+k+v:N)        rel | gate | sinks (| pad)
All six returned tensors are column views of that buffer, so they have the stock
row stride N with unit column stride. That is a hard requirement: the fused
decode/verify attention prologue (kernels/ops/attention/inkling2_attn_prologue.py,
attention_prologue) asserts that q, k, v and rel share one row stride, and it is
taken for decode and for target verify (> 32 verify rows: padded + fused prologue,
i.e. from 16 concurrent requests at 8 verify rows, which the verify CUDA graphs
capture at start-up). An earlier variant with four separate outputs failed there.
Each output element is the same K=2304 BF16 dot product with FP32 accumulation as
the stock F.linear; bench_tc.py found the split bit-equal to the stacked GEMM at
M = 1024, 8192 and 16384, and tools/gpu_check_prefill.py checks it on real weights.

Scope: only the three-distinct-sources case with M >= INKLING_PREFILL_QKVR_MIN_M
(default 128 rows per source), BF16 CUDA tensors, no bias, and only when the stock
projection for that shape is plain F.linear (no compile, no batch-invariant mode,
no cutedsl/gemv BF16 backend, no small-M row-major/tiled weight). Everything else,
including decode and verify, calls the previously installed _project_sources:
the stock method, or inkling_kernels.qkvr_gemv's wrapper when that overlay is on
(it owns M <= 16). The two compose in either patch order.

CUDA graphs: no host sync, no data-dependent shapes; outputs come from torch.empty
via F.linear / torch.mm as in the stock path.

Env:
  INKLING_PREFILL_QKVR_MIN_M  rows per source from which the split applies (default and
                              floor 128: at 64 rows cuBLAS picks other kernels for the
                              k/v GEMMs and the result differs, K5 GPU check)
"""

import torch

TARGET_MODULE = "sglang.srt.models.inkling_common.inkling2_attn"
TARGET_CLASS = "Inkling2Attention"
TARGET_ATTR = "_project_sources"
UNQUANT_MODULE = "sglang.srt.layers.quantization.unquant"
MARK = "inkling_prefill_qkvr_split"


def column_plan(split_sizes, n_rows):
    """((q_lo, q_hi), (k_lo, k_hi), (v_lo, v_hi), (rest_lo, rest_hi)) row ranges
    of the qkvr weight, for split_sizes [q, k, v, rel, gate, sinks(, pad)]."""
    q, k, v = split_sizes[0], split_sizes[1], split_sizes[2]
    if sum(split_sizes) != n_rows or len(split_sizes) < 6:
        return None
    return (0, q), (q, q + k), (q + k, q + k + v), (q + k + v, n_rows)


def split_project(xq, xk, xv, weight, plan, rest_sizes):
    """(q, k, v, r, gate, sinks) of the column-split projection, all column views
    of one [M, N] buffer (row stride N, like the stock split views)."""
    (q0, q1), (k0, k1), (v0, v1), (r0, r1) = plan
    m = xq.shape[0]
    out = torch.empty((m, weight.shape[0]), dtype=xq.dtype, device=xq.device)
    torch.mm(xq, weight[q0:q1].t(), out=out[:, q0:q1])
    torch.mm(xk, weight[k0:k1].t(), out=out[:, k0:k1])
    torch.mm(xv, weight[v0:v1].t(), out=out[:, v0:v1])
    torch.mm(xq, weight[r0:r1].t(), out=out[:, r0:r1])
    rest = out[:, r0:r1]
    r, gate, sinks = rest.split(rest_sizes, dim=-1)[:3]
    return out[:, q0:q1], out[:, k0:k1], out[:, v0:v1], r, gate, sinks


def _stock_is_plain_linear(layer, unquant):
    """True when UnquantizedLinearMethod.apply would end in F.linear for M > 16."""
    method = getattr(layer, "quant_method", None)
    base = getattr(unquant, "UnquantizedLinearMethod", None) if unquant is not None else None
    if method is None:
        return False
    if base is not None:
        # isinstance, not the exact type: inkling_pico12 swaps in per-layer subclasses
        # that only take <= 16 rows and fall back to the stock apply above that.
        if not isinstance(method, base):
            return False
    elif not any(c.__name__ == "UnquantizedLinearMethod" for c in type(method).__mro__):
        return False
    if getattr(layer, "small_m_rowmajor", False) or getattr(layer, "small_m_tiled_weight", None) is not None:
        return False
    if unquant is not None:
        try:
            backend = unquant.get_bf16_gemm_backend()
            if backend.is_cutedsl() or backend.is_gemv():
                return False
        except Exception:  # noqa: BLE001
            return False
        inv = getattr(unquant, "is_batch_invariant_mode_enabled", None)
        if inv is not None and inv():
            return False
        if getattr(unquant, "_use_aiter", False):
            return False
    if torch.compiler.is_compiling():
        return False
    return True


def make_project_sources(prev, unquant, cfg):
    min_m = cfg["min_m"]
    allow_cpu = cfg.get("allow_cpu", False)

    def _project_sources(self, sources):
        q_src, k_src, v_src = sources.q, sources.k, sources.v
        if len({id(q_src), id(k_src), id(v_src)}) != 3:
            return prev(self, sources)
        m = q_src.shape[0]
        if m < min_m:
            return prev(self, sources)
        layer = self.qkvr
        weight = layer.weight
        plan = self.__dict__.get("_inkling_prefill_qkvr_plan", False)
        if plan is False:
            plan = column_plan(list(self.split_sizes), weight.shape[0])
            self.__dict__["_inkling_prefill_qkvr_plan"] = plan
        if (
            plan is None
            or q_src.ndim != 2
            or not (q_src.shape == k_src.shape == v_src.shape)
            or not (q_src.dtype == k_src.dtype == v_src.dtype == weight.dtype == torch.bfloat16)
            or weight.dim() != 2
            or weight.shape[1] != q_src.shape[1]
            or getattr(layer, "bias", None) is not None
            or not (allow_cpu or (q_src.is_cuda and weight.is_cuda))
            or not _stock_is_plain_linear(layer, unquant)
        ):
            return prev(self, sources)
        rest_sizes = list(self.split_sizes[3:])
        return split_project(q_src, k_src, v_src, weight, plan, rest_sizes)

    _project_sources.__wrapped__ = prev
    setattr(_project_sources, MARK, cfg)
    # inkling_kernels.qkvr_gemv and inkling_pico12 detect themselves through these
    # attributes; keep them visible through this wrapper so neither re-wraps.
    for attr in ("inkling_cfg", "inkling_pico12"):
        if hasattr(prev, attr):
            setattr(_project_sources, attr, getattr(prev, attr))
    return _project_sources


def patch(attn_module, cfg):
    import importlib

    cls = getattr(attn_module, TARGET_CLASS, None)
    prev = getattr(cls, TARGET_ATTR, None) if cls is not None else None
    if prev is None:
        raise RuntimeError(
            f"inkling_prefill.qkvr_split: {TARGET_MODULE}.{TARGET_CLASS}.{TARGET_ATTR} is missing"
        )
    if getattr(prev, MARK, None) is not None:
        return False
    unquant = importlib.import_module(UNQUANT_MODULE)
    if not callable(getattr(unquant, "get_bf16_gemm_backend", None)):
        raise RuntimeError(
            f"inkling_prefill.qkvr_split: {UNQUANT_MODULE}.get_bf16_gemm_backend is missing"
        )
    setattr(cls, TARGET_ATTR, make_project_sources(prev, unquant, cfg))
    return True
