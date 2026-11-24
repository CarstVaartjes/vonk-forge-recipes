"""Draft temperature: the draft samples at tau x T instead of T (option ``tau:X`` or
INKLING_DRAFT_TAU=X).

Only the proposal changes. The target distribution, its temperature and the accept
rule are untouched.

How it is wired. ``resolve_draft_params`` (dspark_draft) produces the two temperature
tensors the draft uses:

  temperatures       [bs]  the batch's temperatures clamped at 1e-5; the accept path
                           turns the kept step logits into q with it
                           (SoftmaxTemp(step_logits, draft_block.temperatures), or the
                           draft temperatures of block-wise rejection)
  step_temperatures  [bs]  what the Markov step samples at (Gumbel-max on
                           value / temperature); 0 on argmax rows

The patch multiplies both by tau, in place, after the stock function. They are the
same FP32 product, so the q the accept reads is exactly the law the token was drawn
from (Gumbel-max at temperature t draws from softmax(value / t)). The target side
reads sampling_info.temperatures, which is never written here. Speculative sampling
(and block verification) returns the target law p for any proposal q as long as the
accept reads the q the token was drawn from, so the output law is unchanged; only the
acceptance moves.

Greedy rows keep step temperature 0 (0 x tau = 0): their argmax draft is unchanged.
A batch without sampling_info (all greedy) is left exactly as the stock function wrote
it.

Two engine paths read ``temperatures`` as the TARGET temperature as well and are
refused while tau != 1:

  - the sampling-accept CUDA graph (SGLANG_OPT_DSV41_SAMPLING_ACCEPT_GRAPH, default off
    unless SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION): it builds the target
    temperatures from the draft params. Its constructor raises.
  - the vocab-sharded accept (TP > 1): prepare() already refuses TP > 1.
"""

from __future__ import annotations

import functools
import inspect

DRAFT_MODULE = "sglang.srt.speculative.dspark_components.dspark_draft"
SAMPLER_MODULE = "sglang.srt.speculative.dspark_components.dspark_draft_sampler"
SAG_MODULE = "sglang.srt.speculative.dspark_components.dspark_sampling_accept_graph"

_MARK = "__inkling_draft_tau__"
LABEL = "dspark_draft.resolve_draft_params"
LABEL_SAMPLER = "dspark_draft_sampler.resolve_draft_params"
LABEL_SAG = "DsparkSamplingAcceptGraph.__init__ (refuses)"

TAU_MIN = 0.05
TAU_MAX = 4.0


def check_tau(value: float) -> float:
    """Validated tau (raises ValueError)."""
    tau = float(value)
    if not (TAU_MIN <= tau <= TAU_MAX):
        raise ValueError(f"draft tau must be in [{TAU_MIN}, {TAU_MAX}], got {value!r}")
    return tau


def make_resolve(orig, tau: float):
    """The patched resolve_draft_params: the stock result, both temperature tensors
    scaled by tau in place (sampled batches only)."""
    params = list(inspect.signature(orig).parameters)
    if params != ["sampling_info", "bs", "device", "out"]:
        raise TypeError(f"resolve_draft_params signature changed: {params}")
    tau = check_tau(tau)

    @functools.wraps(orig)
    def resolve_draft_params(*, sampling_info, bs, device=None, out=None):
        result = orig(sampling_info=sampling_info, bs=bs, device=device, out=out)
        if sampling_info is not None:
            # Same FP32 product in both: the sampled law and the q of the accept agree.
            result.temperatures.mul_(tau)
            result.step_temperatures.mul_(tau)
        return result

    setattr(resolve_draft_params, _MARK, tau)
    return resolve_draft_params


def is_patched(fn) -> bool:
    return getattr(fn, _MARK, None) is not None


def refusing_sag_init(orig_init, tau: float, error_cls):
    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        raise error_cls(
            f"draft tau {tau} is incompatible with the sampling-accept CUDA graph "
            "(SGLANG_OPT_DSV41_SAMPLING_ACCEPT_GRAPH): it reads the draft temperatures "
            "as the target temperatures. Unset that flag (and "
            "SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION) or drop tau."
        )

    setattr(__init__, _MARK, tau)
    return __init__
