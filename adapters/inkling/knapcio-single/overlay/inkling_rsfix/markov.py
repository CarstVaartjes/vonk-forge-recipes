"""SGLang PR #51681 ("Prevent Philox rounding from selecting a masked token") for the
DSpark Markov draft sampler ``sglang.kernels.ops.speculative.dspark.markov_sample``
(pinned image text, see _sources).

Stock bug: a 32-bit Philox word w becomes ``u = (w + 0.5) * 2**-32`` in FP32. Words from
0xFFFFFF80 up round to 2**32, so u == 1 exactly; the exponential noise
``-log1p(-u)`` is +inf and that token's key is -inf. The token can then never be drawn
(a ~3e-8 hole per token and draw in the draft law, which the verify's q does not know
about), and when it is the only finite-logit token it ties the masked (-inf) tokens and
the lowest index, a masked token, wins. The upper half of the words also only resolves
1 - u to 2**-24.

Fix (the PR's code): fold the word before the FP32 conversion,
``folded = (min(w, ~w) + 0.5) * 2**-32`` in (0, 0.5]; the lower half uses
``-log1p(-folded)``, the upper half ``-log(folded)`` (folded is then 1 - u). Both are
finite and positive. Philox keys, counters and word order are unchanged; a seeded draw
can change only where the corrected upper-half noise changes the winner.

Rebound by inkling_rsfix: _markov_sample_partial_kernel (looked up by _launch_partial at
call time; this copy calls the fixed _select_block below), _select_block and
_exponential_from_philox as module attributes, and MarkovSampleStep.torch (the CPU
reference, only reached with CPU tensors) as a classmethod.
"""

from typing import Callable, Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Bound from the stock module by inkling_rsfix.apply() before any kernel here is used
# (no import of the stock module from this file: no import cycle with the post-import hook).
_STOCK_DEPS = ("_NO_INDEX", "_UNIFORM_SCALE", "_philox", "_round_to_step_dtype")


@triton.jit
def _exponential_from_philox(bits):
    # For u = (bits + 0.5) * 2**-32, folding the word before the FP32
    # conversion retains 1-u even when u would round to 1.
    lower = bits < 0x80000000
    folded = (tl.minimum(bits, ~bits).to(tl.float32) + 0.5) * _UNIFORM_SCALE
    return tl.where(lower, -libdevice.log1p(-folded), -tl.log(folded))


@triton.jit
def _markov_sample_partial_kernel(
    B,
    X,
    T,
    OFFSET,
    L,
    P,
    BS,
    XS,
    LS,
    WIDTH,
    PADDED,
    VOCAB_START,
    PARTS,
    KEY,
    STEP,
    softcap,
    vocab_limit,
    BLOCK: tl.constexpr,
    ROUND: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    STORE: tl.constexpr,
    SELECT: tl.constexpr,
    SAMPLE: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
    SOFTCAP: tl.constexpr = False,
    LIMIT: tl.constexpr = False,
):
    row, part = tl.program_id(0), tl.program_id(1)
    i = part * BLOCK + tl.arange(0, BLOCK)
    valid = i < WIDTH
    in_shard = i < PADDED
    value = tl.load(X + row * XS + i, in_shard, 0).to(tl.float32)
    if HAS_BIAS:
        value += tl.load(B + row * BS + i, valid, 0).to(tl.float32)
    value = _round_to_step_dtype(value, ROUND)
    if SOFTCAP:
        # softcap_inplace_logits on the rounded step logits.
        value = _round_to_step_dtype(libdevice.tanh(value / softcap) * softcap, ROUND)
    if LIMIT:
        value = tl.where(i + VOCAB_START < vocab_limit, value, -float("inf"))
    if STORE:
        tl.store(L + row * LS + i, value.to(L.dtype.element_ty), in_shard)
    if SELECT:
        _select_block(
            value,
            T,
            OFFSET,
            P,
            row,
            part,
            i,
            valid,
            PARTS,
            KEY,
            STEP,
            VOCAB_START,
            BLOCK,
            SAMPLE,
            PHILOX_ROUNDS,
        )


@triton.jit
def _select_block(
    value,
    T,
    OFFSET,
    P,
    row,
    part,
    i,
    valid,
    PARTS,
    KEY,
    STEP,
    VOCAB_START,
    BLOCK: tl.constexpr,
    SAMPLE: tl.constexpr,
    PHILOX_ROUNDS: tl.constexpr,
):
    key = value
    if SAMPLE:
        temperature = tl.load(T + row)
    else:
        temperature = 0.0
    if temperature != 0.0:
        # One Philox call yields four words; word k goes to token 4 * q + k.
        quad = (part * BLOCK + VOCAB_START) // 4 + tl.arange(0, BLOCK // 4)
        lane = tl.zeros([BLOCK // 4], tl.int64)
        w0, w1, w2, w3 = tl.philox(
            KEY,
            quad.to(tl.uint32),
            (lane + row).to(tl.uint32),
            (lane + STEP).to(tl.uint32),
            ((lane + tl.load(OFFSET)) & 0xFFFFFFFF).to(tl.uint32),
            PHILOX_ROUNDS,
        )
        bits = tl.interleave(tl.interleave(w0, w2), tl.interleave(w1, w3))
        exp_noise = _exponential_from_philox(bits)
        key = value / temperature - tl.log(exp_noise)
    nan = valid & (key != key)
    has_nan = tl.sum(nan.to(tl.int32), 0) > 0
    maximum = tl.max(tl.where(valid & ~nan, key, -float("inf")), 0)
    wins = tl.where(has_nan, nan, valid & (key == maximum))
    idx = tl.min(tl.where(wins, i + VOCAB_START, _NO_INDEX), 0)
    maximum = tl.where(has_nan, float("nan"), maximum)
    tl.store(P + (row * PARTS + part) * 2, maximum)
    tl.store(P + (row * PARTS + part) * 2 + 1, idx.to(tl.float32, bitcast=True))


# MarkovSampleStep.torch with the PR's change; installed as a classmethod.
def markov_torch(
    cls,
    *,
    base_logits: torch.Tensor,
    bias: torch.Tensor,
    temperatures: torch.Tensor,
    key: int,
    offset: torch.Tensor,
    step: int,
    step_dtype: torch.dtype,
    vocab_start: int = 0,
    gather: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    gather_logits: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    logits_out: Optional[torch.Tensor] = None,
    has_sampled_rows: Optional[bool] = None,
    softcap: Optional[float] = None,
    vocab_limit: Optional[int] = None,
) -> torch.Tensor:
    # None draws the noise for every row: reading the temperatures on the
    # host would break a graph capture.
    cls._check(
        base_logits,
        bias,
        temperatures,
        offset,
        step_dtype,
        vocab_start,
        gather,
        gather_logits,
        logits_out,
    )
    rows, width = bias.shape
    value = base_logits.to(torch.float32, copy=True)
    value[:, :width] += bias.float()
    value = value.to(step_dtype).float()
    if softcap is not None:
        value = (torch.tanh(value / softcap) * softcap).to(step_dtype).float()
    if vocab_limit is not None:
        columns = vocab_start + torch.arange(value.shape[1], device=value.device)
        value[:, columns >= vocab_limit] = float("-inf")
    if gather_logits is not None and logits_out is not None:
        full = gather_logits(value.to(step_dtype))[:, : logits_out.shape[1]]
        return cls.torch(
            base_logits=full,
            bias=torch.zeros_like(full),
            temperatures=temperatures,
            key=key,
            offset=offset,
            step=step,
            step_dtype=step_dtype,
            logits_out=logits_out,
            has_sampled_rows=has_sampled_rows,
        )
    padded = base_logits.shape[1]
    device = base_logits.device
    if logits_out is not None:
        logits_out.copy_(value)

    keys = value
    if has_sampled_rows is not False:
        quads = vocab_start // 4 + torch.arange(
            triton.cdiv(padded, 4), device=device, dtype=torch.int64
        )
        words = _philox(
            key,
            quads[None, :],
            torch.arange(rows, device=device, dtype=torch.int64)[:, None],
            torch.full((1, 1), step, device=device, dtype=torch.int64),
            offset.reshape(1, 1) & 0xFFFFFFFF,
        )
        words = torch.broadcast_tensors(*words)
        bits = torch.stack(words, dim=-1).reshape(rows, -1)[:, :padded]
        # _philox returns uint32 words stored in int64 tensors.
        folded = (
            torch.minimum(bits, 0xFFFFFFFF - bits).float() + 0.5
        ) * _UNIFORM_SCALE.value
        exp_noise = torch.where(
            bits < 0x80000000, -torch.log1p(-folded), -torch.log(folded)
        )
        sampled = temperatures != 0
        safe_temperatures = torch.where(sampled, temperatures, 1.0)[:, None]
        noised = value / safe_temperatures - torch.log(exp_noise)
        keys = torch.where(sampled[:, None], noised, value)
    cols = torch.arange(padded, device=device)
    keys = torch.where(cols[None, :] < width, keys, -torch.inf)

    if width == 0:
        local_values = torch.full((rows,), -torch.inf, device=device)
        local_indices = torch.full(
            (rows,), _NO_INDEX.value, dtype=torch.int32, device=device
        )
    else:
        local = keys.argmax(dim=1)
        local_values = keys.gather(1, local[:, None]).squeeze(1)
        local_indices = (local + vocab_start).to(torch.int32)
    partial = torch.stack(
        [local_values.float(), local_indices.view(torch.float32)], dim=1
    )
    gathered = partial if gather is None else gather(partial)
    gathered = gathered.view(-1, rows, 2)
    values = gathered[..., 0].transpose(0, 1)
    indices = gathered[..., 1].contiguous().view(torch.int32) & _NO_INDEX.value
    indices = indices.transpose(0, 1).long()
    # Lowest index among the NaNs if any, else among the maxima.
    valid = indices != _NO_INDEX.value
    nan = valid & values.isnan()
    has_nan = nan.any(dim=-1, keepdim=True)
    finite = torch.where(valid & ~nan, values, -torch.inf)
    maximum = finite.amax(dim=-1, keepdim=True)
    wins = torch.where(has_nan, nan, valid & (finite == maximum))
    return torch.where(wins, indices, _NO_INDEX.value).amin(dim=-1)
