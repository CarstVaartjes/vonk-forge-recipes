# inkling_dfaccept: sparse-q fused sampled accept for DFLASH

Import-time overlay for the DFLASH worker (`SPEC=dflash2`).

| switch | value | effect |
|---|---|---|
| `INKLING_DFLASH_FUSED_ACCEPT` | unset / `0` | nothing is patched |
| | `1` | `DFlashWorkerV2._selector_sampling_accept` takes the fused path on eligible calls |
| `INKLING_DFLASH_FUSED_MIN_BS` | default 16 | smallest batch (requests) served fused |

`scripts/launch.sh` sets `DFLASH_FUSED=1` by default with `SPEC=dflash2` and the overlay on.

## What it replaces

The stock accept scatters the selector's q (16 probabilities per draft slot) into a dense `[bs, 7, V]` FP32 buffer and
runs DSpark's `accept_sampling`, which materialises `softmax(z / T)` for every target row (`[bs, 8, V]`). The fused
path (`core.py`, the structure of `inkling_vcap/fused_accept.py`) computes:

1. per target row, the max and log-sum-exp of `z / T` (`inkling_vcap.fused_accept`'s statistics kernel);
2. `p(d_j)` by a gather and `q(d_j)` as the selector's value at the candidate equal to `d_j`;
3. the accept walk on `[bs, 7]` tensors (chain, or block verification with residual masses whose sparse `Q` is
   subtracted in-kernel);
4. the bonus by an exponential race over the stop row with the slot's 16 `(id, q)` pairs subtracted in-kernel, with
   the `inkling_rsfix` rules (Philox words folded before the FP32 conversion, a finite key floor for positive mass,
   the last positive-target lane when no lane has positive residual).

Eligible calls: plain temperature sampling for every request (no top-k / top-p / min-p; a greedy row is top-k 1, so
it sends its batch to stock), no verify cutoff (the verify cap's capped calls take the stock accept), TP1, CUDA, not
under capture, at least `INKLING_DFLASH_FUSED_MIN_BS` requests.

## Exactness

Same law as the stock accept: the same formulas, `p` from a log-sum-exp instead of a materialised softmax (FP32
order), `q` exactly the selector's values, and a different sampler for the bonus that is exact up to the 64-bit
uniform grid and FP32 math.

## Checks (`tools/gpu_check_dfaccept.py`, GB10, 2026-11-24)

- Statistics and masses vs a dense reference (max mass difference 4.8e-7); accept lengths identical for the same
  coins, chain and block verification.
- Bonus law: 2e5 draws per case, chi-square z within +-0.5; no zero-probability token drawn; the fallback lane.
- Joint law of (accept length, bonus), stock `accept_sampling` vs fused, 1e5 draws each: total variation 0.0086
  (chain) and 0.0047 (block verification), within sampling noise.
- Accept time at V = 201,024: bs 8 0.81 -> 0.53 ms, bs 24 2.44 -> 1.04 ms. Serving A/B (verify cap off): sampled c24
  per-request decode +2.3 %; c8 inconclusive, hence the 16-request minimum.
