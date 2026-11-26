# inkling_pfuse: prefill glue fusion (attention-residual mixes, loop messages, conv + residual)

Switch: `INKLING_PFUSE=loopmix,resnorm,sconvres` (or `all`); unset = nothing is imported.
`INKLING_PFUSE_MIN_ROWS` (default 512, floor 64): smallest EXTEND / MIXED batch served by the fused path.
Decode, target verify, smaller batches and anything with an unexpected dtype / layout run the stock code.

Status 2026-11-21: written, reviewed, GPU-checked on a GB10 Spark
(ALL OK, see Results). Not yet A/B-tested end to end in the server (the patched `_forward` was only install-checked).

## Where the elementwise remainder goes (stock kernels, one 8192-token chunk)

The R2PP prefill trace holds no prefill GPU work (the profiler window closed at the launch of the prefill batch,
like the older P2 trace; `with_stack` Python events fill it). `tools/attrib_prefill.py` rebuilds the remainder
instead: it walks the 48 layer visits of a chunk exactly like `Inkling2CausalLLM._run_loop` (attn-res blocks,
loop bank, block boundaries), records every elementwise / glue kernel with its shape, and times each distinct call
with the stock kernel on GB10 (cold L2, CUDA events, median). Table: see "Results".

Every stock kernel already runs at 224-238 GB/s: the remainder is not slow kernels, it is bytes. The byte
hogs are the FP32 [T, 2304] intermediates (75 MB each at 8K): every attention-residual mix writes one per query and
the following RMSNorm reads it back; the loop message path of the 24 core visits in passes 2-3 writes and reads
seven of them per attention side (3 messages, 3 controller outputs, plus the dest re-read per controller).

## Levers

| lever | replaces (per visit) | bytes at 8K (stock -> fused) | exactness |
|---|---|---|---|
| `loopmix` | attention side of the 24 loop visits: attn_res(state) -> FP32, attn_res(bank) -> 3x FP32, `_controller_norm`, `_controller_down`, `_controller_mix` -> 3x FP32, RMSNorm -> BF16; mlp side: attn_res(state + addend), attn_res(bank), controllers | attn side ~62 B -> ~23 B, mlp side ~28 B -> ~21 B (B = one BF16 [T, D] = 37.7 MB) | E2 (FP32 row-reduction order) |
| `resnorm` | attn_res + RMSNorm of the other visits (attention input of local and global layers, mlp input) | saves 4 B per query (the FP32 mix write + read) | E2 |
| `sconvres` | mlp short conv (BF16 out) + `residual_add` / boundary add | saves 2 B per visit | bit-identical |

Kernels (`_triton_mix.py`, `_triton_sconv.py`):

- `_resnorm_kernel` (one row per program): attention-residual statistics, mix, RMS statistics and the normalized
  BF16 output in three passes over the row's sources (re-reads hit L1/L2); the FP32 mix never leaves registers.
- `_loop_stats_kernel` (one row per program): state mix (dest) and bank mix (messages) plus the controller RMS
  statistics; writes the controller's BF16 normed input (exactly what stock `_controller_norm` writes) and the
  softmax weights of both mixes (a few floats per row).
- stock `_controller_down` on that input, launched with the stock parameters (so the hidden is computed by the same
  code from the same BF16 input).
- `_controller_down_split` (default): the stock kernel's code reading the normed input as [ND + NM, T, D] (the
  local layers' shared dest is written and read once, not three times); partials bit-identical to the stock
  kernel (GPU check `split`).
- `_loop_mix1_kernel` (default; `_loop_mix_kernel` is the 3-controllers-per-program original): rebuilds dest and messages from the BF16 sources with the stored
  weights (same FP32 op sequence as the stats kernel, so the values are identical), hidden = bf16(sum of the
  split-K partials in group order), out = `(1 - tanh(h Wf)) dest + tanh(h Wg) msg` (stock formula, same `tl.dot`
  K = 64), then the RMSNorm in a second pass (attention side) or stores FP32 out for the stock mlp_norm (mlp side,
  one controller: recomputing would read more than the FP32 round trip).
- `_sconv_residual_kernel`: copy of the stock `_sconv_prefill_kernel` whose store is the residual update.

Exactness argument. Same weights, same per-element formulas (attn_res: `fma(prob_n, x_n, acc)` in source order,
times `1 / sum e`; controllers: stock `_controller_mix` expression; RMSNorm `(x * rstd) * w`), same BF16 rounding
points (controller normed input, controller hidden, final BF16 outputs), all kernels compiled with
`enable_fp_fusion=False` like the stock Triton kernels. What differs is the order of the FP32 row reductions
(attn-res sum of squares and query dots, the RMS statistics), so dest / message / norm values differ at FP32
rounding level and a BF16 output can flip by one ulp. `tools/gpu_check_pfuse.py` measures, per case, the error of
stock and fused against an FP64 reference of the same math with the same BF16 rounding points: the fused error is
required to be no larger than 1.5x stock (max and mean).

## Patch point and coexistence

One patch: `Inkling2DecoderLayer._forward` (sglang/srt/models/inkling2.py), the stock body with the
attention-input and mlp-input parts (and the mlp conv + finalize) replaced when eligible. Everything between them
(`self.attn`, qkvr, rel bias, FA4, gate_sink, wo, attn conv, MoE) is the stock call, so `inkling_prefill`
(qkvr_split, bias_layout, quant_once, partial_bf16), `inkling_kernels`, `inkling_pico12` and `inkling_fuse`
(which wraps `Inkling2DecoderLayer.forward`, not `_forward`) compose unchanged. Outputs keep the stock layouts:
the three attention sources of a loop / global visit are one contiguous [3, T, D] BF16 tensor (so `_stacked` and
`qkvr_split` see adjacent sources as before).

`install()` hashes the source of every function / class the fused path replaces or relies on (32 targets incl.
`_forward`, `_attention_input`, `_run_loop`, `block_attn_res`, `Inkling2AttnResState.*`, `LoopMessageBank.mix`,
`message_controllers` and its kernels, `attn_res` and the CuTe kernel class, `rms_norm`, the sconv path, the
residual add) in the installed sglang sources without importing sglang; any difference raises, and the combined
sitecustomize exits 78.

CUDA graphs: shapes are static per captured prefill graph; the only Python branches are on mode, row count,
dtypes and layouts; no host sync; buffers come from the torch allocator. `gpu_check_pfuse.py --sections graphs`
captures all three kernels and compares replays with eager bit for bit.

## sitecustomize snippet

Add after the `INKLING_PREFILL` block (order is irrelevant: no shared patch target):

```python
    if _on("INKLING_PFUSE"):  # prefill glue fusion: patches Inkling2DecoderLayer._forward (extend only)
        import inkling_pfuse
        inkling_pfuse.install()
```

and in `scripts/launch.sh` export `INKLING_PFUSE=all` (or a subset).

## Tools

- `tools/attrib_prefill.py [--tokens 8192]`: the per-kernel attribution above (stock kernels only).
- `tools/bench_loop.py`, `tools/sweep_mix.py`: per-kernel times of the fused pipelines; mix-kernel launch sweep.
- `tools/gpu_check_pfuse.py [--sections split,resnorm,loop,sconvres,graphs,install] [--tokens 8192,4097,1025,513]`:
  correctness (bit-equality rate, error vs FP64 for stock and fused), microbench stock vs fused at the first
  token count, CUDA-graph replay, install in a subprocess. Exit 1 on any failure.

## Results (GB10, 2026-11-21; logs in results/kbench/round2/pfuse/runs/)

Attribution of the remainder at 8192 tokens (stock kernels, `attrib8k.log`): 614 ms per chunk of the 744 ms
closing term, every kernel at 215-300 GB/s (L2-assisted for the small ones). By kind: loop controllers 233 ms
(48 calls), attn_res 143 (144), RMSNorm 74 (114), gate_sink 42 (48), short convs 42 (192), q/k/rel clips 29
(144), residual_add 26 (40), KV store (approximated) 15, v_norm 5, boundary add 4, DSpark aux copy 1. The other
~130 ms are not in this walk (embedding, tau, sconv cache prepare/update, launch gaps, LM-head side).

Correctness (`check4.log`, 8192 / 4097 / 1025 / 513 rows, real queries, controller and norm weights of layers
7 / 11 / 12 / 18, random BF16 sources; FP64 reference of the same math with the same BF16 rounding points on the
first 1024 rows):

| case | bit-equal to stock (all rows) | err/max vs FP64, stock -> fused | mean err, stock -> fused |
|---|---:|---|---|
| resnorm attn local / global Q=3 / mlp +addend / mlp boundary N=5 | 99.99 % | 2.14e-3 -> 2.14e-3, 3.36e-3 -> 3.36e-3, 2.00e-3 -> 2.00e-3, 2.91e-3 -> 2.91e-3 | identical to 3 digits |
| loopmix attn local / global (BF16 normed out) | 99.53 % / 99.60 % | 2.81e-3 -> 2.82e-3, 2.39e-3 -> 2.39e-3 | 1.44e-3 / 9.59e-4 both |
| loopmix mlp (BF16 after stock mlp_norm) | 99.55 % | 3.40e-3 -> 3.40e-3 | 1.02e-3 both |
| loopmix mlp FP32 mix output | 38.7 % | 1.61e-3 -> 1.37e-3 | 1.05e-5 -> 7.96e-6 |
| split down vs stock down (all shapes) | 100 % | | |
| sconvres (T = 8192, 4097, 3001, 2049, 2048; cached + fresh sequence; with / without partial) | 100 % | | |
| CUDA graph capture + 3 replays + replay after input change (all kernels) | bit-equal to eager | | |
| install (INKLING_PFUSE=all, real sglang import) | patched | | |

The differing BF16 outputs are 1-ulp flips from the FP32 row-sum order (large "max ulp" values in the logs are
values crossing zero); against the FP64 reference the fused error equals the stock error in every case (all 4
token counts).

Microbench at 8192 rows (cold L2, median of 20; stock = the stock kernel sequence it replaces):

| group | stock ms | fused ms | saved ms | calls / 8K chunk | saved ms / chunk |
|---|---:|---:|---:|---:|---:|
| resnorm attn local (N=3) | 1.325 | 0.727 | 0.60 | 19 | 11.4 |
| resnorm attn global (N=3, Q=3) | 2.984 | 1.300 | 1.68 | 4 | 6.7 |
| resnorm mlp (N=4 + addend) | 1.644 | 1.161 | 0.48 | 23 | 11.1 |
| loopmix attn local | 10.946 | 5.699 | 5.25 | 20 | 104.9 |
| loopmix attn global | 11.622 | 6.550 | 5.07 | 4 | 20.3 |
| loopmix mlp | 4.573 | 3.848 | 0.73 | 24 | 17.4 |
| sconvres | 0.998 | 0.672 | 0.33 | 47 | 15.4 |
| **total** | | | | | **187** |

Loop attn-local pipeline now: stats 1.91 ms + split down 0.79 + mix (two-pass, norm inside) 3.0 (`bench_loop.py`);
the stats kernel runs at ~200 GB/s of its bytes, the mix kernel at ~100-120 GB/s of its DRAM bytes (L2 re-reads of
the sources and weights): the remaining headroom is in the mix kernel.

End-to-end estimate (per-chunk saving x chunks against the shipped TTFTs; not yet measured in the server):

| prompt | shipped | with inkling_pfuse | speedup |
|---|---:|---:|---:|
| 8K | 1.92 s (4,267 tok/s) | ~1.73 s (~4,730 tok/s) | ~1.11x |
| 32K | 8.31 s (3,943 tok/s) | ~7.56 s (~4,330 tok/s) | ~1.10x |
| 128K | 46.86 s (2,797 tok/s) | ~43.9 s (~2,990 tok/s) | ~1.07x |
