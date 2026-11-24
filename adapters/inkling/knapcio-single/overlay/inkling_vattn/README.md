# inkling_vattn: DSpark target-verify attention through the compact split-KV kernel at every batch size

Switch: `INKLING_VATTN=1` (levers `cap,bias`) or an explicit list `INKLING_VATTN=cap[,bias]`; unset = nothing is
imported. Requires the stock switch `SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1` (refuses to start without it).
`INKLING_VATTN_MAX_TOKENS` (default 384 = 48 requests x 8 verify rows) is the new row cap.

## Why verify left the split-KV kernel above c4

`Inkling2Attention._forward` (models/inkling_common/inkling2_attn.py) takes the compact verify path only when
`tokens <= VERIFY_REL_BIAS_MAX_TOKENS = 32`, i.e. at most 4 requests x 8 DSpark rows. Above that it uses the
padded FA4 path: the prologue writes q and the relative bias padded from 3 to 4 heads per KV group
(`[t, 32, 1024]` BF16), FA4 SM120 runs a shear kernel (`flash_attn cute shearing`, copies the bias into
`[t, 32, 1280]`), the mainloop reads the sheared bias (2 bytes per score) with 8 rows x 4 heads = 32 useful rows
in a 128-row packed-GQA tile, then `gate_sink`. (`VERIFY_ROUTED_EPILOGUE_MAX_TOKENS = 32` in inkling2.py is the
MoE routed epilogue, unrelated to attention.) The compact path is the Triton `_decode_partials` kernel: one CTA per
(request, KV head, split) holds all 8 rows x 3 q heads in one 32-row tile, so each request's K/V is read once per
visit; `combine_gate_sink` fuses the split reduction with the gate / sink epilogue. Nothing in it depends on the
batch size (grid `(batch * 8, splits)`, partials allocated per call, captured in the verify graphs at bs 1-4
already).

## Levers

| lever | what changes |
|---|---|
| `cap` | module global `VERIFY_REL_BIAS_MAX_TOKENS` 32 -> 384: every verify batch up to 48 requests takes the stock compact path (stock prologue, `_decode_partials` with its own launch heuristics, `combine_gate_sink`). A retuned launch (splits / tile / stages, `ops.stock_bias_partials`) measured within noise at the whole-visit level and is not used. |
| `bias` | compact verify only: the prologue is run with `rel_proj = [I_16 \| 0]` (16 x 128), so it stores the rel-clip output r itself in `rel[..., :16]` (a 6 KiB instead of 48 KiB row per token), and `ops.inkernel_bias_partials` computes the bias per key tile in-kernel (`BF16(dot(r, proj[:, dbase : dbase + WB]))`, one m16n8k16 K-step like the prologue's own dot, then `tl.gather` of each row's diagonal); the `[t, 24, 1024]` tensor is neither written nor read. Only above 32 rows (`INKLING_VATTN_BIAS_MIN_TOKENS`, default 33); needs `cap` to matter. |

## Results (GB10, Spark_06, 2026-11-21)

Whole verify visit (prologue + attention + gate/sink epilogue), cold (256 MiB read before each rep), CUDA-graph
replays, median us; GB/s = K/V bytes attended / time (4 KiB per key). Contexts jittered per request (-0..210).
`stock` = shipped path above 32 rows (padded prologue, FA4 shear + mainloop, gate_sink); `cap` = the compact stock
path (what `cap` alone gives); `cap+bias` = both levers (`bias` applies above 32 rows only).

| requests | ctx | layer | stock us (GB/s) | cap us (GB/s) | cap+bias us (GB/s) | prologue stock / compact / bias us |
|---:|---:|---|---:|---:|---:|---:|
| 8 | 700 | SWA | 199 (102) | 124 (164) | 118 (173) | 16.4 / 14.5 / 14.4 |
| 8 | 700 | global | 208 (98) | 124 (164) | 115 (177) | |
| 8 | 1500 | SWA | 289 (117) | 194 (175) | 194 (175) | |
| 8 | 1500 | global | 328 (142) | 255 (182) | 238 (196) | |
| 16 | 700 | SWA | 375 (108) | 263 (154) | 237 (172) | 32.5 / 24.6 / 18.6 |
| 16 | 1500 | global | 665 (140) | 505 (184) | 471 (197) | |
| 24 | 700 | SWA | 561 (108) | 386 (158) | 346 (176) | 60.4 / 47.2 / 26.8 |
| 24 | 1500 | SWA | 847 (120) | 567 (179) | 536 (189) | |
| 24 | 1500 | global | 899 (155) | 749 (186) | 681 (205) | |
| 48 | 700 | SWA | 1187 (102) | 781 (155) | 675 (179) | 157.7 / 125.0 / 56.4 |
| 48 | 1500 | SWA | 1781 (114) | 1154 (176) | 1045 (194) | |
| 48 | 1500 | global | 1859 (150) | 1472 (189) | 1360 (205) | |
| 8 | 32K | global | 4798 (223) | 4608 (232) | 4581 (234) | |
| 1 | 128K | global | 2374 (226)* | 2310 (232) | 2310 (232) | |
| 4 | 128K | global | 9321 (230)* | 9144 (235) | 9144 (235) | |

\* at 1-4 requests the shipped engine already uses the compact path (the `cap` column); the FA4 row is only for
reference. Attention kernels alone (no prologue, `vattn_bench.py`, ctx 1K, cold): FA4 path 130-147 GB/s at 8-48
requests, compact kernel 167-195 GB/s, in-kernel bias 206-214 GB/s. At 32K-128K every path is within 2-6 % of the
~235 GB/s DRAM floor (the compact kernel reads each request's K/V once for all 8 rows x 3 q heads).

Per-step estimate (48 visits; shipped-stack profile R2P-s05: verify attention incl. prologue 7.2 ms/step at c8,
21.4 ms/step at c24, contexts ~0.3-2K): the visit ratios above (0.62-0.69 cap, 0.59-0.68 cap+bias) give
**c8: -2.2..-2.9 ms/step (~2-2.5 %)**, **c24: -6.4..-8.2 ms/step (~4-4.7 %)**; `bias` adds ~0.3 ms (c8) /
~1.5 ms (c24) on top of `cap`. c1-c4: unchanged (already compact; `bias` is off at <= 32 rows, where it measured
equal or up to 2-4 us/visit slower).

Accuracy vs an FP32 reference (no TF32), every row, bs 4/8/24/48 x ctx 1K/4K/32K x SWA/global + bs 1 at
32K/128K, real layer-0 `proj`: max |out| error stock FA4 6.2e-4..3.5e-3, compact 7.7e-4..3.5e-3, in-kernel
6.7e-4..3.5e-3 (BF16 P rounding dominates; same in all three); max |LSE| error stock 1.9e-6..6.1e-5, compact
and in-kernel 1.9e-6..3.8e-6. Gated BF16 output vs reference: identical maxima for all three (3.0e-3..7.8e-3).
`gpu_check_patch.py` (real stock prologue, synthetic projections and conv state): the identity-projection prologue
gives q / k / v / conv windows bit-identical and `BF16(r . proj)` bit-identical to the stock rel bias; in-kernel
vs stock compact attention bit-identical at 1-2 requests, <= 1 BF16 ulp (2.4e-4 at |out| 0.2) above.

## Math and exactness

Same math as stock: s = q.k * scale + bias in FP32, bias = BF16(FP32 dot of r and proj) for distances 0..1023
(0 beyond on global layers), causal + window 1023 on SWA layers, FP32 online softmax (exp2), P rounded to BF16,
FP32 partials, `combine_gate_sink` epilogue. `cap` replaces FA4's arithmetic with the compact kernel's (FP32
accumulation order and tile boundaries only). `bias` produces the bias with the same K = 16 MMA as the prologue
(bit-identical bias in the GPU check); the rest of its kernel is the stock compact kernel (outputs within 1 BF16
ulp of it, FP32 rounding of the fused score update).

## Patch points (patch.py)

- `sglang.srt.models.inkling_common.inkling2_attn`: `VERIFY_REL_BIAS_MAX_TOKENS` (read by `_forward` at call
  time) and, with `bias`, the module name `attention_prologue` (bound by `from ... import`): a compact verify call
  (verify windows given, `padded_group == heads // kv_heads`, no shear, SM12x, BF16 proj) runs the stock prologue
  with the identity projection and records `(rel, proj)`.
- `sglang.kernels.ops.attention.inkling2_attn.rel_bias_partials` (imported by the FA backend at call time): a
  `rel_bias` with a live record (data pointer, shape; several may be pending, e.g. interleaved layers) runs the
  in-kernel kernel with the recorded proj; a 128-wide rel bias without a record raises; everything else goes to
  the original. Only installed with `bias`.
- Install-time `check_sources` looks for the exact code the patches rely on (cap constant and its use, the compact
  branch and its prologue call, the backend's compact dispatch, `_decode_partials`' indexing / bias load, the
  prologue's compact rel store); signatures of `attention_prologue` and `rel_bias_partials` are checked at patch
  time. A mismatch raises (exit 78 through sitecustomize).

## CUDA graphs

Both wrappers are Python-only at call time (no host sync); the kernels' grids and the launch config are pure
functions of static shapes, partials are allocated per call (graph pool), so the verify graphs (bs 1..48) capture
them like the stock calls. The identity projection is created on the first eager verify visit (SGLang's full-graph
backend runs eager warm-up steps before capturing), followed by one device synchronize so no stream can read it
unfilled; a visit captured before any eager one keeps the stock bias (logged once; the stock compact path). Graph padding requests (seqlen 1 < 8 rows) are handled as in the stock kernel.

## Arming (sitecustomize.py)

```python
    if _on("INKLING_VATTN"):
        import inkling_vattn
        inkling_vattn.install()
```

anywhere in the try-block (it patches different targets than the other packages), and forward `INKLING_VATTN`
(and optionally `INKLING_VATTN_MAX_TOKENS`) into the container in `scripts/launch.sh`. Start-up log:
`[inkling_vattn pid=...] enabled cap,bias max_tokens=384 bias_min_tokens=33`. `inkling_vattn.patch.STATS` counts the calls taken
(at capture time for graphed shapes).

## Coexistence

`inkling_vcap` (dead rows' experts remapped) does not touch attention; `inkling_attn` / `inkling_prefill` patch
prefill only (`RelLogitsProj`, FA4 varlen); `inkling_pico12` / `inkling_kernels` wrap projections. No shared
patch targets.

## Tools

In `results/kbench/round2/vattn/`: `vattn_bench.py` (correctness vs FP32 reference + cold
microbench + launch sweep), `gpu_check_patch.py` (the patches on the real modules: real prologue with identity
projection bit-equal, in-kernel vs stock attention, CUDA-graph replay with padding requests, capture-first
fallback), `compile_check.py` (sm_121 registers / spills / shared memory).

## Risks / open items

- Compact verify (stock, and this overlay above 32 rows) assumes a fixed number of verify rows per request
  (DSpark: 8) and ignores `cu_seqlens_q`; a ragged verify batch would assert (non-divisible) or group rows wrongly.
  The stock engine has the same assumption at <= 32 rows.
- Validated with direct calls and CUDA-graph capture on the real sglang modules, TP1, BF16 KV, page 128. Not yet
  validated in a full server (two-batch overlap, TP > 1, MXFP8 KV are excluded by the stock compact condition or
  untested). Serving A/B: docs/experiments.md.
- Long-context c1-c4 verify (global layers) is already at the DRAM floor (224-235 GB/s at 32K-128K); the only
  remaining lever there is fewer bytes (FP8 KV, lossy): 128K c1 global visits read 8 x 537 MB = 4.3 GB/step
  (~18.5 ms at 232 GB/s); FP8 K/V would save ~9.3 ms/step (+0.4 ms on SWA layers), i.e. ~58 -> ~49 ms per step,
  roughly +18-20 % tok/s at 128K (+4-5 % at 32K). Not built.

## Checks

- Kernel: bias parity is now checked against the real prologue (bit-identical);
  old global tiles (`dbase >= 1024`) skip the bias dot and gather; identity matrix never created inside a capture;
  registers / spills / smem checked on sm_121 (`compile_check.py`: in-kernel 232-249 regs, 0 spills, 19-30 KB smem).
- Integration: the single pending slot became a per-tensor record table (A, B, A order tested); the
  identity matrix is synchronized once after creation; `cap` no longer wraps the launcher (pure constant change);
  the 128-wide marker check only exists with `bias`; exactness wording qualified; fixed-width verify assumption
  listed under risks.
