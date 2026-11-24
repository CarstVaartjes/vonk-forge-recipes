# inkling_kernels: target-side kernel levers for Inkling-2-Mini on one GB10

Two import-time overlays for `lmsysorg/sglang:inkling-2-mini` (sglang
0.0.0.dev1+g03fde9cc3.d20261120, flashinfer 0.7.4, triton 3.8.0), TP1, DSpark
verify of 8 rows per request. Levers #7 and #8 of the source survey (docs/experiments.md).

Nothing here has run on a GPU yet. The overlay's hook, env parsing, startup checks
and the lever A wrapper's view layout were tested on CPU (lever A's math was stubbed
for that test). The Triton kernel, the FlashInfer calls and every timing need the
two GPU scripts below.

```
overlay/
  inkling_kernels/__init__.py   install(), env parsing, startup source checks
  inkling_kernels/_hook.py      post-import hook (patches apply when SGLang imports the target)
  inkling_kernels/qkvr_gemv.py  lever A: Triton column-split GEMV + _project_sources wrapper
  inkling_kernels/moe_tiles.py  lever B: INKLING_MOE_TILES parser + _tiles wrapper
  sitecustomize.py              standalone loader (single-track testing)
  gpu_check_qkvr.py             GPU correctness/timing check for lever A
  bench_moe_tiles.py            GPU tile sweep for lever B (NVFP4, optional MXFP8)
```

## Activation

`INKLING_KERNELS=qkvr_gemv,moe_tiles` (either or both). Unset or empty: `install()` does
nothing and nothing is imported. `install()` checks the installed sglang sources for
the patch targets without importing sglang, registers post-import hooks, and prints
one line per process:

```
[inkling_kernels pid=1234] enabled qkvr_gemv(max_m=16,tile=32x256,warps=4) moe_tiles(<=8:swap_ab=1,tile_n=64,tile_k=256)
```

The patches apply when SGLang imports the target module (scheduler/model process).
`INKLING_KERNELS_VERBOSE=1` adds an "applied ..." line at that point. Failures:
- bad env, unknown lever, or a missing target file or function in the sglang
  sources: `install()` raises `RuntimeError`. The standalone `sitecustomize.py`
  then prints `FATAL` and exits the process with code 78.
- the target attribute is missing at import time: the patch raises inside the import,
  so the model import fails with a traceback.

For a combined loader, call `inkling_kernels.install()` (or `install(["qkvr_gemv"])`)
and let `RuntimeError` stop the process. `inkling_kernels.STATE["applied"]` lists the
patches that were applied in this process.

### Docker additions

```
-v /path/to/repo/overlay:/opt/inkling-overlay:ro \
-e PYTHONPATH=/opt/inkling-overlay \
-e INKLING_KERNELS=qkvr_gemv,moe_tiles \
-e INKLING_MOE_TILES='8:swap_ab=1,tile_n=64,tile_k=256' \
```

`PYTHONPATH` puts our `sitecustomize.py` ahead of site-packages, so it shadows any
sitecustomize the image ships. Check first with
`docker run --rm lmsysorg/sglang:inkling-2-mini python3 -c "import sitecustomize; print(sitecustomize.__file__)"`.
If that prints a path, the combined loader must import it explicitly.

## Lever A: `qkvr_gemv`

**Patch point.** `sglang.srt.models.inkling_common.inkling2_attn.Inkling2Attention._project_sources`.
The class attribute is replaced; the stock function stays reachable as `__wrapped__`.

**Problem.** When q, k and v come from three distinct tensors (global layers, and every
looped core layer in passes 2-3), stock code stacks them to `[3M, 2304]` and runs the
whole qkvr weight `[8600, 2304]` (39.6 MB BF16). At c1 verify `3M = 24 > MAX_M = 16`,
so the call leaves the sm12x Triton GEMV for cuBLAS `F.linear`, and two thirds of the
outputs are discarded.

**Change.** One Triton kernel over the 8600 output columns in 32-column CTAs. A CTA
in the k columns `[3072, 4096)` multiplies the k source, a CTA in the v columns
`[4096, 5120)` the v source, and every other CTA (q, r, gate, sinks, pad) the q source.
Each weight column is read once, at M rows (no 3x FLOPs). Results are written into a
`[3M, 8600]` buffer at exactly the places the stock code reads, then split with the
stock `split_sizes`. The six returned views have the stock shapes, strides and storage
offsets (CPU-verified against the real `_project_sources` source).

**When it runs.** Only in the three-distinct-sources case, with `0 < M <= INKLING_QKVR_MAX_M`
(default 16 = the stock GEMV bound), BF16 2-D sources, the weight
`[sum(split_sizes), K]`, and only where the stock sm12x GEMV route is enabled.
The last condition is `unquant._sm12x_bf16_gemv_eligible` on a one-row view, so the
overlay also steps aside under `--bf16-gemm-backend` other than auto, deterministic
inference, batch-invariant mode and torch.compile tracing. Anything else calls the
stock method.

**Exactness.** BF16 operands, FP32 accumulation, BF16 out. The default tile (BN 32,
BK 256, 4 warps) is the stock GEMV's, so each output element should be bit-identical
to the stock GEMV run on that source alone; `gpu_check_qkvr.py` asserts this at
M <= 16. Against today's cuBLAS path at 3M > 16 the result differs by FP32 summation
order only (E2): T=0 can flip a near-tie token. The check reports max abs/rel
difference against stock and against an FP64 reference.

**CUDA graphs.** The grid is static, there is no host sync and no data-dependent
shape, and the output comes from `torch.empty` like stock `F.linear`. The check
captures the kernel and replays it. Triton compiles each row bucket (MP 16/32/64)
on first use, during SGLang's eager warm-up before capture, as the stock GEMV does.

**Knobs.**
- `INKLING_QKVR_MAX_M` (1..64, default 16): raise it to 32 or 64 only if
  `gpu_check_qkvr.py` shows the kernel beating stock at M=32/64 (c4/c8).
- `INKLING_QKVR_TILE=BN,BK,WARPS` (default `32,256,4`): a non-default BK changes the
  summation order. Use only if the timing table shows a clear win.

**Expected gain.** 28 of the 48 QKVR calls per c1 step move from cuBLAS (~175 GB/s
per the stock file's docstring) to the GEMV (~240 GB/s). That is about 60 us per call,
or ~1.5-2 ms per step if those numbers hold at M=24. The real cuBLAS speed at 24 rows
is unknown, and `gpu_check_qkvr.py` measures it.

## Lever B: `moe_tiles`

**Patch point.** `sglang.kernels.ops.gemm.inkling2_grouped_sm120._tiles`, a module
attribute that `nvfp4_expert_matmul` looks up on every call. The stock table stays
reachable as `__wrapped__`. Only the NVFP4 layers 0-19 use it. The MXFP8 call
(layers 20-23) passes no tile arguments in this SGLang.

**Grammar.** `INKLING_MOE_TILES='MAXTOK[@WIDTH]:key=val,...;...'` with keys `swap_ab`
(0/1), `tile_n` and `tile_k`.
- `tokens` = `ids.shape[0]`: 8 at c1 verify, 1 at plain decode, 64 at c8 verify.
- `WIDTH` = unpacked K: 2304 for w13, 1152 for w2.
- A call takes the rule with the smallest `MAXTOK >= tokens` among the rules whose
  WIDTH is absent or matches. At equal MAXTOK the width-specific rule wins.
- Keys left out keep the stock value. Token counts no rule covers keep the stock table,
  so prefill is untouched unless you write a rule with a large bound.
- Stock today: tokens <= 128 gives `swap_ab=1, tile_n=32, tile_k=128`.

**Exactness and graphs.** The tile choice is host-side and evaluated at capture time,
so each graph keeps the tiles it was captured with. Whether a tile change is bit-exact
is measured by the bench, which compares every config bitwise with the stock config
on identical inputs. Use only configs the bench reports as `exact: yes` for an E0 arm.
A new tile config JIT-compiles a CUTLASS kernel on first use, possibly for minutes.
Run the bench first in the same container (or with the same JIT cache) so serving
start-up does not pay for it.

## Running the GPU scripts

Run them while the Inkling server is stopped or idle. The bench allocates about 1 GB
(all 128 experts of w13 + w2) plus a 96 MB L2-flush buffer, and the timings are
meaningless with serving traffic on the GPU.

Inside a running container that has the mount:

```
docker exec -it <container> python3 /opt/inkling-overlay/gpu_check_qkvr.py
docker exec -it <container> python3 /opt/inkling-overlay/bench_moe_tiles.py --mxfp8 --json /tmp/moe_tiles.json
```

One-off container:

```
docker run --rm --gpus all --ipc=host \
  -v /path/to/repo/overlay:/opt/inkling-overlay:ro \
  lmsysorg/sglang:inkling-2-mini \
  python3 /opt/inkling-overlay/gpu_check_qkvr.py

docker run --rm --gpus all --ipc=host \
  -v /path/to/repo/overlay:/opt/inkling-overlay:ro \
  -v /path/to/jit-cache:/root/.cache \
  lmsysorg/sglang:inkling-2-mini \
  python3 /opt/inkling-overlay/bench_moe_tiles.py --mxfp8
```

Do not set `INKLING_KERNELS` for these runs. Both scripts call the code directly.
Mount `/root/.cache` (FlashInfer + Triton JIT caches) if the user in the image is
root and you want compiled tiles to survive the container.

### `gpu_check_qkvr.py`

Options: `--ms 1,2,4,8,16,32,64`, `--iters 200`, `--tiles "32,256,4;64,256,4;..."`.

For every M it prints:
- lever A vs stock: max abs/rel difference and the number of differing elements;
- whether lever A is bit-equal to the per-source stock GEMV (M <= 16);
- the FP64 error of both;
- the result of a CUDA-graph replay.

Then it runs the real (unpatched) `_project_sources` against the patched one on a
stub module, for adjacent and separate source layouts, checking the views' shapes
and strides. Last comes a timing table with medians in us and weight GB/s for:
stock stacked 3M, 3x per-source stock, and lever A per tile variant. It exits 1 and
prints `FAILURES:` on any failed check, otherwise `ALL CHECKS PASSED`. If
`inkling2_attn` cannot be imported outside a server, only the stub step is skipped,
with a note.

### `bench_moe_tiles.py`

Inputs are built with SGLang's own `route_metadata` and `nvfp4_routed_quantize` /
`mxfp8_routed_quantize`, random NVFP4 weights for all 128 experts with swizzled E4M3
block scales, in the exact call form of `inkling2_grouped_sm120.py`.
- Scenarios `T:U` = MoE tokens : distinct experts. Default
  `1:8,8:20,8:27,8:40,64:100,64:128`; the decode profile fits about 27 distinct experts at M=8.
- Default grid: `swap_ab {1,0} x tile_n {32,64,128,256} x tile_k {128,256}`. `--full`
  adds tile_n 192 and tile_k 64.
- At start it prints the FlashInfer signature and the source lines mentioning tile
  checks.
- Each config: rejected (with the error), or timed (median/min/p90 us, total and weight
  GB/s, speed-up vs stock) and checked bitwise against stock.
- At the end it prints the best bit-exact config per token bucket and a ready
  `INKLING_MOE_TILES=...` string.
- An illegal tile that faults the GPU poisons the process. The bench then stops and
  says so; rerun with `--isolate` (one subprocess per config).
- `--mxfp8` introspects `moe_gemm_mxfp8_nt_groupwise`. It sweeps any tile parameters
  it finds, otherwise it times the default only (report-only; no overlay exists for
  MXFP8 because SGLang passes no tile arguments).

## Test plan

1. `gpu_check_qkvr.py` must print `ALL CHECKS PASSED`. Record the M=8 row: stock(3M)
   us vs lever A us. If lever A is not faster at M=8, drop lever A.
2. `bench_moe_tiles.py --mxfp8 --json /tmp/moe_tiles.json` (add `--isolate` if it
   reports a fatal CUDA error). Keep the suggested string only if it saves real time
   at `8:27`.
   - Per-step estimate at c1: 44 NVFP4 visits x (w13 saving + w2 saving).
3. Boot with `INKLING_KERNELS=qkvr_gemv INKLING_KERNELS_VERBOSE=1`. Check the
   `enabled` and `applied qkvr_gemv` lines and that CUDA-graph capture completes.
   Then repeat with `moe_tiles` alone, then both.
4. Exactness gate:
   - moe_tiles with bit-exact tiles is E0: T=0 outputs must equal the baseline.
   - qkvr_gemv is E2 against today's cuBLAS path: use the bracket rule. X must
     match A or A' per task.
5. Speed A/B per arm with the usual harness at c1 and c8, plus long thinking. Use
   per-step ms (expect lever A ~-1.5 to -2 ms at c1) and tok/s.
   - For c4/c8, rerun the check with `INKLING_QKVR_MAX_M` 32/64 if its timing table
     favours the kernel there.
6. Rollback: unset `INKLING_KERNELS` (or drop the `-e`), restart.
