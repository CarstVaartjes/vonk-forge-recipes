# DESIGN — the GLM-5.2 NVFP4 KV-cache port, end to end

This is the architecture writeup: what the port changes, why the record is 400
bytes and not something smaller, how the write and read paths work, and what
keeps a wrong-layout read from ever silently corrupting output. Companion docs:

- [LAYOUT-SPEC.md](LAYOUT-SPEC.md) — the byte-level record (single source of truth)
- [BENCHMARKS.md](BENCHMARKS.md) — every measured number, with methodology
- [OPERATIONS.md](OPERATIONS.md) — running the deployed stack
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — failure modes we actually hit

Code map (everything under `port/`):

| file | role |
|---|---|
| `apply_nvfp4_plumbing.py` | anchor-patcher: registers `nvfp4_ds_mla` in vLLM + rewires the flashmla_sparse overlay |
| `nvfp4_glm_kernels.py` | Triton store kernel, prefill gather-dequant, rows-dequant oracle, GPU selftest |
| `b12x-nvfp4/` | patched b12x CuTe kernel set (traits, io, io_mg, kernel, prefill_mg, nvfp4 expansion stage, vLLM-side helpers) |
| `build-nvfp4.sh` / `deploy-nvfp4.sh` | per-node image build; gated cluster swap with automatic rollback |
| `test-b12x-nvfp4.py` | pre-boot GPU gate: b12x read path vs a torch oracle |

---

## 1. The problem: the record IS the context limit

GLM-5.2's sparse-MLA main KV cache stores one 656-byte `fp8_ds_mla` record per
token (512 e4m3 NoPE + 16 B of fp32 tile scales + 128 B bf16 RoPE). At our
serving config the KV pool is a fixed **10.95 GB** per GPU
(`--kv-cache-memory-bytes 10950000000`, gmu 0.91 — both razor-tuned on GB10),
which caps the pool at **200,064 tokens**. That number, not compute, is what
held the stack at 200K context.

NVFP4 replaces the fp8 payload with 4-bit E2M1 values under per-32-dim UE8M0
block scales: **400 B/token** (61% of 656). Same 10.95 GB now holds **317,312
tokens** (measured at the 200K config; 317,279 at the live 316K config) —
**+58.6% pool**, serving at **316K context**. The naive 656/400 ratio predicts
~328K; the gap is per-token pool overhead the port leaves untouched (the DSA
indexer cache shares the budget, see §6).

### Why 400 bytes and not 0xdfi's 368

The 0xdfi reference deployment uses a 368 B/token variant with a different
scale/rope packing. We deliberately kept 400:

- **The record shape transfers 1:1.** `nvfp4_ds_mla` keeps fp8_ds_mla's exact
  structure — `[payload | 16 scale bytes | 128 B bf16 rope]`, one contiguous
  record per token, no split footer. Every piece of per-token indexing logic
  (record walk, scale region size, rope tail) ports with a stride change.
  Per-32 UE8M0 granularity is what makes the scale region come out at exactly
  16 bytes (512/32 = 16), the same size the fp8 record reserves for its 4 fp32
  tile scales.
- **The rope half is byte-identical.** Bytes [272:400) of our record are the
  same bf16 bits fp8_ds_mla stores at [528:656). The read side's rope handling
  (smem staging in decode, global/L2 reads in the MG lane) needs only an
  offset change, and rope precision is untouched — important because rope
  carries positional signal.
- **The dequant idiom already existed in the stack.** The overlay's fp8ds
  footer kernels (sparse_mla_kernels.py, DSV4 448-dim variant) already
  dequantize as `tl.exp2(encoded_scale - 127.0)` over UE8M0 block-scale bytes;
  our read path is the same walk with a 32-wide block and a nibble decode.

Result: a **mechanical, low-risk port** — every changed constant is derivable
from the layout table, and the fp8 KNOWNGOOD path stays byte-identical (see
§5). The extra 32 B/token vs 0xdfi is the price of that; 368 B is a possible
follow-up once the 400 B stack has soaked.

---

## 2. Write path: one Triton kernel behind an existing custom op

vLLM's MLA layer routes every cache insert through the
`torch.ops.vllm.unified_mla_kv_cache_update` custom op, which dispatches to
`do_kv_cache_update` on the attention backend impl. That gave us a single,
clean seam: the plumbing patcher overrides `do_kv_cache_update` on the
flashmla_sparse overlay impl (`apply_nvfp4_plumbing.py`,
`flashmla:store_override`) — `nvfp4_ds_mla` goes to our Triton store kernel,
everything else falls through to `super()`.

The store kernel (`nvfp4_glm_kernels.py::_store_nvfp4_glm_kv_kernel`, one
program per token) does, per 32-dim NoPE block:

1. `amax = max(|x|)` in fp32;
2. `E = ceil(log2(amax / 6.0))` (6.0 = E2M1 max magnitude), clamped to
   [-127, 127]; scale byte = `E + 127` (UE8M0);
3. values packed two-per-byte with the hardware
   `cvt.rn.satfinite.e2m1x2.f32` instruction — `satfinite` clamps
   `|x / 2^E| > 6` to the ±6 code, so no explicit clamp is needed;
4. the 64 rope bf16s are copied as raw uint16 (bit-exact, no arithmetic).

A zero block encodes scale byte 0 and all-zero nibbles and round-trips to
exact zeros (a deliberate deviation from the danielwoz epsilon-floor, which
encodes zero blocks as byte 1 — byte 0 is what our dequant and the b12x
UE8M0 helper convention expect).

Provenance: the quant scheme, kernel structure, and the inline-asm E2M1 packer
come from **danielwoz/vllm-dspark-nvfp4** (Apache-2.0), vendored in this repo
under `danielwoz/`. Differences: 448 -> 512 NoPE dims, per-64 -> per-32
blocks, split-footer -> per-token contiguous scales, and **no in-kernel RoPE**
— GLM's vLLM path ropes `k_pe` before the cache write, so the whole
cos/sin-cache machinery is dropped and the rope half is a raw copy.

### The rope-fusion pass is double-guarded

vLLM's compile-time fusion pass `mla_rope_kvcache_cat_fusion` pattern-matches
rope + cache-write and replaces them with a fused C++ op
(`concat_and_cache_mla_rope_fused`) — which only knows how to write 656-byte
fp8 records. If it ever matched an nvfp4 layer it would scribble fp8-layout
bytes into a 400 B-stride cache: silent, total corruption. So the patcher
installs two guards:

- **skip at registration** (`fusion:registration_skip`): layers with
  `kv_cache_dtype == "nvfp4_ds_mla"` are excluded from the pattern
  registration loop, so the fusion never fires;
- **raise at runtime** (`fusion:runtime_guard`): the fused impl itself raises
  if it is ever entered with nvfp4 — belt-and-suspenders in case a future
  rebase regresses the registration skip.

The double guard costs nothing (the unfused Triton store is our write path
anyway) and turns the worst silent-corruption scenario in the port into a
loud boot-time crash.

---

## 3. Read paths

There are three readers of the main KV cache, and they are not the same code.

### 3a. Decode + mixed-batch prefill: the b12x CuTe lane (production)

The interesting one. With TP4, each rank runs 16 query heads — below the
overlay's `MIN_HEADS_FOR_BF16_PREFILL = 32` — so flashmla_sparse routes BOTH
decode and prefill through the "mixed-batch" quantized-record path, which
lands in b12x's CuTe `run_unified_prefill_mg` (via
`b12x_sparse_helpers.b12x_glm_mla_attention`). This is the production lane;
its per-step cost is the serving speed.

The port uses an **expansion-stage design** rather than teaching the math
pipeline to read nibbles:

- **IO stage (changed):** the IO warps bulk-gather each candidate token's
  packed 272-byte payload row (256 B E2M1 nibbles + 16 B UE8M0 scales)
  **densely into the FRONT of the unchanged 64x528 B `kv_fp8` smem buffer**.
  Zero extra shared memory: the staging region (64x272 = 17,408 B) fits
  inside the existing 33,792 B buffer. The mbarrier `expect_tx` count drops
  33,792 -> 17,408 (the transaction count must equal the bulk-copied bytes
  exactly). Alignment falls out of the layout: 272 = 16x17 and 400 = 16x25,
  so every `cp.async.bulk` source, destination, and size stays 16 B-aligned,
  and the rope's gmem offset (272) is too.
- **Expansion stage (new, `b12x-nvfp4/nvfp4.py::s0k_nvfp4_expand_kv_smem`):**
  before the first math stage, the 256 math threads expand the staging rows
  **in place** to standard 528 B GLM fp8 compute rows — 512 e4m3 bytes plus 4
  inline fp32 tile scales at [512:528), exactly the bytes the fp8 bulk copy
  would have landed. One thread per (token, 128-dim tile) = 64x4 = 256.
  Because output overlaps input, it is two-phase: phase 1 pulls every
  thread's source bytes into registers (17 u32), a CTA-math barrier, then
  phase 2 decodes and writes. The decode itself is
  `nvfp4_tile_scale_residuals` (max of the tile's 4 UE8M0 exponents -> one
  fp32 tile scale, plus f16x2 residuals `2^(e_i - max e)` per block) feeding
  b12x upstream's `e2m1x8_mul_residual_to_e4m3x8` (from `b12x/cute/fp4.py`),
  which multiplies 8 nibbles by the residual and re-encodes as e4m3, nibble i
  -> byte i.
- **Math stages S1–S7 (unchanged):** downstream of the expansion the kernel
  is running on byte-identical inputs to the fp8 path — same smem row format,
  same ARBITRARY_FP32 scale contract, same MMA schedule. **This is why the
  engine step time is unchanged (~131 ms vs ~127 ms):** the port adds a short
  register-resident stage between the mbarrier wait and S1 and removes 16 KB
  of gmem traffic per chunk, and the two roughly wash.

Trait deltas tell the whole story (`b12x-nvfp4/traits.py`, GLM_NSA + nvfp4):
`kv_gmem_stride` 656 -> 400, `kv_pack_stride` 0 -> 272 (nonzero = staging
mode), `bulk_tx_bytes` 41,984 -> 25,600 in the decode kernel (64x(272+128);
the MG lane computes 64x272 since it reads rope from global/L2). Every
compute-side constant — `kv_smem_stride` 528, `quant_tile` 128, `num_scales`
4, thread counts — is **unchanged**, which is the design goal stated as a
number.

The b12x split decode kernel (`b12x-nvfp4/kernel.py`) got the same treatment
(same traits, same expansion call between mbarrier wait and S1) so the lane
stays correct even where routing sends it there; on our config the MG lane is
what serves.

### 3b. Separate-prefill lane: Triton gather-dequant

For long-prompt chunked prefill the fp8 path gathers cache rows into a bf16
workspace with a C++ op, `ops.cp_gather_and_upconvert_fp8_kv_cache` — which
only parses 656 B records. The patcher branches that call site
(`flashmla:prefill_gather`) to
`nvfp4_glm_kernels.gather_dequant_nvfp4_glm`: a Triton kernel that walks
`block_table[r][t // block_size]`, dequantizes per LAYOUT-SPEC, and writes
bf16 [n, 576] rows (rope again as a raw uint16 copy, bit-exact). Grid is
(token-tile, request) with tiling on axis 0 because a 317K-token request would
otherwise exceed the 65,535 CUDA grid-Y limit; block offsets promote to int64
because the pool crosses 2^31 bytes.

An escape hatch, `GLM52_NVFP4_FORCE_SEPARATE=1`
(`flashmla:nvfp4_separate_hatch`), forces nvfp4 prefill down this lane instead
of the b12x mixed-batch lane — kept as a rescue path if the CuTe lane ever
misbehaves in production.

### 3c. Precision: per-32 scales realized as per-128 tile-max shift

The compute pipeline's scale contract is 4 fp32 scales per 128-dim tile. The
expansion stage realizes our 16 per-32 UE8M0 scales inside that contract:
tile scale = `2^(max(e0..e3) - 127)`, and each block's payload is shifted by
`2^(e_i - max e)` during the e4m3 re-encode, so
`payload x tile_scale == e2m1 x 2^(e_i - 127)` — the exact LAYOUT-SPEC
dequant. The shift is **bit-exact whenever a block's scale is within 2^8 of
its tile max** (e4m3 absorbs the shift in its exponent range). On real GLM
latents, blocks within a 128-dim tile are magnitude-correlated and validation
saw **100% bitmatch**; the adversarial worst case (a block sitting far below
its tile max) rounds/flushes that block's smallest values with error bounded
by ~`2^-9 x tile_amax` per element. The oracle gate's tolerances (§5) budget
for exactly this. Rope is bit-exact everywhere by construction.

---

## 4. Plumbing: registering a dtype nobody can accidentally misread

`apply_nvfp4_plumbing.py` performs every edit as an exact-anchor replace —
if an anchor is not found exactly once, it aborts loudly; edits whose
replacement is already present are skipped (idempotent re-runs). Two targets:

**Image side** (5 files in site-packages/vllm):

- `config/cache.py`: `"nvfp4_ds_mla"` joins the `CacheDType` literal (this is
  what lets `--kv-cache-dtype nvfp4_ds_mla` parse);
- `utils/torch_utils.py`: storage dtype uint8 + `is_quantized_kv_cache`
  predicate;
- `v1/kv_cache_interface.py`: page math — `block_size * 400`. This single
  number is what moves the boot print from 200,064 to 317,xxx tokens;
- `model_executor/layers/attention/mla_attention.py`: the **auto-convert
  guard**. Upstream "helpfully" rewrites any quantized cache dtype to
  `fp8_ds_mla` for MLA backends; without this one-line widening the server
  would have accepted `--kv-cache-dtype nvfp4_ds_mla` and silently served
  fp8 at 200K — the wrong-launcher failure mode with the right launcher;
- `compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py`: the double
  guard from §2.

**Overlay side** (a COPY of the glm-triton overlay's `flashmla_sparse.py`,
never the KNOWNGOOD dir): backend dtype list, 400 B cache shape, metadata
builder flag, impl asserts + workspace, the prefill gather branch, the b12x
`kv_layout` threading, the store override, and the no-fallback raise.

Two invariants run through all of it:

- **Layout inference by record stride, mismatch = hard error.** Both the
  vLLM-side helper (`b12x_sparse_helpers`) and b12x proper
  (`traits.infer_kv_layout`) resolve the layout from the cache tensor's last
  dim (656 vs 400) and/or an explicit `kv_layout` argument; if both are given
  they must agree or the call raises. A 400 B cache can never be walked with
  656 B strides by accident.
- **No silent fp8 fallback.** If the b12x nvfp4 path fails, the decode does
  NOT fall through to the compiled FlashMLA kernel (fp8-record-only) — it
  raises (`flashmla:no_fp8_fallback`). The `kv_layout` kwarg is only added to
  the b12x call on the nvfp4 path, so the fp8 call stays byte-identical and
  keeps working against a pre-port b12x install; a `TypeError` from an
  un-ported b12x is converted to a loud "port missing" error. The compile key
  gains a `kv_layout` field only off the fp8 default, so cached fp8 kernels
  are untouched.

---

## 5. Deployment safety: three gates and a parked rollback

`build-nvfp4.sh` rebuilds deterministically on every node (base image
`vllm-node-tf5-glm52-b12x:probe-modded` -> `...:nvfp4-v1`; overlay
`/var/tmp/glm-triton` copied to `/var/tmp/glm-triton-nvfp4` and patched — the
KNOWNGOOD dir is never touched). `deploy-nvfp4.sh` then swaps the cluster
behind three gates, each with an automatic path back to the fp8 launcher:

1. **Gate 0 — import/signature dry-check, no GPU, BEFORE teardown.** Asserts
   `run_unified_prefill_mg` accepts `kv_layout`, the helpers' stride map is
   right, and every kernel entry point imports — inside the candidate image
   with the serving env (`GLM52_B12X_MLA=1`) set. Failure costs nothing: the
   fp8 stack is still serving. (This gate exists because its absence cost us
   a boot cycle — see TROUBLESHOOTING #1.)
2. **Gate 1 — Triton store/gather roundtrip selftest** on the freed GPU
   (`nvfp4_glm_kernels.py::selftest`): rope bit-identity, per-block error
   bounds against the *stored* scale bytes, zero-block encoding, slot-mapping
   sentinels, gather-vs-rows bit parity. Runs in the teardown window because
   gmu 0.91 leaves no GB10 headroom to test beside a serving engine
   (TROUBLESHOOTING #3). Failure -> `bash speednight-k5.sh` automatically.
3. **Gate 2 — b12x-vs-oracle numerical gate** (`test-b12x-nvfp4.py`), with
   the overlay mounted exactly as serving mounts it: store 2,048 tokens with
   OUR kernel, read with b12x's CuTe kernel, compare against a torch oracle
   that attends over the same dequantized bytes (so quantization loss cancels
   and the comparison isolates addressing/dequant bugs). Includes a
   rope-only-query probe that instantly catches a wrong rope gmem offset —
   an offset bug there reads scale/nibble bytes as bf16 and destroys the
   correlation. Failure -> automatic rollback, same as gate 1.

Only then does the NVFP4 launcher boot. Rollback stays one command at any
later time: the fp8 image, overlay dir, and launcher are all parked intact.

## 6. What the port does NOT touch

- **The DSA indexer cache.** The sparse-attention indexer keeps its own
  separate per-token cache, untouched by the port — which is also why pool
  growth is +58.6% rather than the naive record ratio of +64%.
- **The fp8 KNOWNGOOD path.** Every b12x change is behind trait defaults
  (`kv_layout=FP8_DS_MLA`, `kv_pack_stride=0`) chosen so the fp8 trace and
  PTX are byte-identical to the pre-port build; the overlay edits widen
  conditions rather than change fp8 behavior; the fp8 launcher, image, and
  overlay dir are physically separate artifacts.
- **Weights, TP layout, MTP draft, scheduler config** — identical to the fp8
  stack; only `--kv-cache-dtype`, the image tag, the overlay dir, and
  `--max-model-len` differ between the launchers.

## 7. Provenance / credits

| piece | source |
|---|---|
| Quant scheme, store-kernel structure, inline-asm E2M1 packer | danielwoz/vllm-dspark-nvfp4 (Apache-2.0; vendored under `danielwoz/`), building on the DSpark fork lineage and Keys' NVFP4 image (v0.21, `keys-ref/`) |
| E2M1/UE8M0 CuTe register helpers (`e2m1x8_mul_residual_to_e4m3x8` etc.) | b12x upstream `b12x/cute/fp4.py` (FlashInfer team, Apache-2.0) |
| fp8_ds_mla 656 B layout, Triton record-walk/dequant structure | jasl/vllm deepseek_v4 path via CosmicRaisins/glm-5.2-gb10 overlays (`overlays/`) |
| Recipe + target numbers (42 tok/s, 368 B variant) | 0xdfi/GLM-5.2-1M-4x-DGX-Spark |
| Base serving stack | QuantTrio GLM-5.2 AWQ, Zatz/back199640/ciprianveg recipe lineage |
