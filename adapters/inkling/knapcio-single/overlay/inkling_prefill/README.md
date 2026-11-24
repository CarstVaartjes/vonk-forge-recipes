# inkling_prefill: exact prefill levers for Inkling-2-Mini on GB10

Switch: `INKLING_PREFILL=qkvr_split,bias_layout,quant_once,partial_bf16` (or `all`); unset = nothing is imported.
Each lever applies only to prefill / extend shapes (and `qkvr_split` also to target-verify graphs with >= 128 rows,
i.e. 16 or more concurrent requests); every other call goes to whatever was installed before (stock, or the decode
overlays). A bad value or a missing patch target stops the server at start-up (exit 78). Analysis behind the levers:
`docs/prefill/PREFILL.md`.

| lever | patch point | what | check on GB10 (`tools/gpu_check_prefill.py`, real weights) |
|---|---|---|---|
| `qkvr_split` | `Inkling2Attention._project_sources` | the looped layers' stacked [3M, 2304] x [2304, 8600] projection (three sources, a third of each output kept) becomes four GEMMs on the column blocks each source needs, written into one [M, 8600] buffer with the stock strides; from 128 rows per source (`INKLING_PREFILL_QKVR_MIN_M`, refused below 128) | bit-identical at every size >= 128 incl. all prefill-graph buckets and verify row counts; that call 62-64 % faster |
| `bias_layout` | `Inkling2Attention._forward`, `RelLogitsProj.forward`, the SM120 FA4 wrapper | one Triton kernel computes the relative-position bias directly in FA4's sheared layout (with the -inf / left pads) in the same workspace the stock shear uses; FA4 is called with the presheared bias | sheared buffer, attention output and LSE bit-identical; graph capture / replay checked |
| `quant_once` | `experts_sm120.nvfp4_routed_quantize` (top-k > 1, >= 256 tokens) | quantise each token's NVFP4 activation once, then gather values, amax and scale words into the routed layout | values, amax, every scale byte and the w13 grouped-GEMM output bit-identical |
| `partial_bf16` | `routed_partial` inside `_forward_sm120` | the MoE combine stores BF16 directly instead of FP32 followed by a cast | bit-identical |

Measured end to end (speedbench longctx, cold): 1K 2,151 -> 2,363, 8K 3,408 -> 4,253, 32K 3,235 -> 3,958, 128K
2,409 -> 2,749 tok/s with all four levers (`docs/experiments.md`). Tests: `tests/prefill/`.
