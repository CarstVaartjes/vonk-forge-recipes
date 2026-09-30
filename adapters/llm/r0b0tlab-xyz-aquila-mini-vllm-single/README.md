# XYZ-Aquila-mini NVFP4 — SM121 vLLM runtime

Digest-pinned **vLLM 0.25.0** container for serving
[`r0b0tlab/XYZ-Aquila-mini-NVFP4`](https://huggingface.co/r0b0tlab/XYZ-Aquila-mini-NVFP4)
on **NVIDIA GB10 / SM121**.

## Pull

```bash
docker pull ghcr.io/r0b0tlab/xyz-aquila-mini-nvfp4-vllm:v0.25.0-sm121-702f4814
# digest sha256:502c6946029085a69decfd490f1338a982e2d5697eda419ad09f12feb3c531f0
```

## Run

```bash
docker run --gpus all --ipc=host --network host \
  -e PORT=18082 \
  -e MODEL_ID=/models/model \
  -e SERVED_MODEL_NAME=xyz-aquila-nvfp4 \
  -e QUANTIZATION=modelopt_mixed \
  -e KV_CACHE_DTYPE=fp8 \
  -e ATTENTION_BACKEND=flashinfer \
  -e MOE_BACKEND=flashinfer_b12x \
  -e LINEAR_BACKEND=flashinfer_cutlass \
  -e LANGUAGE_MODEL_ONLY=1 \
  -e GPU_MEMORY_UTILIZATION=0.85 \
  -e MAX_MODEL_LEN=32768 \
  -e MAX_NUM_SEQS=16 \
  -v /path/to/XYZ-Aquila-mini-NVFP4:/models/model:ro \
  ghcr.io/r0b0tlab/xyz-aquila-mini-nvfp4-vllm:v0.25.0-sm121-702f4814
```

### Long context (max-context / NIAH)

Weights support **262,144** tokens. To stress long context:

- set `MAX_MODEL_LEN` toward 65536 / 131072 / 262144
- use `MAX_NUM_SEQS=1` (or low)
Long-context tip: this GB10 admits full native **262144** with `MAX_NUM_SEQS=1`,
`GPU_MEMORY_UTILIZATION=0.85`, `MAX_NUM_BATCHED_TOKENS=16384` (chunked prefill).

**NIAH_MAX_CONTEXT protocol:** depths = 25% / 50% / 90% of served `(max_model_len − 64)`,
discovered from `/v1/models`. When M=262144 this is a full native max-context test.

Measured max-context NIAH: **3/3 PASS** at 65,520 / 131,040 / 235,872 prompt tokens.

## Native markers

Expect logs containing:

- `FlashInferCutlassNvFp4LinearKernel`
- `FLASHINFER_B12X` MoE backend
- FlashInfer attention `arch=sm121`
- FP8 KV

Reject Marlin / emulation paths for production claims.

## Measured results (GB10)

### Quality

| Suite | Score |
|-------|------:|
| BFCL multi_turn_base 200 | **64%** campaign · **68%** core-subset |
| BFCL AST-600 micro | **33.3%** · **33.5%** core-subset |
| AST multiple / parallel / par_mult | 35.5% / 48.5% / 16.5% (core-subset) |
| API canaries | **5/5** |
| NIAH @ served 32k (legacy window) | **3/3** |
| NIAH MAX_CONTEXT_TEST @ native 262k | **3/3** @ 65.5k / 131k / 236k |

### Throughput (vllm bench, 1024/256)

| c | out tok/s | total tok/s | mean TTFT ms | failed |
|--:|----------:|------------:|-------------:|-------:|
| 1 | 64.6 | 325 | 694 | 0 |
| 4 | 164.5 | 829 | 641 | 0 |
| 8 | 230.5 | 1162 | 1009 | 0 |
| 16 | 305.7 | 1541 | 1854 | 0 |

Full stylized board: [docs/evidence-report.html](docs/evidence-report.html)

## Bench client

```bash
docker pull ghcr.io/r0b0tlab/r0b0bench:v1.0.0-rc1
# profiles: core | core-subset (both include systems block)
```

https://github.com/r0b0tlab/r0b0bench

## License

Apache-2.0 for packaging scripts where applicable. Model weights: see HF card / upstream.
