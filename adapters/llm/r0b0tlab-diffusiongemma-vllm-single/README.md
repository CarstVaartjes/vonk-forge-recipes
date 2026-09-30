# DiffusionGemma 26B A4B IT NVFP4 on SM121 vLLM

Optimized NVIDIA GB10 / SM121 vLLM container and benchmark report for `nvidia/diffusiongemma-26B-A4B-it-NVFP4`.

## Current status

- Repo: https://github.com/r0b0tlab/diffusiongemma-26b-nvfp4-sm121-vllm
- Container image: `ghcr.io/r0b0tlab/vllm-diffusiongemma-26b-nvfp4-sm121:spark-dgemma-patched`
- HTML report: `publication/html/index.html`
- Native NVFP4 MoE backend: `FLASHINFER_CUTLASS`
- Attention backend: `TRITON_ATTN`
- KV cache: `fp8`
- Max model len tested: `65536`
- Max concurrency tested: `c16`

> Note: GitHub Container Registry created the package as private by default for this org. The image has been pushed; public anonymous pull requires changing package visibility to Public in GitHub package settings.

## What this repo publishes

- A reproducible vLLM OpenAI-compatible serving recipe for DiffusionGemma NVFP4 on SM121.
- A Docker image built/pinned from the day-0 vLLM DiffusionGemma support line.
- Native-backend evidence: no MARLIN/emulation for NVFP4 quant kernels.
- Benchmark artifacts: concurrency c1/c2/c4/c5/c8/c16, depth sweep, dLLM stream metrics, telemetry, and quality spot-checks.
- A standalone HTML report under `publication/html/index.html`.

## Model

- Model ID: `nvidia/diffusiongemma-26B-A4B-it-NVFP4`
- Pinned revision: `08763e523f7cac09886956a3d7053a51e2e320dc`
- Base: Google DeepMind DiffusionGemma 26B A4B IT
- Quantization: NVIDIA ModelOpt NVFP4
- KV cache quantization: FP8
- Architecture: discrete diffusion language model, 256-token canvases, Gemma 4 MoE backbone
- Parameters: 25.2B total / 3.8B active
- License: Apache-2.0 plus Gemma terms and prohibited-use policy

## Benchmark highlights

- c1 output throughput: 146.32 tok/s
- c5 output throughput: 235.43 tok/s
- c16 output throughput: 242.93 tok/s
- c16 status: 64/64 requests OK
- Longest verified prompt: 64,034 prompt tokens
- Correctness smoke: PASS for capital/math chat prompts
- Streaming shape: 3 chunks/request; average TTFT ~2.18s on the sampled dLLM stream checks
- Telemetry sample during c16 run: max 75 C, max 60.07 W, max GPU util 96%

## Quick start

Download the model to the host and point `MODEL_DIR` at it, then serve:

```bash
export MODEL_DIR=/path/to/nvidia/diffusiongemma-26B-A4B-it-NVFP4
export IMAGE=ghcr.io/r0b0tlab/vllm-diffusiongemma-26b-nvfp4-sm121:spark-dgemma-patched
MAX_MODEL_LEN=65536 MAX_NUM_SEQS=16 KV_CACHE_DTYPE=fp8 bash scripts/serve.sh
```

Verify the native kernel path:

```bash
bash scripts/verify_native_kernels.sh
```

Run benchmarks after serving:

```bash
python3 scripts/http_chat_bench.py --requests-per-concurrency 4 --max-tokens 256
python3 scripts/bench_dllm_metrics.py --base-url http://127.0.0.1:30000/v1 --model /model
python3 scripts/http_depth_sweep.py --base-url http://127.0.0.1:30000/v1 --model /model
```

## Native publication rule

Do **not** publish or cite benchmark numbers as SM121-native if logs show MARLIN, emulation, unsupported quant kernels, failed correctness, missing telemetry, or harness failures. Fix and rerun.

## Important day-0 finding

The official `vllm/vllm-openai:gemma4` image that was available during this run did not include `DiffusionGemmaForBlockDiffusion` in the model registry. The working container therefore uses the Spark day-0 vLLM wheel plus the DiffusionGemma patch set, then verifies the final runtime through actual model loading and native-kernel logs.

## Credits

Google DeepMind (DiffusionGemma), NVIDIA (NVFP4 ModelOpt quantization), vLLM/Red Hat (day-0 dLLM integration), FlashInfer, NVIDIA CUTLASS, Hugging Face Hub, spark-vllm-docker, and r0b0tlab tooling.
