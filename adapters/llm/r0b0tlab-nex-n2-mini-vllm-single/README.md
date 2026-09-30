# nex-n2-mini-nvfp4

NVFP4-quantized [Nex-N2-mini](https://huggingface.co/Nex-AGI/Nex-N2-mini) (Qwen3.5-MoE-35B) served via vLLM on NVIDIA Blackwell (SM 12.1).

[![Docker Image](https://img.shields.io/badge/ghcr.io-r0b0tlab%2Fnex--n2--mini--nvfp4-blue)](https://github.com/r0b0tlab/nex-n2-mini-nvfp4/pkgs/container/nex-n2-mini-nvfp4)
[![HuggingFace](https://img.shields.io/badge🤗-HuggingFace-yellow)](https://huggingface.co/r0b0tlab/nex-n2-mini-nvfp4)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

## Quick Start

```bash
docker run -d --name nex-n2-mini-nvfp4 \
  --gpus all \
  --shm-size=8g \
  -e HF_TOKEN=***  -v nex-n2-model:/mnt/model \
  -p 8000:8000 \
  ghcr.io/r0b0tlab/nex-n2-mini-nvfp4:latest
```

Model downloads from HuggingFace on first start (~24 GB). Subsequent starts use the cached volume.

### With pre-downloaded model

```bash
hf download r0b0tlab/nex-n2-mini-nvfp4 --local-dir ./nex-n2-mini-nvfp4

docker run -d --name nex-n2-mini-nvfp4 \
  --gpus all \
  --shm-size=8g \
  -v $(pwd)/nex-n2-mini-nvfp4:/mnt/model:ro \
  -p 8000:8000 \
  ghcr.io/r0b0tlab/nex-n2-mini-nvfp4:latest
```

### Docker Compose

```bash
echo "HF_TOKEN=*** compose up -d
```

## Verify

```bash
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"r0b0tlab/nex-n2-mini-nvfp4","messages":[{"role":"user","content":"Hello!"}],"max_tokens":100}'
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | (required) | HuggingFace token for model download |
| `MODEL_DIR` | `/mnt/model` | Model weights path |
| `PORT` | `8000` | API server port |
| `GPU_MEM_UTIL` | `0.90` | GPU memory utilization |
| `MAX_MODEL_LEN` | `32768` | Max sequence length |
| `MAX_BATCHED_TOKENS` | `32768` | Max batch tokens (>= 2096 required) |
| `VLLM_EXTRA_ARGS` | (empty) | Extra vLLM CLI args |

## Model Details

| Property | Value |
|---|---|
| Base model | [Nex-AGI/Nex-N2-mini](https://huggingface.co/Nex-AGI/Nex-N2-mini) |
| Architecture | Qwen3_5MoeForConditionalGeneration |
| Parameters | 35B total / 3B active (MoE, 256 experts, top-8) |
| Layers | 40 (30 linear attention + 10 full attention) |
| Quantization | NVFP4 via [ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) 0.44.0 |
| Quant scope | MLP-only (expert projections + shared expert) |
| Group size | 16 |
| Calibration | 128 samples from CNN/DailyMail |
| Original size | ~70 GB (BF16) |
| NVFP4 size | ~22.1 GiB (3.2× compression) |
| KV cache | FP8 e4m3 with calibrated per-layer scales |
| Vision encoder | ViT (BF16, not quantized) |

## Benchmarks

Tested on NVIDIA GB10 (Blackwell SM 12.1), vLLM v0.22.0, FlashInfer CUTLASS NVFP4.

### Throughput (llama-benchy, 3 runs per test)

| Test | Throughput | Peak t/s |
|---|---|---|
| pp2048 | 1,974 t/s | — |
| tg128 | **33.35 t/s** | 38.33 |
| pp2048 @ d4096 | 4,007 t/s | — |
| pp2048 @ d8192 | 4,793 t/s | — |
| pp2048 @ d16384 | 5,017 t/s | — |

Decode throughput is stable at 32–33 t/s across all context depths (0–16K). Only 2.8% degradation.

### Concurrency Scaling

| Concurrency | Aggregate t/s | Per-request t/s | Power | Temp |
|---|---|---|---|---|
| C1 | 28.5 | 28.6 | 20.4 W | 45°C |
| C2 | 51.6 | 25.8 | 18.4 W | 46°C |
| C4 | 105.3 | 26.3 | 20.2 W | 47°C |
| C8 | **185.5** | 23.2 | 22.1 W | 48°C |

6.5× scaling factor at C8. 8.42 t/s/W efficiency. Peak 23.3W at 48°C.

### SM121 Backend Verification

- ✅ NVFP4 GEMM: `FlashInferCutlassNvFp4LinearKernel`
- ✅ MoE: `FLASHINFER_CUTLASS` (selected from 7 candidates)
- ✅ Attention: `FLASHINFER`
- ✅ KV cache: FP8 e4m3 (calibrated)
- ✅ Expert calibration: 10,240/10,240 (100%)

### Capability Tests: 13/13 Passed

Math (3/3), Reasoning (3/3), Coding (3/3), Knowledge (2/2), Instruction (2/2).

## Serving Stack

- **vLLM** v0.22.0 (Docker, aarch64)
- **FlashInfer CUTLASS** NVFP4 linear kernel + MoE backend
- **FP8 KV cache** with per-layer calibrated scales (k: 0.016–0.038, v: 0.010–0.040)
- **Calibrated scales** improve throughput ~10–15% vs uncalibrated FP8

## Known Issues

- `max_num_batched_tokens` must be >= 2096 (Mamba block alignment)
- NVFP4 KV cache unavailable (requires `torch.nvfp4`, NVIDIA-internal only)
- Vision encoder profiling takes ~4–5 min on first startup
- Requires NVIDIA Blackwell GPU (SM 12.1) for NVFP4 kernels

## License

Apache 2.0. Base model: [Nex-AGI/Nex-N2-mini](https://huggingface.co/Nex-AGI/Nex-N2-mini).

## Links

- **Model weights**: [r0b0tlab/nex-n2-mini-nvfp4 on HuggingFace](https://huggingface.co/r0b0tlab/nex-n2-mini-nvfp4)
- **Base model**: [Nex-AGI/Nex-N2-mini](https://huggingface.co/Nex-AGI/Nex-N2-mini)
- **Quantization tool**: [NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
- **Serving engine**: [vLLM](https://github.com/vllm-project/vllm)
