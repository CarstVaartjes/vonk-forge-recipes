# AGENTS.md — nex-n2-mini-nvfp4

## Project Overview

Docker container for serving the NVFP4-quantized Nex-N2-mini model (Qwen3.5-MoE-35B fine-tune) via vLLM on NVIDIA Blackwell (SM 12.1) hardware. The container auto-downloads the model from HuggingFace on first startup.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  ghcr.io/r0b0tlab/nex-n2-mini-nvfp4            │
│  ┌───────────────────────────────────────────┐  │
│  │  startup.sh                               │  │
│  │  ├─ Check /mnt/model for weights          │  │
│  │  ├─ If missing → hf download from HF      │  │
│  │  └─ Launch vLLM API server                │  │
│  └───────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────┐  │
│  │  vLLM v0.22.0                             │  │
│  │  ├─ NVFP4 weights (ModelOpt 0.44.0)       │  │
│  │  ├─ FP8 KV cache (calibrated scales)      │  │
│  │  ├─ FlashInfer CUTLASS NVFP4 GEMM         │  │
│  │  ├─ FlashInfer CUTLASS MoE                │  │
│  │  └─ FlashInfer attention backend          │  │
│  └───────────────────────────────────────────┘  │
│  Base: vllm/vllm-openai:v0.22.0-aarch64        │
└─────────────────────────────────────────────────┘
         │                    │
    :8000 (API)        /mnt/model (volume)
```

## Build & Run

### Quick Start (auto-downloads model on first run)

```bash
docker run -d --name nex-n2-mini-nvfp4 \
  --gpus all \
  --shm-size=8g \
  -e HF_TOKEN=***  -v nex-n2-model:/mnt/model \
  -p 8000:8000 \
  ghcr.io/r0b0tlab/nex-n2-mini-nvfp4:latest
```

### With pre-downloaded model

```bash
# Download model first
hf download r0b0tlab/nex-n2-mini-nvfp4 --local-dir ./nex-n2-mini-nvfp4

# Mount it
docker run -d --name nex-n2-mini-nvfp4 \
  --gpus all \
  --shm-size=8g \
  -v $(pwd)/nex-n2-mini-nvfp4:/mnt/model:ro \
  -p 8000:8000 \
  ghcr.io/r0b0tlab/nex-n2-mini-nvfp4:latest
```

### Docker Compose

```bash
echo "HF_TOKEN=***docker compose up -d
```

### Build from source

```bash
git clone https://github.com/r0b0tlab/nex-n2-mini-nvfp4.git
cd nex-n2-mini-nvfp4
docker build -t nex-n2-mini-nvfp4 .
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | (required for download) | HuggingFace token for model download |
| `MODEL_DIR` | `/mnt/model` | Path to model weights |
| `PORT` | `8000` | vLLM API server port |
| `GPU_MEM_UTIL` | `0.90` | GPU memory utilization fraction |
| `MAX_MODEL_LEN` | `32768` | Maximum sequence length |
| `MAX_BATCHED_TOKENS` | `32768` | Max tokens per batch (must be >= 2096 for Mamba alignment) |
| `VLLM_EXTRA_ARGS` | (empty) | Additional vLLM CLI arguments |

## Server Verification

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "r0b0tlab/nex-n2-mini-nvfp4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## NVFP4 Quantization Pitfalls

These are critical gotchas discovered during quantization. If you're re-quantizing or modifying the checkpoint, read this section.

### 1. lm_head.weight MUST be present

ModelOpt drops `lm_head.weight` during export. Without it, the model produces gibberish. Manually copy from the original BF16 model:

```python
import torch
from safetensors.torch import load_file, save_file
bf16 = load_file("original/model.safetensors")
lm_head = {"lm_head.weight": bf16["lm_head.weight"]}
save_file(lm_head, "nvfp4/model-00003-of-00003.safetensors")
```

### 2. text_config must be in config.json

ModelOpt strips `text_config` from `config.json` during export. Copy it from the original model config. Required fields: `num_hidden_layers`, `num_experts_per_tok`, `num_experts`, `hidden_size`, etc.

### 3. Mamba block alignment

This model has hybrid linear-attention (Mamba) layers. `max_num_batched_tokens` must be >= 2096 (the Mamba cache alignment block size). Lower values cause a runtime assertion failure.

### 4. Vision encoder must be BF16

Exclude `model.visual.*` from quantization. The vision encoder weights must be copied separately from the BF16 model into the NVFP4 checkpoint. Add `language_model.model.visual.*` to the exclude list in `hf_quant_config.json`.

### 5. Weight key prefix

ModelOpt exports keys as `model.layers.*` but vLLM's `ForConditionalGeneration` expects `language_model.model.layers.*`. All weight keys must be prefixed accordingly.

### 6. torch.nvfp4 does not exist in public PyTorch

NVFP4 KV cache requires `torch.nvfp4` dtype which is NVIDIA-internal only. Public PyTorch (2.11, 2.12) does not have it. Use FP8 KV cache with calibrated scales as the fallback.

### 7. Expert calibration

All 10,240 experts (256 per layer × 40 layers) must be 100% calibrated. Use 128 calibration samples from CNN/DailyMail. Check calibration coverage:

```python
import json
with open("model.safetensors.index.json") as f:
    idx = json.load(f)
experts = [k for k in idx["weight_map"] if "experts" in k]
print(f"Expert tensors: {len(experts)}")
```

## Benchmark Reproduction

```bash
# Capability tests
for prompt in "What is 17*23+45-12?" "Write a palindrome function in Python" "Explain TCP vs UDP"; do
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"r0b0tlab/nex-n2-mini-nvfp4\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":500}" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'][:200])"
done

# Throughput (tg128)
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"r0b0tlab/nex-n2-mini-nvfp4","prompt":"Hello","max_tokens":128,"temperature":0}'
```

## Key Results

| Metric | Value |
|---|---|
| Decode (tg128) | 33.35 t/s |
| Prefill (pp2048 @ d16384) | 5,017 t/s |
| Concurrent C8 aggregate | 185.5 t/s |
| Model size | 22.1 GiB (3.2× compression from 70 GB BF16) |
| Power (peak) | 23.3W at 48°C |
| Capability tests | 13/13 passed |

## File Structure

```
nex-n2-mini-nvfp4-container/
├── AGENTS.md                 # This file — agent development guidelines
├── README.md                 # User-facing documentation + quickstart
├── Dockerfile                # Container definition (vLLM + HF download)
├── docker-compose.yaml       # One-command deployment
├── scripts/
│   ├── startup.sh            # Entrypoint — model download + vLLM launch
│   └── healthcheck.sh        # Docker HEALTHCHECK
├── .dockerignore
└── LICENSE                   # Apache 2.0
```
