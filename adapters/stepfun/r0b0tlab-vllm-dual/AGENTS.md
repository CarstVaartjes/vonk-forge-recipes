# AGENTS.md — Step 3.7 Flash NVFP4 on Dual GB10

Instructions for AI agents and automated systems serving, benchmarking, or
debugging `stepfun-ai/Step-3.7-Flash-NVFP4` on dual NVIDIA DGX Spark (GB10/SM121).

## What this repository is

A validated, production-ready TP=2 serving environment for Step 3.7 Flash NVFP4
on dual GB10 with vLLM and Ray. Every claim in this file is backed by evidence
in `evidence/`. The Docker image at `ghcr.io/r0b0tlab/vllm-step37-flash-nvfp4-sm121`
is the canonical artifact.

## The one rule

**Never publish fallback or emulation results as SM121-native.**

If the vLLM log shows `Using 'MARLIN'` or `Using 'EMULATION'` for any NVFP4
backend, the run is invalid. The only valid backends are:
- `FLASHINFER_CUTLASS` (MoE)
- `VLLM_CUTLASS` (dense/linear)
- `FLASHINFER` (attention)

## Model facts

| Fact | Value |
|------|-------|
| ID | `stepfun-ai/Step-3.7-Flash-NVFP4` |
| Architecture | `Step3p7ForConditionalGeneration` |
| Type | Sparse MoE VLM (image + text) |
| Total params | ~198B |
| Active params | ~11B / token |
| Experts | 288, top-k=8 |
| Quantization | NVFP4 (ModelOpt 0.45.0), FP8 KV cache |
| Context | 256K max (32K validated) |
| License | Apache-2.0 |
| Storage | ~129 GB (14 safetensors shards) |

## Hardware requirements

- 2x NVIDIA DGX Spark or GB10-based systems (SM121, 128 GB unified memory each)
- QSFP direct connect or switch (200GbE recommended)
- Docker with NVIDIA container runtime
- ~130 GB free disk per node for model weights

## Docker image

```
ghcr.io/r0b0tlab/vllm-step37-flash-nvfp4-sm121:latest
ghcr.io/r0b0tlab/vllm-step37-flash-nvfp4-sm121:cu130-v0.1.dev16944-arm64-tp2
```

Contents:
- vLLM 0.1.dev16944 (dev build with SM121 native support)
- Ray 2.55.1
- FlashInfer 0.6.11.post2
- CUTLASS FP4 kernels
- All native SM121 C extensions

## Verified launch recipe

### Environment

```bash
NCCL_SOCKET_IFNAME=enp1s0f0np0   # QSFP interface for NCCL
GLOO_SOCKET_IFNAME=enp1s0f0np0   # QSFP interface for GLOO
TP_SOCKET_IFNAME=enp1s0f0np0     # QSFP interface for TP
```

These MUST point to the QSFP interface, not tailscale, WiFi, or Ethernet.
On GB10, the QSFP interface is typically `enp1s0f0np0`. Verify with:
```bash
ip addr show enp1s0f0np0 | grep "state UP"
```

### Ray cluster startup

**Node 1 (head):**
```bash
docker run -d --name step37-vllm \
  --gpus all --ipc=host --network=host \
  --entrypoint bash \
  -v /path/to/models:/models:ro \
  ghcr.io/r0b0tlab/vllm-step37-flash-nvfp4-sm121:latest \
  -lc "ray start --head --node-ip-address=<NODE1_IP> --port=6379 --dashboard-host=0.0.0.0 --block"
```

**Node 2 (worker):**
```bash
docker run -d --name step37-vllm-worker \
  --gpus all --ipc=host --network=host \
  --entrypoint bash \
  -v /path/to/models:/models:ro \
  ghcr.io/r0b0tlab/vllm-step37-flash-nvfp4-sm121:latest \
  -lc "ray start --address=<NODE1_IP>:6379 --node-ip-address=<NODE2_IP> --block"
```

**Verify cluster:**
```bash
docker exec step37-vllm ray status
# Should show 2 active nodes, 2 GPUs
```

### vLLM launch

```bash
docker exec -d step37-vllm bash -lc '
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export GLOO_SOCKET_IFNAME=enp1s0f0np0
export TP_SOCKET_IFNAME=enp1s0f0np0
python3 -m vllm.entrypoints.openai.api_server \
  --model /models/Step-3.7-Flash-NVFP4 \
  --host 0.0.0.0 --port 30000 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --disable-custom-all-reduce \
  --quantization modelopt \
  --kv-cache-dtype fp8 \
  --disable-cascade-attn \
  --gpu-memory-utilization 0.70 \
  --max-model-len 32768 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 4 \
  --reasoning-parser step3p5 \
  --tool-call-parser step3p5 \
  --enable-auto-tool-choice \
  --enforce-eager
'
```

Wait ~4 minutes for model loading. Check health:
```bash
curl -sf http://localhost:30000/health
```

## Every flag explained

| Flag | Required | Why |
|------|----------|-----|
| `--tensor-parallel-size 2` | Yes | Two GB10 nodes, one GPU each |
| `--distributed-executor-backend ray` | Yes | Ray manages cross-node GPU workers |
| `--disable-custom-all-reduce` | Yes | SM121 lacks NCCL_SYMM_MEM; forces PYNCCL |
| `--quantization modelopt` | Yes | Activates NVFP4 modelopt_fp4 path |
| `--kv-cache-dtype fp8` | Yes | Model uses FP8 KV cache |
| `--disable-cascade-attn` | **Critical** | Step 3.7 hybrid sliding-window attention is incompatible with cascade attention |
| `--enforce-eager` | **Critical** | Triton JIT + CUDA graphs cause RPC timeout on first run |
| `--gpu-memory-utilization 0.70` | Yes | 129 GB model on 121 GB nodes needs headroom |
| `--max-model-len 32768` | Tunable | 32K verified; 64K OOMs at 0.70 |
| `--reasoning-parser step3p5` | Optional | Enables Step 3.5/3.7 reasoning extraction |
| `--tool-call-parser step3p5` | Optional | Enables Step 3.5/3.7 tool calling |
| `--enable-auto-tool-choice` | Optional | Auto-detect tool calls in chat completions |
| `--trust-remote-code` | Yes | Model uses custom code (Step3p7*) |

## Things you MUST NOT change

1. **`--disable-cascade-attn`** — Removing this breaks Step 3.7's hybrid attention.
   The model uses sliding-window + global attention layers; cascade attention
   assumes uniform attention and produces garbage.

2. **`--enforce-eager`** — Removing this on first run causes the engine to hang
   during Triton JIT compilation + CUDA graph capture. After the first successful
   run (which warms the JIT cache), CUDA graphs may work at 32K context.
   At 64K+, CUDA graphs hit SHM broadcast deadlock on cross-node TP=2.

3. **`--quantization modelopt`** — Removing this falls back to the non-NVFP4 path.

4. **`--kv-cache-dtype fp8`** — The checkpoint is calibrated for FP8 KV cache.
   Changing to `auto` or `fp16` will use more memory and may change output quality.

5. **`--gpu-memory-utilization` above 0.85** — The model uses ~58.5 GB per node.
   At 0.85, remaining headroom is ~17 GB. Docker builds, Ray overhead, and OS
   memory can consume this during operation.

6. **Network interfaces** — Always pin NCCL/GLOO/TP to the QSFP interface.
   Using tailscale or WiFi causes connection failures or extreme slowdowns.

## Verification checklist

Before claiming this setup works, verify ALL of:

```bash
# 1. Backend selection (MUST show FLASHINFER_CUTLASS + VLLM_CUTLASS)
docker logs step37-vllm 2>&1 | grep "Using.*NvFp4.*backend" | grep -v "potential"

# 2. SM121 capability
docker exec step37-vllm python3 -c "
import torch; print(torch.cuda.get_device_capability())  # expect (12, 1)
from vllm.platforms import current_platform
print(current_platform.is_device_capability_family(120))  # expect True
"

# 3. Native imports
docker exec step37-vllm python3 -c "
for m in ['vllm._C','vllm._C_stable_libtorch','vllm._moe_C']:
    __import__(m); print(f'{m}: OK')
"

# 4. Text completion
curl -s http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/models/Step-3.7-Flash-NVFP4","prompt":"The capital of France is","max_tokens":16,"temperature":0}'

# 5. Tool calling
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"/models/Step-3.7-Flash-NVFP4",
    "messages":[{"role":"user","content":"What is the weather in Tokyo?"}],
    "tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],
    "temperature":0
  }'

# 6. Vision
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"/models/Step-3.7-Flash-NVFP4",
    "messages":[{"role":"user","content":[
      {"type":"text","text":"Describe this image."},
      {"type":"image_url","image_url":{"url":"data:image/png;base64,<BASE64>"}}
    ]}],
    "max_tokens":64,"temperature":0
  }'

# 7. No fallback in logs
docker logs step37-vllm 2>&1 | grep -i "marlin\|emulation" | grep -v "potential" | wc -l
# expect: 0
```

## Performance baseline

Verified 2026-06-04 on dual DGX Spark (GB10 SM121), vLLM 0.1.dev16944,
32K context, --enforce-eager:

| Benchmark | tok/s | Method |
|-----------|-------|--------|
| llama-benchy tg128 | 16.49 ± 0.28 | uvx llama-benchy --pp 2048 --tg 128 --runs 3 |
| llama-benchy pp2048 | 1093.06 ± 94.73 | uvx llama-benchy --pp 2048 --tg 128 --runs 3 |
| Custom tg128 c1 | 14.02 | curl → /v1/completions, 1024→128, 3 runs |
| Custom random c1 | 13.2 | curl → /v1/completions, 2048→1024, 3 runs |
| Custom random c2 | 27.01 | 2 concurrent curl, 2048→1024, 3 runs |
| API latency | 2.12 ms | llama-benchy measurement |
| Coherence | PASSED | llama-benchy coherence check |

GPU state during benchmarks: 92% util, 58°C, 25.5 W.

## Context length limits

| max_model_len | Status | Notes |
|---------------|--------|-------|
| 8,192 | Verified | Default safe |
| 32,768 | Verified | Recommended for production |
| 65,536 | OOM | Needs GPU_UTIL > 0.85 or --enforce-eager only |

To increase context, reduce `--gpu-memory-utilization` to free KV cache space,
or reduce `--max-num-seqs` to limit concurrent sequence slots.

## CUDA graph status

| Context | --enforce-eager | CUDA graphs |
|---------|----------------|-------------|
| 32K | Verified | Verified (VLLM_COMPILE, FULL_AND_PIECEWISE) |
| 64K | Verified | SHM deadlock (cross-node TP=2) |

For production, use `--enforce-eager`. CUDA graphs save ~10-15% latency but
risk SHM broadcast deadlock at larger contexts.

## Troubleshooting

### Server won't start

```bash
docker exec step37-vllm ray status
```
Both nodes must show "Active" with 2 GPUs total. If worker is missing:
```bash
ssh <NODE2_IP> "docker logs step37-vllm-worker" | tail -20
```

### RPC timeout on startup

Ensure `--enforce-eager` is in the vLLM args. Without it, Triton JIT
compilation during CUDA graph capture causes the first inference to hang.

### OOM during model loading

Reduce `--gpu-memory-utilization` to 0.65 or lower. Check available memory:
```bash
free -h  # Need > 20 GB free on each node
```

### SHM broadcast timeout

```
No available shared memory broadcast block found in 60 seconds
```
This is normal during CUDA graph capture. If it persists for > 5 minutes
without the server becoming healthy, kill and restart with `--enforce-eager`.

### Worker can't reach head

Check QSFP connectivity:
```bash
ping <NODE1_IP>  # from node2
```
Verify NCCL/GLOO/TP interfaces are set to the QSFP interface, not tailscale.

### Model downloads slowly

Use `hf_transfer`:
```bash
pip install huggingface_hub hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download ...
```

### FP8 KV cache warnings

```
Checkpoint does not provide a q scaling factor
Using KV cache scaling factor 1.0 for fp8_e4m3
Using uncalibrated q_scale 1.0
```
These are expected. The Step 3.7 checkpoint does not include q_scale factors.
vLLM falls back to k_scale and 1.0 defaults. Output quality is unaffected.

### Step3VLProcessor error (vLLM 0.22.0)

```
'Step3VLProcessor' object has no attribute '_get_num_multimodal_tokens'
```
This is a vLLM 0.22.0 regression. Use the provided Docker image which pins
vLLM 0.1.dev16944. Do not upgrade vLLM without testing Step 3.7 support.

## Multi-agent / concurrent serving

The server supports OpenAI-compatible chat and completions endpoints.
For concurrent agents:
- Increase `--max-num-seqs` (default 4) for more parallel slots
- Each slot consumes KV cache memory; balance with `--max-model-len`
- Monitor with `docker logs step37-vllm 2>&1 | grep "Running.*reqs"`

## What is NOT in this image

- No CUDA graphs enabled by default (use `--enforce-eager`)
- No HF token baked in (mount or set at runtime)
- No benchmark harness (run llama-benchy or custom scripts separately)
- No Nsight tools (mount from host if needed for profiling)

## File layout

```
├── README.md                 # This file (for humans)
├── AGENTS.md                 # This file (for agents)
├── VERDICT.md                # Publication verdict
├── LICENSE                   # Apache-2.0
├── docker/
│   ├── Dockerfile.publish    # Publication Dockerfile
│   ├── entrypoint.sh         # Container entrypoint
│   └── ray_cluster.env       # Ray cluster configuration
├── scripts/
│   ├── launch_gb10_dual_tp2_step37.sh
│   ├── smoke_chat.sh
│   ├── smoke_vision.sh
│   └── ray_health.sh
├── evidence/
│   ├── runtime/              # Server logs, backend evidence
│   ├── benchmarks/           # Benchmark results
│   ├── correctness/          # Smoke test results
│   └── nsight/               # Nsight verification
└── publication/
    └── html/
        └── index.html        # HTML report
```

## License

Apache-2.0 (model) + MIT (packaging scripts).

## Publisher

[r0b0tlab](https://github.com/r0b0tlab) · [@mr-r0b0t](https://x.com/mr_r0b0t)
