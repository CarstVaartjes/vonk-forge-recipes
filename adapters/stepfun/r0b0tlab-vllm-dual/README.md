# Step 3.7 Flash NVFP4 — SM121-Verified TP=2 on Dual GB10

**198B parameter sparse MoE VLM served natively on dual NVIDIA DGX Spark with vLLM**

[![SM121 Verified](https://img.shields.io/badge/SM121-Native%20NVFP4-00ff88)](https://github.com/r0b0tlab/step37-flash-nvfp4-sm121-vllm-docker)
[![Docker](https://img.shields.io/badge/GHCR-ghcr.io%2Fr0b0tlab%2Fvllm--step37--flash--nvfp4--sm121-blue)](https://ghcr.io/r0b0tlab/vllm-step37-flash-nvfp4-sm121)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE)

## What This Is

A plug-and-play Docker container for running [StepFun Step 3.7 Flash NVFP4](https://huggingface.co/stepfun-ai/Step-3.7-Flash-NVFP4) on **dual NVIDIA GB10 / DGX Spark** (SM121 Blackwell) with **vLLM tensor-parallel 2** over Ray. Verified native Blackwell CUTLASS execution — zero Marlin, zero emulation.

| Spec | Value |
|------|-------|
| Model | stepfun-ai/Step-3.7-Flash-NVFP4 |
| Architecture | Step3p7ForConditionalGeneration |
| Total params | ~198B |
| Active params | ~11B / token (288 experts, top-k=8) |
| Modality | Image + Text (VLM) |
| Quantization | NVFP4 (ModelOpt 0.45.0), FP8 KV cache |
| Context | 32K validated (256K max) |
| License | Apache-2.0 |

## Performance (Dual GB10, TP=2)

| Benchmark | tok/s | Notes |
|-----------|-------|-------|
| llama-benchy tg128 | **16.49** (peak 17.00) | Single-stream decode |
| llama-benchy pp2048 | **1093.06** | Prefill throughput |
| Custom tg128 c1 | 14.02 | 1024→128 tokens |
| Custom random c1 | 13.2 | 2048→1024 tokens |
| Custom random c2 | 27.01 | 2 concurrent, aggregate |

## Quick Start

### Prerequisites

- 2x NVIDIA DGX Spark / GB10 (SM121) connected via QSFP
- Docker with NVIDIA container runtime
- Passwordless SSH between nodes (for Ray worker)
- ~130GB free disk per node (model weights)
- [HuggingFace token](https://huggingface.co/settings/tokens) (for model download)

### 1. Download the Model

On **both nodes**:

```bash
pip install huggingface_hub hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
  stepfun-ai/Step-3.7-Flash-NVFP4 \
  --local-dir /path/to/models/Step-3.7-Flash-NVFP4
```

### 2. Start the Server

**Node 1 (head):**
```bash
docker run -d --name step37-vllm \
  --gpus all --ipc=host --network=host \
  --entrypoint bash \
  -v /path/to/models:/models:ro \
  ghcr.io/r0b0tlab/vllm-step37-flash-nvfp4-sm121:latest \
  -lc "ray start --head --node-ip-address=<NODE1_IP> --port=6379 --block"
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

**Launch vLLM (on node 1):**
```bash
docker exec -d step37-vllm bash -lc '
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export GLOO_SOCKET_IFNAME=enp1s0f0np0
export TP_SOCKET_IFNAME=enp1s0f0np0
python3 -m vllm.entrypoints.openai.api_server \
  --model /models/Step-3.7-Flash-NVFP4 \
  --host 0.0.0.0 --port 30000 \
  --trust-remote-code --tensor-parallel-size 2 \
  --distributed-executor-backend ray \
  --disable-custom-all-reduce --quantization modelopt \
  --kv-cache-dtype fp8 --disable-cascade-attn \
  --gpu-memory-utilization 0.90 --max-model-len 65536 \
  --max-num-batched-tokens 32768 --max-num-seqs 4 \
  --reasoning-parser step3p5 --tool-call-parser step3p5 \
  --enable-auto-tool-choice --enforce-eager
'
```

### 3. Verify

```bash
# Health check
curl http://localhost:30000/health

# Text completion
curl http://localhost:30000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/models/Step-3.7-Flash-NVFP4","prompt":"Hello","max_tokens":32}'

# Tool calling
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/models/Step-3.7-Flash-NVFP4","messages":[{"role":"user","content":"What is 2+2?"}],"tools":[{"type":"function","function":{"name":"calculate","parameters":{"type":"object","properties":{"expr":{"type":"string"}}}}}]}'

# Backend verification (should show FLASHINFER_CUTLASS, not Marlin)
docker logs step37-vllm 2>&1 | grep "Using.*NvFp4"
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Node 1 (DGX Spark A)          Node 2 (DGX Spark B)    │
│  ┌───────────────────┐         ┌───────────────────┐    │
│  │  Ray Head          │◄──QSFP──►│  Ray Worker       │    │
│  │  vLLM API Server   │  200GbE  │  TP1 GPU Worker   │    │
│  │  TP0 GPU Worker    │         │                   │    │
│  │  GB10 SM121        │         │  GB10 SM121       │    │
│  └───────────────────┘         └───────────────────┘    │
│         :30000                                           │
└─────────────────────────────────────────────────────────┘
```

## Verified SM121 Backends

The vLLM runtime auto-selects native Blackwell CUTLASS backends:

| Component | Backend Selected | Fallback |
|-----------|-----------------|----------|
| NVFP4 MoE | FLASHINFER_CUTLASS | None (0 Marlin, 0 emulation) |
| NVFP4 Dense | VLLM_CUTLASS | None |
| Attention | FLASHINFER | None |
| VIT Encoder | FLASH_ATTN | None |
| All-Reduce | PYNCCL | Expected (SM121) |

Platform verification:
- `torch.cuda.get_device_capability()` → `(12, 1)` (SM121)
- `is_device_capability_family(120)` → `True`
- `vllm._C`, `vllm._C_stable_libtorch`, `vllm._moe_C` → all OK

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NCCL_SOCKET_IFNAME` | `enp1s0f0np0` | NCCL network interface (QSFP) |
| `GLOO_SOCKET_IFNAME` | `enp1s0f0np0` | GLOO network interface |
| `TP_SOCKET_IFNAME` | `enp1s0f0np0` | TP network interface |
| `MODEL` | `stepfun-ai/Step-3.7-Flash-NVFP4` | Model path or HF ID |
| `PORT` | `8000` | API server port |
| `GPU_UTIL` | `0.70` | GPU memory utilization |
| `MAX_MODEL_LEN` | `8192` | Maximum context length |
| `MAX_SEQS` | `4` | Maximum concurrent sequences |

### Key Launch Flags

| Flag | Why It Exists |
|------|--------------|
| `--tensor-parallel-size 2` | Dual GB10 TP=2 |
| `--distributed-executor-backend ray` | Ray for cross-node TP |
| `--disable-custom-all-reduce` | SM121 uses PYNCCL (NCCL_SYMM_MEM unavailable) |
| `--quantization modelopt` | NVFP4 modelopt_fp4 quantization |
| `--kv-cache-dtype fp8` | FP8 KV cache (model default) |
| `--disable-cascade-attn` | **REQUIRED** — Step 3.7 uses hybrid sliding-window attention |
| `--enforce-eager` | **REQUIRED** on first run — Triton JIT + CUDA graphs cause RPC timeout |
| `--gpu-memory-utilization 0.90` | Conservative for 129GB model on 121GB nodes |

### Context Length vs GPU Memory

| max_model_len | Status | GPU Memory | Notes |
|---------------|--------|------------|-------|
| 8,192 | Verified | ~60 GiB | Safe default |
| 32,768 | Verified | ~85 GiB | Recommended max |
| 65,536 | OOM | >121 GiB | Needs GPU_UTIL > 0.85 |

## Known Limitations

1. **CUDA graphs at 64K context**: SHM broadcast deadlock on cross-node TP=2. Use `--enforce-eager` for 64K+.
2. **Upstream recommends TP=4+EP**: Step 3.7's official recipe uses TP=4 with expert parallelism. TP=2 is our validated target for dual GB10.
3. **First-run Triton JIT**: The first inference request triggers Triton kernel compilation, which can cause RPC timeout with CUDA graphs. Always use `--enforce-eager` on first launch.
4. **PYNCCL all-reduce**: SM121 does not support NCCL_SYMM_MEM, QUICK_REDUCE, or FLASHINFER all-reduce. PYNCCL is the only available backend.
5. **FP8 KV scaling**: The checkpoint does not provide q_scale factors. vLLM uses k_scale as fallback. This produces a warning but works correctly.

## Evidence

All validation evidence is in the `evidence/` directory:

- `evidence/runtime/backend-selection.txt` — Backend selection logs
- `evidence/benchmarks/benchmark_matrix.json` — Custom benchmark results
- `evidence/benchmarks/llama-benchy-32k.md` — llama-benchy baseline
- `evidence/correctness/vision_smoke.json` — Vision smoke test
- `evidence/nsight/native_sm121_verdict.md` — SM121 verification report

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Server won't start | `docker exec step37-vllm ray status` — check both nodes connected |
| RPC timeout on startup | Ensure `--enforce-eager` is set |
| OOM during loading | Reduce `--gpu-memory-utilization` or `--max-model-len` |
| PYNCCL warning | Expected on SM121, not an error |
| Vision 500 error | Image URL must be accessible from inside the container |
| Ray worker can't connect | Check QSFP link: `ping <NODE2_IP>`, verify `NCCL_SOCKET_IFNAME` |
| SHM broadcast timeout | Normal during CUDA graph capture; wait or use `--enforce-eager` |

## Credits

- **StepFun AI** — Step 3.7 Flash model architecture and weights
- **HuggingFace** — Model hosting and hub
- **NVIDIA** — ModelOpt quantization, CUTLASS, FlashInfer, GB10 hardware
- **vLLM** — Serving framework
- **r0b0tlab** — SM121 validation, Docker packaging, benchmark publication

## Links

- Model: [stepfun-ai/Step-3.7-Flash-NVFP4](https://huggingface.co/stepfun-ai/Step-3.7-Flash-NVFP4)
- Docker: [ghcr.io/r0b0tlab/vllm-step37-flash-nvfp4-sm121](https://ghcr.io/r0b0tlab/vllm-step37-flash-nvfp4-sm121)
- Publisher: [r0b0tlab](https://github.com/r0b0tlab) · [@mr-r0b0t](https://x.com/mr_r0b0t)

---

*SM121-verified on dual NVIDIA DGX Spark · vLLM 0.1.dev16944 · 2026-06-04*
