# VERDICT — Step 3.7 Flash NVFP4 SM121 Optimization

Status: **GO — SM121-verified native NVFP4 TP=2 dual-GB10**

## Date: 2026-06-04

## Summary

Step 3.7 Flash NVFP4 (198B total, 11B active, 288 experts, top-k=8) runs natively on dual GB10 SM121 with vLLM TP=2 over Ray. All backends are native Blackwell CUTLASS — zero Marlin, zero emulation.

## Evidence

### Backend Selection (Primary Gate)
Both TP ranks selected native Blackwell backends:
- **MoE**: FLASHINFER_CUTLASS on TP0 and TP1
- **Dense/Linear**: VLLM_CUTLASS on TP0 and TP1
- **Attention**: FLASHINFER on TP0 and TP1
- **VIT**: FLASH_ATTN
- **All-reduce**: PYNCCL (expected on SM121)
- **Zero Marlin. Zero emulation. Zero fallback.**

### Platform Capability
- SM Capability: (12, 1) — SM121 confirmed
- `is_device_capability_family(120)`: True — vLLM routes to native Blackwell path
- All native imports: vllm._C, vllm._C_stable_libtorch, vllm._moe_C OK
- FlashInfer 0.6.11.post2, CUTLASS available

### Nsight Evidence
- NVTX trace: 29,670 NCCL all-reduces (TP=2 communication active)
- CCCL:cub::DeviceRadixSort kernels (CUTLASS library)
- nsys 2025.3.2 limitation: cannot trace Ray worker GPU kernels across nodes
- Full trace at evidence/nsight/step37_tp2_full_trace.nsys-rep (6.5MB)

### Correctness
| Test | Result |
|------|--------|
| Text completion | "Paris" — correct factual answer |
| Tool calling | get_weather({"city":" Paris"}) — correct structured output |
| Vision (base64 PNG) | "The image is red" — correct color identification |

### Performance (dual GB10, TP=2, --enforce-eager, max_model_len=8192)
| Benchmark | Mean tok/s |
|-----------|-----------|
| tg128 c1 | 14.02 |
| random c1 (2048→1024) | 13.2 |
| random c2 (2048→1024) | 27.01 aggregate |

### Server Configuration
- Image: step37-stepfun-ray:arm64 (vLLM 0.1.dev16944)
- TP=2 via Ray over QSFP (192.168.100.10/11)
- modelopt_fp4 quantization, FP8 KV cache
- --enforce-eager (Triton JIT + CUDA graphs cause RPC timeout on first run)
- --disable-cascade-attn (required for hybrid sliding-window attention)
- --max-model-len 8192 (conservative initial)
- GPU memory utilization: 0.70

## Known Limitations
1. nsys cannot trace Ray worker GPU kernels across nodes (SM121 architecture limitation with distributed TP)
2. --enforce-eager required (no CUDA graphs on first run due to Triton JIT)
3. max_model_len=8192 (conservative; model supports 256K)
4. Upstream recommends TP=4+EP; TP=2 is our validated target
5. Vision test uses synthetic base64 PNG (no external URL access from server)

## Release Artifacts
- Docker image: step37-stepfun-ray:arm64
- Model: stepfun-ai/Step-3.7-Flash-NVFP4 (129GB, 14 shards)
- Evidence: evidence/ directory with runtime, benchmarks, nsight, correctness
- Scripts: scripts/ directory with launch, smoke, benchmark tools
