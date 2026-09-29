# AGENTS.md — Agents-A1 NVFP4 SM121 vLLM

These instructions apply to all work in this repository.

## Hard gate

Never publish, tag, upload, or describe a result as **SM121-native NVFP4** unless the evidence shows all of the following:

- Runtime device capability is `(12, 1)` on NVIDIA GB10 / SM121.
- `vllm._C`, `vllm._C_stable_libtorch`, and `vllm._moe_C` import successfully in the final container.
- `torch.ops._C.cutlass_scaled_mm_supports_fp4(121)` and `(120)` both return `True`.
- No packaged fallback artifacts are present: `_vllm_fa2_C`, `_vllm_fa3_C`, or `*marlin*.so`.
- Runtime logs select `FlashInferCutlassNvFp4LinearKernel` and `FLASHINFER_CUTLASS`/native NVFP4 paths.
- Runtime logs do not use MARLIN, EMULATION, or generic fallback kernels as the active backend.

## Canonical model

- Base model: `InternScience/Agents-A1`
- Quantized checkpoint: `r0b0tlab/Agents-A1-NVFP4` once uploaded, or a local mount at `/models/Agents-A1-NVFP4`.
- Quantization: NVIDIA ModelOpt NVFP4, group size 16, `modelopt` runtime quantization.
- First release policy: MLP/MoE-only NVFP4; attention, linear-attention/GDN, vision tower, embeddings, `lm_head`, and MTP-sensitive modules remain BF16.

## Canonical container commands

Build:

```bash
docker build -t ghcr.io/r0b0tlab/agents-a1-nvfp4-sm121-vllm:latest -f docker/Dockerfile .
```

Audit:

```bash
docker run --rm --gpus all ghcr.io/r0b0tlab/agents-a1-nvfp4-sm121-vllm:latest audit
```

Serve local model mount:

```bash
docker run --rm --gpus all --ipc=host --name agents-a1-nvfp4-vllm \
  -p 18080:8000 \
  -v /home/r0b0tdgx/work/agents-a1-nvfp4/Agents-A1-NVFP4:/models/Agents-A1-NVFP4:ro \
  ghcr.io/r0b0tlab/agents-a1-nvfp4-sm121-vllm:latest
```

Probe:

```bash
curl -s http://127.0.0.1:18080/v1/models
curl -s http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"agents-a1-nvfp4","messages":[{"role":"user","content":"Say READY and nothing else."}],"max_tokens":8}'
```

## Documentation discipline

- Keep every claim evidence-backed under `evidence/`.
- Do not hide the inherited vLLM version. If the runtime is a patched development build rather than an upstream release tag, say so directly.
- Credit InternScience for Agents-A1, NVIDIA for ModelOpt/NVFP4 tooling, vLLM/FlashInfer for serving kernels, and Hugging Face for distribution.
- Do not put secrets, tokens, or local-only credentials in README files, scripts, evidence, or commits.
