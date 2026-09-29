# Docker — Step 3.7 Flash NVFP4 SM121

## Published Image

```
ghcr.io/r0b0tlab/vllm-step37-flash-nvfp4-sm121:latest
ghcr.io/r0b0tlab/vllm-step37-flash-nvfp4-sm121:cu130-v0.1.dev16944-arm64-tp2
```

## What's Inside

| Component | Version |
|-----------|---------|
| vLLM | 0.1.dev16944+ge9c8946e7 |
| Ray | 2.55.1 |
| FlashInfer | 0.6.11.post2 |
| CUDA | 13.0 |
| Base | NVIDIA CUDA Ubuntu ARM64 |

## Quick Start (Dual GB10)

See [README.md](../README.md#quick-start) for the full walkthrough.

## Dockerfile

`Dockerfile.publish` extends the working vLLM dev image with Ray and
an entrypoint script. To build locally:

```bash
docker build -f docker/Dockerfile.publish -t step37-local .
```

## Entrypoint

The container's entrypoint starts vLLM with the verified SM121 flags.
Override behavior with environment variables:

```bash
docker run ... -e MAX_MODEL_LEN=16384 -e GPU_UTIL=0.75 ...
```

## Image Architecture

The image is `linux/arm64` only. GB10 / DGX Spark is an ARM64 (aarch64)
system. Do not attempt to run on x86_64.

## Network Requirements

The container uses `--network=host` and `--ipc=host`. It needs:

- Access to the QSFP interface for NCCL/GLOO/TP communication
- Port 30000 (or custom PORT) for the API server
- Port 6379 for Ray (head node)
- Port 8265 for Ray dashboard (head node)

## Model Mount

Mount the model directory from the host:

```bash
-v /path/to/models/Step-3.7-Flash-NVFP4:/models/Step-3.7-Flash-NVFP4:ro
```

The model must be downloaded on both nodes (Ray workers read weights locally).
