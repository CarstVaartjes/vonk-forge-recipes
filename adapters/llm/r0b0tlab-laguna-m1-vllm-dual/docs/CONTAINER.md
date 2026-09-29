# Container runtime (canonical)

## Headline registry image

| Field | Value |
|-------|--------|
| **Registry image (this project)** | `ghcr.io/r0b0tlab/vllm-laguna-m1-nvfp4-sm121:cu130-sm121-arm64-dda4668b` |
| **Also tagged** | `ghcr.io/r0b0tlab/vllm-laguna-m1-nvfp4-sm121:latest` |
| **Digest pin** | `sha256:6e2dfa4c48ac73965890a864dc1510017f42981e743b3f2834b4b189ad7712a8` |
| **Docker name** | `laguna-m1-vllm` |
| **Launch** | `scripts/serve_tp2_cluster.sh` |
| **Publish** | `bash scripts/publish_ghcr_laguna_runtime.sh` (retag from SM121 vLLM build; does **not** overwrite `vllm-dsv4-flash-gb10`) |

LagunaForCausalLM, Poolside parsers, FlashInfer CUTLASS NVFP4 MoE. **Weights and Poolside parsers are serve-time flags** (`scripts/serve_tp2_cluster.sh`).

**GHCR package (org):** https://github.com/orgs/r0b0tlab/packages/container/package/vllm-laguna-m1-nvfp4-sm121  
**Setup (visibility + repo link):** `docs/GHCR_PACKAGE.md`

**Lineage:** Same SM121 vLLM layers as `ghcr.io/r0b0tlab/vllm-dsv4-flash-gb10` (DeepSeek-V4-Flash publication build, digest above). DSV4 repo keeps its own package; this is a **separate** GHCR package for Laguna M.1 users.

## Dual GB10

```bash
export MODEL_DIR=/path/to/Laguna-M.1-NVFP4
export IMAGE=ghcr.io/r0b0tlab/vllm-laguna-m1-nvfp4-sm121:cu130-sm121-arm64-dda4668b
export NAME=laguna-m1-vllm
export PORT=30100
export KV_CACHE_DTYPE=nvfp4
export MAX_MODEL_LEN=8192
bash scripts/serve_tp2_cluster.sh
```

## Single-node dev

`scripts/serve.sh` — same default `IMAGE`.