# Third-party notices

## DeepSeek AI
- Model: DeepSeek-V4-Flash / DeepSeek-V4-Flash-DSpark
- Revision under test: `913f0657a874f76844e2e91cbe706dbcaceeb6d7`
- Model weights and tokenizer are **not** included in this repository
- Governed by DeepSeek’s published model license/terms for that revision

## vLLM
- https://github.com/vllm-project/vllm
- License: Apache License 2.0
- Upstream baseline under test: `568afb3a13806beb53bb2e6bd518269357b237c0`
- Package under test: `0.26.0+dspark.sm121.2` (r0b0tlab knownfix overlay on top of audited parent)

## FlashInfer
- https://github.com/flashinfer-ai/flashinfer
- Sparse MLA / related kernel stack used at runtime
- License/copyright: upstream FlashInfer authors

## NVIDIA
- DGX Spark / GB10 platform, CUDA 13.0, NCCL (`nvidia-nccl-cu13` 2.30.4)
- Trademarks and platform docs remain NVIDIA property
- No NVIDIA endorsement implied

## PyTorch
- `2.11.0+cu130`
- BSD-style license (upstream PyTorch)

## B12X MoE components
- FlashInfer B12X MXFP4 / W4A16 fused MoE path
- Copyright remains with respective upstream package authors
