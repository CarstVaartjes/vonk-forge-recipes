# Family-aware qualification coverage matrix — 2026-09-24

Derived from exact catalog identities by `tools/build-family-aware-coverage`.

This artifact records catalog structure, byte-bound execution stacks, batch assignments and typed recovery references. It does not claim cache readiness, structural qualification, Controller deployment or physical Spark acceptance.

## Sources

| Input | Identity |
|---|---|
| Plan | `docs/recipe-qualification-plan-2026-09-24.md` |
| Catalog | CarstVaartjes/vonk-forge-recipes — 85 recipe documents |
| Catalog release | v1.0.16 at `7b23f1a1569e16f8f2728ddd6846cf4f7c58f0f7` |
| Catalog source commit | `a0ffd8735270c35b0c99b5b04cfe8956ce8c351d` |
| Qualification authority | `nl-family-aware-20260924` |
| Scope | 81 recipes within 2 Sparks; 4 larger topologies listed for audit only |

## Derivation

Every recipe and Model document is validated through the canonical contract; its content digest must match both the producer index and the authored tree document.
`runtime_stack_sha256` binds the full runtime/build contract and the bytes of every regular file in its build context, Dockerfile and patches at the accepted catalog source commit.
Specialized forks stay distinct from generic engine evidence. Every recipe retains its own inference and declared assertion checks.
Paired single-node batches use two distinct lanes under one whole-fleet profile/apply and preserve independent smoke and recovery results. Dual-node rows have exclusive batches.
Recovery references bind explicit members and exact stack/topology identities. A physical receipt and member-use validation remain required before sharing recovery evidence.

## Coverage groups

| Coverage group | In-scope rows | Specialized forks | Dual-node rows |
|---|---:|---:|---:|
| `vllm` | 21 | 0 | 2 |
| `sglang` | 4 | 3 | 2 |
| `specialized-fork-engines` | 13 | 12 | 5 |
| `image-video-workflows` | 32 | 2 | 0 |
| `three-d-pipelines` | 8 | 0 | 0 |
| `audio-multimodal-pipelines` | 3 | 2 | 0 |

## Stack and family matrix

One row per catalog recipe. In-scope rows show their authority sequence and batch lane; wider topologies remain visible for audit. `stack` is the short prefix of the full source-byte-bound runtime identity.

| Seq | Recipe | Nodes | Group | Engine | Base image | Build context | Families | Quantization | Artifacts | Rep | Batch/lane | Authored source matches published |
|---:|---|---:|---|---|---|---|---|---|---:|---|---|---|
| 1 | `vonk-forge/step1x-3d-geometry-pytorch-single` | 1 | `three-d-pipelines` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@0e1392f431f8` | `adapters/three-d/step1x-3d` | dinov2-with-registers, step1x-3d | none | 19.4 GiB | yes | batch-001/L1 | yes |
| 2 | `vonk-forge/step1x-3d-label-geometry-pytorch-single` | 1 | `three-d-pipelines` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@0e1392f431f8` | `adapters/three-d/step1x-3d` | clip-vit-large-patch14, dinov2-with-registers, step1x-3d | none | 25.8 GiB | yes | batch-001/L2 | yes |
| 3 | `vonk-forge/flux-2-klein-4b-nvfp4-comfyui-single` | 1 | `image-video-workflows` | `comfyui` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/media/comfyui-core` | flux-2 | nvfp4 | 10.1 GiB | yes | batch-002/L1 | yes |
| 4 | `vonk-forge/skintokens-pytorch-single` | 1 | `three-d-pipelines` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@0e1392f431f8` | `adapters/three-d/skintokens` | skintokens | none | 1.5 GiB | yes | batch-002/L2 | yes |
| 5 | `vonk-forge/trellis-2-4b-pytorch-single` | 1 | `three-d-pipelines` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@0e1392f431f8` | `adapters/three-d/trellis2-native` | trellis | none | 16.4 GiB | yes | batch-003/L1 | yes |
| 6 | `vonk-forge/triposg-pytorch-single` | 1 | `three-d-pipelines` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@0e1392f431f8` | `adapters/three-d/triposg` | bria-rmbg, triposg | none | 7.6 GiB | yes | batch-003/L2 | yes |
| 7 | `vonk-forge/flux-2-klein-4b-comfyui-single` | 1 | `image-video-workflows` | `comfyui` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/media/comfyui-core` | flux-2 | none | 15.0 GiB | yes | batch-004/L1 | yes |
| 8 | `vonk-forge/pixal3d-pytorch-single` | 1 | `three-d-pipelines` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@0e1392f431f8` | `adapters/three-d/trellis2-native` | pixal3d | none | 23.5 GiB | yes | batch-004/L2 | yes |
| 9 | `vonk-forge/ornith-1-5-35b-a3b-nvfp4-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@1c8e60a0841b` | `adapters/llm/vllm-openai` | ornith-1-5 | modelopt-mixed-w4a16-nvfp4-fp8 | 21.8 GiB | yes | batch-005/L1 | yes |
| 10 | `vonk-forge/lfm2-5-vl-3b-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@1c8e60a0841b` | `adapters/liquidai/lfm25-vl-vllm` | lfm2-5 | none | 5.8 GiB | yes | batch-005/L2 | yes |
| 11 | `vonk-forge/lfm2-5-vl-3b-vllm028-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@41b54fb42c66` | `adapters/liquidai/lfm25-vl-vllm-028` | lfm2-5 | none | 5.8 GiB | yes | batch-006/L1 | yes |
| 12 | `vonk-forge/qwen3-5-9b-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@41b54fb42c66` | `adapters/llm/qwen35-vllm` | qwen | none | 18.0 GiB | yes | batch-006/L2 | yes |
| 13 | `vonk-forge/qwen3-6-35b-a3b-nvfp4-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@41b54fb42c66` | `adapters/nvidia/qwen36-35b-vllm` | nvidia-qwen | nvfp4 | 21.9 GiB | yes | batch-007/L1 | yes |
| 14 | `vonk-forge/qwen3-8-27b-fp8-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@41b54fb42c66` | `adapters/llm/vllm-openai-028` | qwen | fp8 | 28.8 GiB | yes | batch-007/L2 | yes |
| 15 | `vonk-forge/laguna-xs-2-1-nvfp4-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@1c8e60a0841b` | `adapters/llm/laguna-vllm` | poolside | nvfp4-fp8-kv-cache | 20.1 GiB | yes | batch-008/L1 | yes |
| 16 | `vonk-forge/wan-2-2-ti2v-5b-comfyui-single` | 1 | `image-video-workflows` | `comfyui` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/media/comfyui-core` | wan-2-2 | fp8 | 16.9 GiB | yes | batch-008/L2 | yes |
| 17 | `vonk-forge/moss-vl-realtime-11b-pytorch-single` | 1 | `audio-multimodal-pipelines` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/video/moss-vl-realtime` | moss-vl | none | 21.1 GiB | yes | batch-009/L1 | yes |
| 18 | `vonk-forge/ltx-2-19b-dev-bf16-diffusers-single` | 1 | `image-video-workflows` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@a52783d8d73a` | `adapters/video/ltx23-sync-native-disk` | ltx | none | 93.8 GiB | vonk-forge/ltx-2-3-22b-distilled-1-1-diffusers-single | batch-009/L2 | no |
| 19 | `vonk-forge/ltx-2-19b-dev-fp4-pytorch-single` | 1 | `image-video-workflows` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@a52783d8d73a` | `adapters/video/ltx2-pytorch` | ltx | fp4 | 72.1 GiB | yes | batch-010/L1 | yes |
| 20 | `vonk-forge/nemotron-3-5-lightning-30b-a3b-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@1c8e60a0841b` | `adapters/nvidia/nemotron` | nemotron | nvfp4 | 20.1 GiB | yes | batch-010/L2 | yes |
| 21 | `vonk-forge/nemotron-3-nano-30b-a3b-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@871fc9b75a97` | `adapters/nvidia/nemotron-vllm-0-20-0` | nemotron | nvfp4-fp8-kv-selective-bf16 | 18.0 GiB | yes | batch-011/L1 | yes |
| 22 | `vonk-forge/nemotron-3-nano-omni-30b-a3b-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@871fc9b75a97` | `adapters/nvidia/nemotron-vllm-0-20-0` | nemotron | nvfp4 | 20.9 GiB | yes | batch-011/L2 | yes |
| 23 | `vonk-forge/qwen3-8-27b-nvfp4-dspark-sglang-single` | 1 | `sglang` | `sglang` | `docker.io/lmsysorg/sglang@3c0abdf41ef2` | `adapters/qwen/qwen38-27b-dspark-single` | qwen3-8-radixark | none, nvfp4-w4a4-bf16-lmhead | 25.6 GiB | yes | batch-012/L1 | yes |
| 24 | `vonk-forge/ltx-2-3-22b-distilled-1-1-diffusers-single` | 1 | `image-video-workflows` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@a52783d8d73a` | `adapters/video/ltx23-sync-native-disk` | ltx | none | 89.3 GiB | yes | batch-012/L2 | no |
| 25 | `vonk-forge/qwen3-6-27b-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@41b54fb42c66` | `adapters/llm/vllm-openai-028` | qwen | none | 51.8 GiB | yes | batch-013/L1 | yes |
| 26 | `vonk-forge/ltx-2-19b-distilled-fp8-diffusers-single` | 1 | `image-video-workflows` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@a52783d8d73a` | `adapters/video/ltx2-sync-native` | ltx | fp8-scaled-mm, none | 71.6 GiB | yes | batch-013/L2 | no |
| 27 | `vonk-forge/step1x-3d-texture-pytorch-single` | 1 | `three-d-pipelines` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@0e1392f431f8` | `adapters/three-d/step1x-3d` | sdxl-vae, stable-diffusion-xl, step1x-3d | none | 91.2 GiB | yes | batch-014/L1 | yes |
| 28 | `vonk-forge/gemma-4-26b-a4b-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@1c8e60a0841b` | `adapters/llm/vllm-openai` | gemma | none | 48.1 GiB | yes | batch-014/L2 | yes |
| 29 | `vonk-forge/gemma-4-26b-a4b-vllm028-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@41b54fb42c66` | `adapters/google/gemma4-vllm-028` | gemma | none | 48.1 GiB | yes | batch-015/L1 | yes |
| 30 | `vonk-forge/qwen-image-2512-fp8-lightning-comfyui-single` | 1 | `image-video-workflows` | `comfyui` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/media/comfyui-qwen-image-2512-fp8-lightning` | qwen-image | fp8, none | 29.6 GiB | yes | batch-015/L2 | yes |
| 31 | `vonk-forge/qwen-image-edit-2511-fp8mixed-comfyui-single` | 1 | `image-video-workflows` | `comfyui` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/media/comfyui-qwen-image-edit-2511-fp8mixed` | qwen-image-edit | fp8mixed | 28.1 GiB | yes | batch-016/L1 | yes |
| 32 | `vonk-forge/qwen-image-edit-2511-int8-convrot-comfyui-single` | 1 | `image-video-workflows` | `comfyui` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/media/comfyui-qwen-image-edit-2511-int8-convrot` | qwen-image-edit | int8_tensorwise_convrot | 28.1 GiB | yes | batch-016/L2 | yes |
| 33 | `vonk-forge/ui-mate-27b-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@41b54fb42c66` | `adapters/llm/ui-mate-vllm` | ui-mate | none | 51.0 GiB | yes | batch-017/L1 | yes |
| 34 | `vonk-forge/muse-glimmer-30b-bf16-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@ae1de325b8ea` | `adapters/llm/muse-glimmer-vllm` | muse-glimmer | none | 55.5 GiB | yes | batch-017/L2 | yes |
| 35 | `vonk-forge/qwen3-8-27b-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@41b54fb42c66` | `adapters/llm/vllm-openai-028` | qwen | none | 51.8 GiB | yes | batch-018/L1 | yes |
| 36 | `vonk-forge/ltx-2-5-22b-distilled-fp8-cast-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/video/ltx25-diffusers-fp8-canary` | ltx | none | 65.3 GiB | yes | batch-018/L2 | yes |
| 37 | `vonk-forge/mova-360p-diffusers-single` | 1 | `image-video-workflows` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@a52783d8d73a` | `adapters/video/mova-pytorch` | mova | none | 72.4 GiB | yes | batch-019/L1 | yes |
| 38 | `vonk-forge/deepseek-v4-flash-0731-ds4-dspark-latency-single` | 1 | `specialized-fork-engines` | `ds4` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/deepseek/ds4` | deepseek-flash | iq2_xxs-q2_k-mixed | 86.3 GiB | yes | batch-019/L2 | yes |
| 39 | `vonk-forge/deepseek-v4-flash-0731-ds4-single` | 1 | `specialized-fork-engines` | `ds4` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/deepseek/ds4` | deepseek-flash | iq2_xxs-q2_k-mixed | 80.8 GiB | yes | batch-020/L1 | yes |
| 40 | `vonk-forge/ltx-2-19b-distilled-diffusers-single` | 1 | `image-video-workflows` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@a52783d8d73a` | `adapters/video/ltx2-sync-native` | ltx | none | 86.7 GiB | yes | batch-020/L2 | no |
| 41 | `vonk-forge/laguna-s-2-1-nvfp4-vllm-low-memory-canary-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@1c8e60a0841b` | `adapters/llm/laguna-s-vllm` | poolside | nvfp4 | 92.9 GiB | yes | batch-021/L1 | yes |
| 42 | `vonk-forge/mova-720p-diffusers-single` | 1 | `image-video-workflows` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@a52783d8d73a` | `adapters/video/mova-pytorch` | mova | none | 72.4 GiB | vonk-forge/mova-360p-diffusers-single | batch-021/L2 | yes |
| 43 | `vonk-forge/nemotron-3-5-lightning-dspark-lowmem-canary-single` | 1 | `specialized-fork-engines` | `vllm` | `docker.io/vllm/vllm-openai@1c8e60a0841b` | `adapters/nvidia/nemotron` | nemotron | nvfp4, w4a16-nvfp4 | 21.4 GiB | yes | batch-022/L1 | yes |
| 44 | `vonk-forge/nemotron-3-super-120b-a12b-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@1c8e60a0841b` | `adapters/llm/vllm-openai` | nemotron | none, nvfp4 | 80.3 GiB | yes | batch-022/L2 | yes |
| 45 | `vonk-forge/nvidia-qwen-image-flash-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/image/nvidia-qwen-image-flash-diffusers` | qwen-image | none | 53.7 GiB | yes | batch-023/L1 | yes |
| 46 | `vonk-forge/qwen-image-2512-comfyui-single` | 1 | `image-video-workflows` | `comfyui` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/media/comfyui-qwen-image-2512-bf16` | qwen-image | fp8 | 47.0 GiB | yes | batch-023/L2 | yes |
| 47 | `vonk-forge/qwen-image-2512-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/image/qwen-image-2512-diffusers` | qwen-image | bf16 | 53.7 GiB | yes | batch-024/L1 | yes |
| 48 | `vonk-forge/qwen-image-2512-lightning-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/image/qwen-image-lightning-diffusers` | qwen-image | bf16, none | 54.5 GiB | yes | batch-024/L2 | yes |
| 49 | `vonk-forge/qwen-image-edit-2511-comfyui-single` | 1 | `image-video-workflows` | `comfyui` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/media/comfyui-core` | qwen-image-edit | bf16 | 47.0 GiB | yes | batch-025/L1 | yes |
| 50 | `vonk-forge/qwen-image-edit-2511-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/image/qwen-image-edit-2511-diffusers` | qwen-image-edit | bf16 | 53.8 GiB | yes | batch-025/L2 | yes |
| 51 | `vonk-forge/qwen-image-edit-2511-lightning-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/image/qwen-image-lightning-diffusers` | qwen-image, qwen-image-edit | bf16, none | 54.5 GiB | yes | batch-026/L1 | yes |
| 52 | `vonk-forge/qwen-image-layered-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/image/qwen-image-layered-diffusers` | qwen-image | bf16 | 53.8 GiB | yes | batch-026/L2 | yes |
| 53 | `vonk-forge/wan-2-2-i2v-14b-comfyui-single` | 1 | `image-video-workflows` | `comfyui` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/media/comfyui-core` | wan-2-2 | fp8 | 33.1 GiB | yes | batch-027/L1 | yes |
| 54 | `vonk-forge/wan-2-2-t2v-14b-comfyui-single` | 1 | `image-video-workflows` | `comfyui` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/media/comfyui-core` | wan-2-2 | fp8 | 33.1 GiB | yes | batch-027/L2 | yes |
| 55 | `vonk-forge/wan-dancer-14b-disk-offload-pytorch-single` | 1 | `image-video-workflows` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/video/wan-dancer-diffsynth-disk` | wan-dancer | none | 79.8 GiB | yes | batch-028/L1 | yes |
| 56 | `vonk-forge/deepseek-v4-flash-0731-sparkinfer-target-only-canary-single` | 1 | `specialized-fork-engines` | `vllm` | `ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@2e077489a83a` | `adapters/deepseek/sparkinfer-target-only-single` | deepseek-flash | reap-k216-exl3-trellis-3_0bpw | 99.5 GiB | yes | batch-028/L2 | yes |
| 57 | `vonk-forge/ling-3-0-flash-dspark-sglang-single` | 1 | `sglang` | `sglang` | `docker.io/lmsysorg/sglang@fc960102f1d2` | `adapters/ling/flash-dspark-single` | ling-3-0 | int4-group32, none | 74.3 GiB | yes | batch-029/L1 | yes |
| 58 | `vonk-forge/deepseek-v4-flash-0731-mia-sparkinfer-single` | 1 | `specialized-fork-engines` | `vllm` | `ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@2e077489a83a` | `adapters/deepseek/mia-sparkinfer-single` | deepseek-flash | reap-k216-exl3-trellis-3_0bpw | 99.5 GiB | yes | batch-029/L2 | yes |
| 59 | `vonk-forge/hunyuan-video-foley-xl-pytorch-single` | 1 | `audio-multimodal-pipelines` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/audio/hunyuan-video-foley-native` | clap, hunyuan-video-foley, siglip2 | none | 19.5 GiB | yes | batch-030/L1 | yes |
| 60 | `vonk-forge/hunyuan-video-foley-xxl-pytorch-single` | 1 | `audio-multimodal-pipelines` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/audio/hunyuan-video-foley-native` | clap, hunyuan-video-foley, siglip2 | none | 19.5 GiB | yes | batch-030/L2 | yes |
| 61 | `vonk-forge/hunyuanocr-1-5-vllm-dflash-single` | 1 | `specialized-fork-engines` | `pytorch-pipeline` | `docker.io/vllm/vllm-openai@1c8e60a0841b` | `adapters/ocr/hunyuanocr-1-5-vllm-dflash` | hunyuanocr | none | 2.4 GiB | yes | batch-031/L1 | yes |
| 62 | `vonk-forge/hunyuan3d-omni-pytorch-single` | 1 | `three-d-pipelines` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@a52783d8d73a` | `adapters/three-d/hunyuan3d-omni` | dinov2, hunyuan3d | none | 26.2 GiB | yes | batch-031/L2 | yes |
| 63 | `vonk-forge/hunyuan-video-15-distilled-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/video/hunyuan-video-15-native` | hunyuan-video | none | 49.7 GiB | yes | batch-032/L1 | yes |
| 64 | `vonk-forge/hunyuan-video-15-i2v-step-distilled-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/video/hunyuan-video-15-native` | hunyuan-video | none | 32.3 GiB | yes | batch-032/L2 | yes |
| 65 | `vonk-forge/hunyuan-video-15-t2v-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/video/hunyuan-video-15-native` | hunyuan-video | none | 49.7 GiB | yes | batch-033/L1 | yes |
| 66 | `vonk-forge/minimax-h3-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/video/minimax-h3-modular-diffusers` | minimax-h3 | none | 464.2 GiB | yes | batch-033/L2 | yes |
| 67 | `vonk-forge/minimax-h3-fl2va-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/video/minimax-h3-fl2va-modular-diffusers` | minimax-h3 | none | 134.2 GiB | yes | batch-034/L1 | yes |
| 68 | `vonk-forge/ltx-2-5-22b-distilled-bf16-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/video/ltx25-diffusers` | ltx | none | 65.3 GiB | yes | batch-034/L2 | yes |
| 69 | `vonk-forge/nemotron-3-5-lightning-30b-a3b-vllm-dspark-latency-single` | 1 | `specialized-fork-engines` | `vllm` | `docker.io/vllm/vllm-openai@1c8e60a0841b` | `adapters/nvidia/nemotron` | nemotron | nvfp4, w4a16-nvfp4 | 21.4 GiB | yes | batch-035/L1 | yes |
| 70 | `vonk-forge/wan-dancer-14b-pytorch-single` | 1 | `image-video-workflows` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/video/wan-dancer-native` | wan-dancer | none | 79.8 GiB | yes | batch-035/L2 | yes |
| 71 | `vonk-forge/deepseek-v4-flash-0731-sparkinfer-single` | 1 | `specialized-fork-engines` | `vllm` | `ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@2e077489a83a` | `adapters/deepseek/sparkinfer-single` | deepseek-flash | reap-k216-exl3-trellis-3_0bpw | 99.5 GiB | yes | batch-036/L1 | yes |
| 72 | `vonk-forge/laguna-s-2-1-nvfp4-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@1c8e60a0841b` | `adapters/llm/laguna-s-vllm` | poolside | nvfp4 | 92.9 GiB | yes | batch-036/L2 | yes |
| 73 | `vonk-forge/deepseek-v4-flash-0731-mia-dual` | 2 | `specialized-fork-engines` | `vllm` | `ghcr.io/anemll/dspark-vllm-gx10@a83948492cf1` | `adapters/deepseek/mia-vllm` | deepseek-flash | moe-experts-fp4-remaining-fp8 | 155.4 GiB | yes | batch-037 (exclusive dual) | yes |
| 74 | `vonk-forge/deepseek-v4-flash-vision-exp-mia-dual` | 2 | `specialized-fork-engines` | `vllm` | `ghcr.io/anemll/dspark-vllm-gx10@a83948492cf1` | `adapters/deepseek/mia-vllm-vision` | deepseek-flash | moe-experts-fp4-remaining-fp8-vision-bf16 | 156.3 GiB | yes | batch-038 (exclusive dual) | yes |
| 75 | `vonk-forge/glm-5-3-flash-exl3-dflash2-vllm-dual` | 2 | `specialized-fork-engines` | `vllm` | `docker.io/vllm/vllm-openai@905c02933be6` | `adapters/glm/mia-exl3-dflash2-dual` | glm-5-3-dflash, glm-5-3-mia | exl3-tr3-4bpw-plus-bf16-dflash2, none | 168.0 GiB | yes | batch-039 (exclusive dual) | yes |
| 76 | `vonk-forge/inkling-small-nvfp4-sglang-dual` | 2 | `sglang` | `sglang` | `docker.io/lmsysorg/sglang@bbedab8cbf2d` | `adapters/inkling/small-dual` | inkling | nvfp4 | 159.0 GiB | yes | batch-040 (exclusive dual) | yes |
| 77 | `vonk-forge/qwen3-8-flash-next-nvfp4-sglang-dual` | 2 | `sglang` | `sglang` | `docker.io/lmsysorg/sglang@14ed58251858` | `adapters/qwen/flash-next-sglang-dual` | qwen3-8-radixark | nvfp4-w4a4-routed-experts | 126.0 GiB | yes | batch-041 (exclusive dual) | yes |
| 78 | `vonk-forge/qwen3-8-flash-next-nvfp4-vllm-dual` | 2 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@3b0e188ffceb` | `adapters/qwen/flash-next-vllm-dual` | qwen3-8-radixark | nvfp4-w4a4-routed-experts | 126.0 GiB | yes | batch-042 (exclusive dual) | yes |
| 79 | `vonk-forge/glm-5-3-flash-nvfp4-vllm-dual` | 2 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@905c02933be6` | `adapters/glm/glm53-sm121` | glm-5-3-libertai | nvfp4-weight-only-routed-experts | 181.3 GiB | yes | batch-043 (exclusive dual) | yes |
| 80 | `vonk-forge/glm-5-3-flash-nvfp4-ablit-l15-43-dflash2-vllm-dual` | 2 | `specialized-fork-engines` | `vllm` | `ghcr.io/tonyd2wild/vllm-glm53-flash@4def0ef644cb` | `adapters/glm/tonyd2wild-dflash2-dual` | glm-5-3-dflash, glm-5-3-redhat-compressed | compressed-tensors-nvfp4-w4a4, none | 188.7 GiB | yes | batch-044 (exclusive dual) | yes |
| 81 | `vonk-forge/glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual` | 2 | `specialized-fork-engines` | `vllm` | `ghcr.io/drowzeys/keys-vllm-glm53-flash-nvfp4-ablit@f722ec19d826` | `adapters/glm/drowzeys-glm53-1m` | glm-5-3-libertai | nvfp4-weight-only-routed-experts | 181.3 GiB | yes | batch-045 (exclusive dual) | yes |
| — | `vonk-forge/glm-5-2-aqlm-vllm-triple` | 3 | `specialized-fork-engines` | `vllm` | `ghcr.io/miaai-lab/glm-5.2-nvfp4-triple-dgx-sparks@f8f350d46b33` | `adapters/glm/mia-triple` | glm-5-2 | nvfp4_aqlm_hybrid | 272.5 GiB | — | — | yes |
| — | `vonk-forge/glm-5-2-quanttrio-vllm-four` | 4 | `specialized-fork-engines` | `vllm` | `ghcr.io/drowzeys/vllm-node-tf5-glm52-b12x@e006935eb4f8` | `adapters/glm/eugr-four` | glm-5-2 | compressed_tensors_w4a16_w8a16 | 377.7 GiB | — | — | yes |
| — | `vonk-forge/glm-5-3-flash-nvfp4-vllm-four` | 4 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@905c02933be6` | `adapters/glm/tonyd2wild-glm53-tp4-current` | glm-5-3-dflash, glm-5-3-libertai | none, nvfp4-weight-only-routed-experts | 183.5 GiB | — | — | yes |
| — | `vonk-forge/inkling-975b-a41b-nvfp4-sglang-eight` | 8 | `sglang` | `sglang` | `docker.io/lmsysorg/sglang@c60f221f8f42` | `adapters/llm/inkling-sglang-eight` | inkling | nvfp4 | 551.4 GiB | — | — | yes |

## Representative selection

Each risk group combines exact runtime stack, model family, quantization and node count. Its representative has the smallest declared artifact footprint; every recipe still needs its own inference and assertions.

| Risk group | Group | Representative | Members | Artifacts |
|---|---|---|---:|---:|
| `04695e134095279e` | `specialized-fork-engines` | `vonk-forge/deepseek-v4-flash-0731-sparkinfer-target-only-canary-single` | 1 | 99.5 GiB |
| `14fa8f71592be856` | `sglang` | `vonk-forge/ling-3-0-flash-dspark-sglang-single` | 1 | 74.3 GiB |
| `20ce701a369ffdf8` | `image-video-workflows` | `vonk-forge/hunyuan-video-15-i2v-step-distilled-diffusers-single` | 1 | 32.3 GiB |
| `20f9964b270e5a89` | `three-d-pipelines` | `vonk-forge/trellis-2-4b-pytorch-single` | 1 | 16.4 GiB |
| `26d15b6035efcbf6` | `image-video-workflows` | `vonk-forge/ltx-2-5-22b-distilled-fp8-cast-diffusers-single` | 1 | 65.3 GiB |
| `28c3b7614bd9755b` | `vllm` | `vonk-forge/laguna-s-2-1-nvfp4-vllm-low-memory-canary-single` | 1 | 92.9 GiB |
| `29c55b8037667c48` | `sglang` | `vonk-forge/inkling-small-nvfp4-sglang-dual` | 1 | 159.0 GiB |
| `2e07b746b2d12d66` | `image-video-workflows` | `vonk-forge/qwen-image-edit-2511-fp8mixed-comfyui-single` | 1 | 28.1 GiB |
| `30567cd5b0cbe03d` | `specialized-fork-engines` | `vonk-forge/glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual` | 1 | 181.3 GiB |
| `314cd475cbb678a8` | `image-video-workflows` | `vonk-forge/ltx-2-19b-distilled-fp8-diffusers-single` | 1 | 71.6 GiB |
| `34127fcc22b3e304` | `image-video-workflows` | `vonk-forge/flux-2-klein-4b-comfyui-single` | 1 | 15.0 GiB |
| `3681b04dad19e271` | `audio-multimodal-pipelines` | `vonk-forge/hunyuan-video-foley-xxl-pytorch-single` | 1 | 19.5 GiB |
| `397e48ec81253c57` | `image-video-workflows` | `vonk-forge/nvidia-qwen-image-flash-diffusers-single` | 1 | 53.7 GiB |
| `3ae0f31f76e5ad39` | `image-video-workflows` | `vonk-forge/wan-2-2-t2v-14b-comfyui-single` | 1 | 33.1 GiB |
| `3d2c793b016e0194` | `image-video-workflows` | `vonk-forge/qwen-image-2512-fp8-lightning-comfyui-single` | 1 | 29.6 GiB |
| `3e0d8775b97fe733` | `image-video-workflows` | `vonk-forge/hunyuan-video-15-t2v-diffusers-single` | 1 | 49.7 GiB |
| `4134db56b51bb733` | `specialized-fork-engines` | `vonk-forge/deepseek-v4-flash-vision-exp-mia-dual` | 1 | 156.3 GiB |
| `4424de3538b49dae` | `image-video-workflows` | `vonk-forge/minimax-h3-fl2va-diffusers-single` | 1 | 134.2 GiB |
| `44a36628ffba8a4c` | `image-video-workflows` | `vonk-forge/qwen-image-2512-diffusers-single` | 1 | 53.7 GiB |
| `45795e60b5bfd65a` | `image-video-workflows` | `vonk-forge/qwen-image-2512-comfyui-single` | 1 | 47.0 GiB |
| `4ba9dd2e0a3ad13d` | `image-video-workflows` | `vonk-forge/ltx-2-3-22b-distilled-1-1-diffusers-single` | 2 | 89.3 GiB |
| `4cb8319ea6b94858` | `vllm` | `vonk-forge/lfm2-5-vl-3b-vllm-single` | 1 | 5.8 GiB |
| `542c3de1f5e55387` | `vllm` | `vonk-forge/laguna-s-2-1-nvfp4-vllm-single` | 1 | 92.9 GiB |
| `5576df725249604c` | `vllm` | `vonk-forge/glm-5-3-flash-nvfp4-vllm-dual` | 1 | 181.3 GiB |
| `577750fdc9296e0f` | `image-video-workflows` | `vonk-forge/mova-360p-diffusers-single` | 2 | 72.4 GiB |
| `58daf673d2932135` | `image-video-workflows` | `vonk-forge/qwen-image-edit-2511-int8-convrot-comfyui-single` | 1 | 28.1 GiB |
| `5b33f4dbba43faef` | `vllm` | `vonk-forge/nemotron-3-nano-omni-30b-a3b-vllm-single` | 1 | 20.9 GiB |
| `5c9dc83b6586f4c9` | `specialized-fork-engines` | `vonk-forge/glm-5-3-flash-exl3-dflash2-vllm-dual` | 1 | 168.0 GiB |
| `5f8e69fac5786b3a` | `three-d-pipelines` | `vonk-forge/triposg-pytorch-single` | 1 | 7.6 GiB |
| `65614268fe5125cf` | `specialized-fork-engines` | `vonk-forge/nemotron-3-5-lightning-30b-a3b-vllm-dspark-latency-single` | 1 | 21.4 GiB |
| `6703a6793f2e27a5` | `three-d-pipelines` | `vonk-forge/step1x-3d-texture-pytorch-single` | 1 | 91.2 GiB |
| `6706e7f079306f04` | `image-video-workflows` | `vonk-forge/ltx-2-19b-distilled-diffusers-single` | 1 | 86.7 GiB |
| `6b1f1cfb9db555df` | `three-d-pipelines` | `vonk-forge/hunyuan3d-omni-pytorch-single` | 1 | 26.2 GiB |
| `713a0cc8251b2416` | `vllm` | `vonk-forge/muse-glimmer-30b-bf16-vllm-single` | 1 | 55.5 GiB |
| `715d96cf1ed7b18e` | `vllm` | `vonk-forge/ui-mate-27b-vllm-single` | 1 | 51.0 GiB |
| `717b1eebdd18b012` | `image-video-workflows` | `vonk-forge/qwen-image-edit-2511-diffusers-single` | 1 | 53.8 GiB |
| `7bc249e34dbd1eee` | `image-video-workflows` | `vonk-forge/wan-dancer-14b-pytorch-single` | 1 | 79.8 GiB |
| `7f6971c0d85ab631` | `specialized-fork-engines` | `vonk-forge/deepseek-v4-flash-0731-mia-dual` | 1 | 155.4 GiB |
| `7fc67bf7ded957d8` | `sglang` | `vonk-forge/qwen3-8-flash-next-nvfp4-sglang-dual` | 1 | 126.0 GiB |
| `8028122b3d253439` | `image-video-workflows` | `vonk-forge/hunyuan-video-15-distilled-diffusers-single` | 1 | 49.7 GiB |
| `82d3da6ac38b7f14` | `vllm` | `vonk-forge/qwen3-6-27b-vllm-single` | 1 | 51.8 GiB |
| `877792057d387be9` | `vllm` | `vonk-forge/lfm2-5-vl-3b-vllm028-single` | 1 | 5.8 GiB |
| `951911a0fd6049db` | `vllm` | `vonk-forge/qwen3-5-9b-vllm-single` | 1 | 18.0 GiB |
| `a3db633d8a3e1e7e` | `image-video-workflows` | `vonk-forge/ltx-2-19b-dev-fp4-pytorch-single` | 1 | 72.1 GiB |
| `a50eab82d20c6d14` | `vllm` | `vonk-forge/ornith-1-5-35b-a3b-nvfp4-vllm-single` | 1 | 21.8 GiB |
| `a5d0549e4b56292d` | `specialized-fork-engines` | `vonk-forge/hunyuanocr-1-5-vllm-dflash-single` | 1 | 2.4 GiB |
| `a6dbf285f7f5a52f` | `image-video-workflows` | `vonk-forge/minimax-h3-diffusers-single` | 1 | 464.2 GiB |
| `a7aa238b388e157b` | `three-d-pipelines` | `vonk-forge/step1x-3d-label-geometry-pytorch-single` | 1 | 25.8 GiB |
| `a8372413acfb8749` | `vllm` | `vonk-forge/laguna-xs-2-1-nvfp4-vllm-single` | 1 | 20.1 GiB |
| `a85d3d684dfee401` | `specialized-fork-engines` | `vonk-forge/glm-5-3-flash-nvfp4-ablit-l15-43-dflash2-vllm-dual` | 1 | 188.7 GiB |
| `a9b7b77fdfc4769f` | `three-d-pipelines` | `vonk-forge/pixal3d-pytorch-single` | 1 | 23.5 GiB |
| `adfcf0f563a7e738` | `vllm` | `vonk-forge/gemma-4-26b-a4b-vllm028-single` | 1 | 48.1 GiB |
| `aef95919ac18b47c` | `specialized-fork-engines` | `vonk-forge/nemotron-3-5-lightning-dspark-lowmem-canary-single` | 1 | 21.4 GiB |
| `b1f48c056a9b1f36` | `specialized-fork-engines` | `vonk-forge/deepseek-v4-flash-0731-ds4-single` | 1 | 80.8 GiB |
| `b2ae04a0550d79bc` | `image-video-workflows` | `vonk-forge/wan-dancer-14b-disk-offload-pytorch-single` | 1 | 79.8 GiB |
| `b5f82ac9ba835b4b` | `image-video-workflows` | `vonk-forge/qwen-image-layered-diffusers-single` | 1 | 53.8 GiB |
| `b63ce156cf6655df` | `specialized-fork-engines` | `vonk-forge/deepseek-v4-flash-0731-mia-sparkinfer-single` | 1 | 99.5 GiB |
| `b8ead7f5f15c0177` | `specialized-fork-engines` | `vonk-forge/deepseek-v4-flash-0731-sparkinfer-single` | 1 | 99.5 GiB |
| `c0380c9df0146da2` | `image-video-workflows` | `vonk-forge/qwen-image-edit-2511-comfyui-single` | 1 | 47.0 GiB |
| `c36ecc306c28d9d2` | `vllm` | `vonk-forge/qwen3-8-27b-fp8-vllm-single` | 1 | 28.8 GiB |
| `c4576eaee8ff6507` | `vllm` | `vonk-forge/qwen3-8-27b-vllm-single` | 1 | 51.8 GiB |
| `c5561f37101545b0` | `vllm` | `vonk-forge/nemotron-3-super-120b-a12b-vllm-single` | 1 | 80.3 GiB |
| `c7f76a8204ec4e0e` | `specialized-fork-engines` | `vonk-forge/deepseek-v4-flash-0731-ds4-dspark-latency-single` | 1 | 86.3 GiB |
| `c9b3c7c91deab10e` | `image-video-workflows` | `vonk-forge/qwen-image-edit-2511-lightning-diffusers-single` | 1 | 54.5 GiB |
| `cb150d76b0d7f3c8` | `sglang` | `vonk-forge/qwen3-8-27b-nvfp4-dspark-sglang-single` | 1 | 25.6 GiB |
| `cb73a8dcd9642107` | `vllm` | `vonk-forge/gemma-4-26b-a4b-vllm-single` | 1 | 48.1 GiB |
| `cc4920ba6549b1f5` | `image-video-workflows` | `vonk-forge/flux-2-klein-4b-nvfp4-comfyui-single` | 1 | 10.1 GiB |
| `ce99e746deb2daf7` | `image-video-workflows` | `vonk-forge/qwen-image-2512-lightning-diffusers-single` | 1 | 54.5 GiB |
| `d2be802506b647b9` | `vllm` | `vonk-forge/nemotron-3-nano-30b-a3b-vllm-single` | 1 | 18.0 GiB |
| `da20026d0b5c8b3e` | `vllm` | `vonk-forge/qwen3-6-35b-a3b-nvfp4-vllm-single` | 1 | 21.9 GiB |
| `dce5ef52254c8b84` | `three-d-pipelines` | `vonk-forge/step1x-3d-geometry-pytorch-single` | 1 | 19.4 GiB |
| `de931df4f9c6dc06` | `audio-multimodal-pipelines` | `vonk-forge/moss-vl-realtime-11b-pytorch-single` | 1 | 21.1 GiB |
| `e26e2351c339480c` | `vllm` | `vonk-forge/nemotron-3-5-lightning-30b-a3b-vllm-single` | 1 | 20.1 GiB |
| `ebe178aa48cca2cf` | `image-video-workflows` | `vonk-forge/wan-2-2-ti2v-5b-comfyui-single` | 1 | 16.9 GiB |
| `f0453bf3cd4afcb5` | `audio-multimodal-pipelines` | `vonk-forge/hunyuan-video-foley-xl-pytorch-single` | 1 | 19.5 GiB |
| `f2c9a5aa85d134f3` | `vllm` | `vonk-forge/qwen3-8-flash-next-nvfp4-vllm-dual` | 1 | 126.0 GiB |
| `f47fbbac229cafb0` | `image-video-workflows` | `vonk-forge/ltx-2-5-22b-distilled-bf16-diffusers-single` | 1 | 65.3 GiB |
| `fc54bdef255c8524` | `three-d-pipelines` | `vonk-forge/skintokens-pytorch-single` | 1 | 1.5 GiB |
| `fccd3961103e72db` | `image-video-workflows` | `vonk-forge/wan-2-2-i2v-14b-comfyui-single` | 1 | 33.1 GiB |

## Executable batches

The authority owns exact recipe/lane assignments. Paired singles share one whole-fleet profile and apply, smoke in separate lanes, then wait at a cleanup/reconciliation barrier before the next batch. Dual-node assignments reserve both Sparks exclusively.

| Batch | Mode | Lane 1 | Lane 2 | Shared stack | Shared Models | Combined artifacts |
|---|---|---|---|---|---|---:|
| `batch-001` | `paired-single` | `vonk-forge/step1x-3d-geometry-pytorch-single` (rep) | `vonk-forge/step1x-3d-label-geometry-pytorch-single` (rep) | no | yes | 45.2 GiB |
| `batch-002` | `paired-single` | `vonk-forge/flux-2-klein-4b-nvfp4-comfyui-single` (rep) | `vonk-forge/skintokens-pytorch-single` (rep) | no | no | 11.6 GiB |
| `batch-003` | `paired-single` | `vonk-forge/trellis-2-4b-pytorch-single` (rep) | `vonk-forge/triposg-pytorch-single` (rep) | no | no | 24.0 GiB |
| `batch-004` | `paired-single` | `vonk-forge/flux-2-klein-4b-comfyui-single` (rep) | `vonk-forge/pixal3d-pytorch-single` (rep) | no | no | 38.5 GiB |
| `batch-005` | `paired-single` | `vonk-forge/ornith-1-5-35b-a3b-nvfp4-vllm-single` (rep) | `vonk-forge/lfm2-5-vl-3b-vllm-single` (rep) | no | no | 27.6 GiB |
| `batch-006` | `paired-single` | `vonk-forge/lfm2-5-vl-3b-vllm028-single` (rep) | `vonk-forge/qwen3-5-9b-vllm-single` (rep) | no | no | 23.8 GiB |
| `batch-007` | `paired-single` | `vonk-forge/qwen3-6-35b-a3b-nvfp4-vllm-single` (rep) | `vonk-forge/qwen3-8-27b-fp8-vllm-single` (rep) | no | no | 50.6 GiB |
| `batch-008` | `paired-single` | `vonk-forge/laguna-xs-2-1-nvfp4-vllm-single` (rep) | `vonk-forge/wan-2-2-ti2v-5b-comfyui-single` (rep) | no | no | 37.0 GiB |
| `batch-009` | `paired-single` | `vonk-forge/moss-vl-realtime-11b-pytorch-single` (rep) | `vonk-forge/ltx-2-19b-dev-bf16-diffusers-single` | no | no | 115.0 GiB |
| `batch-010` | `paired-single` | `vonk-forge/ltx-2-19b-dev-fp4-pytorch-single` (rep) | `vonk-forge/nemotron-3-5-lightning-30b-a3b-vllm-single` (rep) | no | no | 92.2 GiB |
| `batch-011` | `paired-single` | `vonk-forge/nemotron-3-nano-30b-a3b-vllm-single` (rep) | `vonk-forge/nemotron-3-nano-omni-30b-a3b-vllm-single` (rep) | no | no | 38.9 GiB |
| `batch-012` | `paired-single` | `vonk-forge/qwen3-8-27b-nvfp4-dspark-sglang-single` (rep) | `vonk-forge/ltx-2-3-22b-distilled-1-1-diffusers-single` (rep) | no | no | 114.9 GiB |
| `batch-013` | `paired-single` | `vonk-forge/qwen3-6-27b-vllm-single` (rep) | `vonk-forge/ltx-2-19b-distilled-fp8-diffusers-single` (rep) | no | no | 123.4 GiB |
| `batch-014` | `paired-single` | `vonk-forge/step1x-3d-texture-pytorch-single` (rep) | `vonk-forge/gemma-4-26b-a4b-vllm-single` (rep) | no | no | 139.3 GiB |
| `batch-015` | `paired-single` | `vonk-forge/gemma-4-26b-a4b-vllm028-single` (rep) | `vonk-forge/qwen-image-2512-fp8-lightning-comfyui-single` (rep) | no | no | 77.7 GiB |
| `batch-016` | `paired-single` | `vonk-forge/qwen-image-edit-2511-fp8mixed-comfyui-single` (rep) | `vonk-forge/qwen-image-edit-2511-int8-convrot-comfyui-single` (rep) | no | no | 56.2 GiB |
| `batch-017` | `paired-single` | `vonk-forge/ui-mate-27b-vllm-single` (rep) | `vonk-forge/muse-glimmer-30b-bf16-vllm-single` (rep) | no | no | 106.5 GiB |
| `batch-018` | `paired-single` | `vonk-forge/qwen3-8-27b-vllm-single` (rep) | `vonk-forge/ltx-2-5-22b-distilled-fp8-cast-diffusers-single` (rep) | no | no | 117.0 GiB |
| `batch-019` | `paired-single` | `vonk-forge/mova-360p-diffusers-single` (rep) | `vonk-forge/deepseek-v4-flash-0731-ds4-dspark-latency-single` (rep) | no | no | 158.7 GiB |
| `batch-020` | `paired-single` | `vonk-forge/deepseek-v4-flash-0731-ds4-single` (rep) | `vonk-forge/ltx-2-19b-distilled-diffusers-single` (rep) | no | no | 167.4 GiB |
| `batch-021` | `paired-single` | `vonk-forge/laguna-s-2-1-nvfp4-vllm-low-memory-canary-single` (rep) | `vonk-forge/mova-720p-diffusers-single` | no | no | 165.2 GiB |
| `batch-022` | `paired-single` | `vonk-forge/nemotron-3-5-lightning-dspark-lowmem-canary-single` (rep) | `vonk-forge/nemotron-3-super-120b-a12b-vllm-single` (rep) | no | no | 101.7 GiB |
| `batch-023` | `paired-single` | `vonk-forge/nvidia-qwen-image-flash-diffusers-single` (rep) | `vonk-forge/qwen-image-2512-comfyui-single` (rep) | no | no | 100.8 GiB |
| `batch-024` | `paired-single` | `vonk-forge/qwen-image-2512-diffusers-single` (rep) | `vonk-forge/qwen-image-2512-lightning-diffusers-single` (rep) | no | yes | 108.3 GiB |
| `batch-025` | `paired-single` | `vonk-forge/qwen-image-edit-2511-comfyui-single` (rep) | `vonk-forge/qwen-image-edit-2511-diffusers-single` (rep) | no | no | 100.8 GiB |
| `batch-026` | `paired-single` | `vonk-forge/qwen-image-edit-2511-lightning-diffusers-single` (rep) | `vonk-forge/qwen-image-layered-diffusers-single` (rep) | no | no | 108.3 GiB |
| `batch-027` | `paired-single` | `vonk-forge/wan-2-2-i2v-14b-comfyui-single` (rep) | `vonk-forge/wan-2-2-t2v-14b-comfyui-single` (rep) | no | no | 66.3 GiB |
| `batch-028` | `paired-single` | `vonk-forge/wan-dancer-14b-disk-offload-pytorch-single` (rep) | `vonk-forge/deepseek-v4-flash-0731-sparkinfer-target-only-canary-single` (rep) | no | no | 179.3 GiB |
| `batch-029` | `paired-single` | `vonk-forge/ling-3-0-flash-dspark-sglang-single` (rep) | `vonk-forge/deepseek-v4-flash-0731-mia-sparkinfer-single` (rep) | no | no | 173.8 GiB |
| `batch-030` | `paired-single` | `vonk-forge/hunyuan-video-foley-xl-pytorch-single` (rep) | `vonk-forge/hunyuan-video-foley-xxl-pytorch-single` (rep) | no | yes | 39.0 GiB |
| `batch-031` | `paired-single` | `vonk-forge/hunyuanocr-1-5-vllm-dflash-single` (rep) | `vonk-forge/hunyuan3d-omni-pytorch-single` (rep) | no | no | 28.7 GiB |
| `batch-032` | `paired-single` | `vonk-forge/hunyuan-video-15-distilled-diffusers-single` (rep) | `vonk-forge/hunyuan-video-15-i2v-step-distilled-diffusers-single` (rep) | no | no | 82.0 GiB |
| `batch-033` | `paired-single` | `vonk-forge/hunyuan-video-15-t2v-diffusers-single` (rep) | `vonk-forge/minimax-h3-diffusers-single` (rep) | no | no | 514.0 GiB |
| `batch-034` | `paired-single` | `vonk-forge/minimax-h3-fl2va-diffusers-single` (rep) | `vonk-forge/ltx-2-5-22b-distilled-bf16-diffusers-single` (rep) | no | no | 199.4 GiB |
| `batch-035` | `paired-single` | `vonk-forge/nemotron-3-5-lightning-30b-a3b-vllm-dspark-latency-single` (rep) | `vonk-forge/wan-dancer-14b-pytorch-single` (rep) | no | no | 101.1 GiB |
| `batch-036` | `paired-single` | `vonk-forge/deepseek-v4-flash-0731-sparkinfer-single` (rep) | `vonk-forge/laguna-s-2-1-nvfp4-vllm-single` (rep) | no | no | 192.4 GiB |
| `batch-037` | `exclusive-dual` | `vonk-forge/deepseek-v4-flash-0731-mia-dual` (rep) | — | yes | no | 155.4 GiB |
| `batch-038` | `exclusive-dual` | `vonk-forge/deepseek-v4-flash-vision-exp-mia-dual` (rep) | — | yes | no | 156.3 GiB |
| `batch-039` | `exclusive-dual` | `vonk-forge/glm-5-3-flash-exl3-dflash2-vllm-dual` (rep) | — | yes | no | 168.0 GiB |
| `batch-040` | `exclusive-dual` | `vonk-forge/inkling-small-nvfp4-sglang-dual` (rep) | — | yes | no | 159.0 GiB |
| `batch-041` | `exclusive-dual` | `vonk-forge/qwen3-8-flash-next-nvfp4-sglang-dual` (rep) | — | yes | no | 126.0 GiB |
| `batch-042` | `exclusive-dual` | `vonk-forge/qwen3-8-flash-next-nvfp4-vllm-dual` (rep) | — | yes | no | 126.0 GiB |
| `batch-043` | `exclusive-dual` | `vonk-forge/glm-5-3-flash-nvfp4-vllm-dual` (rep) | — | yes | no | 181.3 GiB |
| `batch-044` | `exclusive-dual` | `vonk-forge/glm-5-3-flash-nvfp4-ablit-l15-43-dflash2-vllm-dual` (rep) | — | yes | no | 188.7 GiB |
| `batch-045` | `exclusive-dual` | `vonk-forge/glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual` (rep) | — | yes | no | 181.3 GiB |

## Outside available capacity

These recipes require more than two Sparks and remain visible for catalog audit.

| Recipe | Nodes | Group | Reason |
|---|---:|---|---|
| `vonk-forge/glm-5-2-aqlm-vllm-triple` | 3 | `specialized-fork-engines` | requires more than two Sparks; catalog audit only |
| `vonk-forge/glm-5-2-quanttrio-vllm-four` | 4 | `specialized-fork-engines` | requires more than two Sparks; catalog audit only |
| `vonk-forge/glm-5-3-flash-nvfp4-vllm-four` | 4 | `vllm` | requires more than two Sparks; catalog audit only |
| `vonk-forge/inkling-975b-a41b-nvfp4-sglang-eight` | 8 | `sglang` | requires more than two Sparks; catalog audit only |

## Typed recovery coverage

These definitions are validated against the current Pydantic authority contract. Sharing still requires the exact representative receipt and a member-use record that binds recipe, package, Models, stack, topology, runtime image, platform/agent builds, target nodes and smoke receipt.

| Coverage ID | Failure mode | Representative | Members | Shared | Model review |
|---|---|---|---:|---|---|
| `e7ad8f1614fa5aa2` | `dual-host-restart` | `vonk-forge/qwen3-8-flash-next-nvfp4-vllm-dual` | 1 | no | n/a |
| `2447c28b8e9fdbb6` | `dual-host-restart` | `vonk-forge/glm-5-3-flash-exl3-dflash2-vllm-dual` | 1 | no | n/a |
| `3a9a47af3aaafd82` | `dual-host-restart` | `vonk-forge/inkling-small-nvfp4-sglang-dual` | 1 | no | n/a |
| `0e1d72dbd67aed25` | `dual-host-restart` | `vonk-forge/glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual` | 1 | no | n/a |
| `2b51d1349ae3f8e6` | `dual-host-restart` | `vonk-forge/deepseek-v4-flash-0731-mia-dual` | 1 | no | n/a |
| `14766703c3081e55` | `dual-host-restart` | `vonk-forge/glm-5-3-flash-nvfp4-ablit-l15-43-dflash2-vllm-dual` | 1 | no | n/a |
| `27868f7bf1719e80` | `dual-host-restart` | `vonk-forge/glm-5-3-flash-nvfp4-vllm-dual` | 1 | no | n/a |
| `a3dab006b3d1dfef` | `dual-host-restart` | `vonk-forge/deepseek-v4-flash-vision-exp-mia-dual` | 1 | no | n/a |
| `81a4d3e7311df4c5` | `dual-host-restart` | `vonk-forge/qwen3-8-flash-next-nvfp4-sglang-dual` | 1 | no | n/a |
| `4788aaee4b5ac010` | `dual-rank-loss-recovery` | `vonk-forge/qwen3-8-flash-next-nvfp4-vllm-dual` | 1 | no | n/a |
| `0fe3abfea57b5957` | `dual-rank-loss-recovery` | `vonk-forge/glm-5-3-flash-exl3-dflash2-vllm-dual` | 1 | no | n/a |
| `f22b441b8ccd0a59` | `dual-rank-loss-recovery` | `vonk-forge/inkling-small-nvfp4-sglang-dual` | 1 | no | n/a |
| `cc3da824f444134a` | `dual-rank-loss-recovery` | `vonk-forge/glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual` | 1 | no | n/a |
| `1990facff7069a91` | `dual-rank-loss-recovery` | `vonk-forge/deepseek-v4-flash-0731-mia-dual` | 1 | no | n/a |
| `2a72e60b1f2f42f9` | `dual-rank-loss-recovery` | `vonk-forge/glm-5-3-flash-nvfp4-ablit-l15-43-dflash2-vllm-dual` | 1 | no | n/a |
| `dec6eb2540e86922` | `dual-rank-loss-recovery` | `vonk-forge/glm-5-3-flash-nvfp4-vllm-dual` | 1 | no | n/a |
| `17290e7c68662175` | `dual-rank-loss-recovery` | `vonk-forge/deepseek-v4-flash-vision-exp-mia-dual` | 1 | no | n/a |
| `9c124690497e549c` | `dual-rank-loss-recovery` | `vonk-forge/qwen3-8-flash-next-nvfp4-sglang-dual` | 1 | no | n/a |
| `dc0f23d422f59311` | `single-host-restart` | `vonk-forge/hunyuan-video-15-i2v-step-distilled-diffusers-single` | 1 | no | n/a |
| `14895d5f282df7b5` | `single-host-restart` | `vonk-forge/ornith-1-5-35b-a3b-nvfp4-vllm-single` | 1 | no | n/a |
| `7655e723ee062434` | `single-host-restart` | `vonk-forge/ltx-2-19b-dev-bf16-diffusers-single` | 1 | no | n/a |
| `972bd2b180bd8857` | `single-host-restart` | `vonk-forge/ltx-2-3-22b-distilled-1-1-diffusers-single` | 1 | no | n/a |
| `098f3d6616d0b796` | `single-host-restart` | `vonk-forge/deepseek-v4-flash-0731-sparkinfer-single` | 1 | no | n/a |
| `2bba8e1cd753ab25` | `single-host-restart` | `vonk-forge/qwen-image-edit-2511-comfyui-single` | 1 | no | n/a |
| `80e03c7bcdd499a5` | `single-host-restart` | `vonk-forge/nvidia-qwen-image-flash-diffusers-single` | 1 | no | n/a |
| `9bbed0b3e98da72c` | `single-host-restart` | `vonk-forge/deepseek-v4-flash-0731-ds4-dspark-latency-single` | 1 | no | n/a |
| `c9572c88417cb6a0` | `single-host-restart` | `vonk-forge/qwen-image-edit-2511-lightning-diffusers-single` | 1 | no | n/a |
| `b3e432f0d4d6695d` | `single-host-restart` | `vonk-forge/hunyuan3d-omni-pytorch-single` | 1 | no | n/a |
| `44f72a88832600a9` | `single-host-restart` | `vonk-forge/qwen-image-2512-diffusers-single` | 1 | no | n/a |
| `f00b7407b4dbe383` | `single-host-restart` | `vonk-forge/qwen-image-2512-lightning-diffusers-single` | 1 | no | n/a |
| `10a698acaceed6eb` | `single-host-restart` | `vonk-forge/mova-720p-diffusers-single` | 1 | no | n/a |
| `a958082b0f8eb641` | `single-host-restart` | `vonk-forge/mova-360p-diffusers-single` | 1 | no | n/a |
| `176a7fc1e18c6590` | `single-host-restart` | `vonk-forge/nemotron-3-5-lightning-dspark-lowmem-canary-single` | 1 | no | n/a |
| `2cd1edcaffbbb2fa` | `single-host-restart` | `vonk-forge/deepseek-v4-flash-0731-sparkinfer-target-only-canary-single` | 1 | no | n/a |
| `9d4cda4fc0f8b59f` | `single-host-restart` | `vonk-forge/hunyuan-video-foley-xl-pytorch-single` | 1 | no | n/a |
| `047bbd9163878882` | `single-host-restart` | `vonk-forge/lfm2-5-vl-3b-vllm028-single` | 1 | no | n/a |
| `baa923e02ca134e3` | `single-host-restart` | `vonk-forge/hunyuan-video-15-t2v-diffusers-single` | 1 | no | n/a |
| `cb4fcb42262e5244` | `single-host-restart` | `vonk-forge/laguna-s-2-1-nvfp4-vllm-single` | 1 | no | n/a |
| `802ee16f2e4cc14f` | `single-host-restart` | `vonk-forge/moss-vl-realtime-11b-pytorch-single` | 1 | no | n/a |
| `2513669007435c45` | `single-host-restart` | `vonk-forge/ltx-2-5-22b-distilled-fp8-cast-diffusers-single` | 1 | no | n/a |
| `c82b92bca5fd7b67` | `single-host-restart` | `vonk-forge/minimax-h3-diffusers-single` | 1 | no | n/a |
| `abde5f9e13d95b3d` | `single-host-restart` | `vonk-forge/nemotron-3-nano-30b-a3b-vllm-single` | 1 | no | n/a |
| `84cc874a097d44b2` | `single-host-restart` | `vonk-forge/ltx-2-19b-distilled-fp8-diffusers-single` | 1 | no | n/a |
| `5013b360f39b801b` | `single-host-restart` | `vonk-forge/ltx-2-19b-distilled-diffusers-single` | 1 | no | n/a |
| `b059d8206d454237` | `single-host-restart` | `vonk-forge/ltx-2-5-22b-distilled-bf16-diffusers-single` | 1 | no | n/a |
| `26672cf138e08a71` | `single-host-restart` | `vonk-forge/nemotron-3-nano-omni-30b-a3b-vllm-single` | 1 | no | n/a |
| `9234d31d40d27076` | `single-host-restart` | `vonk-forge/deepseek-v4-flash-0731-ds4-single` | 1 | no | n/a |
| `b77b7c8aea89c448` | `single-host-restart` | `vonk-forge/gemma-4-26b-a4b-vllm028-single` | 1 | no | n/a |
| `d11361ff14a65705` | `single-host-restart` | `vonk-forge/hunyuanocr-1-5-vllm-dflash-single` | 1 | no | n/a |
| `da78a35b0c350408` | `single-host-restart` | `vonk-forge/triposg-pytorch-single` | 1 | no | n/a |
| `01d11206ea7a5a6b` | `single-host-restart` | `vonk-forge/laguna-s-2-1-nvfp4-vllm-low-memory-canary-single` | 1 | no | n/a |
| `4afb2762471d44f9` | `single-host-restart` | `vonk-forge/laguna-xs-2-1-nvfp4-vllm-single` | 1 | no | n/a |
| `29819c7d35e021ce` | `single-host-restart` | `vonk-forge/flux-2-klein-4b-comfyui-single` | 1 | no | n/a |
| `aeec665874d0eebf` | `single-host-restart` | `vonk-forge/flux-2-klein-4b-nvfp4-comfyui-single` | 1 | no | n/a |
| `5b3ec130e9f5e386` | `single-host-restart` | `vonk-forge/wan-2-2-ti2v-5b-comfyui-single` | 1 | no | n/a |
| `4f8fc22fe9525da0` | `single-host-restart` | `vonk-forge/ling-3-0-flash-dspark-sglang-single` | 1 | no | n/a |
| `22b937f9b480b7b6` | `single-host-restart` | `vonk-forge/wan-dancer-14b-pytorch-single` | 1 | no | n/a |
| `0f953f1cecd6f040` | `single-host-restart` | `vonk-forge/step1x-3d-label-geometry-pytorch-single` | 1 | no | n/a |
| `47f2b805ad002075` | `single-host-restart` | `vonk-forge/qwen3-6-27b-vllm-single` | 1 | no | n/a |
| `8340e7e034cf48d5` | `single-host-restart` | `vonk-forge/nemotron-3-super-120b-a12b-vllm-single` | 1 | no | n/a |
| `4503ed55994eb51c` | `single-host-restart` | `vonk-forge/ui-mate-27b-vllm-single` | 1 | no | n/a |
| `b8d57e128710abcf` | `single-host-restart` | `vonk-forge/pixal3d-pytorch-single` | 1 | no | n/a |
| `daedf7fe8274c753` | `single-host-restart` | `vonk-forge/lfm2-5-vl-3b-vllm-single` | 1 | no | n/a |
| `7a26c3feb87b0587` | `single-host-restart` | `vonk-forge/qwen3-8-27b-vllm-single` | 1 | no | n/a |
| `b302d23df7a2c2b5` | `single-host-restart` | `vonk-forge/hunyuan-video-15-distilled-diffusers-single` | 1 | no | n/a |
| `0ae607d9c1b03402` | `single-host-restart` | `vonk-forge/qwen-image-edit-2511-fp8mixed-comfyui-single` | 1 | no | n/a |
| `aa9a25be3a7fe62c` | `single-host-restart` | `vonk-forge/gemma-4-26b-a4b-vllm-single` | 1 | no | n/a |
| `71372fbb861c1dc4` | `single-host-restart` | `vonk-forge/step1x-3d-texture-pytorch-single` | 1 | no | n/a |
| `68a4a9fc29ae0c7a` | `single-host-restart` | `vonk-forge/minimax-h3-fl2va-diffusers-single` | 1 | no | n/a |
| `5a28aa1f78b2c06a` | `single-host-restart` | `vonk-forge/qwen3-5-9b-vllm-single` | 1 | no | n/a |
| `030b9d20a00c8a9c` | `single-host-restart` | `vonk-forge/wan-2-2-i2v-14b-comfyui-single` | 1 | no | n/a |
| `41f3f9703ef97278` | `single-host-restart` | `vonk-forge/ltx-2-19b-dev-fp4-pytorch-single` | 1 | no | n/a |
| `9be3f2966a553be7` | `single-host-restart` | `vonk-forge/qwen-image-layered-diffusers-single` | 1 | no | n/a |
| `1b81c34d1c0cda26` | `single-host-restart` | `vonk-forge/qwen-image-edit-2511-diffusers-single` | 1 | no | n/a |
| `519c0f763bf545a4` | `single-host-restart` | `vonk-forge/wan-dancer-14b-disk-offload-pytorch-single` | 1 | no | n/a |
| `2f544c2066f5879c` | `single-host-restart` | `vonk-forge/qwen-image-2512-comfyui-single` | 1 | no | n/a |
| `0e5a521307a65ea1` | `single-host-restart` | `vonk-forge/muse-glimmer-30b-bf16-vllm-single` | 1 | no | n/a |
| `18be7e00caa560e2` | `single-host-restart` | `vonk-forge/wan-2-2-t2v-14b-comfyui-single` | 1 | no | n/a |
| `9299e55f9d43034b` | `single-host-restart` | `vonk-forge/step1x-3d-geometry-pytorch-single` | 1 | no | n/a |
| `8d85349f0b3bb9d6` | `single-host-restart` | `vonk-forge/qwen-image-2512-fp8-lightning-comfyui-single` | 1 | no | n/a |
| `c22aedb43b2f56a6` | `single-host-restart` | `vonk-forge/nemotron-3-5-lightning-30b-a3b-vllm-dspark-latency-single` | 1 | no | n/a |
| `4b0332dd79de131d` | `single-host-restart` | `vonk-forge/trellis-2-4b-pytorch-single` | 1 | no | n/a |
| `fedb188dbd167084` | `single-host-restart` | `vonk-forge/deepseek-v4-flash-0731-mia-sparkinfer-single` | 1 | no | n/a |
| `0e0d4aa0f81a028a` | `single-host-restart` | `vonk-forge/qwen3-8-27b-fp8-vllm-single` | 1 | no | n/a |
| `e50213bb3da4e413` | `single-host-restart` | `vonk-forge/nemotron-3-5-lightning-30b-a3b-vllm-single` | 1 | no | n/a |
| `8eae8cb0b9413e9e` | `single-host-restart` | `vonk-forge/qwen3-8-27b-nvfp4-dspark-sglang-single` | 1 | no | n/a |
| `62edac4312f8830f` | `single-host-restart` | `vonk-forge/skintokens-pytorch-single` | 1 | no | n/a |
| `faa8ac90ddd40f90` | `single-host-restart` | `vonk-forge/qwen3-6-35b-a3b-nvfp4-vllm-single` | 1 | no | n/a |
| `3042976b6b5c95d5` | `single-host-restart` | `vonk-forge/qwen-image-edit-2511-int8-convrot-comfyui-single` | 1 | no | n/a |
| `0eb24676e0db93e2` | `single-host-restart` | `vonk-forge/hunyuan-video-foley-xxl-pytorch-single` | 1 | no | n/a |

## Published source comparison

The working catalog index matches the accepted release index.

The following rows differ from the accepted recipe or build-source identity.

| Recipe | Difference | Published stack | Working stack |
|---|---|---|---|
| `vonk-forge/ltx-2-19b-dev-bf16-diffusers-single` | authored source differs from published in: authored build source | `0ce64ce4e51f2135` | `488290dc79fc76b2` |
| `vonk-forge/ltx-2-3-22b-distilled-1-1-diffusers-single` | authored source differs from published in: authored build source | `0ce64ce4e51f2135` | `488290dc79fc76b2` |
| `vonk-forge/ltx-2-19b-distilled-fp8-diffusers-single` | authored source differs from published in: authored build source | `619a141fab505bcf` | `51f8214b9cf16aed` |
| `vonk-forge/ltx-2-19b-distilled-diffusers-single` | authored source differs from published in: authored build source | `619a141fab505bcf` | `51f8214b9cf16aed` |

## Evidence boundaries

- Cache readiness remains owned by the live Controller cache surface.
- Structural validity does not prove an image build or runtime acceptance.
- Controller and runner deployment require separate live evidence.
- Physical inference, declared assertions and recovery receipts remain recipe-specific evidence.
- Declared footprints do not replace a fresh whole-fleet admission preview.
