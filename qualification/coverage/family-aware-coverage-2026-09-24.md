# Family-aware qualification coverage matrix — 2026-09-24

Derived from exact catalog identities by `tools/build-family-aware-coverage`.

This artifact answers the plan's coverage-matrix implementation item. It records structural identity, representative selection and scheduling bounds only. It is **not** cache readiness, structural qualification, Controller deployment or physical Spark acceptance.

## Sources

| Input | Identity |
|---|---|
| Plan | `docs/recipe-qualification-plan-2026-09-24.md` |
| Catalog | CarstVaartjes/vonk-forge-recipes — 85 recipe documents |
| Qualification authority | `nl-sequential-2c118a99` |
| Scope | 81 recipes within 2 Sparks; 4 larger topologies listed for audit only |

## Derivation

- Every indexed recipe and Model document is validated through the canonical contract, and its canonical content digest must equal the producer index digest and the authored tree document.
- `stack_id` is the canonical digest of the exact engine, entrypoint, base image repository/digest/platform, build context, Dockerfile, target and declared build patches. It is the reuse boundary.
- `coverage_group` is the primary engine/workflow group. Declared forks are cross-cutting: a non-stock engine image or a declared fork marker in the recipe slug or build context is recorded in `specialized_forks` and cannot be satisfied by generic engine evidence.
- `risk_group_id` adds model family, quantization and node count to the stack.
- A risk group's representative is the ready-to-build member with the smallest declared artifact footprint; cache readiness is not decided here.
- Paired batches place two single-Spark rows on the two Sparks in authority order. Dual-Spark rows are exclusive. Every batch keeps distinct lane identities and a barrier.
- Recovery sharing is proposed by declared failure signature only and stays unenforceable until typed runner/authority support exists.

## Coverage groups

The required checks and variant boundaries for each group remain owned by the plan; this table only records which exact identities fall in each group.

| Coverage group | In-scope rows | Specialized-forks flag | Exclusive rows |
|---|---:|---:|---:|
| `vllm` | 21 | 0 | 2 |
| `sglang` | 4 | 3 | 2 |
| `specialized-fork-engines` | 13 | 12 | 5 |
| `image-video-workflows` | 32 | 2 | 0 |
| `three-d-pipelines` | 8 | 0 | 0 |
| `audio-multimodal-pipelines` | 3 | 2 | 0 |

## Stack and family matrix

One row per catalog recipe. `seq` is the authority sequence for in-scope rows; the four larger topologies are listed last in ascending node count and remain owned by the plan inventory's printed order. `rep` names the risk-group representative, `repr` marks a representative, and `batch` is `P<n>/L<lane>` for a paired lane, `X` for an exclusive dual-Spark row, or `-` for an out-of-scope topology.

| seq | Recipe | Nodes | Group | Engine | Base image | Build context | Model families | Quantization | Artifacts | rep | repr | batch |
|---:|---|---:|---|---|---|---|---|---|---:|---|---|---|
| 1 | `vonk-forge/step1x-3d-geometry-pytorch-single` | 1 | `three-d-pipelines` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@0e1392f431f8` | `adapters/three-d/step1x-3d` | dinov2-with-registers, step1x-3d | none | 19.4 GiB |  | yes | P1/L1 |
| 2 | `vonk-forge/step1x-3d-label-geometry-pytorch-single` | 1 | `three-d-pipelines` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@0e1392f431f8` | `adapters/three-d/step1x-3d` | clip-vit-large-patch14, dinov2-with-registers, step1x-3d | none | 25.8 GiB |  | yes | P1/L2 |
| 3 | `vonk-forge/flux-2-klein-4b-nvfp4-comfyui-single` | 1 | `image-video-workflows` | `comfyui` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/media/comfyui-core` | flux-2 | nvfp4 | 10.1 GiB |  | yes | P2/L1 |
| 4 | `vonk-forge/skintokens-pytorch-single` | 1 | `three-d-pipelines` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@0e1392f431f8` | `adapters/three-d/skintokens` | skintokens | none | 1.5 GiB |  | yes | P2/L2 |
| 5 | `vonk-forge/trellis-2-4b-pytorch-single` | 1 | `three-d-pipelines` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@0e1392f431f8` | `adapters/three-d/trellis2-native` | trellis | none | 16.4 GiB |  | yes | P3/L1 |
| 6 | `vonk-forge/triposg-pytorch-single` | 1 | `three-d-pipelines` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@0e1392f431f8` | `adapters/three-d/triposg` | bria-rmbg, triposg | none | 7.6 GiB |  | yes | P3/L2 |
| 7 | `vonk-forge/flux-2-klein-4b-comfyui-single` | 1 | `image-video-workflows` | `comfyui` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/media/comfyui-core` | flux-2 | none | 15.0 GiB |  | yes | P4/L1 |
| 8 | `vonk-forge/pixal3d-pytorch-single` | 1 | `three-d-pipelines` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@0e1392f431f8` | `adapters/three-d/trellis2-native` | pixal3d | none | 23.5 GiB |  | yes | P4/L2 |
| 9 | `vonk-forge/ornith-1-5-35b-a3b-nvfp4-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@1c8e60a0841b` | `adapters/llm/vllm-openai` | ornith-1-5 | modelopt-mixed-w4a16-nvfp4-fp8 | 21.8 GiB |  | yes | P5/L1 |
| 10 | `vonk-forge/lfm2-5-vl-3b-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@1c8e60a0841b` | `adapters/liquidai/lfm25-vl-vllm` | lfm2-5 | none | 5.8 GiB |  | yes | P5/L2 |
| 11 | `vonk-forge/lfm2-5-vl-3b-vllm028-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@41b54fb42c66` | `adapters/liquidai/lfm25-vl-vllm-028` | lfm2-5 | none | 5.8 GiB |  | yes | P6/L1 |
| 12 | `vonk-forge/qwen3-5-9b-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@41b54fb42c66` | `adapters/llm/qwen35-vllm` | qwen | none | 18.0 GiB |  | yes | P6/L2 |
| 13 | `vonk-forge/qwen3-6-35b-a3b-nvfp4-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@41b54fb42c66` | `adapters/nvidia/qwen36-35b-vllm` | nvidia-qwen | nvfp4 | 21.9 GiB |  | yes | P7/L1 |
| 14 | `vonk-forge/qwen3-8-27b-fp8-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@41b54fb42c66` | `adapters/llm/vllm-openai-028` | qwen | fp8 | 28.8 GiB |  | yes | P7/L2 |
| 15 | `vonk-forge/laguna-xs-2-1-nvfp4-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@1c8e60a0841b` | `adapters/llm/laguna-vllm` | poolside | nvfp4-fp8-kv-cache | 20.1 GiB |  | yes | P8/L1 |
| 16 | `vonk-forge/wan-2-2-ti2v-5b-comfyui-single` | 1 | `image-video-workflows` | `comfyui` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/media/comfyui-core` | wan-2-2 | fp8 | 16.9 GiB |  | yes | P8/L2 |
| 17 | `vonk-forge/moss-vl-realtime-11b-pytorch-single` | 1 | `audio-multimodal-pipelines` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/video/moss-vl-realtime` | moss-vl | none | 21.1 GiB |  | yes | P9/L1 |
| 18 | `vonk-forge/ltx-2-19b-dev-bf16-diffusers-single` | 1 | `image-video-workflows` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@a52783d8d73a` | `adapters/video/ltx23-sync-native-disk` | ltx | none | 93.8 GiB | vonk-forge/ltx-2-3-22b-distilled-1-1-diffusers-single |  | P9/L2 |
| 19 | `vonk-forge/ltx-2-19b-dev-fp4-pytorch-single` | 1 | `image-video-workflows` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@a52783d8d73a` | `adapters/video/ltx2-pytorch` | ltx | fp4 | 72.1 GiB |  | yes | P10/L1 |
| 20 | `vonk-forge/nemotron-3-5-lightning-30b-a3b-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@1c8e60a0841b` | `adapters/nvidia/nemotron` | nemotron | nvfp4 | 20.1 GiB |  | yes | P10/L2 |
| 21 | `vonk-forge/nemotron-3-nano-30b-a3b-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@871fc9b75a97` | `adapters/nvidia/nemotron-vllm-0-20-0` | nemotron | nvfp4-fp8-kv-selective-bf16 | 18.0 GiB |  | yes | P11/L1 |
| 22 | `vonk-forge/nemotron-3-nano-omni-30b-a3b-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@871fc9b75a97` | `adapters/nvidia/nemotron-vllm-0-20-0` | nemotron | nvfp4 | 20.9 GiB |  | yes | P11/L2 |
| 23 | `vonk-forge/qwen3-8-27b-nvfp4-dspark-sglang-single` | 1 | `sglang` | `sglang` | `docker.io/lmsysorg/sglang@3c0abdf41ef2` | `adapters/qwen/qwen38-27b-dspark-single` | qwen3-8-radixark | none, nvfp4-w4a4-bf16-lmhead | 25.6 GiB |  | yes | P12/L1 |
| 24 | `vonk-forge/ltx-2-3-22b-distilled-1-1-diffusers-single` | 1 | `image-video-workflows` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@a52783d8d73a` | `adapters/video/ltx23-sync-native-disk` | ltx | none | 89.3 GiB |  | yes | P12/L2 |
| 25 | `vonk-forge/qwen3-6-27b-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@41b54fb42c66` | `adapters/llm/vllm-openai-028` | qwen | none | 51.8 GiB |  | yes | P13/L1 |
| 26 | `vonk-forge/ltx-2-19b-distilled-fp8-diffusers-single` | 1 | `image-video-workflows` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@a52783d8d73a` | `adapters/video/ltx2-sync-native` | ltx | fp8-scaled-mm, none | 71.6 GiB |  | yes | P13/L2 |
| 27 | `vonk-forge/step1x-3d-texture-pytorch-single` | 1 | `three-d-pipelines` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@0e1392f431f8` | `adapters/three-d/step1x-3d` | sdxl-vae, stable-diffusion-xl, step1x-3d | none | 91.2 GiB |  | yes | P14/L1 |
| 28 | `vonk-forge/gemma-4-26b-a4b-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@1c8e60a0841b` | `adapters/llm/vllm-openai` | gemma | none | 48.1 GiB |  | yes | P14/L2 |
| 29 | `vonk-forge/gemma-4-26b-a4b-vllm028-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@41b54fb42c66` | `adapters/google/gemma4-vllm-028` | gemma | none | 48.1 GiB |  | yes | P15/L1 |
| 30 | `vonk-forge/qwen-image-2512-fp8-lightning-comfyui-single` | 1 | `image-video-workflows` | `comfyui` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/media/comfyui-qwen-image-2512-fp8-lightning` | qwen-image | fp8, none | 29.6 GiB |  | yes | P15/L2 |
| 31 | `vonk-forge/qwen-image-edit-2511-fp8mixed-comfyui-single` | 1 | `image-video-workflows` | `comfyui` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/media/comfyui-qwen-image-edit-2511-fp8mixed` | qwen-image-edit | fp8mixed | 28.1 GiB |  | yes | P16/L1 |
| 32 | `vonk-forge/qwen-image-edit-2511-int8-convrot-comfyui-single` | 1 | `image-video-workflows` | `comfyui` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/media/comfyui-qwen-image-edit-2511-int8-convrot` | qwen-image-edit | int8_tensorwise_convrot | 28.1 GiB |  | yes | P16/L2 |
| 33 | `vonk-forge/ui-mate-27b-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@41b54fb42c66` | `adapters/llm/ui-mate-vllm` | ui-mate | none | 51.0 GiB |  | yes | P17/L1 |
| 34 | `vonk-forge/muse-glimmer-30b-bf16-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@ae1de325b8ea` | `adapters/llm/muse-glimmer-vllm` | muse-glimmer | none | 55.5 GiB |  | yes | P17/L2 |
| 35 | `vonk-forge/qwen3-8-27b-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@41b54fb42c66` | `adapters/llm/vllm-openai-028` | qwen | none | 51.8 GiB | vonk-forge/qwen3-6-27b-vllm-single |  | P18/L1 |
| 36 | `vonk-forge/ltx-2-5-22b-distilled-fp8-cast-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/video/ltx25-diffusers-fp8-canary` | ltx | none | 65.3 GiB |  | yes | P18/L2 |
| 37 | `vonk-forge/mova-360p-diffusers-single` | 1 | `image-video-workflows` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@a52783d8d73a` | `adapters/video/mova-pytorch` | mova | none | 72.4 GiB |  | yes | P19/L1 |
| 38 | `vonk-forge/deepseek-v4-flash-0731-ds4-dspark-latency-single` | 1 | `specialized-fork-engines` | `ds4` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/deepseek/ds4` | deepseek-flash | iq2_xxs-q2_k-mixed | 86.3 GiB | vonk-forge/deepseek-v4-flash-0731-ds4-single |  | P19/L2 |
| 39 | `vonk-forge/deepseek-v4-flash-0731-ds4-single` | 1 | `specialized-fork-engines` | `ds4` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/deepseek/ds4` | deepseek-flash | iq2_xxs-q2_k-mixed | 80.8 GiB |  | yes | P20/L1 |
| 40 | `vonk-forge/ltx-2-19b-distilled-diffusers-single` | 1 | `image-video-workflows` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@a52783d8d73a` | `adapters/video/ltx2-sync-native` | ltx | none | 86.7 GiB |  | yes | P20/L2 |
| 41 | `vonk-forge/laguna-s-2-1-nvfp4-vllm-low-memory-canary-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@1c8e60a0841b` | `adapters/llm/laguna-s-vllm` | poolside | nvfp4 | 92.9 GiB |  | yes | P21/L1 |
| 42 | `vonk-forge/mova-720p-diffusers-single` | 1 | `image-video-workflows` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@a52783d8d73a` | `adapters/video/mova-pytorch` | mova | none | 72.4 GiB | vonk-forge/mova-360p-diffusers-single |  | P21/L2 |
| 43 | `vonk-forge/nemotron-3-5-lightning-dspark-lowmem-canary-single` | 1 | `specialized-fork-engines` | `vllm` | `docker.io/vllm/vllm-openai@1c8e60a0841b` | `adapters/nvidia/nemotron` | nemotron | nvfp4, w4a16-nvfp4 | 21.4 GiB | vonk-forge/nemotron-3-5-lightning-30b-a3b-vllm-dspark-latency-single |  | P22/L1 |
| 44 | `vonk-forge/nemotron-3-super-120b-a12b-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@1c8e60a0841b` | `adapters/llm/vllm-openai` | nemotron | none, nvfp4 | 80.3 GiB |  | yes | P22/L2 |
| 45 | `vonk-forge/nvidia-qwen-image-flash-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/image/nvidia-qwen-image-flash-diffusers` | qwen-image | none | 53.7 GiB |  | yes | P23/L1 |
| 46 | `vonk-forge/qwen-image-2512-comfyui-single` | 1 | `image-video-workflows` | `comfyui` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/media/comfyui-qwen-image-2512-bf16` | qwen-image | fp8 | 47.0 GiB |  | yes | P23/L2 |
| 47 | `vonk-forge/qwen-image-2512-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/image/qwen-image-2512-diffusers` | qwen-image | bf16 | 53.7 GiB |  | yes | P24/L1 |
| 48 | `vonk-forge/qwen-image-2512-lightning-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/image/qwen-image-lightning-diffusers` | qwen-image | bf16, none | 54.5 GiB |  | yes | P24/L2 |
| 49 | `vonk-forge/qwen-image-edit-2511-comfyui-single` | 1 | `image-video-workflows` | `comfyui` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/media/comfyui-core` | qwen-image-edit | bf16 | 47.0 GiB |  | yes | P25/L1 |
| 50 | `vonk-forge/qwen-image-edit-2511-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/image/qwen-image-edit-2511-diffusers` | qwen-image-edit | bf16 | 53.8 GiB |  | yes | P25/L2 |
| 51 | `vonk-forge/qwen-image-edit-2511-lightning-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/image/qwen-image-lightning-diffusers` | qwen-image, qwen-image-edit | bf16, none | 54.5 GiB |  | yes | P26/L1 |
| 52 | `vonk-forge/qwen-image-layered-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/image/qwen-image-layered-diffusers` | qwen-image | bf16 | 53.8 GiB |  | yes | P26/L2 |
| 53 | `vonk-forge/wan-2-2-i2v-14b-comfyui-single` | 1 | `image-video-workflows` | `comfyui` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/media/comfyui-core` | wan-2-2 | fp8 | 33.1 GiB | vonk-forge/wan-2-2-ti2v-5b-comfyui-single |  | P27/L1 |
| 54 | `vonk-forge/wan-2-2-t2v-14b-comfyui-single` | 1 | `image-video-workflows` | `comfyui` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/media/comfyui-core` | wan-2-2 | fp8 | 33.1 GiB | vonk-forge/wan-2-2-ti2v-5b-comfyui-single |  | P27/L2 |
| 55 | `vonk-forge/wan-dancer-14b-disk-offload-pytorch-single` | 1 | `image-video-workflows` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/video/wan-dancer-diffsynth-disk` | wan-dancer | none | 79.8 GiB |  | yes | P28/L1 |
| 56 | `vonk-forge/deepseek-v4-flash-0731-sparkinfer-target-only-canary-single` | 1 | `specialized-fork-engines` | `vllm` | `ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@2e077489a83a` | `adapters/deepseek/sparkinfer-target-only-single` | deepseek-flash | reap-k216-exl3-trellis-3_0bpw | 99.5 GiB |  | yes | P28/L2 |
| 57 | `vonk-forge/ling-3-0-flash-dspark-sglang-single` | 1 | `sglang` | `sglang` | `docker.io/lmsysorg/sglang@fc960102f1d2` | `adapters/ling/flash-dspark-single` | ling-3-0 | int4-group32, none | 74.3 GiB |  | yes | P29/L1 |
| 58 | `vonk-forge/deepseek-v4-flash-0731-mia-sparkinfer-single` | 1 | `specialized-fork-engines` | `vllm` | `ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@2e077489a83a` | `adapters/deepseek/mia-sparkinfer-single` | deepseek-flash | reap-k216-exl3-trellis-3_0bpw | 99.5 GiB |  | yes | P29/L2 |
| 59 | `vonk-forge/hunyuan-video-foley-xl-pytorch-single` | 1 | `audio-multimodal-pipelines` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/audio/hunyuan-video-foley-native` | clap, hunyuan-video-foley, siglip2 | none | 19.5 GiB |  | yes | P30/L1 |
| 60 | `vonk-forge/hunyuan-video-foley-xxl-pytorch-single` | 1 | `audio-multimodal-pipelines` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/audio/hunyuan-video-foley-native` | clap, hunyuan-video-foley, siglip2 | none | 19.5 GiB | vonk-forge/hunyuan-video-foley-xl-pytorch-single |  | P30/L2 |
| 61 | `vonk-forge/hunyuanocr-1-5-vllm-dflash-single` | 1 | `specialized-fork-engines` | `pytorch-pipeline` | `docker.io/vllm/vllm-openai@1c8e60a0841b` | `adapters/ocr/hunyuanocr-1-5-vllm-dflash` | hunyuanocr | none | 2.4 GiB |  | yes | P31/L1 |
| 62 | `vonk-forge/hunyuan3d-omni-pytorch-single` | 1 | `three-d-pipelines` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@a52783d8d73a` | `adapters/three-d/hunyuan3d-omni` | dinov2, hunyuan3d | none | 26.2 GiB |  | yes | P31/L2 |
| 63 | `vonk-forge/hunyuan-video-15-distilled-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/video/hunyuan-video-15-native` | hunyuan-video | none | 49.7 GiB | vonk-forge/hunyuan-video-15-i2v-step-distilled-diffusers-single |  | P32/L1 |
| 64 | `vonk-forge/hunyuan-video-15-i2v-step-distilled-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/video/hunyuan-video-15-native` | hunyuan-video | none | 32.3 GiB |  | yes | P32/L2 |
| 65 | `vonk-forge/hunyuan-video-15-t2v-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/video/hunyuan-video-15-native` | hunyuan-video | none | 49.7 GiB | vonk-forge/hunyuan-video-15-i2v-step-distilled-diffusers-single |  | P33/L1 |
| 66 | `vonk-forge/minimax-h3-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/video/minimax-h3-modular-diffusers` | minimax-h3 | none | 464.2 GiB |  | yes | P33/L2 |
| 67 | `vonk-forge/minimax-h3-fl2va-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/video/minimax-h3-fl2va-modular-diffusers` | minimax-h3 | none | 134.2 GiB |  | yes | P34/L1 |
| 68 | `vonk-forge/ltx-2-5-22b-distilled-bf16-diffusers-single` | 1 | `image-video-workflows` | `diffusers` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/video/ltx25-diffusers` | ltx | none | 65.3 GiB |  | yes | P34/L2 |
| 69 | `vonk-forge/nemotron-3-5-lightning-30b-a3b-vllm-dspark-latency-single` | 1 | `specialized-fork-engines` | `vllm` | `docker.io/vllm/vllm-openai@1c8e60a0841b` | `adapters/nvidia/nemotron` | nemotron | nvfp4, w4a16-nvfp4 | 21.4 GiB |  | yes | P35/L1 |
| 70 | `vonk-forge/wan-dancer-14b-pytorch-single` | 1 | `image-video-workflows` | `pytorch-pipeline` | `nvcr.io/nvidia/cuda@36050649ad1a` | `adapters/video/wan-dancer-native` | wan-dancer | none | 79.8 GiB |  | yes | P35/L2 |
| 71 | `vonk-forge/deepseek-v4-flash-0731-sparkinfer-single` | 1 | `specialized-fork-engines` | `vllm` | `ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@2e077489a83a` | `adapters/deepseek/sparkinfer-single` | deepseek-flash | reap-k216-exl3-trellis-3_0bpw | 99.5 GiB |  | yes | P36/L1 |
| 72 | `vonk-forge/laguna-s-2-1-nvfp4-vllm-single` | 1 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@1c8e60a0841b` | `adapters/llm/laguna-s-vllm` | poolside | nvfp4 | 92.9 GiB | vonk-forge/laguna-s-2-1-nvfp4-vllm-low-memory-canary-single |  | P36/L2 |
| 73 | `vonk-forge/deepseek-v4-flash-0731-mia-dual` | 2 | `specialized-fork-engines` | `vllm` | `ghcr.io/anemll/dspark-vllm-gx10@a83948492cf1` | `adapters/deepseek/mia-vllm` | deepseek-flash | moe-experts-fp4-remaining-fp8 | 155.4 GiB |  | yes | X |
| 74 | `vonk-forge/deepseek-v4-flash-vision-exp-mia-dual` | 2 | `specialized-fork-engines` | `vllm` | `ghcr.io/anemll/dspark-vllm-gx10@a83948492cf1` | `adapters/deepseek/mia-vllm-vision` | deepseek-flash | moe-experts-fp4-remaining-fp8-vision-bf16 | 156.3 GiB |  | yes | X |
| 75 | `vonk-forge/glm-5-3-flash-exl3-dflash2-vllm-dual` | 2 | `specialized-fork-engines` | `vllm` | `docker.io/vllm/vllm-openai@905c02933be6` | `adapters/glm/mia-exl3-dflash2-dual` | glm-5-3-dflash, glm-5-3-mia | exl3-tr3-4bpw-plus-bf16-dflash2, none | 168.0 GiB |  | yes | X |
| 76 | `vonk-forge/inkling-small-nvfp4-sglang-dual` | 2 | `sglang` | `sglang` | `docker.io/lmsysorg/sglang@bbedab8cbf2d` | `adapters/inkling/small-dual` | inkling | nvfp4 | 159.0 GiB |  | yes | X |
| 77 | `vonk-forge/qwen3-8-flash-next-nvfp4-sglang-dual` | 2 | `sglang` | `sglang` | `docker.io/lmsysorg/sglang@14ed58251858` | `adapters/qwen/flash-next-sglang-dual` | qwen3-8-radixark | nvfp4-w4a4-routed-experts | 126.0 GiB |  | yes | X |
| 78 | `vonk-forge/qwen3-8-flash-next-nvfp4-vllm-dual` | 2 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@3b0e188ffceb` | `adapters/qwen/flash-next-vllm-dual` | qwen3-8-radixark | nvfp4-w4a4-routed-experts | 126.0 GiB |  | yes | X |
| 79 | `vonk-forge/glm-5-3-flash-nvfp4-vllm-dual` | 2 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@905c02933be6` | `adapters/glm/glm53-sm121` | glm-5-3-libertai | nvfp4-weight-only-routed-experts | 181.3 GiB |  | yes | X |
| 80 | `vonk-forge/glm-5-3-flash-nvfp4-ablit-l15-43-dflash2-vllm-dual` | 2 | `specialized-fork-engines` | `vllm` | `ghcr.io/tonyd2wild/vllm-glm53-flash@4def0ef644cb` | `adapters/glm/tonyd2wild-dflash2-dual` | glm-5-3-dflash, glm-5-3-redhat-compressed | compressed-tensors-nvfp4-w4a4, none | 188.7 GiB |  | yes | X |
| 81 | `vonk-forge/glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual` | 2 | `specialized-fork-engines` | `vllm` | `ghcr.io/drowzeys/keys-vllm-glm53-flash-nvfp4-ablit@f722ec19d826` | `adapters/glm/drowzeys-glm53-1m` | glm-5-3-libertai | nvfp4-weight-only-routed-experts | 181.3 GiB |  | yes | X |
| — | `vonk-forge/glm-5-2-aqlm-vllm-triple` | 3 | `specialized-fork-engines` | `vllm` | `ghcr.io/miaai-lab/glm-5.2-nvfp4-triple-dgx-sparks@f8f350d46b33` | `adapters/glm/mia-triple` | glm-5-2 | nvfp4_aqlm_hybrid | 272.5 GiB | — |  | — |
| — | `vonk-forge/glm-5-2-quanttrio-vllm-four` | 4 | `specialized-fork-engines` | `vllm` | `ghcr.io/drowzeys/vllm-node-tf5-glm52-b12x@e006935eb4f8` | `adapters/glm/eugr-four` | glm-5-2 | compressed_tensors_w4a16_w8a16 | 377.7 GiB | — |  | — |
| — | `vonk-forge/glm-5-3-flash-nvfp4-vllm-four` | 4 | `vllm` | `vllm` | `docker.io/vllm/vllm-openai@905c02933be6` | `adapters/glm/tonyd2wild-glm53-tp4-current` | glm-5-3-dflash, glm-5-3-libertai | none, nvfp4-weight-only-routed-experts | 183.5 GiB | — |  | — |
| — | `vonk-forge/inkling-975b-a41b-nvfp4-sglang-eight` | 8 | `sglang` | `sglang` | `docker.io/lmsysorg/sglang@c60f221f8f42` | `adapters/llm/inkling-sglang-eight` | inkling | nvfp4 | 551.4 GiB | — |  | — |

## Representative selection

Each risk group below is one exact stack plus model family, quantization and node count. The representative is the member with the smallest declared artifact footprint; every other member keeps its own physical inference and assertions.

| Risk group | Group | Representative | Members | Artifacts |
|---|---|---|---|---:|
| `04952749bc7672cc` | `specialized-fork-engines` | `vonk-forge/deepseek-v4-flash-0731-mia-dual` | 1 | 155.4 GiB |
| `0934c39674725032` | `image-video-workflows` | `vonk-forge/qwen-image-edit-2511-lightning-diffusers-single` | 1 | 54.5 GiB |
| `0aa7abdd1bd35b9e` | `vllm` | `vonk-forge/muse-glimmer-30b-bf16-vllm-single` | 1 | 55.5 GiB |
| `136e635112bfbac4` | `image-video-workflows` | `vonk-forge/ltx-2-5-22b-distilled-bf16-diffusers-single` | 1 | 65.3 GiB |
| `15110c5ed832f900` | `audio-multimodal-pipelines` | `vonk-forge/hunyuan-video-foley-xl-pytorch-single` | 2 | 19.5 GiB |
| `19471f6e436a6077` | `image-video-workflows` | `vonk-forge/hunyuan-video-15-i2v-step-distilled-diffusers-single` | 3 | 32.3 GiB |
| `1c9d923f77b5ddf6` | `image-video-workflows` | `vonk-forge/qwen-image-edit-2511-comfyui-single` | 1 | 47.0 GiB |
| `1e360ce06b619f31` | `image-video-workflows` | `vonk-forge/wan-2-2-ti2v-5b-comfyui-single` | 3 | 16.9 GiB |
| `23cfcd6f3c01e147` | `specialized-fork-engines` | `vonk-forge/deepseek-v4-flash-0731-sparkinfer-target-only-canary-single` | 1 | 99.5 GiB |
| `24f6d9b4e340984a` | `vllm` | `vonk-forge/qwen3-6-35b-a3b-nvfp4-vllm-single` | 1 | 21.9 GiB |
| `2bb32ce6e2a2ad16` | `image-video-workflows` | `vonk-forge/wan-dancer-14b-pytorch-single` | 1 | 79.8 GiB |
| `2f703c9be1449572` | `image-video-workflows` | `vonk-forge/qwen-image-edit-2511-fp8mixed-comfyui-single` | 1 | 28.1 GiB |
| `2fc30811cab8528b` | `three-d-pipelines` | `vonk-forge/step1x-3d-label-geometry-pytorch-single` | 1 | 25.8 GiB |
| `32b43bce37ada244` | `vllm` | `vonk-forge/qwen3-6-27b-vllm-single` | 2 | 51.8 GiB |
| `3970ec72e3eb0b8b` | `specialized-fork-engines` | `vonk-forge/glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual` | 1 | 181.3 GiB |
| `3a1f0ac6d127de18` | `image-video-workflows` | `vonk-forge/ltx-2-19b-dev-fp4-pytorch-single` | 1 | 72.1 GiB |
| `3a84e7bf7fb52c23` | `vllm` | `vonk-forge/laguna-xs-2-1-nvfp4-vllm-single` | 1 | 20.1 GiB |
| `4193d56ba930c1d0` | `vllm` | `vonk-forge/ui-mate-27b-vllm-single` | 1 | 51.0 GiB |
| `465ad1e179c1ef2b` | `three-d-pipelines` | `vonk-forge/hunyuan3d-omni-pytorch-single` | 1 | 26.2 GiB |
| `49809d90751326ff` | `specialized-fork-engines` | `vonk-forge/deepseek-v4-flash-0731-mia-sparkinfer-single` | 1 | 99.5 GiB |
| `4aa6a7f6c2d91fd9` | `image-video-workflows` | `vonk-forge/ltx-2-19b-distilled-fp8-diffusers-single` | 1 | 71.6 GiB |
| `4f4ba262c1fe5451` | `specialized-fork-engines` | `vonk-forge/nemotron-3-5-lightning-30b-a3b-vllm-dspark-latency-single` | 2 | 21.4 GiB |
| `5399957259fec455` | `vllm` | `vonk-forge/qwen3-8-27b-fp8-vllm-single` | 1 | 28.8 GiB |
| `586e8937fc1513e0` | `image-video-workflows` | `vonk-forge/qwen-image-2512-diffusers-single` | 1 | 53.7 GiB |
| `58d5aaafee83c639` | `vllm` | `vonk-forge/nemotron-3-nano-30b-a3b-vllm-single` | 1 | 18.0 GiB |
| `59477203c727c468` | `image-video-workflows` | `vonk-forge/ltx-2-5-22b-distilled-fp8-cast-diffusers-single` | 1 | 65.3 GiB |
| `59ce7ba2a7616e49` | `image-video-workflows` | `vonk-forge/nvidia-qwen-image-flash-diffusers-single` | 1 | 53.7 GiB |
| `6172df60131c6c94` | `image-video-workflows` | `vonk-forge/wan-dancer-14b-disk-offload-pytorch-single` | 1 | 79.8 GiB |
| `67294747a8a7bb4f` | `sglang` | `vonk-forge/qwen3-8-flash-next-nvfp4-sglang-dual` | 1 | 126.0 GiB |
| `69159192c39b275a` | `image-video-workflows` | `vonk-forge/qwen-image-layered-diffusers-single` | 1 | 53.8 GiB |
| `69c8ba0005a5f996` | `three-d-pipelines` | `vonk-forge/triposg-pytorch-single` | 1 | 7.6 GiB |
| `6e2908dfe9601224` | `sglang` | `vonk-forge/ling-3-0-flash-dspark-sglang-single` | 1 | 74.3 GiB |
| `7d6d8046bbeb85b3` | `sglang` | `vonk-forge/qwen3-8-27b-nvfp4-dspark-sglang-single` | 1 | 25.6 GiB |
| `7dd3708ea3ba4bba` | `vllm` | `vonk-forge/ornith-1-5-35b-a3b-nvfp4-vllm-single` | 1 | 21.8 GiB |
| `8420c91db663fbc7` | `image-video-workflows` | `vonk-forge/ltx-2-19b-distilled-diffusers-single` | 1 | 86.7 GiB |
| `8473d8c0de998423` | `three-d-pipelines` | `vonk-forge/skintokens-pytorch-single` | 1 | 1.5 GiB |
| `859cd12ec38151b4` | `vllm` | `vonk-forge/qwen3-5-9b-vllm-single` | 1 | 18.0 GiB |
| `8ad586ee5b6375df` | `vllm` | `vonk-forge/nemotron-3-5-lightning-30b-a3b-vllm-single` | 1 | 20.1 GiB |
| `8af495846af6ccd1` | `specialized-fork-engines` | `vonk-forge/glm-5-3-flash-nvfp4-ablit-l15-43-dflash2-vllm-dual` | 1 | 188.7 GiB |
| `8c86a7f2269a1600` | `vllm` | `vonk-forge/nemotron-3-super-120b-a12b-vllm-single` | 1 | 80.3 GiB |
| `8e8af52a211cd88d` | `sglang` | `vonk-forge/inkling-small-nvfp4-sglang-dual` | 1 | 159.0 GiB |
| `9447fe670d1fd793` | `image-video-workflows` | `vonk-forge/qwen-image-2512-fp8-lightning-comfyui-single` | 1 | 29.6 GiB |
| `98f6598c21470f06` | `three-d-pipelines` | `vonk-forge/trellis-2-4b-pytorch-single` | 1 | 16.4 GiB |
| `9906ac901263b6af` | `vllm` | `vonk-forge/nemotron-3-nano-omni-30b-a3b-vllm-single` | 1 | 20.9 GiB |
| `9a1cbd1d789aa36b` | `vllm` | `vonk-forge/gemma-4-26b-a4b-vllm028-single` | 1 | 48.1 GiB |
| `9afc6b3b3e925605` | `three-d-pipelines` | `vonk-forge/pixal3d-pytorch-single` | 1 | 23.5 GiB |
| `9b0dc1f3c49d2bc6` | `image-video-workflows` | `vonk-forge/flux-2-klein-4b-nvfp4-comfyui-single` | 1 | 10.1 GiB |
| `a15fd544f18315a0` | `specialized-fork-engines` | `vonk-forge/deepseek-v4-flash-0731-ds4-single` | 2 | 80.8 GiB |
| `a47ffb49ae949548` | `vllm` | `vonk-forge/lfm2-5-vl-3b-vllm028-single` | 1 | 5.8 GiB |
| `aefcbbc789dcee15` | `vllm` | `vonk-forge/laguna-s-2-1-nvfp4-vllm-low-memory-canary-single` | 2 | 92.9 GiB |
| `b33b475a595aae49` | `specialized-fork-engines` | `vonk-forge/deepseek-v4-flash-0731-sparkinfer-single` | 1 | 99.5 GiB |
| `b4b6236a1dfa797d` | `image-video-workflows` | `vonk-forge/ltx-2-3-22b-distilled-1-1-diffusers-single` | 2 | 89.3 GiB |
| `baa15f7fb82a38ab` | `vllm` | `vonk-forge/lfm2-5-vl-3b-vllm-single` | 1 | 5.8 GiB |
| `bacc52a8de739935` | `specialized-fork-engines` | `vonk-forge/deepseek-v4-flash-vision-exp-mia-dual` | 1 | 156.3 GiB |
| `bb209d392efbacec` | `audio-multimodal-pipelines` | `vonk-forge/moss-vl-realtime-11b-pytorch-single` | 1 | 21.1 GiB |
| `bb93860883775bf8` | `vllm` | `vonk-forge/qwen3-8-flash-next-nvfp4-vllm-dual` | 1 | 126.0 GiB |
| `bbf7425488807d18` | `image-video-workflows` | `vonk-forge/qwen-image-edit-2511-diffusers-single` | 1 | 53.8 GiB |
| `c4fc00a1c700067d` | `image-video-workflows` | `vonk-forge/qwen-image-2512-lightning-diffusers-single` | 1 | 54.5 GiB |
| `c7bb70eb1801bda4` | `three-d-pipelines` | `vonk-forge/step1x-3d-geometry-pytorch-single` | 1 | 19.4 GiB |
| `c7c29c9ab2db9d11` | `vllm` | `vonk-forge/glm-5-3-flash-nvfp4-vllm-dual` | 1 | 181.3 GiB |
| `c88db88b41124eef` | `image-video-workflows` | `vonk-forge/qwen-image-2512-comfyui-single` | 1 | 47.0 GiB |
| `ce5261d206a88e3a` | `image-video-workflows` | `vonk-forge/mova-360p-diffusers-single` | 2 | 72.4 GiB |
| `d5f1ff5e52b5ce36` | `image-video-workflows` | `vonk-forge/flux-2-klein-4b-comfyui-single` | 1 | 15.0 GiB |
| `d69877068b9d3951` | `image-video-workflows` | `vonk-forge/minimax-h3-fl2va-diffusers-single` | 1 | 134.2 GiB |
| `db74a74c3915283c` | `vllm` | `vonk-forge/gemma-4-26b-a4b-vllm-single` | 1 | 48.1 GiB |
| `e20763439f96ec46` | `specialized-fork-engines` | `vonk-forge/hunyuanocr-1-5-vllm-dflash-single` | 1 | 2.4 GiB |
| `e63a602a175f4ad9` | `image-video-workflows` | `vonk-forge/minimax-h3-diffusers-single` | 1 | 464.2 GiB |
| `eb570b231af23f01` | `three-d-pipelines` | `vonk-forge/step1x-3d-texture-pytorch-single` | 1 | 91.2 GiB |
| `f46940e69f9242cc` | `image-video-workflows` | `vonk-forge/qwen-image-edit-2511-int8-convrot-comfyui-single` | 1 | 28.1 GiB |
| `f7d478748072001c` | `specialized-fork-engines` | `vonk-forge/glm-5-3-flash-exl3-dflash2-vllm-dual` | 1 | 168.0 GiB |

## Paired single-Spark batches

Two independent single-Spark recipes share the two Sparks under one whole-fleet profile. A batch is admissible only after fresh per-lane previews prove fit, and while both Sparks are healthy and idle for inference. With fewer than two available Sparks only exclusive single-lane work is admissible.

| Batch | Lane 1 | Lane 2 | Shared stack | Shared models | Combined artifacts |
|---:|---|---|---|---|---:|
| 1 | `vonk-forge/step1x-3d-geometry-pytorch-single` (rep) | `vonk-forge/step1x-3d-label-geometry-pytorch-single` (rep) | yes | yes | 45.2 GiB |
| 2 | `vonk-forge/flux-2-klein-4b-nvfp4-comfyui-single` (rep) | `vonk-forge/skintokens-pytorch-single` (rep) | no | no | 11.6 GiB |
| 3 | `vonk-forge/trellis-2-4b-pytorch-single` (rep) | `vonk-forge/triposg-pytorch-single` (rep) | no | no | 24.0 GiB |
| 4 | `vonk-forge/flux-2-klein-4b-comfyui-single` (rep) | `vonk-forge/pixal3d-pytorch-single` (rep) | no | no | 38.5 GiB |
| 5 | `vonk-forge/ornith-1-5-35b-a3b-nvfp4-vllm-single` (rep) | `vonk-forge/lfm2-5-vl-3b-vllm-single` (rep) | no | no | 27.6 GiB |
| 6 | `vonk-forge/lfm2-5-vl-3b-vllm028-single` (rep) | `vonk-forge/qwen3-5-9b-vllm-single` (rep) | no | no | 23.8 GiB |
| 7 | `vonk-forge/qwen3-6-35b-a3b-nvfp4-vllm-single` (rep) | `vonk-forge/qwen3-8-27b-fp8-vllm-single` (rep) | no | no | 50.6 GiB |
| 8 | `vonk-forge/laguna-xs-2-1-nvfp4-vllm-single` (rep) | `vonk-forge/wan-2-2-ti2v-5b-comfyui-single` (rep) | no | no | 37.0 GiB |
| 9 | `vonk-forge/moss-vl-realtime-11b-pytorch-single` (rep) | `vonk-forge/ltx-2-19b-dev-bf16-diffusers-single` | no | no | 115.0 GiB |
| 10 | `vonk-forge/ltx-2-19b-dev-fp4-pytorch-single` (rep) | `vonk-forge/nemotron-3-5-lightning-30b-a3b-vllm-single` (rep) | no | no | 92.2 GiB |
| 11 | `vonk-forge/nemotron-3-nano-30b-a3b-vllm-single` (rep) | `vonk-forge/nemotron-3-nano-omni-30b-a3b-vllm-single` (rep) | yes | no | 38.9 GiB |
| 12 | `vonk-forge/qwen3-8-27b-nvfp4-dspark-sglang-single` (rep) | `vonk-forge/ltx-2-3-22b-distilled-1-1-diffusers-single` (rep) | no | no | 114.9 GiB |
| 13 | `vonk-forge/qwen3-6-27b-vllm-single` (rep) | `vonk-forge/ltx-2-19b-distilled-fp8-diffusers-single` (rep) | no | no | 123.4 GiB |
| 14 | `vonk-forge/step1x-3d-texture-pytorch-single` (rep) | `vonk-forge/gemma-4-26b-a4b-vllm-single` (rep) | no | no | 139.3 GiB |
| 15 | `vonk-forge/gemma-4-26b-a4b-vllm028-single` (rep) | `vonk-forge/qwen-image-2512-fp8-lightning-comfyui-single` (rep) | no | no | 77.7 GiB |
| 16 | `vonk-forge/qwen-image-edit-2511-fp8mixed-comfyui-single` (rep) | `vonk-forge/qwen-image-edit-2511-int8-convrot-comfyui-single` (rep) | no | no | 56.2 GiB |
| 17 | `vonk-forge/ui-mate-27b-vllm-single` (rep) | `vonk-forge/muse-glimmer-30b-bf16-vllm-single` (rep) | no | no | 106.5 GiB |
| 18 | `vonk-forge/qwen3-8-27b-vllm-single` | `vonk-forge/ltx-2-5-22b-distilled-fp8-cast-diffusers-single` (rep) | no | no | 117.0 GiB |
| 19 | `vonk-forge/mova-360p-diffusers-single` (rep) | `vonk-forge/deepseek-v4-flash-0731-ds4-dspark-latency-single` | no | no | 158.7 GiB |
| 20 | `vonk-forge/deepseek-v4-flash-0731-ds4-single` (rep) | `vonk-forge/ltx-2-19b-distilled-diffusers-single` (rep) | no | no | 167.4 GiB |
| 21 | `vonk-forge/laguna-s-2-1-nvfp4-vllm-low-memory-canary-single` (rep) | `vonk-forge/mova-720p-diffusers-single` | no | no | 165.2 GiB |
| 22 | `vonk-forge/nemotron-3-5-lightning-dspark-lowmem-canary-single` | `vonk-forge/nemotron-3-super-120b-a12b-vllm-single` (rep) | no | no | 101.7 GiB |
| 23 | `vonk-forge/nvidia-qwen-image-flash-diffusers-single` (rep) | `vonk-forge/qwen-image-2512-comfyui-single` (rep) | no | no | 100.8 GiB |
| 24 | `vonk-forge/qwen-image-2512-diffusers-single` (rep) | `vonk-forge/qwen-image-2512-lightning-diffusers-single` (rep) | no | yes | 108.3 GiB |
| 25 | `vonk-forge/qwen-image-edit-2511-comfyui-single` (rep) | `vonk-forge/qwen-image-edit-2511-diffusers-single` (rep) | no | no | 100.8 GiB |
| 26 | `vonk-forge/qwen-image-edit-2511-lightning-diffusers-single` (rep) | `vonk-forge/qwen-image-layered-diffusers-single` (rep) | no | no | 108.3 GiB |
| 27 | `vonk-forge/wan-2-2-i2v-14b-comfyui-single` | `vonk-forge/wan-2-2-t2v-14b-comfyui-single` | yes | no | 66.3 GiB |
| 28 | `vonk-forge/wan-dancer-14b-disk-offload-pytorch-single` (rep) | `vonk-forge/deepseek-v4-flash-0731-sparkinfer-target-only-canary-single` (rep) | no | no | 179.3 GiB |
| 29 | `vonk-forge/ling-3-0-flash-dspark-sglang-single` (rep) | `vonk-forge/deepseek-v4-flash-0731-mia-sparkinfer-single` (rep) | no | no | 173.8 GiB |
| 30 | `vonk-forge/hunyuan-video-foley-xl-pytorch-single` (rep) | `vonk-forge/hunyuan-video-foley-xxl-pytorch-single` | yes | yes | 39.0 GiB |
| 31 | `vonk-forge/hunyuanocr-1-5-vllm-dflash-single` (rep) | `vonk-forge/hunyuan3d-omni-pytorch-single` (rep) | no | no | 28.7 GiB |
| 32 | `vonk-forge/hunyuan-video-15-distilled-diffusers-single` | `vonk-forge/hunyuan-video-15-i2v-step-distilled-diffusers-single` (rep) | yes | no | 82.0 GiB |
| 33 | `vonk-forge/hunyuan-video-15-t2v-diffusers-single` | `vonk-forge/minimax-h3-diffusers-single` (rep) | no | no | 514.0 GiB |
| 34 | `vonk-forge/minimax-h3-fl2va-diffusers-single` (rep) | `vonk-forge/ltx-2-5-22b-distilled-bf16-diffusers-single` (rep) | no | no | 199.4 GiB |
| 35 | `vonk-forge/nemotron-3-5-lightning-30b-a3b-vllm-dspark-latency-single` (rep) | `vonk-forge/wan-dancer-14b-pytorch-single` (rep) | no | no | 101.1 GiB |
| 36 | `vonk-forge/deepseek-v4-flash-0731-sparkinfer-single` (rep) | `vonk-forge/laguna-s-2-1-nvfp4-vllm-single` | no | no | 192.4 GiB |

## Exclusive schedule

Dual-Spark rows reserve both Sparks; disruptive recovery checks run exclusively after the other lane is stopped and its release is observed.

| Kind | Recipe | Nodes | Group |
|---|---|---:|---|
| `dual-spark` | `vonk-forge/deepseek-v4-flash-0731-mia-dual` | 2 | `specialized-fork-engines` |
| `dual-spark` | `vonk-forge/deepseek-v4-flash-vision-exp-mia-dual` | 2 | `specialized-fork-engines` |
| `dual-spark` | `vonk-forge/glm-5-3-flash-exl3-dflash2-vllm-dual` | 2 | `specialized-fork-engines` |
| `dual-spark` | `vonk-forge/inkling-small-nvfp4-sglang-dual` | 2 | `sglang` |
| `dual-spark` | `vonk-forge/qwen3-8-flash-next-nvfp4-sglang-dual` | 2 | `sglang` |
| `dual-spark` | `vonk-forge/qwen3-8-flash-next-nvfp4-vllm-dual` | 2 | `vllm` |
| `dual-spark` | `vonk-forge/glm-5-3-flash-nvfp4-vllm-dual` | 2 | `vllm` |
| `dual-spark` | `vonk-forge/glm-5-3-flash-nvfp4-ablit-l15-43-dflash2-vllm-dual` | 2 | `specialized-fork-engines` |
| `dual-spark` | `vonk-forge/glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual` | 2 | `specialized-fork-engines` |
| `outside-available-capacity` | `vonk-forge/glm-5-2-aqlm-vllm-triple` | 3 | `specialized-fork-engines` |
| `outside-available-capacity` | `vonk-forge/glm-5-2-quanttrio-vllm-four` | 4 | `specialized-fork-engines` |
| `outside-available-capacity` | `vonk-forge/glm-5-3-flash-nvfp4-vllm-four` | 4 | `vllm` |
| `outside-available-capacity` | `vonk-forge/inkling-975b-a41b-nvfp4-sglang-eight` | 8 | `sglang` |

## Recovery-coverage mapping

Grouped by exact build identity plus declared failure signature, because equivalence for platform recovery means an identical runtime stack and topology. The exact operator checkpoint sequence stays owned by [the plan](docs/recipe-qualification-plan-2026-09-24.md#existing-sequential-execution-gates). Shared recovery evidence is a proposal only, is never enforceable until typed runner support exists, and any cluster spanning more than one recipe still needs an explicit model-specific recovery review.

| Signature | Stack | Nodes | Backend | Rank loss | Recovery | Families | Recipes | Shared candidate | Dedicated | Model review |
|---|---|---:|---|---|---|---|---:|---|---|---|
| `07acab803a129b6f` | `b669f42d4932ae87` | 1 | `local` | `not-applicable` | `restart-entrypoint` | qwen-image-edit | 1 | no | none | n/a |
| `0a3c1d34ea70b185` | `698e2594cfd78adf` | 1 | `local` | `not-applicable` | `restart-entrypoint` | deepseek-flash | 2 | no | 1 row | required |
| `0d4dd7cc68ccd249` | `3b8814db6a852836` | 1 | `local` | `not-applicable` | `restart-entrypoint` | ltx | 1 | no | none | n/a |
| `0ddb490972854cf9` | `b6f989e1c4772d61` | 1 | `local` | `not-applicable` | `restart-entrypoint` | qwen-image | 1 | no | none | n/a |
| `10b0315f69a27bb1` | `639ea2ba3f6b3b09` | 1 | `local` | `not-applicable` | `restart-entrypoint` | qwen-image | 1 | no | none | n/a |
| `117c103b77d9d725` | `1d6cca51c9b17034` | 1 | `local` | `not-applicable` | `restart-entrypoint` | ltx | 2 | 2 rows | none | required |
| `118e96ca83fd1535` | `af7395228e9b7f0d` | 1 | `local` | `not-applicable` | `restart-entrypoint` | dinov2, hunyuan3d | 1 | no | none | n/a |
| `11c8fdb27de9c718` | `bc4fbfac5164a9f5` | 1 | `local` | `not-applicable` | `restart-entrypoint` | nemotron | 1 | no | none | n/a |
| `14959268d80e8120` | `941682cafa60f4b2` | 1 | `local` | `not-applicable` | `restart-entrypoint` | flux-2, qwen-image-edit, wan-2-2 | 6 | 6 rows | none | required |
| `1f9b825d57c49aea` | `a1044addb6048d37` | 1 | `local` | `not-applicable` | `restart-entrypoint` | ltx | 1 | no | none | n/a |
| `23ca0151854ffbcc` | `312257fa56593f3f` | 1 | `local` | `not-applicable` | `restart-entrypoint` | poolside | 1 | no | none | n/a |
| `2fe11bccaba77ce6` | `505bb91b41b4bcab` | 1 | `local` | `not-applicable` | `restart-entrypoint` | qwen-image | 1 | no | none | n/a |
| `33778e200fd8d92f` | `737082202898906a` | 1 | `local` | `not-applicable` | `restart-entrypoint` | ui-mate | 1 | no | none | n/a |
| `343a50adead42fd0` | `0795d38b73ff89cc` | 1 | `local` | `not-applicable` | `restart-entrypoint` | poolside | 2 | 2 rows | none | required |
| `3c10458b0023816c` | `05af3eb20a261835` | 1 | `local` | `not-applicable` | `restart-entrypoint` | deepseek-flash | 1 | no | 1 row | n/a |
| `3fae1dcb7747f443` | `e7137bdc5d990f3d` | 1 | `local` | `not-applicable` | `restart-entrypoint` | qwen-image-edit | 1 | no | none | n/a |
| `400005a6699b1507` | `29dcf8c3f65cc216` | 1 | `local` | `not-applicable` | `restart-entrypoint` | clap, hunyuan-video-foley, siglip2 | 2 | no | 2 rows | required |
| `4310670b2e30e94d` | `5dbce3d22855f6b8` | 1 | `local` | `not-applicable` | `restart-entrypoint` | qwen | 1 | no | none | n/a |
| `4b1853473905d3f5` | `246af82943a84e90` | 1 | `local` | `not-applicable` | `restart-entrypoint` | gemma, ornith-1-5 | 2 | 2 rows | none | required |
| `559c7d96eb28a69b` | `591c7a3e15d74d8d` | 1 | `local` | `not-applicable` | `restart-entrypoint` | moss-vl | 1 | no | none | n/a |
| `5854a14fa378cfb3` | `2e1e0a962cfd9038` | 1 | `local` | `not-applicable` | `restart-entrypoint` | nemotron | 2 | 2 rows | none | required |
| `5f3b9e4dd0fa9999` | `59e2bcb6f0f3c877` | 1 | `local` | `not-applicable` | `restart-entrypoint` | bria-rmbg, triposg | 1 | no | none | n/a |
| `5fbf5f8c074c41d1` | `748875a51bb5b552` | 1 | `local` | `not-applicable` | `restart-entrypoint` | muse-glimmer | 1 | no | none | n/a |
| `6004a5d2de254ebd` | `e5ee88e34827516a` | 2 | `mp` | `withdraw-endpoint` | `restart-worker-then-entrypoint` | deepseek-flash | 1 | no | 1 row | n/a |
| `60af82f4ae5b243f` | `4b68d058fd8c01e2` | 1 | `local` | `not-applicable` | `restart-entrypoint` | nemotron | 1 | no | none | n/a |
| `64fd91477de8b8d4` | `5a98a6a093c821c8` | 2 | `mp` | `withdraw-endpoint` | `restart-worker-then-entrypoint` | glm-5-3-libertai | 1 | no | 1 row | n/a |
| `6546ec4055e3c016` | `2c5d957947f36422` | 1 | `local` | `not-applicable` | `restart-entrypoint` | lfm2-5 | 1 | no | none | n/a |
| `6af8a66b05fdfb81` | `fe605c04239ce8f0` | 1 | `local` | `not-applicable` | `restart-entrypoint` | qwen-image-edit | 1 | no | none | n/a |
| `70c2be4e3a4047a0` | `deb57f29404b86f1` | 1 | `local` | `not-applicable` | `restart-entrypoint` | hunyuanocr | 1 | no | 1 row | n/a |
| `716ff7d18c24ae90` | `3db192b11481d7a5` | 1 | `local` | `not-applicable` | `restart-entrypoint` | nemotron | 2 | no | 2 rows | required |
| `76bf06d339c83220` | `5f9981c2eea787e8` | 1 | `local` | `not-applicable` | `restart-entrypoint` | deepseek-flash | 1 | no | 1 row | n/a |
| `7816601b1436bae6` | `8eda5e824c7c9de6` | 1 | `local` | `not-applicable` | `restart-entrypoint` | qwen-image, qwen-image-edit | 2 | 2 rows | none | required |
| `793fa6961bf1403c` | `0a66ab1f629bcbda` | 1 | `local` | `not-applicable` | `restart-entrypoint` | qwen3-8-radixark | 1 | no | 1 row | n/a |
| `7abf0df1c98ae035` | `a892ae1bd706602d` | 1 | `local` | `not-applicable` | `restart-entrypoint` | ling-3-0 | 1 | no | 1 row | n/a |
| `7f633752e5cbfd30` | `f740cacb1bf7fa2e` | 1 | `local` | `not-applicable` | `restart-entrypoint` | pixal3d, trellis | 2 | 2 rows | none | required |
| `81ccd0837da596d6` | `5751bc86debb1dcf` | 1 | `local` | `not-applicable` | `restart-entrypoint` | lfm2-5 | 1 | no | none | n/a |
| `867c7237f0c75eac` | `05cf80b2e9596839` | 1 | `local` | `not-applicable` | `restart-entrypoint` | qwen | 3 | 3 rows | none | required |
| `8a461624adc89ca2` | `b80aaab3f530c083` | 2 | `native` | `withdraw-endpoint` | `restart-worker-then-entrypoint` | qwen3-8-radixark | 1 | no | none | n/a |
| `8a9f8262f738ac27` | `ea8513566fba73db` | 1 | `local` | `not-applicable` | `restart-entrypoint` | nvidia-qwen | 1 | no | none | n/a |
| `8f37723a39a5ee16` | `a53c0bc5de206755` | 2 | `mp` | `withdraw-endpoint` | `restart-worker-then-entrypoint` | glm-5-3-dflash, glm-5-3-redhat-compressed | 1 | no | 1 row | n/a |
| `926734c51ed852da` | `93f0317b0fd9ed1f` | 2 | `mp` | `withdraw-endpoint` | `restart-worker-then-entrypoint` | deepseek-flash | 1 | no | 1 row | n/a |
| `97564b842b085b0f` | `eeb2c262a28ee227` | 2 | `ray` | `withdraw-endpoint` | `restart-worker-then-entrypoint` | glm-5-3-libertai | 1 | no | none | n/a |
| `986a7805aac40950` | `2f0bc4170f805896` | 1 | `local` | `not-applicable` | `restart-entrypoint` | ltx | 1 | no | none | n/a |
| `99d8445ac9d8bb85` | `d8fa3ac3bc56dc30` | 1 | `local` | `not-applicable` | `restart-entrypoint` | qwen-image | 1 | no | none | n/a |
| `a0c970cc4bf662c4` | `b2e9dcb6d24791aa` | 1 | `local` | `not-applicable` | `restart-entrypoint` | wan-dancer | 1 | no | 1 row | n/a |
| `a1bc25bae04d5f38` | `0f2b8f0e0e7a659a` | 1 | `local` | `not-applicable` | `restart-entrypoint` | hunyuan-video | 3 | 3 rows | none | required |
| `b3754860320fccaf` | `53dfd7fac4641a1a` | 1 | `local` | `not-applicable` | `restart-entrypoint` | deepseek-flash | 1 | no | 1 row | n/a |
| `b50d5202b98a3af7` | `46fe542f29231208` | 2 | `native` | `withdraw-endpoint` | `restart-worker-then-entrypoint` | qwen3-8-radixark | 1 | no | 1 row | n/a |
| `c33c26e934e3445e` | `35415bfc23aa9746` | 2 | `native` | `withdraw-endpoint` | `restart-worker-then-entrypoint` | inkling | 1 | no | none | n/a |
| `c7ea20f951ec79f2` | `73ae50d2534b614c` | 1 | `local` | `not-applicable` | `restart-entrypoint` | wan-dancer | 1 | no | 1 row | n/a |
| `c94df845941a3608` | `0e19e5d517e2949f` | 1 | `local` | `not-applicable` | `restart-entrypoint` | qwen-image | 1 | no | none | n/a |
| `d099ad5d423e0144` | `eccc20cb41ed56e4` | 1 | `local` | `not-applicable` | `restart-entrypoint` | minimax-h3 | 1 | no | none | n/a |
| `e0822af3be34f3bb` | `f083ef3e7f3069cf` | 1 | `local` | `not-applicable` | `restart-entrypoint` | gemma | 1 | no | none | n/a |
| `e340267d27df6f1b` | `f02bc020c58cc941` | 1 | `local` | `not-applicable` | `restart-entrypoint` | minimax-h3 | 1 | no | none | n/a |
| `ef7e51c0912c89fc` | `9ada5d0daba33ad5` | 1 | `local` | `not-applicable` | `restart-entrypoint` | skintokens | 1 | no | none | n/a |
| `f8d6c153647600dd` | `130b0be2b78bd594` | 1 | `local` | `not-applicable` | `restart-entrypoint` | clip-vit-large-patch14, dinov2-with-registers, sdxl-vae, stable-diffusion-xl, step1x-3d | 3 | 3 rows | none | required |
| `fbae195e0d7e7085` | `a2c5434cda9104d7` | 1 | `local` | `not-applicable` | `restart-entrypoint` | mova | 2 | 2 rows | none | required |
| `fc18812788600136` | `2bb4cf772ad5a449` | 1 | `local` | `not-applicable` | `restart-entrypoint` | ltx | 2 | 2 rows | none | required |
| `ffdfa2b3f5ff6398` | `749d0d3547ad3799` | 2 | `mp` | `withdraw-endpoint` | `restart-worker-then-entrypoint` | glm-5-3-dflash, glm-5-3-mia | 1 | no | 1 row | n/a |

## Bound-authority divergence

The current catalog index is byte-identical to the index bound by the authority.

Every in-scope row matches its bound authority identity.

## Not claimed here

- Cache readiness: missing model or image assets remain actionable blockers owned by the plan's file-closure section and the live cache surface.
- Structural validity: passing contract validation is not compilation, build or image evidence.
- Deployment: the installed runner is still the sequential one-recipe path.
- Physical acceptance: every recipe still needs its own attributable inference, smoke and recovery evidence.
- Capacity: declared artifact footprints are recipe declarations, not a live fit decision; admission still requires a fresh whole-fleet preview.
