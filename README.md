# Vonk Forge models and recipes

## The recipe book at a glance

A **recipe** is one tested way to run a model on NVIDIA DGX Spark: the model files, the engine and its pinned software, the launch settings and the number of Sparks. Pick a model family, then an engine and a Spark count.

- **Versions:** a recipe runs the upstream project's version when it publishes one (otherwise the recipe's own), with the release date. Only the newest revision is kept, refreshed in place; exact pins (commits, image digests, file hashes) sit underneath.
- **Options:** settings such as context length are declared per recipe and say whether changing them needs a restart or a rebuild.
- **Licences are informational:** the licence and any regional limits are shown so you can decide; they never block a recipe.
- **Full catalog:** browse everything at [vonkforge.ai/recipes](https://vonkforge.ai/recipes).

<!-- overview:start -->
**90 recipes** for **29 model families**, using **96 Models**. This section is generated from `recipes/`, `models/` and [`creators.json`](creators.json) by `tools/build-readme-overview`; do not edit it by hand.

### Engines and models

Number of recipes per model family and engine.

| Model family | vLLM | SGLang | TensorRT-LLM | llama.cpp | TensorFold | ds4 | diffusers | ComfyUI | pytorch-pipeline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek V4 Flash | 5 |  |  |  |  | 3 |  |  |  |
| FLUX.2 |  |  |  |  |  |  |  | 2 |  |
| Gemma | 2 |  |  |  |  |  |  |  |  |
| GLM | 9 |  |  |  |  |  |  |  |  |
| Hunyuan3D |  |  |  |  |  |  |  |  | 1 |
| HunyuanOCR |  |  |  |  |  |  |  |  | 1 |
| HunyuanVideo |  |  |  |  |  |  | 3 |  |  |
| HunyuanVideo Foley |  |  |  |  |  |  |  |  | 2 |
| Inkling |  | 2 |  |  |  |  |  |  |  |
| Laguna | 3 |  |  |  |  |  |  |  |  |
| LFM2.5 | 2 |  |  |  |  |  |  |  |  |
| Ling |  | 1 |  |  |  |  |  |  |  |
| LTX |  |  |  |  |  |  | 2 |  | 5 |
| MiMo | 1 |  |  |  |  |  |  |  |  |
| MiniMax H3 |  |  |  |  |  |  | 2 |  |  |
| MOSS-VL |  |  |  |  |  |  |  |  | 1 |
| MOVA |  |  |  |  |  |  |  |  | 2 |
| Muse Glimmer | 1 |  |  |  |  |  |  |  |  |
| Nemotron | 6 |  |  |  |  |  |  |  |  |
| Ornith | 1 |  |  |  |  |  |  |  |  |
| Pixal3D |  |  |  |  |  |  |  |  | 1 |
| Qwen (text and vision) | 7 | 1 |  |  | 1 |  |  |  |  |
| Qwen Image |  |  |  |  |  |  | 6 | 5 |  |
| SkinTokens |  |  |  |  |  |  |  |  | 1 |
| Step1X-3D |  |  |  |  |  |  |  |  | 3 |
| TRELLIS |  |  |  |  |  |  |  |  | 1 |
| TripoSG |  |  |  |  |  |  |  |  | 1 |
| UI-Mate | 1 |  |  |  |  |  |  |  |  |
| Wan |  |  |  |  |  |  |  | 3 | 2 |
| **Total** | **38** | **4** | **0** | **0** | **1** | **3** | **13** | **10** | **21** |

### Spark variants

Which Spark counts each family runs on, and who provides them.

| Model family | 1 Spark | 2 Sparks | 3 Sparks | 4 Sparks | 8 Sparks |
| --- | --- | --- | --- | --- | --- |
| DeepSeek V4 Flash | 0xSero, antirez, MiaAI-Lab | MiaAI-Lab |  |  |  |
| FLUX.2 | black-forest-labs, Comfy-Org |  |  |  |  |
| Gemma | google |  |  |  |  |
| GLM | r0b0tlab | drowzeys (keyz), MiaAI-Lab, r0b0tlab, tonyd2wild | MiaAI-Lab | drowzeys (keyz), tonyd2wild |  |
| Hunyuan3D | tencent |  |  |  |  |
| HunyuanOCR | tencent |  |  |  |  |
| HunyuanVideo | hunyuanvideo-community |  |  |  |  |
| HunyuanVideo Foley | tencent |  |  |  |  |
| Inkling |  | sgl-project |  |  | sgl-project |
| Laguna | poolside |  |  |  |  |
| LFM2.5 | LiquidAI |  |  |  |  |
| Ling | MiaAI-Lab |  |  |  |  |
| LTX | Lightricks |  |  |  |  |
| MiMo |  | tonyd2wild |  |  |  |
| MiniMax H3 | MiniMaxAI |  |  |  |  |
| MOSS-VL | OpenMOSS-Team |  |  |  |  |
| MOVA | OpenMOSS-Team |  |  |  |  |
| Muse Glimmer | meta-models |  |  |  |  |
| Nemotron | NVIDIA playbooks |  |  |  |  |
| Ornith | ornith-ai |  |  |  |  |
| Pixal3D | TencentARC |  |  |  |  |
| Qwen (text and vision) | MiaAI-Lab, NVIDIA playbooks, Qwen | MiaAI-Lab |  |  |  |
| Qwen Image | Comfy-Org, lightx2v, NVIDIA playbooks, Qwen |  |  |  |  |
| SkinTokens | VAST-AI |  |  |  |  |
| Step1X-3D | stepfun-ai |  |  |  |  |
| TRELLIS | microsoft |  |  |  |  |
| TripoSG | VAST-AI |  |  |  |  |
| UI-Mate | Tencent |  |  |  |  |
| Wan | Comfy-Org, modelscope, Wan-Video |  |  |  |  |

### Tracked creators

Recipes are credited to the creator whose repository they come from (the recipe's source reference), or else to a tracked creator named in its attribution.

| Creator | Focus | Engines | Recipes |
| --- | --- | --- | ---: |
| [MiaAI-Lab](https://github.com/MiaAI-Lab) | Spark cookbooks for SGLang, vLLM and TensorFold; DSpark and EXL3 builds; Qwen, GLM, DeepSeek, Ling. | SGLang, TensorFold, vLLM | 11 |
| [tonyd2wild](https://github.com/tonyd2wild) | Multi-Spark (2 and 4) vLLM recipes for large MoE models: GLM, MiMo. | vLLM | 3 |
| [r0b0tlab](https://github.com/r0b0tlab) | vLLM on GB10/SM121: EXL3 kernels, DFlash2 speculative decoding, GLM and Nemotron. | vLLM | 2 |
| [drowzeys (keyz)](https://github.com/drowzeys) | Large multi-Spark vLLM builds with prebuilt images: GLM, MiMo, abliterated variants. | vLLM | 2 |
| [NVIDIA playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) | Official DGX Spark playbooks and NVIDIA model releases: Nemotron, NVFP4 checkpoints. | diffusers, vLLM | 8 |
| [eugr](https://github.com/eugr) | spark-vllm-docker: vLLM container builds for dual DGX Spark; llama-benchy benchmarking. |  | 0 |
| [sfxnz](https://github.com/sfxnz) | Dual-Spark vLLM TP=2 cookbooks: Qwen3.8, GLM 5.3, DeepSeek V4.1 EXL3. |  | 0 |
| [0xSero](https://github.com/0xSero) | SparkInfer builds and local-ai-recipe-kit; DeepSeek V4 Flash on one Spark. | vLLM | 2 |
| [antirez](https://github.com/antirez/ds4) | ds4: the DeepSeek 4 Flash inference engine (Metal, CUDA, ROCm). | ds4 | 3 |

<details><summary>Other upstream sources</summary>

| Source | Recipes |
| --- | ---: |
| [black-forest-labs](https://huggingface.co/black-forest-labs) | 1 |
| [Comfy-Org](https://huggingface.co/Comfy-Org) | 8 |
| [google](https://huggingface.co/google) | 2 |
| [hunyuanvideo-community](https://huggingface.co/hunyuanvideo-community) | 3 |
| [Lightricks](https://huggingface.co/Lightricks) | 7 |
| [lightx2v](https://huggingface.co/lightx2v) | 3 |
| [LiquidAI](https://huggingface.co/LiquidAI) | 2 |
| [meta-models](https://huggingface.co/meta-models) | 1 |
| [microsoft](https://github.com/microsoft) | 1 |
| [MiniMaxAI](https://huggingface.co/MiniMaxAI) | 2 |
| [modelscope](https://github.com/modelscope) | 1 |
| [OpenMOSS-Team](https://huggingface.co/OpenMOSS-Team) | 3 |
| [ornith-ai](https://huggingface.co/ornith-ai) | 1 |
| [poolside](https://huggingface.co/poolside) | 3 |
| [Qwen](https://huggingface.co/Qwen) | 7 |
| [sgl-project](https://github.com/sgl-project) | 2 |
| [stepfun-ai](https://huggingface.co/stepfun-ai) | 3 |
| [tencent](https://huggingface.co/tencent) | 5 |
| [TencentARC](https://github.com/TencentARC) | 1 |
| [VAST-AI](https://huggingface.co/VAST-AI) | 2 |
| [Wan-Video](https://github.com/Wan-Video) | 1 |

</details>

### Recipes by family

<details><summary>DeepSeek V4 Flash (8)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [deepseek-v4-1-flash-ds4-single](recipes/deepseek-v4-1-flash-ds4-single.json) | 1.0.0 | 2026-09-29 | ds4 | 1 | antirez |
| [deepseek-v4-1-flash-exl3-mia-dual](recipes/deepseek-v4-1-flash-exl3-mia-dual.json) | 1.0.1 | 2026-09-19 | vLLM | 2 | MiaAI-Lab |
| [deepseek-v4-flash-0731-ds4-dspark-latency-single](recipes/deepseek-v4-flash-0731-ds4-dspark-latency-single.json) | 1.3.0 | 2026-09-20 | ds4 | 1 | antirez |
| [deepseek-v4-flash-0731-ds4-single](recipes/deepseek-v4-flash-0731-ds4-single.json) | 2.2.0 | 2026-09-20 | ds4 | 1 | antirez |
| [deepseek-v4-flash-0731-mia-sparkinfer-single](recipes/deepseek-v4-flash-0731-mia-sparkinfer-single.json) | 1.1.5 | 2026-09-06 | vLLM | 1 | MiaAI-Lab |
| [deepseek-v4-flash-0731-sparkinfer-single](recipes/deepseek-v4-flash-0731-sparkinfer-single.json) | 0.1.6 | 2026-09-03 | vLLM | 1 | 0xSero |
| [deepseek-v4-flash-0731-sparkinfer-target-only-canary-single](recipes/deepseek-v4-flash-0731-sparkinfer-target-only-canary-single.json) | 0.1.5 | 2026-09-03 | vLLM | 1 | 0xSero |
| [deepseek-v4-flash-vision-exp-mia-dual](recipes/deepseek-v4-flash-vision-exp-mia-dual.json) | 1.2.1 | 2026-09-16 | vLLM | 2 | MiaAI-Lab |

</details>

<details><summary>FLUX.2 (2)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [flux-2-klein-4b-comfyui-single](recipes/flux-2-klein-4b-comfyui-single.json) | 1.1.0 | 2026-09-28 | ComfyUI | 1 | Comfy-Org |
| [flux-2-klein-4b-nvfp4-comfyui-single](recipes/flux-2-klein-4b-nvfp4-comfyui-single.json) | 1.1.1 | 2026-09-28 | ComfyUI | 1 | black-forest-labs |

</details>

<details><summary>Gemma (2)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [gemma-4-26b-a4b-vllm-single](recipes/gemma-4-26b-a4b-vllm-single.json) | 2.1.0 | 2026-09-28 | vLLM | 1 | google |
| [gemma-4-26b-a4b-vllm028-single](recipes/gemma-4-26b-a4b-vllm028-single.json) | 1.1.1 | 2026-09-28 | vLLM | 1 | google |

</details>

<details><summary>GLM (9)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [glm-5-2-aqlm-vllm-triple](recipes/glm-5-2-aqlm-vllm-triple.json) | 4.5 | 2026-07-27 | vLLM | 3 | MiaAI-Lab |
| [glm-5-2-quanttrio-vllm-four](recipes/glm-5-2-quanttrio-vllm-four.json) | 2.0.6 | 2026-09-03 | vLLM | 4 | drowzeys (keyz) |
| [glm-5-3-flash-exl3-dflash2-r0b0tlab-vllm-single](recipes/glm-5-3-flash-exl3-dflash2-r0b0tlab-vllm-single.json) | 1.0.1 | 2026-09-20 | vLLM | 1 | r0b0tlab |
| [glm-5-3-flash-exl3-dflash2-vllm-dual](recipes/glm-5-3-flash-exl3-dflash2-vllm-dual.json) | 1.7.3 | 2026-09-28 | vLLM | 2 | MiaAI-Lab |
| [glm-5-3-flash-nvfp4-ablit-l15-43-dflash2-vllm-dual](recipes/glm-5-3-flash-nvfp4-ablit-l15-43-dflash2-vllm-dual.json) | 1.0.9 | 2026-09-24 | vLLM | 2 | tonyd2wild |
| [glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual](recipes/glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual.json) | 1.1.6 | 2026-09-06 | vLLM | 2 | drowzeys (keyz) |
| [glm-5-3-flash-nvfp4-vllm-dual](recipes/glm-5-3-flash-nvfp4-vllm-dual.json) | 1.3.2 | 2026-09-28 | vLLM | 2 | MiaAI-Lab |
| [glm-5-3-flash-nvfp4-vllm-four](recipes/glm-5-3-flash-nvfp4-vllm-four.json) | 1.3.3 | 2026-09-21 | vLLM | 4 | tonyd2wild |
| [glm-5-3-flash-nvidia-nvfp4-dflash2-vllm-dual](recipes/glm-5-3-flash-nvidia-nvfp4-dflash2-vllm-dual.json) | 1.0.2 | 2026-09-11 | vLLM | 2 | r0b0tlab |

</details>

<details><summary>Hunyuan3D (1)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [hunyuan3d-omni-pytorch-single](recipes/hunyuan3d-omni-pytorch-single.json) | 2.0.13 | 2026-09-28 | pytorch-pipeline | 1 | tencent |

</details>

<details><summary>HunyuanOCR (1)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [hunyuanocr-1-5-vllm-dflash-single](recipes/hunyuanocr-1-5-vllm-dflash-single.json) | 1.3.0 | 2026-09-28 | pytorch-pipeline | 1 | tencent |

</details>

<details><summary>HunyuanVideo (3)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [hunyuan-video-15-distilled-diffusers-single](recipes/hunyuan-video-15-distilled-diffusers-single.json) | 1.5.0 | 2026-09-28 | diffusers | 1 | hunyuanvideo-community |
| [hunyuan-video-15-i2v-step-distilled-diffusers-single](recipes/hunyuan-video-15-i2v-step-distilled-diffusers-single.json) | 1.5.0 | 2026-09-28 | diffusers | 1 | hunyuanvideo-community |
| [hunyuan-video-15-t2v-diffusers-single](recipes/hunyuan-video-15-t2v-diffusers-single.json) | 1.5.0 | 2026-09-28 | diffusers | 1 | hunyuanvideo-community |

</details>

<details><summary>HunyuanVideo Foley (2)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [hunyuan-video-foley-xl-pytorch-single](recipes/hunyuan-video-foley-xl-pytorch-single.json) | 1.4.8 | 2026-09-28 | pytorch-pipeline | 1 | tencent |
| [hunyuan-video-foley-xxl-pytorch-single](recipes/hunyuan-video-foley-xxl-pytorch-single.json) | 1.4.8 | 2026-09-28 | pytorch-pipeline | 1 | tencent |

</details>

<details><summary>Inkling (2)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [inkling-975b-a41b-nvfp4-sglang-eight](recipes/inkling-975b-a41b-nvfp4-sglang-eight.json) | 0.5.20 | 2026-09-18 | SGLang | 8 | sgl-project |
| [inkling-small-nvfp4-sglang-dual](recipes/inkling-small-nvfp4-sglang-dual.json) | 0.5.20 | 2026-09-18 | SGLang | 2 | sgl-project |

</details>

<details><summary>Laguna (3)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [laguna-s-2-1-nvfp4-vllm-low-memory-canary-single](recipes/laguna-s-2-1-nvfp4-vllm-low-memory-canary-single.json) | 1.1.0 | 2026-09-28 | vLLM | 1 | poolside |
| [laguna-s-2-1-nvfp4-vllm-single](recipes/laguna-s-2-1-nvfp4-vllm-single.json) | 1.1.0 | 2026-09-28 | vLLM | 1 | poolside |
| [laguna-xs-2-1-nvfp4-vllm-single](recipes/laguna-xs-2-1-nvfp4-vllm-single.json) | 1.1.0 | 2026-09-28 | vLLM | 1 | poolside |

</details>

<details><summary>LFM2.5 (2)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [lfm2-5-vl-3b-vllm-single](recipes/lfm2-5-vl-3b-vllm-single.json) | 1.2.0 | 2026-09-28 | vLLM | 1 | LiquidAI |
| [lfm2-5-vl-3b-vllm028-single](recipes/lfm2-5-vl-3b-vllm028-single.json) | 1.2.1 | 2026-09-28 | vLLM | 1 | LiquidAI |

</details>

<details><summary>Ling (1)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [ling-3-0-flash-dspark-sglang-single](recipes/ling-3-0-flash-dspark-sglang-single.json) | 1.1.0 | 2026-09-28 | SGLang | 1 | MiaAI-Lab |

</details>

<details><summary>LTX (7)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [ltx-2-19b-dev-bf16-diffusers-single](recipes/ltx-2-19b-dev-bf16-diffusers-single.json) | 3.2.10 | 2026-09-28 | pytorch-pipeline | 1 | Lightricks |
| [ltx-2-19b-dev-fp4-pytorch-single](recipes/ltx-2-19b-dev-fp4-pytorch-single.json) | 2.3.7 | 2026-09-28 | pytorch-pipeline | 1 | Lightricks |
| [ltx-2-19b-distilled-diffusers-single](recipes/ltx-2-19b-distilled-diffusers-single.json) | 3.1.10 | 2026-09-28 | pytorch-pipeline | 1 | Lightricks |
| [ltx-2-19b-distilled-fp8-diffusers-single](recipes/ltx-2-19b-distilled-fp8-diffusers-single.json) | 3.1.10 | 2026-09-28 | pytorch-pipeline | 1 | Lightricks |
| [ltx-2-3-22b-distilled-1-1-diffusers-single](recipes/ltx-2-3-22b-distilled-1-1-diffusers-single.json) | 3.2.11 | 2026-09-28 | pytorch-pipeline | 1 | Lightricks |
| [ltx-2-5-22b-distilled-bf16-diffusers-single](recipes/ltx-2-5-22b-distilled-bf16-diffusers-single.json) | 1.1.8 | 2026-09-28 | diffusers | 1 | Lightricks |
| [ltx-2-5-22b-distilled-fp8-cast-diffusers-single](recipes/ltx-2-5-22b-distilled-fp8-cast-diffusers-single.json) | 1.0.8 | 2026-09-28 | diffusers | 1 | Lightricks |

</details>

<details><summary>MiMo (1)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [mimo-v2-6-flash-rl-vllm-dual](recipes/mimo-v2-6-flash-rl-vllm-dual.json) | 1.0.1 | 2026-09-22 | vLLM | 2 | tonyd2wild |

</details>

<details><summary>MiniMax H3 (2)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [minimax-h3-diffusers-single](recipes/minimax-h3-diffusers-single.json) | 3.0.9 | 2026-09-28 | diffusers | 1 | MiniMaxAI |
| [minimax-h3-fl2va-diffusers-single](recipes/minimax-h3-fl2va-diffusers-single.json) | 1.0.9 | 2026-09-28 | diffusers | 1 | MiniMaxAI |

</details>

<details><summary>MOSS-VL (1)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [moss-vl-realtime-11b-pytorch-single](recipes/moss-vl-realtime-11b-pytorch-single.json) | 1.2.5 | 2026-09-28 | pytorch-pipeline | 1 | OpenMOSS-Team |

</details>

<details><summary>MOVA (2)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [mova-360p-diffusers-single](recipes/mova-360p-diffusers-single.json) | 2.0.12 | 2026-09-28 | pytorch-pipeline | 1 | OpenMOSS-Team |
| [mova-720p-diffusers-single](recipes/mova-720p-diffusers-single.json) | 2.0.12 | 2026-09-28 | pytorch-pipeline | 1 | OpenMOSS-Team |

</details>

<details><summary>Muse Glimmer (1)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [muse-glimmer-30b-bf16-vllm-single](recipes/muse-glimmer-30b-bf16-vllm-single.json) | 1.1.0 | 2026-09-28 | vLLM | 1 | meta-models |

</details>

<details><summary>Nemotron (6)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [nemotron-3-5-lightning-30b-a3b-vllm-dspark-latency-single](recipes/nemotron-3-5-lightning-30b-a3b-vllm-dspark-latency-single.json) | 1.2.0 | 2026-09-28 | vLLM | 1 | NVIDIA playbooks |
| [nemotron-3-5-lightning-30b-a3b-vllm-single](recipes/nemotron-3-5-lightning-30b-a3b-vllm-single.json) | 1.4.0 | 2026-09-28 | vLLM | 1 | NVIDIA playbooks |
| [nemotron-3-5-lightning-dspark-lowmem-canary-single](recipes/nemotron-3-5-lightning-dspark-lowmem-canary-single.json) | 1.1.0 | 2026-09-28 | vLLM | 1 | NVIDIA playbooks |
| [nemotron-3-nano-30b-a3b-vllm-single](recipes/nemotron-3-nano-30b-a3b-vllm-single.json) | 2.1.0 | 2026-09-28 | vLLM | 1 | NVIDIA playbooks |
| [nemotron-3-nano-omni-30b-a3b-vllm-single](recipes/nemotron-3-nano-omni-30b-a3b-vllm-single.json) | 2.1.0 | 2026-09-28 | vLLM | 1 | NVIDIA playbooks |
| [nemotron-3-super-120b-a12b-vllm-single](recipes/nemotron-3-super-120b-a12b-vllm-single.json) | 2.2.0 | 2026-09-28 | vLLM | 1 | NVIDIA playbooks |

</details>

<details><summary>Ornith (1)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [ornith-1-5-35b-a3b-nvfp4-vllm-single](recipes/ornith-1-5-35b-a3b-nvfp4-vllm-single.json) | 1.1.0 | 2026-09-28 | vLLM | 1 | ornith-ai |

</details>

<details><summary>Pixal3D (1)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [pixal3d-pytorch-single](recipes/pixal3d-pytorch-single.json) | 2.0.10 | 2026-09-28 | pytorch-pipeline | 1 | TencentARC |

</details>

<details><summary>Qwen (text and vision) (9)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [qwen3-5-9b-vllm-single](recipes/qwen3-5-9b-vllm-single.json) | 1.2.0 | 2026-09-28 | vLLM | 1 | Qwen |
| [qwen3-6-27b-vllm-single](recipes/qwen3-6-27b-vllm-single.json) | 1.2.1 | 2026-09-28 | vLLM | 1 | Qwen |
| [qwen3-6-35b-a3b-nvfp4-vllm-single](recipes/qwen3-6-35b-a3b-nvfp4-vllm-single.json) | 1.1.0 | 2026-09-28 | vLLM | 1 | NVIDIA playbooks |
| [qwen3-8-27b-fp8-vllm-single](recipes/qwen3-8-27b-fp8-vllm-single.json) | 1.2.1 | 2026-09-28 | vLLM | 1 | Qwen |
| [qwen3-8-27b-nvfp4-dspark-sglang-single](recipes/qwen3-8-27b-nvfp4-dspark-sglang-single.json) | 1.2.1 | 2026-09-09 | SGLang | 1 | MiaAI-Lab |
| [qwen3-8-27b-vllm-single](recipes/qwen3-8-27b-vllm-single.json) | 1.2.1 | 2026-09-28 | vLLM | 1 | Qwen |
| [qwen3-8-flash-next-nvfp4-vllm-dual](recipes/qwen3-8-flash-next-nvfp4-vllm-dual.json) | 2.0.1 | 2026-09-27 | vLLM | 2 | MiaAI-Lab |
| [qwen3-8-flash-next-nvfp4-vllm-single](recipes/qwen3-8-flash-next-nvfp4-vllm-single.json) | 1.0.1 | 2026-09-29 | vLLM | 1 | MiaAI-Lab |
| [qwen3-8-flash-next-tensorfold-single](recipes/qwen3-8-flash-next-tensorfold-single.json) | 0.3.7 | 2026-09-29 | TensorFold | 1 | MiaAI-Lab |

</details>

<details><summary>Qwen Image (11)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [nvidia-qwen-image-flash-diffusers-single](recipes/nvidia-qwen-image-flash-diffusers-single.json) | 1.2.9 | 2026-09-28 | diffusers | 1 | NVIDIA playbooks |
| [qwen-image-2512-comfyui-single](recipes/qwen-image-2512-comfyui-single.json) | 1.1.0 | 2026-09-28 | ComfyUI | 1 | Comfy-Org |
| [qwen-image-2512-diffusers-single](recipes/qwen-image-2512-diffusers-single.json) | 1.2.9 | 2026-09-28 | diffusers | 1 | Qwen |
| [qwen-image-2512-fp8-lightning-comfyui-single](recipes/qwen-image-2512-fp8-lightning-comfyui-single.json) | 1.1.1 | 2026-09-28 | ComfyUI | 1 | lightx2v |
| [qwen-image-2512-lightning-diffusers-single](recipes/qwen-image-2512-lightning-diffusers-single.json) | 1.2.11 | 2026-09-28 | diffusers | 1 | lightx2v |
| [qwen-image-edit-2511-comfyui-single](recipes/qwen-image-edit-2511-comfyui-single.json) | 1.1.1 | 2026-09-28 | ComfyUI | 1 | Comfy-Org |
| [qwen-image-edit-2511-diffusers-single](recipes/qwen-image-edit-2511-diffusers-single.json) | 1.1.10 | 2026-09-28 | diffusers | 1 | Qwen |
| [qwen-image-edit-2511-fp8mixed-comfyui-single](recipes/qwen-image-edit-2511-fp8mixed-comfyui-single.json) | 1.1.1 | 2026-09-28 | ComfyUI | 1 | Comfy-Org |
| [qwen-image-edit-2511-int8-convrot-comfyui-single](recipes/qwen-image-edit-2511-int8-convrot-comfyui-single.json) | 1.1.1 | 2026-09-28 | ComfyUI | 1 | Comfy-Org |
| [qwen-image-edit-2511-lightning-diffusers-single](recipes/qwen-image-edit-2511-lightning-diffusers-single.json) | 1.2.11 | 2026-09-28 | diffusers | 1 | lightx2v |
| [qwen-image-layered-diffusers-single](recipes/qwen-image-layered-diffusers-single.json) | 1.1.12 | 2026-09-28 | diffusers | 1 | Qwen |

</details>

<details><summary>SkinTokens (1)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [skintokens-pytorch-single](recipes/skintokens-pytorch-single.json) | 2.0.13 | 2026-09-28 | pytorch-pipeline | 1 | VAST-AI |

</details>

<details><summary>Step1X-3D (3)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [step1x-3d-geometry-pytorch-single](recipes/step1x-3d-geometry-pytorch-single.json) | 1.2.16 | 2026-09-28 | pytorch-pipeline | 1 | stepfun-ai |
| [step1x-3d-label-geometry-pytorch-single](recipes/step1x-3d-label-geometry-pytorch-single.json) | 1.2.16 | 2026-09-28 | pytorch-pipeline | 1 | stepfun-ai |
| [step1x-3d-texture-pytorch-single](recipes/step1x-3d-texture-pytorch-single.json) | 1.2.16 | 2026-09-28 | pytorch-pipeline | 1 | stepfun-ai |

</details>

<details><summary>TRELLIS (1)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [trellis-2-4b-pytorch-single](recipes/trellis-2-4b-pytorch-single.json) | 2.0.10 | 2026-09-28 | pytorch-pipeline | 1 | microsoft |

</details>

<details><summary>TripoSG (1)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [triposg-pytorch-single](recipes/triposg-pytorch-single.json) | 2.0.14 | 2026-09-28 | pytorch-pipeline | 1 | VAST-AI |

</details>

<details><summary>UI-Mate (1)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [ui-mate-27b-vllm-single](recipes/ui-mate-27b-vllm-single.json) | 1.2.1 | 2026-09-28 | vLLM | 1 | Tencent |

</details>

<details><summary>Wan (5)</summary>

| Recipe | Version | Released | Engine | Sparks | Creator |
| --- | --- | --- | --- | ---: | --- |
| [wan-2-2-i2v-14b-comfyui-single](recipes/wan-2-2-i2v-14b-comfyui-single.json) | 1.1.0 | 2026-09-28 | ComfyUI | 1 | Comfy-Org |
| [wan-2-2-t2v-14b-comfyui-single](recipes/wan-2-2-t2v-14b-comfyui-single.json) | 1.1.0 | 2026-09-28 | ComfyUI | 1 | Comfy-Org |
| [wan-2-2-ti2v-5b-comfyui-single](recipes/wan-2-2-ti2v-5b-comfyui-single.json) | 1.1.0 | 2026-09-28 | ComfyUI | 1 | Comfy-Org |
| [wan-dancer-14b-disk-offload-pytorch-single](recipes/wan-dancer-14b-disk-offload-pytorch-single.json) | 1.0.7 | 2026-09-28 | pytorch-pipeline | 1 | modelscope |
| [wan-dancer-14b-pytorch-single](recipes/wan-dancer-14b-pytorch-single.json) | 1.0.12 | 2026-09-28 | pytorch-pipeline | 1 | Wan-Video |

</details>
<!-- overview:end -->

## Contracts


This repository defines **what a model is** and **how to run it** in [Vonk Forge](https://vonkforge.ai). Authors write two kinds of JSON document, validated by the shared [Pydantic contracts](contracts/src/vonk_forge_contracts).

- **Model:** one exact model version and variant, its capabilities and its files.
- **Recipe:** the model files, software, settings and hardware needed to run it.

Several recipes can use the same Model—for example, with different engines or with one Spark versus two. A recipe can also use several Models when it needs companion weights.

## Model contract

[`ModelDefinition`](contracts/src/vonk_forge_contracts/model.py) describes a specific set of model files.

| Field | Contents |
| --- | --- |
| `identity` | Publisher and unique name; family, model, version and variant for browsing and grouping. |
| `metadata` | Description and tags. |
| `modalities` | The kinds of data the model handles: text, images, audio, video, 3D or embeddings. |
| `source` | Where the files come from, with an exact source revision. |
| `requires_token` | Whether downloading the files needs a provider account token (a gated repository). The token itself stays in Controller secrets. |
| `dependencies` | Exact references to companion Models. |
| `format` | Numerical precision and quantization. |
| `license` | Usage terms, attribution and any territorial notice. |
| `files` | Each file’s ID, path, content hash, byte size and purpose, such as weights or tokenizer. |
| `capabilities` | The names of the features the model supports. |

Family, model, version and variant names are **data**, not Python classes. Adding a new family or version does not require changing the contract. File hashes and sizes live here once; recipes reference them.

[See a complete Model example →](contracts/src/vonk_forge_contracts/examples/model-definition.json)

## Recipe contract

[`RecipeDefinition`](contracts/src/vonk_forge_contracts/recipe.py) describes one way to run the selected model files.

| Field | Contents |
| --- | --- |
| `identity`, `metadata` | Publisher, unique name, title, description and tags. |
| `models` | Exact Model references, selected file IDs, and where each Spark role reads those files. |
| `execution` | How to build the container image: a pinned ARM64 base image, the build context, Dockerfile, patches and the hosts the build may reach. |
| `runtime` | Engine, launch command, arguments, environment and stop timeout. |
| `settings` | Generation, embedding or job settings, including whether changing a value needs a restart or rebuild. |
| `topology` | Number of Sparks, their roles, parallelism and start order, and each role's memory and disk needs. |
| `interfaces` | How an application uses the model: an API or a file-based job. |
| `validation` | Representative requests or job inputs and their expected results. |
| `release` | The version the recipe runs and its date: the upstream project's version when it publishes one, otherwise the recipe's own. |
| `provenance` | Where the recipe came from and who should be credited. |

Both documents declare their `kind` (`model` or `recipe`). Pydantic checks their structure; the shared resolver checks that a recipe references the right Models (by the `document_sha256` of their JSON) and files. Running the declared tests checks actual model behavior.

The contracts have one semantic version, `CONTRACT_VERSION`, which is also the version of the published library release (`v2.0.0`). Recipe and Model changes update that release in place; an additive contract change publishes a new minor release and a breaking one a new major release.

Each recipe is one JSON file. Its downloadable package adds the exact Model snapshots, build sources and test fixtures it needs; it contains no model weights or container images.

Examples: [build from source](contracts/src/vonk_forge_contracts/examples/recipe-source-build.json) · [two Sparks](contracts/src/vonk_forge_contracts/examples/recipe-dual.json) · [file-based job](contracts/src/vonk_forge_contracts/examples/recipe-job.json). These use synthetic data to show the structure.

## What the Controller handles

The repository contains definitions and build sources, not model weights or container images. The Controller caches model files and images separately on local storage, distributes them to the selected Sparks, and manages starting, stopping and progress. Runtime defaults such as writable engine caches belong to the platform, so every recipe does not have to repeat them.

## Where to look

- [Create and update recipes](docs/recipe-authoring.md): the standard workflow for agents and maintainers, including upstream refreshes and version notes.
- [Pydantic definitions](contracts/src/vonk_forge_contracts): the shared source of truth for Model and Recipe structure.
- [Generated JSON Schemas](contracts/src/vonk_forge_contracts/schema): editor and non-Python tooling support.
- [Contract guide](contracts/README.md): validation, package reuse and Controller integration details.
- [Public catalog](https://vonkforge.ai/recipes): browse models and recipes.

This repository defines the first-release catalog contract. Consumers use the same shared definitions; there is no parallel legacy catalog format.
