# October 2026 model additions

These twelve recipes cover the ten model families selected in the September
19–October 17 release review. LightOnOCR has separate 0.8B, 1B and 4B recipes.
The fifteen immutable model manifests include Nemotron's draft, FLUX's prompt
upsampler and the classifier loaded by FLUX's reference implementation.

These are executable candidates. Source and contract checks do not establish
container startup, performance, quality, or physical Spark qualification. No
workload deployment or checkpoint download is part of this publication.

| Model | Selected recipe | Sparks | Selection basis and remaining uncertainty |
| --- | --- | ---: | --- |
| Nemotron 3.5 Super VL 120B-A12B | NVFP4, vLLM, LiLiCorr seven-token draft | 1 | NVIDIA's Spark-specific launch configuration, including Mamba cache flags. The recommended v0.32.0 tag was unavailable; the documented immutable nightly has an ARM64 image. |
| LFM2.5-24B-A2B | BF16, vLLM | 1 | The 48 GB checkpoint fits without quantization. Native Lfm2Moe support is present in the pinned engine. FP8 may improve throughput, but no comparative Spark result was established. |
| Mellum2.1-12B-A2.5B-Thinking | BF16, vLLM | 1 | Upstream Qwen3 reasoning and Hermes tool parsers; the engine registers Mellum through its Transformers MoE backend. |
| MiMo-V2.6-Distill-Qwen-9B | BF16, SGLang | 1 | Follows Xiaomi's SGLang launch and MiMo reasoning parser. No tool-calling capability is claimed without a verified parser configuration. |
| Hy-Memory-30B-A3B | BF16, vLLM | 1 | Follows the model card's vLLM route. The pinned runtime contains HYV3 support. No parsed reasoning or tool-use claim is made. |
| LightOnOCR-3-0.8B | BF16, vLLM | 1 | Smallest document model; image input and thinking disabled. |
| LightOnOCR-3-1B | BF16, vLLM | 1 | Separate architecture/checkpoint from the 0.8B and 4B variants; image-input qualification is required. |
| LightOnOCR-3-4B | BF16, vLLM | 1 | Largest released OCR variant; same document-oriented interface with its own immutable assets. |
| Qwen-Image-2.1-Turbo | BF16, Diffusers | 1 | QwenImage21Pipeline at a source revision containing saved-sigma support; retains the checkpoint's eight-step schedule and prefix KV cache. |
| FLUX 3 Image Klein | BF16, native PyTorch reference pipeline | 1 | Distilled 16-step flow-map generation, required local prompt upsampling, and complete offline classifier assets. Uses the reference package's vLLM 0.21 dependency family. |
| Step 5 Preview | 0xSero EXL3 K4 experts/K8 hybrid body, vLLM, MTP2 | 4 | Dedicated ARM64 image and launch measured upstream on four Sparks. Upstream reports approximately 26–30 tokens/s; Vonk has not reproduced that measurement. |
| Muse Spark Open 816B-A42B | Unsloth NVFP4, dedicated Muse vLLM image | 8 | Approximately 533 GB of checkpoint assets requires a larger fleet. Marlin/FlashInfer is the SM121 candidate path; the eight-node adaptation has no physical acceptance evidence. No speculative draft is enabled before baseline qualification. |

## Access and reproducibility

The official FLUX checkpoint requires a provider token. Its immutable config was
successfully read using the existing Controller credential on October 17.
The official Muse NVFP4 checkpoint still returned HTTP 403 after the access
request, so this batch uses Unsloth's accessible distribution. These checks do
not promise future access or license eligibility for other accounts. Tokens
remain in Controller secrets and are absent from recipe packages.

Every image base is pinned to a verified linux/arm64 manifest. Model files use
provider LFS SHA-256 values or SHA-256 of the exact small-file bytes. The FLUX
source archive has a verified SHA-256; Step 5's image lock is derived from its
upstream launcher. The two image adapters consume the canonical typed job
manifest and reject changed or undeclared input bytes.

The image recipes expose text-to-image at 1024 × 1024. Editing and layout are
model capabilities but are not exposed by these recipes. Their estimates are
90 GB (Qwen) and 100 GB (FLUX), not measurements. Serving recipes estimate
allocated unified memory plus 8 GB of runtime/host overhead. Nemotron also
includes the library's 17 GB separate-drafter allowance and uses 0.70 GPU
utilization to preserve the Spark OS floor. Informational
reserve bytes are not an additional allocation. Four- and eight-Spark recipes
cannot run on the current two-Spark fleet. Their runtime smoke checks remain
in the recipe; they are outside the current one/two-Spark qualification campaign.

## Primary sources

- [NVIDIA model card and Spark launch](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4/tree/59b8721646c38111a242da18218859fb2b454322)
- [LiquidAI model card](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B/tree/1769516ac588c4c558089c94f27d96b0dc67b183)
- [JetBrains model card](https://huggingface.co/JetBrains/Mellum2.1-12B-A2.5B-Thinking/tree/92ddae9fc7665e9f801d141d2e5a6b2caf2460c4)
- [Xiaomi model card](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/tree/2367e865d009c13ac81713a2878291d33ab28177)
- [Tencent model card](https://huggingface.co/tencent/Hy-Memory-30B-A3B/tree/64a460a94d5498b7316fb02395c72d01c47a7347)
- [LightOnOCR 3](https://huggingface.co/lightonai/LightOnOCR-3-4B/tree/b71010f095dc3735de7ee6b9969bff8f8f50b39a)
- [Qwen Turbo sampling contract](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/tree/fc13ba1986fcb905854dc8de2f52eee395f49b5b)
- [FLUX reference implementation](https://github.com/black-forest-labs/flux-3-image-klein/tree/e53f4ea697fe45db8f38fca91a90f5623f7f8f37)
- [Step 5 four-Spark launcher and measurements](https://github.com/0xSero/Step-5-Preview-Four-Sparks/tree/662949a6c1244c1e8a5c44ed62cedb89a207cbcc)
- [Muse vLLM deployment](https://recipes.vllm.ai/meta-models/Muse-Spark-Open-816B-A42B)
