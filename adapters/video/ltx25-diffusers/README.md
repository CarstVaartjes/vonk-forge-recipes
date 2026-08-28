# LTX 2.5 gated-model preflight

This adapter uses the official gated
`Lightricks/LTX-2.5-Diffusers` repository at commit
`426936f8b22dc28e4def61e515478b0b7e4a53cc`.

Before installation, the operator must:

1. Sign in to Hugging Face and review the binding
   [LTX-2 Community License Agreement](https://github.com/Lightricks/LTX-2/blob/a95ab856bf29407b6b066ede0abe1846050db56c/LICENSE-2_x).
2. Open the [official model page](https://huggingface.co/Lightricks/LTX-2.5-Diffusers), review the privacy/contact-sharing gate, and choose whether to request access.
3. Configure Vonk's existing Hugging Face download credential with a read token
   belonging to the same approved account. Do not put the token in a recipe,
   build argument, environment entry, prompt, or log.

Installation should fail before downloading weights when the account has not
accepted the gate or the credential is missing. The signed recipe downloads
only the 28 pinned distilled/convolutional-decoder files. It excludes the full
transformer and all optional diffusion-decoder, prompt-enhancer, upsampler,
duration-head, and LoRA assets.

Every job requires one UTF-8 text prompt (the common Prompt field). An optional
`/inputs/request.json` selects seed and memory profile:

```json
{
  "seed": 42,
  "profile": "bf16-model-offload"
}
```

Supported profiles are `bf16-model-offload`, `fp8-cast-model-offload`, and the
lower-CUDA-residency fallback `fp8-cast-sequential-offload`. DGX Spark CPU and
GPU allocations share the same 128 GB physical memory. Offload therefore
controls CUDA residency and allocator headroom; it does not create additional
physical RAM.

The filtered snapshot binds the four transformer shards implied by the pinned
index shared with the complete `transformer_full` component, and validates the
downloaded index before loading any weights. The authenticated physical install
is the final gate for that closure because Hugging Face does not expose gated
index contents to anonymous clients.

NVFP4 is intentionally not an install profile. Upstream `ltx-kernels` 1.3.0
only declares SM100a, SM110a, and SM120a; NVIDIA requires SM121-real for DGX
Spark. A later NVFP4 candidate needs explicit upstream SM121 support, a fully
pinned CUDA-devel toolchain, and repeatable physical-GB10 output evidence.
