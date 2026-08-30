# LTX 2.5 gated-model preflight

This adapter uses the official gated `Lightricks/LTX-2.5-Diffusers`
repository at commit `426936f8b22dc28e4def61e515478b0b7e4a53cc`.

## Required before build and installation

1. Sign in to Hugging Face and review the exact pinned
   [LTX-2 Community License Agreement](https://github.com/Lightricks/LTX-2/blob/a95ab856bf29407b6b066ede0abe1846050db56c/LICENSE-2_x),
   whose SHA-256 is
   `be75acae5c99b0fb16ed6cfbf8f731e5121a729bef112d20337699407e796451`.
2. Open the [official model page](https://huggingface.co/Lightricks/LTX-2.5-Diffusers),
   review the contact-sharing gate, and request access if the account is not yet
   approved.
3. Put a Hugging Face read token for that same account in a file owned by the
   current user with mode `0600`. Never put the token in a recipe, command-line
   argument, build argument, environment entry, prompt, receipt, or log.
4. From this adapter directory, run the bounded preflight before requesting a
   recipe build or installation:

   ```shell
   python3 preflight.py \
     --token-file /secure/path/huggingface-token \
     --accept-license-sha256 be75acae5c99b0fb16ed6cfbf8f731e5121a729bef112d20337699407e796451
   ```

The preflight acknowledges that exact license revision and reads only the
505-byte gated `audio_vae/config.json` at the pinned model commit. It rejects
cross-origin redirects, verifies the immutable Git blob, never prints the
token, and reports actionable `401`, `403`, and `404` failures. A successful
JSON receipt proves that this token's account can read the pinned gated object;
it does not grant access or modify either account.

Configure the same approved account token through every target Spark agent's
existing root-owned `huggingface_curl_config` before installation. The signed
recipe makes the same 505-byte object its first artifact, ahead of the full
70,090,051,372-byte runtime closure. A missing, invalid, or unapproved Spark
credential therefore stops before the model artifact is transferred. The
current agent reports such an automatic failure as a generic managed-artifact
error; rerun this preflight for the clear account/token diagnosis.

The runtime closure remains the same 28 pinned distilled/convolutional-decoder
files. It excludes the full transformer and all optional diffusion-decoder,
prompt-enhancer, upsampler, duration-head, and LoRA assets. The probe is a
deliberate 505-byte duplicate of one runtime-required config so the runtime
snapshot remains a complete `/models/target` tree.

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
