# Hy3-295B W4A4 NVFP4 on dual GB10

Audited Hy3 W4A4 v3 reproducibility and benchmark evidence for native vLLM on two NVIDIA GB10 systems.

- Model: [https://huggingface.co/r0b0tlab/Hy3-295B-NVFP4](https://huggingface.co/r0b0tlab/Hy3-295B-NVFP4)
- Runtime: `ghcr.io/r0b0tlab/vllm-v0250-cu130-sm121:v0.25.0-cu130-sm121-arm64-702f4814-r2`
- Model index SHA-256: `738e3ad2c8d16372e3d681d5793c78676fd256d7a4b43d9ca560f268d26adb26`
- Native path: `HYV3ForCausalLM` + `modelopt_mixed` + `FLASHINFER_CUTLASS`
- KV cache: FP8

## Quality — base O0/eager, non-MTP

| Task | N | Metric |
|---|---:|---:|
| gsm8k | 1319 | 95.91% |
| arc_challenge | 1172 | 96.16% |
| piqa | 1838 | 94.61% |
| winogrande | 1267 | 83.90% |
| truthfulqa_mc1 | 817 | 82.25% |
| mmlu_abstract_algebra | 100 | 74.00% |
| mmlu_business_ethics | 100 | 82.00% |
| mmlu_clinical_knowledge | 265 | 92.83% |
| mmlu_college_biology | 144 | 98.61% |
| mmlu_computer_security | 100 | 88.00% |
| mmlu_conceptual_physics | 235 | 94.89% |
| mmlu_high_school_world_history | 237 | 93.25% |
| mmlu_international_law | 121 | 92.56% |
| ifeval | 541 | 91.13% |
| humaneval | 164 | 84.15% |

These quality results are preserved unchanged from the prior publication run. IFEval reports official strict/loose instruction-following metrics in `benchmarks/quality/summary.json`. HumanEval is sandbox-executed pass@1.

## Base-profile concurrency

Highest safe passing concurrency: **c8**.

| Concurrency | Aggregate output tok/s | Per-request tok/s | Mean wall s | Mean output tokens |
|---|---:|---:|---:|---:|
| c1 | 8.73 | 8.73 | 14.66 | 128 |
| c2 | 16.67 | 8.37 | 15.36 | 256 |
| c4 | 32.03 | 8.02 | 15.98 | 512 |
| c8 | 50.94 | 6.37 | 20.10 | 1024 |

## Separate MTP-1 profile

MTP-1 passed at utilization 0.735; measured acceptance 52.97%; c1 mean 11.04 output tok/s under profile `mtp1_o0_eager_sync`. Forced 128/512/1024-token decoding passed 3/3 at each length, representative stability passed 100/100, and llama-benchy 0.4.0 completed.

## Separate practical base-AR context profile

A base-AR O0/eager/synchronous profile passed a **49,152-token configured window** with **49,968 measured KV-token capacity** and **3,904 MiB FP8 KV**. Near-window retrieval passed with needles near the beginning, middle, and end at up to 48,896 prompt tokens. This is a practical tested boundary, not a claim of architectural maximum context, and it is not the MTP profile.

## Reproduce

See `scripts/`, `runtime/runtime.json`, curated structured summaries, and `MANIFEST.sha256`. The launch scripts use one fail-closed two-rank owner, explicit FP8 KV, zero container swap, and a 16 GiB per-node `MemAvailable` floor.

## Attribution

Hy3 is from Tencent. Quantization/runtime work builds on NVIDIA ModelOpt, NVIDIA CUDA/GB10, FlashInfer, and vLLM; evaluation uses Hugging Face Datasets, lm-eval, and llama-benchy. Review upstream licenses and the base model license before use. This repository contains benchmark/reproducibility assets, not model weights.

## Appendix: methodology and claim boundaries

Quality, base concurrency, MTP-1, and practical base-AR context are distinct evidence profiles. NVFP4 describes model weights; all validated KV-cache profiles use FP8. MTP acceptance is claimed only because cumulative counters prove positive drafted and accepted tokens. Public evidence is curated summary data only: no raw private logs, telemetry streams, or per-item response dumps. Raw model weights remain in the existing Hugging Face repository and are not duplicated here.
